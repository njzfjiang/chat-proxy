from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .config import ProxyConfig, load_config
from .context_builder import (
    context_packet_from_snapshot,
    render_to_openai_messages,
    build_web_chat_context,
)
from .parsing import (
    SseTextAccumulator,
    extract_chat_completion_text,
    extract_token_usage,
    is_tool_continuation,
    last_user_text,
    message_id_for,
    prepare_request_body_for_upstream,
    request_id_for,
    resolve_conversation,
    sanitize_headers,
)
from .storage import ChatProxyStore
from .daily_summary import date_key_for, update_daily_summary
from .summary import inject_rolling_summary, update_conversation_summary
from .tool_registry import resolve_tools_policy


HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
    "host",
    "content-length",
    "accept-encoding",
    "content-encoding",
}


def create_app(config: ProxyConfig | None = None) -> FastAPI:
    cfg = config or load_config()
    store = ChatProxyStore(cfg.db_path)
    store.initialize()

    app = FastAPI(title="chat-proxy", version="0.1.0")
    app.state.config = cfg
    app.state.store = store
    app.state.summary_tasks = set()
    app.state.daily_summary_tasks = set()

    @app.get("/healthz")
    def healthz() -> dict[str, Any]:
        return {
            "ok": True,
            "upstream_base": cfg.upstream_base,
            "db_path": str(cfg.db_path),
        }

    @app.get("/admin/daily-summary/{date_key}")
    def daily_summary(date_key: str) -> dict[str, Any]:
        row = store.get_daily_summary(date_key)
        candidates = []
        if row:
            candidates = store.get_daily_memory_candidates(
                date_key=date_key,
                summary_version=int(row["version"]),
            )
        return {
            "date_key": date_key,
            "summary": row,
            "memory_candidates": candidates,
        }

    @app.get("/admin/daily-summary")
    def today_daily_summary() -> dict[str, Any]:
        date_key = date_key_for(_now(), cfg.daily_summary_timezone)
        row = store.get_daily_summary(date_key)
        candidates = []
        if row:
            candidates = store.get_daily_memory_candidates(
                date_key=date_key,
                summary_version=int(row["version"]),
            )
        return {
            "date_key": date_key,
            "summary": row,
            "memory_candidates": candidates,
        }

    @app.get("/admin/requests")
    def admin_requests(
        conversation_id: str | None = None,
        request_id: str | None = None,
        limit: int = 20,
        include_payloads: bool = False,
    ) -> dict[str, Any]:
        return {
            "requests": [
                _request_payload(row, include_payloads=include_payloads)
                for row in store.list_requests(
                    conversation_id=conversation_id,
                    request_id=request_id,
                    limit=limit,
                )
            ]
        }

    @app.get("/conversations")
    def conversations(
        limit: int = 50,
        include_archived: bool = False,
    ) -> dict[str, Any]:
        return {
            "conversations": [
                _conversation_payload(row)
                for row in store.list_conversations(
                    limit=limit,
                    include_archived=include_archived,
                )
            ]
        }

    @app.post("/conversations")
    async def create_conversation(request: Request):
        body = await _read_json_object(request)
        if isinstance(body, JSONResponse):
            return body
        now = _now()
        conversation_id = (
            str(body.get("conversation_id") or "").strip()
            or f"conv_{uuid4().hex}"
        )
        client_id = _optional_body_str(body, "client_id")
        assistant_key = _optional_body_str(body, "assistant_key")
        provider_key = _optional_body_str(body, "provider_key")
        title = _optional_body_str(body, "title") or assistant_key
        metadata = {
            "conversation_id": conversation_id,
            "client_key": client_id,
            "assistant_key": assistant_key,
            "provider_key": provider_key,
            "mode_hint": _optional_body_str(body, "mode_hint"),
        }
        store.upsert_conversation(
            conversation_id=conversation_id,
            now=now,
            resolver="webapp",
            client_key=client_id,
            assistant_key=assistant_key,
            title=title,
            metadata=metadata,
        )
        return {
            "conversation_id": conversation_id,
            "created_at": now,
            "client_id": client_id,
            "assistant_key": assistant_key,
            "provider_key": provider_key,
            "title": title,
        }

    @app.patch("/conversations/{conversation_id}")
    async def update_conversation(conversation_id: str, request: Request):
        body = await _read_json_object(request)
        if isinstance(body, JSONResponse):
            return body
        title = _optional_body_str(body, "title") if "title" in body else None
        archived = body.get("archived") if "archived" in body else None
        if archived is not None and not isinstance(archived, bool):
            return JSONResponse(
                {"error": "archived must be a boolean."},
                status_code=400,
            )
        row = store.update_conversation(
            conversation_id=conversation_id,
            now=_now(),
            title=title,
            archived=archived,
        )
        if row is None:
            return JSONResponse({"error": "Conversation not found."}, status_code=404)
        return {"conversation": _conversation_payload(row)}

    @app.post("/conversations/{conversation_id}/branches")
    async def branch_conversation(conversation_id: str, request: Request):
        body = await _read_json_object(request)
        if isinstance(body, JSONResponse):
            return body
        source_message_id = body.get("source_message_id")
        if not isinstance(source_message_id, int):
            return JSONResponse(
                {"error": "source_message_id must be an integer."},
                status_code=400,
            )
        now = _now()
        branch_id = (
            _optional_body_str(body, "conversation_id")
            or f"conv_{uuid4().hex}"
        )
        source = store.get_conversation(conversation_id)
        if source is None:
            return JSONResponse({"error": "Conversation not found."}, status_code=404)
        title = (
            _optional_body_str(body, "title")
            or f"{source.get('title') or source.get('assistant_key') or 'Conversation'} branch"
        )
        store.upsert_conversation(
            conversation_id=branch_id,
            now=now,
            resolver="branch",
            client_key=source.get("client_key"),
            assistant_key=source.get("assistant_key"),
            title=title,
            metadata={
                "source_conversation_id": conversation_id,
                "source_message_id": source_message_id,
            },
        )
        messages = store.get_conversation_messages_through(
            conversation_id=conversation_id,
            through_id=source_message_id,
        )
        if not messages:
            return JSONResponse({"error": "Source message not found."}, status_code=404)
        for message in messages:
            store.insert_message(
                timestamp=str(message.get("timestamp") or now),
                role=str(message.get("role") or "user"),
                content=str(message.get("content") or ""),
                conversation_title=title,
                conversation_id=branch_id,
                message_id=f"branch:{branch_id}:{message.get('id')}",
                kind=str(message.get("kind") or "chat"),
            )
        return {
            "conversation_id": branch_id,
            "source_conversation_id": conversation_id,
            "source_message_id": source_message_id,
            "copied_message_count": len(messages),
            "title": title,
        }

    @app.get("/conversations/{conversation_id}/messages")
    def conversation_messages(
        conversation_id: str,
        limit: int = 50,
        before_id: int | None = None,
        after_id: int | None = None,
        kind: str | None = None,
    ) -> dict[str, Any]:
        messages = store.get_conversation_messages(
            conversation_id=conversation_id,
            limit=limit,
            before_id=before_id,
            after_id=after_id,
            kind=kind,
        )
        return {
            "conversation_id": conversation_id,
            "messages": [_message_payload(row) for row in messages],
        }

    @app.delete("/conversations/{conversation_id}/messages/{message_id}")
    def delete_message(conversation_id: str, message_id: int) -> dict[str, Any]:
        deleted = store.delete_message(
            conversation_id=conversation_id,
            message_id=message_id,
        )
        if not deleted:
            return JSONResponse({"error": "Message not found."}, status_code=404)
        store.touch_conversation(conversation_id=conversation_id, now=_now())
        return {
            "conversation_id": conversation_id,
            "message_id": message_id,
            "deleted": True,
        }

    @app.get("/conversations/{conversation_id}/rolling-short")
    def rolling_short(conversation_id: str) -> dict[str, Any]:
        return {
            "conversation_id": conversation_id,
            "summary": store.get_summary(conversation_id),
        }

    @app.get("/conversations/{conversation_id}/rolling-short/versions")
    def rolling_short_versions(
        conversation_id: str,
        limit: int = 20,
    ) -> dict[str, Any]:
        return {
            "conversation_id": conversation_id,
            "versions": [
                _summary_version_payload(row)
                for row in store.list_summary_versions(
                    conversation_id=conversation_id,
                    limit=limit,
                )
            ],
        }

    @app.post("/conversations/{conversation_id}/rolling-short/rollback")
    async def rollback_rolling_short(conversation_id: str, request: Request):
        body = await _read_json_object(request)
        if isinstance(body, JSONResponse):
            return body
        raw_version = body.get("version")
        version = None
        if raw_version is not None:
            if not isinstance(raw_version, int):
                return JSONResponse(
                    {"error": "version must be an integer."},
                    status_code=400,
                )
            version = raw_version
        summary = store.rollback_summary(
            conversation_id=conversation_id,
            target_version=version,
            now=_now(),
        )
        if summary is None:
            return JSONResponse(
                {"error": "No summary version available to roll back to."},
                status_code=404,
            )
        return {
            "conversation_id": conversation_id,
            "summary": summary,
        }

    @app.get("/daily-summaries")
    def daily_summaries(
        date_key: str | None = None,
        days_ago: int | None = None,
        limit: int = 30,
    ) -> dict[str, Any]:
        if days_ago is not None:
            try:
                date_key = _date_key_for_days_ago(days_ago, cfg)
            except ValueError as exc:
                return JSONResponse({"error": str(exc)}, status_code=400)
        if date_key:
            row = store.get_daily_summary(date_key)
            candidates = []
            if row:
                candidates = store.get_daily_memory_candidates(
                    date_key=date_key,
                    summary_version=int(row["version"]),
                )
            return {
                "date_key": date_key,
                "summary": row,
                "memory_candidates": candidates,
            }
        return {"summaries": store.list_daily_summaries(limit=limit)}

    @app.post("/daily-summaries/run")
    async def run_daily_summary(request: Request):
        body = await _read_json_object(request)
        if isinstance(body, JSONResponse):
            return body
        try:
            date_key = _daily_run_date_key(body, cfg)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        force = body.get("force") is True
        scan_limit = body.get("scan_limit")
        if scan_limit is not None and not isinstance(scan_limit, int):
            return JSONResponse(
                {"error": "scan_limit must be an integer."},
                status_code=400,
            )
        if force:
            store.delete_daily_summary(date_key)
        now = _local_noon(date_key, cfg.daily_summary_timezone)
        await update_daily_summary(
            cfg=cfg,
            store=store,
            now=now,
            scan_limit=scan_limit or 100000,
        )
        row = store.get_daily_summary(date_key)
        candidates = []
        if row:
            candidates = store.get_daily_memory_candidates(
                date_key=date_key,
                summary_version=int(row["version"]),
            )
        return {
            "date_key": date_key,
            "summary": row,
            "memory_candidates": candidates,
        }

    @app.post("/chat")
    async def chat(request: Request):
        body = await _read_json_object(request)
        if isinstance(body, JSONResponse):
            return body
        headers = _web_chat_headers(request.headers, body, cfg)
        try:
            context_result = build_web_chat_context(
                body=body,
                cfg=cfg,
                store=store,
                headers=headers,
            )
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        upstream_body = context_result.upstream_body
        body_text = json.dumps(upstream_body, ensure_ascii=False, sort_keys=True)
        return await _handle_chat_body(
            app=request.app,
            cfg=cfg,
            store=store,
            incoming_path="/chat/completions",
            incoming_headers=headers,
            body=upstream_body,
            body_text=body_text,
            webapp_mode=True,
            context_snapshot=context_result.snapshot,
        )

    @app.post("/build_context")
    async def build_context(request: Request):
        body = await _read_json_object(request)
        if isinstance(body, JSONResponse):
            return body
        chat_body = _context_builder_chat_body(body)
        include = _context_builder_include(body)
        effective_cfg = _context_builder_config(cfg, include)
        headers = _web_chat_headers(request.headers, chat_body, effective_cfg)
        try:
            context_result = build_web_chat_context(
                body=chat_body,
                cfg=effective_cfg,
                store=store,
                headers=headers,
            )
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)

        identity = resolve_conversation(headers, chat_body)
        has_as_of_cutoff = bool(
            chat_body.get("as_of_message_id") or chat_body.get("as_of_timestamp")
        )
        summary_row = (
            store.get_summary(identity.conversation_id)
            if _include_enabled(include, "rolling_summary", True)
            and not has_as_of_cutoff
            else None
        )
        if has_as_of_cutoff and _include_enabled(include, "rolling_summary", True):
            context_result.snapshot["rolling_short_suppressed_reason"] = (
                "Historical cutoff requested; no as-of summary version is available."
            )
        summary_text = str(summary_row["summary"]) if summary_row else None
        upstream_body = inject_rolling_summary(
            context_result.upstream_body,
            summary_text,
        )
        snapshot = _injected_context_snapshot(
            context_snapshot=context_result.snapshot,
            summary_row=summary_row,
            final_body=upstream_body,
        )
        packet = context_packet_from_snapshot(
            snapshot=snapshot,
            messages=upstream_body.get("messages") or [],
        )
        messages = render_to_openai_messages(packet)
        debug = _context_builder_debug(
            snapshot=snapshot,
            body=body,
            chat_body=chat_body,
            include=include,
        )
        return {
            "messages": messages,
            "debug": debug,
            "context_packet": packet.to_dict(),
        }

    @app.post("/chat/completions")
    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        return await _handle_chat_completions(request, cfg, store)

    web_dist = Path(__file__).resolve().parent.parent / "web" / "dist"
    if web_dist.exists():
        app.mount("/", StaticFiles(directory=web_dist, html=True), name="web")

    return app


async def _handle_chat_completions(
    request: Request,
    cfg: ProxyConfig,
    store: ChatProxyStore,
):
    body_bytes = await request.body()
    body_text = body_bytes.decode("utf-8", errors="replace")
    try:
        body = json.loads(body_text) if body_text.strip() else {}
    except json.JSONDecodeError as exc:
        return JSONResponse(
            {"error": f"Request body must be a JSON object: {exc}"},
            status_code=400,
        )
    if not isinstance(body, dict):
        return JSONResponse(
            {"error": "Request body must be a JSON object."},
            status_code=400,
        )

    return await _handle_chat_body(
        app=request.app,
        cfg=cfg,
        store=store,
        incoming_path=str(request.url.path),
        incoming_headers=dict(request.headers),
        body=body,
        body_text=body_text,
        webapp_mode=False,
        context_snapshot=None,
    )


async def _handle_chat_body(
    *,
    app: FastAPI,
    cfg: ProxyConfig,
    store: ChatProxyStore,
    incoming_path: str,
    incoming_headers: dict[str, str],
    body: dict[str, Any],
    body_text: str,
    webapp_mode: bool,
    context_snapshot: dict[str, Any] | None,
):
    request_id = request_id_for(body_text, incoming_headers)
    duplicate = _duplicate_request_response(store, incoming_headers, request_id)
    if duplicate is not None:
        return duplicate
    identity = resolve_conversation(incoming_headers, body)
    prepared_body = prepare_request_body_for_upstream(incoming_headers, body)
    summary_row = store.get_summary(identity.conversation_id)
    summary_text = str(summary_row["summary"]) if summary_row else None
    upstream_body = inject_rolling_summary(prepared_body.body, summary_text)
    now = _now()
    model_id = str(upstream_body.get("model") or body.get("model") or "") or None
    user_text = last_user_text(body)
    reused_user_message = _reuses_user_message_for_tool_continuation(
        store=store,
        conversation_id=identity.conversation_id,
        body=body,
        user_text=user_text,
    )
    conversation_title = _conversation_title(identity)
    if webapp_mode and user_text:
        conversation_title = _auto_conversation_title(
            current_title=conversation_title,
            assistant_key=identity.assistant_key,
            user_text=user_text,
        )

    metadata = {
        "request_id": request_id,
        "conversation_resolver": identity.resolver,
        "conversation_metadata": identity.metadata or {},
        "upstream_body_mode": prepared_body.mode,
        "rolling_summary_injected": bool(summary_text and summary_text.strip()),
        "injected_context_snapshot": _injected_context_snapshot(
            context_snapshot=context_snapshot,
            summary_row=summary_row,
            final_body=upstream_body,
        ),
        "stripped_metadata": prepared_body.stripped_metadata or {},
        "path": incoming_path,
        "user_message_reused_for_tool_continuation": reused_user_message,
    }

    store.upsert_conversation(
        conversation_id=identity.conversation_id,
        now=now,
        resolver=identity.resolver,
        client_key=identity.client_key,
        assistant_key=identity.assistant_key,
        title=conversation_title,
        metadata=identity.metadata,
    )
    store.insert_request_pending(
        request_id=request_id,
        conversation_id=identity.conversation_id,
        now=now,
        provider_key=identity.provider_key,
        model_id=model_id,
        request_headers=sanitize_headers(incoming_headers),
        request_json=body,
        metadata=metadata,
    )

    if user_text and not reused_user_message:
        store.insert_message(
            timestamp=now,
            role="user",
            content=user_text,
            conversation_title=conversation_title,
            conversation_id=identity.conversation_id,
            message_id=message_id_for(
                request_id=request_id,
                conversation_id=identity.conversation_id,
                role="user",
                content=user_text,
            ),
            request_id=request_id,
        )

    upstream_url = _upstream_url(cfg.upstream_base, incoming_path)
    headers = _forward_headers(incoming_headers, cfg)

    if upstream_body.get("stream") is True:
        return await _stream_upstream(
            app=app,
            cfg=cfg,
            store=store,
            request_id=request_id,
            conversation_id=identity.conversation_id,
            conversation_title=conversation_title,
            upstream_url=upstream_url,
            headers=headers,
            body=upstream_body,
        )

    async with httpx.AsyncClient(timeout=None) as client:
        try:
            response = await client.post(upstream_url, headers=headers, json=upstream_body)
        except Exception as exc:  # httpx errors should be persisted before bubbling to client.
            store.complete_request(
                request_id=request_id,
                now=_now(),
                status="error",
                http_status=None,
                error_text=str(exc),
            )
            return JSONResponse({"error": str(exc)}, status_code=502)

    response_text = response.text
    response_payload: Any
    try:
        response_payload = response.json()
    except json.JSONDecodeError:
        response_payload = response_text

    status = "error" if response.status_code >= 400 else "completed"
    store.complete_request(
        request_id=request_id,
        now=_now(),
        status=status,
        http_status=response.status_code,
        response_headers=sanitize_headers(dict(response.headers)),
        response_json=response_payload,
        error_text=response_text if response.status_code >= 400 else None,
    )

    if isinstance(response_payload, dict):
        assistant_text = extract_chat_completion_text(response_payload)
        token_usage = extract_token_usage(response_payload)
        if assistant_text:
            store.insert_message(
                timestamp=_now(),
                role="assistant",
                content=assistant_text,
                conversation_title=conversation_title,
                conversation_id=identity.conversation_id,
                message_id=message_id_for(
                    request_id=request_id,
                    conversation_id=identity.conversation_id,
                    role="assistant",
                    content=assistant_text,
                ),
                request_id=request_id,
                token_usage=token_usage,
            )
            _schedule_summary_update(
                app=app,
                cfg=cfg,
                store=store,
                conversation_id=identity.conversation_id,
            )

    if isinstance(response_payload, (dict, list)):
        return JSONResponse(
            content=response_payload,
            status_code=response.status_code,
            headers=_response_headers(response.headers),
        )

    return Response(
        content=response.content,
        status_code=response.status_code,
        headers=_response_headers(response.headers),
        media_type=response.headers.get("content-type"),
    )


def _reuses_user_message_for_tool_continuation(
    *,
    store: ChatProxyStore,
    conversation_id: str,
    body: dict[str, Any],
    user_text: str | None,
) -> bool:
    if not user_text or not is_tool_continuation(body):
        return False
    latest_user = store.get_latest_message_by_role(
        conversation_id=conversation_id,
        role="user",
    )
    if latest_user is None:
        return False
    return str(latest_user.get("content") or "").strip() == user_text.strip()


async def _stream_upstream(
    *,
    app: FastAPI,
    cfg: ProxyConfig,
    store: ChatProxyStore,
    request_id: str,
    conversation_id: str,
    conversation_title: str | None,
    upstream_url: str,
    headers: dict[str, str],
    body: dict[str, Any],
):
    client = httpx.AsyncClient(timeout=None)
    accumulator = SseTextAccumulator()
    raw_chunks: list[str] = []
    synthetic_done_emitted = False

    try:
        upstream = await client.send(
            client.build_request("POST", upstream_url, headers=headers, json=body),
            stream=True,
        )
    except Exception as exc:
        await client.aclose()
        store.complete_request(
            request_id=request_id,
            now=_now(),
            status="error",
            http_status=None,
            error_text=str(exc),
        )
        return JSONResponse({"error": str(exc)}, status_code=502)

    async def body_iter():
        nonlocal synthetic_done_emitted
        status = "completed"
        error_text = None
        try:
            try:
                async for chunk in upstream.aiter_bytes():
                    if chunk:
                        accumulator.add_bytes(chunk)
                        raw_chunks.append(chunk.decode("utf-8", errors="replace"))
                        yield chunk
                if upstream.status_code >= 400:
                    status = "error"
                    error_text = "".join(raw_chunks)
            except (asyncio.CancelledError, GeneratorExit):
                status = "cancelled"
                error_text = "client disconnected while streaming"
                raise
            except httpx.RemoteProtocolError as exc:
                status = "upstream_incomplete" if raw_chunks else "error"
                error_text = str(exc)
                if not raw_chunks:
                    raise
            except Exception as exc:
                status = "error"
                error_text = str(exc)
                raise

            accumulator.finish()
            if (
                upstream.status_code < 400
                and accumulator.saw_data
                and not accumulator.done_received
            ):
                synthetic_done_emitted = True
                yield b"data: [DONE]\n\n"
        finally:
            await upstream.aclose()
            await client.aclose()
            store.complete_request(
                request_id=request_id,
                now=_now(),
                status=status,
                http_status=upstream.status_code,
                response_headers=sanitize_headers(dict(upstream.headers)),
                response_json={
                    "stream": "".join(raw_chunks),
                    "sse_done_received": accumulator.done_received,
                    "sse_finish_reason": accumulator.finish_reason,
                    "synthetic_done_emitted": synthetic_done_emitted,
                },
                error_text=error_text,
            )
            assistant_text = accumulator.text.strip()
            if assistant_text:
                store.insert_message(
                    timestamp=_now(),
                    role="assistant",
                    content=assistant_text,
                    conversation_title=conversation_title,
                    conversation_id=conversation_id,
                    message_id=message_id_for(
                        request_id=request_id,
                        conversation_id=conversation_id,
                        role="assistant",
                        content=assistant_text,
                    ),
                    request_id=request_id,
                    token_usage=accumulator.usage,
                )
                _schedule_summary_update(
                    app=app,
                    cfg=cfg,
                    store=store,
                    conversation_id=conversation_id,
                )

    response_headers = _response_headers(upstream.headers)
    response_headers.setdefault("cache-control", "no-cache")
    response_headers.setdefault("x-accel-buffering", "no")
    return StreamingResponse(
        body_iter(),
        status_code=upstream.status_code,
        media_type=upstream.headers.get("content-type", "text/event-stream"),
        headers=response_headers,
    )


async def _read_json_object(request: Request) -> dict[str, Any] | JSONResponse:
    body_bytes = await request.body()
    body_text = body_bytes.decode("utf-8", errors="replace")
    try:
        body = json.loads(body_text) if body_text.strip() else {}
    except json.JSONDecodeError as exc:
        return JSONResponse(
            {"error": f"Request body must be a JSON object: {exc}"},
            status_code=400,
        )
    if not isinstance(body, dict):
        return JSONResponse(
            {"error": "Request body must be a JSON object."},
            status_code=400,
        )
    return body


def _injected_context_snapshot(
    *,
    context_snapshot: dict[str, Any] | None,
    summary_row: dict[str, Any] | None,
    final_body: dict[str, Any],
) -> dict[str, Any]:
    snapshot = dict(context_snapshot or {"source": "proxy", "components": []})
    components = list(snapshot.get("components") or [])
    summary_text = str(summary_row.get("summary") or "") if summary_row else ""
    if summary_text.strip():
        components.insert(
            0,
            {
                "name": "rolling_short",
                "message_count": 1,
                "chars": len(summary_text),
                "version": summary_row.get("version"),
                "last_message_id": summary_row.get("last_message_id"),
            },
        )
    snapshot["components"] = components
    snapshot["rolling_short_injected"] = bool(summary_text.strip())
    snapshot["final_message_count"] = len(final_body.get("messages") or [])
    snapshot["final_chars"] = _message_list_chars(final_body.get("messages"))
    return snapshot


def _message_list_chars(messages: Any) -> int:
    if not isinstance(messages, list):
        return 0
    total = 0
    for message in messages:
        if isinstance(message, dict):
            total += len(str(message.get("content") or ""))
    return total


def _context_builder_chat_body(body: dict[str, Any]) -> dict[str, Any]:
    if isinstance(body.get("messages"), list) or str(body.get("user_text") or "").strip():
        return dict(body)

    recent_turns = body.get("recent_turns")
    if not isinstance(recent_turns, list):
        recent_turns = []
    include = _context_builder_include(body)
    include_recent = _include_enabled(include, "recent_turns", True)
    messages = _context_builder_messages(recent_turns, include_recent=include_recent)
    last_user = _last_user_turn_text(messages)

    chat_body: dict[str, Any] = {
        "client_id": str(body.get("user_id") or "").strip() or "context_builder",
        "conversation_id": str(body.get("conversation_id") or "").strip(),
        "assistant_key": "kai",
        "model": str(body.get("model") or "").strip(),
    }
    if not chat_body["conversation_id"]:
        chat_body.pop("conversation_id")
    if not chat_body["model"]:
        chat_body.pop("model")

    runtime_hint = _context_builder_runtime_hint(body)
    if runtime_hint and _include_enabled(include, "system", True):
        chat_body["system_prompt"] = runtime_hint

    params = body.get("params")
    if isinstance(params, dict):
        chat_body["params"] = params

    if messages:
        chat_body["messages"] = messages
    elif last_user:
        chat_body["user_text"] = last_user
    else:
        task_hint = str(body.get("task_hint") or "").strip()
        if task_hint:
            chat_body["user_text"] = task_hint
    task_hint = str(body.get("task_hint") or "").strip()
    if task_hint:
        chat_body["task_hint"] = task_hint
    for key in ("as_of_message_id", "as_of_timestamp"):
        value = body.get(key)
        if value is not None and str(value).strip():
            chat_body[key] = value
    return chat_body


def _context_builder_messages(
    recent_turns: list[Any],
    *,
    include_recent: bool,
) -> list[dict[str, str]]:
    messages = []
    for turn in recent_turns:
        if not isinstance(turn, dict):
            continue
        role = str(turn.get("role") or "").strip()
        if role == "tool":
            role = "assistant"
        if role not in {"system", "user", "assistant"}:
            continue
        content = str(turn.get("content") or "").strip()
        if not content:
            continue
        messages.append({"role": role, "content": content})
    if include_recent:
        return messages
    last_user = _last_user_turn_text(messages)
    return [{"role": "user", "content": last_user}] if last_user else []


def _last_user_turn_text(messages: list[dict[str, str]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            return str(message.get("content") or "").strip()
    return ""


def _context_builder_include(body: dict[str, Any]) -> dict[str, Any]:
    include = body.get("include")
    return dict(include) if isinstance(include, dict) else {}


def _context_builder_runtime_hint(body: dict[str, Any]) -> str:
    parts = []
    for key in ("mode", "locale", "timestamp", "task_hint"):
        value = str(body.get(key) or "").strip()
        if value:
            parts.append(f"{key}: {value}")
    runtime_prefs = _clean_mapping(body.get("runtime_prefs"))
    if runtime_prefs:
        parts.append("runtime_prefs: " + _mapping_inline(runtime_prefs))
    guardrails = _clean_mapping(body.get("guardrails"))
    if guardrails:
        parts.append("guardrails: " + _mapping_inline(guardrails))
    if not parts:
        return ""
    return "[Runtime Hints]\n" + "\n".join(f"- {part}" for part in parts)


def _context_builder_config(
    cfg: ProxyConfig,
    include: dict[str, Any],
) -> ProxyConfig:
    updates: dict[str, Any] = {}
    if not _include_enabled(include, "worldbook", True):
        updates["worldbook_enabled"] = False
    if not _include_enabled(include, "core_anchors", True):
        updates["core_anchors_enabled"] = False
    if not _include_enabled(include, "mother_memory", True):
        updates["mother_memory_enabled"] = False
    return replace(cfg, **updates) if updates else cfg


def _include_enabled(
    include: dict[str, Any],
    key: str,
    default: bool,
) -> bool:
    value = include.get(key, default)
    return value is not False


def _context_builder_debug(
    *,
    snapshot: dict[str, Any],
    body: dict[str, Any],
    chat_body: dict[str, Any],
    include: dict[str, Any],
) -> dict[str, Any]:
    components = [
        component
        for component in snapshot.get("components") or []
        if isinstance(component, dict)
    ]
    injected_by_layer = {
        str(component.get("name") or f"component_{index}"): _rough_tokens(
            int(component.get("chars") or 0)
        )
        for index, component in enumerate(components)
        if int(component.get("message_count") or 0) > 0
    }
    retrieved_not_injected_by_layer = {
        str(component.get("name") or f"component_{index}"): _rough_tokens(
            int(component.get("retrieved_chars") or component.get("chars") or 0)
        )
        for index, component in enumerate(components)
        if int(component.get("message_count") or 0) <= 0
        and (
            int(component.get("result_count") or 0) > 0
            or bool(component.get("items"))
        )
    }
    by_layer = {
        str(component.get("name") or f"component_{index}"): _rough_tokens(
            int(component.get("chars") or 0)
        )
        for index, component in enumerate(components)
    }
    notes = [
        "Preview only: /build_context does not call the upstream LLM or persist turns.",
    ]
    if "messages" not in body and "user_text" not in body:
        notes.append("Request was normalized from ContextBuilderRequest-like fields.")
    if include.get("chatlog_history"):
        notes.append("chatlog_history is not implemented in this thin endpoint yet.")
    if include.get("health_data"):
        notes.append("health_data is not implemented in this thin endpoint yet.")
    suppressed_summary = snapshot.get("rolling_short_suppressed_reason")
    if suppressed_summary:
        notes.append(str(suppressed_summary))
    return {
        "included_layers": [
            str(component.get("name"))
            for component in components
            if component.get("name") and int(component.get("message_count") or 0) > 0
        ],
        "source_ids": _context_builder_source_ids(components),
        "token_estimates": {
            "total": _rough_tokens(int(snapshot.get("final_chars") or 0)),
            "injected_total": sum(injected_by_layer.values()),
            "retrieved_not_injected_total": sum(
                retrieved_not_injected_by_layer.values()
            ),
            "injected_by_layer": injected_by_layer,
            "retrieved_not_injected_by_layer": retrieved_not_injected_by_layer,
            "by_layer": by_layer,
        },
        "truncated": {"by_layer": {}},
        "notes": notes,
        "mode": snapshot.get("mode"),
        "model": snapshot.get("model") or chat_body.get("model"),
        "source": snapshot.get("source"),
        "request_meta": _context_builder_request_meta(body),
        "tool_context": resolve_tools_policy(
            body.get("tools_policy") if isinstance(body.get("tools_policy"), dict) else None
        ),
    }


def _context_builder_source_ids(
    components: list[dict[str, Any]],
) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for index, component in enumerate(components):
        layer = str(component.get("name") or f"component_{index}")
        out[layer] = _component_source_ids(component)
    return out


def _component_source_ids(component: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    layer = str(component.get("name") or "").strip()

    for raw_id in component.get("message_ids") or []:
        ids.append(f"message:{raw_id}")

    if layer == "rolling_short":
        version = component.get("version")
        if version is not None:
            ids.append(f"rolling_summary:{version}")
        last_message_id = component.get("last_message_id")
        if last_message_id is not None:
            ids.append(f"message:{last_message_id}")

    for item in component.get("items") or []:
        if not isinstance(item, dict):
            continue
        if layer == "core_anchors":
            value = item.get("anchor_key")
            if value:
                ids.append(f"core_anchor:{value}")
        elif layer == "mother_memory":
            value = item.get("path")
            if value:
                ids.append(f"mother:{value}")
        elif layer == "wb_snippets":
            value = item.get("id") or item.get("name")
            source = item.get("source")
            if value and source:
                ids.append(f"worldbook:{source}#{value}")
            elif value:
                ids.append(f"worldbook:{value}")
            elif source:
                ids.append(f"worldbook:{source}")
        elif layer == "kmlog_search":
            value = item.get("id")
            if value is not None:
                ids.append(f"kmlog:{value}")
        elif item.get("id") is not None:
            ids.append(f"{layer}:{item.get('id')}")

    return _dedupe_debug_ids(ids)


def _dedupe_debug_ids(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw_item in items:
        item = str(raw_item or "").strip()
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _rough_tokens(chars: int) -> int:
    if chars <= 0:
        return 0
    return max(1, round(chars / 4))


def _context_builder_request_meta(body: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in (
        "user_id",
        "conversation_id",
        "frontend",
        "mode",
        "locale",
        "task_hint",
        "timestamp",
        "as_of_message_id",
        "as_of_timestamp",
    ):
        value = body.get(key)
        if value is not None:
            out[key] = value
    for key in ("runtime_prefs", "guardrails"):
        value = body.get(key)
        if isinstance(value, dict):
            out[key] = _clean_mapping(value)
    return out


def _clean_mapping(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    out = {}
    for key, raw_item in value.items():
        name = str(key or "").strip()
        if not name or raw_item is None:
            continue
        if isinstance(raw_item, (str, int, float, bool)):
            out[name] = raw_item
    return out


def _mapping_inline(value: dict[str, Any]) -> str:
    return ", ".join(f"{key}={item}" for key, item in value.items())


def _web_chat_to_upstream_body(
    body: dict[str, Any],
    cfg: ProxyConfig,
    store: ChatProxyStore,
    headers: dict[str, str],
) -> dict[str, Any]:
    return build_web_chat_context(
        body=body,
        cfg=cfg,
        store=store,
        headers=headers,
    ).upstream_body


def _web_chat_headers(
    headers: Any,
    body: dict[str, Any],
    cfg: ProxyConfig,
) -> dict[str, str]:
    out = dict(headers)
    header_map = {
        "client_id": "X-Kelivo-Client-Id",
        "conversation_id": "X-Kelivo-Conversation-Id",
        "request_id": "X-Kelivo-Request-Id",
        "assistant_key": "X-Kelivo-Assistant-Key",
        "provider_key": "X-Kelivo-Provider-Key",
    }
    for body_key, header_key in header_map.items():
        value = _optional_body_str(body, body_key)
        if value:
            out[header_key] = value
    if "X-Kelivo-Provider-Key" not in out and cfg.provider_key:
        out["X-Kelivo-Provider-Key"] = cfg.provider_key
    out.setdefault("content-type", "application/json")
    return out


def _conversation_payload(row: dict[str, Any]) -> dict[str, Any]:
    content = str(row.get("last_message_content") or "")
    return {
        "conversation_id": row.get("conversation_id"),
        "created_at": row.get("created_at"),
        "updated_at": row.get("updated_at"),
        "resolver": row.get("resolver"),
        "client_id": row.get("client_key"),
        "assistant_key": row.get("assistant_key"),
        "title": row.get("title"),
        "archived_at": row.get("archived_at"),
        "metadata": _decode_json(row.get("metadata_json")),
        "message_count": int(row.get("message_count") or 0),
        "last_message_id": row.get("last_message_id"),
        "last_message_at": row.get("last_message_at"),
        "last_message_role": row.get("last_message_role"),
        "last_message_preview": content[:240],
        "rolling_summary": row.get("rolling_summary"),
        "rolling_summary_status": row.get("rolling_summary_status"),
        "rolling_summary_version": row.get("rolling_summary_version"),
    }


def _message_payload(row: dict[str, Any]) -> dict[str, Any]:
    payload = dict(row)
    payload["token_usage"] = _decode_json(payload.pop("token_usage_json", None))
    return payload


def _request_payload(
    row: dict[str, Any],
    *,
    include_payloads: bool = False,
) -> dict[str, Any]:
    payload = {
        "request_id": row.get("request_id"),
        "conversation_id": row.get("conversation_id"),
        "created_at": row.get("created_at"),
        "updated_at": row.get("updated_at"),
        "provider_key": row.get("provider_key"),
        "model_id": row.get("model_id"),
        "status": row.get("status"),
        "http_status": row.get("http_status"),
        "metadata": _debug_metadata(_decode_json(row.get("metadata_json"))),
        "error_text": row.get("error_text"),
    }
    if include_payloads:
        payload["request"] = _decode_json(row.get("request_json"))
        payload["response"] = _decode_json(row.get("response_json"))
        payload["metadata"] = _decode_json(row.get("metadata_json"))
    return payload


def _summary_version_payload(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row.get("id"),
        "conversation_id": row.get("conversation_id"),
        "version": row.get("version"),
        "summary": row.get("summary"),
        "created_at": row.get("created_at"),
        "last_message_id": row.get("last_message_id"),
        "previous_last_message_id": row.get("previous_last_message_id"),
        "source_message_count": row.get("source_message_count"),
        "model_id": row.get("model_id"),
        "metadata": _decode_json(row.get("metadata_json")),
    }


def _debug_metadata(metadata: Any) -> dict[str, Any]:
    if not isinstance(metadata, dict):
        return {}
    return {
        "conversation_resolver": metadata.get("conversation_resolver"),
        "upstream_body_mode": metadata.get("upstream_body_mode"),
        "rolling_summary_injected": metadata.get("rolling_summary_injected"),
        "injected_context_snapshot": metadata.get("injected_context_snapshot"),
        "path": metadata.get("path"),
    }


def _optional_body_str(body: dict[str, Any], key: str) -> str | None:
    value = body.get(key)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _decode_json(value: Any) -> Any:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None


def _duplicate_request_response(
    store: ChatProxyStore,
    headers: dict[str, str],
    request_id: str,
) -> Response | None:
    if not _has_explicit_request_id(headers):
        return None
    existing = store.get_request(request_id)
    if not existing:
        return None
    status = str(existing.get("status") or "")
    if status != "completed":
        return JSONResponse(
            {
                "error": "duplicate request is still pending or failed",
                "request_id": request_id,
                "status": status,
            },
            status_code=409,
        )
    payload = _decode_json(existing.get("response_json"))
    http_status = int(existing.get("http_status") or 200)
    response_headers = _decode_json(existing.get("response_headers_json"))
    headers_out = (
        _plain_headers(response_headers)
        if isinstance(response_headers, dict)
        else {}
    )
    if isinstance(payload, (dict, list)):
        return JSONResponse(
            content=payload,
            status_code=http_status,
            headers=headers_out,
        )
    return Response(
        content="" if payload is None else str(payload),
        status_code=http_status,
        headers=headers_out,
    )


def _has_explicit_request_id(headers: dict[str, str]) -> bool:
    lowered = {key.lower(): value for key, value in headers.items()}
    return bool(
        (
            lowered.get("x-request-id")
            or lowered.get("x-kelivo-request-id")
            or lowered.get("request-id")
            or ""
        ).strip()
    )


def _plain_headers(headers: dict[str, Any]) -> dict[str, str]:
    return {str(key): str(value) for key, value in headers.items()}


def _upstream_url(upstream_base: str, incoming_path: str) -> str:
    path = incoming_path
    if path.startswith("/v1/"):
        path = path[3:]
    if not path.startswith("/"):
        path = f"/{path}"
    return f"{upstream_base.rstrip('/')}{path}"


def _forward_headers(headers: dict[str, str], cfg: ProxyConfig) -> dict[str, str]:
    out: dict[str, str] = {}
    for key, value in headers.items():
        if key.lower() in HOP_BY_HOP_HEADERS:
            continue
        if key.lower() == "authorization" and not _is_provider_authorization(value):
            continue
        out[key] = value
    out.setdefault("content-type", "application/json")
    if cfg.upstream_api_key and not _has_provider_authorization(out):
        out["authorization"] = f"Bearer {cfg.upstream_api_key}"
    return out


def _has_provider_authorization(headers: dict[str, str]) -> bool:
    for key, value in headers.items():
        if key.lower() == "authorization" and _is_provider_authorization(value):
            return True
    return False


def _is_provider_authorization(value: str) -> bool:
    return value.strip().lower().startswith("bearer ")


def _response_headers(headers: httpx.Headers) -> dict[str, str]:
    out: dict[str, str] = {}
    for key, value in headers.items():
        if key.lower() in HOP_BY_HOP_HEADERS:
            continue
        out[key] = value
    return out


def _conversation_title(identity) -> str | None:
    if not identity.metadata:
        return identity.assistant_key
    return (
        identity.metadata.get("conversation_title")
        or identity.assistant_key
        or identity.metadata.get("assistant")
        or identity.metadata.get("assistant_key")
    )


def _auto_conversation_title(
    *,
    current_title: str | None,
    assistant_key: str | None,
    user_text: str,
) -> str | None:
    if current_title and current_title.strip() and current_title != assistant_key:
        return current_title
    cleaned = " ".join(user_text.split())
    if not cleaned:
        return current_title or assistant_key
    return cleaned[:48]


def _daily_run_date_key(body: dict[str, Any], cfg: ProxyConfig) -> str:
    raw_date = _optional_body_str(body, "date_key")
    if raw_date:
        try:
            return datetime.strptime(raw_date, "%Y-%m-%d").date().isoformat()
        except ValueError as exc:
            raise ValueError("date_key must be formatted as YYYY-MM-DD.") from exc
    raw_days_ago = body.get("days_ago", 0)
    if not isinstance(raw_days_ago, int):
        raise ValueError("days_ago must be an integer.")
    return _date_key_for_days_ago(raw_days_ago, cfg)


def _date_key_for_days_ago(days_ago: int, cfg: ProxyConfig) -> str:
    if days_ago < 0 or days_ago > 365:
        raise ValueError("days_ago must be between 0 and 365.")
    tz = _zoneinfo(cfg.daily_summary_timezone)
    target = datetime.now(tz).date() - timedelta(days=days_ago)
    return target.isoformat()


def _local_noon(date_key: str, timezone_name: str) -> str:
    tz = _zoneinfo(timezone_name)
    return datetime.strptime(date_key, "%Y-%m-%d").replace(
        hour=12,
        tzinfo=tz,
    ).isoformat()


def _zoneinfo(timezone_name: str):
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

    try:
        return ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError:
        return ZoneInfo("America/Toronto")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _schedule_summary_update(
    *,
    app: FastAPI,
    cfg: ProxyConfig,
    store: ChatProxyStore,
    conversation_id: str,
) -> None:
    if not cfg.summary_enabled:
        _schedule_daily_summary_update(app=app, cfg=cfg, store=store)
        return
    task = asyncio.create_task(
        update_conversation_summary(
            cfg=cfg,
            store=store,
            conversation_id=conversation_id,
            now=_now(),
        )
    )
    app.state.summary_tasks.add(task)
    task.add_done_callback(app.state.summary_tasks.discard)
    _schedule_daily_summary_update(app=app, cfg=cfg, store=store)


def _schedule_daily_summary_update(
    *,
    app: FastAPI,
    cfg: ProxyConfig,
    store: ChatProxyStore,
) -> None:
    if not cfg.daily_summary_enabled:
        return
    task = asyncio.create_task(
        update_daily_summary(
            cfg=cfg,
            store=store,
            now=_now(),
        )
    )
    app.state.daily_summary_tasks.add(task)
    task.add_done_callback(app.state.daily_summary_tasks.discard)
