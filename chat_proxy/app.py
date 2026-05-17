from __future__ import annotations

import json
import asyncio
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from .config import ProxyConfig, load_config
from .parsing import (
    OPENAI_CHAT_COMPLETION_BODY_KEYS,
    SseTextAccumulator,
    extract_chat_completion_text,
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

    @app.get("/conversations")
    def conversations(limit: int = 50) -> dict[str, Any]:
        return {
            "conversations": [
                _conversation_payload(row)
                for row in store.list_conversations(limit=limit)
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
            "messages": messages,
        }

    @app.get("/conversations/{conversation_id}/rolling-short")
    def rolling_short(conversation_id: str) -> dict[str, Any]:
        return {
            "conversation_id": conversation_id,
            "summary": store.get_summary(conversation_id),
        }

    @app.get("/daily-summaries")
    def daily_summaries(
        date_key: str | None = None,
        limit: int = 30,
    ) -> dict[str, Any]:
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

    @app.post("/chat")
    async def chat(request: Request):
        body = await _read_json_object(request)
        if isinstance(body, JSONResponse):
            return body
        try:
            upstream_body = _web_chat_to_upstream_body(body, cfg)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        headers = _web_chat_headers(request.headers, body, cfg)
        body_text = json.dumps(upstream_body, ensure_ascii=False, sort_keys=True)
        return await _handle_chat_body(
            app=request.app,
            cfg=cfg,
            store=store,
            incoming_path="/chat/completions",
            incoming_headers=headers,
            body=upstream_body,
            body_text=body_text,
        )

    @app.post("/chat/completions")
    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        return await _handle_chat_completions(request, cfg, store)

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
    conversation_title = _conversation_title(identity)

    metadata = {
        "request_id": request_id,
        "conversation_resolver": identity.resolver,
        "conversation_metadata": identity.metadata or {},
        "upstream_body_mode": prepared_body.mode,
        "rolling_summary_injected": bool(summary_text and summary_text.strip()),
        "stripped_metadata": prepared_body.stripped_metadata or {},
        "path": incoming_path,
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

    user_text = last_user_text(body)
    if user_text:
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
        status = "completed"
        error_text = None
        try:
            async for chunk in upstream.aiter_bytes():
                if chunk:
                    accumulator.add_bytes(chunk)
                    raw_chunks.append(chunk.decode("utf-8", errors="replace"))
                    yield chunk
            if upstream.status_code >= 400:
                status = "error"
                error_text = "".join(raw_chunks)
        except Exception as exc:
            status = "error"
            error_text = str(exc)
            raise
        finally:
            await upstream.aclose()
            await client.aclose()
            store.complete_request(
                request_id=request_id,
                now=_now(),
                status=status,
                http_status=upstream.status_code,
                response_headers=sanitize_headers(dict(upstream.headers)),
                response_json={"stream": "".join(raw_chunks)},
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
                )
                _schedule_summary_update(
                    app=app,
                    cfg=cfg,
                    store=store,
                    conversation_id=conversation_id,
                )

    return StreamingResponse(
        body_iter(),
        status_code=upstream.status_code,
        media_type=upstream.headers.get("content-type", "text/event-stream"),
        headers=_response_headers(upstream.headers),
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


def _web_chat_to_upstream_body(
    body: dict[str, Any],
    cfg: ProxyConfig,
) -> dict[str, Any]:
    upstream = {
        key: value
        for key, value in body.items()
        if key in OPENAI_CHAT_COMPLETION_BODY_KEYS
    }
    messages = body.get("messages")
    user_text = str(body.get("user_text") or "").strip()
    system_prompt = str(body.get("system_prompt") or "").strip()
    if isinstance(messages, list):
        upstream_messages = list(messages)
    elif user_text:
        upstream_messages = [{"role": "user", "content": user_text}]
    else:
        raise ValueError("POST /chat requires messages or user_text.")
    if system_prompt:
        upstream_messages = [
            {"role": "system", "content": system_prompt},
            *upstream_messages,
        ]
    upstream["messages"] = upstream_messages
    if not str(upstream.get("model") or "").strip():
        upstream["model"] = cfg.chat_model
    return upstream


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
        out[key] = value
    out.setdefault("content-type", "application/json")
    if cfg.upstream_api_key and not any(
        key.lower() == "authorization" for key in out
    ):
        out["authorization"] = f"Bearer {cfg.upstream_api_key}"
    return out


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
