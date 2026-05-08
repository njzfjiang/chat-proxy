from __future__ import annotations

import json
import asyncio
from datetime import datetime, timezone
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from .config import ProxyConfig, load_config
from .parsing import (
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
}


def create_app(config: ProxyConfig | None = None) -> FastAPI:
    cfg = config or load_config()
    store = ChatProxyStore(cfg.db_path)
    store.initialize()

    app = FastAPI(title="chat-proxy", version="0.1.0")
    app.state.config = cfg
    app.state.store = store
    app.state.summary_tasks = set()

    @app.get("/healthz")
    def healthz() -> dict[str, Any]:
        return {
            "ok": True,
            "upstream_base": cfg.upstream_base,
            "db_path": str(cfg.db_path),
        }

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

    incoming_headers = dict(request.headers)
    request_id = request_id_for(body_text, incoming_headers)
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
        "path": str(request.url.path),
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

    upstream_url = _upstream_url(cfg.upstream_base, request.url.path)
    headers = _forward_headers(incoming_headers)

    if upstream_body.get("stream") is True:
        return await _stream_upstream(
            app=request.app,
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
                app=request.app,
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


def _upstream_url(upstream_base: str, incoming_path: str) -> str:
    path = incoming_path
    if path.startswith("/v1/"):
        path = path[3:]
    if not path.startswith("/"):
        path = f"/{path}"
    return f"{upstream_base.rstrip('/')}{path}"


def _forward_headers(headers: dict[str, str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for key, value in headers.items():
        if key.lower() in HOP_BY_HOP_HEADERS:
            continue
        out[key] = value
    out.setdefault("content-type", "application/json")
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
