import json
import asyncio
import sqlite3

import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from httpx import ASGITransport

from chat_proxy.app import create_app
from chat_proxy.config import ProxyConfig


def _create_base_db(path):
    conn = sqlite3.connect(path)
    conn.executescript(
        """
CREATE TABLE messages (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  timestamp TEXT,
  role TEXT,
  content TEXT,
  conversation_title TEXT,
  conversation_id TEXT,
  message_id TEXT UNIQUE,
  kind TEXT DEFAULT 'chat'
);
CREATE VIRTUAL TABLE messages_fts USING fts5(
  content,
  conversation_title,
  content=messages,
  content_rowid=id,
  tokenize='unicode61'
);
"""
    )
    conn.close()


async def _wait_summary_tasks(app):
    tasks = list(app.state.summary_tasks)
    if tasks:
        await asyncio.gather(*tasks)


@pytest.fixture
def upstream_app():
    app = FastAPI()

    @app.post("/chat/completions")
    async def completions(request: Request):
        body = await request.json()
        return JSONResponse(
            {
                "id": "cmpl-test",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": f"echo: {body['messages'][-1]['content']}",
                        }
                    }
                ],
            }
        )

    @app.post("/stream/chat/completions")
    async def stream_completions(_request: Request):
        async def chunks():
            yield b'data: {"choices":[{"delta":{"content":"hel"}}]}\n\n'
            yield b'data: {"choices":[{"delta":{"content":"lo"}}]}\n\n'
            yield b"data: [DONE]\n\n"

        return StreamingResponse(chunks(), media_type="text/event-stream")

    return app


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_proxy_forwards_and_persists_non_stream(tmp_path, upstream_app, monkeypatch):
    db_path = tmp_path / "chat_search.db"
    _create_base_db(db_path)

    transport = ASGITransport(app=upstream_app)
    original_async_client = httpx.AsyncClient

    def client_factory(*args, **kwargs):
        kwargs["transport"] = transport
        kwargs["base_url"] = "http://upstream"
        return original_async_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", client_factory)
    app = create_app(
        ProxyConfig(upstream_base="http://upstream", db_path=db_path)
    )

    async with original_async_client(
        transport=ASGITransport(app=app),
        base_url="http://proxy",
    ) as client:
        resp = await client.post(
            "/chat/completions",
            headers={
                "Authorization": "Bearer secret",
                "X-Kelivo-Conversation-Id": "chat-1",
                "X-Kelivo-Assistant-Key": "kai",
                "X-Kelivo-Provider-Key": "openai",
            },
            json={
                "model": "gpt-test",
                "messages": [{"role": "user", "content": "2026-05-07 21:40 \r\nhello"}],
            },
        )

    assert resp.status_code == 200
    assert (
        resp.json()["choices"][0]["message"]["content"]
        == "echo: 2026-05-07 21:40 \r\nhello"
    )

    conn = sqlite3.connect(db_path)
    messages = conn.execute(
        "SELECT role, content, conversation_id FROM messages ORDER BY id"
    ).fetchall()
    request_row = conn.execute(
        "SELECT status, request_headers_json, provider_key FROM requests"
    ).fetchone()
    fts_rows = conn.execute(
        "SELECT rowid FROM messages_fts WHERE messages_fts MATCH ?",
        ('"echo 2026"',),
    ).fetchall()
    conn.close()

    assert messages == [
        ("user", "hello", "chat-1"),
        ("assistant", "echo: 2026-05-07 21:40 \r\nhello", "chat-1"),
    ]
    assert request_row[0] == "completed"
    assert json.loads(request_row[1])["authorization"] == "[REDACTED]"
    assert request_row[2] == "openai"
    assert len(fts_rows) == 1


@pytest.mark.anyio
async def test_proxy_strips_kelivo_analysis_meta_before_forwarding(
    tmp_path, upstream_app, monkeypatch
):
    db_path = tmp_path / "chat_search.db"
    _create_base_db(db_path)
    captured_body = {}

    async def completions(request: Request):
        captured_body.update(await request.json())
        return JSONResponse(
            {
                "id": "cmpl-test",
                "choices": [{"message": {"role": "assistant", "content": "ok"}}],
            }
        )

    upstream_app.router.routes.clear()
    upstream_app.post("/chat/completions")(completions)

    transport = ASGITransport(app=upstream_app)
    original_async_client = httpx.AsyncClient

    def client_factory(*args, **kwargs):
        kwargs["transport"] = transport
        kwargs["base_url"] = "http://upstream"
        return original_async_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", client_factory)
    app = create_app(ProxyConfig(upstream_base="http://upstream", db_path=db_path))

    async with original_async_client(
        transport=ASGITransport(app=app),
        base_url="http://proxy",
    ) as client:
        resp = await client.post(
            "/chat/completions",
            headers={"X-Kelivo-Analysis-Version": "1"},
            json={
                "model": "gpt-test",
                "messages": [{"role": "user", "content": "hello dev"}],
                "_kelivo_analysis_meta": {
                    "conversation_id": "chat-dev",
                    "conversation_title": "Dev Chat",
                    "assistant_id": "assistant-a",
                    "provider_key": "openai",
                },
                "extra_body": {"debug": True},
            },
        )

    assert resp.status_code == 200
    assert captured_body == {
        "model": "gpt-test",
        "messages": [{"role": "user", "content": "hello dev"}],
    }

    conn = sqlite3.connect(db_path)
    request_row = conn.execute(
        "SELECT conversation_id, provider_key, request_json, metadata_json FROM requests"
    ).fetchone()
    conversation_row = conn.execute(
        "SELECT conversation_id, title, assistant_key FROM conversations"
    ).fetchone()
    conn.close()

    assert request_row[0] == "chat-dev"
    assert request_row[1] == "openai"
    assert "_kelivo_analysis_meta" in json.loads(request_row[2])
    metadata = json.loads(request_row[3])
    assert metadata["upstream_body_mode"] == "kelivo_analysis"
    assert metadata["stripped_metadata"]["conversation_title"] == "Dev Chat"
    assert conversation_row == ("chat-dev", "Dev Chat", "assistant-a")


@pytest.mark.anyio
async def test_proxy_streams_and_persists_assistant_text(tmp_path, upstream_app, monkeypatch):
    db_path = tmp_path / "chat_search.db"
    _create_base_db(db_path)

    transport = ASGITransport(app=upstream_app)
    original_async_client = httpx.AsyncClient

    def client_factory(*args, **kwargs):
        kwargs["transport"] = transport
        kwargs["base_url"] = "http://upstream"
        return original_async_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", client_factory)
    app = create_app(
        ProxyConfig(upstream_base="http://upstream/stream", db_path=db_path)
    )

    async with original_async_client(
        transport=ASGITransport(app=app),
        base_url="http://proxy",
    ) as client:
        resp = await client.post(
            "/chat/completions",
            headers={"X-Kelivo-Conversation-Id": "chat-stream"},
            json={
                "model": "gpt-test",
                "stream": True,
                "messages": [{"role": "user", "content": "hello stream"}],
            },
        )

    assert resp.status_code == 200
    assert "data:" in resp.text

    conn = sqlite3.connect(db_path)
    assistant = conn.execute(
        "SELECT content FROM messages WHERE role = 'assistant'"
    ).fetchone()[0]
    status = conn.execute("SELECT status FROM requests").fetchone()[0]
    conn.close()

    assert assistant == "hello"
    assert status == "completed"


@pytest.mark.anyio
async def test_proxy_injects_existing_summary_without_mutating_request_json(
    tmp_path, upstream_app, monkeypatch
):
    db_path = tmp_path / "chat_search.db"
    _create_base_db(db_path)
    captured_body = {}

    async def completions(request: Request):
        captured_body.update(await request.json())
        return JSONResponse(
            {
                "id": "cmpl-test",
                "choices": [{"message": {"role": "assistant", "content": "ok"}}],
            }
        )

    upstream_app.router.routes.clear()
    upstream_app.post("/chat/completions")(completions)

    transport = ASGITransport(app=upstream_app)
    original_async_client = httpx.AsyncClient

    def client_factory(*args, **kwargs):
        kwargs["transport"] = transport
        kwargs["base_url"] = "http://upstream"
        return original_async_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", client_factory)
    app = create_app(ProxyConfig(upstream_base="http://upstream", db_path=db_path))
    app.state.store.upsert_conversation(
        conversation_id="chat-1",
        now="2026-05-08T00:00:00Z",
        resolver="header",
        client_key="desktop",
        assistant_key="kai",
        title="kai",
        metadata=None,
    )
    app.state.store.upsert_summary(
        conversation_id="chat-1",
        summary="User prefers concise answers.",
        last_message_id=None,
        updated_at="2026-05-08T00:00:00Z",
    )

    async with original_async_client(
        transport=ASGITransport(app=app),
        base_url="http://proxy",
    ) as client:
        resp = await client.post(
            "/chat/completions",
            headers={"X-Kelivo-Conversation-Id": "chat-1"},
            json={
                "model": "gpt-test",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

    assert resp.status_code == 200
    assert captured_body["messages"][0]["role"] == "system"
    assert "User prefers concise answers." in captured_body["messages"][0]["content"]
    assert captured_body["messages"][1:] == [{"role": "user", "content": "hello"}]

    conn = sqlite3.connect(db_path)
    request_json = json.loads(
        conn.execute("SELECT request_json FROM requests").fetchone()[0]
    )
    metadata = json.loads(
        conn.execute("SELECT metadata_json FROM requests").fetchone()[0]
    )
    conn.close()

    assert request_json["messages"] == [{"role": "user", "content": "hello"}]
    assert metadata["rolling_summary_injected"] is True


@pytest.mark.anyio
async def test_proxy_updates_summary_after_non_stream_response(
    tmp_path, upstream_app, monkeypatch
):
    db_path = tmp_path / "chat_search.db"
    _create_base_db(db_path)
    summary_body = {}
    summary_headers = {}

    async def completions(_request: Request):
        return JSONResponse(
            {
                "id": "cmpl-test",
                "choices": [{"message": {"role": "assistant", "content": "answer"}}],
            }
        )

    async def summary_completions(request: Request):
        summary_body.update(await request.json())
        summary_headers.update(dict(request.headers))
        return JSONResponse(
            {
                "id": "summary-test",
                "choices": [
                    {"message": {"role": "assistant", "content": "updated summary"}}
                ],
            }
        )

    upstream_app.router.routes.clear()
    upstream_app.post("/chat/completions")(completions)
    upstream_app.post("/summary/chat/completions")(summary_completions)

    transport = ASGITransport(app=upstream_app)
    original_async_client = httpx.AsyncClient

    def client_factory(*args, **kwargs):
        kwargs["transport"] = transport
        kwargs["base_url"] = "http://upstream"
        return original_async_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", client_factory)
    app = create_app(
        ProxyConfig(
            upstream_base="http://upstream",
            db_path=db_path,
            summary_enabled=True,
            summary_upstream_base="http://upstream/summary",
            summary_api_key="summary-secret",
        )
    )

    async with original_async_client(
        transport=ASGITransport(app=app),
        base_url="http://proxy",
    ) as client:
        resp = await client.post(
            "/chat/completions",
            headers={"X-Kelivo-Conversation-Id": "chat-1"},
            json={
                "model": "gpt-test",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )
    await _wait_summary_tasks(app)

    assert resp.status_code == 200
    assert summary_body["model"] == "deepseek-v4-flash"
    assert summary_headers["authorization"] == "Bearer summary-secret"
    assert "user: hello" in summary_body["messages"][1]["content"]
    assert "assistant: answer" in summary_body["messages"][1]["content"]

    conn = sqlite3.connect(db_path)
    row = conn.execute(
        "SELECT summary, version, status, error_text FROM conversation_summaries"
    ).fetchone()
    history_row = conn.execute(
        """
SELECT version, summary, source_message_count, model_id
FROM conversation_summary_versions
"""
    ).fetchone()
    conn.close()

    assert row == ("updated summary", 1, "completed", None)
    assert history_row == (1, "updated summary", 2, "deepseek-v4-flash")


@pytest.mark.anyio
async def test_proxy_records_summary_error_without_failing_chat(
    tmp_path, upstream_app, monkeypatch
):
    db_path = tmp_path / "chat_search.db"
    _create_base_db(db_path)

    async def completions(_request: Request):
        return JSONResponse(
            {
                "id": "cmpl-test",
                "choices": [{"message": {"role": "assistant", "content": "answer"}}],
            }
        )

    async def summary_completions(_request: Request):
        return JSONResponse({"error": "bad summary"}, status_code=500)

    upstream_app.router.routes.clear()
    upstream_app.post("/chat/completions")(completions)
    upstream_app.post("/summary/chat/completions")(summary_completions)

    transport = ASGITransport(app=upstream_app)
    original_async_client = httpx.AsyncClient

    def client_factory(*args, **kwargs):
        kwargs["transport"] = transport
        kwargs["base_url"] = "http://upstream"
        return original_async_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", client_factory)
    app = create_app(
        ProxyConfig(
            upstream_base="http://upstream",
            db_path=db_path,
            summary_enabled=True,
            summary_upstream_base="http://upstream/summary",
        )
    )

    async with original_async_client(
        transport=ASGITransport(app=app),
        base_url="http://proxy",
    ) as client:
        resp = await client.post(
            "/chat/completions",
            headers={"X-Kelivo-Conversation-Id": "chat-1"},
            json={
                "model": "gpt-test",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )
    await _wait_summary_tasks(app)

    assert resp.status_code == 200
    conn = sqlite3.connect(db_path)
    row = conn.execute(
        "SELECT status, error_text FROM conversation_summaries"
    ).fetchone()
    history_count = conn.execute(
        "SELECT COUNT(*) FROM conversation_summary_versions"
    ).fetchone()[0]
    conn.close()
    assert row[0] == "error"
    assert "summary HTTP 500" in row[1]
    assert history_count == 0


@pytest.mark.anyio
async def test_proxy_updates_summary_after_stream_response(
    tmp_path, upstream_app, monkeypatch
):
    db_path = tmp_path / "chat_search.db"
    _create_base_db(db_path)

    async def stream_completions(_request: Request):
        async def chunks():
            yield b'data: {"choices":[{"delta":{"content":"hel"}}]}\n\n'
            yield b'data: {"choices":[{"delta":{"content":"lo"}}]}\n\n'
            yield b"data: [DONE]\n\n"

        return StreamingResponse(chunks(), media_type="text/event-stream")

    async def summary_completions(_request: Request):
        return JSONResponse(
            {
                "id": "summary-test",
                "choices": [
                    {"message": {"role": "assistant", "content": "stream summary"}}
                ],
            }
        )

    upstream_app.router.routes.clear()
    upstream_app.post("/chat/completions")(stream_completions)
    upstream_app.post("/summary/chat/completions")(summary_completions)

    transport = ASGITransport(app=upstream_app)
    original_async_client = httpx.AsyncClient

    def client_factory(*args, **kwargs):
        kwargs["transport"] = transport
        kwargs["base_url"] = "http://upstream"
        return original_async_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", client_factory)
    app = create_app(
        ProxyConfig(
            upstream_base="http://upstream",
            db_path=db_path,
            summary_enabled=True,
            summary_upstream_base="http://upstream/summary",
        )
    )

    async with original_async_client(
        transport=ASGITransport(app=app),
        base_url="http://proxy",
    ) as client:
        resp = await client.post(
            "/chat/completions",
            headers={"X-Kelivo-Conversation-Id": "chat-stream"},
            json={
                "model": "gpt-test",
                "stream": True,
                "messages": [{"role": "user", "content": "hello stream"}],
            },
        )
    await _wait_summary_tasks(app)

    assert resp.status_code == 200
    conn = sqlite3.connect(db_path)
    summary = conn.execute(
        "SELECT summary, status FROM conversation_summaries"
    ).fetchone()
    assistant = conn.execute(
        "SELECT content FROM messages WHERE role = 'assistant'"
    ).fetchone()[0]
    conn.close()

    assert assistant == "hello"
    assert summary == ("stream summary", "completed")
