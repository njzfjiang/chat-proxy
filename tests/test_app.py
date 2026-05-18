import json
import asyncio
import sqlite3
from datetime import datetime

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
    daily_tasks = list(app.state.daily_summary_tasks)
    if daily_tasks:
        await asyncio.gather(*daily_tasks)


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
                "usage": {
                    "prompt_tokens": 8,
                    "completion_tokens": 5,
                    "total_tokens": 13,
                },
            }
        )

    @app.post("/stream/chat/completions")
    async def stream_completions(_request: Request):
        async def chunks():
            yield b'data: {"choices":[{"delta":{"content":"hel"}}]}\n\n'
            yield b'data: {"choices":[{"delta":{"content":"lo"}}]}\n\n'
            yield b'data: {"usage":{"prompt_tokens":6,"completion_tokens":2,"total_tokens":8}}\n\n'
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


@pytest.mark.anyio
async def test_proxy_updates_daily_summary_without_conversation_summary(
    tmp_path, upstream_app, monkeypatch
):
    db_path = tmp_path / "chat_search.db"
    _create_base_db(db_path)
    daily_body = {}

    async def completions(_request: Request):
        return JSONResponse(
            {
                "id": "cmpl-test",
                "choices": [{"message": {"role": "assistant", "content": "answer"}}],
            }
        )

    async def daily_completions(request: Request):
        daily_body.update(await request.json())
        return JSONResponse(
            {
                "id": "daily-test",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(
                                {
                                    "summary": "Day overview\n- Talked about memory audit.",
                                    "memory_candidates": [
                                        {
                                            "label": "Memory audit first",
                                            "evidence": "The user said candidates should be audited.",
                                            "domain": "rule",
                                            "function": "daily_context",
                                            "primary_mother": "F",
                                            "secondary_mother": "E",
                                            "importance": 3,
                                            "confidence": "high",
                                            "source_message_ids": [1, 2],
                                        }
                                    ],
                                }
                            ),
                        }
                    }
                ],
            }
        )

    upstream_app.router.routes.clear()
    upstream_app.post("/chat/completions")(completions)
    upstream_app.post("/daily/chat/completions")(daily_completions)

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
            daily_summary_enabled=True,
            daily_summary_upstream_base="http://upstream/daily",
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
                "messages": [{"role": "user", "content": "audit memory first"}],
            },
        )
        await _wait_summary_tasks(app)
        admin = await client.get("/admin/daily-summary")

    assert resp.status_code == 200
    assert daily_body["model"] == "deepseek-v4-flash"
    assert "audit memory first" in daily_body["messages"][1]["content"]
    assert "answer" in daily_body["messages"][1]["content"]
    assert admin.status_code == 200
    assert admin.json()["summary"]["status"] == "completed"
    assert admin.json()["memory_candidates"][0]["label"] == "Memory audit first"

    conn = sqlite3.connect(db_path)
    row = conn.execute(
        "SELECT summary, version, status FROM daily_summaries"
    ).fetchone()
    candidate = conn.execute(
        """
SELECT label, domain, function, primary_mother, target_layer, status
FROM daily_memory_candidates
"""
    ).fetchone()
    conversation_count = conn.execute(
        "SELECT COUNT(*) FROM conversation_summaries"
    ).fetchone()[0]
    conn.close()

    assert row == ("Day overview\n- Talked about memory audit.", 1, "completed")
    assert candidate == (
        "Memory audit first",
        "rule",
        "daily_context",
        "F",
        "mem0",
        "candidate",
    )
    assert conversation_count == 0


@pytest.mark.anyio
async def test_manual_daily_summary_run_endpoint_backfills_date(
    tmp_path, upstream_app, monkeypatch
):
    db_path = tmp_path / "chat_search.db"
    _create_base_db(db_path)
    daily_body = {}

    async def daily_completions(request: Request):
        daily_body.update(await request.json())
        return JSONResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "summary": "Day overview\n- Manual run worked.",
                                    "memory_candidates": [],
                                }
                            )
                        }
                    }
                ]
            }
        )

    upstream_app.router.routes.clear()
    upstream_app.post("/chat/completions")(daily_completions)

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
            daily_summary_enabled=True,
            daily_summary_upstream_base="http://upstream",
            daily_summary_timezone="America/Toronto",
        )
    )
    app.state.store.insert_message(
        timestamp="2026-05-15T16:00:00+00:00",
        role="user",
        content="manual daily note",
        conversation_title="kai",
        conversation_id="chat-1",
        message_id="manual-daily-1",
    )

    async with original_async_client(
        transport=ASGITransport(app=app),
        base_url="http://proxy",
    ) as client:
        resp = await client.post(
            "/daily-summaries/run",
            json={"date_key": "2026-05-15", "force": True},
        )

    assert resp.status_code == 200
    assert resp.json()["date_key"] == "2026-05-15"
    assert resp.json()["summary"]["summary"].startswith("Day overview")
    assert "manual daily note" in daily_body["messages"][1]["content"]


@pytest.mark.anyio
async def test_daily_summary_can_be_read_by_days_ago(tmp_path):
    db_path = tmp_path / "chat_search.db"
    _create_base_db(db_path)
    app = create_app(
        ProxyConfig(
            upstream_base="http://upstream",
            db_path=db_path,
            daily_summary_timezone="America/Toronto",
        )
    )
    today = datetime.now().date().isoformat()
    app.state.store.upsert_daily_summary(
        date_key=today,
        summary="Today summary",
        last_message_id=None,
        updated_at="2026-05-17T00:00:00Z",
    )

    async with httpx.AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://proxy",
    ) as client:
        resp = await client.get("/daily-summaries?days_ago=0")

    assert resp.status_code == 200
    assert resp.json()["date_key"] == today
    assert resp.json()["summary"]["summary"] == "Today summary"


@pytest.mark.anyio
async def test_web_chat_endpoint_uses_explicit_identity_and_exposes_history(
    tmp_path, upstream_app, monkeypatch
):
    db_path = tmp_path / "chat_search.db"
    _create_base_db(db_path)
    captured_body = {}
    captured_headers = {}

    async def completions(request: Request):
        captured_body.update(await request.json())
        captured_headers.update(dict(request.headers))
        return JSONResponse(
            {
                "id": "cmpl-test",
                "choices": [
                    {"message": {"role": "assistant", "content": "web answer"}}
                ],
                "usage": {
                    "prompt_tokens": 8,
                    "completion_tokens": 5,
                    "total_tokens": 13,
                },
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
    app = create_app(
        ProxyConfig(
            upstream_base="http://upstream",
            db_path=db_path,
            upstream_api_key="backend-secret",
            chat_model="backend-chat-model",
            provider_key="deepseek",
        )
    )

    async with original_async_client(
        transport=ASGITransport(app=app),
        base_url="http://proxy",
    ) as client:
        resp = await client.post(
            "/chat",
            json={
                "client_id": "phone-1",
                "conversation_id": "mobile-chat",
                "request_id": "req-mobile-1",
                "assistant_key": "kai",
                "system_prompt": "Reply gently.",
                "user_text": "hello mobile",
            },
        )
        conversations = await client.get("/conversations")
        messages = await client.get("/conversations/mobile-chat/messages")
        rolling = await client.get("/conversations/mobile-chat/rolling-short")

    assert resp.status_code == 200
    assert captured_body == {
        "model": "backend-chat-model",
        "messages": [
            {"role": "system", "content": "Reply gently."},
            {"role": "user", "content": "hello mobile"},
        ],
    }
    assert captured_headers["authorization"] == "Bearer backend-secret"
    assert conversations.json()["conversations"][0]["conversation_id"] == "mobile-chat"
    assert conversations.json()["conversations"][0]["client_id"] == "phone-1"
    assert conversations.json()["conversations"][0]["title"] == "hello mobile"
    assert [row["role"] for row in messages.json()["messages"]] == [
        "user",
        "assistant",
    ]
    assert messages.json()["messages"][0]["content"] == "hello mobile"
    assert messages.json()["messages"][1]["token_usage"] == {
        "prompt_tokens": 8,
        "completion_tokens": 5,
        "total_tokens": 13,
    }
    assert rolling.json() == {"conversation_id": "mobile-chat", "summary": None}


@pytest.mark.anyio
async def test_backend_api_key_replaces_basic_auth_from_reverse_proxy(
    tmp_path, upstream_app, monkeypatch
):
    db_path = tmp_path / "chat_search.db"
    _create_base_db(db_path)
    captured_headers = {}

    async def completions(request: Request):
        captured_headers.update(dict(request.headers))
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
    app = create_app(
        ProxyConfig(
            upstream_base="http://upstream",
            db_path=db_path,
            upstream_api_key="backend-secret",
        )
    )

    async with original_async_client(
        transport=ASGITransport(app=app),
        base_url="http://proxy",
    ) as client:
        resp = await client.post(
            "/chat/completions",
            headers={
                "Authorization": "Basic dXNlcjpwYXNz",
                "X-Kelivo-Conversation-Id": "basic-auth-chat",
            },
            json={
                "model": "gpt-test",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

    assert resp.status_code == 200
    assert captured_headers["authorization"] == "Bearer backend-secret"


@pytest.mark.anyio
async def test_web_chat_uses_db_recent_turns_for_context(
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
                "choices": [
                    {"message": {"role": "assistant", "content": "context answer"}}
                ],
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
    app = create_app(
        ProxyConfig(
            upstream_base="http://upstream",
            db_path=db_path,
            chat_model="backend-chat-model",
            chat_recent_k=2,
        )
    )
    app.state.store.insert_message(
        timestamp="2026-05-17T00:00:00Z",
        role="user",
        content="older user",
        conversation_title="kai",
        conversation_id="mobile-chat",
        message_id="old-user",
    )
    app.state.store.insert_message(
        timestamp="2026-05-17T00:00:01Z",
        role="assistant",
        content="recent assistant",
        conversation_title="kai",
        conversation_id="mobile-chat",
        message_id="recent-assistant",
    )

    async with original_async_client(
        transport=ASGITransport(app=app),
        base_url="http://proxy",
    ) as client:
        resp = await client.post(
            "/chat",
            json={
                "client_id": "phone-1",
                "conversation_id": "mobile-chat",
                "request_id": "req-mobile-context",
                "assistant_key": "kai",
                "user_text": "new mobile text",
            },
        )

    assert resp.status_code == 200
    assert captured_body["messages"] == [
        {"role": "user", "content": "older user"},
        {"role": "assistant", "content": "recent assistant"},
        {"role": "user", "content": "new mobile text"},
    ]
    conn = sqlite3.connect(db_path)
    metadata = json.loads(
        conn.execute("SELECT metadata_json FROM requests").fetchone()[0]
    )
    conn.close()
    snapshot = metadata["injected_context_snapshot"]
    assert snapshot["mode"] == "db_recent_turns"
    recent = next(
        component
        for component in snapshot["components"]
        if component["name"] == "recent_turns"
    )
    current_user = next(
        component
        for component in snapshot["components"]
        if component["name"] == "current_user"
    )
    assert recent["message_count"] == 2
    assert current_user["chars"] == len("new mobile text")


@pytest.mark.anyio
async def test_web_chat_injects_matching_worldbook_snippets(
    tmp_path, upstream_app, monkeypatch
):
    db_path = tmp_path / "chat_search.db"
    _create_base_db(db_path)
    worldbook_path = tmp_path / "worldbook.json"
    second_worldbook_path = tmp_path / "second-worldbook.json"
    worldbook_path.write_text(
        json.dumps(
            {
                "data": {
                    "entries": [
                        {
                            "id": "city",
                            "name": "City memory",
                            "enabled": True,
                            "priority": 10,
                            "content": "Hangzhou and spring context.",
                            "keywords": ["杭州"],
                        },
                        {
                            "id": "quiet",
                            "name": "Quiet entry",
                            "enabled": True,
                            "priority": 99,
                            "content": "Should not appear.",
                            "keywords": ["not-triggered"],
                        },
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    second_worldbook_path.write_text(
        json.dumps(
            {
                "data": {
                    "name": "Second book",
                    "entries": [
                        {
                            "id": "long-term",
                            "name": "Long-term memory",
                            "enabled": True,
                            "priority": 50,
                            "content": "Long-term memory context.",
                            "keywords": ["春天"],
                        }
                    ],
                }
            }
        ),
        encoding="utf-8",
    )
    captured_body = {}

    async def completions(request: Request):
        captured_body.update(await request.json())
        return JSONResponse(
            {
                "id": "cmpl-test",
                "choices": [
                    {"message": {"role": "assistant", "content": "worldbook answer"}}
                ],
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
    app = create_app(
        ProxyConfig(
            upstream_base="http://upstream",
            db_path=db_path,
            worldbook_enabled=True,
            worldbook_path=worldbook_path,
            worldbook_paths=(worldbook_path, second_worldbook_path),
        )
    )

    async with original_async_client(
        transport=ASGITransport(app=app),
        base_url="http://proxy",
    ) as client:
        resp = await client.post(
            "/chat",
            json={
                "client_id": "phone-1",
                "conversation_id": "mobile-chat",
                "request_id": "req-worldbook",
                "assistant_key": "kai",
                "user_text": "想聊聊杭州的春天",
            },
        )
        debug = await client.get("/admin/requests?conversation_id=mobile-chat")
        debug_by_request = await client.get("/admin/requests?request_id=req-worldbook")

    assert resp.status_code == 200
    assert captured_body["messages"][0]["role"] == "system"
    assert "Long-term memory" in captured_body["messages"][0]["content"]
    assert "Long-term memory context." in captured_body["messages"][0]["content"]
    snapshot = debug.json()["requests"][0]["metadata"]["injected_context_snapshot"]
    wb = next(component for component in snapshot["components"] if component["name"] == "wb_snippets")
    assert wb["items"][0]["id"] == "long-term"
    assert wb["items"][0]["book_name"] == "Second book"
    assert wb["items"][0]["keyword"] == "春天"
    assert debug_by_request.json()["requests"][0]["request_id"] == "req-worldbook"
    assert (
        debug_by_request.json()["requests"][0]["metadata"]["injected_context_snapshot"]
        == snapshot
    )


@pytest.mark.anyio
async def test_web_chat_replays_completed_explicit_request_id(
    tmp_path, upstream_app, monkeypatch
):
    db_path = tmp_path / "chat_search.db"
    _create_base_db(db_path)
    call_count = 0

    async def completions(_request: Request):
        nonlocal call_count
        call_count += 1
        return JSONResponse(
            {
                "id": "cmpl-test",
                "choices": [
                    {"message": {"role": "assistant", "content": f"answer {call_count}"}}
                ],
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
    payload = {
        "client_id": "phone-1",
        "conversation_id": "mobile-chat",
        "request_id": "req-mobile-1",
        "model": "gpt-test",
        "user_text": "hello mobile",
    }

    async with original_async_client(
        transport=ASGITransport(app=app),
        base_url="http://proxy",
    ) as client:
        first = await client.post("/chat", json=payload)
        second = await client.post("/chat", json=payload)

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json() == second.json()
    assert call_count == 1


@pytest.mark.anyio
async def test_conversation_can_be_renamed_and_archived(tmp_path):
    db_path = tmp_path / "chat_search.db"
    _create_base_db(db_path)
    app = create_app(ProxyConfig(upstream_base="http://upstream", db_path=db_path))

    async with httpx.AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://proxy",
    ) as client:
        created = await client.post(
            "/conversations",
            json={
                "conversation_id": "chat-archive",
                "client_id": "phone-1",
                "assistant_key": "kai",
                "title": "Old title",
            },
        )
        renamed = await client.patch(
            "/conversations/chat-archive",
            json={"title": "New title"},
        )
        archived = await client.patch(
            "/conversations/chat-archive",
            json={"archived": True},
        )
        visible = await client.get("/conversations")
        with_archived = await client.get("/conversations?include_archived=true")

    assert created.status_code == 200
    assert renamed.status_code == 200
    assert renamed.json()["conversation"]["title"] == "New title"
    assert archived.status_code == 200
    assert archived.json()["conversation"]["archived_at"]
    assert visible.json()["conversations"] == []
    assert with_archived.json()["conversations"][0]["conversation_id"] == "chat-archive"


@pytest.mark.anyio
async def test_conversation_can_branch_through_message(tmp_path):
    db_path = tmp_path / "chat_search.db"
    _create_base_db(db_path)
    app = create_app(ProxyConfig(upstream_base="http://upstream", db_path=db_path))
    app.state.store.upsert_conversation(
        conversation_id="source-chat",
        now="2026-05-17T00:00:00Z",
        resolver="webapp",
        client_key="phone-1",
        assistant_key="kai",
        title="Source",
        metadata=None,
    )
    first_id = app.state.store.insert_message(
        timestamp="2026-05-17T00:00:01Z",
        role="user",
        content="first",
        conversation_title="Source",
        conversation_id="source-chat",
        message_id="source-1",
    )
    second_id = app.state.store.insert_message(
        timestamp="2026-05-17T00:00:02Z",
        role="assistant",
        content="second",
        conversation_title="Source",
        conversation_id="source-chat",
        message_id="source-2",
    )
    app.state.store.insert_message(
        timestamp="2026-05-17T00:00:03Z",
        role="user",
        content="third",
        conversation_title="Source",
        conversation_id="source-chat",
        message_id="source-3",
    )

    async with httpx.AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://proxy",
    ) as client:
        branched = await client.post(
            "/conversations/source-chat/branches",
            json={
                "conversation_id": "branch-chat",
                "source_message_id": second_id,
                "title": "Branch",
            },
        )
        branch_messages = await client.get("/conversations/branch-chat/messages")

    assert first_id is not None
    assert branched.status_code == 200
    assert branched.json()["copied_message_count"] == 2
    assert [row["content"] for row in branch_messages.json()["messages"]] == [
        "first",
        "second",
    ]


@pytest.mark.anyio
async def test_conversation_message_can_be_deleted(tmp_path):
    db_path = tmp_path / "chat_search.db"
    _create_base_db(db_path)
    app = create_app(ProxyConfig(upstream_base="http://upstream", db_path=db_path))
    app.state.store.upsert_conversation(
        conversation_id="delete-chat",
        now="2026-05-17T00:00:00Z",
        resolver="webapp",
        client_key="phone-1",
        assistant_key="kai",
        title="Delete",
        metadata=None,
    )
    message_id = app.state.store.insert_message(
        timestamp="2026-05-17T00:00:01Z",
        role="user",
        content="remove this",
        conversation_title="Delete",
        conversation_id="delete-chat",
        message_id="delete-1",
    )

    async with httpx.AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://proxy",
    ) as client:
        deleted = await client.delete(
            f"/conversations/delete-chat/messages/{message_id}"
        )
        messages = await client.get("/conversations/delete-chat/messages")

    assert deleted.status_code == 200
    assert deleted.json()["deleted"] is True
    assert messages.json()["messages"] == []


@pytest.mark.anyio
async def test_rolling_summary_versions_and_rollback(tmp_path):
    db_path = tmp_path / "chat_search.db"
    _create_base_db(db_path)
    app = create_app(ProxyConfig(upstream_base="http://upstream", db_path=db_path))
    app.state.store.upsert_summary(
        conversation_id="summary-chat",
        summary="first summary",
        last_message_id=None,
        updated_at="2026-05-17T00:00:01Z",
    )
    app.state.store.upsert_summary(
        conversation_id="summary-chat",
        summary="second summary",
        last_message_id=None,
        updated_at="2026-05-17T00:00:02Z",
    )

    async with httpx.AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://proxy",
    ) as client:
        versions = await client.get(
            "/conversations/summary-chat/rolling-short/versions"
        )
        rollback = await client.post(
            "/conversations/summary-chat/rolling-short/rollback",
            json={"version": 1},
        )
        rolling = await client.get("/conversations/summary-chat/rolling-short")

    assert versions.status_code == 200
    assert [row["version"] for row in versions.json()["versions"]] == [2, 1]
    assert rollback.status_code == 200
    assert rollback.json()["summary"]["version"] == 3
    assert rolling.json()["summary"]["summary"] == "first summary"
