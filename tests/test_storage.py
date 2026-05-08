import sqlite3

from chat_proxy.storage import ChatProxyStore


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


def test_storage_initializes_side_tables_and_fts_triggers(tmp_path):
    db_path = tmp_path / "chat_search.db"
    _create_base_db(db_path)

    store = ChatProxyStore(db_path)
    store.initialize()
    store.upsert_conversation(
        conversation_id="chat-1",
        now="2026-05-08T00:00:00Z",
        resolver="header",
        client_key="desktop",
        assistant_key="kai",
        title="kai",
        metadata={"conversation_id": "chat-1"},
    )
    store.insert_request_pending(
        request_id="req-1",
        conversation_id="chat-1",
        now="2026-05-08T00:00:01Z",
        provider_key=None,
        model_id="gpt-test",
        request_headers={"authorization": "[REDACTED]"},
        request_json={"messages": []},
        metadata={"ok": True},
    )
    store.complete_request(
        request_id="req-1",
        now="2026-05-08T00:00:02Z",
        status="completed",
        http_status=200,
        response_json={"ok": True},
    )
    store.insert_message(
        timestamp="2026-05-08T00:00:03Z",
        role="user",
        content="needle text",
        conversation_title="kai",
        conversation_id="chat-1",
        message_id="msg-1",
    )
    store.insert_message(
        timestamp="2026-05-08T00:00:03Z",
        role="user",
        content="needle text",
        conversation_title="kai",
        conversation_id="chat-1",
        message_id="msg-1",
    )

    conn = sqlite3.connect(db_path)
    assert conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0] == 1
    assert conn.execute("SELECT status FROM requests").fetchone()[0] == "completed"
    assert conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 1
    rows = conn.execute(
        "SELECT rowid FROM messages_fts WHERE messages_fts MATCH ?",
        ('"needle text"',),
    ).fetchall()
    conn.close()
    assert len(rows) == 1
