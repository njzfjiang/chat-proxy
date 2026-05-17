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


def test_storage_upserts_summary_and_reads_recent_messages(tmp_path):
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
        metadata=None,
    )
    store.upsert_conversation(
        conversation_id="chat-2",
        now="2026-05-08T00:00:00Z",
        resolver="header",
        client_key="desktop",
        assistant_key="kai",
        title="kai",
        metadata=None,
    )

    first_id = store.insert_message(
        timestamp="2026-05-08T00:00:01Z",
        role="user",
        content="first",
        conversation_title="kai",
        conversation_id="chat-1",
        message_id="msg-1",
    )
    second_id = store.insert_message(
        timestamp="2026-05-08T00:00:02Z",
        role="assistant",
        content="second",
        conversation_title="kai",
        conversation_id="chat-1",
        message_id="msg-2",
    )
    store.insert_message(
        timestamp="2026-05-08T00:00:03Z",
        role="user",
        content="other chat",
        conversation_title="kai",
        conversation_id="chat-2",
        message_id="msg-3",
    )

    store.upsert_summary(
        conversation_id="chat-1",
        summary="old summary",
        last_message_id=first_id,
        updated_at="2026-05-08T00:00:04Z",
    )
    store.upsert_summary(
        conversation_id="chat-1",
        summary="new summary",
        last_message_id=second_id,
        updated_at="2026-05-08T00:00:05Z",
    )

    summary = store.get_summary("chat-1")
    assert summary["summary"] == "new summary"
    assert summary["version"] == 2
    assert summary["last_message_id"] == second_id
    assert summary["status"] == "completed"

    conn = sqlite3.connect(db_path)
    versions = conn.execute(
        """
SELECT version, summary, last_message_id, previous_last_message_id
FROM conversation_summary_versions
WHERE conversation_id = ?
ORDER BY version
""",
        ["chat-1"],
    ).fetchall()
    conn.close()
    assert versions == [
        (1, "old summary", first_id, None),
        (2, "new summary", second_id, first_id),
    ]

    recent = store.get_recent_messages(conversation_id="chat-1", limit=30)
    assert [row["content"] for row in recent] == ["first", "second"]
    after_first = store.get_recent_messages(
        conversation_id="chat-1",
        limit=30,
        after_message_id=first_id,
    )
    assert [row["content"] for row in after_first] == ["second"]

    rollback = store.rollback_summary(
        conversation_id="chat-1",
        target_version=1,
        now="2026-05-08T00:00:06Z",
    )
    assert rollback["summary"] == "old summary"
    assert rollback["version"] == 3
    versions_after_rollback = store.list_summary_versions(
        conversation_id="chat-1",
        limit=10,
    )
    assert [row["version"] for row in versions_after_rollback] == [3, 2, 1]
    assert "rollback" in versions_after_rollback[0]["metadata_json"]


def test_storage_auto_classifies_inserted_message_kind(tmp_path):
    db_path = tmp_path / "chat_search.db"
    _create_base_db(db_path)

    store = ChatProxyStore(db_path)
    store.initialize()
    store.insert_message(
        timestamp="2026-05-08T00:00:01Z",
        role="assistant",
        content="HttpException: HTTP 404:",
        conversation_title="kai",
        conversation_id="chat-1",
        message_id="msg-noise",
    )
    store.insert_message(
        timestamp="2026-05-08T00:00:02Z",
        role="assistant",
        content="HttpException: HTTP 404:",
        conversation_title="kai",
        conversation_id="chat-1",
        message_id="msg-explicit-chat",
        kind="chat",
    )

    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT message_id, kind FROM messages ORDER BY id"
    ).fetchall()
    conn.close()

    assert rows == [("msg-noise", "noise"), ("msg-explicit-chat", "chat")]


def test_storage_deletes_message_from_messages_and_fts(tmp_path):
    db_path = tmp_path / "chat_search.db"
    _create_base_db(db_path)

    store = ChatProxyStore(db_path)
    store.initialize()
    message_id = store.insert_message(
        timestamp="2026-05-08T00:00:01Z",
        role="user",
        content="delete me needle",
        conversation_title="kai",
        conversation_id="chat-1",
        message_id="msg-delete",
    )

    assert message_id is not None
    assert store.delete_message(conversation_id="chat-1", message_id=message_id)
    assert not store.delete_message(conversation_id="chat-1", message_id=message_id)

    conn = sqlite3.connect(db_path)
    message_count = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
    fts_rows = conn.execute(
        "SELECT rowid FROM messages_fts WHERE messages_fts MATCH ?",
        ('"delete me needle"',),
    ).fetchall()
    conn.close()

    assert message_count == 0
    assert fts_rows == []


def test_storage_upserts_daily_summary_and_candidates(tmp_path):
    db_path = tmp_path / "chat_search.db"
    _create_base_db(db_path)

    store = ChatProxyStore(db_path)
    store.initialize()
    first_id = store.insert_message(
        timestamp="2026-05-16T12:00:00+00:00",
        role="user",
        content="first daily note",
        conversation_title="kai",
        conversation_id="chat-1",
        message_id="msg-1",
    )
    second_id = store.insert_message(
        timestamp="2026-05-16T12:01:00+00:00",
        role="assistant",
        content="second daily note",
        conversation_title="kai",
        conversation_id="chat-1",
        message_id="msg-2",
    )

    version = store.upsert_daily_summary(
        date_key="2026-05-16",
        summary="Day overview\n- Built daily summaries.",
        last_message_id=second_id,
        updated_at="2026-05-16T12:02:00+00:00",
        candidates=[
            {
                "label": "Daily summary module",
                "evidence": "Implemented auditable candidates.",
                "domain": "infra_asset",
                "function": "infra_reference",
                "primary_mother": "D",
                "secondary_mother": "E",
                "importance": 3,
                "confidence": "medium",
                "source_message_ids": [first_id, second_id],
            }
        ],
        source_message_count=2,
        model_id="summary-model",
    )

    summary = store.get_daily_summary("2026-05-16")
    candidates = store.get_daily_memory_candidates(
        date_key="2026-05-16",
        summary_version=version,
    )
    after_first = store.get_messages_after(limit=30, after_message_id=first_id)

    assert summary["summary"].startswith("Day overview")
    assert summary["version"] == 1
    assert summary["last_message_id"] == second_id
    assert candidates[0]["label"] == "Daily summary module"
    assert candidates[0]["domain"] == "infra_asset"
    assert candidates[0]["target_layer"] == "wb"
    assert candidates[0]["status"] == "candidate"
    assert candidates[0]["source_message_ids_json"] == f"[{first_id}, {second_id}]"
    assert [row["content"] for row in after_first] == ["second daily note"]


def test_storage_deletes_daily_summary_history_and_candidates(tmp_path):
    db_path = tmp_path / "chat_search.db"
    _create_base_db(db_path)

    store = ChatProxyStore(db_path)
    store.initialize()
    message_id = store.insert_message(
        timestamp="2026-05-16T12:00:00+00:00",
        role="user",
        content="first daily note",
        conversation_title="kai",
        conversation_id="chat-1",
        message_id="msg-1",
    )
    store.upsert_daily_summary(
        date_key="2026-05-16",
        summary="Day overview\n- Built daily summaries.",
        last_message_id=message_id,
        updated_at="2026-05-16T12:02:00+00:00",
        candidates=[
            {
                "label": "Daily summary module",
                "evidence": "Implemented auditable candidates.",
                "domain": "infra_asset",
                "function": "infra_reference",
                "primary_mother": "D",
            }
        ],
    )

    store.delete_daily_summary("2026-05-16")

    conn = sqlite3.connect(db_path)
    summary_count = conn.execute("SELECT COUNT(*) FROM daily_summaries").fetchone()[0]
    version_count = conn.execute(
        "SELECT COUNT(*) FROM daily_summary_versions"
    ).fetchone()[0]
    candidate_count = conn.execute(
        "SELECT COUNT(*) FROM daily_memory_candidates"
    ).fetchone()[0]
    message_count = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
    conn.close()

    assert summary_count == 0
    assert version_count == 0
    assert candidate_count == 0
    assert message_count == 1


def test_storage_migrates_daily_memory_candidates_target_layer(tmp_path):
    db_path = tmp_path / "chat_search.db"
    _create_base_db(db_path)

    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
CREATE TABLE daily_memory_candidates (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  date_key TEXT NOT NULL,
  summary_version INTEGER NOT NULL,
  label TEXT NOT NULL,
  evidence TEXT,
  domain TEXT NOT NULL,
  function TEXT NOT NULL,
  primary_mother TEXT NOT NULL,
  secondary_mother TEXT,
  importance INTEGER,
  confidence TEXT,
  source_message_ids_json TEXT,
  status TEXT NOT NULL DEFAULT 'candidate',
  metadata_json TEXT,
  created_at TEXT NOT NULL
);
"""
    )
    conn.close()

    store = ChatProxyStore(db_path)
    store.initialize()

    conn = sqlite3.connect(db_path)
    columns = {
        row[1]
        for row in conn.execute("PRAGMA table_info(daily_memory_candidates)")
    }
    indexes = {
        row[1]
        for row in conn.execute("PRAGMA index_list(daily_memory_candidates)")
    }
    conn.close()

    assert "target_layer" in columns
    assert "idx_daily_memory_candidates_target_layer" in indexes
