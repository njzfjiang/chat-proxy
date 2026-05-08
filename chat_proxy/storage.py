from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Mapping


class ChatProxyStore:
    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)

    def connect(self) -> sqlite3.Connection:
        if not self.db_path.exists():
            raise RuntimeError(f"SQLite DB not found: {self.db_path}")
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def initialize(self) -> None:
        with self.connect() as conn:
            conn.executescript(
                """
CREATE TABLE IF NOT EXISTS conversations (
  conversation_id TEXT PRIMARY KEY,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  resolver TEXT NOT NULL,
  client_key TEXT,
  assistant_key TEXT,
  title TEXT,
  metadata_json TEXT
);

CREATE TABLE IF NOT EXISTS requests (
  request_id TEXT PRIMARY KEY,
  conversation_id TEXT NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  provider_key TEXT,
  model_id TEXT,
  status TEXT NOT NULL,
  http_status INTEGER,
  request_headers_json TEXT,
  response_headers_json TEXT,
  request_json TEXT NOT NULL,
  response_json TEXT,
  metadata_json TEXT,
  error_text TEXT,
  FOREIGN KEY(conversation_id) REFERENCES conversations(conversation_id)
);

CREATE INDEX IF NOT EXISTS idx_requests_conversation_created
  ON requests(conversation_id, created_at);
CREATE INDEX IF NOT EXISTS idx_requests_status ON requests(status);
CREATE INDEX IF NOT EXISTS idx_conversations_updated_at
  ON conversations(updated_at);

CREATE INDEX IF NOT EXISTS messages_timestamp_idx ON messages(timestamp);
CREATE INDEX IF NOT EXISTS messages_role_idx ON messages(role);
CREATE INDEX IF NOT EXISTS messages_conversation_idx ON messages(conversation_id);
CREATE INDEX IF NOT EXISTS messages_kind_idx ON messages(kind);

CREATE TRIGGER IF NOT EXISTS messages_ai AFTER INSERT ON messages BEGIN
  INSERT INTO messages_fts(rowid, content, conversation_title)
  VALUES (new.id, new.content, new.conversation_title);
END;

CREATE TRIGGER IF NOT EXISTS messages_ad AFTER DELETE ON messages BEGIN
  INSERT INTO messages_fts(messages_fts, rowid, content, conversation_title)
  VALUES('delete', old.id, old.content, old.conversation_title);
END;

CREATE TRIGGER IF NOT EXISTS messages_au AFTER UPDATE ON messages BEGIN
  INSERT INTO messages_fts(messages_fts, rowid, content, conversation_title)
  VALUES('delete', old.id, old.content, old.conversation_title);
  INSERT INTO messages_fts(rowid, content, conversation_title)
  VALUES (new.id, new.content, new.conversation_title);
END;
"""
            )

    def upsert_conversation(
        self,
        *,
        conversation_id: str,
        now: str,
        resolver: str,
        client_key: str | None,
        assistant_key: str | None,
        title: str | None,
        metadata: Mapping[str, Any] | None,
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                """
INSERT INTO conversations(
  conversation_id, created_at, updated_at, resolver,
  client_key, assistant_key, title, metadata_json
)
VALUES(?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(conversation_id) DO UPDATE SET
  updated_at = excluded.updated_at,
  resolver = excluded.resolver,
  client_key = COALESCE(excluded.client_key, conversations.client_key),
  assistant_key = COALESCE(excluded.assistant_key, conversations.assistant_key),
  title = COALESCE(excluded.title, conversations.title),
  metadata_json = COALESCE(excluded.metadata_json, conversations.metadata_json)
""",
                [
                    conversation_id,
                    now,
                    now,
                    resolver,
                    client_key,
                    assistant_key,
                    title,
                    _json(metadata),
                ],
            )

    def insert_request_pending(
        self,
        *,
        request_id: str,
        conversation_id: str,
        now: str,
        provider_key: str | None,
        model_id: str | None,
        request_headers: Mapping[str, Any],
        request_json: Mapping[str, Any],
        metadata: Mapping[str, Any] | None,
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                """
INSERT OR REPLACE INTO requests(
  request_id, conversation_id, created_at, updated_at,
  provider_key, model_id, status, request_headers_json,
  request_json, metadata_json
)
VALUES(?, ?, ?, ?, ?, ?, 'pending', ?, ?, ?)
""",
                [
                    request_id,
                    conversation_id,
                    now,
                    now,
                    provider_key,
                    model_id,
                    _json(request_headers),
                    _json(request_json),
                    _json(metadata),
                ],
            )

    def complete_request(
        self,
        *,
        request_id: str,
        now: str,
        status: str,
        http_status: int | None,
        response_headers: Mapping[str, Any] | None = None,
        response_json: Any = None,
        error_text: str | None = None,
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                """
UPDATE requests SET
  updated_at = ?,
  status = ?,
  http_status = COALESCE(?, http_status),
  response_headers_json = COALESCE(?, response_headers_json),
  response_json = COALESCE(?, response_json),
  error_text = COALESCE(?, error_text)
WHERE request_id = ?
""",
                [
                    now,
                    status,
                    http_status,
                    _json(response_headers),
                    _json(response_json),
                    error_text,
                    request_id,
                ],
            )

    def insert_message(
        self,
        *,
        timestamp: str,
        role: str,
        content: str,
        conversation_title: str | None,
        conversation_id: str,
        message_id: str,
        kind: str = "chat",
    ) -> None:
        if not content.strip():
            return
        with self.connect() as conn:
            conn.execute(
                """
INSERT OR IGNORE INTO messages(
  timestamp, role, content, conversation_title,
  conversation_id, message_id, kind
)
VALUES(?, ?, ?, ?, ?, ?, ?)
""",
                [
                    timestamp,
                    role,
                    content,
                    conversation_title,
                    conversation_id,
                    message_id,
                    kind,
                ],
            )


def _json(value: Any) -> str | None:
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False, sort_keys=True)
