from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Mapping

from .message_kind import classify_message_kind, normalize_message_kind
from .memory_target import choose_target_layer, normalize_target_layer


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
  archived_at TEXT,
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

CREATE TABLE IF NOT EXISTS conversation_summaries (
  conversation_id TEXT PRIMARY KEY,
  summary TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  version INTEGER NOT NULL DEFAULT 0,
  last_message_id INTEGER,
  status TEXT,
  error_text TEXT,
  FOREIGN KEY(conversation_id) REFERENCES conversations(conversation_id)
);

CREATE TABLE IF NOT EXISTS conversation_summary_versions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  conversation_id TEXT NOT NULL,
  version INTEGER NOT NULL,
  summary TEXT NOT NULL,
  created_at TEXT NOT NULL,
  last_message_id INTEGER,
  previous_last_message_id INTEGER,
  source_message_count INTEGER,
  model_id TEXT,
  metadata_json TEXT,
  FOREIGN KEY(conversation_id) REFERENCES conversations(conversation_id),
  UNIQUE(conversation_id, version)
);

CREATE TABLE IF NOT EXISTS daily_summaries (
  date_key TEXT PRIMARY KEY,
  summary TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  version INTEGER NOT NULL DEFAULT 0,
  last_message_id INTEGER,
  status TEXT,
  error_text TEXT
);

CREATE TABLE IF NOT EXISTS daily_summary_versions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  date_key TEXT NOT NULL,
  version INTEGER NOT NULL,
  summary TEXT NOT NULL,
  created_at TEXT NOT NULL,
  last_message_id INTEGER,
  previous_last_message_id INTEGER,
  source_message_count INTEGER,
  model_id TEXT,
  metadata_json TEXT,
  UNIQUE(date_key, version)
);

CREATE TABLE IF NOT EXISTS daily_memory_candidates (
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
  target_layer TEXT,
  source_message_ids_json TEXT,
  status TEXT NOT NULL DEFAULT 'candidate',
  metadata_json TEXT,
  created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_requests_conversation_created
  ON requests(conversation_id, created_at);
CREATE INDEX IF NOT EXISTS idx_requests_status ON requests(status);
CREATE INDEX IF NOT EXISTS idx_conversations_updated_at
  ON conversations(updated_at);
CREATE INDEX IF NOT EXISTS idx_conversation_summaries_updated_at
  ON conversation_summaries(updated_at);
CREATE INDEX IF NOT EXISTS idx_conversation_summary_versions_conversation_version
  ON conversation_summary_versions(conversation_id, version);
CREATE INDEX IF NOT EXISTS idx_conversation_summary_versions_conversation_created
  ON conversation_summary_versions(conversation_id, created_at);
CREATE INDEX IF NOT EXISTS idx_conversation_summary_versions_last_message
  ON conversation_summary_versions(last_message_id);
CREATE INDEX IF NOT EXISTS idx_daily_summaries_updated_at
  ON daily_summaries(updated_at);
CREATE INDEX IF NOT EXISTS idx_daily_summary_versions_date_version
  ON daily_summary_versions(date_key, version);
CREATE INDEX IF NOT EXISTS idx_daily_memory_candidates_date_version
  ON daily_memory_candidates(date_key, summary_version);
CREATE INDEX IF NOT EXISTS idx_daily_memory_candidates_status
  ON daily_memory_candidates(status);

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
            _ensure_column(
                conn,
                table_name="conversations",
                column_name="archived_at",
                definition="archived_at TEXT",
            )
            _ensure_column(
                conn,
                table_name="daily_memory_candidates",
                column_name="target_layer",
                definition="target_layer TEXT",
            )
            _ensure_column(
                conn,
                table_name="messages",
                column_name="request_id",
                definition="request_id TEXT",
            )
            _ensure_column(
                conn,
                table_name="messages",
                column_name="token_usage_json",
                definition="token_usage_json TEXT",
            )
            conn.execute(
                """
CREATE INDEX IF NOT EXISTS idx_daily_memory_candidates_target_layer
  ON daily_memory_candidates(target_layer)
"""
            )
            conn.execute(
                """
CREATE INDEX IF NOT EXISTS messages_request_idx ON messages(request_id)
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
  title = CASE
    WHEN excluded.title IS NULL OR excluded.title = '' THEN conversations.title
    WHEN conversations.title IS NULL OR conversations.title = ''
      THEN excluded.title
    WHEN conversations.title = conversations.assistant_key
      THEN excluded.title
    ELSE conversations.title
  END,
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

    def list_conversations(
        self,
        *,
        limit: int = 50,
        include_archived: bool = False,
    ) -> list[dict[str, Any]]:
        safe_limit = max(1, min(limit, 200))
        where = "" if include_archived else "WHERE c.archived_at IS NULL"
        with self.connect() as conn:
            rows = conn.execute(
                f"""
SELECT
  c.conversation_id,
  c.created_at,
  c.updated_at,
  c.resolver,
  c.client_key,
  c.assistant_key,
  c.title,
  c.archived_at,
  c.metadata_json,
  COUNT(m.id) AS message_count,
  MAX(m.id) AS last_message_id,
  MAX(m.timestamp) AS last_message_at,
  (
    SELECT m2.role
    FROM messages m2
    WHERE m2.conversation_id = c.conversation_id
    ORDER BY m2.id DESC
    LIMIT 1
  ) AS last_message_role,
  (
    SELECT m2.content
    FROM messages m2
    WHERE m2.conversation_id = c.conversation_id
    ORDER BY m2.id DESC
    LIMIT 1
  ) AS last_message_content,
  s.summary AS rolling_summary,
  s.status AS rolling_summary_status,
  s.version AS rolling_summary_version
FROM conversations c
LEFT JOIN messages m ON m.conversation_id = c.conversation_id
LEFT JOIN conversation_summaries s ON s.conversation_id = c.conversation_id
{where}
GROUP BY c.conversation_id
ORDER BY COALESCE(MAX(m.timestamp), c.updated_at) DESC, c.updated_at DESC
LIMIT ?
""",
                [safe_limit],
            ).fetchall()
        return [dict(row) for row in rows]

    def update_conversation(
        self,
        *,
        conversation_id: str,
        now: str,
        title: str | None = None,
        archived: bool | None = None,
    ) -> dict[str, Any] | None:
        assignments = ["updated_at = ?"]
        params: list[Any] = [now]
        if title is not None:
            assignments.append("title = ?")
            params.append(title)
        if archived is not None:
            assignments.append("archived_at = ?")
            params.append(now if archived else None)
        params.append(conversation_id)
        with self.connect() as conn:
            conn.execute(
                f"""
UPDATE conversations
SET {', '.join(assignments)}
WHERE conversation_id = ?
""",
                params,
            )
            row = conn.execute(
                """
SELECT conversation_id, created_at, updated_at, resolver,
       client_key, assistant_key, title, archived_at, metadata_json
FROM conversations
WHERE conversation_id = ?
""",
                [conversation_id],
            ).fetchone()
        return dict(row) if row else None

    def touch_conversation(self, *, conversation_id: str, now: str) -> None:
        with self.connect() as conn:
            conn.execute(
                """
UPDATE conversations
SET updated_at = ?
WHERE conversation_id = ?
""",
                [now, conversation_id],
            )

    def get_conversation(self, conversation_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(
                """
SELECT conversation_id, created_at, updated_at, resolver,
       client_key, assistant_key, title, archived_at, metadata_json
FROM conversations
WHERE conversation_id = ?
""",
                [conversation_id],
            ).fetchone()
        return dict(row) if row else None

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

    def get_request(self, request_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(
                """
SELECT request_id, conversation_id, created_at, updated_at,
       provider_key, model_id, status, http_status,
       request_headers_json, response_headers_json, request_json,
       response_json, metadata_json, error_text
FROM requests
WHERE request_id = ?
""",
                [request_id],
            ).fetchone()
        return dict(row) if row else None

    def list_requests(
        self,
        *,
        conversation_id: str | None = None,
        request_id: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        safe_limit = max(1, min(limit, 100))
        clauses: list[str] = []
        params: list[Any] = []
        if conversation_id:
            clauses.append("conversation_id = ?")
            params.append(conversation_id)
        if request_id:
            clauses.append("request_id = ?")
            params.append(request_id)
        params.append(safe_limit)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self.connect() as conn:
            rows = conn.execute(
                f"""
SELECT request_id, conversation_id, created_at, updated_at,
       provider_key, model_id, status, http_status,
       request_json, response_json, metadata_json, error_text
FROM requests
{where}
ORDER BY created_at DESC
LIMIT ?
""",
                params,
            ).fetchall()
        return [dict(row) for row in rows]

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
        kind: str | None = None,
        request_id: str | None = None,
        token_usage: Mapping[str, Any] | None = None,
    ) -> int | None:
        if not content.strip():
            return None
        normalized_kind = normalize_message_kind(kind) or classify_message_kind(
            content=content,
            conversation_title=conversation_title,
        )
        with self.connect() as conn:
            conn.execute(
                """
INSERT OR IGNORE INTO messages(
  timestamp, role, content, conversation_title,
  conversation_id, message_id, kind, request_id, token_usage_json
)
VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
""",
                [
                    timestamp,
                    role,
                    content,
                    conversation_title,
                    conversation_id,
                    message_id,
                    normalized_kind,
                    request_id,
                    _json(token_usage),
                ],
            )
            row = conn.execute(
                "SELECT id FROM messages WHERE message_id = ?",
                [message_id],
            ).fetchone()
            return int(row["id"]) if row else None

    def get_summary(self, conversation_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(
                """
SELECT conversation_id, summary, updated_at, version,
       last_message_id, status, error_text
FROM conversation_summaries
WHERE conversation_id = ?
""",
                [conversation_id],
            ).fetchone()
            return dict(row) if row else None

    def mark_summary_pending(self, *, conversation_id: str, now: str) -> None:
        with self.connect() as conn:
            conn.execute(
                """
INSERT INTO conversation_summaries(
  conversation_id, summary, updated_at, version,
  last_message_id, status, error_text
)
VALUES(?, '', ?, 0, NULL, 'pending', NULL)
ON CONFLICT(conversation_id) DO UPDATE SET
  updated_at = excluded.updated_at,
  status = 'pending',
  error_text = NULL
""",
                [conversation_id, now],
            )

    def mark_summary_error(
        self,
        *,
        conversation_id: str,
        now: str,
        error_text: str,
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                """
INSERT INTO conversation_summaries(
  conversation_id, summary, updated_at, version,
  last_message_id, status, error_text
)
VALUES(?, '', ?, 0, NULL, 'error', ?)
ON CONFLICT(conversation_id) DO UPDATE SET
  updated_at = excluded.updated_at,
  status = 'error',
  error_text = excluded.error_text
""",
                [conversation_id, now, error_text],
            )

    def upsert_summary(
        self,
        *,
        conversation_id: str,
        summary: str,
        last_message_id: int | None,
        updated_at: str,
        source_message_count: int | None = None,
        model_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        with self.connect() as conn:
            existing = conn.execute(
                """
SELECT version, last_message_id
FROM conversation_summaries
WHERE conversation_id = ?
""",
                [conversation_id],
            ).fetchone()
            next_version = int(existing["version"]) + 1 if existing else 1
            previous_last_message_id = (
                existing["last_message_id"] if existing else None
            )
            conn.execute(
                """
INSERT INTO conversation_summaries(
  conversation_id, summary, updated_at, version,
  last_message_id, status, error_text
)
VALUES(?, ?, ?, ?, ?, 'completed', NULL)
ON CONFLICT(conversation_id) DO UPDATE SET
  summary = excluded.summary,
  updated_at = excluded.updated_at,
  version = excluded.version,
  last_message_id = excluded.last_message_id,
  status = 'completed',
  error_text = NULL
""",
                [
                    conversation_id,
                    summary,
                    updated_at,
                    next_version,
                    last_message_id,
                ],
            )
            conn.execute(
                """
INSERT INTO conversation_summary_versions(
  conversation_id, version, summary, created_at,
  last_message_id, previous_last_message_id,
  source_message_count, model_id, metadata_json
)
VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
""",
                [
                    conversation_id,
                    next_version,
                    summary,
                    updated_at,
                    last_message_id,
                    previous_last_message_id,
                    source_message_count,
                    model_id,
                    _json(metadata),
                ],
            )

    def list_summary_versions(
        self,
        *,
        conversation_id: str,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        safe_limit = max(1, min(limit, 100))
        with self.connect() as conn:
            rows = conn.execute(
                """
SELECT id, conversation_id, version, summary, created_at,
       last_message_id, previous_last_message_id,
       source_message_count, model_id, metadata_json
FROM conversation_summary_versions
WHERE conversation_id = ?
ORDER BY version DESC
LIMIT ?
""",
                [conversation_id, safe_limit],
            ).fetchall()
        return [dict(row) for row in rows]

    def rollback_summary(
        self,
        *,
        conversation_id: str,
        target_version: int | None,
        now: str,
    ) -> dict[str, Any] | None:
        with self.connect() as conn:
            current = conn.execute(
                """
SELECT version, last_message_id
FROM conversation_summaries
WHERE conversation_id = ?
""",
                [conversation_id],
            ).fetchone()
            if current is None:
                return None
            current_version = int(current["version"])
            if target_version is None:
                target_version = current_version - 1
            target = conn.execute(
                """
SELECT version, summary, last_message_id, source_message_count, model_id
FROM conversation_summary_versions
WHERE conversation_id = ? AND version = ?
""",
                [conversation_id, target_version],
            ).fetchone()
            if target is None:
                return None
            next_version = current_version + 1
            conn.execute(
                """
UPDATE conversation_summaries
SET summary = ?, updated_at = ?, version = ?,
    last_message_id = ?, status = 'completed', error_text = NULL
WHERE conversation_id = ?
""",
                [
                    target["summary"],
                    now,
                    next_version,
                    target["last_message_id"],
                    conversation_id,
                ],
            )
            conn.execute(
                """
INSERT INTO conversation_summary_versions(
  conversation_id, version, summary, created_at,
  last_message_id, previous_last_message_id,
  source_message_count, model_id, metadata_json
)
VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
""",
                [
                    conversation_id,
                    next_version,
                    target["summary"],
                    now,
                    target["last_message_id"],
                    current["last_message_id"],
                    target["source_message_count"],
                    target["model_id"],
                    _json(
                        {
                            "rollback": True,
                            "rollback_from_version": current_version,
                            "rollback_to_version": target_version,
                        }
                    ),
                ],
            )
            row = conn.execute(
                """
SELECT conversation_id, summary, updated_at, version,
       last_message_id, status, error_text
FROM conversation_summaries
WHERE conversation_id = ?
""",
                [conversation_id],
            ).fetchone()
        return dict(row) if row else None

    def get_recent_messages(
        self,
        *,
        conversation_id: str,
        limit: int = 30,
        after_message_id: int | None = None,
    ) -> list[dict[str, Any]]:
        clauses = ["conversation_id = ?"]
        params: list[Any] = [conversation_id]
        if after_message_id is not None:
            clauses.append("id > ?")
            params.append(after_message_id)
        params.append(limit)
        with self.connect() as conn:
            rows = conn.execute(
                f"""
SELECT id, timestamp, role, content, conversation_title, kind,
       request_id, token_usage_json
FROM messages
WHERE {' AND '.join(clauses)}
ORDER BY id DESC
LIMIT ?
""",
                params,
            ).fetchall()
        return [dict(row) for row in reversed(rows)]

    def get_conversation_messages(
        self,
        *,
        conversation_id: str,
        limit: int = 50,
        before_id: int | None = None,
        after_id: int | None = None,
        kind: str | None = None,
    ) -> list[dict[str, Any]]:
        safe_limit = max(1, min(limit, 500))
        clauses = ["conversation_id = ?"]
        params: list[Any] = [conversation_id]
        if kind:
            clauses.append("kind = ?")
            params.append(kind)
        if after_id is not None:
            clauses.append("id > ?")
            params.append(after_id)
            order = "ASC"
            reverse = False
        else:
            if before_id is not None:
                clauses.append("id < ?")
                params.append(before_id)
            order = "DESC"
            reverse = True
        params.append(safe_limit)
        with self.connect() as conn:
            rows = conn.execute(
                f"""
SELECT id, timestamp, role, content, conversation_title,
       conversation_id, message_id, kind, request_id, token_usage_json
FROM messages
WHERE {' AND '.join(clauses)}
ORDER BY id {order}
LIMIT ?
""",
                params,
            ).fetchall()
        result = [dict(row) for row in rows]
        return list(reversed(result)) if reverse else result

    def delete_message(
        self,
        *,
        conversation_id: str,
        message_id: int,
    ) -> bool:
        with self.connect() as conn:
            cursor = conn.execute(
                """
DELETE FROM messages
WHERE conversation_id = ? AND id = ?
""",
                [conversation_id, message_id],
            )
            return cursor.rowcount > 0

    def get_conversation_messages_through(
        self,
        *,
        conversation_id: str,
        through_id: int,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        safe_limit = max(1, min(limit, 1000))
        with self.connect() as conn:
            rows = conn.execute(
                """
SELECT id, timestamp, role, content, conversation_title,
       conversation_id, message_id, kind, request_id, token_usage_json
FROM messages
WHERE conversation_id = ? AND id <= ?
ORDER BY id DESC
LIMIT ?
""",
                [conversation_id, through_id, safe_limit],
            ).fetchall()
        return [dict(row) for row in reversed(rows)]

    def get_messages_after(
        self,
        *,
        limit: int = 200,
        after_message_id: int | None = None,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if after_message_id is not None:
            clauses.append("id > ?")
            params.append(after_message_id)
        params.append(limit)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self.connect() as conn:
            rows = conn.execute(
                f"""
SELECT id, timestamp, role, content, conversation_title, conversation_id, kind,
       request_id, token_usage_json
FROM messages
{where}
ORDER BY id ASC
LIMIT ?
""",
                params,
            ).fetchall()
        return [dict(row) for row in rows]

    def get_daily_summary(self, date_key: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(
                """
SELECT date_key, summary, updated_at, version,
       last_message_id, status, error_text
FROM daily_summaries
WHERE date_key = ?
""",
                [date_key],
            ).fetchone()
            return dict(row) if row else None

    def list_daily_summaries(self, *, limit: int = 30) -> list[dict[str, Any]]:
        safe_limit = max(1, min(limit, 365))
        with self.connect() as conn:
            rows = conn.execute(
                """
SELECT date_key, summary, updated_at, version,
       last_message_id, status, error_text
FROM daily_summaries
ORDER BY date_key DESC
LIMIT ?
""",
                [safe_limit],
            ).fetchall()
        return [dict(row) for row in rows]

    def get_daily_memory_candidates(
        self,
        *,
        date_key: str,
        summary_version: int | None = None,
    ) -> list[dict[str, Any]]:
        clauses = ["date_key = ?"]
        params: list[Any] = [date_key]
        if summary_version is not None:
            clauses.append("summary_version = ?")
            params.append(summary_version)
        with self.connect() as conn:
            rows = conn.execute(
                f"""
SELECT id, date_key, summary_version, label, evidence, domain,
       function, primary_mother, secondary_mother, importance,
       confidence, target_layer, source_message_ids_json, status, metadata_json,
       created_at
FROM daily_memory_candidates
WHERE {' AND '.join(clauses)}
ORDER BY id
""",
                params,
            ).fetchall()
        return [dict(row) for row in rows]

    def delete_daily_summary(self, date_key: str) -> None:
        with self.connect() as conn:
            conn.execute(
                "DELETE FROM daily_memory_candidates WHERE date_key = ?",
                [date_key],
            )
            conn.execute(
                "DELETE FROM daily_summary_versions WHERE date_key = ?",
                [date_key],
            )
            conn.execute(
                "DELETE FROM daily_summaries WHERE date_key = ?",
                [date_key],
            )

    def mark_daily_summary_pending(self, *, date_key: str, now: str) -> None:
        with self.connect() as conn:
            conn.execute(
                """
INSERT INTO daily_summaries(
  date_key, summary, updated_at, version,
  last_message_id, status, error_text
)
VALUES(?, '', ?, 0, NULL, 'pending', NULL)
ON CONFLICT(date_key) DO UPDATE SET
  updated_at = excluded.updated_at,
  status = 'pending',
  error_text = NULL
""",
                [date_key, now],
            )

    def mark_daily_summary_error(
        self,
        *,
        date_key: str,
        now: str,
        error_text: str,
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                """
INSERT INTO daily_summaries(
  date_key, summary, updated_at, version,
  last_message_id, status, error_text
)
VALUES(?, '', ?, 0, NULL, 'error', ?)
ON CONFLICT(date_key) DO UPDATE SET
  updated_at = excluded.updated_at,
  status = 'error',
  error_text = excluded.error_text
""",
                [date_key, now, error_text],
            )

    def upsert_daily_summary(
        self,
        *,
        date_key: str,
        summary: str,
        last_message_id: int | None,
        updated_at: str,
        candidates: list[Mapping[str, Any]] | None = None,
        source_message_count: int | None = None,
        model_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> int:
        with self.connect() as conn:
            existing = conn.execute(
                """
SELECT version, last_message_id
FROM daily_summaries
WHERE date_key = ?
""",
                [date_key],
            ).fetchone()
            next_version = int(existing["version"]) + 1 if existing else 1
            previous_last_message_id = (
                existing["last_message_id"] if existing else None
            )
            conn.execute(
                """
INSERT INTO daily_summaries(
  date_key, summary, updated_at, version,
  last_message_id, status, error_text
)
VALUES(?, ?, ?, ?, ?, 'completed', NULL)
ON CONFLICT(date_key) DO UPDATE SET
  summary = excluded.summary,
  updated_at = excluded.updated_at,
  version = excluded.version,
  last_message_id = excluded.last_message_id,
  status = 'completed',
  error_text = NULL
""",
                [date_key, summary, updated_at, next_version, last_message_id],
            )
            conn.execute(
                """
INSERT INTO daily_summary_versions(
  date_key, version, summary, created_at,
  last_message_id, previous_last_message_id,
  source_message_count, model_id, metadata_json
)
VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
""",
                [
                    date_key,
                    next_version,
                    summary,
                    updated_at,
                    last_message_id,
                    previous_last_message_id,
                    source_message_count,
                    model_id,
                    _json(metadata),
                ],
            )
            for candidate in candidates or []:
                target_layer = normalize_target_layer(
                    candidate.get("target_layer")
                ) or choose_target_layer(candidate)
                conn.execute(
                    """
INSERT INTO daily_memory_candidates(
  date_key, summary_version, label, evidence, domain,
  function, primary_mother, secondary_mother, importance,
  confidence, target_layer, source_message_ids_json, status, metadata_json,
  created_at
)
VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'candidate', ?, ?)
""",
                    [
                        date_key,
                        next_version,
                        str(candidate.get("label") or "Untitled candidate"),
                        str(candidate.get("evidence") or ""),
                        str(candidate.get("domain") or "everyday_slice"),
                        str(candidate.get("function") or "daily_context"),
                        str(candidate.get("primary_mother") or "E"),
                        _optional_str(candidate.get("secondary_mother")),
                        _optional_int(candidate.get("importance")),
                        _optional_str(candidate.get("confidence")),
                        target_layer,
                        _json(candidate.get("source_message_ids") or []),
                        _json(candidate.get("metadata")),
                        updated_at,
                    ],
                )
        return next_version


def _json(value: Any) -> str | None:
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _ensure_column(
    conn: sqlite3.Connection,
    *,
    table_name: str,
    column_name: str,
    definition: str,
) -> None:
    columns = {
        str(row["name"])
        for row in conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    }
    if column_name not in columns:
        conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {definition}")


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
