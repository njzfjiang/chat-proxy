from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

from .config import ProxyConfig
from .parsing import OPENAI_CHAT_COMPLETION_BODY_KEYS, resolve_conversation
from .storage import ChatProxyStore


@dataclass(frozen=True)
class ContextBuildResult:
    upstream_body: dict[str, Any]
    snapshot: dict[str, Any]


def build_web_chat_context(
    *,
    body: Mapping[str, Any],
    cfg: ProxyConfig,
    store: ChatProxyStore,
    headers: Mapping[str, str],
) -> ContextBuildResult:
    upstream = {
        key: value
        for key, value in body.items()
        if key in OPENAI_CHAT_COMPLETION_BODY_KEYS
    }
    messages = body.get("messages")
    user_text = str(body.get("user_text") or "").strip()
    system_prompt = str(body.get("system_prompt") or "").strip()
    snapshot: dict[str, Any] = {
        "source": "webapp",
        "order": ["system", "wb_snippets", "recent_turns", "current_user"],
        "budgets": {
            "recent_turns": cfg.chat_recent_k,
            "wb_items": cfg.worldbook_max_items,
            "wb_chars_total": cfg.worldbook_chars_total,
        },
        "components": [],
    }

    if isinstance(messages, list):
        upstream_messages = list(messages)
        snapshot["mode"] = "explicit_messages"
        snapshot["components"].append(
            {
                "name": "explicit_messages",
                "message_count": len(messages),
                "chars": _messages_chars(messages),
            }
        )
    elif user_text:
        identity = resolve_conversation(dict(headers), dict(body))
        recent_context, recent_snapshot = _recent_context_messages(
            store=store,
            conversation_id=identity.conversation_id,
            limit=cfg.chat_recent_k,
        )
        base_messages = [
            *recent_context,
            {"role": "user", "content": user_text},
        ]
        wb_messages, wb_snapshot = _worldbook_messages(
            cfg=cfg,
            scan_text=_scan_text(base_messages),
        )
        upstream_messages = [*wb_messages, *base_messages]
        snapshot["mode"] = "db_recent_turns"
        snapshot["conversation_id"] = identity.conversation_id
        snapshot["components"].extend(
            [
                wb_snapshot,
                recent_snapshot,
                {
                    "name": "current_user",
                    "message_count": 1,
                    "chars": len(user_text),
                },
            ]
        )
    else:
        raise ValueError("POST /chat requires messages or user_text.")

    if isinstance(messages, list):
        wb_messages, wb_snapshot = _worldbook_messages(
            cfg=cfg,
            scan_text=_scan_text(upstream_messages),
        )
        if wb_messages:
            upstream_messages = [*wb_messages, *upstream_messages]
        snapshot["components"].insert(0, wb_snapshot)

    if system_prompt:
        upstream_messages = [
            {"role": "system", "content": system_prompt},
            *upstream_messages,
        ]
        snapshot["components"].insert(
            0,
            {
                "name": "system",
                "message_count": 1,
                "chars": len(system_prompt),
            },
        )

    upstream["messages"] = upstream_messages
    if not str(upstream.get("model") or "").strip():
        upstream["model"] = cfg.chat_model
    snapshot["model"] = upstream["model"]
    snapshot["final_message_count_before_rolling"] = len(upstream_messages)
    snapshot["final_chars_before_rolling"] = _messages_chars(upstream_messages)
    return ContextBuildResult(upstream_body=upstream, snapshot=snapshot)


def _worldbook_messages(
    *,
    cfg: ProxyConfig,
    scan_text: str,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    snapshot: dict[str, Any] = {
        "name": "wb_snippets",
        "enabled": cfg.worldbook_enabled,
        "message_count": 0,
        "items": [],
        "chars": 0,
    }
    if not cfg.worldbook_enabled:
        return [], snapshot
    paths = cfg.worldbook_paths or ((cfg.worldbook_path,) if cfg.worldbook_path else ())
    if not paths:
        snapshot["error"] = "CHAT_PROXY_WORLDBOOK_PATHS is not configured."
        return [], snapshot
    entries: list[dict[str, Any]] = []
    errors: list[str] = []
    for path in paths:
        loaded, error = _load_worldbook_entries(path)
        if error:
            errors.append(error)
            continue
        entries.extend(loaded)
    snapshot["sources"] = [str(path) for path in paths]
    if errors:
        snapshot["errors"] = errors
    if not entries:
        return [], snapshot

    matches = [
        match
        for entry in entries
        if (match := _match_worldbook_entry(entry, scan_text)) is not None
    ]
    matches.sort(
        key=lambda item: (
            int(item["entry"].get("priority") or 0),
            len(str(item.get("keyword") or "")),
        ),
        reverse=True,
    )

    chosen: list[dict[str, Any]] = []
    blocks: list[str] = []
    remaining_chars = max(0, cfg.worldbook_chars_total)
    max_items = max(0, cfg.worldbook_max_items)
    for match in matches:
        if len(chosen) >= max_items or remaining_chars <= 0:
            break
        entry = match["entry"]
        content = str(entry.get("content") or "").strip()
        if not content:
            continue
        clipped = content[:remaining_chars].rstrip()
        if not clipped:
            continue
        remaining_chars -= len(clipped)
        blocks.append(f"[{entry.get('name') or entry.get('id') or 'Worldbook'}]\n{clipped}")
        chosen.append(
            {
                "id": entry.get("id"),
                "name": entry.get("name"),
                "book_name": entry.get("_book_name"),
                "source": entry.get("_source_path"),
                "priority": entry.get("priority"),
                "keyword": match.get("keyword"),
                "chars": len(clipped),
            }
        )

    if not blocks:
        return [], snapshot
    content = "Triggered world book snippets:\n\n" + "\n\n".join(blocks)
    snapshot.update(
        {
            "message_count": 1,
            "items": chosen,
            "chars": len(content),
        }
    )
    return [{"role": "system", "content": content}], snapshot


def _recent_context_messages(
    *,
    store: ChatProxyStore,
    conversation_id: str,
    limit: int,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    rows = store.get_recent_messages(
        conversation_id=conversation_id,
        limit=max(0, min(limit, 80)),
    )
    messages: list[dict[str, str]] = []
    message_ids: list[int] = []
    skipped = 0
    for row in rows:
        role = str(row.get("role") or "").strip()
        content = str(row.get("content") or "").strip()
        kind = str(row.get("kind") or "chat").strip()
        if role not in {"user", "assistant", "system"} or kind not in {"chat", ""}:
            skipped += 1
            continue
        if content:
            messages.append({"role": role, "content": content})
            if row.get("id") is not None:
                message_ids.append(int(row["id"]))

    snapshot = {
        "name": "recent_turns",
        "message_count": len(messages),
        "message_ids": message_ids,
        "chars": _messages_chars(messages),
        "skipped": skipped,
    }
    return messages, snapshot


def _messages_chars(messages: list[Any]) -> int:
    total = 0
    for message in messages:
        if isinstance(message, Mapping):
            total += len(str(message.get("content") or ""))
    return total


def _scan_text(messages: list[Any]) -> str:
    parts = []
    for message in messages:
        if isinstance(message, Mapping):
            parts.append(str(message.get("content") or ""))
    return "\n".join(parts)


@lru_cache(maxsize=8)
def _load_worldbook_entries(path: Path) -> tuple[tuple[dict[str, Any], ...], str | None]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return (), f"Could not read worldbook: {exc}"
    entries = payload.get("entries") if isinstance(payload, dict) else None
    if entries is None and isinstance(payload, dict):
        data = payload.get("data")
        if isinstance(data, dict):
            entries = data.get("entries")
    if not isinstance(entries, list):
        return (), f"Worldbook JSON did not contain an entries list: {path}"
    book_name = ""
    if isinstance(payload, dict):
        data = payload.get("data")
        if isinstance(data, dict):
            book_name = str(data.get("name") or "")
        book_name = book_name or str(payload.get("name") or path.name)
    normalized = tuple(
        {
            **entry,
            "_source_path": str(path),
            "_book_name": book_name,
        }
        for entry in entries
        if isinstance(entry, dict)
    )
    return normalized, None


def _match_worldbook_entry(
    entry: Mapping[str, Any],
    scan_text: str,
) -> dict[str, Any] | None:
    if entry.get("enabled") is False:
        return None
    if entry.get("constantActive") is True:
        return {"entry": entry, "keyword": "(constant)"}
    keywords = entry.get("keywords")
    if not isinstance(keywords, list):
        return None
    case_sensitive = entry.get("caseSensitive") is True
    use_regex = entry.get("useRegex") is True
    haystack = scan_text if case_sensitive else scan_text.lower()
    for raw_keyword in keywords:
        keyword = str(raw_keyword or "").strip()
        if not keyword:
            continue
        needle = keyword if case_sensitive else keyword.lower()
        if use_regex:
            flags = 0 if case_sensitive else re.IGNORECASE
            try:
                if re.search(keyword, scan_text, flags=flags):
                    return {"entry": entry, "keyword": keyword}
            except re.error:
                continue
        elif needle in haystack:
            return {"entry": entry, "keyword": keyword}
    return None
