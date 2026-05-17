from __future__ import annotations

import re
from typing import Any, Mapping


VALID_KINDS = {"chat", "summary", "meta", "noise"}

SUMMARY_CARD_RE = re.compile(
    r"(^|\n)##\s+\d+\.\s+.+?\n.*?Tag[：:].*?\n.*?(时间戳|timestamp)[：:].*?\n.*?(发生了什么|what happened)[：:]",
    re.IGNORECASE | re.DOTALL,
)

SUMMARY_MARKERS = (
    "[summary of previous conversation]",
    "summary of previous conversation",
    "## summary",
    "# summary",
    "conversation summary",
    "长期记忆摘要",
    "对话摘要",
    "窗口摘要",
    "总结卡片",
)

SUMMARY_TITLE_MARKERS = {
    "summary",
    "daily summary",
    "rolling summary",
    "conversation summary",
}

NOISE_MARKERS = (
    "http error",
    "http 400",
    "http 401",
    "http 403",
    "http 404",
    "http 429",
    "http 500",
    "http 502",
    "http 503",
    "traceback (most recent call last)",
    "error:",
    "exception:",
    "failed to",
    "connection error",
    "read timed out",
    "rate limit",
    "bad gateway",
    "service unavailable",
)

META_MARKERS = (
    "chat-proxy",
    "kmlog-search",
    "sqlite",
    "fts5",
    "build_sqlite_fts",
    "cleaned_chats.jsonl",
    "manual_daily_summary",
    "daily_summary",
    "rolling summary",
    "import_scripts",
    "schema",
    "database",
)


def normalize_message_kind(value: Any) -> str | None:
    text = str(value or "").strip().lower()
    return text if text in VALID_KINDS else None


def classify_message_kind(
    *,
    content: str,
    conversation_title: str | None = None,
    row: Mapping[str, Any] | None = None,
) -> str:
    if row:
        explicit_kind = normalize_message_kind(row.get("kind"))
        if explicit_kind:
            return explicit_kind

    title = conversation_title or ""
    text = f"{title}\n{content}".lower()

    if _has_any(text, NOISE_MARKERS) or _looks_like_error_noise(content):
        return "noise"
    if (
        SUMMARY_CARD_RE.search(content)
        or _has_any(text, SUMMARY_MARKERS)
        or _looks_like_summary_title(title)
    ):
        return "summary"
    if _has_any(text, META_MARKERS):
        return "meta"
    return "chat"


def _has_any(text: str, markers: tuple[str, ...]) -> bool:
    return any(marker in text for marker in markers)


def _looks_like_summary_title(title: str) -> bool:
    normalized = title.strip().lower()
    return normalized in SUMMARY_TITLE_MARKERS or normalized.endswith(" summary")


def _looks_like_error_noise(content: str) -> bool:
    stripped = content.strip()
    if len(stripped) > 1200:
        return False
    lowered = stripped.lower()
    return (
        lowered.startswith("http") and "error" in lowered
    ) or lowered.startswith(("error ", "error:", "exception ", "exception:"))
