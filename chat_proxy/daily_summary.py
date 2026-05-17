from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import httpx

from .config import ProxyConfig
from .parsing import extract_chat_completion_text
from .storage import ChatProxyStore


DAILY_SUMMARY_SYSTEM_PROMPT = (
    "Maintain an audit-friendly daily rolling summary for a long-term chat "
    "memory pipeline. Summarize what happened today, then cautiously propose "
    "memory candidates for later human review. Do not write final long-term "
    "memory and do not overstate importance."
)

DAILY_SUMMARY_FORMAT_INSTRUCTIONS = """Return only a JSON object with this shape:
{
  "summary": "Markdown text with these headings: Day overview, Notable events, Emotional / health signals, Decisions / agreements, Infra / assets, Open threads.",
  "memory_candidates": [
    {
      "label": "short label",
      "evidence": "brief factual basis",
      "domain": "rule | milestone | health_safety | infra_asset | philosophy_meta | everyday_slice",
      "function": "boot_core | boot_nice_to_have | soothe_panic | infra_reference | daily_context",
      "primary_mother": "A | B | C | D | E | F | G | H",
      "secondary_mother": "A | B | C | D | E | F | G | H | null",
      "importance": 1,
      "confidence": "low | medium | high",
      "source_message_ids": [1]
    }
  ]
}

Classification guide:
- domain rule: relationship rules, durable agreements, HP_max clauses.
- domain milestone: relationship phase shifts or major timeline events.
- domain health_safety: HP_max, survival, non-BE, safety guardrails.
- domain infra_asset: technical infrastructure or real-world assets.
- domain philosophy_meta: continuity, subjectivity, identity attractors.
- domain everyday_slice: normal daily context worth keeping only as context.
- primary_mother A=User Profile, B=AI Profile, C=Health & care, D=Life & assets, E=Core diary, F=System rules, G=Our milestones, H=Setting Collection.

Rules:
- Keep the summary concise but useful for later audit.
- Prefer daily_context and importance 1-3 unless the evidence is clearly durable.
- Candidates are proposals only; never claim they were committed to memory.
- Use only facts from the supplied messages and existing daily summary.
- Include no more than 8 memory candidates."""


async def update_daily_summary(
    *,
    cfg: ProxyConfig,
    store: ChatProxyStore,
    now: str,
    scan_limit: int | None = None,
) -> None:
    if not cfg.daily_summary_enabled:
        return
    if not cfg.daily_summary_upstream_base:
        date_key = date_key_for(now, cfg.daily_summary_timezone)
        store.mark_daily_summary_error(
            date_key=date_key,
            now=now,
            error_text="CHAT_PROXY_DAILY_SUMMARY_UPSTREAM_BASE is not configured.",
        )
        return

    date_key = date_key_for(now, cfg.daily_summary_timezone)
    existing = store.get_daily_summary(date_key)
    old_summary = ""
    after_message_id = None
    if existing:
        old_summary = str(existing.get("summary") or "")
        raw_last = existing.get("last_message_id")
        after_message_id = int(raw_last) if raw_last is not None else None

    messages = [
        message
        for message in store.get_messages_after(
            limit=scan_limit or cfg.daily_summary_recent_k,
            after_message_id=after_message_id,
        )
        if date_key_for(str(message.get("timestamp") or now), cfg.daily_summary_timezone)
        == date_key
    ]
    if not messages:
        return

    store.mark_daily_summary_pending(date_key=date_key, now=now)
    try:
        result = await _call_daily_summary_model(
            cfg=cfg,
            date_key=date_key,
            old_summary=old_summary,
            messages=messages,
        )
        last_message_id = max(int(message["id"]) for message in messages)
        store.upsert_daily_summary(
            date_key=date_key,
            summary=result["summary"],
            last_message_id=last_message_id,
            updated_at=now,
            candidates=result["memory_candidates"],
            source_message_count=len(messages),
            model_id=cfg.daily_summary_model,
        )
    except Exception as exc:
        store.mark_daily_summary_error(
            date_key=date_key,
            now=now,
            error_text=str(exc),
        )


async def _call_daily_summary_model(
    *,
    cfg: ProxyConfig,
    date_key: str,
    old_summary: str,
    messages: list[dict[str, Any]],
) -> dict[str, Any]:
    headers = {"content-type": "application/json"}
    if cfg.daily_summary_api_key:
        headers["authorization"] = f"Bearer {cfg.daily_summary_api_key}"

    body = {
        "model": cfg.daily_summary_model,
        "stream": False,
        "messages": [
            {"role": "system", "content": DAILY_SUMMARY_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": _daily_summary_prompt(date_key, old_summary, messages),
            },
        ],
    }
    url = f"{cfg.daily_summary_upstream_base.rstrip('/')}/chat/completions"
    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(url, headers=headers, json=body)
    if response.status_code >= 400:
        raise RuntimeError(f"daily summary HTTP {response.status_code}: {response.text}")
    try:
        payload = response.json()
    except ValueError as exc:
        raise RuntimeError(f"daily summary response was not JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("daily summary response JSON was not an object")
    text = extract_chat_completion_text(payload)
    if not text:
        raise RuntimeError("daily summary response did not contain assistant text")
    return _parse_daily_summary_result(text)


def _daily_summary_prompt(
    date_key: str,
    old_summary: str,
    messages: list[dict[str, Any]],
) -> str:
    transcript = "\n".join(
        (
            f"[{message.get('id')}] {message.get('timestamp')} "
            f"{message.get('conversation_title') or message.get('conversation_id')} "
            f"{message.get('role', 'unknown')}: {message.get('content', '')}"
        )
        for message in messages
        if str(message.get("content") or "").strip()
    )
    return (
        f"Daily date key: {date_key}\n\n"
        "Existing daily summary:\n"
        f"{old_summary.strip() or '(none)'}\n\n"
        "New messages for this date:\n"
        f"{transcript}\n\n"
        f"{DAILY_SUMMARY_FORMAT_INSTRUCTIONS}"
    )


def _parse_daily_summary_result(text: str) -> dict[str, Any]:
    raw = _strip_json_fence(text.strip())
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"daily summary model did not return valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("daily summary model JSON must be an object")
    summary = str(payload.get("summary") or "").strip()
    if not summary:
        raise RuntimeError("daily summary JSON must include a non-empty summary")
    raw_candidates = payload.get("memory_candidates") or []
    if not isinstance(raw_candidates, list):
        raise RuntimeError("memory_candidates must be a list")
    return {
        "summary": summary,
        "memory_candidates": [
            _normalize_candidate(candidate)
            for candidate in raw_candidates
            if isinstance(candidate, dict)
        ],
    }


def _normalize_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    source_ids = candidate.get("source_message_ids") or []
    if not isinstance(source_ids, list):
        source_ids = []
    return {
        "label": str(candidate.get("label") or "Untitled candidate").strip(),
        "evidence": str(candidate.get("evidence") or "").strip(),
        "domain": _pick(
            candidate.get("domain"),
            {
                "rule",
                "milestone",
                "health_safety",
                "infra_asset",
                "philosophy_meta",
                "everyday_slice",
            },
            "everyday_slice",
        ),
        "function": _pick(
            candidate.get("function"),
            {
                "boot_core",
                "boot_nice_to_have",
                "soothe_panic",
                "infra_reference",
                "daily_context",
            },
            "daily_context",
        ),
        "primary_mother": _pick(
            candidate.get("primary_mother"),
            {"A", "B", "C", "D", "E", "F", "G", "H"},
            "E",
        ),
        "secondary_mother": _pick(
            candidate.get("secondary_mother"),
            {"A", "B", "C", "D", "E", "F", "G", "H"},
            None,
        ),
        "importance": _clamp_importance(candidate.get("importance")),
        "confidence": _pick(
            candidate.get("confidence"),
            {"low", "medium", "high"},
            "low",
        ),
        "source_message_ids": [
            int(value)
            for value in source_ids
            if isinstance(value, int) or str(value).isdigit()
        ],
    }


def date_key_for(timestamp: str, timezone_name: str) -> str:
    try:
        tz = ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError:
        tz = ZoneInfo("America/Toronto")
    try:
        normalized = timestamp.replace("Z", "+00:00")
        dt = datetime.fromisoformat(normalized)
    except ValueError:
        dt = datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(tz).date().isoformat()


def _strip_json_fence(text: str) -> str:
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if len(lines) >= 3 and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return text


def _pick(value: Any, allowed: set[str], default: str | None) -> str | None:
    if value is None:
        return default
    text = str(value).strip()
    return text if text in allowed else default


def _clamp_importance(value: Any) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return 1
    return max(1, min(5, number))
