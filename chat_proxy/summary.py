from __future__ import annotations

from typing import Any, Mapping

import httpx

from .config import ProxyConfig
from .parsing import extract_chat_completion_text
from .storage import ChatProxyStore


SUMMARY_SYSTEM_PROMPT = (
    "Maintain a concise, semi-structured rolling summary of a chat conversation. "
    "Preserve durable facts, decisions, preferences, unresolved tasks, and "
    "important context. Remove transient wording, remove stale details, and "
    "avoid inventing details. Keep the whole summary under about 180 words."
)
SUMMARY_FORMAT_INSTRUCTIONS = """Return exactly these sections:
Now: 1-3 sentences about the current situation.
Key context: up to 5 bullets.
Open threads: up to 5 bullets, or "None".
Style / protocols: up to 5 bullets.

Rules:
- Keep the whole summary under about 180 words.
- Prefer durable facts over recent phrasing.
- Remove stale details.
- Describe concrete actions, tone, and decisions; avoid re-labelling the relationship with generic kink labels (e.g. "dom/sub") or psychological diagnoses.
- Do not introduce new category labels for the relationship; keep using the fox–cat / long-term partner framing implied by the conversation, unless the user explicitly defines another label.
- Return only the updated rolling summary."""
SUMMARY_INJECTION_PREFIX = "Rolling summary of this conversation so far:\n"


def inject_rolling_summary(
    body: Mapping[str, Any],
    summary: str | None,
) -> dict[str, Any]:
    upstream_body = dict(body)
    clean_summary = (summary or "").strip()
    messages = upstream_body.get("messages")
    if not clean_summary or not isinstance(messages, list):
        return upstream_body

    injected = {
        "role": "system",
        "content": f"{SUMMARY_INJECTION_PREFIX}{clean_summary}",
    }
    upstream_body["messages"] = [injected, *messages]
    return upstream_body


async def update_conversation_summary(
    *,
    cfg: ProxyConfig,
    store: ChatProxyStore,
    conversation_id: str,
    now: str,
) -> None:
    if not cfg.summary_enabled:
        return
    if not cfg.summary_upstream_base:
        store.mark_summary_error(
            conversation_id=conversation_id,
            now=now,
            error_text="CHAT_PROXY_SUMMARY_UPSTREAM_BASE is not configured.",
        )
        return

    existing = store.get_summary(conversation_id)
    old_summary = ""
    after_message_id = None
    if existing:
        old_summary = str(existing.get("summary") or "")
        raw_last = existing.get("last_message_id")
        after_message_id = int(raw_last) if raw_last is not None else None

    messages = store.get_recent_messages(
        conversation_id=conversation_id,
        limit=cfg.summary_recent_k,
        after_message_id=after_message_id,
    )
    if not messages:
        return

    store.mark_summary_pending(conversation_id=conversation_id, now=now)
    try:
        new_summary = await _call_summary_model(
            cfg=cfg,
            old_summary=old_summary,
            messages=messages,
        )
        last_message_id = max(int(message["id"]) for message in messages)
        store.upsert_summary(
            conversation_id=conversation_id,
            summary=new_summary,
            last_message_id=last_message_id,
            updated_at=now,
            source_message_count=len(messages),
            model_id=cfg.summary_model,
        )
    except Exception as exc:
        store.mark_summary_error(
            conversation_id=conversation_id,
            now=now,
            error_text=str(exc),
        )


async def _call_summary_model(
    *,
    cfg: ProxyConfig,
    old_summary: str,
    messages: list[dict[str, Any]],
) -> str:
    headers = {"content-type": "application/json"}
    if cfg.summary_api_key:
        headers["authorization"] = f"Bearer {cfg.summary_api_key}"

    body = {
        "model": cfg.summary_model,
        "stream": False,
        "messages": [
            {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
            {"role": "user", "content": _summary_prompt(old_summary, messages)},
        ],
    }
    url = f"{cfg.summary_upstream_base.rstrip('/')}/chat/completions"
    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(url, headers=headers, json=body)
    if response.status_code >= 400:
        raise RuntimeError(f"summary HTTP {response.status_code}: {response.text}")
    try:
        payload = response.json()
    except ValueError as exc:
        raise RuntimeError(f"summary response was not JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("summary response JSON was not an object")
    summary = extract_chat_completion_text(payload)
    if not summary:
        raise RuntimeError("summary response did not contain assistant text")
    return summary.strip()


def _summary_prompt(old_summary: str, messages: list[dict[str, Any]]) -> str:
    transcript = "\n".join(
        f"{message.get('role', 'unknown')}: {message.get('content', '')}"
        for message in messages
        if str(message.get("content") or "").strip()
    )
    return (
        "Existing rolling summary:\n"
        f"{old_summary.strip() or '(none)'}\n\n"
        "New conversation messages:\n"
        f"{transcript}\n\n"
        f"{SUMMARY_FORMAT_INSTRUCTIONS}"
    )
