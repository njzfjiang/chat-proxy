from __future__ import annotations

from typing import Any, Mapping

import httpx

from .config import ProxyConfig
from .parsing import extract_chat_completion_text
from .storage import ChatProxyStore

SUMMARY_SYSTEM_PROMPT = (
    "Maintain a concise, cautious rolling continuity cache for a chat "
    "conversation. This cache is fallible compression, not authoritative memory. "
    "Keep only explicitly supported current context and unresolved threads. "
    "Prefer omission over inference and keep the whole summary under about 180 "
    "words."
)

SUMMARY_FORMAT_INSTRUCTIONS = """Return exactly these sections:
Now: 1-3 sentences about the explicitly supported current situation, or "None".
Current focus & near-term goals (1–14 days): up to 5 bullets containing only intentions or goals the user explicitly stated or confirmed, or "None".
Key context: up to 5 bullets containing only explicitly stated, confirmed, or externally verified context needed for continuity, or "None".
Open threads: up to 5 bullets containing only explicitly unanswered questions, pending decisions, or promised follow-ups, or "None".
Style / protocols: up to 5 bullets containing only user-stated or user-confirmed interaction preferences, boundaries, or care/relationship protocols, or "None".

Rules:
- Keep the whole summary under about 180 words.
- A repeated pattern is not a durable preference, protocol, identity fact, or goal unless the user explicitly states it, confirms it, or the supplied prior summary marks it as externally verified.
- Any section may be "None". Never invent content to avoid an empty section.
- Do not infer near-term goals from mood, emotional support, or actions unless the user explicitly expresses an intention.
- Do not promote a one-off assistant response into a user preference or protocol.
- Current raw conversation is primary evidence. The prior rolling cache may preserve continuity only; it must never override, reinterpret, or strengthen the current conversation.
- If a previous-summary claim is not supported by the current conversation input, preserve it only if it is already marked `[verified]`; otherwise drop it.
- Never add a `[verified]` marker yourself. Only preserve an existing marker supplied by an external verification step.
- Do not treat casual assistant questions, suggestions, or invitations as open threads unless the user explicitly accepted them, deferred a decision, or requested a follow-up.
- Prefer omission over inference.
- Remove stale or completed goals and threads as they clearly resolve.
- Describe concrete actions and explicit decisions without adding psychological diagnoses, generic relationship labels, or inferred motivations.
- Do not retain explicit intimacy details unless they are necessary for a currently unresolved, explicitly stated boundary or safety issue.
- Return only the updated rolling summary."""

SUMMARY_INJECTION_PREFIX = (
    "Rolling continuity cache (derived and fallible; not authoritative memory).\n"
    "Use it only as a continuity hint. Prefer current messages and curated "
    "sources when they conflict:\n"
)


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
