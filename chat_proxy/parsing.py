from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Mapping


META_BLOCK_RE = re.compile(
    r"\[kelivo_meta\](?P<body>.*?)\[/kelivo_meta\]",
    re.IGNORECASE | re.DOTALL,
)
LEADING_TIMESTAMP_RE = re.compile(
    r"^\s*\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}"
    r"(?::\d{2})?(?:\s*(?:Z|[+-]\d{2}:?\d{2}))?\s*(?:\r?\n)+",
)


SENSITIVE_HEADERS = {
    "authorization",
    "proxy-authorization",
    "x-api-key",
    "api-key",
    "cookie",
    "set-cookie",
}

KELIVO_ANALYSIS_META_BODY_KEY = "_kelivo_analysis_meta"
KELIVO_ANALYSIS_VERSION_HEADER = "x-kelivo-analysis-version"
OPENAI_CHAT_COMPLETION_BODY_KEYS = {
    "model",
    "messages",
    "stream",
    "temperature",
    "top_p",
    "max_tokens",
    "max_completion_tokens",
    "n",
    "stop",
    "presence_penalty",
    "frequency_penalty",
    "logit_bias",
    "user",
    "response_format",
    "seed",
    "tools",
    "tool_choice",
    "parallel_tool_calls",
    "stream_options",
    "reasoning_effort",
    "thinking",
    "prediction",
    "modalities",
    "audio",
    "store",
    "metadata",
}


@dataclass(frozen=True)
class ConversationIdentity:
    conversation_id: str
    resolver: str
    client_key: str | None = None
    assistant_key: str | None = None
    provider_key: str | None = None
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class PreparedRequestBody:
    body: dict[str, Any]
    stripped_metadata: dict[str, Any] | None = None
    mode: str = "normal"


def stable_hash(value: str, length: int = 24) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


def sanitize_headers(headers: Mapping[str, str]) -> dict[str, str]:
    sanitized: dict[str, str] = {}
    for key, value in headers.items():
        if key.lower() in SENSITIVE_HEADERS:
            sanitized[key] = "[REDACTED]"
        else:
            sanitized[key] = value
    return sanitized


def extract_text_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
                elif item.get("type") in {"image_url", "input_image"}:
                    parts.append("[image]")
        return "\n".join(part for part in parts if part).strip()
    return json.dumps(content, ensure_ascii=False, sort_keys=True)


def strip_leading_message_timestamp(text: str) -> str:
    return LEADING_TIMESTAMP_RE.sub("", text, count=1).strip()


def prepare_request_body_for_upstream(
    headers: Mapping[str, str],
    body: Mapping[str, Any],
) -> PreparedRequestBody:
    lowered = {key.lower(): value for key, value in headers.items()}
    raw_meta = body.get(KELIVO_ANALYSIS_META_BODY_KEY)
    has_dev_metadata = isinstance(raw_meta, dict) or bool(
        lowered.get(KELIVO_ANALYSIS_VERSION_HEADER, "").strip()
    )
    if not has_dev_metadata:
        return PreparedRequestBody(body=dict(body))

    upstream_body = {
        key: value
        for key, value in body.items()
        if key in OPENAI_CHAT_COMPLETION_BODY_KEYS
    }
    stripped_metadata = (
        {str(key): value for key, value in raw_meta.items()}
        if isinstance(raw_meta, dict)
        else {}
    )
    return PreparedRequestBody(
        body=upstream_body,
        stripped_metadata=stripped_metadata,
        mode="kelivo_analysis",
    )


def last_user_text(body: Mapping[str, Any]) -> str | None:
    messages = body.get("messages")
    if not isinstance(messages, list):
        return None
    for message in reversed(messages):
        if not isinstance(message, dict):
            continue
        if str(message.get("role", "")) != "user":
            continue
        text = strip_leading_message_timestamp(extract_text_content(message.get("content")))
        return text or None
    return None


def first_system_text(body: Mapping[str, Any]) -> str | None:
    messages = body.get("messages")
    if not isinstance(messages, list):
        return None
    for message in messages:
        if not isinstance(message, dict):
            continue
        if str(message.get("role", "")) not in {"system", "developer"}:
            continue
        text = extract_text_content(message.get("content")).strip()
        if text:
            return text
    return None


def parse_meta_block(text: str | None) -> dict[str, str]:
    if not text:
        return {}
    match = META_BLOCK_RE.search(text)
    if not match:
        return {}
    raw = match.group("body").strip()
    if not raw:
        return {}
    if raw.startswith("{"):
        try:
            decoded = json.loads(raw)
            if isinstance(decoded, dict):
                return {str(k): str(v) for k, v in decoded.items() if v is not None}
        except json.JSONDecodeError:
            pass

    meta: dict[str, str] = {}
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, value = line.split("=", 1)
        elif ":" in line:
            key, value = line.split(":", 1)
        else:
            continue
        key = key.strip()
        value = value.strip()
        if key and value:
            meta[key] = value
    return meta


def resolve_conversation(
    headers: Mapping[str, str],
    body: Mapping[str, Any],
) -> ConversationIdentity:
    lowered = {key.lower(): value for key, value in headers.items()}
    header_conversation = lowered.get("x-kelivo-conversation-id", "").strip()
    header_client = (
        lowered.get("x-kelivo-client-id")
        or lowered.get("x-kelivo-client-key")
        or lowered.get("x-client-id")
        or lowered.get("x-client-key")
        or lowered.get("client-id")
        or lowered.get("client-key")
        or ""
    ).strip() or None
    header_assistant = lowered.get("x-kelivo-assistant-key", "").strip() or None
    header_provider = (
        lowered.get("x-kelivo-provider-key")
        or lowered.get("x-provider-key")
        or lowered.get("provider-key")
        or ""
    ).strip() or None
    meta = parse_meta_block(first_system_text(body))
    meta_conversation = (
        meta.get("conversation")
        or meta.get("conversation_id")
        or meta.get("session")
        or meta.get("session_id")
        or ""
    ).strip()
    meta_client = (
        meta.get("client")
        or meta.get("client_id")
        or meta.get("client_key")
        or ""
    ).strip() or None
    meta_assistant = (
        meta.get("assistant")
        or meta.get("assistant_key")
        or meta.get("assistant_id")
        or ""
    ).strip() or None
    meta_provider = (
        meta.get("provider")
        or meta.get("provider_key")
        or meta.get("provider_id")
        or ""
    ).strip() or None
    body_meta = body.get(KELIVO_ANALYSIS_META_BODY_KEY)
    analysis_meta = body_meta if isinstance(body_meta, dict) else {}
    analysis_conversation = (
        analysis_meta.get("conversation_id")
        or analysis_meta.get("session_id")
        or analysis_meta.get("conversation_title")
        or ""
    )
    analysis_conversation = str(analysis_conversation).strip()
    analysis_title = str(analysis_meta.get("conversation_title") or "").strip() or None
    analysis_assistant = str(analysis_meta.get("assistant_id") or "").strip() or None
    analysis_provider = str(analysis_meta.get("provider_key") or "").strip() or None
    analysis_model = str(analysis_meta.get("model_id") or "").strip() or None
    model_id = str(body.get("model") or analysis_model or "unknown-model")

    if analysis_conversation:
        client_key = header_client or meta_client or assigned_client_key(body, model_id)
        assistant_key = header_assistant or analysis_assistant or meta_assistant
        provider_key = header_provider or analysis_provider or meta_provider
        return ConversationIdentity(
            conversation_id=header_conversation or analysis_conversation,
            resolver="kelivo_analysis",
            client_key=client_key,
            assistant_key=assistant_key or analysis_title,
            provider_key=provider_key,
            metadata={
                "conversation_id": header_conversation or analysis_conversation,
                "conversation_title": analysis_title,
                "client_key": client_key,
                "assistant_key": assistant_key,
                "provider_key": provider_key,
                "model_id": analysis_model or model_id,
                "kelivo_analysis": analysis_meta,
                "system_meta": meta,
            },
        )

    if header_conversation:
        return ConversationIdentity(
            conversation_id=header_conversation,
            resolver="header",
            client_key=header_client,
            assistant_key=header_assistant,
            provider_key=header_provider,
            metadata={
                "conversation_id": header_conversation,
                "client_key": header_client,
                "assistant_key": header_assistant,
                "provider_key": header_provider,
            },
        )

    if meta_conversation:
        return ConversationIdentity(
            conversation_id=meta_conversation,
            resolver="system_meta",
            client_key=header_client or meta_client,
            assistant_key=header_assistant or meta_assistant,
            provider_key=header_provider or meta_provider,
            metadata=meta,
        )

    client_key = header_client or meta_client or assigned_client_key(body, model_id)
    assistant_key = header_assistant or meta_assistant or model_id
    raw = f"{client_key}|{assistant_key}|{model_id}"
    return ConversationIdentity(
        conversation_id=f"proxy_assigned_{stable_hash(raw)}",
        resolver="proxy_assigned",
        client_key=client_key,
        assistant_key=assistant_key,
        provider_key=header_provider or meta_provider,
        metadata={
            "client_key": client_key,
            "assistant_key": assistant_key,
            "provider_key": header_provider or meta_provider,
            "model_id": model_id,
        },
    )


def assigned_client_key(body: Mapping[str, Any], model_id: str) -> str:
    user_hint = str(body.get("user") or "").strip()
    raw = f"{user_hint}|{model_id}" if user_hint else model_id
    return f"proxy_assigned_client_{stable_hash(raw, 16)}"


def request_id_for(body_text: str, headers: Mapping[str, str]) -> str:
    lowered = {key.lower(): value for key, value in headers.items()}
    explicit = (
        lowered.get("x-request-id")
        or lowered.get("x-kelivo-request-id")
        or lowered.get("request-id")
    )
    if explicit and explicit.strip():
        return explicit.strip()
    return f"req_{stable_hash(body_text, length=32)}"


def message_id_for(
    *,
    request_id: str,
    conversation_id: str,
    role: str,
    content: str,
) -> str:
    return stable_hash(
        f"{request_id}|{conversation_id}|{role}|{stable_hash(content, 32)}",
        length=40,
    )


def extract_chat_completion_text(response: Mapping[str, Any]) -> str | None:
    choices = response.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            message = first.get("message")
            if isinstance(message, dict):
                text = extract_text_content(message.get("content")).strip()
                if text:
                    return text
            text = extract_text_content(first.get("text")).strip()
            if text:
                return text
    output_text = response.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()
    return None


class SseTextAccumulator:
    def __init__(self) -> None:
        self._buffer = ""
        self._parts: list[str] = []

    @property
    def text(self) -> str:
        return "".join(self._parts)

    def add_bytes(self, chunk: bytes) -> None:
        self._buffer += chunk.decode("utf-8", errors="replace")
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self._consume_line(line.rstrip("\r"))

    def _consume_line(self, line: str) -> None:
        if not line.startswith("data:"):
            return
        payload = line[5:].strip()
        if not payload or payload == "[DONE]":
            return
        try:
            decoded = json.loads(payload)
        except json.JSONDecodeError:
            return
        if not isinstance(decoded, dict):
            return
        choices = decoded.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, dict):
                delta = first.get("delta")
                if isinstance(delta, dict):
                    text = extract_text_content(delta.get("content"))
                    if text:
                        self._parts.append(text)
                message = first.get("message")
                if isinstance(message, dict):
                    text = extract_text_content(message.get("content"))
                    if text:
                        self._parts.append(text)
