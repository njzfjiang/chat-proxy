from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any

import httpx
from pydantic import BaseModel, Field


class ContextEvidence(BaseModel):
    messages: list[dict[str, Any]]
    included_layers: list[str] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)
    layer_status: dict[str, dict[str, Any]] = Field(default_factory=dict)
    context_hash: str


class ContextToolTrace(BaseModel):
    conversation_id: str
    as_of_timestamp: str | None = None
    latency_ms: int
    source_ids: list[str] = Field(default_factory=list)
    included_layers: list[str] = Field(default_factory=list)
    layer_status: dict[str, dict[str, Any]] = Field(default_factory=dict)
    context_hash: str


def _flatten_source_ids(value: Any) -> list[str]:
    found: list[str] = []

    def visit(item: Any) -> None:
        if isinstance(item, str):
            found.append(item)
        elif isinstance(item, list):
            for child in item:
                visit(child)
        elif isinstance(item, dict):
            for child in item.values():
                visit(child)

    visit(value)
    return list(dict.fromkeys(found))


@dataclass
class BuildContextClient:
    base_url: str
    timeout_seconds: float = 10.0
    http_client: httpx.AsyncClient | None = None

    async def retrieve(
        self,
        *,
        prompt: str,
        conversation_id: str,
        as_of_timestamp: str | None = None,
        exclude_conversation_ids: list[str] | None = None,
    ) -> tuple[ContextEvidence, ContextToolTrace]:
        body: dict[str, Any] = {
            "client_id": "pydantic-agent-canary",
            "conversation_id": conversation_id,
            "assistant_key": "agent-canary",
            "user_text": prompt,
            "retrieval_enabled": True,
            "retrieval_inject": True,
            "retrieval_router_enabled": True,
            "retrieval_query_planner_enabled": True,
            "retrieval_renderer_v2_enabled": True,
            "tools_policy": {
                "expose_tools": True,
                "allowed_tool_groups": ["context_read"],
                "write_requires_confirmation": True,
                "tool_mode": "read_only",
            },
        }
        if as_of_timestamp:
            body["as_of_timestamp"] = as_of_timestamp
        exclusions = list(
            dict.fromkeys(
                item.strip()
                for item in [conversation_id, *(exclude_conversation_ids or [])]
                if item and item.strip()
            )
        )
        if exclusions:
            body["retrieval_exclude_conversation_ids"] = exclusions

        started = time.perf_counter()
        if self.http_client is not None:
            response = await self.http_client.post("/build_context", json=body)
        else:
            async with httpx.AsyncClient(
                base_url=self.base_url.rstrip("/"),
                timeout=self.timeout_seconds,
            ) as client:
                response = await client.post("/build_context", json=body)
        latency_ms = round((time.perf_counter() - started) * 1000)
        response.raise_for_status()
        payload = response.json()

        messages = payload.get("messages")
        debug = payload.get("debug")
        if not isinstance(messages, list) or not isinstance(debug, dict):
            raise ValueError("/build_context returned an invalid response shape")

        serialized = json.dumps(messages, ensure_ascii=False, sort_keys=True)
        context_hash = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        included_layers = [
            str(item) for item in debug.get("included_layers", [])
        ]
        source_ids = _flatten_source_ids(debug.get("source_ids", {}))
        raw_layer_status = debug.get("layer_status", {})
        layer_status = (
            raw_layer_status if isinstance(raw_layer_status, dict) else {}
        )
        evidence = ContextEvidence(
            messages=messages,
            included_layers=included_layers,
            source_ids=source_ids,
            layer_status=layer_status,
            context_hash=context_hash,
        )
        trace = ContextToolTrace(
            conversation_id=conversation_id,
            as_of_timestamp=as_of_timestamp,
            latency_ms=latency_ms,
            source_ids=source_ids,
            included_layers=included_layers,
            layer_status=layer_status,
            context_hash=context_hash,
        )
        return evidence, trace
