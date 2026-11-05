from __future__ import annotations

import hashlib
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from .context_client import ContextEvidence


ARTIFACT_SCHEMA_VERSION = 1


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_run_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"agent-canary-{timestamp}-{uuid.uuid4().hex[:8]}"


def default_output_path(run_id: str) -> Path:
    configured = os.getenv("AGENT_CANARY_OUTPUT", "").strip()
    if configured:
        return Path(configured).expanduser()
    return Path("benchmark_outputs") / "agent_canary" / f"{run_id}.jsonl"


def redact_destination(base_url: str) -> str:
    parts = urlsplit(base_url)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))


def evidence_snapshot(evidence: ContextEvidence) -> dict[str, Any]:
    messages: list[dict[str, Any]] = []
    for message in evidence.messages:
        content = str(message.get("content") or "")
        messages.append(
            {
                "role": str(message.get("role") or ""),
                "chars": len(content),
                "content_hash": hashlib.sha256(content.encode("utf-8")).hexdigest(),
                "content": content,
            }
        )
    return {
        "context_hash": evidence.context_hash,
        "included_layers": evidence.included_layers,
        "source_ids": evidence.source_ids,
        "layer_status": evidence.layer_status,
        "messages": messages,
    }


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_APPEND | os.O_CREAT | os.O_WRONLY
    fd = os.open(path, flags, 0o600)
    with os.fdopen(fd, "a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True, default=str))
        handle.write("\n")
        handle.flush()
