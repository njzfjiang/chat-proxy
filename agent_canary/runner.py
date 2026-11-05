from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import uuid
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel
from pydantic_ai import UsageLimits
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from dotenv import load_dotenv

from .agent import AgentDeps, build_agent
from .artifacts import (
    ARTIFACT_SCHEMA_VERSION,
    append_jsonl,
    default_output_path,
    evidence_snapshot,
    new_run_id,
    redact_destination,
    utc_now,
)
from .context_client import BuildContextClient


def _required_env(name: str, fallback: str | None = None) -> str:
    value = os.getenv(name) or (os.getenv(fallback) if fallback else None)
    if not value:
        suffix = f" or {fallback}" if fallback else ""
        raise ValueError(f"Missing required environment variable: {name}{suffix}")
    return value


def _jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "__dict__"):
        return vars(value)
    return str(value)


def _load_cases(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.prompt:
        return [
            {
                "id": args.case_id or "adhoc",
                "prompt": args.prompt,
                "conversation_id": args.conversation_id
                or f"agent-canary-{uuid.uuid4().hex}",
                "as_of_timestamp": args.as_of_timestamp,
            }
        ]
    with Path(args.manifest).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError("Manifest must contain a JSON list")
    return payload


def _validate_context_requirements(
    case: dict[str, Any],
    trace: Any,
    evidence: Any | None = None,
) -> None:
    missing_layers = [
        str(layer)
        for layer in case.get("required_layers", [])
        if str(layer) not in trace.included_layers
    ]
    if missing_layers:
        status = trace.layer_status.get("kmlog_search", {})
        raise RuntimeError(
            "Context preflight missing required injected layers: "
            + ", ".join(missing_layers)
            + f"; kmlog_search_status={json.dumps(status, sort_keys=True)}"
        )

    missing_prefixes = [
        str(prefix)
        for prefix in case.get("required_source_prefixes", [])
        if not any(source_id.startswith(str(prefix)) for source_id in trace.source_ids)
    ]
    if missing_prefixes:
        raise RuntimeError(
            "Context preflight missing required source prefixes: "
            + ", ".join(missing_prefixes)
        )

    missing_source_ids = [
        str(source_id)
        for source_id in case.get("required_source_ids", [])
        if str(source_id) not in trace.source_ids
    ]
    if missing_source_ids:
        raise RuntimeError(
            "Context preflight missing required source IDs: "
            + ", ".join(missing_source_ids)
        )

    forbidden_layers = [
        str(layer)
        for layer in case.get("forbidden_layers", [])
        if str(layer) in trace.included_layers
    ]
    if forbidden_layers:
        raise RuntimeError(
            "Context preflight included forbidden layers: "
            + ", ".join(forbidden_layers)
        )

    forbidden_prefixes = [
        str(prefix)
        for prefix in case.get("forbidden_source_prefixes", [])
        if any(source_id.startswith(str(prefix)) for source_id in trace.source_ids)
    ]
    if forbidden_prefixes:
        raise RuntimeError(
            "Context preflight included forbidden source prefixes: "
            + ", ".join(forbidden_prefixes)
        )

    required_visible_text = [
        str(value).strip()
        for value in case.get("required_visible_text", [])
        if str(value).strip()
    ]
    if required_visible_text:
        messages = getattr(evidence, "messages", []) if evidence is not None else []
        visible_text = "\n".join(
            str(message.get("content") or "")
            for message in messages
            if isinstance(message, dict)
        )
        missing_visible_text = [
            value for value in required_visible_text if value not in visible_text
        ]
        if missing_visible_text:
            raise RuntimeError(
                "Context preflight missing required visible text: "
                + ", ".join(missing_visible_text)
            )


def _evaluate_answer_expectations(
    case: dict[str, Any],
    answer: Any,
) -> dict[str, dict[str, Any]]:
    checks: dict[str, dict[str, Any]] = {}
    for field in ("evidence_status", "claim_type"):
        expected_key = f"expected_{field}"
        if expected_key not in case:
            continue
        expected = str(case[expected_key])
        actual = str(getattr(answer, field))
        checks[field] = {
            "expected": expected,
            "actual": actual,
            "passed": actual == expected,
        }
    return checks


async def _run(args: argparse.Namespace) -> int:
    base_url = os.getenv("AGENT_CANARY_PROXY_BASE", "http://127.0.0.1:8787")
    model_name = _required_env("AGENT_CANARY_MODEL", "CHAT_PROXY_CHAT_MODEL")
    api_key = _required_env("AGENT_CANARY_API_KEY", "CHAT_PROXY_UPSTREAM_API_KEY")
    upstream_base = _required_env(
        "AGENT_CANARY_UPSTREAM_BASE", "CHAT_PROXY_UPSTREAM_BASE"
    )
    timeout = float(os.getenv("AGENT_CANARY_TIMEOUT_SECONDS", "15"))

    provider = OpenAIProvider(base_url=upstream_base, api_key=api_key)
    model = OpenAIChatModel(model_name, provider=provider)
    agent = build_agent(model)
    cases = _load_cases(args)
    run_id = new_run_id()
    output_path = Path(args.output).expanduser() if args.output else default_output_path(run_id)

    for case in cases:
        prompt = str(case["prompt"])
        conversation_id = str(case["conversation_id"])
        exclude_conversation_ids = list(
            dict.fromkeys(
                str(item).strip()
                for item in [
                    conversation_id,
                    *case.get("exclude_conversation_ids", []),
                ]
                if str(item).strip()
            )
        )
        context_client = BuildContextClient(base_url, timeout)
        started_at = utc_now()
        stage = "preflight"
        prefetched_context = None
        case_timeout: float | None = None
        try:
            case_timeout = float(case.get("timeout_seconds", timeout))
            if case_timeout <= 0:
                raise ValueError("timeout_seconds must be positive")
            prefetched_context = await context_client.retrieve(
                prompt=prompt,
                conversation_id=conversation_id,
                as_of_timestamp=case.get("as_of_timestamp"),
                exclude_conversation_ids=exclude_conversation_ids,
            )
            _validate_context_requirements(
                case,
                prefetched_context[1],
                prefetched_context[0],
            )
            stage = "agent"
            deps = AgentDeps(
                context_client=context_client,
                prompt=prompt,
                conversation_id=str(case["conversation_id"]),
                as_of_timestamp=case.get("as_of_timestamp"),
                exclude_conversation_ids=exclude_conversation_ids,
                prefetched_context=prefetched_context,
            )
            result = await agent.run(
                prompt,
                deps=deps,
                usage_limits=UsageLimits(request_limit=3, tool_calls_limit=1),
                timeout=case_timeout,
            )
            if len(deps.tool_traces) != 1:
                raise RuntimeError(
                    f"Expected exactly one context tool call, got {len(deps.tool_traces)}"
                )
            expectation_checks = _evaluate_answer_expectations(case, result.output)
            record = {
                "schema_version": ARTIFACT_SCHEMA_VERSION,
                "run_id": run_id,
                "status": "completed",
                "started_at": started_at,
                "completed_at": utc_now(),
                "case_id": case.get("id"),
                "prompt": prompt,
                "conversation_id": deps.conversation_id,
                "as_of_timestamp": deps.as_of_timestamp,
                "exclude_conversation_ids": exclude_conversation_ids,
                "model": model_name,
                "destination": redact_destination(upstream_base),
                "timeout_seconds": case_timeout,
                "answer": result.output.model_dump(mode="json"),
                "expectation_checks": expectation_checks,
                "evidence": evidence_snapshot(prefetched_context[0]),
                "tool_traces": [
                    trace.model_dump(mode="json") for trace in deps.tool_traces
                ],
                "usage": _jsonable(result.usage),
            }
            append_jsonl(output_path, record)
            console_record = {
                key: value for key, value in record.items() if key != "evidence"
            }
            console_record["artifact_path"] = str(output_path.resolve())
            print(
                json.dumps(
                    console_record,
                    ensure_ascii=False,
                    sort_keys=True,
                    default=str,
                )
            )
        except Exception as exc:
            failure_record: dict[str, Any] = {
                "schema_version": ARTIFACT_SCHEMA_VERSION,
                "run_id": run_id,
                "status": "error",
                "stage": stage,
                "started_at": started_at,
                "completed_at": utc_now(),
                "case_id": case.get("id"),
                "prompt": prompt,
                "conversation_id": str(case.get("conversation_id") or ""),
                "as_of_timestamp": case.get("as_of_timestamp"),
                "exclude_conversation_ids": exclude_conversation_ids,
                "model": model_name,
                "destination": redact_destination(upstream_base),
                "timeout_seconds": case_timeout,
                "error": {"type": type(exc).__name__, "message": str(exc)},
            }
            if prefetched_context is not None:
                failure_record["evidence"] = evidence_snapshot(prefetched_context[0])
                failure_record["preflight_trace"] = prefetched_context[1].model_dump(
                    mode="json"
                )
            append_jsonl(output_path, failure_record)
            raise
    return 0


def main() -> int:
    load_dotenv()
    os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Run the isolated retrieval agent")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--prompt")
    source.add_argument("--manifest")
    parser.add_argument("--case-id")
    parser.add_argument("--conversation-id")
    parser.add_argument("--as-of-timestamp")
    parser.add_argument(
        "--output",
        help="JSONL artifact path (default: benchmark_outputs/agent_canary/<run-id>.jsonl)",
    )
    args = parser.parse_args()
    try:
        return asyncio.run(_run(args))
    except (ValueError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
