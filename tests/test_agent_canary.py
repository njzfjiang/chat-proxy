import json
from pathlib import Path

import httpx
import pytest

pytest.importorskip("pydantic_ai")

from pydantic_ai import models
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage

from agent_canary.agent import AgentDeps, build_agent
from agent_canary.artifacts import append_jsonl, evidence_snapshot, redact_destination
from agent_canary.context_client import (
    BuildContextClient,
    ContextEvidence,
    ContextToolTrace,
)
from agent_canary.runner import (
    _evaluate_answer_expectations,
    _jsonable,
    _validate_context_requirements,
)


@pytest.fixture(autouse=True)
def disable_real_model_requests():
    previous = models.ALLOW_MODEL_REQUESTS
    models.ALLOW_MODEL_REQUESTS = False
    try:
        yield
    finally:
        models.ALLOW_MODEL_REQUESTS = previous


@pytest.mark.anyio
async def test_agent_calls_read_only_context_once_and_records_server_sources():
    requests = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        body = json.loads(request.content)
        assert body["tools_policy"] == {
            "expose_tools": True,
            "allowed_tool_groups": ["context_read"],
            "write_requires_confirmation": True,
            "tool_mode": "read_only",
        }
        assert body["retrieval_enabled"] is True
        assert body["retrieval_inject"] is True
        assert body["retrieval_router_enabled"] is True
        assert body["retrieval_query_planner_enabled"] is True
        assert body["retrieval_renderer_v2_enabled"] is True
        assert body["retrieval_exclude_conversation_ids"] == [
            "agent-canary-test",
            "old-test-conversation",
        ]
        return httpx.Response(
            200,
            json={
                "messages": [
                    {"role": "system", "content": "Evidence from the proxy."},
                    {"role": "user", "content": body["user_text"]},
                ],
                "debug": {
                    "included_layers": ["kmlog_search", "current_user"],
                    "layer_status": {
                        "kmlog_search": {
                            "enabled": True,
                            "inject": True,
                            "message_count": 1,
                        }
                    },
                    "source_ids": {
                        "kmlog_search": ["message:28955"],
                        "current_user": [],
                    },
                },
                "context_packet": {},
            },
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://proxy"
    ) as client:
        deps = AgentDeps(
            context_client=BuildContextClient("http://proxy", http_client=client),
            prompt="What happened?",
            conversation_id="agent-canary-test",
            as_of_timestamp="2026-10-22T18:26:00+00:00",
            exclude_conversation_ids=["old-test-conversation"],
        )
        result = await build_agent(TestModel()).run("What happened?", deps=deps)

    assert result.output.evidence_status in {"supported", "insufficient"}
    assert len(requests) == 1
    assert requests[0].url.path == "/build_context"
    assert len(deps.tool_traces) == 1
    assert deps.tool_traces[0].source_ids == ["message:28955"]
    assert deps.tool_traces[0].layer_status["kmlog_search"]["message_count"] == 1
    assert len(deps.tool_traces[0].context_hash) == 64


@pytest.mark.anyio
async def test_context_client_rejects_invalid_response_shape():
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"messages": "not-a-list", "debug": {}})

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://proxy"
    ) as client:
        context_client = BuildContextClient("http://proxy", http_client=client)
        with pytest.raises(ValueError, match="invalid response shape"):
            await context_client.retrieve(
                prompt="test",
                conversation_id="agent-canary-test",
            )


def test_run_usage_property_is_json_serializable():
    payload = _jsonable(RunUsage(requests=2, input_tokens=10, output_tokens=4))
    assert isinstance(payload, dict)
    assert payload["requests"] == 2


def test_context_preflight_requires_injected_kmlog_evidence():
    class Trace:
        included_layers = ["wb_snippets", "current_user"]
        source_ids = ["worldbook:book#item"]
        layer_status = {
            "kmlog_search": {
                "enabled": True,
                "inject": True,
                "skipped_reason": "router did not select chat_history_search",
            }
        }

    with pytest.raises(RuntimeError, match="kmlog_search"):
        _validate_context_requirements(
            {
                "required_layers": ["kmlog_search"],
                "required_source_prefixes": ["kmlog:"],
                "required_source_ids": ["kmlog:28955"],
            },
            Trace(),
        )


def test_context_preflight_requires_locked_original_source_id():
    class Trace:
        included_layers = ["kmlog_search", "current_user"]
        source_ids = ["kmlog:49831"]
        layer_status = {
            "kmlog_search": {
                "enabled": True,
                "inject": True,
                "message_count": 1,
            }
        }

    with pytest.raises(RuntimeError, match="kmlog:28955"):
        _validate_context_requirements(
            {
                "required_layers": ["kmlog_search"],
                "required_source_prefixes": ["kmlog:"],
                "required_source_ids": ["kmlog:28955"],
            },
            Trace(),
        )


def test_context_preflight_rejects_forbidden_historical_injection():
    class Trace:
        included_layers = ["kmlog_search", "current_user"]
        source_ids = ["kmlog:8423"]
        layer_status = {"kmlog_search": {"message_count": 1}}

    with pytest.raises(RuntimeError, match="forbidden layers"):
        _validate_context_requirements(
            {
                "forbidden_layers": ["kmlog_search"],
                "forbidden_source_prefixes": ["kmlog:"],
            },
            Trace(),
        )


def test_context_preflight_requires_answer_bearing_visible_text():
    class Trace:
        included_layers = ["kmlog_search", "current_user"]
        source_ids = ["kmlog:7553"]
        layer_status = {"kmlog_search": {"message_count": 1}}

    evidence = ContextEvidence(
        messages=[
            {
                "role": "system",
                "content": "id=7553 excerpt: 只有一个真的。",
            }
        ],
        included_layers=Trace.included_layers,
        source_ids=Trace.source_ids,
        context_hash="c" * 64,
    )

    with pytest.raises(RuntimeError, match="匣中有门"):
        _validate_context_requirements(
            {
                "required_source_ids": ["kmlog:7553"],
                "required_visible_text": ["只有一个真的", "匣中有门，心中有钥"],
            },
            Trace(),
            evidence,
        )


def test_context_preflight_accepts_locked_source_with_visible_evidence():
    class Trace:
        included_layers = ["kmlog_search", "current_user"]
        source_ids = ["kmlog:7553"]
        layer_status = {"kmlog_search": {"message_count": 1}}

    evidence = ContextEvidence(
        messages=[
            {
                "role": "system",
                "content": (
                    "后来改稿这一段基本都删掉了。只有一个真的。"
                    "匣中有门，心中有钥。"
                ),
            }
        ],
        included_layers=Trace.included_layers,
        source_ids=Trace.source_ids,
        context_hash="d" * 64,
    )

    _validate_context_requirements(
        {
            "required_source_ids": ["kmlog:7553"],
            "required_visible_text": [
                "后来改稿这一段基本都删掉了",
                "只有一个真的",
                "匣中有门，心中有钥",
            ],
        },
        Trace(),
        evidence,
    )


def test_answer_expectations_are_recorded_without_raising():
    class Answer:
        evidence_status = "supported"
        claim_type = "unknown"

    checks = _evaluate_answer_expectations(
        {
            "expected_evidence_status": "insufficient",
            "expected_claim_type": "personal_statement",
        },
        Answer(),
    )

    assert checks == {
        "evidence_status": {
            "expected": "insufficient",
            "actual": "supported",
            "passed": False,
        },
        "claim_type": {
            "expected": "personal_statement",
            "actual": "unknown",
            "passed": False,
        },
    }


def test_checked_in_manifest_covers_reviewed_canary_controls():
    manifest_path = Path(__file__).parents[1] / "agent_canary" / "seeds.json"
    cases = json.loads(manifest_path.read_text(encoding="utf-8"))
    by_id = {case["id"]: case for case in cases}

    assert set(by_id) == {
        "tang-forged-order",
        "guan-yi-old-draft-key",
        "ai-self-over-perfect-response",
        "plot-evidence-handoff-rejection",
    }
    assert by_id["tang-forged-order"]["required_source_ids"] == ["kmlog:28955"]
    assert by_id["guan-yi-old-draft-key"]["required_source_ids"] == [
        "kmlog:7553"
    ]
    assert by_id["guan-yi-old-draft-key"]["required_visible_text"] == [
        "后来改稿这一段基本都删掉了",
        "只有一个真的",
        "匣中有门，心中有钥",
    ]
    assert by_id["guan-yi-old-draft-key"]["timeout_seconds"] == 30
    assert by_id["ai-self-over-perfect-response"]["required_source_ids"] == [
        "kmlog:18108"
    ]
    rejection = by_id["plot-evidence-handoff-rejection"]
    assert rejection["forbidden_layers"] == ["kmlog_search"]
    assert rejection["as_of_timestamp"] == "2025-12-10T00:00:08.335+00:00"
    assert rejection["timeout_seconds"] == 30


@pytest.mark.anyio
async def test_agent_uses_prefetched_context_without_second_http_request():
    evidence = ContextEvidence(
        messages=[{"role": "system", "content": "Prefetched evidence."}],
        included_layers=["kmlog_search"],
        source_ids=["kmlog:42"],
        context_hash="a" * 64,
    )
    trace = ContextToolTrace(
        conversation_id="agent-canary-test",
        latency_ms=12,
        source_ids=["kmlog:42"],
        included_layers=["kmlog_search"],
        context_hash="a" * 64,
    )
    deps = AgentDeps(
        context_client=BuildContextClient("http://must-not-be-called.invalid"),
        prompt="What happened?",
        conversation_id="agent-canary-test",
        prefetched_context=(evidence, trace),
    )

    await build_agent(TestModel()).run("What happened?", deps=deps)

    assert deps.prefetched_context is None
    assert deps.tool_traces == [trace]


def test_artifact_snapshot_and_jsonl_preserve_visible_evidence(tmp_path):
    evidence = ContextEvidence(
        messages=[{"role": "system", "content": "伪造书信与异常药方"}],
        included_layers=["kmlog_search"],
        source_ids=["kmlog:27581"],
        context_hash="b" * 64,
    )
    snapshot = evidence_snapshot(evidence)
    output = tmp_path / "canary.jsonl"

    append_jsonl(output, {"status": "completed", "evidence": snapshot})
    append_jsonl(output, {"status": "error", "stage": "preflight"})

    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 2
    assert records[0]["evidence"]["messages"][0]["content"] == "伪造书信与异常药方"
    assert records[0]["evidence"]["messages"][0]["chars"] == 9
    assert len(records[0]["evidence"]["messages"][0]["content_hash"]) == 64
    assert records[1] == {"stage": "preflight", "status": "error"}


def test_destination_redaction_removes_query_and_fragment():
    assert (
        redact_destination("https://example.test/v1?api_key=secret#fragment")
        == "https://example.test/v1"
    )
