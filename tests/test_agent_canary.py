import json

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
from agent_canary.runner import _jsonable, _validate_context_requirements


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
            },
            Trace(),
        )


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
