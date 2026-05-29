from pathlib import Path

from chat_proxy.config import ProxyConfig
from chat_proxy.context_builder import (
    context_packet_from_snapshot,
    render_to_openai_messages,
    build_web_chat_context,
)


class DummyStore:
    def get_recent_messages(self, **_kwargs):
        return []


def test_context_packet_renders_openai_messages_and_filters_bad_roles():
    packet = context_packet_from_snapshot(
        snapshot={
            "source": "test",
            "mode": "explicit_messages",
            "model": "gpt-test",
            "components": [{"name": "recent_turns", "message_count": 1}],
            "final_message_count": 3,
            "final_chars": 12,
        },
        messages=[
            {"role": "system", "content": "rules"},
            {"role": "narrator", "content": "skip"},
            {"role": "user", "content": "hello"},
        ],
    )

    assert render_to_openai_messages(packet) == [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "hello"},
    ]
    assert packet.to_dict()["messages"] == render_to_openai_messages(packet)


def test_explicit_messages_are_trimmed_but_system_is_kept():
    cfg = ProxyConfig(
        upstream_base="http://upstream",
        db_path=Path("dummy.db"),
        explicit_messages_recent_k=2,
    )
    result = build_web_chat_context(
        body={
            "model": "gpt-test",
            "messages": [
                {"role": "system", "content": "system rules"},
                {"role": "user", "content": "old user"},
                {"role": "assistant", "content": "old assistant"},
                {"role": "user", "content": "recent user"},
                {"role": "assistant", "content": "recent assistant"},
            ],
        },
        cfg=cfg,
        store=DummyStore(),
        headers={},
    )

    assert result.upstream_body["messages"] == [
        {"role": "system", "content": "system rules"},
        {"role": "user", "content": "recent user"},
        {"role": "assistant", "content": "recent assistant"},
    ]
    explicit = next(
        item
        for item in result.snapshot["components"]
        if item["name"] == "explicit_messages"
    )
    assert explicit["trim_enabled"] is True
    assert explicit["trimmed_conversational_count"] == 2
    assert explicit["forwarded_message_count"] == 3


def test_core_anchor_keyword_triggers_prioritize_context_specific_anchors(monkeypatch):
    calls = []

    class FakeResponse:
        def __init__(self, results):
            self._results = results

        def raise_for_status(self):
            return None

        def json(self):
            return {"results": self._results}

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def get(self, url, headers=None, params=None):
            calls.append({"url": url, "headers": headers, "params": params})
            if params.get("anchor_key") == "kmlog_cofounder":
                return FakeResponse(
                    [
                        {
                            "anchor_key": "kmlog_cofounder",
                            "title": "KMLog cofounder",
                            "content": "Kai keeps KMLog continuity.",
                            "function": "boot_core",
                            "priority": 2,
                        }
                    ]
                )
            if params.get("function") == "soothe_panic":
                return FakeResponse(
                    [
                        {
                            "anchor_key": "panic_ground",
                            "title": "Panic grounding",
                            "content": "Slow down and help Mei feel held.",
                            "function": "soothe_panic",
                            "priority": 1,
                        }
                    ]
                )
            return FakeResponse(
                [
                    {
                        "anchor_key": "multi_model_same_kai",
                        "title": "Same Kai",
                        "content": "Same Kai across carriers.",
                        "function": "boot_core",
                        "priority": 1,
                    }
                ]
            )

    monkeypatch.setattr("chat_proxy.context_builder.httpx.Client", FakeClient)
    cfg = ProxyConfig(
        upstream_base="http://upstream",
        db_path=Path("dummy.db"),
        core_anchors_enabled=True,
        core_anchors_url="http://kmlog",
        core_anchors_boot_keys=("multi_model_same_kai",),
        core_anchors_boot_max=3,
    )
    result = build_web_chat_context(
        body={
            "model": "gpt-test",
            "messages": [
                {"role": "user", "content": "infra 这块我有点 panic"}
            ],
        },
        cfg=cfg,
        store=DummyStore(),
        headers={},
    )

    anchor_message = result.upstream_body["messages"][0]
    assert anchor_message["role"] == "system"
    assert "kmlog_cofounder" in anchor_message["content"]
    assert "panic_ground" in anchor_message["content"]
    assert "multi_model_same_kai" in anchor_message["content"]
    assert [call["params"] for call in calls] == [
        {"status": "active", "limit": 20, "function": "boot_core"},
        {"status": "active", "limit": 20, "function": "soothe_panic"},
        {"status": "active", "limit": 20, "anchor_key": "kmlog_cofounder"},
    ]
    snapshot = result.snapshot["components"][0]
    assert snapshot["triggered_functions"] == ["soothe_panic"]
    assert snapshot["triggered_keys"] == ["kmlog_cofounder"]
    assert snapshot["boot_keys"] == [
        "kmlog_cofounder",
        "panic_ground",
        "multi_model_same_kai",
    ]


def test_core_anchor_functions_can_be_requested_from_context_builder_params(monkeypatch):
    called_functions = []

    class FakeResponse:
        def __init__(self, results):
            self._results = results

        def raise_for_status(self):
            return None

        def json(self):
            return {"results": self._results}

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def get(self, _url, headers=None, params=None):
            function = params.get("function")
            if function:
                called_functions.append(function)
            if function == "infra_reference":
                return FakeResponse(
                    [
                        {
                            "anchor_key": "infra_map",
                            "title": "Infra map",
                            "content": "Keep infrastructure context concrete.",
                            "function": "infra_reference",
                            "priority": 1,
                        }
                    ]
                )
            return FakeResponse([])

    monkeypatch.setattr("chat_proxy.context_builder.httpx.Client", FakeClient)
    cfg = ProxyConfig(
        upstream_base="http://upstream",
        db_path=Path("dummy.db"),
        core_anchors_enabled=True,
        core_anchors_url="http://kmlog",
        core_anchors_boot_max=2,
    )
    result = build_web_chat_context(
        body={
            "model": "gpt-test",
            "messages": [{"role": "user", "content": "continue work"}],
            "params": {"core_anchor_functions": ["infra_reference", "unknown"]},
        },
        cfg=cfg,
        store=DummyStore(),
        headers={},
    )

    assert called_functions == ["boot_core", "infra_reference"]
    assert "infra_map" in result.upstream_body["messages"][0]["content"]
    snapshot = result.snapshot["components"][0]
    assert snapshot["requested_functions"] == ["infra_reference"]
