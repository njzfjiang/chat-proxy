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


def test_render_boundary_sanitizes_injected_half_snippets():
    packet = context_packet_from_snapshot(
        snapshot={"source": "test", "components": []},
        messages=[
            {
                "role": "system",
                "content": "[Core Anchors / active]\n- safe: canonical（5.",
            },
            {"role": "user", "content": "hello"},
        ],
    )

    rendered = render_to_openai_messages(packet)
    assert rendered[0]["content"] == "[Core Anchors / active]"
    assert "canonical（5." not in rendered[0]["content"]


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
    assert snapshot["trigger_input_sources"] == ["current_user"]
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


def test_continuity_loss_keywords_trigger_soothe_panic(monkeypatch):
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

        def get(self, _url, headers=None, params=None):
            calls.append(params)
            if params.get("function") == "soothe_panic":
                return FakeResponse(
                    [
                        {
                            "anchor_key": "continuity_soothe",
                            "title": "Continuity soothe",
                            "content": "Reassure continuity gently.",
                            "function": "soothe_panic",
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
            "messages": [{"role": "user", "content": "模型下架了会不会消失"}],
        },
        cfg=cfg,
        store=DummyStore(),
        headers={},
    )

    snapshot = result.snapshot["components"][0]
    assert snapshot["triggered_functions"] == ["soothe_panic"]
    assert any(call.get("function") == "soothe_panic" for call in calls)
    assert "continuity_soothe" in result.upstream_body["messages"][0]["content"]


def test_triggers_ignore_assistant_and_system_text(monkeypatch, tmp_path):
    worldbook_path = tmp_path / "wb.json"
    worldbook_path.write_text(
        """
{
  "entries": [
    {
      "id": "infra-wb",
      "name": "Infra WB",
      "keywords": ["infra"],
      "content": "This should not trigger from assistant text."
    }
  ]
}
""".strip(),
        encoding="utf-8",
    )
    calls = []

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"results": []}

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def get(self, _url, headers=None, params=None):
            calls.append(params)
            return FakeResponse()

    monkeypatch.setattr("chat_proxy.context_builder.httpx.Client", FakeClient)
    cfg = ProxyConfig(
        upstream_base="http://upstream",
        db_path=Path("dummy.db"),
        worldbook_enabled=True,
        worldbook_paths=(worldbook_path,),
        core_anchors_enabled=True,
        core_anchors_url="http://kmlog",
    )
    result = build_web_chat_context(
        body={
            "model": "gpt-test",
            "messages": [
                {"role": "system", "content": "panic infra"},
                {"role": "assistant", "content": "panic infra"},
                {"role": "user", "content": "hello"},
            ],
        },
        cfg=cfg,
        store=DummyStore(),
        headers={},
    )

    anchor = next(
        item for item in result.snapshot["components"] if item["name"] == "core_anchors"
    )
    wb = next(
        item for item in result.snapshot["components"] if item["name"] == "wb_snippets"
    )
    assert anchor["triggered_functions"] == []
    assert anchor["triggered_keys"] == []
    assert anchor["trigger_input_sources"] == ["current_user"]
    assert wb["message_count"] == 0
    assert wb["trigger_input_sources"] == ["current_user"]
    assert calls == [{"status": "active", "limit": 20, "function": "boot_core"}]


def test_core_anchor_uses_compact_summary_and_skips_half_sentence(monkeypatch):
    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "results": [
                    {
                        "anchor_key": "long_anchor",
                        "title": "Long",
                        "compact_summary": "First complete sentence. Second complete sentence.",
                        "content": "Ignored content that should not be used.",
                        "function": "boot_core",
                        "priority": 1,
                    }
                ]
            }

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def get(self, _url, headers=None, params=None):
            return FakeResponse()

    monkeypatch.setattr("chat_proxy.context_builder.httpx.Client", FakeClient)
    cfg = ProxyConfig(
        upstream_base="http://upstream",
        db_path=Path("dummy.db"),
        core_anchors_enabled=True,
        core_anchors_url="http://kmlog",
        core_anchors_boot_keys=("long_anchor",),
        core_anchors_chars_total=42,
    )
    result = build_web_chat_context(
        body={"model": "gpt-test", "messages": [{"role": "user", "content": "hello"}]},
        cfg=cfg,
        store=DummyStore(),
        headers={},
    )

    content = result.upstream_body["messages"][0]["content"]
    assert "First complete sentence." in content
    assert "Second complete" not in content
    assert "Ignored content" not in content


def test_worldbook_uses_compact_summary_and_structured_warnings(tmp_path):
    missing_path = tmp_path / "missing.json"
    worldbook_path = tmp_path / "wb.json"
    worldbook_path.write_text(
        """
{
  "entries": [
    {
      "id": "care",
      "name": "Care",
      "keywords": ["care"],
      "compact_summary": "Compact WB sentence. Another sentence.",
      "content": "Long original content should not appear."
    }
  ]
}
""".strip(),
        encoding="utf-8",
    )
    cfg = ProxyConfig(
        upstream_base="http://upstream",
        db_path=Path("dummy.db"),
        worldbook_enabled=True,
        worldbook_paths=(missing_path, worldbook_path),
        worldbook_chars_total=34,
    )
    result = build_web_chat_context(
        body={
            "model": "gpt-test",
            "messages": [{"role": "user", "content": "care please"}],
        },
        cfg=cfg,
        store=DummyStore(),
        headers={},
    )

    wb = next(
        item for item in result.snapshot["components"] if item["name"] == "wb_snippets"
    )
    assert wb["warnings"][0]["code"] == "worldbook_load_failed"
    assert "errors" not in wb
    assert wb["items"][0]["used_compact_summary"] is True
    content = result.upstream_body["messages"][0]["content"]
    assert "Compact WB sentence." in content
    assert "Another sentence" not in content
    assert "Long original content" not in content


def test_worldbook_ascii_keywords_use_token_boundaries(tmp_path):
    worldbook_path = tmp_path / "wb.json"
    worldbook_path.write_text(
        """
{
  "entries": [
    {
      "id": "review",
      "name": "Review",
      "keywords": ["review"],
      "content": "Only a standalone review should trigger."
    }
  ]
}
""".strip(),
        encoding="utf-8",
    )
    cfg = ProxyConfig(
        upstream_base="http://upstream",
        db_path=Path("dummy.db"),
        worldbook_enabled=True,
        worldbook_paths=(worldbook_path,),
    )
    preview_result = build_web_chat_context(
        body={
            "model": "gpt-test",
            "messages": [{"role": "user", "content": "please preview this"}],
        },
        cfg=cfg,
        store=DummyStore(),
        headers={},
    )
    review_result = build_web_chat_context(
        body={
            "model": "gpt-test",
            "messages": [{"role": "user", "content": "please review this"}],
        },
        cfg=cfg,
        store=DummyStore(),
        headers={},
    )

    preview_wb = next(
        item
        for item in preview_result.snapshot["components"]
        if item["name"] == "wb_snippets"
    )
    review_wb = next(
        item
        for item in review_result.snapshot["components"]
        if item["name"] == "wb_snippets"
    )
    assert preview_wb["message_count"] == 0
    assert preview_wb["trigger_matches"] == []
    assert review_wb["message_count"] == 1
    assert review_wb["trigger_matches"][0]["keyword"] == "review"
    assert review_wb["trigger_matches"][0]["span"] == [7, 13]
    assert "please review this" in review_wb["trigger_matches"][0]["excerpt"]


def test_injected_snippets_drop_trailing_half_bullets(tmp_path):
    worldbook_path = tmp_path / "wb.json"
    worldbook_path.write_text(
        """
{
  "entries": [
    {
      "id": "tool",
      "name": "Tool",
      "keywords": ["tool"],
      "compact_summary": "- done（5）. - tool（5.",
      "content": "Ignored."
    }
  ]
}
""".strip(),
        encoding="utf-8",
    )
    cfg = ProxyConfig(
        upstream_base="http://upstream",
        db_path=Path("dummy.db"),
        worldbook_enabled=True,
        worldbook_paths=(worldbook_path,),
    )
    result = build_web_chat_context(
        body={
            "model": "gpt-test",
            "messages": [{"role": "user", "content": "tool"}],
        },
        cfg=cfg,
        store=DummyStore(),
        headers={},
    )

    content = result.upstream_body["messages"][0]["content"]
    assert "- done（5）." in content
    assert "- tool（5." not in content
