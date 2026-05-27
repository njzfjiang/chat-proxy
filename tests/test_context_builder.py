from pathlib import Path

from chat_proxy.config import ProxyConfig
from chat_proxy.context_builder import build_web_chat_context


class DummyStore:
    def get_recent_messages(self, **_kwargs):
        return []


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
