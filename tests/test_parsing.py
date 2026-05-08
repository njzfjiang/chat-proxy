from chat_proxy.parsing import (
    SseTextAccumulator,
    extract_chat_completion_text,
    last_user_text,
    message_id_for,
    prepare_request_body_for_upstream,
    resolve_conversation,
)


def test_resolve_conversation_prefers_headers():
    identity = resolve_conversation(
        {
            "X-Kelivo-Conversation-Id": "chat-1",
            "X-Kelivo-Client-Id": "desktop",
            "X-Kelivo-Assistant-Key": "kai",
            "X-Kelivo-Provider-Key": "openai",
        },
        {"model": "gpt-test", "messages": []},
    )

    assert identity.conversation_id == "chat-1"
    assert identity.resolver == "header"
    assert identity.client_key == "desktop"
    assert identity.assistant_key == "kai"
    assert identity.provider_key == "openai"


def test_resolve_conversation_uses_system_meta():
    identity = resolve_conversation(
        {},
        {
            "model": "gpt-test",
            "messages": [
                {
                    "role": "system",
                    "content": "[kelivo_meta]\nclient=phone\nassistant=kai\nprovider=deepseek\nconversation=kai-main\n[/kelivo_meta]",
                }
            ],
        },
    )

    assert identity.conversation_id == "kai-main"
    assert identity.resolver == "system_meta"
    assert identity.client_key == "phone"
    assert identity.assistant_key == "kai"
    assert identity.provider_key == "deepseek"


def test_resolve_conversation_uses_kelivo_analysis_meta():
    identity = resolve_conversation(
        {"X-Kelivo-Analysis-Version": "1"},
        {
            "model": "gpt-test",
            "_kelivo_analysis_meta": {
                "conversation_id": "chat-dev",
                "conversation_title": "Dev Chat",
                "assistant_id": "assistant-a",
                "provider_key": "openai",
                "model_id": "gpt-4.1",
            },
            "messages": [],
        },
    )

    assert identity.conversation_id == "chat-dev"
    assert identity.resolver == "kelivo_analysis"
    assert identity.assistant_key == "assistant-a"
    assert identity.provider_key == "openai"
    assert identity.metadata["conversation_title"] == "Dev Chat"


def test_resolve_conversation_dev_meta_falls_back_to_system_client():
    identity = resolve_conversation(
        {"X-Kelivo-Analysis-Version": "1"},
        {
            "model": "gpt-test",
            "_kelivo_analysis_meta": {
                "conversation_id": "chat-dev",
                "assistant_id": "assistant-a",
            },
            "messages": [
                {
                    "role": "system",
                    "content": "[kelivo_meta]\nclient=desktop-prod\nprovider=deepseek\n[/kelivo_meta]",
                }
            ],
        },
    )

    assert identity.conversation_id == "chat-dev"
    assert identity.resolver == "kelivo_analysis"
    assert identity.client_key == "desktop-prod"
    assert identity.provider_key == "deepseek"
    assert identity.metadata["system_meta"]["client"] == "desktop-prod"


def test_resolve_conversation_dev_meta_accepts_client_key_header_alias():
    identity = resolve_conversation(
        {
            "X-Kelivo-Analysis-Version": "1",
            "X-Client-Key": "desktop-custom",
        },
        {
            "model": "gpt-test",
            "_kelivo_analysis_meta": {"conversation_id": "chat-dev"},
            "messages": [],
        },
    )

    assert identity.client_key == "desktop-custom"


def test_resolve_conversation_dev_meta_generates_client_when_missing():
    identity = resolve_conversation(
        {"X-Kelivo-Analysis-Version": "1"},
        {
            "model": "gpt-test",
            "_kelivo_analysis_meta": {"conversation_id": "chat-dev"},
            "messages": [],
        },
    )

    assert identity.client_key.startswith("proxy_assigned_client_")
    assert identity.metadata["client_key"] == identity.client_key


def test_prepare_request_body_for_upstream_strips_kelivo_analysis_meta():
    prepared = prepare_request_body_for_upstream(
        {"X-Kelivo-Analysis-Version": "1"},
        {
            "model": "gpt-test",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": True,
            "_kelivo_analysis_meta": {"turn_id": "turn-1"},
            "extra_body": {"debug": True},
        },
    )

    assert prepared.mode == "kelivo_analysis"
    assert prepared.stripped_metadata == {"turn_id": "turn-1"}
    assert prepared.body == {
        "model": "gpt-test",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": True,
    }


def test_resolve_conversation_assigns_fallback():
    identity = resolve_conversation({}, {"model": "gpt-test", "messages": []})

    assert identity.conversation_id.startswith("proxy_assigned_")
    assert identity.resolver == "proxy_assigned"


def test_last_user_text_handles_string_and_multimodal_content():
    assert (
        last_user_text(
            {
                "messages": [
                    {"role": "user", "content": "old"},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "hello"},
                            {"type": "image_url", "image_url": {"url": "x"}},
                        ],
                    },
                ]
            }
        )
        == "hello\n[image]"
    )


def test_last_user_text_strips_leading_timestamp_line():
    assert (
        last_user_text(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "2026-05-07 21:40 \r\nmessage",
                    },
                ]
            }
        )
        == "message"
    )


def test_last_user_text_allows_empty_messages():
    assert last_user_text({"messages": []}) is None
    assert last_user_text({}) is None


def test_extract_chat_completion_text():
    assert (
        extract_chat_completion_text(
            {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}
        )
        == "ok"
    )


def test_message_id_is_stable():
    first = message_id_for(
        request_id="req-1",
        conversation_id="chat-1",
        role="user",
        content="hello",
    )
    second = message_id_for(
        request_id="req-1",
        conversation_id="chat-1",
        role="user",
        content="hello",
    )

    assert first == second


def test_sse_accumulator_reads_openai_delta_text():
    acc = SseTextAccumulator()
    acc.add_bytes(b'data: {"choices":[{"delta":{"content":"hel"}}]}\n\n')
    acc.add_bytes(b'data: {"choices":[{"delta":{"content":"lo"}}]}\n\n')
    acc.add_bytes(b"data: [DONE]\n\n")

    assert acc.text == "hello"
