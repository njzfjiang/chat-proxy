from chat_proxy.summary import _summary_prompt


def test_summary_prompt_requires_short_semi_structured_format():
    prompt = _summary_prompt(
        "Old context",
        [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ],
    )

    assert "Now: 1-3 sentences" in prompt
    assert "Key context: up to 5 bullets" in prompt
    assert "Open threads: up to 5 bullets" in prompt
    assert "Style / protocols: up to 5 bullets" in prompt
    assert "under about 180 words" in prompt
    assert "Return only the updated rolling summary." in prompt
    assert "user: hello" in prompt
    assert "assistant: hi" in prompt
