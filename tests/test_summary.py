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
    assert 'Any section may be "None"' in prompt
    assert "Prefer omission over inference" in prompt
    assert "Current raw conversation is primary evidence" in prompt
    assert "must never override, reinterpret, or strengthen" in prompt
    assert "Do not infer near-term goals from mood" in prompt
    assert "Do not promote a one-off assistant response" in prompt
    assert "Do not treat casual assistant questions" in prompt
    assert "unless the user explicitly accepted them" in prompt
    assert "Never add a `[verified]` marker yourself" in prompt
    assert "otherwise drop it." in prompt
    assert "mark it uncertain" not in prompt
    assert "still extract at least 1 bullet" not in prompt
    assert "Return only the updated rolling summary." in prompt
    assert "user: hello" in prompt
    assert "assistant: hi" in prompt


def test_summary_prompt_requires_explicit_user_support_for_durable_claims():
    prompt = _summary_prompt(
        "Style / protocols:\n- Assistant inferred that reassurance is always preferred.",
        [
            {"role": "user", "content": "I am tired today."},
            {"role": "assistant", "content": "I will keep things gentle."},
        ],
    )

    assert "A repeated pattern is not a durable preference" in prompt
    assert "only intentions or goals the user explicitly stated or confirmed" in prompt
    assert "user-stated or user-confirmed interaction preferences" in prompt
    assert "unless they are necessary for a currently unresolved" in prompt
