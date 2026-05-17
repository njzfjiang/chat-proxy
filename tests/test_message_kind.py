from chat_proxy.message_kind import classify_message_kind


def test_classifies_http_errors_as_noise():
    assert (
        classify_message_kind(
            content='HttpException: HTTP 401: {"error": "Incorrect API key"}',
            conversation_title="Kai",
        )
        == "noise"
    )


def test_classifies_numbered_summary_cards_as_summary():
    assert (
        classify_message_kind(
            content="""----------

## 172. Window length and lag

**Tag:** Setting
**Timestamp:** 2025-12-08
**What happened:**

The user noticed the window was long.
""",
            conversation_title="Kai",
        )
        == "summary"
    )


def test_classifies_summary_mechanism_discussion_as_meta():
    assert (
        classify_message_kind(
            content="Let's tune the rolling summary design and SQLite import.",
            conversation_title="Kai",
        )
        == "meta"
    )
