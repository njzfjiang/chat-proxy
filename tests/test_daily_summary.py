import json

from chat_proxy.daily_summary import (
    _daily_summary_prompt,
    _parse_daily_summary_result,
    date_key_for,
)


def test_daily_summary_prompt_includes_memory_mapping_schema():
    prompt = _daily_summary_prompt(
        "2026-05-16",
        "Old day",
        [
            {
                "id": 12,
                "timestamp": "2026-05-16T16:00:00+00:00",
                "conversation_title": "Kai",
                "conversation_id": "chat-1",
                "role": "user",
                "content": "We agreed to audit memory candidates first.",
            }
        ],
    )

    assert "Daily date key: 2026-05-16" in prompt
    assert "memory_candidates" in prompt
    assert "health_safety" in prompt
    assert "daily_context" in prompt
    assert "primary_mother A=User Profile" in prompt
    assert "[12]" in prompt
    assert "audit memory candidates" in prompt


def test_parse_daily_summary_result_normalizes_candidates():
    result = _parse_daily_summary_result(
        json.dumps(
            {
                "summary": "Day overview\n- Built the first version.",
                "memory_candidates": [
                    {
                        "label": "Daily summary module",
                        "evidence": "Implemented audit candidates.",
                        "domain": "infra_asset",
                        "function": "infra_reference",
                        "primary_mother": "D",
                        "secondary_mother": "E",
                        "importance": 9,
                        "confidence": "medium",
                        "source_message_ids": ["1", 2, "x"],
                    },
                    {
                        "label": "Bad labels fall back",
                        "domain": "unknown",
                        "function": "unknown",
                        "primary_mother": "Z",
                    },
                ],
            }
        )
    )

    assert result["summary"].startswith("Day overview")
    first = result["memory_candidates"][0]
    assert first["domain"] == "infra_asset"
    assert first["function"] == "infra_reference"
    assert first["primary_mother"] == "D"
    assert first["importance"] == 5
    assert first["source_message_ids"] == [1, 2]

    second = result["memory_candidates"][1]
    assert second["domain"] == "everyday_slice"
    assert second["function"] == "daily_context"
    assert second["primary_mother"] == "E"


def test_date_key_for_uses_configured_timezone():
    assert date_key_for("2026-05-16T03:30:00+00:00", "America/Toronto") == "2026-05-15"
