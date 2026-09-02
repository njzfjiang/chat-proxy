import json

from chat_proxy.context_selection_benchmark import _write_outputs


def test_write_outputs_summarizes_string_boolean_fields(tmp_path):
    results = [
        {
            "theme": "memory",
            "message_id": "1",
            "status": "ok",
            "error": "",
            "future_leak": "False",
            "expected_retrieval": "True",
            "expected_no_context": "False",
            "retrieval_result_count": "2",
            "recent_turn_count": "3",
            "total_token_estimate": "100",
        },
        {
            "theme": "social",
            "message_id": "2",
            "status": "ok",
            "error": "",
            "future_leak": "False",
            "expected_retrieval": "False",
            "expected_no_context": "True",
            "retrieval_result_count": "0",
            "recent_turn_count": "1",
            "total_token_estimate": "20",
        },
    ]

    paths = _write_outputs(tmp_path, results)
    summary = json.loads(paths["summary_json"].read_text(encoding="utf-8"))

    assert summary["future_leak_count"] == 0
    assert summary["expected_retrieval_seed_count"] == 1
    assert summary["expected_retrieval_nonempty_count"] == 1
    assert summary["unexpected_retrieval_count"] == 0
    assert summary["no_context_seed_count"] == 1
    assert summary["no_context_with_retrieval_count"] == 0
