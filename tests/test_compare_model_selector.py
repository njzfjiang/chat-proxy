from benchmarks.context_selection.compare_model_selector import compare_selector_results


def test_compare_selector_results_reports_id_changes_without_quality_claims():
    selector_rows = [
        {
            "seed_id": "10",
            "status": "ok",
            "selected": [{"id": 2}, {"id": 3}],
            "usage": {"total_tokens": 20},
            "elapsed_ms": 100,
        },
        {
            "seed_id": "20",
            "status": "ok",
            "selected": [],
            "reject_all_reason": "No match.",
            "usage": {"total_tokens": 10},
            "elapsed_ms": 300,
        },
        {
            "seed_id": "30",
            "status": "error",
            "selected": [],
            "elapsed_ms": 200,
        },
    ]
    evidence_rows = [
        {"message_id": "10", "retrieval_selected_after_budget_ids": "1|2"},
        {"message_id": "20", "retrieval_selected_after_budget_ids": "8"},
        {"message_id": "30", "retrieval_selected_after_budget_ids": "9"},
    ]

    result = compare_selector_results(selector_rows, evidence_rows)

    assert result["changed_seed_count"] == 2
    assert result["error_count"] == 1
    assert result["model_only_selected_count"] == 1
    assert result["seed_with_model_only_count"] == 1
    assert result["reject_all_seed_ids"] == ["20"]
    assert result["total_tokens"] == 30
    assert result["average_elapsed_ms"] == 200.0
    assert result["comparisons"][0]["overlap_ids"] == [2]
    assert result["comparisons"][0]["model_only_ids"] == [3]
