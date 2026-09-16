from benchmarks.context_selection.build_selector_abc import evaluate_seed


def _candidate(candidate_id, text, terms=()):
    return {
        "id": candidate_id,
        "timestamp": "2026-01-01T00:00:00Z",
        "role": "user",
        "content_preview": text,
        "matched_excerpt": text,
        "content_hash": str(candidate_id),
        "evidence_version": 1,
        "body_matched_terms": list(terms),
        "evidence_excerpts": [{"text": text}],
    }


def test_selector_abc_uses_common_pool_and_shared_render_budget():
    snapshot = {
        "candidate_items": [
            _candidate(1, "generic background"),
            _candidate(2, "FISTA experiment completed", ("FISTA",)),
        ],
        "rerank_input_ids": [1],
        "plan": {
            "sources": ["chat_history_search"],
            "required_terms": [],
            "optional_terms": ["FISTA"],
        },
    }
    selector_row = {
        "seed_id": "10",
        "theme": "course_project",
        "input_candidate_ids": [1, 2],
        "selected": [
            {
                "id": 2,
                "evidence_excerpt_indices": [0],
                "reason": "direct event",
            }
        ],
    }

    result = evaluate_seed(
        snapshot=snapshot,
        selector_row=selector_row,
        selection_limit=1,
        max_candidate_chars=600,
        total_chars=30,
    )

    groups = result["groups"]
    assert groups["a_backend_pool_rule"]["selected_after_budget_ids"] == [1]
    assert groups["b_common_pool_rule"]["selected_after_budget_ids"] == [2]
    assert groups["c_common_pool_model"]["selected_after_budget_ids"] == [2]
    assert groups["b_common_pool_rule"]["input_ids"] == [1, 2]
    assert groups["c_common_pool_model"]["input_ids"] == [1, 2]
    assert groups["b_common_pool_rule"]["rendered_body_chars"] <= 30
    assert groups["c_common_pool_model"]["rendered_body_chars"] <= 30
