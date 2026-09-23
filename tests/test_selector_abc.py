from benchmarks.context_selection.build_selector_abc import evaluate_seed
from benchmarks.context_selection.run_local_evidence_ab import (
    MANIFEST_CODE_PATHS,
    ROOT,
)


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


def test_selector_abc_rule_and_model_use_the_same_visible_candidate_body():
    hidden_tail = "SOURCE_ONLY_TAIL"
    source_text = "visible " + ("x" * 700) + hidden_tail
    snapshot = {
        "candidate_items": [_candidate(1, source_text)],
        "rerank_input_ids": [1],
        "plan": {"sources": ["chat_history_search"]},
    }
    selector_row = {
        "seed_id": "10",
        "theme": "visibility_parity",
        "input_candidate_ids": [1],
        "selected": [
            {
                "id": 1,
                "evidence_excerpt_indices": [0],
                "reason": "only visible candidate",
            }
        ],
    }

    result = evaluate_seed(
        snapshot=snapshot,
        selector_row=selector_row,
        selection_limit=1,
        max_candidate_chars=600,
        total_chars=1200,
    )

    groups = result["groups"]
    rule = groups["b_common_pool_rule"]
    model = groups["c_common_pool_model"]
    assert rule["candidate_representation"] == "selector_visible_v1"
    assert model["candidate_representation"] == "selector_visible_v1"
    assert rule["rendered_content"] == model["rendered_content"]
    assert rule["rendered_items"][0]["source_chars"] == 600
    assert model["rendered_items"][0]["source_chars"] == 600
    assert hidden_tail not in rule["rendered_content"]
    assert hidden_tail not in model["rendered_content"]


def test_selector_abc_audits_legacy_declared_candidates_without_visible_evidence():
    invalid = _candidate(2, "not serializable")
    invalid["evidence_excerpts"] = []
    snapshot = {
        "candidate_items": [_candidate(1, "visible evidence"), invalid],
        "rerank_input_ids": [1],
        "plan": {"sources": ["chat_history_search"]},
    }
    selector_row = {
        "seed_id": "10",
        "theme": "legacy_manifest",
        "input_candidate_ids": [1, 2],
        "selected": [
            {
                "id": 1,
                "evidence_excerpt_indices": [0],
                "reason": "only serializable candidate",
            }
        ],
    }

    result = evaluate_seed(
        snapshot=snapshot,
        selector_row=selector_row,
        selection_limit=1,
        max_candidate_chars=600,
        total_chars=1200,
    )

    audit = result["candidate_pool_audit"]
    assert audit["declared_input_ids"] == [1, 2]
    assert audit["visible_input_ids"] == [1]
    assert audit["dropped_nonserializable_ids"] == [2]
    assert result["groups"]["b_common_pool_rule"]["input_ids"] == [1]
    assert result["groups"]["c_common_pool_model"]["input_ids"] == [1]


def test_evidence_benchmark_manifest_tracks_runner_and_renderer():
    assert ROOT / "benchmarks/context_selection/run_local_evidence_ab.py" in (
        MANIFEST_CODE_PATHS
    )
    assert ROOT / "chat_proxy/retrieval_evidence.py" in MANIFEST_CODE_PATHS
