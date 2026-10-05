import json
from pathlib import Path

from benchmarks.context_selection.audit_required_term_filter import audit_cases
from benchmarks.context_selection.run_local_evidence_ab import _apply_gold_labels


ROOT = Path(__file__).resolve().parents[1]
ANNOTATIONS = ROOT / "benchmarks/context_selection/annotations"


def test_reviewed_gold_label_overrides_8422_without_mutating_input():
    seeds = [
        {
            "seed_message_id": "8422",
            "message_id": "8422",
            "expected_context": "recent+episodic",
        },
        {
            "seed_message_id": "17699",
            "message_id": "17699",
            "expected_context": "episodic",
        },
    ]

    updated, applied = _apply_gold_labels(
        seeds, ANNOTATIONS / "gold_labels_v2.json"
    )

    assert seeds[0]["expected_context"] == "recent+episodic"
    assert updated[0]["expected_context"] == "recent"
    assert seeds[1]["expected_context"] == "episodic"
    assert updated[1]["expected_context"] == "recent"
    assert applied == [
        {
            "seed_id": "8422",
            "old_expected_context": "recent+episodic",
            "new_expected_context": "recent",
        },
        {
            "seed_id": "17699",
            "old_expected_context": "episodic",
            "new_expected_context": "recent",
        },
    ]


def test_candidate_change_annotations_cover_the_saved_14_seed_delta():
    payload = json.loads(
        (ANNOTATIONS / "candidate_change_annotations_v1.json").read_text(
            encoding="utf-8"
        )
    )

    assert {item["seed_id"] for item in payload["labels"]} == {
        "831",
        "4249",
        "4239",
        "5705",
        "985",
        "10497",
        "10891",
        "12762",
        "522",
        "17699",
        "33394",
        "10741",
        "18108",
        "31847",
    }


def test_required_term_audit_accepts_known_positive_paraphrases():
    result = audit_cases()

    assert result["case_count"] == 4
    assert result["false_negative_count"] == 0
    assert result["mismatch_count"] == 0
    assert result["results"][0]["case_id"] == "exact_terms_control"
    assert all(item["observed_accept"] is True for item in result["results"])
    assert all(
        item["creative_match"]["acceptance"] == "semantic_evidence_coherence"
        for item in result["results"][1:]
    )
