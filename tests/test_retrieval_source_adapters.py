from chat_proxy.retrieval_source_adapters import (
    recent_goal_candidate,
    reviewed_memory_candidate,
)


def test_recent_goal_adapter_preserves_current_state_lifecycle():
    candidate = recent_goal_candidate(
        {
            "id": "J-CSC2555",
            "title": "CSC2555 project",
            "body": "- Finish the evaluation table.",
            "owner": "Shared",
            "area": "course",
            "status": "active",
            "created_at": "2026-08-20",
            "review_on": "2026-09-05",
        },
        source_file="Recent Goals(Current).md",
        source_revision="abc123",
        source_updated_at="2026-09-01T12:00:00Z",
        match_value="course_project",
        match_detail="router selected recent goals",
        retrieval_score=2.0,
        rank=1,
        evaluated_at="2026-09-01T13:00:00Z",
    ).to_dict()

    assert candidate["item"]["source_type"] == "recent_goals"
    assert candidate["item"]["epistemic_role"] == "current_state"
    assert candidate["item"]["source_id"] == "J-CSC2555"
    assert candidate["item"]["review_after"] == "2026-09-05"
    assert candidate["freshness"]["state"] == "current"
    assert candidate["injectable"]["allowed"] is False
    assert candidate["injectable"]["policy"] == "selection_only_v1"


def test_recent_goal_adapter_marks_review_due_item_as_aging():
    candidate = recent_goal_candidate(
        {
            "id": "J-OLD",
            "title": "Old goal",
            "status": "active",
            "created_at": "2026-08-01",
            "review_on": "2026-08-20",
        },
        source_file="Recent Goals(Current).md",
        source_revision="abc123",
        source_updated_at="2026-08-21T00:00:00Z",
        match_value="memory_infra",
        match_detail="router selected recent goals",
        retrieval_score=1.0,
        rank=1,
        evaluated_at="2026-09-01T00:00:00Z",
    )

    assert candidate.freshness.state.value == "aging"


def test_reviewed_adapter_preserves_provenance_and_confidence():
    candidate = reviewed_memory_candidate(
        {
            "id": 42,
            "title": "Course milestone",
            "content": "The report draft was completed.",
            "domain": "course",
            "function": "retrieval",
            "topic_key": "course.csc2555",
            "layer_role": "retrieval_summary",
            "canonical_ref": "J-CSC2555",
            "confidence": 0.9,
            "explicitness": "explicit_user_said",
            "status": "active",
            "source_candidate_ids_json": "[7]",
            "source_message_ids_json": "[101, 102]",
            "reviewer": "human",
            "reviewed_at": "2026-08-30T10:00:00Z",
            "updated_at": "2026-08-30T10:00:00Z",
            "review_after": "2026-09-30T00:00:00Z",
        },
        match_value="course_project",
        match_detail="router selected reviewed memory",
        retrieval_score=3.0,
        rank=1,
        evaluated_at="2026-09-01T00:00:00Z",
    ).to_dict()

    assert candidate["item"]["source_type"] == "reviewed_memory"
    assert candidate["item"]["epistemic_role"] == "validated_memory"
    assert candidate["item"]["provenance"]["candidate_ids"] == ["7"]
    assert [
        ref["source_id"] for ref in candidate["item"]["provenance"]["source_refs"]
    ] == ["message:101", "message:102"]
    assert candidate["evidence_confidence"] == 0.9
    assert candidate["retrieval_score_method"] == "deterministic_source_match_v1"
