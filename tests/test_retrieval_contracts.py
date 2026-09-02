import json

import pytest

from chat_proxy.retrieval_contracts import (
    EpistemicRole,
    FreshnessAssessment,
    FreshnessState,
    InjectionDecision,
    MatchKind,
    MatchReason,
    MemoryItem,
    Provenance,
    RetrievalCandidate,
    Sensitivity,
    SourceReference,
    SourceType,
)


def _reviewed_item() -> MemoryItem:
    return MemoryItem(
        source_type=SourceType.REVIEWED_MEMORY,
        epistemic_role=EpistemicRole.VALIDATED_MEMORY,
        source_id="reviewed:22",
        topic_key="project:genai-day2night",
        content="The day-to-night project report was submitted.",
        status="active",
        observed_at="2026-08-12",
        updated_at="2026-08-13T00:00:00Z",
        review_after="2026-10-01",
        valid_from="2026-08-12",
        provenance=Provenance(
            source_refs=(
                SourceReference(
                    source_type=SourceType.CHAT_HISTORY,
                    source_id="message:32149",
                    relation="supporting_evidence",
                ),
            ),
            candidate_ids=("candidate:83",),
            promoted_from=("daily:2026-08-12",),
            reviewer="human",
        ),
        supersedes=("reviewed:9",),
        attributes={
            "domain": "course_project",
            "layer_role": "retrieval_summary",
        },
    )


def _freshness(state: FreshnessState, reason: str) -> FreshnessAssessment:
    return FreshnessAssessment(
        state=state,
        policy="role-aware-v1",
        reason=reason,
        evaluated_at="2026-09-01T12:00:00Z",
        score=0.9 if state is FreshnessState.CURRENT else 0.2,
    )


def _match() -> MatchReason:
    return MatchReason(
        kind=MatchKind.STRUCTURED_FILTER,
        value="topic_key=project:genai-day2night",
        detail="The routed project topic matched the reviewed item.",
    )


def test_memory_item_contains_source_metadata_but_not_retrieval_decisions():
    payload = _reviewed_item().to_dict()

    assert payload["source_type"] == "reviewed_memory"
    assert payload["epistemic_role"] == "validated_memory"
    assert payload["provenance"]["source_refs"][0]["source_id"] == "message:32149"
    assert payload["supersedes"] == ["reviewed:9"]
    assert payload["attributes"]["layer_role"] == "retrieval_summary"
    assert "freshness" not in payload
    assert "retrieval_score" not in payload
    assert "injectable" not in payload
    json.dumps(payload, ensure_ascii=False)


def test_same_item_can_have_query_specific_retrieval_decisions():
    item = _reviewed_item()
    current = RetrievalCandidate(
        item=item,
        freshness=_freshness(FreshnessState.CURRENT, "Project fact is still current."),
        retrieval_score=4.2,
        retrieval_score_method="structured-route-v1",
        evidence_confidence=0.95,
        evidence_confidence_reason="Human-reviewed with message provenance.",
        match_reason=_match(),
        injectable=InjectionDecision(
            allowed=True,
            policy="standard-memory-v1",
            reason="Validated project memory is allowed for this query.",
            signals={"project_query": True},
        ),
        rank=1,
    )
    gated = RetrievalCandidate(
        item=item,
        freshness=_freshness(FreshnessState.STALE, "Query asks for current status."),
        retrieval_score=0.4,
        retrieval_score_method="structured-route-v1",
        evidence_confidence=0.95,
        evidence_confidence_reason="The evidence is reliable even when stale.",
        match_reason=_match(),
        injectable=InjectionDecision(
            allowed=False,
            policy="current-state-precedence-v1",
            reason="A newer J item supersedes this project state for injection.",
            signals={"newer_current_state": True},
        ),
        rank=3,
    )

    assert current.item is gated.item
    assert current.freshness.state is FreshnessState.CURRENT
    assert gated.freshness.state is FreshnessState.STALE
    assert current.injectable.allowed is True
    assert gated.injectable.allowed is False
    json.dumps(current.to_dict(), ensure_ascii=False)
    json.dumps(gated.to_dict(), ensure_ascii=False)


def test_explicit_intimacy_is_an_item_sensitivity_but_injection_is_policy_result():
    item = MemoryItem(
        source_type=SourceType.INTIMACY_MEMORY,
        epistemic_role=EpistemicRole.STRUCTURED_CONTEXT,
        source_id="mother:I-1.1",
        content="Preference content.",
        sensitivity=Sensitivity.EXPLICIT_INTIMACY,
    )
    candidate = RetrievalCandidate(
        item=item,
        freshness=FreshnessAssessment(
            state=FreshnessState.TIMELESS,
            policy="stable-preference-v1",
            reason="Stable preference; age alone does not make it stale.",
            evaluated_at="2026-09-01T12:00:00Z",
        ),
        match_reason=MatchReason(
            kind=MatchKind.SOURCE_ROUTE,
            value="intimacy_memory",
            detail="Relationship context was detected without an explicit cue.",
        ),
        injectable=InjectionDecision(
            allowed=False,
            policy="explicit-intimacy-gate-v1",
            reason="An explicit intimacy cue is required.",
            signals={
                "relationship_context": True,
                "explicit_intimacy_cue": False,
            },
        ),
    )

    assert item.sensitivity is Sensitivity.EXPLICIT_INTIMACY
    assert candidate.injectable.allowed is False


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"retrieval_score": float("nan")}, "retrieval_score must be finite"),
        (
            {"retrieval_score": 1.0},
            "retrieval_score_method is required",
        ),
        (
            {"evidence_confidence": 1.1, "evidence_confidence_reason": "bad"},
            "evidence_confidence must be between 0 and 1",
        ),
        ({"rank": 0}, "rank must be a positive integer"),
    ],
)
def test_retrieval_candidate_validates_ambiguous_scores(kwargs, message):
    with pytest.raises(ValueError, match=message):
        RetrievalCandidate(
            item=_reviewed_item(),
            freshness=_freshness(FreshnessState.CURRENT, "Current."),
            match_reason=_match(),
            injectable=InjectionDecision(
                allowed=True,
                policy="test",
                reason="test decision",
            ),
            **kwargs,
        )


def test_memory_item_rejects_self_supersession_and_mutable_metadata_is_copied():
    source_attributes = {"nested": {"area": "study_work"}}
    item = MemoryItem(
        source_type=SourceType.RECENT_GOALS,
        epistemic_role=EpistemicRole.CURRENT_STATE,
        source_id="J-20260823-002",
        content="Prepare CV and project list.",
        attributes=source_attributes,
    )
    source_attributes["nested"]["area"] = "mutated"

    assert item.to_dict()["attributes"]["nested"]["area"] == "study_work"
    with pytest.raises(ValueError, match="cannot supersede itself"):
        MemoryItem(
            source_type=SourceType.RECENT_GOALS,
            epistemic_role=EpistemicRole.CURRENT_STATE,
            source_id="J-1",
            content="Current goal.",
            supersedes=("J-1",),
        )


def test_temporal_fields_require_iso_dates_or_timestamps():
    with pytest.raises(ValueError, match="updated_at must be an ISO-8601"):
        MemoryItem(
            source_type=SourceType.RECENT_GOALS,
            epistemic_role=EpistemicRole.CURRENT_STATE,
            source_id="J-1",
            content="Current goal.",
            updated_at="last Tuesday",
        )


def test_rolling_summary_is_a_derived_cache_not_current_state_memory():
    item = MemoryItem(
        source_type=SourceType.ROLLING_SUMMARY,
        epistemic_role=EpistemicRole.DERIVED_CONTEXT_CACHE,
        source_id="rolling:conversation-1:v3",
        content="Now: User explicitly said they are resting.\nOpen threads: None.",
        observed_at="2026-09-01T12:00:00Z",
        supersedes=("rolling:conversation-1:v2",),
        attributes={
            "conversation_id": "conversation-1",
            "version": 3,
            "last_message_id": 42,
        },
    )

    payload = item.to_dict()
    assert payload["source_type"] == "rolling_summary"
    assert payload["epistemic_role"] == "derived_context_cache"
    assert payload["supersedes"] == ["rolling:conversation-1:v2"]
