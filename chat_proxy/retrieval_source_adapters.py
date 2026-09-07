from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Mapping

from .retrieval_contracts import (
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


def recent_goal_candidate(
    item: Mapping[str, Any],
    *,
    source_file: str,
    source_revision: str,
    source_updated_at: str,
    match_value: str,
    match_detail: str,
    retrieval_score: float,
    rank: int,
    evaluated_at: str,
) -> RetrievalCandidate:
    item_id = _required(item, "id")
    title = _required(item, "title")
    body = str(item.get("body") or "").strip()
    content = f"{title}\n{body}".strip()
    status = str(item.get("status") or "active").strip()
    freshness = _lifecycle_freshness(
        evaluated_at=evaluated_at,
        expires_at=_optional(item.get("expires_at")),
        review_after=_optional(item.get("review_on")),
        policy="recent_goals_lifecycle",
    )
    memory_item = MemoryItem(
        source_type=SourceType.RECENT_GOALS,
        epistemic_role=EpistemicRole.CURRENT_STATE,
        source_id=item_id,
        topic_key=_optional(item.get("area")),
        content=content,
        status=status,
        observed_at=_optional(item.get("created_at")),
        updated_at=source_updated_at,
        expires_at=_optional(item.get("expires_at")),
        review_after=_optional(item.get("review_on")),
        provenance=Provenance(
            source_refs=(
                SourceReference(
                    source_type=SourceType.RECENT_GOALS,
                    source_id=source_file,
                    relation="source_document",
                    attributes={"revision": source_revision},
                ),
            ),
        ),
        sensitivity=Sensitivity.PERSONAL,
        attributes={
            "title": title,
            "owner": _optional(item.get("owner")),
            "area": _optional(item.get("area")),
        },
    )
    return _selection_only_candidate(
        memory_item,
        freshness=freshness,
        match_value=match_value,
        match_detail=match_detail,
        retrieval_score=retrieval_score,
        rank=rank,
    )


def reviewed_memory_candidate(
    item: Mapping[str, Any],
    *,
    match_value: str,
    match_detail: str,
    retrieval_score: float,
    rank: int,
    evaluated_at: str,
) -> RetrievalCandidate:
    item_id = _required(item, "id")
    title = _required(item, "title")
    content = _required(item, "content")
    source_id = f"reviewed:{item_id}"
    source_message_ids = _json_ids(item.get("source_message_ids_json"))
    source_candidate_ids = _json_ids(item.get("source_candidate_ids_json"))
    superseded_by = _optional(item.get("superseded_by_item_id"))
    freshness = _lifecycle_freshness(
        evaluated_at=evaluated_at,
        expires_at=_optional(item.get("expires_at")),
        review_after=_optional(item.get("review_after")),
        policy="reviewed_memory_lifecycle",
    )
    provenance = Provenance(
        source_refs=tuple(
            SourceReference(
                source_type=SourceType.CHAT_HISTORY,
                source_id=f"message:{message_id}",
                relation="supporting_message",
            )
            for message_id in source_message_ids
        ),
        candidate_ids=tuple(source_candidate_ids),
        reviewer=_optional(item.get("reviewer")),
        attributes={"canonical_ref": _optional(item.get("canonical_ref"))},
    )
    memory_item = MemoryItem(
        source_type=SourceType.REVIEWED_MEMORY,
        epistemic_role=EpistemicRole.VALIDATED_MEMORY,
        source_id=source_id,
        topic_key=_optional(item.get("topic_key")),
        content=f"{title}\n{content}".strip(),
        status=_optional(item.get("status")),
        observed_at=_optional(item.get("reviewed_at")),
        updated_at=_optional(item.get("updated_at")),
        expires_at=_optional(item.get("expires_at")),
        review_after=_optional(item.get("review_after")),
        provenance=provenance,
        sensitivity=Sensitivity.PERSONAL,
        superseded_by=(f"reviewed:{superseded_by}",) if superseded_by else (),
        attributes={
            "title": title,
            "domain": _optional(item.get("domain")),
            "function": _optional(item.get("function")),
            "primary_mother": _optional(item.get("primary_mother")),
            "secondary_mother": _optional(item.get("secondary_mother")),
            "layer_role": _optional(item.get("layer_role")),
            "explicitness": _optional(item.get("explicitness")),
        },
    )
    confidence = _bounded_optional_float(item.get("confidence"))
    candidate = _selection_only_candidate(
        memory_item,
        freshness=freshness,
        match_value=match_value,
        match_detail=match_detail,
        retrieval_score=retrieval_score,
        rank=rank,
    )
    if confidence is None:
        return candidate
    return RetrievalCandidate(
        item=candidate.item,
        freshness=candidate.freshness,
        match_reason=candidate.match_reason,
        injectable=candidate.injectable,
        retrieval_score=candidate.retrieval_score,
        retrieval_score_method=candidate.retrieval_score_method,
        evidence_confidence=confidence,
        evidence_confidence_reason="reviewed memory confidence field",
        rank=candidate.rank,
    )


def _selection_only_candidate(
    item: MemoryItem,
    *,
    freshness: FreshnessAssessment,
    match_value: str,
    match_detail: str,
    retrieval_score: float,
    rank: int,
) -> RetrievalCandidate:
    return RetrievalCandidate(
        item=item,
        freshness=freshness,
        match_reason=MatchReason(
            kind=MatchKind.SOURCE_ROUTE,
            value=match_value,
            detail=match_detail,
        ),
        injectable=InjectionDecision(
            allowed=False,
            policy="selection_only_v1",
            reason="Source adapter is enabled for selection evaluation only.",
        ),
        retrieval_score=retrieval_score,
        retrieval_score_method="deterministic_source_match_v1",
        rank=rank,
    )


def _lifecycle_freshness(
    *,
    evaluated_at: str,
    expires_at: str | None,
    review_after: str | None,
    policy: str,
) -> FreshnessAssessment:
    evaluated = _parse_temporal(evaluated_at)
    if expires_at and _parse_temporal(expires_at) <= evaluated:
        state = FreshnessState.STALE
        reason = "source item reached its expiry"
    elif review_after and _parse_temporal(review_after) <= evaluated:
        state = FreshnessState.AGING
        reason = "source item is due for review"
    else:
        state = FreshnessState.CURRENT
        reason = "source lifecycle is active and within review bounds"
    return FreshnessAssessment(
        state=state,
        policy=policy,
        reason=reason,
        evaluated_at=evaluated_at,
    )


def _parse_temporal(value: str) -> datetime:
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        return parsed
    return parsed.replace(tzinfo=None)


def _required(item: Mapping[str, Any], key: str) -> str:
    value = str(item.get(key) or "").strip()
    if not value:
        raise ValueError(f"{key} must be non-empty.")
    return value


def _optional(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _json_ids(value: Any) -> list[str]:
    if isinstance(value, list):
        raw_values = value
    else:
        try:
            raw_values = json.loads(str(value or "[]"))
        except (TypeError, ValueError, json.JSONDecodeError):
            return []
    if not isinstance(raw_values, list):
        return []
    return [text for raw in raw_values if (text := str(raw).strip())]


def _bounded_optional_float(value: Any) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if 0.0 <= parsed <= 1.0 else None
