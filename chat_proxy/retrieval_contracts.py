from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Mapping


class SourceType(StrEnum):
    RECENT_TURNS = "recent_turns"
    ROLLING_SUMMARY = "rolling_summary"
    RECENT_GOALS = "recent_goals"
    REVIEWED_MEMORY = "reviewed_memory"
    MOTHER_MEMORY = "mother_memory"
    CORE_ANCHOR = "core_anchor"
    WORLDBOOK = "worldbook"
    INTIMACY_MEMORY = "intimacy_memory"
    CHAT_HISTORY = "chat_history"


class EpistemicRole(StrEnum):
    CURRENT_STATE = "current_state"
    DERIVED_CONTEXT_CACHE = "derived_context_cache"
    VALIDATED_MEMORY = "validated_memory"
    STABLE_SEMANTIC = "stable_semantic"
    INVARIANT = "invariant"
    STRUCTURED_CONTEXT = "structured_context"
    EPISODIC_EVIDENCE = "episodic_evidence"


class Sensitivity(StrEnum):
    STANDARD = "standard"
    PERSONAL = "personal"
    SENSITIVE = "sensitive"
    EXPLICIT_INTIMACY = "explicit_intimacy"


class FreshnessState(StrEnum):
    CURRENT = "current"
    AGING = "aging"
    STALE = "stale"
    TIMELESS = "timeless"
    UNKNOWN = "unknown"


class MatchKind(StrEnum):
    SOURCE_ROUTE = "source_route"
    STRUCTURED_FILTER = "structured_filter"
    TOPIC = "topic"
    KEYWORD = "keyword"
    EXPLICIT_REFERENCE = "explicit_reference"
    FALLBACK = "fallback"


@dataclass(frozen=True)
class SourceReference:
    source_type: SourceType
    source_id: str
    relation: str = "evidence"
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_enum(self.source_type, SourceType, "source_type")
        object.__setattr__(
            self, "source_id", _required_text(self.source_id, "source_id")
        )
        object.__setattr__(self, "relation", _required_text(self.relation, "relation"))
        object.__setattr__(self, "attributes", _freeze_json_mapping(self.attributes))

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_type": self.source_type.value,
            "source_id": self.source_id,
            "relation": self.relation,
            "attributes": _thaw_json(self.attributes),
        }


@dataclass(frozen=True)
class Provenance:
    source_refs: tuple[SourceReference, ...] = ()
    candidate_ids: tuple[str, ...] = ()
    promoted_from: tuple[str, ...] = ()
    author: str | None = None
    reviewer: str | None = None
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_refs", tuple(self.source_refs))
        if any(
            not isinstance(reference, SourceReference) for reference in self.source_refs
        ):
            raise TypeError("source_refs must contain SourceReference values.")
        object.__setattr__(
            self, "candidate_ids", _normalized_ids(self.candidate_ids, "candidate_ids")
        )
        object.__setattr__(
            self, "promoted_from", _normalized_ids(self.promoted_from, "promoted_from")
        )
        object.__setattr__(self, "author", _optional_text(self.author))
        object.__setattr__(self, "reviewer", _optional_text(self.reviewer))
        object.__setattr__(self, "attributes", _freeze_json_mapping(self.attributes))

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_refs": [reference.to_dict() for reference in self.source_refs],
            "candidate_ids": list(self.candidate_ids),
            "promoted_from": list(self.promoted_from),
            "author": self.author,
            "reviewer": self.reviewer,
            "attributes": _thaw_json(self.attributes),
        }


@dataclass(frozen=True)
class MemoryItem:
    source_type: SourceType
    epistemic_role: EpistemicRole
    source_id: str
    content: str
    topic_key: str | None = None
    status: str | None = None
    observed_at: str | None = None
    updated_at: str | None = None
    expires_at: str | None = None
    review_after: str | None = None
    valid_from: str | None = None
    valid_until: str | None = None
    provenance: Provenance = field(default_factory=Provenance)
    sensitivity: Sensitivity = Sensitivity.PERSONAL
    supersedes: tuple[str, ...] = ()
    superseded_by: tuple[str, ...] = ()
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_enum(self.source_type, SourceType, "source_type")
        _require_enum(self.epistemic_role, EpistemicRole, "epistemic_role")
        _require_enum(self.sensitivity, Sensitivity, "sensitivity")
        source_id = _required_text(self.source_id, "source_id")
        object.__setattr__(self, "source_id", source_id)
        object.__setattr__(self, "content", _required_text(self.content, "content"))
        for name in (
            "observed_at",
            "updated_at",
            "expires_at",
            "review_after",
            "valid_from",
            "valid_until",
        ):
            object.__setattr__(
                self,
                name,
                _optional_iso_temporal(getattr(self, name), name),
            )
        object.__setattr__(self, "topic_key", _optional_text(self.topic_key))
        object.__setattr__(self, "status", _optional_text(self.status))
        supersedes = _normalized_ids(self.supersedes, "supersedes")
        superseded_by = _normalized_ids(self.superseded_by, "superseded_by")
        if source_id in supersedes or source_id in superseded_by:
            raise ValueError("A memory item cannot supersede itself.")
        object.__setattr__(self, "supersedes", supersedes)
        object.__setattr__(self, "superseded_by", superseded_by)
        object.__setattr__(self, "attributes", _freeze_json_mapping(self.attributes))

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_type": self.source_type.value,
            "epistemic_role": self.epistemic_role.value,
            "source_id": self.source_id,
            "topic_key": self.topic_key,
            "content": self.content,
            "status": self.status,
            "observed_at": self.observed_at,
            "updated_at": self.updated_at,
            "expires_at": self.expires_at,
            "review_after": self.review_after,
            "valid_from": self.valid_from,
            "valid_until": self.valid_until,
            "provenance": self.provenance.to_dict(),
            "sensitivity": self.sensitivity.value,
            "supersedes": list(self.supersedes),
            "superseded_by": list(self.superseded_by),
            "attributes": _thaw_json(self.attributes),
        }


@dataclass(frozen=True)
class FreshnessAssessment:
    state: FreshnessState
    policy: str
    reason: str
    evaluated_at: str
    score: float | None = None

    def __post_init__(self) -> None:
        _require_enum(self.state, FreshnessState, "state")
        object.__setattr__(self, "policy", _required_text(self.policy, "policy"))
        object.__setattr__(self, "reason", _required_text(self.reason, "reason"))
        object.__setattr__(
            self,
            "evaluated_at",
            _required_iso_temporal(self.evaluated_at, "evaluated_at"),
        )
        if self.score is not None:
            _bounded_score(self.score, "freshness score")

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "policy": self.policy,
            "reason": self.reason,
            "evaluated_at": self.evaluated_at,
            "score": self.score,
        }


@dataclass(frozen=True)
class MatchReason:
    kind: MatchKind
    value: str
    detail: str
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_enum(self.kind, MatchKind, "kind")
        object.__setattr__(self, "value", _required_text(self.value, "value"))
        object.__setattr__(self, "detail", _required_text(self.detail, "detail"))
        object.__setattr__(self, "attributes", _freeze_json_mapping(self.attributes))

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "value": self.value,
            "detail": self.detail,
            "attributes": _thaw_json(self.attributes),
        }


@dataclass(frozen=True)
class InjectionDecision:
    allowed: bool
    policy: str
    reason: str
    signals: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.allowed, bool):
            raise TypeError("allowed must be a boolean.")
        object.__setattr__(self, "policy", _required_text(self.policy, "policy"))
        object.__setattr__(self, "reason", _required_text(self.reason, "reason"))
        object.__setattr__(self, "signals", _freeze_json_mapping(self.signals))

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "policy": self.policy,
            "reason": self.reason,
            "signals": _thaw_json(self.signals),
        }


@dataclass(frozen=True)
class RetrievalCandidate:
    item: MemoryItem
    freshness: FreshnessAssessment
    match_reason: MatchReason
    injectable: InjectionDecision
    retrieval_score: float | None = None
    retrieval_score_method: str | None = None
    evidence_confidence: float | None = None
    evidence_confidence_reason: str | None = None
    rank: int | None = None
    supporting_matches: tuple[MatchReason, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.item, MemoryItem):
            raise TypeError("item must be a MemoryItem.")
        if not isinstance(self.freshness, FreshnessAssessment):
            raise TypeError("freshness must be a FreshnessAssessment.")
        if not isinstance(self.match_reason, MatchReason):
            raise TypeError("match_reason must be a MatchReason.")
        if not isinstance(self.injectable, InjectionDecision):
            raise TypeError("injectable must be an InjectionDecision.")
        if self.retrieval_score is not None:
            if not math.isfinite(self.retrieval_score):
                raise ValueError("retrieval_score must be finite.")
            if not _optional_text(self.retrieval_score_method):
                raise ValueError(
                    "retrieval_score_method is required when retrieval_score is set."
                )
        object.__setattr__(
            self,
            "retrieval_score_method",
            _optional_text(self.retrieval_score_method),
        )
        if self.evidence_confidence is not None:
            _bounded_score(self.evidence_confidence, "evidence_confidence")
            if not _optional_text(self.evidence_confidence_reason):
                raise ValueError(
                    "evidence_confidence_reason is required when "
                    "evidence_confidence is set."
                )
        object.__setattr__(
            self,
            "evidence_confidence_reason",
            _optional_text(self.evidence_confidence_reason),
        )
        if self.rank is not None and self.rank < 1:
            raise ValueError("rank must be a positive integer.")
        object.__setattr__(self, "supporting_matches", tuple(self.supporting_matches))
        if any(not isinstance(match, MatchReason) for match in self.supporting_matches):
            raise TypeError("supporting_matches must contain MatchReason values.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "item": self.item.to_dict(),
            "freshness": self.freshness.to_dict(),
            "retrieval_score": self.retrieval_score,
            "retrieval_score_method": self.retrieval_score_method,
            "evidence_confidence": self.evidence_confidence,
            "evidence_confidence_reason": self.evidence_confidence_reason,
            "match_reason": self.match_reason.to_dict(),
            "injectable": self.injectable.to_dict(),
            "rank": self.rank,
            "supporting_matches": [
                match.to_dict() for match in self.supporting_matches
            ],
        }


def _required_text(value: Any, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field_name} must be non-empty.")
    return text


def _optional_text(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _required_iso_temporal(value: Any, field_name: str) -> str:
    text = _required_text(value, field_name)
    normalized = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError(
            f"{field_name} must be an ISO-8601 date or timestamp."
        ) from exc
    return text


def _optional_iso_temporal(value: Any, field_name: str) -> str | None:
    text = _optional_text(value)
    return _required_iso_temporal(text, field_name) if text else None


def _normalized_ids(values: Any, field_name: str) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, str | bytes):
        raise TypeError(f"{field_name} must be an iterable of IDs, not a string.")
    normalized: list[str] = []
    seen = set()
    for raw_value in values:
        value = _required_text(raw_value, field_name)
        if value in seen:
            continue
        seen.add(value)
        normalized.append(value)
    return tuple(normalized)


def _bounded_score(value: float, field_name: str) -> None:
    if not math.isfinite(value) or value < 0 or value > 1:
        raise ValueError(f"{field_name} must be between 0 and 1.")


def _require_enum(value: Any, enum_type: type[StrEnum], field_name: str) -> None:
    if not isinstance(value, enum_type):
        raise TypeError(f"{field_name} must be a {enum_type.__name__} value.")


def _freeze_json_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("attributes/signals must be a mapping.")
    json.dumps(value, ensure_ascii=False)
    return _freeze_json(dict(value))


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, list | tuple):
        return tuple(_freeze_json(item) for item in value)
    return value


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value
