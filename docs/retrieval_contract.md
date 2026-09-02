# Typed retrieval evidence contract

The retrieval contract separates canonical source facts from query-time
selection decisions.

## Canonical source layer

`MemoryItem` answers: **what is this item?** It contains:

- source identity: `source_type`, `source_id`, `epistemic_role`, `topic_key`;
- source lifecycle: `status`, `expires_at`, `review_after`;
- semantic time: `observed_at`, `valid_from`, `valid_until`;
- change history: `supersedes`, `superseded_by`;
- structured audit data: `provenance` and `sensitivity`;
- source-native metadata in `attributes`, such as J `area`, reviewed-memory
  `layer_role`, or a Mother section path.

All temporal values use validated ISO-8601 date or timestamp strings so source
adapters can preserve date-only J fields without inventing a timezone.

`expires_at` is an operational retrieval cutoff. `valid_until` says when the
fact stopped being true. `review_after` requests maintenance but does not by
itself invalidate the item.

`Provenance` contains typed `SourceReference` objects plus candidate IDs,
promotion lineage, author/reviewer identity, and JSON-safe attributes. It is not
a free-form source label.

## Retrieval-time layer

`RetrievalCandidate` answers: **why was this item selected for this query?** It
wraps one unchanged `MemoryItem` with:

- a role-aware `FreshnessAssessment`, including policy, evaluation time, and
  reason;
- `retrieval_score` plus a required scoring-method name;
- `evidence_confidence` plus a required explanation;
- a primary structured `MatchReason` and optional supporting matches;
- an `InjectionDecision` with policy, reason, and gate signals;
- a query-local rank.

Freshness and injection eligibility are never stored on `MemoryItem`. The same
item may therefore be current and injectable for one query, but stale or gated
for another, without mutating canonical memory.

## Source roles

| Source | Default epistemic role |
|---|---|
| recent turns / J | `current_state` |
| rolling summary | `derived_context_cache` |
| reviewed memory | `validated_memory` |
| Mother | `stable_semantic` |
| Core Anchors | `invariant` |
| World Book / gated I view | `structured_context` |
| raw chat history | `episodic_evidence` |

These roles are not globally comparable relevance scores. Routing and
precedence policies should compare candidates within the requested fact class.

### Rolling summary representation

A rolling summary is a fallible derived cache, not validated memory. Represent
each version with:

- `source_type=rolling_summary`;
- `epistemic_role=derived_context_cache`;
- `source_id=rolling:<conversation_id>:v<version>`;
- `observed_at=<summary updated_at>`;
- source-range metadata such as `previous_last_message_id`, `last_message_id`,
  `source_message_count`, and `model_id` in provenance/attributes;
- a `supersedes` link to the previous rolling version.

Rolling claims cannot supersede canonical memory. Their freshness is short and
conversation-local. Unsupported `Key context` or `Style / protocols` claims
should be omitted or gated rather than promoted into stable facts. Derived
summary text must never create an `explicit_intimacy_cue`; that signal comes
only from the current raw user turn.

## Sensitive-source rule

An I-section item carries `sensitivity=explicit_intimacy`, but that does not make
it permanently injectable. `InjectionDecision.signals` must record independent
signals such as `relationship_context` and `explicit_intimacy_cue`; policy may
allow injection only when the explicit cue is true.

The implementation is in `chat_proxy/retrieval_contracts.py`. Adapters should
construct these objects at their boundary; this initial contract does not alter
current context-selection behavior.
