# Streaming and retrieval evidence follow-up

## Streaming diagnostics

`done-short-circuit-v2` preserves upstream bytes, adding an SSE delimiter if the
upstream terminal sentinel has no blank-line termination. UTF-8 decoding for
stored text is incremental. Responses include `x-chat-proxy-request-id`.

`stream_terminal` describes upstream consumption and is emitted before database
persistence finishes. `stream_delivery` describes ASGI sends, including byte count,
SHA-256, disconnect notification, exception type and `asgi_final_sent`.
A successful ASGI send is **not** an acknowledgement from nginx or Kelivo.
An absent delivery log means the request has not exited that response path (or
the process/logging failed), not necessarily that the network is at fault.

Capture untruncated logs with `journalctl --no-pager -l -o cat`, using a narrow
time range. Check nginx HTTP version, buffering, compression and timeout settings;
do not infer the cause from HTTP/1.0 alone. Do not publish tokens or private URLs.

## Optional retrieval evidence

Proxy requests `include_evidence=true` from `/search` by default. Set the request
body's `retrieval_evidence_enabled=false` to use the legacy backend search path.
Older backends may ignore this optional field; `evidence_version` in the snapshot
distinguishes supported responses. Existing backend clients default to legacy.

SQLite evidence mode unions a legacy pool with independent body LIKE pools,
bounded to at most 10 terms and 200 rows per term. This is bounded candidate
recall, not exhaustive recall or semantic search. Full-body matches are checked
with Latin entity boundaries; title-only results cannot satisfy proxy required
entity filtering. Responses retain the legacy prefix and add matching excerpts,
character offsets, body/title terms and a full-content hash. Context uses matching
excerpts. Hash deduplication happens only after entity filtering.

Snapshots expose candidate pool IDs, backend result IDs, selected pre-budget IDs,
selected post-budget IDs and aggregate filter counts. Offsets refer to the original
body, not to budget-clipped context. A matching term elsewhere in the body does not
guarantee that every term survives the final character budget.

Planner, curated routing, kind policy and historical source availability are not
changed. Compare against `context_selection_prod_router_planner_fixed_v2.zip` with
the same seeds and data snapshot; nonempty context is not a relevance/recall score.
This patch's fixed-case tests do not constitute a production benchmark rerun.

## Follow-up: query and excerpt fixes

The optional evidence backend now retains one-character non-ASCII alphanumeric
queries (including CJK `药`). Queries with no evidence terms still reach the
legacy candidate pool. Overlapping/adjacent windows are merged instead of repeated;
the first three matching terms in planner order select the windows, rather than
the first three source offsets. Each excerpt retains original character offsets.

The proxy splits the body-character budget across evidence candidates before
clipping, and omits a truncated fragment smaller than 40 characters (naturally
short complete excerpts remain valid). Legacy responses retain their old budget
behavior. Budgeting is measured in Python Unicode characters, not UTF-8 bytes.
This simple allocation can leave budget unused; it is not an optimal evidence
packer and does not ensure every matched entity survives clipping.

The 32-seed A/B rerun is in
`benchmark_outputs/context_selection_prod_evidence_ab_v4/`. Chat presence recovered
from 17/23 to 18/23, with 0/6 no-context historical triggers and zero detected
recent/chat future leaks. All four course/project seeds retain five excerpts.
Curated selections are unchanged. Presence and candidate count are not relevance
metrics; no answer model was called. The frozen Mother snapshot limitation remains.
