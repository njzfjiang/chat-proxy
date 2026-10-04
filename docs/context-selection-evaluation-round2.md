# Context selection evaluation: round 2

This round changes evaluation artifacts only. It does not modify the database,
the online retrieval path, or the creative-writing required-term rule. No model
calls were made.

## Gold review: seed 8422

The prior seed label expected `recent+episodic`. Manual review does not support
the episodic requirement:

- The seed explicitly says the earlier plot discussion occurred in a DeepSeek
  window whose context had already been lost.
- Messages `8420` and `8421` establish only that the user is writing a
  historical-style novel and is stuck.
- A timestamp-bounded search before `2025-12-10T00:00:08.335` found no indexed
  message describing how the character could receive the evidence.
- Messages `7553` and `7554` contain later concrete fiction material but are
  dated in April 2026 and are invalid future evidence for this seed.

The reviewed gold is therefore `expected_context=recent`, with historical chat
retrieval expected to abstain. This decision comes from the source conversation
and as-of corpus search, not from v11 returning an empty result.

## Report A: six-seed model selector

The prompt-v2, thinking-disabled experiment remains a six-seed result:

- 6/6 requests succeeded.
- Average latency was 2.35 seconds, versus 27.48 seconds in the saved earlier
  run (about 11.7 times faster).
- Reported reasoning tokens fell from 32,475 to 0.
- `8422` and `11847` correctly rejected all candidates in this review.
- `10741` gained more direct self-identity evidence.
- `831` remained incomplete and redundant; `12762` lost useful scope evidence.

The prompt changed at the same time as the thinking setting. These measurements
do not isolate a thinking effect and do not generalize to all 32 seeds.

## Report B: v11 32-seed retrieval

The saved v11 retrieval output is unchanged:

- 32/32 seeds completed with no errors.
- No detected future leaks.
- Correcting the `8422` gold changes the expected historical-retrieval
  denominator from 23 to 22; 17 of those 22 seeds have non-empty historical
  chat retrieval.
- Compared with v8, only `8422` changes final evidence IDs, from five
  DeepSeek-centered distractions to an empty result.
- All 81 selected evidence items have traceable body-term matches, but that is
  lexical auditability, not relevance accuracy.

The 17/22 figure is retrieval presence under the reviewed labels. It is not an
answer-quality or precision score.

A complete gold-v2 rerun is stored in
`context_selection_prod_evidence_gold_v12_20261004`. It produced zero final
historical-chat candidate-ID differences and zero historical-chat
final-injected-ID differences versus v11 across all 32 seeds. Its summary
records 32 successful seeds, 22 expected historical-retrieval seeds, 17
non-empty historical retrievals, and zero future leaks. Current Recent Goals
results changed for six seeds (`831`, `4249`, `4239`, `5705`, `12762`, and
`10741`), while Mother, Core Anchors, World Book, and reviewed-memory results
were unchanged. The historical retrieval denominator change therefore comes
from the reviewed label, but v12 must not be treated as a frozen all-source
comparison because Recent Goals is mutable.

## Report C: 14 changed-candidate seeds

Manual review compares the legacy-backend and evidence-backend final visible
snippets for the 14 seeds whose candidate IDs changed within v11:

| Review outcome | Seeds |
|---|---:|
| Evidence arm preferred | 5 |
| Legacy arm preferred | 2 |
| Neither arm complete | 6 |
| Materially equivalent | 1 |

Notable findings:

- `831`: the evidence arm repeats memory-layer architecture and still misses
  current J-area/source-state evidence, including message `5066`.
- `12762`: the evidence arm keeps a useful paper/prototype requirement but
  loses message `12761` and spends three slots on Cloud-related material.
- `10497` and `18108`: the legacy arm preserves more task-critical evidence.
- `17699`: both arms are unrelated; historical retrieval should abstain.
- `33394`: the evidence arm clearly improves retrieval of the earlier Qdrant
  plus keyword-search design.

The complete per-arm labels are stored in
`candidate_change_annotations_v1.json`.

## Required-term known positives

The creative-writing hard AND was audited with one exact-term control and three
known-positive paraphrases:

- The exact `剧情` + `人物` + `证据` control passes.
- All three semantically relevant paraphrases are rejected because they use
  concrete names and synonyms such as `密信` and `线索` instead of all three
  generic query terms.

This is a concrete false-negative mechanism, but this round deliberately leaves
the rule unchanged. The next retrieval-rule proposal should be evaluated
against these cases and the 32-seed regression set before entering the online
path.
