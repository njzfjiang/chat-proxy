# Context selection evaluation: round 3

This round changes the online retrieval path and its evaluation artifacts. It
does not modify the database and makes no model calls.

## Rule change

Creative-writing retrieval keeps the exact required-term path, but adds a
bounded semantic path for paraphrases:

- Backend evidence lookup adds seven terms: `情节`, `下一幕`, `线索`, `密信`,
  `信件`, `书信`, and `拼图`.
- A candidate passes the semantic path only when it contains both an evidence
  concept and a concrete acquisition, discovery, or handoff action.
- Narrative vocabulary alone is insufficient. This rejects meta-discussion
  that happens to mention words such as `剧情` and `证据` in separate passages.
- Trace output records concept matches and missing concept groups.

Long, self-contained narration without a question or explicit recall request
no longer selects chat-history search merely because it contains `之前` or
`那时候`. Other matched domains still retain their own retrieval routes.

## Three-layer checks

The three paraphrase fixtures pass all local layers:

1. present in the simulated backend top 20 response;
2. accepted by the proxy coherence filter;
3. retained in the final 1,200-character visible context.

Against the read-only production database snapshot, message `7553` was ranked
13th in the backend top 20, survived the filter, and remained in the final five
visible items at 925 total rendered characters. Message `7554` ranked third in
the backend top 20 but did not survive into the final five, so backend recall
and final visibility are reported separately.

## Rejection controls

- `8422`: 20 backend candidates, zero rerank outputs, zero visible historical
  items in both saved backend arms.
- `17699`: routing stops before the backend and emits zero historical items.
  Manual review had already found that both prior arms were unrelated, so its
  reviewed gold label changes from `episodic` to `recent`.

## Full regression

The 32-seed model-free run completed 32/32 seeds with zero errors and zero
detected future leaks. Compared with the saved v12 evidence arm:

- `17699` is the only changed rerank result, from five unrelated items to none.
- `8422` remains empty.
- `831`, `12762`, `10497`, and `18108` retain the same selected IDs.
- The other 27 seeds also retain the same selected IDs.

After both reviewed gold overrides, 21 seeds expect historical retrieval and
16 have non-empty results. The remaining five expected-retrieval misses are
unchanged from v12; this round does not claim to solve them.

These checks establish retrieval behavior for the saved corpus and fixtures.
They do not measure answer quality or justify enabling an agent path yet.
