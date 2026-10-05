# Context selection benchmark artifacts

`benchmark_outputs/` is intentionally ignored because snapshots can contain
chat-derived evidence. Commit benchmark code, contracts, and manual labels;
keep raw runs local or transfer them only as an explicitly reviewed package.

## Current review set

Keep these artifacts while the selector and planner changes are under review:

- `context_selection_prod_router_planner_fixed_v2`: fixed 32-seed baseline and
  seed hydration source.
- `context_selection_prod_evidence_sentence_v8`: pre-creative-planner
  comparison point.
- `context_selection_prod_evidence_sentence_v11`: current 32-seed local run.
  Compared with v8, only seed `8422` changed; it now returns no retrieved chat
  evidence.
- `context_selection_prod_evidence_gold_v12_20261004`: 32-seed rerun with the
  reviewed gold override. Historical chat candidate IDs are identical to v11
  for all 32 seeds. Current Recent Goals results changed for six seeds because
  that source is mutable, so v12 is not a frozen all-source comparison.
- `context_selection_prod_evidence_anchor_v6`: common candidate source plus
  saved selector runs. Preserve this directory because model calls are not
  reproduced by local benchmark commands.
- `context_selection_prod_evidence_8422_gold_v2_20261004`: smaller model-free,
  single-seed verification of the same reviewed `8422` gold override.
- `benchmarks/context_selection/annotations/priority_annotations_v1.json`:
  blind labels for the earlier six-seed A/B/C comparison.
- `benchmarks/context_selection/annotations/priority_annotations_v2.json`:
  labels for the prompt-v2, thinking-disabled selector output. These labels do
  not isolate the thinking setting and do not estimate 32-seed accuracy.
- `benchmarks/context_selection/annotations/candidate_change_annotations_v1.json`:
  per-arm manual review for the 14 v11 seeds whose final candidate IDs changed.
- `benchmarks/context_selection/annotations/gold_labels_v2.json`: reviewed gold
  overrides applied by the local runner.
- `docs/context-selection-evaluation-round2.md`: separate reports for the
  six-seed selector, v11 32-seed retrieval, 14 changed-candidate review, and
  required-term known-positive audit.
- `docs/context-selection-evaluation-round3.md`: concept-aware creative
  filtering, `8422`/`17699` rejection controls, real-backend positive checks,
  and the 32-seed model-free regression.
- `benchmarks/context_selection/audit_creative_positive_trace.py`: read-only
  real-backend trace generator for backend top 20, rerank decisions, and final
  visible context. Raw outputs remain ignored because they contain chat-derived
  evidence.
- `context_selection_renderer_budget_v21_20261005` and its matching zip:
  renderer-change 32-seed baseline retained for shadow review. It contains the
  per-seed evidence and legacy outputs, retrieval snapshots, reports, and
  manifest; all 32 seeds succeeded with no future leaks or contexts over 1,200
  characters.

## Reproducing a local seed

The runner never calls the summary model in this mode. It clones the source
SQLite database into a temporary database and uses an in-process backend:

```powershell
python benchmarks/context_selection/run_local_evidence_ab.py `
  --seed-id 8422 `
  --output benchmark_outputs/context_selection_prod_evidence_8422_head_<date>
```

`--seed-id` may be repeated. The output directory must not already exist.

## Superseded local runs

These are reasonable cleanup candidates after the current review is complete:

- Early routing and curation pilots from `context_selection_pilot*` through
  `context_selection_prod_router_planner_fixed`; retain
  `context_selection_prod_router_planner_fixed_v2` as the baseline.
- `context_selection_prod_evidence_ab`, `_v2`, and `_v3`; retain `_v4` as the
  completed evidence A/B milestone.
- `context_selection_prod_evidence_sentence_v7`, `_v9`, and `_v10`; retain
  `_v8` and `_v11` as the before/after pair.
- `context_selection_prod_evidence_8422_head_20260923`; retain the reviewed
  gold-v2 rerun after the older diagnostic report is no longer referenced.
- Root archives for superseded runs, including
  `context_selection_prod_evidence_trace_v5.zip` and
  `context_selection_prod_evidence_sentence_v7.zip`, after confirming no
  external review still references them.

Do not use the existing `context_selection_prod_evidence_anchor_v6.zip` as the
current review package: the directory received later selector outputs after the
archive was created. Rebuild a deliberately scoped package if raw evidence is
needed for external review.

The three `query_clusters_rebuilt*` directories account for roughly 11 MiB but
represent corpus-analysis provenance. Keep them until seed-generation
provenance is documented elsewhere; deleting them saves little and makes the
early benchmark construction harder to audit.
