"""Summarize the saved pilot A/B without claiming labeled retrieval quality."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "benchmark_outputs/context_selection_prod_evidence_ab_v4"
BASE = ROOT / "benchmark_outputs/context_selection_prod_router_planner_fixed_v2"


def rows(path):
    with (path / "results.csv").open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main():
    paths = [BASE, RUN / "legacy_backend", RUN / "evidence"]
    summaries = [json.loads((p / "summary.json").read_text(encoding="utf-8")) for p in paths]
    old, legacy, evidence = [rows(p) for p in paths]
    assert [r["message_id"] for r in old] == [r["message_id"] for r in evidence]
    metrics = ["seed_count", "error_count", "expected_retrieval_nonempty_count",
               "expected_retrieval_any_source_nonempty_count", "curated_nonempty_count",
               "no_context_with_any_source_count", "future_leak_count"]
    lines = ["# Context selection: body evidence A/B", "",
             "32 original seeds; router, planner and curated sources enabled. No answer model calls.", "",
             "| Metric | Original ZIP baseline | Current proxy, legacy backend | Evidence backend |",
             "|---|---:|---:|---:|"]
    for key in metrics:
        lines.append(f"| {key} | " + " | ".join(str(s[key]) for s in summaries) + " |")
    changes = []
    for a, b, c in zip(old, legacy, evidence):
        if b["retrieval_source_ids"] != c["retrieval_source_ids"]:
            changes.append({"seed_id": c["message_id"], "theme": c["theme"],
                            "baseline": a["retrieval_source_ids"],
                            "legacy": b["retrieval_source_ids"], "evidence": c["retrieval_source_ids"]})
    lines += ["", "## Changed final chat candidates", "",
              "| Seed | Theme | Current legacy IDs | Evidence IDs |", "|---|---|---|---|"]
    for change in changes:
        lines.append(f"| {change['seed_id']} | {change['theme']} | {change['legacy'].replace('|', ', ')} | {change['evidence'].replace('|', ', ')} |")
    snapshots = json.loads((RUN / "evidence/retrieval_snapshots.json").read_text(encoding="utf-8"))
    assert len(snapshots) == len(evidence)
    active = [s for s in snapshots if "backend_result_ids" in s]
    assert all(s.get("evidence_version") == 1 for s in active)
    evidence_items = [item for s in snapshots for item in s.get("items", [])]
    body_supported = sum(bool(item.get("body_matched_terms")) for item in evidence_items)
    unchanged_curated = all(all(a[key] == b[key] for key in (
        "mother_items_json", "core_items_json", "worldbook_items_json",
        "reviewed_memory_items_json", "recent_goals_items_json")) for a, b in zip(legacy, evidence))
    result = {"changed_seed_count": len(changes), "changes": changes,
              "evidence_searches": len(active), "selected_items": len(evidence_items),
              "selected_items_with_body_terms": body_supported,
              "curated_identical_between_ab": unchanged_curated,
              "legacy_ids_equal_original_count": sum(a["retrieval_source_ids"] == b["retrieval_source_ids"] for a, b in zip(old, legacy))}
    lines += ["", "## Interpretation and limits", "",
              f"- Final chat candidate IDs/order changed for {len(changes)}/32 seeds.",
              f"- Evidence v1 verified in all {len(active)} executed chat searches.",
              f"- Selected items with full-body lexical matches: {body_supported}/{len(evidence_items)}.",
              f"- Curated selections identical between A/B: {unchanged_curated}.",
              "- Nonempty candidates and lexical support are NOT precision, recall or answer quality.",
              "- Full-content hashes remove preview-prefix deduplication, not semantic duplicates.",
              "- Mother uses frozen production DB sections; local Markdown lazy refresh is disabled.",
              "- Initial aborted runs hit read-only indexing / cross-path Mother uniqueness errors; excluded.",
              "- Source snapshots are current, not valid historical as-of curated evidence.",
              "- Both A/B arms use the current proxy; the legacy arm is not a checkout of the original code.",
              "- In-process transport does not measure HTTP timeout or deployment behavior.",
              "- Manifest records the DB, ZIP, seed CSV and relevant code hashes."]
    (RUN / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (RUN / "comparison.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
