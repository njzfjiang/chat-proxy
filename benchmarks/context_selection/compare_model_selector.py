"""Compare saved model-selector output with the rule-based evidence selection."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _parse_ids(value: str | None) -> list[int]:
    return [int(item) for item in (value or "").split("|") if item.strip()]


def compare_selector_results(
    selector_rows: list[dict[str, Any]],
    evidence_rows: list[dict[str, str]],
) -> dict[str, Any]:
    evidence_by_seed = {str(row["message_id"]): row for row in evidence_rows}
    comparisons = []
    for selector_row in selector_rows:
        seed_id = str(selector_row.get("seed_id") or "")
        evidence_row = evidence_by_seed.get(seed_id)
        if evidence_row is None:
            raise RuntimeError(f"selector seed {seed_id!r} is missing from evidence results")
        rule_ids = _parse_ids(evidence_row.get("retrieval_selected_after_budget_ids"))
        model_ids = [
            int(item["id"])
            for item in selector_row.get("selected") or []
            if isinstance(item, dict) and item.get("id") is not None
        ]
        rule_set = set(rule_ids)
        model_set = set(model_ids)
        comparisons.append(
            {
                "seed_id": seed_id,
                "status": selector_row.get("status"),
                "rule_ids": rule_ids,
                "model_ids": model_ids,
                "overlap_ids": sorted(rule_set & model_set),
                "model_only_ids": [item for item in model_ids if item not in rule_set],
                "rule_only_ids": [item for item in rule_ids if item not in model_set],
                "reject_all_reason": selector_row.get("reject_all_reason"),
            }
        )

    ok_rows = [row for row in selector_rows if row.get("status") == "ok"]
    successful = [row for row in comparisons if row["status"] == "ok"]
    changed = [
        row for row in successful if row["model_ids"] != row["rule_ids"]
    ]
    zero_overlap = [
        row
        for row in successful
        if row["model_ids"] and row["rule_ids"] and not row["overlap_ids"]
    ]
    reject_all = [row for row in successful if not row["model_ids"]]
    return {
        "seed_count": len(selector_rows),
        "ok_count": len(ok_rows),
        "error_count": len(selector_rows) - len(ok_rows),
        "changed_seed_count": len(changed),
        "zero_overlap_seed_count": len(zero_overlap),
        "model_only_selected_count": sum(
            len(row["model_only_ids"]) for row in successful
        ),
        "seed_with_model_only_count": sum(
            bool(row["model_only_ids"]) for row in successful
        ),
        "reject_all_seed_ids": [row["seed_id"] for row in reject_all],
        "selected_count": sum(len(row["model_ids"]) for row in successful),
        "total_tokens": sum(
            int((row.get("usage") or {}).get("total_tokens") or 0)
            for row in selector_rows
        ),
        "average_elapsed_ms": round(
            sum(float(row.get("elapsed_ms") or 0) for row in selector_rows)
            / max(1, len(selector_rows)),
            1,
        ),
        "comparisons": comparisons,
    }


def _render_markdown(summary: dict[str, Any]) -> str:
    reject_ids = ", ".join(summary["reject_all_seed_ids"]) or "none"
    return "\n".join(
        [
            "# Model Selector Comparison",
            "",
            "This report compares selection IDs only. It does not establish "
            "relevance or answer accuracy.",
            "",
            f"- Seeds: {summary['seed_count']} "
            f"({summary['ok_count']} ok, {summary['error_count']} errors)",
            f"- Changed selections: {summary['changed_seed_count']}",
            f"- Zero-overlap selections: {summary['zero_overlap_seed_count']}",
            f"- Model-only selected IDs: {summary['model_only_selected_count']} "
            f"across {summary['seed_with_model_only_count']} seeds",
            f"- Reject-all seeds: {reject_ids}",
            f"- Selected items: {summary['selected_count']}",
            f"- Total selector tokens: {summary['total_tokens']}",
            f"- Average selector latency: {summary['average_elapsed_ms']} ms",
            "",
        ]
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selections", type=Path, required=True)
    parser.add_argument("--evidence-results", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    selector_rows = json.loads(args.selections.read_text(encoding="utf-8"))
    if not isinstance(selector_rows, list):
        raise RuntimeError("selections JSON must be a list")
    summary = compare_selector_results(selector_rows, _read_rows(args.evidence_results))
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.write_text(_render_markdown(summary), encoding="utf-8")
    public_summary = {
        key: value for key, value in summary.items() if key != "comparisons"
    }
    print(json.dumps(public_summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
