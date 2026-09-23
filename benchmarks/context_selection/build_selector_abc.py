"""Build a same-pool, same-budget A/B/C selector comparison without model calls."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN = ROOT / "benchmark_outputs/context_selection_prod_evidence_anchor_v6"
sys.path.insert(0, str(ROOT))

from chat_proxy.context_builder import _rerank_kmlog_results
from chat_proxy.retrieval_evidence import contains_kmlog_term, render_kmlog_results
from chat_proxy.retrieval_model_selector import (
    build_selector_candidates,
    resolve_selector_evidence,
)
from chat_proxy.retrieval_planner import RetrievalPlan


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _plan_from_snapshot(snapshot: Mapping[str, Any]) -> RetrievalPlan:
    raw = snapshot.get("plan") or {}
    return RetrievalPlan(
        sources=tuple(raw.get("sources") or ()),
        search_query=raw.get("search_query"),
        matched_domains=tuple(raw.get("matched_domains") or ()),
        matched_terms=tuple(raw.get("matched_terms") or ()),
        required_terms=tuple(raw.get("required_terms") or ()),
        optional_terms=tuple(raw.get("optional_terms") or ()),
        reasons=tuple(raw.get("reasons") or ()),
    )


def _items_by_ids(
    candidates: list[Mapping[str, Any]], ids: list[Any]
) -> list[dict[str, Any]]:
    by_id = {candidate.get("id"): candidate for candidate in candidates}
    missing = [candidate_id for candidate_id in ids if candidate_id not in by_id]
    if missing:
        raise RuntimeError(f"candidate IDs are missing from snapshot: {missing}")
    return [dict(by_id[candidate_id]) for candidate_id in ids]


def _model_selected_candidates(
    resolved: list[Mapping[str, Any]],
    source_candidates: list[Mapping[str, Any]],
    required_terms: tuple[str, ...],
) -> list[dict[str, Any]]:
    source_by_id = {candidate.get("id"): candidate for candidate in source_candidates}
    result = []
    for selected in resolved:
        candidate = dict(source_by_id[selected["id"]])
        visible_text = "\n...\n".join(selected.get("evidence") or [])
        candidate["matched_excerpt"] = visible_text
        candidate["content_preview"] = visible_text
        candidate["planner_required_matches"] = [
            term for term in required_terms if contains_kmlog_term(visible_text, term)
        ]
        result.append(candidate)
    return result


def _selector_visible_candidates(
    model_candidates: list[Mapping[str, Any]],
    source_candidates: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Build rule-selector inputs from exactly the evidence serialized for the model."""
    source_by_id = {candidate.get("id"): candidate for candidate in source_candidates}
    result = []
    for visible in model_candidates:
        candidate_id = visible.get("id")
        source = source_by_id.get(candidate_id)
        if source is None:
            raise RuntimeError(
                f"selector-visible candidate {candidate_id!r} has no source candidate"
            )
        visible_excerpts = [
            dict(excerpt)
            for excerpt in visible.get("evidence_excerpts") or []
            if isinstance(excerpt, Mapping)
        ]
        visible_text = "\n...\n".join(
            str(excerpt.get("text") or "") for excerpt in visible_excerpts
        )
        candidate = {
            key: source.get(key)
            for key in (
                "match_type",
                "relevance",
                "token_hits",
                "evidence_origin",
                "title_only",
                "duplicate_count",
                "duplicate_provenance",
                "match_spans",
            )
            if key in source
        }
        candidate.update(
            {
                "id": candidate_id,
                "timestamp": visible.get("timestamp"),
                "role": visible.get("role"),
                "conversation_title": visible.get("conversation_title"),
                "body_matched_terms": list(
                    visible.get("body_matched_terms") or []
                ),
                "source_message_ids": list(
                    visible.get("source_message_ids") or []
                ),
                "evidence_excerpts": visible_excerpts,
                "evidence_version": 1,
                "matched_excerpt": visible_text,
                "content_preview": visible_text,
                # Deduplicate on model-visible evidence, not hidden source text.
                "content_hash": hashlib.sha256(
                    visible_text.encode("utf-8")
                ).hexdigest(),
            }
        )
        result.append(candidate)
    return result


def _render_group(
    *, input_ids: list[Any], selected: list[dict[str, Any]], total_chars: int
) -> dict[str, Any]:
    content, stats = render_kmlog_results(selected, total_chars=total_chars)
    return {
        "input_ids": input_ids,
        "selected_before_budget_ids": [item.get("id") for item in selected],
        "selected_after_budget_ids": stats["selected_after_budget_ids"],
        "rendered_body_chars": stats["budget_used_chars"],
        "rendered_content": content,
        "rendered_items": stats["items"],
        "budget_dropped": stats["budget_dropped"],
    }


def evaluate_seed(
    *,
    snapshot: Mapping[str, Any],
    selector_row: Mapping[str, Any],
    selection_limit: int,
    max_candidate_chars: int,
    total_chars: int,
) -> dict[str, Any]:
    raw_candidates = snapshot.get("candidate_items") or []
    if not isinstance(raw_candidates, list):
        raise RuntimeError("snapshot candidate_items must be a list")
    common_ids = list(selector_row.get("input_candidate_ids") or [])
    common_pool = _items_by_ids(raw_candidates, common_ids)
    plan = _plan_from_snapshot(snapshot)

    baseline_input_ids = list(snapshot.get("rerank_input_ids") or [])
    baseline_input = _items_by_ids(raw_candidates, baseline_input_ids)
    baseline_selected, baseline_stats = _rerank_kmlog_results(
        baseline_input, plan=plan, limit=selection_limit
    )
    model_candidates = build_selector_candidates(common_pool, max_candidate_chars)
    visible_ids = [candidate.get("id") for candidate in model_candidates]
    dropped_nonserializable_ids = [
        candidate_id for candidate_id in common_ids if candidate_id not in visible_ids
    ]
    visible_common_pool = _selector_visible_candidates(
        model_candidates, common_pool
    )
    expanded_selected, expanded_stats = _rerank_kmlog_results(
        visible_common_pool, plan=plan, limit=selection_limit
    )

    resolved = resolve_selector_evidence(
        selector_row, model_candidates, common_pool
    )
    model_selected = _model_selected_candidates(
        resolved, visible_common_pool, plan.required_terms
    )

    return {
        "seed_id": str(selector_row.get("seed_id") or ""),
        "theme": selector_row.get("theme"),
        "candidate_pool_audit": {
            "declared_input_ids": common_ids,
            "visible_input_ids": visible_ids,
            "dropped_nonserializable_ids": dropped_nonserializable_ids,
        },
        "groups": {
            "a_backend_pool_rule": {
                **_render_group(
                    input_ids=baseline_input_ids,
                    selected=baseline_selected,
                    total_chars=total_chars,
                ),
                "filter_stats": baseline_stats,
            },
            "b_common_pool_rule": {
                **_render_group(
                    input_ids=visible_ids,
                    selected=expanded_selected,
                    total_chars=total_chars,
                ),
                "filter_stats": expanded_stats,
                "candidate_representation": "selector_visible_v1",
            },
            "c_common_pool_model": {
                **_render_group(
                    input_ids=visible_ids,
                    selected=model_selected,
                    total_chars=total_chars,
                ),
                "reject_all_reason": selector_row.get("reject_all_reason"),
                "selection_audit": resolved,
                "candidate_representation": "selector_visible_v1",
            },
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--selections", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--total-chars", type=int, default=1200)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    selections_path = args.selections or args.run_dir / "model_selector_b/selections.json"
    selector_dir = selections_path.parent
    selector_manifest = json.loads(
        (selector_dir / "manifest.json").read_text(encoding="utf-8")
    )
    selections = json.loads(selections_path.read_text(encoding="utf-8"))
    evidence_dir = args.run_dir / "evidence"
    snapshots_path = evidence_dir / "retrieval_snapshots.json"
    rows = _read_rows(evidence_dir / "results.csv")
    snapshots = json.loads(
        snapshots_path.read_text(encoding="utf-8")
    )
    if len(rows) != len(snapshots):
        raise RuntimeError("results.csv and retrieval_snapshots.json do not align")
    snapshots_by_seed = {
        str(row["message_id"]): snapshot for row, snapshot in zip(rows, snapshots)
    }
    output_dir = args.output_dir or selector_dir / "abc_same_budget"
    if output_dir.exists():
        raise RuntimeError("Output exists; choose a new directory")

    results = []
    for selector_row in selections:
        if selector_row.get("status") != "ok":
            continue
        seed_id = str(selector_row.get("seed_id") or "")
        if seed_id not in snapshots_by_seed:
            raise RuntimeError(f"selector seed {seed_id!r} has no snapshot")
        results.append(
            evaluate_seed(
                snapshot=snapshots_by_seed[seed_id],
                selector_row=selector_row,
                selection_limit=int(selector_manifest["selection_limit"]),
                max_candidate_chars=int(selector_manifest["max_candidate_chars"]),
                total_chars=args.total_chars,
            )
        )

    def ids(row: Mapping[str, Any], group: str) -> list[Any]:
        return row["groups"][group]["selected_after_budget_ids"]

    a_vs_b_changed = [
        row["seed_id"]
        for row in results
        if ids(row, "a_backend_pool_rule") != ids(row, "b_common_pool_rule")
    ]
    b_vs_c_changed = [
        row["seed_id"]
        for row in results
        if ids(row, "b_common_pool_rule") != ids(row, "c_common_pool_model")
    ]
    summary = {
        "seed_count": len(results),
        "total_chars_per_group": args.total_chars,
        "input_representation_adjusted_seed_ids": [
            row["seed_id"]
            for row in results
            if row["candidate_pool_audit"]["dropped_nonserializable_ids"]
        ],
        "a_vs_b_changed_seed_count": len(a_vs_b_changed),
        "a_vs_b_changed_seed_ids": a_vs_b_changed,
        "b_vs_c_changed_seed_count": len(b_vs_c_changed),
        "b_vs_c_changed_seed_ids": b_vs_c_changed,
        "model_reject_all_seed_ids": [
            row["seed_id"]
            for row in results
            if not ids(row, "c_common_pool_model")
        ],
        "quality_claim": "none; this artifact isolates pool and selector behavior",
    }
    output_dir.mkdir(parents=True)
    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "source_selector_manifest": selector_manifest,
                "input_selections_sha256": _sha256(selections_path),
                "input_snapshots_sha256": _sha256(snapshots_path),
                "total_chars_per_group": args.total_chars,
                "code_sha256": {
                    "benchmarks/context_selection/build_selector_abc.py": _sha256(
                        Path(__file__)
                    ),
                    "chat_proxy/context_builder.py": _sha256(
                        ROOT / "chat_proxy/context_builder.py"
                    ),
                    "chat_proxy/retrieval_evidence.py": _sha256(
                        ROOT / "chat_proxy/retrieval_evidence.py"
                    ),
                },
                "group_contract": {
                    "a": "backend top pool, rule selector, shared renderer",
                    "b": "serializable model-visible common pool, rule selector, shared renderer",
                    "c": "serializable model-visible common pool, model selector, shared renderer",
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
