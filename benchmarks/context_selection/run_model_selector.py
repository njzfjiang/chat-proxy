"""Run a bounded model selector over a saved evidence candidate pool."""
from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN = ROOT / "benchmark_outputs/context_selection_prod_evidence_anchor_v6"
sys.path.insert(0, str(ROOT))

from chat_proxy.config import load_config
from chat_proxy.retrieval_model_selector import (
    SELECTOR_PROMPT_VERSION,
    build_selector_candidates,
    select_retrieval_candidates,
)

def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _selected_evidence(
    selection: dict[str, Any], candidates: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    by_id = {candidate.get("id"): candidate for candidate in candidates}
    result = []
    for item in selection.get("selected") or []:
        candidate = by_id[item["id"]]
        excerpts = candidate.get("evidence_excerpts") or []
        result.append(
            {
                **item,
                "role": candidate.get("role"),
                "timestamp": candidate.get("timestamp"),
                "source_message_ids": candidate.get("source_message_ids") or [],
                "evidence": [
                    excerpts[index].get("text")
                    for index in item["evidence_excerpt_indices"]
                ],
            }
        )
    return result


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    evidence_dir = args.run_dir / "evidence"
    results_path = evidence_dir / "results.csv"
    snapshots_path = evidence_dir / "retrieval_snapshots.json"
    rows = _read_rows(results_path)
    snapshots = json.loads(snapshots_path.read_text(encoding="utf-8"))
    if not isinstance(snapshots, list) or len(rows) != len(snapshots):
        raise RuntimeError("results.csv and retrieval_snapshots.json do not align")

    cfg = load_config()
    eligible = []
    for row, snapshot in zip(rows, snapshots):
        candidates = snapshot.get("candidate_items") or []
        if not isinstance(candidates, list) or not candidates:
            continue
        bounded = candidates[: args.max_candidates]
        model_candidates = build_selector_candidates(
            bounded, args.max_candidate_chars
        )
        if model_candidates:
            eligible.append((row, snapshot, bounded, model_candidates))

    manifest = {
        "selector_prompt_version": SELECTOR_PROMPT_VERSION,
        "model": cfg.summary_model,
        "input_results_sha256": _sha256(results_path),
        "input_snapshots_sha256": _sha256(snapshots_path),
        "eligible_seed_count": len(eligible),
        "selection_limit": args.limit,
        "max_candidates": args.max_candidates,
        "max_candidate_chars": args.max_candidate_chars,
        "candidate_counts": {
            row["message_id"]: len(model_candidates)
            for row, _, _, model_candidates in eligible
        },
    }
    if args.dry_run:
        return {"dry_run": True, "manifest": manifest, "results": []}
    if args.output_dir.exists():
        raise RuntimeError("Output exists; choose a new directory")
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    output_rows = []
    for index, (row, snapshot, candidates, _) in enumerate(eligible, start=1):
        started = time.perf_counter()
        record = {
            "seed_id": row["message_id"],
            "theme": row.get("theme"),
            "input_candidate_ids": [candidate.get("id") for candidate in candidates],
        }
        try:
            selection = await select_retrieval_candidates(
                cfg=cfg,
                query=str(snapshot.get("original_query") or row.get("text") or ""),
                candidates=candidates,
                limit=args.limit,
                max_candidate_chars=args.max_candidate_chars,
                timeout_seconds=args.timeout,
            )
            record.update(
                status="ok",
                selected=_selected_evidence(selection, candidates),
                reject_all_reason=selection.get("reject_all_reason"),
                usage=selection.get("usage") or {},
            )
        except Exception as exc:
            record.update(status="error", error=str(exc))
        record["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 1)
        output_rows.append(record)
        (args.output_dir / "selections.json").write_text(
            json.dumps(output_rows, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            f"selector: {index}/{len(eligible)} seed={row['message_id']} "
            f"status={record['status']}",
            flush=True,
        )

    summary = {
        "seed_count": len(output_rows),
        "ok_count": sum(row["status"] == "ok" for row in output_rows),
        "error_count": sum(row["status"] != "ok" for row in output_rows),
        "reject_all_count": sum(
            row["status"] == "ok" and not row.get("selected") for row in output_rows
        ),
        "selected_count": sum(len(row.get("selected") or []) for row in output_rows),
        "total_tokens": sum(
            int((row.get("usage") or {}).get("total_tokens") or 0)
            for row in output_rows
        ),
        "average_elapsed_ms": round(
            sum(row["elapsed_ms"] for row in output_rows) / max(1, len(output_rows)),
            1,
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return {"dry_run": False, "manifest": manifest, "summary": summary}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--max-candidates", type=int, default=80)
    parser.add_argument("--max-candidate-chars", type=int, default=600)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.output_dir is None:
        args.output_dir = args.run_dir / "model_selector_b"
    result = asyncio.run(_run(args))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
