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
from chat_proxy.context_builder import _rerank_kmlog_results, _timestamp_at_or_after
from chat_proxy.retrieval_model_selector import (
    SELECTOR_PROMPT_VERSION,
    SELECTOR_SYSTEM_PROMPT,
    build_selector_candidates,
    resolve_selector_evidence,
    select_retrieval_candidates,
)
from chat_proxy.retrieval_planner import RetrievalPlan


EMPTY_PLAN = RetrievalPlan((), None, (), (), (), (), ())

def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    evidence_dir = args.run_dir / "evidence"
    results_path = evidence_dir / "results.csv"
    snapshots_path = evidence_dir / "retrieval_snapshots.json"
    rows = _read_rows(results_path)
    snapshots = json.loads(snapshots_path.read_text(encoding="utf-8"))
    if not isinstance(snapshots, list) or len(rows) != len(snapshots):
        raise RuntimeError("results.csv and retrieval_snapshots.json do not align")

    cfg = load_config()
    requested_seed_ids = set(args.seed_id)
    available_seed_ids = {str(row["message_id"]) for row in rows}
    missing_seed_ids = sorted(requested_seed_ids - available_seed_ids)
    if missing_seed_ids:
        raise RuntimeError(f"requested seed IDs are missing: {missing_seed_ids}")
    eligible = []
    for row, snapshot in zip(rows, snapshots):
        if requested_seed_ids and str(row["message_id"]) not in requested_seed_ids:
            continue
        candidates = snapshot.get("candidate_items") or []
        if not isinstance(candidates, list) or not candidates:
            continue
        as_of_timestamp = str(snapshot.get("as_of_timestamp") or "").strip()
        future_ids = [
            candidate.get("id")
            for candidate in candidates
            if isinstance(candidate, dict)
            and as_of_timestamp
            and _timestamp_at_or_after(candidate.get("timestamp"), as_of_timestamp)
        ]
        if future_ids:
            raise RuntimeError(
                f"seed {row['message_id']} candidate pool crosses historical cutoff: "
                f"{future_ids}"
            )
        prepared, preparation_stats = _rerank_kmlog_results(
            candidates, plan=EMPTY_PLAN, limit=len(candidates)
        )
        bounded = prepared[: args.max_candidates]
        model_candidates = build_selector_candidates(
            bounded, args.max_candidate_chars
        )
        if model_candidates:
            model_candidate_ids = {
                candidate["id"] for candidate in model_candidates
            }
            bounded = [
                candidate
                for candidate in bounded
                if candidate.get("id") in model_candidate_ids
            ]
            eligible.append(
                (row, snapshot, bounded, model_candidates, preparation_stats)
            )

    manifest = {
        "selector_prompt_version": SELECTOR_PROMPT_VERSION,
        "model": cfg.summary_model,
        "input_results_sha256": _sha256(results_path),
        "input_snapshots_sha256": _sha256(snapshots_path),
        "eligible_seed_count": len(eligible),
        "requested_seed_ids": list(args.seed_id),
        "selection_limit": args.limit,
        "max_candidates": args.max_candidates,
        "max_candidate_chars": args.max_candidate_chars,
        "request": {
            "temperature": 0,
            "thinking_mode": args.thinking,
            "max_output_tokens": args.max_output_tokens,
            "timeout_seconds": args.timeout,
        },
        "selector_system_prompt_sha256": hashlib.sha256(
            SELECTOR_SYSTEM_PROMPT.encode("utf-8")
        ).hexdigest(),
        "code_sha256": {
            "chat_proxy/retrieval_model_selector.py": _sha256(
                ROOT / "chat_proxy/retrieval_model_selector.py"
            ),
            "benchmarks/context_selection/run_model_selector.py": _sha256(
                Path(__file__)
            ),
        },
        "model_payload_sha256": hashlib.sha256(
            json.dumps(
                [
                    {
                        "seed_id": row["message_id"],
                        "query": str(
                            snapshot.get("original_query") or row.get("text") or ""
                        ),
                        "candidates": model_candidates,
                    }
                    for row, snapshot, _, model_candidates, _ in eligible
                ],
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
        "candidate_counts": {
            row["message_id"]: len(model_candidates)
            for row, _, _, model_candidates, _ in eligible
        },
        "candidate_preparation_stats": {
            row["message_id"]: stats for row, _, _, _, stats in eligible
        },
    }
    if not args.live:
        return {"dry_run": True, "manifest": manifest, "results": []}
    if args.output_dir.exists():
        raise RuntimeError("Output exists; choose a new directory")
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    output_rows = []
    for index, (row, snapshot, candidates, model_candidates, _) in enumerate(
        eligible, start=1
    ):
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
                thinking_mode=args.thinking,
                max_output_tokens=args.max_output_tokens,
            )
            record.update(
                status="ok",
                selected=resolve_selector_evidence(
                    selection,
                    model_candidates,
                    candidates,
                ),
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
    parser.add_argument(
        "--thinking",
        choices=("default", "disabled", "enabled"),
        default="disabled",
    )
    parser.add_argument("--max-output-tokens", type=int, default=1200)
    parser.add_argument("--seed-id", action="append", default=[])
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Deprecated compatibility flag; dry-run is now the default.",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Send the manifest-previewed payload to the configured upstream.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.live and args.dry_run:
        raise SystemExit("--live and --dry-run are mutually exclusive")
    if args.output_dir is None:
        args.output_dir = args.run_dir / "model_selector_b"
    result = asyncio.run(_run(args))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
