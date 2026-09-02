from __future__ import annotations

import argparse
import asyncio
import csv
import gc
import json
import sqlite3
import tempfile
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote

import httpx
from httpx import ASGITransport

from .app import create_app
from .config import load_config


def _clone_database_read_only(source_path: Path, destination_path: Path) -> None:
    resolved = source_path.expanduser().resolve()
    uri = f"file:{quote(resolved.as_posix(), safe='/:')}?mode=ro&immutable=1"
    with sqlite3.connect(uri, uri=True) as source:
        with sqlite3.connect(destination_path) as destination:
            source.backup(destination)


def _load_seed_rows(seed_path: Path, db_path: Path) -> list[dict[str, str]]:
    with seed_path.open("r", encoding="utf-8-sig", newline="") as handle:
        seeds = [dict(row) for row in csv.DictReader(handle)]
    message_ids = [int(row["message_id"]) for row in seeds]
    if not message_ids:
        raise ValueError("Seed CSV contains no rows.")

    resolved = db_path.expanduser().resolve()
    uri = f"file:{quote(resolved.as_posix(), safe='/:')}?mode=ro&immutable=1"
    placeholders = ",".join("?" for _ in message_ids)
    with sqlite3.connect(uri, uri=True) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            f"""
SELECT id, timestamp, content, conversation_id
FROM messages
WHERE id IN ({placeholders})
""",
            message_ids,
        ).fetchall()
    by_id = {int(row["id"]): dict(row) for row in rows}
    missing = sorted(set(message_ids) - set(by_id))
    if missing:
        raise ValueError(f"Seed message IDs are missing from the DB: {missing}")

    hydrated = []
    for seed in seeds:
        source = by_id[int(seed["message_id"])]
        hydrated.append(
            {
                **seed,
                "timestamp": str(source.get("timestamp") or seed.get("timestamp") or ""),
                "conversation_id": str(source.get("conversation_id") or ""),
                "text": str(source.get("content") or seed.get("text") or ""),
            }
        )
    return hydrated


def _component(packet: dict[str, Any], name: str) -> dict[str, Any]:
    for item in packet.get("components") or []:
        if isinstance(item, dict) and item.get("name") == name:
            return item
    return {}


def _recent_ids(debug: dict[str, Any]) -> list[int]:
    values = (debug.get("source_ids") or {}).get("recent_turns") or []
    return [int(value.split(":", 1)[1]) for value in values if value.startswith("message:")]


def _is_true(value: Any) -> bool:
    return value is True or str(value).strip().lower() in {"1", "true", "yes"}


async def _run_benchmark(
    *,
    db_path: Path,
    seeds: list[dict[str, str]],
    retrieval_enabled: bool,
    router_enabled: bool = False,
    query_planner_enabled: bool = False,
    curated_sources: bool = False,
) -> list[dict[str, Any]]:
    base_cfg = load_config()
    with tempfile.TemporaryDirectory(prefix="context-selection-benchmark-") as tmp:
        temp_db = Path(tmp) / "benchmark.db"
        _clone_database_read_only(db_path, temp_db)
        cfg = replace(
            base_cfg,
            db_path=temp_db,
            worldbook_enabled=curated_sources and base_cfg.worldbook_enabled,
            worldbook_path=base_cfg.worldbook_path if curated_sources else None,
            worldbook_paths=base_cfg.worldbook_paths if curated_sources else (),
            core_anchors_enabled=curated_sources,
            mother_memory_enabled=curated_sources,
            mother_memory_inject_enabled=False,
            summary_enabled=False,
            retrieval_enabled=retrieval_enabled,
            retrieval_inject_enabled=False,
            retrieval_router_enabled=router_enabled,
            retrieval_query_planner_enabled=query_planner_enabled,
        )
        app = create_app(cfg)
        results = []
        async with httpx.AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://benchmark",
            timeout=30,
        ) as client:
            for seed in seeds:
                response = await client.post(
                    "/build_context",
                    json={
                        "conversation_id": seed["conversation_id"],
                        "user_text": seed["text"],
                        "as_of_message_id": int(seed["message_id"]),
                        "as_of_timestamp": seed["timestamp"],
                        "retrieval_enabled": retrieval_enabled,
                        "retrieval_inject": False,
                        "retrieval_router_enabled": router_enabled,
                        "retrieval_query_planner_enabled": query_planner_enabled,
                        "mother_memory_enabled": curated_sources,
                        "mother_memory_inject": False,
                        "core_anchors_inject": False,
                        "worldbook_inject": False,
                        "include": {
                            "worldbook": curated_sources,
                            "core_anchors": curated_sources,
                            "mother_memory": curated_sources,
                            "rolling_summary": False,
                        },
                    },
                )
                if response.status_code != 200:
                    results.append(
                        {
                            **seed,
                            "status": "error",
                            "error": response.text,
                        }
                    )
                    continue
                payload = response.json()
                debug = payload["debug"]
                packet = payload["context_packet"]
                recent = _component(packet, "recent_turns")
                retrieval = _component(packet, "kmlog_search")
                mother = _component(packet, "mother_memory")
                core = _component(packet, "core_anchors")
                worldbook = _component(packet, "wb_snippets")
                recent_ids = _recent_ids(debug)
                retrieval_items = retrieval.get("items") or []
                plan = retrieval.get("plan") or {}
                future_recent_ids = [
                    value for value in recent_ids if value >= int(seed["message_id"])
                ]
                future_retrieval_ids = [
                    item.get("id")
                    for item in retrieval_items
                    if str(item.get("timestamp") or "") >= seed["timestamp"]
                ]
                expected = set(seed["expected_context"].split("+"))
                results.append(
                    {
                        **seed,
                        "status": "ok",
                        "error": retrieval.get("error", ""),
                        "expected_recent": "recent" in expected,
                        "expected_retrieval": bool({"episodic", "semantic"} & expected),
                        "expected_no_context": expected == {"none"},
                        "recent_turn_count": int(recent.get("message_count") or 0),
                        "recent_source_ids": "|".join(map(str, recent_ids)),
                        "retrieval_result_count": int(retrieval.get("result_count") or 0),
                        "retrieval_source_ids": "|".join(
                            str(item.get("id")) for item in retrieval_items
                        ),
                        "retrieval_items_json": json.dumps(
                            [
                                {
                                    "id": item.get("id"),
                                    "timestamp": item.get("timestamp"),
                                    "relevance": item.get("relevance"),
                                    "token_hits": item.get("token_hits"),
                                    "preview": item.get("content_preview"),
                                }
                                for item in retrieval_items
                            ],
                            ensure_ascii=False,
                        ),
                        "retrieval_chars": int(retrieval.get("chars") or 0),
                        "router_enabled": bool(retrieval.get("router_enabled")),
                        "query_planner_enabled": bool(
                            retrieval.get("query_planner_enabled")
                        ),
                        "router_sources": "|".join(plan.get("sources") or []),
                        "router_domains": "|".join(
                            plan.get("matched_domains") or []
                        ),
                        "planned_search_query": retrieval.get("search_query", ""),
                        "router_skipped_reason": retrieval.get(
                            "skipped_reason", ""
                        ),
                        "mother_result_count": int(mother.get("result_count") or 0),
                        "mother_paths": "|".join(
                            str(item.get("path")) for item in mother.get("items") or []
                        ),
                        "mother_items_json": json.dumps(
                            mother.get("items") or [], ensure_ascii=False
                        ),
                        "core_result_count": len(core.get("items") or []),
                        "core_anchor_keys": "|".join(
                            str(item.get("anchor_key"))
                            for item in core.get("items") or []
                        ),
                        "core_items_json": json.dumps(
                            core.get("items") or [], ensure_ascii=False
                        ),
                        "worldbook_result_count": len(worldbook.get("items") or []),
                        "worldbook_items": "|".join(
                            str(item.get("name") or item.get("id"))
                            for item in worldbook.get("items") or []
                        ),
                        "worldbook_items_json": json.dumps(
                            worldbook.get("items") or [], ensure_ascii=False
                        ),
                        "total_token_estimate": int(
                            (debug.get("token_estimates") or {}).get("total") or 0
                        ),
                        "future_recent_ids": "|".join(map(str, future_recent_ids)),
                        "future_retrieval_ids": "|".join(
                            str(value) for value in future_retrieval_ids
                        ),
                        "future_leak": bool(future_recent_ids or future_retrieval_ids),
                        "retrieval_relevant": "",
                        "recent_relevant": "",
                        "review_notes": "",
                    }
                )
        del app
        gc.collect()
    return results


def _write_outputs(output_dir: Path, results: list[dict[str, Any]]) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.csv"
    summary_path = output_dir / "summary.json"
    report_path = output_dir / "report.md"

    fieldnames: list[str] = []
    for row in results:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with results_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    ok_rows = [row for row in results if row.get("status") == "ok"]
    errors = [row for row in results if row.get("status") != "ok" or row.get("error")]
    by_theme: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in ok_rows:
        by_theme[str(row.get("theme") or "unknown")].append(row)
    summary = {
        "seed_count": len(results),
        "ok_count": len(ok_rows),
        "error_count": len(errors),
        "router_enabled": any(
            _is_true(row.get("router_enabled")) for row in ok_rows
        ),
        "query_planner_enabled": any(
            _is_true(row.get("query_planner_enabled")) for row in ok_rows
        ),
        "curated_sources_enabled": any(
            int(row.get("mother_result_count") or 0) > 0
            or int(row.get("core_result_count") or 0) > 0
            or int(row.get("worldbook_result_count") or 0) > 0
            for row in ok_rows
        ),
        "mother_nonempty_count": sum(
            int(row.get("mother_result_count") or 0) > 0 for row in ok_rows
        ),
        "core_nonempty_count": sum(
            int(row.get("core_result_count") or 0) > 0 for row in ok_rows
        ),
        "worldbook_nonempty_count": sum(
            int(row.get("worldbook_result_count") or 0) > 0 for row in ok_rows
        ),
        "curated_nonempty_count": sum(
            any(
                int(row.get(key) or 0) > 0
                for key in (
                    "mother_result_count",
                    "core_result_count",
                    "worldbook_result_count",
                )
            )
            for row in ok_rows
        ),
        "expected_retrieval_curated_nonempty_count": sum(
            _is_true(row.get("expected_retrieval"))
            and any(
                int(row.get(key) or 0) > 0
                for key in (
                    "mother_result_count",
                    "core_result_count",
                    "worldbook_result_count",
                )
            )
            for row in ok_rows
        ),
        "no_context_with_curated_count": sum(
            _is_true(row.get("expected_no_context"))
            and any(
                int(row.get(key) or 0) > 0
                for key in (
                    "mother_result_count",
                    "core_result_count",
                    "worldbook_result_count",
                )
            )
            for row in ok_rows
        ),
        "future_leak_count": sum(_is_true(row.get("future_leak")) for row in ok_rows),
        "retrieval_nonempty_count": sum(
            int(row.get("retrieval_result_count") or 0) > 0 for row in ok_rows
        ),
        "expected_retrieval_seed_count": sum(
            _is_true(row.get("expected_retrieval")) for row in ok_rows
        ),
        "expected_retrieval_nonempty_count": sum(
            _is_true(row.get("expected_retrieval"))
            and int(row.get("retrieval_result_count") or 0) > 0
            for row in ok_rows
        ),
        "unexpected_retrieval_count": sum(
            not _is_true(row.get("expected_retrieval"))
            and int(row.get("retrieval_result_count") or 0) > 0
            for row in ok_rows
        ),
        "no_context_seed_count": sum(
            _is_true(row.get("expected_no_context")) for row in ok_rows
        ),
        "no_context_with_retrieval_count": sum(
            _is_true(row.get("expected_no_context"))
            and int(row.get("retrieval_result_count") or 0) > 0
            for row in ok_rows
        ),
        "themes": {
            theme: {
                "count": len(rows),
                "retrieval_nonempty": sum(
                    int(row.get("retrieval_result_count") or 0) > 0 for row in rows
                ),
                "curated_nonempty": sum(
                    any(
                        int(row.get(key) or 0) > 0
                        for key in (
                            "mother_result_count",
                            "core_result_count",
                            "worldbook_result_count",
                        )
                    )
                    for row in rows
                ),
                "average_recent_turns": round(
                    sum(int(row.get("recent_turn_count") or 0) for row in rows)
                    / len(rows),
                    2,
                ),
                "average_token_estimate": round(
                    sum(int(row.get("total_token_estimate") or 0) for row in rows)
                    / len(rows),
                    2,
                ),
            }
            for theme, rows in sorted(by_theme.items())
        },
        "limitations": [
            "This pilot evaluates selection only and does not call an answer model.",
            (
                "Curated sources use current snapshots and are not valid "
                "historical as-of evidence."
                if any(
                    int(row.get("mother_result_count") or 0) > 0
                    or int(row.get("core_result_count") or 0) > 0
                    or int(row.get("worldbook_result_count") or 0) > 0
                    for row in ok_rows
                )
                else "Mutable curated sources and rolling summary are disabled."
            ),
            "Retrieval relevance still requires manual labels in results.csv.",
        ],
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Context-selection benchmark pilot",
        "",
        f"- Seeds: {summary['seed_count']}",
        f"- Successful builds: {summary['ok_count']}",
        f"- Errors: {summary['error_count']}",
        f"- Router enabled: {summary['router_enabled']}",
        f"- Query planner enabled: {summary['query_planner_enabled']}",
        f"- Curated-source candidates present: {summary['curated_sources_enabled']}",
        f"- Mother-memory non-empty: {summary['mother_nonempty_count']}",
        f"- Core-anchor non-empty: {summary['core_nonempty_count']}",
        f"- World Book non-empty: {summary['worldbook_nonempty_count']}",
        f"- Any curated source non-empty: {summary['curated_nonempty_count']}",
        (
            "- Expected-retrieval seeds with curated candidates: "
            f"{summary['expected_retrieval_curated_nonempty_count']} / "
            f"{summary['expected_retrieval_seed_count']}"
        ),
        (
            "- No-context seeds with curated candidates: "
            f"{summary['no_context_with_curated_count']} / "
            f"{summary['no_context_seed_count']}"
        ),
        f"- Future leaks: {summary['future_leak_count']}",
        f"- Non-empty retrievals: {summary['retrieval_nonempty_count']}",
        (
            "- Expected-retrieval seeds with candidates: "
            f"{summary['expected_retrieval_nonempty_count']} / "
            f"{summary['expected_retrieval_seed_count']}"
        ),
        f"- Unexpected non-empty retrievals: {summary['unexpected_retrieval_count']}",
        (
            "- No-context seeds with retrieval candidates: "
            f"{summary['no_context_with_retrieval_count']} / "
            f"{summary['no_context_seed_count']}"
        ),
        "",
        "Retrieval presence is not retrieval correctness. Fill `retrieval_relevant`",
        "and `recent_relevant` in `results.csv` before calculating precision-like",
        "metrics.",
        "",
        "## By theme",
        "",
        "| Theme | N | Chat retrieval | Curated | Avg recent turns | Avg tokens |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for theme, values in summary["themes"].items():
        lines.append(
            f"| {theme} | {values['count']} | {values['retrieval_nonempty']} | "
            f"{values['curated_nonempty']} | "
            f"{values['average_recent_turns']} | {values['average_token_estimate']} |"
        )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "results_csv": results_path,
        "summary_json": summary_path,
        "report_md": report_path,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a no-model context-selection benchmark with historical cutoffs."
    )
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark_outputs/context_selection_pilot"),
    )
    parser.add_argument(
        "--no-retrieval",
        action="store_true",
        help="Benchmark recent-turn selection without calling KMLog search.",
    )
    parser.add_argument(
        "--router",
        action="store_true",
        help="Enable deterministic source routing and query planning.",
    )
    parser.add_argument(
        "--query-planner",
        action="store_true",
        help="Use the router plan to rewrite the KMLog search query.",
    )
    parser.add_argument(
        "--curated-sources",
        action="store_true",
        help=(
            "Retrieve current mother-memory, Core Anchor, and World Book "
            "snapshots without injecting them."
        ),
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    seeds = _load_seed_rows(args.seed, args.db)
    results = asyncio.run(
        _run_benchmark(
            db_path=args.db,
            seeds=seeds,
            retrieval_enabled=not args.no_retrieval,
            router_enabled=args.router,
            query_planner_enabled=args.query_planner,
            curated_sources=args.curated_sources,
        )
    )
    paths = _write_outputs(args.output_dir, results)
    print(json.dumps({key: str(value.resolve()) for key, value in paths.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
