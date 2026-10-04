"""Run selection-only A/B against the sibling backend without network access."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib
import json
import sqlite3
import sys
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT.parent / "kmlog-search"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(BACKEND))

from chat_proxy import context_selection_benchmark as benchmark
from chat_proxy import context_builder


MANIFEST_CODE_PATHS = (
    ROOT / "benchmarks/context_selection/run_local_evidence_ab.py",
    ROOT / "chat_proxy/context_builder.py",
    ROOT / "chat_proxy/retrieval_planner.py",
    ROOT / "chat_proxy/retrieval_evidence.py",
    ROOT / "chat_proxy/context_selection_benchmark.py",
    BACKEND / "servers/app.py",
    BACKEND / "servers/search_sqlite.py",
    BACKEND / "servers/message_search.py",
)
DEFAULT_GOLD_LABELS = (
    ROOT / "benchmarks/context_selection/annotations/gold_labels_v2.json"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a local, model-free context selection benchmark."
    )
    parser.add_argument(
        "--seed-id",
        action="append",
        default=[],
        help="Run only this original or rematched message ID; repeat as needed.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "benchmark_outputs/context_selection_prod_evidence_sentence_v11",
        help="Output directory; must not already exist.",
    )
    parser.add_argument(
        "--gold-labels",
        type=Path,
        default=DEFAULT_GOLD_LABELS,
        help="JSON overrides for reviewed expected_context labels.",
    )
    return parser.parse_args()


def _select_seeds(
    seeds: list[dict[str, str]], seed_ids: list[str]
) -> list[dict[str, str]]:
    requested = {str(value).strip() for value in seed_ids if str(value).strip()}
    if not requested:
        return seeds
    selected = [
        seed
        for seed in seeds
        if requested
        & {
            str(seed.get("seed_message_id") or ""),
            str(seed.get("message_id") or ""),
        }
    ]
    matched = {
        requested_id
        for seed in selected
        for requested_id in requested
        if requested_id
        in {
            str(seed.get("seed_message_id") or ""),
            str(seed.get("message_id") or ""),
        }
    }
    missing = sorted(requested - matched)
    if missing:
        raise ValueError(f"Seed IDs were not found: {', '.join(missing)}")
    return selected


def _apply_gold_labels(
    seeds: list[dict[str, str]], path: Path
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    labels = payload.get("labels") or []
    by_id = {str(label["seed_id"]): label for label in labels}
    applied: list[dict[str, str]] = []
    updated: list[dict[str, str]] = []
    matched: set[str] = set()
    for seed in seeds:
        aliases = {
            str(seed.get("seed_message_id") or ""),
            str(seed.get("message_id") or ""),
        }
        label_id = next((value for value in aliases if value in by_id), None)
        if label_id is None:
            updated.append(seed)
            continue
        label = by_id[label_id]
        old_context = str(seed.get("expected_context") or "")
        new_context = str(label["expected_context"])
        updated.append({**seed, "expected_context": new_context})
        applied.append(
            {
                "seed_id": label_id,
                "old_expected_context": old_context,
                "new_expected_context": new_context,
            }
        )
        matched.add(label_id)
    missing = sorted(set(by_id) - matched)
    if missing:
        raise ValueError(f"Gold label seed IDs were not found: {', '.join(missing)}")
    return updated, applied


def main():
    args = _parse_args()
    db = BACKEND / "chat_data/chat_search_prod.db"
    baseline = ROOT / "benchmark_outputs/context_selection_prod_router_planner_fixed_v2"
    output = args.output.resolve()
    if output.exists():
        raise RuntimeError("Output exists; preserve prior runs by choosing a new directory")
    cfg = benchmark.load_config()
    backend_app = importlib.import_module("servers.app")
    backend_module = importlib.import_module(backend_app.search_messages.__module__)

    working_db = []
    original_clone = benchmark._clone_database_read_only

    def clone(source, destination):
        original_clone(source, destination)
        working_db[:] = [destination]

    def connect():
        if not working_db:
            raise RuntimeError("Backend cannot access the source before cloning")
        conn = sqlite3.connect(working_db[0])
        conn.row_factory = sqlite3.Row
        return conn

    # Source is read-only; curated lazy indexing writes only to the runner's clone.
    backend_module.get_connection = connect
    # Freeze the already-ingested Mother snapshot. Reingesting Windows Markdown
    # into a VPS-derived DB can collide on globally unique section paths.
    def frozen_mother(*args, **kwargs):
        with connect() as conn:
            count = conn.execute("SELECT count(*) FROM memory_mother_sections").fetchone()[0]
        if not count:
            raise RuntimeError("Frozen Mother snapshot is empty")
        return {"refreshed": False, "section_count": count}

    backend_module.ensure_mother_markdown_ingested = frozen_mother
    backend_app.auth = lambda token: None
    client = TestClient(backend_app.app)
    cfg = replace(cfg, db_path=db, retrieval_router_enabled=True,
                  retrieval_query_planner_enabled=True, recent_goals_enabled=True,
                  reviewed_memory_enabled=True, mother_memory_enabled=True,
                  upstream_base="http://model-disabled.invalid", upstream_api_key=None,
                  kmlog_search_url="http://local-benchmark", kmlog_search_api_key=None,
                  recent_goals_url="http://local-benchmark", recent_goals_api_key=None,
                  reviewed_memory_url="http://local-benchmark", reviewed_memory_api_key=None,
                  mother_memory_url="http://local-benchmark", mother_memory_api_key=None,
                  core_anchors_url="http://local-benchmark", core_anchors_api_key=None,
                  summary_enabled=False, daily_summary_enabled=False)
    all_seeds, applied_gold_labels = _apply_gold_labels(
        benchmark._load_seed_rows(baseline / "results.csv", db),
        args.gold_labels.resolve(),
    )
    seeds = _select_seeds(all_seeds, args.seed_id)
    original_client = httpx.Client
    original_search = context_builder._kmlog_search_messages
    output.mkdir(parents=True)

    def fingerprint(path):
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    public_cfg = {key: str(value) if isinstance(value, Path) else value
                  for key, value in asdict(cfg).items()
                  if not any(word in key for word in ("key", "url", "upstream", "provider"))}
    (output / "manifest.json").write_text(json.dumps({
        "trace_schema_version": 2,
        "db": str(db), "db_sha256": fingerprint(db),
        "baseline_csv_sha256": fingerprint(baseline / "results.csv"),
        "baseline_zip_sha256": fingerprint(baseline.with_suffix(".zip")),
        "gold_labels_sha256": fingerprint(args.gold_labels.resolve()),
        "applied_gold_labels": applied_gold_labels,
        "selected_seed_ids": [seed["seed_message_id"] for seed in seeds],
        "config": public_cfg,
        "notes": ["Current curated snapshots; not historical as-of evidence",
                  "Mother lazy refresh disabled; use existing production DB sections",
                  "A/B toggles backend include_evidence; both use current proxy reranker",
                  "Evidence arm uses sentence-aware clipping and dynamic budget reallocation",
                  "In-process HTTP transport; timeouts/network performance not evaluated"],
        "code_sha256": {
            str(path.relative_to(ROOT.parent)): fingerprint(path)
            for path in MANIFEST_CODE_PATHS
        },
    }, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    try:
        for evidence in (False, True):
            mode = "evidence" if evidence else "legacy_backend"
            snapshots = []

            def dispatch(request):
                if request.url.host != "local-benchmark":
                    raise RuntimeError("Non-local backend request blocked")
                content = request.content
                if request.url.path == "/search":
                    payload = json.loads(content)
                    payload["include_evidence"] = evidence
                    content = json.dumps(payload).encode()
                return client.request(request.method, str(request.url),
                                      content=content, headers={"content-type": "application/json"})

            def client_factory(*args, **kwargs):
                kwargs["transport"] = httpx.MockTransport(dispatch)
                return original_client(*args, **kwargs)

            def capture(**kwargs):
                messages, snapshot = original_search(**kwargs)
                snapshots.append(snapshot)
                print(f"{mode}: {len(snapshots)}/{len(seeds)}", flush=True)
                return messages, snapshot

            with patch.object(benchmark, "load_config", return_value=cfg), \
                 patch.object(benchmark, "_clone_database_read_only", side_effect=clone), \
                 patch.object(httpx, "Client", side_effect=client_factory), \
                 patch.object(context_builder, "_kmlog_search_messages", side_effect=capture):
                results = asyncio.run(benchmark._run_benchmark(
                    db_path=db, seeds=seeds, retrieval_enabled=True,
                    router_enabled=True, query_planner_enabled=True, curated_sources=True))
            benchmark._write_outputs(output / mode, results)
            (output / mode / "retrieval_snapshots.json").write_text(
                json.dumps(snapshots, ensure_ascii=False, indent=2), encoding="utf-8")
    finally:
        client.close()


if __name__ == "__main__":
    main()
