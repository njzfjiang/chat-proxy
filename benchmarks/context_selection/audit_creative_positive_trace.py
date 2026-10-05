"""Capture a real-backend creative positive trace without model calls."""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT.parent / "kmlog-search"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(BACKEND))

from chat_proxy.context_builder import _rerank_kmlog_results
from chat_proxy.retrieval_evidence import render_kmlog_results
from chat_proxy.retrieval_planner import plan_retrieval
from chat_proxy.retrieval_semantics import creative_evidence_expansion_terms
from servers import search_sqlite as backend
from servers.message_search import search_with_evidence

DEFAULT_DB = BACKEND / "chat_data/chat_search_prod.db"
DEFAULT_OUTPUT = ROOT / "benchmark_outputs/context_selection_creative_positive_v1"
SEARCH_QUERY = "剧情 人物 证据"
PLANNER_QUERY = "推进剧情时，人物怎么拿到证据"
TARGET_IDS = (7553, 7554)
CODE_PATHS = (
    ROOT / "chat_proxy/context_builder.py",
    ROOT / "chat_proxy/retrieval_evidence.py",
    ROOT / "chat_proxy/retrieval_semantics.py",
    Path(__file__).resolve(),
    BACKEND / "servers/message_search.py",
    BACKEND / "servers/search_sqlite.py",
)


def _fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _git_state() -> dict[str, object]:
    revision = subprocess.run(
        ["git", "-c", f"safe.directory={ROOT.as_posix()}", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            [
                "git",
                "-c",
                f"safe.directory={ROOT.as_posix()}",
                "status",
                "--porcelain",
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return {"commit": revision, "dirty": dirty}


def _working_tree_patch() -> str:
    return subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={ROOT.as_posix()}",
            "diff",
            "--binary",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout


def capture_trace(*, db_path: Path, output_dir: Path) -> dict[str, Path]:
    db_path = db_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        raise RuntimeError("Output exists; choose a new directory to preserve prior runs")
    output_dir.mkdir(parents=True)

    uri = f"file:{quote(db_path.as_posix(), safe='/:')}?mode=ro&immutable=1"

    def connect() -> sqlite3.Connection:
        connection = sqlite3.connect(uri, uri=True)
        connection.row_factory = sqlite3.Row
        return connection

    original_connect = backend.get_connection
    backend.get_connection = connect
    try:
        evidence_terms = creative_evidence_expansion_terms()
        response = search_with_evidence(
            SEARCH_QUERY,
            legacy_search=backend.search_messages,
            connection_factory=connect,
            limit=20,
            mode="auto",
            kinds=["chat"],
            evidence_terms=evidence_terms,
            candidate_limit=80,
            include_candidate_results=True,
        )
    finally:
        backend.get_connection = original_connect

    backend_results = response["results"]
    plan = plan_retrieval(PLANNER_QUERY)
    ranked, filter_stats = _rerank_kmlog_results(
        backend_results,
        plan=plan,
        limit=5,
    )
    rendered_context, render_stats = render_kmlog_results(
        ranked,
        total_chars=1200,
    )

    selected_ids = response["selected_ids"]
    backend_payload = {
        "query": SEARCH_QUERY,
        "evidence_terms": evidence_terms,
        "limit": 20,
        "candidate_limit": 80,
        "candidate_count": response["candidate_count"],
        "candidate_ids": response["candidate_ids"],
        "selected_ids": selected_ids,
        "target_positions": {
            str(message_id): (
                selected_ids.index(message_id) + 1
                if message_id in selected_ids
                else None
            )
            for message_id in TARGET_IDS
        },
        "results": backend_results,
    }
    rerank_payload = {
        "planner_query": PLANNER_QUERY,
        "plan": plan.to_dict(),
        "input_ids": [item.get("id") for item in backend_results],
        "output_ids": [item.get("id") for item in ranked],
        "filter_stats": filter_stats,
        "output_items": ranked,
    }
    final_payload = {
        "selected_after_budget_ids": render_stats["selected_after_budget_ids"],
        "budget_total_chars": render_stats["budget_total_chars"],
        "budget_excerpt_chars": render_stats["budget_excerpt_chars"],
        "budget_wrapper_chars": render_stats["budget_wrapper_chars"],
        "budget_used_chars": render_stats["budget_used_chars"],
        "chars": len(rendered_context),
        "items": render_stats["items"],
        "content": rendered_context,
    }
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_calls": False,
        "database_mode": "read_only_immutable",
        "database": {
            "path": str(db_path),
            "size": db_path.stat().st_size,
            "mtime": datetime.fromtimestamp(
                db_path.stat().st_mtime, timezone.utc
            ).isoformat(),
        },
        "git": _git_state(),
        "code_sha256": {
            str(path.relative_to(ROOT.parent)): _fingerprint(path)
            for path in CODE_PATHS
        },
        "target_ids": list(TARGET_IDS),
        "artifacts": [
            "backend_top20.json",
            "rerank_trace.json",
            "final_visible_context.json",
            "final_visible_context.txt",
            "working_tree.patch",
        ],
    }

    paths = {
        "manifest": output_dir / "manifest.json",
        "backend_top20": output_dir / "backend_top20.json",
        "rerank_trace": output_dir / "rerank_trace.json",
        "final_visible_context": output_dir / "final_visible_context.json",
        "final_visible_text": output_dir / "final_visible_context.txt",
        "working_tree_patch": output_dir / "working_tree.patch",
    }
    _write_json(paths["manifest"], manifest)
    _write_json(paths["backend_top20"], backend_payload)
    _write_json(paths["rerank_trace"], rerank_payload)
    _write_json(paths["final_visible_context"], final_payload)
    paths["final_visible_text"].write_text(rendered_context + "\n", encoding="utf-8")
    paths["working_tree_patch"].write_text(
        _working_tree_patch(),
        encoding="utf-8",
    )
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Capture backend, rerank, and visible creative-positive traces."
    )
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    paths = capture_trace(db_path=args.db, output_dir=args.output)
    print(
        json.dumps(
            {name: str(path) for name, path in paths.items()},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
