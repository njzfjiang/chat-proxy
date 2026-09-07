from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote

from .context_selection_benchmark import _load_seed_rows

_SYNTHETIC_USER_PREFIX = "The following context is provided by the system."


def _normalize_content(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _load_probes(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    probes = payload.get("probes") if isinstance(payload, dict) else payload
    if not isinstance(probes, list) or not probes:
        raise ValueError("Probe JSON must contain a non-empty 'probes' list.")

    normalized = []
    for index, probe in enumerate(probes, start=1):
        if not isinstance(probe, dict):
            raise ValueError(f"Probe {index} must be an object.")
        seed_id = str(probe.get("seed_message_id") or "").strip()
        label = str(probe.get("label") or "").strip()
        groups = probe.get("all_of")
        if not seed_id or not label or not isinstance(groups, list) or not groups:
            raise ValueError(
                f"Probe {index} requires seed_message_id, label, and all_of."
            )
        clean_groups = []
        for group in groups:
            if not isinstance(group, list):
                raise ValueError(f"Probe {index} all_of entries must be lists.")
            terms = [str(term).strip() for term in group if str(term).strip()]
            if not terms:
                raise ValueError(f"Probe {index} contains an empty term group.")
            clean_groups.append(terms)
        roles = probe.get("roles", ["user", "assistant"])
        if not isinstance(roles, list) or not roles:
            raise ValueError(f"Probe {index} roles must be a non-empty list.")
        normalized.append(
            {
                **probe,
                "seed_message_id": seed_id,
                "label": label,
                "all_of": clean_groups,
                "roles": [str(role) for role in roles],
                "expected_corpus": str(
                    probe.get("expected_corpus") or "unknown"
                ).lower(),
            }
        )
    return normalized


def _probe_rows(
    conn: sqlite3.Connection,
    *,
    seed: dict[str, str],
    probe: dict[str, Any],
    preview_limit: int,
) -> dict[str, Any]:
    clauses = [
        "kind = 'chat'",
        "julianday(timestamp) < julianday(?)",
        "id != ?",
        "content NOT LIKE ?",
    ]
    params: list[Any] = [
        seed["timestamp"],
        int(seed["message_id"]),
        f"{_SYNTHETIC_USER_PREFIX}%",
    ]

    roles = probe["roles"]
    clauses.append(f"role IN ({','.join('?' for _ in roles)})")
    params.extend(roles)
    for group in probe["all_of"]:
        clauses.append(
            "("
            + " OR ".join("instr(lower(content), lower(?)) > 0" for _ in group)
            + ")"
        )
        params.extend(group)

    rows = conn.execute(
        f"""
SELECT id, timestamp, role, conversation_id, content
FROM messages
WHERE {' AND '.join(clauses)}
ORDER BY julianday(timestamp) DESC, id DESC
""",
        params,
    ).fetchall()
    unique_rows = []
    seen_content = set()
    for row in rows:
        item = dict(row)
        key = _normalize_content(item["content"])
        if key in seen_content:
            continue
        seen_content.add(key)
        unique_rows.append(item)

    previews = [
        {
            "id": row["id"],
            "timestamp": row["timestamp"],
            "role": row["role"],
            "conversation_id": row["conversation_id"],
            "preview": _normalize_content(row["content"])[:500],
        }
        for row in unique_rows[:preview_limit]
    ]
    return {
        "seed_message_id": seed["seed_message_id"],
        "actual_message_id": seed["message_id"],
        "message_id_rematched": seed["message_id_rematched"],
        "seed_timestamp": seed["timestamp"],
        "theme": seed.get("theme", ""),
        "seed_note": seed.get("note", ""),
        "probe_label": probe["label"],
        "all_of_json": json.dumps(probe["all_of"], ensure_ascii=False),
        "roles": "|".join(roles),
        "expected_corpus": probe["expected_corpus"],
        "raw_match_count": len(rows),
        "unique_match_count": len(unique_rows),
        "corpus_status": "present" if unique_rows else "missing",
        "matched_ids": "|".join(str(row["id"]) for row in previews),
        "previews_json": json.dumps(previews, ensure_ascii=False),
    }


def run_probe(
    *,
    db_path: Path,
    seed_path: Path,
    probes_path: Path,
    preview_limit: int = 5,
) -> list[dict[str, Any]]:
    seeds = _load_seed_rows(seed_path, db_path)
    seeds_by_original_id = {seed["seed_message_id"]: seed for seed in seeds}
    probes = _load_probes(probes_path)
    missing_seed_ids = sorted(
        {
            probe["seed_message_id"]
            for probe in probes
            if probe["seed_message_id"] not in seeds_by_original_id
        }
    )
    if missing_seed_ids:
        raise ValueError(f"Probe seed IDs are absent from seed CSV: {missing_seed_ids}")

    resolved = db_path.expanduser().resolve()
    uri = f"file:{quote(resolved.as_posix(), safe='/:')}?mode=ro&immutable=1"
    with sqlite3.connect(uri, uri=True) as conn:
        conn.row_factory = sqlite3.Row
        return [
            _probe_rows(
                conn,
                seed=seeds_by_original_id[probe["seed_message_id"]],
                probe=probe,
                preview_limit=preview_limit,
            )
            for probe in probes
        ]


def write_outputs(output_dir: Path, rows: list[dict[str, Any]]) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "corpus_probe_results.csv"
    summary_path = output_dir / "corpus_probe_summary.json"
    report_path = output_dir / "corpus_probe_report.md"

    with results_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    expected_present = [row for row in rows if row["expected_corpus"] == "present"]
    summary = {
        "probe_count": len(rows),
        "present_count": sum(row["corpus_status"] == "present" for row in rows),
        "missing_count": sum(row["corpus_status"] == "missing" for row in rows),
        "expected_present_count": len(expected_present),
        "expected_present_confirmed_count": sum(
            row["corpus_status"] == "present" for row in expected_present
        ),
        "synthetic_user_prefix_filtered": _SYNTHETIC_USER_PREFIX,
        "deduplication": "whitespace-normalized exact content",
        "limitation": (
            "A missing lexical probe is not proof that no semantically equivalent "
            "evidence exists."
        ),
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# Context-selection corpus probe",
        "",
        f"- Probes: {summary['probe_count']}",
        f"- Present: {summary['present_count']}",
        f"- Missing: {summary['missing_count']}",
        (
            "- Expected-present confirmed: "
            f"{summary['expected_present_confirmed_count']} / "
            f"{summary['expected_present_count']}"
        ),
        "- Cutoff: messages strictly before each seed timestamp",
        "- Exact duplicate content across conversations is counted once",
        "- Synthetic system context stored as user content is excluded",
        "",
        "| Seed | Probe | Expected | Status | Raw | Unique |",
        "|---:|---|---|---|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['seed_message_id']} | {row['probe_label']} | "
            f"{row['expected_corpus']} | {row['corpus_status']} | "
            f"{row['raw_match_count']} | {row['unique_match_count']} |"
        )
    lines.extend(["", f"Limitation: {summary['limitation']}"])
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "results_csv": results_path,
        "summary_json": summary_path,
        "report_md": report_path,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Probe whether lexical evidence exists before benchmark seeds."
    )
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=Path)
    parser.add_argument("--probes", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--preview-limit", type=int, default=5)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    rows = run_probe(
        db_path=args.db,
        seed_path=args.seed,
        probes_path=args.probes,
        preview_limit=max(1, args.preview_limit),
    )
    paths = write_outputs(args.output_dir, rows)
    print(
        json.dumps(
            {key: str(value.resolve()) for key, value in paths.items()},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
