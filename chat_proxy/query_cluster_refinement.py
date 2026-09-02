from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from .query_clustering import ClusterSummary, QueryDocument, cluster_queries


@dataclass(frozen=True)
class RefinedItem:
    parent_cluster_id: str
    subcluster_id: int
    cluster_id: str
    document: QueryDocument
    summary: ClusterSummary


def _cluster_sort_key(value: str) -> tuple[int, str]:
    try:
        return int(value), value
    except ValueError:
        return 2**31 - 1, value


def load_assignments(path: Path) -> dict[str, list[QueryDocument]]:
    grouped: dict[str, list[QueryDocument]] = defaultdict(list)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            parent_id = str(row.get("cluster_id") or "").strip()
            text = str(row.get("text") or "").strip()
            if not parent_id or not text:
                continue
            titles = {
                title.strip()
                for title in str(row.get("conversation_titles") or "").split("|")
                if title.strip()
            }
            grouped[parent_id].append(
                QueryDocument(
                    text=text,
                    representative_id=int(row["message_id"]),
                    occurrence_count=max(1, int(row.get("occurrences") or 1)),
                    first_timestamp=str(row.get("first_timestamp") or ""),
                    last_timestamp=str(row.get("last_timestamp") or ""),
                    conversation_titles=titles,
                )
            )
    return dict(grouped)


def load_review_annotations(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {
            str(row.get("cluster_id") or "").strip(): dict(row)
            for row in csv.DictReader(handle)
            if str(row.get("cluster_id") or "").strip()
        }


def refine_clusters(
    grouped: dict[str, list[QueryDocument]],
    *,
    selected: Sequence[str],
    subclusters: int,
    seed: int = 42,
    dimensions: int = 96,
    min_df: int = 2,
    top_terms: int = 12,
    examples: int = 4,
) -> list[RefinedItem]:
    refined: list[RefinedItem] = []
    for parent_id in selected:
        documents = grouped.get(parent_id, [])
        if len(documents) < 2:
            raise ValueError(f"Parent cluster {parent_id!r} has fewer than two queries.")
        labels, summaries = cluster_queries(
            documents,
            clusters=subclusters,
            seed=seed,
            dimensions=dimensions,
            min_df=min_df,
            top_terms=top_terms,
            examples_per_cluster=examples,
            profile="content",
            min_topic_chars=1,
        )
        summary_by_id = {summary.cluster_id: summary for summary in summaries}
        for document, subcluster_id in zip(documents, labels):
            refined.append(
                RefinedItem(
                    parent_cluster_id=parent_id,
                    subcluster_id=subcluster_id,
                    cluster_id=f"{parent_id}.{subcluster_id}",
                    document=document,
                    summary=summary_by_id[subcluster_id],
                )
            )
    return refined


def write_refined_outputs(
    output_dir: Path,
    *,
    refined: Sequence[RefinedItem],
    annotations: dict[str, dict[str, str]],
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    assignments_path = output_dir / "refined_assignments.csv"
    review_path = output_dir / "refined_cluster_review.csv"
    report_path = output_dir / "refined_report.md"

    ordered = sorted(
        refined,
        key=lambda item: (
            _cluster_sort_key(item.parent_cluster_id),
            item.subcluster_id,
            item.document.representative_id,
        ),
    )
    with assignments_path.open("w", encoding="utf-8-sig", newline="") as handle:
        fields = [
            "parent_cluster_id",
            "cluster_id",
            "message_id",
            "occurrences",
            "first_timestamp",
            "last_timestamp",
            "conversation_titles",
            "text",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in ordered:
            document = item.document
            writer.writerow(
                {
                    "parent_cluster_id": item.parent_cluster_id,
                    "cluster_id": item.cluster_id,
                    "message_id": document.representative_id,
                    "occurrences": document.occurrence_count,
                    "first_timestamp": document.first_timestamp,
                    "last_timestamp": document.last_timestamp,
                    "conversation_titles": " | ".join(
                        sorted(document.conversation_titles)
                    ),
                    "text": document.text,
                }
            )

    groups: dict[str, RefinedItem] = {}
    for item in ordered:
        groups.setdefault(item.cluster_id, item)

    review_fields = [
        "parent_cluster_id",
        "cluster_id",
        "weighted_turns",
        "unique_queries",
        "parent_suggested_label",
        "suggested_label",
        "query_form",
        "parent_memory_type",
        "memory_type",
        "parent_context_need",
        "context_need",
        "top_terms",
        "notes",
    ]
    with review_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=review_fields)
        writer.writeheader()
        for cluster_id, item in groups.items():
            parent_id = item.parent_cluster_id
            parent = annotations.get(parent_id, {})
            summary = item.summary
            writer.writerow(
                {
                    "parent_cluster_id": parent_id,
                    "cluster_id": cluster_id,
                    "weighted_turns": summary.weighted_turns,
                    "unique_queries": summary.unique_queries,
                    "parent_suggested_label": parent.get("suggested_label", ""),
                    "suggested_label": "",
                    "query_form": "",
                    "parent_memory_type": parent.get("memory_type", ""),
                    "memory_type": "",
                    "parent_context_need": parent.get("context_need", ""),
                    "context_need": "",
                    "top_terms": " | ".join(summary.top_terms),
                    "notes": "",
                }
            )

    lines = [
        "# Refined user-turn query clusters",
        "",
        "Parent annotations are hints only. Label each child by topic and use",
        "`query_form` for narration, inquiry, continuation, status, or social turns.",
        "",
    ]
    current_parent = None
    for cluster_id, item in groups.items():
        parent_id = item.parent_cluster_id
        parent = annotations.get(parent_id, {})
        if parent_id != current_parent:
            current_parent = parent_id
            lines.extend(
                [
                    f"## Parent cluster {parent_id}: {parent.get('suggested_label', '')}",
                    "",
                    (
                        "Parent memory/context labels: "
                        f"{parent.get('memory_type', '')} / {parent.get('context_need', '')}"
                    ),
                    "",
                ]
            )
        summary = item.summary
        lines.extend(
            [
                f"### Cluster {cluster_id}",
                "",
                (
                    f"{summary.weighted_turns} turns; "
                    f"{summary.unique_queries} unique queries."
                ),
                "",
                f"Top terms: {', '.join(summary.top_terms)}",
                "",
            ]
        )
        for example in summary.examples:
            text = str(example["text"])
            if len(text) > 280:
                text = text[:277].rstrip() + "..."
            lines.append(f"- `{example['message_id']}`: {text}")
        lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return {
        "assignments_csv": assignments_path,
        "review_csv": review_path,
        "report_md": report_path,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Re-cluster selected parent groups from query clustering outputs."
    )
    parser.add_argument("--assignments", required=True, type=Path)
    parser.add_argument("--review", required=True, type=Path)
    parser.add_argument(
        "--parents",
        required=True,
        help="Comma-separated parent cluster IDs, for example 1,2,3,5,6,8.",
    )
    parser.add_argument("--subclusters", type=int, default=8)
    parser.add_argument("--dimensions", type=int, default=96)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--top-terms", type=int, default=12)
    parser.add_argument("--examples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark_outputs/query_clusters_refined"),
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    selected = [value.strip() for value in args.parents.split(",") if value.strip()]
    if not selected:
        raise ValueError("At least one parent cluster ID is required.")
    grouped = load_assignments(args.assignments)
    annotations = load_review_annotations(args.review)
    refined = refine_clusters(
        grouped,
        selected=selected,
        subclusters=args.subclusters,
        seed=args.seed,
        dimensions=args.dimensions,
        min_df=args.min_df,
        top_terms=args.top_terms,
        examples=args.examples,
    )
    paths = write_refined_outputs(
        args.output_dir,
        refined=refined,
        annotations=annotations,
    )
    for name, path in paths.items():
        print(f"{name}: {path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
