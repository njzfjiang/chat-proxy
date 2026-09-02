import csv

from chat_proxy.query_cluster_refinement import (
    load_assignments,
    load_review_annotations,
    refine_clusters,
    write_refined_outputs,
)


def _write_csv(path, fieldnames, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_refinement_preserves_parent_annotations(tmp_path):
    assignments = tmp_path / "assignments.csv"
    review = tmp_path / "review.csv"
    _write_csv(
        assignments,
        [
            "cluster_id",
            "message_id",
            "occurrences",
            "first_timestamp",
            "last_timestamp",
            "conversation_titles",
            "text",
        ],
        [
            {
                "cluster_id": "6",
                "message_id": index,
                "occurrences": 1,
                "first_timestamp": "2026-01-01",
                "last_timestamp": "2026-01-01",
                "conversation_titles": "test",
                "text": text,
            }
            for index, text in enumerate(
                [
                    "整理 memory 记忆库和锚点",
                    "检查 memory vault 里的记忆",
                    "修复 cloud project deployment",
                    "检查 project code 和测试",
                ],
                start=1,
            )
        ],
    )
    _write_csv(
        review,
        ["cluster_id", "suggested_label", "memory_type", "context_need"],
        [
            {
                "cluster_id": "6",
                "suggested_label": "technical",
                "memory_type": "episodic",
                "context_need": "mixed",
            }
        ],
    )

    grouped = load_assignments(assignments)
    annotations = load_review_annotations(review)
    refined = refine_clusters(grouped, selected=["6"], subclusters=2, dimensions=2)
    paths = write_refined_outputs(
        tmp_path / "output", refined=refined, annotations=annotations
    )

    rows = list(
        csv.DictReader(paths["review_csv"].open("r", encoding="utf-8-sig"))
    )
    assert len(rows) == 2
    assert {row["parent_cluster_id"] for row in rows} == {"6"}
    assert {row["parent_suggested_label"] for row in rows} == {"technical"}
    assert {row["parent_memory_type"] for row in rows} == {"episodic"}
    assert all(row["cluster_id"].startswith("6.") for row in rows)
    assert all(row["query_form"] == "" for row in rows)
    assert paths["assignments_csv"].is_file()
    assert paths["report_md"].is_file()
