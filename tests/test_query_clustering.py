import csv
import json
import sqlite3

from chat_proxy.query_clustering import (
    cluster_queries,
    content_focus_text,
    load_user_queries,
    normalize_query_text,
    write_cluster_outputs,
)


def _create_messages_db(path):
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
CREATE TABLE messages (
  id INTEGER PRIMARY KEY,
  timestamp TEXT,
  role TEXT,
  content TEXT,
  conversation_title TEXT,
  conversation_id TEXT,
  kind TEXT
)
"""
        )
        conn.executemany(
            "INSERT INTO messages VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (1, "2026-01-01T00:00:00", "user", "部署 proxy", "Infra", "a", "chat"),
                (2, "2026-01-01T00:01:00", "user", "部署   proxy", "Infra", "a", "chat"),
                (3, "2026-01-01T00:02:00", "assistant", "done", "Infra", "a", "chat"),
                (4, "2026-01-02T00:00:00", "user", "修复数据库服务", "Infra", "b", None),
                (5, "2026-01-03T00:00:00", "user", "今天想喝红茶", "Daily", "c", "chat"),
                (6, "2026-01-04T00:00:00", "user", "晚上做什么饭", "Daily", "c", "chat"),
                (7, "2026-01-05T00:00:00", "user", "ignore noise", "Daily", "c", "noise"),
            ],
        )


def test_load_user_queries_filters_and_deduplicates(tmp_path):
    db_path = tmp_path / "messages.db"
    _create_messages_db(db_path)

    documents = load_user_queries(db_path, min_chars=4)

    assert len(documents) == 4
    proxy = next(item for item in documents if item.text == "部署 proxy")
    assert proxy.occurrence_count == 2
    assert proxy.conversation_ids == {"a"}
    assert normalize_query_text("  hello\n  world ") == "hello world"
    assert content_focus_text("Feb 2, 2026 at 9:18 AM 小猫哈哈哈 AWS部署") == "aws部署"


def test_cluster_queries_and_write_reviewable_outputs(tmp_path):
    db_path = tmp_path / "messages.db"
    _create_messages_db(db_path)
    documents = load_user_queries(db_path, min_chars=4)

    labels, summaries = cluster_queries(
        documents,
        clusters=2,
        seed=7,
        min_df=1,
        top_terms=4,
        examples_per_cluster=2,
        min_topic_chars=4,
    )
    paths = write_cluster_outputs(
        tmp_path / "out",
        db_path=db_path,
        documents=documents,
        labels=labels,
        summaries=summaries,
        settings={"clusters": 2},
    )

    assert len(labels) == len(documents)
    assert [item.cluster_id for item in summaries] == [1, 2]
    assert sum(item.weighted_turns for item in summaries) == 5
    assert all(item.top_terms for item in summaries)

    payload = json.loads(paths["summary_json"].read_text(encoding="utf-8"))
    assert payload["weighted_turns"] == 5
    assert len(payload["clusters"]) == 2
    assert "Cluster 1" in paths["report_md"].read_text(encoding="utf-8")

    with paths["assignments_csv"].open(encoding="utf-8-sig", newline="") as handle:
        assignments = list(csv.DictReader(handle))
    assert len(assignments) == 4
    assert {row["cluster_id"] for row in assignments} == {"1", "2"}

    with paths["review_csv"].open(encoding="utf-8-sig", newline="") as handle:
        review_rows = list(csv.DictReader(handle))
    assert review_rows[0]["memory_type"] == ""
    assert review_rows[0]["context_need"] == ""
