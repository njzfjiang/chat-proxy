import csv
import json
import sqlite3

from chat_proxy.context_selection_benchmark import _load_seed_rows, _write_outputs


def test_write_outputs_summarizes_string_boolean_fields(tmp_path):
    results = [
        {
            "theme": "memory",
            "message_id": "1",
            "status": "ok",
            "error": "",
            "future_leak": "False",
            "expected_retrieval": "True",
            "expected_no_context": "False",
            "retrieval_result_count": "2",
            "recent_turn_count": "3",
            "total_token_estimate": "100",
        },
        {
            "theme": "social",
            "message_id": "2",
            "status": "ok",
            "error": "",
            "future_leak": "False",
            "expected_retrieval": "False",
            "expected_no_context": "True",
            "retrieval_result_count": "0",
            "recent_turn_count": "1",
            "total_token_estimate": "20",
        },
    ]

    paths = _write_outputs(tmp_path, results)
    summary = json.loads(paths["summary_json"].read_text(encoding="utf-8"))

    assert summary["future_leak_count"] == 0
    assert summary["rematched_seed_count"] == 0
    assert summary["expected_retrieval_seed_count"] == 1
    assert summary["expected_retrieval_nonempty_count"] == 1
    assert summary["unexpected_retrieval_count"] == 0
    assert summary["no_context_seed_count"] == 1
    assert summary["no_context_with_retrieval_count"] == 0
    assert summary["expected_retrieval_any_source_nonempty_count"] == 1
    assert summary["no_context_with_any_source_count"] == 0


def test_write_outputs_counts_curated_source_errors(tmp_path):
    paths = _write_outputs(
        tmp_path,
        [
            {
                "theme": "course_project",
                "status": "ok",
                "error": "",
                "source_errors_json": '{"reviewed_memory": "500 error"}',
                "recent_turn_count": 0,
                "total_token_estimate": 0,
            }
        ],
    )

    summary = json.loads(paths["summary_json"].read_text(encoding="utf-8"))
    assert summary["error_count"] == 1


def test_load_seed_rows_rematches_exact_text_when_integer_ids_differ(tmp_path):
    db_path = tmp_path / "messages.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute("""
CREATE TABLE messages (
    id INTEGER PRIMARY KEY,
    timestamp TEXT,
    role TEXT,
    content TEXT,
    conversation_id TEXT
)
""")
        conn.executemany(
            "INSERT INTO messages VALUES (?, ?, ?, ?, ?)",
            [
                (7, "2026-01-01T00:00:00", "user", "different", "wrong"),
                (19, "2026-02-02T00:00:00", "user", "seed\ntext", "right"),
            ],
        )
    seed_path = tmp_path / "seed.csv"
    with seed_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("message_id", "timestamp", "text", "theme")
        )
        writer.writeheader()
        writer.writerow(
            {
                "message_id": "7",
                "timestamp": "2026-02-02T00:00:00",
                "text": "seed text",
                "theme": "test",
            }
        )

    rows = _load_seed_rows(seed_path, db_path)

    assert rows[0]["seed_message_id"] == "7"
    assert rows[0]["message_id"] == "19"
    assert rows[0]["message_id_rematched"] == "true"
    assert rows[0]["conversation_id"] == "right"
