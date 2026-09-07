import csv
import json
import sqlite3

from chat_proxy.context_selection_corpus_probe import run_probe, write_outputs


def test_probe_uses_and_or_groups_cutoff_filters_and_deduplicates(tmp_path):
    db_path = tmp_path / "messages.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute("""
CREATE TABLE messages (
    id INTEGER PRIMARY KEY,
    timestamp TEXT,
    role TEXT,
    kind TEXT,
    content TEXT,
    conversation_id TEXT
)
""")
        conn.executemany(
            "INSERT INTO messages VALUES (?, ?, ?, ?, ?, ?)",
            [
                (1, "2026-01-01T00:00:00", "user", "chat", "convex ISTA", "a"),
                (2, "2026-01-01T00:00:00", "user", "chat", "convex ISTA", "b"),
                (3, "2026-01-02T00:00:00", "assistant", "chat", "convex ADMM", "a"),
                (4, "2026-01-03T00:00:00", "user", "chat", "convex ISTA", "seed"),
                (5, "2026-01-04T00:00:00", "user", "chat", "convex FISTA", "a"),
                (
                    6,
                    "2026-01-01T00:00:00",
                    "user",
                    "chat",
                    "The following context is provided by the system. convex ISTA",
                    "synthetic",
                ),
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
                "message_id": 4,
                "timestamp": "2026-01-03T00:00:00",
                "text": "convex ISTA",
                "theme": "course_project",
            }
        )

    probes_path = tmp_path / "probes.json"
    probes_path.write_text(
        json.dumps(
            {
                "probes": [
                    {
                        "seed_message_id": "4",
                        "label": "convex algorithms",
                        "all_of": [["convex"], ["ISTA", "FISTA", "ADMM"]],
                        "expected_corpus": "present",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    rows = run_probe(db_path=db_path, seed_path=seed_path, probes_path=probes_path)

    assert rows[0]["raw_match_count"] == 3
    assert rows[0]["unique_match_count"] == 2
    assert rows[0]["corpus_status"] == "present"
    assert set(rows[0]["matched_ids"].split("|")) == {"2", "3"}

    paths = write_outputs(tmp_path / "output", rows)
    summary = json.loads(paths["summary_json"].read_text(encoding="utf-8"))
    assert summary["expected_present_confirmed_count"] == 1
