"""Audit known-positive paraphrases against the current required-term filter."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from chat_proxy.context_builder import _rerank_kmlog_results

CASES = (
    ROOT
    / "benchmarks/context_selection/annotations/hard_filter_known_positives_v1.json"
)


def audit_cases(path: Path = CASES) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    results = []
    for index, case in enumerate(payload["cases"], start=1):
        candidate_id = f"known-positive-{index}"
        ranked, stats = _rerank_kmlog_results(
            [
                {
                    "id": candidate_id,
                    "content_preview": case["candidate"],
                    "conversation_title": "",
                }
            ],
            plan=SimpleNamespace(
                required_terms=case["required_terms"],
                optional_terms=[],
                matched_domains=("creative_writing",),
            ),
            limit=1,
        )
        observed_accept = bool(ranked)
        filter_reason = next(
            (
                reason
                for reason in stats["filter_reasons"]
                if reason.get("id") == candidate_id
            ),
            None,
        )
        results.append(
            {
                "case_id": case["case_id"],
                "expected_accept": bool(case["expected_accept"]),
                "observed_accept": observed_accept,
                "matches_expectation": observed_accept
                == bool(case["expected_accept"]),
                "filter_reason": filter_reason,
            }
        )
    return {
        "case_count": len(results),
        "false_negative_count": sum(
            item["expected_accept"] and not item["observed_accept"]
            for item in results
        ),
        "results": results,
    }


def main() -> None:
    print(json.dumps(audit_cases(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
