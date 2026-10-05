from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any


CREATIVE_EVIDENCE_EXPANSION_TERMS = (
    "情节",
    "下一幕",
    "线索",
    "密信",
    "信件",
    "书信",
    "拼图",
)

_CREATIVE_CONCEPT_TERMS = {
    "narrative": ("剧情", "情节", "下一幕", "故事线", "伏笔"),
    "evidence": (
        "证据",
        "线索",
        "密信",
        "信件",
        "书信",
        "物证",
        "拼图",
    ),
    "transfer": (
        "拿到",
        "得到",
        "取得",
        "获得",
        "发现",
        "找到",
        "交给",
        "交到",
        "递给",
        "展开",
        "掰开",
    ),
}


def creative_evidence_expansion_terms() -> list[str]:
    return list(CREATIVE_EVIDENCE_EXPANSION_TERMS)


def match_creative_evidence(
    haystack: str,
    *,
    required_terms: Iterable[str],
    required_matches: Iterable[str],
    matched_terms: Iterable[str] = (),
    keyword_match: Callable[[str, str], bool],
) -> dict[str, Any]:
    required = list(required_terms)
    exact_matches = list(required_matches)
    body_matches = {term.casefold() for term in matched_terms}
    concept_matches = {
        concept: [
            term
            for term in terms
            if term.casefold() in body_matches or keyword_match(haystack, term)
        ]
        for concept, terms in _CREATIVE_CONCEPT_TERMS.items()
    }
    exact_all = bool(required) and len(exact_matches) == len(required)
    semantic_coherence = bool(concept_matches["evidence"]) and bool(
        concept_matches["transfer"]
    )
    return {
        "accepted": exact_all or semantic_coherence,
        "acceptance": (
            "exact_required_terms"
            if exact_all
            else "semantic_evidence_coherence"
            if semantic_coherence
            else None
        ),
        "concept_matches": concept_matches,
        "missing_concepts": [
            concept
            for concept in ("evidence", "transfer")
            if (concept == "evidence" and not concept_matches["evidence"])
            or (concept == "transfer" and not concept_matches["transfer"])
        ],
    }
