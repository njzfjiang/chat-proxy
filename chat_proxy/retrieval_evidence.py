from __future__ import annotations

import re
from typing import Any, Mapping


def render_kmlog_results(
    results: list[Any], *, total_chars: int
) -> tuple[str, dict[str, Any]]:
    remaining_chars = max(0, total_chars)
    evidence_item_budget = remaining_chars // max(1, len(results))
    budget_dropped: list[dict[str, Any]] = []
    items: list[dict[str, Any]] = []
    blocks: list[str] = []
    for result_index, raw_item in enumerate(results):
        if not isinstance(raw_item, Mapping):
            budget_dropped.append({"id": None, "reason": "invalid_result"})
            continue
        if remaining_chars <= 0:
            budget_dropped.append(
                {"id": raw_item.get("id"), "reason": "budget_exhausted"}
            )
            continue
        preview = str(
            raw_item.get("matched_excerpt")
            or raw_item.get("content_preview")
            or ""
        ).strip()
        if not preview:
            budget_dropped.append(
                {"id": raw_item.get("id"), "reason": "empty_excerpt"}
            )
            continue
        item_budget = remaining_chars
        if raw_item.get("evidence_version") == 1:
            remaining_slots = max(1, len(results) - result_index)
            item_budget = remaining_chars // remaining_slots
            clipped = clip_kmlog_evidence(raw_item, item_budget)
        else:
            clipped = preview[:item_budget].rstrip()
        if not clipped:
            budget_dropped.append(
                {"id": raw_item.get("id"), "reason": "below_minimum_fragment"}
            )
            continue
        remaining_chars -= len(clipped)
        title = str(raw_item.get("conversation_title") or "").strip()
        role = str(raw_item.get("role") or "").strip()
        timestamp = str(raw_item.get("timestamp") or "").strip()
        body_matched_terms = list(raw_item.get("body_matched_terms", []))
        required_matches = list(raw_item.get("planner_required_matches", []))
        visible_matched_terms = [
            term for term in body_matched_terms if contains_kmlog_term(clipped, term)
        ]
        visible_required_terms = [
            term for term in required_matches if contains_kmlog_term(clipped, term)
        ]
        missing_required_terms = [
            term for term in required_matches if term not in visible_required_terms
        ]
        blocks.append(
            "\n".join(
                part
                for part in [
                    f"- id={raw_item.get('id')} {timestamp} {role}".strip(),
                    f"  title: {title}" if title else "",
                    f"  excerpt: {clipped}",
                ]
                if part
            )
        )
        items.append(
            {
                "id": raw_item.get("id"),
                "title": title,
                "role": role,
                "timestamp": timestamp,
                "match_type": raw_item.get("match_type"),
                "relevance": raw_item.get("relevance"),
                "token_hits": raw_item.get("token_hits"),
                "evidence_version": raw_item.get("evidence_version"),
                "body_matched_terms": body_matched_terms,
                "title_matched_terms": raw_item.get("title_matched_terms", []),
                "evidence_origin": raw_item.get("evidence_origin"),
                "title_only": raw_item.get("title_only"),
                "content_hash": raw_item.get("content_hash"),
                "source_message_ids": raw_item.get("source_message_ids", []),
                "duplicate_count": raw_item.get("duplicate_count"),
                "duplicate_provenance": raw_item.get("duplicate_provenance", []),
                "match_spans": raw_item.get("match_spans", []),
                "planner_required_matches": required_matches,
                "planner_optional_matches": raw_item.get(
                    "planner_optional_matches", []
                ),
                "chars": len(clipped),
                "source_chars": len(preview),
                "budget_limit": item_budget,
                "truncated": len(clipped) < len(preview),
                "visible_matched_terms": visible_matched_terms,
                "visible_required_terms": visible_required_terms,
                "missing_required_terms": missing_required_terms,
                "required_coverage": (
                    len(visible_required_terms) / len(required_matches)
                    if required_matches
                    else None
                ),
                "content_preview": clipped,
            }
        )

    content = ""
    if blocks:
        content = "Retrieved chat log snippets:\n\n" + "\n\n".join(blocks)
    return content, {
        "items": items,
        "result_count": len(items),
        "selected_after_budget_ids": [item["id"] for item in items],
        "budget_total_chars": max(0, total_chars),
        "budget_per_evidence_item_chars": evidence_item_budget,
        "budget_strategy": "remaining_equal_share",
        "budget_dropped": budget_dropped,
        "budget_used_chars": sum(item["chars"] for item in items),
        "required_terms_visible_count": sum(
            len(item["visible_required_terms"]) for item in items
        ),
        "required_terms_missing_count": sum(
            len(item["missing_required_terms"]) for item in items
        ),
        "chars": len(content),
    }


def clip_kmlog_excerpt(preview: str, budget: int) -> str:
    if budget < min(40, len(preview)):
        return ""
    return preview[: max(0, budget)].rstrip()


def clip_kmlog_evidence(item: Mapping[str, Any], budget: int) -> str:
    text = str(
        item.get("matched_excerpt") or item.get("content_preview") or ""
    ).strip()
    if budget <= 0 or not text:
        return ""
    if len(text) <= budget:
        return text

    required_terms = _dedupe_casefolded(
        list(item.get("planner_required_matches") or [])
    )
    priority_terms = required_terms or _dedupe_casefolded(
        list(item.get("body_matched_terms") or [])
    )
    anchors = []
    for term in priority_terms:
        span = find_kmlog_term(text, term)
        if span is not None:
            anchors.append((term, *span))
    if not anchors:
        return clip_kmlog_excerpt(text, budget)

    separator = "\n...\n"
    selected = []
    minimum_chars = 0
    for anchor in anchors:
        added = len(anchor[0]) + (len(separator) if selected else 0)
        if minimum_chars + added > budget:
            continue
        selected.append(anchor)
        minimum_chars += added
    if not selected:
        return clip_kmlog_excerpt(text, budget)

    groups: list[dict[str, int]] = []
    for _, start, end in selected:
        sentence_start, sentence_end = _sentence_bounds(text, start, end)
        if (
            groups
            and groups[-1]["sentence_start"] == sentence_start
            and groups[-1]["sentence_end"] == sentence_end
        ):
            groups[-1]["anchor_start"] = min(groups[-1]["anchor_start"], start)
            groups[-1]["anchor_end"] = max(groups[-1]["anchor_end"], end)
        else:
            groups.append(
                {
                    "sentence_start": sentence_start,
                    "sentence_end": sentence_end,
                    "anchor_start": start,
                    "anchor_end": end,
                }
            )

    content_budget = budget - len(separator) * (len(groups) - 1)
    widths = [group["anchor_end"] - group["anchor_start"] for group in groups]
    if sum(widths) > content_budget:
        groups = [
            {
                "sentence_start": start,
                "sentence_end": end,
                "anchor_start": start,
                "anchor_end": end,
            }
            for _, start, end in selected
        ]
        content_budget = budget - len(separator) * (len(groups) - 1)
        widths = [group["anchor_end"] - group["anchor_start"] for group in groups]
    full_sentence_widths = [
        group["sentence_end"] - group["sentence_start"] for group in groups
    ]
    if sum(full_sentence_widths) <= content_budget:
        widths = full_sentence_widths
    else:
        remaining = content_budget - sum(widths)
        for index in range(len(widths)):
            share = remaining // (len(widths) - index)
            widths[index] += share
            remaining -= share

    windows = []
    for group, width in zip(groups, widths):
        windows.append(
            _sentence_or_centered_window(
                text, group["anchor_start"], group["anchor_end"], width
            )
        )

    merged = []
    for start, end in sorted(windows):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return separator.join(text[start:end].strip() for start, end in merged).rstrip()


def contains_kmlog_term(text: str, term: Any) -> bool:
    return find_kmlog_term(text, term) is not None


def find_kmlog_term(text: str, term: Any) -> tuple[int, int] | None:
    value = str(term or "").strip()
    if not value:
        return None
    pattern = re.escape(value)
    if value[0].isascii() and value[0].isalnum():
        pattern = r"(?<![a-zA-Z0-9_])" + pattern
    if value[-1].isascii() and value[-1].isalnum():
        pattern += r"(?![a-zA-Z0-9_])"
    match = re.search(pattern, text, re.IGNORECASE)
    return (match.start(), match.end()) if match else None


def _sentence_or_centered_window(
    text: str, start: int, end: int, width: int
) -> tuple[int, int]:
    sentence_start, sentence_end = _sentence_bounds(text, start, end)
    if sentence_end - sentence_start <= width:
        return sentence_start, sentence_end

    center = (start + end) // 2
    window_start = max(0, center - width // 2)
    window_end = min(len(text), window_start + width)
    return max(0, window_end - width), window_end


def _sentence_bounds(text: str, start: int, end: int) -> tuple[int, int]:
    left_matches = list(re.finditer(r"[。！？!?;；\n]|(?<=[a-z0-9])[.](?=\s)", text[:start], re.I))
    sentence_start = left_matches[-1].end() if left_matches else 0
    right = re.search(r"[。！？!?;；\n]|[.](?=\s|$)", text[end:], re.I)
    sentence_end = end + right.end() if right else len(text)
    while sentence_start < sentence_end and text[sentence_start].isspace():
        sentence_start += 1
    return sentence_start, sentence_end


def _dedupe_casefolded(values: list[Any]) -> list[str]:
    result = []
    seen = set()
    for value in values:
        term = str(value or "").strip()
        key = term.casefold()
        if term and key not in seen:
            seen.add(key)
            result.append(term)
    return result
