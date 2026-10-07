from __future__ import annotations

import time
from typing import Any, Mapping

from .config import ProxyConfig
from .context_builder import _kmlog_search_messages
from .parsing import last_user_text


def run_retrieval_shadow(
    *,
    request_id: str,
    request_body: Mapping[str, Any],
    cfg: ProxyConfig,
    active_component: Mapping[str, Any] | None,
    candidate_used_for_answer: bool = False,
) -> dict[str, Any]:
    started = time.monotonic()
    query = _shadow_query(request_body, active_component)
    if not query:
        return {
            "status": "skipped",
            "reason": "no_query_text",
            "request_id": request_id,
            "candidate_used_for_answer": candidate_used_for_answer,
        }

    candidate_body = dict(request_body)
    candidate_body.update(
        {
            "retrieval_enabled": True,
            "retrieval_inject": False,
            "retrieval_router_enabled": True,
            "retrieval_query_planner_enabled": True,
            "retrieval_evidence_enabled": True,
            "retrieval_candidate_results_enabled": False,
            "retrieval_renderer_v2_enabled": True,
        }
    )
    _, candidate = _kmlog_search_messages(
        body=candidate_body,
        cfg=cfg,
        query=query,
    )
    active = _trace_projection(active_component)
    projected_candidate = _trace_projection(candidate)
    status = "error" if candidate.get("error") else "completed"
    return {
        "status": status,
        "request_id": request_id,
        "mode": "active_vs_candidate_v2",
        "candidate_used_for_answer": candidate_used_for_answer,
        "duration_ms": round((time.monotonic() - started) * 1000),
        "active": active,
        "candidate": projected_candidate,
        "diff": _trace_diff(active, projected_candidate),
        "review": {
            "status": "pending",
            "fact_completeness": None,
            "irrelevant_injection": None,
            "source_role": None,
            "notes": None,
        },
    }


def kmlog_component(snapshot: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(snapshot, Mapping):
        return None
    components = snapshot.get("components")
    if not isinstance(components, list):
        return None
    for component in components:
        if isinstance(component, Mapping) and component.get("name") == "kmlog_search":
            return dict(component)
    return None


def _shadow_query(
    body: Mapping[str, Any], active_component: Mapping[str, Any] | None
) -> str:
    if isinstance(active_component, Mapping):
        active_query = str(active_component.get("original_query") or "").strip()
        if active_query:
            return active_query
    return str(body.get("user_text") or last_user_text(dict(body)) or "").strip()


def _trace_projection(component: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(component, Mapping):
        return {"available": False}
    keys = (
        "enabled",
        "inject",
        "router_enabled",
        "query_planner_enabled",
        "renderer",
        "skipped_reason",
        "error",
        "latency_ms",
        "temporal_scope",
        "as_of_timestamp",
        "before",
        "backend_candidate_count",
        "candidate_pool_ids",
        "backend_selected_ids",
        "backend_result_ids",
        "rerank_input_ids",
        "rerank_output_ids",
        "selected_before_budget_ids",
        "selected_after_budget_ids",
        "final_injected_ids",
        "planner_filter_stats",
        "budget_total_chars",
        "budget_excerpt_chars",
        "budget_wrapper_chars",
        "budget_used_chars",
        "chars",
    )
    projection = {key: component.get(key) for key in keys if key in component}
    projection["available"] = True
    projection["items"] = [
        {
            key: item.get(key)
            for key in (
                "id",
                "role",
                "timestamp",
                "title",
                "content_hash",
                "content_preview",
                "visible_matched_terms",
                "visible_required_terms",
                "missing_required_terms",
                "visible_creative_terms",
                "missing_creative_terms",
                "chars",
                "truncated",
            )
            if key in item
        }
        for item in component.get("items", [])
        if isinstance(item, Mapping)
    ]
    return projection


def _trace_diff(
    active: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, Any]:
    active_ids = list(active.get("selected_after_budget_ids") or [])
    candidate_ids = list(candidate.get("selected_after_budget_ids") or [])
    active_set = set(active_ids)
    candidate_set = set(candidate_ids)
    return {
        "same_selected_ids": active_ids == candidate_ids,
        "added_ids": [
            item_id for item_id in candidate_ids if item_id not in active_set
        ],
        "removed_ids": [
            item_id for item_id in active_ids if item_id not in candidate_set
        ],
        "retained_ids": [item_id for item_id in candidate_ids if item_id in active_set],
        "active_chars": active.get("chars"),
        "candidate_chars": candidate.get("chars"),
        "active_latency_ms": active.get("latency_ms"),
        "candidate_latency_ms": candidate.get("latency_ms"),
        "candidate_error": candidate.get("error"),
    }
