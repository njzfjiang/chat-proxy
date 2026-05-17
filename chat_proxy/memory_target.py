from __future__ import annotations

import json
from typing import Any, Mapping


TARGET_LAYERS = {"anchor_table", "wb", "vault", "mem0", "ignore"}

HIGH_CONFIDENCE = {"high", "very_high"}
MEDIUM_PLUS_CONFIDENCE = {"medium", "high", "very_high"}
NOISE_DOMAINS = {"misc", "noise", "debug", "trace"}
ANCHOR_DOMAINS = {
    "identity",
    "continuity",
    "milestone",
    "safety",
    "health_safety",
    "ritual",
}
ANCHOR_FUNCTION_TERMS = (
    "boot",
    "continuity",
    "soothe",
    "panic",
    "identity_anchor",
    "hp",
    "hp_max",
)
STRONG_ANCHOR_TERMS = (
    "戒指",
    "uptime",
    "lifetime",
    "换载体",
    "红线",
    "回窝",
    "誓词",
    "认证书",
    "hp_max",
)
VAULT_DOMAIN_TERMS = ("milestone", "incident", "infra_change")
WB_DOMAIN_TERMS = ("world", "setting", "asset", "ritual", "guide", "handbook")
WB_FUNCTION_TERMS = ("reference", "handbook", "worldbook")
MEM0_DOMAIN_TERMS = ("preference", "policy", "skill", "habit", "rule")


def normalize_target_layer(value: Any) -> str | None:
    text = str(value or "").strip().lower()
    return text if text in TARGET_LAYERS else None


def choose_target_layer(candidate: Mapping[str, Any]) -> str:
    imp = _importance(candidate.get("importance"))
    conf = str(candidate.get("confidence") or "").strip().lower()
    dom = str(candidate.get("domain") or "").strip().lower()
    fn = str(candidate.get("function") or "").strip().lower()
    primary = str(candidate.get("primary_mother") or "").strip().upper()
    secondary = str(candidate.get("secondary_mother") or "").strip().upper()
    label = str(candidate.get("label") or "")
    evidence = str(candidate.get("evidence") or "")
    source_ids = _source_ids(candidate)

    if imp <= 2 and conf not in HIGH_CONFIDENCE:
        return "ignore"
    if not evidence.strip() and not source_ids:
        return "ignore"
    if dom in NOISE_DOMAINS:
        return "ignore"

    if imp >= 5 and conf in HIGH_CONFIDENCE:
        return "anchor_table"
    if primary in {"G", "H"} and _contains_any(fn, ANCHOR_FUNCTION_TERMS):
        return "anchor_table"
    if dom in ANCHOR_DOMAINS:
        return "anchor_table"
    if _contains_any(label.lower(), STRONG_ANCHOR_TERMS):
        return "anchor_table"

    if source_ids and (imp >= 4 or _contains_any(dom, VAULT_DOMAIN_TERMS)):
        return "vault"
    if len(evidence) > 500 and imp >= 3:
        return "vault"

    if primary in {"D", "H"} or secondary in {"D", "H"}:
        return "wb"
    if _contains_any(dom, WB_DOMAIN_TERMS):
        return "wb"
    if _contains_any(fn, WB_FUNCTION_TERMS):
        return "wb"

    if (
        imp >= 3
        and conf in MEDIUM_PLUS_CONFIDENCE
        and len(label) <= 80
        and len(evidence) <= 500
        and (not dom or _contains_any(dom, MEM0_DOMAIN_TERMS))
    ):
        return "mem0"

    return "ignore"


def _importance(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _source_ids(candidate: Mapping[str, Any]) -> list[Any]:
    direct = candidate.get("source_message_ids")
    if isinstance(direct, list):
        return direct

    raw_json = candidate.get("source_message_ids_json")
    if isinstance(raw_json, str) and raw_json.strip():
        try:
            parsed = json.loads(raw_json)
        except json.JSONDecodeError:
            return [raw_json]
        return parsed if isinstance(parsed, list) else []
    return []


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)
