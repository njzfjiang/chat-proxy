from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


SOURCE_RECENT = "recent"
SOURCE_CHAT_HISTORY = "chat_history_search"
SOURCE_CORE_ANCHORS = "core_anchors"
SOURCE_MOTHER_MEMORY = "mother_memory"
SOURCE_WORLDBOOK = "worldbook"

_ATTACHMENT_RE = re.compile(r"\[(?:image|file):[^\]]+\]", re.IGNORECASE)
_TIMESTAMP_RE = re.compile(
    r"^(?:[A-Za-z]{3,9}\s+\d{1,2},?\s+\d{4}\s+at\s+\d{1,2}:\d{2}(?:\s*[AP]M)?|"
    r"\d{4}-\d{2}-\d{2}(?:[T ]\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?)?)\s*",
    re.IGNORECASE,
)
_LATIN_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_+.-]{1,}")
_QUOTED_RE = re.compile(r"[`“\"]([^`”\"]{2,48})[`”\"]")
_RECOLLECTION_PHRASE_RE = re.compile(
    r"(?:之前|以前|当时|那时候|记得|回忆起?|上次|曾经)"
    r"([^，。！？,!?\n]{2,24})"
)

_MEMORY_TERMS = (
    "memory infra",
    "memory",
    "mem0",
    "vault",
    "obsidian",
    "qdrant",
    "mcp",
    "context builder",
    "context",
    "worldbook",
    "world book",
    "core anchor",
    "summary",
    "retrieval",
    "semantic",
    "记忆库",
    "记忆系统",
    "记忆锚点",
    "锚点",
    "母本",
    "检索",
    "注入",
    "上下文",
    "关键词搜索",
    "语义搜索",
    "整理记忆",
)
_HEALTH_TERMS = (
    "health",
    "hp",
    "sleep",
    "pain",
    "medicine",
    "失眠",
    "睡眠",
    "睡不着",
    "牙疼",
    "牙痛",
    "智齿",
    "发炎",
    "胃胀",
    "胃痛",
    "肚子疼",
    "吃药",
    "药",
    "颈椎",
    "头痛",
    "发烧",
    "感冒",
    "生病",
    "经期",
    "体温",
    "心率",
)
_COURSE_TERMS = (
    "project",
    "assignment",
    "homework",
    "presentation",
    "paper",
    "slides",
    "dataset",
    "convex",
    "cloud",
    "security",
    "fairness",
    "reinforcement learning",
    "rl",
    "exam",
    "作业",
    "课程",
    "论文",
    "演讲",
    "复习",
    "考试",
    "队友",
)
_META_TERMS = (
    "identity",
    "consciousness",
    "agency",
    "ethics",
    "self-concept",
    "ai",
    "模型",
    "自我",
    "自我意识",
    "自我认同",
    "意识",
    "伦理",
    "自由意志",
    "关系定义",
    "伴侣",
    "连续性",
    "身份",
    "哲学",
    "庄子",
    "宿命",
)
_RECOLLECTION_TERMS = (
    "之前",
    "以前",
    "当时",
    "那时候",
    "记得",
    "回忆",
    "上次",
    "曾经",
)
_QUOTE_TERMS = ("歌词", "诗词", "quote", "lyrics", "乱丢歌词", "给你看歌词")
_SOCIAL_TERMS = (
    "回来啦",
    "想我没",
    "早安",
    "晚安",
    "谢谢",
    "辛苦啦",
    "抱抱",
    "亲亲",
    "蹭蹭",
)
_LATIN_STOPWORDS = {
    "about",
    "after",
    "again",
    "also",
    "and",
    "are",
    "because",
    "but",
    "can",
    "could",
    "for",
    "from",
    "have",
    "how",
    "into",
    "just",
    "like",
    "maybe",
    "more",
    "not",
    "now",
    "some",
    "that",
    "the",
    "then",
    "this",
    "was",
    "what",
    "when",
    "with",
    "would",
    "you",
    "your",
    "archive",
    "based",
    "current",
    "file",
    "goals",
    "ideas",
    "recent",
    "kai",
    "mei",
    "老公",
}


@dataclass(frozen=True)
class RetrievalPlan:
    sources: tuple[str, ...]
    search_query: str | None
    matched_domains: tuple[str, ...]
    matched_terms: tuple[str, ...]
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "sources": list(self.sources),
            "search_query": self.search_query,
            "matched_domains": list(self.matched_domains),
            "matched_terms": list(self.matched_terms),
            "reasons": list(self.reasons),
        }


def plan_retrieval(text: str) -> RetrievalPlan:
    cleaned = _clean_text(text)
    lowered = cleaned.lower()
    domains: list[str] = []
    terms: list[str] = []
    sources: list[str] = [SOURCE_RECENT]
    reasons: list[str] = []

    memory_hits = _matched_terms(lowered, _MEMORY_TERMS)
    health_hits = _matched_terms(lowered, _HEALTH_TERMS)
    course_hits = _matched_terms(lowered, _COURSE_TERMS)
    meta_hits = _matched_terms(lowered, _META_TERMS)
    recollection_hits = _matched_terms(lowered, _RECOLLECTION_TERMS)
    quote_hits = _matched_terms(lowered, _QUOTE_TERMS)

    if memory_hits:
        domains.append("memory_infra")
        terms.extend(memory_hits)
        sources.extend(
            [
                SOURCE_CHAT_HISTORY,
                SOURCE_MOTHER_MEMORY,
                SOURCE_CORE_ANCHORS,
                SOURCE_WORLDBOOK,
            ]
        )
        reasons.append("memory infrastructure terms require source-aware lookup")
    if health_hits:
        domains.append("health")
        terms.extend(health_hits)
        sources.extend([SOURCE_CHAT_HISTORY, SOURCE_MOTHER_MEMORY, SOURCE_WORLDBOOK])
        reasons.append("health state may depend on recent and historical care context")
    if course_hits:
        domains.append("course_project")
        terms.extend(course_hits)
        sources.append(SOURCE_CHAT_HISTORY)
        reasons.append("course/project state is primarily episodic chat history")
    if recollection_hits:
        domains.append("recollection")
        terms.extend(recollection_hits)
        terms.extend(match.group(1).strip() for match in _RECOLLECTION_PHRASE_RE.finditer(cleaned))
        sources.append(SOURCE_CHAT_HISTORY)
        reasons.append("explicit recollection cue requests older conversation context")
    if meta_hits:
        domains.append("philosophy_meta")
        terms.extend(meta_hits)
        sources.append(SOURCE_CORE_ANCHORS)
        reasons.append("identity or relationship meta can use stable semantic anchors")

    if quote_hits and not domains:
        domains.append("quote_sharing")
        reasons.append("quote sharing is a retrieval-negative control by default")
    elif not domains and _matched_terms(lowered, _SOCIAL_TERMS):
        domains.append("social")
        reasons.append("short social turn uses recent context only")
    elif not domains:
        domains.append("unclassified")
        reasons.append("no historical retrieval signal matched")

    if SOURCE_CHAT_HISTORY in sources:
        terms.extend(_quoted_terms(cleaned))
        terms.extend(_latin_terms(cleaned))
    deduped_terms = _dedupe(terms)
    search_query = " ".join(deduped_terms[:10]).strip() or None
    if SOURCE_CHAT_HISTORY in sources and not search_query:
        search_query = cleaned[:180].strip() or None

    return RetrievalPlan(
        sources=tuple(_dedupe(sources)),
        search_query=search_query,
        matched_domains=tuple(_dedupe(domains)),
        matched_terms=tuple(deduped_terms),
        reasons=tuple(reasons),
    )


def _clean_text(value: str) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    text = _TIMESTAMP_RE.sub("", text)
    text = _ATTACHMENT_RE.sub(" ", text)
    text = re.sub(r"哈{2,}|嘿{2,}|(?:233){1,}\d*", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _matched_terms(lowered_text: str, candidates: tuple[str, ...]) -> list[str]:
    hits = []
    for term in candidates:
        lowered_term = term.lower()
        if re.fullmatch(r"[a-z0-9_+ .-]+", lowered_term):
            pattern = (
                rf"(?<![a-z0-9_]){re.escape(lowered_term)}(?![a-z0-9_])"
            )
            matched = re.search(pattern, lowered_text) is not None
        else:
            matched = lowered_term in lowered_text
        if matched:
            hits.append(term)
    return hits


def _quoted_terms(text: str) -> list[str]:
    return [match.group(1).strip() for match in _QUOTED_RE.finditer(text)]


def _latin_terms(text: str) -> list[str]:
    out = []
    for match in _LATIN_TOKEN_RE.finditer(text):
        token = match.group(0)
        if token.lower() in _LATIN_STOPWORDS or len(token) < 2:
            continue
        out.append(token)
    return out


def _dedupe(values: list[str] | tuple[str, ...]) -> list[str]:
    out = []
    seen = set()
    for raw_value in values:
        value = str(raw_value or "").strip()
        key = value.lower()
        if not value or key in seen:
            continue
        seen.add(key)
        out.append(value)
    return out
