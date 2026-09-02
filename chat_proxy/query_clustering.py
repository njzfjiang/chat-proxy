from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sqlite3
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence
from urllib.parse import quote

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.preprocessing import Normalizer


_MONTHS = (
    "jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|"
    "january|february|march|april|june|july|august|september|october|"
    "november|december"
)
_LEADING_TIMESTAMP_RE = re.compile(
    rf"^(?:(?:{_MONTHS})\s+\d{{1,2}},?\s+\d{{4}}\s+at\s+\d{{1,2}}:\d{{2}}(?:\s*[ap]m)?|"
    r"\d{4}-\d{2}-\d{2}(?:[t ]\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?)?)\s*",
    re.IGNORECASE,
)
_ATTACHMENT_RE = re.compile(r"\[(?:image|file):[^\]]+\]", re.IGNORECASE)
_STYLE_PATTERNS = (
    r"_?\(:_」∠\)_?",
    r"\(\s*´▽｀\s*\)",
    r"哈{2,}",
    r"嘿{2,}",
    r"(?:233){1,}\d*",
    r"kai|g老师",
    r"小猫|猫猫|老公|狐狸|夫人|老婆|老师|许先生|桂郎|坏猫",
    r"动动耳朵|动耳朵|眨眨眼|眨眼|尾巴|凑过去|贴过去|"
    r"亲(?:一下|一口|亲)?|抱(?:住|一下)?|摸(?:摸|一下)?|"
    r"蹭(?:蹭|一下)?|咬(?:一下|一口)?|挠(?:一下)?|戳(?:一下)?|捏(?:一下)?",
)
_STYLE_RE = re.compile("|".join(f"(?:{pattern})" for pattern in _STYLE_PATTERNS), re.IGNORECASE)
_LATIN_TOKEN_RE = re.compile(r"[a-z][a-z0-9_+.-]{1,}", re.IGNORECASE)
_CJK_RUN_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]{2,}")
_TOPIC_STOPWORDS = {
    "一下",
    "一个",
    "一些",
    "什么",
    "这个",
    "那个",
    "可以",
    "但是",
    "不过",
    "然后",
    "所以",
    "因为",
    "可能",
    "还是",
    "就是",
    "已经",
    "现在",
    "感觉",
    "觉得",
    "知道",
    "好的",
    "好呀",
    "其实",
    "自己",
    "我们",
    "你们",
    "他们",
    "我的",
    "你的",
    "他的",
    "时候",
    "有点",
    "不是",
    "怎么",
    "继续",
}


@dataclass
class QueryDocument:
    text: str
    representative_id: int
    occurrence_count: int = 1
    first_timestamp: str = ""
    last_timestamp: str = ""
    conversation_ids: set[str] = field(default_factory=set)
    conversation_titles: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class ClusterSummary:
    cluster_id: int
    unique_queries: int
    weighted_turns: int
    conversation_count: int
    top_terms: tuple[str, ...]
    examples: tuple[dict[str, Any], ...]


def normalize_query_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def content_focus_text(value: Any) -> str:
    text = normalize_query_text(value).lower()
    text = _LEADING_TIMESTAMP_RE.sub("", text)
    text = _ATTACHMENT_RE.sub(" ", text)
    text = _STYLE_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip(" ，。！？,.!?~～—-_()（）[]【】")


def topic_analyzer(value: str) -> list[str]:
    text = content_focus_text(value)
    tokens = [
        token
        for match in _LATIN_TOKEN_RE.finditer(text)
        if (token := match.group(0).lower()) not in ENGLISH_STOP_WORDS
    ]
    for match in _CJK_RUN_RE.finditer(text):
        run = match.group(0)
        for size in (2, 3, 4):
            tokens.extend(
                token
                for index in range(len(run) - size + 1)
                if (token := run[index : index + size]) not in _TOPIC_STOPWORDS
            )
    return tokens


def load_user_queries(
    db_path: Path,
    *,
    min_chars: int = 6,
    after: str | None = None,
    before: str | None = None,
    immutable: bool = False,
) -> list[QueryDocument]:
    resolved = db_path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"SQLite database not found: {resolved}")

    query = """
SELECT id, timestamp, content, conversation_id, conversation_title
FROM messages
WHERE role = 'user'
  AND COALESCE(kind, 'chat') = 'chat'
  AND TRIM(COALESCE(content, '')) <> ''
"""
    params: list[str] = []
    if after:
        query += " AND timestamp >= ?"
        params.append(after)
    if before:
        query += " AND timestamp < ?"
        params.append(before)
    query += " ORDER BY id"

    suffix = "?mode=ro"
    if immutable:
        suffix += "&immutable=1"
    uri = f"file:{quote(resolved.as_posix(), safe='/:')}{suffix}"
    grouped: dict[str, QueryDocument] = {}
    with sqlite3.connect(uri, uri=True) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(query, params)
        for row in rows:
            text = normalize_query_text(row["content"])
            if len(text) < max(1, min_chars):
                continue
            timestamp = str(row["timestamp"] or "")
            conversation_id = str(row["conversation_id"] or "").strip()
            title = str(row["conversation_title"] or "").strip()
            existing = grouped.get(text)
            if existing is None:
                grouped[text] = QueryDocument(
                    text=text,
                    representative_id=int(row["id"]),
                    first_timestamp=timestamp,
                    last_timestamp=timestamp,
                    conversation_ids={conversation_id} if conversation_id else set(),
                    conversation_titles={title} if title else set(),
                )
                continue
            existing.occurrence_count += 1
            if timestamp and (
                not existing.first_timestamp or timestamp < existing.first_timestamp
            ):
                existing.first_timestamp = timestamp
            if timestamp and timestamp > existing.last_timestamp:
                existing.last_timestamp = timestamp
            if conversation_id:
                existing.conversation_ids.add(conversation_id)
            if title:
                existing.conversation_titles.add(title)
    return list(grouped.values())


def cluster_queries(
    documents: Sequence[QueryDocument],
    *,
    clusters: int,
    seed: int = 42,
    max_features: int = 50_000,
    min_df: int = 2,
    top_terms: int = 12,
    examples_per_cluster: int = 5,
    profile: str = "content",
    dimensions: int = 128,
    min_topic_chars: int = 8,
) -> tuple[list[int], list[ClusterSummary]]:
    if len(documents) < 2:
        raise ValueError("At least two unique queries are required for clustering.")
    if profile not in {"content", "raw"}:
        raise ValueError("profile must be 'content' or 'raw'.")
    prepared = [
        content_focus_text(document.text) if profile == "content" else document.text
        for document in documents
    ]
    clusterable_indices = [
        index
        for index, text in enumerate(prepared)
        if (
            len(text) >= max(1, min_topic_chars)
            and (topic_analyzer(text) if profile == "content" else text.strip())
        )
    ]
    low_content_indices = sorted(set(range(len(documents))) - set(clusterable_indices))
    if len(clusterable_indices) < 2:
        raise ValueError("At least two queries contain clusterable topic text.")
    cluster_count = max(2, min(clusters, len(clusterable_indices)))
    effective_min_df = max(1, min(min_df, len(clusterable_indices)))
    vectorizer_kwargs: dict[str, Any] = {
        "min_df": effective_min_df,
        "max_df": 0.95,
        "max_features": max_features,
        "sublinear_tf": True,
        "norm": "l2",
    }
    if profile == "content":
        vectorizer = TfidfVectorizer(analyzer=topic_analyzer, **vectorizer_kwargs)
    else:
        vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 5),
            **vectorizer_kwargs,
        )
    matrix = vectorizer.fit_transform(prepared[index] for index in clusterable_indices)
    max_dimensions = min(matrix.shape[0] - 1, matrix.shape[1] - 1)
    if max_dimensions >= 2:
        svd = TruncatedSVD(
            n_components=max(2, min(dimensions, max_dimensions)),
            random_state=seed,
        )
        cluster_matrix = Normalizer(copy=False).fit_transform(svd.fit_transform(matrix))
    else:
        cluster_matrix = matrix
    model = MiniBatchKMeans(
        n_clusters=cluster_count,
        random_state=seed,
        n_init=10,
        batch_size=min(2048, max(256, len(clusterable_indices))),
        max_iter=200,
    )
    weights = [documents[index].occurrence_count for index in clusterable_indices]
    raw_labels = model.fit_predict(cluster_matrix, sample_weight=weights)
    feature_names = vectorizer.get_feature_names_out()

    members: dict[int, list[int]] = defaultdict(list)
    for local_index, raw_label in enumerate(raw_labels):
        members[int(raw_label)].append(local_index)
    raw_order = sorted(
        members,
        key=lambda label: (
            -sum(
                documents[clusterable_indices[local_index]].occurrence_count
                for local_index in members[label]
            ),
            label,
        ),
    )
    remap = {raw_label: rank + 1 for rank, raw_label in enumerate(raw_order)}
    labels = [0] * len(documents)
    for local_index, raw_label in enumerate(raw_labels):
        labels[clusterable_indices[local_index]] = remap[int(raw_label)]

    summaries: list[ClusterSummary] = []
    if low_content_indices:
        summaries.append(
            ClusterSummary(
                cluster_id=0,
                unique_queries=len(low_content_indices),
                weighted_turns=sum(
                    documents[index].occurrence_count for index in low_content_indices
                ),
                conversation_count=len(
                    {
                        conversation_id
                        for index in low_content_indices
                        for conversation_id in documents[index].conversation_ids
                    }
                ),
                top_terms=("low-content/short/social-only",),
                examples=tuple(
                    {
                        "message_id": documents[index].representative_id,
                        "occurrences": documents[index].occurrence_count,
                        "timestamp": documents[index].last_timestamp,
                        "text": documents[index].text,
                    }
                    for index in low_content_indices[:examples_per_cluster]
                ),
            )
        )
    for raw_label in raw_order:
        local_indices = members[raw_label]
        indices = [clusterable_indices[local_index] for local_index in local_indices]
        mean_tfidf = np.asarray(matrix[local_indices].mean(axis=0)).ravel()
        ranked_features = mean_tfidf.argsort()[::-1]
        terms: list[str] = []
        for feature_index in ranked_features:
            term = str(feature_names[feature_index]).strip()
            if not term or term in terms:
                continue
            terms.append(term)
            if len(terms) >= top_terms:
                break

        distances = model.transform(cluster_matrix[local_indices])[:, raw_label]
        representative_indices = sorted(
            range(len(indices)),
            key=lambda local_index: (
                float(distances[local_index]),
                documents[indices[local_index]].representative_id,
            ),
        )[:examples_per_cluster]
        examples = tuple(
            {
                "message_id": documents[indices[local_index]].representative_id,
                "occurrences": documents[indices[local_index]].occurrence_count,
                "timestamp": documents[indices[local_index]].last_timestamp,
                "text": documents[indices[local_index]].text,
            }
            for local_index in representative_indices
        )
        conversation_ids = {
            conversation_id
            for index in indices
            for conversation_id in documents[index].conversation_ids
        }
        summaries.append(
            ClusterSummary(
                cluster_id=remap[raw_label],
                unique_queries=len(indices),
                weighted_turns=sum(
                    documents[index].occurrence_count for index in indices
                ),
                conversation_count=len(conversation_ids),
                top_terms=tuple(terms),
                examples=examples,
            )
        )
    return labels, summaries


def write_cluster_outputs(
    output_dir: Path,
    *,
    db_path: Path,
    documents: Sequence[QueryDocument],
    labels: Sequence[int],
    summaries: Sequence[ClusterSummary],
    settings: dict[str, Any],
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_json = output_dir / "summary.json"
    report_md = output_dir / "report.md"
    assignments_csv = output_dir / "assignments.csv"
    review_csv = output_dir / "cluster_review.csv"

    payload = {
        "source_db": str(db_path.expanduser().resolve()),
        "settings": settings,
        "unique_queries": len(documents),
        "weighted_turns": sum(document.occurrence_count for document in documents),
        "clusters": [
            {
                "cluster_id": summary.cluster_id,
                "unique_queries": summary.unique_queries,
                "weighted_turns": summary.weighted_turns,
                "conversation_count": summary.conversation_count,
                "top_terms": list(summary.top_terms),
                "examples": list(summary.examples),
            }
            for summary in summaries
        ],
    }
    summary_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    with assignments_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "cluster_id",
                "message_id",
                "occurrences",
                "first_timestamp",
                "last_timestamp",
                "conversation_count",
                "conversation_titles",
                "text",
            ],
        )
        writer.writeheader()
        for document, label in sorted(
            zip(documents, labels),
            key=lambda item: (item[1], item[0].representative_id),
        ):
            writer.writerow(
                {
                    "cluster_id": label,
                    "message_id": document.representative_id,
                    "occurrences": document.occurrence_count,
                    "first_timestamp": document.first_timestamp,
                    "last_timestamp": document.last_timestamp,
                    "conversation_count": len(document.conversation_ids),
                    "conversation_titles": " | ".join(
                        sorted(document.conversation_titles)
                    ),
                    "text": document.text,
                }
            )

    with review_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "cluster_id",
                "weighted_turns",
                "unique_queries",
                "suggested_label",
                "memory_type",
                "context_need",
                "top_terms",
                "notes",
            ],
        )
        writer.writeheader()
        for summary in summaries:
            writer.writerow(
                {
                    "cluster_id": summary.cluster_id,
                    "weighted_turns": summary.weighted_turns,
                    "unique_queries": summary.unique_queries,
                    "suggested_label": "",
                    "memory_type": "",
                    "context_need": "",
                    "top_terms": " | ".join(summary.top_terms),
                    "notes": "",
                }
            )

    lines = [
        "# User-turn query clusters",
        "",
        f"- Source DB: `{db_path.expanduser().resolve()}`",
        f"- Unique normalized queries: {len(documents)}",
        f"- Weighted user turns: {sum(item.occurrence_count for item in documents)}",
        f"- Clusters: {len(summaries)}",
        "",
        "Cluster IDs are ordered by weighted turn count. Top terms are character",
        "n-grams or mixed-language topic tokens and should be treated as hints; use",
        "the examples when naming a cluster. Cluster 0 contains turns with no topic",
        "text left after content-focused cleanup.",
        "",
    ]
    for summary in summaries:
        lines.extend(
            [
                f"## Cluster {summary.cluster_id}",
                "",
                (
                    f"{summary.weighted_turns} turns; {summary.unique_queries} unique "
                    f"queries; {summary.conversation_count} conversations."
                ),
                "",
                f"Top terms: {', '.join(summary.top_terms)}",
                "",
            ]
        )
        for example in summary.examples:
            text = str(example["text"])
            if len(text) > 280:
                text = text[:277].rstrip() + "..."
            lines.append(
                f"- `{example['message_id']}` ({example['occurrences']}x): {text}"
            )
        lines.append("")
    report_md.write_text("\n".join(lines), encoding="utf-8")
    return {
        "summary_json": summary_json,
        "report_md": report_md,
        "assignments_csv": assignments_csv,
        "review_csv": review_csv,
    }


def _sample_documents(
    documents: Sequence[QueryDocument], max_documents: int, seed: int
) -> list[QueryDocument]:
    if max_documents <= 0 or len(documents) <= max_documents:
        return list(documents)
    rng = random.Random(seed)
    return [documents[index] for index in sorted(rng.sample(range(len(documents)), max_documents))]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Cluster real user chat turns from a KMLog-compatible SQLite DB."
    )
    parser.add_argument("--db", required=True, type=Path, help="SQLite database path")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark_outputs/query_clusters"),
    )
    parser.add_argument("--clusters", type=int, default=30)
    parser.add_argument("--min-chars", type=int, default=6)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--min-topic-chars", type=int, default=8)
    parser.add_argument("--max-features", type=int, default=50_000)
    parser.add_argument("--dimensions", type=int, default=128)
    parser.add_argument("--max-documents", type=int, default=0)
    parser.add_argument("--top-terms", type=int, default=12)
    parser.add_argument("--examples", type=int, default=5)
    parser.add_argument(
        "--profile",
        choices=("content", "raw"),
        default="content",
        help="content removes common chat-style markers; raw preserves surface style.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--after")
    parser.add_argument("--before")
    parser.add_argument(
        "--immutable",
        action="store_true",
        help="Skip SQLite locking for a static offline DB copy; ignores WAL changes.",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    documents = load_user_queries(
        args.db,
        min_chars=args.min_chars,
        after=args.after,
        before=args.before,
        immutable=args.immutable,
    )
    documents = _sample_documents(documents, args.max_documents, args.seed)
    labels, summaries = cluster_queries(
        documents,
        clusters=args.clusters,
        seed=args.seed,
        max_features=args.max_features,
        min_df=args.min_df,
        top_terms=args.top_terms,
        examples_per_cluster=args.examples,
        profile=args.profile,
        dimensions=args.dimensions,
        min_topic_chars=args.min_topic_chars,
    )
    paths = write_cluster_outputs(
        args.output_dir,
        db_path=args.db,
        documents=documents,
        labels=labels,
        summaries=summaries,
        settings={
            "clusters": args.clusters,
            "min_chars": args.min_chars,
            "min_df": args.min_df,
            "min_topic_chars": args.min_topic_chars,
            "max_features": args.max_features,
            "dimensions": args.dimensions,
            "max_documents": args.max_documents,
            "seed": args.seed,
            "after": args.after,
            "before": args.before,
            "immutable": args.immutable,
            "profile": args.profile,
        },
    )
    print(
        json.dumps(
            {key: str(path.resolve()) for key, path in paths.items()},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
