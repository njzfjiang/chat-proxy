from chat_proxy.retrieval_planner import (
    SOURCE_CHAT_HISTORY,
    SOURCE_CORE_ANCHORS,
    SOURCE_MOTHER_MEMORY,
    SOURCE_RECENT_GOALS,
    SOURCE_REVIEWED_MEMORY,
    SOURCE_WORLDBOOK,
    plan_retrieval,
)
from chat_proxy.context_builder import (
    _match_worldbook_entry,
    _rank_typed_source_items,
    _rerank_kmlog_results,
)


def test_memory_infra_routes_to_history_and_stable_memory_sources():
    plan = plan_retrieval("context builder 的 semantic retrieval 要怎么改")

    assert "memory_infra" in plan.matched_domains
    assert SOURCE_CHAT_HISTORY in plan.sources
    assert SOURCE_RECENT_GOALS in plan.sources
    assert SOURCE_REVIEWED_MEMORY in plan.sources
    assert SOURCE_MOTHER_MEMORY in plan.sources
    assert SOURCE_CORE_ANCHORS in plan.sources
    assert "context builder" in plan.search_query


def test_health_routes_to_history_mother_memory_and_worldbook():
    plan = plan_retrieval("我昨晚又睡不着，还吃了药")

    assert "health" in plan.matched_domains
    assert SOURCE_CHAT_HISTORY in plan.sources
    assert SOURCE_MOTHER_MEMORY in plan.sources
    assert SOURCE_WORLDBOOK in plan.sources


def test_identity_meta_routes_to_core_without_user_profile_mother_section():
    plan = plan_retrieval("模型的自我认同会不会随对话改变")

    assert SOURCE_CORE_ANCHORS in plan.sources
    assert SOURCE_MOTHER_MEMORY not in plan.sources


def test_model_term_alone_does_not_trigger_identity_meta_or_core():
    plan = plan_retrieval(
        "队友写了一个 CycleGAN training module，模型输出需要重新 evaluation"
    )

    assert "course_project" in plan.matched_domains
    assert "philosophy_meta" not in plan.matched_domains
    assert SOURCE_CORE_ANCHORS not in plan.sources


def test_course_query_keeps_entities_and_drops_logistics_terms():
    plan = plan_retrieval(
        "DHL 到了，队友说 5090 跑 CycleGAN 和 Pix2Pix iteration 快很多"
    )

    assert plan.required_terms == ("cyclegan", "pix2pix")
    assert "dhl" not in plan.search_query.lower()
    assert "队友" not in plan.search_query


def test_unnamed_project_query_expands_deliverable_synonyms():
    plan = plan_retrieval("老师要求做 prototyping，之后还要交作业和演讲")

    assert "prototype" in plan.optional_terms
    assert "presentation" in plan.optional_terms
    assert "assignment" in plan.optional_terms


def test_chat_reranker_requires_project_entity_and_deduplicates():
    plan = plan_retrieval("DHL 到了，CycleGAN 和 Pix2Pix 跑得更快")
    rows = [
        {"id": 1, "content_preview": "DHL 快递正在配送"},
        {"id": 2, "content_preview": "CycleGAN 10-shot training run"},
        {"id": 3, "content_preview": "CycleGAN 10-shot training run"},
        {"id": 4, "content_preview": "Pix2Pix evaluation finished"},
        {
            "id": 5,
            "content_preview": (
                "The following context is provided by the system. CycleGAN"
            ),
        },
    ]

    ranked, stats = _rerank_kmlog_results(rows, plan=plan, limit=5)

    assert [item["id"] for item in ranked] == [2, 4]
    assert stats == {
        "synthetic_filtered": 1,
        "duplicate_filtered": 1,
        "query_echo_filtered": 0,
        "entity_filtered": 1,
        "filter_reasons": [
            {"id": 1, "reason": "missing_required_term", "required_terms": ["cyclegan", "pix2pix"]},
            {"id": 3, "reason": "duplicate_content", "duplicate_of": 2},
            {"id": 5, "reason": "synthetic_context"},
        ],
    }


def test_quote_and_social_turns_do_not_search_chat_history():
    assert SOURCE_CHAT_HISTORY not in plan_retrieval("给你看歌词").sources
    assert SOURCE_CHAT_HISTORY not in plan_retrieval("晚安，抱抱").sources


def test_quote_preference_statement_does_not_become_recollection_query():
    plan = plan_retrieval("其实我以前最喜欢这首歌，给你乱丢歌词")

    assert plan.matched_domains == ("quote_sharing",)
    assert SOURCE_CHAT_HISTORY not in plan.sources


def test_explicit_request_to_recall_a_quote_can_search_history():
    plan = plan_retrieval("你还记得我以前给你看过哪句歌词吗？")

    assert "recollection" in plan.matched_domains
    assert SOURCE_CHAT_HISTORY in plan.sources


def test_ascii_router_terms_require_token_boundaries():
    plan = plan_retrieval("I said the fairness project needs another pass")

    assert "course_project" in plan.matched_domains
    assert "philosophy_meta" not in plan.matched_domains


def test_course_project_uses_j_then_reviewed_then_chat_fallback():
    plan = plan_retrieval("CSC2555 project 现在做到哪一步了")

    assert plan.sources.index(SOURCE_RECENT_GOALS) < plan.sources.index(
        SOURCE_REVIEWED_MEMORY
    )
    assert plan.sources.index(SOURCE_REVIEWED_MEMORY) < plan.sources.index(
        SOURCE_CHAT_HISTORY
    )
    assert "recent goals, then reviewed memory" in plan.reasons[0]


def test_typed_source_ranking_rejects_generic_project_only_match():
    plan = plan_retrieval("这个 project 到底要做什么")
    items = [{"title": "Daily care", "body": "Rest when a project deadline is close."}]

    assert (
        _rank_typed_source_items(
            items,
            plan=plan,
            fields=("title", "body"),
            limit=4,
        )
        == []
    )


def test_typed_source_ranking_keeps_specific_course_anchor():
    plan = plan_retrieval("genAI fairness project 做完了吗")
    items = [
        {
            "title": "Course follow-up",
            "body": "Compress the Algo Fairness and GenAI project notes.",
        }
    ]

    ranked = _rank_typed_source_items(
        items,
        plan=plan,
        fields=("title", "body"),
        limit=4,
    )

    assert ranked[0][0] >= 2
    assert "genai" in {term.lower() for term in ranked[0][2]}


def test_motion_phrase_does_not_trigger_recollection():
    plan = plan_retrieval("抱住你腰贴过去")

    assert "recollection" not in plan.matched_domains
    assert SOURCE_CHAT_HISTORY not in plan.sources


def test_recollection_uses_story_subject_instead_of_discussion_source():
    plan = plan_retrieval(
        "我之前在和deepseek讨论这个，但是现在推进剧情需要某个人物拿到某片证据"
    )

    assert "creative_writing" in plan.matched_domains
    assert "recollection" in plan.matched_domains
    assert plan.search_query == "剧情 人物 证据"
    assert plan.required_terms == ("剧情", "人物", "证据")
    assert plan.optional_terms == ()
    assert "deepseek" not in plan.search_query.casefold()
    assert "之前" not in plan.search_query


def test_recollection_keeps_topic_named_after_discussion_source():
    plan = plan_retrieval("我之前和DeepSeek讨论FISTA convergence，还记得吗")

    assert "fista" in plan.search_query.casefold()
    assert "convergence" in plan.search_query.casefold()
    assert "deepseek" not in plan.search_query.casefold()


def test_entity_acquisition_recollection_uses_entity_and_evidence_terms():
    plan = plan_retrieval("之前关颐是怎么拿到线索的？")

    assert plan.matched_domains == ("recollection",)
    assert plan.search_query == "关颐 线索 拿到"
    assert plan.required_terms == ("关颐",)
    assert plan.optional_terms == ("线索", "拿到")
    assert "之前" not in plan.search_query
    assert "怎么" not in plan.search_query


def test_entity_acquisition_recollection_preserves_specific_object():
    plan = plan_retrieval("之前谢常流是怎么拿到断剑线索的？")

    assert plan.search_query == "谢常流 断剑 线索 拿到"
    assert plan.required_terms == ("谢常流", "断剑")
    assert plan.optional_terms == ("线索", "拿到")


def test_entity_acquisition_does_not_promote_generic_pronouns():
    plan = plan_retrieval("之前你是怎么拿到证据的？")

    assert plan.required_terms == ()
    assert "你是" not in plan.required_terms


def test_entity_acquisition_recollection_filters_weak_cue_matches():
    plan = plan_retrieval("之前关颐是怎么拿到线索的？")
    rows = [
        {"id": 0, "role": "user", "content_preview": "之前关颐是怎么拿到线索的？"},
        {"id": 1, "content_preview": "之前只是讨论了别的事情。"},
        {"id": 2, "content_preview": "关颐在唱片夹层里发现纸条，拿到了关键线索。"},
        {"id": 3, "content_preview": "关颐脑中拼图合上，随后展开了泛黄的信。"},
    ]

    ranked, stats = _rerank_kmlog_results(
        rows,
        plan=plan,
        limit=5,
        query_text="之前关颐是怎么拿到线索的？",
    )

    assert [item["id"] for item in ranked] == [2, 3]
    assert stats["entity_filtered"] == 1
    assert stats["query_echo_filtered"] == 1
    assert stats["filter_reasons"] == [
        {"id": 0, "reason": "query_echo"},
        {
            "id": 1,
            "reason": "missing_required_term",
            "required_terms": ["关颐"],
        },
    ]


def test_entity_acquisition_requires_object_and_renders_answer_bearing_context():
    from chat_proxy.retrieval_evidence import render_kmlog_results

    plan = plan_retrieval("之前谢常流是怎么拿到断剑线索的？")
    rows = [
        {
            "id": 1,
            "evidence_version": 1,
            "content_hash": "generic",
            "matched_excerpt": "谢常流后来抵达京城，继续追查师兄死亡。",
            "body_matched_terms": ["谢常流"],
        },
        {
            "id": 2,
            "evidence_version": 1,
            "content_hash": "direct",
            "matched_excerpt": (
                "无关的前情说明。" * 20
                + "\n- 谢常流在京城找铸剑铺/黑市消息，查到："
                + "\n  - 宁州断剑的钢材和纹路，只有云欢楼背后的兵器铺能拿到。"
                + "\n后续还有其他剧情建议。" * 20
            ),
            "body_matched_terms": ["谢常流", "断剑", "线索", "拿到"],
        },
    ]

    ranked, stats = _rerank_kmlog_results(rows, plan=plan, limit=5)
    content, render_stats = render_kmlog_results(ranked, total_chars=320)

    assert [item["id"] for item in ranked] == [2]
    assert stats["entity_filtered"] == 1
    assert stats["filter_reasons"][0]["missing_required_terms"] == ["断剑"]
    assert "铸剑铺/黑市消息" in content
    assert "宁州断剑" in content
    assert "云欢楼" in content
    assert "能拿到" in content
    assert render_stats["selected_after_budget_ids"] == [2]


def test_creative_writing_reranker_keeps_a_conservative_coherence_gate():
    plan = plan_retrieval("推进剧情时，人物怎么拿到证据")
    rows = [
        {"id": 1, "content_preview": "剧情里出现了一份证据"},
        {"id": 2, "content_preview": "这个人物推动了剧情"},
        {"id": 3, "content_preview": "人物在剧情里拿到了证据"},
    ]

    ranked, stats = _rerank_kmlog_results(rows, plan=plan, limit=5)

    assert [item["id"] for item in ranked] == [3]
    assert stats["entity_filtered"] == 2
    assert stats["filter_reasons"][0]["missing_creative_concepts"] == [
        "transfer"
    ]
    assert stats["filter_reasons"][1]["missing_creative_concepts"] == [
        "evidence",
        "transfer",
    ]


def test_creative_writing_reranker_accepts_evidence_paraphrases():
    plan = plan_retrieval("推进剧情时，人物怎么拿到证据")
    rows = [
        {"id": 1, "content_preview": "关颐得设法让叶长依拿到那封密信，否则下一幕无法成立。"},
        {"id": 2, "content_preview": "她必须把线索交到他手里，但直接见面会暴露身份。"},
        {"id": 3, "content_preview": "关颐脑中最后一块拼图合上。她掰开指节，展开泛黄的信。"},
    ]

    ranked, stats = _rerank_kmlog_results(rows, plan=plan, limit=5)

    assert [item["id"] for item in ranked] == [1, 2, 3]
    assert stats["entity_filtered"] == 0
    assert all(
        item["creative_match"]["acceptance"] == "semantic_evidence_coherence"
        for item in ranked
    )


def test_long_self_contained_recollection_narration_skips_history():
    plan = plan_retrieval(
        "之前家里的树在一场雪里倒了，我拿它写过作文。"
        "那时候我还学过滑冰，总是站不起来，家里人坐在旁边笑。"
        "后来回头看，这两件事都挺有意思，也让我发现大家表达关心的方式很不一样。"
        "现在讲起来已经没有当时那么难受，只觉得是很久以前的一段生活记录。"
        "作文里写了树，现实里讲的是滑冰，两个片段并没有需要继续追查的前情。"
        "我只是忽然想起这些细节，所以顺手把它们完整地讲出来。"
    )

    assert "recollection" not in plan.matched_domains
    assert SOURCE_CHAT_HISTORY not in plan.sources


def test_creative_evidence_survives_backend_filter_and_visible_budget(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    from chat_proxy import context_builder
    from chat_proxy.config import ProxyConfig
    from chat_proxy.retrieval_semantics import CREATIVE_EVIDENCE_EXPANSION_TERMS

    rows = [
        {
            "id": 200,
            "evidence_version": 1,
            "content_hash": "negative",
            "matched_excerpt": "今天只是在闲聊角色外观，没有任何可用线索。",
            "body_matched_terms": ["角色"],
        },
        {
            "id": 101,
            "evidence_version": 1,
            "content_hash": "positive-1",
            "matched_excerpt": "关颐得设法让叶长依拿到那封密信，否则下一幕无法成立。",
            "body_matched_terms": ["下一幕", "密信"],
        },
        {
            "id": 102,
            "evidence_version": 1,
            "content_hash": "positive-2",
            "matched_excerpt": "她必须把线索交到他手里，但直接见面会暴露身份。",
            "body_matched_terms": ["线索"],
        },
        {
            "id": 103,
            "evidence_version": 1,
            "content_hash": "positive-3",
            "matched_excerpt": "关颐脑中最后一块拼图合上。她掰开指节，展开泛黄的信。",
            "body_matched_terms": ["拼图"],
        },
    ]
    captured = {}

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def post(self, *args, **kwargs):
            captured.update(kwargs["json"])
            return SimpleNamespace(
                raise_for_status=lambda: None,
                json=lambda: {
                    "results": rows,
                    "evidence_version": 1,
                    "candidate_ids": [item["id"] for item in rows],
                    "selected_ids": [item["id"] for item in rows],
                    "candidate_count": len(rows),
                },
            )

    monkeypatch.setattr(context_builder.httpx, "Client", FakeClient)
    cfg = ProxyConfig(
        upstream_base="http://disabled",
        db_path=tmp_path / "unused.db",
        retrieval_enabled=True,
        retrieval_inject_enabled=True,
        retrieval_router_enabled=True,
        retrieval_query_planner_enabled=True,
        kmlog_search_url="http://test",
        kmlog_search_limit=5,
        kmlog_search_chars_total=1200,
    )

    messages, snapshot = context_builder._kmlog_search_messages(
        body={}, cfg=cfg, query="推进剧情时，人物怎么拿到证据"
    )

    assert captured["evidence_terms"] == list(CREATIVE_EVIDENCE_EXPANSION_TERMS)
    assert snapshot["backend_result_ids"] == [200, 101, 102, 103]
    assert snapshot["rerank_input_ids"] == [200, 101, 102, 103]
    assert snapshot["rerank_output_ids"] == [101, 102, 103]
    assert snapshot["final_injected_ids"] == [101, 102, 103]
    assert snapshot["chars"] <= 1200
    assert len(messages) == 1
    assert all(
        marker in messages[0]["content"] for marker in ("密信", "线索", "拼图")
    )


def test_self_contained_recollection_stops_before_backend(tmp_path, monkeypatch):
    from chat_proxy import context_builder
    from chat_proxy.config import ProxyConfig

    class UnexpectedClient:
        def __init__(self, *args, **kwargs):
            raise AssertionError("self-contained narration must not call the backend")

    monkeypatch.setattr(context_builder.httpx, "Client", UnexpectedClient)
    cfg = ProxyConfig(
        upstream_base="http://disabled",
        db_path=tmp_path / "unused.db",
        retrieval_enabled=True,
        retrieval_inject_enabled=True,
        retrieval_router_enabled=True,
        retrieval_query_planner_enabled=True,
        kmlog_search_url="http://test",
    )
    query = (
        "之前家里的树在一场雪里倒了，我拿它写过作文。"
        "那时候我还学过滑冰，总是站不起来，家里人坐在旁边笑。"
        "后来回头看，这两件事都挺有意思，也让我发现大家表达关心的方式很不一样。"
        "现在讲起来已经没有当时那么难受，只觉得是很久以前的一段生活记录。"
        "作文里写了树，现实里讲的是滑冰，两个片段并没有需要继续追查的前情。"
        "我只是忽然想起这些细节，所以顺手把它们完整地讲出来。"
    )

    messages, snapshot = context_builder._kmlog_search_messages(
        body={}, cfg=cfg, query=query
    )

    assert messages == []
    assert snapshot["skipped_reason"] == "router did not select chat_history_search"
    assert snapshot.get("final_injected_ids", []) == []


def test_router_mode_can_exclude_constant_worldbook_entries():
    entry = {
        "id": "always-on",
        "enabled": True,
        "constantActive": True,
        "content": "generic style",
    }

    assert _match_worldbook_entry(entry, "失眠") is not None
    assert _match_worldbook_entry(entry, "失眠", allow_constant=False) is None


def test_evidence_filter_uses_body_matches_and_full_content_hash():
    from types import SimpleNamespace

    plan = SimpleNamespace(required_terms=["FISTA"], optional_terms=[])
    rows = [
        {"id": 1, "content_preview": "same prefix", "conversation_title": "FISTA",
         "evidence_version": 1, "body_matched_terms": [], "content_hash": "one"},
        {"id": 2, "content_preview": "same prefix", "evidence_version": 1,
         "body_matched_terms": ["FISTA"], "matched_excerpt": "FISTA converged", "content_hash": "two"},
        {"id": 3, "content_preview": "same prefix", "evidence_version": 1,
         "body_matched_terms": ["FISTA"], "matched_excerpt": "FISTA failed", "content_hash": "three"},
        {"id": 4, "content_preview": "same prefix", "evidence_version": 1,
         "body_matched_terms": ["FISTA"], "content_hash": "two"},
    ]
    ranked, stats = _rerank_kmlog_results(rows, plan=plan, limit=5)
    assert [r["id"] for r in ranked] == [2, 3]
    assert stats["entity_filtered"] == 1
    assert stats["duplicate_filtered"] == 1


def test_rejected_legacy_row_does_not_poison_dedup():
    from types import SimpleNamespace

    plan = SimpleNamespace(required_terms=["FISTA"], optional_terms=[])
    rows = [{"id": 1, "content_preview": "same", "conversation_title": "other"},
            {"id": 2, "content_preview": "same", "conversation_title": "FISTA"}]
    ranked, _ = _rerank_kmlog_results(rows, plan=plan, limit=5)
    assert [r["id"] for r in ranked] == [2]


def test_excerpt_budget_does_not_emit_tiny_trailing_fragments():
    from chat_proxy.context_builder import _clip_kmlog_excerpt

    assert _clip_kmlog_excerpt("long evidence " * 30, 1) == ""
    assert _clip_kmlog_excerpt("short fact", 20) == "short fact"
    assert len(_clip_kmlog_excerpt("x" * 720, 240)) == 240


def test_evidence_budget_prioritizes_required_anchor_over_body_matches():
    from chat_proxy.context_builder import _clip_kmlog_evidence

    excerpt = "early dataset " + "x" * 600 + " convex anchor " + "y" * 500
    item = {
        "matched_excerpt": excerpt,
        "planner_required_matches": ["convex"],
        "body_matched_terms": ["dataset", "convex"],
    }

    clipped = _clip_kmlog_evidence(item, 240)

    assert len(clipped) <= 240
    assert "convex" in clipped
    assert "dataset" not in clipped


def test_evidence_budget_uses_body_anchors_without_required_terms():
    from chat_proxy.context_builder import _clip_kmlog_evidence

    excerpt = "early dataset " + "x" * 600 + " trailing text"
    item = {
        "matched_excerpt": excerpt,
        "planner_required_matches": [],
        "body_matched_terms": ["dataset"],
    }

    clipped = _clip_kmlog_evidence(item, 120)

    assert "dataset" in clipped


def test_evidence_budget_prioritizes_required_terms_when_all_anchors_do_not_fit():
    from chat_proxy.context_builder import _clip_kmlog_evidence

    item = {
        "matched_excerpt": "required " + "x" * 100 + " optional-extra-long",
        "planner_required_matches": ["required"],
        "body_matched_terms": ["optional-extra-long"],
    }

    clipped = _clip_kmlog_evidence(item, 12)

    assert "required" in clipped
    assert "optional-extra-long" not in clipped


def test_evidence_budget_prefers_a_complete_short_sentence():
    from chat_proxy.context_builder import _clip_kmlog_evidence

    item = {
        "matched_excerpt": (
            "Unrelated setup with several words. "
            "FISTA converged after twelve steps. "
            "Unrelated trailing discussion with several words."
        ),
        "planner_required_matches": ["FISTA"],
        "body_matched_terms": ["FISTA"],
    }

    clipped = _clip_kmlog_evidence(item, 45)

    assert clipped == "FISTA converged after twelve steps."


def test_evidence_budget_groups_required_terms_by_sentence():
    from chat_proxy.context_builder import _clip_kmlog_evidence

    item = {
        "matched_excerpt": (
            "LASSO is a convex formulation. "
            "Methods include ISTA, accelerated FISTA, and ADMM. "
            "Unrelated trailing explanation."
        ),
        "planner_required_matches": ["convex", "ISTA", "FISTA", "ADMM"],
        "body_matched_terms": ["dataset"],
    }

    clipped = _clip_kmlog_evidence(item, 90)

    assert clipped == (
        "LASSO is a convex formulation.\n...\n"
        "Methods include ISTA, accelerated FISTA, and ADMM."
    )


def test_evidence_items_share_the_total_budget(tmp_path, monkeypatch):
    from chat_proxy import context_builder
    from chat_proxy.config import ProxyConfig

    rows = [{"id": i, "evidence_version": 1, "content_preview": "prefix",
             "matched_excerpt": "x" * 720, "body_matched_terms": ["FISTA"]}
            for i in range(5)]

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def post(self, *args, **kwargs):
            from types import SimpleNamespace
            return SimpleNamespace(raise_for_status=lambda: None,
                                   json=lambda: {"results": rows, "evidence_version": 1})

    monkeypatch.setattr(context_builder.httpx, "Client", FakeClient)
    cfg = ProxyConfig(upstream_base="http://disabled", db_path=tmp_path / "unused.db",
                      retrieval_enabled=True, kmlog_search_url="http://test",
                      kmlog_search_chars_total=1200,
                      retrieval_renderer_v2_enabled=True)
    _, snapshot = context_builder._kmlog_search_messages(body={}, cfg=cfg, query="FISTA")
    assert snapshot["selected_after_budget_ids"] == list(range(5))
    assert snapshot["trace_version"] == 2
    assert snapshot["temporal_scope"] == "current_index"
    assert snapshot["request_payload"] == {
        "query": "FISTA",
        "limit": 5,
        "mode": "auto",
        "kinds": ["chat"],
        "include_evidence": True,
    }
    assert snapshot["rerank_input_ids"] == list(range(5))
    assert snapshot["rerank_output_ids"] == list(range(5))
    assert snapshot["cutoff_filtered_ids"] == []
    assert snapshot["budget_total_chars"] == 1200
    assert snapshot["budget_excerpt_chars"] < 1200
    assert snapshot["budget_used_chars"] + snapshot["budget_wrapper_chars"] <= 1200
    assert snapshot["chars"] <= 1200
    assert snapshot["budget_dropped"] == []
    assert snapshot["final_injected_ids"] == []
    assert all(item["source_chars"] == 720 for item in snapshot["items"])
    assert all(item["truncated"] is True for item in snapshot["items"])
    assert all(item["visible_matched_terms"] == [] for item in snapshot["items"])
    assert snapshot["required_terms_visible_count"] == 0
    assert snapshot["required_terms_missing_count"] == 0


def test_evidence_renderer_reallocates_unused_item_budget_within_total():
    from chat_proxy.retrieval_evidence import render_kmlog_results

    rows = [
        {"id": 1, "evidence_version": 1, "matched_excerpt": "short fact"},
        {"id": 2, "evidence_version": 1, "matched_excerpt": "x" * 200},
    ]

    content, snapshot = render_kmlog_results(rows, total_chars=100)

    assert snapshot["selected_after_budget_ids"] == [1]
    assert snapshot["items"][0]["chars"] == 10
    assert snapshot["budget_dropped"] == [
        {"id": 2, "reason": "below_minimum_fragment"}
    ]
    assert len(content) <= 100
    assert snapshot["budget_used_chars"] + snapshot["budget_wrapper_chars"] <= 100
    assert snapshot["budget_strategy"] == "remaining_equal_share"


def test_semantic_renderer_prioritizes_evidence_and_action_anchors():
    from chat_proxy.retrieval_evidence import render_kmlog_results

    rows = [
        {
            "id": 7553,
            "evidence_version": 1,
            "matched_excerpt": (
                "关颐脑中最后一块拼图合上。她掰开僵硬的指节，展开泛黄的信。"
                + "无关分析" * 80
                + "最后才泛泛提到人物塑造。"
            ),
            "body_matched_terms": ["拼图", "人物"],
            "planner_required_matches": ["人物"],
            "creative_match": {
                "accepted": True,
                "acceptance": "semantic_evidence_coherence",
                "concept_matches": {
                    "narrative": [],
                    "evidence": ["拼图"],
                    "transfer": ["掰开", "展开"],
                },
                "missing_concepts": [],
            },
        }
    ]

    content, snapshot = render_kmlog_results(rows, total_chars=120)

    assert "拼图" in content
    assert "掰开" in content
    assert "展开" in content
    assert snapshot["items"][0]["visible_creative_terms"] == [
        "拼图",
        "掰开",
        "展开",
    ]
    assert snapshot["items"][0]["missing_creative_terms"] == []
