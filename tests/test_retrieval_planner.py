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
        "entity_filtered": 1,
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


def test_router_mode_can_exclude_constant_worldbook_entries():
    entry = {
        "id": "always-on",
        "enabled": True,
        "constantActive": True,
        "content": "generic style",
    }

    assert _match_worldbook_entry(entry, "失眠") is not None
    assert _match_worldbook_entry(entry, "失眠", allow_constant=False) is None
