from chat_proxy.retrieval_planner import (
    SOURCE_CHAT_HISTORY,
    SOURCE_CORE_ANCHORS,
    SOURCE_MOTHER_MEMORY,
    SOURCE_WORLDBOOK,
    plan_retrieval,
)
from chat_proxy.context_builder import _match_worldbook_entry


def test_memory_infra_routes_to_history_and_stable_memory_sources():
    plan = plan_retrieval("context builder 的 semantic retrieval 要怎么改")

    assert "memory_infra" in plan.matched_domains
    assert SOURCE_CHAT_HISTORY in plan.sources
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


def test_quote_and_social_turns_do_not_search_chat_history():
    assert SOURCE_CHAT_HISTORY not in plan_retrieval("给你看歌词").sources
    assert SOURCE_CHAT_HISTORY not in plan_retrieval("晚安，抱抱").sources


def test_ascii_router_terms_require_token_boundaries():
    plan = plan_retrieval("I said the fairness project needs another pass")

    assert "course_project" in plan.matched_domains
    assert "philosophy_meta" not in plan.matched_domains


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
    assert (
        _match_worldbook_entry(entry, "失眠", allow_constant=False) is None
    )
