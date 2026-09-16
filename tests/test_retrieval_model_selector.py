import asyncio

import pytest

from chat_proxy.config import ProxyConfig
from chat_proxy.retrieval_model_selector import (
    build_selector_candidates,
    parse_selector_result,
    resolve_selector_evidence,
    select_retrieval_candidates,
)


def _candidates():
    return [
        {
            "id": 10,
            "role": "user",
            "evidence_excerpts": [{"text": "direct event evidence"}],
        },
        {
            "id": 20,
            "role": "assistant",
            "evidence_excerpts": [{"text": "background advice"}],
        },
    ]


def test_selector_result_accepts_known_ids_and_reject_all():
    candidates = build_selector_candidates(_candidates(), 600)
    selected = parse_selector_result(
        '```json\n{"selected":[{"id":10,"evidence_excerpt_indices":[0],'
        '"reason":"direct report"}],"reject_all_reason":null}\n```',
        candidates,
        limit=5,
    )
    rejected = parse_selector_result(
        '{"selected":[],"reject_all_reason":"No event match."}',
        candidates,
        limit=5,
    )

    assert selected["selected"][0]["id"] == 10
    assert selected["reject_all_reason"] is None
    assert rejected == {"selected": [], "reject_all_reason": "No event match."}


def test_selector_preserves_original_excerpt_indices():
    candidates = build_selector_candidates(
        [
            {
                "id": 10,
                "evidence_excerpts": [
                    {"text": ""},
                    {"text": "usable second excerpt"},
                ],
            }
        ],
        600,
    )

    result = parse_selector_result(
        '{"selected":[{"id":10,"evidence_excerpt_indices":[1],'
        '"reason":"second excerpt"}],"reject_all_reason":null}',
        candidates,
        limit=5,
    )

    assert result["selected"][0]["evidence_excerpt_indices"] == [1]


def test_selector_exports_only_the_model_visible_slice():
    source = [
        {
            "id": 10,
            "role": "user",
            "evidence_excerpts": [
                {"text": "a" * 240},
                {"text": "b" * 240},
                {"text": "c" * 240},
            ],
        }
    ]
    visible = build_selector_candidates(source, 600)
    selection = {
        "selected": [
            {
                "id": 10,
                "evidence_excerpt_indices": [2],
                "reason": "third excerpt",
            }
        ]
    }

    resolved = resolve_selector_evidence(selection, visible, source)

    assert resolved[0]["evidence"] == ["c" * 120]
    assert resolved[0]["evidence_slices"][0]["source_end"] == 120
    assert resolved[0]["evidence_slices"][0]["source_excerpt_chars"] == 240
    assert len(resolved[0]["evidence_slices"][0]["source_excerpt_sha256"]) == 64


def test_selector_slice_offsets_reference_the_untrimmed_source():
    source = [{"id": 10, "evidence_excerpts": [{"text": "  visible text  "}]}]
    visible = build_selector_candidates(source, 600)
    selection = {
        "selected": [
            {
                "id": 10,
                "evidence_excerpt_indices": [0],
                "reason": "direct",
            }
        ]
    }

    resolved = resolve_selector_evidence(selection, visible, source)
    evidence_slice = resolved[0]["evidence_slices"][0]

    assert evidence_slice["text"] == "visible text"
    assert evidence_slice["source_start"] == 2
    assert evidence_slice["source_end"] == 14
    assert evidence_slice["source_excerpt_chars"] == 16


@pytest.mark.parametrize(
    "payload",
    [
        '{"selected":[{"id":99,"evidence_excerpt_indices":[0],"reason":"x"}]}',
        '{"selected":[{"id":10,"evidence_excerpt_indices":[9],"reason":"x"}]}',
        '{"selected":[{"id":10,"evidence_excerpt_indices":[0],"reason":""}]}',
        '{"selected":[],"reject_all_reason":""}',
    ],
)
def test_selector_result_rejects_unverifiable_output(payload):
    with pytest.raises(RuntimeError):
        parse_selector_result(
            payload, build_selector_candidates(_candidates(), 600), limit=5
        )


def test_selector_call_reuses_summary_configuration(monkeypatch, tmp_path):
    captured = {}

    class FakeResponse:
        status_code = 200

        def json(self):
            return {
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"selected":[{"id":10,"evidence_excerpt_indices":[0],'
                                '"reason":"direct"}],"reject_all_reason":null}'
                            )
                        }
                    }
                ],
                "usage": {"total_tokens": 12},
            }

    class FakeClient:
        def __init__(self, *args, **kwargs):
            captured["timeout"] = kwargs.get("timeout")

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def post(self, url, headers, json):
            captured.update(url=url, headers=headers, body=json)
            return FakeResponse()

    monkeypatch.setattr(
        "chat_proxy.retrieval_model_selector.httpx.AsyncClient", FakeClient
    )
    cfg = ProxyConfig(
        upstream_base="http://disabled",
        db_path=tmp_path / "unused.db",
        summary_upstream_base="https://summary.example/v1",
        summary_api_key="secret",
        summary_model="small-selector",
    )

    result = asyncio.run(
        select_retrieval_candidates(
            cfg=cfg,
            query="Which event?",
            candidates=_candidates(),
            thinking_mode="disabled",
            max_output_tokens=320,
        )
    )

    assert captured["url"] == "https://summary.example/v1/chat/completions"
    assert captured["headers"]["authorization"] == "Bearer secret"
    assert captured["body"]["model"] == "small-selector"
    assert captured["body"]["thinking"] == {"type": "disabled"}
    assert captured["body"]["max_tokens"] == 320
    assert result["selected"][0]["id"] == 10
    assert result["usage"] == {"total_tokens": 12}
