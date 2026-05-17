from chat_proxy.memory_target import choose_target_layer


def test_choose_target_layer_ignores_low_value_candidate():
    assert (
        choose_target_layer(
            {
                "importance": 2,
                "confidence": "medium",
                "evidence": "Minor daily aside.",
                "source_message_ids": [1],
            }
        )
        == "ignore"
    )


def test_choose_target_layer_promotes_strong_anchor():
    assert (
        choose_target_layer(
            {
                "label": "红线 uptime anchor",
                "evidence": "Durable continuity anchor.",
                "domain": "continuity",
                "function": "boot_core",
                "primary_mother": "G",
                "importance": 5,
                "confidence": "high",
                "source_message_ids": [1],
            }
        )
        == "anchor_table"
    )


def test_choose_target_layer_promotes_auditable_infra_to_vault():
    assert (
        choose_target_layer(
            {
                "label": "DB rebuild",
                "evidence": "Rebuilt the SQLite FTS database.",
                "domain": "infra_asset",
                "function": "infra_reference",
                "importance": 4,
                "confidence": "medium",
                "source_message_ids": [10, 11],
            }
        )
        == "vault"
    )


def test_choose_target_layer_promotes_reference_to_wb():
    assert (
        choose_target_layer(
            {
                "label": "Search database guide",
                "evidence": "Builder creates SQLite FTS and kind indexes.",
                "domain": "infra_asset",
                "function": "infra_reference",
                "primary_mother": "D",
                "importance": 3,
                "confidence": "medium",
                "source_message_ids": [20],
            }
        )
        == "wb"
    )


def test_choose_target_layer_promotes_short_rule_to_mem0():
    assert (
        choose_target_layer(
            {
                "label": "Review memory candidates first",
                "evidence": "Candidates should be audited before committing.",
                "domain": "rule",
                "function": "daily_context",
                "importance": 3,
                "confidence": "high",
                "source_message_ids": [30],
            }
        )
        == "mem0"
    )
