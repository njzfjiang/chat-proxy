from chat_proxy.config import load_config, load_dotenv


def test_load_dotenv_sets_missing_values_without_overriding(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "CHAT_PROXY_UPSTREAM_BASE=https://from-env-file.example/v1",
                "CHAT_PROXY_UPSTREAM_API_KEY='chat-key'",
                "CHAT_PROXY_CHAT_MODEL=chat-model",
                "CHAT_PROXY_CHAT_RECENT_K=17",
                "CHAT_PROXY_EXPLICIT_MESSAGES_RECENT_K=11",
                "CHAT_PROXY_PROVIDER_KEY=deepseek",
                f"CHAT_PROXY_WORLDBOOK_PATH={tmp_path / 'worldbook.json'}",
                f"CHAT_PROXY_WORLDBOOK_PATHS={tmp_path / 'wb2.json'};{tmp_path / 'wb3.json'}",
                "CHAT_PROXY_WORLDBOOK_MAX_ITEMS=3",
                "CHAT_PROXY_WORLDBOOK_CHARS_TOTAL=900",
                "CHAT_PROXY_RETRIEVAL_ENABLED=true",
                "CHAT_PROXY_RETRIEVAL_INJECT_ENABLED=false",
                "CHAT_PROXY_KMLOG_SEARCH_URL=http://127.0.0.1:8013",
                "CHAT_PROXY_KMLOG_SEARCH_API_KEY='search-key'",
                "CHAT_PROXY_KMLOG_SEARCH_LIMIT=7",
                "CHAT_PROXY_KMLOG_SEARCH_CHARS_TOTAL=1500",
                "CHAT_PROXY_KMLOG_SEARCH_TIMEOUT_SECONDS=2.5",
                "CHAT_PROXY_SUMMARY_ENABLED=true",
                "CHAT_PROXY_SUMMARY_MODEL='deepseek-v4-flash'",
                "CHAT_PROXY_SUMMARY_API_KEY=\"abc#123\"",
                "CHAT_PROXY_DAILY_SUMMARY_ENABLED=true",
                "CHAT_PROXY_DAILY_SUMMARY_RECENT_K=125",
                "CHAT_PROXY_PORT=9999 # inline comment",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("CHAT_PROXY_ENV_FILE", str(env_file))
    monkeypatch.setenv("CHAT_PROXY_UPSTREAM_BASE", "https://real-env.example/v1")
    monkeypatch.delenv("CHAT_PROXY_SUMMARY_ENABLED", raising=False)
    monkeypatch.delenv("CHAT_PROXY_UPSTREAM_API_KEY", raising=False)
    monkeypatch.delenv("CHAT_PROXY_CHAT_MODEL", raising=False)
    monkeypatch.delenv("CHAT_PROXY_CHAT_RECENT_K", raising=False)
    monkeypatch.delenv("CHAT_PROXY_EXPLICIT_MESSAGES_RECENT_K", raising=False)
    monkeypatch.delenv("CHAT_PROXY_PROVIDER_KEY", raising=False)
    monkeypatch.delenv("CHAT_PROXY_WORLDBOOK_PATH", raising=False)
    monkeypatch.delenv("CHAT_PROXY_WORLDBOOK_PATHS", raising=False)
    monkeypatch.delenv("CHAT_PROXY_WORLDBOOK_MAX_ITEMS", raising=False)
    monkeypatch.delenv("CHAT_PROXY_WORLDBOOK_CHARS_TOTAL", raising=False)
    monkeypatch.delenv("CHAT_PROXY_RETRIEVAL_ENABLED", raising=False)
    monkeypatch.delenv("CHAT_PROXY_RETRIEVAL_INJECT_ENABLED", raising=False)
    monkeypatch.delenv("CHAT_PROXY_KMLOG_SEARCH_URL", raising=False)
    monkeypatch.delenv("CHAT_PROXY_KMLOG_SEARCH_API_KEY", raising=False)
    monkeypatch.delenv("CHAT_PROXY_KMLOG_SEARCH_LIMIT", raising=False)
    monkeypatch.delenv("CHAT_PROXY_KMLOG_SEARCH_CHARS_TOTAL", raising=False)
    monkeypatch.delenv("CHAT_PROXY_KMLOG_SEARCH_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("CHAT_PROXY_SUMMARY_MODEL", raising=False)
    monkeypatch.delenv("CHAT_PROXY_SUMMARY_API_KEY", raising=False)
    monkeypatch.delenv("CHAT_PROXY_DAILY_SUMMARY_ENABLED", raising=False)
    monkeypatch.delenv("CHAT_PROXY_DAILY_SUMMARY_RECENT_K", raising=False)
    monkeypatch.delenv("CHAT_PROXY_PORT", raising=False)

    cfg = load_config()

    assert cfg.upstream_base == "https://real-env.example/v1"
    assert cfg.upstream_api_key == "chat-key"
    assert cfg.chat_model == "chat-model"
    assert cfg.chat_recent_k == 17
    assert cfg.explicit_messages_recent_k == 11
    assert cfg.provider_key == "deepseek"
    assert cfg.worldbook_enabled is True
    assert cfg.worldbook_path == tmp_path / "worldbook.json"
    assert cfg.worldbook_paths == (
        tmp_path / "worldbook.json",
        tmp_path / "wb2.json",
        tmp_path / "wb3.json",
    )
    assert cfg.worldbook_max_items == 3
    assert cfg.worldbook_chars_total == 900
    assert cfg.retrieval_enabled is True
    assert cfg.retrieval_inject_enabled is False
    assert cfg.kmlog_search_url == "http://127.0.0.1:8013"
    assert cfg.kmlog_search_api_key == "search-key"
    assert cfg.kmlog_search_limit == 7
    assert cfg.kmlog_search_chars_total == 1500
    assert cfg.kmlog_search_timeout_seconds == 2.5
    assert cfg.summary_enabled is True
    assert cfg.summary_model == "deepseek-v4-flash"
    assert cfg.summary_api_key == "abc#123"
    assert cfg.daily_summary_enabled is True
    assert cfg.daily_summary_upstream_base is None
    assert cfg.daily_summary_api_key == "abc#123"
    assert cfg.daily_summary_model == "deepseek-v4-flash"
    assert cfg.daily_summary_recent_k == 125
    assert cfg.port == 9999


def test_load_dotenv_ignores_missing_file(monkeypatch, tmp_path):
    missing = tmp_path / "missing.env"
    monkeypatch.setenv("CHAT_PROXY_ENV_FILE", str(missing))

    load_dotenv()
