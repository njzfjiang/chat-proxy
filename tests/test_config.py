from chat_proxy.config import load_config, load_dotenv


def test_load_dotenv_sets_missing_values_without_overriding(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "CHAT_PROXY_UPSTREAM_BASE=https://from-env-file.example/v1",
                "CHAT_PROXY_UPSTREAM_API_KEY='chat-key'",
                "CHAT_PROXY_CHAT_MODEL=chat-model",
                "CHAT_PROXY_PROVIDER_KEY=deepseek",
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
    monkeypatch.delenv("CHAT_PROXY_PROVIDER_KEY", raising=False)
    monkeypatch.delenv("CHAT_PROXY_SUMMARY_MODEL", raising=False)
    monkeypatch.delenv("CHAT_PROXY_SUMMARY_API_KEY", raising=False)
    monkeypatch.delenv("CHAT_PROXY_DAILY_SUMMARY_ENABLED", raising=False)
    monkeypatch.delenv("CHAT_PROXY_DAILY_SUMMARY_RECENT_K", raising=False)
    monkeypatch.delenv("CHAT_PROXY_PORT", raising=False)

    cfg = load_config()

    assert cfg.upstream_base == "https://real-env.example/v1"
    assert cfg.upstream_api_key == "chat-key"
    assert cfg.chat_model == "chat-model"
    assert cfg.provider_key == "deepseek"
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
