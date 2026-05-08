from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path


DEFAULT_DB_PATH = (
    Path(r"C:\Users\ellat\Desktop\K_Space\kmlog-search\chat_data")
    / "chat_search.db"
)
ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class ProxyConfig:
    upstream_base: str
    db_path: Path
    host: str = "127.0.0.1"
    port: int = 8787
    summary_enabled: bool = False
    summary_upstream_base: str | None = None
    summary_api_key: str | None = None
    summary_model: str = "deepseek-v4-flash"
    summary_recent_k: int = 30


def load_config() -> ProxyConfig:
    load_dotenv()

    upstream = os.getenv("CHAT_PROXY_UPSTREAM_BASE", "").strip()
    if not upstream:
        upstream = "https://api.openai.com/v1"

    db_path = Path(os.getenv("CHAT_PROXY_DB", str(DEFAULT_DB_PATH))).expanduser()
    host = os.getenv("CHAT_PROXY_HOST", "127.0.0.1").strip() or "127.0.0.1"
    port = int(os.getenv("CHAT_PROXY_PORT", "8787"))
    summary_enabled = os.getenv("CHAT_PROXY_SUMMARY_ENABLED", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    summary_upstream = os.getenv("CHAT_PROXY_SUMMARY_UPSTREAM_BASE", "").strip()
    summary_api_key = os.getenv("CHAT_PROXY_SUMMARY_API_KEY", "").strip()
    summary_model = (
        os.getenv("CHAT_PROXY_SUMMARY_MODEL", "deepseek-v4-flash").strip()
        or "deepseek-v4-flash"
    )
    summary_recent_k = int(os.getenv("CHAT_PROXY_SUMMARY_RECENT_K", "30"))

    return ProxyConfig(
        upstream_base=upstream.rstrip("/"),
        db_path=db_path,
        host=host,
        port=port,
        summary_enabled=summary_enabled,
        summary_upstream_base=summary_upstream.rstrip("/") or None,
        summary_api_key=summary_api_key or None,
        summary_model=summary_model,
        summary_recent_k=summary_recent_k,
    )


def load_dotenv(path: Path | str | None = None) -> None:
    env_path = Path(
        path or os.getenv("CHAT_PROXY_ENV_FILE", ".env")
    ).expanduser()
    if not env_path.exists() or not env_path.is_file():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not ENV_KEY_RE.match(key):
            continue
        os.environ.setdefault(key, _parse_env_value(value.strip()))


def _parse_env_value(value: str) -> str:
    quoted = len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}
    if not quoted and "#" in value:
        value = value.split("#", 1)[0].rstrip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
    return value
