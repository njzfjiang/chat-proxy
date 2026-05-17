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
    upstream_api_key: str | None = None
    chat_model: str = "deepseek-v4-flash"
    chat_recent_k: int = 20
    provider_key: str | None = None
    worldbook_enabled: bool = False
    worldbook_path: Path | None = None
    worldbook_paths: tuple[Path, ...] = ()
    worldbook_max_items: int = 2
    worldbook_chars_total: int = 800
    summary_enabled: bool = False
    summary_upstream_base: str | None = None
    summary_api_key: str | None = None
    summary_model: str = "deepseek-v4-flash"
    summary_recent_k: int = 30
    daily_summary_enabled: bool = False
    daily_summary_upstream_base: str | None = None
    daily_summary_api_key: str | None = None
    daily_summary_model: str = "deepseek-v4-flash"
    daily_summary_recent_k: int = 200
    daily_summary_timezone: str = "America/Toronto"


def load_config() -> ProxyConfig:
    load_dotenv()

    upstream = os.getenv("CHAT_PROXY_UPSTREAM_BASE", "").strip()
    if not upstream:
        upstream = "https://api.openai.com/v1"

    db_path = Path(os.getenv("CHAT_PROXY_DB", str(DEFAULT_DB_PATH))).expanduser()
    host = os.getenv("CHAT_PROXY_HOST", "127.0.0.1").strip() or "127.0.0.1"
    port = int(os.getenv("CHAT_PROXY_PORT", "8787"))
    upstream_api_key = (
        os.getenv("CHAT_PROXY_UPSTREAM_API_KEY", "").strip()
        or os.getenv("CHAT_PROXY_API_KEY", "").strip()
    )
    chat_model = (
        os.getenv("CHAT_PROXY_CHAT_MODEL", "").strip()
        or os.getenv("CHAT_PROXY_MODEL", "").strip()
        or "deepseek-v4-flash"
    )
    chat_recent_k = int(os.getenv("CHAT_PROXY_CHAT_RECENT_K", "20"))
    provider_key = os.getenv("CHAT_PROXY_PROVIDER_KEY", "").strip()
    worldbook_path_raw = os.getenv("CHAT_PROXY_WORLDBOOK_PATH", "").strip()
    worldbook_paths_raw = os.getenv("CHAT_PROXY_WORLDBOOK_PATHS", "").strip()
    worldbook_path = Path(worldbook_path_raw).expanduser() if worldbook_path_raw else None
    worldbook_paths = _parse_path_list(worldbook_paths_raw)
    if worldbook_path:
        worldbook_paths = (worldbook_path, *worldbook_paths)
    worldbook_enabled = _env_bool("CHAT_PROXY_WORLDBOOK_ENABLED") or bool(worldbook_paths)
    worldbook_max_items = int(os.getenv("CHAT_PROXY_WORLDBOOK_MAX_ITEMS", "2"))
    worldbook_chars_total = int(os.getenv("CHAT_PROXY_WORLDBOOK_CHARS_TOTAL", "800"))
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
    daily_summary_enabled = _env_bool("CHAT_PROXY_DAILY_SUMMARY_ENABLED")
    daily_summary_upstream = (
        os.getenv("CHAT_PROXY_DAILY_SUMMARY_UPSTREAM_BASE", "").strip()
        or summary_upstream
    )
    daily_summary_api_key = (
        os.getenv("CHAT_PROXY_DAILY_SUMMARY_API_KEY", "").strip()
        or summary_api_key
    )
    daily_summary_model = (
        os.getenv("CHAT_PROXY_DAILY_SUMMARY_MODEL", "").strip()
        or summary_model
        or "deepseek-v4-flash"
    )
    daily_summary_recent_k = int(
        os.getenv("CHAT_PROXY_DAILY_SUMMARY_RECENT_K", "200")
    )
    daily_summary_timezone = (
        os.getenv("CHAT_PROXY_DAILY_SUMMARY_TIMEZONE", "America/Toronto").strip()
        or "America/Toronto"
    )

    return ProxyConfig(
        upstream_base=upstream.rstrip("/"),
        db_path=db_path,
        host=host,
        port=port,
        upstream_api_key=upstream_api_key or None,
        chat_model=chat_model,
        chat_recent_k=chat_recent_k,
        provider_key=provider_key or None,
        worldbook_enabled=worldbook_enabled,
        worldbook_path=worldbook_path,
        worldbook_paths=worldbook_paths,
        worldbook_max_items=worldbook_max_items,
        worldbook_chars_total=worldbook_chars_total,
        summary_enabled=summary_enabled,
        summary_upstream_base=summary_upstream.rstrip("/") or None,
        summary_api_key=summary_api_key or None,
        summary_model=summary_model,
        summary_recent_k=summary_recent_k,
        daily_summary_enabled=daily_summary_enabled,
        daily_summary_upstream_base=daily_summary_upstream.rstrip("/") or None,
        daily_summary_api_key=daily_summary_api_key or None,
        daily_summary_model=daily_summary_model,
        daily_summary_recent_k=daily_summary_recent_k,
        daily_summary_timezone=daily_summary_timezone,
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


def _env_bool(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _parse_path_list(value: str) -> tuple[Path, ...]:
    paths: list[Path] = []
    seen: set[str] = set()
    for raw_item in re.split(r"[;\n]", value):
        item = raw_item.strip().strip('"').strip("'")
        if not item:
            continue
        path = Path(item).expanduser()
        key = str(path).lower()
        if key in seen:
            continue
        seen.add(key)
        paths.append(path)
    return tuple(paths)
