from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


DEFAULT_DB_PATH = (
    Path(r"C:\Users\ellat\Desktop\K_Space\kmlog-search\chat_data")
    / "chat_search.db"
)


@dataclass(frozen=True)
class ProxyConfig:
    upstream_base: str
    db_path: Path
    host: str = "127.0.0.1"
    port: int = 8787


def load_config() -> ProxyConfig:
    upstream = os.getenv("CHAT_PROXY_UPSTREAM_BASE", "").strip()
    if not upstream:
        upstream = "https://api.openai.com/v1"

    db_path = Path(os.getenv("CHAT_PROXY_DB", str(DEFAULT_DB_PATH))).expanduser()
    host = os.getenv("CHAT_PROXY_HOST", "127.0.0.1").strip() or "127.0.0.1"
    port = int(os.getenv("CHAT_PROXY_PORT", "8787"))

    return ProxyConfig(
        upstream_base=upstream.rstrip("/"),
        db_path=db_path,
        host=host,
        port=port,
    )
