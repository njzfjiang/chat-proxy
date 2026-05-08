from __future__ import annotations

import uvicorn

from .app import create_app
from .config import load_config


def main() -> None:
    cfg = load_config()
    uvicorn.run(
        create_app(cfg),
        host=cfg.host,
        port=cfg.port,
        reload=False,
    )


if __name__ == "__main__":
    main()
