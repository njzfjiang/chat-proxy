from __future__ import annotations

import argparse
import asyncio
from datetime import datetime
from typing import Sequence
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .config import load_config
from .daily_summary import update_daily_summary
from .storage import ChatProxyStore


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Manually run the daily summary job for a local date."
    )
    parser.add_argument(
        "--date",
        required=True,
        help="Local date to summarize, formatted as YYYY-MM-DD.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete the existing daily summary for this date before summarizing.",
    )
    parser.add_argument(
        "--scan-limit",
        type=int,
        default=100000,
        help="Maximum messages to scan while backfilling an older date.",
    )
    args = parser.parse_args(argv)

    date_key = _validate_date(args.date)
    cfg = load_config()
    store = ChatProxyStore(cfg.db_path)
    store.initialize()

    if args.force:
        store.delete_daily_summary(date_key)

    now = _local_noon(date_key, cfg.daily_summary_timezone)
    asyncio.run(
        update_daily_summary(
            cfg=cfg,
            store=store,
            now=now,
            scan_limit=args.scan_limit,
        )
    )

    row = store.get_daily_summary(date_key)
    if not row:
        print(f"No messages found for {date_key}, or daily summaries are disabled.")
        return 0

    candidates = store.get_daily_memory_candidates(
        date_key=date_key,
        summary_version=int(row["version"]),
    )
    print(
        f"{date_key}: {row['status']} v{row['version']} "
        f"last_message_id={row['last_message_id']} "
        f"candidates={len(candidates)}"
    )
    if row.get("error_text"):
        print(row["error_text"])
    return 0


def _validate_date(value: str) -> str:
    try:
        parsed = datetime.strptime(value, "%Y-%m-%d")
    except ValueError as exc:
        raise SystemExit("--date must be formatted as YYYY-MM-DD") from exc
    return parsed.date().isoformat()


def _local_noon(date_key: str, timezone_name: str) -> str:
    try:
        tz = ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError:
        tz = ZoneInfo("America/Toronto")
    return datetime.strptime(date_key, "%Y-%m-%d").replace(
        hour=12,
        tzinfo=tz,
    ).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
