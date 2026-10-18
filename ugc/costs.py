"""Estimated-spend ledger and daily cap. Estimates only; fal bills the real amount."""
from __future__ import annotations

import json
import threading
from datetime import date, datetime, timezone

from . import config

_lock = threading.Lock()


class CapExceeded(RuntimeError):
    pass


def estimate_images(n: int) -> float:
    return round(n * config.IMAGE_USD_EACH, 4)


def estimate_video(seconds: float) -> float:
    return round(seconds * config.VIDEO_USD_PER_SEC, 4)


def _path():
    return config.DATA_DIR / "ledger.jsonl"


def today_total() -> float:
    p = _path()
    if not p.exists():
        return 0.0
    today = date.today().isoformat()
    total = 0.0
    for line in p.read_text().splitlines():
        row = json.loads(line)
        if row["ts"].startswith(today):
            total += row["usd"]
    return round(total, 4)


def guard(estimate: float) -> None:
    spent = today_total()
    if spent + estimate > config.DAILY_CAP_USD:
        raise CapExceeded(
            f"estimated ${estimate:.2f} would take today's spend (${spent:.2f}) over the "
            f"${config.DAILY_CAP_USD:.2f} cap (UGC_DAILY_CAP_USD)"
        )


def record(kind: str, usd: float, note: str = "") -> None:
    with _lock:
        _path().parent.mkdir(parents=True, exist_ok=True)
        row = {"ts": datetime.now(timezone.utc).astimezone().isoformat(), "kind": kind, "usd": usd, "note": note}
        with _path().open("a") as f:
            f.write(json.dumps(row) + "\n")
