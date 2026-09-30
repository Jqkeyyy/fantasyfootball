"""Small, dependency-free host and source checks for the operations page."""

from __future__ import annotations

import json
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import polars as pl

from ffapp.config import Settings


def host_status(settings: Settings, *, now: datetime | None = None) -> dict[str, object]:
    current = now or datetime.now(UTC)
    total, used, free = shutil.disk_usage(settings.data_root)
    app_reachable = False
    try:
        with urlopen("http://127.0.0.1:8501/_stcore/health", timeout=2) as response:
            app_reachable = response.status == 200 and response.read().strip() == b"ok"
    except (OSError, URLError):
        pass
    return {
        "checked_at": current.isoformat(),
        "dashboard": "Online" if app_reachable else "Unreachable",
        "disk_used_pct": used / total if total else 0.0,
        "disk_free_gb": free / 1024**3,
        "github_token": "Configured"
        if os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN")
        else "Anonymous",
    }


def source_status(settings: Settings, *, now: datetime | None = None) -> pl.DataFrame:
    """Summarize the newest cache metadata for every upstream source."""
    current = now or datetime.now(UTC)
    newest: dict[str, tuple[datetime, Path, int | None]] = {}
    for path in settings.data_root.glob("raw/**/*.meta.json"):
        try:
            payload = json.loads(path.read_text())
            fetched = datetime.fromisoformat(str(payload["fetched_at_utc"]).replace("Z", "+00:00"))
            source = str(payload.get("source") or path.parent.name)
            rows = int(payload["rows"]) if payload.get("rows") is not None else None
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            continue
        if source not in newest or fetched > newest[source][0]:
            newest[source] = (fetched, path, rows)
    rows_out = []
    for source, (fetched, path, rows) in sorted(newest.items()):
        age = max(0.0, (current - fetched.astimezone(UTC)).total_seconds() / 3600)
        rows_out.append(
            {
                "source": source,
                "status": "Fresh" if age <= 36 else "Stale",
                "age_hours": age,
                "rows": rows,
                "latest_cache": path.stem.removesuffix(".meta"),
            }
        )
    return (
        pl.DataFrame(rows_out)
        if rows_out
        else pl.DataFrame(
            schema={
                "source": pl.String,
                "status": pl.String,
                "age_hours": pl.Float64,
                "rows": pl.Int64,
                "latest_cache": pl.String,
            }
        )
    )


__all__ = ["host_status", "source_status"]
