"""Pure data helpers for the automation control center."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from ffapp.config import LeagueConfig, Settings


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _age_hours(stamp: object, now: datetime) -> float | None:
    if not isinstance(stamp, str):
        return None
    try:
        parsed = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    except ValueError:
        return None
    return max(0.0, (now - parsed.astimezone(UTC)).total_seconds() / 3600)


def league_operation_row(
    settings: Settings, league: LeagueConfig, *, now: datetime | None = None
) -> dict[str, object]:
    """Summarize one league's latest automation run and materialized coverage."""
    current = now or datetime.now(UTC)
    output = settings.data_root / "outputs" / league.slug
    manifest = _read_json(output / "refresh_runs" / "latest.json")
    gameday = _read_json(output / "gameday.json")
    alerts = _read_json(output / "alerts" / "latest.json")
    steps = manifest.get("steps", [])
    problems = (
        [
            f"{step.get('name')}: {str(step.get('detail') or '')[:120]}"
            for step in steps
            if isinstance(step, dict) and step.get("status") in {"degraded", "failed"}
        ]
        if isinstance(steps, list)
        else []
    )

    projected = 0
    total = 0
    coverage_path = output / "projection_coverage.parquet"
    if coverage_path.exists():
        coverage = pl.read_parquet(coverage_path)
        if not coverage.is_empty():
            latest = (
                coverage.select("season", "week")
                .unique()
                .sort(["season", "week"], descending=True)
                .row(0)
            )
            coverage = coverage.filter(
                (pl.col("season") == latest[0]) & (pl.col("week") == latest[1])
            )
            projected = coverage.filter(pl.col("projected")).height
            total = coverage.height

    alert_rows = alerts.get("alerts", [])
    return {
        "league": league.display_name,
        "slug": league.slug,
        "status": str(manifest.get("status", "missing")),
        "season": manifest.get("season"),
        "week": manifest.get("week"),
        "last_refresh": manifest.get("generated_at_utc"),
        "age_hours": _age_hours(manifest.get("generated_at_utc"), current),
        "coverage": f"{projected}/{total}" if total else "missing",
        "problems": ", ".join(problems) if problems else "None",
        "decision_alerts": len(alert_rows) if isinstance(alert_rows, list) else 0,
        "gameday_status": str(gameday.get("status", "pending")),
        "gameday_checked": gameday.get("checked_at"),
    }


def recent_run_rows(
    settings: Settings, league: LeagueConfig, *, limit: int = 5
) -> list[dict[str, object]]:
    """Load the newest immutable refresh manifests for a league."""
    directory = settings.data_root / "outputs" / league.slug / "refresh_runs"
    rows: list[dict[str, object]] = []
    if not directory.exists():
        return rows
    for path in sorted(directory.glob("*.json"), reverse=True):
        if path.name == "latest.json":
            continue
        payload = _read_json(path)
        if not payload:
            continue
        steps = payload.get("steps", [])
        problems = (
            [
                str(step.get("name"))
                for step in steps
                if isinstance(step, dict) and step.get("status") in {"degraded", "failed"}
            ]
            if isinstance(steps, list)
            else []
        )
        rows.append(
            {
                "league": league.display_name,
                "generated_at": payload.get("generated_at_utc"),
                "season": payload.get("season"),
                "week": payload.get("week"),
                "status": payload.get("status"),
                "problems": ", ".join(problems) if problems else "None",
            }
        )
        if len(rows) >= limit:
            break
    return rows


__all__ = ["league_operation_row", "recent_run_rows"]
