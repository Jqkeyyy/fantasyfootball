"""Calendar helpers for unattended in-season refreshes."""

from __future__ import annotations

from datetime import UTC, datetime

import polars as pl


def current_projection_week(
    schedule: pl.DataFrame, season: int, *, now: datetime | None = None
) -> int:
    """Return the earliest regular-season week that still has an unplayed kickoff."""
    current_time = now or datetime.now(UTC)
    rows = schedule.filter(pl.col("season") == season)
    if "season_type" in rows.columns:
        rows = rows.filter(pl.col("season_type") == "REG")
    if rows.is_empty():
        raise ValueError(f"Schedule has no regular-season rows for {season}")
    weeks_with_future_game = [
        int(row["week"])
        for row in rows.iter_rows(named=True)
        if row.get("kickoff_utc") is not None
        and datetime.fromisoformat(str(row["kickoff_utc"]).replace("Z", "+00:00")) >= current_time
    ]
    if weeks_with_future_game:
        return min(weeks_with_future_game)
    max_week = rows.select(pl.col("week").max().cast(pl.Int64)).item()
    if not isinstance(max_week, int):
        raise ValueError(f"Schedule has no valid week number for {season}")
    return max_week


__all__ = ["current_projection_week"]
