from __future__ import annotations

from datetime import UTC, datetime

import polars as pl

from ffapp.tools.weekly_clock import current_projection_week


def test_current_projection_week_advances_after_last_game_finishes() -> None:
    schedule = pl.DataFrame(
        {
            "season": [2026, 2026, 2026],
            "week": [1, 1, 2],
            "season_type": ["REG"] * 3,
            "kickoff_utc": [
                "2026-09-11T00:20:00+00:00",
                "2026-09-15T03:00:00+00:00",
                "2026-09-18T00:20:00+00:00",
            ],
        }
    )

    week = current_projection_week(
        schedule, 2026, now=datetime(2026, 9, 15, 12, tzinfo=UTC)
    )

    assert week == 2
