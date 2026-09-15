from __future__ import annotations

import polars as pl

from ffapp.tools.weekly_alerts import build_weekly_alerts


def _snapshot(mean: list[float], active: list[float], waiver: list[float]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "player_id": ["starter", "free"],
            "player_name": ["Starter", "Free Agent"],
            "mean": mean,
            "p_active": active,
            "is_starter": [True, False],
            "is_rostered": [True, False],
            "waiver_upgrade": waiver,
        }
    )


def test_alerts_capture_new_out_change_and_waiver_threshold() -> None:
    previous = _snapshot([15.0, 8.0], [1.0, 1.0], [0.0, 2.0])
    current = _snapshot([10.0, 12.0], [0.05, 1.0], [0.0, 4.0])

    alerts = build_weekly_alerts(current, previous)

    assert set(alerts["kind"]) == {
        "starter_ruled_out",
        "starter_projection_changed",
        "waiver_upgrade",
    }


def test_alerts_do_not_repeat_unchanged_conditions() -> None:
    current = _snapshot([10.0, 12.0], [0.05, 1.0], [0.0, 4.0])

    alerts = build_weekly_alerts(current, current)

    assert alerts.is_empty()


def test_projection_change_does_not_compare_different_weeks() -> None:
    previous = _snapshot([15.0, 8.0], [1.0, 1.0], [0.0, 0.0]).with_columns(
        pl.lit(2026).alias("season"), pl.lit(2).alias("week")
    )
    current = _snapshot([9.0, 8.0], [1.0, 1.0], [0.0, 0.0]).with_columns(
        pl.lit(2026).alias("season"), pl.lit(3).alias("week")
    )

    assert build_weekly_alerts(current, previous).is_empty()
