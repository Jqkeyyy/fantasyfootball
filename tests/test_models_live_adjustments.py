from __future__ import annotations

import polars as pl

from ffapp.models.live_adjustments import (
    apply_adjustments,
    build_injury_adjustments,
    build_role_adjustments,
    expand_role_adjustments,
    parse_injury_duration,
)


def test_parse_injury_duration_understands_ranges_and_plain_language() -> None:
    assert parse_injury_duration("Expected to miss 2-4 weeks") == (2, 4)
    assert parse_injury_duration("Re-evaluated in 10-14 days") == (2, 2)
    assert parse_injury_duration("Considered week-to-week") == (1, 3)
    assert parse_injury_duration("Season-ending surgery") == (99, 99)
    assert parse_injury_duration(None) is None


def test_injury_curve_uses_reported_range_and_explains_it() -> None:
    projections = pl.DataFrame(
        {
            "player_id": ["p1"] * 5,
            "season": [2026] * 5,
            "week": [3, 4, 5, 6, 7],
            "mean": [10.0] * 5,
        }
    )
    players = pl.DataFrame(
        {
            "player_id": ["p1"],
            "full_name": ["Injured Player"],
            "injury_status": ["Out"],
            "injury_body_part": ["Hamstring"],
            "injury_notes": ["Expected to miss 2-4 weeks"],
        }
    )

    audit = build_injury_adjustments(projections, players, anchor_week=3)

    assert audit.sort("week")["multiplier"].to_list() == [0.0, 0.0, 1 / 3, 2 / 3]
    assert "reported 2-4 week recovery" in audit["reason"][0]


def test_unknown_out_status_uses_conservative_recovery_curve() -> None:
    projections = pl.DataFrame(
        {
            "player_id": ["p1"] * 4,
            "season": [2026] * 4,
            "week": [3, 4, 5, 6],
        }
    )
    players = pl.DataFrame(
        {
            "player_id": ["p1"],
            "injury_status": ["Out"],
            "injury_notes": [None],
        }
    )

    audit = build_injury_adjustments(projections, players, anchor_week=3)

    assert audit.sort("week")["multiplier"].to_list() == [0.0, 0.45, 0.80, 0.95]
    assert "no return date" in audit["reason"][0]


def _usage_rows(include_target_week: bool = False) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    baseline = [(2025, week, 0.40, 0.10) for week in range(13, 17)]
    recent = [(2026, 1, 0.75, 0.22), (2026, 2, 0.80, 0.25)]
    future = [(2026, 3, 0.01, 0.01)] if include_target_week else []
    for season, week, snap, target in baseline + recent + future:
        rows.append(
            {
                "player_id": "p1",
                "season": season,
                "week": week,
                "offense_snap_pct": snap,
                "target_share": target,
                "carry_share": 0.0,
                "rz_touch_share": target,
            }
        )
    return pl.DataFrame(rows)


def test_role_change_requires_corroboration_and_ignores_target_week() -> None:
    players = pl.DataFrame(
        {"player_id": ["p1"], "full_name": ["Riser"], "position": ["WR"]}
    )

    without_future = build_role_adjustments(
        _usage_rows(), players, season=2026, target_week=3
    )
    with_future = build_role_adjustments(
        _usage_rows(include_target_week=True), players, season=2026, target_week=3
    )

    assert without_future.to_dicts() == with_future.to_dicts()
    assert without_future.height == 1
    assert 1.0 < without_future["multiplier"][0] <= 1.075
    assert without_future["early_season_weight"][0] == 0.5
    assert "early-season strength 50%" in without_future["reason"][0]
    assert "snap share" in without_future["reason"][0]
    assert "target share" in without_future["reason"][0]


def test_role_change_does_not_react_to_one_signal() -> None:
    usage = _usage_rows().with_columns(
        pl.lit(0.10).alias("target_share"), pl.lit(0.10).alias("rz_touch_share")
    )
    players = pl.DataFrame(
        {"player_id": ["p1"], "full_name": ["One Signal"], "position": ["WR"]}
    )

    result = build_role_adjustments(usage, players, season=2026, target_week=3)

    assert result.is_empty()


def test_role_effect_fades_and_injury_is_applied_after_it() -> None:
    projections = pl.DataFrame(
        {
            "player_id": ["p1"] * 5,
            "season": [2026] * 5,
            "week": [3, 4, 5, 6, 7],
            "mean": [10.0] * 5,
            "p_active": [0.95] * 5,
        }
    )
    role = pl.DataFrame(
        {
            "player_id": ["p1"],
            "player_name": ["Player"],
            "season": [2026],
            "week": [3],
            "adjustment_type": ["role"],
            "multiplier": [1.12],
            "direction": ["up"],
            "confidence": [0.8],
            "reason": ["two-game role shift"],
        }
    )
    expanded = expand_role_adjustments(role, projections, anchor_week=3)
    assert expanded.sort("week")["multiplier"].to_list() == [1.12, 1.09, 1.06, 1.03]

    injury = role.with_columns(
        pl.lit("injury").alias("adjustment_type"),
        pl.lit(0.0).alias("multiplier"),
        pl.lit("down").alias("direction"),
    )
    adjusted, _ = apply_adjustments(
        projections.filter(pl.col("week") == 3), pl.concat([role, injury])
    )
    assert adjusted["mean"][0] == 0.0
    assert adjusted["p_active"][0] == 0.0
