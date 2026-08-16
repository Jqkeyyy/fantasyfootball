from __future__ import annotations

import polars as pl
import pytest

from ffapp.models import ros_consensus


def _season_points(name: str, player_id: str, points: float, position: str = "RB") -> pl.DataFrame:
    return pl.DataFrame(
        {"player_id": [player_id], "position": [position], "team": ["KC"], "points": [points]}
    )


def test_resolve_remaining_value_subtracts_actuals_for_flat_trend() -> None:
    """A source whose real trend is 'flat' (a static preseason snapshot,
    per check_sources' own detection) is treated as full-season -- the
    safer default -- and real actuals-to-date are subtracted."""
    season_points = {
        "espn": pl.DataFrame(
            {"player_id": ["p1"], "position": ["RB"], "team": ["KC"], "points": [220.0]}
        )
    }
    trend_by_source = {"espn": "flat"}
    actuals = pl.DataFrame({"player_id": ["p1"], "actual_points_to_date": [80.0]})

    result = ros_consensus.resolve_remaining_value(season_points, trend_by_source, actuals)
    row = result.row(0, named=True)
    assert row["points"] == pytest.approx(140.0)
    assert row["branch"] == "subtracted"


def test_resolve_remaining_value_uses_directly_for_declining_trend() -> None:
    season_points = {
        "cbs": pl.DataFrame(
            {"player_id": ["p1"], "position": ["RB"], "team": ["KC"], "points": [140.0]}
        )
    }
    trend_by_source = {"cbs": "declining"}
    actuals = pl.DataFrame({"player_id": ["p1"], "actual_points_to_date": [80.0]})

    result = ros_consensus.resolve_remaining_value(season_points, trend_by_source, actuals)
    row = result.row(0, named=True)
    assert row["points"] == pytest.approx(140.0)
    assert row["branch"] == "ros_direct"


def test_resolve_remaining_value_defaults_to_subtracted_when_trend_unknown() -> None:
    """insufficient_data (or a source missing from trend_by_source
    entirely) defaults to the safer full-season branch, per requirement 1's
    own explicit instruction."""
    season_points = {
        "fftoday": pl.DataFrame(
            {"player_id": ["p1"], "position": ["RB"], "team": ["KC"], "points": [200.0]}
        )
    }
    actuals = pl.DataFrame({"player_id": ["p1"], "actual_points_to_date": [50.0]})

    result = ros_consensus.resolve_remaining_value(season_points, {}, actuals)
    row = result.row(0, named=True)
    assert row["points"] == pytest.approx(150.0)
    assert row["branch"] == "subtracted"


def test_resolve_remaining_value_clips_at_zero() -> None:
    season_points = {
        "espn": pl.DataFrame(
            {"player_id": ["p1"], "position": ["RB"], "team": ["KC"], "points": [60.0]}
        )
    }
    actuals = pl.DataFrame({"player_id": ["p1"], "actual_points_to_date": [90.0]})

    result = ros_consensus.resolve_remaining_value(season_points, {"espn": "flat"}, actuals)
    assert result.row(0, named=True)["points"] == pytest.approx(0.0)


def test_resolve_remaining_value_no_actuals_row_treated_as_zero_scored() -> None:
    """A player with no real logged actuals-to-date row (e.g. a rookie
    with 0 games played yet) subtracts nothing, not null."""
    season_points = {
        "espn": pl.DataFrame(
            {"player_id": ["rookie"], "position": ["WR"], "team": ["KC"], "points": [90.0]}
        )
    }
    actuals = pl.DataFrame(
        {"player_id": [], "actual_points_to_date": []},
        schema={"player_id": pl.String, "actual_points_to_date": pl.Float64},
    )
    result = ros_consensus.resolve_remaining_value(season_points, {"espn": "flat"}, actuals)
    assert result.row(0, named=True)["points"] == pytest.approx(90.0)


def test_aggregate_remaining_value_trims_and_reports_n_sources() -> None:
    resolved = pl.DataFrame(
        {
            "join_key": ["p1|rb", "p1|rb", "p1|rb"],
            "player_name": ["P One"] * 3,
            "position": ["RB"] * 3,
            "points": [100.0, 110.0, 105.0],
        }
    )
    result = ros_consensus.aggregate_remaining_value(resolved)
    row = result.row(0, named=True)
    assert row["n_sources"] == 3
    assert row["season_consensus_ros_points"] == pytest.approx(105.0, abs=1.0)
