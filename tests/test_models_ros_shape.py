from __future__ import annotations

import polars as pl
import pytest

from ffapp.models import ros_shape


def test_frozen_defense_ratings_uses_latest_week_at_or_before_anchor() -> None:
    dpa = pl.DataFrame(
        {
            "season": [2026] * 4,
            "week": [1, 2, 3, 4],
            "defteam": ["BUF", "BUF", "BUF", "BUF"],
            "position_group": ["WR"] * 4,
            "adj_epa_allowed": [0.0, 0.1, 0.2, 0.9],  # week 4 is in the future -- must not leak in
            "n_plays": [20, 22, 21, 25],
        }
    )
    result = ros_shape.frozen_defense_ratings(dpa, season=2026, as_of_week=3, position_group="WR")
    row = result.row(0, named=True)
    assert row["defteam"] == "BUF"
    assert row["frozen_adj_epa_allowed"] == pytest.approx(0.2)  # week 3, not week 4


def test_future_week_opponents_excludes_a_real_bye() -> None:
    schedule = pl.DataFrame(
        {
            "season": [2026, 2026, 2026],
            "week": [5, 6, 7],
            "home_team": ["KC", "BUF", "KC"],
            "away_team": ["DEN", "MIA", "LAC"],
            "season_type": ["REG", "REG", "REG"],
        }
    )
    # KC plays week 5 (home vs DEN) and week 7 (home vs LAC); week 6 is a
    # real bye (KC appears as neither home nor away -- that row is BUF's
    # own game against MIA, unrelated to KC).
    result = ros_shape.future_week_opponents(schedule, season=2026, team="KC", weeks=[5, 6, 7])
    assert result["week"].to_list() == [5, 7]
    assert result["opponent_team"].to_list() == ["DEN", "LAC"]


def test_allocate_season_consensus_sums_to_the_real_level() -> None:
    weeks_with_opponents = pl.DataFrame({"week": [5, 6, 7], "opponent_team": ["DEN", "LAC", "LV"]})
    frozen_ratings = {
        "WR": pl.DataFrame(
            {
                "defteam": ["DEN", "LAC", "LV"],
                "frozen_adj_epa_allowed": [0.3, -0.1, 0.0],  # DEN = easiest, LAC = hardest
                "frozen_n_plays": [20, 22, 21],
            }
        )
    }
    result = ros_shape.allocate_season_consensus(
        season_consensus_ros_points=30.0,
        position="WR",
        team="KC",
        weeks_with_opponents=weeks_with_opponents,
        frozen_ratings_by_group={"WR": frozen_ratings["WR"]},
    )
    assert result["mean"].sum() == pytest.approx(30.0)
    # Easier matchup (higher adj_epa_allowed) gets a bigger share.
    by_week = dict(zip(result["week"].to_list(), result["mean"].to_list(), strict=True))
    assert by_week[5] > by_week[6]  # DEN (easiest) > LAC (hardest)


def test_allocate_season_consensus_handles_flat_ratings_evenly() -> None:
    """Every real opponent equally tough -> an equal split across weeks
    (10.0 each of 30.0 across 3 weeks) -- proves the allocator doesn't
    introduce spurious variation when there's genuinely none."""
    weeks_with_opponents = pl.DataFrame({"week": [5, 6, 7], "opponent_team": ["A", "B", "C"]})
    frozen_ratings = pl.DataFrame(
        {
            "defteam": ["A", "B", "C"],
            "frozen_adj_epa_allowed": [0.0, 0.0, 0.0],
            "frozen_n_plays": [20, 20, 20],
        }
    )
    result = ros_shape.allocate_season_consensus(
        season_consensus_ros_points=30.0, position="WR", team="KC",
        weeks_with_opponents=weeks_with_opponents, frozen_ratings_by_group={"WR": frozen_ratings},
    )
    for value in result["mean"].to_list():
        assert value == pytest.approx(10.0)
