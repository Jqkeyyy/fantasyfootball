from __future__ import annotations

import polars as pl
import pytest

from ffapp.models import efficiency


def _player_week_features() -> pl.DataFrame:
    """Two WRs on different teams (wr1, wr2 -- two distinct real teams so
    the league-average opponent adjustment Task 3 needs has more than one
    data point to average) and one RB (rb1, wr1's own teammate), two real
    weeks each. def_adj_* values are deliberately non-trivial (not all
    equal, not all zero) so Task 3's tests have real signal to check --
    Tasks 1-2 don't examine these columns at all."""
    return pl.DataFrame(
        {
            "player_id": ["wr1", "wr1", "wr2", "wr2", "rb1", "rb1"],
            "season": [2025] * 6,
            "week": [1, 2, 1, 2, 1, 2],
            "team": ["KC", "KC", "BUF", "BUF", "KC", "KC"],
            "position": ["WR", "WR", "WR", "WR", "RB", "RB"],
            "def_adj_ypt_allowed_wr": [2.0, 2.0, 0.0, 0.0, 2.0, 2.0],
            "def_adj_ypt_allowed_te": [0.0] * 6,
            "def_adj_ypt_allowed_rb_receiving": [0.0] * 6,
            "def_adj_ypt_allowed_rb_rushing": [1.0] * 6,
            "def_adj_ypt_allowed_qb_rushing": [0.0] * 6,
            "def_adj_td_rate_allowed_wr": [0.04, 0.04, 0.0, 0.0, 0.04, 0.04],
            "def_adj_td_rate_allowed_te": [0.0] * 6,
            "def_adj_td_rate_allowed_rb_receiving": [0.0] * 6,
            "def_adj_td_rate_allowed_rb_rushing": [0.02] * 6,
            "def_adj_td_rate_allowed_qb_rushing": [0.0] * 6,
        }
    )


def _player_week_usage() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "player_id": ["wr1", "wr1", "wr2", "wr2", "rb1", "rb1"],
            "season": [2025] * 6,
            "week": [1, 2, 1, 2, 1, 2],
            "targets": [8, 10, 6, 6, 1, 1],
            "carries": [0, 0, 0, 0, 15, 18],
        }
    )


def _player_week_stats() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "player_id": ["wr1", "wr1", "wr2", "wr2", "rb1", "rb1"],
            "season": [2025] * 6,
            "week": [1, 2, 1, 2, 1, 2],
            "receiving_yards": [100, 120, 60, 60, 5, 8],
            "receiving_tds": [1, 1, 0, 0, 0, 0],
            "rushing_yards": [0, 0, 0, 0, 75, 90],
            "rushing_tds": [0, 0, 0, 0, 1, 1],
        }
    )


def _build_table() -> pl.DataFrame:
    return efficiency.build_efficiency_table(
        _player_week_features(), _player_week_usage(), _player_week_stats()
    )


def test_build_efficiency_table_computes_real_outcome_for_a_week_with_touches() -> None:
    result = _build_table()

    wr1_week1 = result.filter((pl.col("player_id") == "wr1") & (pl.col("week") == 1)).row(
        0, named=True
    )
    assert wr1_week1["real_yards_per_target"] == pytest.approx(100 / 8)

    rb1_week1 = result.filter((pl.col("player_id") == "rb1") & (pl.col("week") == 1)).row(
        0, named=True
    )
    assert rb1_week1["real_yards_per_carry"] == pytest.approx(75 / 15)


def test_build_efficiency_table_real_outcome_is_null_with_no_touches_that_week() -> None:
    result = _build_table()

    # wr1 has 0 real carries in both weeks -- real_yards_per_carry must be
    # null, not a fabricated 0.
    wr1_week1 = result.filter((pl.col("player_id") == "wr1") & (pl.col("week") == 1)).row(
        0, named=True
    )
    assert wr1_week1["real_yards_per_carry"] is None


def test_build_efficiency_table_trailing_raw_is_null_in_a_players_first_tracked_week() -> None:
    result = _build_table()

    wr1_week1 = result.filter((pl.col("player_id") == "wr1") & (pl.col("week") == 1)).row(
        0, named=True
    )
    assert wr1_week1["trailing_raw_yards_per_target"] is None
    assert wr1_week1["_n_touches_yards_per_target"] == 0


def test_build_efficiency_table_trailing_raw_is_a_ratio_of_cumulative_sums_not_a_mean_of_weekly_ratios(  # noqa: E501
) -> None:
    result = _build_table()

    wr1_week2 = result.filter((pl.col("player_id") == "wr1") & (pl.col("week") == 2)).row(
        0, named=True
    )
    # Week 1's real cumulative sum (through week 1 only, week 2's own
    # outcome must never leak in): 100 yards / 8 targets = 12.5.
    assert wr1_week2["trailing_raw_yards_per_target"] == pytest.approx(100 / 8)
    assert wr1_week2["_n_touches_yards_per_target"] == 8


def test_build_efficiency_table_league_mean_is_a_ratio_of_pooled_sums_not_a_mean_of_player_ratios(  # noqa: E501
) -> None:
    result = _build_table()

    wr1_week2 = result.filter((pl.col("player_id") == "wr1") & (pl.col("week") == 2)).row(
        0, named=True
    )
    # Pooled across BOTH real WRs' own week-1 real values: (100+60) yards
    # / (8+6) targets = 160/14 -- NOT the naive mean of each player's own
    # week-1 ratio ((100/8 + 60/6)/2 = 11.25), which would wrongly
    # equal-weight wr1's 8-target week and wr2's 6-target week.
    expected = (100 + 60) / (8 + 6)
    assert wr1_week2["league_mean_yards_per_target"] == pytest.approx(expected)
    naive_wrong_value = (100 / 8 + 60 / 6) / 2
    assert wr1_week2["league_mean_yards_per_target"] != pytest.approx(naive_wrong_value)
