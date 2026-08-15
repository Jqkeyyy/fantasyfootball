from __future__ import annotations

import polars as pl
import pytest

from ffapp.models import opportunity


def _player_week_features() -> pl.DataFrame:
    """One team (KC), one real week, three players at three different
    positions -- WR (in PASS_CATCHERS_AND_RB, not RB_QB), RB (in both),
    QB (in RB_QB, not PASS_CATCHERS_AND_RB). Share values are deliberately
    non-null/non-zero for every player regardless of position eligibility
    (matching the real data: features.usage's windowing computes shares for
    every row, not just eligible positions) -- this is what exercises the
    position-gating logic, not just null propagation."""
    return pl.DataFrame(
        {
            "player_id": ["wr1", "rb1", "qb1"],
            "season": [2025, 2025, 2025],
            "week": [1, 1, 1],
            "team": ["KC", "KC", "KC"],
            "position": ["WR", "RB", "QB"],
            "target_share_ewm_3": [0.25, 0.10, 0.01],
            "carry_share_ewm_3": [0.02, 0.60, 0.08],
            "rz_touch_share_ewm_6": [0.15, 0.30, 0.05],
        }
    )


def _player_week_usage() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "player_id": ["wr1", "rb1", "qb1"],
            "season": [2025, 2025, 2025],
            "week": [1, 1, 1],
            "targets": [8, 3, 0],
            "carries": [1, 15, 4],
            "rz_targets": [1, 0, 0],
            "rz_carries": [0, 3, 1],
        }
    )


def _stage1_predictions() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "team": ["KC"],
            "season": [2025],
            "week": [1],
            "predicted_team_plays": [55.0],
            "predicted_pass_attempts": [30.0],
            "predicted_rush_attempts": [25.0],
        }
    )


def test_build_opportunity_table_computes_expected_targets_for_eligible_positions() -> None:
    result = opportunity.build_opportunity_table(
        _player_week_features(), _player_week_usage(), _stage1_predictions()
    )

    wr = result.filter(pl.col("player_id") == "wr1").row(0, named=True)
    rb = result.filter(pl.col("player_id") == "rb1").row(0, named=True)
    assert wr["expected_targets"] == pytest.approx(0.25 * 30.0)  # WR is in PASS_CATCHERS_AND_RB
    assert rb["expected_targets"] == pytest.approx(0.10 * 30.0)  # RB is in PASS_CATCHERS_AND_RB


def test_build_opportunity_table_nulls_expected_targets_for_ineligible_position() -> None:
    result = opportunity.build_opportunity_table(
        _player_week_features(), _player_week_usage(), _stage1_predictions()
    )

    qb = result.filter(pl.col("player_id") == "qb1").row(0, named=True)
    # QB is NOT in PASS_CATCHERS_AND_RB -- must be null even though
    # target_share_ewm_3 itself has a real (non-null) value of 0.01.
    assert qb["expected_targets"] is None


def test_build_opportunity_table_computes_expected_carries_for_eligible_positions() -> None:
    result = opportunity.build_opportunity_table(
        _player_week_features(), _player_week_usage(), _stage1_predictions()
    )

    rb = result.filter(pl.col("player_id") == "rb1").row(0, named=True)
    qb = result.filter(pl.col("player_id") == "qb1").row(0, named=True)
    assert rb["expected_carries"] == pytest.approx(0.60 * 25.0)  # RB is in RB_QB
    assert qb["expected_carries"] == pytest.approx(0.08 * 25.0)  # QB is in RB_QB


def test_build_opportunity_table_nulls_expected_carries_for_ineligible_position() -> None:
    result = opportunity.build_opportunity_table(
        _player_week_features(), _player_week_usage(), _stage1_predictions()
    )

    wr = result.filter(pl.col("player_id") == "wr1").row(0, named=True)
    # WR is NOT in RB_QB -- must be null even though carry_share_ewm_3
    # itself has a real (non-null) value of 0.02.
    assert wr["expected_carries"] is None


def test_build_opportunity_table_computes_expected_rz_touches_for_eligible_positions() -> None:
    result = opportunity.build_opportunity_table(
        _player_week_features(), _player_week_usage(), _stage1_predictions()
    )

    wr = result.filter(pl.col("player_id") == "wr1").row(0, named=True)
    qb = result.filter(pl.col("player_id") == "qb1").row(0, named=True)
    assert wr["expected_rz_touches"] == pytest.approx(0.15 * 55.0)  # WR is in PASS_CATCHERS_AND_RB
    assert qb["expected_rz_touches"] is None  # QB is NOT in PASS_CATCHERS_AND_RB


def test_build_opportunity_table_derives_real_rz_touches_from_targets_and_carries() -> None:
    result = opportunity.build_opportunity_table(
        _player_week_features(), _player_week_usage(), _stage1_predictions()
    )

    rb = result.filter(pl.col("player_id") == "rb1").row(0, named=True)
    assert rb["rz_touches"] == 3  # rz_targets=0 + rz_carries=3, real counts carried through


def test_build_opportunity_table_carries_real_target_counts_unmodified() -> None:
    result = opportunity.build_opportunity_table(
        _player_week_features(), _player_week_usage(), _stage1_predictions()
    )

    wr = result.filter(pl.col("player_id") == "wr1").row(0, named=True)
    assert wr["targets"] == 8
    assert wr["carries"] == 1


def _baseline_fixture_table() -> pl.DataFrame:
    """Two WRs, two weeks each -- enough to exercise both the pooled
    league-mean (two players at the same position, same week) and the
    per-player trailing ewm_4 (two real weeks for the same player, so the
    shift is provably not leaking week 2's own outcome)."""
    return pl.DataFrame(
        {
            "player_id": ["wrA", "wrA", "wrB", "wrB"],
            "season": [2025, 2025, 2025, 2025],
            "week": [1, 2, 1, 2],
            "position": ["WR", "WR", "WR", "WR"],
            "targets": [6, 10, 8, 9],
            "carries": [0, 0, 0, 0],
            "rz_touches": [1, 2, 0, 1],
        }
    )


def test_add_opportunity_baselines_adds_all_six_columns() -> None:
    result = opportunity.add_opportunity_baselines(_baseline_fixture_table())

    for target_column in opportunity.TARGET_COLUMNS:
        assert f"{target_column}_league_mean" in result.columns
        assert f"{target_column}_b2_ewm_4" in result.columns


def test_add_opportunity_baselines_b2_never_leaks_the_target_week() -> None:
    result = opportunity.add_opportunity_baselines(_baseline_fixture_table())

    week2 = result.filter((pl.col("player_id") == "wrA") & (pl.col("week") == 2)).row(
        0, named=True
    )
    # week 2's b2 baseline must be built only from week 1's real outcome (6),
    # never week 2's own (10) -- with a single prior week, ewm_4 of one
    # point equals that point exactly.
    assert week2["targets_b2_ewm_4"] == pytest.approx(6.0)


def test_add_opportunity_baselines_league_mean_pools_across_players_at_the_position() -> None:
    result = opportunity.add_opportunity_baselines(_baseline_fixture_table())

    week2_wrA = result.filter((pl.col("player_id") == "wrA") & (pl.col("week") == 2)).row(
        0, named=True
    )
    # week 2's pooled mean is week 1's real values across BOTH WRs: (6+8)/2 = 7
    assert week2_wrA["targets_league_mean"] == pytest.approx(7.0)


def test_to_predictions_frame_reshapes_to_one_row_per_predictor() -> None:
    table = pl.DataFrame(
        {
            "player_id": ["wr1"],
            "season": [2025],
            "week": [1],
            "position": ["WR"],
            "team": ["KC"],
            "targets": [8],
            "expected_targets": [7.5],
            "targets_b2_ewm_4": [6.0],
            "targets_league_mean": [7.0],
        }
    )

    result = opportunity.to_predictions_frame(
        table,
        real_column="targets",
        composition_column="expected_targets",
        trailing_raw_column="targets_b2_ewm_4",
        league_mean_column="targets_league_mean",
    )

    assert result.height == 3  # one row per predictor
    by_predictor = {row["predictor"]: row for row in result.to_dicts()}
    assert by_predictor["opportunity_composition"]["prediction"] == pytest.approx(7.5)
    assert by_predictor["trailing_raw"]["prediction"] == pytest.approx(6.0)
    assert by_predictor["league_mean"]["prediction"] == pytest.approx(7.0)
    for row in result.to_dicts():
        assert row["target"] == 8  # the real outcome, same on every predictor's own row
