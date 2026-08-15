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
