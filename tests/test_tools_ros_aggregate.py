from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from ffapp.config import DEFAULT_CORRELATION_SETTINGS, RosCalibration
from ffapp.tools import ros_aggregate


def _projections_ros() -> pl.DataFrame:
    rows = []
    for week, mean in [(5, 15.0), (6, 12.0), (7, 18.0)]:
        rows.append(
            {
                "player_id": "p1",
                "season": 2026,
                "week": week,
                "position": "RB",
                "team": "KC",
                "opponent_team": "DEN",
                "mean": mean,
                "q10": mean - 6,
                "q25": mean - 3,
                "q50": mean,
                "q75": mean + 3,
                "q90": mean + 6,
                "is_current_week": week == 5,
            }
        )
    return pl.DataFrame(rows)


def _single_week_row(week: int, mean: float, *, is_current: bool) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "player_id": "p1",
                "season": 2026,
                "week": week,
                "position": "RB",
                "team": "KC",
                "opponent_team": "DEN",
                "mean": mean,
                "q10": mean - 6,
                "q25": mean - 3,
                "q50": mean,
                "q75": mean + 3,
                "q90": mean + 6,
                "is_current_week": is_current,
            }
        ]
    )


def test_aggregate_ros_points_roughly_matches_sum_of_means() -> None:
    """With no injury risk (p_miss=0) and full health (p_active_now=1),
    ros_points should land close to the simple sum of each week's own
    mean (15+12+18=45) -- the Monte Carlo shouldn't systematically bias
    the level away from what the shape function already set."""
    result = ros_aggregate.aggregate_ros(
        _projections_ros(),
        p_active_now={"p1": 1.0},
        p_miss_now={"p1": 0.0},
        position_by_player={"p1": "RB"},
        calibration=RosCalibration(
            within_player_week_correlation={"RB": 0.3}, recovery_prob={"RB": 0.5}
        ),
        playoff_weeks=[7],
        ros_sims=5000,
        default_recovery_prob=0.5,
        correlation=DEFAULT_CORRELATION_SETTINGS,
        rng=np.random.default_rng(0),
    )
    row = result.row(0, named=True)
    assert row["ros_points"] == pytest.approx(45.0, rel=0.05)
    assert row["expected_games"] == pytest.approx(3.0, rel=0.02)


def test_aggregate_ros_playoff_weeks_value_is_separate_column() -> None:
    result = ros_aggregate.aggregate_ros(
        _projections_ros(),
        p_active_now={"p1": 1.0},
        p_miss_now={"p1": 0.0},
        position_by_player={"p1": "RB"},
        calibration=RosCalibration(
            within_player_week_correlation={"RB": 0.3}, recovery_prob={"RB": 0.5}
        ),
        playoff_weeks=[7],
        ros_sims=5000,
        default_recovery_prob=0.5,
        correlation=DEFAULT_CORRELATION_SETTINGS,
        rng=np.random.default_rng(1),
    )
    row = result.row(0, named=True)
    assert row["playoff_weeks_value"] == pytest.approx(18.0, rel=0.1)
    assert row["playoff_weeks_value"] < row["ros_points"]  # never folded into the main total


def test_aggregate_ros_reduces_expected_games_with_real_injury_risk() -> None:
    healthy = ros_aggregate.aggregate_ros(
        _projections_ros(),
        p_active_now={"p1": 1.0},
        p_miss_now={"p1": 0.0},
        position_by_player={"p1": "RB"},
        calibration=RosCalibration(
            within_player_week_correlation={"RB": 0.3}, recovery_prob={"RB": 0.5}
        ),
        playoff_weeks=[],
        ros_sims=5000,
        default_recovery_prob=0.5,
        correlation=DEFAULT_CORRELATION_SETTINGS,
        rng=np.random.default_rng(2),
    )
    risky = ros_aggregate.aggregate_ros(
        _projections_ros(),
        p_active_now={"p1": 1.0},
        p_miss_now={"p1": 0.4},
        position_by_player={"p1": "RB"},
        calibration=RosCalibration(
            within_player_week_correlation={"RB": 0.3}, recovery_prob={"RB": 0.5}
        ),
        playoff_weeks=[],
        ros_sims=5000,
        default_recovery_prob=0.5,
        correlation=DEFAULT_CORRELATION_SETTINGS,
        rng=np.random.default_rng(3),
    )
    assert risky.row(0, named=True)["expected_games"] < healthy.row(0, named=True)["expected_games"]


def test_aggregate_ros_p10_below_p90() -> None:
    result = ros_aggregate.aggregate_ros(
        _projections_ros(),
        p_active_now={"p1": 1.0},
        p_miss_now={"p1": 0.1},
        position_by_player={"p1": "RB"},
        calibration=RosCalibration(
            within_player_week_correlation={"RB": 0.3}, recovery_prob={"RB": 0.5}
        ),
        playoff_weeks=[],
        ros_sims=5000,
        default_recovery_prob=0.5,
        correlation=DEFAULT_CORRELATION_SETTINGS,
        rng=np.random.default_rng(4),
    )
    row = result.row(0, named=True)
    assert row["ros_p10"] < row["ros_p50"] < row["ros_p90"]


def test_aggregate_ros_p_active_now_discounts_future_weeks_not_current_week() -> None:
    """`p_active_now` reflects the anchor week's already-unconditional
    consensus `mean` (`models.predict.project_week`'s own docstring for
    `baseline_b2`/`consensus_b3`) -- multiplying it into the current
    week's contribution would double-count. It must discount future
    weeks (whose `mean` comes from `models.ros_shape`, which never bakes
    in availability) but leave the current week's own contribution
    untouched (`p_miss_now=0.0` here, so the separate hazard-persistence
    mask never fires either, isolating the `p_active_now` effect alone).
    """
    p_active_now = {"p1": 0.5}
    common_kwargs = dict(
        p_active_now=p_active_now,
        p_miss_now={"p1": 0.0},
        position_by_player={"p1": "RB"},
        calibration=RosCalibration(
            within_player_week_correlation={"RB": 0.3}, recovery_prob={"RB": 0.5}
        ),
        playoff_weeks=[],
        ros_sims=5000,
        default_recovery_prob=0.5,
        correlation=DEFAULT_CORRELATION_SETTINGS,
    )

    current_only = ros_aggregate.aggregate_ros(
        _single_week_row(week=5, mean=15.0, is_current=True),
        rng=np.random.default_rng(5),
        **common_kwargs,
    )
    # Current/anchor week: NOT discounted by p_active_now.
    assert current_only.row(0, named=True)["ros_points"] == pytest.approx(15.0, rel=0.05)

    future_only = ros_aggregate.aggregate_ros(
        _single_week_row(week=6, mean=12.0, is_current=False),
        rng=np.random.default_rng(6),
        **common_kwargs,
    )
    # Future week: discounted by p_active_now (0.5 * 12 = 6.0).
    assert future_only.row(0, named=True)["ros_points"] == pytest.approx(6.0, rel=0.05)

    combined = ros_aggregate.aggregate_ros(
        _projections_ros(),
        rng=np.random.default_rng(7),
        **common_kwargs,
    )
    # 15 (current, undiscounted) + 0.5*12 + 0.5*18 (future, discounted) = 30.0 --
    # NOT 0.5*(15+12+18)=22.5, which is what uniformly applying p_active_now to
    # every week (the pre-fix bug) would have produced.
    assert combined.row(0, named=True)["ros_points"] == pytest.approx(30.0, rel=0.05)
