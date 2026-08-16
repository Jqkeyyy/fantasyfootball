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
                "player_id": "p1", "season": 2026, "week": week, "position": "RB",
                "team": "KC", "opponent_team": "DEN",
                "mean": mean, "q10": mean - 6, "q25": mean - 3, "q50": mean,
                "q75": mean + 3, "q90": mean + 6, "is_current_week": week == 5,
            }
        )
    return pl.DataFrame(rows)


def test_aggregate_ros_points_roughly_matches_sum_of_means() -> None:
    """With no injury risk (p_miss=0) and full health (p_active_now=1),
    ros_points should land close to the simple sum of each week's own
    mean (15+12+18=45) -- the Monte Carlo shouldn't systematically bias
    the level away from what the shape function already set."""
    result = ros_aggregate.aggregate_ros(
        _projections_ros(),
        p_active_now={"p1": 1.0}, p_miss_now={"p1": 0.0}, position_by_player={"p1": "RB"},
        calibration=RosCalibration(
            within_player_week_correlation={"RB": 0.3}, recovery_prob={"RB": 0.5}
        ),
        playoff_weeks=[7], ros_sims=5000, default_recovery_prob=0.5,
        correlation=DEFAULT_CORRELATION_SETTINGS, rng=np.random.default_rng(0),
    )
    row = result.row(0, named=True)
    assert row["ros_points"] == pytest.approx(45.0, rel=0.05)
    assert row["expected_games"] == pytest.approx(3.0, rel=0.02)


def test_aggregate_ros_playoff_weeks_value_is_separate_column() -> None:
    result = ros_aggregate.aggregate_ros(
        _projections_ros(),
        p_active_now={"p1": 1.0}, p_miss_now={"p1": 0.0}, position_by_player={"p1": "RB"},
        calibration=RosCalibration(
            within_player_week_correlation={"RB": 0.3}, recovery_prob={"RB": 0.5}
        ),
        playoff_weeks=[7], ros_sims=5000, default_recovery_prob=0.5,
        correlation=DEFAULT_CORRELATION_SETTINGS, rng=np.random.default_rng(1),
    )
    row = result.row(0, named=True)
    assert row["playoff_weeks_value"] == pytest.approx(18.0, rel=0.1)
    assert row["playoff_weeks_value"] < row["ros_points"]  # never folded into the main total


def test_aggregate_ros_reduces_expected_games_with_real_injury_risk() -> None:
    healthy = ros_aggregate.aggregate_ros(
        _projections_ros(), p_active_now={"p1": 1.0}, p_miss_now={"p1": 0.0},
        position_by_player={"p1": "RB"},
        calibration=RosCalibration(
            within_player_week_correlation={"RB": 0.3}, recovery_prob={"RB": 0.5}
        ),
        playoff_weeks=[], ros_sims=5000, default_recovery_prob=0.5,
        correlation=DEFAULT_CORRELATION_SETTINGS, rng=np.random.default_rng(2),
    )
    risky = ros_aggregate.aggregate_ros(
        _projections_ros(), p_active_now={"p1": 1.0}, p_miss_now={"p1": 0.4},
        position_by_player={"p1": "RB"},
        calibration=RosCalibration(
            within_player_week_correlation={"RB": 0.3}, recovery_prob={"RB": 0.5}
        ),
        playoff_weeks=[], ros_sims=5000, default_recovery_prob=0.5,
        correlation=DEFAULT_CORRELATION_SETTINGS, rng=np.random.default_rng(3),
    )
    assert risky.row(0, named=True)["expected_games"] < healthy.row(0, named=True)["expected_games"]


def test_aggregate_ros_p10_below_p90() -> None:
    result = ros_aggregate.aggregate_ros(
        _projections_ros(), p_active_now={"p1": 1.0}, p_miss_now={"p1": 0.1},
        position_by_player={"p1": "RB"},
        calibration=RosCalibration(
            within_player_week_correlation={"RB": 0.3}, recovery_prob={"RB": 0.5}
        ),
        playoff_weeks=[], ros_sims=5000, default_recovery_prob=0.5,
        correlation=DEFAULT_CORRELATION_SETTINGS, rng=np.random.default_rng(4),
    )
    row = result.row(0, named=True)
    assert row["ros_p10"] < row["ros_p50"] < row["ros_p90"]
