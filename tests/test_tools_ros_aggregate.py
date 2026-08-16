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


def _projections_ros_with_null_week() -> pl.DataFrame:
    """Same 3-week shape as `_projections_ros`, but week 6's `mean`/quantile
    grid is honestly null -- matching how `models.predict_ros
    .project_week_range` actually leaves a real (player, week) null when no
    empirical error-quantile bucket exists yet for that position/tau (that
    module's own "never guess, leave it null" comment). Regression fixture
    for the real bugs found in Task 13's own verification: a null week used
    to propagate into a NaN `ros_points` for the player's entire season
    (fixed by excluding the row from that week's own marginals), and
    `expected_games` used to still count the null week as a real game even
    after that fix (fixed separately, see the test below)."""
    rows = []
    for week, mean in [(5, 15.0), (6, None), (7, 18.0)]:
        if mean is None:
            rows.append(
                {
                    "player_id": "p1",
                    "season": 2026,
                    "week": week,
                    "position": "RB",
                    "team": "KC",
                    "opponent_team": "DEN",
                    "mean": None,
                    "q10": None,
                    "q25": None,
                    "q50": None,
                    "q75": None,
                    "q90": None,
                    "is_current_week": False,
                }
            )
        else:
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
    return pl.DataFrame(
        rows,
        schema={
            "player_id": pl.String,
            "season": pl.Int64,
            "week": pl.Int64,
            "position": pl.String,
            "team": pl.String,
            "opponent_team": pl.String,
            "mean": pl.Float64,
            "q10": pl.Float64,
            "q25": pl.Float64,
            "q50": pl.Float64,
            "q75": pl.Float64,
            "q90": pl.Float64,
            "is_current_week": pl.Boolean,
        },
    )


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


def test_aggregate_ros_null_week_excluded_from_points_and_expected_games() -> None:
    """Regression test for two real bugs found live during Task 13's own
    verification, both in `aggregate_ros`. The most severe: a single
    null-`mean` (player, week) -- the honest "no empirical error-quantile
    bucket yet" case `models.predict_ros.project_week_range` documents
    producing -- used to propagate into a NaN `ros_points` for the
    player's ENTIRE season, which then sorted to the very top of the real
    board (49 of 622 real players) ahead of every real, well-formed
    projection, since NaN sorts before real numbers by default. Fixed by
    excluding that one week's row from its own week's marginals rather
    than fabricating a value. The quieter follow-up bug that first fix
    introduced: `expected_games` was still computed over the full week
    list, so the excluded week still silently counted as a real game even
    though it contributed zero points -- fixed by gating `expected_games`
    with the same per-(week, player) "had a real projection" mask the
    points exclusion already uses.
    """
    common_kwargs = dict(
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
    )

    result = ros_aggregate.aggregate_ros(
        _projections_ros_with_null_week(),
        rng=np.random.default_rng(8),
        **common_kwargs,
    )
    row = result.row(0, named=True)

    # (a) Locks in the original, severe bug's fix: a real, finite ros_points
    # (and every quantile), not NaN -- a NaN here is exactly what sorted to
    # rank 1 on the real board before the fix.
    assert not np.isnan(row["ros_points"])
    assert not np.isnan(row["ros_p10"])
    assert not np.isnan(row["ros_p50"])
    assert not np.isnan(row["ros_p90"])
    # (c) Sorts correctly: a real, bounded value strictly between 0 and the
    # fully-populated 3-week total (45) -- never the NaN-sorts-first failure
    # mode the reviewer's own repro described, and never a fabricated value
    # for the missing week either. Only the two real weeks (15 + 18 = 33)
    # contribute.
    assert 0.0 < row["ros_points"] < 45.0
    assert row["ros_points"] == pytest.approx(33.0, rel=0.05)

    # (b) Locks in the residual-defect fix: expected_games must reflect only
    # the two real weeks with an actual projection (2.0), not all three
    # weeks present in the input (3.0) -- the null week must not silently
    # count as a game just because it's zeroed out of the points sum.
    assert row["expected_games"] == pytest.approx(2.0, rel=0.05)

    fully_populated = ros_aggregate.aggregate_ros(
        _projections_ros(),
        rng=np.random.default_rng(9),
        **common_kwargs,
    )
    # Same real per-week means (15/12/18) and the same p_active_now/p_miss_now,
    # differing only in whether week 6 has a real projection -- a measurably
    # lower expected_games than the fully-populated case proves the null week
    # is excluded from expected_games, not just from points.
    assert row["expected_games"] < fully_populated.row(0, named=True)["expected_games"]
    assert fully_populated.row(0, named=True)["expected_games"] == pytest.approx(3.0, rel=0.02)
