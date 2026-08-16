"""ROS Monte Carlo aggregation (`SPEC-ADDENDUM-04.md` §D.2). Composes,
per real player: task 2.2's cross-player correlated weekly sampling
(`sim.week`, via `sim.persistence.simulate_week_with_common_factor` for
the added within-player layer -- requirement 2), task 2.3/2.4's
persistent-duration injury sampling (`sim.season.simulate_availability`
-- requirement 3), and `models.predict_ros`'s own already-shaped weekly
quantile grids (current week unchanged, future weeks per
`models.ros_shape`). `p_play[w] = p_active_now x P(available in week w
| hazard persistence)`, applied multiplicatively at this aggregation
stage only -- SPEC-ADDENDUM-04.md §D.2's own literal pseudocode, and the
one place in this pipeline availability is applied (never baked into the
shape function, which would double-count).

**`p_active_now` is applied to future weeks only, never to the current/
anchor week.** `projections_ros`'s current-week row (`is_current_week`,
Task 8's own output) carries a `mean` from `models.predict.project_week`
with `projection_source in {"baseline_b2", "consensus_b3"}` -- that
function's own docstring is explicit that this `mean` "is already an
unconditional quantity by construction (no `p_active` re-multiplication
... that would double-count the same playing-time signal ... already
reflects)". `p_active_now` is exactly that same Part-A model
(`models.availability.predict_p_active`) evaluated on the anchor week, so
multiplying it into the current week's already-unconditional mean here
would double-count. Future weeks are unaffected -- their `mean` comes
from `models.ros_shape`, which never bakes in availability (Task 7), so
`p_active_now` is correctly applied there exactly once. The hazard-driven
`available` mask (from `p_miss_now` via `simulate_availability`) is a
genuinely different signal (multi-week persistence, no B3 equivalent)
and is applied to every week including the current one, unchanged.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from ffapp.config import CorrelationSettings, RosCalibration
from ffapp.sim.persistence import simulate_week_with_common_factor
from ffapp.sim.season import simulate_availability
from ffapp.sim.week import PlayerMarginal

_QUANTILE_ALPHAS = (0.10, 0.25, 0.50, 0.75, 0.90)
_Q_COLUMNS = {0.10: "q10", 0.25: "q25", 0.50: "q50", 0.75: "q75", 0.90: "q90"}


def aggregate_ros(
    projections_ros: pl.DataFrame,
    p_active_now: dict[str, float],
    p_miss_now: dict[str, float],
    position_by_player: dict[str, str],
    calibration: RosCalibration,
    *,
    playoff_weeks: list[int],
    ros_sims: int,
    default_recovery_prob: float,
    correlation: CorrelationSettings,
    rng: np.random.Generator,
) -> pl.DataFrame:
    players = sorted(projections_ros["player_id"].unique().to_list())
    n_players = len(players)
    if n_players == 0:
        return pl.DataFrame(
            schema={
                "player_id": pl.String, "ros_points": pl.Float64, "ros_p10": pl.Float64,
                "ros_p50": pl.Float64, "ros_p90": pl.Float64, "expected_games": pl.Float64,
                "playoff_weeks_value": pl.Float64,
            }
        )

    weeks = sorted(projections_ros["week"].unique().to_list())
    n_weeks = len(weeks)
    player_index = {pid: i for i, pid in enumerate(players)}

    positions = [position_by_player.get(pid, "") for pid in players]
    rho = np.array(
        [calibration.within_player_week_correlation.get(pos, 0.0) for pos in positions]
    )
    recovery = np.array(
        [calibration.recovery_prob.get(pos, default_recovery_prob) for pos in positions]
    )
    p_miss = np.tile(
        np.array([p_miss_now.get(pid, 0.0) for pid in players]), (n_weeks, 1)
    )
    p_active = np.array([p_active_now.get(pid, 1.0) for pid in players])

    # `p_active_now` reflects the anchor week's already-unconditional
    # consensus mean (see module docstring) -- applying it again there
    # would double-count. Build a per-week multiplier that is 1.0 for
    # the current week and the real `p_active` array for every other
    # (future) week, instead of broadcasting `p_active` uniformly.
    is_current_week_by_week = {
        week: bool(
            projections_ros.filter(pl.col("week") == week)["is_current_week"].any()
        )
        for week in weeks
    }
    p_active_by_week = np.array(
        [
            np.ones(n_players) if is_current_week_by_week[week] else p_active
            for week in weeks
        ]
    )  # (n_weeks, n_players)

    available = simulate_availability(
        p_miss, season_sims=ros_sims, recovery_prob=recovery, rng=rng
    )  # (ros_sims, n_weeks, n_players)

    player_factor = rng.standard_normal((ros_sims, n_players))
    totals = np.zeros((ros_sims, n_weeks, n_players))
    for week_idx, week in enumerate(weeks):
        week_rows = projections_ros.filter(pl.col("week") == week).sort(
            pl.col("player_id").map_elements(
                lambda p: player_index.get(p, -1), return_dtype=pl.Int64
            )
        )
        present_ids = week_rows["player_id"].to_list()
        marginals = [
            PlayerMarginal(
                player_id=row["player_id"], position=row["position"], team=row["team"],
                opponent_team=row.get("opponent_team"),
                alphas=list(_QUANTILE_ALPHAS),
                quantile_values=[row[_Q_COLUMNS[a]] for a in _QUANTILE_ALPHAS],
            )
            for row in week_rows.iter_rows(named=True)
        ]
        present_idx = [player_index[pid] for pid in present_ids]
        week_rho = rho[present_idx]
        week_factor = player_factor[:, present_idx]
        scores = simulate_week_with_common_factor(
            marginals, correlation, week_sims=ros_sims, player_factor=week_factor,
            rho=week_rho, rng=rng,
        )
        for local_i, global_i in enumerate(present_idx):
            totals[:, week_idx, global_i] = scores[:, local_i]

    actual = totals * available * p_active_by_week[None, :, :]
    season_totals = actual.sum(axis=1)  # (ros_sims, n_players)
    expected_games = (available * p_active_by_week[None, :, :]).sum(axis=1).mean(axis=0)

    playoff_idx = [weeks.index(w) for w in playoff_weeks if w in weeks]
    playoff_value = (
        actual[:, playoff_idx, :].sum(axis=1).mean(axis=0)
        if playoff_idx
        else np.zeros(n_players)
    )

    ros_points = season_totals.mean(axis=0)
    ros_p10 = np.quantile(season_totals, 0.10, axis=0)
    ros_p50 = np.quantile(season_totals, 0.50, axis=0)
    ros_p90 = np.quantile(season_totals, 0.90, axis=0)

    return pl.DataFrame(
        {
            "player_id": players,
            "ros_points": ros_points.tolist(),
            "ros_p10": ros_p10.tolist(),
            "ros_p50": ros_p50.tolist(),
            "ros_p90": ros_p90.tolist(),
            "expected_games": expected_games.tolist(),
            "playoff_weeks_value": playoff_value.tolist(),
        }
    )


__all__ = ["aggregate_ros"]
