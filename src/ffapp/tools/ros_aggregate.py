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

**Every column this function emits is a per-player marginal** (`ros_points`,
`ros_p10`/`ros_p50`/`ros_p90`, `expected_games`, `playoff_weeks_value` --
each summed/quantiled independently over `player_index`, never over a
joint cross-player draw). `sim.week`'s copula machinery
(`sim.persistence.simulate_week_with_common_factor`, reused here for the
within-player week-to-week correlation only) is real and correctly wired,
but its cross-player correlation layer has no effect on any output
column of this function -- worth a reader knowing rather than assuming
these season quantiles encode real cross-player dependence (final review
fix wave, M3).
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
    default_p_active_by_position: dict[str, float] | None = None,
    default_p_miss_by_position: dict[str, float] | None = None,
) -> pl.DataFrame:
    players = sorted(projections_ros["player_id"].unique().to_list())
    n_players = len(players)
    if n_players == 0:
        return pl.DataFrame(
            schema={
                "player_id": pl.String,
                "ros_points": pl.Float64,
                "ros_p10": pl.Float64,
                "ros_p50": pl.Float64,
                "ros_p90": pl.Float64,
                "expected_games": pl.Float64,
                "playoff_weeks_value": pl.Float64,
            }
        )

    weeks = sorted(projections_ros["week"].unique().to_list())
    n_weeks = len(weeks)
    player_index = {pid: i for i, pid in enumerate(players)}

    positions = [position_by_player.get(pid, "") for pid in players]
    rho = np.array([calibration.within_player_week_correlation.get(pos, 0.0) for pos in positions])
    recovery = np.array(
        [calibration.recovery_prob.get(pos, default_recovery_prob) for pos in positions]
    )
    # Real bug found in whole-branch review (C1, the most severe finding in
    # this plan's review history): a player entirely missing from
    # `p_active_now`/`p_miss_now` -- a real, common case for any player thin
    # on recent anchor-week data -- used to silently default to
    # p_active=1.0/p_miss=0.0, i.e. "definitely healthy," the exact opposite
    # of this project's own "honest null, never guess" convention (Task 8's
    # `apply_empirical_error_quantiles`, this same function's own null-mean/
    # null-quantile guard below). Real measured impact on the live
    # rogan-radinator-league board: 299 of 622 ranked players (48%) had no
    # anchor-week row and all 299 got the undiscounted maximum
    # expected_games.
    #
    # The real fallback, per player, in order: (1) their own real per-player
    # value; (2) `default_p_active_by_position`/`default_p_miss_by_position`'s
    # real positional base rate for their position
    # (`sim.injury.positional_base_rate`/
    # `models.baselines.positional_availability_base_rate`, passed in by the
    # caller); (3) if neither is available, this is a genuine "we know
    # nothing" case -- `has_availability_info` below marks that player for
    # full exclusion (zero points, zero expected_games via the same
    # `has_projection` masking the null-mean/null-quantile guard already
    # uses), rather than guessing.
    default_p_active_by_position = default_p_active_by_position or {}
    default_p_miss_by_position = default_p_miss_by_position or {}
    p_active_values: list[float] = []
    p_miss_values: list[float] = []
    has_availability_info: list[bool] = []
    for pid, pos in zip(players, positions, strict=True):
        active_val = p_active_now.get(pid)
        if active_val is None:
            active_val = default_p_active_by_position.get(pos)
        miss_val = p_miss_now.get(pid)
        if miss_val is None:
            miss_val = default_p_miss_by_position.get(pos)
        has_availability_info.append(active_val is not None and miss_val is not None)
        p_active_values.append(active_val if active_val is not None else 1.0)
        p_miss_values.append(miss_val if miss_val is not None else 0.0)

    p_miss = np.tile(np.array(p_miss_values), (n_weeks, 1))
    p_active = np.array(p_active_values)
    has_availability_info_arr = np.array(has_availability_info, dtype=bool)

    # `p_active_now` reflects the anchor week's already-unconditional
    # consensus mean (see module docstring) -- applying it again there
    # would double-count. Build a per-week multiplier that is 1.0 for
    # the current week and the real `p_active` array for every other
    # (future) week, instead of broadcasting `p_active` uniformly.
    is_current_week_by_week = {
        week: bool(projections_ros.filter(pl.col("week") == week)["is_current_week"].any())
        for week in weeks
    }
    p_active_by_week = np.array(
        [np.ones(n_players) if is_current_week_by_week[week] else p_active for week in weeks]
    )  # (n_weeks, n_players)

    available = simulate_availability(
        p_miss, season_sims=ros_sims, recovery_prob=recovery, rng=rng
    )  # (ros_sims, n_weeks, n_players)

    player_factor = rng.standard_normal((ros_sims, n_players))
    totals = np.zeros((ros_sims, n_weeks, n_players))
    # Real bug found in review (not by any unit test), separate from and quieter
    # than the NaN-sort bug above: excluding a null-`mean` (player, week) row
    # from `marginals`/`totals` correctly zeroes its POINTS contribution, but
    # `expected_games` was originally computed over the full `weeks` list with
    # no awareness of which (player, week) cells actually had a real
    # projection -- a player missing one week's projection still had that week
    # counted toward `expected_games`, silently understating their real
    # points-per-game (`ros_points / expected_games`) with no visible flag
    # anywhere. `has_projection` tracks, per week per player, whether that
    # player was actually present in `week_rows` this week (i.e. survived the
    # `mean.is_not_null()` filter) -- the same condition the points exclusion
    # above already uses -- so both quantities are gated identically.
    has_projection = np.zeros((n_weeks, n_players), dtype=bool)
    for week_idx, week in enumerate(weeks):
        # Real gap found live during task 13's own e2e verification, not by any
        # unit test: `models.predict_ros.project_week_range`'s own module
        # docstring documents an honest null `mean`/quantile grid for a real
        # (player, week) whose position/tau bucket has no empirical error-
        # quantile history yet ("never guess, leave it null" -- see that
        # module's own comment). `sim.week.PlayerMarginal`/the copula
        # machinery propagate that single null into a NaN for the player's
        # *entire* season total once summed, which then sorted to the very
        # top of the board (NaN comparisons in the ranking sort put these
        # players at rank 1, ahead of every real projection -- confirmed live:
        # 49 of 622 real rogan-radinator-league players, all with `vor_ros`
        # NaN, occupied ranks 1-49 before this fix). Excluding a null-`mean`
        # (player, week) row from this week's own marginals -- rather than
        # fabricating a specific point value for it -- leaves that player's
        # `totals` at this week's already-zero-initialised default, the same
        # numeric treatment a bye/unavailable week already gets elsewhere in
        # this same aggregation; every other real week for that player is
        # untouched.
        # Fix 3 (final review fix wave): the original guard only checked
        # `mean`, but `baselines.apply_empirical_error_quantiles` (called
        # upstream in predict_ros.py) can return a null q10..q90 value for a
        # position whose empirical error bucket is empty, even when `mean`
        # itself is real. Such a row used to pass the mean-only guard, then
        # `PlayerMarginal.quantile_values` would carry a `None`, propagating
        # into a NaN through the copula machinery -- the exact same
        # severity-1 "NaN sorts to rank 1" bug class the mean-only guard
        # already fixed, through the one door it left open. Widened to
        # require every quantile column non-null too.
        week_rows = (
            projections_ros.filter(pl.col("week") == week)
            .filter(
                pl.col("mean").is_not_null()
                & pl.col("q10").is_not_null()
                & pl.col("q25").is_not_null()
                & pl.col("q50").is_not_null()
                & pl.col("q75").is_not_null()
                & pl.col("q90").is_not_null()
            )
            .sort(
                pl.col("player_id").map_elements(
                    lambda p: player_index.get(p, -1), return_dtype=pl.Int64
                )
            )
        )
        present_ids = week_rows["player_id"].to_list()
        marginals = [
            PlayerMarginal(
                player_id=row["player_id"],
                position=row["position"],
                team=row["team"],
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
            marginals,
            correlation,
            week_sims=ros_sims,
            player_factor=week_factor,
            rho=week_rho,
            rng=rng,
        )
        for local_i, global_i in enumerate(present_idx):
            totals[:, week_idx, global_i] = scores[:, local_i]
            has_projection[week_idx, global_i] = True

    # Fix 1 continued: a player with no real per-player p_active/p_miss AND
    # no positional fallback (`has_availability_info_arr[i] is False`) gets
    # BOTH their totals (points) and their has_projection (games) zeroed
    # here -- an honest, visible "we don't know" for that player's entire
    # season, exactly the same numeric treatment (zero, not fabricated) a
    # null-mean/null-quantile week already gets, just applied across every
    # week for the whole player rather than a single week.
    totals = totals * has_availability_info_arr[None, None, :]
    has_projection = has_projection & has_availability_info_arr[None, :]

    actual = totals * available * p_active_by_week[None, :, :]
    season_totals = actual.sum(axis=1)  # (ros_sims, n_players)
    # Gated by `has_projection` so a (player, week) excluded above for having
    # no real projection doesn't still count as an expected game for that
    # player -- the same real week is now excluded from both the points sum
    # and the games-played sum, never one without the other. Weeks/players
    # that DO have a real projection are completely unaffected (their mask
    # value is True, i.e. a no-op multiplier).
    expected_games = (
        (available * p_active_by_week[None, :, :] * has_projection[None, :, :])
        .sum(axis=1)
        .mean(axis=0)
    )

    playoff_idx = [weeks.index(w) for w in playoff_weeks if w in weeks]
    playoff_value = (
        actual[:, playoff_idx, :].sum(axis=1).mean(axis=0) if playoff_idx else np.zeros(n_players)
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
