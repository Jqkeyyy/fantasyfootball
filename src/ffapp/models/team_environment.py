"""Decomposed model v2, Stage 1: team environment (SPEC.md §11.4; not a
numbered TASKS.md task -- see docs/design-model-v2-stage1-team-environment.md
for the full design). Predicts a team's own `team_plays` and `pass_rate`
for a week from Vegas lines, pace, and PROE.

`build_team_environment_table` reshapes team-week rows into the shape
`evaluation.backtest.run_walk_forward_backtest` already expects
(`player_id`/`position`/`availability_flag`) -- the same trick
`models.dst.build_dst_table` already uses, so the harness itself is never
touched and nothing else that depends on it (points, dst, availability,
quantiles) can regress.

`pass_attempts`/`rush_attempts` are never modeled directly -- they're
derived (`team_plays * pass_rate` / `team_plays * (1 - pass_rate)`) so the
two always sum to the predicted total exactly, by construction.
"""

from __future__ import annotations

import polars as pl

from ffapp.features.build import lag_shift_join
from ffapp.models.baselines import pooled_rolling_mean

TRAILING_FEATURE_COLUMNS = [
    "proe_ewm_5",
    "neutral_pace_ewm_8",
]
CURRENT_FEATURE_COLUMNS = [
    "implied_team_total",
    "spread",
    "opponent_neutral_pace_ewm_8",
]
FEATURE_COLUMNS = TRAILING_FEATURE_COLUMNS + CURRENT_FEATURE_COLUMNS

TARGET_COLUMNS = ["team_plays", "pass_rate"]


def build_team_environment_table(team_context_features: pl.DataFrame) -> pl.DataFrame:
    """One row per real `(team, season, week)` from `team_context_features`
    (`features.team_context.build_team_context_features`'s own output),
    reshaped for the walk-forward harness: `player_id`/`position`/
    `availability_flag` added (DST-style), `plays` renamed to
    `team_plays`, trailing features lag-shifted one week, current-week
    features (Vegas lines) joined directly.
    """
    targets = team_context_features.select(
        "team", "season", "week", pl.col("plays").alias("team_plays"), "pass_rate"
    )
    shifted = lag_shift_join(targets, team_context_features, "team", TRAILING_FEATURE_COLUMNS)
    with_current = shifted.join(
        team_context_features.select("team", "season", "week", *CURRENT_FEATURE_COLUMNS),
        on=["team", "season", "week"],
        how="left",
    )
    return with_current.with_columns(
        pl.col("team").alias("player_id"),
        pl.lit("TEAM_ENV").alias("position"),
        pl.lit(True).alias("availability_flag"),
    )


def add_team_environment_baselines(table: pl.DataFrame) -> pl.DataFrame:
    """Two baselines per target, following this project's established B0/B2
    pattern (SPEC §12.3) at team grain instead of player grain:

    - `*_league_mean` (B0-equivalent, sanity floor): every team pooled
      together, via `models.baselines.pooled_rolling_mean`.
    - `*_b2_ewm_4` (the real bar, same span as every other B2 in this
      project -- see `models.dst.add_dst_b2_ewm_4`): this team's own
      trailing `ewm_4`, `.shift(1)`'d so the target week's own outcome
      never leaks in.

    Stage 1's model must beat `*_b2_ewm_4` on MAE to be considered
    working -- see the design doc.
    """
    with_league_means = table
    for target_column in TARGET_COLUMNS:
        with_league_means = pooled_rolling_mean(
            with_league_means, "position", target_column, f"{target_column}_league_mean"
        )

    sorted_table = with_league_means.sort(["team", "season", "week"])
    with_b2 = sorted_table
    for target_column in TARGET_COLUMNS:
        with_b2 = with_b2.with_columns(
            pl.col(target_column)
            .ewm_mean(span=4)
            .shift(1)
            .over(["team", "season"])
            .alias(f"{target_column}_b2_ewm_4")
        )
    return with_b2


__all__ = [
    "CURRENT_FEATURE_COLUMNS",
    "FEATURE_COLUMNS",
    "TARGET_COLUMNS",
    "TRAILING_FEATURE_COLUMNS",
    "add_team_environment_baselines",
    "build_team_environment_table",
]
