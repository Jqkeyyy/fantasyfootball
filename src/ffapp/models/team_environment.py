"""Decomposed model v2, Stage 1: team environment (SPEC.md §11.4; not a
numbered TASKS.md task -- see docs/design-model-v2-stage1-team-environment.md
for the full design). Predicts a team's own `team_plays` and `pass_rate`
for a week from Vegas lines, pace, and PROE.

`build_team_environment_table` reshapes team-week rows into the shape
`evaluation.backtest.run_walk_forward_backtest` already expects
(`player_id`/`position`/`availability_flag`) -- the same trick
`models.dst.build_dst_table` already uses, so the harness itself is never
touched and nothing else that depends on it (points, dst, availability,
quantiles) can regress. Trailing features (`TRAILING_FEATURE_COLUMNS`) are
lagged with a plain positional `.shift(1).over(["team", "season"])`, not
`features.build.lag_shift_join`'s week-arithmetic shift (`week +
lag_weeks`) -- matching how this same table's baselines
(`add_team_environment_baselines`) and `features.team_context.
add_opponent_pace`'s internal lag already work. Week-arithmetic lag leaves
season-openers *and* bye-week returns null (no row exists at `week -
lag_weeks`), while the baselines on those same rows stay populated under a
positional shift -- an inconsistency inside this table's own values that a
mixed lag strategy would otherwise introduce.

`pass_attempts`/`rush_attempts` are never modeled directly -- they're
derived (`team_plays * pass_rate` / `team_plays * (1 - pass_rate)`) so the
two always sum to the predicted total exactly, by construction.

**Monotonic constraint sign, verified rather than assumed** (same
precedent `models.points`'s own module docstring already established --
"SPEC lists def_adj_epa_allowed_* under 'decreasing'... confirmed
empirically... not SPEC's own literal bullet placement"): the design doc
states `neutral_pace_ewm_8`/`opponent_neutral_pace_ewm_8` as *increasing*
on `team_plays`, but `neutral_pace_sec` (`interim.build.add_neutral_pace`)
is seconds-per-play, not plays-per-second -- a *higher* value means a
*slower* game, i.e. *fewer* plays, the opposite sign. Confirmed empirically
against real cached 2015-2025 data: `corr(own lagged neutral_pace_ewm_8,
plays) = -0.1291` (n=5702), `corr(opponent_neutral_pace_ewm_8, plays) =
-0.0543` (n=5468) -- both negative, so both are registered as *decreasing*
(`_DECREASING_FEATURES`), not increasing. `proe_ewm_5` on `pass_rate` was
checked the same way and confirmed correctly increasing: `corr(own lagged
proe_ewm_5, pass_rate) = +0.2524` (n=5202) -- left as-is in
`_INCREASING_FEATURES`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import lightgbm as lgb
import polars as pl

from ffapp.config import LightGBMSettings
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

# Signs verified empirically against real data, not trusted from the
# design doc's literal wording -- see module docstring.
_INCREASING_FEATURES = {
    "team_plays": set(),
    "pass_rate": {"proe_ewm_5"},
}
_DECREASING_FEATURES = {
    "team_plays": {"neutral_pace_ewm_8", "opponent_neutral_pace_ewm_8"},
    "pass_rate": set(),
}


def monotone_constraints(target_column: str) -> list[int]:
    """LightGBM's own `monotone_constraints` vector, aligned 1:1 with
    `FEATURE_COLUMNS`'s order -- `1` (increasing) for this target's own
    `_INCREASING_FEATURES` entry, `-1` (decreasing, LightGBM's own
    convention: the target should *decrease* as the feature increases)
    for `_DECREASING_FEATURES`, `0` (unconstrained) everywhere else.
    Mirrors `models.points.monotone_constraints`'s own shape. See module
    docstring for the real correlations behind each sign."""
    increasing = _INCREASING_FEATURES[target_column]
    decreasing = _DECREASING_FEATURES[target_column]
    return [
        1 if column in increasing else -1 if column in decreasing else 0
        for column in FEATURE_COLUMNS
    ]


def build_team_environment_table(team_context_features: pl.DataFrame) -> pl.DataFrame:
    """One row per real `(team, season, week)` from `team_context_features`
    (`features.team_context.build_team_context_features`'s own output),
    reshaped for the walk-forward harness: `player_id`/`position`/
    `availability_flag` added (DST-style), `plays` renamed to
    `team_plays`, trailing features lag-shifted one week (a positional
    `.shift(1).over(["team", "season"])`, not week-arithmetic -- see
    module docstring), current-week features (Vegas lines) joined
    directly.
    """
    targets = team_context_features.select(
        "team", "season", "week", pl.col("plays").alias("team_plays"), "pass_rate"
    )
    sorted_features = team_context_features.sort(["team", "season", "week"]).with_columns(
        [
            pl.col(column).shift(1).over(["team", "season"]).alias(column)
            for column in TRAILING_FEATURE_COLUMNS
        ]
    )
    shifted = targets.join(
        sorted_features.select("team", "season", "week", *TRAILING_FEATURE_COLUMNS),
        on=["team", "season", "week"],
        how="left",
    )
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


def to_feature_frame(rows: pl.DataFrame) -> Any:
    """Polars -> pandas only at this fit/predict boundary (CLAUDE.md's own
    convention, same as `models.points`/`models.dst`). No categorical
    columns here -- every Stage 1 feature is numeric."""
    return rows.select(FEATURE_COLUMNS).to_pandas()


@dataclass
class FittedTeamEnvironmentModel:
    booster: lgb.LGBMRegressor
    target_column: str


def fit_team_environment_model(
    train_rows: pl.DataFrame, *, target_column: str, lightgbm_params: LightGBMSettings
) -> FittedTeamEnvironmentModel:
    booster = lgb.LGBMRegressor(
        n_estimators=lightgbm_params.n_estimators,
        learning_rate=lightgbm_params.learning_rate,
        num_leaves=lightgbm_params.num_leaves,
        min_child_samples=lightgbm_params.min_child_samples,
        subsample=lightgbm_params.subsample,
        colsample_bytree=lightgbm_params.colsample_bytree,
        reg_lambda=lightgbm_params.reg_lambda,
        monotone_constraints=monotone_constraints(target_column),
        verbosity=-1,
    )
    booster.fit(to_feature_frame(train_rows), train_rows[target_column].to_numpy())
    return FittedTeamEnvironmentModel(booster=booster, target_column=target_column)


def predict_team_environment(model: FittedTeamEnvironmentModel, rows: pl.DataFrame) -> pl.Series:
    predictions = model.booster.predict(to_feature_frame(rows))
    return pl.Series("prediction", predictions, dtype=pl.Float64)


def derive_attempts(team_plays: pl.Series, pass_rate: pl.Series) -> tuple[pl.Series, pl.Series]:
    """`pass_attempts`/`rush_attempts` are never modeled directly -- always
    derived from the two predicted quantities, so they sum to
    `team_plays` exactly by construction (see module docstring)."""
    pass_attempts = (team_plays * pass_rate).rename("pass_attempts")
    rush_attempts = (team_plays * (1 - pass_rate)).rename("rush_attempts")
    return pass_attempts, rush_attempts


class TeamEnvironmentPredictor:
    """A `evaluation.backtest.Predictor` wrapping
    `fit_team_environment_model`/`predict_team_environment`, exercised via
    the same `run_walk_forward_backtest` harness every other predictor in
    this project uses -- construct one per target (`team_plays`,
    `pass_rate`), each with its own `name`."""

    def __init__(self, *, name: str, target_column: str, lightgbm_params: LightGBMSettings) -> None:
        self.name = name
        self.target_column = target_column
        self.lightgbm_params = lightgbm_params

    def fit(self, train_rows: pl.DataFrame) -> Any:
        return fit_team_environment_model(
            train_rows, target_column=self.target_column, lightgbm_params=self.lightgbm_params
        )

    def predict(self, fitted: Any, target_rows: pl.DataFrame) -> pl.Series:
        return predict_team_environment(fitted, target_rows)


__all__ = [
    "CURRENT_FEATURE_COLUMNS",
    "FEATURE_COLUMNS",
    "TARGET_COLUMNS",
    "TRAILING_FEATURE_COLUMNS",
    "FittedTeamEnvironmentModel",
    "TeamEnvironmentPredictor",
    "add_team_environment_baselines",
    "build_team_environment_table",
    "derive_attempts",
    "fit_team_environment_model",
    "monotone_constraints",
    "predict_team_environment",
    "to_feature_frame",
]
