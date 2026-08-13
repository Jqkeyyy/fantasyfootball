"""Conditional points model v1 -- Part B of SPEC §11.2's hurdle
architecture (task 1.15; SPEC §11.3).

    E[points] = P(plays) x E[points | plays]

This module builds `E[points | plays]`: one LightGBM regressor per
position (QB, RB, WR, TE -- the only positions `player_week_features`
covers, task 1.9's own `SKILL_POSITIONS`; DST and K get their own,
entirely different models later, SPEC §11.6/§11.7, TASKS.md 2.7+),
trained only on rows where the player actually recorded a snap
(`availability_flag=True`) -- SPEC §11.2's own "Part B... trained only
on rows where the player played," taken literally: a non-played row's
`target` is always exactly 0 by construction (task 1.9), which would
just teach the regressor to predict zero for a huge share of rows it
will never actually be asked to score once Part A (`models.availability`)
handles that separately.

Reuses `evaluation.backtest`'s existing walk-forward harness via
`PointsPredictor`, the same pattern `AvailabilityPredictor` (task 1.14)
already established.

**Features:** the usage/team-context/situation blocks from SPEC §10.2
(`COMMON_FEATURE_COLUMNS`) plus `xfp` and the position's own opponent-
adjustment columns (`features.opponent.POSITION_TO_GROUPS`, reused
directly rather than a second mapping that could drift). `position`
itself is *not* a feature -- it's constant within any one position's own
model, so it would carry zero information (unlike
`models.availability`'s single cross-position classifier, where
`position` is a genuinely informative categorical).

**`xfp` substitution, not the raw column:** SPEC §11.3 names `xfp`
literally, but the raw per-week `xfp` (ffopportunity's own expected
points from *that week's actual* recorded targets/carries) is only
knowable once the game has been played -- using it directly to predict
that same week's own points would violate the as_of contract (CLAUDE.md
rule 2) and is exactly the "feature whose value at inference time
differs in kind from its value in training" failure mode CLAUDE.md's own
standing rules call out. `features.usage` (task 1.6) already resolved
this: no raw `xfp` feature was ever registered, only trailing/lagged
forms (`xfp_per_game_ewm_4`, `xfp_per_game_season_to_date`), which *are*
legitimately known before kickoff. Those are what `xfp`'s own monotonic
constraint applies to here.

**Monotonic constraint sign, verified rather than assumed:** SPEC lists
`def_adj_epa_allowed_*` under "decreasing," but its own parenthetical
("where lower means a tougher defence") and its own explicit "verify
sign at implementation" instruction both point the other way -- a
tougher defence (lower value) should suppress points, i.e. the
relationship is *increasing*, not decreasing. Confirmed empirically
against real 2015-2025 data: positive correlation with actual points at
every position (QB 0.023, RB 0.026, WR 0.018, TE 0.024, ~14k-34k rows
each) -- `_INCREASING_FEATURES` uses "increasing," SPEC's stated
semantic and the real data, not SPEC's own literal bullet placement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl

from ffapp.config import LightGBMSettings
from ffapp.features import opponent
from ffapp.interim.build import SKILL_POSITIONS

COMMON_FEATURE_COLUMNS = [
    "adot_ewm_8",
    "air_yards_share_ewm_4",
    "carry_share_ewm_3",
    "carry_share_ewm_8",
    "cpoe_ewm_4",
    "designed_rush_share_ewm_6",
    "gz_carry_share_ewm_6",
    "pass_attempts_ewm_4",
    "plays_per_game_ewm_5",
    "rush_yards_per_game_ewm_6",
    "rz_touch_share_ewm_6",
    "sack_rate_taken_ewm_4",
    "snap_pct_ewm_3",
    "snap_pct_ewm_8",
    "snap_pct_prior_season",
    "snap_pct_trend",
    "target_share_ewm_3",
    "target_share_ewm_8",
    "target_share_season_to_date",
    "weeks_in_current_role",
    "wopr_ewm_4",
    "xfp_minus_actual_ewm_6",
    "xfp_per_game_ewm_4",
    "xfp_per_game_season_to_date",
    "points_std_std_8",
    "teammate_vacated_target_share",
    "teammate_vacated_carry_share",
    "neutral_pace_ewm_8",
    "ol_continuity_ewm_5",
    "proe_ewm_5",
    "team_epa_off_ewm_8",
    "team_success_off_ewm_8",
    "implied_team_total",
    "spread",
    "is_home",
    "rest_days",
    "is_primetime",
    "week_number",
    "wind_mph",
    "precip_prob",
    "temp_f",
    "is_dome",
    "report_status",
    "practice_participation",
    "weeks_since_return",
]
CATEGORICAL_COLUMNS = ["report_status", "practice_participation"]
TARGET_COLUMN = "target"
AVAILABILITY_COLUMN = "availability_flag"

_OPPONENT_ADJ_METRICS = (
    "def_adj_epa_allowed",
    "def_adj_success_allowed",
    "def_adj_ypt_allowed",
    "def_adj_td_rate_allowed",
)

# SPEC §11.3's own list, mapped onto every real windowed variant of each
# named base signal (all measure the same underlying quantity at
# different trailing windows) -- see module docstring for `xfp`'s
# substitution and `def_adj_epa_allowed_*`'s corrected sign.
_INCREASING_FEATURES = {
    "target_share_ewm_3",
    "target_share_ewm_8",
    "target_share_season_to_date",
    "carry_share_ewm_3",
    "carry_share_ewm_8",
    "snap_pct_ewm_3",
    "snap_pct_ewm_8",
    "rz_touch_share_ewm_6",
    "implied_team_total",
    "xfp_per_game_ewm_4",
    "xfp_per_game_season_to_date",
}


def opponent_feature_columns(position: str) -> list[str]:
    """The real per-position opponent-adjustment columns (task 1.8/1.9),
    reusing `features.opponent.POSITION_TO_GROUPS` directly rather than a
    second, potentially-drifting position -> group mapping."""
    groups = opponent.POSITION_TO_GROUPS[position]
    return [f"{metric}_{group.lower()}" for group in groups for metric in _OPPONENT_ADJ_METRICS]


def _epa_allowed_columns(position: str) -> set[str]:
    return {
        f"def_adj_epa_allowed_{group.lower()}" for group in opponent.POSITION_TO_GROUPS[position]
    }


def feature_columns(position: str) -> list[str]:
    """SPEC §11.3's full feature list for one position's own model."""
    return COMMON_FEATURE_COLUMNS + opponent_feature_columns(position)


def monotone_constraints(position: str) -> list[int]:
    """LightGBM's own `monotone_constraints` vector, aligned 1:1 with
    `feature_columns(position)`'s order -- `1` (increasing) for
    `_INCREASING_FEATURES` and this position's own `def_adj_epa_allowed_*`
    columns, `0` (unconstrained) everywhere else. SPEC names no
    "decreasing" feature once `def_adj_epa_allowed_*`'s sign is
    corrected (see module docstring)."""
    increasing = _INCREASING_FEATURES | _epa_allowed_columns(position)
    return [1 if column in increasing else 0 for column in feature_columns(position)]


def _to_feature_frame(rows: pl.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Polars -> pandas only at this fit/predict boundary (CLAUDE.md's
    own convention), same as `models.availability`."""
    pdf = rows.select(columns).to_pandas()
    for column in CATEGORICAL_COLUMNS:
        if column in pdf.columns:
            pdf[column] = pdf[column].astype("category")
    return pdf


@dataclass
class FittedPointsModels:
    boosters: dict[str, lgb.LGBMRegressor]


def fit_points_model(
    train_rows: pl.DataFrame, *, lightgbm_params: LightGBMSettings
) -> FittedPointsModels:
    """SPEC §11.3: one regressor per position, fit only on rows where the
    player actually played (see module docstring). A position with zero
    real played rows in `train_rows` (a genuine early-backtest-week edge
    case) is simply skipped -- `predict_points` leaves those rows
    honestly null rather than guessing.
    """
    played = train_rows.filter(pl.col(AVAILABILITY_COLUMN))
    boosters: dict[str, lgb.LGBMRegressor] = {}
    for position in SKILL_POSITIONS:
        position_rows = played.filter(pl.col("position") == position)
        if position_rows.is_empty():
            continue
        columns = feature_columns(position)
        booster = lgb.LGBMRegressor(
            n_estimators=lightgbm_params.n_estimators,
            learning_rate=lightgbm_params.learning_rate,
            num_leaves=lightgbm_params.num_leaves,
            min_child_samples=lightgbm_params.min_child_samples,
            subsample=lightgbm_params.subsample,
            colsample_bytree=lightgbm_params.colsample_bytree,
            reg_lambda=lightgbm_params.reg_lambda,
            monotone_constraints=monotone_constraints(position),
            verbosity=-1,
        )
        booster.fit(
            _to_feature_frame(position_rows, columns),
            position_rows[TARGET_COLUMN].to_numpy(),
            categorical_feature=[c for c in CATEGORICAL_COLUMNS if c in columns],
        )
        boosters[position] = booster
    return FittedPointsModels(boosters=boosters)


def predict_points(model: FittedPointsModels, rows: pl.DataFrame) -> pl.Series:
    """`E[points | plays]` for every row, regardless of that row's own
    real `availability_flag` -- the conditional expectation answers "if
    this player plays," not only rows where they happened to (combining
    it with Part A's `p_active` into an unconditional expectation is a
    later task, SPEC §11.2's own hurdle formula). A row whose position
    has no fitted booster (never seen in `train_rows`) predicts honestly
    null, not a guessed value.
    """
    predictions = np.full(rows.height, np.nan, dtype=float)
    row_positions = rows["position"].to_numpy()
    for position, booster in model.boosters.items():
        mask = row_positions == position
        if not mask.any():
            continue
        columns = feature_columns(position)
        subset = rows.filter(pl.Series(mask))
        predictions[mask] = np.asarray(booster.predict(_to_feature_frame(subset, columns)))
    return pl.Series("prediction", predictions, dtype=pl.Float64).fill_nan(None)


class PointsPredictor:
    """A `evaluation.backtest.Predictor` wrapping
    `fit_points_model`/`predict_points` -- exercised via the same
    `run_walk_forward_backtest` harness every other predictor uses."""

    name = "conditional_points_lightgbm"

    def __init__(self, lightgbm_params: LightGBMSettings) -> None:
        self.lightgbm_params = lightgbm_params

    def fit(self, train_rows: pl.DataFrame) -> Any:
        return fit_points_model(train_rows, lightgbm_params=self.lightgbm_params)

    def predict(self, fitted: Any, target_rows: pl.DataFrame) -> pl.Series:
        return predict_points(fitted, target_rows)


__all__ = [
    "AVAILABILITY_COLUMN",
    "CATEGORICAL_COLUMNS",
    "COMMON_FEATURE_COLUMNS",
    "TARGET_COLUMN",
    "FittedPointsModels",
    "PointsPredictor",
    "feature_columns",
    "fit_points_model",
    "monotone_constraints",
    "opponent_feature_columns",
    "predict_points",
]
