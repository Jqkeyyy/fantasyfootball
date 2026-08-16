"""Anchored residual model (task 1.20; `SPEC-ADDENDUM-04.md` §B,
supersedes SPEC §11.3 alongside task 1.15).

Reopened by ADDENDUM-04 §A's five diagnostics (task 1.15's own hyper-
parameter search never distinguished an architecture problem from a
feature/target/measurement problem): A.3 confirmed the direct points
model's predictions are genuinely compressed relative to actuals
(std(pred)/std(actual) ≈ 0.57 at every position, well under the
addendum's own 0.6 threshold) and A.4 ruled out the training objective
as the cause (a `lambdarank` run on the same features didn't fix it
either). Anchoring is the structural response to compression, not a
tuning fix: rather than predicting points from scratch (where a
compressed/undertrained model can land arbitrarily far from B2), this
predicts the *residual* against B2 (`models.baselines.add_b2_ewm_4`,
"the real bar for did my features do anything") and adds it back:

    target  = actual_points − B2(player, week)
    predict = B2(player, week) + model(features)

A model that has learned nothing outputs ≈0 for the residual and the
composed prediction is exactly B2 -- the floor is the baseline by
construction, not a second independently-trainable quantity that can
fall below it. A.5 found real per-position variation (RB/WR close to B2,
QB/TE clearly behind) -- the blend weight below (`fit_blend_weight`)
lets each position fall back toward B2 automatically rather than forcing
one global answer.

**Feature set:** `models.points.feature_columns(position)` (the same
SPEC §11.3 feature list, unchanged) plus three fit-time-only columns
(`add_points_history_features`, this module): the anchor itself
(`b2_ewm_4` -- SPEC-ADDENDUM-04 §A.2's "does the model have its own
strongest predictor as an input?", literally yes for B2's own value,
letting the model learn how much to trust/discount it) and two genuinely
new signals not already present anywhere in `COMMON_FEATURE_COLUMNS`:
`ewm_points_8` (a slower trailing points average than B2's own span-4)
and `points_last_week` (single-week recency). None of the three is
written to `player_week_features.parquet` -- computed in-memory from the
already-existing `target` column, same seam `ADDENDUM-01 §A.6` already
protects for the real fantasy-points target itself.

**Monotonic constraints:** the base feature set reuses
`points.monotone_constraints(position)` unchanged -- those signs were
already verified against raw points and the underlying mechanism
(more target share/tougher matchup pushes points the same direction)
holds regardless of what the points are measured relative to. The three
new points-history features are left **unconstrained** -- `points.py`'s
own module docstring insists a monotonic sign be verified before
imposing it, and a residual's relationship to its own anchor is
genuinely ambiguous a priori (mean reversion would push it negative,
momentum would push it positive); absent that check, unconstrained is
the honest default, not a guess in either direction.

**Blend weight** (`fit_blend_weight`/`apply_blend_weight`, ADDENDUM-04
§B): `final = w × (B2 + residual_model) + (1 − w) × B2`, `w` fit per
position on a season range strictly earlier than whatever range gets
reported (the caller's responsibility -- these two functions are season-
range-agnostic by design so the honesty of the split lives in one place,
the evaluation script, not duplicated logic here). Grid search (not
closed-form) over `WEIGHT_GRID`, maximizing mean weekly Spearman -- the
project's own literal acceptance metric (SPEC §12.4/§11.3), not MAE,
which a compression-prone residual model could win by shrinking `w`
toward 0 even where real ranking signal exists.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import lightgbm as lgb
import numpy as np
import polars as pl

from ffapp.config import LightGBMSettings
from ffapp.interim.build import SKILL_POSITIONS
from ffapp.models import points

B2_COLUMN = "b2_ewm_4"
POINTS_HISTORY_COLUMNS = ["ewm_points_8", "points_last_week"]
WEIGHT_GRID: tuple[float, ...] = tuple(round(w, 2) for w in np.arange(0.0, 1.001, 0.1))


def add_points_history_features(features: pl.DataFrame) -> pl.DataFrame:
    """Two fit-time-only columns beyond `b2_ewm_4` (assumed already
    present, `models.baselines.add_b2_ewm_4`): `ewm_points_8` and
    `points_last_week`, both trailing strictly through week W-1
    (`.shift(1).over(["player_id", "season"])`, same convention
    `add_b2_ewm_4` itself uses) so the as_of contract holds. Never
    written to `player_week_features.parquet` -- see module docstring.
    """
    sorted_df = features.sort(["player_id", "season", "week"])
    return sorted_df.with_columns(
        pl.col("target")
        .ewm_mean(span=8)
        .shift(1)
        .over(["player_id", "season"])
        .alias("ewm_points_8"),
        pl.col("target").shift(1).over(["player_id", "season"]).alias("points_last_week"),
    )


def residual_feature_columns(position: str) -> list[str]:
    """`points.feature_columns(position)` plus the anchor and the two
    new points-history columns -- see module docstring."""
    return points.feature_columns(position) + [B2_COLUMN, *POINTS_HISTORY_COLUMNS]


def residual_monotone_constraints(position: str) -> list[int]:
    """`points.monotone_constraints(position)`'s own vector, extended
    with `0` (unconstrained) for the three new points-history columns --
    see module docstring for why their sign isn't imposed."""
    return points.monotone_constraints(position) + [0, 0, 0]


@dataclass
class FittedResidualModels:
    boosters: dict[str, lgb.LGBMRegressor]


def fit_residual_model(
    train_rows: pl.DataFrame, *, lightgbm_params: LightGBMSettings
) -> FittedResidualModels:
    """One regressor per position, trained on the RESIDUAL
    (`target - b2_ewm_4`), same "only rows where the player actually
    played" scope `fit_points_model` already established (SPEC §11.2
    Part B) -- a non-played row's `target` is 0 by construction, so its
    residual (`-b2_ewm_4`) is not information about matchup/role-change/
    game-script, just "this player didn't play," which task 1.14's
    availability model already handles separately. Rows with a null
    `b2_ewm_4` (a player's own first tracked week -- no trailing history
    yet to anchor to) are also excluded from training, the same
    "genuinely unknowable, not a value to guess at" precedent
    `features.usage`'s own first-week nulls already established.
    """
    played = train_rows.filter(pl.col(points.AVAILABILITY_COLUMN)).drop_nulls([B2_COLUMN])
    boosters: dict[str, lgb.LGBMRegressor] = {}
    for position in SKILL_POSITIONS:
        position_rows = played.filter(pl.col("position") == position)
        if position_rows.is_empty():
            continue
        columns = residual_feature_columns(position)
        residual_target = (
            position_rows[points.TARGET_COLUMN] - position_rows[B2_COLUMN]
        ).to_numpy()
        booster = lgb.LGBMRegressor(
            n_estimators=lightgbm_params.n_estimators,
            learning_rate=lightgbm_params.learning_rate,
            num_leaves=lightgbm_params.num_leaves,
            min_child_samples=lightgbm_params.min_child_samples,
            subsample=lightgbm_params.subsample,
            colsample_bytree=lightgbm_params.colsample_bytree,
            reg_lambda=lightgbm_params.reg_lambda,
            monotone_constraints=residual_monotone_constraints(position),
            verbosity=-1,
        )
        booster.fit(
            points.to_feature_frame(position_rows, columns),
            residual_target,
            categorical_feature=[c for c in points.CATEGORICAL_COLUMNS if c in columns],
        )
        boosters[position] = booster
    return FittedResidualModels(boosters=boosters)


def predict_residual_points(model: FittedResidualModels, rows: pl.DataFrame) -> pl.Series:
    """`B2(player, week) + model(features)` -- the COMPOSED prediction,
    matching `predict_points`'s own contract of returning `E[points|plays]`
    directly (so this can substitute for `points.PointsPredictor`
    anywhere v1's conditional points model is used, task 1.18's own
    composition unchanged). A row with a null `b2_ewm_4` (no trailing
    history to anchor to, same as fit-time) or whose position has no
    fitted booster predicts honestly null, not a guessed value.
    """
    predictions = np.full(rows.height, np.nan, dtype=float)
    row_positions = rows["position"].to_numpy()
    b2_values = rows[B2_COLUMN].fill_null(float("nan")).to_numpy().astype(float)
    for position, booster in model.boosters.items():
        mask = (row_positions == position) & ~np.isnan(b2_values)
        if not mask.any():
            continue
        columns = residual_feature_columns(position)
        subset = rows.filter(pl.Series(mask))
        residual_pred = np.asarray(booster.predict(points.to_feature_frame(subset, columns)))
        predictions[mask] = b2_values[mask] + residual_pred
    return pl.Series("prediction", predictions, dtype=pl.Float64).fill_nan(None)


class ResidualPredictor:
    """A `evaluation.backtest.Predictor` wrapping
    `fit_residual_model`/`predict_residual_points` -- exercised via the
    same `run_walk_forward_backtest` harness every other predictor in
    this project uses."""

    name = "anchored_residual"

    def __init__(self, lightgbm_params: LightGBMSettings) -> None:
        self.lightgbm_params = lightgbm_params

    def fit(self, train_rows: pl.DataFrame) -> Any:
        return fit_residual_model(train_rows, lightgbm_params=self.lightgbm_params)

    def predict(self, fitted: Any, target_rows: pl.DataFrame) -> pl.Series:
        return predict_residual_points(fitted, target_rows)


# --- blend weight (ADDENDUM-04 §B) -------------------------------------------------------


def _pivot_residual_and_b2(
    predictions: pl.DataFrame, *, residual_name: str, b2_name: str
) -> pl.DataFrame:
    """One row per (season, week, player_id) carrying both predictors'
    own real predictions side by side (`residual_pred`, `b2_pred`) plus
    `position`/`target` -- the shape both `fit_blend_weight` and
    `apply_blend_weight` need to compute a per-row blend without a
    second walk-forward pass. Rows where either predictor is null
    (e.g. a first tracked week with no `b2_ewm_4`) are dropped -- no
    real blend to compute."""
    residual_rows = predictions.filter(pl.col("predictor") == residual_name).select(
        "player_id",
        "season",
        "week",
        "position",
        "team",
        "played",
        "target",
        pl.col("prediction").alias("residual_pred"),
    )
    b2_rows = predictions.filter(pl.col("predictor") == b2_name).select(
        "player_id", "season", "week", pl.col("prediction").alias("b2_pred")
    )
    return residual_rows.join(b2_rows, on=["player_id", "season", "week"], how="inner").drop_nulls(
        ["residual_pred", "b2_pred"]
    )


def _mean_weekly_spearman(df: pl.DataFrame, pred_col: str) -> float:
    weekly = (
        df.group_by(["season", "week"])
        .agg(pl.corr(pred_col, "target", method="spearman").alias("rho"))
        .filter(pl.col("rho").is_not_null() & pl.col("rho").is_not_nan())
    )
    rho_values = weekly["rho"].to_list()
    return float(np.mean(rho_values)) if rho_values else float("-inf")


def fit_blend_weight(
    dev_predictions: pl.DataFrame,
    *,
    residual_name: str = ResidualPredictor.name,
    b2_name: str = "b2_ewm_4",
    weight_grid: tuple[float, ...] = WEIGHT_GRID,
) -> dict[str, float]:
    """Per-position `w ∈ [0, 1]` maximizing mean weekly Spearman (SPEC
    §12.4's own ranking metric) on `dev_predictions` -- real walk-forward
    out-of-sample predictions from a season range the CALLER must ensure
    is strictly earlier than whatever range metrics get reported on
    (SPEC §12.5; this function is season-range-agnostic by design, see
    module docstring). Grid search over `weight_grid`, not closed-form --
    the objective has no closed form and the grid is cheap. A position
    with no real rows in `dev_predictions` gets `w=0.0` (falls back to
    B2 outright, never a guessed positive weight)."""
    pivoted = _pivot_residual_and_b2(dev_predictions, residual_name=residual_name, b2_name=b2_name)
    weights: dict[str, float] = {}
    for position in SKILL_POSITIONS:
        pos_df = pivoted.filter(pl.col("position") == position)
        if pos_df.is_empty():
            weights[position] = 0.0
            continue
        best_w, best_rho = 0.0, float("-inf")
        for w in weight_grid:
            blended = w * pos_df["residual_pred"] + (1 - w) * pos_df["b2_pred"]
            scored = pos_df.with_columns(blended.alias("_blend_pred"))
            mean_rho = _mean_weekly_spearman(scored, "_blend_pred")
            if mean_rho > best_rho:
                best_rho, best_w = mean_rho, w
        weights[position] = best_w
    return weights


def apply_blend_weight(
    predictions: pl.DataFrame,
    weight_by_position: dict[str, float],
    *,
    residual_name: str = ResidualPredictor.name,
    b2_name: str = "b2_ewm_4",
    blended_name: str = "anchored_residual_blend",
) -> pl.DataFrame:
    """Appends real `blended_name` predictor rows -- `w × residual_pred +
    (1 − w) × b2_pred` per row's own position, using each position's
    already-fitted `w` (`fit_blend_weight`). A position missing from
    `weight_by_position` defaults to `w=0.0` (B2 outright), never a
    guessed weight. Output matches `predictions`' own schema exactly, so
    it can be concatenated straight back in for a shared evaluation
    pass."""
    pivoted = _pivot_residual_and_b2(predictions, residual_name=residual_name, b2_name=b2_name)
    w_expr = pl.col("position").replace_strict(
        weight_by_position, default=0.0, return_dtype=pl.Float64
    )
    blended = pivoted.with_columns(
        (w_expr * pl.col("residual_pred") + (1 - w_expr) * pl.col("b2_pred")).alias("prediction")
    )
    return blended.with_columns(pl.lit(blended_name).alias("predictor")).select(predictions.columns)


__all__ = [
    "B2_COLUMN",
    "POINTS_HISTORY_COLUMNS",
    "WEIGHT_GRID",
    "FittedResidualModels",
    "ResidualPredictor",
    "add_points_history_features",
    "apply_blend_weight",
    "fit_blend_weight",
    "fit_residual_model",
    "predict_residual_points",
    "residual_feature_columns",
    "residual_monotone_constraints",
]
