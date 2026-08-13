"""DST model (SPEC.md §11.6; task 2.7).

SPEC's own words: "Separate model, different feature set... Do not
attempt to project individual defensive players. Project team defence
scoring directly." One LightGBM regressor (not per-position, unlike
`models.points` -- DST has only one "position") on
`features.dst.FEATURE_COLUMNS`, exercised through the same
`evaluation.backtest.run_walk_forward_backtest` harness every other
predictor in this project uses -- `build_dst_table` reshapes DST's own
team-keyed rows into that harness's expected `player_id`/`position`/
`availability_flag` shape (matching Sleeper's own real convention,
already established in `scoring.stats.build_dst_stat_frame`, that a
team defence's roster entity *is* its team abbreviation) rather than
writing a second, parallel walk-forward loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import lightgbm as lgb
import pandas as pd
import polars as pl

from ffapp.config import LightGBMSettings
from ffapp.features.dst import FEATURE_COLUMNS

TARGET_COLUMN = "target"
_BOOLEAN_COLUMNS = ("is_home", "is_dome")


def build_dst_table(dst_features: pl.DataFrame, target: pl.DataFrame) -> pl.DataFrame:
    """Join `features.dst.build_dst_features`'s own output with the real
    per-team-week target (`scoring.engine.score_stat_line` applied to
    `scoring.stats.build_dst_stat_frame`'s raw columns -- see the module
    docstring for the reasoning behind the added columns).
    """
    return dst_features.join(target, on=["team", "season", "week"], how="inner").with_columns(
        pl.col("team").alias("player_id"),
        pl.lit("DST").alias("position"),
        pl.lit(True).alias("availability_flag"),
    )


def to_feature_frame(rows: pl.DataFrame) -> pd.DataFrame:
    """Polars -> pandas only at this fit/predict boundary (CLAUDE.md's
    own convention, same as `models.points`/`models.availability`)."""
    pdf = rows.select(FEATURE_COLUMNS).to_pandas()
    for column in _BOOLEAN_COLUMNS:
        if column in pdf.columns:
            pdf[column] = pdf[column].astype(float)
    return pdf


@dataclass
class FittedDstModel:
    booster: lgb.LGBMRegressor


def fit_dst_model(train_rows: pl.DataFrame, *, lightgbm_params: LightGBMSettings) -> FittedDstModel:
    booster = lgb.LGBMRegressor(
        n_estimators=lightgbm_params.n_estimators,
        learning_rate=lightgbm_params.learning_rate,
        num_leaves=lightgbm_params.num_leaves,
        min_child_samples=lightgbm_params.min_child_samples,
        subsample=lightgbm_params.subsample,
        colsample_bytree=lightgbm_params.colsample_bytree,
        reg_lambda=lightgbm_params.reg_lambda,
        verbosity=-1,
    )
    booster.fit(to_feature_frame(train_rows), train_rows[TARGET_COLUMN].to_numpy())
    return FittedDstModel(booster=booster)


def predict_dst(model: FittedDstModel, rows: pl.DataFrame) -> pl.Series:
    predictions = model.booster.predict(to_feature_frame(rows))
    return pl.Series("prediction", predictions, dtype=pl.Float64)


class DstPredictor:
    """A `evaluation.backtest.Predictor` wrapping
    `fit_dst_model`/`predict_dst`, exercised via the same
    `run_walk_forward_backtest` harness every other predictor uses."""

    name = "dst_lightgbm"

    def __init__(self, lightgbm_params: LightGBMSettings) -> None:
        self.lightgbm_params = lightgbm_params

    def fit(self, train_rows: pl.DataFrame) -> Any:
        return fit_dst_model(train_rows, lightgbm_params=self.lightgbm_params)

    def predict(self, fitted: Any, target_rows: pl.DataFrame) -> pl.Series:
        return predict_dst(fitted, target_rows)


def add_dst_b2_ewm_4(dst_table: pl.DataFrame) -> pl.DataFrame:
    """The DST model's own comparison baseline -- TASKS.md 2.7's own
    literal acceptance bar is "beats B2 for DST". Exactly
    `models.baselines.add_b2_ewm_4`'s shape (a team's own trailing
    `ewm_4` of its own real points, `.shift(1)`'d so the target week's
    own outcome never leaks in) applied to `build_dst_table`'s own
    `player_id`/`target` instead of `player_week_features.parquet`'s --
    DST rows live in their own table (task 2.7), not
    `player_week_features.parquet` (task 1.9's own `SKILL_POSITIONS`
    scope, which never included DST)."""
    sorted_df = dst_table.sort(["player_id", "season", "week"])
    return sorted_df.with_columns(
        pl.col(TARGET_COLUMN)
        .ewm_mean(span=4)
        .shift(1)
        .over(["player_id", "season"])
        .alias("dst_b2_ewm_4")
    )


def weekly_streamer_list(predictions: pl.DataFrame, *, season: int, week: int) -> pl.DataFrame:
    """SPEC §11.6/TASKS.md 2.7's own literal ask: "produces a weekly
    streamer list" -- every DST ranked by predicted points, descending,
    for one real (season, week). `predictions` is expected already
    scoped to a single predictor (e.g. `run_walk_forward_backtest`'s own
    output filtered to `predictor == DstPredictor.name`, or called with
    a single-element `predictors` list) -- this function doesn't
    disambiguate multiple predictors' rows for the same team-week
    itself.

    Filtering this ranking down to only currently *unrostered* teams (a
    genuine waiver-wire streamer list, not just a full-field ranking) is
    task 2.6's own job (SPEC §14.4) -- it needs real Sleeper roster
    data, blocked on this network (see HANDOFF §7). This ranks the full
    32-team field, exactly as useful once 2.6's ownership filter is
    applied on top.
    """
    return (
        predictions.filter((pl.col("season") == season) & (pl.col("week") == week))
        .sort("prediction", descending=True)
        .select(pl.col("player_id").alias("team"), "prediction")
    )


__all__ = [
    "TARGET_COLUMN",
    "DstPredictor",
    "FittedDstModel",
    "add_dst_b2_ewm_4",
    "build_dst_table",
    "fit_dst_model",
    "predict_dst",
    "to_feature_frame",
    "weekly_streamer_list",
]
