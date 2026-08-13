"""Availability model -- Part A of SPEC §11.2's hurdle architecture (task 1.14).

    E[points] = P(plays) x E[points | plays]

This module builds `P(plays)` (`p_active`): a LightGBM binary classifier
(target = `availability_flag`, SPEC §11.1's own "recorded >=1 offensive
snap") plus isotonic-regression calibration on a held-out slice of the
training data -- SPEC §11.2: "raw GBM probabilities are not well
calibrated, and this probability is multiplied through everything
downstream."

`AvailabilityPredictor` plugs this into `evaluation.backtest`'s existing
walk-forward harness (task 1.12) with `target_column="availability_flag"`
rather than a second, parallel walk-forward loop -- the harness itself
never assumed it was predicting fantasy points, only that "target" is
some real per-row outcome.

Calibration happens *inside* `fit()`, not against `target_rows`: the most
recent `calibration_weeks` weeks of `train_rows` (still strictly prior to
the week being predicted) are held out from the classifier's own fit and
used only to fit the isotonic calibrator. This keeps calibration
genuinely walk-forward-safe without the harness needing to know anything
about it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
from sklearn.isotonic import IsotonicRegression

from ffapp.config import LightGBMSettings

FEATURE_COLUMNS = [
    "report_status",
    "practice_participation",
    "weeks_since_return",
    "depth_chart_rank",
    "snap_pct_trend",
    "position",
    "age",
]
CATEGORICAL_COLUMNS = ["report_status", "practice_participation", "position"]
TARGET_COLUMN = "availability_flag"

DEFAULT_CALIBRATION_WEEKS = 4


def _to_feature_frame(rows: pl.DataFrame) -> pd.DataFrame:
    """Polars -> pandas only at this fit/predict boundary (CLAUDE.md's
    own convention) -- LightGBM's sklearn API needs pandas/numpy, and
    `category` dtype is how it recognises `CATEGORICAL_COLUMNS` without
    a separate `categorical_feature` index list that could drift out of
    sync with `FEATURE_COLUMNS`."""
    pdf = rows.select(FEATURE_COLUMNS).to_pandas()
    for column in CATEGORICAL_COLUMNS:
        pdf[column] = pdf[column].astype("category")
    return pdf


def calibration_split(
    train_rows: pl.DataFrame, calibration_weeks: int
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Splits `train_rows` into (fit_rows, calibration_rows) by real
    (season, week) order -- the most recent `calibration_weeks` real
    weeks become the calibration set, everything before them is what the
    classifier itself is fit on. Both halves are still strictly prior to
    whatever week the caller is ultimately predicting -- this split
    happens entirely inside `train_rows`, never touching the target
    week.
    """
    weeks = train_rows.select("season", "week").unique().sort(["season", "week"])
    n_weeks = weeks.height
    # Never hold out more than half the real weeks -- `calibration_weeks`
    # is used as-is whenever there's enough data for it (the common
    # case); only a genuinely small `train_rows` shrinks it, leaving
    # `fit_availability_model`'s own both-empty fallback for the extreme
    # (a single real week) rather than this function trying to.
    effective_calibration_weeks = min(calibration_weeks, max(1, n_weeks // 2))
    cutoff_idx = max(0, n_weeks - effective_calibration_weeks)
    cutoff = weeks.row(cutoff_idx, named=True)
    is_calibration = (pl.col("season") > cutoff["season"]) | (
        (pl.col("season") == cutoff["season"]) & (pl.col("week") >= cutoff["week"])
    )
    return train_rows.filter(~is_calibration), train_rows.filter(is_calibration)


@dataclass
class FittedAvailabilityModel:
    booster: lgb.LGBMClassifier
    calibrator: IsotonicRegression


def fit_availability_model(
    train_rows: pl.DataFrame,
    *,
    lightgbm_params: LightGBMSettings,
    calibration_weeks: int = DEFAULT_CALIBRATION_WEEKS,
) -> FittedAvailabilityModel:
    """SPEC §11.2 Part A: fits the classifier, then calibrates its raw
    probabilities with isotonic regression on a held-out tail slice of
    `train_rows` (see module docstring). Falls back to fitting the
    calibrator directly on the classifier's own fit rows when
    `train_rows` spans too few real weeks to hold any out (a real early-
    backtest-week edge case, not a design choice) -- still better than
    crashing or skipping calibration outright.
    """
    fit_rows, calibration_rows = calibration_split(train_rows, calibration_weeks)
    if fit_rows.is_empty() or calibration_rows.is_empty():
        fit_rows = calibration_rows = train_rows

    booster = lgb.LGBMClassifier(
        n_estimators=lightgbm_params.n_estimators,
        learning_rate=lightgbm_params.learning_rate,
        num_leaves=lightgbm_params.num_leaves,
        min_child_samples=lightgbm_params.min_child_samples,
        subsample=lightgbm_params.subsample,
        colsample_bytree=lightgbm_params.colsample_bytree,
        reg_lambda=lightgbm_params.reg_lambda,
        objective="binary",
        verbosity=-1,
    )
    booster.fit(
        _to_feature_frame(fit_rows),
        fit_rows[TARGET_COLUMN].cast(pl.Int8).to_numpy(),
        categorical_feature=CATEGORICAL_COLUMNS,
    )

    raw_p = np.asarray(booster.predict_proba(_to_feature_frame(calibration_rows)))[:, 1]
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    calibrator.fit(raw_p, calibration_rows[TARGET_COLUMN].cast(pl.Int8).to_numpy())

    return FittedAvailabilityModel(booster=booster, calibrator=calibrator)


def predict_p_active(model: FittedAvailabilityModel, rows: pl.DataFrame) -> pl.Series:
    """`p_active`: the classifier's raw probability, passed through the
    isotonic calibrator fit alongside it."""
    raw_p = np.asarray(model.booster.predict_proba(_to_feature_frame(rows)))[:, 1]
    calibrated = model.calibrator.predict(raw_p)
    return pl.Series("p_active", calibrated, dtype=pl.Float64)


class AvailabilityPredictor:
    """A `evaluation.backtest.Predictor` wrapping
    `fit_availability_model`/`predict_p_active` -- exercised via
    `run_walk_forward_backtest(..., target_column="availability_flag")`,
    the same harness every other predictor in this project uses.
    """

    name = "availability_lightgbm"

    def __init__(
        self,
        lightgbm_params: LightGBMSettings,
        *,
        calibration_weeks: int = DEFAULT_CALIBRATION_WEEKS,
    ) -> None:
        self.lightgbm_params = lightgbm_params
        self.calibration_weeks = calibration_weeks

    def fit(self, train_rows: pl.DataFrame) -> Any:
        return fit_availability_model(
            train_rows,
            lightgbm_params=self.lightgbm_params,
            calibration_weeks=self.calibration_weeks,
        )

    def predict(self, fitted: Any, target_rows: pl.DataFrame) -> pl.Series:
        return predict_p_active(fitted, target_rows)


__all__ = [
    "CATEGORICAL_COLUMNS",
    "DEFAULT_CALIBRATION_WEEKS",
    "FEATURE_COLUMNS",
    "TARGET_COLUMN",
    "AvailabilityPredictor",
    "FittedAvailabilityModel",
    "calibration_split",
    "fit_availability_model",
    "predict_p_active",
]
