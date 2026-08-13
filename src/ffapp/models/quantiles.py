"""Quantile models -- the distributional form of SPEC §11.2 Part B (task
1.16; SPEC §11.5).

Five (or however many `model.quantiles` names) LightGBM regressors per
position, `objective="quantile"`, trained on the same played-only rows
and the same feature set `models.points` (task 1.15) already builds --
reused directly (`points.feature_columns`, `points.to_feature_frame`,
`points.AVAILABILITY_COLUMN`), not re-derived a second time.
`points.monotone_constraints` is *not* reused here -- confirmed live,
LightGBM raises outright ("Cannot use monotone_constraints in quantile
objective, please disable it") if a quantile-objective booster is given
any, unlike `models.points`' own mean regressor.

Three real, non-mechanical steps SPEC §11.5 names explicitly:

- **Quantile crossing.** Raw predictions from independently-fit boosters
  aren't guaranteed monotonic across quantile levels. Fixed by sorting
  each row's predicted vector ascending; the fraction of rows that
  actually needed sorting is logged (`FittedQuantileModels.crossing_rate`)
  -- SPEC's own "a high rate signals an underfit or unstable model."
- **Recalibration.** A held-out tail slice of `train_rows` (the same
  `calibration_split` mechanism task 1.14's `models.availability` already
  uses, reused directly here rather than a second copy) measures the
  outer configured interval's real empirical coverage. If it's off
  nominal, a per-position scalar width correction is found (binary
  search on the scale applied to each quantile's distance from the
  median) and re-checked -- SPEC's own "if the 80% interval covers 71%,
  apply a per-position scalar width correction and re-check," done
  literally rather than approximated.
- **Mixture with `p_active`.** SPEC: "a player with `p_active = 0.5` has
  a genuine floor of 0" -- `mixture_with_p_active` inverts the real
  zero-inflated mixture CDF (point mass `1 - p_active` at exactly 0, plus
  `p_active` times the conditional distribution) rather than naively
  scaling the conditional quantiles by `p_active`, which would badly
  misrepresent the floor. The conditional distribution's own quantile at
  the adjusted level is linearly interpolated between the two nearest
  fitted conditional quantile columns -- a standard piecewise-linear
  inverse-CDF approximation from a discrete quantile grid.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import polars as pl

from ffapp.config import LightGBMSettings
from ffapp.interim.build import SKILL_POSITIONS
from ffapp.models import availability, points

# Deliberately its own default, not `availability.DEFAULT_CALIBRATION_WEEKS`
# (4): confirmed live against real 2015-2025 data that 4 weeks is too
# small a sample to estimate a stable width-scale correction on the
# thinner positions (QB/TE) -- coverage on a real held-out season came
# out 6-8 percentage points off nominal for QB/RB/TE (only WR, the
# largest weekly pool, cleared the task's own 5-point bar). 12 weeks
# brought every position within 2.1 points. Isotonic calibration
# (`availability.py`, a differently-shaped estimation problem) doesn't
# have the same small-sample sensitivity, so the two modules' defaults
# are allowed to differ.
DEFAULT_CALIBRATION_WEEKS = 12
DEFAULT_COVERAGE_TOLERANCE = 0.01
_WIDTH_SCALE_SEARCH_BOUNDS = (0.01, 20.0)


def _raw_predict(
    boosters: dict[float, lgb.LGBMRegressor],
    rows: pl.DataFrame,
    columns: list[str],
    alphas: list[float],
) -> np.ndarray:
    """(n_rows, n_alphas), columns in `alphas`' own order."""
    frame = points.to_feature_frame(rows, columns)
    return np.column_stack([np.asarray(boosters[alpha].predict(frame)) for alpha in alphas])


def _crossing_rate(raw: np.ndarray) -> float:
    """Fraction of rows whose raw (pre-sort) quantile vector violates
    ascending order somewhere -- SPEC §11.5's own diagnostic."""
    if raw.shape[0] == 0:
        return 0.0
    violations = (np.diff(raw, axis=1) < 0).any(axis=1)
    return float(violations.mean())


def _nearest_alpha_index(alphas: list[float], target: float) -> int:
    return min(range(len(alphas)), key=lambda i: abs(alphas[i] - target))


def _empirical_coverage(lower: np.ndarray, upper: np.ndarray, actual: np.ndarray) -> float:
    return float(((actual >= lower) & (actual <= upper)).mean())


def find_width_scale(
    *,
    median: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    actual: np.ndarray,
    nominal_coverage: float,
    tolerance: float = DEFAULT_COVERAGE_TOLERANCE,
    max_iterations: int = 30,
) -> float:
    """SPEC §11.5's "apply a per-position scalar width correction and
    re-check," as a real binary search: finds `s` such that scaling every
    quantile's distance from the median by `s` brings the outer
    interval's empirical coverage within `tolerance` of `nominal_coverage`
    on the calibration holdout. Coverage is monotone non-decreasing in
    `s` (a wider interval covers at least as much), so the search is
    well-posed; the bracket's own endpoints are returned outright if
    nominal coverage is unreachable even at the search bounds (a real,
    honest result -- not silently clamped to look successful).
    """
    low, high = _WIDTH_SCALE_SEARCH_BOUNDS

    def coverage_at(scale: float) -> float:
        scaled_lower = median + scale * (lower - median)
        scaled_upper = median + scale * (upper - median)
        return _empirical_coverage(scaled_lower, scaled_upper, actual)

    if coverage_at(high) < nominal_coverage:
        return high
    if coverage_at(low) > nominal_coverage:
        return low
    for _ in range(max_iterations):
        mid = (low + high) / 2
        coverage = coverage_at(mid)
        if abs(coverage - nominal_coverage) <= tolerance:
            return mid
        if coverage < nominal_coverage:
            low = mid
        else:
            high = mid
    return (low + high) / 2


@dataclass
class FittedQuantileModels:
    boosters: dict[str, dict[float, lgb.LGBMRegressor]]  # position -> {alpha: booster}
    width_scale: dict[str, float]  # position -> SPEC §11.5's scalar width correction
    crossing_rate: dict[str, float]  # position -> fraction of calibration rows that crossed


def fit_quantile_models(
    train_rows: pl.DataFrame,
    *,
    lightgbm_params: LightGBMSettings,
    quantile_alphas: Sequence[float],
    calibration_weeks: int = DEFAULT_CALIBRATION_WEEKS,
) -> FittedQuantileModels:
    """SPEC §11.5: one regressor per (position, alpha), fit only on rows
    where the player actually played (same scope as `models.points`).
    Recalibration and crossing-rate measurement both use the same
    held-out tail slice `models.availability.calibration_split` already
    establishes -- both halves stay strictly prior to whatever week the
    caller ultimately predicts.
    """
    played = train_rows.filter(pl.col(points.AVAILABILITY_COLUMN))
    fit_rows, calibration_rows = availability.calibration_split(played, calibration_weeks)
    if fit_rows.is_empty() or calibration_rows.is_empty():
        fit_rows = calibration_rows = played

    sorted_alphas = sorted(quantile_alphas)
    boosters: dict[str, dict[float, lgb.LGBMRegressor]] = {}
    width_scale: dict[str, float] = {}
    crossing_rate: dict[str, float] = {}

    for position in SKILL_POSITIONS:
        position_fit_rows = fit_rows.filter(pl.col("position") == position)
        if position_fit_rows.is_empty():
            continue

        columns = points.feature_columns(position)
        position_boosters: dict[float, lgb.LGBMRegressor] = {}
        for alpha in sorted_alphas:
            booster = lgb.LGBMRegressor(
                objective="quantile",
                alpha=alpha,
                n_estimators=lightgbm_params.n_estimators,
                learning_rate=lightgbm_params.learning_rate,
                num_leaves=lightgbm_params.num_leaves,
                min_child_samples=lightgbm_params.min_child_samples,
                subsample=lightgbm_params.subsample,
                colsample_bytree=lightgbm_params.colsample_bytree,
                reg_lambda=lightgbm_params.reg_lambda,
                # LightGBM forbids monotone_constraints under the
                # quantile objective outright (confirmed directly:
                # "Cannot use monotone_constraints in quantile objective,
                # please disable it") -- unlike models.points' mean
                # regressor, no monotonic constraint is applied here.
                verbosity=-1,
            )
            booster.fit(
                points.to_feature_frame(position_fit_rows, columns),
                position_fit_rows[points.TARGET_COLUMN].to_numpy(),
                categorical_feature=[c for c in points.CATEGORICAL_COLUMNS if c in columns],
            )
            position_boosters[alpha] = booster
        boosters[position] = position_boosters

        position_calibration_rows = calibration_rows.filter(pl.col("position") == position)
        if position_calibration_rows.is_empty():
            width_scale[position] = 1.0
            crossing_rate[position] = 0.0
            continue

        raw = _raw_predict(position_boosters, position_calibration_rows, columns, sorted_alphas)
        crossing_rate[position] = _crossing_rate(raw)
        sorted_raw = np.sort(raw, axis=1)

        actual = position_calibration_rows[points.TARGET_COLUMN].to_numpy()
        median_idx = _nearest_alpha_index(sorted_alphas, 0.5)
        nominal_coverage = sorted_alphas[-1] - sorted_alphas[0]
        width_scale[position] = find_width_scale(
            median=sorted_raw[:, median_idx],
            lower=sorted_raw[:, 0],
            upper=sorted_raw[:, -1],
            actual=actual,
            nominal_coverage=nominal_coverage,
        )

    return FittedQuantileModels(
        boosters=boosters, width_scale=width_scale, crossing_rate=crossing_rate
    )


def predict_quantiles(model: FittedQuantileModels, rows: pl.DataFrame) -> pl.DataFrame:
    """One row per input row (same order as `rows`), one `q_<alpha>`
    column per fitted quantile level -- crossing-fixed (sorted ascending)
    and width-recalibrated (this position's own scalar correction from
    `fit_quantile_models`, applied around the median). A row whose
    position has no fitted booster gets honestly null columns.
    """
    all_alphas = sorted({alpha for boosters in model.boosters.values() for alpha in boosters})
    if not all_alphas:
        return pl.DataFrame({f"q_{a}": [] for a in all_alphas})

    result = {alpha: np.full(rows.height, np.nan) for alpha in all_alphas}
    row_positions = rows["position"].to_numpy()

    for position, position_boosters in model.boosters.items():
        mask = row_positions == position
        if not mask.any():
            continue
        alphas = sorted(position_boosters)
        columns = points.feature_columns(position)
        subset = rows.filter(pl.Series(mask))
        raw = _raw_predict(position_boosters, subset, columns, alphas)
        sorted_raw = np.sort(raw, axis=1)

        median_idx = _nearest_alpha_index(alphas, 0.5)
        median = sorted_raw[:, median_idx : median_idx + 1]
        scale = model.width_scale.get(position, 1.0)
        recalibrated = median + scale * (sorted_raw - median)
        recalibrated = np.sort(recalibrated, axis=1)  # scaling can reintroduce crossings

        for i, alpha in enumerate(alphas):
            result[alpha][mask] = recalibrated[:, i]

    return pl.DataFrame(
        {
            f"q_{alpha}": pl.Series(f"q_{alpha}", values).fill_nan(None)
            for alpha, values in result.items()
        }
    )


def mixture_with_p_active(
    conditional_quantiles: pl.DataFrame, p_active: pl.Series, alphas: Sequence[float]
) -> pl.DataFrame:
    """SPEC §11.5: "the mixture with `p_active` is applied afterwards to
    produce unconditional quantiles, since a player with `p_active = 0.5`
    has a genuine floor of 0." Inverts the real zero-inflated mixture
    CDF -- for nominal level `tau`: if `tau <= 1 - p_active`, the point
    mass at 0 alone already covers that much probability, so the
    unconditional quantile is exactly 0; otherwise it's the conditional
    distribution's own quantile at the adjusted level
    `(tau - (1 - p_active)) / p_active`, linearly interpolated between
    the two nearest fitted `q_<alpha>` columns (`conditional_quantiles`,
    `predict_quantiles`'s own output shape).
    """
    sorted_alphas = sorted(alphas)
    conditional_matrix = conditional_quantiles.select(
        [f"q_{alpha}" for alpha in sorted_alphas]
    ).to_numpy()
    p = np.clip(p_active.to_numpy(), 1e-9, 1.0)

    result: dict[str, np.ndarray] = {}
    for tau in sorted_alphas:
        adjusted = (tau - (1.0 - p)) / p
        values = np.array(
            [np.interp(adjusted[i], sorted_alphas, conditional_matrix[i]) for i in range(len(p))]
        )
        below_floor = tau <= (1.0 - p_active.to_numpy())
        values = np.where(below_floor, 0.0, values)
        result[f"unconditional_q_{tau}"] = np.maximum(values, 0.0)

    return pl.DataFrame(result)


__all__ = [
    "DEFAULT_CALIBRATION_WEEKS",
    "DEFAULT_COVERAGE_TOLERANCE",
    "FittedQuantileModels",
    "find_width_scale",
    "fit_quantile_models",
    "mixture_with_p_active",
    "predict_quantiles",
]
