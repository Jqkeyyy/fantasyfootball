"""Evaluation metrics (SPEC.md §12.4-§12.5; task 1.13).

Consumes `evaluation.backtest.run_walk_forward_backtest`'s own output
shape directly: one row per (player_id, season, week, predictor), with
`team`, `position`, `target` (the real actual outcome), and `prediction`.

Every metric is reported as a `MetricResult`: a value, its own real
observation count, and a bootstrap confidence interval -- SPEC §12.5's
own statistical-honesty rule ("state the number of observations behind
every reported metric," "report bootstrap confidence intervals") applied
uniformly, not as an afterthought bolted onto a subset of metrics.

**Bootstrap resampling is by week, never by row** (SPEC §12.5: "rows
within a week are correlated"). Two flavours exist because the two
families of metrics reduce differently:

- `bootstrap_ci_over_rows`: for a statistic that reduces directly over
  rows (MAE, RMSE) -- resamples *weeks* with replacement, duplicating
  every row of a week drawn more than once, then reapplies the
  statistic to the resampled row set.
- `bootstrap_ci_over_weekly_values`: for a statistic that's already been
  reduced to one real value per week (Spearman-per-week, top-k
  precision-per-week, start/sit accuracy-per-week) -- resamples
  *indices* into that pre-computed per-week array. Re-deriving a
  per-week reduction (e.g. Spearman, itself a `group_by(week)`) inside
  a row-duplicating bootstrap would silently collapse back to one row
  per week on the next `group_by`, undoing the resampling weight
  entirely -- a real, easy-to-miss bug this split avoids by construction.

**"Startable"** (SPEC §12.4: "players rostered and above a projection
threshold -- accuracy on the deep bench does not matter") and **top-k
precision**'s own `k` both come from `startable_counts_from_predictions`,
which reuses `tools.vor`'s already-tested SPEC §9.4 fixed-point
convergence over a real `LeagueFormat` (CLAUDE.md rule 5: never hardcode
a starting lineup shape) -- one shared, predictor-agnostic per-position
pool size, computed once from real `target` values pooled across the
whole evaluation, not re-derived per predictor or per week. Sharing one
pool means an accuracy comparison across predictors on "startable" rows
is comparing the same rows for every predictor.

**Distribution metrics** (`pinball_loss`, `interval_coverage`) are pure
numeric primitives, not yet wired to any real data -- no quantile model
exists until task 1.16 (SPEC §11.5). Calibration plots are task 1.17's
job (SPEC §12.6).

**Lineup regret is not implemented here.** SPEC §12.4's
`optimal_lineup_points - model_recommended_lineup_points` needs
`sim/lineup.py`'s real ILP optimiser (SPEC §13.1), which is task 2.1 --
ordered *after* this task in TASKS.md. Confirmed with you rather than
guessed: defer the metric, revisit once 2.1 lands.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import polars as pl

from ffapp.league_format import LeagueFormat
from ffapp.tools import vor

DEFAULT_N_BOOTSTRAP = 1000
DEFAULT_CI = 0.95


@dataclass(frozen=True)
class MetricResult:
    """`n_obs` is the count behind `value` in whatever unit that metric is
    actually averaged over: player-week rows for row-level metrics (MAE,
    RMSE), real (season, week) observations for week-level metrics
    (Spearman, top-k precision) -- SPEC §12.5's "state the number of
    observations" applied honestly to each metric's own real unit, rather
    than forcing a single unit that would misrepresent one family or the
    other.
    """

    metric: str
    predictor: str
    position: str | None  # None = every position pooled
    scope: str  # "all" or "startable"
    value: float
    n_obs: int
    ci_low: float
    ci_high: float


# --- bootstrap ------------------------------------------------------------------------


def bootstrap_ci_over_rows(
    df: pl.DataFrame,
    statistic: Callable[[pl.DataFrame], float],
    *,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    ci: float = DEFAULT_CI,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """Resamples `df`'s own real (season, week) keys with replacement,
    duplicating every row of a week drawn more than once, and reapplies
    `statistic` to each resampled row set. Correct for a `statistic` that
    is itself a plain reduction over rows (mean absolute error, mean
    squared error) -- duplicate rows correctly reweight a pooled mean.
    """
    generator = rng if rng is not None else np.random.default_rng()
    weeks = df.select("season", "week").unique()
    n_weeks = weeks.height
    if n_weeks == 0:
        return (float("nan"), float("nan"))

    values = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        draw_idx = generator.integers(0, n_weeks, size=n_weeks)
        sampled_weeks = weeks[draw_idx]
        resampled = sampled_weeks.join(df, on=["season", "week"], how="left")
        values[i] = statistic(resampled)

    alpha = (1 - ci) / 2
    return (float(np.nanquantile(values, alpha)), float(np.nanquantile(values, 1 - alpha)))


def bootstrap_ci_over_weekly_values(
    weekly_values: Sequence[float] | np.ndarray,
    *,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    ci: float = DEFAULT_CI,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """For a statistic already reduced to one real value per (season,
    week) -- resamples *indices* into that array with replacement and
    averages each draw, rather than re-deriving the per-week reduction
    inside every bootstrap iteration (see module docstring for why that
    would silently undo the resampling weight for a `group_by`-based
    per-week statistic).
    """
    arr = np.asarray(weekly_values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return (float("nan"), float("nan"))

    generator = rng if rng is not None else np.random.default_rng()
    draws = generator.integers(0, arr.size, size=(n_bootstrap, arr.size))
    means = arr[draws].mean(axis=1)
    alpha = (1 - ci) / 2
    return (float(np.quantile(means, alpha)), float(np.quantile(means, 1 - alpha)))


# --- accuracy (SPEC §12.4) -------------------------------------------------------------


def _mae(df: pl.DataFrame) -> float:
    return float(df.select((pl.col("prediction") - pl.col("target")).abs().mean()).item())


def _rmse(df: pl.DataFrame) -> float:
    return float(df.select(((pl.col("prediction") - pl.col("target")) ** 2).mean().sqrt()).item())


def _filter_startable(df: pl.DataFrame, startable_counts: dict[str, int]) -> pl.DataFrame:
    """Keeps only rows whose player was among the real top-N (by actual
    `target`) at their position that week, N = `startable_counts[position]`
    -- the ground-truth pool, shared by every predictor (see module
    docstring)."""
    ranked = df.with_columns(
        pl.col("target")
        .rank(method="ordinal", descending=True)
        .over(["season", "week", "position"])
        .alias("_true_rank")
    )
    max_rank = pl.col("position").replace_strict(startable_counts, default=0, return_dtype=pl.Int64)
    return ranked.filter(pl.col("_true_rank") <= max_rank).drop("_true_rank")


def _metric_result(
    metric: str,
    predictor: str,
    position: str | None,
    scope: str,
    df: pl.DataFrame,
    statistic: Callable[[pl.DataFrame], float],
    *,
    n_bootstrap: int,
    ci: float,
    rng: np.random.Generator | None,
) -> MetricResult:
    value = statistic(df) if not df.is_empty() else float("nan")
    ci_low, ci_high = bootstrap_ci_over_rows(df, statistic, n_bootstrap=n_bootstrap, ci=ci, rng=rng)
    return MetricResult(
        metric=metric,
        predictor=predictor,
        position=position,
        scope=scope,
        value=value,
        n_obs=df.height,
        ci_low=ci_low,
        ci_high=ci_high,
    )


def _accuracy_for_scope(
    df: pl.DataFrame,
    predictor: str,
    scope: str,
    *,
    n_bootstrap: int,
    ci: float,
    rng: np.random.Generator | None,
) -> list[MetricResult]:
    results = [
        _metric_result(
            "mae", predictor, None, scope, df, _mae, n_bootstrap=n_bootstrap, ci=ci, rng=rng
        ),
        _metric_result(
            "rmse", predictor, None, scope, df, _rmse, n_bootstrap=n_bootstrap, ci=ci, rng=rng
        ),
    ]
    for position in sorted(df["position"].unique().to_list()):
        pos_df = df.filter(pl.col("position") == position)
        results.append(
            _metric_result(
                "mae",
                predictor,
                position,
                scope,
                pos_df,
                _mae,
                n_bootstrap=n_bootstrap,
                ci=ci,
                rng=rng,
            )
        )
        results.append(
            _metric_result(
                "rmse",
                predictor,
                position,
                scope,
                pos_df,
                _rmse,
                n_bootstrap=n_bootstrap,
                ci=ci,
                rng=rng,
            )
        )
    return results


def accuracy_metrics(
    predictions: pl.DataFrame,
    *,
    startable_counts: dict[str, int] | None = None,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    ci: float = DEFAULT_CI,
    rng: np.random.Generator | None = None,
) -> list[MetricResult]:
    """SPEC §12.4 Accuracy: MAE and RMSE, overall and per position, on all
    rows and (if `startable_counts` is given) on startable rows only.
    `predictions` must have `prediction` non-null on any row scored --
    rows with a null `prediction` (e.g. a baseline's own honest first-
    tracked-week null, task 1.10) are excluded before computing.
    """
    results: list[MetricResult] = []
    for predictor in sorted(predictions["predictor"].unique().to_list()):
        pred_df = predictions.filter(
            (pl.col("predictor") == predictor) & pl.col("prediction").is_not_null()
        )
        results.extend(
            _accuracy_for_scope(pred_df, predictor, "all", n_bootstrap=n_bootstrap, ci=ci, rng=rng)
        )
        if startable_counts is not None:
            startable_df = _filter_startable(pred_df, startable_counts)
            results.extend(
                _accuracy_for_scope(
                    startable_df, predictor, "startable", n_bootstrap=n_bootstrap, ci=ci, rng=rng
                )
            )
    return results


# --- ranking (SPEC §12.4) ---------------------------------------------------------------


def _weekly_spearman(df: pl.DataFrame) -> pl.DataFrame:
    """A week where every player at a position got the *same* predicted
    value (e.g. B0's pooled positional mean -- one number per position
    per week, not per player) has undefined correlation: polars' `corr`
    returns float `NaN` for zero-variance input, not a real `null` --
    `.drop_nulls` alone doesn't catch it, so `is_not_nan` is checked too.
    """
    return (
        df.group_by(["season", "week"])
        .agg(pl.corr("prediction", "target", method="spearman").alias("rho"))
        .filter(pl.col("rho").is_not_null() & pl.col("rho").is_not_nan())
    )


def _weekly_top_k_precision(df: pl.DataFrame, k: int) -> pl.DataFrame:
    with_ranks = df.with_columns(
        pl.col("target")
        .rank(method="ordinal", descending=True)
        .over(["season", "week"])
        .alias("_true_rank"),
        pl.col("prediction")
        .rank(method="ordinal", descending=True)
        .over(["season", "week"])
        .alias("_pred_rank"),
    )
    return (
        with_ranks.group_by(["season", "week"])
        .agg(
            ((pl.col("_true_rank") <= k) & (pl.col("_pred_rank") <= k)).sum().alias("_hits"),
            pl.len().alias("_n"),
        )
        .filter(pl.col("_n") >= k)  # fewer than k real candidates -- no real top-k to score
        .with_columns((pl.col("_hits") / k).alias("precision"))
    )


def ranking_metrics(
    predictions: pl.DataFrame,
    *,
    startable_counts: dict[str, int] | None = None,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    ci: float = DEFAULT_CI,
    rng: np.random.Generator | None = None,
) -> list[MetricResult]:
    """SPEC §12.4 Ranking: Spearman rank correlation within each (position,
    week), averaged across weeks; and top-k precision (of the real top-k
    at a position that week, how many were in the predicted top-k), k
    from `startable_counts` (see module docstring).
    """
    results: list[MetricResult] = []
    for predictor in sorted(predictions["predictor"].unique().to_list()):
        pred_df = predictions.filter(
            (pl.col("predictor") == predictor) & pl.col("prediction").is_not_null()
        )
        for position in sorted(pred_df["position"].unique().to_list()):
            pos_df = pred_df.filter(pl.col("position") == position)

            weekly_rho = _weekly_spearman(pos_df)
            rho_values = weekly_rho["rho"].to_list()
            rho_value = float(np.mean(rho_values)) if rho_values else float("nan")
            rho_low, rho_high = bootstrap_ci_over_weekly_values(
                rho_values, n_bootstrap=n_bootstrap, ci=ci, rng=rng
            )
            results.append(
                MetricResult(
                    metric="spearman",
                    predictor=predictor,
                    position=position,
                    scope="all",
                    value=rho_value,
                    n_obs=weekly_rho.height,
                    ci_low=rho_low,
                    ci_high=rho_high,
                )
            )

            k = (startable_counts or {}).get(position, 0)
            if k > 0:
                weekly_prec = _weekly_top_k_precision(pos_df, k)
                prec_values = weekly_prec["precision"].to_list()
                prec_value = float(np.mean(prec_values)) if prec_values else float("nan")
                prec_low, prec_high = bootstrap_ci_over_weekly_values(
                    prec_values, n_bootstrap=n_bootstrap, ci=ci, rng=rng
                )
                results.append(
                    MetricResult(
                        metric=f"top_{k}_precision",
                        predictor=predictor,
                        position=position,
                        scope="all",
                        value=prec_value,
                        n_obs=weekly_prec.height,
                        ci_low=prec_low,
                        ci_high=prec_high,
                    )
                )
    return results


# --- startable pool size (SPEC §9.4 reused, CLAUDE.md rule 5) --------------------------


def startable_counts_from_predictions(
    predictions: pl.DataFrame, league_format: LeagueFormat
) -> dict[str, int]:
    """The per-position startable-pool size `accuracy_metrics`/
    `ranking_metrics` both use -- reuses `tools.vor`'s already-tested SPEC
    §9.4 fixed-point convergence directly, from real `target` points
    pooled across every (season, week) in `predictions` (any one
    predictor's rows -- `target` is predictor-independent). One stable
    pool size for the whole evaluation, not one that reflows week to
    week.
    """
    any_predictor = predictions["predictor"][0]
    one_predictor = predictions.filter(pl.col("predictor") == any_predictor)
    points_by_position = {
        position: sorted(
            one_predictor.filter(pl.col("position") == position)["target"].to_list(), reverse=True
        )
        for position in one_predictor["position"].unique().to_list()
    }
    return vor.startable_counts(points_by_position, league_format)


# --- distribution (SPEC §12.4) ----------------------------------------------------------


def pinball_loss(
    target: Sequence[float] | np.ndarray,
    predicted: Sequence[float] | np.ndarray,
    quantile: float,
) -> float:
    """SPEC §12.4 Distribution: pinball loss at a single `quantile` (0-1).
    Not yet exercised against real data -- no quantile model exists until
    task 1.16 (SPEC §11.5); implemented now so this module doesn't need
    revisiting when it lands.
    """
    t = np.asarray(target, dtype=float)
    p = np.asarray(predicted, dtype=float)
    diff = t - p
    loss = np.where(diff >= 0, quantile * diff, (quantile - 1) * diff)
    return float(loss.mean())


def interval_coverage(
    target: Sequence[float] | np.ndarray,
    lower: Sequence[float] | np.ndarray,
    upper: Sequence[float] | np.ndarray,
) -> float:
    """SPEC §12.4 Distribution: empirical coverage of a predicted
    [lower, upper] interval -- the fraction of real `target` values that
    actually fell inside their own predicted interval. Same task-1.16
    status as `pinball_loss`.
    """
    t = np.asarray(target, dtype=float)
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    return float(((t >= lo) & (t <= hi)).mean())


def brier_score(
    predicted_prob: Sequence[float] | np.ndarray, actual: Sequence[float] | np.ndarray
) -> float:
    """Mean squared error of a predicted probability against a real
    binary (0/1) outcome -- SPEC §11.2's own calibration metric for the
    availability model (task 1.14): "Brier score beats a positional
    base-rate predictor."
    """
    p = np.asarray(predicted_prob, dtype=float)
    a = np.asarray(actual, dtype=float)
    return float(np.mean((p - a) ** 2))


def calibration_curve(
    predicted_prob: Sequence[float] | np.ndarray,
    actual: Sequence[float] | np.ndarray,
    *,
    n_bins: int = 10,
) -> list[tuple[float, float, int]]:
    """Buckets rows into `n_bins` equal-width probability bins; returns
    `(mean predicted probability, mean actual outcome, n rows)` per
    non-empty bin, ascending -- SPEC §12.4's "predicted quantile vs
    realised frequency" calibration check, applied to a binary
    probability (task 1.14's availability model) rather than a points
    quantile: is the model's stated confidence trustworthy? A
    "near-diagonal" curve (SPEC §11.2/task 1.14's own acceptance bar)
    means the two columns of each tuple are close.
    """
    p = np.asarray(predicted_prob, dtype=float)
    a = np.asarray(actual, dtype=float)
    bin_idx = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    result: list[tuple[float, float, int]] = []
    for b in range(n_bins):
        mask = bin_idx == b
        if not mask.any():
            continue
        result.append((float(p[mask].mean()), float(a[mask].mean()), int(mask.sum())))
    return result


# --- decision quality (SPEC §12.4) -------------------------------------------------------


def _weekly_start_sit_accuracy(df: pl.DataFrame) -> pl.DataFrame:
    """Every real pairwise choice between two flex-eligible teammates
    (same real NFL `team`, same `season`/`week`) in `df`: did the model's
    pick (the higher `prediction`) actually outscore the alternative?
    Exact ties in either `prediction` or `target` are excluded -- no real
    pick, nothing to score against.
    """
    left = df.select(
        "season",
        "week",
        "team",
        pl.col("player_id").alias("player_id_a"),
        pl.col("prediction").alias("prediction_a"),
        pl.col("target").alias("target_a"),
    )
    right = df.select(
        "season",
        "week",
        "team",
        pl.col("player_id").alias("player_id_b"),
        pl.col("prediction").alias("prediction_b"),
        pl.col("target").alias("target_b"),
    )
    pairs = (
        left.join(right, on=["season", "week", "team"], how="inner")
        .filter(pl.col("player_id_a") < pl.col("player_id_b"))
        .filter(pl.col("prediction_a") != pl.col("prediction_b"))
        .filter(pl.col("target_a") != pl.col("target_b"))
    )
    if pairs.is_empty():
        return pl.DataFrame(
            schema={
                "season": pl.Int64,
                "week": pl.Int64,
                "accuracy": pl.Float64,
                "n_pairs": pl.Int64,
            }
        )

    model_picks_a = pl.col("prediction_a") > pl.col("prediction_b")
    a_outscored_b = pl.col("target_a") > pl.col("target_b")
    correct = (model_picks_a & a_outscored_b) | (~model_picks_a & ~a_outscored_b)
    return (
        pairs.with_columns(correct.alias("_correct"))
        .group_by(["season", "week"])
        .agg(pl.col("_correct").mean().alias("accuracy"), pl.len().alias("n_pairs"))
    )


def start_sit_accuracy(
    predictions: pl.DataFrame,
    flex_eligible_positions: set[str],
    *,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    ci: float = DEFAULT_CI,
    rng: np.random.Generator | None = None,
) -> list[MetricResult]:
    """SPEC §12.4 Decision quality: "construct realistic pairwise choices
    (two flex-eligible players on the same roster) and measure how often
    the model's pick outscored the alternative." This backtest spans
    multiple real NFL seasons independent of any one fantasy league's
    actual historical rosters (which don't span the full validation
    range consistently), so "same roster" is operationalised as "same
    real NFL team" -- confirmed with you rather than guessed: a genuine
    flex decision a manager could actually face (a bye week or injury
    forcing a choice between two teammates), with real data available
    for every backtest week regardless of which fantasy league is being
    evaluated.
    """
    results: list[MetricResult] = []
    for predictor in sorted(predictions["predictor"].unique().to_list()):
        pred_df = predictions.filter(
            (pl.col("predictor") == predictor)
            & pl.col("prediction").is_not_null()
            & pl.col("position").is_in(list(flex_eligible_positions))
        )
        weekly = _weekly_start_sit_accuracy(pred_df)
        accuracy_values = weekly["accuracy"].to_list()
        value = float(np.mean(accuracy_values)) if accuracy_values else float("nan")
        ci_low, ci_high = bootstrap_ci_over_weekly_values(
            accuracy_values, n_bootstrap=n_bootstrap, ci=ci, rng=rng
        )
        n_obs = int(weekly["n_pairs"].sum()) if not weekly.is_empty() else 0
        results.append(
            MetricResult(
                metric="start_sit_accuracy",
                predictor=predictor,
                position=None,
                scope="all",
                value=value,
                n_obs=n_obs,
                ci_low=ci_low,
                ci_high=ci_high,
            )
        )
    return results


__all__ = [
    "DEFAULT_CI",
    "DEFAULT_N_BOOTSTRAP",
    "MetricResult",
    "accuracy_metrics",
    "bootstrap_ci_over_rows",
    "bootstrap_ci_over_weekly_values",
    "brier_score",
    "calibration_curve",
    "interval_coverage",
    "pinball_loss",
    "ranking_metrics",
    "start_sit_accuracy",
    "startable_counts_from_predictions",
]
