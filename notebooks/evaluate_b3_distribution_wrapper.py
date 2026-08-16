# notebooks/evaluate_b3_distribution_wrapper.py
"""Real coverage evaluation for `models.predict.project_week`'s own
`consensus_b3` distribution wrapper (`SPEC-ADDENDUM-04.md` §D's own
blocking work item -- see `docs/JOURNAL.md`'s 2026-08-16 entry). Two
real options, both evaluated against real 2021-2024 B3/actual data and
task 1.16's own established coverage bar (TASKS.md: within 5 percentage
points of nominal per position; the real achieved number was 2.1pp):

(a) **Empirical B3-error spread**: the empirical distribution of
    `actual - b3_points` (real historical residuals), per position, fit
    on a dev slice (2021-2022) and applied additively to B3's own point
    estimate on a held-out slice (2023-2024). No model at all -- purely
    non-parametric.

(b) **v1's quantile spread applied around B3's point estimate**: exactly
    `models.predict.project_week`'s own current interim implementation
    -- task 1.16's already-validated unconditional quantile grid,
    recentered so its own median lands on B3's point estimate instead of
    the points model's own conditional median. Unlike a true walk-forward
    evaluation (task 1.16's own, already validated separately), this
    reuses a SINGLE quantile-model fit on all real training data through
    the dev/test boundary rather than refitting weekly -- a real,
    documented simplification: this evaluation is about whether v1's
    already-validated SPREAD SHAPE transfers coherently onto a
    different-source mean, not a re-validation of the quantile model
    itself.

Scratch, per CLAUDE.md's notebooks/ convention -- not imported by src/.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from ffapp.config import DEFAULT_QUANTILES, load_settings
from ffapp.interim.build import SKILL_POSITIONS
from ffapp.models import availability, points, quantiles

DEV_SEASONS = [2021, 2022]
TEST_SEASONS = [2023, 2024]
INTERVALS = [(0.10, 0.90, 0.80), (0.25, 0.75, 0.50)]  # (lower_tau, upper_tau, nominal_coverage)


def _coverage(actual: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    mask = ~(np.isnan(lower) | np.isnan(upper))
    if mask.sum() == 0:
        return float("nan")
    return float(((actual[mask] >= lower[mask]) & (actual[mask] <= upper[mask])).mean())


def main() -> None:
    settings = load_settings()
    schedule = pl.read_parquet(settings.data_root / "interim" / "schedule.parquet")
    features = pl.read_parquet(settings.data_root / "features" / "player_week_features.parquet")
    b3 = pl.read_parquet(settings.data_root / "interim" / "b3_predictions.parquet")

    schedule = schedule.filter(pl.col("season_type") == "REG")
    reg_weeks = schedule.select("season", "week").unique()
    features = features.join(reg_weeks, on=["season", "week"], how="inner").filter(
        pl.col("position").is_in(SKILL_POSITIONS)
    )

    with_b3 = features.join(b3, on=["player_id", "season", "week"], how="inner")
    print(f"Real rows with both features and a real B3 value: {with_b3.height}")

    dev = with_b3.filter(pl.col("season").is_in(DEV_SEASONS))
    test = with_b3.filter(pl.col("season").is_in(TEST_SEASONS))
    print(f"Dev (fit): {dev.height} rows, seasons {DEV_SEASONS}")
    print(f"Test (held out, report on this): {test.height} rows, seasons {TEST_SEASONS}")

    # --- Option (a): empirical B3-error spread, per position --------------------------
    print("\n=== Option (a): empirical B3-error spread ===")
    dev_with_error = dev.with_columns((pl.col("target") - pl.col("b3_points")).alias("error"))

    error_quantiles: dict[str, dict[float, float]] = {}
    for position in SKILL_POSITIONS:
        pos_errors = dev_with_error.filter(pl.col("position") == position)["error"].drop_nulls()
        error_quantiles[position] = {
            tau: float(pos_errors.quantile(tau, interpolation="linear") or 0.0)
            for tau in DEFAULT_QUANTILES
        }
        print(f"  {position} (n={pos_errors.len()}): {error_quantiles[position]}")

    for lower_tau, upper_tau, nominal in INTERVALS:
        print(f"\n  --- {int(nominal * 100)}% interval coverage (option a) ---")
        for position in SKILL_POSITIONS:
            pos_test = test.filter(pl.col("position") == position)
            if pos_test.is_empty():
                continue
            actual = pos_test["target"].to_numpy()
            b3_points_arr = pos_test["b3_points"].to_numpy()
            lower = np.maximum(b3_points_arr + error_quantiles[position][lower_tau], 0.0)
            upper = np.maximum(b3_points_arr + error_quantiles[position][upper_tau], 0.0)
            cov = _coverage(actual, lower, upper)
            print(
                f"    {position}: coverage={cov * 100:.1f}% "
                f"(nominal {int(nominal * 100)}%, off by {abs(cov - nominal) * 100:.1f}pp, "
                f"n={pos_test.height})"
            )

    # --- Option (b): v1's quantile spread, recentered on B3 ---------------------------
    print("\n=== Option (b): v1 quantile spread recentered on B3 ===")
    train_rows = features.filter(
        (pl.col("season") < DEV_SEASONS[0]) | (pl.col("season").is_in(DEV_SEASONS))
    ).filter(pl.col("season") >= settings.seasons.train_start)
    print(f"Single fit on real training rows through {DEV_SEASONS[-1]}: {train_rows.height} rows")

    availability_model = availability.fit_availability_model(
        train_rows, lightgbm_params=settings.model.lightgbm
    )
    quantile_model = quantiles.fit_quantile_models(
        train_rows, lightgbm_params=settings.model.lightgbm, quantile_alphas=DEFAULT_QUANTILES
    )
    print(f"Real crossing rate per position: {quantile_model.crossing_rate}")
    print(f"Real width_scale per position: {quantile_model.width_scale}")

    p_active = availability.predict_p_active(availability_model, test)
    conditional_quantiles = quantiles.predict_quantiles(quantile_model, test)
    unconditional = quantiles.mixture_with_p_active(
        conditional_quantiles, p_active, DEFAULT_QUANTILES
    )

    b3_mean = test["b3_points"].to_numpy()
    center = unconditional["unconditional_q_0.5"].to_numpy()
    delta = b3_mean - center

    recentered: dict[float, np.ndarray] = {}
    for tau in DEFAULT_QUANTILES:
        recentered[tau] = np.maximum(unconditional[f"unconditional_q_{tau}"].to_numpy() + delta, 0.0)

    test_positions = test["position"].to_numpy()
    actual_all = test["target"].to_numpy()

    for lower_tau, upper_tau, nominal in INTERVALS:
        print(f"\n  --- {int(nominal * 100)}% interval coverage (option b) ---")
        for position in SKILL_POSITIONS:
            mask = test_positions == position
            if not mask.any():
                continue
            cov = _coverage(actual_all[mask], recentered[lower_tau][mask], recentered[upper_tau][mask])
            print(
                f"    {position}: coverage={cov * 100:.1f}% "
                f"(nominal {int(nominal * 100)}%, off by {abs(cov - nominal) * 100:.1f}pp, "
                f"n={mask.sum()})"
            )


if __name__ == "__main__":
    main()
