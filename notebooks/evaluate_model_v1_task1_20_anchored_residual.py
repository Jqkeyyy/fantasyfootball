# notebooks/evaluate_model_v1_task1_20_anchored_residual.py
"""Real evaluation of task 1.20's anchored residual model
(SPEC-ADDENDUM-04.md §B) against real 2021-2024 data. Scratch, per
CLAUDE.md's notebooks/ convention -- not imported by anything under src/.

Two real walk-forward passes, both through the SAME
`evaluation.backtest.run_walk_forward_backtest` harness every other model
in this project uses (no cadence shortcut, no dev-only untuned config --
the shipped `config/settings.yml` LightGBM params throughout):

1. DEV_SEASONS (2018-2020, task 1.15's own established dev range) --
   real out-of-sample `anchored_residual`/`b2_ewm_4` predictions, used
   ONLY to fit each position's own blend weight `w`
   (`models.residual.fit_blend_weight`) via a Spearman-maximizing grid
   search. Never touched for reporting -- SPEC §12.5's "no reusing
   validation data" preserved by construction, the same season-range
   split `tune_points_v1.py` and this session's own ADDENDUM-04 §A
   diagnostics already established.
2. REPORT_SEASONS (2021-2024, task 1.15's own established validation
   range) -- real out-of-sample predictions, scored with the DEV-fitted
   `w` applied. 2025 stays fully held out, same precedent.

Reports, per ADDENDUM-04 §A.1's own finding (Spearman on ALL rows is the
wrong population -- SPEC §12.4 says startable rows only): Spearman on
BOTH the unrestricted population and the real startable-only population
(SPEC §9.4 fixed point), for `anchored_residual` (raw, unblended),
`anchored_residual_blend` (the real production candidate), `b2_ewm_4`,
and `consensus_b3` (if `data/interim/b3_predictions.parquet` exists --
this evaluation's own real B3 materialization, see
notebooks/_scratch_materialize_b3.py). Also reports MAE (for context)
and real SPEC §12.4 lineup regret for the same predictors, and the
fitted per-position `w` itself -- the honest readout of how much the
model actually contributes anywhere (ADDENDUM-04 §B's own explicit
requirement).
"""

from __future__ import annotations

import polars as pl

from ffapp.config import load_primary_league, load_settings
from ffapp.evaluation.backtest import BaselinePredictor, run_walk_forward_backtest
from ffapp.evaluation.metrics import (
    accuracy_metrics,
    lineup_regret,
    ranking_metrics,
    startable_counts_from_predictions,
)
from ffapp.league_format import parse_league_format
from ffapp.models import baselines, residual

DEV_SEASONS = [2018, 2019, 2020]
REPORT_SEASONS = [2021, 2022, 2023, 2024]
B3_PATH_NAME = "b3_predictions.parquet"


def _filter_startable(predictions: pl.DataFrame, startable_counts: dict[str, int]) -> pl.DataFrame:
    """Restricts to each real (season, week, position)'s own top-N by
    REAL `target`, N from the shared SPEC §9.4 fixed point --
    `evaluation.metrics._filter_startable`'s own logic (private, not
    exported), but that function is always called on ONE predictor's
    already-isolated rows first (`accuracy_metrics`'s own usage). This
    table stacks MULTIPLE predictors' rows for the same real player-week,
    so ranking directly over the stacked table would break real ties on
    identical `target` values across predictor copies via ordinal-rank's
    row-order tie-break -- silently picking a DIFFERENT "startable" row
    set per predictor even though the real startable population is
    predictor-independent (a real bug caught before it reached the
    reported numbers, not shipped). Fixed by computing the startable
    (season, week, position, player_id) key set from ONE reference
    predictor's own real target values, then joining that key set back
    onto every predictor's own rows -- the same "any_predictor" pattern
    `startable_counts_from_predictions` itself already uses."""
    reference = predictions.filter(pl.col("predictor") == "b2_ewm_4")
    ranked = reference.with_columns(
        pl.col("target")
        .rank(method="ordinal", descending=True)
        .over(["season", "week", "position"])
        .alias("_true_rank")
    )
    max_rank = pl.col("position").replace_strict(startable_counts, default=0, return_dtype=pl.Int64)
    startable_keys = ranked.filter(pl.col("_true_rank") <= max_rank).select(
        "player_id", "season", "week", "position"
    )
    return predictions.join(startable_keys, on=["player_id", "season", "week", "position"], how="inner")


def _print_spearman(predictions: pl.DataFrame, label: str) -> None:
    print(f"\n--- Spearman ({label}) ---")
    for result in ranking_metrics(predictions):
        if result.metric == "spearman":
            print(
                f"  {result.predictor} / {result.position}: rho={result.value:.4f} "
                f"(n={result.n_obs}, ci=[{result.ci_low:.4f}, {result.ci_high:.4f}])"
            )


def _print_mae(predictions: pl.DataFrame, label: str) -> None:
    print(f"\n--- MAE ({label}, all rows, pooled) ---")
    for result in accuracy_metrics(predictions):
        if result.metric == "mae" and result.position is None and result.scope == "all":
            print(
                f"  {result.predictor}: mae={result.value:.4f} "
                f"(n={result.n_obs}, ci=[{result.ci_low:.4f}, {result.ci_high:.4f}])"
            )


def main() -> None:
    settings = load_settings()
    schedule = pl.read_parquet(settings.data_root / "interim" / "schedule.parquet")
    features = pl.read_parquet(settings.data_root / "features" / "player_week_features.parquet")

    features = baselines.add_b2_ewm_4(features)
    features = residual.add_points_history_features(features)

    league = load_primary_league()
    fmt = parse_league_format(league)

    residual_predictors = [
        BaselinePredictor("b2_ewm_4", "b2_ewm_4"),
        residual.ResidualPredictor(settings.model.lightgbm),
    ]

    # --- Step 1: fit the blend weight on DEV_SEASONS (never reported on). ---
    print(f"Fitting blend weight on real dev seasons {DEV_SEASONS} (out-of-sample walk-forward)...")
    dev_predictions = run_walk_forward_backtest(
        features,
        schedule,
        residual_predictors,
        validation_seasons=DEV_SEASONS,
        train_start=settings.seasons.train_start,
        min_train_rows=settings.model.min_train_rows,
    )
    weight_by_position = residual.fit_blend_weight(dev_predictions)
    print(f"Fitted per-position blend weight w: {weight_by_position}")

    # --- Step 2: real out-of-sample predictions on REPORT_SEASONS. ---
    print(f"\nRunning real walk-forward backtest on report seasons {REPORT_SEASONS}...")
    report_predictions = run_walk_forward_backtest(
        features,
        schedule,
        residual_predictors,
        validation_seasons=REPORT_SEASONS,
        train_start=settings.seasons.train_start,
        min_train_rows=settings.model.min_train_rows,
    )
    blended = residual.apply_blend_weight(report_predictions, weight_by_position)
    all_predictions = pl.concat([report_predictions, blended], how="vertical_relaxed")

    # --- Step 3: fold in real B3 consensus, if materialized. ---
    b3_path = settings.data_root / "interim" / B3_PATH_NAME
    if b3_path.exists():
        b3 = pl.read_parquet(b3_path)
        lookup = report_predictions.filter(pl.col("predictor") == "b2_ewm_4").select(
            "player_id", "season", "week", "position", "team", "played", "target"
        )
        b3_rows = (
            lookup.join(b3, on=["player_id", "season", "week"], how="inner")
            .with_columns(
                pl.lit("consensus_b3").alias("predictor"), pl.col("b3_points").alias("prediction")
            )
            .select(all_predictions.columns)
            .drop_nulls("prediction")
        )
        print(f"\nFolded in {b3_rows.height} real consensus_b3 rows.")
        all_predictions = pl.concat([all_predictions, b3_rows], how="vertical_relaxed")
    else:
        print(f"\nNo {B3_PATH_NAME} found -- consensus_b3 omitted from this report.")

    # Cache the real out-of-sample predictions to scratch -- lets the
    # reporting step below be re-run (e.g. after fixing a reporting-only
    # bug) without re-paying the real walk-forward LightGBM refit cost.
    cache_path = settings.data_root / "outputs" / "_scratch_task1_20_predictions.parquet"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    all_predictions.write_parquet(cache_path)
    print(f"\nCached real predictions to {cache_path}")

    # --- Step 4: report. ---
    startable_counts = startable_counts_from_predictions(all_predictions, fmt)
    print(f"\nStartable pool sizes (SPEC SS9.4 fixed point): {startable_counts}")

    _print_spearman(all_predictions, "all rows")
    startable = _filter_startable(all_predictions, startable_counts)
    _print_spearman(startable, "startable rows only")

    _print_mae(all_predictions, "all rows")

    print("\n--- Lineup regret (SPEC SS12.4) ---")
    for result in lineup_regret(all_predictions, fmt):
        print(
            f"  {result.predictor}: regret={result.value:.4f} pts/week "
            f"(n_weeks={result.n_obs}, ci=[{result.ci_low:.4f}, {result.ci_high:.4f}])"
        )

    print(f"\nFitted per-position blend weight w (repeated for visibility): {weight_by_position}")


if __name__ == "__main__":
    main()
