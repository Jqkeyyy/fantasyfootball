# notebooks/evaluate_efficiency_v2_stage3.py
"""One-off script: real evaluation of model v2 Stage 3 (efficiency
priors) against 2021-2025 data. Scratch, per CLAUDE.md's notebooks/
convention -- not imported by anything under src/.

Unlike Stage 2, this stage doesn't depend on Stage 1's or Stage 2's own
predictions -- SPEC's own Stage 3 input list names only "player
efficiency history" and "opponent adjusted rates," both already real,
already-available columns (models.efficiency.build_efficiency_table's
own job) -- so this script does not re-run any other stage's model
first.

Routes every predictor (the shrunk model AND both baselines) through
evaluation.backtest.run_walk_forward_backtest +
evaluation.backtest.BaselinePredictor from the start -- the exact
discipline Stage 2's own final review had to retrofit after a real
row-set-mismatch bug, applied proactively here. Filters to rows with a
real, defined per-touch outcome (>=1 real touch of that type that week)
BEFORE calling the harness -- the harness itself does not drop a null
target_column, and accuracy_metrics only filters on a null prediction,
not a null real outcome."""

import polars as pl

from ffapp.config import load_settings
from ffapp.evaluation.backtest import BaselinePredictor, run_walk_forward_backtest
from ffapp.evaluation.metrics import accuracy_metrics
from ffapp.features.usage import PASS_CATCHERS_AND_RB, RB_QB
from ffapp.models import efficiency

settings = load_settings()

# --- Load real data. ---
schedule = pl.read_parquet(settings.data_root / "interim" / "schedule.parquet")
player_week_features = pl.read_parquet(
    settings.data_root / "features" / "player_week_features.parquet"
)
player_week_usage = pl.read_parquet(settings.data_root / "interim" / "player_week_usage.parquet")
player_week_stats = pl.read_parquet(settings.data_root / "interim" / "player_week_stats.parquet")

# Real regular season only, upstream of everything else -- same
# convention Stage 1/2's own (corrected) evaluation scripts already
# established.
schedule = schedule.filter(pl.col("season_type") == "REG")
_reg_weeks = schedule.select("season", "week").unique()
player_week_features = player_week_features.join(_reg_weeks, on=["season", "week"], how="inner")
player_week_usage = player_week_usage.join(_reg_weeks, on=["season", "week"], how="inner")
player_week_stats = player_week_stats.join(_reg_weeks, on=["season", "week"], how="inner")

# --- Build Stage 3's own table. ---
table = efficiency.build_efficiency_table(
    player_week_features, player_week_usage, player_week_stats
)

# run_walk_forward_backtest needs availability_flag (task 1.9's own
# column, not produced by build_efficiency_table) -- joined in here,
# matching the same separation Stage 1/2 both keep between their own
# table-building functions and their evaluation scripts.
table = table.join(
    player_week_features.select("player_id", "season", "week", "availability_flag"),
    on=["player_id", "season", "week"],
    how="left",
)

validation_seasons = [2021, 2022, 2023, 2024, 2025]

# --- Score each output through the real walk-forward harness, once per
# target, with all three predictors (shrunk model, trailing_raw,
# league_mean) reading from the SAME position-eligible, real-outcome-
# defined input table -- so every predictor for a given target is built
# from the identical underlying rows from the start. NOTE: this does not
# mean every predictor's final SCORED n is identical --
# evaluation.metrics.accuracy_metrics still filters each predictor
# independently on a null `prediction` (standard, expected behavior,
# same precedent as Stage 1's own evaluation script), and trailing_raw
# is null for a player's own cold-start weeks in a way the shrinkage
# formula's degenerate case never is, so trailing_raw is scored on fewer
# rows than shrunk_model/league_mean -- see docs/JOURNAL.md's Stage 3
# entry for the real common-support numbers and why the headline result
# holds either way. ---
for target_column, eligible_positions in [
    ("yards_per_target", PASS_CATCHERS_AND_RB),
    ("td_rate_per_target", PASS_CATCHERS_AND_RB),
    ("yards_per_carry", RB_QB),
    ("td_rate_per_carry", RB_QB),
]:
    scoped = (
        table.filter(pl.col("position").is_in(eligible_positions))
        .filter(pl.col(f"real_{target_column}").is_not_null())
        .select(
            "player_id",
            "season",
            "week",
            "position",
            "team",
            "availability_flag",
            pl.col(f"real_{target_column}").alias(target_column),
            pl.col(f"expected_{target_column}").alias("_shrunk_model"),
            pl.col(f"trailing_raw_{target_column}").alias("_trailing_raw"),
            pl.col(f"league_mean_{target_column}").alias("_league_mean"),
        )
    )
    predictors = [
        BaselinePredictor(name="shrunk_model", column="_shrunk_model"),
        BaselinePredictor(name="trailing_raw", column="_trailing_raw"),
        BaselinePredictor(name="league_mean", column="_league_mean"),
    ]
    predictions = run_walk_forward_backtest(
        scoped,
        schedule,
        predictors,
        validation_seasons=validation_seasons,
        train_start=settings.seasons.train_start,
        min_train_rows=settings.model.min_train_rows,
        target_column=target_column,
    )
    print(f"\n=== {target_column} ===")
    for result in accuracy_metrics(predictions):
        if result.scope == "all" and result.position is None:
            print(
                f"{result.predictor}: {result.metric}={result.value:.4f} "
                f"(n={result.n_obs}, ci=[{result.ci_low:.4f}, {result.ci_high:.4f}])"
            )
