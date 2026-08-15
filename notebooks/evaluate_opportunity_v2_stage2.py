# notebooks/evaluate_opportunity_v2_stage2.py
"""One-off script: real evaluation of model v2 Stage 2 (opportunity)
against 2021-2025 data. Scratch, per CLAUDE.md's notebooks/ convention --
not imported by anything under src/.

Corrected per the final whole-branch review: routes every predictor
(the composition AND both baselines) through
`evaluation.backtest.run_walk_forward_backtest` +
`evaluation.backtest.BaselinePredictor`, exactly the way Stage 1's own
`evaluate_team_environment_v2_stage1.py` already does, so every predictor
is scored on the identical row set (real REG-season rows,
`validation_seasons=[2021..2025]`, position-eligible for that output)
instead of three different row sets reshaped independently
(`build_opportunity_table` emits every base-table row for all three
outputs regardless of position eligibility or validation window; the
prior version of this script reshaped that unfiltered table and let
`accuracy_metrics`'s own per-predictor `prediction.is_not_null()` filter
implicitly define each predictor's row set -- which is NOT the same row
set across predictors, since the composition is null outside its eligible
positions while the two baseline columns are computed, and thus non-null,
for every position and every season 2015-2025)."""

import polars as pl

from ffapp.config import DEFAULT_LIGHTGBM_SETTINGS, load_settings
from ffapp.evaluation.backtest import BaselinePredictor, run_walk_forward_backtest
from ffapp.evaluation.metrics import accuracy_metrics
from ffapp.features import team_context
from ffapp.features.usage import PASS_CATCHERS_AND_RB, RB_QB
from ffapp.models import opportunity, team_environment

settings = load_settings()

# --- Load real data. ---
team_week_context = pl.read_parquet(settings.data_root / "interim" / "team_week_context.parquet")
schedule = pl.read_parquet(settings.data_root / "interim" / "schedule.parquet")
snap_counts = pl.read_parquet(
    settings.data_root / "raw" / "nflverse" / "snap_counts_2015-2025.parquet"
)
injuries = pl.read_parquet(settings.data_root / "interim" / "injuries.parquet")
player_week_features = pl.read_parquet(
    settings.data_root / "features" / "player_week_features.parquet"
)
player_week_usage = pl.read_parquet(settings.data_root / "interim" / "player_week_usage.parquet")

# Real regular season only, upstream of everything else -- both
# player_week_features.parquet and player_week_usage.parquet carry real
# postseason weeks (19-22), confirmed present for real 2024 data. Same
# scoping tools.sos and Stage 1's own (corrected) evaluation script already
# established, applied proactively here from the start.
schedule = schedule.filter(pl.col("season_type") == "REG")
_reg_weeks = schedule.select("season", "week").unique()
team_week_context = team_week_context.join(_reg_weeks, on=["season", "week"], how="inner")
player_week_features = player_week_features.join(_reg_weeks, on=["season", "week"], how="inner")
player_week_usage = player_week_usage.join(_reg_weeks, on=["season", "week"], how="inner")

# --- Rebuild Stage 1's own table (same real data, same real construction
# its own evaluation script already uses) so its backtest can be re-run to
# capture its own out-of-sample predictions this time, not just accuracy. ---
usage_features_for_stage1 = player_week_features.select(
    "player_id", "season", "week", "team", "target_share_ewm_3", "carry_share_ewm_3"
)
team_context_features = team_context.build_team_context_features(
    team_week_context, schedule, snap_counts, injuries, usage_features_for_stage1
)
stage1_table = team_environment.build_team_environment_table(team_context_features)

validation_seasons = [2021, 2022, 2023, 2024, 2025]

# --- Re-run Stage 1's own backtest, once per target, keeping the full
# predictions this time (not just accuracy_metrics's summary). ---
stage1_predicted_team_plays = run_walk_forward_backtest(
    stage1_table,
    schedule,
    [
        team_environment.TeamEnvironmentPredictor(
            name="team_env_model",
            target_column="team_plays",
            lightgbm_params=DEFAULT_LIGHTGBM_SETTINGS,
        )
    ],
    validation_seasons=validation_seasons,
    train_start=settings.seasons.train_start,
    min_train_rows=settings.model.min_train_rows,
    target_column="team_plays",
).select(
    pl.col("player_id").alias("team"),
    "season",
    "week",
    pl.col("prediction").alias("predicted_team_plays"),
)

stage1_predicted_pass_rate = run_walk_forward_backtest(
    stage1_table,
    schedule,
    [
        team_environment.TeamEnvironmentPredictor(
            name="team_env_model",
            target_column="pass_rate",
            lightgbm_params=DEFAULT_LIGHTGBM_SETTINGS,
        )
    ],
    validation_seasons=validation_seasons,
    train_start=settings.seasons.train_start,
    min_train_rows=settings.model.min_train_rows,
    target_column="pass_rate",
).select(
    pl.col("player_id").alias("team"),
    "season",
    "week",
    pl.col("prediction").alias("predicted_pass_rate"),
)

stage1_predictions = stage1_predicted_team_plays.join(
    stage1_predicted_pass_rate, on=["team", "season", "week"], how="inner"
)
predicted_pass_attempts, predicted_rush_attempts = team_environment.derive_attempts(
    stage1_predictions["predicted_team_plays"], stage1_predictions["predicted_pass_rate"]
)
stage1_predictions = stage1_predictions.with_columns(
    predicted_pass_attempts.alias("predicted_pass_attempts"),
    predicted_rush_attempts.alias("predicted_rush_attempts"),
)

# --- Build Stage 2's own table and baselines. ---
table = opportunity.build_opportunity_table(
    player_week_features, player_week_usage, stage1_predictions
)
table = opportunity.add_opportunity_baselines(table)

# `run_walk_forward_backtest` needs `availability_flag` (task 1.9's own
# column, not produced by `build_opportunity_table`) -- joined in here
# rather than added to that module, matching the same separation Stage 1
# keeps between its own table-building functions and its evaluation script.
table = table.join(
    player_week_features.select("player_id", "season", "week", "availability_flag"),
    on=["player_id", "season", "week"],
    how="left",
)

# --- Score each output through the real walk-forward harness, once per
# target, with all three predictors (the composition AND both baselines)
# reading from the SAME position-eligibility-scoped features table -- so
# every predictor is scored on the identical row set, restricted to the
# same real validation window Stage 1 itself uses. Position eligibility
# matches `build_opportunity_table`'s own formula gating exactly (see that
# module's docstring): targets/rz_touches -> PASS_CATCHERS_AND_RB,
# carries -> RB_QB.
for target_column, composition_column, eligible_positions in [
    ("targets", "expected_targets", PASS_CATCHERS_AND_RB),
    ("carries", "expected_carries", RB_QB),
    ("rz_touches", "expected_rz_touches", PASS_CATCHERS_AND_RB),
]:
    scoped = table.filter(pl.col("position").is_in(eligible_positions)).select(
        "player_id",
        "season",
        "week",
        "position",
        "team",
        "availability_flag",
        target_column,
        pl.col(composition_column).alias("_opportunity_composition"),
        pl.col(f"{target_column}_b2_ewm_4").alias("_trailing_raw"),
        pl.col(f"{target_column}_league_mean").alias("_league_mean"),
    )
    predictors = [
        BaselinePredictor(name="opportunity_composition", column="_opportunity_composition"),
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
