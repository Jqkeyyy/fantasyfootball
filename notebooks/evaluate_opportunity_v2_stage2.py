# notebooks/evaluate_opportunity_v2_stage2.py
"""One-off script: real evaluation of model v2 Stage 2 (opportunity)
against 2021-2025 data. Scratch, per CLAUDE.md's notebooks/ convention --
not imported by anything under src/."""

import polars as pl

from ffapp.config import DEFAULT_LIGHTGBM_SETTINGS, load_settings
from ffapp.evaluation.backtest import run_walk_forward_backtest
from ffapp.evaluation.metrics import accuracy_metrics
from ffapp.features import team_context
from ffapp.models import opportunity, team_environment

settings = load_settings()

# --- Load real data. ---
team_week_context = pl.read_parquet(settings.data_root / "interim" / "team_week_context.parquet")
schedule = pl.read_parquet(settings.data_root / "interim" / "schedule.parquet")
snap_counts = pl.read_parquet(settings.data_root / "raw" / "nflverse" / "snap_counts_2015-2025.parquet")
injuries = pl.read_parquet(settings.data_root / "interim" / "injuries.parquet")
player_week_features = pl.read_parquet(settings.data_root / "features" / "player_week_features.parquet")
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
    pl.col("player_id").alias("team"), "season", "week", pl.col("prediction").alias("predicted_team_plays")
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
    pl.col("player_id").alias("team"), "season", "week", pl.col("prediction").alias("predicted_pass_rate")
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
table = opportunity.build_opportunity_table(player_week_features, player_week_usage, stage1_predictions)
table = opportunity.add_opportunity_baselines(table)

for target_column, composition_column in [
    ("targets", "expected_targets"),
    ("carries", "expected_carries"),
    ("rz_touches", "expected_rz_touches"),
]:
    predictions = opportunity.to_predictions_frame(
        table,
        real_column=target_column,
        composition_column=composition_column,
        trailing_raw_column=f"{target_column}_b2_ewm_4",
        league_mean_column=f"{target_column}_league_mean",
    )
    print(f"\n=== {target_column} ===")
    for result in accuracy_metrics(predictions):
        if result.scope == "all" and result.position is None:
            print(
                f"{result.predictor}: {result.metric}={result.value:.4f} "
                f"(n={result.n_obs}, ci=[{result.ci_low:.4f}, {result.ci_high:.4f}])"
            )
