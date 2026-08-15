# notebooks/evaluate_team_environment_v2_stage1.py
"""One-off script: real walk-forward evaluation of model v2 Stage 1 against
2021-2025 data. Scratch, per CLAUDE.md's notebooks/ convention -- not
imported by anything under src/."""

import polars as pl

from ffapp.config import DEFAULT_LIGHTGBM_SETTINGS, load_settings
from ffapp.evaluation.backtest import BaselinePredictor, run_walk_forward_backtest
from ffapp.evaluation.metrics import accuracy_metrics
from ffapp.features import team_context
from ffapp.models import team_environment

settings = load_settings()

team_week_context = pl.read_parquet(settings.data_root / "interim" / "team_week_context.parquet")
schedule = pl.read_parquet(settings.data_root / "interim" / "schedule.parquet")
snap_counts = pl.read_parquet(settings.data_root / "raw" / "nflverse" / "snap_counts_2015-2025.parquet")
injuries = pl.read_parquet(settings.data_root / "interim" / "injuries.parquet")
usage_features_path = settings.data_root / "features" / "player_week_features.parquet"
usage_features = pl.read_parquet(usage_features_path).select(
    "player_id", "season", "week", "team", "target_share_ewm_3", "carry_share_ewm_3"
)

# Real regular season only, upstream of everything else: nflverse numbers
# real postseason weeks 19-22 (WC/DIV/CON/SB), and team_week_context carries
# real rows for them too -- no fantasy league plays through them, so they
# must not leak into training or validation. `schedule.season_type` is the
# real source of this ("REG"/"POST" etc, `tools.sos` already established
# this exact scoping convention for the same reason). `team_week_context`
# itself has no `season_type` column, so it's scoped by joining against
# `schedule`'s own real (season, week) pairs rather than hardcoding a week
# cutoff (CLAUDE.md rule 5).
schedule = schedule.filter(pl.col("season_type") == "REG")
_reg_weeks = schedule.select("season", "week").unique()
team_week_context = team_week_context.join(_reg_weeks, on=["season", "week"], how="inner")

team_context_features = team_context.build_team_context_features(
    team_week_context, schedule, snap_counts, injuries, usage_features
)
table = team_environment.build_team_environment_table(team_context_features)
table = team_environment.add_team_environment_baselines(table)

validation_seasons = [2021, 2022, 2023, 2024, 2025]

for target_column, baseline_league_col, baseline_b2_col in [
    ("team_plays", "team_plays_league_mean", "team_plays_b2_ewm_4"),
    ("pass_rate", "pass_rate_league_mean", "pass_rate_b2_ewm_4"),
]:
    predictors = [
        team_environment.TeamEnvironmentPredictor(
            name="team_env_model",
            target_column=target_column,
            lightgbm_params=DEFAULT_LIGHTGBM_SETTINGS,
        ),
        BaselinePredictor(name="league_mean", column=baseline_league_col),
        BaselinePredictor(name="trailing_ewm_4", column=baseline_b2_col),
    ]
    predictions = run_walk_forward_backtest(
        table,
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
