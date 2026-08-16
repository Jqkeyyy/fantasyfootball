"""Multi-week ROS projection composer (`SPEC-ADDENDUM-04.md` §D.1's
amended horizon split, `docs/JOURNAL.md`'s 2026-08-16 entry;
TASKS.md 1.21). Writes `outputs/<league_slug>/projections_ros.parquet`
(task-level acceptance: every row carries `as_of_utc`).

Deliberately thin -- every real piece of math already lives in
`models.predict` (current week, unchanged), `models.ros_consensus`
(season-long level), and `models.ros_shape` (weekly shape). This module's
only real job is the seam between them: call `project_week` exactly once
for the anchor week, resolve/aggregate/allocate exactly once for every
remaining week at once (not once per week -- the season-long consensus
fetch is one real network round-trip per source, not `through_week -
from_week` of them), and combine both into one output schema.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

import polars as pl

from ffapp.config import DEFAULT_LIGHTGBM_SETTINGS, LightGBMSettings, Settings
from ffapp.ids import mapping
from ffapp.models import baselines, predict, ros_consensus, ros_shape

OUTPUT_COLUMNS = [
    "player_id",
    "season",
    "week",
    "position",
    "team",
    "opponent_team",
    "mean",
    "q10",
    "q25",
    "q50",
    "q75",
    "q90",
    "is_current_week",
    "as_of_utc",
]

_OUTPUT_SCHEMA = {
    "player_id": pl.String,
    "season": pl.Int64,
    "week": pl.Int64,
    "position": pl.String,
    "team": pl.String,
    "opponent_team": pl.String,
    "mean": pl.Float64,
    "q10": pl.Float64,
    "q25": pl.Float64,
    "q50": pl.Float64,
    "q75": pl.Float64,
    "q90": pl.Float64,
    "is_current_week": pl.Boolean,
    "as_of_utc": pl.String,
}

_Q_COLUMN_NAMES = {0.10: "q10", 0.25: "q25", 0.50: "q50", 0.75: "q75", 0.90: "q90"}


def _current_week_rows(
    features: pl.DataFrame,
    season: int,
    week: int,
    players_dim: pl.DataFrame,
    b3_historical: pl.DataFrame,
    *,
    train_start: int,
    min_train_rows: int,
    lightgbm_params: LightGBMSettings,
    code_version: str | None,
    now: datetime,
    quantile_alphas: Sequence[float],
    offline: bool | None,
    settings: Settings | None,
) -> pl.DataFrame:
    result = predict.project_week(
        features,
        season,
        week,
        train_start=train_start,
        min_train_rows=min_train_rows,
        lightgbm_params=lightgbm_params,
        code_version=code_version,
        now=now,
        quantile_alphas=quantile_alphas,
        projection_source="consensus_b3",
        players_dim=players_dim,
        b3_historical=b3_historical,
        offline=offline,
        settings=settings,
    )
    if result.is_empty():
        return pl.DataFrame(schema=_OUTPUT_SCHEMA)
    # Both `team` and `position` come from `features` -- `project_week`'s
    # own output (`_current_week_rows`'s `result`) never carries either
    # (it only ever needs `player_id`/`season`/`week` to key its own
    # internal joins), so both must be attached here, not just `team`.
    team_by_player = features.filter(
        (pl.col("season") == season) & (pl.col("week") == week)
    ).select("player_id", "team", "position")
    return (
        result.join(team_by_player, on="player_id", how="left")
        .with_columns(
            pl.lit(None, dtype=pl.String).alias("opponent_team"),
            pl.lit(True).alias("is_current_week"),
        )
        .select(*OUTPUT_COLUMNS[:-1], "as_of_utc")
    )


def project_week_range(
    features: pl.DataFrame,
    schedule: pl.DataFrame,
    defense_position_allowed: pl.DataFrame,
    season: int,
    from_week: int,
    through_week: int,
    league_slug: str,
    scoring_settings: dict[str, float],
    players_dim: pl.DataFrame,
    b3_historical: pl.DataFrame,
    actuals_to_date: pl.DataFrame,
    season_points_by_source: dict[str, pl.DataFrame],
    trend_by_source: dict[str, str],
    quantile_alphas: Sequence[float],
    now: datetime,
    *,
    train_start: int,
    min_train_rows: int,
    lightgbm_params: LightGBMSettings | None,
    code_version: str | None,
    offline: bool | None,
    settings: Settings | None,
) -> pl.DataFrame:
    """`season_points_by_source` is `models.ros_consensus.fetch_season_consensus`'s
    own real output, fetched exactly once by the caller (CLI/log job) for
    this whole horizon -- passed in rather than fetched here, so this
    function stays a pure composer, testable without a real network call.
    """
    # `predict.project_week` (and everything underneath it -- the
    # availability model is fit unconditionally, task 1.14/1.16) requires
    # a real `LightGBMSettings`, never `None`, for every real caller
    # elsewhere in this codebase (`cli.py`, `tools.prediction_log`).
    # `None` is accepted at this function's own boundary only so tests can
    # mock `project_week` out entirely without constructing one; resolved
    # to the project's own real default the moment it's actually needed.
    resolved_lightgbm_params = (
        lightgbm_params if lightgbm_params is not None else DEFAULT_LIGHTGBM_SETTINGS
    )
    current = _current_week_rows(
        features,
        season,
        from_week,
        players_dim,
        b3_historical,
        train_start=train_start,
        min_train_rows=min_train_rows,
        lightgbm_params=resolved_lightgbm_params,
        code_version=code_version,
        now=now,
        quantile_alphas=quantile_alphas,
        offline=offline,
        settings=settings,
    )
    if current.is_empty():
        return pl.DataFrame(schema=_OUTPUT_SCHEMA)

    future_weeks = list(range(from_week + 1, through_week + 1))
    if not future_weeks:
        return current

    resolved = ros_consensus.resolve_remaining_value(
        season_points_by_source, trend_by_source, actuals_to_date
    )
    # `join_key` only exists after `dedupe_to_one_row_per_name_position`
    # (confirmed: `ids.mapping.build_players_dim`'s own raw output has no
    # such column -- `add_b3_fp_weekly_consensus` already established this
    # exact call as the real way to get one, reused here rather than a
    # second name-normalization scheme).
    keyed_players_dim = mapping.dedupe_to_one_row_per_name_position(players_dim)
    with_identity = resolved.join(
        keyed_players_dim.select("player_id", "join_key", pl.col("full_name").alias("player_name")),
        on="player_id",
        how="left",
    )
    aggregated = ros_consensus.aggregate_remaining_value(with_identity)
    level_by_player = (
        with_identity.select("player_id", "join_key", "position", "team")
        .unique(subset=["player_id"])
        .join(
            aggregated.select("join_key", "season_consensus_ros_points"), on="join_key", how="inner"
        )
    )

    train_rows_with_b3 = features.filter(
        (pl.col("season") < season) | ((pl.col("season") == season) & (pl.col("week") < from_week))
    ).join(
        b3_historical.select("player_id", "season", "week", "b3_points"),
        on=["player_id", "season", "week"],
        how="inner",
    )
    error_quantiles = baselines.empirical_error_quantiles(
        train_rows_with_b3, "b3_points", quantile_alphas
    )

    dpa_groups: dict[str, pl.DataFrame] = {}
    for group in defense_position_allowed["position_group"].unique().to_list():
        dpa_groups[group] = ros_shape.frozen_defense_ratings(
            defense_position_allowed, season=season, as_of_week=from_week, position_group=group
        )

    future_frames: list[pl.DataFrame] = []
    for row in level_by_player.iter_rows(named=True):
        weeks_with_opponents = ros_shape.future_week_opponents(
            schedule, season=season, team=row["team"], weeks=future_weeks
        )
        if weeks_with_opponents.is_empty():
            continue
        allocated = ros_shape.allocate_season_consensus(
            row["season_consensus_ros_points"],
            row["position"],
            row["team"],
            weeks_with_opponents,
            dpa_groups,
        )
        allocated = allocated.join(weeks_with_opponents, on="week", how="left")
        for tau, column_name in _Q_COLUMN_NAMES.items():
            offset = error_quantiles.get(row["position"], {}).get(tau, 0.0)
            allocated = allocated.with_columns(
                (pl.col("mean") + offset).clip(lower_bound=0.0).alias(column_name)
            )
        future_frames.append(
            allocated.with_columns(
                pl.lit(row["player_id"]).alias("player_id"),
                pl.lit(season).alias("season"),
                pl.lit(row["position"]).alias("position"),
                pl.lit(row["team"]).alias("team"),
                pl.lit(False).alias("is_current_week"),
                pl.lit(now.isoformat()).alias("as_of_utc"),
            ).select(*OUTPUT_COLUMNS)
        )

    future = (
        pl.concat(future_frames, how="vertical_relaxed")
        if future_frames
        else pl.DataFrame(schema=_OUTPUT_SCHEMA)
    )
    return pl.concat([current, future], how="vertical_relaxed")


__all__ = ["OUTPUT_COLUMNS", "project_week_range"]
