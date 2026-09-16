"""Pure data preparation for the chopped-player bid calculator."""

from __future__ import annotations

import polars as pl

from ffapp.config import LeagueConfig


def is_chopped_league(league: LeagueConfig) -> bool:
    return bool(
        league.league_cache.get("league_type") == 3
        or league.league_cache.get("disable_trades")
    )


def build_player_values(
    weekly_projections: pl.DataFrame,
    ros_projections: pl.DataFrame | None,
    players_dim: pl.DataFrame,
    *,
    season: int,
    week: int,
) -> pl.DataFrame:
    """Build one canonical projection row per Sleeper player."""
    identities = (
        players_dim.filter(pl.col("sleeper_id").is_not_null())
        .sort("search_rank")
        .unique(subset=["player_id"], keep="first")
        .select(
            "player_id",
            "sleeper_id",
            pl.col("full_name").alias("player_name"),
            "position",
            "team",
        )
    )
    current = (
        weekly_projections.filter(
            (pl.col("season") == season)
            & (pl.col("week") == week)
            & pl.col("mean").is_not_null()
        )
        .select("player_id", pl.col("mean").alias("current_week_projection"))
        .unique(subset=["player_id"], keep="last")
    )
    values = identities.join(current, on="player_id", how="inner")
    if ros_projections is not None and not ros_projections.is_empty():
        ros = (
            ros_projections.filter(
                (pl.col("season") == season)
                & (pl.col("week") >= week)
                & pl.col("mean").is_not_null()
            )
            .group_by("player_id")
            .agg(pl.col("mean").mean().alias("ros_projection_ppg"))
        )
        values = values.join(ros, on="player_id", how="left")
    else:
        values = values.with_columns(pl.lit(None, dtype=pl.Float64).alias("ros_projection_ppg"))
    return values.with_columns(
        pl.coalesce("ros_projection_ppg", "current_week_projection").alias("projection_ppg")
    )


__all__ = ["build_player_values", "is_chopped_league"]
