"""Pure helpers for the week-by-week predictions vs actuals page."""

from __future__ import annotations

import polars as pl

SCORED_STATUS = "Scored"
INJURY_STATUS = "Injury excluded"

COMPARISON_SCHEMA = {
    "season": pl.Int64,
    "week": pl.Int64,
    "player_id": pl.String,
    "player": pl.String,
    "position": pl.String,
    "predicted": pl.Float64,
    "actual": pl.Float64,
    "difference": pl.Float64,
    "miss": pl.Float64,
    "injured": pl.Boolean,
}


def prediction_comparison(review: pl.DataFrame, *, include_injured: bool = False) -> pl.DataFrame:
    """One row per player-week with a verified pregame prediction and a real score.

    `difference` is actual minus predicted, so a positive value means the
    player beat the model. Confirmed in-game injuries are left out unless
    asked for, because no pregame forecast could have seen them coming.
    """
    if review.is_empty():
        return pl.DataFrame(schema=COMPARISON_SCHEMA)
    statuses = [SCORED_STATUS, INJURY_STATUS] if include_injured else [SCORED_STATUS]
    return (
        review.filter(
            pl.col("status").is_in(statuses)
            & pl.col("projected").is_not_null()
            & pl.col("actual").is_not_null()
        )
        .select(
            pl.col("season").cast(pl.Int64),
            pl.col("week").cast(pl.Int64),
            pl.col("player_id").cast(pl.String),
            pl.col("player").cast(pl.String),
            pl.col("position").cast(pl.String),
            pl.col("projected").cast(pl.Float64).alias("predicted"),
            pl.col("actual").cast(pl.Float64),
            (pl.col("actual") - pl.col("projected")).cast(pl.Float64).alias("difference"),
            (pl.col("actual") - pl.col("projected")).abs().cast(pl.Float64).alias("miss"),
            (pl.col("status") == INJURY_STATUS).alias("injured"),
        )
        .sort(["season", "week", "predicted"], descending=[True, True, True])
    )


def weekly_accuracy(comparison: pl.DataFrame) -> pl.DataFrame:
    """How close the model was each week, newest week first."""
    if comparison.is_empty():
        return pl.DataFrame(
            schema={
                "season": pl.Int64,
                "week": pl.Int64,
                "players": pl.UInt32,
                "average_miss": pl.Float64,
                "average_difference": pl.Float64,
                "within_five": pl.Float64,
                "predicted_total": pl.Float64,
                "actual_total": pl.Float64,
            }
        )
    return (
        comparison.group_by("season", "week")
        .agg(
            pl.len().alias("players"),
            pl.col("miss").mean().alias("average_miss"),
            pl.col("difference").mean().alias("average_difference"),
            (pl.col("miss") <= 5).mean().alias("within_five"),
            pl.col("predicted").sum().alias("predicted_total"),
            pl.col("actual").sum().alias("actual_total"),
        )
        .sort(["season", "week"], descending=True)
    )


def player_history(comparison: pl.DataFrame, player_id: str) -> pl.DataFrame:
    """One player's predicted and actual points in week order."""
    return comparison.filter(pl.col("player_id") == player_id).sort(["season", "week"])


__all__ = [
    "COMPARISON_SCHEMA",
    "player_history",
    "prediction_comparison",
    "weekly_accuracy",
]
