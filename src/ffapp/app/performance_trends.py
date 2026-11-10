"""Weekly matched-sample trends from verified, saved pregame predictions."""

from __future__ import annotations

import polars as pl

POSITIONS = ("QB", "RB", "WR", "TE", "DEFENSE", "K")


def performance_trends(
    review: pl.DataFrame, *, cohort: str = "Everything", include_injured: bool = True
) -> pl.DataFrame:
    """Rank before filtering outcomes; never replace missing top-100 results with lower ranks."""
    if review.is_empty():
        return pl.DataFrame()
    rows = review.filter(
        pl.col("status").is_in(
            ["Scored", "Injury excluded", "Awaiting actuals", "Awaiting real actuals"]
        )
        & pl.col("projected").is_finite()
    ).with_columns(
        pl.col("position")
        .str.to_uppercase()
        .replace({"DEF": "DEFENSE", "DST": "DEFENSE", "D/ST": "DEFENSE"})
    )
    if cohort == "Top 100":
        rows = (
            rows.sort(
                ["season", "week", "projected", "player_id"],
                descending=[False, False, True, False],
            )
            .group_by("season", "week", maintain_order=True)
            .head(100)
        )
    elif cohort in POSITIONS:
        rows = rows.filter(pl.col("position") == cohort)
    elif cohort != "Everything":
        raise ValueError(f"Unknown cohort: {cohort}")
    if rows.is_empty():
        return pl.DataFrame()
    statuses = ["Scored", "Injury excluded"] if include_injured else ["Scored"]
    rows = (
        rows.with_columns(
            (pl.col("status").is_in(statuses) & pl.col("actual").is_finite()).alias("matched")
        )
        .with_columns(
            pl.when("matched").then("projected").alias("prediction"),
            pl.when("matched").then("actual").alias("outcome"),
        )
        .with_columns((pl.col("prediction") - pl.col("outcome")).alias("error"))
    )
    return (
        rows.group_by("season", "week")
        .agg(
            pl.len().alias("selected"),
            pl.col("matched").sum().alias("matched"),
            pl.col("prediction").mean().alias("predicted_average"),
            pl.col("outcome").mean().alias("actual_average"),
            pl.when(pl.col("matched").sum() > 0)
            .then(pl.col("prediction").sum())
            .alias("predicted_total"),
            pl.when(pl.col("matched").sum() > 0)
            .then(pl.col("outcome").sum())
            .alias("actual_total"),
            pl.col("error").abs().mean().alias("average_miss"),
            pl.col("error").mean().alias("bias"),
        )
        .sort("season", "week")
        .with_columns(
            (pl.col("selected") - pl.col("matched")).alias("unscored_or_excluded"),
            pl.col("matched").truediv(pl.col("selected")).alias("coverage"),
            (
                pl.col("season").cast(pl.String)
                + " W"
                + pl.col("week").cast(pl.String).str.pad_start(2, "0")
            ).alias("Week"),
        )
    )
