"""Multiweek roster construction, bye, playoff, and target analysis."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import polars as pl


def player_values(
    ros: pl.DataFrame,
    names: Mapping[str, str],
    *,
    current_week: int,
    playoff_week_start: int,
) -> pl.DataFrame:
    """Aggregate weekly forecasts into near-term and playoff player values."""
    future = ros.filter(pl.col("week") >= current_week)
    if future.is_empty():
        return pl.DataFrame()
    return (
        future.group_by("player_id")
        .agg(
            pl.col("position").drop_nulls().first(),
            pl.col("team").drop_nulls().first(),
            pl.col("mean").sum().alias("ros_points"),
            pl.when(pl.col("week") < current_week + 4)
            .then(pl.col("mean"))
            .otherwise(0.0)
            .sum()
            .alias("next_4_points"),
            pl.when(pl.col("week") >= playoff_week_start)
            .then(pl.col("mean"))
            .otherwise(0.0)
            .sum()
            .alias("playoff_points"),
            pl.len().alias("games_remaining"),
        )
        .with_columns(
            pl.col("player_id")
            .replace_strict(names, default=pl.col("player_id"))
            .alias("player_name")
        )
    )


def roster_strategy(
    values: pl.DataFrame,
    roster_players: Mapping[str, Sequence[str]],
    my_team_id: str,
    starter_counts: Mapping[str, int],
    *,
    unavailable_player_ids: set[str] | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Return position health, acquisition targets, and trade chips."""
    if values.is_empty():
        return pl.DataFrame(), pl.DataFrame(), pl.DataFrame()
    owner_rows = [
        {"player_id": player_id, "team_id": team_id}
        for team_id, player_ids in roster_players.items()
        for player_id in player_ids
    ]
    owned = pl.DataFrame(owner_rows, schema={"player_id": pl.String, "team_id": pl.String})
    pool = values.join(owned, on="player_id", how="left")
    owned_pool = pool.drop_nulls("team_id").with_columns(
        pl.col("position").replace_strict(starter_counts, default=1).alias("starters_needed"),
        pl.col("next_4_points")
        .rank(method="ordinal", descending=True)
        .over("team_id", "position")
        .alias("position_rank"),
    )
    counts = owned_pool.group_by("team_id", "position").agg(pl.len().alias("roster_count"))
    team_position = (
        owned_pool.filter(pl.col("position_rank") <= pl.col("starters_needed"))
        .group_by("team_id", "position")
        .agg(pl.col("next_4_points").sum().alias("starter_value"))
        .join(counts, on=["team_id", "position"], how="left")
    )
    league_baseline = team_position.group_by("position").agg(
        pl.col("starter_value").median().alias("league_median"),
        pl.col("roster_count").median().alias("median_count"),
    )
    mine = league_baseline.join(
        team_position.filter(pl.col("team_id") == my_team_id),
        on="position",
        how="left",
    )
    missing_positions = set(starter_counts) - set(mine["position"].drop_nulls().to_list())
    if missing_positions:
        mine = pl.concat(
            [
                mine,
                league_baseline.filter(
                    pl.col("position").is_in(list(missing_positions))
                ).with_columns(
                    pl.lit(my_team_id).alias("team_id"),
                    pl.lit(0.0).alias("starter_value"),
                    pl.lit(0).alias("roster_count"),
                ),
            ],
            how="diagonal_relaxed",
        )
    health = (
        mine.with_columns(
            pl.col("position").replace_strict(starter_counts, default=1).alias("starters_needed"),
            (pl.col("starter_value").fill_null(0) - pl.col("league_median").fill_null(0)).alias(
                "edge"
            ),
        )
        .with_columns(
            pl.when(pl.col("edge") <= -8)
            .then(pl.lit("Priority need"))
            .when(pl.col("edge") < 4)
            .then(pl.lit("Average"))
            .otherwise(pl.lit("Strength"))
            .alias("status"),
            (pl.col("roster_count").fill_null(0) - pl.col("starters_needed") - 1)
            .clip(lower_bound=0)
            .alias("surplus"),
        )
        .sort("edge")
    )
    needs = health.filter(pl.col("status") == "Priority need")["position"].to_list()
    if not needs:
        needs = health.sort("edge").head(2)["position"].to_list()
    my_ids = set(roster_players.get(my_team_id, []))
    targets = (
        pool.filter(
            pl.col("position").is_in(needs)
            & ~pl.col("player_id").is_in(list(my_ids))
            & ~pl.col("player_id").is_in(list(unavailable_player_ids or set()))
        )
        .with_columns(
            pl.when(pl.col("team_id").is_null())
            .then(pl.lit("Waiver"))
            .otherwise(pl.lit("Trade"))
            .alias("path"),
            pl.when(pl.col("team_id").is_null())
            .then(pl.lit("Available"))
            .otherwise(pl.col("team_id"))
            .alias("owner"),
        )
        .sort(["position", "next_4_points"], descending=[False, True])
        .group_by("position", maintain_order=True)
        .head(8)
        .sort("next_4_points", descending=True)
    )
    surplus_positions = health.filter(pl.col("surplus") > 0)["position"].to_list()
    chips = (
        pool.filter(pl.col("team_id") == my_team_id)
        .filter(pl.col("position").is_in(surplus_positions))
        .sort(["position", "next_4_points"], descending=[False, False])
    )
    return health, targets, chips


def bye_pressure(
    ros: pl.DataFrame,
    my_player_ids: Sequence[str],
    *,
    current_week: int,
    final_week: int,
) -> pl.DataFrame:
    """Count roster players without a projection in each remaining week."""
    roster = ros.filter(pl.col("player_id").is_in(list(my_player_ids)))
    rows = []
    total = len(set(my_player_ids))
    for week in range(current_week, final_week + 1):
        playing = roster.filter((pl.col("week") == week) & (pl.col("mean") > 0))[
            "player_id"
        ].n_unique()
        rows.append(
            {
                "week": week,
                "players_available": playing,
                "players_without_game": max(0, total - playing),
            }
        )
    return pl.DataFrame(rows)


__all__ = ["bye_pressure", "player_values", "roster_strategy"]
