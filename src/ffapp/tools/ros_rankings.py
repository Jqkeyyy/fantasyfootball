"""Rest-of-season VOR over the CURRENT free-agent pool, and week-over-
week rank-change tracking (`SPEC-ADDENDUM-04.md` §D.3/§D.4/§D.5).
Reuses `tools.vor`'s already-shipped fixed point (task 0.9) unmodified --
the real difference from the preseason draft board is only which players
and which points column feed it: this module's real job is the scoping
(`tools.waivers.free_agent_pool`, task 2.6, reused directly rather than
recomputing "who's rostered" a second way) and the rank-over-time diff.
"""

from __future__ import annotations

import polars as pl

from ffapp.league_format import LeagueFormat
from ffapp.tools import vor
from ffapp.tools.waivers import free_agent_pool


def current_free_agent_projections(
    ros_points_table: pl.DataFrame,
    players_dim: pl.DataFrame,
    rostered_ids: set[str],
    eligible_positions: set[str],
) -> pl.DataFrame:
    """`ros_points_table` (`tools.ros_aggregate.aggregate_ros`'s own
    output -- `player_id, ros_points`, and today `ros_p10, ros_p50,
    ros_p90, expected_games, playoff_weeks_value`) scoped to real current
    free agents -- `tools.waivers.free_agent_pool`'s own already-shipped
    scoping, joined onto the real ROS points. Preserves every real column
    `ros_points_table` carries, not just a hardcoded subset."""
    pool = free_agent_pool(players_dim, rostered_ids, eligible_positions)
    joined = pool.join(ros_points_table, on="player_id", how="inner")
    extra_columns = [c for c in ros_points_table.columns if c != "player_id"]
    return joined.select("player_id", "position", *extra_columns)


def build_ros_board(
    ros_points_table: pl.DataFrame,
    players_dim: pl.DataFrame,
    rostered_ids: set[str],
    eligible_positions: set[str],
    league_format: LeagueFormat,
    *,
    replacement_overrides: dict[str, float] | None = None,
) -> pl.DataFrame:
    """SPEC §9.4's fixed point (`tools.vor.compute_vor`), replacement
    level computed over `ros_points_table`'s own real remaining-value
    scope and the CURRENT free-agent pool -- `SPEC-ADDENDUM-04.md` §D.3's
    own explicit correction to using August's preseason pool. Ranked by
    `vor_ros` descending, never by raw `ros_points` (§D.3: "never by raw
    projected points")."""
    scoped = current_free_agent_projections(
        ros_points_table, players_dim, rostered_ids, eligible_positions
    )
    with_vor = vor.compute_vor(
        scoped,
        league_format,
        points_column="ros_points",
        replacement_overrides=replacement_overrides,
    ).rename({"vor": "vor_ros"})
    return with_vor.sort("vor_ros", descending=True)


def rank_change(current_board: pl.DataFrame, previous_board: pl.DataFrame | None) -> pl.DataFrame:
    """`player_id, rank, rank_change` -- `rank_change` is
    `previous_rank - current_rank` (positive = moved up), null for a
    player with no real logged previous board (the first real run ever,
    or a genuinely new free agent this week) -- SPEC-ADDENDUM-04.md §D.5:
    "that last column is the one you will actually look at," so a
    misleading guessed value here would be the single worst mistake this
    function could make."""
    ranked = current_board.with_columns(
        pl.col("vor_ros").rank(method="ordinal", descending=True).cast(pl.Int64).alias("rank")
    ).select("player_id", "rank")
    if previous_board is None or previous_board.is_empty():
        return ranked.with_columns(pl.lit(None, dtype=pl.Int64).alias("rank_change"))

    previous_ranked = previous_board.with_columns(
        pl.col("vor_ros").rank(method="ordinal", descending=True).cast(pl.Int64).alias("rank")
    ).select("player_id", pl.col("rank").alias("_previous_rank"))

    return (
        ranked.join(previous_ranked, on="player_id", how="left")
        .with_columns((pl.col("_previous_rank") - pl.col("rank")).alias("rank_change"))
        .drop("_previous_rank")
    )


__all__ = ["build_ros_board", "current_free_agent_projections", "rank_change"]
