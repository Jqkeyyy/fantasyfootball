"""Pure data adapters for the lineup-aware trade analyzer page."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import polars as pl

from ffapp.sim.season import Matchup, Roster, SimPlayer


def build_trade_rosters(
    projections_ros: pl.DataFrame,
    roster_players: Mapping[str, Sequence[str]],
    *,
    from_week: int,
) -> tuple[list[Roster], dict[str, float]]:
    """Convert weekly ROS projections into one lineup-aware player per roster."""
    future = projections_ros.filter(pl.col("week") >= from_week)
    if future.is_empty():
        return [], {}
    players = future.group_by("player_id").agg(
        pl.col("position").drop_nulls().first(),
        pl.col("team").drop_nulls().first(),
        pl.col("opponent_team").drop_nulls().first(),
        pl.col("mean").mean(),
        pl.col("q10").mean(),
        pl.col("q25").mean(),
        pl.col("q50").mean(),
        pl.col("q75").mean(),
        pl.col("q90").mean(),
        pl.col("mean").sum().alias("ros_points"),
    )
    by_id = {str(row["player_id"]): row for row in players.iter_rows(named=True)}
    rosters: list[Roster] = []
    for team_id, player_ids in roster_players.items():
        sim_players: list[SimPlayer] = []
        for player_id in player_ids:
            row = by_id.get(player_id)
            if row is None:
                continue
            sim_players.append(
                SimPlayer(
                    player_id=player_id,
                    position=str(row["position"]),
                    team=str(row["team"]),
                    opponent_team=(
                        str(row["opponent_team"]) if row["opponent_team"] is not None else None
                    ),
                    mean=float(row["mean"]),
                    alphas=(0.10, 0.25, 0.50, 0.75, 0.90),
                    quantile_values=tuple(
                        float(row[name]) for name in ("q10", "q25", "q50", "q75", "q90")
                    ),
                    p_miss=0.0,
                )
            )
        rosters.append(Roster(team_id=str(team_id), players=sim_players))

    replacement = players.group_by("position").agg(pl.col("ros_points").median().alias("base"))
    baseline = dict(zip(replacement["position"], replacement["base"], strict=True))
    vor = {
        player_id: max(0.0, float(row["ros_points"]) - float(baseline[str(row["position"])]))
        for player_id, row in by_id.items()
    }
    return rosters, vor


def matchup_schedule(
    payload_by_week: Mapping[int, Sequence[Mapping[str, object]]],
) -> list[Matchup]:
    """Convert Sleeper matchup rows into simulator pairings."""
    schedule: list[Matchup] = []
    for week, rows in sorted(payload_by_week.items()):
        grouped: dict[object, list[str]] = {}
        for row in rows:
            matchup_id = row.get("matchup_id")
            roster_id = row.get("roster_id")
            if matchup_id is not None and roster_id is not None:
                grouped.setdefault(matchup_id, []).append(str(roster_id))
        for team_ids in grouped.values():
            if len(team_ids) == 2:
                schedule.append(Matchup(week=week, home=team_ids[0], away=team_ids[1]))
    return schedule


__all__ = ["build_trade_rosters", "matchup_schedule"]
