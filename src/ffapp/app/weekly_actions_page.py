"""Pure helpers for the weekly action cockpit."""

from __future__ import annotations

import math
from collections.abc import Iterable

import polars as pl

from ffapp.features.opponent import team_opponent
from ffapp.ids import mapping
from ffapp.league_format import LeagueFormat
from ffapp.projections.aggregate import add_join_key
from ffapp.sim.lineup import PlayerProjection, optimal_lineup
from ffapp.sim.season import SimPlayer
from ffapp.tools.waivers import build_waiver_board, value_added

_QUANTILE_ALPHAS = (0.10, 0.25, 0.50, 0.75, 0.90)


def _number(value: object) -> float:
    if not isinstance(value, int | float):
        raise TypeError(f"Expected a numeric projection value, got {value!r}")
    return float(value)


def explain_player(row: dict[str, object]) -> list[str]:
    """Turn a ranked row into factual, deliberately non-causal decision evidence."""
    mean = row.get("proj_mean")
    floor = row.get("floor")
    ceiling = row.get("ceiling")
    p_active = row.get("p_active")
    source = row.get("projection_source") or "unknown"
    explanations = [
        f"Projection: {_number(mean):.1f} points from {source}." if mean is not None else
        f"Projection: unavailable from {source}.",
    ]
    if floor is not None and ceiling is not None:
        explanations.append(
            f"Uncertainty: {_number(floor):.1f} floor to {_number(ceiling):.1f} ceiling "
            f"({_number(ceiling) - _number(floor):.1f}-point range)."
        )
    if p_active is not None:
        explanations.append(f"Availability estimate: {_number(p_active):.0%} chance to play.")
    grade = row.get("matchup_grade")
    sample = row.get("n_plays_behind_matchup_grade")
    opponent = row.get("opponent") or "unknown opponent"
    if grade is None:
        explanations.append(f"Matchup context: no supported grade for {opponent}. ")
    else:
        support = f" across {int(_number(sample))} plays" if sample is not None else ""
        explanations.append(
            f"Matchup context: grade {grade} versus {opponent}{support}; "
            "shown as context, not claimed as the projection's cause."
        )
    return explanations


def projection_supported_format(fmt: LeagueFormat, positions: set[str]) -> LeagueFormat:
    """Restrict a league format to positions covered by weekly projections."""
    return LeagueFormat(
        n_teams=fmt.n_teams,
        starters={
            position: count for position, count in fmt.starters.items() if position in positions
        },
        flex_slots=dict(fmt.flex_slots),
        flex_eligible={
            slot: [position for position in eligible if position in positions]
            for slot, eligible in fmt.flex_eligible.items()
        },
        bench=fmt.bench,
        ir=fmt.ir,
        playoff_week_start=fmt.playoff_week_start,
        waiver_budget=fmt.waiver_budget,
    )


def player_projections(rankings: pl.DataFrame, player_ids: Iterable[str]) -> list[PlayerProjection]:
    """Convert ranked rows for a roster into the lineup engine's input."""
    wanted = set(player_ids)
    rows = rankings.filter(
        pl.col("player_id").is_in(list(wanted))
        & pl.col("proj_mean").is_not_null()
        & pl.col("median").is_not_null()
        & pl.col("ceiling").is_not_null()
    ).iter_rows(named=True)
    return [
        PlayerProjection(
            player_id=str(row["player_id"]),
            position=str(row["position"]),
            mean=float(row["proj_mean"]),
            median=float(row["median"]),
            ceiling=float(row["ceiling"]),
        )
        for row in rows
    ]


def sim_players(rankings: pl.DataFrame, player_ids: Iterable[str]) -> list[SimPlayer]:
    """Convert ranked rows for a roster into start/sit simulation inputs."""
    wanted = set(player_ids)
    rows = rankings.filter(
        pl.col("player_id").is_in(list(wanted))
        & pl.col("proj_mean").is_not_null()
        & pl.col("floor").is_not_null()
        & pl.col("lower_quartile").is_not_null()
        & pl.col("median").is_not_null()
        & pl.col("upper_quartile").is_not_null()
        & pl.col("ceiling").is_not_null()
    ).iter_rows(named=True)
    return [
        SimPlayer(
            player_id=str(row["player_id"]),
            position=str(row["position"]),
            team=str(row["team"]),
            opponent_team=str(row["opponent"]) if row["opponent"] is not None else None,
            mean=float(row["proj_mean"]),
            alphas=_QUANTILE_ALPHAS,
            quantile_values=(
                float(row["floor"]),
                float(row["lower_quartile"]),
                float(row["median"]),
                float(row["upper_quartile"]),
                float(row["ceiling"]),
            ),
            p_miss=max(0.0, min(1.0, 1.0 - float(row["p_active"]))),
        )
        for row in rows
    ]


def recommended_lineup(
    rankings: pl.DataFrame,
    roster_ids: set[str],
    current_starter_ids: set[str],
    fmt: LeagueFormat,
) -> pl.DataFrame:
    """Return the projection-optimal supported lineup with current-start flags."""
    supported = projection_supported_format(fmt, set(rankings["position"].unique().to_list()))
    projections = player_projections(rankings, roster_ids)
    lineup = optimal_lineup(projections, supported)
    by_id = {row["player_id"]: row for row in rankings.iter_rows(named=True)}
    rows: list[dict[str, object]] = []
    for slot, player_id in lineup.slots.items():
        player = by_id[player_id]
        rows.append(
            {
                "slot": slot,
                "player_id": player_id,
                "player_name": player["player_name"],
                "position": player["position"],
                "team": player["team"],
                "opponent": player["opponent"],
                "projected_points": player["proj_mean"],
                "floor": player["floor"],
                "ceiling": player["ceiling"],
                "currently_starting": player_id in current_starter_ids,
            }
        )
    schema = {
        "slot": pl.String,
        "player_id": pl.String,
        "player_name": pl.String,
        "position": pl.String,
        "team": pl.String,
        "opponent": pl.String,
        "projected_points": pl.Float64,
        "floor": pl.Float64,
        "ceiling": pl.Float64,
        "currently_starting": pl.Boolean,
    }
    return pl.DataFrame(rows, schema=schema) if rows else pl.DataFrame(schema=schema)


def waiver_recommendations(
    rankings: pl.DataFrame,
    roster_ids: set[str],
    fmt: LeagueFormat,
    *,
    current_week: int,
    remaining_budget: int,
    playoff_weight: float,
    aggressiveness: float,
    limit: int = 15,
    candidates_per_position: int = 10,
    ros_projections: pl.DataFrame | None = None,
    opponent_roster_ids: list[set[str]] | None = None,
) -> pl.DataFrame:
    """Rank current free agents by value relative to the user's roster."""
    supported = projection_supported_format(fmt, set(rankings["position"].unique().to_list()))
    projected = rankings.filter(pl.col("proj_mean").is_not_null())
    projection_by_player = dict(
        zip(projected["player_id"].to_list(), projected["proj_mean"].to_list(), strict=True)
    )
    projection_basis = "current_week"
    if ros_projections is not None and not ros_projections.is_empty():
        future = ros_projections.filter(pl.col("week") >= current_week)
        if (
            not future.is_empty()
            and int(_number(future.select(pl.col("week").min()).item())) == current_week
        ):
            weighted = future.with_columns(
                pl.when(pl.col("week") >= fmt.playoff_week_start)
                .then(pl.lit(playoff_weight))
                .otherwise(pl.lit(1.0))
                .alias("_weight")
            ).group_by("player_id").agg(
                ((pl.col("mean") * pl.col("_weight")).sum() / pl.col("_weight").sum()).alias(
                    "weighted_mean"
                )
            )
            projection_by_player.update(
                dict(zip(weighted["player_id"], weighted["weighted_mean"], strict=True))
            )
            projection_basis = "multiweek_ros"

    def roster_projections(ids: set[str]) -> list[PlayerProjection]:
        rows = rankings.filter(pl.col("player_id").is_in(list(ids))).iter_rows(named=True)
        return [
            PlayerProjection(
                player_id=str(row["player_id"]),
                position=str(row["position"]),
                mean=float(projection_by_player[str(row["player_id"])]),
                median=float(projection_by_player[str(row["player_id"])]),
                ceiling=float(projection_by_player[str(row["player_id"])]),
            )
            for row in rows
            if str(row["player_id"]) in projection_by_player
        ]

    roster = roster_projections(roster_ids)
    free_agents = (
        projected.filter(pl.col("owner_status") == "free_agent")
        .sort("proj_mean", descending=True)
        .group_by("position", maintain_order=True)
        .head(candidates_per_position)
        .select(pl.col("player_id").alias("sleeper_id"), "player_id", "position")
    )
    board = build_waiver_board(
        free_agents,
        projection_by_player,
        roster,
        supported,
        current_week=current_week,
        remaining_budget=remaining_budget,
        playoff_weight=playoff_weight,
        aggressiveness=aggressiveness,
    )
    names = rankings.select("player_id", "player_name", "team", "opponent", "proj_mean")
    drop_names = rankings.select(
        pl.col("player_id").alias("drop_candidate"),
        pl.col("player_name").alias("drop_player"),
    )
    opponents = [roster_projections(ids) for ids in (opponent_roster_ids or [])]
    competition_rows: list[dict[str, object]] = []
    for player_id in board["player_id"].to_list():
        row = rankings.filter(pl.col("player_id") == player_id).row(0, named=True)
        mean = _number(projection_by_player[str(player_id)])
        candidate = PlayerProjection(str(player_id), str(row["position"]), mean, mean, mean)
        opponent_values = [value_added(team, candidate, fmt)[0] for team in opponents]
        interested = sum(value > 0 for value in opponent_values)
        competition_rows.append(
            {
                "player_id": player_id,
                "competing_teams": interested,
                "max_opponent_need": max(opponent_values, default=0.0),
            }
        )
    competition = pl.DataFrame(
        competition_rows,
        schema={
            "player_id": pl.String,
            "competing_teams": pl.Int64,
            "max_opponent_need": pl.Float64,
        },
    )
    result = (
        board.join(names, on="player_id", how="left")
        .join(drop_names, on="drop_candidate", how="left")
        .join(competition, on="player_id", how="left")
        .filter(pl.col("value_added_per_week") > 0)
        .head(limit)
    )
    return result.with_columns(
        pl.struct("suggested_bid", "competing_teams")
        .map_elements(
            lambda row: min(
                remaining_budget,
                math.ceil(
                    _number(row["suggested_bid"])
                    * (1.0 + 0.1 * _number(row["competing_teams"]))
                ),
            ),
            return_dtype=pl.Int64,
        )
        .alias("suggested_bid"),
        pl.lit(projection_basis).alias("projection_basis"),
    )


def streaming_recommendations(
    weekly_points: pl.DataFrame,
    players_dim: pl.DataFrame,
    schedule: pl.DataFrame,
    rostered_sleeper_ids: set[str],
    *,
    season: int,
    week: int,
    limit_per_position: int = 5,
) -> pl.DataFrame:
    """Rank available K/DST options from league-scored current-week projections."""
    current = weekly_points.filter(
        (pl.col("season") == season)
        & (pl.col("week") == week)
        & pl.col("position").is_in(["K", "DST"])
    )
    kickers = add_join_key(current.filter(pl.col("position") == "K")).join(
        mapping.dedupe_to_one_row_per_name_position(players_dim).select(
            "join_key", "player_id", "sleeper_id"
        ),
        on="join_key",
        how="left",
    )
    defenses = current.filter(pl.col("position") == "DST").with_columns(
        pl.col("team").alias("player_id"),
        pl.col("team").alias("sleeper_id"),
    )
    candidates = pl.concat(
        [
            kickers.select(
                "player_id", "sleeper_id", "player_name", "position", "team", "points"
            ),
            defenses.select(
                "player_id", "sleeper_id", "player_name", "position", "team", "points"
            ),
        ],
        how="vertical_relaxed",
    ).filter(
        pl.col("sleeper_id").is_not_null()
        & ~pl.col("sleeper_id").is_in(list(rostered_sleeper_ids))
    )
    opponents = (
        team_opponent(schedule)
        .filter((pl.col("season") == season) & (pl.col("week") == week))
        .select("team", "opponent")
    )
    return (
        candidates.join(opponents, on="team", how="left")
        .sort(["position", "points"], descending=[False, True])
        .group_by("position", maintain_order=True)
        .head(limit_per_position)
        .with_columns(pl.int_range(1, pl.len() + 1).over("position").alias("rank"))
        .select("rank", "player_name", "position", "team", "opponent", "points")
    )


__all__ = [
    "explain_player",
    "player_projections",
    "projection_supported_format",
    "recommended_lineup",
    "sim_players",
    "streaming_recommendations",
    "waiver_recommendations",
]
