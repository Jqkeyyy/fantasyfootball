"""Pure helpers for the weekly action cockpit."""

from __future__ import annotations

import math
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any

import polars as pl

from ffapp.app.weekly_rankings_page import UNAVAILABLE_STATUSES
from ffapp.features.opponent import team_opponent
from ffapp.ids import mapping
from ffapp.league_format import LeagueFormat
from ffapp.projections.aggregate import add_join_key
from ffapp.sim.lineup import (
    PlayerProjection,
    legal_swap_pairs,
    optimal_lineup,
    optimal_lineup_points,
)
from ffapp.sim.season import SimPlayer
from ffapp.tools.waivers import build_waiver_board, value_added

_QUANTILE_ALPHAS = (0.10, 0.25, 0.50, 0.75, 0.90)

ACTION_INBOX_SCHEMA = {
    "priority": pl.String,
    "category": pl.String,
    "action": pl.String,
    "expected_gain": pl.Float64,
    "confidence": pl.Float64,
    "confidence_label": pl.String,
    "evidence": pl.String,
    "risk": pl.String,
}

LINEUP_DECISION_SCHEMA = {
    "slot": pl.String,
    "start": pl.String,
    "sit": pl.String,
    "expected_gain": pl.Float64,
    "start_floor": pl.Float64,
    "sit_floor": pl.Float64,
    "start_ceiling": pl.Float64,
    "sit_ceiling": pl.Float64,
    "confidence": pl.Float64,
    "confidence_label": pl.String,
    "lock_time": pl.String,
    "why": pl.String,
    "risk": pl.String,
}


def _number(value: object) -> float:
    if not isinstance(value, int | float):
        raise TypeError(f"Expected a numeric projection value, got {value!r}")
    return float(value)


def recommendation_confidence(
    *,
    expected_gain: float,
    floor: float,
    ceiling: float,
    p_active: float,
) -> tuple[float, str]:
    """Score decision support from edge, availability, and projection width."""
    spread = max(0.0, ceiling - floor)
    edge_score = min(1.0, max(0.0, expected_gain) / 5.0)
    uncertainty_score = max(0.0, 1.0 - spread / 35.0)
    score = min(1.0, 0.45 * edge_score + 0.35 * p_active + 0.20 * uncertainty_score)
    label = "High" if score >= 0.72 else "Medium" if score >= 0.50 else "Low"
    return score, label


def explain_player(row: dict[str, object]) -> list[str]:
    """Turn a ranked row into factual, deliberately non-causal decision evidence."""
    mean = row.get("proj_mean")
    floor = row.get("floor")
    ceiling = row.get("ceiling")
    p_active = row.get("p_active")
    source = row.get("projection_source") or "unknown"
    explanations = [
        f"Projection: {_number(mean):.1f} points from {source}."
        if mean is not None
        else f"Projection: unavailable from {source}.",
    ]
    if floor is not None and ceiling is not None:
        explanations.append(
            f"Uncertainty: {_number(floor):.1f} floor to {_number(ceiling):.1f} ceiling "
            f"({_number(ceiling) - _number(floor):.1f}-point range)."
        )
    if p_active is not None:
        explanations.append(f"Availability estimate: {_number(p_active):.0%} chance to play.")
    if mean is not None and floor is not None and ceiling is not None and p_active is not None:
        score, label = recommendation_confidence(
            expected_gain=max(0.0, _number(mean) - _number(floor)),
            floor=_number(floor),
            ceiling=_number(ceiling),
            p_active=_number(p_active),
        )
        explanations.append(
            f"Confidence: {label} ({score:.0%}), based on availability and projection range."
        )
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


EMPTY_SPOT = "your empty lineup spot"


def swap_action(start_name: object, sit_name: object | None) -> str:
    """Plain-language start/sit instruction; `None` means filling an empty slot."""
    if sit_name is None:
        return f"Start {start_name} in {EMPTY_SPOT}"
    return f"Start {start_name} over {sit_name}"


def _projection(row: dict[str, Any] | None) -> float:
    """A ranked row's projection, with an empty slot or missing projection as 0."""
    value = None if row is None else row.get("proj_mean")
    return float(value) if isinstance(value, int | float) else 0.0


def lineup_decisions(
    lineup: pl.DataFrame,
    rankings: pl.DataFrame,
    current_starter_ids: set[str],
    fmt: LeagueFormat,
    *,
    kickoff_by_team: dict[str, datetime] | None = None,
    now: datetime | None = None,
) -> pl.DataFrame:
    """Pair optimizer changes into actionable, unlocked start/sit decisions."""
    current_time = now or datetime.now(UTC)
    kickoffs = kickoff_by_team or {}
    by_id = {str(row["player_id"]): row for row in rankings.iter_rows(named=True)}
    rows: list[dict[str, object]] = []
    for incoming, outgoing in lineup_swap_rows(lineup, rankings, current_starter_ids, fmt):
        incoming_rank = by_id[str(incoming["player_id"])]
        incoming_kickoff = kickoffs.get(str(incoming_rank.get("team")))
        outgoing_kickoff = None if outgoing is None else kickoffs.get(str(outgoing.get("team")))
        if any(
            stamp is not None and stamp <= current_time
            for stamp in (incoming_kickoff, outgoing_kickoff)
        ):
            continue
        gain = _projection(incoming_rank) - _projection(outgoing)
        if gain <= 0:
            continue
        p_active = min(
            _number(incoming_rank.get("p_active") or 0.0),
            1.0 if outgoing is None else _number(outgoing.get("p_active") or 0.0),
        )
        floor = _number(incoming_rank["floor"])
        ceiling = _number(incoming_rank["ceiling"])
        confidence, label = recommendation_confidence(
            expected_gain=gain,
            floor=floor,
            ceiling=ceiling,
            p_active=p_active,
        )
        lock = min(
            (stamp for stamp in (incoming_kickoff, outgoing_kickoff) if stamp is not None),
            default=None,
        )
        if outgoing is None:
            why = (
                f"A starting spot is empty, so {incoming_rank['player_name']} adds "
                f"{_projection(incoming_rank):.1f} projected points."
            )
        else:
            reshuffle = (
                f" {incoming_rank['player_name']} goes in a FLEX spot; if "
                f"{outgoing['player_name']} isn't the one in FLEX, slide your FLEX "
                f"{outgoing['position']} into {outgoing['player_name']}'s "
                f"{outgoing['position']} spot."
                if incoming_rank["position"] != outgoing["position"]
                else ""
            )
            why = (
                f"{_projection(incoming_rank):.1f} vs {_projection(outgoing):.1f} projected "
                f"points; {p_active:.0%} minimum availability.{reshuffle}"
            )
        rows.append(
            {
                "slot": incoming["slot"],
                "start": incoming_rank["player_name"],
                "sit": None if outgoing is None else outgoing["player_name"],
                "expected_gain": gain,
                "start_floor": floor,
                "sit_floor": 0.0 if outgoing is None else outgoing["floor"],
                "start_ceiling": ceiling,
                "sit_ceiling": 0.0 if outgoing is None else outgoing["ceiling"],
                "confidence": confidence,
                "confidence_label": label,
                "lock_time": lock.isoformat() if lock is not None else "Unknown",
                "why": why,
                "risk": (
                    f"Recommended player's range is {floor:.1f}-{ceiling:.1f}; "
                    f"the edge is {gain:+.1f}."
                ),
            }
        )
    return (
        pl.DataFrame(rows, schema=LINEUP_DECISION_SCHEMA).sort("expected_gain", descending=True)
        if rows
        else pl.DataFrame(schema=LINEUP_DECISION_SCHEMA)
    )


def lineup_swap_rows(
    lineup: pl.DataFrame,
    rankings: pl.DataFrame,
    current_starter_ids: set[str],
    fmt: LeagueFormat,
) -> list[tuple[dict[str, Any], dict[str, Any] | None]]:
    """(recommended lineup row, ranked row to sit) pairs, each a legal swap.

    The sit side is `None` when the recommended player fills an empty slot.
    """
    if lineup.is_empty():
        return []
    supported = projection_supported_format(fmt, set(rankings["position"].unique().to_list()))
    by_id = {str(row["player_id"]): row for row in rankings.iter_rows(named=True)}
    adds = {
        str(row["player_id"]): row
        for row in lineup.filter(~pl.col("player_id").is_in(list(current_starter_ids)))
        .sort("projected_points", descending=True)
        .iter_rows(named=True)
        if str(row["player_id"]) in by_id
    }
    recommended = set(lineup["player_id"].to_list())
    sits = sorted(
        (player_id for player_id in current_starter_ids - recommended if player_id in by_id),
        key=lambda player_id: _projection(by_id[player_id]),
    )
    positions = {
        player_id: str(by_id[player_id]["position"])
        for player_id in current_starter_ids | set(adds)
        if player_id in by_id
    }
    return [
        (adds[start], None if sit is None else by_id[sit])
        for start, sit in legal_swap_pairs(list(adds), sits, positions, supported)
    ]


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
    available = rankings
    if "injury_status" in available.columns:
        available = available.filter(
            ~pl.col("injury_status").fill_null("").is_in(list(UNAVAILABLE_STATUSES))
        )
    rows = available.filter(
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
    available = rankings
    if "injury_status" in available.columns:
        available = available.filter(
            ~pl.col("injury_status").fill_null("").is_in(list(UNAVAILABLE_STATUSES))
        )
    rows = available.filter(
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


def build_action_inbox(
    lineup: pl.DataFrame,
    rankings: pl.DataFrame,
    current_starter_ids: set[str],
    waivers: pl.DataFrame,
    *,
    fmt: LeagueFormat,
    pipeline_status: str,
    alerts: list[dict[str, object]] | None = None,
) -> pl.DataFrame:
    """Build a small, ordered list of decisions that deserve attention now."""
    rows: list[dict[str, object]] = []
    if pipeline_status != "healthy":
        rows.append(
            {
                "priority": "NOW",
                "category": "data",
                "action": f"Review {pipeline_status} pipeline checks before acting",
                "expected_gain": None,
                "confidence": None,
                "confidence_label": "Data check",
                "evidence": "At least one freshness, coverage, or source check is not healthy.",
                "risk": "Inputs may be stale or incomplete until the next healthy refresh.",
            }
        )
    for add, drop in lineup_swap_rows(lineup, rankings, current_starter_ids, fmt):
        gain = float(add["projected_points"]) - _projection(drop)
        spread = max(0.0, float(add["ceiling"]) - float(add["floor"]))
        confidence = max(
            0.0,
            min(1.0, 1.0 - spread / max(1.0, float(add["projected_points"]) * 4.0)),
        )
        confidence_label = (
            "High" if confidence >= 0.72 else "Medium" if confidence >= 0.50 else "Low"
        )
        rows.append(
            {
                "priority": "NOW" if gain >= 2.0 else "WATCH",
                "category": "lineup",
                "action": swap_action(
                    add["player_name"], None if drop is None else drop["player_name"]
                ),
                "expected_gain": gain,
                "confidence": confidence,
                "confidence_label": confidence_label,
                "evidence": (
                    f"{float(add['projected_points']):.1f} vs {_projection(drop):.1f} "
                    f"projected points; {float(add['floor']):.1f}-{float(add['ceiling']):.1f} "
                    "range for the recommended starter."
                ),
                "risk": (
                    f"Recommended starter range: {float(add['floor']):.1f}-"
                    f"{float(add['ceiling']):.1f}."
                ),
            }
        )
    for row in waivers.head(5).iter_rows(named=True):
        gain = float(row["value_added_per_week"])
        rows.append(
            {
                "priority": "NOW" if gain >= 2.0 else "WATCH",
                "category": "waiver",
                "action": (
                    f"Add {row['player_name']}"
                    + (f"; drop {row['drop_player']}" if row.get("drop_player") else "")
                ),
                "expected_gain": gain,
                "confidence": None,
                "confidence_label": "Roster value",
                "evidence": (
                    f"Suggested bid {row['suggested_bid']}; "
                    f"{row['competing_teams']} competing roster(s); "
                    f"{row['projection_basis']} basis."
                ),
                "risk": "FAAB demand and future role can change before waivers process.",
            }
        )
    for alert in (alerts or [])[:5]:
        message = str(alert.get("message") or "Projection changed since the last refresh")
        if not any(row["action"] == message for row in rows):
            rows.append(
                {
                    "priority": "WATCH",
                    "category": "change",
                    "action": message,
                    "expected_gain": None,
                    "confidence": None,
                    "confidence_label": "Changed input",
                    "evidence": "Detected by comparison with the previous refresh snapshot.",
                    "risk": "Recheck availability and kickoff status before acting.",
                }
            )
    order = {"NOW": 0, "WATCH": 1}
    rows.sort(
        key=lambda row: (
            order[str(row["priority"])],
            0 if row["category"] == "data" else 1,
            -_number(row["expected_gain"]) if row["expected_gain"] is not None else 0.0,
        )
    )
    return (
        pl.DataFrame(rows, schema=ACTION_INBOX_SCHEMA)
        if rows
        else pl.DataFrame(schema=ACTION_INBOX_SCHEMA)
    )


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
    if "injury_status" in projected.columns:
        projected = projected.filter(
            ~pl.col("injury_status").fill_null("").is_in(list(UNAVAILABLE_STATUSES))
        )
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
            weighted = (
                future.with_columns(
                    pl.when(pl.col("week") >= fmt.playoff_week_start)
                    .then(pl.lit(playoff_weight))
                    .otherwise(pl.lit(1.0))
                    .alias("_weight")
                )
                .group_by("player_id")
                .agg(
                    ((pl.col("mean") * pl.col("_weight")).sum() / pl.col("_weight").sum()).alias(
                        "weighted_mean"
                    )
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
    # Competition changes bid sizing, not roster-relative value ordering.
    # Restrict the expensive opponent-lineup solves to rows that can actually
    # appear in the rendered recommendation table.
    board = board.filter(pl.col("value_added_per_week") > 0).head(limit)
    names = rankings.select(
        "player_id",
        "player_name",
        "team",
        "opponent",
        "proj_mean",
        "p_active",
        "floor",
        "ceiling",
    )
    drop_names = rankings.select(
        pl.col("player_id").alias("drop_candidate"),
        pl.col("player_name").alias("drop_player"),
    )
    opponents = [roster_projections(ids) for ids in (opponent_roster_ids or [])]
    opponent_context = [(team, optimal_lineup_points(team, supported)) for team in opponents]
    competition_rows: list[dict[str, object]] = []
    for player_id in board["player_id"].to_list():
        row = rankings.filter(pl.col("player_id") == player_id).row(0, named=True)
        mean = _number(projection_by_player[str(player_id)])
        candidate = PlayerProjection(str(player_id), str(row["position"]), mean, mean, mean)
        opponent_values = [
            value_added(team, candidate, supported, baseline_points=baseline)[0]
            for team, baseline in opponent_context
        ]
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
    )
    result = result.with_columns(
        pl.struct("suggested_bid", "competing_teams")
        .map_elements(
            lambda row: min(
                remaining_budget,
                math.ceil(
                    _number(row["suggested_bid"]) * (1.0 + 0.1 * _number(row["competing_teams"]))
                ),
            ),
            return_dtype=pl.Int64,
        )
        .alias("suggested_bid"),
        pl.lit(projection_basis).alias("projection_basis"),
    )
    if result.is_empty():
        return result
    position_counts: dict[str, int] = {}
    planned: list[dict[str, object]] = []
    for priority, row in enumerate(result.iter_rows(named=True), start=1):
        position = str(row["position"])
        position_counts[position] = position_counts.get(position, 0) + 1
        bid = int(_number(row["suggested_bid"]))
        floor = _number(row.get("floor") or 0.0)
        ceiling = _number(row.get("ceiling") or floor)
        p_active = _number(row.get("p_active") or 0.0)
        confidence, label = recommendation_confidence(
            expected_gain=_number(row["value_added_per_week"]),
            floor=floor,
            ceiling=ceiling,
            p_active=p_active,
        )
        drop = str(row.get("drop_player") or "your lowest-value bench player")
        role = (
            "Primary"
            if position_counts[position] == 1
            else f"Fallback {position_counts[position] - 1}"
        )
        row.update(
            {
                "claim_priority": priority,
                "claim_role": f"{role} {position}",
                "roster_need": f"Upgrade {drop}",
                "bid_floor": max(0, math.floor(bid * 0.75)),
                "bid_ceiling": bid,
                "confidence": confidence,
                "confidence_label": label,
                "why": (
                    f"Adds {_number(row['value_added_per_week']):.1f} projected lineup "
                    f"points per week; {int(_number(row['competing_teams']))} competing "
                    f"roster(s); {projection_basis.replace('_', ' ')} projection basis."
                ),
            }
        )
        planned.append(row)
    return pl.DataFrame(planned)


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
            kickers.select("player_id", "sleeper_id", "player_name", "position", "team", "points"),
            defenses.select("player_id", "sleeper_id", "player_name", "position", "team", "points"),
        ],
        how="vertical_relaxed",
    ).filter(
        pl.col("sleeper_id").is_not_null() & ~pl.col("sleeper_id").is_in(list(rostered_sleeper_ids))
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
    "EMPTY_SPOT",
    "lineup_swap_rows",
    "swap_action",
    "build_action_inbox",
    "explain_player",
    "player_projections",
    "projection_supported_format",
    "recommended_lineup",
    "sim_players",
    "streaming_recommendations",
    "waiver_recommendations",
]
