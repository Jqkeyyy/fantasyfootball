"""League-aware FAAB guidance for players released by chopped rosters."""

from __future__ import annotations

import math
from functools import cache
from typing import Any

import polars as pl

from ffapp.league_format import LeagueFormat
from ffapp.sim.lineup import PlayerProjection, optimal_lineup, slot_instances
from ffapp.tools.waivers import suggested_bid


def remaining_faab(roster: dict[str, Any], total_budget: int) -> int:
    settings = roster.get("settings")
    used = int(settings.get("waiver_budget_used", 0)) if isinstance(settings, dict) else 0
    return max(0, total_budget - used)


def chopped_candidates(
    transactions: list[dict[str, Any]], rostered_sleeper_ids: set[str]
) -> pl.DataFrame:
    """Return chopped drops that remain unrostered, newest chop first."""
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    ordered = sorted(transactions, key=lambda row: int(row.get("created") or 0), reverse=True)
    for transaction in ordered:
        if transaction.get("type") != "chopped" or transaction.get("status") != "complete":
            continue
        roster_ids = transaction.get("roster_ids") or []
        source_roster_id = int(roster_ids[0]) if roster_ids else None
        drops = transaction.get("drops") or {}
        if not isinstance(drops, dict):
            continue
        for raw_player_id in drops:
            sleeper_id = str(raw_player_id)
            if sleeper_id in seen or sleeper_id in rostered_sleeper_ids:
                continue
            seen.add(sleeper_id)
            rows.append(
                {
                    "sleeper_id": sleeper_id,
                    "chopped_week": int(transaction.get("_week") or 0),
                    "chopped_at_ms": int(transaction.get("created") or 0),
                    "source_roster_id": source_roster_id,
                    "transaction_id": str(transaction.get("transaction_id") or ""),
                }
            )
    schema = {
        "sleeper_id": pl.String,
        "chopped_week": pl.Int64,
        "chopped_at_ms": pl.Int64,
        "source_roster_id": pl.Int64,
        "transaction_id": pl.String,
    }
    return pl.DataFrame(rows, schema=schema) if rows else pl.DataFrame(schema=schema)


def _percentile(values: list[int], percentile: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    weight = index - lower
    return int(round(ordered[lower] * (1.0 - weight) + ordered[upper] * weight))


def _roster_players(
    roster: dict[str, Any], by_sleeper_id: dict[str, dict[str, Any]]
) -> list[PlayerProjection]:
    result: list[PlayerProjection] = []
    for raw_id in roster.get("players") or []:
        row = by_sleeper_id.get(str(raw_id))
        if row is None or row.get("projection_ppg") is None:
            continue
        mean = float(row["projection_ppg"])
        result.append(
            PlayerProjection(
                player_id=str(row["player_id"]),
                position=str(row["position"]),
                mean=mean,
                median=mean,
                ceiling=mean,
            )
        )
    return result


def _fast_lineup_points(players: list[PlayerProjection], fmt: LeagueFormat) -> float:
    """Exact in-process assignment DP used for bulk opponent valuation."""
    slots = slot_instances(fmt)

    @cache
    def best(slot_index: int, used_mask: int) -> float:
        if slot_index >= len(slots):
            return 0.0
        _, eligible = slots[slot_index]
        result = best(slot_index + 1, used_mask)
        for player_index, player in enumerate(players):
            bit = 1 << player_index
            if used_mask & bit or player.position not in eligible:
                continue
            result = max(
                result,
                player.mean + best(slot_index + 1, used_mask | bit),
            )
        return result

    return best(0, 0)


def build_chopped_bid_board(
    candidates: pl.DataFrame,
    player_values: pl.DataFrame,
    rosters: list[dict[str, Any]],
    my_roster_id: int,
    fmt: LeagueFormat,
    *,
    current_week: int,
    total_budget: int,
    reserve_chops: int = 3,
    aggressiveness: float = 1.0,
    season_end_week: int = 17,
) -> pl.DataFrame:
    """Estimate value, market pressure, and a bid range for chopped drops.

    A roster's raw willingness is its share of the current chopped cohort's
    positive rest-of-season lineup value. A configurable reserve discount
    keeps some FAAB for future chopped rosters. The market estimate is the
    75th percentile of surviving opponents' discounted willingness; this is
    intentionally an estimate, while ``max_bid`` remains the user's hard
    value-based ceiling.
    """
    if candidates.is_empty():
        return _empty_bid_board()
    if total_budget < 0 or reserve_chops < 0 or aggressiveness <= 0:
        raise ValueError("Budget/reserve must be non-negative and aggressiveness positive")

    candidate_values = candidates.join(player_values, on="sleeper_id", how="inner")
    candidate_values = candidate_values.filter(pl.col("projection_ppg").is_not_null())
    if candidate_values.is_empty():
        return _empty_bid_board()

    active_rosters = [roster for roster in rosters if roster.get("players")]
    roster_by_id = {int(roster["roster_id"]): roster for roster in active_rosters}
    if my_roster_id not in roster_by_id:
        raise ValueError("Your roster is not active in this chopped league")

    by_sleeper_id = {
        str(row["sleeper_id"]): row
        for row in player_values.filter(pl.col("sleeper_id").is_not_null()).iter_rows(named=True)
    }
    lineups = {
        roster_id: _roster_players(roster, by_sleeper_id)
        for roster_id, roster in roster_by_id.items()
    }
    baseline_points = {
        roster_id: _fast_lineup_points(lineup, fmt) for roster_id, lineup in lineups.items()
    }
    my_baseline = optimal_lineup(lineups[my_roster_id], fmt)
    weeks_left = max(1, season_end_week - current_week + 1)
    values_by_roster: dict[
        int, dict[str, tuple[float, float, float, str | None]]
    ] = {}
    for roster_id, lineup in lineups.items():
        roster_values: dict[str, tuple[float, float, float, str | None]] = {}
        weakest = min(lineup, key=lambda player: player.mean) if lineup else None
        for row in candidate_values.iter_rows(named=True):
            mean = float(row["projection_ppg"])
            candidate = PlayerProjection(
                player_id=str(row["player_id"]),
                position=str(row["position"]),
                mean=mean,
                median=mean,
                ceiling=mean,
            )
            if roster_id == my_roster_id:
                with_candidate = optimal_lineup([*lineup, candidate], fmt)
                weekly_added = with_candidate.total_points - my_baseline.total_points
                started_ids = set(with_candidate.slots.values())
                benched = [player for player in lineup if player.player_id not in started_ids]
                drop = min(benched, key=lambda player: player.mean).player_id if benched else None
            else:
                weekly_added = (
                    _fast_lineup_points([*lineup, candidate], fmt)
                    - baseline_points[roster_id]
                )
                drop = None
            lineup_gain = max(0.0, weekly_added) if weekly_added > 1e-6 else 0.0
            depth_gain = 0.0
            if lineup_gain == 0.0 and weakest is not None:
                # Chopped leagues have no trades and lose an entire roster weekly;
                # a strong bench replacement has real survival value even when it
                # does not immediately crack the optimal lineup. Keep that value
                # deliberately smaller than a weekly starter upgrade.
                depth_gain = max(0.0, candidate.mean - weakest.mean) * 0.15
                if depth_gain > 0:
                    drop = weakest.player_id
            roster_values[str(row["sleeper_id"])] = (
                (lineup_gain + depth_gain) * weeks_left,
                lineup_gain,
                depth_gain,
                drop,
            )
        values_by_roster[roster_id] = roster_values

    reserve_discount = 1.0 / (1.0 + 0.15 * reserve_chops)
    raw_bids: dict[int, dict[str, int]] = {}
    reserved_bids: dict[int, dict[str, int]] = {}
    for roster_id, values in values_by_roster.items():
        total_value = sum(value[0] for value in values.values() if value[0] > 0)
        budget = remaining_faab(roster_by_id[roster_id], total_budget)
        raw_bids[roster_id] = {}
        reserved_bids[roster_id] = {}
        for sleeper_id, (candidate_value, _, _, _) in values.items():
            raw = suggested_bid(
                candidate_value,
                total_value,
                budget,
                aggressiveness=aggressiveness,
            )
            raw_bids[roster_id][sleeper_id] = raw
            reserved_bids[roster_id][sleeper_id] = (
                min(budget, max(1, round(raw * reserve_discount))) if raw > 0 else 0
            )

    my_budget = remaining_faab(roster_by_id[my_roster_id], total_budget)
    name_by_player_id = {
        str(row["player_id"]): str(row["player_name"])
        for row in player_values.filter(pl.col("player_name").is_not_null()).iter_rows(named=True)
    }
    output: list[dict[str, Any]] = []
    for row in candidate_values.iter_rows(named=True):
        sleeper_id = str(row["sleeper_id"])
        my_ros_value, lineup_gain, depth_gain, drop_id = values_by_roster[my_roster_id][
            sleeper_id
        ]
        value_bid = reserved_bids[my_roster_id][sleeper_id]
        max_bid = min(my_budget, raw_bids[my_roster_id][sleeper_id])
        opponent_bids = [
            reserved_bids[roster_id][sleeper_id]
            for roster_id in roster_by_id
            if roster_id != my_roster_id
            and values_by_roster[roster_id][sleeper_id][0] > 0
        ]
        market_bid = _percentile(opponent_bids, 0.75)
        competing_teams = len(opponent_bids)
        if my_ros_value <= 0:
            recommendation = "No lineup upgrade"
            recommended_bid = 0
        elif market_bid + 1 > max_bid:
            recommendation = "Pass above max"
            recommended_bid = 0
        else:
            recommendation = "Bid"
            recommended_bid = min(max_bid, max(value_bid, market_bid + 1))
        output.append(
            {
                "sleeper_id": sleeper_id,
                "player_id": str(row["player_id"]),
                "player_name": str(row["player_name"]),
                "position": str(row["position"]),
                "team": str(row["team"]) if row.get("team") is not None else None,
                "chopped_week": int(row["chopped_week"]),
                "current_week_projection": float(row["current_week_projection"]),
                "projection_ppg": float(row["projection_ppg"]),
                "lineup_gain_ppg": lineup_gain,
                "depth_gain_ppg": depth_gain,
                "ros_lineup_value": my_ros_value,
                "value_bid": value_bid,
                "market_bid": market_bid,
                "recommended_bid": recommended_bid,
                "max_bid": max_bid,
                "competing_teams": competing_teams,
                "highest_opponent_estimate": max(opponent_bids, default=0),
                "drop_player": name_by_player_id.get(str(drop_id)) if drop_id else None,
                "value_basis": (
                    "starter upgrade"
                    if lineup_gain > 0
                    else "depth upgrade"
                    if depth_gain > 0
                    else "no upgrade"
                ),
                "recommendation": recommendation,
            }
        )
    return pl.DataFrame(output, schema=_BID_SCHEMA).sort(
        ["recommended_bid", "ros_lineup_value"], descending=True
    )


_BID_SCHEMA = {
    "sleeper_id": pl.String,
    "player_id": pl.String,
    "player_name": pl.String,
    "position": pl.String,
    "team": pl.String,
    "chopped_week": pl.Int64,
    "current_week_projection": pl.Float64,
    "projection_ppg": pl.Float64,
    "lineup_gain_ppg": pl.Float64,
    "depth_gain_ppg": pl.Float64,
    "ros_lineup_value": pl.Float64,
    "value_bid": pl.Int64,
    "market_bid": pl.Int64,
    "recommended_bid": pl.Int64,
    "max_bid": pl.Int64,
    "competing_teams": pl.Int64,
    "highest_opponent_estimate": pl.Int64,
    "drop_player": pl.String,
    "value_basis": pl.String,
    "recommendation": pl.String,
}


def _empty_bid_board() -> pl.DataFrame:
    return pl.DataFrame(schema=_BID_SCHEMA)


__all__ = [
    "build_chopped_bid_board",
    "chopped_candidates",
    "remaining_faab",
]
