"""League-aware FAAB guidance for players released by chopped rosters."""

from __future__ import annotations

import math
from functools import cache
from typing import Any

import polars as pl

from ffapp.league_format import LeagueFormat
from ffapp.sim.lineup import PlayerProjection, optimal_lineup, slot_instances
from ffapp.tools.waivers import suggested_bid


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
            (pl.col("season") == season) & (pl.col("week") == week) & pl.col("mean").is_not_null()
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


# Market price model, fit on 145 resolved chopped-player waivers from this account's two
# chopped leagues (2026 weeks 2-8). Winning prices follow points above positional
# replacement far more than raw projection: an elite QB sells for most of a team's
# FAAB while a merely good QB often goes unclaimed. Prices are expressed as a share of
# the average remaining FAAB per active team, so they shrink as the league spends down.
# Leave-one-week-out backtest: average miss $50 vs $64 for the per-team share model.
REPLACEMENT_PPG = {"QB": 13.0, "RB": 3.5, "WR": 6.0, "TE": 6.5}
DEFAULT_REPLACEMENT_PPG = 6.0
PRICE_TIER_EDGES = (0.0, 3.0, 6.0, 9.0)
# Deciles (0%, 10%, ..., 100%) of winning price / average remaining FAAB, per tier.
PRICE_SHARE_DECILES = (
    (0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.024, 0.055, 0.215),
    (0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.003, 0.013, 0.036, 0.178, 0.445),
    (0.000, 0.000, 0.000, 0.002, 0.011, 0.025, 0.055, 0.117, 0.177, 0.365, 0.590),
    (0.006, 0.092, 0.166, 0.220, 0.251, 0.261, 0.285, 0.313, 0.375, 0.408, 0.550),
    (0.182, 0.200, 0.275, 0.404, 0.604, 0.720, 0.784, 0.800, 0.807, 0.831, 0.880),
)
# Price quantile each bid targets. Ordinary players are bid near the typical price,
# where the old rule overpaid most; scarce top tiers are bid further up the range.
CONSERVATIVE_QUANTILE = 0.5
RECOMMENDED_QUANTILE = (0.6, 0.6, 0.6, 0.75, 0.85)
# Never recommend less than this share of a player's value to you: low-tier players are
# usually unclaimed but sometimes sell for a lot, and a token bid forfeits those. Backtest:
# 25% floor plus value shots won 74 players / 140 points above replacement for $1,691,
# against 52 / 104 for $3,226 under the original model.
VALUE_FLOOR_SHARE = 0.25
VALUE_SHOT_CONSERVATIVE_SHARE = 0.6


def price_tier(position: str, projection_ppg: float) -> int:
    """0 (at/below replacement) .. 4 (elite), from points above positional replacement."""
    above = projection_ppg - REPLACEMENT_PPG.get(position, DEFAULT_REPLACEMENT_PPG)
    return sum(above >= edge for edge in PRICE_TIER_EDGES)


def market_price(tier: int, average_faab: float, quantile: float) -> int:
    """Estimated winning price at ``quantile`` of this tier's historical price range."""
    deciles = PRICE_SHARE_DECILES[tier]
    position = min(max(quantile, 0.0), 1.0) * 10
    lower = math.floor(position)
    upper = min(lower + 1, 10)
    share = deciles[lower] + (deciles[upper] - deciles[lower]) * (position - lower)
    return int(round(share * max(average_faab, 0.0)))


def market_win_probability(tier: int, average_faab: float, bid: int) -> float:
    """Historical chance that ``bid`` beats the winning price for this tier."""
    if bid <= 0:
        return 0.0
    if average_faab <= 0:
        return 1.0
    share = bid / average_faab
    deciles = PRICE_SHARE_DECILES[tier]
    # Highest decile whose price is still below the bid (ties at $0 mean unclaimed).
    below = [index for index, value in enumerate(deciles) if value < share]
    if not below:
        return 0.0
    index = below[-1]
    if index == 10:
        return 1.0
    low, high = deciles[index], deciles[index + 1]
    fraction = (share - low) / (high - low) if high > low else 1.0
    return min(1.0, (index + fraction) / 10)


def _win_chance(bid: int, tier: int, average_faab: float, lock_bid: int) -> float:
    """Certain once a bid exceeds every interested competitor's remaining FAAB."""
    if bid > 0 and bid >= lock_bid:
        return 1.0
    return market_win_probability(tier, average_faab, bid)


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
    opponent_aggression: dict[int, float] | None = None,
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
    opponent_count = max(1, len(baseline_points) - 1)
    weaker_teams = sum(
        points < baseline_points[my_roster_id]
        for roster_id, points in baseline_points.items()
        if roster_id != my_roster_id
    )
    survival_urgency = 1.0 - weaker_teams / opponent_count
    weeks_left = max(1, season_end_week - current_week + 1)
    values_by_roster: dict[int, dict[str, tuple[float, float, float, str | None]]] = {}
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
                    _fast_lineup_points([*lineup, candidate], fmt) - baseline_points[roster_id]
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
    aggression_by_roster = opponent_aggression or {}
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
            learned_multiplier = aggression_by_roster.get(roster_id, 1.0)
            reserved_bids[roster_id][sleeper_id] = (
                min(budget, max(1, round(raw * reserve_discount * learned_multiplier)))
                if raw > 0
                else 0
            )

    my_budget = remaining_faab(roster_by_id[my_roster_id], total_budget)
    average_faab = sum(
        remaining_faab(roster, total_budget) for roster in roster_by_id.values()
    ) / max(1, len(roster_by_id))
    name_by_player_id = {
        str(row["player_id"]): str(row["player_name"])
        for row in player_values.filter(pl.col("player_name").is_not_null()).iter_rows(named=True)
    }
    output: list[dict[str, Any]] = []
    for row in candidate_values.iter_rows(named=True):
        sleeper_id = str(row["sleeper_id"])
        my_ros_value, lineup_gain, depth_gain, drop_id = values_by_roster[my_roster_id][sleeper_id]
        value_bid = reserved_bids[my_roster_id][sleeper_id]
        max_bid = min(my_budget, raw_bids[my_roster_id][sleeper_id])
        opponent_bids = [
            reserved_bids[roster_id][sleeper_id]
            for roster_id in roster_by_id
            if roster_id != my_roster_id and values_by_roster[roster_id][sleeper_id][0] > 0
        ]
        competitor_budgets = [
            remaining_faab(roster_by_id[roster_id], total_budget)
            for roster_id in roster_by_id
            if roster_id != my_roster_id and values_by_roster[roster_id][sleeper_id][0] > 0
        ]
        richest_competitor_faab = max(competitor_budgets, default=0)
        # No competitor can bid more than the FAAB it has left, so one dollar above the
        # richest interested competitor's budget wins outright (barring a tie-break).
        lock_bid = richest_competitor_faab + 1
        competing_teams = len(opponent_bids)
        tier = price_tier(str(row["position"]), float(row["projection_ppg"]))

        def priced(quantile: float, tier: int = tier, cap: int = richest_competitor_faab) -> int:
            return min(cap, market_price(tier, average_faab, quantile))

        market_bid = priced(RECOMMENDED_QUANTILE[tier])
        if my_ros_value <= 0:
            recommendation = "No lineup upgrade"
            conservative_bid = 0
            recommended_bid = 0
            aggressive_bid = 0
        elif max_bid <= 0:
            recommendation = "No FAAB left"
            conservative_bid = 0
            recommended_bid = 0
            aggressive_bid = 0
        elif market_bid + 1 > max_bid:
            # The market usually prices him above his worth to you, but managers forget or
            # run short; a fair-value bid still lands stars at a discount often enough.
            recommendation = "Value shot"
            recommended_bid = max_bid
            aggressive_bid = max_bid
            conservative_bid = max(1, round(max_bid * VALUE_SHOT_CONSERVATIVE_SHARE))
        else:
            recommendation = "Bid"
            base_recommendation = max(market_bid + 1, round(max_bid * VALUE_FLOOR_SHARE))
            urgency_buffer = round((max_bid - base_recommendation) * survival_urgency * 0.05)
            recommended_bid = min(max_bid, base_recommendation + urgency_buffer)
            conservative_bid = min(recommended_bid, priced(CONSERVATIVE_QUANTILE) + 1)
            aggressive_bid = max(recommended_bid, max_bid)
        if recommended_bid > 0 and lock_bid <= max_bid:
            conservative_bid = min(conservative_bid, lock_bid)
            recommended_bid = min(recommended_bid, lock_bid)
            aggressive_bid = min(aggressive_bid, lock_bid)

        output.append(
            {
                "sleeper_id": sleeper_id,
                "player_id": str(row["player_id"]),
                "player_name": str(row["player_name"]),
                "position": str(row["position"]),
                "team": str(row["team"]) if row.get("team") is not None else None,
                "chopped_week": int(row["chopped_week"]),
                "chopped_at_ms": int(row["chopped_at_ms"]),
                "chopped_transaction_id": str(row["transaction_id"]),
                "current_week_projection": float(row["current_week_projection"]),
                "projection_ppg": float(row["projection_ppg"]),
                "lineup_gain_ppg": lineup_gain,
                "depth_gain_ppg": depth_gain,
                "ros_lineup_value": my_ros_value,
                "value_bid": value_bid,
                "market_bid": market_bid,
                "conservative_bid": conservative_bid,
                "recommended_bid": recommended_bid,
                "aggressive_bid": aggressive_bid,
                "max_bid": max_bid,
                "conservative_win_probability": _win_chance(
                    conservative_bid, tier, average_faab, lock_bid
                ),
                "recommended_win_probability": _win_chance(
                    recommended_bid, tier, average_faab, lock_bid
                ),
                "aggressive_win_probability": _win_chance(
                    aggressive_bid, tier, average_faab, lock_bid
                ),
                "survival_urgency": survival_urgency,
                "faab_after_recommended": max(0, my_budget - recommended_bid),
                "competing_teams": competing_teams,
                "highest_opponent_estimate": max(opponent_bids, default=0),
                "richest_competitor_faab": richest_competitor_faab,
                "lock_bid": lock_bid,
                "teams_can_outbid": sum(budget >= recommended_bid for budget in competitor_budgets)
                if recommended_bid > 0
                else len(competitor_budgets),
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
    "chopped_at_ms": pl.Int64,
    "chopped_transaction_id": pl.String,
    "current_week_projection": pl.Float64,
    "projection_ppg": pl.Float64,
    "lineup_gain_ppg": pl.Float64,
    "depth_gain_ppg": pl.Float64,
    "ros_lineup_value": pl.Float64,
    "value_bid": pl.Int64,
    "market_bid": pl.Int64,
    "conservative_bid": pl.Int64,
    "recommended_bid": pl.Int64,
    "aggressive_bid": pl.Int64,
    "max_bid": pl.Int64,
    "conservative_win_probability": pl.Float64,
    "recommended_win_probability": pl.Float64,
    "aggressive_win_probability": pl.Float64,
    "survival_urgency": pl.Float64,
    "faab_after_recommended": pl.Int64,
    "competing_teams": pl.Int64,
    "highest_opponent_estimate": pl.Int64,
    "richest_competitor_faab": pl.Int64,
    "lock_bid": pl.Int64,
    "teams_can_outbid": pl.Int64,
    "drop_player": pl.String,
    "value_basis": pl.String,
    "recommendation": pl.String,
}


def _empty_bid_board() -> pl.DataFrame:
    return pl.DataFrame(schema=_BID_SCHEMA)


__all__ = [
    "build_player_values",
    "build_chopped_bid_board",
    "chopped_candidates",
    "remaining_faab",
]
