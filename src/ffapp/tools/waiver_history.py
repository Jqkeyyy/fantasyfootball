"""League-specific FAAB outcomes and opponent bidding tendencies."""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Any

import polars as pl

from ffapp.config import LeagueConfig, Settings
from ffapp.ingest import sleeper
from ffapp.tools.artifacts import atomic_write_parquet

OUTCOME_SCHEMA = {
    "transaction_id": pl.String,
    "week": pl.Int64,
    "created_at_ms": pl.Int64,
    "status": pl.String,
    "roster_id": pl.Int64,
    "sleeper_id": pl.String,
    "bid": pl.Int64,
}

PROFILE_SCHEMA = {
    "roster_id": pl.Int64,
    "n_winning_bids": pl.Int64,
    "mean_bid": pl.Float64,
    "median_bid": pl.Float64,
    "p75_bid": pl.Float64,
    "max_bid": pl.Int64,
    "zero_bid_rate": pl.Float64,
    "aggression_multiplier": pl.Float64,
}


def extract_waiver_outcomes(transactions: list[dict[str, Any]]) -> pl.DataFrame:
    """Normalize completed Sleeper waiver awards into one row per added player."""
    rows: list[dict[str, object]] = []
    for transaction in transactions:
        if transaction.get("type") != "waiver":
            continue
        settings = transaction.get("settings") or {}
        raw_bid = settings.get("waiver_bid", 0) if isinstance(settings, dict) else 0
        bid = max(0, int(raw_bid or 0))
        adds = transaction.get("adds") or {}
        if not isinstance(adds, dict):
            continue
        fallback_rosters = transaction.get("roster_ids") or []
        fallback_roster = int(fallback_rosters[0]) if fallback_rosters else None
        for sleeper_id, raw_roster_id in adds.items():
            roster_id = int(raw_roster_id) if raw_roster_id is not None else fallback_roster
            if roster_id is None:
                continue
            rows.append(
                {
                    "transaction_id": str(transaction.get("transaction_id") or ""),
                    "week": int(transaction.get("_week") or 0),
                    "created_at_ms": int(transaction.get("created") or 0),
                    "status": str(transaction.get("status") or "unknown"),
                    "roster_id": roster_id,
                    "sleeper_id": str(sleeper_id),
                    "bid": bid,
                }
            )
    if not rows:
        return pl.DataFrame(schema=OUTCOME_SCHEMA)
    return (
        pl.DataFrame(rows, schema=OUTCOME_SCHEMA)
        .unique(subset=["transaction_id", "sleeper_id"], keep="last")
        .sort(["week", "created_at_ms"])
    )


def build_manager_bid_profiles(outcomes: pl.DataFrame, roster_ids: list[int]) -> pl.DataFrame:
    """Estimate stable manager aggression with shrinkage toward league average."""
    completed = (
        outcomes.filter(pl.col("status") == "complete") if not outcomes.is_empty() else outcomes
    )
    league_bids = completed["bid"].to_list() if not completed.is_empty() else []
    league_median = float(statistics.median(league_bids)) if league_bids else 0.0
    rows: list[dict[str, object]] = []
    for roster_id in sorted(set(roster_ids)):
        bids = completed.filter(pl.col("roster_id") == roster_id)["bid"].to_list()
        n = len(bids)
        median = float(statistics.median(bids)) if bids else 0.0
        raw_multiplier = median / league_median if league_median > 0 and n else 1.0
        raw_multiplier = min(1.75, max(0.50, raw_multiplier))
        reliability = n / (n + 5.0)
        multiplier = 1.0 + reliability * (raw_multiplier - 1.0)
        ordered = sorted(int(value) for value in bids)
        p75_index = math.ceil(0.75 * len(ordered)) - 1 if ordered else 0
        rows.append(
            {
                "roster_id": roster_id,
                "n_winning_bids": n,
                "mean_bid": float(sum(ordered) / n) if n else 0.0,
                "median_bid": median,
                "p75_bid": float(ordered[p75_index]) if ordered else 0.0,
                "max_bid": max(ordered, default=0),
                "zero_bid_rate": sum(value == 0 for value in ordered) / n if n else 0.0,
                "aggression_multiplier": multiplier,
            }
        )
    return pl.DataFrame(rows, schema=PROFILE_SCHEMA)


def profile_multipliers(profiles: pl.DataFrame) -> dict[int, float]:
    return {
        int(row["roster_id"]): float(row["aggression_multiplier"])
        for row in profiles.iter_rows(named=True)
    }


def refresh_waiver_history(
    settings: Settings,
    league: LeagueConfig,
    current_week: int,
    roster_ids: list[int],
    *,
    offline: bool | None,
) -> tuple[Path, int]:
    if league.league_id is None:
        raise ValueError("Sleeper league ID is required for waiver history")
    transactions: list[dict[str, Any]] = []
    for week in range(1, current_week + 1):
        path = sleeper.fetch_transactions(
            league.league_id,
            week,
            offline=offline if week >= current_week - 1 else True,
            settings=settings,
        )
        week_rows: list[dict[str, Any]] = json.loads(path.read_text())
        for row in week_rows:
            row["_week"] = week
        transactions.extend(week_rows)
    outcomes = extract_waiver_outcomes(transactions)
    profiles = build_manager_bid_profiles(outcomes, roster_ids)
    output = settings.data_root / "outputs" / league.slug / "waiver_history"
    outcome_path = output / "latest.parquet"
    atomic_write_parquet(outcomes, outcome_path)
    atomic_write_parquet(profiles, output / "manager_profiles.parquet")
    return outcome_path, outcomes.height


__all__ = [
    "build_manager_bid_profiles",
    "extract_waiver_outcomes",
    "profile_multipliers",
    "refresh_waiver_history",
]
