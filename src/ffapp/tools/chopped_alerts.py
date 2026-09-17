"""Stateful Discord recommendations for newly chopped player pools."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from ffapp.config import LeagueConfig, Settings
from ffapp.draft.pick_order import resolve_my_roster_id
from ffapp.ids import mapping
from ffapp.ingest import nflverse, sleeper
from ffapp.league_format import parse_league_format
from ffapp.sim.chopped_survival import simulate_chopped_survival
from ffapp.tools.artifacts import atomic_write_json, atomic_write_parquet
from ffapp.tools.chopped_bids import (
    build_chopped_bid_board,
    build_player_values,
    chopped_candidates,
)
from ffapp.tools.discord_notifications import NotificationResult, send_discord_message
from ffapp.tools.waiver_history import profile_multipliers
from ffapp.tools.waivers import rostered_sleeper_ids

EVALUATION_SCHEMA = {
    "chopped_transaction_id": pl.String,
    "sleeper_id": pl.String,
    "player_name": pl.String,
    "recommended_bid": pl.Int64,
    "max_bid": pl.Int64,
    "actual_winning_bid": pl.Int64,
    "winning_roster_id": pl.Int64,
    "recommendation_error": pl.Int64,
    "recommended_met_winning_bid": pl.Boolean,
}


@dataclass(frozen=True)
class ChoppedAlertResult:
    status: str
    detail: str
    sent_count: int = 0


def completed_chop_transactions(transactions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    completed = [
        row
        for row in transactions
        if row.get("type") == "chopped"
        and row.get("status") == "complete"
        and isinstance(row.get("drops"), dict)
        and bool(row.get("drops"))
    ]
    return sorted(completed, key=lambda row: int(row.get("created") or 0))


def format_chopped_recommendation(
    league_name: str,
    week: int,
    board: pl.DataFrame,
    *,
    top_n: int = 3,
) -> str:
    lines = [f"🪓 **{league_name} — new Week {week} chopped roster**"]
    lines.append("Top roster-specific claims:")
    for index, row in enumerate(board.head(top_n).iter_rows(named=True), start=1):
        drop = f" · drop {row['drop_player']}" if row.get("drop_player") else ""
        gain = float(row.get("survival_probability_gain") or 0.0)
        survival = float(row.get("survival_probability_after") or 0.0)
        lines.append(
            f"**{index}. {row['player_name']} ({row['position']})** — "
            f"${row['conservative_bid']} / **${row['recommended_bid']}** / "
            f"${row['aggressive_bid']} (max ${row['max_bid']})\n"
            f"Win {float(row['recommended_win_probability']):.0%} · "
            f"survival {survival:.0%} ({gain:+.1%}){drop}"
        )
    lines.append("Bid order: conservative / recommended / aggressive. Submit in Sleeper.")
    return "\n".join(lines)[:2000]


def evaluate_recommendations(recommendations: pl.DataFrame, outcomes: pl.DataFrame) -> pl.DataFrame:
    """Match each recommendation to the first later completed award for that player."""
    if recommendations.is_empty() or outcomes.is_empty():
        return pl.DataFrame(schema=EVALUATION_SCHEMA)
    completed = outcomes.filter(pl.col("status") == "complete").sort("created_at_ms")
    rows: list[dict[str, object]] = []
    for recommendation in recommendations.iter_rows(named=True):
        matches = completed.filter(
            (pl.col("sleeper_id") == str(recommendation["sleeper_id"]))
            & (pl.col("created_at_ms") >= int(recommendation["chopped_at_ms"]))
        )
        if matches.is_empty():
            continue
        outcome = matches.row(0, named=True)
        recommended = int(recommendation["recommended_bid"])
        actual = int(outcome["bid"])
        rows.append(
            {
                "chopped_transaction_id": str(recommendation["chopped_transaction_id"]),
                "sleeper_id": str(recommendation["sleeper_id"]),
                "player_name": str(recommendation["player_name"]),
                "recommended_bid": recommended,
                "max_bid": int(recommendation["max_bid"]),
                "actual_winning_bid": actual,
                "winning_roster_id": int(outcome["roster_id"]),
                "recommendation_error": recommended - actual,
                "recommended_met_winning_bid": recommended >= actual,
            }
        )
    return (
        pl.DataFrame(rows, schema=EVALUATION_SCHEMA)
        if rows
        else pl.DataFrame(schema=EVALUATION_SCHEMA)
    )


def _transaction_key(transaction: dict[str, Any]) -> str:
    transaction_id = str(transaction.get("transaction_id") or "")
    if transaction_id:
        return transaction_id
    return f"{transaction.get('created', 0)}-{(transaction.get('roster_ids') or ['x'])[0]}"


def _load_transactions(
    settings: Settings,
    league: LeagueConfig,
    current_week: int,
    *,
    offline: bool | None,
) -> list[dict[str, Any]]:
    if league.league_id is None:
        return []
    transactions: list[dict[str, Any]] = []
    for week in range(1, current_week + 1):
        path = sleeper.fetch_transactions(
            league.league_id,
            week,
            offline=offline if week >= current_week - 1 else True,
            settings=settings,
        )
        rows: list[dict[str, Any]] = json.loads(path.read_text())
        for row in rows:
            row["_week"] = week
        transactions.extend(rows)
    return transactions


def _update_evaluation(output: Path) -> None:
    snapshot_paths = sorted((output / "snapshots").glob("*.parquet"))
    outcomes_path = output.parent / "waiver_history" / "latest.parquet"
    if not snapshot_paths or not outcomes_path.exists():
        return
    recommendations = pl.concat(
        [pl.read_parquet(path) for path in snapshot_paths], how="diagonal_relaxed"
    )
    evaluation = evaluate_recommendations(recommendations, pl.read_parquet(outcomes_path))
    atomic_write_parquet(evaluation, output / "evaluation.parquet")


def refresh_chopped_alerts(
    settings: Settings,
    league: LeagueConfig,
    season: int,
    current_week: int,
    rosters: list[dict[str, Any]],
    *,
    offline: bool | None,
) -> ChoppedAlertResult:
    """Notify once per new chopped transaction and preserve its recommendation board."""
    if league.league_id is None or settings.sleeper_username is None:
        return ChoppedAlertResult("skipped", "Sleeper league and username are required")
    output = settings.data_root / "outputs" / league.slug / "chopped_notifications"
    state_path = output / "state.json"
    transactions = _load_transactions(settings, league, current_week, offline=offline)
    chops = completed_chop_transactions(transactions)
    all_keys = [_transaction_key(row) for row in chops]
    if not state_path.exists():
        atomic_write_json({"notified_transaction_ids": all_keys}, state_path)
        _update_evaluation(output)
        return ChoppedAlertResult(
            "initialized", f"Baselined {len(all_keys)} existing chopped transaction(s)"
        )
    state = json.loads(state_path.read_text())
    notified = {str(value) for value in state.get("notified_transaction_ids", [])}
    pending = [row for row in chops if _transaction_key(row) not in notified]
    if not pending:
        _update_evaluation(output)
        return ChoppedAlertResult("skipped", "No new chopped transaction")

    weekly_path = settings.data_root / "outputs" / league.slug / "projections.parquet"
    ros_path = settings.data_root / "outputs" / league.slug / "projections_ros.parquet"
    if not weekly_path.exists():
        return ChoppedAlertResult("failed", f"Missing {weekly_path}")
    players_dim = mapping.build_players_dim(
        nflverse.fetch_player_ids(offline=True, settings=settings),
        sleeper.fetch_players(offline=True, settings=settings),
        mapping.ID_OVERRIDES_PATH,
    )
    player_values = build_player_values(
        pl.read_parquet(weekly_path),
        pl.read_parquet(ros_path) if ros_path.exists() else None,
        players_dim,
        season=season,
        week=current_week,
    )
    user = json.loads(
        sleeper.fetch_user(settings.sleeper_username, offline=True, settings=settings).read_text()
    )
    my_roster_id = resolve_my_roster_id(str(user["user_id"]), rosters)
    fmt = parse_league_format(league)
    profiles_path = (
        settings.data_root / "outputs" / league.slug / "waiver_history" / "manager_profiles.parquet"
    )
    multipliers = (
        profile_multipliers(pl.read_parquet(profiles_path)) if profiles_path.exists() else {}
    )
    sent = 0
    failures: list[str] = []
    for transaction in pending:
        transaction_id = _transaction_key(transaction)
        candidates = chopped_candidates([transaction], rostered_sleeper_ids(rosters))
        if candidates.is_empty():
            notified.add(transaction_id)
            continue
        snapshot_path = output / "snapshots" / f"{transaction_id}.parquet"
        if snapshot_path.exists():
            board = pl.read_parquet(snapshot_path)
        else:
            board = build_chopped_bid_board(
                candidates,
                player_values,
                rosters,
                my_roster_id,
                fmt,
                current_week=current_week,
                total_budget=fmt.waiver_budget or 0,
                opponent_aggression=multipliers,
            )
            if board.is_empty():
                failures.append(f"{transaction_id}: no projected candidates")
                continue
            _, impacts = simulate_chopped_survival(
                rosters,
                player_values,
                my_roster_id,
                fmt,
                candidates["sleeper_id"].to_list(),
                n_sims=settings.simulation.week_sims,
            )
            board = board.join(impacts, on="sleeper_id", how="left").with_columns(
                pl.lit(datetime.now(UTC).isoformat()).alias("recommendation_generated_at_utc")
            )
            atomic_write_parquet(board, snapshot_path)
            atomic_write_parquet(board, output / "latest.parquet")
        result: NotificationResult = send_discord_message(
            format_chopped_recommendation(
                league.display_name, int(transaction.get("_week") or current_week), board
            )
        )
        if result.status == "sent":
            notified.add(transaction_id)
            sent += 1
        else:
            failures.append(f"{transaction_id}: {result.detail}")
    atomic_write_json({"notified_transaction_ids": sorted(notified)}, state_path)
    _update_evaluation(output)
    if failures:
        return ChoppedAlertResult("failed", "; ".join(failures), sent)
    return ChoppedAlertResult("sent", f"Sent {sent} chopped recommendation alert(s)", sent)


__all__ = [
    "ChoppedAlertResult",
    "completed_chop_transactions",
    "evaluate_recommendations",
    "format_chopped_recommendation",
    "refresh_chopped_alerts",
]
