"""Poll Sleeper transactions and send deduplicated, roster-aware Discord alerts."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import polars as pl

from ffapp.config import LeagueConfig, Settings, load_all_leagues, load_settings
from ffapp.draft.pick_order import resolve_my_roster_id
from ffapp.env import load_env
from ffapp.ingest import sleeper
from ffapp.tools.artifacts import atomic_write_json
from ffapp.tools.discord_notifications import send_action_notification, with_dashboard_link


def _player_name(players: dict[str, Any], player_id: object) -> str:
    row = players.get(str(player_id), {})
    name = row.get("full_name") or " ".join(
        part for part in (row.get("first_name"), row.get("last_name")) if part
    )
    return str(name or player_id)


def _team_names(rosters: list[dict[str, Any]], users: list[dict[str, Any]]) -> dict[int, str]:
    users_by_id = {str(user.get("user_id")): user for user in users}
    names: dict[int, str] = {}
    for roster in rosters:
        owner = users_by_id.get(str(roster.get("owner_id")), {})
        raw_metadata = owner.get("metadata")
        metadata: dict[str, Any] = raw_metadata if isinstance(raw_metadata, dict) else {}
        names[int(roster["roster_id"])] = str(
            metadata.get("team_name") or owner.get("display_name") or f"Team {roster['roster_id']}"
        )
    return names


def format_transaction(
    transaction: dict[str, Any],
    players: dict[str, Any],
    team_names: dict[int, str],
    my_roster_id: int,
) -> str:
    """Explain a completed add/drop or trade and why it matters to the user."""
    raw_adds = transaction.get("adds")
    raw_drops = transaction.get("drops")
    adds: dict[str, Any] = raw_adds if isinstance(raw_adds, dict) else {}
    drops: dict[str, Any] = raw_drops if isinstance(raw_drops, dict) else {}
    involved = {
        int(value)
        for value in [*adds.values(), *drops.values(), *(transaction.get("roster_ids") or [])]
        if value is not None
    }
    kind = str(transaction.get("type") or "transaction").replace("_", " ").title()
    lines = [f"🔄 **{kind}: roster move completed**"]
    for roster_id in sorted(involved):
        acquired = [
            _player_name(players, player) for player, team in adds.items() if int(team) == roster_id
        ]
        released = [
            _player_name(players, player)
            for player, team in drops.items()
            if int(team) == roster_id
        ]
        parts = []
        if acquired:
            parts.append("added " + ", ".join(acquired))
        if released:
            parts.append("dropped " + ", ".join(released))
        if parts:
            marker = " **(you)**" if roster_id == my_roster_id else ""
            lines.append(
                f"• **{team_names.get(roster_id, f'Team {roster_id}')}**{marker} "
                + "; ".join(parts)
            )
    if my_roster_id in involved:
        lines.append("**Impact:** Your roster changed. Recheck the lineup and waiver plan.")
    elif drops:
        lines.append(
            "**Impact:** Newly dropped players may become waiver targets after league processing."
        )
    elif str(transaction.get("type")) == "trade":
        lines.append("**Impact:** Trade values and league power rankings have changed.")
    else:
        lines.append(
            "**Impact:** An opponent changed its roster; your strategy pages will update "
            "on refresh."
        )
    return "\n".join(lines)


def completed_new_transactions(
    transactions: list[dict[str, Any]], seen_ids: set[str]
) -> list[dict[str, Any]]:
    return sorted(
        [
            row
            for row in transactions
            if row.get("status") == "complete"
            and str(row.get("transaction_id") or "") not in seen_ids
        ],
        key=lambda row: int(row.get("created") or 0),
    )


def _current_week(settings: Settings, league: LeagueConfig) -> int | None:
    path = settings.data_root / "outputs" / league.slug / "projections.parquet"
    if not path.exists():
        return None
    periods = pl.read_parquet(path, columns=["season", "week"]).filter(
        pl.col("season") == league.season
    )
    if periods.is_empty():
        return None
    latest = periods["week"].max()
    return int(latest) if isinstance(latest, int | float) else None


def run(*, now: datetime | None = None) -> dict[str, int]:
    """Poll every configured league; the first run silently establishes a baseline."""
    load_env()
    settings = load_settings()
    if not settings.sleeper_username:
        return {"checked": 0, "sent": 0}
    state_path = settings.data_root / "outputs" / "transaction_watch_state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"leagues": {}}
    league_state = state.setdefault("leagues", {})
    players = json.loads(sleeper.fetch_players(offline=False, settings=settings).read_text())
    user = json.loads(
        sleeper.fetch_user(settings.sleeper_username, offline=False, settings=settings).read_text()
    )
    checked = sent = 0
    for league in load_all_leagues():
        if not league.league_id or (week := _current_week(settings, league)) is None:
            continue
        transactions: list[dict[str, Any]] = json.loads(
            sleeper.fetch_transactions(
                league.league_id, week, offline=False, settings=settings
            ).read_text()
        )
        ids = {str(row.get("transaction_id")) for row in transactions if row.get("transaction_id")}
        prior = league_state.get(league.slug)
        if prior is None:
            league_state[league.slug] = {"seen_ids": sorted(ids), "week": week}
            checked += 1
            continue
        seen = set(prior.get("seen_ids", [])) if int(prior.get("week", week)) == week else set()
        rosters: list[dict[str, Any]] = json.loads(
            sleeper.fetch_rosters(league.league_id, offline=False, settings=settings).read_text()
        )
        users: list[dict[str, Any]] = json.loads(
            sleeper.fetch_users(league.league_id, offline=False, settings=settings).read_text()
        )
        my_roster_id = resolve_my_roster_id(str(user["user_id"]), rosters)
        names = _team_names(rosters, users)
        for transaction in completed_new_transactions(transactions, seen):
            transaction_id = str(transaction.get("transaction_id"))
            result = send_action_notification(
                with_dashboard_link(
                    f"**{league.display_name} · Week {week}**\n"
                    + format_transaction(transaction, players, names, my_roster_id)
                ),
                key=f"transaction-{league.slug}-{transaction_id}",
                cooldown=7 * 24 * 60 * 60,
            )
            if result.status == "sent":
                sent += 1
        league_state[league.slug] = {"seen_ids": sorted(ids), "week": week}
        checked += 1
    state["last_check_utc"] = (now or datetime.now(UTC)).isoformat()
    atomic_write_json(state, state_path)
    return {"checked": checked, "sent": sent}


if __name__ == "__main__":
    print(run())


__all__ = ["completed_new_transactions", "format_transaction", "run"]
