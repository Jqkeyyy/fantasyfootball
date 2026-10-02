"""Stateful, decision-level alerts produced by each weekly refresh."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from ffapp.app.weekly_actions_page import swap_action
from ffapp.app.weekly_rankings_page import UNAVAILABLE_STATUSES
from ffapp.config import LeagueConfig, Settings
from ffapp.draft.pick_order import resolve_my_roster_id
from ffapp.ids import mapping
from ffapp.ingest import nflverse, sleeper
from ffapp.league_format import LeagueFormat, parse_league_format
from ffapp.sim.lineup import PlayerProjection, legal_swap_pairs, optimal_lineup
from ffapp.tools.artifacts import atomic_write_json, atomic_write_parquet

_PROJECTION_CHANGE = 3.0
_WAIVER_UPGRADE = 3.0
_RULED_OUT_P_ACTIVE = 0.10
_LINEUP_GAIN = 1.5


def _supported_format(fmt: LeagueFormat, positions: set[str]) -> LeagueFormat:
    return LeagueFormat(
        n_teams=fmt.n_teams,
        starters={key: value for key, value in fmt.starters.items() if key in positions},
        flex_slots=dict(fmt.flex_slots),
        flex_eligible={
            key: [position for position in eligible if position in positions]
            for key, eligible in fmt.flex_eligible.items()
        },
        bench=fmt.bench,
        ir=fmt.ir,
        playoff_week_start=fmt.playoff_week_start,
        waiver_budget=fmt.waiver_budget,
    )


def _lineup_swaps(
    snapshot: pl.DataFrame, fmt: LeagueFormat
) -> list[tuple[dict[str, object], dict[str, object] | None, float]]:
    if "is_my_roster" not in snapshot.columns:
        return []
    roster = snapshot.filter(pl.col("is_my_roster") & pl.col("mean").is_not_null())
    if "injury_status" in roster.columns:
        roster = roster.filter(
            ~pl.col("injury_status").fill_null("").is_in(list(UNAVAILABLE_STATUSES))
        )
    if roster.is_empty():
        return []
    supported = _supported_format(fmt, set(roster["position"].drop_nulls().to_list()))
    players = [
        PlayerProjection(
            str(row["player_id"]),
            str(row["position"]),
            float(row["mean"]),
            float(row["mean"]),
            float(row["mean"]),
        )
        for row in roster.iter_rows(named=True)
    ]
    recommended = set(optimal_lineup(players, supported).slots.values())
    current = set(roster.filter(pl.col("is_starter"))["player_id"].to_list())
    by_id = {str(row["player_id"]): row for row in roster.iter_rows(named=True)}
    incoming = sorted(
        recommended - current, key=lambda player: float(by_id[player]["mean"]), reverse=True
    )
    outgoing = sorted(current - recommended, key=lambda player: float(by_id[player]["mean"]))
    positions = {
        player_id: str(by_id[player_id]["position"]) for player_id in current | recommended
    }
    return [
        (
            by_id[start],
            None if sit is None else by_id[sit],
            float(by_id[start]["mean"]) - (0.0 if sit is None else float(by_id[sit]["mean"])),
        )
        for start, sit in legal_swap_pairs(incoming, outgoing, positions, supported)
    ]


def build_weekly_alerts(
    current: pl.DataFrame,
    previous: pl.DataFrame | None,
    *,
    fmt: LeagueFormat | None = None,
    repeat_lineup_actions: bool = False,
) -> pl.DataFrame:
    """Diff enriched projection snapshots into actionable alert rows."""
    prior = previous if previous is not None else pl.DataFrame()
    prior_by_id = {str(row["player_id"]): row for row in prior.iter_rows(named=True)}
    alerts: list[dict[str, object]] = []
    for row in current.iter_rows(named=True):
        player_id = str(row["player_id"])
        old = prior_by_id.get(player_id)
        same_period = bool(
            old
            and (
                "season" not in row
                or "season" not in old
                or (row["season"], row["week"]) == (old["season"], old["week"])
            )
        )
        old_active = float(old["p_active"]) if old and old.get("p_active") is not None else None
        p_active = float(row["p_active"]) if row["p_active"] is not None else 1.0
        if (
            bool(row["is_starter"])
            and p_active <= _RULED_OUT_P_ACTIVE
            and (old_active is None or old_active > _RULED_OUT_P_ACTIVE)
        ):
            alerts.append(
                {
                    "kind": "starter_ruled_out",
                    "player_id": player_id,
                    "player_name": row["player_name"],
                    "message": (
                        f"{row['player_name']} is a current starter with only "
                        f"{p_active:.0%} availability."
                    ),
                    "magnitude": 1.0 - p_active,
                    "priority": "NOW",
                    "confidence": 1.0 - p_active,
                    "why": "Current starter availability crossed the ruled-out threshold.",
                    "risk": "Confirm the final inactive list before kickoff.",
                }
            )
        if (
            bool(row["is_starter"])
            and old
            and same_period
            and old.get("mean") is not None
            and row["mean"] is not None
        ):
            change = float(row["mean"]) - float(old["mean"])
            if abs(change) >= _PROJECTION_CHANGE:
                alerts.append(
                    {
                        "kind": "starter_projection_changed",
                        "player_id": player_id,
                        "player_name": row["player_name"],
                        "message": f"{row['player_name']} projection changed {change:+.1f} points.",
                        "magnitude": abs(change),
                        "priority": "NOW" if change < 0 else "WATCH",
                        "confidence": 0.70,
                        "why": (
                            "The starter projection moved at least three points since the prior "
                            "snapshot."
                        ),
                        "risk": "A projection move can reverse as news and source inputs change.",
                    }
                )
        upgrade = float(row["waiver_upgrade"] or 0.0)
        old_upgrade = float(old.get("waiver_upgrade") or 0.0) if old else 0.0
        unavailable = str(row.get("injury_status") or "") in UNAVAILABLE_STATUSES
        if (
            not bool(row["is_rostered"])
            and not unavailable
            and upgrade >= _WAIVER_UPGRADE
            and old_upgrade < _WAIVER_UPGRADE
        ):
            alerts.append(
                {
                    "kind": "waiver_upgrade",
                    "player_id": player_id,
                    "player_name": row["player_name"],
                    "message": (
                        f"{row['player_name']} is available and projects as a "
                        f"{upgrade:+.1f}-point positional upgrade."
                    ),
                    "magnitude": upgrade,
                    "priority": "NOW",
                    "confidence": 0.65,
                    "why": "The available player crossed the three-point roster-upgrade threshold.",
                    "risk": "Role and waiver competition can change before claims process.",
                }
            )
    if fmt is not None:
        previous_pairs = (
            set()
            if repeat_lineup_actions
            else {
                (str(start["player_id"]), "" if sit is None else str(sit["player_id"]))
                for start, sit, gain in _lineup_swaps(prior, fmt)
                if gain >= _LINEUP_GAIN
            }
        )
        for start, sit, gain in _lineup_swaps(current, fmt):
            pair = (str(start["player_id"]), "" if sit is None else str(sit["player_id"]))
            action = swap_action(start["player_name"], None if sit is None else sit["player_name"])
            if gain < _LINEUP_GAIN or pair in previous_pairs:
                continue
            raw_active = start.get("p_active")
            p_active = float(raw_active) if isinstance(raw_active, int | float) else 0.0
            confidence = min(0.95, 0.45 + gain / 10.0) * p_active
            alerts.append(
                {
                    "kind": "lineup_swap",
                    "player_id": str(start["player_id"]),
                    "player_name": start["player_name"],
                    "message": (f"{action} for a projected {gain:+.1f}-point gain."),
                    "magnitude": gain,
                    "priority": "NOW",
                    "confidence": confidence,
                    "why": "The projection-optimal lineup changed by a meaningful amount.",
                    "risk": (
                        "Check both kickoff times and final availability before changing the "
                        "lineup."
                    ),
                }
            )
    schema = {
        "kind": pl.String,
        "player_id": pl.String,
        "player_name": pl.String,
        "message": pl.String,
        "magnitude": pl.Float64,
        "priority": pl.String,
        "confidence": pl.Float64,
        "why": pl.String,
        "risk": pl.String,
    }
    result = pl.DataFrame(alerts, schema=schema).sort("magnitude", descending=True)
    return pl.concat(
        [
            result.filter(pl.col("kind") != "waiver_upgrade"),
            result.filter(pl.col("kind") == "waiver_upgrade").head(3),
        ],
        how="vertical",
    ).sort("magnitude", descending=True)


def _enriched_snapshot(
    projections: pl.DataFrame,
    features: pl.DataFrame,
    players_dim: pl.DataFrame,
    *,
    season: int,
    week: int,
    rostered_ids: set[str],
    my_roster_ids: set[str],
    starter_ids: set[str],
) -> pl.DataFrame:
    current = projections.filter((pl.col("season") == season) & (pl.col("week") == week))
    identity = features.filter((pl.col("season") == season) & (pl.col("week") == week)).select(
        "player_id", "position"
    )
    names = players_dim.select(
        "player_id",
        pl.col("full_name").alias("player_name"),
        (
            pl.col("injury_status")
            if "injury_status" in players_dim.columns
            else pl.lit(None, dtype=pl.String).alias("injury_status")
        ),
    )
    snapshot = (
        current.select("player_id", "season", "week", "mean", "p_active")
        .join(identity, on="player_id", how="left")
        .join(names, on="player_id", how="left")
    )
    snapshot = snapshot.with_columns(
        pl.col("player_id").is_in(list(rostered_ids)).alias("is_rostered"),
        pl.col("player_id").is_in(list(my_roster_ids)).alias("is_my_roster"),
        pl.col("player_id").is_in(list(starter_ids)).alias("is_starter"),
    )
    starter_floors = (
        snapshot.filter(pl.col("is_starter"))
        .group_by("position")
        .agg(pl.col("mean").min().alias("starter_floor"))
    )
    return snapshot.join(starter_floors, on="position", how="left").with_columns(
        pl.when(~pl.col("is_rostered"))
        .then(pl.col("mean") - pl.col("starter_floor"))
        .otherwise(pl.lit(0.0))
        .fill_null(0.0)
        .alias("waiver_upgrade")
    )


def refresh_weekly_alerts(
    settings: Settings,
    league: LeagueConfig,
    season: int,
    week: int,
    *,
    now: datetime | None = None,
    repeat_lineup_actions: bool = False,
) -> tuple[Path, int]:
    """Build alerts from cached league state, then advance the comparison snapshot."""
    if league.league_id is None or settings.sleeper_username is None:
        raise ValueError("Sleeper league ID and username are required for alerts")
    rosters = json.loads(
        sleeper.fetch_rosters(league.league_id, offline=True, settings=settings).read_text()
    )
    user = json.loads(
        sleeper.fetch_user(settings.sleeper_username, offline=True, settings=settings).read_text()
    )
    roster_id = resolve_my_roster_id(str(user["user_id"]), rosters)
    my_roster = next(row for row in rosters if row.get("roster_id") == roster_id)
    players_dim = mapping.build_players_dim(
        nflverse.fetch_player_ids(offline=True, settings=settings),
        sleeper.fetch_players(offline=True, settings=settings),
        mapping.ID_OVERRIDES_PATH,
    )
    sleeper_to_player = dict(players_dim.select("sleeper_id", "player_id").drop_nulls().iter_rows())
    rostered_ids = {
        sleeper_to_player[player]
        for roster in rosters
        for player in (roster.get("players") or [])
        if player in sleeper_to_player
    }
    starter_ids = {
        sleeper_to_player[player]
        for player in (my_roster.get("starters") or [])
        if player in sleeper_to_player
    }
    my_roster_ids = {
        sleeper_to_player[player]
        for player in (my_roster.get("players") or [])
        if player in sleeper_to_player
    }
    output_dir = settings.data_root / "outputs" / league.slug / "alerts"
    snapshot_path = output_dir / "latest_snapshot.parquet"
    previous = pl.read_parquet(snapshot_path) if snapshot_path.exists() else None
    current = _enriched_snapshot(
        pl.read_parquet(settings.data_root / "outputs" / league.slug / "projections.parquet"),
        pl.read_parquet(settings.data_root / "features" / "player_week_features.parquet"),
        players_dim,
        season=season,
        week=week,
        rostered_ids=rostered_ids,
        my_roster_ids=my_roster_ids,
        starter_ids=starter_ids,
    )
    alerts = build_weekly_alerts(
        current,
        previous,
        fmt=parse_league_format(league),
        repeat_lineup_actions=repeat_lineup_actions,
    )
    generated = now or datetime.now(UTC)
    payload: dict[str, object] = {
        "league_slug": league.slug,
        "season": season,
        "week": week,
        "generated_at_utc": generated.isoformat(),
        "alerts": alerts.to_dicts(),
    }
    path = output_dir / f"{season}-w{week:02d}-{generated:%Y%m%dT%H%M%SZ}.json"
    atomic_write_json(payload, path)
    atomic_write_json(payload, output_dir / "latest.json")
    atomic_write_parquet(current, snapshot_path)
    return path, alerts.height


def enrich_alerts_with_movements(
    settings: Settings, league_slug: str, movements: pl.DataFrame
) -> int:
    """Attach source evidence and add only consequential starter movement alerts."""
    if movements.is_empty():
        return 0
    output_dir = settings.data_root / "outputs" / league_slug / "alerts"
    latest_path = output_dir / "latest.json"
    snapshot_path = output_dir / "latest_snapshot.parquet"
    if not latest_path.exists():
        return 0
    payload = json.loads(latest_path.read_text())
    raw_alerts = payload.get("alerts", [])
    alerts = [row for row in raw_alerts if isinstance(row, dict)]
    names: dict[str, str] = {}
    if snapshot_path.exists():
        snapshot = pl.read_parquet(snapshot_path)
        if {"player_id", "player_name"}.issubset(snapshot.columns):
            names = {
                str(row["player_id"]): str(row["player_name"])
                for row in snapshot.select("player_id", "player_name")
                .drop_nulls()
                .iter_rows(named=True)
            }
    movement_by_id = {str(row["player_id"]): row for row in movements.iter_rows(named=True)}
    changed = 0
    alerted_ids: set[str] = set()
    for alert in alerts:
        player_id = str(alert.get("player_id") or "")
        alerted_ids.add(player_id)
        movement = movement_by_id.get(player_id)
        if alert.get("kind") != "starter_projection_changed" or movement is None:
            continue
        alert["confidence"] = max(
            float(alert.get("confidence") or 0.0), float(movement["confidence"])
        )
        alert["why"] = movement["reason"]
        alert["risk"] = (
            "Only one tracked source confirms this move; verify late news before acting."
            if movement["signal"] == "source-led"
            else "Multiple saved sources support this move, but late news can still reverse it."
        )
        changed += 1
    for movement in movements.iter_rows(named=True):
        player_id = str(movement["player_id"])
        live_delta = _movement_number(movement.get("live_delta"))
        if (
            player_id in alerted_ids
            or not bool(movement.get("was_starting"))
            or live_delta is None
            or abs(live_delta) < _PROJECTION_CHANGE
        ):
            continue
        player_name = names.get(player_id, player_id)
        alerts.append(
            {
                "kind": "starter_projection_changed",
                "player_id": player_id,
                "player_name": player_name,
                "message": (
                    f"{player_name} projection changed {live_delta:+.1f} points from "
                    f"{movement['from_run']} to {movement['to_run']}."
                ),
                "magnitude": abs(live_delta),
                "priority": "NOW" if live_delta < 0 else "WATCH",
                "confidence": float(movement["confidence"]),
                "why": movement["reason"],
                "risk": (
                    "Only one tracked source confirms this move; verify late news before acting."
                    if movement["signal"] == "source-led"
                    else (
                        "Multiple saved sources support this move, but late news can still "
                        "reverse it."
                    )
                ),
            }
        )
        alerted_ids.add(player_id)
        changed += 1
    alerts.sort(key=lambda row: float(row.get("magnitude") or 0.0), reverse=True)
    payload["alerts"] = alerts
    atomic_write_json(payload, latest_path)
    return changed


def _movement_number(value: object) -> float | None:
    return float(value) if isinstance(value, int | float) else None


__all__ = ["build_weekly_alerts", "enrich_alerts_with_movements", "refresh_weekly_alerts"]
