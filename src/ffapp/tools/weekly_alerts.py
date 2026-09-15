"""Stateful, decision-level alerts produced by each weekly refresh."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from ffapp.config import LeagueConfig, Settings
from ffapp.draft.pick_order import resolve_my_roster_id
from ffapp.ids import mapping
from ffapp.ingest import nflverse, sleeper
from ffapp.tools.artifacts import atomic_write_json, atomic_write_parquet

_PROJECTION_CHANGE = 3.0
_WAIVER_UPGRADE = 3.0
_RULED_OUT_P_ACTIVE = 0.10


def build_weekly_alerts(current: pl.DataFrame, previous: pl.DataFrame | None) -> pl.DataFrame:
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
                    }
                )
        upgrade = float(row["waiver_upgrade"] or 0.0)
        old_upgrade = float(old.get("waiver_upgrade") or 0.0) if old else 0.0
        if (
            not bool(row["is_rostered"])
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
                }
            )
    schema = {
        "kind": pl.String,
        "player_id": pl.String,
        "player_name": pl.String,
        "message": pl.String,
        "magnitude": pl.Float64,
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
    starter_ids: set[str],
) -> pl.DataFrame:
    current = projections.filter((pl.col("season") == season) & (pl.col("week") == week))
    identity = features.filter((pl.col("season") == season) & (pl.col("week") == week)).select(
        "player_id", "position"
    )
    names = players_dim.select("player_id", pl.col("full_name").alias("player_name"))
    snapshot = (
        current.select("player_id", "season", "week", "mean", "p_active")
        .join(identity, on="player_id", how="left")
        .join(names, on="player_id", how="left")
    )
    snapshot = snapshot.with_columns(
        pl.col("player_id").is_in(list(rostered_ids)).alias("is_rostered"),
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
        for player in roster.get("players", [])
        if player in sleeper_to_player
    }
    starter_ids = {
        sleeper_to_player[player]
        for player in my_roster.get("starters", [])
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
        starter_ids=starter_ids,
    )
    alerts = build_weekly_alerts(current, previous)
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


__all__ = ["build_weekly_alerts", "refresh_weekly_alerts"]
