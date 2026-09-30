"""Kickoff-aware checks, fresh availability and immutable pregame snapshots."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from ffapp.config import load_all_leagues, load_settings
from ffapp.env import load_env
from ffapp.ids import mapping
from ffapp.ingest import nflverse, sleeper
from ffapp.interim.build import add_kickoff_utc
from ffapp.tools.artifacts import atomic_write_json, atomic_write_parquet
from ffapp.tools.discord_notifications import (
    format_decision_alert,
    send_action_notification,
    with_dashboard_link,
)
from ffapp.tools.weekly_alerts import refresh_weekly_alerts


def due_windows(schedule: pl.DataFrame, now: datetime) -> list[dict[str, Any]]:
    windows = {}
    for row in schedule.to_dicts():
        if row.get("season_type", "REG") != "REG" or not row.get("kickoff_utc"):
            continue
        kickoff = datetime.fromisoformat(row["kickoff_utc"].replace("Z", "+00:00"))
        minutes = (kickoff - now).total_seconds() / 60
        if not 5 <= minutes <= 90:
            continue
        checkpoint = 30 if minutes <= 30 else 90
        key = f"{row['season']}:{row['week']}:{kickoff.isoformat()}:{checkpoint}"
        windows[key] = {
            "key": key,
            "season": row["season"],
            "week": row["week"],
            "kickoff": kickoff.isoformat(),
            "checkpoint": checkpoint,
        }
    return sorted(windows.values(), key=lambda r: r["kickoff"])


def apply_availability(
    projections: pl.DataFrame, ids: pl.DataFrame, players: dict[str, Any], teams: set[str]
) -> tuple[pl.DataFrame, dict[str, str]]:
    """Only explicit Out/IR statuses zero a player; questionable is not ruled out."""
    statuses = {}
    for row in ids.select("player_id", "sleeper_id").drop_nulls().to_dicts():
        player = players.get(row["sleeper_id"], {})
        status = str(player.get("injury_status") or "")
        if player.get("team") in teams and status.lower() in {"out", "ir", "injured reserve"}:
            statuses[row["player_id"]] = status
    columns = [
        c
        for c in ["mean", "p_active", "q10", "q25", "q50", "q75", "q90"]
        if c in projections.columns
    ]
    return projections.with_columns(
        [
            pl.when(pl.col("player_id").is_in(list(statuses)))
            .then(0.0)
            .otherwise(pl.col(c))
            .alias(c)
            for c in columns
        ]
    ), statuses


def save_snapshot(
    output: Path,
    projections: pl.DataFrame,
    features: pl.DataFrame,
    schedule: pl.DataFrame,
    statuses: dict[str, str],
    now: datetime,
) -> None:
    identity = features.select("season", "week", "player_id", "position", "team").unique(
        subset=["season", "week", "player_id"]
    )
    rows = projections.join(identity, on=["season", "week", "player_id"], how="left")
    upcoming = schedule.filter(pl.col("kickoff_utc").str.to_datetime(time_zone="UTC") > now)
    teams = pl.concat(
        [
            upcoming.select("season", "week", pl.col(side).alias("team"))
            for side in ["home_team", "away_team"]
        ]
    ).unique()
    rows = rows.join(teams, on=["season", "week", "team"], how="inner")
    if rows.is_empty():
        return
    rows = rows.rename(
        {"mean": "live_mean", **{f"q{q}": f"live_q{q}" for q in [10, 25, 50, 75, 90]}}
    )
    rows = rows.with_columns(
        pl.lit(now.isoformat()).alias("as_of_utc"),
        pl.lit("kickoff").alias("run_label"),
        pl.lit(None, dtype=pl.Float64).alias("actual_points"),
        pl.col("player_id")
        .replace_strict(statuses, default=None, return_dtype=pl.String)
        .alias("gameday_injury_status"),
    )
    path = output / "prediction_log" / "kickoff" / f"{now:%Y%m%dT%H%M%S%fZ}.parquet"
    atomic_write_parquet(rows, path)


def run() -> None:
    load_env()
    settings = load_settings()
    if not settings.sleeper_username:
        raise ValueError("Sleeper username is required for kickoff refreshes")
    state_path = settings.data_root / "outputs" / "gameday_state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    now = datetime.now(UTC)
    schedule_path = settings.data_root / "interim" / "schedule.parquet"
    schedule = pl.read_parquet(schedule_path)
    # Refresh flexed kickoff times at most every six hours, without rebuilding features.
    if now.timestamp() - state.get("schedule_checked", 0) > 21600:
        try:
            raw = pl.read_parquet(
                nflverse.fetch_schedules(settings.seasons.current, offline=False, settings=settings)
            )
            fresh = add_kickoff_utc(nflverse.normalize_schedule(raw), pl.DataFrame())
            schedule = pl.concat(
                [schedule.filter(pl.col("season") != settings.seasons.current), fresh],
                how="diagonal_relaxed",
            )
            atomic_write_parquet(schedule, schedule_path)
            state["schedule_checked"] = now.timestamp()
        except Exception:
            state["schedule_error"] = now.isoformat()
            send_action_notification(
                "Game-day schedule refresh failed; cached kickoff times remain in use.",
                key="gameday-schedule",
            )
    state["last_check"] = now.isoformat()
    for window in due_windows(schedule, now):
        for league in load_all_leagues():
            if league.season != window["season"] or not league.league_id:
                continue
            key = window["key"] + ":" + league.slug
            previous = state.get(key, {})
            if previous.get("status") == "healthy" or now.timestamp() < previous.get(
                "retry_after", 0
            ):
                continue
            output = settings.data_root / "outputs" / league.slug
            log_path = settings.data_root / "outputs" / "logs" / "gameday.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                # Refresh projections directly so alerts see the final availability override,
                # rather than sending a second notification from the weekly CLI workflow.
                with log_path.open("a") as log:
                    result = subprocess.run(
                        [
                            sys.executable,
                            "-c",
                            "from ffapp.cli import app; app()",
                            "project",
                            "--week",
                            str(window["week"]),
                            "--season",
                            str(window["season"]),
                            "--league",
                            league.slug,
                            "--no-offline",
                        ],
                        stdout=log,
                        stderr=log,
                        timeout=600,
                        check=False,
                    )
                if result.returncode:
                    raise RuntimeError("Projection refresh failed; see server game-day log")
                players_path = sleeper.fetch_players(offline=False, settings=settings)
                players = json.loads(players_path.read_text())
                sleeper.fetch_rosters(league.league_id, offline=False, settings=settings)
                sleeper.fetch_matchups(
                    league.league_id, window["week"], offline=False, settings=settings
                )
                sleeper.fetch_user(settings.sleeper_username, offline=False, settings=settings)
                ids = mapping.build_players_dim(
                    nflverse.fetch_player_ids(offline=True, settings=settings),
                    players_path,
                    mapping.ID_OVERRIDES_PATH,
                )
                stamp = datetime.now(UTC)
                upcoming = schedule.filter(
                    (pl.col("season") == window["season"])
                    & (pl.col("week") == window["week"])
                    & (pl.col("kickoff_utc").str.to_datetime(time_zone="UTC") > stamp)
                )
                teams = set(upcoming["home_team"].to_list() + upcoming["away_team"].to_list())
                path = output / "projections.parquet"
                all_rows = pl.read_parquet(path)
                period = (pl.col("season") == window["season"]) & (pl.col("week") == window["week"])
                adjusted, statuses = apply_availability(
                    all_rows.filter(period), ids, players, teams
                )
                atomic_write_parquet(pl.concat([all_rows.filter(~period), adjusted]), path)
                save_snapshot(
                    output,
                    adjusted,
                    pl.read_parquet(
                        settings.data_root / "features" / "player_week_features.parquet"
                    ),
                    schedule,
                    statuses,
                    stamp,
                )
                alert_path, _ = refresh_weekly_alerts(
                    settings,
                    league,
                    window["season"],
                    window["week"],
                    repeat_lineup_actions=True,
                )
                alerts = json.loads(alert_path.read_text())["alerts"]
                for alert in alerts:
                    checkpoint = str(window["checkpoint"]).replace("m", " minutes")
                    delivery = send_action_notification(
                        with_dashboard_link(
                            f"{league.display_name} · Week {window['week']}\n"
                            f"{checkpoint} before kickoff\n{format_decision_alert(alert)}\n"
                        ),
                        key=key.split(":")[0] + league.slug,
                    )
                    if delivery.status == "failed":
                        raise RuntimeError("Discord delivery failed")
                state[key] = {"status": "healthy", "checked_at": stamp.isoformat()}
                atomic_write_json(
                    {
                        "status": "healthy",
                        "checked_at": stamp.isoformat(),
                        "season": window["season"],
                        "week": window["week"],
                        "checkpoint": window["checkpoint"],
                        "ruled_out": statuses,
                    },
                    output / "gameday.json",
                )
            except Exception as exc:
                state[key] = {
                    "status": "failed",
                    "retry_after": now.timestamp() + 600,
                    "error": type(exc).__name__,
                }
                send_action_notification(
                    f"{league.display_name}: game-day refresh failed. "
                    "Check the site before setting your lineup.",
                    key=key,
                )
            atomic_write_json(state, state_path)
    atomic_write_json(state, state_path)


if __name__ == "__main__":
    run()
