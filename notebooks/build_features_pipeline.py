# notebooks/build_features_pipeline.py
"""Rebuilds `data/interim/*` and `data/features/player_week_features.parquet`
from cached + live raw sources. Not scratch: this is the permanent home for
the raw-to-feature orchestration that HANDOFF.md's own "6. Rebuilding data/
from empty" section otherwise only describes as an ad hoc, undocumented
Python call sequence -- there is still no CLI command for this (README's own
"Known limitations"), so a real script has to exist somewhere.

Re-run this weekly during the season (Tuesday, after Monday Night Football's
stats land upstream) to pull the week's new player_week_stats/usage,
opponent-adjustment inputs, injuries, and near-term weather into the feature
table before `ffapp project`/`ffapp rankings ros` run. Safe to re-run any
time -- every step overwrites its own output file, matching this project's
"re-running for the same scope overwrites cleanly" idempotent convention.

`--no-offline` on ingest calls is implicit: this script always fetches live.
Run it only when you have network access.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import nflreadpy as nfl
import polars as pl
from nflreadpy.downloader import get_downloader

from ffapp.cache.offline import write_sidecar
from ffapp.config import load_primary_league, load_settings
from ffapp.features import build as features_build
from ffapp.ids import mapping
from ffapp.ingest import nflverse, sleeper, weather
from ffapp.interim import build as interim_build

# Open-Meteo's forecast endpoint only serves a limited window ahead of "now"
# (confirmed live 2026-09-05: a 23-day-out request 400'd). Weather for games
# further out than this stays honestly null and gets filled in on a later
# re-run as the season approaches them -- matches how every other in-season
# input in this project is refreshed incrementally, not fetched once.
FORECAST_HORIZON_DAYS = 15


def _season_range(settings) -> tuple[list[int], list[int]]:
    """(historical seasons with real played games, full range including the
    live current season)."""
    hist = list(range(settings.seasons.train_start, settings.seasons.current))
    full = list(range(settings.seasons.train_start, settings.seasons.current + 1))
    return hist, full


def _fetch_rosters_full_range(seasons_hist: list[int], settings) -> pl.DataFrame:
    """Weekly rosters for every season including the live current one.

    `nflreadpy.load_rosters_weekly` gates the current season behind its own
    Labor-Day heuristic and raises for a season it doesn't consider "started"
    yet, even when the real preseason roster file already exists upstream
    (confirmed live 2026-09-05, current=2026, cutover ~2026-09-10). Try the
    normal, fully-cached path first -- once nflreadpy's own gate opens (true
    for every run after that cutover date, i.e. essentially every real
    in-season Tuesday this project will ever run), this is the only branch
    that executes. Only fall back to bypassing nflreadpy's own downloader
    directly for the current season if the normal call actually raises.
    """
    current = settings.seasons.current
    full = [*seasons_hist, current]
    try:
        path = nflverse.fetch_rosters(full, offline=False, settings=settings)
        return pl.read_parquet(path)
    except Exception as exc:
        print(
            f"  normal load_rosters_weekly(seasons={full}) failed ({exc!r}); "
            f"falling back to a direct downloader fetch for {current} alone "
            "(nflreadpy's own season-start gate hasn't opened yet)."
        )
    rosters_hist = nfl.load_rosters_weekly(seasons=seasons_hist)
    downloader = get_downloader()
    rosters_current = downloader.download(
        "nflverse-data", f"weekly_rosters/roster_weekly_{current}", season=current
    )
    rosters_all = pl.concat([rosters_hist, rosters_current], how="diagonal_relaxed")
    path = settings.cache.root / "nflverse" / f"rosters_{full[0]}-{full[-1]}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    rosters_all.write_parquet(path)
    write_sidecar(
        path,
        source="nflverse",
        call=f"load_rosters_weekly(seasons={seasons_hist}) + direct roster_weekly_{current} bypass",
        cache_key="nflverse_rosters",
        rows=rosters_all.height,
    )
    return rosters_all


def main() -> None:
    settings = load_settings()
    seasons_hist, seasons_full = _season_range(settings)
    interim = settings.data_root / "interim"
    interim.mkdir(parents=True, exist_ok=True)

    print(f"seasons: historical={seasons_hist[0]}-{seasons_hist[-1]}, "
          f"full (incl. live)={seasons_full[0]}-{seasons_full[-1]}")

    print("fetching raw tables (live)...")
    pbp = pl.read_parquet(nflverse.fetch_pbp(seasons_hist, offline=False, settings=settings))
    player_stats = pl.read_parquet(
        nflverse.fetch_player_stats(seasons_hist, offline=False, settings=settings)
    )
    team_stats = pl.read_parquet(
        nflverse.fetch_team_stats(seasons_hist, offline=False, settings=settings)
    )
    raw_schedules = pl.read_parquet(
        nflverse.fetch_schedules(seasons_full, offline=False, settings=settings)
    )
    rosters_raw = _fetch_rosters_full_range(seasons_hist, settings)
    snap_counts_raw = pl.read_parquet(
        nflverse.fetch_snap_counts(seasons_hist, offline=False, settings=settings)
    )
    injuries_raw = pl.read_parquet(
        nflverse.fetch_injuries(seasons_hist, offline=False, settings=settings)
    )
    depth_charts_raw = pl.read_parquet(
        nflverse.fetch_depth_charts(seasons_full, offline=False, settings=settings)
    )
    ffopp_raw = pl.read_parquet(
        nflverse.fetch_ff_opportunity(seasons_hist, offline=False, settings=settings)
    )
    crosswalk_path = nflverse.fetch_player_ids(offline=False, settings=settings)
    sleeper_players_path = sleeper.fetch_players(offline=False, settings=settings)
    stadiums = pl.read_csv("config/stadiums.csv")

    players_dim = mapping.build_players_dim(
        crosswalk_path, sleeper_players_path, mapping.ID_OVERRIDES_PATH
    )
    print("players_dim:", players_dim.shape)

    print("schedule...")
    schedule = nflverse.normalize_schedule(raw_schedules)
    schedule = interim_build.add_kickoff_utc(schedule, stadiums)
    null_kickoff = schedule.filter(pl.col("kickoff_utc").is_null()).height
    print(f"schedule: {schedule.shape}, null kickoff_utc rows: {null_kickoff}")
    schedule.write_parquet(interim / "schedule.parquet")

    print("team_week_context...")
    twc = interim_build.build_team_week_context(pbp)
    twc = interim_build.add_proe(twc, pbp)
    twc = interim_build.add_neutral_pace(twc, pbp)
    twc = interim_build.add_schedule_context(twc, schedule)
    print("team_week_context:", twc.shape)
    twc.write_parquet(interim / "team_week_context.parquet")

    print("defense_position_allowed...")
    dpa = interim_build.build_defense_position_allowed(pbp, player_stats)
    dpa = interim_build.add_opponent_adjustment(dpa, pbp, player_stats)
    print("defense_position_allowed:", dpa.shape)
    dpa.write_parquet(interim / "defense_position_allowed.parquet")

    print("player_week_stats...")
    pws = interim_build.build_player_week_stats(player_stats, team_stats, raw_schedules, pbp)
    print("player_week_stats:", pws.shape)
    pws.write_parquet(interim / "player_week_stats.parquet")

    print("player_week_usage...")
    pwu = interim_build.build_player_week_usage(player_stats, snap_counts_raw, pbp, players_dim)
    pwu = interim_build.add_xfp(pwu, ffopp_raw)
    print("player_week_usage:", pwu.shape)
    pwu.write_parquet(interim / "player_week_usage.parquet")

    print("injuries...")
    injuries = nflverse.normalize_injuries(injuries_raw)
    injuries = interim_build.backfill_injury_date_modified(injuries, schedule)
    print("injuries:", injuries.shape)
    injuries.write_parquet(interim / "injuries.parquet")

    now = datetime.now(UTC)
    cutoff = now + timedelta(days=FORECAST_HORIZON_DAYS)

    def _in_forecast_range(kickoff_utc: str | None) -> bool:
        if kickoff_utc is None:
            return False
        return datetime.fromisoformat(kickoff_utc.replace("Z", "+00:00")) <= cutoff

    schedule_for_weather = schedule.filter(
        pl.col("kickoff_utc").map_elements(_in_forecast_range, return_dtype=pl.Boolean)
    )
    print(
        f"weather: fetching {schedule_for_weather.height} of {schedule.height} games "
        f"(historical + within {FORECAST_HORIZON_DAYS}d forecast horizon)..."
    )
    weather_table = weather.fetch_weather_for_schedule(
        schedule_for_weather, stadiums, now=now, offline=False, settings=settings
    )
    print("weather:", weather_table.shape)
    weather_table.write_parquet(interim / "weather.parquet")

    print("player_week_features (final assembly)...")
    primary = load_primary_league()
    scoring_settings = primary.league_cache["scoring_settings"]
    print("using scoring settings from primary league:", primary.slug)

    features = features_build.build_player_week_features(
        rosters_raw,
        schedule,
        pws,
        pwu,
        snap_counts_raw,
        twc,
        dpa,
        injuries,
        weather_table,
        depth_charts_raw,
        scoring_settings,
        registry=None,
    )
    print("player_week_features:", features.shape)
    (settings.data_root / "features").mkdir(parents=True, exist_ok=True)
    features.write_parquet(settings.data_root / "features" / "player_week_features.parquet")

    print("DONE")


if __name__ == "__main__":
    main()
