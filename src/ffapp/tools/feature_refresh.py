"""Incremental raw-to-feature refresh for the live NFL season."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl

from ffapp.config import CONFIG_DIR, LeagueConfig, Settings
from ffapp.features import build as features_build
from ffapp.ids import mapping
from ffapp.ingest import nflverse, sleeper, weather
from ffapp.interim import build as interim_build

FORECAST_HORIZON_DAYS = 15
FetchFrame = Callable[..., Path]


def _read_history_and_current(
    fetcher: FetchFrame,
    historical_seasons: list[int],
    current_season: int,
    *,
    offline: bool | None,
    settings: Settings,
    current_required: bool,
) -> pl.DataFrame:
    """Reuse the historical cache and refresh only the small live-season partition."""
    try:
        historical_path = fetcher(historical_seasons, offline=True, settings=settings)
    except Exception:
        if offline is not False:
            raise
        historical_path = fetcher(historical_seasons, offline=False, settings=settings)
    historical = pl.read_parquet(historical_path)

    try:
        current_path = fetcher(current_season, offline=offline, settings=settings)
        current = pl.read_parquet(current_path)
    except Exception:
        if current_required:
            raise
        return historical
    if current.is_empty():
        return historical
    return pl.concat([historical, current], how="diagonal_relaxed")


def _weather_table(
    schedule: pl.DataFrame,
    stadiums: pl.DataFrame,
    settings: Settings,
    current_season: int,
    *,
    offline: bool | None,
    now: datetime,
) -> pl.DataFrame:
    """Refresh only current-season weather while retaining historical materialization."""
    path = settings.data_root / "interim" / "weather.parquet"
    existing = pl.read_parquet(path) if path.exists() else pl.DataFrame()
    cutoff = now + timedelta(days=FORECAST_HORIZON_DAYS)
    current_games = schedule.filter(
        (pl.col("season") == current_season)
        & pl.col("kickoff_utc").is_not_null()
        & (
            pl.col("kickoff_utc").str.to_datetime(time_zone="UTC", strict=False)
            <= pl.lit(cutoff)
        )
    )
    if current_games.is_empty():
        return existing
    try:
        refreshed = weather.fetch_weather_for_schedule(
            current_games, stadiums, now=now, offline=offline, settings=settings
        )
    except Exception:
        if existing.is_empty():
            raise
        return existing
    if existing.is_empty():
        return refreshed
    return pl.concat(
        [existing.join(refreshed.select("game_id"), on="game_id", how="anti"), refreshed],
        how="diagonal_relaxed",
    )


def refresh_features(
    settings: Settings,
    league: LeagueConfig,
    *,
    offline: bool | None,
    now: datetime | None = None,
) -> dict[str, int]:
    """Refresh live raw partitions and atomically rebuild interim/features outputs."""
    current = settings.seasons.current
    historical = list(range(settings.seasons.train_start, current))
    if not historical:
        raise ValueError("Feature refresh requires at least one historical training season")

    essential = {
        "player_stats": nflverse.fetch_player_stats,
        "team_stats": nflverse.fetch_team_stats,
        "schedules": nflverse.fetch_schedules,
        "pbp": nflverse.fetch_pbp,
        "rosters": nflverse.fetch_rosters,
    }
    optional = {
        "snap_counts": nflverse.fetch_snap_counts,
        "injuries": nflverse.fetch_injuries,
        "depth_charts": nflverse.fetch_depth_charts,
        "ff_opportunity": nflverse.fetch_ff_opportunity,
    }
    raw = {
        name: _read_history_and_current(
            fetcher,
            historical,
            current,
            offline=offline,
            settings=settings,
            current_required=True,
        )
        for name, fetcher in essential.items()
    }
    raw.update(
        {
            name: _read_history_and_current(
                fetcher,
                historical,
                current,
                offline=offline,
                settings=settings,
                current_required=False,
            )
            for name, fetcher in optional.items()
        }
    )

    crosswalk = nflverse.fetch_player_ids(offline=offline, settings=settings)
    sleeper_players = sleeper.fetch_players(offline=offline, settings=settings)
    players_dim = mapping.build_players_dim(crosswalk, sleeper_players, mapping.ID_OVERRIDES_PATH)

    schedule = interim_build.add_kickoff_utc(
        nflverse.normalize_schedule(raw["schedules"]),
        pl.read_csv(CONFIG_DIR / "stadiums.csv"),
    )
    team_context = interim_build.add_schedule_context(
        interim_build.add_neutral_pace(
            interim_build.add_proe(
                interim_build.build_team_week_context(raw["pbp"]), raw["pbp"]
            ),
            raw["pbp"],
        ),
        schedule,
    )
    defense = interim_build.add_opponent_adjustment(
        interim_build.build_defense_position_allowed(raw["pbp"], raw["player_stats"]),
        raw["pbp"],
        raw["player_stats"],
    )
    player_stats = interim_build.build_player_week_stats(
        raw["player_stats"], raw["team_stats"], raw["schedules"], raw["pbp"]
    )
    usage = interim_build.add_xfp(
        interim_build.build_player_week_usage(
            raw["player_stats"], raw["snap_counts"], raw["pbp"], players_dim
        ),
        raw["ff_opportunity"],
    )
    injuries = interim_build.backfill_injury_date_modified(
        nflverse.normalize_injuries(raw["injuries"]), schedule
    )
    stadiums = pl.read_csv(CONFIG_DIR / "stadiums.csv")
    current_time = now or datetime.now(UTC)
    weather_table = _weather_table(
        schedule, stadiums, settings, current, offline=offline, now=current_time
    )
    features = features_build.build_player_week_features(
        raw["rosters"],
        schedule,
        player_stats,
        usage,
        raw["snap_counts"],
        team_context,
        defense,
        injuries,
        weather_table,
        raw["depth_charts"],
        league.league_cache["scoring_settings"],
        registry=None,
    )

    interim_dir = settings.data_root / "interim"
    features_dir = settings.data_root / "features"
    interim_dir.mkdir(parents=True, exist_ok=True)
    features_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        interim_dir / "schedule.parquet": schedule,
        interim_dir / "team_week_context.parquet": team_context,
        interim_dir / "defense_position_allowed.parquet": defense,
        interim_dir / "player_week_stats.parquet": player_stats,
        interim_dir / "player_week_usage.parquet": usage,
        interim_dir / "injuries.parquet": injuries,
        interim_dir / "weather.parquet": weather_table,
        features_dir / "player_week_features.parquet": features,
    }
    for path, frame in outputs.items():
        temporary = path.with_suffix(f"{path.suffix}.tmp")
        frame.write_parquet(temporary)
        temporary.replace(path)

    current_actual_rows = player_stats.filter(pl.col("season") == current).height
    return {
        "player_week_stats": player_stats.height,
        "current_actual_rows": current_actual_rows,
        "features": features.height,
    }


__all__ = ["FORECAST_HORIZON_DAYS", "refresh_features"]
