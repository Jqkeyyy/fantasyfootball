"""ffverse ingestion (SPEC.md §6.1, §6.3, §7; ADDENDUM-01 §C.2 kicker stats).

Two network paths:
    - `_get_csv()`: a raw HTTP GET for the dynastyprocess/ffverse player-id
      crosswalk (the base table for `ids/mapping.py`).
    - `nflreadpy.load_*()`: the `player_stats`/`team_stats`/`schedules` fetchers
      call nflreadpy directly rather than hitting an HTTP endpoint by hand.
      nflreadpy hands back an already-parsed polars DataFrame (no raw bytes to
      archive verbatim), so the DataFrame itself, written to parquet, is this
      project's raw archive -- the moral equivalent of `sleeper.py` re-serialising
      a decoded JSON payload back to disk as its "raw" copy.

Both paths follow the same offline-cache shape: offline reads the cache or raises
`OfflineCacheMiss`; online fetches and archives the raw payload plus a sidecar.

Every `fetch_*` here accepts `seasons: int | list[int]` (nflreadpy's own
`load_*` functions already take either natively) -- a single season for
task 0.5's golden test, or the full historical range for task 1.1's
interim-table build. Schema normalisation for the *simple* tables (pure
1:1 column reshaping -- `schedule`, `injuries`) lives here too, per
§6.3's `fetch()`/`normalise()` pairing; anything that needs real joins or
aggregation across sources (player_week_usage, team_week_context,
defense_position_allowed) lives in `interim/build.py` instead, matching
this project's established ingest/-stays-pure precedent (`projections/
aggregate.py` vs. `ingest/rankings.py`; `scoring/stats.py` vs. this
module) -- CLAUDE.md: no business logic in ingest/ beyond schema
normalisation.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

import nflreadpy as nfl
import polars as pl
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from ffapp.cache.offline import (
    cache_miss,
    check_staleness,
    is_offline,
    read_sidecar,
    write_sidecar,
)
from ffapp.config import Settings
from ffapp.config import load_settings as _load_settings

CROSSWALK_URL = (
    "https://raw.githubusercontent.com/dynastyprocess/data/master/files/db_playerids.csv"
)
USER_AGENT = (
    "ffapp/0.1 (personal fantasy football decision-support tool; "
    "contact via github.com/Jqkeyyy/fantasyfootball)"
)

logger = logging.getLogger("ffapp.ingest.nflverse")

_session: requests.Session | None = None


def _get_session() -> requests.Session:
    global _session
    if _session is None:
        session = requests.Session()
        session.headers.update({"User-Agent": USER_AGENT})
        retry = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        _session = session
    return _session


def _get_csv() -> str:
    """Fetch the crosswalk CSV as text. The only network call in this module."""
    response = _get_session().get(CROSSWALK_URL, timeout=30)
    response.raise_for_status()
    return response.text


def _resolve_settings(settings: Settings | None) -> Settings:
    return settings or _load_settings()


def _raw_dir(settings: Settings) -> Path:
    return settings.cache.root / "nflverse"


def fetch_player_ids(*, offline: bool | None = None, settings: Settings | None = None) -> Path:
    """Fetch the ffverse player-id crosswalk (SPEC §7 step 1) to
    data/raw/nflverse/player_ids.csv.
    """
    settings = _resolve_settings(settings)
    path = _raw_dir(settings) / "player_ids.csv"

    if is_offline(offline):
        if not path.exists():
            raise cache_miss(
                "nflverse",
                "player_ids",
                "",
                "ffapp ingest nflverse --player-ids --no-offline",
            )
        meta = read_sidecar(path)
        if meta is not None:
            verdict = check_staleness(meta, "nflverse_player_ids", settings.cache.staleness_hours)
            if verdict == "stale":
                logger.warning(
                    "player_ids is stale (fetched_at_utc=%s); run ingest nflverse to refresh.",
                    meta["fetched_at_utc"],
                )
        return path

    text = _get_csv()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    rows = max(len(text.splitlines()) - 1, 0)
    write_sidecar(
        path, source="nflverse", call=CROSSWALK_URL, cache_key="nflverse_player_ids", rows=rows
    )
    return path


def _season_label(seasons: int | list[int]) -> str:
    """Filename-safe label for one season (`"2025"`) or a contiguous-ish
    range (`"2015-2026"`) -- used for both the cache filename and the
    human-readable `params` string in cache-miss/staleness messages."""
    season_list = _as_season_list(seasons)
    if len(season_list) == 1:
        return str(season_list[0])
    return f"{min(season_list)}-{max(season_list)}"


def _as_season_list(seasons: int | list[int]) -> list[int]:
    return [seasons] if isinstance(seasons, int) else list(seasons)


def _fetch_nflreadpy_parquet(
    *,
    filename: str,
    call_desc: str,
    cache_key: str,
    load: Callable[[], pl.DataFrame],
    artifact: str,
    params: str,
    offline: bool | None,
    settings: Settings,
) -> Path:
    path = _raw_dir(settings) / filename

    if is_offline(offline):
        if not path.exists():
            raise cache_miss(
                "nflverse",
                artifact,
                params,
                f"ffapp ingest nflverse --{artifact} {params.replace(' ', ' --')} --no-offline",
            )
        meta = read_sidecar(path)
        if meta is not None:
            verdict = check_staleness(meta, cache_key, settings.cache.staleness_hours)
            if verdict == "stale":
                logger.warning(
                    "%s %s is stale (fetched_at_utc=%s); run to refresh.",
                    artifact,
                    params,
                    meta["fetched_at_utc"],
                )
        return path

    df = load()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)
    write_sidecar(path, source="nflverse", call=call_desc, cache_key=cache_key, rows=df.height)
    return path


def fetch_player_stats(
    seasons: int | list[int], *, offline: bool | None = None, settings: Settings | None = None
) -> Path:
    """Fetch nflreadpy's per-player-week stat table for one season or a
    range (task 1.1: seasons 2015-2026) to
    data/raw/nflverse/player_stats_<label>.parquet.
    """
    settings = _resolve_settings(settings)
    label = _season_label(seasons)
    season_list = _as_season_list(seasons)
    return _fetch_nflreadpy_parquet(
        filename=f"player_stats_{label}.parquet",
        call_desc=f"load_player_stats(seasons={season_list})",
        cache_key="nflverse_player_stats",
        load=lambda: nfl.load_player_stats(seasons=season_list),
        artifact="player-stats",
        params=f"seasons={label}",
        offline=offline,
        settings=settings,
    )


def fetch_team_stats(
    seasons: int | list[int], *, offline: bool | None = None, settings: Settings | None = None
) -> Path:
    """Fetch nflreadpy's per-team-week stat table (DST scoring inputs) for
    one season or a range to data/raw/nflverse/team_stats_<label>.parquet.
    """
    settings = _resolve_settings(settings)
    label = _season_label(seasons)
    season_list = _as_season_list(seasons)
    return _fetch_nflreadpy_parquet(
        filename=f"team_stats_{label}.parquet",
        call_desc=f"load_team_stats(seasons={season_list})",
        cache_key="nflverse_team_stats",
        load=lambda: nfl.load_team_stats(seasons=season_list),
        artifact="team-stats",
        params=f"seasons={label}",
        offline=offline,
        settings=settings,
    )


def fetch_pbp(
    seasons: int | list[int], *, offline: bool | None = None, settings: Settings | None = None
) -> Path:
    """Fetch nflreadpy's play-by-play (needed to derive genuine
    defensive/return touchdowns -- see scoring/stats.py's
    `_defensive_return_tds` -- and team_week_context/defense_position_allowed's
    EPA/success-rate aggregations, task 1.1) for one season or a range to
    data/raw/nflverse/pbp_<label>.parquet.
    """
    settings = _resolve_settings(settings)
    label = _season_label(seasons)
    season_list = _as_season_list(seasons)
    return _fetch_nflreadpy_parquet(
        filename=f"pbp_{label}.parquet",
        call_desc=f"load_pbp(seasons={season_list})",
        cache_key="nflverse_pbp",
        load=lambda: nfl.load_pbp(seasons=season_list),
        artifact="pbp",
        params=f"seasons={label}",
        offline=offline,
        settings=settings,
    )


def fetch_schedules(
    seasons: int | list[int], *, offline: bool | None = None, settings: Settings | None = None
) -> Path:
    """Fetch nflreadpy's schedule (game scores, spread/total lines, rest
    days) for one season or a range to
    data/raw/nflverse/schedules_<label>.parquet.
    """
    settings = _resolve_settings(settings)
    label = _season_label(seasons)
    season_list = _as_season_list(seasons)
    return _fetch_nflreadpy_parquet(
        filename=f"schedules_{label}.parquet",
        call_desc=f"load_schedules(seasons={season_list})",
        cache_key="nflverse_schedules",
        load=lambda: nfl.load_schedules(seasons=season_list),
        artifact="schedules",
        params=f"seasons={label}",
        offline=offline,
        settings=settings,
    )


def fetch_snap_counts(
    seasons: int | list[int], *, offline: bool | None = None, settings: Settings | None = None
) -> Path:
    """Fetch nflreadpy's PFR-sourced snap counts (offense_snaps/offense_pct,
    task 1.1's player_week_usage input) for one season or a range to
    data/raw/nflverse/snap_counts_<label>.parquet.
    """
    settings = _resolve_settings(settings)
    label = _season_label(seasons)
    season_list = _as_season_list(seasons)
    return _fetch_nflreadpy_parquet(
        filename=f"snap_counts_{label}.parquet",
        call_desc=f"load_snap_counts(seasons={season_list})",
        cache_key="nflverse_snap_counts",
        load=lambda: nfl.load_snap_counts(seasons=season_list),
        artifact="snap-counts",
        params=f"seasons={label}",
        offline=offline,
        settings=settings,
    )


def fetch_depth_charts(
    seasons: int | list[int], *, offline: bool | None = None, settings: Settings | None = None
) -> Path:
    """Fetch nflreadpy's weekly depth charts for one season or a range to
    data/raw/nflverse/depth_charts_<label>.parquet.
    """
    settings = _resolve_settings(settings)
    label = _season_label(seasons)
    season_list = _as_season_list(seasons)
    return _fetch_nflreadpy_parquet(
        filename=f"depth_charts_{label}.parquet",
        call_desc=f"load_depth_charts(seasons={season_list})",
        cache_key="nflverse_depth_charts",
        load=lambda: nfl.load_depth_charts(seasons=season_list),
        artifact="depth-charts",
        params=f"seasons={label}",
        offline=offline,
        settings=settings,
    )


def fetch_rosters(
    seasons: int | list[int], *, offline: bool | None = None, settings: Settings | None = None
) -> Path:
    """Fetch nflreadpy's weekly rosters (position/team history, ids) for one
    season or a range to data/raw/nflverse/rosters_<label>.parquet.
    """
    settings = _resolve_settings(settings)
    label = _season_label(seasons)
    season_list = _as_season_list(seasons)
    return _fetch_nflreadpy_parquet(
        filename=f"rosters_{label}.parquet",
        call_desc=f"load_rosters(seasons={season_list})",
        cache_key="nflverse_rosters",
        load=lambda: nfl.load_rosters(seasons=season_list),
        artifact="rosters",
        params=f"seasons={label}",
        offline=offline,
        settings=settings,
    )


def fetch_injuries(
    seasons: int | list[int], *, offline: bool | None = None, settings: Settings | None = None
) -> Path:
    """Fetch nflreadpy's weekly official injury report for one season or a
    range to data/raw/nflverse/injuries_<label>.parquet.
    """
    settings = _resolve_settings(settings)
    label = _season_label(seasons)
    season_list = _as_season_list(seasons)
    return _fetch_nflreadpy_parquet(
        filename=f"injuries_{label}.parquet",
        call_desc=f"load_injuries(seasons={season_list})",
        cache_key="nflverse_injuries",
        load=lambda: nfl.load_injuries(seasons=season_list),
        artifact="injuries",
        params=f"seasons={label}",
        offline=offline,
        settings=settings,
    )


def normalize_schedule(raw: pl.DataFrame) -> pl.DataFrame:
    """nflreadpy's `load_schedules()` output -> `interim/schedule.parquet`
    (SPEC §6.2). `home_rest`/`away_rest` are already nflverse's own
    precomputed columns, not derived here.

    `kickoff_utc` and `home_implied_total`/`away_implied_total` are left
    null -- both need real work task 1.3 owns, not guessed here: converting
    `gametime` (local kickoff time) to UTC needs a per-stadium timezone
    lookup (`config/stadiums.csv`, task 1.3's own deliverable), and the
    implied-total formula needs `spread_line`'s sign convention verified
    first (SPEC §10.2/1.3: "positive spread = home favoured (verify sign at
    ingest and document)" -- not yet verified). `kickoff_utc` is explicitly
    the as_of boundary (SPEC §6.2) -- guessing it wrong here would be a
    silent leakage bug, the single most expensive failure mode this project
    has (CLAUDE.md rule 2), so it stays null rather than approximated.
    """
    return raw.select(
        "game_id",
        "season",
        "week",
        pl.col("game_type").alias("season_type"),
        "home_team",
        "away_team",
        "gameday",
        "gametime",
        pl.lit(None, dtype=pl.Utf8).alias("kickoff_utc"),
        "spread_line",
        "total_line",
        pl.lit(None, dtype=pl.Float64).alias("home_implied_total"),
        pl.lit(None, dtype=pl.Float64).alias("away_implied_total"),
        "roof",
        "surface",
        "stadium_id",
        "home_rest",
        "away_rest",
    )


def normalize_injuries(raw: pl.DataFrame) -> pl.DataFrame:
    """nflreadpy's `load_injuries()` output -> `interim/injuries.parquet`
    (SPEC §6.2). `player_id` <- `gsis_id` -- nflverse's own primary key,
    matching this project's canonical `player_id` scheme directly
    (`ids.mapping.assign_canonical_id`: `player_id = gsis_id` when
    present). Rows with no `gsis_id` (rare; a practice-squad-only player
    nflverse hasn't linked yet) are kept, not dropped -- CLAUDE.md rule 4 --
    with a null `player_id`, same as any other unresolved-id case elsewhere
    in this project.

    `season`/`week` are cast `Int32` -- confirmed live, nflreadpy's own
    `load_injuries()` hands both back as `Float64` (no nulls, just an
    upstream schema quirk unique to this one source), which would silently
    break any join against the other five interim tables' `Int32`
    season/week columns (the exact `SchemaError` this project's own tests
    for this module hit while under construction).
    """
    return raw.select(
        pl.col("gsis_id").alias("player_id"),
        pl.col("season").cast(pl.Int32),
        pl.col("week").cast(pl.Int32),
        "report_status",
        "practice_status",
        "report_primary_injury",
        "date_modified",
    )


__all__ = [
    "CROSSWALK_URL",
    "fetch_depth_charts",
    "fetch_injuries",
    "fetch_pbp",
    "fetch_player_ids",
    "fetch_player_stats",
    "fetch_rosters",
    "fetch_schedules",
    "fetch_snap_counts",
    "fetch_team_stats",
    "normalize_injuries",
    "normalize_schedule",
]
