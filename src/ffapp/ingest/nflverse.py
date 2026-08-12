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
    """Fetch nflreadpy's *weekly* rosters (position/team/status history,
    ids) for one season or a range to
    data/raw/nflverse/rosters_<label>.parquet.

    Uses `load_rosters_weekly`, not `load_rosters` (task 1.1 originally
    called the latter, a real bug found and fixed in task 1.9 -- this
    docstring already said "weekly" but the implementation didn't match
    it). Confirmed live: `load_rosters` returns one row per player per
    *season* (~3k rows/season across the real 2015-2025 range, 33,195
    total) -- nowhere near enough granularity for SPEC §11.1's "every
    player on an active roster *that week*." `load_rosters_weekly`
    returns real per-week snapshots (46,579 rows for 2024 alone).
    """
    settings = _resolve_settings(settings)
    label = _season_label(seasons)
    season_list = _as_season_list(seasons)
    return _fetch_nflreadpy_parquet(
        filename=f"rosters_{label}.parquet",
        call_desc=f"load_rosters_weekly(seasons={season_list})",
        cache_key="nflverse_rosters",
        load=lambda: nfl.load_rosters_weekly(seasons=season_list),
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
    precomputed columns, not derived here. `weekday` (e.g. "Thursday",
    "Monday") isn't in SPEC §6.2's own column table, but nflverse's raw
    source already carries it at zero cost -- kept for task 1.9's
    `is_primetime` derivation, which needs day-of-week and has no other
    source for it.

    `home_implied_total`/`away_implied_total` = `total_line/2 ±
    spread_line/2` (SPEC §6.2's own formula) -- safe now that
    `spread_line`'s sign convention has been verified against real data
    (task 1.3): across all 3,028 real completed games 2015-2025,
    `spread_line` correlates +0.44 with the actual home-away score margin,
    and the extreme cases are unambiguous (e.g. DAL spread_line +22.0 at
    home, won by 25; MIA spread_line -18.0 at home, lost 0-43) --
    **positive `spread_line` means the home team is favoured by that many
    points**, confirming SPEC's own stated assumption rather than
    overturning it.

    `kickoff_utc` is still left null -- converting `gametime` (local
    kickoff time) to UTC needs a per-stadium timezone lookup
    (`config/stadiums.csv`, this same task's other deliverable, built
    after this function). It's explicitly the as_of boundary (SPEC §6.2)
    -- guessing it wrong here would be a silent leakage bug, the single
    most expensive failure mode this project has (CLAUDE.md rule 2), so it
    stays null rather than approximated even one task early.
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
        "weekday",
        pl.lit(None, dtype=pl.Utf8).alias("kickoff_utc"),
        "spread_line",
        "total_line",
        (pl.col("total_line") / 2 + pl.col("spread_line") / 2).alias("home_implied_total"),
        (pl.col("total_line") / 2 - pl.col("spread_line") / 2).alias("away_implied_total"),
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

    `practice_status` is stripped and blanked-to-null -- confirmed live,
    213 real rows across 2015-2025 carry a literal `"\n"` (or `"\n    "`)
    instead of an empty designation, which is a missing value with extra
    steps, not a distinct real category.

    `team` is kept even though it isn't in SPEC §6.2's own column list for
    this table -- task 1.4's `interim.build.backfill_injury_date_modified`
    needs it to resolve each row's own game date against `schedule`, and
    it costs nothing extra since the raw source already carries it
    (same justified-addition precedent as task 0.12's provenance columns).
    """
    return raw.select(
        pl.col("gsis_id").alias("player_id"),
        pl.col("season").cast(pl.Int32),
        pl.col("week").cast(pl.Int32),
        "team",
        "report_status",
        pl.col("practice_status").str.strip_chars().replace("", None),
        "report_primary_injury",
        "date_modified",
    )


# SPEC §6.1: ffopportunity is CC-BY-SA (share-alike propagates to derived
# data), distinct from the plain CC-BY the rest of nflverse's data carries
# -- written to its own data/raw/ffopportunity/ directory (not mixed into
# data/raw/nflverse/) so the licence obligation stays unambiguous, per
# SPEC's own instruction to record each source's licence at ingest time.
FF_OPPORTUNITY_LICENSE = """ffopportunity (ffverse) -- CC BY-SA 4.0

https://creativecommons.org/licenses/by-sa/4.0/

Precomputed expected fantasy points (xfp) from an xgboost model over
nflverse play-by-play. Share-alike: any derived data or output built from
this source inherits the same licence terms and must be attributed, and
shared under equivalent terms if ever redistributed. See SPEC.md §6.1 and
§16.5 (public-readiness audit) before any public release.

Source: https://github.com/ffverse/ffopportunity
"""


def _ffopportunity_dir(settings: Settings) -> Path:
    return settings.cache.root / "ffopportunity"


def fetch_ff_opportunity(
    seasons: int | list[int], *, offline: bool | None = None, settings: Settings | None = None
) -> Path:
    """Fetch ffopportunity's weekly expected-fantasy-points data (task 1.2's
    `xfp` source) for one season or a range to
    data/raw/ffopportunity/weekly_<label>.parquet, alongside a `LICENSE.txt`
    recording its CC-BY-SA terms (see module-level `FF_OPPORTUNITY_LICENSE`).
    """
    settings = _resolve_settings(settings)
    label = _season_label(seasons)
    season_list = _as_season_list(seasons)
    raw_dir = _ffopportunity_dir(settings)
    path = raw_dir / f"weekly_{label}.parquet"
    call_desc = f"load_ff_opportunity(seasons={season_list}, stat_type='weekly')"

    if is_offline(offline):
        if not path.exists():
            raise cache_miss(
                "ffopportunity",
                "weekly",
                f"seasons={label}",
                f"ffapp ingest nflverse --ff-opportunity seasons={label} --no-offline",
            )
        meta = read_sidecar(path)
        if meta is not None:
            verdict = check_staleness(meta, "ffopportunity_weekly", settings.cache.staleness_hours)
            if verdict == "stale":
                logger.warning(
                    "ffopportunity weekly seasons=%s is stale (fetched_at_utc=%s); run to refresh.",
                    label,
                    meta["fetched_at_utc"],
                )
        return path

    df = nfl.load_ff_opportunity(seasons=season_list, stat_type="weekly")
    raw_dir.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)
    write_sidecar(
        path,
        source="ffopportunity",
        call=call_desc,
        cache_key="ffopportunity_weekly",
        rows=df.height,
    )
    license_path = raw_dir / "LICENSE.txt"
    if not license_path.exists():
        license_path.write_text(FF_OPPORTUNITY_LICENSE, encoding="utf-8")
    return path


__all__ = [
    "CROSSWALK_URL",
    "FF_OPPORTUNITY_LICENSE",
    "fetch_depth_charts",
    "fetch_ff_opportunity",
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
