"""Consensus ranking/projection ingestion (SPEC.md §6.1, §9.1-9.2; task 0.7).

Fetch + schema-normalisation only -- CLAUDE.md: no business logic in ingest/,
no network calls outside ingest/. Each source's `normalize_<source>()` maps
that source's own stat vocabulary onto this project's existing canonical
stat-column names (`scoring/keymap.py`'s `STAT_KEY_MAP`, the same columns
`scoring/stats.py` already produces from nflverse actuals), so
`projections/aggregate.py` can call `scoring.engine.score_stat_line()`
unmodified for every source -- one scoring engine, not a second one for
projections.

ESPN: the per-athlete `.../athletes/{id}/projections` endpoint (ESPN's public
sports.core API) turned out to be actual historical stats, not projections,
and requires already knowing every athlete's ESPN id. The bulk
`leaguedefaults` endpoint (`X-Fantasy-Filter` header, undocumented but stable
and widely used -- see cwendt94/espn-api) returns every rostered-relevant
player's `stats` list in one call; the season-total projection is the row
with `scoringPeriodId == 0, statSourceId == 1, statSplitTypeId == 0`
(confirmed live against a real 2026 payload -- `appliedTotal` on that row is
a full-season projected fantasy-point total, e.g. 365.3 for a top RB).
`ESPN_STAT_ID_MAP` is cwendt94/espn-api's community-maintained
`PLAYER_STATS_MAP` (widely used, actively maintained), cross-checked against
that same real payload -- id 24's magnitude matched a season of rushing
yards, id 41/53 matched receptions, etc.

Known gap: ESPN's DST stat ids (points-allowed buckets, 89-92/121-125) are
season-long *frequency* counts, not the single per-game `points_allowed`
scalar `scoring/keymap.py`'s `_points_allowed_bucket` expects -- that bucket
model is actuals-shaped (one game, one score) and doesn't translate to a
season projection. DST rows are normalised with defensive stats that do map
cleanly (sacks, INTs, forced/recovered fumbles, safeties, return TDs) and
`points_allowed` left null, so points-allowed-bucket keys simply contribute
zero for DST from this source -- acceptable per SPEC §9.4's own note that
DST's VOR is "almost always tiny," not a case this project is betting on.
"""

from __future__ import annotations

import io
import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import lxml.html
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

USER_AGENT = (
    "ffapp/0.1 (personal fantasy football decision-support tool; "
    "contact via github.com/Jqkeyyy/fantasyfootball)"
)

ESPN_URL_TEMPLATE = (
    "https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl/seasons/{season}"
    "/segments/0/leaguedefaults/3"
)
ESPN_PLAYER_LIMIT = 3000

logger = logging.getLogger("ffapp.ingest.rankings")

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


def _get_espn_json(season: int) -> dict[str, Any]:
    """Fetch the bulk ESPN player projections payload. The only network call
    this function makes."""
    filt = {
        "players": {
            "limit": ESPN_PLAYER_LIMIT,
            "sortDraftRanks": {"sortPriority": 1, "sortAsc": True, "value": "STANDARD"},
        }
    }
    response = _get_session().get(
        ESPN_URL_TEMPLATE.format(season=season),
        headers={"X-Fantasy-Filter": json.dumps(filt)},
        timeout=30,
    )
    response.raise_for_status()
    result: dict[str, Any] = response.json()
    return result


def _resolve_settings(settings: Settings | None) -> Settings:
    return settings or _load_settings()


def _raw_dir(settings: Settings) -> Path:
    return settings.cache.root / "rankings"


def _use_offline_cache(
    path: Path, *, artifact: str, params: str, cache_key: str, settings: Settings
) -> bool:
    """True if the offline-cached artefact at `path` should be returned as-is.

    Raises OfflineCacheMiss if offline and nothing is cached. Warns (or raises
    StaleCacheError under FFAPP_CACHE_STRICT, via check_staleness) if what's
    cached is stale.
    """
    if not path.exists():
        raise cache_miss(
            "rankings",
            artifact,
            params,
            f"ffapp ingest rankings --{artifact} {params.replace(' ', ' --')} --no-offline",
        )
    meta = read_sidecar(path)
    if meta is not None:
        verdict = check_staleness(meta, cache_key, settings.cache.staleness_hours)
        if verdict == "stale":
            logger.warning(
                "%s %s is stale (fetched_at_utc=%s); run ingest rankings to refresh.",
                artifact,
                params,
                meta["fetched_at_utc"],
            )
    return True


def _fetch_json(
    *,
    filename: str,
    call_desc: str,
    cache_key: str,
    load: Callable[[], dict[str, Any]],
    row_count: Callable[[dict[str, Any]], int],
    artifact: str,
    params: str,
    offline: bool | None,
    settings: Settings,
) -> Path:
    path = _raw_dir(settings) / filename

    if is_offline(offline):
        _use_offline_cache(
            path, artifact=artifact, params=params, cache_key=cache_key, settings=settings
        )
        return path

    payload = load()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    write_sidecar(
        path, source="rankings", call=call_desc, cache_key=cache_key, rows=row_count(payload)
    )
    return path


def _fetch_text(
    *,
    filename: str,
    call_desc: str,
    cache_key: str,
    load: Callable[[], str],
    row_count: Callable[[str], int],
    artifact: str,
    params: str,
    offline: bool | None,
    settings: Settings,
) -> Path:
    path = _raw_dir(settings) / filename

    if is_offline(offline):
        _use_offline_cache(
            path, artifact=artifact, params=params, cache_key=cache_key, settings=settings
        )
        return path

    text = load()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    write_sidecar(
        path, source="rankings", call=call_desc, cache_key=cache_key, rows=row_count(text)
    )
    return path


def fetch_espn(
    season: int, *, offline: bool | None = None, settings: Settings | None = None
) -> Path:
    """Fetch ESPN's bulk fantasy player projections for `season` to
    data/raw/rankings/espn_<season>.json.
    """
    settings = _resolve_settings(settings)
    return _fetch_json(
        filename=f"espn_{season}.json",
        call_desc=f"GET {ESPN_URL_TEMPLATE.format(season=season)}",
        cache_key="rankings_espn",
        load=lambda: _get_espn_json(season),
        row_count=lambda payload: len(payload.get("players", [])),
        artifact="espn",
        params=f"season={season}",
        offline=offline,
        settings=settings,
    )


# ESPN's defaultPositionId -> this project's canonical position code. NOT the
# widely-cited cwendt94/espn-api community mapping (0=QB, 4=WR, 6=TE, 17=K) --
# that doesn't match ESPN's current live payload. Verified directly against a
# real 500-player leaguedefaults response: Josh Allen (QB) has
# defaultPositionId=1, real WRs have 3, George Kittle/Travis Kelce (TE) have
# 4, a real kicker (Brandon Aubrey) has 5, and DST count matches 32 teams at
# 16. Under the stale community map every real QB/WR/K was silently dropped
# and every TE mislabeled as WR. Only fantasy-rosterable positions are
# listed; everything else (HC, punters, IDP slots this league doesn't use,
# ...) is dropped by normalize_espn.
ESPN_POSITION_MAP: dict[int, str] = {
    1: "QB",
    2: "RB",
    3: "WR",
    4: "TE",
    5: "K",
    16: "DST",
}

# ESPN's proTeamId -> team abbreviation (ESPN's own PRO_TEAM_MAP, e.g.
# cwendt94/espn-api's constant.py). Carried as ESPN's own raw abbreviation;
# reconciling against Sleeper/nflverse team codes is a downstream id-mapping
# concern, not this ingest module's job.
ESPN_TEAM_MAP: dict[int, str] = {
    0: "FA",
    1: "ATL",
    2: "BUF",
    3: "CHI",
    4: "CIN",
    5: "CLE",
    6: "DAL",
    7: "DEN",
    8: "DET",
    9: "GB",
    10: "TEN",
    11: "IND",
    12: "KC",
    13: "LV",
    14: "LAR",
    15: "MIA",
    16: "MIN",
    17: "NE",
    18: "NO",
    19: "NYG",
    20: "NYJ",
    21: "PHI",
    22: "ARI",
    23: "PIT",
    24: "LAC",
    25: "SF",
    26: "SEA",
    27: "TB",
    28: "WSH",
    29: "CAR",
    30: "JAX",
    33: "BAL",
    34: "HOU",
}

# ESPN's numeric stat id -> this project's canonical stat column name
# (scoring/keymap.py's STAT_KEY_MAP vocabulary). Source: cwendt94/espn-api's
# community-maintained PLAYER_STATS_MAP, cross-checked against a real 2026
# projections payload (see module docstring).
ESPN_STAT_ID_MAP: dict[str, str] = {
    "3": "passing_yards",
    "4": "passing_tds",
    "19": "passing_2pt_conversions",
    "20": "passing_interceptions",
    "64": "sacks_suffered",
    "24": "rushing_yards",
    "25": "rushing_tds",
    "26": "rushing_2pt_conversions",
    # Real payloads use id 53 for receptions, not the community map's 41 --
    # confirmed live (George Kittle/Travis Kelce season-total rows carry 53,
    # never 41). Both kept, so if 41 ever does appear it still maps.
    "41": "receptions",
    "53": "receptions",
    "42": "receiving_yards",
    "43": "receiving_tds",
    "44": "receiving_2pt_conversions",
    "68": "fumbles_total",
    "72": "fumbles_lost_total",
    "63": "fumble_recovery_tds",
    "86": "pat_made",
    "88": "pat_missed",
    "83": "fg_made",
    "85": "fg_missed",
    # DST (see module docstring for the points-allowed gap).
    "95": "def_interceptions",
    "96": "fumble_recovery_opp",
    "99": "def_sacks",
    "106": "def_fumbles_forced",
    "98": "def_safeties",
    "94": "def_return_tds",
}


def normalize_espn(payload: dict[str, Any], *, season: int) -> pl.DataFrame:
    """Extract each player's season-total projection row (`scoringPeriodId
    == 0, statSourceId == 1, statSplitTypeId == 0`) into the canonical
    per-player projection schema.
    """
    rows: list[dict[str, Any]] = []
    for entry in payload.get("players", []):
        player = entry["player"]
        position = ESPN_POSITION_MAP.get(player.get("defaultPositionId"))
        if position is None:
            continue

        season_total = next(
            (
                s
                for s in player.get("stats", [])
                if s.get("scoringPeriodId") == 0
                and s.get("statSourceId") == 1
                and s.get("statSplitTypeId") == 0
            ),
            None,
        )
        if season_total is None:
            continue

        row: dict[str, Any] = {
            "source": "espn",
            "season": season,
            "player_name": player["fullName"],
            "position": position,
            "team": ESPN_TEAM_MAP.get(player.get("proTeamId"), None),
        }
        stats = season_total.get("stats", {})
        for stat_id, column in ESPN_STAT_ID_MAP.items():
            if stat_id in stats:
                row[column] = float(stats[stat_id])
        rows.append(row)

    # infer_schema_length=None: pl.DataFrame(list[dict]) otherwise infers its
    # schema from only the first 100 rows and silently drops any column that
    # first appears later -- confirmed live, kickers rank low enough in a
    # real rank-sorted ESPN payload that fg_made/pat_made vanished entirely
    # with no error (see the regression test for the full story).
    return pl.DataFrame(rows, infer_schema_length=None)


# --- FantasyPros -------------------------------------------------------------
#
# Ranks-only (SPEC §9.2's "ecr_type: ro" -- Expert Consensus Rank -- case, not
# per-stat projections). Sourced from the dynastyprocess/data GitHub repo's
# db_fpecr_latest.csv -- the same CSV pipeline task 0.3 already trusts for the
# player-id crosswalk, so no scraping/ToS concerns and no new fetch mechanism.
# The repo also has db_fpecr.csv.gz/db_fpecr.parquet/db_fpecr.rds; the plain
# CSV is the only one this ingest module needs to parse without extra
# dependencies. Confirmed live: this CSV mixes several page types (dynasty,
# best-ball, redraft-overall, per-position) under one `ecr_type` column --
# redraft *positional* rank is `ecr_type == "rp"` (SPEC's algorithm needs
# positional rank, "reference_curve[position(p)][rank[s][p]]", not overall
# rank), and its own `pos` column already uses this project's canonical
# position codes (QB/RB/WR/TE/K/DST), confirmed against real rows for all
# six.

FANTASYPROS_URL = (
    "https://raw.githubusercontent.com/dynastyprocess/data/master/files/db_fpecr_latest.csv"
)
FANTASYPROS_POSITIONS = frozenset({"QB", "RB", "WR", "TE", "K", "DST"})


def _get_fantasypros_csv() -> str:
    """Fetch the FantasyPros ECR CSV as text. The only network call this
    function makes."""
    response = _get_session().get(FANTASYPROS_URL, timeout=30)
    response.raise_for_status()
    return response.text


def fetch_fantasypros(*, offline: bool | None = None, settings: Settings | None = None) -> Path:
    """Fetch FantasyPros expert consensus rankings to
    data/raw/rankings/fantasypros.csv.

    Unlike the other sources here, this isn't season-parameterised -- the
    upstream CSV is always "latest" (it carries its own `scrape_date` column
    instead), same as task 0.3's player-id crosswalk fetch.
    """
    settings = _resolve_settings(settings)
    return _fetch_text(
        filename="fantasypros.csv",
        call_desc=f"GET {FANTASYPROS_URL}",
        cache_key="rankings_fantasypros",
        load=_get_fantasypros_csv,
        row_count=lambda text: max(len(text.splitlines()) - 1, 0),
        artifact="fantasypros",
        params="",
        offline=offline,
        settings=settings,
    )


def normalize_fantasypros(csv_text: str, *, season: int) -> pl.DataFrame:
    """Extract redraft positional-rank rows (`ecr_type == "rp"`, one of the
    six fantasy-relevant positions) into the canonical per-player ranking
    schema.
    """
    # null_values=["NA"]: this repo's CSVs use the literal string "NA" for
    # missing values, not empty cells -- same gotcha task 0.3's crosswalk
    # fetch already hit (HANDOFF.md §5), confirmed again live here.
    raw = pl.read_csv(io.StringIO(csv_text), null_values=["NA"])
    filtered = raw.filter(
        (pl.col("ecr_type") == "rp") & pl.col("pos").is_in(list(FANTASYPROS_POSITIONS))
    )
    return filtered.select(
        pl.lit("fantasypros").alias("source"),
        pl.lit(season).alias("season"),
        pl.col("player").alias("player_name"),
        pl.col("pos").alias("position"),
        pl.col("team"),
        pl.col("ecr").alias("rank"),
        pl.col("sd").alias("rank_sd"),
        pl.col("best").alias("rank_best"),
        pl.col("worst").alias("rank_worst"),
    )


# --- FantasySharks ------------------------------------------------------------
#
# Real per-stat season projections via a `format=json` query param FantasySharks
# itself exposes next to its CSV/XML export buttons on the projections page --
# confirmed live, no HTML table scraping needed. One request per position
# bucket: `pos=ALL` covers QB/RB/WR/TE together (confirmed live -- despite the
# UI calling "ALL" "All without IDP", it excludes K/DST too), `pos=PK` and
# `pos=D` are separate requests. Neither the PK nor the D payload carries its
# own `Pos` field (confirmed live), so normalize_fantasysharks assigns K/DST
# itself per bucket rather than reading it from the row.
#
# Names arrive "Last, First" ("Allen, Josh"; DST as "Seahawks, Seattle") --
# reversed to natural order at normalise time.
#
# Known gap, same shape as ESPN's: DST's `PtsAllow` is a season total, not the
# single per-game scalar `scoring/keymap.py`'s `_points_allowed_bucket`
# expects, so `points_allowed` is left unset here too (see the ESPN section's
# docstring for the full reasoning -- SPEC §9.4 already treats DST's VOR as
# near-zero).

FANTASYSHARKS_URL_TEMPLATE = (
    "https://www.fantasysharks.com/apps/Projections/SeasonProjections.php?pos={pos}&format=json"
)
FANTASYSHARKS_POS_BUCKETS = ("ALL", "PK", "D")


def _get_fantasysharks_json(pos: str) -> list[dict[str, Any]]:
    """Fetch one position bucket's FantasySharks projections. The only
    network call this function makes."""
    response = _get_session().get(
        FANTASYSHARKS_URL_TEMPLATE.format(pos=pos),
        timeout=30,
    )
    response.raise_for_status()
    result: list[dict[str, Any]] = response.json()
    return result


def fetch_fantasysharks(*, offline: bool | None = None, settings: Settings | None = None) -> Path:
    """Fetch FantasySharks season projections (all position buckets) to
    data/raw/rankings/fantasysharks.json, keyed by bucket
    (`{"ALL": [...], "PK": [...], "D": [...]}`) -- each list is that bucket's
    raw, unmodified response.
    """
    settings = _resolve_settings(settings)
    return _fetch_json(
        filename="fantasysharks.json",
        call_desc=f"GET {FANTASYSHARKS_URL_TEMPLATE} for pos in {FANTASYSHARKS_POS_BUCKETS}",
        cache_key="rankings_fantasysharks",
        load=lambda: {pos: _get_fantasysharks_json(pos) for pos in FANTASYSHARKS_POS_BUCKETS},
        row_count=lambda payload: sum(len(rows) for rows in payload.values()),
        artifact="fantasysharks",
        params="",
        offline=offline,
        settings=settings,
    )


def _reverse_last_first(name: str) -> str:
    """ "Allen, Josh" -> "Josh Allen"; leaves names with no comma untouched."""
    if "," not in name:
        return name
    last, _, first = name.partition(",")
    return f"{first.strip()} {last.strip()}"


def _fantasysharks_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


# FantasySharks field name -> this project's canonical stat column name.
# Split by position bucket since PK/D rows carry entirely different fields
# than the ALL (QB/RB/WR/TE) bucket, confirmed live.
_FANTASYSHARKS_SKILL_STAT_MAP = {
    "PassYards": "passing_yards",
    "PassTD": "passing_tds",
    "Int": "passing_interceptions",
    "RushYards": "rushing_yards",
    "RushTD": "rushing_tds",
    "Rec": "receptions",
    "RecYards": "receiving_yards",
    "RecTD": "receiving_tds",
}
_FANTASYSHARKS_K_STAT_MAP = {
    "XP": "pat_made",
    "FG": "fg_made",
    "Miss": "fg_missed",
}
_FANTASYSHARKS_DST_STAT_MAP = {
    "TD": "def_return_tds",
    "Int": "def_interceptions",
    "Fum": "fumble_recovery_opp",
    "Sack": "def_sacks",
}


def _normalize_fantasysharks_rows(
    rows: list[dict[str, Any]], *, position: str | None, stat_map: dict[str, str], season: int
) -> list[dict[str, Any]]:
    normalized = []
    for row in rows:
        out: dict[str, Any] = {
            "source": "fantasysharks",
            "season": season,
            "player_name": _reverse_last_first(row["Name"]),
            "position": position if position is not None else row.get("Pos"),
            "team": row.get("Team"),
        }
        for field, column in stat_map.items():
            if field in row:
                out[column] = _fantasysharks_float(row[field])
        normalized.append(out)
    return normalized


def normalize_fantasysharks(
    payload: dict[str, list[dict[str, Any]]], *, season: int
) -> pl.DataFrame:
    """Combine all three FantasySharks position buckets into the canonical
    per-player projection schema."""
    rows: list[dict[str, Any]] = []
    rows += _normalize_fantasysharks_rows(
        payload.get("ALL", []),
        position=None,
        stat_map=_FANTASYSHARKS_SKILL_STAT_MAP,
        season=season,
    )
    rows += _normalize_fantasysharks_rows(
        payload.get("PK", []), position="K", stat_map=_FANTASYSHARKS_K_STAT_MAP, season=season
    )
    rows += _normalize_fantasysharks_rows(
        payload.get("D", []), position="DST", stat_map=_FANTASYSHARKS_DST_STAT_MAP, season=season
    )
    # infer_schema_length=None -- see normalize_espn's docstring for why this
    # matters whenever heterogeneous-keyed rows are combined into one frame.
    return pl.DataFrame(rows, infer_schema_length=None)


# --- CBS Sports ----------------------------------------------------------
#
# Replacement for NFL.com (blocked -- see the top-level module note). Real
# server-rendered per-stat projections at
# cbssports.com/fantasy/football/stats/{POS}/{season}/season/projections/ppr/,
# one page per position, no JS rendering needed -- confirmed live for all six
# positions. Player name is duplicated short+long inside td[0]
# (`CellPlayerName--short` / `CellPlayerName--long`); the long span's link
# text is the clean full name. DST rows have no such span -- td[0] is just
# the team name text directly, confirmed live (Denver's DST row).
#
# Column mapping is positional, not name-based: the same header text ("yds",
# "td", "avg") means a different stat depending on which stat group it's
# under (passing vs rushing vs receiving), confirmed live per position page,
# and TE has no rushing columns at all. UnexpectedColumnLayoutError guards
# against a future CBS layout change silently mis-mapping every column on a
# page (CLAUDE.md rule 4's spirit -- a wrong-but-plausible number is worse
# than a loud failure).
#
# Known gaps, same shape as ESPN's/FantasySharks': DST's `pts` (points
# allowed) is a season total, not the per-game scalar
# `_points_allowed_bucket` expects, so `points_allowed` is left unset. CBS's
# FG buckets are "1-19/20-29/30-39/40-49/50+" (no 50-59/60+ split); "50+" is
# mapped to `fg_made_50_59` as the closer approximation (60+ kicks are rare)
# and `fg_made_60_` is left unset. DST's "fum" column's exact meaning
# (forced vs. some other fumble stat) isn't documented anywhere on the page;
# mapped to `def_fumbles_forced` as the best guess from the label alone --
# revisit if a real disagreement ever traces to it.

CBS_POSITIONS = ("QB", "RB", "WR", "TE", "K", "DST")
CBS_URL_TEMPLATE = (
    "https://www.cbssports.com/fantasy/football/stats/{pos}/{season}/season/projections/ppr/"
)


class UnexpectedColumnLayoutError(Exception):
    """A scraped source's column count doesn't match what its parser
    expects. Column mapping for CBS and FFToday is positional (the same
    header text means different stats on different position pages), so a
    layout change must fail loudly rather than silently mis-map every stat
    on that page."""


# Ordered canonical column name per position, aligned to each body row's
# `<td>` cells *after* the player-name/team cell (index 0). None = present
# on the page but not mapped to a scoring stat.
_CBS_COLUMNS: dict[str, list[str | None]] = {
    "QB": [
        None,
        None,
        None,
        "passing_yards",
        None,
        "passing_tds",
        "passing_interceptions",
        None,
        None,
        "rushing_yards",
        None,
        "rushing_tds",
        "fumbles_lost_total",
        None,
        None,
    ],
    "RB": [
        None,
        None,
        "rushing_yards",
        None,
        "rushing_tds",
        None,
        "receptions",
        "receiving_yards",
        None,
        None,
        "receiving_tds",
        "fumbles_lost_total",
        None,
        None,
    ],
    "WR": [
        None,
        None,
        "receptions",
        "receiving_yards",
        None,
        None,
        "receiving_tds",
        None,
        "rushing_yards",
        None,
        "rushing_tds",
        "fumbles_lost_total",
        None,
        None,
    ],
    "TE": [
        None,
        None,
        "receptions",
        "receiving_yards",
        None,
        None,
        "receiving_tds",
        "fumbles_lost_total",
        None,
        None,
    ],
    "K": [
        None,
        "fg_made",
        None,
        None,
        "fg_made_0_19",
        None,
        "fg_made_20_29",
        None,
        "fg_made_30_39",
        None,
        "fg_made_40_49",
        None,
        "fg_made_50_59",
        None,
        "pat_made",
        None,
        None,
        None,
    ],
    "DST": [
        "def_interceptions",
        "def_safeties",
        "def_sacks",
        None,
        "fumble_recovery_opp",
        "def_fumbles_forced",
        "def_return_tds",
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    ],
}


def _get_cbs_html(pos: str, season: int) -> str:
    """Fetch one position's CBS Sports projections page. The only network
    call this function makes."""
    response = _get_session().get(CBS_URL_TEMPLATE.format(pos=pos, season=season), timeout=30)
    response.raise_for_status()
    return response.text


def fetch_cbs(
    season: int, *, offline: bool | None = None, settings: Settings | None = None
) -> Path:
    """Fetch CBS Sports season projections (all six position pages) to
    data/raw/rankings/cbs_<season>.json, keyed by position -- each value is
    that page's raw, unmodified HTML.
    """
    settings = _resolve_settings(settings)
    return _fetch_json(
        filename=f"cbs_{season}.json",
        call_desc=f"GET {CBS_URL_TEMPLATE} for pos in {CBS_POSITIONS}",
        cache_key="rankings_cbs",
        load=lambda: {pos: _get_cbs_html(pos, season) for pos in CBS_POSITIONS},
        row_count=lambda payload: len(payload),
        artifact="cbs",
        params=f"season={season}",
        offline=offline,
        settings=settings,
    )


def _xpath_elements(node: lxml.html.HtmlElement, expr: str) -> list[lxml.html.HtmlElement]:
    """lxml's `.xpath()` return type is a wide union (elements, strings,
    numbers, booleans) depending on the expression; this project only ever
    uses it for element-returning expressions, so narrow it here once
    instead of fighting the stubs' union type at every call site."""
    return cast("list[lxml.html.HtmlElement]", node.xpath(expr))


def _xpath_strings(node: lxml.html.HtmlElement, expr: str) -> list[str]:
    return cast("list[str]", node.xpath(expr))


def _cbs_player_name_and_team(row: lxml.html.HtmlElement) -> tuple[str, str | None]:
    long_name = _xpath_strings(row, './/span[@class="CellPlayerName--long"]//a/text()')
    if not long_name:
        # DST rows have no CellPlayerName span -- td[0] is just the team name.
        return _xpath_elements(row, "./td[1]")[0].text_content().strip(), None
    team = _xpath_strings(
        row, './/span[@class="CellPlayerName--long"]//span[@class="CellPlayerName-team"]/text()'
    )
    return long_name[0].strip(), team[0].strip() if team else None


def _parse_scraped_float(value: str) -> float | None:
    """Shared numeric-cell parser for CBS and FFToday: both use
    comma-grouped thousands ("3,787") and CBS additionally uses an em dash
    for "no data" (confirmed live, e.g. a kicker's `lng` column)."""
    value = value.strip().replace(",", "")
    if value in ("", "—"):
        return None
    return float(value)


def normalize_cbs(payload: dict[str, str], *, season: int) -> pl.DataFrame:
    """Parse each position's raw HTML table into the canonical per-player
    projection schema. Raises UnexpectedColumnLayoutError if a page's column
    count doesn't match `_CBS_COLUMNS` for that position.
    """
    rows: list[dict[str, Any]] = []
    for position, html in payload.items():
        columns = _CBS_COLUMNS[position]
        tree = lxml.html.fromstring(html)
        for body_row in _xpath_elements(tree, '//tbody/tr[contains(@class,"TableBase-bodyTr")]'):
            cells = _xpath_elements(body_row, "./td")
            stat_cells = cells[1:]
            if len(stat_cells) != len(columns):
                raise UnexpectedColumnLayoutError(
                    f"CBS {position} page: expected {len(columns)} stat columns, "
                    f"got {len(stat_cells)}. The site's layout likely changed; "
                    "_CBS_COLUMNS needs re-verifying against the real page."
                )
            player_name, team = _cbs_player_name_and_team(body_row)
            row: dict[str, Any] = {
                "source": "cbs",
                "season": season,
                "player_name": player_name,
                "position": position,
                "team": team,
            }
            for column, cell in zip(columns, stat_cells, strict=True):
                if column is not None:
                    row[column] = _parse_scraped_float(cell.text_content())
            rows.append(row)

    # infer_schema_length=None -- see normalize_espn's docstring for why this
    # matters whenever heterogeneous-keyed rows are combined into one frame.
    return pl.DataFrame(rows, infer_schema_length=None)


# --- FFToday ---------------------------------------------------------------
#
# Replacement for Yahoo (blocked -- see the top-level module note). Real
# per-stat projections at fftoday.com/rankings/playerproj.php, one PosID per
# position (10/20/30/40/80/99 -- confirmed live by fetching each and reading
# the page's own <title>). The page is server-rendered, but the real data
# table is nested inside a malformed outer <td> alongside several small
# layout tables (confirmed live) -- found here as the nested table with the
# most rows, since the real player list always dwarfs any layout table on
# the page. Player/team come from fixed column indices (no CBS-style
# short/long name duplication here); DST rows have no separate team column,
# same as CBS's DST rows -- td[1] is the team name directly.
#
# Column mapping is positional for the same reason as CBS's (see that
# section's note) -- UnexpectedColumnLayoutError guards both sources.
#
# Known gap, same shape as the other three sources': DST's `PA` (points
# allowed) is a season total, not the per-game scalar
# `_points_allowed_bucket` expects, so `points_allowed` is left unset.

FFTODAY_POSITIONS = ("QB", "RB", "WR", "TE", "K", "DST")
FFTODAY_POS_ID = {"QB": 10, "RB": 20, "WR": 30, "TE": 40, "K": 80, "DST": 99}
FFTODAY_URL_TEMPLATE = (
    "https://www.fftoday.com/rankings/playerproj.php?Season={season}&PosID={pos_id}"
)

# Ordered canonical column name per position, aligned to a data row's full
# `<td>` cell list (index 0 = the leading "Chg" cell present on every page).
# "__name__"/"__team__" mark the player-name/team cells; None = present on
# the page but not mapped to a scoring stat.
_FFTODAY_COLUMNS: dict[str, list[str | None]] = {
    "QB": [
        None,
        "__name__",
        "__team__",
        None,
        None,
        None,
        "passing_yards",
        "passing_tds",
        "passing_interceptions",
        None,
        "rushing_yards",
        "rushing_tds",
        None,
    ],
    "RB": [
        None,
        "__name__",
        "__team__",
        None,
        None,
        "rushing_yards",
        "rushing_tds",
        "receptions",
        "receiving_yards",
        "receiving_tds",
        None,
    ],
    "WR": [
        None,
        "__name__",
        "__team__",
        None,
        "receptions",
        "receiving_yards",
        "receiving_tds",
        None,
        "rushing_yards",
        "rushing_tds",
        None,
    ],
    "TE": [
        None,
        "__name__",
        "__team__",
        None,
        "receptions",
        "receiving_yards",
        "receiving_tds",
        None,
    ],
    "K": [
        None,
        "__name__",
        "__team__",
        None,
        "fg_made",
        None,
        None,
        "pat_made",
        None,
        None,
    ],
    "DST": [
        None,
        "__name__",
        None,
        "def_sacks",
        "fumble_recovery_opp",
        "def_interceptions",
        "def_return_tds",
        None,
        None,
        None,
        "def_safeties",
        None,
        None,
    ],
}
# Every position's data table has this many header rows (a group-header row
# e.g. "Passing/Rushing/Fantasy", then the real column-label row) before
# player data starts -- confirmed live across all six position pages.
_FFTODAY_HEADER_ROWS = 2


def _get_fftoday_html(pos: str, season: int) -> str:
    """Fetch one position's FFToday projections page. The only network call
    this function makes."""
    response = _get_session().get(
        FFTODAY_URL_TEMPLATE.format(season=season, pos_id=FFTODAY_POS_ID[pos]), timeout=30
    )
    response.raise_for_status()
    return response.text


def fetch_fftoday(
    season: int, *, offline: bool | None = None, settings: Settings | None = None
) -> Path:
    """Fetch FFToday season projections (all six position pages) to
    data/raw/rankings/fftoday_<season>.json, keyed by position -- each value
    is that page's raw, unmodified HTML.
    """
    settings = _resolve_settings(settings)
    return _fetch_json(
        filename=f"fftoday_{season}.json",
        call_desc=f"GET {FFTODAY_URL_TEMPLATE} for pos in {FFTODAY_POSITIONS}",
        cache_key="rankings_fftoday",
        load=lambda: {pos: _get_fftoday_html(pos, season) for pos in FFTODAY_POSITIONS},
        row_count=lambda payload: len(payload),
        artifact="fftoday",
        params=f"season={season}",
        offline=offline,
        settings=settings,
    )


def _fftoday_data_table(html: str) -> lxml.html.HtmlElement:
    """The real player-data table is nested inside a malformed outer <td>
    alongside several small layout tables -- identified as whichever nested
    table has the most rows, since the real player list always dwarfs any
    layout table on the page (confirmed live)."""
    tree = lxml.html.fromstring(html)
    tables = _xpath_elements(tree, "//table")
    return max(tables, key=lambda t: len(_xpath_elements(t, "./tr")))


def normalize_fftoday(payload: dict[str, str], *, season: int) -> pl.DataFrame:
    """Parse each position's raw HTML table into the canonical per-player
    projection schema. Raises UnexpectedColumnLayoutError if a page's column
    count doesn't match `_FFTODAY_COLUMNS` for that position.
    """
    rows: list[dict[str, Any]] = []
    for position, html in payload.items():
        columns = _FFTODAY_COLUMNS[position]
        data_rows = _xpath_elements(_fftoday_data_table(html), "./tr")[_FFTODAY_HEADER_ROWS:]
        for data_row in data_rows:
            cells = _xpath_elements(data_row, "./td")
            if len(cells) != len(columns):
                raise UnexpectedColumnLayoutError(
                    f"FFToday {position} page: expected {len(columns)} columns, "
                    f"got {len(cells)}. The site's layout likely changed; "
                    "_FFTODAY_COLUMNS needs re-verifying against the real page."
                )
            row: dict[str, Any] = {
                "source": "fftoday",
                "season": season,
                "position": position,
                "team": None,
            }
            for column, cell in zip(columns, cells, strict=True):
                if column == "__name__":
                    row["player_name"] = cell.text_content().strip()
                elif column == "__team__":
                    row["team"] = cell.text_content().strip()
                elif column is not None:
                    row[column] = _parse_scraped_float(cell.text_content())
            rows.append(row)

    # infer_schema_length=None -- see normalize_espn's docstring for why this
    # matters whenever heterogeneous-keyed rows are combined into one frame.
    return pl.DataFrame(rows, infer_schema_length=None)


# --- ADP (FantasyFootballCalculator) ---------------------------------------

# FantasyPros' own ADP page (fantasypros.com/nfl/adp/overall.php) renders its
# full player table client-side -- the server response only SSRs a 5-row
# teaser (confirmed live), the same "empty JS-mount shell" problem HANDOFF.md
# §4 already hit with Sharp Football Analysis. FantasyFootballCalculator's
# public JSON API has no such gap: real per-player ADP aggregated from actual
# recent live drafts (not an editorial ranking), with `high`/`low`/`stdev`
# already computed server-side -- better than SPEC §9.6's own high/low->sd
# fallback formula, so `adp_sd` is taken directly rather than derived. No
# auth, documented, widely used by other fantasy tools.
FFC_URL_TEMPLATE = (
    "https://fantasyfootballcalculator.com/api/v1/adp/{scoring}?teams={teams}&year={season}"
)

# FantasyFootballCalculator's own position codes -> this project's canonical
# ones (same DEF->DST/PK->K aliasing FantasySharks' "PK" bucket already
# needed).
FFC_POSITION_MAP: dict[str, str] = {
    "QB": "QB",
    "RB": "RB",
    "WR": "WR",
    "TE": "TE",
    "PK": "K",
    "DEF": "DST",
}


def _get_ffc_json(season: int, teams: int, scoring: str) -> dict[str, Any]:
    """Fetch FantasyFootballCalculator's ADP payload. The only network call
    this function makes."""
    response = _get_session().get(
        FFC_URL_TEMPLATE.format(scoring=scoring, teams=teams, season=season), timeout=30
    )
    response.raise_for_status()
    result: dict[str, Any] = response.json()
    return result


def fetch_adp(
    season: int,
    *,
    teams: int,
    scoring: str = "ppr",
    offline: bool | None = None,
    settings: Settings | None = None,
) -> Path:
    """Fetch consensus ADP (with spread) for `season` to
    data/raw/rankings/adp_<season>_<teams>_<scoring>.json.

    `teams` should be the league's own `LeagueFormat.n_teams` -- ADP shifts
    materially with league size (a 10-team league's pick 100 sits in a very
    different part of the player pool than a 12-team league's pick 100).
    `scoring` defaults to "ppr" (FantasyFootballCalculator's own vocabulary:
    standard/half-ppr/ppr) as the closest approximation to this project's
    real leagues (both score a full point per reception) -- unlike
    `apply_league_scoring`'s per-stat rescale, this is a coarse format
    match, not a real rescore: ADP reflects actual human drafts across many
    real league formats, not a single stat line this project can recompute.
    """
    settings = _resolve_settings(settings)
    return _fetch_json(
        filename=f"adp_{season}_{teams}_{scoring}.json",
        call_desc=f"GET {FFC_URL_TEMPLATE.format(scoring=scoring, teams=teams, season=season)}",
        cache_key="rankings_adp",
        load=lambda: _get_ffc_json(season, teams, scoring),
        row_count=lambda payload: len(payload.get("players", [])),
        artifact="adp",
        params=f"season={season} teams={teams} scoring={scoring}",
        offline=offline,
        settings=settings,
    )


def normalize_adp(payload: dict[str, Any], *, season: int) -> pl.DataFrame:
    """Extract each player's consensus ADP + spread into the canonical
    per-player ADP schema (`player_name`, `position`, `team`, `adp`,
    `adp_sd`, `adp_high`, `adp_low`, `times_drafted`, `bye_week`). Positions
    outside `FFC_POSITION_MAP` (none in real payloads today, but a future
    site change could add one) are dropped rather than guessed at.

    `bye_week` rides along for free -- FFC's payload already carries a `bye`
    field per player (task 0.12 needs it for SPEC §9.7's identity columns;
    no separate schedule source required for it).
    """
    rows: list[dict[str, Any]] = []
    for player in payload.get("players", []):
        position = FFC_POSITION_MAP.get(player.get("position"))
        if position is None:
            continue
        rows.append(
            {
                "source": "fantasyfootballcalculator",
                "season": season,
                "player_name": player["name"],
                "position": position,
                "team": player.get("team"),
                "adp": player["adp"],
                "adp_sd": player["stdev"],
                "adp_high": player["high"],
                "adp_low": player["low"],
                "times_drafted": player.get("times_drafted"),
                "bye_week": player.get("bye"),
            }
        )
    return pl.DataFrame(rows, infer_schema_length=None)


__all__ = [
    "CBS_POSITIONS",
    "CBS_URL_TEMPLATE",
    "ESPN_POSITION_MAP",
    "ESPN_STAT_ID_MAP",
    "ESPN_TEAM_MAP",
    "ESPN_URL_TEMPLATE",
    "FANTASYPROS_POSITIONS",
    "FANTASYPROS_URL",
    "FANTASYSHARKS_POS_BUCKETS",
    "FANTASYSHARKS_URL_TEMPLATE",
    "FFC_POSITION_MAP",
    "FFC_URL_TEMPLATE",
    "FFTODAY_POSITIONS",
    "FFTODAY_POS_ID",
    "FFTODAY_URL_TEMPLATE",
    "UnexpectedColumnLayoutError",
    "fetch_adp",
    "fetch_cbs",
    "fetch_espn",
    "fetch_fantasypros",
    "fetch_fantasysharks",
    "fetch_fftoday",
    "normalize_adp",
    "normalize_cbs",
    "normalize_espn",
    "normalize_fantasypros",
    "normalize_fantasysharks",
    "normalize_fftoday",
]
