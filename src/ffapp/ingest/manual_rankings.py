"""Manually-exported overall rankings ("Pure Rankings" tab; task follow-up to
0.7's per-source aggregation).

CBS/ESPN/FantasySharks/FFToday have no native published rank in this app's
existing scrapers (`ingest/rankings.py` fetches their raw *stat projections*
instead, SPEC §9.2) -- confirmed live against CBS's own real "Fantasy
Experts" rankings page, which disagrees with a rank derived from applying
this league's scoring to CBS's stat projections (correct math, different
question, see `draft/board.py`'s own history). The project owner instead
downloaded each site's own real cheat-sheet/rankings export by hand and
dropped the files in `rankings/` at the repo root -- this module reads
those, schema-normalised the same way `ingest/rankings.py` normalises a
live fetch. Refreshing is manual: re-download and overwrite the same
filenames whenever fresher data is wanted (SPEC-ADDENDUM-02's offline-first
posture already treats "the source of truth is whatever's on disk" as
normal for this project).

Every export already carries a genuine *overall* rank (not just
positional) -- `RK`/`Rank`/`Overall Rank` -- which is the whole point:
`draft/board.py`'s "Pure Rankings" Consensus tab needs a real overall
order, not one this app invents from stat projections.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from ffapp.config import REPO_ROOT
from ffapp.ingest.rankings import FANTASYPROS_POSITIONS

MANUAL_RANKINGS_DIR = REPO_ROOT / "rankings"

_FILENAMES = {
    "cbs": "CBS_Fantasy_Expert_Rankings.xlsx",
    "draftsharks": "DraftSharks_PPR_Rankings.xlsx",
    "espn": "ESPN_PPR_Top300_Cheat_Sheet.xlsx",
    "fantasypros": "FantasyPros_PPR_Rankings.xlsx",
    "fantasysharks": "1_081420261202.csv",
    "fftoday": "FFToday_PPR_Cheatsheet.xlsx",
    "footballguys": "FootballGuys_Rankings.xlsx",
}

_CANONICAL_SCHEMA = pl.Schema(
    {
        "source": pl.Utf8,
        "player_name": pl.Utf8,
        "position": pl.Utf8,
        "team": pl.Utf8,
        "rank": pl.Float64,
    }
)


def _manual_file_path(source: str) -> Path:
    return MANUAL_RANKINGS_DIR / _FILENAMES[source]


def _finalize(df: pl.DataFrame, *, source: str) -> pl.DataFrame:
    """Filter to the six fantasy-relevant positions and cast to the
    canonical schema every `normalize_manual_*` function below returns."""
    return (
        df.filter(pl.col("position").is_in(FANTASYPROS_POSITIONS))
        .select(list(_CANONICAL_SCHEMA))
        .cast(_CANONICAL_SCHEMA)
    )


# --- CBS -----------------------------------------------------------------------
#
# No `team` column at all, and player names are abbreviated ("J. Gibbs") with
# no way to disambiguate same-initial players here -- resolved downstream in
# `draft/board.py` against `players_dim` (needs the full crosswalk, which is
# business logic, not schema normalisation -- CLAUDE.md's ingest/ purity
# rule). `player_name` stays abbreviated after this function; only the
# I-prefix repair below is schema normalisation (fixing this file's own raw
# export artefact), not name resolution.
#
# Real export bug, confirmed live: a player whose name ends in a roman-
# numeral suffix ("Kenneth Walker III" -> "K. Walker III") loses the last
# character of that suffix into the *next* cell -- "K. Walker II" in Player,
# "IRB" in Position, instead of "K. Walker III" / "RB". Every real
# occurrence in this file (K. Walker, L. Burden, O. Gadsden) follows the
# same pattern: Position starts with "I" followed by a real position code.
# Reversed here by moving that leading "I" back onto the player name, which
# is the only cell it could have overflowed from.

_CBS_POSITIONS = frozenset({"QB", "RB", "WR", "TE", "K", "DST"})


def fetch_manual_cbs() -> pl.DataFrame:
    return pl.read_excel(_manual_file_path("cbs"))


def normalize_manual_cbs(raw: pl.DataFrame) -> pl.DataFrame:
    bled = pl.col("Position").str.starts_with("I") & pl.col("Position").str.slice(1).is_in(
        _CBS_POSITIONS
    )
    repaired = raw.with_columns(
        pl.when(bled).then(pl.col("Player") + "I").otherwise(pl.col("Player")).alias("Player"),
        pl.when(bled)
        .then(pl.col("Position").str.slice(1))
        .otherwise(pl.col("Position"))
        .alias("Position"),
    )
    return _finalize(
        repaired.select(
            pl.lit("cbs").alias("source"),
            pl.col("Player").alias("player_name"),
            pl.col("Position").alias("position"),
            pl.lit(None, dtype=pl.Utf8).alias("team"),
            pl.col("Rank").alias("rank"),
        ),
        source="cbs",
    )


# --- ESPN ------------------------------------------------------------------------


def fetch_manual_espn() -> pl.DataFrame:
    return pl.read_excel(_manual_file_path("espn"))


def normalize_manual_espn(raw: pl.DataFrame) -> pl.DataFrame:
    return _finalize(
        raw.select(
            pl.lit("espn").alias("source"),
            pl.col("Player").alias("player_name"),
            pl.col("Position").alias("position"),
            pl.col("Team").alias("team"),
            pl.col("Overall Rank").alias("rank"),
        ),
        source="espn",
    )


# --- FantasyPros -------------------------------------------------------------


def fetch_manual_fantasypros() -> pl.DataFrame:
    return pl.read_excel(_manual_file_path("fantasypros"))


def normalize_manual_fantasypros(raw: pl.DataFrame) -> pl.DataFrame:
    return _finalize(
        raw.select(
            pl.lit("fantasypros").alias("source"),
            pl.col("Player").alias("player_name"),
            pl.col("Position").alias("position"),
            pl.col("Team").alias("team"),
            pl.col("RK").alias("rank"),
        ),
        source="fantasypros",
    )


# --- FFToday -----------------------------------------------------------------
#
# Position is spelled "D/ST", not "DST" -- the only one of these seven
# exports to use the slash form.


def fetch_manual_fftoday() -> pl.DataFrame:
    return pl.read_excel(_manual_file_path("fftoday"))


def normalize_manual_fftoday(raw: pl.DataFrame) -> pl.DataFrame:
    return _finalize(
        raw.select(
            pl.lit("fftoday").alias("source"),
            pl.col("Player").alias("player_name"),
            pl.col("Position").replace({"D/ST": "DST"}).alias("position"),
            pl.col("Team").alias("team"),
            pl.col("Rank").alias("rank"),
        ),
        source="fftoday",
    )


# --- FootballGuys --------------------------------------------------------------
#
# Position uses "PK"/"TD", same two non-standard tokens `ingest/rankings.py`'s
# own live FootballGuys scraper already normalises (`FOOTBALLGUYS_POSITION_MAP`).
# `Rank` is this file's own ranking; `Consensus Rank` is FootballGuys' own
# separate cross-site meta-consensus stat -- not used here, since this app
# builds its own consensus from all seven real sources instead.


def fetch_manual_footballguys() -> pl.DataFrame:
    return pl.read_excel(_manual_file_path("footballguys"))


def normalize_manual_footballguys(raw: pl.DataFrame) -> pl.DataFrame:
    return _finalize(
        raw.select(
            pl.lit("footballguys").alias("source"),
            pl.col("Player").alias("player_name"),
            pl.col("Position").replace({"PK": "K", "TD": "DST"}).alias("position"),
            pl.col("Team").alias("team"),
            pl.col("Rank").alias("rank"),
        ),
        source="footballguys",
    )


# --- DraftSharks ---------------------------------------------------------------
#
# Position uses "DEF", not "DST".


def fetch_manual_draftsharks() -> pl.DataFrame:
    return pl.read_excel(_manual_file_path("draftsharks"))


def normalize_manual_draftsharks(raw: pl.DataFrame) -> pl.DataFrame:
    return _finalize(
        raw.select(
            pl.lit("draftsharks").alias("source"),
            pl.col("Player").alias("player_name"),
            pl.col("Position").replace({"DEF": "DST"}).alias("position"),
            pl.col("Team").alias("team"),
            pl.col("RK").alias("rank"),
        ),
        source="draftsharks",
    )


# --- FantasySharks ---------------------------------------------------------------
#
# A CSV, not xlsx (`skip_rows=1`: the real file's first line is a title row,
# "Draft Planner Cheat Sheet,VBD2,...", not the header). Player name is
# split First/Last; Position includes real IDP codes (DL/DB/LB) this
# league doesn't start -- dropped by `_finalize`'s position filter, same as
# every other source's non-fantasy-relevant rows. Position uses "D", not
# "DST".


def fetch_manual_fantasysharks() -> pl.DataFrame:
    return pl.read_csv(_manual_file_path("fantasysharks"), skip_rows=1)


def normalize_manual_fantasysharks(raw: pl.DataFrame) -> pl.DataFrame:
    return _finalize(
        raw.select(
            pl.lit("fantasysharks").alias("source"),
            (pl.col("First Name") + " " + pl.col("Last Name")).alias("player_name"),
            pl.col("Position").replace({"D": "DST"}).alias("position"),
            pl.col("Team").alias("team"),
            pl.col("Rank").alias("rank"),
        ),
        source="fantasysharks",
    )


MANUAL_RANKING_FETCHERS = {
    "cbs": (fetch_manual_cbs, normalize_manual_cbs),
    "draftsharks": (fetch_manual_draftsharks, normalize_manual_draftsharks),
    "espn": (fetch_manual_espn, normalize_manual_espn),
    "fantasypros": (fetch_manual_fantasypros, normalize_manual_fantasypros),
    "fantasysharks": (fetch_manual_fantasysharks, normalize_manual_fantasysharks),
    "fftoday": (fetch_manual_fftoday, normalize_manual_fftoday),
    "footballguys": (fetch_manual_footballguys, normalize_manual_footballguys),
}


__all__ = [
    "MANUAL_RANKINGS_DIR",
    "MANUAL_RANKING_FETCHERS",
    "fetch_manual_cbs",
    "fetch_manual_draftsharks",
    "fetch_manual_espn",
    "fetch_manual_fantasypros",
    "fetch_manual_fantasysharks",
    "fetch_manual_fftoday",
    "fetch_manual_footballguys",
    "normalize_manual_cbs",
    "normalize_manual_draftsharks",
    "normalize_manual_espn",
    "normalize_manual_fantasypros",
    "normalize_manual_fantasysharks",
    "normalize_manual_fftoday",
    "normalize_manual_footballguys",
]
