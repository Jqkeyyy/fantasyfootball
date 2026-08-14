"""Consensus projection aggregation (SPEC.md §9.1-9.2; task 0.7).

Applies league scoring to each source's per-stat projections individually
(§9.2 point 2: rescale BEFORE aggregating, not after -- generic-PPR-then-
correct is a different, wrong operation for any league whose scoring isn't
a scalar multiple of PPR), maps ranks-only sources onto the value scale via
a reference curve built from the point-providing sources, then aggregates
with a 20% trimmed mean, keeping per-player dispersion and coverage.

Business logic lives here, not in ingest/rankings.py (CLAUDE.md's ingest/
purity rule) -- ingest/ only fetches and schema-normalises each source;
score_stat_line() is the same function scoring/stats.py uses for nflverse
actuals, called here unmodified.

Player matching across sources is a normalized (name, position) key, not
task 0.3's canonical player_id crosswalk -- a deliberate simplification for
this task (see HANDOFF.md §4). Resolving projections to canonical
player_id is a reasonable follow-up, not required by 0.7's acceptance bar.
"""

from __future__ import annotations

import logging
import statistics
from collections.abc import Sequence

import polars as pl

from ffapp.ids.mapping import normalize_name
from ffapp.scoring.engine import score_stat_line
from ffapp.scoring.keymap import STAT_KEY_MAP, DirectStat

logger = logging.getLogger("ffapp.projections.aggregate")

TRIM = 0.2

# DST rows arrive under wildly different spellings per source -- confirmed
# live against every real per-stat source's own actual output: ESPN uses
# "{Nickname} D/ST" ("Texans D/ST"), FantasySharks/FFToday use the full
# "{City} {Nickname}" ("Houston Texans"), CBS uses the city alone, or a
# disambiguated "L.A."/"N.Y." prefix for the two shared-city pairs
# ("L.A. Rams", "N.Y. Jets"), and the ADP source (FantasyFootballCalculator)
# uses "{City} Defense" ("Houston Defense", "LA Rams Defense" -- no periods,
# unlike CBS's own "L.A."/"N.Y." style). Before this table existed, none of
# these spellings normalized to the same join_key, so every DST silently
# never aggregated across sources at all -- each board row was really one
# source's single, uncross-validated opinion (n_sources=1, dispersion=0.0
# on every one of the 32 real teams, confirmed live), *and* ADP/survival
# probability came back null for every DST too, same root cause.
#
# 26 of the 32 ADP names below are directly confirmed live; ADP coverage is
# thin for 6 teams (ARI, CAR, IND, KC, LV, TB -- real, not a bug, same as
# any other thin-ADP-coverage player elsewhere on the board). Those 6 follow
# the exact "{City} Defense" pattern all 26 confirmed ones share, using the
# same city spelling as the CBS column, so they're filled in rather than
# left as a real gap in a table that exists specifically to close gaps.
_DST_TEAMS: list[tuple[str, str, str, str, str]] = [
    ("ARI", "Arizona Cardinals", "Cardinals D/ST", "Arizona", "Arizona Defense"),
    ("ATL", "Atlanta Falcons", "Falcons D/ST", "Atlanta", "Atlanta Defense"),
    ("BAL", "Baltimore Ravens", "Ravens D/ST", "Baltimore", "Baltimore Defense"),
    ("BUF", "Buffalo Bills", "Bills D/ST", "Buffalo", "Buffalo Defense"),
    ("CAR", "Carolina Panthers", "Panthers D/ST", "Carolina", "Carolina Defense"),
    ("CHI", "Chicago Bears", "Bears D/ST", "Chicago", "Chicago Defense"),
    ("CIN", "Cincinnati Bengals", "Bengals D/ST", "Cincinnati", "Cincinnati Defense"),
    ("CLE", "Cleveland Browns", "Browns D/ST", "Cleveland", "Cleveland Defense"),
    ("DAL", "Dallas Cowboys", "Cowboys D/ST", "Dallas", "Dallas Defense"),
    ("DEN", "Denver Broncos", "Broncos D/ST", "Denver", "Denver Defense"),
    ("DET", "Detroit Lions", "Lions D/ST", "Detroit", "Detroit Defense"),
    ("GB", "Green Bay Packers", "Packers D/ST", "Green Bay", "Green Bay Defense"),
    ("HOU", "Houston Texans", "Texans D/ST", "Houston", "Houston Defense"),
    ("IND", "Indianapolis Colts", "Colts D/ST", "Indianapolis", "Indianapolis Defense"),
    ("JAX", "Jacksonville Jaguars", "Jaguars D/ST", "Jacksonville", "Jacksonville Defense"),
    ("KC", "Kansas City Chiefs", "Chiefs D/ST", "Kansas City", "Kansas City Defense"),
    ("LV", "Las Vegas Raiders", "Raiders D/ST", "Las Vegas", "Las Vegas Defense"),
    ("LAR", "Los Angeles Rams", "Rams D/ST", "L.A. Rams", "LA Rams Defense"),
    ("LAC", "Los Angeles Chargers", "Chargers D/ST", "L.A. Chargers", "LA Chargers Defense"),
    ("MIA", "Miami Dolphins", "Dolphins D/ST", "Miami", "Miami Defense"),
    ("MIN", "Minnesota Vikings", "Vikings D/ST", "Minnesota", "Minnesota Defense"),
    ("NE", "New England Patriots", "Patriots D/ST", "New England", "New England Defense"),
    ("NO", "New Orleans Saints", "Saints D/ST", "New Orleans", "New Orleans Defense"),
    ("NYG", "New York Giants", "Giants D/ST", "N.Y. Giants", "NY Giants Defense"),
    ("NYJ", "New York Jets", "Jets D/ST", "N.Y. Jets", "NY Jets Defense"),
    ("PHI", "Philadelphia Eagles", "Eagles D/ST", "Philadelphia", "Philadelphia Defense"),
    ("PIT", "Pittsburgh Steelers", "Steelers D/ST", "Pittsburgh", "Pittsburgh Defense"),
    ("SF", "San Francisco 49ers", "49ers D/ST", "San Francisco", "San Francisco Defense"),
    ("SEA", "Seattle Seahawks", "Seahawks D/ST", "Seattle", "Seattle Defense"),
    ("TB", "Tampa Bay Buccaneers", "Buccaneers D/ST", "Tampa Bay", "Tampa Bay Defense"),
    ("TEN", "Tennessee Titans", "Titans D/ST", "Tennessee", "Tennessee Defense"),
    ("WSH", "Washington Commanders", "Commanders D/ST", "Washington", "Washington Defense"),
]

DST_TEAM_ABBREVIATIONS: dict[str, str] = {
    normalize_name(variant): abbrev
    for abbrev, full_name, espn_name, cbs_name, adp_name in _DST_TEAMS
    for variant in (full_name, espn_name, cbs_name, adp_name)
}
DST_CANONICAL_NAMES: dict[str, str] = {
    abbrev: full_name for abbrev, full_name, *_rest in _DST_TEAMS
}

_DIRECT_STAT_COLUMNS = {
    spec.column for spec in STAT_KEY_MAP.values() if isinstance(spec, DirectStat)
}
# Columns referenced by DerivedStat compute closures in keymap.py, not
# already covered by _DIRECT_STAT_COLUMNS -- enumerated by hand from each
# closure's own column references (no way to introspect a closure's column
# reads generically): FG per-yard/legacy-bucket scoring reads fg_made_list;
# DST points-allowed buckets read points_allowed; fgmiss/xpmiss sum a
# miss+blocked pair; the six DST-gated "team defense" keys each read one
# team_stats-shaped column.
_DERIVED_STAT_EXTRA_COLUMNS = {
    "fg_made_list",
    "points_allowed",
    "pat_missed",
    "pat_blocked",
    "fg_missed",
    "fg_blocked",
    "def_sacks",
    "def_interceptions",
    "def_fumbles_forced",
    "fumble_recovery_opp",
    "def_safeties",
    "special_teams_tds",
    "special_teams_forced_fumbles",
    "special_teams_fumble_recoveries",
}
_ALL_SCORING_STAT_COLUMNS = _DIRECT_STAT_COLUMNS | _DERIVED_STAT_EXTRA_COLUMNS


def _with_missing_stat_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Add any scoring-relevant column score_stat_line might reference that
    this source's normalized output doesn't have, as an all-null column --
    score_stat_line's DirectStat/DerivedStat paths both `fill_null(0)`
    whatever they read, so a stat this particular source never publishes
    simply contributes 0 to that source's points, rather than
    score_stat_line raising ColumnNotFoundError on `stats[column]` for a
    column that isn't present at all (not just null).
    """
    missing = _ALL_SCORING_STAT_COLUMNS - set(df.columns)
    if not missing:
        return df
    return df.with_columns(
        pl.lit(None, dtype=pl.Utf8 if col == "fg_made_list" else pl.Float64).alias(col)
        for col in missing
    )


def apply_league_scoring(df: pl.DataFrame, scoring_settings: dict[str, float]) -> pl.DataFrame:
    """Add a `points` column to a per-stat-source normalized projections
    DataFrame (one of ingest/rankings.py's normalize_* outputs) by applying
    league scoring via the same score_stat_line the actuals pipeline uses.
    """
    filled = _with_missing_stat_columns(df)
    points = score_stat_line(filled, scoring_settings)
    return df.with_columns(points.alias("points"))


def _canonicalize_dst_player_names(df: pl.DataFrame) -> pl.DataFrame:
    """Rewrite `player_name` to one canonical "City Nickname" form per real
    NFL team for every DST row (see `DST_TEAM_ABBREVIATIONS`'s own module
    -level comment) -- every source's own spelling collapses onto the same
    string before `add_join_key` builds a join_key from it, so DST rows
    finally aggregate across sources the way every other position already
    does. A DST name this table doesn't recognise (a future source-side
    spelling change) is left as-is and logged, not silently dropped or
    crashed on -- no worse than the pre-fix behaviour for that one row.
    """
    if "position" not in df.columns or not (df["position"] == "DST").any():
        return df

    normalized = pl.col("player_name").map_elements(normalize_name, return_dtype=pl.Utf8)
    abbrev = normalized.replace_strict(DST_TEAM_ABBREVIATIONS, default=None, return_dtype=pl.Utf8)
    canonical = abbrev.replace_strict(DST_CANONICAL_NAMES, default=None, return_dtype=pl.Utf8)

    is_dst = pl.col("position") == "DST"
    unresolved = df.filter(is_dst & canonical.is_null())
    if unresolved.height:
        logger.warning(
            "%d DST row(s) have an unrecognised team name, left uncanonicalised: %s",
            unresolved.height,
            unresolved["player_name"].to_list(),
        )

    return df.with_columns(
        pl.when(is_dst & canonical.is_not_null())
        .then(canonical)
        .otherwise(pl.col("player_name"))
        .alias("player_name")
    )


# A source's own nickname/given-name spelling for a player normalizes to a
# different string than the spelling every other source uses, so the two
# never share a join_key and the same real player shows up twice on the
# board -- same root cause as the DST spelling problem above, just for an
# individual player instead of all 32 teams. Confirmed live: one source
# spells Kenneth Walker III as "Ken Walker III", which strips to "ken
# walker" -- distinct from every other source's "kenneth walker". Keyed by
# the alias's own normalized form; add one entry per confirmed case rather
# than attempting general nickname resolution.
_PLAYER_NAME_ALIASES: dict[str, str] = {
    "ken walker": "Kenneth Walker III",
}


def _canonicalize_aliased_player_names(df: pl.DataFrame) -> pl.DataFrame:
    """Rewrite `player_name` to the canonical spelling for any row matching
    `_PLAYER_NAME_ALIASES`, before `add_join_key` derives a join_key from it.
    """
    normalized = pl.col("player_name").map_elements(normalize_name, return_dtype=pl.Utf8)
    canonical = normalized.replace_strict(_PLAYER_NAME_ALIASES, default=None, return_dtype=pl.Utf8)
    return df.with_columns(
        pl.when(canonical.is_not_null())
        .then(canonical)
        .otherwise(pl.col("player_name"))
        .alias("player_name")
    )


def add_join_key(df: pl.DataFrame) -> pl.DataFrame:
    """Cross-source player-matching key: normalized name + position (see
    module docstring for why this isn't the canonical player_id crosswalk).

    DST rows are canonicalised first (`_canonicalize_dst_player_names`) --
    every source spells DST team names differently, and without this the
    same real team's rows from different sources would build different
    join_keys and never aggregate together at all. Individual-player alias
    spellings (`_PLAYER_NAME_ALIASES`) are canonicalised the same way right
    after.
    """
    canonicalized = _canonicalize_dst_player_names(df)
    aliased = _canonicalize_aliased_player_names(canonicalized)
    return aliased.with_columns(
        (
            pl.col("player_name").map_elements(normalize_name, return_dtype=pl.Utf8)
            + "|"
            + pl.col("position")
        ).alias("join_key")
    )


def rank_within_position(df: pl.DataFrame) -> pl.DataFrame:
    """Add an ordinal `rank` column (1 = best), ranking `points` descending
    within each `position` -- for a point-projection source, which never
    publishes a native rank of its own, this is the only way to express
    that source's opinion as a rank in the same shape a ranks-only source
    (FantasyPros, FootballGuys, DraftSharks) already carries natively.
    """
    return df.with_columns(
        pl.col("points")
        .rank(method="ordinal", descending=True)
        .over("position")
        .cast(pl.Int64)
        .alias("rank")
    )


def build_reference_curve(point_sources: Sequence[pl.DataFrame]) -> pl.DataFrame:
    """SPEC §9.2: for each position, the median points-at-positional-rank
    across every point-providing source (each already scored via
    `apply_league_scoring`). Returns columns [position, rank, ref_points].
    """
    ranked = [rank_within_position(df) for df in point_sources]
    combined = pl.concat(
        [df.select(["position", "rank", "points"]) for df in ranked], how="vertical"
    )
    return combined.group_by(["position", "rank"]).agg(
        pl.col("points").median().alias("ref_points")
    )


def map_ranks_to_points(rank_df: pl.DataFrame, reference_curve: pl.DataFrame) -> pl.DataFrame:
    """For a ranks-only source (e.g. FantasyPros), impute `points` for each
    row via the reference curve at that row's (position, rounded rank).
    Rows whose (position, rank) has no reference-curve entry get
    `points = null` -- thin coverage at the tail, per SPEC §9.2 flagged via
    `coverage`, not dropped from the source's own output.
    """
    with_int_rank = rank_df.with_columns(pl.col("rank").round(0).cast(pl.Int64))
    return with_int_rank.join(reference_curve, on=["position", "rank"], how="left").rename(
        {"ref_points": "points"}
    )


def _trimmed_mean(values: list[float]) -> float:
    n = len(values)
    trim_n = int(n * TRIM)
    ordered = sorted(values)
    trimmed = ordered[trim_n : n - trim_n] if n - 2 * trim_n > 0 else ordered
    return statistics.mean(trimmed)


def _dispersion(values: list[float]) -> float:
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def aggregate_projections(
    scored_sources: Sequence[pl.DataFrame], *, n_sources: int
) -> pl.DataFrame:
    """SPEC §9.2's final step: per player (join key = normalized name +
    position), `proj_points` (20% trimmed mean), `dispersion` (population
    stdev), `n_sources`, and `coverage` (`n_sources / n_sources_total`)
    across every source's `points` for that player.

    Every source's DataFrame here must already have a `points` column (via
    `apply_league_scoring` or `map_ranks_to_points`) and a `join_key`
    column (via `add_join_key`). Rows with `points = null` (a source that
    doesn't cover this player, or a ranks-only row past the reference
    curve's tail) don't count toward coverage.

    Grouped by `join_key` alone, not also by the literal `player_name`
    string -- confirmed live via task 0.14's replay testing: one source
    spelling a player "James Cook" and another spelling him "James Cook
    III" both normalize to the same join_key, but grouping by the raw name
    string too split one real player into two separate board rows (each
    getting only part of his real source coverage). `position` is already
    fully determined by `join_key` (it's literally part of how the key is
    built), so `.first()` on both is safe -- `position` never actually
    varies within a group; `player_name` picks whichever source's spelling
    happened to come first, an arbitrary but deterministic tie-break.
    """
    combined = pl.concat(
        [df.select(["join_key", "player_name", "position", "points"]) for df in scored_sources],
        how="vertical",
    ).filter(pl.col("points").is_not_null())

    return (
        combined.group_by("join_key")
        .agg(
            pl.col("player_name").first(),
            pl.col("position").first(),
            pl.col("points").alias("_points_list"),
        )
        .with_columns(
            pl.col("_points_list").list.len().alias("n_sources"),
            pl.col("_points_list")
            .map_elements(_trimmed_mean, return_dtype=pl.Float64)
            .alias("proj_points"),
            pl.col("_points_list")
            .map_elements(_dispersion, return_dtype=pl.Float64)
            .alias("dispersion"),
        )
        .with_columns((pl.col("n_sources") / n_sources).alias("coverage"))
        .drop("_points_list")
    )


def aggregate_source_ranks(source_frames: Sequence[pl.DataFrame]) -> pl.DataFrame:
    """The "no model" counterpart to `aggregate_projections`: per player,
    each source's own positional rank side by side (`rank_<source>`), plus
    `avg_rank`/`median_rank`/`rank_sd`/`n_sources` across whichever sources
    actually covered that player. No league scoring, no VOR -- just each
    source's own rank opinion, averaged.

    Every frame here must already have `join_key` (`add_join_key`) and
    `rank` (a ranks-only source's native rank, or a point source's own
    `rank_within_position`) columns. Grouped by `join_key` alone, same
    reasoning as `aggregate_projections`.

    `rank` is cast to Float64 here, not trusted as-is -- a ranks-only
    source's native rank is a real fractional consensus value (e.g.
    FantasyPros' ecr), while `rank_within_position` returns Int64 (needed
    to match `build_reference_curve`'s own join dtype, untouched here) --
    concatenating the two as-is raises a polars SchemaError, confirmed live
    running the real pipeline. Float64 loses nothing for either side.
    """
    combined = pl.concat(
        [
            df.select(
                "join_key",
                "player_name",
                "position",
                "source",
                pl.col("rank").cast(pl.Float64),
            )
            for df in source_frames
        ],
        how="vertical",
    ).filter(pl.col("rank").is_not_null())

    stats = (
        combined.group_by("join_key")
        .agg(
            pl.col("player_name").first(),
            pl.col("position").first(),
            pl.col("rank").alias("_rank_list"),
        )
        .with_columns(
            pl.col("_rank_list").list.len().alias("n_sources"),
            pl.col("_rank_list")
            .map_elements(statistics.mean, return_dtype=pl.Float64)
            .alias("avg_rank"),
            pl.col("_rank_list")
            .map_elements(statistics.median, return_dtype=pl.Float64)
            .alias("median_rank"),
            pl.col("_rank_list")
            .map_elements(_dispersion, return_dtype=pl.Float64)
            .alias("rank_sd"),
        )
        .drop("_rank_list")
    )

    wide = combined.pivot(on="source", index="join_key", values="rank", aggregate_function="first")
    wide = wide.rename({col: f"rank_{col}" for col in wide.columns if col != "join_key"})

    return stats.join(wide, on="join_key").sort("avg_rank")


__all__ = [
    "DST_CANONICAL_NAMES",
    "DST_TEAM_ABBREVIATIONS",
    "TRIM",
    "add_join_key",
    "aggregate_projections",
    "aggregate_source_ranks",
    "apply_league_scoring",
    "build_reference_curve",
    "map_ranks_to_points",
    "rank_within_position",
]
