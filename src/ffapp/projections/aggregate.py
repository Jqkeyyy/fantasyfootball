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

import statistics
from collections.abc import Sequence

import polars as pl

from ffapp.ids.mapping import normalize_name
from ffapp.scoring.engine import score_stat_line
from ffapp.scoring.keymap import STAT_KEY_MAP, DirectStat

TRIM = 0.2

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


def add_join_key(df: pl.DataFrame) -> pl.DataFrame:
    """Cross-source player-matching key: normalized name + position (see
    module docstring for why this isn't the canonical player_id crosswalk).
    """
    return df.with_columns(
        (
            pl.col("player_name").map_elements(normalize_name, return_dtype=pl.Utf8)
            + "|"
            + pl.col("position")
        ).alias("join_key")
    )


def build_reference_curve(point_sources: Sequence[pl.DataFrame]) -> pl.DataFrame:
    """SPEC §9.2: for each position, the median points-at-positional-rank
    across every point-providing source (each already scored via
    `apply_league_scoring`). Returns columns [position, rank, ref_points].
    """
    ranked = [
        df.with_columns(
            pl.col("points")
            .rank(method="ordinal", descending=True)
            .over("position")
            .cast(pl.Int64)
            .alias("rank")
        )
        for df in point_sources
    ]
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


__all__ = [
    "TRIM",
    "add_join_key",
    "aggregate_projections",
    "apply_league_scoring",
    "build_reference_curve",
    "map_ranks_to_points",
]
