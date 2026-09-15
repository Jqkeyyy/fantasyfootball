"""Fantasy-relevance-aware coverage audits for weekly projection sources."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from ffapp.tools.artifacts import atomic_write_parquet

RELEVANT_SEARCH_RANK = 300


def build_projection_coverage(
    projections: pl.DataFrame,
    players_dim: pl.DataFrame,
    rostered_sleeper_ids: set[str],
    *,
    relevant_search_rank: int = RELEVANT_SEARCH_RANK,
) -> pl.DataFrame:
    """Classify source omissions by roster and search relevance."""
    identity = players_dim.select(
        "player_id", "full_name", "position", "sleeper_id", "search_rank"
    ).unique("player_id")
    result = projections.select(
        "season", "week", "player_id", pl.col("mean").is_not_null().alias("projected")
    ).join(identity, on="player_id", how="left")
    rostered = pl.col("sleeper_id").is_in(list(rostered_sleeper_ids))
    top_search = pl.col("search_rank").is_not_null() & (
        pl.col("search_rank") <= relevant_search_rank
    )
    return result.with_columns(
        rostered.alias("rostered"),
        (rostered | top_search).alias("fantasy_relevant"),
        pl.when(pl.col("projected"))
        .then(pl.lit("projected"))
        .when(rostered)
        .then(pl.lit("rostered_missing"))
        .when(top_search)
        .then(pl.lit("top_search_missing"))
        .when(pl.col("full_name").is_null())
        .then(pl.lit("identity_unresolved"))
        .otherwise(pl.lit("deep_source_omission"))
        .alias("coverage_reason"),
    )


def write_projection_coverage(coverage: pl.DataFrame, path: Path) -> pl.DataFrame:
    """Upsert coverage by season/week, mirroring weekly projection artifacts."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = pl.read_parquet(path)
        keys = coverage.select("season", "week").unique()
        existing = existing.join(keys, on=["season", "week"], how="anti")
        combined = pl.concat([existing, coverage], how="diagonal_relaxed")
    else:
        combined = coverage
    atomic_write_parquet(combined, path)
    return combined


__all__ = [
    "RELEVANT_SEARCH_RANK",
    "build_projection_coverage",
    "write_projection_coverage",
]
