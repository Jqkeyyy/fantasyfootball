"""Player age feature (SPEC.md §11.2; task 1.14).

No dedicated earlier task owns this block (same precedent as
`features.situation`'s own docstring) -- `age` is mechanical derivation
from a column (`birth_date`) `rosters` already carries (real nflverse
weekly roster data, task 1.9's own `ingest.nflverse.fetch_rosters`), not
a new ingestion.

Keyed directly by `gsis_id`/`player_id` -- unlike the draft board's
`projections.games_played.player_ages_from_players_dim`, which resolves
age through a normalized name `join_key` only because *that* context's
external ranking sources have no `gsis_id` of their own. Here, every row
already carries the real canonical id, so no name-matching detour (and
its associated silent-drop risk, CLAUDE.md rule 4) is needed.
"""

from __future__ import annotations

import polars as pl

from ffapp.features.registry import FeatureSpec, register

SOURCE_TABLE = "rosters"
_ALL_OFFENSE = ["QB", "RB", "WR", "TE"]
_AS_OF_UTC_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
_DAYS_PER_YEAR = 365.25


def add_player_age(grid: pl.DataFrame, rosters: pl.DataFrame) -> pl.DataFrame:
    """`age`: fractional years as of this row's own `as_of_utc` (that
    week's real kickoff) -- requires `_add_as_of_utc` to already have run.
    Computed from `rosters`' own real `birth_date`, one lookup per
    `player_id` (birth dates don't vary by season/week, so `rosters` is
    deduplicated to one row per player first).
    """
    birth_dates = (
        rosters.select(pl.col("gsis_id").alias("player_id"), "birth_date")
        .drop_nulls()
        .unique(subset="player_id")
    )
    as_of_date = (
        pl.col("as_of_utc").str.strptime(pl.Datetime(time_zone="UTC"), _AS_OF_UTC_FORMAT).dt.date()
    )
    return (
        grid.join(birth_dates, on="player_id", how="left")
        .with_columns(
            ((as_of_date - pl.col("birth_date")).dt.total_days() / _DAYS_PER_YEAR).alias("age")
        )
        .drop("birth_date")
    )


def build_player_bio_features(
    grid: pl.DataFrame,
    rosters: pl.DataFrame,
    *,
    registry: dict[str, FeatureSpec] | None = None,
) -> pl.DataFrame:
    """Adds `age` and registers its `FeatureSpec`. `lag_weeks=1` despite
    the direct join -- same convention as `features.situation`/
    `features.opponent`/`features.depth_chart`: a player's real birth
    date, known long before kickoff, is exactly as safe for training as
    a genuinely lag-shifted feature."""
    result = add_player_age(grid, rosters)
    register(
        FeatureSpec(
            name="age",
            description="fractional age in years as of this row's own as_of_utc",
            positions=_ALL_OFFENSE,
            window=None,
            source_table=SOURCE_TABLE,
            available_at_inference=True,
            lag_weeks=1,
        ),
        registry=registry,
    )
    return result


__all__ = ["SOURCE_TABLE", "add_player_age", "build_player_bio_features"]
