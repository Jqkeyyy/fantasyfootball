"""Depth chart position feature (SPEC.md §11.2; task 1.14).

nflverse's weekly depth charts (`ingest.nflverse.fetch_depth_charts`,
task 0.x) were ingested but never normalized into a feature. SPEC §11.2
names "depth chart position" as an availability-model input (a starter
is more likely to record a snap than a third-stringer) -- this module
builds the pipeline that ingest task left undone.

Joined directly onto the target week, the same as `features.situation`/
`features.opponent` (see `features/build.py`'s own module docstring for
why): a team's real depth chart is genuinely public before that week's
kickoff, not data leaking from the game's own outcome.
"""

from __future__ import annotations

import polars as pl

from ffapp.features.registry import FeatureSpec, register

SOURCE_TABLE = "depth_charts"
_ALL_OFFENSE = ["QB", "RB", "WR", "TE"]
_OFFENSE_FORMATION = "Offense"


def normalize_depth_charts(raw: pl.DataFrame) -> pl.DataFrame:
    """nflreadpy's real weekly depth-chart rows -> one row per
    (player_id, season, week): `depth_chart_rank`, the player's own best
    (lowest-numbered) real offensive depth slot that week.

    Scoped to `formation == "Offense"` -- a player's Special Teams depth
    slot (punt returner, etc.) says nothing about offensive availability,
    SPEC §11.2's actual question here. A player can have more than one
    real `Offense` row the same week (eligible at more than one slot,
    e.g. a WR also listed as an emergency RB) -- the lowest rank (their
    most senior real role) wins, not an arbitrary one.
    """
    offense = raw.filter(
        (pl.col("formation") == _OFFENSE_FORMATION) & pl.col("gsis_id").is_not_null()
    ).with_columns(pl.col("depth_team").cast(pl.Int64).alias("depth_chart_rank"))
    return (
        offense.group_by(["gsis_id", "season", "week"])
        .agg(pl.col("depth_chart_rank").min())
        .rename({"gsis_id": "player_id"})
    )


def add_depth_chart_position(grid: pl.DataFrame, depth_charts: pl.DataFrame) -> pl.DataFrame:
    """`depth_chart_rank` (SPEC §11.2's "depth chart position"): the
    player's own best real offensive depth slot that week, joined
    directly (see module docstring). A player absent from that week's
    real depth chart (a late elevation, a real data gap) stays honestly
    null, not a guessed rank."""
    return grid.join(
        normalize_depth_charts(depth_charts).select(
            "player_id", "season", "week", "depth_chart_rank"
        ),
        on=["player_id", "season", "week"],
        how="left",
    )


def build_depth_chart_features(
    grid: pl.DataFrame,
    depth_charts: pl.DataFrame,
    *,
    registry: dict[str, FeatureSpec] | None = None,
) -> pl.DataFrame:
    """Adds `depth_chart_rank` and registers its `FeatureSpec`. `lag_weeks=1`
    despite the direct (unshifted) join -- same convention as
    `features.situation`/`features.opponent`: `lag_weeks` gates
    `assert_training_lag` (SPEC §10.1: "usable in a training matrix"), and
    a real pre-kickoff public fact, joined directly, is exactly as safe
    for training as a genuinely lag-shifted one -- it never sees the
    target week's own outcome.
    """
    result = add_depth_chart_position(grid, depth_charts)
    register(
        FeatureSpec(
            name="depth_chart_rank",
            description="player's own best real offensive depth-chart slot that week (1 = starter)",
            positions=_ALL_OFFENSE,
            window=None,
            source_table=SOURCE_TABLE,
            available_at_inference=True,
            lag_weeks=1,
        ),
        registry=registry,
    )
    return result


__all__ = [
    "SOURCE_TABLE",
    "add_depth_chart_position",
    "build_depth_chart_features",
    "normalize_depth_charts",
]
