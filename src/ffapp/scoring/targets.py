"""Derive league-specific model targets from shared weekly stat lines."""

from __future__ import annotations

import polars as pl

from ffapp.scoring.engine import score_stat_line

_KEYS = ["player_id", "season", "week"]


class DuplicateStatRowsError(ValueError):
    """The canonical stat table has more than one row for a player-week."""


def apply_league_scoring_target(
    features: pl.DataFrame,
    player_week_stats: pl.DataFrame,
    scoring_settings: dict[str, float],
) -> pl.DataFrame:
    """Return features with ``target`` recomputed for one league.

    A player in the feature row universe without a stat row is retained with a
    zero target. This is the hurdle model's intentional inactive-player
    convention, not a missing-data imputation.
    """
    duplicates = player_week_stats.group_by(_KEYS).len().filter(pl.col("len") > 1)
    if not duplicates.is_empty():
        sample = duplicates.select(_KEYS).head(5).to_dicts()
        raise DuplicateStatRowsError(
            "player_week_stats must contain one row per player-week; "
            f"duplicate keys include {sample}"
        )

    scored = player_week_stats.with_columns(
        score_stat_line(player_week_stats, scoring_settings).alias("target")
    ).select(*_KEYS, "target")
    base = features.drop("target") if "target" in features.columns else features
    return base.join(scored, on=_KEYS, how="left").with_columns(
        pl.col("target").fill_null(0.0)
    )


__all__ = ["DuplicateStatRowsError", "apply_league_scoring_target"]
