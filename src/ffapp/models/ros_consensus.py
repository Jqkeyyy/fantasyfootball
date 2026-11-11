"""Season-long consensus resolution for future weeks
(`SPEC-ADDENDUM-04.md` §D.1's amendment -- see `docs/JOURNAL.md`'s
2026-08-16 entry). FantasyPros' weekly archive only ever publishes the
CURRENT week; there is no future-week weekly consensus to fetch. The six
other real sources this project already knows how to fetch
(`tools.prediction_log.SEASON_SOURCES`) return real season-long totals
instead -- this module resolves each one to a real REMAINING-season
value (full-season minus real actuals-to-date, unless
`tools.prediction_log.season_source_trend` has already confirmed that
source is a genuine rest-of-season signal) and aggregates them with the
same trimmed mean `projections.aggregate.aggregate_projections` already
uses for the draft board -- one real number, this player's own real
remaining-season consensus level. `models.ros_shape` (a separate module)
allocates that level across real remaining weeks; this module never
looks at individual weeks at all.
"""

from __future__ import annotations

from datetime import datetime

import polars as pl

from ffapp.config import Settings
from ffapp.projections.aggregate import aggregate_projections, build_reference_curve
from ffapp.tools.prediction_log import (
    POINT_SOURCES,
    RANK_SOURCES,
    SEASON_SOURCES,
    fetch_point_source,
    fetch_rank_source,
)

_DEFAULT_BRANCH = "subtracted"
_MIN_LIVE_SOURCES = 2
# A "season total" source whose typical value is under this fraction of the
# other sources' is really publishing a per-game number (seen live from CBS:
# ~25 for a player the others put near 200). Subtracting season actuals from
# it zeroes out exactly the best players.
_OFF_SCALE_RATIO = 0.35


def _off_scale_sources(season_points: dict[str, pl.DataFrame]) -> set[str]:
    """Sources whose median value is far below the median of the others."""
    medians = {
        name: float(df["points"].median())  # type: ignore[arg-type]
        for name, df in season_points.items()
        if not df.is_empty() and df["points"].median() is not None
    }
    off: set[str] = set()
    for name, median in medians.items():
        others = [m for other, m in medians.items() if other != name]
        if len(others) < 2:
            continue
        reference = sorted(others)[len(others) // 2]
        if reference > 0 and median < _OFF_SCALE_RATIO * reference:
            off.add(name)
    return off


def fetch_season_consensus(
    season: int,
    scoring_settings: dict[str, float],
    players_dim: pl.DataFrame,
    *,
    offline: bool | None,
    settings: Settings,
    now: datetime,
) -> dict[str, pl.DataFrame]:
    """Real live fetch of the six season-long sources, reusing
    `tools.prediction_log`'s own real fetchers (promoted to public in
    this same task) rather than a third copy of the same seven-source
    fetch logic (`draft.board` has the first, `prediction_log` the
    second, both for their own real purposes). Returns
    `{source_name: player_id/position/team/points}`, `SEASON_SOURCES`
    only -- `fantasypros`/weekly sources are the current week's own
    unchanged job, not this module's.
    """
    point_results: dict[str, pl.DataFrame] = {}
    real_ranked = []
    for name in POINT_SOURCES:
        result = fetch_point_source(
            name, season, scoring_settings, players_dim, offline=offline, settings=settings, now=now
        )
        point_results[name] = result.points
        if result.ranked is not None and result.ranked.height > 0:
            real_ranked.append(result.ranked)
    reference_curve = (
        build_reference_curve(real_ranked)
        if real_ranked
        else pl.DataFrame(
            schema={"position": pl.String, "rank": pl.Int64, "ref_points": pl.Float64}
        )
    )

    for name in RANK_SOURCES:
        result = fetch_rank_source(
            name, season, reference_curve, players_dim, offline=offline, settings=settings, now=now
        )
        point_results[name] = result.points

    return {name: point_results[name] for name in SEASON_SOURCES}


def resolve_remaining_value(
    season_points: dict[str, pl.DataFrame],
    trend_by_source: dict[str, str],
    actuals_to_date: pl.DataFrame,
) -> pl.DataFrame:
    """Per source, per real player: `points` already converted to a real
    remaining-season value, plus which real `branch` was taken (§D's
    requirement 1, "log which branch each source took"). `trend_by_source`
    is `tools.prediction_log.season_source_trend`'s own real per-source
    `trend` ("declining"/"flat"/"insufficient_data") -- only "declining"
    (a real, already-confirmed rest-of-season signal) is used directly;
    every other case, INCLUDING a source entirely absent from
    `trend_by_source` (never yet logged), defaults to the safer
    full-season branch and subtracts `actuals_to_date`, clipped at 0 (a
    remaining season can't be worth negative points). `actuals_to_date`:
    `player_id, actual_points_to_date` -- a player with no real row there
    (no games logged yet, e.g. a rookie) subtracts 0, not null.
    """
    usable = {
        name: df
        for name, df in season_points.items()
        if not df.is_empty() and name not in _off_scale_sources(season_points)
    }
    # Once at least `_MIN_LIVE_SOURCES` sources are confirmed to update for
    # games already played, the static preseason ones only add bias:
    # "preseason total minus actuals" punishes any player beating his
    # preseason projection (a 25-PPG breakout ends up with almost nothing
    # left), so drop them.
    live = [name for name in usable if trend_by_source.get(name) == "declining"]
    if len(live) >= _MIN_LIVE_SOURCES:
        usable = {name: usable[name] for name in live}

    frames: list[pl.DataFrame] = []
    for name, points_df in usable.items():
        trend = trend_by_source.get(name, _DEFAULT_BRANCH)
        branch = "ros_direct" if trend == "declining" else "subtracted"
        with_actuals = points_df.join(actuals_to_date, on="player_id", how="left").with_columns(
            pl.col("actual_points_to_date").fill_null(0.0)
        )
        if branch == "ros_direct":
            resolved_points = pl.col("points")
        else:
            resolved_points = (pl.col("points") - pl.col("actual_points_to_date")).clip(
                lower_bound=0.0
            )
        frames.append(
            with_actuals.with_columns(
                resolved_points.alias("points"),
                pl.lit(name).alias("source"),
                pl.lit(branch).alias("branch"),
            ).select("player_id", "position", "team", "points", "source", "branch")
        )
    if not frames:
        return pl.DataFrame(
            schema={
                "player_id": pl.String,
                "position": pl.String,
                "team": pl.String,
                "points": pl.Float64,
                "source": pl.String,
                "branch": pl.String,
            }
        )
    return pl.concat(frames, how="vertical_relaxed")


def aggregate_remaining_value(resolved: pl.DataFrame) -> pl.DataFrame:
    """Thin wrapper around `projections.aggregate.aggregate_projections`
    -- `resolved` must already carry `join_key`/`player_name`/`position`/
    `points` per source (the caller resolves `player_id` -> `join_key`/
    `player_name` via `players_dim` first, same pattern
    `tools.prediction_log._resolve_to_player_id` already establishes;
    this function's own real job is only the trim + rename, not player
    resolution). `aggregate_projections` accepts a sequence of per-source
    frames and concatenates internally, so passing the single already-
    stacked `resolved` frame as a one-element sequence is correct --
    every row still carries its own real `source`, `n_sources`/dispersion
    are still computed per real player across every source that covered
    them. Renames `aggregate_projections`'s own generic `proj_points` to
    `season_consensus_ros_points` so it never gets silently confused with
    the current week's own `b3_mean`/`mean`.
    """
    n_sources = resolved["source"].n_unique() if "source" in resolved.columns else 1
    aggregated = aggregate_projections([resolved], n_sources=max(n_sources, 1))
    return aggregated.rename({"proj_points": "season_consensus_ros_points"})


__all__ = [
    "aggregate_remaining_value",
    "fetch_season_consensus",
    "resolve_remaining_value",
]
