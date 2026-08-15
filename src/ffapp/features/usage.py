"""The player usage feature block (SPEC.md §10.2 "Player usage"; task 1.6).

Windows are computed within-season only (grouped by `player_id` AND
`season`), not spanning the off-season -- a deliberate reading of SPEC's
own design, not stated explicitly there: `prior_season` exists as its own
window type specifically to carry a signal into early-season rows where
an in-season `ewm`/`std` window is still sparse or entirely absent (week 1
has zero prior in-season games). If `ewm_k` already smoothed across the
off-season, `prior_season` would be redundant with it; SPEC offers both
side by side because they answer different questions ("how has this
player performed *this season* so far" vs. "how did they perform *last
season*"), not because one subsumes the other.

Every feature here is a *trailing* value as of the end of its own row's
week, inclusive of that week's own game -- standard rolling-stats
practice, and exactly why every feature registered here declares
`lag_weeks=1`: a trailing stat computed through week W can only be used
to predict week W+1 onward, never week W itself. Task 1.9's feature-table
assembly is what actually performs that one-week shift when it joins a
player's history onto a target week; this module only computes the
trailing values themselves.

`xfp_minus_actual` and `points_std` need "actual" points, which this
project always means as *league-scored* points (CLAUDE.md rule 5 --
nothing hardcodes a league's format), computed via the same validated
`scoring.engine.score_stat_line` task 0.5's golden test checked against
Sleeper's own numbers (>=99% agreement, both real leagues) -- not a
generic PPR proxy. `scoring_settings` is a required parameter throughout
this module for exactly that reason.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import polars as pl

from ffapp.features.registry import FeatureSpec, register
from ffapp.scoring.engine import score_stat_line

SOURCE_TABLE = "player_week_usage"

_ALL_OFFENSE = ["QB", "RB", "WR", "TE"]
_PASS_CATCHERS = ["WR", "TE"]
PASS_CATCHERS_AND_RB = ["WR", "TE", "RB"]
_RB_ONLY = ["RB"]
RB_QB = ["RB", "QB"]
_QB_ONLY = ["QB"]

SNAP_PCT_CHANGE_THRESHOLD = 0.15


def _sort(df: pl.DataFrame) -> pl.DataFrame:
    return df.sort(["player_id", "season", "week"])


def ewm(df: pl.DataFrame, value_col: str, span: int, out_col: str) -> pl.DataFrame:
    """Exponentially weighted mean, span `span`, within-season, trailing
    through (and including) each row's own week -- see module docstring."""
    return _sort(df).with_columns(
        pl.col(value_col).ewm_mean(span=span).over(["player_id", "season"]).alias(out_col)
    )


def rolling_std(df: pl.DataFrame, value_col: str, window: int, out_col: str) -> pl.DataFrame:
    """Rolling standard deviation over the trailing `window` games,
    within-season (SPEC's `std_k`, e.g. `points_std`'s `std_8`). Needs at
    least 2 games to produce a real value; null before that, not 0 --
    a single data point has no variance to report."""
    return _sort(df).with_columns(
        pl.col(value_col)
        .rolling_std(window_size=window, min_samples=2)
        .over(["player_id", "season"])
        .alias(out_col)
    )


def season_to_date(df: pl.DataFrame, value_col: str, out_col: str) -> pl.DataFrame:
    """Cumulative mean within the season, through and including each row's
    own week."""
    sorted_df = _sort(df)
    return sorted_df.with_columns(
        (
            pl.col(value_col).cum_sum().over(["player_id", "season"])
            / pl.col(value_col).cum_count().over(["player_id", "season"])
        ).alias(out_col)
    )


def prior_season(df: pl.DataFrame, value_col: str, out_col: str) -> pl.DataFrame:
    """Each row's value from *last* season's average -- a single constant
    per (player, current season), the same for every week of that season.
    Null for a player's first tracked season (no prior season to look
    up)."""
    season_avg = df.group_by(["player_id", "season"]).agg(pl.col(value_col).mean().alias(out_col))
    lookup = season_avg.with_columns((pl.col("season") + 1).alias("season"))
    return df.join(lookup, on=["player_id", "season"], how="left")


def _apply_window(df: pl.DataFrame, raw_col: str, window: str, out_col: str) -> pl.DataFrame:
    if window.startswith("ewm_"):
        return ewm(df, raw_col, int(window.removeprefix("ewm_")), out_col)
    if window.startswith("std_"):
        return rolling_std(df, raw_col, int(window.removeprefix("std_")), out_col)
    if window == "season_to_date":
        return season_to_date(df, raw_col, out_col)
    if window == "prior_season":
        return prior_season(df, raw_col, out_col)
    raise ValueError(f"Unknown window type: {window!r}")


@dataclass(frozen=True)
class _WindowedFeature:
    raw_column: str
    name_base: str
    description: str
    positions: list[str]
    windows: list[str] = field(default_factory=list)


_WINDOWED_FEATURES = [
    _WindowedFeature(
        "offense_snap_pct",
        "snap_pct",
        "offensive snaps / team offensive snaps",
        _ALL_OFFENSE,
        ["ewm_3", "ewm_8", "prior_season"],
    ),
    _WindowedFeature(
        "target_share",
        "target_share",
        "targets / team pass attempts",
        PASS_CATCHERS_AND_RB,
        ["ewm_3", "ewm_8", "season_to_date"],
    ),
    _WindowedFeature(
        "air_yards_share",
        "air_yards_share",
        "player air yards / team air yards",
        _PASS_CATCHERS,
        ["ewm_4"],
    ),
    _WindowedFeature(
        "wopr",
        "wopr",
        "1.5*target_share + 0.7*air_yards_share",
        PASS_CATCHERS_AND_RB,
        ["ewm_4"],
    ),
    _WindowedFeature("adot", "adot", "air yards / targets", _PASS_CATCHERS, ["ewm_8"]),
    _WindowedFeature(
        "carry_share", "carry_share", "carries / team rush attempts", RB_QB, ["ewm_3", "ewm_8"]
    ),
    _WindowedFeature(
        "rz_touch_share",
        "rz_touch_share",
        "(rz targets + rz carries) / team rz touches",
        PASS_CATCHERS_AND_RB,
        ["ewm_6"],
    ),
    _WindowedFeature(
        "gz_carry_share",
        "gz_carry_share",
        "carries inside 5 / team carries inside 5",
        _RB_ONLY,
        ["ewm_6"],
    ),
    _WindowedFeature(
        "xfp",
        "xfp_per_game",
        "ffopportunity expected fantasy points",
        _ALL_OFFENSE,
        ["ewm_4", "season_to_date"],
    ),
    _WindowedFeature("attempts", "pass_attempts", "QB pass attempts, volume", _QB_ONLY, ["ewm_4"]),
    _WindowedFeature(
        "passing_cpoe", "cpoe", "completion percentage over expected", _QB_ONLY, ["ewm_4"]
    ),
    _WindowedFeature(
        "sack_rate_taken",
        "sack_rate_taken",
        "sacks / (pass attempts + sacks)",
        _QB_ONLY,
        ["ewm_4"],
    ),
    _WindowedFeature(
        "designed_rush_share",
        "designed_rush_share",
        "QB non-scramble rush attempts / team rush attempts",
        _QB_ONLY,
        ["ewm_6"],
    ),
    _WindowedFeature(
        "rushing_yards",
        "rush_yards_per_game",
        "QB rushing yards, floor signal",
        _QB_ONLY,
        ["ewm_6"],
    ),
]


def add_actual_points(
    player_week_stats: pl.DataFrame, scoring_settings: dict[str, float]
) -> pl.DataFrame:
    """League-scored actual points per player-week (see module docstring
    for why this must be league-scored, not a generic proxy)."""
    return player_week_stats.with_columns(
        score_stat_line(player_week_stats, scoring_settings).alias("actual_points")
    )


def _raw_metrics(
    player_week_usage: pl.DataFrame,
    player_week_stats: pl.DataFrame,
    scoring_settings: dict[str, float],
) -> pl.DataFrame:
    """Join `player_week_usage` with the QB-specific raw stats and
    league-scored `actual_points` from `player_week_stats`, and derive the
    two raw ratios that need both sources: `sack_rate_taken` and the
    `xfp_minus_actual` residual (both computed here, once, rather than
    inside the generic windowing loop, since neither is a straight
    passthrough column)."""
    stats_with_points = add_actual_points(player_week_stats, scoring_settings).select(
        "player_id",
        "season",
        "week",
        "attempts",
        "passing_cpoe",
        "sacks_suffered",
        "rushing_yards",
        "actual_points",
    )
    merged = player_week_usage.join(
        stats_with_points, on=["player_id", "season", "week"], how="left"
    )
    return merged.with_columns(
        pl.when((pl.col("attempts") + pl.col("sacks_suffered")) > 0)
        .then(pl.col("sacks_suffered") / (pl.col("attempts") + pl.col("sacks_suffered")))
        .otherwise(None)
        .alias("sack_rate_taken"),
        (pl.col("xfp") - pl.col("actual_points")).alias("_xfp_minus_actual_raw"),
    )


def weeks_in_current_role(
    df: pl.DataFrame,
    *,
    snap_pct_col: str = "offense_snap_pct",
    threshold: float = SNAP_PCT_CHANGE_THRESHOLD,
) -> pl.DataFrame:
    """SPEC §10.2: "weeks since snap_pct changed by >15pp." Read as a
    week-over-week jump detector (a discrete role change -- a backup
    elevated to starter, a starter benched -- shows up as one large
    single-week delta), not a comparison against a rolling baseline, and
    reset each season for consistency with every other window in this
    module. A player's first tracked week of a season has no prior week
    to compare to, so `weeks_in_current_role` starts at 0 there -- their
    entire known history so far *is* "the current role."
    """
    sorted_df = _sort(df)
    with_delta = sorted_df.with_columns(
        (pl.col(snap_pct_col) - pl.col(snap_pct_col).shift(1).over(["player_id", "season"]))
        .abs()
        .alias("_delta")
    )
    with_change_week = with_delta.with_columns(
        pl.when(pl.col("_delta") > threshold)
        .then(pl.col("week"))
        .otherwise(None)
        .forward_fill()
        .over(["player_id", "season"])
        .alias("_last_change_week")
    )
    return with_change_week.with_columns(
        pl.when(pl.col("_last_change_week").is_null())
        .then(pl.col("week") - pl.col("week").first().over(["player_id", "season"]))
        .otherwise(pl.col("week") - pl.col("_last_change_week"))
        .alias("weeks_in_current_role")
    ).drop(["_delta", "_last_change_week"])


def build_usage_features(
    player_week_usage: pl.DataFrame,
    player_week_stats: pl.DataFrame,
    scoring_settings: dict[str, float],
    *,
    registry: dict[str, FeatureSpec] | None = None,
) -> pl.DataFrame:
    """Assemble every SPEC §10.2 "Player usage" feature (task 1.6) and
    register each one's `FeatureSpec` (task 1.5's registry). Every
    registered feature declares `lag_weeks=1` and
    `available_at_inference=True` -- see module docstring for why the lag
    is uniform, and none of these features have an in-season availability
    gap the way route participation does (SPEC §10.5).
    """
    result = _raw_metrics(player_week_usage, player_week_stats, scoring_settings)

    for feature in _WINDOWED_FEATURES:
        for window in feature.windows:
            out_col = f"{feature.name_base}_{window}"
            result = _apply_window(result, feature.raw_column, window, out_col)
            register(
                FeatureSpec(
                    name=out_col,
                    description=feature.description,
                    positions=feature.positions,
                    window=window,
                    source_table=SOURCE_TABLE,
                    available_at_inference=True,
                    lag_weeks=1,
                ),
                registry=registry,
            )

    # snap_pct_trend: derived from two already-computed windows above, not
    # its own raw-value window (SPEC's own Windows column: "--").
    result = result.with_columns(
        (pl.col("snap_pct_ewm_3") - pl.col("snap_pct_ewm_8")).alias("snap_pct_trend")
    )
    register(
        FeatureSpec(
            name="snap_pct_trend",
            description="snap_pct ewm_3 minus ewm_8",
            positions=_ALL_OFFENSE,
            window=None,
            source_table=SOURCE_TABLE,
            available_at_inference=True,
            lag_weeks=1,
        ),
        registry=registry,
    )

    # xfp_minus_actual: ewm_6 of the raw per-week residual computed in
    # _raw_metrics.
    result = ewm(result, "_xfp_minus_actual_raw", 6, "xfp_minus_actual_ewm_6")
    register(
        FeatureSpec(
            name="xfp_minus_actual_ewm_6",
            description=(
                "xfp minus actual league-scored points -- efficiency residual, regresses "
                "hard, use as a negative indicator of sustainability"
            ),
            positions=_ALL_OFFENSE,
            window="ewm_6",
            source_table=SOURCE_TABLE,
            available_at_inference=True,
            lag_weeks=1,
        ),
        registry=registry,
    )

    # points_std: std_8 of actual league-scored points.
    result = rolling_std(result, "actual_points", 8, "points_std_std_8")
    register(
        FeatureSpec(
            name="points_std_std_8",
            description="volatility of own league-scored points",
            positions=_ALL_OFFENSE,
            window="std_8",
            source_table=SOURCE_TABLE,
            available_at_inference=True,
            lag_weeks=1,
        ),
        registry=registry,
    )

    result = weeks_in_current_role(result)
    register(
        FeatureSpec(
            name="weeks_in_current_role",
            description="weeks since snap_pct changed by more than 15 percentage points",
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
    "PASS_CATCHERS_AND_RB",
    "RB_QB",
    "SNAP_PCT_CHANGE_THRESHOLD",
    "SOURCE_TABLE",
    "add_actual_points",
    "build_usage_features",
    "ewm",
    "prior_season",
    "rolling_std",
    "season_to_date",
    "weeks_in_current_role",
]
