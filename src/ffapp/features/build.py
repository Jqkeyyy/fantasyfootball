"""Build-time as_of assertions and wide-table assembly (SPEC.md §10.1,
§6.2; tasks 1.5 and 1.9).

Two unrelated jobs share this module, matching SPEC's own repo-layout
comment (`build.py # assembles the wide table`):

- The two build-time assertions SPEC §10.1 names explicitly (task 1.5):
  every feature in a training matrix must carry at least one week of lag,
  and every feature an inference model actually uses must be marked
  usable at live inference time. Both raise immediately on the first
  violation rather than collecting every one -- this is a leakage gate,
  not a lint report; a single mis-specified feature is enough to
  invalidate a training run (CLAUDE.md rule 2).
- `build_player_week_features` (task 1.9): assembles
  `features/player_week_features.parquet`, the wide modelling table that
  every downstream task in Phase 1 reads from.

Task 1.9's central mechanic is *lag-shifting*. `features.usage`'s own
output rows, and *most* of `features.team_context`'s, are
*through-and-including* their own week -- both modules' docstrings say
explicitly that this module is where the one-week shift onto a *target*
week actually happens. `features.opponent`'s rows are different in kind:
task 1.8's own walk-forward design means a `defense_position_allowed`
row for (defteam, season, week) already only used that defense's *prior*
weeks, so it's already "as of" the target week and needs no further
shift -- joined directly. `features.situation`'s rows describe the
target week's own game (is it at home, what's the weather, what does the
injury report say as of the Friday before kickoff) -- also joined
directly, for the same reason: these are legitimately known *before*
that week's kickoff, not data leaking from the outcome itself.
`features.depth_chart` (task 1.14) is the same case again: a team's real
depth chart is public before kickoff.

`features.team_context.CURRENT_WEEK_COLUMNS` (Vegas line, teammate
vacated shares) are the same "current week" case as `opponent`/
`situation`, despite sharing `team_context.SOURCE_TABLE` with the
windowed columns that do need the shift -- a real bug found by this
task's own end-to-end verification: a blanket shift-everything-from-
team_context rule staples last week's injury news and Vegas line onto
this week's row, silently one week stale for every single row in the
real table. `build_player_week_features` splits the two explicitly
rather than shifting by `source_table` alone.
"""

from __future__ import annotations

from collections.abc import Iterable

import polars as pl

from ffapp.features import depth_chart, opponent, player_bio, situation, team_context, usage
from ffapp.features.registry import FEATURE_REGISTRY, FeatureSpec
from ffapp.interim.build import SKILL_POSITIONS
from ffapp.scoring.engine import score_stat_line

ACTIVE_ROSTER_STATUSES = ("ACT", "INA")


class LeakageError(Exception):
    """A feature's `lag_weeks` or `available_at_inference` violates the as_of contract."""


def assert_training_lag(specs: Iterable[FeatureSpec]) -> None:
    """SPEC §10.1: "every feature in a training matrix has `lag_weeks >= 1`
    relative to the target week." `lag_weeks` is the feature's own
    declared minimum lag (SPEC's own field description: "1 means uses
    data through week W-1") -- `lag_weeks < 1` means the feature can see
    the target week's own data, the textbook leakage case.
    """
    for spec in specs:
        if spec.lag_weeks < 1:
            raise LeakageError(
                f"Feature {spec.name!r} has lag_weeks={spec.lag_weeks}, but every "
                "feature in a training matrix must have lag_weeks >= 1 (SPEC §10.1) "
                "-- it would see data from the target week itself."
            )


def assert_inference_availability(specs: Iterable[FeatureSpec]) -> None:
    """SPEC §10.1: "every feature used by an inference model has
    `available_at_inference=True`." A training-only feature (e.g. route
    participation, in-season -- SPEC §10.5) reaching a live inference
    model is the specific failure mode this guards against: the model
    would have learned to rely on a signal that silently isn't there when
    it's actually asked to predict.
    """
    for spec in specs:
        if not spec.available_at_inference:
            raise LeakageError(
                f"Feature {spec.name!r} has available_at_inference=False, but it "
                "is used by an inference model (SPEC §10.1) -- it may be training-only "
                "(e.g. route participation in-season, SPEC §10.5)."
            )


def _team_scheduled_weeks(schedule: pl.DataFrame) -> pl.DataFrame:
    """One row per (team, season, week) a team has a real scheduled game
    -- a bye week has no row, so anything joined against this naturally
    drops byes rather than needing an explicit filter."""
    home = schedule.select("season", "week", pl.col("home_team").alias("team"))
    away = schedule.select("season", "week", pl.col("away_team").alias("team"))
    return pl.concat([home, away], how="vertical_relaxed").unique()


def _row_grid(rosters: pl.DataFrame, schedule: pl.DataFrame) -> pl.DataFrame:
    """SPEC §11.1's row universe: every player on an active roster that
    week, scoped to `SKILL_POSITIONS` and to weeks the player's own team
    actually has a scheduled game (a bye week has no fantasy relevance at
    all, not even a `target=0` row).

    `status` must be `"ACT"` or `"INA"`, not `"ACT"` alone -- a real,
    load-bearing bug found and fixed by end-to-end verification against
    real data: confirmed live, a player ruled `Out` for a game (e.g. NYG's
    real 2024 rookie WR1 Malik Nabers, hamstring, real weeks 5-6) has real
    roster `status="INA"` those specific weeks, not `"ACT"` -- `"ACT"`
    alone silently dropped him from the row universe entirely for exactly
    the weeks SPEC §11.1 most wants included ("every player on an active
    roster that week, including those who did not play... excluding them
    creates survivorship bias"). `"INA"` means "on the 53-man active
    roster, inactive for this game" -- still the *active* roster, just not
    dressed. `ACTIVE_ROSTER_STATUSES` deliberately excludes `"DEV"`
    (practice squad -- a distinct, smaller roster, not the 53-man active
    one) and `"RES"` (injured reserve -- genuinely off the active roster,
    not merely inactive for one game).

    `rosters` must be the real *weekly* roster snapshot
    (`ingest.nflverse.fetch_rosters`, which itself was a real bug fixed in
    this same task -- see that function's own docstring: it originally
    called `load_rosters` (season-level, ~3k rows/season) instead of
    `load_rosters_weekly` (~46k rows/season for 2024 alone), nowhere near
    the weekly granularity this row universe needs).
    """
    active = rosters.filter(
        pl.col("status").is_in(ACTIVE_ROSTER_STATUSES)
        & pl.col("position").is_in(SKILL_POSITIONS)
        & pl.col("gsis_id").is_not_null()
    ).select(pl.col("gsis_id").alias("player_id"), "season", "week", "team", "position")
    return active.join(_team_scheduled_weeks(schedule), on=["team", "season", "week"], how="inner")


def _add_target_and_availability(
    grid: pl.DataFrame,
    player_week_stats: pl.DataFrame,
    player_week_usage: pl.DataFrame,
    scoring_settings: dict[str, float],
) -> pl.DataFrame:
    """`target` (SPEC §11.1: league-scored actual points for every row,
    0 for a player who didn't play -- not a dropped row; excluding them
    is the survivorship-bias failure SPEC calls out explicitly).
    `availability_flag` = 1 if the player recorded at least one offensive
    snap (SPEC §11.2's own availability-model target definition), 0
    otherwise -- including players entirely absent from
    `player_week_usage` (never played, no row to look up)."""
    points = player_week_stats.with_columns(
        score_stat_line(player_week_stats, scoring_settings).alias("target")
    ).select("player_id", "season", "week", "target")
    snaps = player_week_usage.select(
        "player_id",
        "season",
        "week",
        (pl.col("offense_snaps") > 0).alias("availability_flag"),
    )
    return (
        grid.join(points, on=["player_id", "season", "week"], how="left")
        .join(snaps, on=["player_id", "season", "week"], how="left")
        .with_columns(
            pl.col("target").fill_null(0.0),
            pl.col("availability_flag").fill_null(False),
        )
    )


def _add_as_of_utc(grid: pl.DataFrame, schedule: pl.DataFrame) -> pl.DataFrame:
    """SPEC §6.2: `player_week_features.parquet` "carries `as_of_utc`."
    SPEC §12.1 defines it precisely as "the kickoff time of the first
    game of that week" -- the safety-margin/snapshot mechanics themselves
    are task 1.11's own `evaluation.snapshot`, not this table's job; this
    only stamps the raw boundary timestamp every row needs."""
    as_of = schedule.group_by(["season", "week"]).agg(
        pl.col("kickoff_utc").min().alias("as_of_utc")
    )
    return grid.join(as_of, on=["season", "week"], how="left")


def _registered_feature_columns(registry: dict[str, FeatureSpec], source_table: str) -> list[str]:
    return [name for name, spec in registry.items() if spec.source_table == source_table]


def _lag_shift_join(
    grid: pl.DataFrame,
    feature_table: pl.DataFrame,
    group_key: str,
    feature_columns: list[str],
    *,
    lag_weeks: int = 1,
) -> pl.DataFrame:
    """Joins `feature_table` onto `grid` so a grid row at week W receives
    `feature_table`'s row at week `W - lag_weeks` (same `group_key`,
    same season) -- the trailing-through-week value becomes the *next*
    week's feature. This is what `usage`/`team_context`'s own
    `lag_weeks=1` declaration actually means in practice; see module
    docstring for why `opponent`/`situation` skip this and join directly.
    """
    shifted = feature_table.select(group_key, "season", "week", *feature_columns).with_columns(
        (pl.col("week") + lag_weeks).alias("week")
    )
    return grid.join(shifted, on=[group_key, "season", "week"], how="left")


def build_player_week_features(
    rosters: pl.DataFrame,
    schedule: pl.DataFrame,
    player_week_stats: pl.DataFrame,
    player_week_usage: pl.DataFrame,
    snap_counts: pl.DataFrame,
    team_week_context: pl.DataFrame,
    defense_position_allowed: pl.DataFrame,
    injuries: pl.DataFrame,
    weather: pl.DataFrame,
    depth_charts: pl.DataFrame,
    scoring_settings: dict[str, float],
    *,
    registry: dict[str, FeatureSpec] | None = None,
) -> pl.DataFrame:
    """Task 1.9: assembles `features/player_week_features.parquet`
    (SPEC §6.2) -- the wide modelling table. See module docstring for the
    lag-shift design that's the actual substance of this function; the
    rest is orchestration over tasks 1.6-1.8's own feature blocks plus
    the Situation block (no dedicated earlier task; see `features.situation`'s
    own docstring for why it lives there).
    """
    effective_registry = registry if registry is not None else FEATURE_REGISTRY

    grid = _row_grid(rosters, schedule)
    grid = _add_target_and_availability(
        grid, player_week_stats, player_week_usage, scoring_settings
    )
    grid = _add_as_of_utc(grid, schedule)

    grid = situation.build_situation_features(
        grid, schedule, injuries, weather, registry=effective_registry
    )
    grid = opponent.build_opponent_features(
        grid, schedule, defense_position_allowed, registry=effective_registry
    )
    grid = depth_chart.build_depth_chart_features(grid, depth_charts, registry=effective_registry)
    grid = player_bio.build_player_bio_features(grid, rosters, registry=effective_registry)

    usage_features = usage.build_usage_features(
        player_week_usage, player_week_stats, scoring_settings, registry=effective_registry
    )
    usage_cols = _registered_feature_columns(effective_registry, usage.SOURCE_TABLE)
    grid = _lag_shift_join(grid, usage_features, "player_id", usage_cols)

    team_context_features = team_context.build_team_context_features(
        team_week_context, snap_counts, injuries, usage_features, registry=effective_registry
    )
    team_context_cols = _registered_feature_columns(effective_registry, team_context.SOURCE_TABLE)
    # team_context.CURRENT_WEEK_COLUMNS (Vegas line, vacated shares) are
    # already current-week facts -- joined directly, like
    # opponent/situation, not shifted like the windowed columns.
    trailing_cols = [c for c in team_context_cols if c not in team_context.CURRENT_WEEK_COLUMNS]
    current_cols = [c for c in team_context_cols if c in team_context.CURRENT_WEEK_COLUMNS]
    grid = _lag_shift_join(grid, team_context_features, "team", trailing_cols)
    grid = grid.join(
        team_context_features.select("team", "season", "week", *current_cols),
        on=["team", "season", "week"],
        how="left",
    )

    return grid


__all__ = [
    "ACTIVE_ROSTER_STATUSES",
    "LeakageError",
    "assert_inference_availability",
    "assert_training_lag",
    "build_player_week_features",
]
