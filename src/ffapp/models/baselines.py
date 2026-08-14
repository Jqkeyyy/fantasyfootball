"""Baselines B0-B3 (SPEC.md §12.3; task 1.10).

Every evaluation report compares the real model against all four --
SPEC's own words: "these are the yardstick; a buggy baseline flatters
the model." All four are walk-forward by construction: a baseline's
prediction for week W uses only data strictly before week W, matching
CLAUDE.md rule 1 and this project's established "no random splits, only
trailing history" convention.

B0-B2 are computed directly from `features/player_week_features.parquet`'s
own `target` column (task 1.9) -- no new data. B3 needs a real external
source (`ingest.rankings`'s FantasyPros weekly-archive functions,
this same task).
"""

from __future__ import annotations

import polars as pl

from ffapp.ids import mapping
from ffapp.projections.aggregate import add_join_key


def _sort(df: pl.DataFrame) -> pl.DataFrame:
    return df.sort(["player_id", "season", "week"])


def add_b1_season_to_date_mean(player_week_features: pl.DataFrame) -> pl.DataFrame:
    """B1: "player's season-to-date mean" -- SPEC calls this "naive but
    surprisingly hard." The player's own cumulative mean of `target`,
    trailing strictly through week W-1 (never week W's own outcome) --
    `.cum_sum()`/`.cum_count()` then `.shift(1)` so the target week's own
    contribution is excluded, not merely un-double-counted. Null for a
    player's first tracked week of a season, resetting each season for
    consistency with every other trailing window in this project
    (`features.usage`'s own module docstring explains the same
    within-season convention)."""
    sorted_df = _sort(player_week_features)
    prior_sum = pl.col("target").cum_sum().shift(1).over(["player_id", "season"])
    prior_count = pl.col("target").cum_count().shift(1).over(["player_id", "season"])
    return sorted_df.with_columns((prior_sum / prior_count).alias("b1_season_to_date_mean"))


def add_b2_ewm_4(player_week_features: pl.DataFrame) -> pl.DataFrame:
    """B2: "player's own ewm_4 of own points" -- SPEC calls this "the real
    bar for did my features do anything." Same trailing-through-W-1
    exclusion as B1 (`.ewm_mean(span=4)` then `.shift(1)`), so a huge or
    tiny week-W outcome never leaks into week W's own prediction."""
    sorted_df = _sort(player_week_features)
    return sorted_df.with_columns(
        pl.col("target").ewm_mean(span=4).shift(1).over(["player_id", "season"]).alias("b2_ewm_4")
    )


def pooled_rolling_mean(
    df: pl.DataFrame, group_column: str, target_column: str, output_column: str
) -> pl.DataFrame:
    """Shared machinery behind B0, the availability base-rate baseline
    (task 1.14), and model v2 Stage 1's own league-mean baseline: pooled
    (across every row sharing the same `group_column` value),
    season-to-date-through-week-W-1 mean of `target_column`. A season's
    own week 1 has no current-season trailing data, so it falls back to
    the *entire* prior season's own mean (same fallback convention as
    `features.usage.prior_season`); the very first tracked season's own
    week 1 has neither and stays honestly null, same precedent as task
    1.7's `proe` and task 1.8's opponent adjustment for the identical
    reason.
    """
    by_week = (
        df.group_by([group_column, "season", "week"])
        .agg(
            pl.col(target_column).cast(pl.Float64).sum().alias("_week_sum"),
            pl.len().alias("_week_n"),
        )
        .sort([group_column, "season", "week"])
    )
    prior_sum = pl.col("_week_sum").cum_sum().shift(1).over([group_column, "season"])
    prior_n = pl.col("_week_n").cum_sum().shift(1).over([group_column, "season"])
    with_current_season_mean = by_week.with_columns(
        (prior_sum / prior_n).alias("_current_season_mean")
    )

    prior_season_mean = (
        df.group_by([group_column, "season"])
        .agg(pl.col(target_column).cast(pl.Float64).mean().alias("_prior_season_mean"))
        .with_columns((pl.col("season") + 1).alias("season"))
    )

    combined = (
        with_current_season_mean.join(prior_season_mean, on=[group_column, "season"], how="left")
        .with_columns(
            pl.coalesce(["_current_season_mean", "_prior_season_mean"]).alias(output_column)
        )
        .select(group_column, "season", "week", output_column)
    )

    return df.join(combined, on=[group_column, "season", "week"], how="left")


def add_b0_positional_mean(player_week_features: pl.DataFrame) -> pl.DataFrame:
    """B0: "positional weekly mean" -- SPEC calls this the "sanity floor."
    Pooled across *every* player at a position (not per-player, the
    distinguishing feature from B1/B2) -- see `pooled_rolling_mean` for
    the shared season-to-date/prior-season-fallback mechanics.
    """
    return pooled_rolling_mean(player_week_features, "position", "target", "b0_positional_mean")


def add_availability_base_rate(player_week_features: pl.DataFrame) -> pl.DataFrame:
    """The availability model's own comparison baseline (SPEC §11.2/task
    1.14's own acceptance bar: "Brier score beats a positional base-rate
    predictor") -- exactly B0's shape (`pooled_rolling_mean`),
    applied to `availability_flag` instead of `target`: "what fraction of
    this position's players were active, on average, through last week."
    """
    return pooled_rolling_mean(
        player_week_features, "position", "availability_flag", "availability_base_rate"
    )


def add_b3_fp_weekly_consensus(fp_weekly: pl.DataFrame, players_dim: pl.DataFrame) -> pl.DataFrame:
    """B3: "public consensus weekly projections" -- SPEC calls this "the
    bar for should I use my model at all." Resolves
    `ingest.rankings.normalize_fp_weekly`'s own `(player_name, pos, team,
    b3_points, season, week)` rows to this project's canonical
    `player_id`, via the same normalized `(name, position)` `join_key`
    already used for cross-source player matching elsewhere in this
    project (`projections.aggregate.add_join_key` for the FantasyPros
    side, `ids.mapping.dedupe_to_one_row_per_name_position` for the
    `players_dim` side -- the identical pattern
    `projections.games_played.player_ages_from_players_dim` and
    `draft.board._team_by_join_key` already use for two other sources,
    applied here to a third).

    A FantasyPros row that doesn't resolve to a real `player_id` is
    dropped, not failed loudly -- unlike CLAUDE.md rule 4's join-drop
    rule (which protects the real training/target pipeline), B3 is an
    independent, best-effort comparison table joined onto real player-
    weeks later; an unresolved name just means that one row of B3 stays
    absent (visibly null downstream, same honest-gap precedent as B0's
    own first-season null), not a real player silently vanishing from
    the model's own training data.
    """
    with_key = add_join_key(fp_weekly.rename({"pos": "position"}))
    resolved = mapping.dedupe_to_one_row_per_name_position(players_dim).select(
        "join_key", "player_id"
    )
    return (
        with_key.join(resolved, on="join_key", how="left")
        .drop_nulls("player_id")
        .select("player_id", "season", "week", "b3_points")
    )


__all__ = [
    "add_availability_base_rate",
    "add_b0_positional_mean",
    "add_b1_season_to_date_mean",
    "add_b2_ewm_4",
    "add_b3_fp_weekly_consensus",
    "pooled_rolling_mean",
]
