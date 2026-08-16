"""Future-week shape allocation (`SPEC-ADDENDUM-04.md` §D.1's amendment,
`docs/JOURNAL.md`'s 2026-08-16 entry): "consensus supplies the level;
the pipeline supplies the shape." `models.ros_consensus` resolves a real
season-long remaining-value LEVEL per player; this module allocates that
level across the player's own real remaining weeks (byes excluded
entirely), shaped by each real future opponent's own already-validated
opponent-adjusted rating (`defense_position_allowed`'s `adj_epa_allowed`,
task 1.8), FROZEN at whatever is actually known as of the anchor week --
never a future week's own rating, which doesn't exist in the real table
regardless (task 1.8's own table only ever has rows for real played
weeks). Deliberately does not touch availability/injury at all -- that
is the aggregation stage's own separate `p_play[w]` multiplier
(`tools.ros_aggregate`); baking it in here too would double-count.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from ffapp.features.opponent import POSITION_TO_GROUPS, team_opponent
from ffapp.interim.build import RELOCATED_TEAM_ALIASES

_MATCHUP_WEIGHT_SCALE = 0.15
_MATCHUP_WEIGHT_CLIP = 0.3


def frozen_defense_ratings(
    defense_position_allowed: pl.DataFrame, *, season: int, as_of_week: int, position_group: str
) -> pl.DataFrame:
    """Each real team's own latest known `adj_epa_allowed` for
    `position_group`, frozen at `as_of_week` -- the most recent real row
    at or before `as_of_week`. Made an explicit filter rather than relying
    only on "no future row exists yet" (true today, but this makes the
    freeze a real, testable contract rather than an accident of data
    availability)."""
    scoped = defense_position_allowed.filter(
        (pl.col("season") == season)
        & (pl.col("position_group") == position_group)
        & (pl.col("week") <= as_of_week)
    )
    return (
        scoped.sort(["defteam", "week"])
        .group_by("defteam", maintain_order=True)
        .agg(pl.all().last())
        .select(
            "defteam",
            pl.col("adj_epa_allowed").alias("frozen_adj_epa_allowed"),
            pl.col("n_plays").alias("frozen_n_plays"),
        )
    )


def future_week_opponents(
    schedule: pl.DataFrame, *, season: int, team: str, weeks: list[int]
) -> pl.DataFrame:
    """`week, opponent_team` for every real week in `weeks` that `team`
    actually has a game -- a real bye contributes no row at all (SPEC-
    ADDENDUM-04.md §D.2: "bye weeks contribute zero"), consistent with
    `tools.sos.team_position_group_schedule`'s own same real convention.
    """
    opponents = team_opponent(schedule).filter(
        (pl.col("season") == season) & (pl.col("team") == team) & pl.col("week").is_in(weeks)
    )
    return opponents.select("week", pl.col("opponent").alias("opponent_team")).sort("week")


def _matchup_weight(frozen_value: float | None, mean: float, std: float) -> float:
    if frozen_value is None or std <= 0:
        return 1.0
    z = (frozen_value - mean) / std
    clipped = np.clip(z * _MATCHUP_WEIGHT_SCALE, -_MATCHUP_WEIGHT_CLIP, _MATCHUP_WEIGHT_CLIP)
    return 1.0 + float(clipped)


def allocate_season_consensus(
    season_consensus_ros_points: float,
    position: str,
    team: str,
    weeks_with_opponents: pl.DataFrame,
    frozen_ratings_by_group: dict[str, pl.DataFrame],
) -> pl.DataFrame:
    """Allocates `season_consensus_ros_points` across `weeks_with_opponents`'s
    real rows (`week, opponent_team`) proportional to a bounded matchup
    weight (`_matchup_weight`, a documented judgment-call scale/clip pair
    -- same status as `config.WaiverSettings.aggressiveness`, no SPEC-given
    value exists to fit against). A player's own relevant real position
    groups (`features.opponent.POSITION_TO_GROUPS` -- QB/RB get two,
    WR/TE get one) are averaged when more than one applies. Weights are
    normalized so `Σ mean == season_consensus_ros_points` exactly (up to
    floating point) -- the real level is always preserved by construction,
    only its real week-to-week shape varies.
    """
    groups = POSITION_TO_GROUPS.get(position, [])
    if not groups or weeks_with_opponents.is_empty():
        n = weeks_with_opponents.height
        if n == 0:
            return pl.DataFrame(schema={"week": pl.Int64, "mean": pl.Float64})
        even_share = season_consensus_ros_points / n
        return weeks_with_opponents.select("week").with_columns(pl.lit(even_share).alias("mean"))

    aliased = weeks_with_opponents.with_columns(
        pl.col("opponent_team").replace(RELOCATED_TEAM_ALIASES).alias("_defteam")
    )
    group_weights: list[pl.Series] = []
    for group in groups:
        ratings = frozen_ratings_by_group.get(group)
        if ratings is None or ratings.is_empty():
            group_weights.append(pl.Series([1.0] * aliased.height))
            continue
        values = ratings["frozen_adj_epa_allowed"].to_numpy()
        mean, std = float(np.mean(values)), float(np.std(values))
        joined = aliased.join(ratings, left_on="_defteam", right_on="defteam", how="left")
        weights = [
            _matchup_weight(v, mean, std) for v in joined["frozen_adj_epa_allowed"].to_list()
        ]
        group_weights.append(pl.Series(weights))

    combined_weight = group_weights[0]
    for extra in group_weights[1:]:
        combined_weight = (combined_weight + extra) / 2.0
    total_weight = float(combined_weight.sum())
    shares = (combined_weight / total_weight).to_numpy() if total_weight > 0 else None
    if shares is None:
        n = aliased.height
        shares = np.full(n, 1.0 / n)

    return aliased.select("week").with_columns(
        pl.Series("mean", shares * season_consensus_ros_points)
    )


__all__ = [
    "allocate_season_consensus",
    "frozen_defense_ratings",
    "future_week_opponents",
]
