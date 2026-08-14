"""Streaming-aware replacement level for DST/K (not a numbered TASKS.md task --
direct request 2026-08-14, following up on task 0.9's VOR).

`tools.vor`'s replacement level is "the Nth-best preseason-projected season
total" -- a reasonable proxy for RB/WR/TE, where an injured starter's real
backup is genuinely worse. It is the wrong proxy for DST and K: the project
owner doesn't draft either for season-long value, they stream the best
matchup available off waivers every week (confirmed live against this
league's own real 2021-2025 scoring: a simple "start whichever available
team had the best matchup that week" strategy outscored even the single
best *drafted* DST/K, in real regular-season points, every one of the last
five seasons -- see `docs/JOURNAL.md`'s 2026-08-14 entry for the full
numbers). SPEC §9.4 itself expects DST/K VOR to come out "almost always
tiny" -- that only holds if replacement level reflects what's actually
achievable by streaming, which the standard fixed-point baseline (built for
non-streamable positions) does not capture.

This computes that replacement level empirically, from this league's own
real historical scoring (reusing `scoring.stats.build_stat_frame`/
`scoring.engine.score_stat_line` unmodified -- no new scoring logic), rather
than guessing a constant. Two documented judgment calls, not asked to be
tuned precisely (the project owner doesn't care about DST/K's exact rank,
only that neither shows up early):

- **Per season, the `n_drafted` teams with the best real season total are
  treated as "already drafted"** -- a hindsight proxy (good defenses/kickers
  tend to get drafted, whether or not by this specific league), applied
  before the weekly simulation so a team already accounted for isn't also
  counted as a streaming option.
- **`availability_rank=3`, not the single best-available score each week**
  -- a light, deliberately conservative haircut for imperfect real-world
  streaming (bye weeks, FAAB timing, not always winning the waiver claim
  against the other real managers also chasing matchups), not the
  best-case upper bound.

Scoped to `season_type == "REG"` throughout, same reasoning as
`tools.sos`: nflverse's raw team/player stat tables include real NFL
postseason games no fantasy league plays through.
"""

from __future__ import annotations

import statistics

import polars as pl

from ffapp.scoring.engine import score_stat_line
from ffapp.scoring.stats import build_stat_frame

DEFAULT_SEASONS = list(range(2021, 2026))
DEFAULT_AVAILABILITY_RANK = 3
STREAMING_POSITIONS = ("DST", "K")
_REG_SEASON_TYPE = "REG"


def score_historical_stats(
    player_stats: pl.DataFrame,
    team_stats: pl.DataFrame,
    schedules: pl.DataFrame,
    pbp: pl.DataFrame,
    scoring_settings: dict[str, float],
) -> pl.DataFrame:
    """This league's own real historical points, every position, via the
    same `build_stat_frame`/`score_stat_line` pipeline the golden test and
    the DST model's own training already use -- no new scoring logic.
    `player_stats`/`team_stats` are filtered to `season_type == "REG"`
    before joining; `schedules`/`pbp` need no filter of their own, since
    postseason rows in either simply never find a matching (season, week,
    team) row in the now-REG-only base frame.
    """
    reg_player_stats = player_stats.filter(pl.col("season_type") == _REG_SEASON_TYPE)
    reg_team_stats = team_stats.filter(pl.col("season_type") == _REG_SEASON_TYPE)
    combined = build_stat_frame(reg_player_stats, reg_team_stats, schedules, pbp)
    return combined.with_columns(score_stat_line(combined, scoring_settings).alias("points"))


def _season_streaming_total(
    season_rows: pl.DataFrame, *, drafted_ids: set[str], availability_rank: int
) -> float:
    total = 0.0
    for week in sorted(season_rows["week"].unique().to_list()):
        available = season_rows.filter(
            (pl.col("week") == week) & (~pl.col("player_id").is_in(list(drafted_ids)))
        )
        scores = sorted(available["points"].to_list(), reverse=True)
        if len(scores) >= availability_rank:
            total += scores[availability_rank - 1]
    return total


def streaming_replacement_points(
    scored_stats: pl.DataFrame,
    *,
    position: str,
    n_drafted: int,
    seasons: list[int] = DEFAULT_SEASONS,
    availability_rank: int = DEFAULT_AVAILABILITY_RANK,
) -> float:
    """The real, empirical replacement level for a streamable position:
    the average, across `seasons`, of a season spent always starting the
    `availability_rank`-th best-scoring team not among that season's
    `n_drafted` best (by real season total). `scored_stats` is
    `score_historical_stats`'s own output (needs `position`, `season`,
    `week`, `player_id`, `points`).
    """
    per_season_totals: list[float] = []
    for season in seasons:
        season_rows = scored_stats.filter(
            (pl.col("position") == position) & (pl.col("season") == season)
        )
        season_totals = (
            season_rows.group_by("player_id")
            .agg(pl.col("points").sum().alias("season_total"))
            .sort("season_total", descending=True)
        )
        drafted_ids = set(season_totals["player_id"][:n_drafted].to_list())
        per_season_totals.append(
            _season_streaming_total(
                season_rows, drafted_ids=drafted_ids, availability_rank=availability_rank
            )
        )
    return statistics.mean(per_season_totals)


def streaming_replacement_overrides(
    scored_stats: pl.DataFrame,
    *,
    n_drafted_by_position: dict[str, int],
    seasons: list[int] = DEFAULT_SEASONS,
    availability_rank: int = DEFAULT_AVAILABILITY_RANK,
) -> dict[str, float]:
    """`{position: replacement_points}` for every position in
    `n_drafted_by_position` -- built for `STREAMING_POSITIONS` (DST/K), but
    driven entirely by its input rather than hardcoding that pair, so a
    caller scoping to just one (or a league without a standalone K/DST
    slot at all) doesn't need a second code path."""
    return {
        position: streaming_replacement_points(
            scored_stats,
            position=position,
            n_drafted=n_drafted,
            seasons=seasons,
            availability_rank=availability_rank,
        )
        for position, n_drafted in n_drafted_by_position.items()
    }


__all__ = [
    "DEFAULT_AVAILABILITY_RANK",
    "DEFAULT_SEASONS",
    "STREAMING_POSITIONS",
    "score_historical_stats",
    "streaming_replacement_overrides",
    "streaming_replacement_points",
]
