"""Strength of schedule and matchup views (SPEC.md §14.5; task 2.8).

Three real deliverables, all built on top of `defense_position_allowed`
(task 1.8's already walk-forward-safe ridge-adjusted rates) and
`schedule`:

1. **Positional SOS** (`positional_sos`) -- full-season / rest-of-season
   / fantasy-playoff sums of opponent-adjusted difficulty per team, per
   `week`-range helper (`full_season_weeks`/`rest_of_season_weeks`
   /`playoff_weeks`).
2. **Schedule grid** (`schedule_grid`) -- the same per-(team, week) data
   pivoted wide for a heatmap.
3. **Matchup detail** (`matchup_detail`) -- the per-`position_group`
   breakdown for one real player-week row.

All three share one underlying per-(team, week) table
(`team_position_group_schedule`) rather than three independent
computations, so "which teams played whom" is derived once.

**Sign convention, deliberately not inverted:** every `adj_epa_allowed`
value here means exactly what it means everywhere else in this codebase
(`features.opponent`, `models.points`'s own monotonic-increasing
constraint on the same columns) -- higher = the opponent allows more
EPA = an easier real matchup. `sos_value` is a plain sum of that same
quantity, so "higher `sos_value`" also means "easier schedule." Inverting
it into a "difficulty" score here would make this the one place in the
codebase where a bigger number means something worse.

**Confidence threshold is per-`position_group`, computed from real data
-- not SPEC's `k=250` reused directly.** That was the first real design
here, and it was wrong: caught live, not in a fixture. SPEC §10.4's
`k~250` is the ridge shrinkage's own blend weight input, `w = n_plays /
(n_plays + k)`, where that `n_plays` is a team's *cumulative trailing*
play count *within the current season so far* -- a different, much
larger quantity than the single-week `n_plays` this module (and
`defense_position_allowed.parquet`'s own column) actually reports (task
1.8's own entry in `docs/JOURNAL.md` names this distinction explicitly:
"a different, legitimate, unrelated quantity"). Verified against real
2025 data after building this: WR's own real single-week `n_plays` never
exceeds 37 (mean ~18); QB_rushing/RB_receiving/TE run in the *single
digits* per week. A flat 250-play threshold greys out every real cell,
at every position, every week -- confirmed live in a running Streamlit
session, not assumed. Position-group weekly volumes also differ from
each other by an order of magnitude, so no single fixed number (however
small) would work across all of them at once.

Fixed with `LOW_CONFIDENCE_PERCENTILE` (0.25): the confidence floor for
a given `(season, position_group)` is the bottom quartile of *that
group's own* real single-week `n_plays` distribution, computed fresh
from real data (`position_group_confidence_thresholds`) rather than one
constant borrowed from an unrelated part of the pipeline -- the same
"let the real distribution define what's thin" approach
`weekly_rankings_page.add_matchup_grade`'s own quintile buckets already
use elsewhere in this codebase.

**Real regular season only:** every week-range helper filters
`schedule.season_type == "REG"`. nflverse numbers real NFL postseason
games 19+ (`WC`/`DIV`/`CON`/`SB`), and both `player_week_features.parquet`
and `defense_position_allowed.parquet` carry real rows for those weeks
too (confirmed live: `defense_position_allowed`'s own max week is 22) --
a *fantasy* SOS/schedule grid should never include real NFL elimination
games no fantasy league plays through, so this module scopes to `REG`
explicitly rather than trusting an unfiltered `schedule.week` range the
way `evaluation.backtest._validation_weeks` (a different task's own
already-shipped code, out of scope to change here) currently does not.

**Playoff weeks stop at 17, literally.** SPEC §14.5: "fantasy playoffs
(weeks `playoff_week_start` through 17)" -- not through the season's own
real last regular-season week, which is 18 for 2021+ seasons. Followed
literally rather than generalised, since fantasy leagues conventionally
end by week 17 regardless of the NFL's own schedule length.

**Relocated-franchise join, same gotcha task 1.3 already fixed once.**
`schedule` keeps each game's period-accurate team code (the Rams as
`STL` in 2015); `defense_position_allowed` (built from `pbp`, task 1.8)
is already backfilled to the current franchise code (`LA`). Joining a
team's real opponent onto `defense_position_allowed` without aliasing
through `interim.build.RELOCATED_TEAM_ALIASES` first would silently drop
every relocated franchise's pre-move season (CLAUDE.md rule 4) -- the
exact bug `docs/JOURNAL.md` task 1.3's own entry documents finding and
fixing for `team_week_context`. Reused directly here, not re-derived.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import polars as pl

from ffapp.features.opponent import POSITION_TO_GROUPS, team_opponent
from ffapp.interim.build import RELOCATED_TEAM_ALIASES

LOW_CONFIDENCE_PERCENTILE = 0.25

_RATE_COLUMNS = [
    "adj_epa_allowed",
    "adj_success_allowed",
    "adj_ypt_allowed",
    "adj_td_rate_allowed",
]

_REG_SEASON_TYPE = "REG"
_LAST_FANTASY_PLAYOFF_WEEK = 17


def _real_weeks(schedule: pl.DataFrame, *, season: int) -> list[int]:
    """Real regular-season week numbers for `season` -- never hardcoded
    (CLAUDE.md rule 5), scoped to `season_type == "REG"` (see module
    docstring)."""
    return sorted(
        schedule.filter((pl.col("season") == season) & (pl.col("season_type") == _REG_SEASON_TYPE))[
            "week"
        ]
        .unique()
        .to_list()
    )


def full_season_weeks(schedule: pl.DataFrame, *, season: int) -> list[int]:
    """SPEC §14.5's "full season" SOS range."""
    return _real_weeks(schedule, season=season)


def rest_of_season_weeks(schedule: pl.DataFrame, *, season: int, as_of_week: int) -> list[int]:
    """SPEC §14.5's "rest of season" SOS range: real weeks strictly after
    the last one already played."""
    return [w for w in _real_weeks(schedule, season=season) if w > as_of_week]


def playoff_weeks(schedule: pl.DataFrame, *, season: int, playoff_week_start: int) -> list[int]:
    """SPEC §14.5's "fantasy playoffs" range, `playoff_week_start`
    through 17 literally (see module docstring). `playoff_week_start <=
    0` means the league config has no real playoff start configured
    (`LeagueFormat`'s own `or 0` fallback) -- an honestly empty range,
    not a guessed one."""
    if playoff_week_start <= 0:
        return []
    real_weeks = set(_real_weeks(schedule, season=season))
    return [w for w in range(playoff_week_start, _LAST_FANTASY_PLAYOFF_WEEK + 1) if w in real_weeks]


def _quantile_threshold(
    n_plays: pl.Series, *, percentile: float = LOW_CONFIDENCE_PERCENTILE
) -> float:
    values = n_plays.drop_nulls()
    if values.is_empty():
        return 0.0
    quantile = values.quantile(percentile, interpolation="linear")
    return float(quantile) if quantile is not None else 0.0


def position_group_confidence_thresholds(
    defense_position_allowed: pl.DataFrame, *, season: int
) -> dict[str, float]:
    """Every real `position_group`'s own confidence floor for `season` --
    the bottom quartile of that group's own real single-week `n_plays`
    values (see module docstring). Computed once per season, across every
    real position group at once, so a caller building the matchup-detail
    view (which only ever sees one player-week row at a time, with no
    access to the full distribution itself) can look up the right floor
    per group without recomputing it per row.
    """
    return {
        group: _quantile_threshold(
            defense_position_allowed.filter(
                (pl.col("season") == season) & (pl.col("position_group") == group)
            )["n_plays"]
        )
        for group in defense_position_allowed["position_group"].unique().to_list()
    }


def team_position_group_schedule(
    defense_position_allowed: pl.DataFrame,
    schedule: pl.DataFrame,
    *,
    season: int,
    position_group: str,
) -> pl.DataFrame:
    """One row per (team, week) with a real game that season: that team's
    real opponent and the opponent's real opponent-adjusted rates for
    `position_group` (task 1.8). A real bye has no row at all (`team_opponent`'s
    own documented behaviour) -- callers summing/pivoting over weeks must
    treat an absent week as "no game," never as zero.
    """
    opponents = team_opponent(schedule).filter(pl.col("season") == season)
    aliased = opponents.with_columns(
        pl.col("opponent").replace(RELOCATED_TEAM_ALIASES).alias("_defteam")
    )
    position_rows_all = defense_position_allowed.filter(
        (pl.col("season") == season) & (pl.col("position_group") == position_group)
    )
    threshold = _quantile_threshold(position_rows_all["n_plays"])
    position_rows = position_rows_all.select("defteam", "week", *_RATE_COLUMNS, "n_plays")

    joined = aliased.join(
        position_rows, left_on=["_defteam", "week"], right_on=["defteam", "week"], how="left"
    )
    return joined.with_columns(
        (pl.col("n_plays") >= threshold).fill_null(False).alias("confident")
    ).select("team", "week", "opponent", *_RATE_COLUMNS, "n_plays", "confident")


def positional_sos(team_schedule: pl.DataFrame, *, weeks: Sequence[int]) -> pl.DataFrame:
    """Sums `adj_epa_allowed` (SPEC's own headline rate outcome) across
    `weeks` per team -- SPEC §14.5's "sum the opponent-adjusted
    difficulty... across a week range," applied to whichever real range
    the caller already resolved (`full_season_weeks`/
    `rest_of_season_weeks`/`playoff_weeks`). A week absent for a team (a
    real bye) contributes nothing to the sum, never a guessed average --
    see module docstring for the sign convention (higher = easier).

    `confident` rolls up `team_schedule`'s own already-correctly-scaled
    per-week flags (a majority of the real included weeks must themselves
    be confident) rather than re-deriving a second, independent
    n_plays-based check here -- exactly the mismatched-scale mistake this
    module's own docstring documents fixing once already.
    """
    scoped = team_schedule.filter(pl.col("week").is_in(list(weeks)))
    return (
        scoped.group_by("team")
        .agg(
            pl.col("adj_epa_allowed").sum().alias("sos_value"),
            pl.col("week").n_unique().alias("n_weeks_included"),
            pl.col("n_plays").sum().alias("total_n_plays"),
            (pl.col("confident").cast(pl.Float64).mean() >= 0.5).alias("confident"),
        )
        .sort("sos_value", descending=True)
    )


def schedule_grid(team_schedule: pl.DataFrame) -> pl.DataFrame:
    """Wide teams x weeks pivot of `adj_epa_allowed` for a heatmap (SPEC
    §14.5's own "teams (rows) x weeks (columns)"). A missing (team, week)
    cell (a real bye) pivots to null -- the page renders that as
    blocked-out (SPEC's own "bye weeks rendered as blocked-out cells"),
    never zero, since zero is itself a real, different value ("this
    defense allows exactly average EPA at this position group")."""
    return team_schedule.pivot(on="week", index="team", values="adj_epa_allowed").sort("team")


def schedule_grid_confidence(team_schedule: pl.DataFrame) -> pl.DataFrame:
    """Wide teams x weeks pivot of `confident` -- the page's own greying
    mask (SPEC §14.5: "grey out grades with `n_plays` below a confidence
    threshold"), kept as a separate pivot from `schedule_grid`'s own cell
    value so "what to show" and "whether to trust it" stay independently
    testable, matching `weekly_rankings_page.add_matchup_grade`'s own
    value/sample-size split."""
    return team_schedule.pivot(on="week", index="team", values="confident").sort("team")


def matchup_detail(
    feature_row: Mapping[str, object],
    position: str,
    *,
    confidence_thresholds: Mapping[str, float] | None = None,
) -> list[dict[str, object]]:
    """Per-`position_group` breakdown for one real player-week row from
    `player_week_features.parquet` (task 1.9) -- SPEC §14.5's own
    "Matchup detail view": "opponent's adjusted rates... the sample size
    behind the estimate." A WR/TE gets one group, RB/QB get two
    (`features.opponent.POSITION_TO_GROUPS`) -- returned as a list of
    real components rather than `weekly_rankings_page.add_matchup_grade`'s
    single n_plays-weighted combined value, since the detail view's whole
    point (per SPEC's own "required honesty") is showing what a summary
    grade hides, not repeating it.

    `confidence_thresholds` is `position_group_confidence_thresholds`'s
    own output for the row's season -- a single player-week row has no
    access to the real distribution needed to compute its own threshold,
    so the caller resolves it once per season and passes it through. A
    group missing from `confidence_thresholds` (or no thresholds passed
    at all) falls back to `0.0` -- "trust any real recorded sample," the
    same permissive default a genuinely thin group would get from its own
    real bottom-quartile threshold anyway when every real week is thin.
    """
    thresholds = confidence_thresholds or {}
    groups = POSITION_TO_GROUPS.get(position, [])
    detail: list[dict[str, object]] = []
    for group in groups:
        suffix = group.lower()
        n_plays_raw = feature_row.get(f"def_n_plays_{suffix}")
        n_plays = int(n_plays_raw) if n_plays_raw is not None else None  # type: ignore[call-overload]
        threshold = thresholds.get(group, 0.0)
        detail.append(
            {
                "position_group": group,
                "adj_epa_allowed": feature_row.get(f"def_adj_epa_allowed_{suffix}"),
                "adj_success_allowed": feature_row.get(f"def_adj_success_allowed_{suffix}"),
                "adj_ypt_allowed": feature_row.get(f"def_adj_ypt_allowed_{suffix}"),
                "adj_td_rate_allowed": feature_row.get(f"def_adj_td_rate_allowed_{suffix}"),
                "n_plays": n_plays,
                "confident": n_plays is not None and n_plays >= threshold,
            }
        )
    return detail


__all__ = [
    "LOW_CONFIDENCE_PERCENTILE",
    "full_season_weeks",
    "matchup_detail",
    "playoff_weeks",
    "position_group_confidence_thresholds",
    "positional_sos",
    "rest_of_season_weeks",
    "schedule_grid",
    "schedule_grid_confidence",
    "team_position_group_schedule",
]
