"""Sleeper scoring key -> stat mapping (SPEC.md §8.1-8.3).

Each Sleeper `scoring_settings` key maps to either:
    - `DirectStat`: one stat column, multiplied by the key's point value.
    - `DerivedStat`: a small rule (bonus thresholds, FG per-yard/legacy-bucket
      scoring, DST points-allowed buckets) that computes this key's own point
      contribution.

Column-naming convention for the per-player-week stat frame `score_stat_line`
expects. Most names are nflreadpy's own `load_player_stats`/`load_team_stats`
column names, confirmed by live inspection rather than assumed (see HANDOFF.md
§4) -- `passing_yards`, `def_sacks`, `fg_made_list`, etc. A handful of columns are
this project's own invention, computed at normalisation time because nflreadpy
doesn't provide them directly:

    - `points_allowed`: the DST's opponent's final score for the week (joined from
      `load_schedules`), used to pick the matching `pts_allow_*` bucket.
    - `opponent_blocked_kicks`: credit for the defense that forced a block, derived
      by joining each team-week to its opponent's own `fg_blocked` + `pat_blocked`
      that game (nflreadpy records a blocked kick on the *kicking* team's row, not
      the blocking defense's).
    - `def_return_tds`: genuine defensive/return touchdowns, derived from
      play-by-play (`return_touchdown == 1` and `td_team == defteam`). team_stats'
      own `def_tds` and `fumble_recovery_tds` columns are both unreliable for this
      -- confirmed live, `fumble_recovery_tds` conflates a genuine defensive score
      with an offensive player recovering their *own* team's fumble and scoring
      (real case: HOU week 15 2025), which is not a defensive event.
    - `def_fumbles_forced` / `fumble_recovery_opp` / `special_teams_forced_fumbles`
      / `special_teams_fumble_recoveries`: nflreadpy's own `def_fumbles_forced` /
      `fumble_recovery_opp` team_stats columns lump forced fumbles and recoveries
      from special-teams plays (punts, kickoffs) in with ordinary scrimmage-play
      defense -- but Sleeper scores those as two separate key pairs
      (`ff`/`fum_rec` vs. `def_st_ff`/`def_st_fum_rec`). Confirmed live: PIT's real
      week-1-2025 kickoff-return fumble was already being counted under the
      general team_stats columns, so adding separate special-teams credit on top
      would have double-counted it. All four columns here are instead derived from
      play-by-play's structured fumble fields (`forced_fumble_player_1_team`,
      `fumbled_1_team`, `fumble_recovery_1_team`, `special_teams_play`) -- see
      `scoring/stats.py`'s `_forced_fumbles`/`_opponent_fumble_recoveries`. A
      recovery only counts as a turnover when it's the *opponent's* fumble; a team
      recovering its own loose ball isn't a defensive event.

`load_player_stats` gives one row per individual player-week; `load_team_stats`
gives one row per team-week. Both feed the same per-player-week stat frame here --
DST is modelled as its own row (Sleeper drafts it as a roster entity), sourced from
`load_team_stats` rather than aggregated from individual defenders by hand.
`fg_made_list` is a semicolon-delimited string of kick distances ("25;43;32"), null
when no field goals were made -- not a native list column.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import polars as pl


@dataclass(frozen=True)
class DirectStat:
    """A Sleeper scoring key that multiplies one stat column by the key's point value."""

    column: str


@dataclass(frozen=True)
class DerivedStat:
    """A Sleeper scoring key computed by a rule instead of a single column multiply.

    `compute(stats, value)` returns this key's own per-row point contribution. It
    must not reference any other scoring key's value -- `score_stat_line` calls one
    `DerivedStat` per active key and sums the results.
    """

    compute: Callable[[pl.DataFrame, float], pl.Series]


StatSpec = DirectStat | DerivedStat


def _parse_distances(fg_made_list: str | None) -> list[int]:
    """Parse nflreadpy's `fg_made_list` ("25;43;32", or null for no makes)."""
    if not fg_made_list:
        return []
    return [int(d) for d in fg_made_list.split(";") if d]


def _fg_yards(stats: pl.DataFrame, value: float) -> pl.Series:
    """Total yardage of every made field goal, for per-yard (`fgm_yds`) scoring."""
    yards = stats["fg_made_list"].map_elements(
        lambda s: sum(_parse_distances(s)), return_dtype=pl.Int64
    )
    return (yards * value).cast(pl.Float64)


def _fg_legacy_bucket(lo: int, hi: int | None) -> Callable[[pl.DataFrame, float], pl.Series]:
    """Count of made field goals whose distance falls in [lo, hi] (hi=None: lo+),
    parsed from `fg_made_list` -- for `fgm_50p`, the legacy unbounded 50+ bucket
    nflreadpy doesn't provide as a single ready-made column (it only splits
    50-59 vs 60+)."""

    def count_in_range(fg_made_list: str | None) -> int:
        distances = _parse_distances(fg_made_list)
        if hi is None:
            return sum(1 for d in distances if d >= lo)
        return sum(1 for d in distances if lo <= d <= hi)

    def compute(stats: pl.DataFrame, value: float) -> pl.Series:
        counts = stats["fg_made_list"].map_elements(count_in_range, return_dtype=pl.Int64)
        return (counts * value).cast(pl.Float64)

    return compute


def _points_allowed_bucket(lo: int, hi: int | None) -> Callable[[pl.DataFrame, float], pl.Series]:
    """`value` if `points_allowed` falls in [lo, hi] this row (hi=None: lo+), else 0."""

    def compute(stats: pl.DataFrame, value: float) -> pl.Series:
        points_allowed = stats["points_allowed"]
        if hi is None:
            in_bucket = points_allowed >= lo
        else:
            in_bucket = (points_allowed >= lo) & (points_allowed <= hi)
        return in_bucket.fill_null(False).cast(pl.Float64) * value

    return compute


def _yardage_bonus(column: str, threshold: int) -> Callable[[pl.DataFrame, float], pl.Series]:
    """`value` if `column` meets `threshold` this row, else 0 -- a flat bonus, not scaled."""

    def compute(stats: pl.DataFrame, value: float) -> pl.Series:
        met = (stats[column].fill_null(0) >= threshold).cast(pl.Float64)
        return met * value

    return compute


def _sum_columns(*columns: str) -> Callable[[pl.DataFrame, float], pl.Series]:
    """`value` times the sum of several columns -- for `fgmiss`/`xpmiss`, where a
    blocked kick counts as a miss for the kicker's own scoring alongside a
    regular miss, confirmed live: every kicker disagreement in the primary
    league's real golden-test run had `fg_missed`/`pat_missed` == 0 but
    `fg_blocked`/`pat_blocked` == 1, and was short by exactly one miss's worth
    of points."""

    def compute(stats: pl.DataFrame, value: float) -> pl.Series:
        series = [stats[column].fill_null(0).cast(pl.Float64) for column in columns]
        total = series[0]
        for s in series[1:]:
            total = total + s
        return total * value

    return compute


def _te_reception_bonus(stats: pl.DataFrame, value: float) -> pl.Series:
    is_te = (stats["position"] == "TE").cast(pl.Float64)
    return stats["receptions"].fill_null(0).cast(pl.Float64) * is_te * value


def _dst_only(column: str) -> Callable[[pl.DataFrame, float], pl.Series]:
    """DST-only credit -- only the DST's own row (position == "DST"). Two distinct
    real bugs needed this same gate, both confirmed live:

    1. Special-teams events (`def_st_*` vs `st_*`): Sleeper scores the same
       underlying event twice, once for the team and once for the individual
       player who did it, and both keys are simultaneously non-zero in real
       league scoring. An unfixed version of this bug inflated a real DST's
       weekly score by exactly 6 points (one uncredited-twice return TD).
    2. "Team defense" keys (`sack`/`int`/`ff`/`fum_rec`/`safe`): nflreadpy's
       `load_player_stats` carries IDP-style columns (`def_sacks`,
       `fumble_recovery_opp`, etc.) on *every* individual player row, not just
       defenders -- an offensive player can show a non-zero value on a broken or
       trick play. Without this gate, a skill player's own stray value picks up
       credit meant only for the DST roster entity. Confirmed live: Trey Benson
       (RB) picked up a stray +2.0 from `fum_rec`, Sam Darnold (QB) a stray +1.0
       from `ff`, both from their own individual stat line.
    """

    def compute(stats: pl.DataFrame, value: float) -> pl.Series:
        is_dst = (stats["position"] == "DST").cast(pl.Float64)
        return stats[column].fill_null(0).cast(pl.Float64) * is_dst * value

    return compute


def _individual_special_teams_credit(column: str) -> Callable[[pl.DataFrame, float], pl.Series]:
    """Individual-player credit (`st_*`) for a special-teams event -- every row
    except the DST's own. See `_dst_only` for why this gate exists."""

    def compute(stats: pl.DataFrame, value: float) -> pl.Series:
        is_individual = (stats["position"] != "DST").cast(pl.Float64)
        return stats[column].fill_null(0).cast(pl.Float64) * is_individual * value

    return compute


FG_YARDAGE_KEY = "fgm_yds"
FG_BUCKET_KEYS = frozenset(
    {"fgm_0_19", "fgm_20_29", "fgm_30_39", "fgm_40_49", "fgm_50_59", "fgm_60p", "fgm_50p"}
)

_POINTS_ALLOWED_BUCKETS: dict[str, tuple[int, int | None]] = {
    "pts_allow_0": (0, 0),
    "pts_allow_1_6": (1, 6),
    "pts_allow_7_13": (7, 13),
    "pts_allow_14_20": (14, 20),
    "pts_allow_21_27": (21, 27),
    "pts_allow_28_34": (28, 34),
    "pts_allow_35p": (35, None),
}

STAT_KEY_MAP: dict[str, StatSpec] = {
    # Passing
    "pass_yd": DirectStat("passing_yards"),
    "pass_td": DirectStat("passing_tds"),
    "pass_int": DirectStat("passing_interceptions"),
    "pass_2pt": DirectStat("passing_2pt_conversions"),
    "pass_sack": DirectStat("sacks_suffered"),
    # Rushing
    "rush_yd": DirectStat("rushing_yards"),
    "rush_td": DirectStat("rushing_tds"),
    "rush_2pt": DirectStat("rushing_2pt_conversions"),
    # Receiving
    "rec": DirectStat("receptions"),
    "rec_yd": DirectStat("receiving_yards"),
    "rec_td": DirectStat("receiving_tds"),
    "rec_2pt": DirectStat("receiving_2pt_conversions"),
    # Fumbles (any player)
    "fum": DirectStat("fumbles_total"),
    "fum_lost": DirectStat("fumbles_lost_total"),
    "fum_rec_td": DirectStat("fumble_recovery_tds"),
    # Kicking
    "xpm": DirectStat("pat_made"),
    # A blocked kick counts as a miss for the kicker's own scoring -- see
    # _sum_columns's docstring.
    "xpmiss": DerivedStat(_sum_columns("pat_missed", "pat_blocked")),
    "fgmiss": DerivedStat(_sum_columns("fg_missed", "fg_blocked")),
    "fgm_yds": DerivedStat(_fg_yards),
    "fgm_0_19": DirectStat("fg_made_0_19"),
    "fgm_20_29": DirectStat("fg_made_20_29"),
    "fgm_30_39": DirectStat("fg_made_30_39"),
    "fgm_40_49": DirectStat("fg_made_40_49"),
    "fgm_50_59": DirectStat("fg_made_50_59"),
    "fgm_60p": DirectStat("fg_made_60_"),
    "fgm_50p": DerivedStat(_fg_legacy_bucket(50, None)),
    # Team defense / DST (one row per team-week, from load_team_stats). nflreadpy's
    # load_player_stats carries the same IDP-style column names for individual
    # players too, so each must be gated to the DST's own row -- see _dst_only.
    "sack": DerivedStat(_dst_only("def_sacks")),
    "int": DerivedStat(_dst_only("def_interceptions")),
    "ff": DerivedStat(_dst_only("def_fumbles_forced")),
    "fum_rec": DerivedStat(_dst_only("fumble_recovery_opp")),
    "safe": DerivedStat(_dst_only("def_safeties")),
    "blk_kick": DirectStat("opponent_blocked_kicks"),
    # def_return_tds is derived from play-by-play (scoring/stats.py), not
    # team_stats' own def_tds/fumble_recovery_tds columns -- both are unreliable
    # for DST credit (confirmed live -- see stats.py's module docstring).
    "def_td": DirectStat("def_return_tds"),
    # Special teams -- team defense credit (position == "DST" rows only) vs
    # individual player credit (every other row). Both keys read the same
    # underlying column and are simultaneously non-zero in real league scoring, so
    # each must be gated to its own row type -- see _dst_only's docstring for the
    # real double-counting bug this fixes.
    "def_st_td": DerivedStat(_dst_only("special_teams_tds")),
    "st_td": DerivedStat(_individual_special_teams_credit("special_teams_tds")),
    # Forced-fumble/recovery credit. Known gap: always zero for now (see module
    # docstring), but still position-gated against the same future double-count.
    "def_st_ff": DerivedStat(_dst_only("special_teams_forced_fumbles")),
    "def_st_fum_rec": DerivedStat(_dst_only("special_teams_fumble_recoveries")),
    "st_ff": DerivedStat(_individual_special_teams_credit("special_teams_forced_fumbles")),
    "st_fum_rec": DerivedStat(_individual_special_teams_credit("special_teams_fumble_recoveries")),
    # DST points-allowed buckets
    **{
        key: DerivedStat(_points_allowed_bucket(lo, hi))
        for key, (lo, hi) in _POINTS_ALLOWED_BUCKETS.items()
    },
    # Bonuses (SPEC §8.2's "typical keys"; not present in either real league yet)
    "bonus_rec_te": DerivedStat(_te_reception_bonus),
    "bonus_rush_yd_100": DerivedStat(_yardage_bonus("rushing_yards", 100)),
    "bonus_rush_yd_200": DerivedStat(_yardage_bonus("rushing_yards", 200)),
    "bonus_rec_yd_100": DerivedStat(_yardage_bonus("receiving_yards", 100)),
    "bonus_rec_yd_200": DerivedStat(_yardage_bonus("receiving_yards", 200)),
    "bonus_pass_yd_300": DerivedStat(_yardage_bonus("passing_yards", 300)),
    "bonus_pass_yd_400": DerivedStat(_yardage_bonus("passing_yards", 400)),
}

__all__ = [
    "FG_BUCKET_KEYS",
    "FG_YARDAGE_KEY",
    "STAT_KEY_MAP",
    "DerivedStat",
    "DirectStat",
    "StatSpec",
]
