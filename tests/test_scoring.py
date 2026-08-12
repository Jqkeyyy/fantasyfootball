"""Tests for scoring/keymap.py and scoring/engine.py (SPEC.md §8).

Column names for the per-player-week stat frame these tests build by hand are
nflreadpy's own `load_player_stats`/`load_team_stats` column names, confirmed by
live inspection (see HANDOFF.md §4) rather than assumed -- e.g. `passing_yards`,
`def_sacks`, `fg_made_list`. `points_allowed` and the special-teams-credit columns
are this project's own invention (not present in nflreadpy), derived at
normalisation time -- see HANDOFF.md §4/§5 for what's real vs. derived vs. a known
gap.
"""

from __future__ import annotations

import polars as pl
import pytest

from ffapp.config import LEAGUES_DIR, load_league
from ffapp.scoring import engine


def _stats(**columns: list[object]) -> pl.DataFrame:
    return pl.DataFrame(columns)


# --- direct stats -------------------------------------------------------------


def test_score_stat_line_applies_direct_stat() -> None:
    stats = _stats(passing_yards=[250, 100])

    points = engine.score_stat_line(stats, {"pass_yd": 0.04})

    assert points.to_list() == pytest.approx([10.0, 4.0])


def test_score_stat_line_sums_multiple_keys() -> None:
    stats = _stats(passing_yards=[250], passing_tds=[2])

    points = engine.score_stat_line(stats, {"pass_yd": 0.04, "pass_td": 4})

    assert points.to_list() == pytest.approx([10.0 + 8.0])


def test_score_stat_line_ignores_zero_value_keys_even_if_unmapped() -> None:
    """A scoring key present at 0.0 with no STAT_KEY_MAP entry must not raise or
    require a matching column -- Sleeper's real payloads include every possible
    key, most zeroed out (see config/leagues/*.yml)."""
    stats = _stats(passing_yards=[100])

    points = engine.score_stat_line(stats, {"pass_yd": 0.04, "totally_unknown_key": 0.0})

    assert points.to_list() == pytest.approx([4.0])


def test_score_stat_line_fills_null_stat_as_zero() -> None:
    stats = _stats(passing_yards=[None])

    points = engine.score_stat_line(stats, {"pass_yd": 0.04})

    assert points.to_list() == pytest.approx([0.0])


# --- unhandled_keys / the blocking guardrail (CLAUDE.md rule 3) ----------------


def test_unhandled_keys_returns_nonzero_unmapped_keys() -> None:
    assert engine.unhandled_keys({"totally_unknown_key": 5.0}) == ["totally_unknown_key"]


def test_unhandled_keys_ignores_zero_value_unmapped_keys() -> None:
    assert engine.unhandled_keys({"totally_unknown_key": 0.0}) == []


def test_unhandled_keys_ignores_mapped_keys() -> None:
    assert engine.unhandled_keys({"pass_yd": 0.04}) == []


def test_score_stat_line_raises_for_unhandled_nonzero_key() -> None:
    stats = _stats(passing_yards=[100])

    with pytest.raises(engine.UnhandledScoringKeysError):
        engine.score_stat_line(stats, {"totally_unknown_key": 5.0})


# --- field goal distance buckets and per-yard scoring (ADDENDUM-01 §C) ---------
# `fg_made_0_19` .. `fg_made_60_` are nflreadpy's own bucket-count columns, and their
# boundaries match Sleeper's fgm_* buckets exactly, so those keys are plain direct
# stats. Per-yard (`fgm_yds`) and the legacy unbounded `fgm_50p` bucket need the real
# distance of every kick, which nflreadpy exposes as `fg_made_list` -- a
# semicolon-delimited string ("25;43;32"), null when no field goals were made, not a
# native list column.


def test_score_stat_line_fg_bucket_scoring_uses_nflreadpy_bucket_column_directly() -> None:
    stats = _stats(fg_made_40_49=[1, 0])

    points = engine.score_stat_line(stats, {"fgm_40_49": 4.0})

    assert points.to_list() == pytest.approx([4.0, 0.0])


def test_score_stat_line_fg_yardage_scoring_sums_distances_from_semicolon_string() -> None:
    stats = _stats(fg_made_list=["25;45", None])

    points = engine.score_stat_line(stats, {"fgm_yds": 0.1})

    assert points.to_list() == pytest.approx([7.0, 0.0])


def test_score_stat_line_fg_50_plus_bucket_is_unbounded() -> None:
    stats = _stats(fg_made_list=["61", "59", None])

    points = engine.score_stat_line(stats, {"fgm_50p": 6.0})

    assert points.to_list() == pytest.approx([6.0, 6.0, 0.0])


def test_score_stat_line_raises_on_conflicting_fg_schemes() -> None:
    stats = _stats(fg_made_list=["45"], fg_made_40_49=[1])

    with pytest.raises(engine.ConflictingFieldGoalSchemeError):
        engine.score_stat_line(stats, {"fgm_yds": 0.1, "fgm_40_49": 4.0})


def test_score_stat_line_allows_zero_valued_sibling_fg_scheme() -> None:
    """Mirrors real data: rogan-radinator-league carries every fgm_* bucket key at
    0.0 alongside a non-zero fgm_yds -- only a *non-zero* conflict should raise."""
    stats = _stats(fg_made_list=["45"], fg_made_40_49=[1])

    points = engine.score_stat_line(stats, {"fgm_yds": 0.1, "fgm_40_49": 0.0})

    assert points.to_list() == pytest.approx([4.5])


# --- blocked kicks count as a miss for the kicker's own scoring -----------------
# Confirmed live: every remaining kicker disagreement in the primary league's real
# golden-test run (task 0.5) had `fg_missed`/`pat_missed` == 0 but `fg_blocked`/
# `pat_blocked` == 1, and computed was exactly 1.0 point higher than Sleeper's own
# score in every case -- Sleeper penalises a blocked kick the same as a missed one
# (fgmiss/xpmiss), but the original mapping only read fg_missed/pat_missed, never
# the blocked count.


def test_score_stat_line_fgmiss_counts_a_blocked_field_goal_as_a_miss() -> None:
    stats = _stats(fg_missed=[0], fg_blocked=[1])

    points = engine.score_stat_line(stats, {"fgmiss": -1.0})

    assert points.to_list() == pytest.approx([-1.0])


def test_score_stat_line_fgmiss_sums_regular_misses_and_blocks() -> None:
    stats = _stats(fg_missed=[2], fg_blocked=[1])

    points = engine.score_stat_line(stats, {"fgmiss": -1.0})

    assert points.to_list() == pytest.approx([-3.0])


def test_score_stat_line_xpmiss_counts_a_blocked_extra_point_as_a_miss() -> None:
    stats = _stats(pat_missed=[0], pat_blocked=[1])

    points = engine.score_stat_line(stats, {"xpmiss": -1.0})

    assert points.to_list() == pytest.approx([-1.0])


# --- DST points-allowed buckets -------------------------------------------------


def test_score_stat_line_points_allowed_bucket_scores_matching_bucket_only() -> None:
    stats = _stats(points_allowed=[10, 3])

    points = engine.score_stat_line(stats, {"pts_allow_7_13": 4.0, "pts_allow_1_6": 7.0})

    assert points.to_list() == pytest.approx([4.0, 7.0])


def test_score_stat_line_points_allowed_zero_bucket_is_exact() -> None:
    stats = _stats(points_allowed=[0, 1])

    points = engine.score_stat_line(stats, {"pts_allow_0": 10.0})

    assert points.to_list() == pytest.approx([10.0, 0.0])


def test_score_stat_line_points_allowed_35_plus_is_unbounded() -> None:
    stats = _stats(points_allowed=[35, 50, 34])

    points = engine.score_stat_line(stats, {"pts_allow_35p": -4.0})

    assert points.to_list() == pytest.approx([-4.0, -4.0, 0.0])


# --- bonuses (SPEC §8.2's "typical keys"; not present in either real league yet) -


def test_score_stat_line_bonus_rush_yd_100_is_threshold_not_scaled() -> None:
    stats = _stats(rushing_yards=[99, 100, 150])

    points = engine.score_stat_line(stats, {"bonus_rush_yd_100": 3.0})

    assert points.to_list() == pytest.approx([0.0, 3.0, 3.0])


def test_score_stat_line_bonus_rec_te_multiplies_receptions_for_tight_ends_only() -> None:
    stats = _stats(receptions=[5, 5], position=["TE", "WR"])

    points = engine.score_stat_line(stats, {"bonus_rec_te": 0.5})

    assert points.to_list() == pytest.approx([2.5, 0.0])


# --- special-teams dual credit: def_st_* (team) vs st_* (individual) ------------
# Sleeper scores a special-teams TD/forced-fumble/recovery twice: once as team-
# defense credit (def_st_*) and once as individual-player credit (st_*). Both keys
# are simultaneously non-zero in real league scoring (rogan-radinator-league and
# bdff-chopped both do this), and both read the same underlying column
# (special_teams_tds etc.) -- naively mapping both keys straight to that column
# double-counts every real event, confirmed live: NE's actual week-2-2025
# special_teams_tds=1 inflated NE's DST score from a correct 13 to a computed 19
# before this position gate existed. Each key must only fire on its own row type.


def test_score_stat_line_credits_def_st_td_only_on_the_dst_row() -> None:
    stats = _stats(special_teams_tds=[1, 1], position=["DST", "WR"])

    points = engine.score_stat_line(stats, {"def_st_td": 6.0})

    assert points.to_list() == pytest.approx([6.0, 0.0])


def test_score_stat_line_credits_st_td_only_on_non_dst_rows() -> None:
    stats = _stats(special_teams_tds=[1, 1], position=["DST", "WR"])

    points = engine.score_stat_line(stats, {"st_td": 6.0})

    assert points.to_list() == pytest.approx([0.0, 6.0])


def test_score_stat_line_does_not_double_count_a_real_special_teams_td() -> None:
    """Both def_st_td and st_td active simultaneously (the normal real-league
    case) must credit the DST row exactly once, not twice."""
    stats = _stats(special_teams_tds=[1], position=["DST"])

    points = engine.score_stat_line(stats, {"def_st_td": 6.0, "st_td": 6.0})

    assert points.to_list() == pytest.approx([6.0])


def test_score_stat_line_gates_special_teams_forced_fumble_and_recovery_credit_too() -> None:
    stats = _stats(
        special_teams_forced_fumbles=[1],
        special_teams_fumble_recoveries=[1],
        position=["DST"],
    )

    points = engine.score_stat_line(
        stats,
        {"def_st_ff": 1.0, "st_ff": 1.0, "def_st_fum_rec": 1.0, "st_fum_rec": 1.0},
    )

    assert points.to_list() == pytest.approx([2.0])  # def_st_ff + def_st_fum_rec, not x2


# --- "Team defense" keys (sack/int/ff/fum_rec/safe) must not fire on player rows
# nflreadpy's `load_player_stats` includes IDP-style columns (def_sacks,
# def_interceptions, def_fumbles_forced, fumble_recovery_opp, def_safeties) on
# every individual player row, not just defenders -- an offensive player can show
# a non-zero value on a broken/trick play. Sleeper's "Team defense" keys are meant
# for the DST roster entity only. Confirmed live: Trey Benson (RB) picked up a
# stray +2.0 from `fum_rec` and Sam Darnold (QB) a stray +1.0 from `ff`, both from
# their own individual `fumble_recovery_opp`/`def_fumbles_forced` values.


def test_score_stat_line_credits_fum_rec_only_on_the_dst_row() -> None:
    stats = _stats(fumble_recovery_opp=[1, 1], position=["DST", "RB"])

    points = engine.score_stat_line(stats, {"fum_rec": 2.0})

    assert points.to_list() == pytest.approx([2.0, 0.0])


def test_score_stat_line_credits_ff_only_on_the_dst_row() -> None:
    stats = _stats(def_fumbles_forced=[1, 1], position=["DST", "QB"])

    points = engine.score_stat_line(stats, {"ff": 1.0})

    assert points.to_list() == pytest.approx([1.0, 0.0])


def test_score_stat_line_credits_sack_int_safe_only_on_the_dst_row() -> None:
    stats = _stats(
        def_sacks=[1, 1],
        def_interceptions=[1, 1],
        def_safeties=[1, 1],
        position=["DST", "WR"],
    )

    points = engine.score_stat_line(stats, {"sack": 1.0, "int": 2.0, "safe": 2.0})

    assert points.to_list() == pytest.approx([5.0, 0.0])


# --- real league data (task 0.4's literal acceptance criterion) ----------------


@pytest.mark.parametrize("slug", ["rogan-radinator-league", "bdff-chopped"])
def test_unhandled_keys_is_empty_for_real_league_scoring(slug: str) -> None:
    league = load_league(slug, leagues_dir=LEAGUES_DIR)
    scoring = league.league_cache["scoring_settings"]

    assert engine.unhandled_keys(scoring) == []
