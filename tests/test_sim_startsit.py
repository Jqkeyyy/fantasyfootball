"""Task 2.5's own literal acceptance bar (SPEC §14.3, TASKS.md): "a
constructed heavy-underdog scenario recommends the higher-variance
option and a heavy-favourite scenario recommends the floor." All
fixtures are small and hand-verifiable, composing tasks 2.1/2.2/2.4's
already-fixture-tested building blocks -- no live-data dependency.
"""

from __future__ import annotations

import numpy as np
import pytest

from ffapp.config import CorrelationSettings
from ffapp.league_format import LeagueFormat
from ffapp.sim.season import SimPlayer
from ffapp.sim.startsit import enumerate_candidate_lineups, evaluate_start_sit

_NO_CORRELATION = CorrelationSettings(
    qb_pass_catcher=0.0, same_team_rb_rb=0.0, player_vs_opposing_dst=0.0
)
_ALPHAS = (0.10, 0.25, 0.50, 0.75, 0.90)


def _player(
    player_id: str,
    position: str,
    *,
    team: str = "AAA",
    mean: float = 10.0,
    values: tuple[float, ...] = (5.0, 8.0, 10.0, 12.0, 15.0),
) -> SimPlayer:
    return SimPlayer(
        player_id=player_id,
        position=position,
        team=team,
        opponent_team=None,
        mean=mean,
        alphas=_ALPHAS,
        quantile_values=values,
        p_miss=0.0,
    )


def _flex_only_format() -> LeagueFormat:
    return LeagueFormat(
        n_teams=2,
        starters={},
        flex_slots={"FLEX": 1, "SUPER_FLEX": 0, "REC_FLEX": 0},
        flex_eligible={"FLEX": ["RB", "WR"]},
        bench=0,
        ir=0,
        playoff_week_start=99,
        waiver_budget=None,
    )


def _qb_plus_flex_format() -> LeagueFormat:
    return LeagueFormat(
        n_teams=2,
        starters={"QB": 1},
        flex_slots={"FLEX": 1, "SUPER_FLEX": 0, "REC_FLEX": 0},
        flex_eligible={"FLEX": ["RB", "WR"]},
        bench=0,
        ir=0,
        playoff_week_start=99,
        waiver_budget=None,
    )


# --- enumerate_candidate_lineups -------------------------------------------------------


def test_baseline_candidate_is_the_projection_optimal_lineup() -> None:
    roster = [_player("rb1", "RB", mean=10.0), _player("rb2", "RB", mean=8.0)]

    candidates = enumerate_candidate_lineups(roster, _flex_only_format())

    baseline = candidates[0]
    assert baseline.swapped_slot is None
    assert baseline.swapped_in is None
    assert baseline.swapped_out is None
    assert baseline.lineup.slots["FLEX_1"] == "rb1"


def test_swap_variants_are_generated_only_for_flex_eligible_slots() -> None:
    """QB_1 is a dedicated slot with two real candidates (qb1/qb2), but
    SPEC §14.3 step 3 only asks for swap variants "at each flex-eligible
    slot" -- no QB_1 swap should ever appear, even though bench depth
    exists there."""
    roster = [
        _player("qb1", "QB", mean=20.0),
        _player("qb2", "QB", mean=15.0),
        _player("rb1", "RB", mean=10.0),
        _player("rb2", "RB", mean=8.0),
    ]

    candidates = enumerate_candidate_lineups(roster, _qb_plus_flex_format())

    assert len(candidates) == 2  # baseline + exactly one FLEX_1 swap
    assert all(c.swapped_slot != "QB_1" for c in candidates)
    swap = candidates[1]
    assert swap.swapped_slot == "FLEX_1"
    assert swap.swapped_out == "rb1"
    assert swap.swapped_in == "rb2"


def test_swap_variant_delta_projected_points_matches_the_mean_difference() -> None:
    roster = [_player("rb1", "RB", mean=10.0), _player("rb2", "RB", mean=8.0)]

    candidates = enumerate_candidate_lineups(roster, _flex_only_format())

    baseline, swap = candidates
    assert swap.lineup.total_points - baseline.lineup.total_points == pytest.approx(8.0 - 10.0)


def test_no_swap_variants_when_no_flex_slot_is_active() -> None:
    fmt = LeagueFormat(
        n_teams=2,
        starters={"RB": 1},
        flex_slots={"FLEX": 0, "SUPER_FLEX": 0, "REC_FLEX": 0},
        flex_eligible={},
        bench=0,
        ir=0,
        playoff_week_start=99,
        waiver_budget=None,
    )
    roster = [_player("rb1", "RB", mean=10.0), _player("rb2", "RB", mean=8.0)]

    candidates = enumerate_candidate_lineups(roster, fmt)

    assert len(candidates) == 1


# --- evaluate_start_sit: SPEC's own acceptance bar --------------------------------------


def test_heavy_underdog_recommends_the_higher_variance_option() -> None:
    """SPEC §14.3: "if you are a heavy underdog, the highest-P(win)
    lineup is not the highest-projected one -- it is the higher-variance
    one, because you need an outcome in the tail." `safe` has the higher
    mean (10 > 8) and is what the projection-optimal lineup starts; but
    against a heavy-favourite opponent (mean 50, tight), only `alt`'s
    wide right tail (up to 60) has any real chance of winning."""
    safe = _player("safe", "RB", mean=10.0, values=(8.0, 9.0, 10.0, 11.0, 12.0))
    alt = _player("alt", "RB", mean=8.0, values=(0.0, 2.0, 8.0, 20.0, 60.0))
    opponent = [_player("strong", "RB", mean=50.0, values=(45.0, 48.0, 50.0, 52.0, 55.0))]
    rng = np.random.default_rng(0)

    result = evaluate_start_sit(
        [safe, alt], opponent, _flex_only_format(), _NO_CORRELATION, week_sims=5000, rng=rng
    )

    assert result.recommended.swapped_in == "alt"
    baseline = next(c for c in result.candidates if c.swapped_slot is None)
    swap = next(c for c in result.candidates if c.swapped_in == "alt")
    assert swap.p_win > baseline.p_win
    assert swap.delta_p_win > 0.0


def test_heavy_favourite_recommends_the_floor() -> None:
    """SPEC §14.3: "If you are a heavy favourite, the correct play is
    the safe floor." Against a weak opponent (mean 2, tight), `safe`
    (mean 10, tight) already wins nearly every sim; `alt`'s wide lower
    tail (down to 0) only introduces a real chance of an upset loss,
    never a benefit."""
    safe = _player("safe", "RB", mean=10.0, values=(8.0, 9.0, 10.0, 11.0, 12.0))
    alt = _player("alt", "RB", mean=8.0, values=(0.0, 2.0, 8.0, 20.0, 60.0))
    opponent = [_player("weak", "RB", mean=2.0, values=(1.0, 1.5, 2.0, 2.5, 3.0))]
    rng = np.random.default_rng(0)

    result = evaluate_start_sit(
        [safe, alt], opponent, _flex_only_format(), _NO_CORRELATION, week_sims=5000, rng=rng
    )

    assert result.recommended.swapped_slot is None
    assert result.recommended.p_win > 0.95


def test_candidates_are_ranked_by_p_win_descending() -> None:
    safe = _player("safe", "RB", mean=10.0, values=(8.0, 9.0, 10.0, 11.0, 12.0))
    alt = _player("alt", "RB", mean=8.0, values=(0.0, 2.0, 8.0, 20.0, 60.0))
    opponent = [_player("mid", "RB", mean=20.0, values=(15.0, 18.0, 20.0, 22.0, 25.0))]
    rng = np.random.default_rng(0)

    result = evaluate_start_sit(
        [safe, alt], opponent, _flex_only_format(), _NO_CORRELATION, week_sims=3000, rng=rng
    )

    p_wins = [c.p_win for c in result.candidates]
    assert p_wins == sorted(p_wins, reverse=True)
    assert result.recommended is result.candidates[0]
