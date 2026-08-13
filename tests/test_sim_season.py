"""Task 2.4's own literal acceptance bar (SPEC §13.4): "lineups are set
on *projections* and results drawn from *samples* (assert this in a
test -- it is the most commonly botched detail), and playoff odds sum
sensibly across the league." All fixtures here are small and
hand-verifiable -- SPEC's own pseudocode has no live-data dependency,
composing tasks 2.1-2.3's already-fixture-tested building blocks.
"""

from __future__ import annotations

import numpy as np
import pytest

from ffapp.config import CorrelationSettings
from ffapp.league_format import LeagueFormat
from ffapp.sim.season import (
    Matchup,
    Roster,
    SimPlayer,
    simulate_availability,
    simulate_season,
    simulate_team_week_totals,
)

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
    p_miss: float = 0.0,
) -> SimPlayer:
    return SimPlayer(
        player_id=player_id,
        position=position,
        team=team,
        opponent_team=None,
        mean=mean,
        alphas=_ALPHAS,
        quantile_values=values,
        p_miss=p_miss,
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


# --- simulate_availability -----------------------------------------------------------


def test_simulate_availability_shape() -> None:
    p_miss = np.zeros((5, 3))
    rng = np.random.default_rng(0)

    result = simulate_availability(p_miss, season_sims=100, recovery_prob=0.5, rng=rng)

    assert result.shape == (100, 5, 3)


def test_simulate_availability_is_always_true_when_p_miss_is_zero() -> None:
    p_miss = np.zeros((8, 2))
    rng = np.random.default_rng(0)

    result = simulate_availability(p_miss, season_sims=50, recovery_prob=0.5, rng=rng)

    assert result.all()


def test_simulate_availability_has_persistence_not_independent_weekly_rerolls() -> None:
    """SPEC's own words: "a multi-week injury keeps the player out for a
    sampled duration, not resampled independently each week." With a low
    weekly hazard (0.3) but a low weekly recovery chance (0.02, expected
    duration ~50 weeks), a player who misses one week should almost
    always still be out the following week -- far more often than the
    bare 0.3 weekly hazard alone would predict if each week were an
    independent re-roll."""
    rng = np.random.default_rng(0)
    p_miss = np.full((10, 1), 0.3)

    availability = simulate_availability(p_miss, season_sims=5000, recovery_prob=0.02, rng=rng)
    missed = ~availability[:, :, 0]

    missed_this_week = missed[:, :-1]
    missed_next_week = missed[:, 1:]
    n_missed = missed_this_week.sum()
    continuation_rate = (missed_this_week & missed_next_week).sum() / n_missed

    assert continuation_rate > 0.8


# --- simulate_team_week_totals: the "projections, not actuals" bar -------------------


def test_lineups_are_set_from_projections_not_from_the_sampled_actuals() -> None:
    """The task's own most-commonly-botched-detail bar. WR1's `mean` is
    deliberately far below RB1's even though WR1's real quantile grid
    would almost always outscore RB1's if sampled -- a lineup optimiser
    that peeked at the actual sample would start WR1 far more often (and
    score much higher); one that correctly uses only the projection will
    always start RB1, so the team's simulated total must track RB1's own
    marginal distribution, never WR1's, regardless of what WR1's sample
    would have scored that path."""
    rb1 = _player("rb1", "RB", mean=10.0, values=(8.0, 9.0, 10.0, 11.0, 12.0))
    wr1 = _player("wr1", "WR", mean=1.0, values=(40.0, 45.0, 50.0, 55.0, 60.0))
    team = Roster(team_id="A", players=[rb1, wr1])
    rng = np.random.default_rng(0)

    totals, lineups = simulate_team_week_totals(
        [team],
        _flex_only_format(),
        _NO_CORRELATION,
        remaining_weeks=[1, 2, 3],
        season_sims=2000,
        recovery_prob=0.5,
        rng=rng,
    )

    assert lineups["A"].slots["FLEX_1"] == "rb1"
    # RB1's real grid tops out at 12; if WR1's sample (35-65 range) had
    # ever been used instead, the mean total would be far higher than 12.
    assert totals["A"].mean() < 12.0
    assert totals["A"].mean() > 8.0


def test_simulate_team_week_totals_returns_shape_season_sims_by_n_weeks() -> None:
    team = Roster(team_id="A", players=[_player("rb1", "RB")])
    rng = np.random.default_rng(0)

    totals, _ = simulate_team_week_totals(
        [team],
        _flex_only_format(),
        _NO_CORRELATION,
        remaining_weeks=[1, 2, 3, 4],
        season_sims=300,
        recovery_prob=0.5,
        rng=rng,
    )

    assert totals["A"].shape == (300, 4)


# --- simulate_season: playoff odds sum sensibly ---------------------------------------


def _four_team_league() -> tuple[list[Roster], list[Matchup], LeagueFormat]:
    fmt = LeagueFormat(
        n_teams=4,
        starters={},
        flex_slots={"FLEX": 1, "SUPER_FLEX": 0, "REC_FLEX": 0},
        flex_eligible={"FLEX": ["RB", "WR"]},
        bench=0,
        ir=0,
        playoff_week_start=4,
        waiver_budget=None,
    )
    teams = [
        Roster(team_id=name, players=[_player(f"{name}_rb", "RB", team=name, mean=10.0 + i)])
        for i, name in enumerate(["A", "B", "C", "D"])
    ]
    schedule = [
        Matchup(week=1, home="A", away="B"),
        Matchup(week=1, home="C", away="D"),
        Matchup(week=2, home="A", away="C"),
        Matchup(week=2, home="B", away="D"),
        Matchup(week=3, home="A", away="D"),
        Matchup(week=3, home="B", away="C"),
    ]
    return teams, schedule, fmt


def test_playoff_and_title_odds_sum_sensibly_across_the_league() -> None:
    teams, schedule, fmt = _four_team_league()
    rng = np.random.default_rng(0)

    result = simulate_season(
        teams,
        schedule,
        fmt,
        _NO_CORRELATION,
        remaining_weeks=[1, 2, 3, 4, 5],
        playoff_week_start=4,
        n_playoff_teams=4,
        season_sims=500,
        recovery_prob=0.5,
        rng=rng,
    )

    assert sum(result.p_playoffs.values()) == pytest.approx(4.0)
    assert sum(result.p_title.values()) == pytest.approx(1.0)
    assert set(result.expected_wins) == {"A", "B", "C", "D"}
    for p in result.p_playoffs.values():
        assert 0.0 <= p <= 1.0
    for p in result.p_title.values():
        assert 0.0 <= p <= 1.0


def test_expected_wins_sum_to_the_real_number_of_regular_season_matchups() -> None:
    """3 regular-season weeks x 2 matchups/week = 6 total real wins
    awarded across the league every single sim -- summed expected wins
    across all teams must equal that constant exactly."""
    teams, schedule, fmt = _four_team_league()
    rng = np.random.default_rng(1)

    result = simulate_season(
        teams,
        schedule,
        fmt,
        _NO_CORRELATION,
        remaining_weeks=[1, 2, 3, 4, 5],
        playoff_week_start=4,
        n_playoff_teams=4,
        season_sims=500,
        recovery_prob=0.5,
        rng=rng,
    )

    assert sum(result.expected_wins.values()) == pytest.approx(6.0)


def test_simulate_season_rejects_a_non_power_of_two_playoff_field() -> None:
    teams, schedule, fmt = _four_team_league()
    rng = np.random.default_rng(0)

    with pytest.raises(ValueError):
        simulate_season(
            teams,
            schedule,
            fmt,
            _NO_CORRELATION,
            remaining_weeks=[1, 2, 3, 4, 5],
            playoff_week_start=4,
            n_playoff_teams=3,
            season_sims=100,
            recovery_prob=0.5,
            rng=rng,
        )
