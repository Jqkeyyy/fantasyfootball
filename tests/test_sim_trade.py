"""Task 2.9's own literal acceptance bar (SPEC §14.6, TASKS.md): "uses
common random numbers across the before/after runs and reports both
sides' deltas." Small hand-verifiable fixtures, composing tasks 2.1-2.4's
already-fixture-tested `simulate_season` -- no live-data dependency.
"""

from __future__ import annotations

import pytest

from ffapp.config import CorrelationSettings
from ffapp.league_format import LeagueFormat
from ffapp.sim.season import Matchup, Roster, SimPlayer
from ffapp.sim.trade import TradeProposal, analyze_trade, apply_trade

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
        n_teams=4,
        starters={},
        flex_slots={"FLEX": 1, "SUPER_FLEX": 0, "REC_FLEX": 0},
        flex_eligible={"FLEX": ["RB", "WR"]},
        bench=1,
        ir=0,
        playoff_week_start=4,
        waiver_budget=None,
    )


def _four_team_league() -> tuple[list[Roster], list[Matchup], LeagueFormat]:
    fmt = _flex_only_format()
    teams = [
        Roster(
            team_id=name,
            players=[
                _player(f"{name}_rb", "RB", team=name, mean=10.0 + i),
                _player(f"{name}_bench", "RB", team=name, mean=1.0),
            ],
        )
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


_SIM_KWARGS = dict(
    remaining_weeks=[1, 2, 3, 4, 5],
    playoff_week_start=4,
    n_playoff_teams=4,
    season_sims=300,
    recovery_prob=0.5,
)


# --- apply_trade -------------------------------------------------------------------------


def test_apply_trade_swaps_named_players_between_the_two_rosters() -> None:
    teams, _, _ = _four_team_league()
    proposal = TradeProposal(
        team_a="A", team_b="B", players_from_a=["A_bench"], players_from_b=["B_rb"]
    )

    result = apply_trade(teams, proposal)

    a = next(t for t in result if t.team_id == "A")
    b = next(t for t in result if t.team_id == "B")
    assert {p.player_id for p in a.players} == {"A_rb", "B_rb"}
    assert {p.player_id for p in b.players} == {"B_bench", "A_bench"}


def test_apply_trade_leaves_uninvolved_teams_unchanged() -> None:
    teams, _, _ = _four_team_league()
    proposal = TradeProposal(
        team_a="A", team_b="B", players_from_a=["A_bench"], players_from_b=["B_rb"]
    )

    result = apply_trade(teams, proposal)

    c = next(t for t in result if t.team_id == "C")
    original_c = next(t for t in teams if t.team_id == "C")
    assert c is original_c


def test_apply_trade_raises_for_a_player_not_on_the_proposed_side() -> None:
    teams, _, _ = _four_team_league()
    proposal = TradeProposal(
        team_a="A", team_b="B", players_from_a=["not_on_a_roster"], players_from_b=["B_rb"]
    )

    with pytest.raises(ValueError):
        apply_trade(teams, proposal)


def test_apply_trade_raises_for_an_unknown_team_id() -> None:
    teams, _, _ = _four_team_league()
    proposal = TradeProposal(
        team_a="A", team_b="nonexistent", players_from_a=["A_bench"], players_from_b=[]
    )

    with pytest.raises(ValueError):
        apply_trade(teams, proposal)


# --- analyze_trade -----------------------------------------------------------------------


def test_analyze_trade_reports_both_sides() -> None:
    teams, schedule, fmt = _four_team_league()
    proposal = TradeProposal(
        team_a="A", team_b="B", players_from_a=["A_bench"], players_from_b=["B_bench"]
    )

    result = analyze_trade(
        teams,
        schedule,
        fmt,
        _NO_CORRELATION,
        proposal,
        vor_by_player={},
        rng_seed=0,
        **_SIM_KWARGS,
    )

    assert result.team_a.team_id == "A"
    assert result.team_b.team_id == "B"


def test_analyze_trade_uses_the_same_seed_for_both_runs() -> None:
    """A no-op-ish trade (swapping two functionally identical bench
    players with the same mean/grid) should net out close to zero on
    every metric -- if the before/after runs used *different* random
    seeds, the two independent Monte Carlo estimates would show real
    sampling noise instead of nearly cancelling out."""
    teams, schedule, fmt = _four_team_league()
    proposal = TradeProposal(
        team_a="A", team_b="B", players_from_a=["A_bench"], players_from_b=["B_bench"]
    )

    result = analyze_trade(
        teams,
        schedule,
        fmt,
        _NO_CORRELATION,
        proposal,
        vor_by_player={},
        rng_seed=42,
        **_SIM_KWARGS,
    )

    assert result.team_a.delta_expected_wins == pytest.approx(0.0, abs=1e-9)
    assert result.team_a.delta_p_playoffs == pytest.approx(0.0, abs=1e-9)
    assert result.team_a.delta_p_title == pytest.approx(0.0, abs=1e-9)


def test_analyze_trade_naive_vor_delta_is_the_sum_of_ros_vor_swapped() -> None:
    teams, schedule, fmt = _four_team_league()
    proposal = TradeProposal(
        team_a="A", team_b="B", players_from_a=["A_bench"], players_from_b=["B_rb"]
    )
    vor_by_player = {"A_bench": 2.0, "B_rb": 15.0}

    result = analyze_trade(
        teams,
        schedule,
        fmt,
        _NO_CORRELATION,
        proposal,
        vor_by_player=vor_by_player,
        rng_seed=0,
        **_SIM_KWARGS,
    )

    assert result.team_a.naive_vor_delta == pytest.approx(15.0 - 2.0)
    assert result.team_b.naive_vor_delta == pytest.approx(2.0 - 15.0)


def test_analyze_trade_position_deltas_reflect_what_each_side_gained_and_lost() -> None:
    teams, schedule, fmt = _four_team_league()
    proposal = TradeProposal(
        team_a="A", team_b="B", players_from_a=["A_bench"], players_from_b=["B_rb"]
    )

    result = analyze_trade(
        teams,
        schedule,
        fmt,
        _NO_CORRELATION,
        proposal,
        vor_by_player={},
        rng_seed=0,
        **_SIM_KWARGS,
    )

    # Both sides traded one RB for one RB -- net position count unchanged.
    assert result.team_a.position_deltas == {}
    assert result.team_b.position_deltas == {}
    assert result.team_a.summary() == "no position change"


def test_analyze_trade_upgrading_a_startable_slot_improves_expected_wins() -> None:
    """A real, meaningful trade: team A gives up its bench RB (mean 1.0,
    never starts) for team B's real starter-quality RB (mean 25.0) --
    should visibly raise team A's own expected wins, not just noise
    around zero."""
    teams, schedule, fmt = _four_team_league()
    strong_rb = _player("strong_rb", "RB", team="B", mean=25.0, values=(20, 22, 25, 28, 30))
    teams = [
        Roster(team_id=t.team_id, players=[*t.players, strong_rb]) if t.team_id == "B" else t
        for t in teams
    ]
    proposal = TradeProposal(
        team_a="A", team_b="B", players_from_a=["A_bench"], players_from_b=["strong_rb"]
    )

    result = analyze_trade(
        teams,
        schedule,
        fmt,
        _NO_CORRELATION,
        proposal,
        vor_by_player={},
        rng_seed=0,
        season_sims=2000,
        remaining_weeks=[1, 2, 3, 4, 5],
        playoff_week_start=4,
        n_playoff_teams=4,
        recovery_prob=0.5,
    )

    assert result.team_a.delta_expected_wins > 0.0
