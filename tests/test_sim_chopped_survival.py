from __future__ import annotations

import polars as pl

from ffapp.league_format import LeagueFormat
from ffapp.sim.chopped_survival import simulate_chopped_survival


def _format() -> LeagueFormat:
    return LeagueFormat(
        n_teams=3,
        starters={"QB": 1},
        flex_slots={},
        flex_eligible={},
        bench=1,
        ir=0,
        playoff_week_start=0,
        waiver_budget=100,
    )


def test_elite_candidate_reduces_weak_rosters_elimination_risk() -> None:
    values = pl.DataFrame(
        {
            "sleeper_id": ["weak", "mid", "strong", "elite"],
            "player_id": ["weak", "mid", "strong", "elite"],
            "position": ["QB"] * 4,
            "current_week_projection": [8.0, 16.0, 20.0, 25.0],
        }
    )
    rosters = [
        {"roster_id": 1, "players": ["weak"]},
        {"roster_id": 2, "players": ["mid"]},
        {"roster_id": 3, "players": ["strong"]},
    ]
    teams, impacts = simulate_chopped_survival(
        rosters, values, 1, _format(), ["elite"], n_sims=20_000, seed=7
    )
    mine = teams.filter(pl.col("roster_id") == 1).row(0, named=True)
    elite = impacts.row(0, named=True)
    assert mine["elimination_probability"] > 0.5
    assert elite["survival_probability_gain"] > 0.3
    assert elite["elimination_probability_after"] < mine["elimination_probability"]
