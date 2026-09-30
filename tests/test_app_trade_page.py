from __future__ import annotations

import polars as pl

from ffapp.app.trade_page import (
    build_trade_rosters,
    matchup_schedule,
    standings_from_rosters,
    suggested_trade_packages,
    trade_analysis_blocker,
)
from ffapp.config import LeagueConfig
from ffapp.sim.season import Roster, SimPlayer


def test_build_trade_rosters_aggregates_remaining_weeks() -> None:
    projections = pl.DataFrame(
        {
            "player_id": ["p1", "p1", "p2"],
            "week": [2, 3, 2],
            "position": ["RB", "RB", "WR"],
            "team": ["A", "A", "B"],
            "opponent_team": ["B", "C", "A"],
            "mean": [10.0, 14.0, 8.0],
            "q10": [2.0, 4.0, 1.0],
            "q25": [6.0, 8.0, 4.0],
            "q50": [10.0, 14.0, 8.0],
            "q75": [14.0, 20.0, 12.0],
            "q90": [18.0, 24.0, 15.0],
        }
    )

    rosters, vor = build_trade_rosters(projections, {"1": ["p1"], "2": ["p2"]}, from_week=2)

    assert rosters[0].players[0].mean == 12.0
    assert set(rosters[0].players[0].weekly or {}) == {2, 3}
    assert len(rosters) == 2
    assert set(vor) == {"p1", "p2"}


def test_build_trade_rosters_excludes_null_projection_rows() -> None:
    projections = pl.DataFrame(
        {
            "player_id": ["good", "missing"],
            "week": [3, 3],
            "position": ["RB", "RB"],
            "team": ["A", "A"],
            "opponent_team": ["B", "B"],
            "mean": [10.0, None],
            "q10": [2.0, None],
            "q25": [6.0, None],
            "q50": [10.0, None],
            "q75": [14.0, None],
            "q90": [18.0, None],
        }
    )
    rosters, _ = build_trade_rosters(
        projections, {"1": ["good", "missing"]}, from_week=3
    )
    assert [player.player_id for player in rosters[0].players] == ["good"]


def test_suggested_trade_packages_matches_mutual_needs_and_value() -> None:
    def player(player_id: str, position: str, mean: float) -> SimPlayer:
        return SimPlayer(
            player_id=player_id,
            position=position,
            team="NFL",
            opponent_team=None,
            mean=mean,
            alphas=(0.1, 0.25, 0.5, 0.75, 0.9),
            quantile_values=(mean - 4, mean - 2, mean, mean + 2, mean + 4),
            p_miss=0.0,
        )

    teams = [
        Roster("mine", [player("my_rb", "RB", 20), player("my_wr", "WR", 5)]),
        Roster("other", [player("their_rb", "RB", 5), player("their_wr", "WR", 20)]),
        Roster("third", [player("third_rb", "RB", 12), player("third_wr", "WR", 12)]),
    ]
    ideas = suggested_trade_packages(
        teams,
        {"my_rb": 30, "my_wr": 5, "their_rb": 5, "their_wr": 29},
        {"my_rb": "My RB", "their_wr": "Their WR"},
        {"other": "Other Team"},
        "mine",
    )
    assert ideas.row(0, named=True)["target"] == "Their WR"
    assert ideas.row(0, named=True)["offer"] == "My RB"


def test_matchup_schedule_pairs_rosters_by_matchup_id() -> None:
    schedule = matchup_schedule(
        {2: [{"roster_id": 1, "matchup_id": 5}, {"roster_id": 2, "matchup_id": 5}]}
    )

    assert [(game.week, game.home, game.away) for game in schedule] == [(2, "1", "2")]


def test_trade_analysis_blocks_elimination_leagues() -> None:
    league = LeagueConfig(
        slug="chopped",
        display_name="Chopped",
        is_primary=False,
        league_id="1",
        season=2026,
        league_cache={"league_type": 3, "disable_trades": 1},
        overrides={},
    )

    assert trade_analysis_blocker(league, 0) == "Trades are disabled in this league."


def test_standings_from_rosters_includes_ties_and_decimal_points() -> None:
    wins, points = standings_from_rosters(
        [
            {
                "roster_id": 4,
                "settings": {"wins": 3, "ties": 1, "fpts": 412, "fpts_decimal": 37},
            }
        ]
    )

    assert wins == {"4": 3.5}
    assert points == {"4": 412.37}
