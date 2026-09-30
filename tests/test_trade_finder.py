from ffapp.app.trade_page import find_trade_candidates
from ffapp.league_format import LeagueFormat
from ffapp.sim.season import Roster, SimPlayer, WeekProjection


def _player(player_id: str, position: str, points: float) -> SimPlayer:
    weekly = {
        week: WeekProjection(points, (points - 4, points - 2, points, points + 2, points + 4), None)
        for week in (3, 4)
    }
    return SimPlayer(
        player_id,
        position,
        "NFL",
        None,
        points,
        (0.1, 0.25, 0.5, 0.75, 0.9),
        (points - 4, points - 2, points, points + 2, points + 4),
        0.0,
        weekly,
    )


def test_trade_finder_prefers_package_that_improves_both_lineups() -> None:
    fmt = LeagueFormat(
        n_teams=2,
        starters={"QB": 0, "RB": 1, "WR": 1},
        flex_slots={},
        flex_eligible={},
        bench=4,
        ir=0,
        playoff_week_start=15,
        waiver_budget=100,
    )
    teams = [
        Roster(
            "mine", [_player("rb1", "RB", 20), _player("rb2", "RB", 18), _player("wr1", "WR", 5)]
        ),
        Roster(
            "other", [_player("rb3", "RB", 5), _player("wr2", "WR", 20), _player("wr3", "WR", 18)]
        ),
    ]
    values = {"rb1": 30, "rb2": 25, "wr1": 5, "rb3": 5, "wr2": 29, "wr3": 24}
    result = find_trade_candidates(
        teams,
        values,
        {key: key for key in values},
        {"other": "Other"},
        "mine",
        fmt,
        remaining_weeks=[3, 4],
    )
    assert not result.is_empty()
    best = result.row(0, named=True)
    assert best["mutual_benefit"]
    assert best["your_lineup_gain"] > 0
    assert best["partner_lineup_gain"] > 0
