from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl

from ffapp.app.weekly_actions_page import (
    build_action_inbox,
    explain_player,
    lineup_decisions,
    recommended_lineup,
    streaming_recommendations,
    waiver_recommendations,
)
from ffapp.league_format import LeagueFormat


def _format() -> LeagueFormat:
    return LeagueFormat(
        n_teams=2,
        starters={"RB": 1, "WR": 1, "K": 1},
        flex_slots={"FLEX": 1, "SUPER_FLEX": 0, "REC_FLEX": 0},
        flex_eligible={"FLEX": ["RB", "WR", "TE"]},
        bench=3,
        ir=0,
        playoff_week_start=15,
        waiver_budget=100,
    )


def _rankings() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "player_id": ["rb1", "rb2", "wr1", "wr2", "fa1"],
            "player_name": ["RB One", "RB Two", "WR One", "WR Two", "Free Agent"],
            "position": ["RB", "RB", "WR", "WR", "WR"],
            "team": ["A", "A", "B", "B", "C"],
            "opponent": ["B", "B", "A", "A", "D"],
            "p_active": [1.0] * 5,
            "proj_mean": [20.0, 9.0, 18.0, 5.0, 12.0],
            "floor": [10.0, 3.0, 8.0, 1.0, 4.0],
            "lower_quartile": [15.0, 6.0, 13.0, 3.0, 8.0],
            "median": [20.0, 9.0, 18.0, 5.0, 12.0],
            "upper_quartile": [25.0, 12.0, 23.0, 7.0, 16.0],
            "ceiling": [30.0, 15.0, 28.0, 9.0, 20.0],
            "owner_status": ["my_roster", "my_roster", "my_roster", "my_roster", "free_agent"],
        }
    )


def test_recommended_lineup_uses_supported_positions_and_marks_changes() -> None:
    result = recommended_lineup(
        _rankings(), {"rb1", "rb2", "wr1", "wr2"}, {"rb2", "wr1", "wr2"}, _format()
    )

    assert result.height == 3
    assert "rb1" in result["player_id"].to_list()
    assert result.filter(pl.col("player_id") == "rb1")["currently_starting"].item() is False
    assert not result["slot"].str.starts_with("K").any()


def test_lineup_decisions_pair_unlocked_swap_with_confidence() -> None:
    rankings = _rankings()
    lineup = recommended_lineup(
        rankings, {"rb1", "rb2", "wr1", "wr2"}, {"rb2", "wr1", "wr2"}, _format()
    )
    result = lineup_decisions(
        lineup,
        rankings,
        {"rb2", "wr1", "wr2"},
        _format(),
        kickoff_by_team={"A": datetime.now(UTC) + timedelta(days=1)},
    )

    assert result.row(0, named=True)["start"] == "RB One"
    assert result.row(0, named=True)["sit"] == "WR Two"
    assert result.row(0, named=True)["expected_gain"] == 15.0
    assert result.row(0, named=True)["confidence_label"] in {"Medium", "High"}


def test_lineup_decisions_exclude_locked_players() -> None:
    rankings = _rankings()
    lineup = recommended_lineup(
        rankings, {"rb1", "rb2", "wr1", "wr2"}, {"rb2", "wr1", "wr2"}, _format()
    )
    result = lineup_decisions(
        lineup,
        rankings,
        {"rb2", "wr1", "wr2"},
        _format(),
        kickoff_by_team={"A": datetime.now(UTC) - timedelta(minutes=1)},
    )

    assert result.is_empty()


def test_explain_player_states_source_uncertainty_and_matchup_limits() -> None:
    row = _rankings().row(0, named=True)
    row.update(
        {
            "projection_source": "espn_weekly",
            "matchup_grade": "B",
            "n_plays_behind_matchup_grade": 120,
        }
    )

    explanation = " ".join(explain_player(row))

    assert "20.0 points from espn_weekly" in explanation
    assert "10.0 floor to 30.0 ceiling" in explanation
    assert "context, not claimed" in explanation


def test_waiver_recommendations_rank_only_real_lineup_upgrades() -> None:
    result = waiver_recommendations(
        _rankings(),
        {"rb1", "rb2", "wr1", "wr2"},
        _format(),
        current_week=10,
        remaining_budget=80,
        playoff_weight=1.5,
        aggressiveness=1.0,
    )

    assert result["player_id"].to_list() == ["fa1"]
    assert result["value_added_per_week"].item() == 3.0
    assert result["drop_player"].item() == "WR Two"
    assert result["claim_role"].item() == "Primary WR"
    assert result["bid_floor"].item() <= result["bid_ceiling"].item()
    assert "projected lineup points" in result["why"].item()


def test_waiver_recommendations_exclude_explicitly_unavailable_players() -> None:
    rankings = _rankings().with_columns(
        pl.when(pl.col("player_id") == "fa1")
        .then(pl.lit("Out"))
        .otherwise(pl.lit(None, dtype=pl.String))
        .alias("injury_status")
    )

    result = waiver_recommendations(
        rankings,
        {"rb1", "rb2", "wr1", "wr2"},
        _format(),
        current_week=10,
        remaining_budget=80,
        playoff_weight=1.5,
        aggressiveness=1.0,
    )

    assert result.is_empty()


def test_recommended_lineup_excludes_explicitly_unavailable_players() -> None:
    rankings = _rankings().with_columns(
        pl.when(pl.col("player_id") == "rb1")
        .then(pl.lit("IR"))
        .otherwise(pl.lit(None, dtype=pl.String))
        .alias("injury_status")
    )

    result = recommended_lineup(
        rankings, {"rb1", "rb2", "wr1", "wr2"}, {"rb1", "wr1", "wr2"}, _format()
    )

    assert "rb1" not in result["player_id"].to_list()


def test_waiver_bid_accounts_for_competing_roster_need() -> None:
    no_competition = waiver_recommendations(
        _rankings(),
        {"rb1", "rb2", "wr1", "wr2"},
        _format(),
        current_week=10,
        remaining_budget=80,
        playoff_weight=1.5,
        aggressiveness=1.0,
    )
    with_competition = waiver_recommendations(
        _rankings(),
        {"rb1", "rb2", "wr1", "wr2"},
        _format(),
        current_week=10,
        remaining_budget=80,
        playoff_weight=1.5,
        aggressiveness=1.0,
        opponent_roster_ids=[{"rb2", "wr2"}],
    )

    assert with_competition["competing_teams"].item() == 1
    assert with_competition["suggested_bid"].item() >= no_competition["suggested_bid"].item()


def test_waiver_recommendations_limits_optimization_candidates_per_position() -> None:
    rankings = pl.concat(
        [
            _rankings(),
            pl.DataFrame(
                {
                    "player_id": ["fa2", "fa3"],
                    "player_name": ["Free Two", "Free Three"],
                    "position": ["WR", "WR"],
                    "team": ["D", "E"],
                    "opponent": ["E", "D"],
                    "p_active": [1.0, 1.0],
                    "proj_mean": [11.0, 10.0],
                    "floor": [3.0, 3.0],
                    "lower_quartile": [7.0, 6.0],
                    "median": [11.0, 10.0],
                    "upper_quartile": [15.0, 14.0],
                    "ceiling": [19.0, 18.0],
                    "owner_status": ["free_agent", "free_agent"],
                }
            ),
        ],
        how="vertical",
    )

    result = waiver_recommendations(
        rankings,
        {"rb1", "rb2", "wr1", "wr2"},
        _format(),
        current_week=10,
        remaining_budget=80,
        playoff_weight=1.5,
        aggressiveness=1.0,
        candidates_per_position=1,
    )

    assert result["player_id"].to_list() == ["fa1"]


def test_streamers_exclude_rostered_kickers_and_defenses() -> None:
    points = pl.DataFrame(
        {
            "season": [2026, 2026, 2026],
            "week": [2, 2, 2],
            "player_name": ["K One", "K Two", "Ravens D/ST"],
            "position": ["K", "K", "DST"],
            "team": ["A", "B", "BAL"],
            "points": [10.0, 8.0, 7.0],
        }
    )
    players = pl.DataFrame(
        {
            "player_id": ["k1", "k2"],
            "sleeper_id": ["s1", "s2"],
            "full_name": ["K One", "K Two"],
            "normalized_name": ["k one", "k two"],
            "position": ["K", "K"],
        }
    )
    schedule = pl.DataFrame(
        {
            "season": [2026],
            "week": [2],
            "home_team": ["A"],
            "away_team": ["B"],
        }
    )

    result = streaming_recommendations(
        points, players, schedule, {"s1", "BAL"}, season=2026, week=2
    )

    assert result["player_name"].to_list() == ["K Two"]


def test_action_inbox_prioritizes_data_and_meaningful_lineup_changes() -> None:
    lineup = recommended_lineup(
        _rankings(), {"rb1", "rb2", "wr1", "wr2"}, {"rb2", "wr1", "wr2"}, _format()
    )

    result = build_action_inbox(
        lineup,
        _rankings(),
        {"rb2", "wr1", "wr2"},
        pl.DataFrame(),
        fmt=_format(),
        pipeline_status="degraded",
    )

    assert result["category"].to_list()[:2] == ["data", "lineup"]
    assert "Start RB One over WR Two" in result["action"].to_list()


def test_action_inbox_never_starts_a_rb_over_the_only_te() -> None:
    fmt = LeagueFormat(
        n_teams=2,
        starters={"RB": 1, "TE": 1},
        flex_slots={"FLEX": 1, "SUPER_FLEX": 0, "REC_FLEX": 0},
        flex_eligible={"FLEX": ["RB", "WR", "TE"]},
        bench=3,
        ir=0,
        playoff_week_start=15,
        waiver_budget=100,
    )
    ids = ["rb1", "rb2", "rb3", "te1"]
    points = [15.0, 6.0, 12.0, 3.0]
    rankings = pl.DataFrame(
        {
            "player_id": ids,
            "player_name": ids,
            "position": ["RB", "RB", "RB", "TE"],
            "team": ["A"] * 4,
            "opponent": ["B"] * 4,
            "p_active": [1.0] * 4,
            "proj_mean": points,
            "floor": [p - 3 for p in points],
            "median": points,
            "ceiling": [p + 3 for p in points],
        }
    )
    # te1 is the lowest-projected starter, but it is the only TE -- benched
    # rb3 can only replace the RB in FLEX.
    current = {"rb1", "rb2", "te1"}
    lineup = recommended_lineup(rankings, set(ids), current, fmt)

    result = build_action_inbox(
        lineup, rankings, current, pl.DataFrame(), fmt=fmt, pipeline_status="healthy"
    )

    assert result.filter(pl.col("category") == "lineup")["action"].to_list() == [
        "Start rb3 over rb2"
    ]


def test_lineup_decisions_never_pair_an_illegal_cross_position_swap() -> None:
    fmt = LeagueFormat(
        n_teams=2,
        starters={"QB": 1, "RB": 2, "WR": 2},
        flex_slots={"FLEX": 2, "SUPER_FLEX": 0, "REC_FLEX": 0},
        flex_eligible={"FLEX": ["RB", "WR", "TE"]},
        bench=3,
        ir=0,
        playoff_week_start=15,
        waiver_budget=100,
    )
    ids = ["qb", "rb1", "rb2", "rb3", "rb4", "rb5", "wr1", "wr2", "wr3"]
    positions = ["QB", "RB", "RB", "RB", "RB", "RB", "WR", "WR", "WR"]
    points = [20.0, 15.0, 14.0, 10.0, 9.0, 12.0, 16.0, 4.0, 6.0]
    rankings = pl.DataFrame(
        {
            "player_id": ids,
            "player_name": ids,
            "position": positions,
            "team": ["A"] * 9,
            "opponent": ["B"] * 9,
            "p_active": [1.0] * 9,
            "proj_mean": points,
            "floor": [p - 5 for p in points],
            "median": points,
            "ceiling": [p + 5 for p in points],
        }
    )
    # Four RBs start (both FLEX spots are RBs), so benched rb5 can only replace
    # an RB -- starting it over the weakest starter, wr2, would leave one WR.
    current = {"qb", "rb1", "rb2", "rb3", "rb4", "wr1", "wr2"}
    lineup = recommended_lineup(rankings, set(ids), current, fmt)
    result = lineup_decisions(lineup, rankings, current, fmt)

    pairs = {(row["start"], row["sit"]) for row in result.iter_rows(named=True)}
    assert pairs == {("rb5", "rb4"), ("wr3", "wr2")}


def test_action_inbox_and_moves_fill_an_empty_starting_spot() -> None:
    fmt = LeagueFormat(
        n_teams=2,
        starters={"RB": 2, "WR": 1},
        flex_slots={"FLEX": 0, "SUPER_FLEX": 0, "REC_FLEX": 0},
        flex_eligible={},
        bench=3,
        ir=0,
        playoff_week_start=15,
        waiver_budget=100,
    )
    ids = ["rb1", "rb2", "wr1"]
    points = [15.0, 9.0, 12.0]
    rankings = pl.DataFrame(
        {
            "player_id": ids,
            "player_name": ["RB One", "RB Two", "WR One"],
            "position": ["RB", "RB", "WR"],
            "team": ["A"] * 3,
            "opponent": ["B"] * 3,
            "p_active": [1.0] * 3,
            "proj_mean": points,
            "floor": [p - 3 for p in points],
            "median": points,
            "ceiling": [p + 3 for p in points],
        }
    )
    current = {"rb1", "wr1"}  # second RB slot left empty
    lineup = recommended_lineup(rankings, set(ids), current, fmt)

    inbox = build_action_inbox(
        lineup, rankings, current, pl.DataFrame(), fmt=fmt, pipeline_status="healthy"
    )
    moves = lineup_decisions(lineup, rankings, current, fmt)

    assert inbox.filter(pl.col("category") == "lineup")["action"].to_list() == [
        "Start RB Two in your empty lineup spot"
    ]
    move = moves.row(0, named=True)
    assert (move["start"], move["sit"], move["expected_gain"]) == ("RB Two", None, 9.0)
