from __future__ import annotations

import polars as pl

from ffapp.league_format import LeagueFormat
from ffapp.sim.lineup import PlayerProjection, optimal_lineup_points
from ffapp.tools.chopped_bids import (
    _fast_lineup_points,
    build_chopped_bid_board,
    chopped_candidates,
    market_price,
    market_win_probability,
    price_tier,
    remaining_faab,
)


def _format() -> LeagueFormat:
    return LeagueFormat(
        n_teams=3,
        starters={"QB": 1},
        flex_slots={"FLEX": 0, "SUPER_FLEX": 0, "REC_FLEX": 0},
        flex_eligible={},
        bench=1,
        ir=0,
        playoff_week_start=0,
        waiver_budget=100,
    )


def test_chopped_candidates_reads_real_transaction_type_and_filters_claimed() -> None:
    transactions = [
        {
            "transaction_id": "new",
            "type": "chopped",
            "status": "complete",
            "created": 200,
            "roster_ids": [3],
            "drops": {"available": 3, "claimed": 3},
            "_week": 2,
        },
        {
            "transaction_id": "old",
            "type": "chopped",
            "status": "complete",
            "created": 100,
            "roster_ids": [2],
            "drops": {"available": 2, "older": 2},
            "_week": 1,
        },
        {
            "transaction_id": "ordinary",
            "type": "free_agent",
            "status": "complete",
            "created": 300,
            "drops": {"not-chopped": 1},
            "_week": 2,
        },
    ]

    result = chopped_candidates(transactions, {"claimed"})

    assert result["sleeper_id"].to_list() == ["available", "older"]
    assert result["chopped_week"].to_list() == [2, 1]


def test_remaining_faab_uses_sleeper_budget_spent() -> None:
    roster = {"settings": {"waiver_budget_used": 37}}

    assert remaining_faab(roster, 100) == 63


def test_fast_bulk_lineup_value_matches_established_optimizer_with_flex() -> None:
    fmt = LeagueFormat(
        n_teams=3,
        starters={"QB": 1, "RB": 1, "WR": 1},
        flex_slots={"FLEX": 1, "SUPER_FLEX": 0, "REC_FLEX": 0},
        flex_eligible={"FLEX": ["RB", "WR", "TE"]},
        bench=2,
        ir=0,
        playoff_week_start=0,
        waiver_budget=100,
    )
    players = [
        PlayerProjection("qb", "QB", 20.0, 20.0, 20.0),
        PlayerProjection("rb1", "RB", 15.0, 15.0, 15.0),
        PlayerProjection("rb2", "RB", 13.0, 13.0, 13.0),
        PlayerProjection("wr1", "WR", 14.0, 14.0, 14.0),
        PlayerProjection("wr2", "WR", 12.0, 12.0, 12.0),
        PlayerProjection("te", "TE", 16.0, 16.0, 16.0),
    ]

    assert _fast_lineup_points(players, fmt) == optimal_lineup_points(players, fmt)


def test_build_chopped_bid_board_estimates_market_and_hard_ceiling() -> None:
    candidates = pl.DataFrame(
        {
            "sleeper_id": ["c1", "c2"],
            "chopped_week": [2, 2],
            "chopped_at_ms": [200, 200],
            "source_roster_id": [3, 3],
            "transaction_id": ["tx", "tx"],
        }
    )
    values = pl.DataFrame(
        {
            "sleeper_id": ["mine", "theirs", "c1", "c2"],
            "player_id": ["mine", "theirs", "c1", "c2"],
            "player_name": ["My QB", "Their QB", "Star QB", "Small QB"],
            "position": ["QB", "QB", "QB", "QB"],
            "team": ["A", "B", "C", "D"],
            "current_week_projection": [10.0, 12.0, 20.0, 11.0],
            "projection_ppg": [10.0, 12.0, 20.0, 11.0],
        }
    )
    rosters = [
        {"roster_id": 1, "players": ["mine"], "settings": {"waiver_budget_used": 0}},
        {"roster_id": 2, "players": ["theirs"], "settings": {"waiver_budget_used": 50}},
        {"roster_id": 3, "players": [], "settings": {"waiver_budget_used": 0}},
    ]

    board = build_chopped_bid_board(
        candidates,
        values,
        rosters,
        1,
        _format(),
        current_week=2,
        total_budget=100,
        reserve_chops=0,
        season_end_week=2,
    )

    star = board.filter(pl.col("sleeper_id") == "c1").row(0, named=True)
    small = board.filter(pl.col("sleeper_id") == "c2").row(0, named=True)
    assert star["lineup_gain_ppg"] == 10.0
    # Star QB: 20 ppg is 7 above QB replacement (tier 3); average FAAB is (100 + 50) / 2.
    assert star["market_bid"] == market_price(3, 75.0, 0.75)
    # The only competitor has $50 left, so $51 wins outright; the old value-share bid
    # from our larger $100 budget would have overpaid at $91.
    assert star["richest_competitor_faab"] == 50
    assert star["lock_bid"] == 51
    assert star["recommended_bid"] == 30  # market + 1 plus a small survival-urgency buffer
    assert star["aggressive_bid"] == 32
    assert star["aggressive_bid"] <= star["lock_bid"]
    assert star["teams_can_outbid"] == 1
    assert star["recommended_bid"] <= star["max_bid"]
    assert star["conservative_bid"] <= star["recommended_bid"] <= star["aggressive_bid"]
    assert (
        star["conservative_win_probability"]
        <= star["recommended_win_probability"]
        <= star["aggressive_win_probability"]
    )
    assert star["survival_urgency"] == 1.0
    assert star["faab_after_recommended"] == 70
    assert small["competing_teams"] == 0
    assert small["recommended_bid"] == 1
    assert small["recommended_win_probability"] == 1.0


def test_board_recommends_pass_when_market_exceeds_value_ceiling() -> None:
    candidates = pl.DataFrame(
        {
            "sleeper_id": ["c1"],
            "chopped_week": [2],
            "chopped_at_ms": [200],
            "source_roster_id": [3],
            "transaction_id": ["tx"],
        }
    )
    values = pl.DataFrame(
        {
            "sleeper_id": ["mine", "theirs", "c1"],
            "player_id": ["mine", "theirs", "c1"],
            "player_name": ["My QB", "Their QB", "Candidate"],
            "position": ["QB", "QB", "QB"],
            "team": ["A", "B", "C"],
            "current_week_projection": [19.0, 5.0, 20.0],
            "projection_ppg": [19.0, 5.0, 20.0],
        }
    )
    rosters = [
        {"roster_id": 1, "players": ["mine"], "settings": {"waiver_budget_used": 90}},
        {"roster_id": 2, "players": ["theirs"], "settings": {"waiver_budget_used": 0}},
        {"roster_id": 3, "players": []},
    ]

    board = build_chopped_bid_board(
        candidates,
        values,
        rosters,
        1,
        _format(),
        current_week=2,
        total_budget=100,
        reserve_chops=0,
        season_end_week=2,
    )

    row = board.row(0, named=True)
    assert row["max_bid"] == 10
    assert row["market_bid"] > row["max_bid"]
    assert row["recommended_bid"] == 0
    assert row["recommendation"] == "Pass above max"
    assert row["richest_competitor_faab"] == 100
    assert row["teams_can_outbid"] == 1
    assert row["survival_urgency"] == 0.0
    assert row["faab_after_recommended"] == 10


def test_elite_backup_gets_smaller_depth_insurance_value() -> None:
    candidates = pl.DataFrame(
        {
            "sleeper_id": ["backup"],
            "chopped_week": [2],
            "chopped_at_ms": [200],
            "source_roster_id": [3],
            "transaction_id": ["tx"],
        }
    )
    values = pl.DataFrame(
        {
            "sleeper_id": ["starter", "bench", "other", "backup"],
            "player_id": ["starter", "bench", "other", "backup"],
            "player_name": ["Starter", "Weak Bench", "Other QB", "Elite Backup"],
            "position": ["QB", "WR", "QB", "QB"],
            "team": ["A", "B", "C", "D"],
            "current_week_projection": [20.0, 5.0, 10.0, 18.0],
            "projection_ppg": [20.0, 5.0, 10.0, 18.0],
        }
    )
    rosters = [
        {"roster_id": 1, "players": ["starter", "bench"], "settings": {}},
        {"roster_id": 2, "players": ["other"], "settings": {}},
        {"roster_id": 3, "players": [], "settings": {}},
    ]

    board = build_chopped_bid_board(
        candidates,
        values,
        rosters,
        1,
        _format(),
        current_week=2,
        total_budget=100,
        reserve_chops=0,
        season_end_week=2,
    )

    row = board.row(0, named=True)
    assert row["lineup_gain_ppg"] == 0.0
    assert row["depth_gain_ppg"] == 1.95
    assert row["value_basis"] == "depth upgrade"
    assert row["drop_player"] == "Weak Bench"


def test_learned_opponent_aggression_changes_opponent_estimate() -> None:
    candidates = pl.DataFrame(
        {
            "sleeper_id": ["candidate"],
            "chopped_week": [2],
            "chopped_at_ms": [200],
            "source_roster_id": [3],
            "transaction_id": ["tx"],
        }
    )
    values = pl.DataFrame(
        {
            "sleeper_id": ["mine", "theirs", "candidate"],
            "player_id": ["mine", "theirs", "candidate"],
            "player_name": ["Mine", "Theirs", "Candidate"],
            "position": ["QB", "QB", "QB"],
            "team": ["A", "B", "C"],
            "current_week_projection": [10.0, 10.0, 20.0],
            "projection_ppg": [10.0, 10.0, 20.0],
        }
    )
    rosters = [
        {"roster_id": 1, "players": ["mine"], "settings": {}},
        {"roster_id": 2, "players": ["theirs"], "settings": {}},
        {"roster_id": 3, "players": [], "settings": {}},
    ]
    neutral = build_chopped_bid_board(
        candidates,
        values,
        rosters,
        1,
        _format(),
        current_week=2,
        total_budget=100,
        reserve_chops=3,
        season_end_week=2,
    )
    aggressive = build_chopped_bid_board(
        candidates,
        values,
        rosters,
        1,
        _format(),
        current_week=2,
        total_budget=100,
        reserve_chops=3,
        season_end_week=2,
        opponent_aggression={2: 1.5},
    )
    assert (
        aggressive["highest_opponent_estimate"].item() > neutral["highest_opponent_estimate"].item()
    )


def test_price_tier_uses_points_above_positional_replacement() -> None:
    assert price_tier("QB", 15.0) == 1  # a solid QB is barely above QB replacement
    assert price_tier("RB", 15.0) == 4  # the same points at RB are elite
    assert price_tier("WR", 5.0) == 0


def test_market_price_scales_with_league_faab_and_quantile() -> None:
    assert market_price(4, 500.0, 0.5) == 360
    assert market_price(4, 250.0, 0.5) == 180
    assert market_price(0, 500.0, 0.5) == 0  # low-tier players usually go unclaimed
    assert market_price(2, 500.0, 0.6) < market_price(2, 500.0, 0.9)


def test_market_win_probability_rises_with_bid() -> None:
    assert market_win_probability(4, 500.0, 0) == 0.0
    low = market_win_probability(4, 500.0, 100)
    high = market_win_probability(4, 500.0, 400)
    assert 0.0 < low < high < 1.0
    assert market_win_probability(0, 500.0, 1) >= 0.7  # most tier-0 players go unclaimed
    assert market_win_probability(4, 500.0, 500) == 1.0
