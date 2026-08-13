"""Task 2.6's own literal acceptance bar (TASKS.md, SPEC.md §14.4): "value
is computed relative to your roster (verify: a high-projection player at a
position where you are already deep ranks low)", plus FAAB guidance
calibrated against real bidding history. Small hand-verifiable fixtures for
the calculation layer; `test_build_waiver_board_ranks_by_roster_relative_value_not_raw_projection`
is the acceptance bar's own scenario, built by hand and verified by
inspection -- not just "it runs".
"""

from __future__ import annotations

import polars as pl
import pytest

from ffapp.league_format import LeagueFormat
from ffapp.sim.lineup import PlayerProjection
from ffapp.tools.waivers import (
    build_waiver_board,
    current_form,
    free_agent_pool,
    join_trending,
    ros_value,
    rostered_sleeper_ids,
    suggested_bid,
    value_added,
    weeks_remaining,
)


def _fmt(**overrides: object) -> LeagueFormat:
    base = dict(
        n_teams=10,
        starters={"RB": 1},
        flex_slots={"FLEX": 1, "SUPER_FLEX": 0, "REC_FLEX": 0},
        flex_eligible={"FLEX": ["RB", "WR"]},
        bench=6,
        ir=0,
        playoff_week_start=15,
        waiver_budget=100,
    )
    base.update(overrides)
    return LeagueFormat(**base)  # type: ignore[arg-type]


def _player(player_id: str, position: str, mean: float) -> PlayerProjection:
    return PlayerProjection(
        player_id=player_id, position=position, mean=mean, median=mean, ceiling=mean
    )


class TestRosteredSleeperIds:
    def test_collects_players_across_every_roster(self) -> None:
        rosters = [
            {"roster_id": 1, "players": ["100", "200"]},
            {"roster_id": 2, "players": ["300"]},
        ]
        assert rostered_sleeper_ids(rosters) == {"100", "200", "300"}

    def test_a_roster_with_no_players_key_does_not_crash(self) -> None:
        rosters = [{"roster_id": 1, "players": None}, {"roster_id": 2}]
        assert rostered_sleeper_ids(rosters) == set()


class TestFreeAgentPool:
    def test_excludes_rostered_players_and_scopes_to_eligible_positions(self) -> None:
        players_dim = pl.DataFrame(
            {
                "sleeper_id": ["1", "2", "3", "4"],
                "player_id": ["p1", "p2", "p3", "p4"],
                "position": ["RB", "RB", "WR", "K"],
                "team": ["AAA", "BBB", "CCC", "DDD"],
                "active": [True, True, True, True],
            }
        )
        pool = free_agent_pool(players_dim, rostered_ids={"1"}, eligible_positions={"RB", "WR"})

        assert set(pool["sleeper_id"]) == {"2", "3"}


class TestCurrentForm:
    def test_computes_the_same_ewm_span_4_window_b2_uses_unshifted(self) -> None:
        features = pl.DataFrame(
            {
                "player_id": ["a", "a", "a"],
                "position": ["RB", "RB", "RB"],
                "season": [2025, 2025, 2025],
                "week": [1, 2, 3],
                "target": [10.0, 20.0, 0.0],
            }
        )

        form = current_form(features)

        # polars ewm_mean(span=4) default is adjust=True (bias-corrected weights,
        # matching add_b2_ewm_4's own real behavior, not simple recursive smoothing):
        # alpha=0.4; y2 = (0.36*10 + 0.6*20 + 1*0) / (0.36+0.6+1) = 15.6/1.96
        row = form.filter(pl.col("player_id") == "a").to_dicts()[0]
        assert row["projection_ppg"] == pytest.approx(15.6 / 1.96)
        assert row["week"] == 3

    def test_one_row_per_player_using_their_most_recent_real_week(self) -> None:
        features = pl.DataFrame(
            {
                "player_id": ["a", "a", "b"],
                "position": ["RB", "RB", "WR"],
                "season": [2025, 2025, 2025],
                "week": [1, 2, 1],
                "target": [5.0, 7.0, 3.0],
            }
        )

        form = current_form(features)

        assert form.height == 2
        assert set(form["player_id"]) == {"a", "b"}


class TestValueAdded:
    def test_a_candidate_that_upgrades_a_flex_slot_shows_positive_value_and_names_the_drop(
        self,
    ) -> None:
        fmt = _fmt()
        my_roster = [_player("rb1", "RB", 10.0), _player("wr1", "WR", 8.0)]
        candidate = _player("new_rb", "RB", 15.0)

        added, drop = value_added(my_roster, candidate, fmt)

        assert added == pytest.approx(7.0)  # (10+15) - (10+8)
        assert drop == "wr1"

    def test_a_candidate_that_cannot_crack_the_lineup_adds_nothing_and_names_no_drop(self) -> None:
        fmt = _fmt()
        my_roster = [_player("rb1", "RB", 20.0), _player("wr1", "WR", 18.0)]
        candidate = _player("weak_rb", "RB", 1.0)

        added, drop = value_added(my_roster, candidate, fmt)

        assert added == pytest.approx(0.0)
        assert drop is None


class TestWeeksRemainingAndRosValue:
    def test_weeks_remaining_is_inclusive_of_both_ends(self) -> None:
        assert weeks_remaining(current_week=15, season_end_week=17) == [15, 16, 17]

    def test_ros_value_weights_playoff_weeks_higher(self) -> None:
        # weeks 13-14 regular season (weight 1.0), 15-16 playoffs (weight 2.0)
        value = ros_value(
            value_added_per_week=2.0,
            weeks=[13, 14, 15, 16],
            playoff_week_start=15,
            playoff_weight=2.0,
        )
        assert value == pytest.approx(2.0 * 1.0 + 2.0 * 1.0 + 2.0 * 2.0 + 2.0 * 2.0)


class TestSuggestedBid:
    def test_clamps_to_the_floor_for_a_tiny_but_real_share_of_value(self) -> None:
        bid = suggested_bid(candidate_ros_value=0.01, total_ros_value=1000.0, remaining_budget=50)
        assert bid == 1

    def test_clamps_to_remaining_budget_when_the_raw_bid_would_overshoot(self) -> None:
        # raw = 2.0 * (900/1000) * 50 = 90, clamped down to the real 50 available
        bid = suggested_bid(
            candidate_ros_value=900.0,
            total_ros_value=1000.0,
            remaining_budget=50,
            aggressiveness=2.0,
        )
        assert bid == 50

    def test_a_candidate_with_no_real_value_gets_zero_not_the_floor(self) -> None:
        bid = suggested_bid(candidate_ros_value=0.0, total_ros_value=1000.0, remaining_budget=50)
        assert bid == 0

    def test_no_remaining_budget_bids_zero_rather_than_dividing_by_zero(self) -> None:
        bid = suggested_bid(candidate_ros_value=10.0, total_ros_value=10.0, remaining_budget=0)
        assert bid == 0

    def test_aggressiveness_scales_the_raw_bid_before_clamping(self) -> None:
        low = suggested_bid(
            candidate_ros_value=50.0,
            total_ros_value=100.0,
            remaining_budget=100,
            aggressiveness=0.5,
        )
        high = suggested_bid(
            candidate_ros_value=50.0,
            total_ros_value=100.0,
            remaining_budget=100,
            aggressiveness=1.5,
        )
        assert low < high


class TestJoinTrending:
    def test_ranks_by_position_in_the_trending_list_and_nulls_the_rest(self) -> None:
        board = pl.DataFrame({"sleeper_id": ["1", "2", "3"]})
        trending = join_trending(board, ["2", "1"])

        rows = {r["sleeper_id"]: r["trend_rank"] for r in trending.to_dicts()}
        assert rows["2"] == 1
        assert rows["1"] == 2
        assert rows["3"] is None

    def test_an_empty_trending_list_leaves_every_rank_null(self) -> None:
        board = pl.DataFrame({"sleeper_id": ["1"]})
        trending = join_trending(board, [])

        assert trending.to_dicts()[0]["trend_rank"] is None


class TestBuildWaiverBoard:
    def test_ranks_by_roster_relative_value_not_raw_projection(self) -> None:
        """TASKS.md 2.6's own literal acceptance bar: a high-projection
        player at a position I'm already deep at ranks low; a
        lower-projection player at a thin position ranks high."""
        fmt = _fmt(
            starters={"RB": 2, "TE": 1},
            flex_slots={"FLEX": 0, "SUPER_FLEX": 0, "REC_FLEX": 0},
            flex_eligible={},
        )
        my_roster = [
            _player("rb1", "RB", 20.0),
            _player("rb2", "RB", 18.0),
            _player("te1", "TE", 5.0),  # weak starter -- real need
        ]
        free_agents = pl.DataFrame(
            {
                "sleeper_id": ["fa_rb", "fa_te"],
                "player_id": ["fa_rb", "fa_te"],
                "position": ["RB", "TE"],
            }
        )
        projections = {"fa_rb": 15.0, "fa_te": 9.0}  # fa_rb has the higher raw projection

        board = build_waiver_board(
            free_agents,
            projections,
            my_roster,
            fmt,
            current_week=15,
            season_end_week=17,
            remaining_budget=50,
        )

        rows = {r["player_id"]: r for r in board.to_dicts()}
        assert rows["fa_rb"]["value_added_per_week"] == pytest.approx(0.0)
        assert rows["fa_te"]["value_added_per_week"] > 0
        # sorted descending by ros_value -- the thin-position player ranks first
        assert board["player_id"].to_list()[0] == "fa_te"
        assert rows["fa_rb"]["suggested_bid"] == 0
        assert rows["fa_te"]["suggested_bid"] >= 1

    def test_output_has_every_spec_column(self) -> None:
        fmt = _fmt()
        my_roster = [_player("rb1", "RB", 10.0)]
        free_agents = pl.DataFrame({"sleeper_id": ["1"], "player_id": ["p1"], "position": ["RB"]})
        board = build_waiver_board(
            free_agents,
            {"p1": 12.0},
            my_roster,
            fmt,
            current_week=15,
            season_end_week=17,
            remaining_budget=50,
            trending_ids=["1"],
        )
        expected = {
            "player_id",
            "position",
            "value_added_per_week",
            "ros_value",
            "suggested_bid",
            "trend_rank",
            "drop_candidate",
        }
        assert expected.issubset(set(board.columns))
        assert board.to_dicts()[0]["trend_rank"] == 1

    def test_a_free_agent_with_no_projection_is_silently_excluded_not_crashed_on(self) -> None:
        fmt = _fmt()
        my_roster = [_player("rb1", "RB", 10.0)]
        free_agents = pl.DataFrame(
            {"sleeper_id": ["1", "2"], "player_id": ["p1", "p2"], "position": ["RB", "RB"]}
        )
        board = build_waiver_board(
            free_agents,
            {"p1": 12.0},  # p2 has no projection -- e.g. never played a real game
            my_roster,
            fmt,
            current_week=15,
            season_end_week=17,
            remaining_budget=50,
        )
        assert board["player_id"].to_list() == ["p1"]
