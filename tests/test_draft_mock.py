import random

import polars as pl
import pytest

from ffapp.draft import mock
from ffapp.draft.keepers import KeeperAssignment
from ffapp.draft.pick_order import TradedPick
from ffapp.league_format import LeagueFormat


def _format(**overrides: object) -> LeagueFormat:
    base = dict(
        n_teams=4,
        starters={"QB": 1, "RB": 2, "WR": 2, "TE": 1},
        flex_slots={"FLEX": 1, "SUPER_FLEX": 0, "REC_FLEX": 0},
        flex_eligible={"FLEX": ["RB", "WR", "TE"]},
        bench=3,
        ir=0,
        playoff_week_start=15,
        waiver_budget=None,
    )
    base.update(overrides)
    return LeagueFormat(**base)  # type: ignore[arg-type]


def _pick(position: str, name: str = "Player") -> dict:
    return {"metadata": {"first_name": name, "last_name": "X", "position": position}}


def _board() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "player": ["Ja'Marr Chase", "Bijan Robinson", "Trey McBride", "Josh Allen"],
            "position": ["WR", "RB", "TE", "QB"],
            "team": ["CIN", "ATL", "ARI", "BUF"],
            "vor": [150.0, 140.0, 90.0, 80.0],
            "tier": [1, 1, 1, 2],
            "overall_rank": [1, 2, 3, 4],
            "adp": [3.0, 2.0, 10.0, 15.0],
        }
    )


def _state(**overrides: object) -> mock.MockDraftState:
    base = dict(
        pool=_board()
        .with_columns((pl.col("player") + "|" + pl.col("position")).alias("join_key"))
        .select(
            "join_key",
            pl.col("player").alias("player_name"),
            "position",
            "team",
            "vor",
            "tier",
            "overall_rank",
            "adp",
        ),
        team_rosters={},
        history=[],
        pick_no=1,
        n_teams=4,
        num_rounds=2,
        roster_by_slot={1: 101, 2: 102, 3: 103, 4: 104},
        traded_picks=[],
        season="2026",
        my_roster_id=101,
        my_slot=1,
        league_format=_format(),
        team_names={101: "Me", 102: "Bot2", 103: "Bot3", 104: "Bot4"},
    )
    base.update(overrides)
    return mock.MockDraftState(**base)  # type: ignore[arg-type]


# --- _pick_no_to_round_slot / current_pick_owner -----------------------------------


def test_pick_no_to_round_slot_snake_order() -> None:
    assert mock._pick_no_to_round_slot(1, 4) == (1, 1)
    assert mock._pick_no_to_round_slot(4, 4) == (1, 4)
    assert mock._pick_no_to_round_slot(5, 4) == (2, 4)  # round 2 reverses
    assert mock._pick_no_to_round_slot(8, 4) == (2, 1)
    assert mock._pick_no_to_round_slot(9, 4) == (3, 1)


def test_current_pick_owner_follows_snake_order_with_no_trades() -> None:
    state = _state(pick_no=5)  # round 2, slot 4 -> roster 104

    assert mock.current_pick_owner(state) == 104


def test_current_pick_owner_respects_traded_picks() -> None:
    trade = TradedPick(season="2026", round=1, roster_id=104, owner_id=101, previous_owner_id=104)
    state = _state(pick_no=4, traded_picks=[trade])  # round 1 slot 4, traded to roster 101

    assert mock.current_pick_owner(state) == 101


# --- _need_multiplier ---------------------------------------------------------------


def test_need_multiplier_boosts_unfilled_starter() -> None:
    assert mock._need_multiplier("QB", [], _format()) == mock.NEED_STARTER_BOOST


def test_need_multiplier_boosts_flex_eligible_position_when_flex_open() -> None:
    picks = [_pick("RB"), _pick("RB")]  # both dedicated RB slots filled
    result = mock._need_multiplier("RB", picks, _format())

    assert result == mock.NEED_STARTER_BOOST  # FLEX (RB-eligible) still open


def test_need_multiplier_suppresses_once_bench_soft_cap_reached() -> None:
    # QB has 1 starter slot, no flex eligibility -- soft cap = 1 + BENCH_SOFT_CAP_EXTRA
    picks = [_pick("QB") for _ in range(1 + mock.BENCH_SOFT_CAP_EXTRA)]

    result = mock._need_multiplier("QB", picks, _format())

    assert result == mock.NEED_SUPPRESS_FACTOR


def test_need_multiplier_neutral_between_starter_and_soft_cap() -> None:
    picks = [_pick("QB")]  # starter filled, but under the bench soft cap

    result = mock._need_multiplier("QB", picks, _format())

    assert result == 1.0


# --- _run_multiplier -----------------------------------------------------------------


def test_run_multiplier_boosts_a_running_position() -> None:
    # 16 picks total: 4 WR overall (25% baseline), but the last 8 are 4 WR (50% recent) -> 2x.
    history = (
        [_pick("RB") for _ in range(8)]
        + [_pick("WR") for _ in range(4)]
        + [_pick("TE") for _ in range(4)]
    )

    assert mock._run_multiplier("WR", history) == mock.RUN_BOOST


def test_run_multiplier_neutral_without_a_run() -> None:
    history = [_pick("RB") if i % 2 == 0 else _pick("WR") for i in range(8)]

    assert mock._run_multiplier("RB", history) == 1.0


# --- pick_weights / bot_pick ---------------------------------------------------------


def test_pick_weights_favors_adp_close_to_current_pick() -> None:
    state = _state(pick_no=2)  # Bijan Robinson (adp=2.0) should be closest

    weights = mock.pick_weights(state, 101)
    rows = state.pool.to_dicts()
    by_player = dict(zip((r["player_name"] for r in rows), weights, strict=True))

    assert by_player["Bijan Robinson"] > by_player["Josh Allen"]


def test_bot_pick_is_deterministic_with_a_seeded_rng() -> None:
    state = _state(pick_no=1)
    rng_a = random.Random(42)
    rng_b = random.Random(42)

    choice_a = mock.bot_pick(state, 101, rng=rng_a)
    choice_b = mock.bot_pick(state, 101, rng=rng_b)

    assert choice_a["join_key"] == choice_b["join_key"]


def test_bot_pick_raises_on_empty_pool() -> None:
    state = _state(pool=_state().pool.clear())

    with pytest.raises(mock.PlayerNotAvailableError):
        mock.bot_pick(state, 101, rng=random.Random(1))


# --- record_pick ---------------------------------------------------------------------


def test_record_pick_moves_player_from_pool_to_roster() -> None:
    state = _state()

    pick = mock.record_pick(state, 101, "Bijan Robinson|RB")

    assert pick["player_name"] == "Bijan Robinson"
    assert "Bijan Robinson" not in state.pool["player_name"].to_list()
    assert state.team_rosters[101][0]["join_key"] == "Bijan Robinson|RB"
    assert state.history[-1]["join_key"] == "Bijan Robinson|RB"
    assert state.pick_no == 2


def test_record_pick_raises_for_unknown_join_key() -> None:
    state = _state()

    with pytest.raises(mock.PlayerNotAvailableError):
        mock.record_pick(state, 101, "Nobody|RB")


def test_record_pick_raises_once_draft_is_complete() -> None:
    state = _state(pick_no=9, num_rounds=2, n_teams=4)  # total_picks = 8

    with pytest.raises(mock.DraftCompleteError):
        mock.record_pick(state, 101, "Bijan Robinson|RB")


# --- run_bot_picks_until_user_turn ----------------------------------------------------


def test_run_bot_picks_until_user_turn_stops_on_my_roster() -> None:
    state = _state(pick_no=1, my_roster_id=104)  # slot 4 -> picks last in round 1
    rng = random.Random(7)

    made = mock.run_bot_picks_until_user_turn(state, rng=rng)

    assert len(made) == 3  # rosters 101, 102, 103 picked; 104 is on the clock
    assert mock.current_pick_owner(state) == 104
    assert state.pick_no == 4


def test_run_bot_picks_until_user_turn_stops_at_draft_end() -> None:
    small_board = (
        _board()
        .head(2)
        .with_columns((pl.col("player") + "|" + pl.col("position")).alias("join_key"))
        .select(
            "join_key",
            pl.col("player").alias("player_name"),
            "position",
            "team",
            "vor",
            "tier",
            "overall_rank",
            "adp",
        )
    )
    state = _state(
        pool=small_board,
        n_teams=2,
        num_rounds=1,
        roster_by_slot={1: 101, 2: 102},
        my_roster_id=999,  # never on the clock
    )
    rng = random.Random(3)

    made = mock.run_bot_picks_until_user_turn(state, rng=rng)

    assert len(made) == 2
    assert state.is_complete()


# --- build_mock_draft_state -----------------------------------------------------------


def _keeper_assignment(
    *, pick_no: int, roster_id: int, player_name: str = "Bijan Robinson", position: str = "RB"
) -> KeeperAssignment:
    return KeeperAssignment(
        pick_no=pick_no,
        roster_id=roster_id,
        join_key=f"{player_name.lower()}|{position}",
        player_name=player_name,
        position=position,
        team="ATL",
    )


def _sleeper_adp() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "player_name": ["Ja'Marr Chase"],
            "position": ["WR"],
            "adp": [1.0],
        }
    )


def test_build_mock_draft_state_seeds_keepers_and_removes_from_pool() -> None:
    rosters_raw = [
        {"roster_id": 101, "owner_id": "u1"},
        {"roster_id": 102, "owner_id": "u2"},
    ]
    state = mock.build_mock_draft_state(
        _board(),
        _sleeper_adp(),
        rosters_raw=rosters_raw,
        draft_order={"u1": 1, "u2": 2},
        num_rounds=2,
        traded_picks=[],
        keeper_assignments=[_keeper_assignment(pick_no=1, roster_id=101)],
        league_format=_format(n_teams=2),
        my_roster_id=101,
        team_names={101: "Me", 102: "Bot"},
        season="2026",
    )

    assert "Bijan Robinson" not in state.pool["player_name"].to_list()
    assert state.team_rosters[101][0]["player_name"] == "Bijan Robinson"
    assert state.team_rosters[101][0]["is_keeper"] is True
    assert state.my_slot == 1


def test_build_mock_draft_state_prefers_sleeper_adp_over_board_adp() -> None:
    rosters_raw = [{"roster_id": 101, "owner_id": "u1"}]
    state = mock.build_mock_draft_state(
        _board(),
        _sleeper_adp(),
        rosters_raw=rosters_raw,
        draft_order={"u1": 1},
        num_rounds=2,
        traded_picks=[],
        keeper_assignments=[],
        league_format=_format(n_teams=1),
        my_roster_id=101,
        team_names={101: "Me"},
        season="2026",
    )

    chase = state.pool.filter(pl.col("player_name") == "Ja'Marr Chase").row(0, named=True)
    assert chase["adp"] == 1.0  # sleeper_adp, not the board's own 3.0

    allen = state.pool.filter(pl.col("player_name") == "Josh Allen").row(0, named=True)
    assert allen["adp"] == 15.0  # no sleeper coverage -- falls back to board's own adp


def _manual_sleeper_adp() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "player_name": ["Ja'Marr Chase"],
            "team": ["CIN"],
            "bye_week": [6],
            "adp": [99.0],
        }
    )


def test_build_mock_draft_state_prefers_manual_sleeper_adp_over_live_api_adp() -> None:
    rosters_raw = [{"roster_id": 101, "owner_id": "u1"}]
    state = mock.build_mock_draft_state(
        _board(),
        _sleeper_adp(),
        rosters_raw=rosters_raw,
        draft_order={"u1": 1},
        num_rounds=2,
        traded_picks=[],
        keeper_assignments=[],
        league_format=_format(n_teams=1),
        my_roster_id=101,
        team_names={101: "Me"},
        season="2026",
        manual_sleeper_adp=_manual_sleeper_adp(),
    )

    chase = state.pool.filter(pl.col("player_name") == "Ja'Marr Chase").row(0, named=True)
    assert chase["adp"] == 99.0  # manual file wins over the live API's own 1.0

    # No manual coverage for Bijan Robinson -- falls back to the live API, same as before.
    bijan = state.pool.filter(pl.col("player_name") == "Bijan Robinson").row(0, named=True)
    assert bijan["adp"] == 2.0


def test_build_mock_draft_state_manual_sleeper_adp_joins_by_name_only_no_position_needed() -> None:
    """The real manual export has no Position column at all -- confirms
    the join still resolves a player correctly without one."""
    rosters_raw = [{"roster_id": 101, "owner_id": "u1"}]
    state = mock.build_mock_draft_state(
        _board(),
        _sleeper_adp(),
        rosters_raw=rosters_raw,
        draft_order={"u1": 1},
        num_rounds=2,
        traded_picks=[],
        keeper_assignments=[],
        league_format=_format(n_teams=1),
        my_roster_id=101,
        team_names={101: "Me"},
        season="2026",
        manual_sleeper_adp=_manual_sleeper_adp(),
    )

    assert state.pool.filter(pl.col("player_name") == "Ja'Marr Chase")["adp"][0] == 99.0


def test_build_mock_draft_state_advances_pick_no_past_a_pick1_keeper() -> None:
    rosters_raw = [
        {"roster_id": 101, "owner_id": "u1"},
        {"roster_id": 102, "owner_id": "u2"},
    ]
    state = mock.build_mock_draft_state(
        _board(),
        _sleeper_adp(),
        rosters_raw=rosters_raw,
        draft_order={"u1": 1, "u2": 2},
        num_rounds=2,
        traded_picks=[],
        keeper_assignments=[_keeper_assignment(pick_no=1, roster_id=101)],
        league_format=_format(n_teams=2),
        my_roster_id=101,
        team_names={101: "Me", 102: "Bot"},
        season="2026",
    )

    assert state.pick_no == 2  # pick 1 was already resolved by the keeper
    assert 1 not in [
        c.pick_no for row in mock.draft_grid(state) for c in row if c.player_name is None
    ]


# --- full draft integration -----------------------------------------------------------


def test_a_full_small_draft_completes_and_fills_every_roster() -> None:
    # 4 teams x 1 round == 4 picks, matching the fixture pool's 4 players exactly.
    state = _state(n_teams=4, num_rounds=1)
    rng = random.Random(123)

    while not state.is_complete():
        roster_id = mock.current_pick_owner(state)
        choice = mock.bot_pick(state, roster_id, rng=rng)
        mock.record_pick(state, roster_id, choice["join_key"])

    assert state.pool.height == 0
    total_drafted = sum(len(picks) for picks in state.team_rosters.values())
    assert total_drafted == 4
    assert {roster_id for roster_id in state.team_rosters} == {101, 102, 103, 104}


# --- draft_grid / my_upcoming_picks -----------------------------------------------------


def test_draft_grid_has_one_row_per_round_and_one_cell_per_slot() -> None:
    state = _state(n_teams=4, num_rounds=2)

    rows = mock.draft_grid(state)

    assert len(rows) == 2
    assert all(len(row) == 4 for row in rows)


def test_draft_grid_pick_numbers_follow_snake_order() -> None:
    state = _state(n_teams=4, num_rounds=2)

    rows = mock.draft_grid(state)

    round1_picks = [cell.pick_no for cell in rows[0]]
    round2_picks = [cell.pick_no for cell in rows[1]]
    assert round1_picks == [1, 2, 3, 4]
    assert round2_picks == [8, 7, 6, 5]  # round 2 reverses


def test_draft_grid_marks_is_mine_for_my_roster_only() -> None:
    state = _state(n_teams=4, num_rounds=1, my_roster_id=101)  # slot 1 -> roster 101

    rows = mock.draft_grid(state)

    mine = [cell for cell in rows[0] if cell.is_mine]
    assert len(mine) == 1
    assert mine[0].original_roster_id == 101


def test_draft_grid_flags_current_pick() -> None:
    state = _state(n_teams=4, num_rounds=2, pick_no=5)

    rows = mock.draft_grid(state)

    current_cells = [cell for row in rows for cell in row if cell.is_current]
    assert len(current_cells) == 1
    assert current_cells[0].pick_no == 5


def test_draft_grid_flags_traded_picks() -> None:
    trade = TradedPick(season="2026", round=1, roster_id=104, owner_id=101, previous_owner_id=104)
    state = _state(n_teams=4, num_rounds=1, traded_picks=[trade])

    rows = mock.draft_grid(state)

    traded_cell = next(cell for cell in rows[0] if cell.slot == 4)
    assert traded_cell.original_roster_id == 104
    assert traded_cell.owner_roster_id == 101
    assert traded_cell.is_traded is True
    assert traded_cell.is_mine is True  # now belongs to 101, the "my" roster in this fixture


def test_draft_grid_reflects_a_made_pick() -> None:
    state = _state(n_teams=4, num_rounds=1)
    mock.record_pick(state, 101, "Bijan Robinson|RB")

    rows = mock.draft_grid(state)

    made_cell = next(cell for cell in rows[0] if cell.slot == 1)
    assert made_cell.player_name == "Bijan Robinson"
    assert made_cell.position == "RB"


def test_my_upcoming_picks_excludes_already_made_picks() -> None:
    state = _state(n_teams=4, num_rounds=2, my_roster_id=101)
    mock.record_pick(state, 101, "Bijan Robinson|RB")  # fills pick 1, my own pick

    rows = mock.draft_grid(state)
    upcoming = mock.my_upcoming_picks(rows)

    assert 1 not in upcoming
    assert upcoming == sorted(upcoming)  # ascending


def test_my_upcoming_picks_includes_future_traded_picks() -> None:
    trade = TradedPick(season="2026", round=1, roster_id=104, owner_id=101, previous_owner_id=104)
    state = _state(n_teams=4, num_rounds=1, my_roster_id=101, traded_picks=[trade])

    rows = mock.draft_grid(state)
    upcoming = mock.my_upcoming_picks(rows)

    assert upcoming == [1, 4]  # my own slot-1 pick plus the traded-in slot-4 pick
