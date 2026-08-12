import pytest

from ffapp.draft import pick_order


def _rosters(*owner_by_roster_id: tuple[int, str]) -> list[dict]:
    return [{"roster_id": rid, "owner_id": owner} for rid, owner in owner_by_roster_id]


# --- resolve_my_roster_id -------------------------------------------------------


def test_resolve_my_roster_id_finds_the_matching_roster() -> None:
    rosters = _rosters((5, "user_a"), (7, "user_b"))

    assert pick_order.resolve_my_roster_id("user_b", rosters) == 7


def test_resolve_my_roster_id_raises_if_no_roster_matches() -> None:
    rosters = _rosters((5, "user_a"))

    with pytest.raises(ValueError, match="user_z"):
        pick_order.resolve_my_roster_id("user_z", rosters)


# --- snake_pick_number --------------------------------------------------------


def test_snake_pick_number_odd_round_goes_forward() -> None:
    assert pick_order.snake_pick_number(1, 1, n_teams=10) == 1
    assert pick_order.snake_pick_number(1, 3, n_teams=10) == 3
    assert pick_order.snake_pick_number(1, 10, n_teams=10) == 10


def test_snake_pick_number_even_round_reverses() -> None:
    assert pick_order.snake_pick_number(2, 10, n_teams=10) == 11
    assert pick_order.snake_pick_number(2, 1, n_teams=10) == 20
    assert pick_order.snake_pick_number(2, 3, n_teams=10) == 18  # my slot 3


def test_snake_pick_number_third_round_matches_first() -> None:
    assert pick_order.snake_pick_number(3, 3, n_teams=10) == 23


# --- roster_id_by_slot ---------------------------------------------------------


def test_roster_id_by_slot_joins_draft_order_through_owner_id() -> None:
    draft_order = {"user_a": 1, "user_b": 2}
    rosters = _rosters((5, "user_a"), (9, "user_b"))

    result = pick_order.roster_id_by_slot(draft_order, rosters)

    assert result == {1: 5, 2: 9}


def test_roster_id_by_slot_skips_roster_with_no_owner() -> None:
    draft_order = {"user_a": 1}
    rosters = [{"roster_id": 5, "owner_id": None}, {"roster_id": 6, "owner_id": "user_a"}]

    result = pick_order.roster_id_by_slot(draft_order, rosters)

    assert result == {1: 6}


# --- pick_owner: no trade -------------------------------------------------------


def test_pick_owner_defaults_to_slot_original_owner_when_never_traded() -> None:
    roster_by_slot = {1: 5, 2: 9}

    owner = pick_order.pick_owner(3, 1, roster_by_slot, [], season="2026")

    assert owner == 5


# --- pick_owner: traded, non-chained (real Sleeper shape) ----------------------


def test_pick_owner_follows_a_single_trade() -> None:
    roster_by_slot = {1: 5, 2: 9}
    traded = pick_order.parse_traded_picks(
        [{"season": "2026", "round": 3, "roster_id": 5, "owner_id": 9, "previous_owner_id": 5}]
    )

    owner = pick_order.pick_owner(3, 1, roster_by_slot, traded, season="2026")

    assert owner == 9


def test_pick_owner_ignores_trades_in_other_rounds_or_seasons() -> None:
    roster_by_slot = {1: 5}
    traded = pick_order.parse_traded_picks(
        [
            {"season": "2026", "round": 4, "roster_id": 5, "owner_id": 9, "previous_owner_id": 5},
            {"season": "2025", "round": 3, "roster_id": 5, "owner_id": 9, "previous_owner_id": 5},
        ]
    )

    owner = pick_order.pick_owner(3, 1, roster_by_slot, traded, season="2026")

    assert owner == 5


def test_pick_owner_handles_a_pick_traded_back_to_its_original_owner() -> None:
    """Real edge case seen live in the primary league's own 2026 traded_picks:
    a round-16 record with roster_id == owner_id == 7 -- the pick left and came
    back. Must resolve to the original owner, not error or double-count."""
    roster_by_slot = {1: 7}
    traded = pick_order.parse_traded_picks(
        [{"season": "2026", "round": 16, "roster_id": 7, "owner_id": 7, "previous_owner_id": 3}]
    )

    owner = pick_order.pick_owner(16, 1, roster_by_slot, traded, season="2026")

    assert owner == 7


# --- my_pick_numbers: end-to-end ------------------------------------------------


def test_my_pick_numbers_plain_redraft_with_no_trades() -> None:
    """4-team, 3-round league, no trades: my slot-2 picks are 2, 7 (snake), 10."""
    draft_order = {"me": 2, "a": 1, "b": 3, "c": 4}
    rosters = _rosters((100, "me"), (101, "a"), (102, "b"), (103, "c"))

    picks = pick_order.my_pick_numbers(
        100,
        draft_order=draft_order,
        rosters=rosters,
        traded_picks=[],
        n_teams=4,
        num_rounds=3,
        season="2026",
    )

    assert picks == [2, 7, 10]


def test_my_pick_numbers_accounts_for_a_pick_given_away_and_one_acquired() -> None:
    """Same 4-team/3-round league. I trade away my round-2 pick (slot 2, pick
    7) to roster 101, and acquire roster 103's round-3 pick (slot 4, pick 12)
    in return. My final picks are 2 (round 1, untouched), 10 (round 3, still
    mine), 12 (acquired) -- NOT 7."""
    draft_order = {"me": 2, "a": 1, "b": 3, "c": 4}
    rosters = _rosters((100, "me"), (101, "a"), (102, "b"), (103, "c"))
    traded_picks = pick_order.parse_traded_picks(
        [
            {
                "season": "2026",
                "round": 2,
                "roster_id": 100,
                "owner_id": 101,
                "previous_owner_id": 100,
            },
            {
                "season": "2026",
                "round": 3,
                "roster_id": 103,
                "owner_id": 100,
                "previous_owner_id": 103,
            },
        ]
    )

    picks = pick_order.my_pick_numbers(
        100,
        draft_order=draft_order,
        rosters=rosters,
        traded_picks=traded_picks,
        n_teams=4,
        num_rounds=3,
        season="2026",
    )

    assert picks == [2, 10, 12]
    assert 7 not in picks
