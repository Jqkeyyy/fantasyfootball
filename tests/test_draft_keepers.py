from pathlib import Path

import polars as pl
import pytest

from ffapp.draft import keepers


def _board() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "player": ["Jonathan Taylor", "Bijan Robinson", "Chris Olave"],
            "position": ["RB", "RB", "WR"],
            "team": ["IND", "ATL", "NO"],
        }
    )


def _users_raw() -> list[dict]:
    return [
        {"user_id": "u1", "display_name": "Maybe17"},
        {"user_id": "u2", "display_name": "gavinreilly"},
    ]


def _rosters_raw() -> list[dict]:
    return [
        {"roster_id": 101, "owner_id": "u1"},
        {"roster_id": 102, "owner_id": "u2"},
    ]


# --- parse_pick_notation --------------------------------------------------------------


def test_parse_pick_notation_odd_round() -> None:
    assert keepers.parse_pick_notation("1.5", n_teams=10) == 5


def test_parse_pick_notation_even_round_is_still_chronological() -> None:
    # Real example from the project owner: JT's real keeper cost "2.6" is the
    # 6th pick made chronologically in round 2, overall pick 16 -- NOT the
    # pick a column-6 slot would structurally make in an even (reversed) round.
    assert keepers.parse_pick_notation("2.6", n_teams=10) == 16


def test_parse_pick_notation_double_digit_round_and_pick() -> None:
    assert keepers.parse_pick_notation("10.10", n_teams=10) == 100


# --- load_keeper_config ----------------------------------------------------------------


def test_load_keeper_config_parses_real_yaml(tmp_path: Path) -> None:
    path = tmp_path / "keepers.yml"
    path.write_text(
        "season: 2026\n"
        "league_slug: rogan-radinator-league\n"
        "keepers:\n"
        "  - owner: Maybe17\n"
        "    player: Jonathan Taylor\n"
        '    pick: "2.6"\n',
        encoding="utf-8",
    )

    config = keepers.load_keeper_config(path)

    assert config.season == 2026
    assert config.league_slug == "rogan-radinator-league"
    assert config.entries == [
        keepers.RawKeeperEntry(owner="Maybe17", player="Jonathan Taylor", pick="2.6")
    ]


def test_keeper_config_path_naming(tmp_path: Path) -> None:
    result = keepers.keeper_config_path(tmp_path, league_slug="rogan-radinator-league", season=2026)

    assert result == tmp_path / "keepers_rogan-radinator-league_2026.yml"


# --- resolve_keeper_assignments ---------------------------------------------------------


def test_resolve_keeper_assignments_matches_owner_and_player() -> None:
    entries = [keepers.RawKeeperEntry(owner="Maybe17", player="Jonathan Taylor", pick="2.6")]

    result = keepers.resolve_keeper_assignments(
        entries, _board(), users_raw=_users_raw(), rosters_raw=_rosters_raw(), n_teams=10
    )

    assert len(result) == 1
    assignment = result[0]
    assert assignment.pick_no == 16
    assert assignment.roster_id == 101
    assert assignment.player_name == "Jonathan Taylor"
    assert assignment.position == "RB"
    assert assignment.team == "IND"
    assert assignment.join_key == "jonathan taylor|RB"


def test_resolve_keeper_assignments_owner_lookup_is_case_insensitive() -> None:
    entries = [keepers.RawKeeperEntry(owner="MAYBE17", player="Jonathan Taylor", pick="2.6")]

    result = keepers.resolve_keeper_assignments(
        entries, _board(), users_raw=_users_raw(), rosters_raw=_rosters_raw(), n_teams=10
    )

    assert result[0].roster_id == 101


def test_resolve_keeper_assignments_fuzzy_matches_a_typo_in_player_name() -> None:
    entries = [keepers.RawKeeperEntry(owner="Maybe17", player="Jonathon Tayler", pick="2.6")]

    result = keepers.resolve_keeper_assignments(
        entries, _board(), users_raw=_users_raw(), rosters_raw=_rosters_raw(), n_teams=10
    )

    assert result[0].player_name == "Jonathan Taylor"


def test_resolve_keeper_assignments_raises_for_unknown_owner() -> None:
    entries = [keepers.RawKeeperEntry(owner="NoSuchUser", player="Jonathan Taylor", pick="2.6")]

    with pytest.raises(keepers.KeeperOwnerNotFoundError):
        keepers.resolve_keeper_assignments(
            entries, _board(), users_raw=_users_raw(), rosters_raw=_rosters_raw(), n_teams=10
        )


def test_resolve_keeper_assignments_raises_for_unmatchable_player() -> None:
    entries = [keepers.RawKeeperEntry(owner="Maybe17", player="Zzyzx Nonexistent", pick="2.6")]

    with pytest.raises(keepers.KeeperPlayerNotFoundError):
        keepers.resolve_keeper_assignments(
            entries, _board(), users_raw=_users_raw(), rosters_raw=_rosters_raw(), n_teams=10
        )


def test_resolve_keeper_assignments_resolves_multiple_entries_independently() -> None:
    entries = [
        keepers.RawKeeperEntry(owner="Maybe17", player="Jonathan Taylor", pick="2.6"),
        keepers.RawKeeperEntry(owner="gavinreilly", player="Bijan Robinson", pick="1.5"),
    ]

    result = keepers.resolve_keeper_assignments(
        entries, _board(), users_raw=_users_raw(), rosters_raw=_rosters_raw(), n_teams=10
    )

    by_player = {a.player_name: a for a in result}
    assert by_player["Jonathan Taylor"].roster_id == 101
    assert by_player["Jonathan Taylor"].pick_no == 16
    assert by_player["Bijan Robinson"].roster_id == 102
    assert by_player["Bijan Robinson"].pick_no == 5
