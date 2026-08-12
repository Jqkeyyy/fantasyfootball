from datetime import date

import polars as pl
import pytest

from ffapp.projections import games_played

# --- p_available_baseline -------------------------------------------------


def test_p_available_baseline_returns_position_base_rate_when_age_is_none() -> None:
    assert games_played.p_available_baseline("RB", None) == pytest.approx(
        games_played.POSITION_BASE_AVAILABILITY["RB"]
    )


def test_p_available_baseline_returns_position_base_rate_below_the_aging_cliff() -> None:
    # RB cliff is 27 -- 24 is comfortably below it, so no age decay applies yet.
    assert games_played.p_available_baseline("RB", 24.0) == pytest.approx(
        games_played.POSITION_BASE_AVAILABILITY["RB"]
    )


def test_p_available_baseline_decays_beyond_the_positional_aging_cliff() -> None:
    threshold, decay = games_played.AGE_CLIFF["RB"]
    base = games_played.POSITION_BASE_AVAILABILITY["RB"]
    age = threshold + 3.0

    result = games_played.p_available_baseline("RB", age)

    assert result == pytest.approx(base - decay * 3.0)
    assert result < base


def test_p_available_baseline_floors_at_minimum_availability_for_extreme_age() -> None:
    result = games_played.p_available_baseline("RB", 90.0)

    assert result == pytest.approx(games_played.MIN_AVAILABILITY)


def test_p_available_baseline_dst_ignores_age_entirely() -> None:
    # DST is a team entity, not a person -- no aging cliff applies.
    assert games_played.p_available_baseline("DST", 40.0) == pytest.approx(1.0)
    assert games_played.p_available_baseline("DST", None) == pytest.approx(1.0)


def test_p_available_baseline_raises_on_unknown_position() -> None:
    with pytest.raises(ValueError, match="LB"):
        games_played.p_available_baseline("LB", 25.0)


# --- expected_games --------------------------------------------------------


def test_expected_games_is_17_times_p_available_baseline() -> None:
    assert games_played.expected_games("QB", None) == pytest.approx(
        17 * games_played.POSITION_BASE_AVAILABILITY["QB"]
    )


def test_expected_games_is_below_17_for_an_aging_rb_and_near_17_for_a_young_qb() -> None:
    """TASKS.md 0.8's literal acceptance bar: expected_games must be visibly
    below 17 for the positions/age bands where it should be."""
    old_rb_games = games_played.expected_games("RB", 32.0)
    young_qb_games = games_played.expected_games("QB", 26.0)

    assert old_rb_games < 14.0
    assert young_qb_games > 15.5
    assert young_qb_games < 17.0


# --- player_ages_from_players_dim -------------------------------------------


def test_player_ages_from_players_dim_computes_fractional_age_from_birth_date() -> None:
    players_dim = pl.DataFrame(
        {
            "full_name": ["Test Player"],
            "position": ["RB"],
            "birth_date": ["2000-09-01"],
        }
    )

    ages = games_played.player_ages_from_players_dim(players_dim, as_of=date(2026, 9, 1))

    assert ages["join_key"][0] == "test player|RB"
    assert ages["age"][0] == pytest.approx(26.0, abs=0.01)


def test_player_ages_from_players_dim_is_null_for_missing_birth_date() -> None:
    players_dim = pl.DataFrame(
        {
            "full_name": ["No Birthdate"],
            "position": ["WR"],
            "birth_date": [None],
        }
    )

    ages = games_played.player_ages_from_players_dim(players_dim, as_of=date(2026, 9, 1))

    assert ages["age"][0] is None


# --- add_games_played_adjustment --------------------------------------------


def test_add_games_played_adjustment_computes_ppg_expected_games_and_adjusted_points() -> None:
    projections = pl.DataFrame(
        {
            "join_key": ["young qb|QB"],
            "position": ["QB"],
            "proj_points": [340.0],
        }
    )
    ages = pl.DataFrame({"join_key": ["young qb|QB"], "age": [26.0]})

    result = games_played.add_games_played_adjustment(projections, ages)

    expected_ppg = 340.0 / 17
    expected_games = games_played.expected_games("QB", 26.0)
    row = result.row(0, named=True)
    assert row["proj_ppg"] == pytest.approx(expected_ppg)
    assert row["expected_games"] == pytest.approx(expected_games)
    assert row["proj_points_adj"] == pytest.approx(expected_ppg * expected_games)


def test_add_games_played_adjustment_falls_back_to_position_only_when_age_unmatched() -> None:
    """A player with no ages-table match (e.g. not resolved to the crosswalk)
    still gets expected_games populated via the position-only baseline, not
    a null/dropped row (CLAUDE.md rule 4)."""
    projections = pl.DataFrame(
        {
            "join_key": ["mystery rb|RB"],
            "position": ["RB"],
            "proj_points": [204.0],
        }
    )
    ages = pl.DataFrame(
        {"join_key": [], "age": []}, schema={"join_key": pl.Utf8, "age": pl.Float64}
    )

    result = games_played.add_games_played_adjustment(projections, ages)

    row = result.row(0, named=True)
    assert row["expected_games"] == pytest.approx(games_played.expected_games("RB", None))
    assert row["expected_games"] is not None
