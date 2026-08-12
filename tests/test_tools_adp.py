import polars as pl
import pytest

from ffapp.tools import adp

# --- keeper_sleeper_ids ---------------------------------------------------------


def test_keeper_sleeper_ids_collects_across_rosters() -> None:
    rosters = [
        {"roster_id": 1, "keepers": ["8144"]},
        {"roster_id": 2, "keepers": ["7543"]},
        {"roster_id": 3, "keepers": None},
    ]

    result = adp.keeper_sleeper_ids(rosters)

    assert result == {"8144", "7543"}


def test_keeper_sleeper_ids_handles_missing_key_and_empty_list() -> None:
    rosters = [{"roster_id": 1}, {"roster_id": 2, "keepers": []}]

    result = adp.keeper_sleeper_ids(rosters)

    assert result == set()


# --- keeper_join_keys ------------------------------------------------------------


def _players_dim() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "sleeper_id": ["8144", "7543", "9999"],
            "normalized_name": ["ja marr chase", "justin jefferson", "some other player"],
            "position": ["WR", "WR", "RB"],
        }
    )


def test_keeper_join_keys_resolves_sleeper_ids_to_normalized_join_keys() -> None:
    result = adp.keeper_join_keys({"8144"}, _players_dim())

    assert result == {"ja marr chase|WR"}


def test_keeper_join_keys_ignores_ids_not_in_players_dim() -> None:
    result = adp.keeper_join_keys({"unknown_id"}, _players_dim())

    assert result == set()


# --- exclude_keepers ---------------------------------------------------------------


def test_exclude_keepers_drops_only_matching_rows() -> None:
    df = pl.DataFrame(
        {
            "join_key": ["ja marr chase|WR", "justin jefferson|WR", "some other player|RB"],
            "player_name": ["Ja'Marr Chase", "Justin Jefferson", "Some Other Player"],
        }
    )

    result = adp.exclude_keepers(df, {"ja marr chase|WR"})

    assert result["join_key"].to_list() == ["justin jefferson|WR", "some other player|RB"]


def test_exclude_keepers_with_empty_set_is_a_no_op() -> None:
    df = pl.DataFrame({"join_key": ["a|WR"], "player_name": ["A"]})

    result = adp.exclude_keepers(df, set())

    assert result.height == 1


# --- join_adp -------------------------------------------------------------------


def test_join_adp_adds_columns_and_keeps_unmatched_rows() -> None:
    projections = pl.DataFrame(
        {"join_key": ["covered|RB", "uncovered|RB"], "player_name": ["Covered", "Uncovered"]}
    )
    adp_df = pl.DataFrame(
        {
            "join_key": ["covered|RB"],
            "adp": [12.5],
            "adp_sd": [2.0],
            "adp_high": [8],
            "adp_low": [18],
            "times_drafted": [500],
            "bye_week": [7],
        }
    )

    result = adp.join_adp(projections, adp_df)

    assert result.height == 2  # CLAUDE.md rule 4: never silently drop the unmatched row
    covered = result.filter(pl.col("join_key") == "covered|RB").row(0, named=True)
    assert covered["adp"] == 12.5
    assert covered["adp_sd"] == 2.0
    assert covered["bye_week"] == 7
    uncovered = result.filter(pl.col("join_key") == "uncovered|RB").row(0, named=True)
    assert uncovered["adp"] is None


# --- p_available ------------------------------------------------------------------


def test_p_available_at_the_mean_is_one_half() -> None:
    assert adp.p_available(20, 20.0, 1.0, adp_sd_fallback=8.0) == pytest.approx(0.5)


def test_p_available_is_near_one_when_pick_is_far_before_adp() -> None:
    assert adp.p_available(1, 100.0, 5.0, adp_sd_fallback=8.0) == pytest.approx(1.0, abs=1e-6)


def test_p_available_is_near_zero_when_pick_is_far_after_adp() -> None:
    assert adp.p_available(200, 5.0, 1.0, adp_sd_fallback=8.0) == pytest.approx(0.0, abs=1e-6)


def test_p_available_with_no_adp_data_defaults_to_certain_availability() -> None:
    assert adp.p_available(1, None, None, adp_sd_fallback=8.0) == 1.0


def test_p_available_uses_fallback_sd_when_source_gives_none() -> None:
    with_fallback = adp.p_available(28, 20.0, None, adp_sd_fallback=8.0)
    explicit_same_sd = adp.p_available(28, 20.0, 8.0, adp_sd_fallback=999.0)

    assert with_fallback == pytest.approx(explicit_same_sd)


# --- add_survival_probabilities -----------------------------------------------------


def test_add_survival_probabilities_uses_each_players_own_adp() -> None:
    df = pl.DataFrame(
        {
            "player_name": ["Early ADP", "Late ADP"],
            "adp": [5.0, 100.0],
            "adp_sd": [2.0, 2.0],
        }
    )

    result = adp.add_survival_probabilities(
        df, next_pick=20, after_next_pick=40, adp_sd_fallback=8.0
    )

    early = result.filter(pl.col("player_name") == "Early ADP").row(0, named=True)
    late = result.filter(pl.col("player_name") == "Late ADP").row(0, named=True)
    assert early["p_avail_next"] == pytest.approx(0.0, abs=1e-4)  # long gone by pick 20
    assert late["p_avail_next"] == pytest.approx(1.0, abs=1e-4)  # nowhere near drafted yet
    assert late["p_avail_after_next"] == pytest.approx(1.0, abs=1e-4)


# --- expected_best_available_vor / add_opportunity_cost --------------------------


def _two_player_position(vor_a: float, vor_b: float, *, adp_mean: float, sd: float) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "position": ["RB", "RB"],
            "player_name": ["A", "B"],
            "vor": [vor_a, vor_b],
            "adp": [adp_mean, adp_mean],
            "adp_sd": [sd, sd],
        }
    )


def test_expected_best_available_vor_hand_computed() -> None:
    """Both players have p_avail=0.5 exactly at the pick (adp == pick, sd=1,
    Phi(0)=0.5). Ranked by VOR descending: A (100) then B (60).
    E = 100*0.5*1 + 60*0.5*(1-0.5) = 50 + 15 = 65."""
    df = _two_player_position(100.0, 60.0, adp_mean=20.0, sd=1.0)

    result = adp.expected_best_available_vor(df, pick=20, adp_sd_fallback=8.0)

    assert result == pytest.approx(65.0)


def test_add_opportunity_cost_broadcasts_per_position() -> None:
    df = _two_player_position(100.0, 60.0, adp_mean=20.0, sd=1.0)

    result = adp.add_opportunity_cost(df, fallback_pick=20, adp_sd_fallback=8.0)

    a = result.filter(pl.col("player_name") == "A").row(0, named=True)
    b = result.filter(pl.col("player_name") == "B").row(0, named=True)
    assert a["opportunity_cost"] == pytest.approx(100.0 - 65.0)
    assert b["opportunity_cost"] == pytest.approx(60.0 - 65.0)


def test_add_opportunity_cost_is_near_zero_for_a_player_certain_to_survive() -> None:
    """A lone player at a position, ADP far past the pick -- almost certain
    to still be there, so waiting costs almost nothing."""
    df = pl.DataFrame(
        {
            "position": ["TE"],
            "player_name": ["Solo"],
            "vor": [40.0],
            "adp": [200.0],
            "adp_sd": [5.0],
        }
    )

    result = adp.add_opportunity_cost(df, fallback_pick=20, adp_sd_fallback=8.0)

    assert result.row(0, named=True)["opportunity_cost"] == pytest.approx(0.0, abs=1e-3)


def test_add_opportunity_cost_needs_the_pick_after_the_current_one_not_the_current_one() -> None:
    """Regression guard for the trap found via live testing (see module
    docstring): passing the SAME pick number a player is currently being
    evaluated for (like p_avail_next's own pick) produces a nonsensical
    NEGATIVE opportunity_cost for a very good (but not the single best)
    player -- an artifact of a still-likely-available superstar dominating
    the position's expected-best-available at that same pick, which isn't a
    real alternative (you can only take one player with one pick). Using the
    picker's actual *next* pick after this one flips it to a sensible large
    positive number instead: B's own ADP (15) means he's very likely gone by
    pick 16 too, so taking him now is genuinely costly to defer."""
    df = pl.DataFrame(
        {
            "position": ["WR", "WR"],
            "player_name": ["A", "B"],
            "vor": [100.0, 90.0],
            "adp": [3.0, 15.0],
            "adp_sd": [1.0, 1.0],
        }
    )

    at_current_pick = adp.add_opportunity_cost(df, fallback_pick=3, adp_sd_fallback=8.0)
    at_next_real_pick = adp.add_opportunity_cost(df, fallback_pick=16, adp_sd_fallback=8.0)

    b_now = at_current_pick.filter(pl.col("player_name") == "B").row(0, named=True)
    b_next = at_next_real_pick.filter(pl.col("player_name") == "B").row(0, named=True)
    assert b_now["opportunity_cost"] < 0  # the nonsensical artifact
    assert b_next["opportunity_cost"] > 50  # the sensible, decision-useful number
