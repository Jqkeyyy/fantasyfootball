import polars as pl
import pytest

from ffapp.draft import live
from ffapp.league_format import LeagueFormat


def _pick(name: str, position: str) -> dict:
    first, _, last = name.partition(" ")
    return {"metadata": {"first_name": first, "last_name": last, "position": position}}


def _board() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "player": ["Ja'Marr Chase", "Bijan Robinson", "Trey McBride", "Josh Allen"],
            "position": ["WR", "RB", "TE", "QB"],
            "tier": [1, 1, 1, 2],
            "vor": [150.0, 140.0, 90.0, 80.0],
        }
    )


# --- pick_join_key / drafted_join_keys ------------------------------------------


def test_pick_join_key_matches_the_board_join_key_convention() -> None:
    pick = _pick("Ja'Marr Chase", "WR")

    assert live.pick_join_key(pick) == "jamarr chase|WR"


def test_pick_join_key_returns_none_for_missing_metadata() -> None:
    assert live.pick_join_key({}) is None
    assert live.pick_join_key({"metadata": {}}) is None


def test_drafted_join_keys_skips_unresolvable_picks() -> None:
    picks = [_pick("Ja'Marr Chase", "WR"), {}, _pick("Bijan Robinson", "RB")]

    result = live.drafted_join_keys(picks)

    assert result == {"jamarr chase|WR", "bijan robinson|RB"}


# --- available_pool ---------------------------------------------------------------


def test_available_pool_with_no_picks_returns_the_whole_board() -> None:
    result = live.available_pool(_board(), [])

    assert result.height == 4


def test_available_pool_removes_drafted_players() -> None:
    picks = [_pick("Ja'Marr Chase", "WR")]

    result = live.available_pool(_board(), picks)

    assert "Ja'Marr Chase" not in result["player"].to_list()
    assert result.height == 3


def test_available_pool_never_drops_undrafted_players() -> None:
    picks = [_pick("Ja'Marr Chase", "WR")]

    result = live.available_pool(_board(), picks)

    assert set(result["player"].to_list()) == {"Bijan Robinson", "Trey McBride", "Josh Allen"}


# --- best_available ----------------------------------------------------------------


def test_best_available_returns_top_n_by_existing_vor_order() -> None:
    result = live.best_available(_board(), n=2)

    assert result["player"].to_list() == ["Ja'Marr Chase", "Bijan Robinson"]


# --- tier_depth_remaining -----------------------------------------------------------


def test_tier_depth_remaining_counts_per_position_and_tier() -> None:
    result = live.tier_depth_remaining(_board())

    rows = {
        (row["position"], row["tier"]): row["remaining"] for row in result.iter_rows(named=True)
    }
    assert rows[("WR", 1)] == 1
    assert rows[("RB", 1)] == 1
    assert rows[("TE", 1)] == 1
    assert rows[("QB", 2)] == 1


# --- current_tier_summary -----------------------------------------------------------


def test_current_tier_summary_reports_the_lowest_remaining_tier_per_position() -> None:
    """SPEC's own example: "3 left in RB tier 4" -- the CURRENT tier, not
    every tier that still has players."""
    board = pl.DataFrame(
        {
            "player": ["RB Tier2 A", "RB Tier2 B", "RB Tier4 A", "WR Tier1 A"],
            "position": ["RB", "RB", "RB", "WR"],
            "tier": [2, 2, 4, 1],
        }
    )

    result = live.current_tier_summary(board)

    rows = {
        row["position"]: (row["tier"], row["remaining"]) for row in result.iter_rows(named=True)
    }
    assert rows["RB"] == (2, 2)  # tier 2 is RB's current (lowest) tier, 2 players left in it
    assert rows["WR"] == (1, 1)


# --- positional_run -----------------------------------------------------------------


def test_positional_run_flags_a_position_going_at_double_its_baseline_rate() -> None:
    # 16 picks total: 4 WR overall (25% baseline), but the last 8 are 4 WR (50% recent) -> 2x.
    picks = (
        [_pick(f"RB Guy {i}", "RB") for i in range(8)]
        + [_pick(f"WR Guy {i}", "WR") for i in range(4)]
        + [_pick(f"TE Guy {i}", "TE") for i in range(4)]
    )
    # last 8 picks: 4 WR + 4 TE
    recent_window = picks[-8:]
    assert sum(1 for p in recent_window if p["metadata"]["position"] == "WR") == 4

    result = live.positional_run(picks)

    assert result["WR"]["is_run"] is True
    assert result["WR"]["baseline_rate"] == pytest.approx(4 / 16)
    assert result["WR"]["recent_rate"] == pytest.approx(4 / 8)


def test_positional_run_does_not_flag_a_steady_rate() -> None:
    picks = [_pick(f"Guy {i}", "RB") if i % 2 == 0 else _pick(f"Guy {i}", "WR") for i in range(16)]

    result = live.positional_run(picks)

    assert result["RB"]["is_run"] is False
    assert result["WR"]["is_run"] is False


def test_positional_run_omits_a_position_with_zero_picks_so_far() -> None:
    picks = [_pick("Guy", "RB")]

    result = live.positional_run(picks)

    assert "QB" not in result


def test_positional_run_with_no_picks_returns_empty() -> None:
    assert live.positional_run([]) == {}


# --- starting_lineup_gaps -----------------------------------------------------------


def _format(**overrides: object) -> LeagueFormat:
    base = dict(
        n_teams=10,
        starters={"QB": 1, "RB": 2, "WR": 2, "TE": 1, "K": 1, "DST": 1},
        flex_slots={"FLEX": 1, "SUPER_FLEX": 0, "REC_FLEX": 0},
        flex_eligible={"FLEX": ["RB", "WR", "TE"]},
        bench=6,
        ir=0,
        playoff_week_start=15,
        waiver_budget=None,
    )
    base.update(overrides)
    return LeagueFormat(**base)  # type: ignore[arg-type]


def test_starting_lineup_gaps_with_no_picks_needs_everything() -> None:
    result = live.starting_lineup_gaps([], _format())

    assert result == {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "K": 1, "DST": 1, "FLEX": 1}


def test_starting_lineup_gaps_fills_dedicated_slots_first() -> None:
    picks = [_pick("A", "QB"), _pick("B", "RB")]

    result = live.starting_lineup_gaps(picks, _format())

    assert result["RB"] == 1
    assert "QB" not in result


def test_starting_lineup_gaps_overflow_fills_flex() -> None:
    """3 RBs drafted but only 2 dedicated RB slots -- the 3rd fills FLEX."""
    picks = [_pick("A", "RB"), _pick("B", "RB"), _pick("C", "RB")]

    result = live.starting_lineup_gaps(picks, _format())

    assert "RB" not in result
    assert "FLEX" not in result  # filled by the 3rd RB


def test_starting_lineup_gaps_ignores_positions_with_no_slot_at_all() -> None:
    """A K drafted when K is already full (or not needed) doesn't spill into FLEX --
    K isn't in FLEX's eligible list."""
    picks = [_pick("A", "K"), _pick("B", "K")]

    result = live.starting_lineup_gaps(picks, _format())

    assert "K" not in result  # the 1 K slot is filled
    assert result["FLEX"] == 1  # the extra K does NOT fill flex (not eligible)


def test_starting_lineup_gaps_empty_when_roster_is_full() -> None:
    picks = [
        _pick("QB1", "QB"),
        _pick("RB1", "RB"),
        _pick("RB2", "RB"),
        _pick("WR1", "WR"),
        _pick("WR2", "WR"),
        _pick("TE1", "TE"),
        _pick("K1", "K"),
        _pick("DST1", "DST"),
        _pick("Flex Filler", "WR"),
    ]

    result = live.starting_lineup_gaps(picks, _format())

    assert result == {}
