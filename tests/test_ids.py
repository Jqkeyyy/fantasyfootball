"""Tests for ids/mapping.py (SPEC.md §7).

Fixtures under tests/fixtures/ids/ model the four categories of player the mapping
pipeline must handle:
  - Bijan Robinson: already linked in the crosswalk (sleeper_id present in base).
  - Puka Nacua: crosswalk is missing her sleeper_id, but Sleeper's own dict carries
    the gsis_id, so step 2 (layer_sleeper_ids) fills the gap.
  - Amon-Ra St. Brown: crosswalk spells the name with a hyphen and period
    ("Amon-Ra St. Brown"), Sleeper spells it without ("Amon Ra St Brown"). Neither
    the crosswalk nor Sleeper carries the other's id, so only fuzzy name matching
    (step 3) resolves it.
  - Override Test Player / Zzyzx Unmatched Guy / Deep Bench Guy: no crosswalk row
    and no fuzzy match target exist for any of them. The override fixture resolves
    the first by hand; the other two stay unmatched — one within a "top 300"-style
    relevance cutoff (Zzyzx, search_rank=50) and one outside it (Deep Bench, 9000) —
    to prove unmatched_report's ranking and ffapp ids check's cutoff logic.
"""

import json
from pathlib import Path

import polars as pl
import pytest

from ffapp.config import LeagueConfig
from ffapp.ids import mapping
from ffapp.ingest import nflverse, sleeper

FIXTURES = Path(__file__).parent / "fixtures" / "ids"
CROSSWALK_CSV = FIXTURES / "crosswalk.csv"
SLEEPER_PLAYERS_JSON = FIXTURES / "sleeper_players.json"
OVERRIDES_CSV = FIXTURES / "overrides.csv"


# --- normalize_name ---------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Bijan Robinson", "bijan robinson"),
        ("Michael Pittman Jr.", "michael pittman"),
        ("Amon-Ra St. Brown", "amonra st brown"),
        ("Amon Ra St Brown", "amon ra st brown"),
        ("  Odell   Beckham III ", "odell beckham"),
        ("D'Andre Swift", "dandre swift"),
        # Real duplicate-board-row cases found live (2026-08-17): a
        # diacritic mismatch and the common formal/short first-name
        # pairs, both previously splitting one real player into two
        # separate join_key groups downstream.
        ("Audric Estime", "audric estime"),
        ("Audric Estimé", "audric estime"),
        ("Cam Skattebo", "cameron skattebo"),
        ("Cameron Skattebo", "cameron skattebo"),
        ("Cameron Ward", "cameron ward"),
        ("Cam Ward", "cameron ward"),
        ("Kenny Gainwell", "kenneth gainwell"),
        ("Kenneth Gainwell", "kenneth gainwell"),
        ("Chig Okonkwo", "chigoziem okonkwo"),
        ("Chigoziem Okonkwo", "chigoziem okonkwo"),
        ("Christopher Brooks", "christopher brooks"),
        ("Chris Brooks", "christopher brooks"),
        ("Matt Hibner", "matthew hibner"),
        ("Matthew Hibner", "matthew hibner"),
        ("Mitch Tinsley", "mitchell tinsley"),
        ("Mitchell Tinsley", "mitchell tinsley"),
        ("Gabe Davis", "gabriel davis"),
        ("Gabriel Davis", "gabriel davis"),
        ("Mitch Trubisky", "mitchell trubisky"),
        ("Mitchell Trubisky", "mitchell trubisky"),
        # Real player this project already handles (Ken Walker III via
        # projections.aggregate._PLAYER_NAME_ALIASES) -- confirms the
        # general nickname dict now covers this case too, not just the
        # existing full-name-level alias.
        ("Ken Walker III", "kenneth walker"),
    ],
)
def test_normalize_name(raw: str, expected: str) -> None:
    assert mapping.normalize_name(raw) == expected


# --- load_crosswalk_base -----------------------------------------------------


def test_load_crosswalk_base_parses_expected_columns() -> None:
    base = mapping.load_crosswalk_base(CROSSWALK_CSV)

    assert set(base.columns) >= {
        "gsis_id",
        "sleeper_id",
        "pfr_id",
        "espn_id",
        "full_name",
        "position",
        "team",
        "birth_date",
        "normalized_name",
    }
    assert base.height == 3
    bijan = base.filter(pl.col("full_name") == "Bijan Robinson").row(0, named=True)
    assert bijan["gsis_id"] == "00-0039163"
    assert bijan["sleeper_id"] == "9226"


def test_load_crosswalk_base_treats_na_sentinel_as_null() -> None:
    base = mapping.load_crosswalk_base(CROSSWALK_CSV)

    puka = base.filter(pl.col("full_name") == "Puka Nacua").row(0, named=True)
    assert puka["sleeper_id"] is None


# --- layer_sleeper_ids --------------------------------------------------------


def test_layer_sleeper_ids_fills_gap_using_sleeper_own_gsis_field() -> None:
    base = mapping.load_crosswalk_base(CROSSWALK_CSV)

    layered = mapping.layer_sleeper_ids(base, SLEEPER_PLAYERS_JSON)

    puka = layered.filter(pl.col("gsis_id") == "00-0039164").row(0, named=True)
    assert puka["sleeper_id"] == "9509"


def test_layer_sleeper_ids_adds_new_rows_for_players_absent_from_crosswalk() -> None:
    base = mapping.load_crosswalk_base(CROSSWALK_CSV)

    layered = mapping.layer_sleeper_ids(base, SLEEPER_PLAYERS_JSON)

    zzyzx = layered.filter(pl.col("sleeper_id") == "9999").row(0, named=True)
    assert zzyzx["full_name"] == "Zzyzx Unmatched Guy"
    assert zzyzx["gsis_id"] is None
    assert zzyzx["search_rank"] == 50


def test_layer_sleeper_ids_does_not_touch_already_linked_rows() -> None:
    base = mapping.load_crosswalk_base(CROSSWALK_CSV)

    layered = mapping.layer_sleeper_ids(base, SLEEPER_PLAYERS_JSON)

    bijan_rows = layered.filter(pl.col("gsis_id") == "00-0039163")
    assert bijan_rows.height == 1
    assert bijan_rows.row(0, named=True)["sleeper_id"] == "9226"


def test_layer_sleeper_ids_drops_no_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    """SPEC.md §7 / CLAUDE.md rule 4: every crosswalk row and every Sleeper player
    must survive layering, even the ones that never get linked."""
    base = mapping.load_crosswalk_base(CROSSWALK_CSV)

    layered = mapping.layer_sleeper_ids(base, SLEEPER_PLAYERS_JSON)

    # 3 crosswalk rows + (7 sleeper players - the 2 already linked: Bijan, Puka) = 8
    assert layered.height == 8


# --- fuzzy_match_remainder ----------------------------------------------------


def test_fuzzy_match_remainder_resolves_hyphen_and_period_spelling_difference() -> None:
    base = mapping.load_crosswalk_base(CROSSWALK_CSV)
    layered = mapping.layer_sleeper_ids(base, SLEEPER_PLAYERS_JSON)

    matched = mapping.fuzzy_match_remainder(layered, floor=92)

    st_brown = matched.filter(pl.col("gsis_id") == "00-0037746").row(0, named=True)
    assert st_brown["sleeper_id"] == "7564"


def test_fuzzy_match_remainder_does_not_merge_unrelated_players() -> None:
    base = mapping.load_crosswalk_base(CROSSWALK_CSV)
    layered = mapping.layer_sleeper_ids(base, SLEEPER_PLAYERS_JSON)

    matched = mapping.fuzzy_match_remainder(layered, floor=92)

    zzyzx = matched.filter(pl.col("sleeper_id") == "9999").row(0, named=True)
    assert zzyzx["gsis_id"] is None


def test_fuzzy_match_remainder_merges_without_dropping_or_duplicating_rows() -> None:
    base = mapping.load_crosswalk_base(CROSSWALK_CSV)
    layered = mapping.layer_sleeper_ids(base, SLEEPER_PLAYERS_JSON)

    matched = mapping.fuzzy_match_remainder(layered, floor=92)

    # one merge (St. Brown) collapses 8 rows into 7
    assert matched.height == 7


# --- apply_overrides -----------------------------------------------------------


def test_apply_overrides_forces_player_id_for_matching_source_id() -> None:
    df = pl.DataFrame(
        {
            "gsis_id": [None],
            "sleeper_id": ["7788"],
            "player_id": ["synthetic_deadbeef"],
        }
    )

    overridden = mapping.apply_overrides(df, OVERRIDES_CSV)

    assert overridden.row(0, named=True)["player_id"] == "00-0099999"


def test_apply_overrides_leaves_non_matching_rows_untouched() -> None:
    df = pl.DataFrame(
        {
            "gsis_id": ["00-0039163"],
            "sleeper_id": ["9226"],
            "player_id": ["00-0039163"],
        }
    )

    overridden = mapping.apply_overrides(df, OVERRIDES_CSV)

    assert overridden.row(0, named=True)["player_id"] == "00-0039163"


# --- assign_canonical_id -------------------------------------------------------


def test_assign_canonical_id_uses_gsis_id_when_present() -> None:
    df = pl.DataFrame({"gsis_id": ["00-0039163"], "sleeper_id": ["9226"]})

    assigned = mapping.assign_canonical_id(df)

    assert assigned.row(0, named=True)["player_id"] == "00-0039163"


def test_assign_canonical_id_uses_synthetic_hash_when_gsis_missing() -> None:
    df = pl.DataFrame({"gsis_id": [None], "sleeper_id": ["9999"]})

    assigned = mapping.assign_canonical_id(df)

    player_id = assigned.row(0, named=True)["player_id"]
    assert player_id.startswith("synthetic_")


def test_assign_canonical_id_is_deterministic_across_runs() -> None:
    df = pl.DataFrame({"gsis_id": [None], "sleeper_id": ["9999"]})

    first = mapping.assign_canonical_id(df).row(0, named=True)["player_id"]
    second = mapping.assign_canonical_id(df).row(0, named=True)["player_id"]

    assert first == second


# --- build_players_dim (integration) -------------------------------------------


def test_build_players_dim_resolves_every_player_except_the_genuinely_unmatched() -> None:
    dim = mapping.build_players_dim(
        CROSSWALK_CSV, SLEEPER_PLAYERS_JSON, OVERRIDES_CSV, fuzzy_floor=92
    )

    by_sleeper_id = {row["sleeper_id"]: row for row in dim.iter_rows(named=True)}

    assert not by_sleeper_id["9226"]["player_id"].startswith("synthetic_")  # Bijan
    assert not by_sleeper_id["9509"]["player_id"].startswith("synthetic_")  # Puka
    assert not by_sleeper_id["7564"]["player_id"].startswith("synthetic_")  # St. Brown
    assert by_sleeper_id["7788"]["player_id"] == "00-0099999"  # override
    assert by_sleeper_id["9999"]["player_id"].startswith("synthetic_")  # genuinely unmatched
    assert by_sleeper_id["8888"]["player_id"].startswith("synthetic_")  # genuinely unmatched


# --- unmatched_report ----------------------------------------------------------


def test_unmatched_report_excludes_resolved_and_overridden_players() -> None:
    dim = mapping.build_players_dim(
        CROSSWALK_CSV, SLEEPER_PLAYERS_JSON, OVERRIDES_CSV, fuzzy_floor=92
    )

    unmatched = mapping.unmatched_players(dim)

    # "ATL" is the fixture DST: team defenses have no individual gsis_id in the
    # ffverse crosswalk, so they legitimately stay synthetic forever, harmlessly
    # excluded from the top-N gate by their always-null search_rank.
    sleeper_ids = set(unmatched["sleeper_id"])
    assert sleeper_ids == {"9999", "8888", "ATL"}


def test_unmatched_report_ranks_by_search_rank_ascending() -> None:
    dim = mapping.build_players_dim(
        CROSSWALK_CSV, SLEEPER_PLAYERS_JSON, OVERRIDES_CSV, fuzzy_floor=92
    )

    unmatched = mapping.unmatched_players(dim)

    # nulls (the DST) sort last
    assert unmatched["sleeper_id"].to_list() == ["9999", "8888", "ATL"]


def test_unmatched_report_within_top_n_flags_only_high_relevance_players() -> None:
    dim = mapping.build_players_dim(
        CROSSWALK_CSV, SLEEPER_PLAYERS_JSON, OVERRIDES_CSV, fuzzy_floor=92
    )
    unmatched = mapping.unmatched_players(dim)

    blocking = mapping.within_top_n(unmatched, top_n=300)

    assert blocking["sleeper_id"].to_list() == ["9999"]


def test_unmatched_report_orchestrates_fetch_and_build(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(nflverse, "fetch_player_ids", lambda **kwargs: CROSSWALK_CSV)
    monkeypatch.setattr(sleeper, "fetch_players", lambda **kwargs: SLEEPER_PLAYERS_JSON)
    monkeypatch.setattr(mapping, "ID_OVERRIDES_PATH", OVERRIDES_CSV)

    report = mapping.unmatched_report(2026)

    sleeper_ids = set(report["sleeper_id"])
    assert sleeper_ids == {"9999", "8888", "ATL"}


# --- league-relevance scoping ---------------------------------------------------
# Sleeper's search_rank spans every player it tracks, including retired players and
# IDP positions a given league may not roster at all. `ffapp ids check`'s top-N gate
# needs to be scoped to what this league actually starts, derived from its config
# (CLAUDE.md rule 5: never hardcode league format) rather than a fixed position list.


def test_layer_sleeper_ids_carries_active_flag(tmp_path: Path) -> None:
    crosswalk_path = tmp_path / "crosswalk.csv"
    crosswalk_path.write_text("gsis_id,sleeper_id,pfr_id,espn_id,name,position,team,birthdate\n")
    sleeper_path = tmp_path / "sleeper.json"
    sleeper_path.write_text(
        json.dumps(
            {
                "1": {
                    "full_name": "Test Active Player",
                    "position": "WR",
                    "team": "KC",
                    "gsis_id": None,
                    "espn_id": None,
                    "birth_date": None,
                    "search_rank": 10,
                    "active": True,
                }
            }
        )
    )

    base = mapping.load_crosswalk_base(crosswalk_path)
    layered = mapping.layer_sleeper_ids(base, sleeper_path)

    assert layered.row(0, named=True)["active"] is True


def test_league_relevant_positions_includes_flex_eligible_and_excludes_bench() -> None:
    league = LeagueConfig(
        slug="test-league",
        display_name="Test League",
        is_primary=True,
        league_id="1",
        season=2026,
        league_cache={
            "roster_positions": [
                "QB",
                "RB",
                "RB",
                "WR",
                "WR",
                "TE",
                "FLEX",
                "FLEX",
                "K",
                "DEF",
                "BN",
                "BN",
            ]
        },
        overrides={"flex_eligible": ["RB", "WR", "TE"]},
    )

    positions = mapping.league_relevant_positions(league)

    assert positions == {"QB", "RB", "WR", "TE", "K", "DEF"}


def test_league_relevant_positions_includes_superflex_eligible() -> None:
    league = LeagueConfig(
        slug="test-league",
        display_name="Test League",
        is_primary=True,
        league_id="1",
        season=2026,
        league_cache={"roster_positions": ["QB", "SUPER_FLEX", "WR", "BN"]},
        overrides={"superflex_eligible": ["QB", "RB", "WR", "TE"]},
    )

    positions = mapping.league_relevant_positions(league)

    assert positions == {"QB", "WR", "RB", "TE"}


def test_league_relevant_positions_includes_rec_flex_eligible() -> None:
    league = LeagueConfig(
        slug="test-league",
        display_name="Test League",
        is_primary=True,
        league_id="1",
        season=2026,
        league_cache={"roster_positions": ["QB", "REC_FLEX", "BN"]},
        overrides={"rec_flex_eligible": ["WR", "TE"]},
    )

    positions = mapping.league_relevant_positions(league)

    assert positions == {"QB", "WR", "TE"}


def test_league_relevant_filters_by_position_and_active() -> None:
    df = pl.DataFrame(
        {
            "sleeper_id": ["1", "2", "3", "4"],
            "position": ["WR", "LB", "WR", "RB"],
            "team": ["KC", "KC", "KC", "KC"],
            "active": [True, True, False, None],
        }
    )

    relevant = mapping.league_relevant(df, {"WR", "RB"})

    assert relevant["sleeper_id"].to_list() == ["1"]


def test_league_relevant_excludes_players_with_no_current_nfl_team() -> None:
    """Sleeper's active=True only means "not purged from Sleeper's system" -- a
    cut/unsigned player can still be active=True with team=null. Real fantasy
    relevance requires an actual current NFL team.
    """
    df = pl.DataFrame(
        {
            "sleeper_id": ["1", "2"],
            "position": ["WR", "WR"],
            "team": ["KC", None],
            "active": [True, True],
        }
    )

    relevant = mapping.league_relevant(df, {"WR"})

    assert relevant["sleeper_id"].to_list() == ["1"]


# --- dedupe_to_one_row_per_name_position ----------------------------------------


def test_dedupe_to_one_row_per_name_position_is_a_no_op_when_keys_are_unique() -> None:
    players_dim = pl.DataFrame(
        {
            "normalized_name": ["josh allen", "jahmyr gibbs"],
            "position": ["QB", "RB"],
            "sleeper_id": ["1", "2"],
        }
    )

    result = mapping.dedupe_to_one_row_per_name_position(players_dim)

    assert result.height == 2
    assert set(result["join_key"].to_list()) == {"josh allen|QB", "jahmyr gibbs|RB"}


def test_dedupe_to_one_row_per_name_position_prefers_a_real_sleeper_id() -> None:
    """Real case: an active player and a same-base-name retired relative
    both normalize to the same (name, position) key -- confirmed live with
    "Marvin Harrison Jr." (active, real sleeper_id) and his retired father
    "Marvin Harrison" (crosswalk-only, no sleeper_id)."""
    players_dim = pl.DataFrame(
        {
            "normalized_name": ["marvin harrison", "marvin harrison", "marvin harrison"],
            "position": ["WR", "WR", "WR"],
            "sleeper_id": [None, "11628", None],
            "team": ["FA*", "ARI", "IND"],
        }
    )

    result = mapping.dedupe_to_one_row_per_name_position(players_dim)

    assert result.height == 1
    row = result.row(0, named=True)
    assert row["sleeper_id"] == "11628"
    assert row["team"] == "ARI"


def test_dedupe_to_one_row_per_name_position_breaks_ties_deterministically() -> None:
    """When no row in a collision has a sleeper_id, keep the first one seen
    rather than raising or picking randomly -- a low-stakes fallback, not a
    case this project bets real value on."""
    players_dim = pl.DataFrame(
        {
            "normalized_name": ["ambiguous guy", "ambiguous guy"],
            "position": ["RB", "RB"],
            "sleeper_id": [None, None],
            "team": ["KC", "SF"],
        }
    )

    result = mapping.dedupe_to_one_row_per_name_position(players_dim)

    assert result.height == 1
    assert result.row(0, named=True)["team"] == "KC"
