import re
import subprocess
from pathlib import Path

import polars as pl
import pytest

from ffapp.config import CacheSettings, Settings
from ffapp.draft import board
from ffapp.league_format import LeagueFormat

# --- _streaming_replacement_overrides: board-level position exclusion -------


def test_streaming_replacement_overrides_skips_fetch_entirely_when_positions_excluded() -> None:
    """Once DST/K are excluded from the board itself (settings.draft.
    excluded_positions -- direct request 2026-08-14), there's nothing to
    compute a streaming replacement level for. This must return early
    without ever touching real nflverse data, so it stays safe to call
    with no cached historical tables at all."""
    league_format = LeagueFormat(
        n_teams=10,
        starters={"QB": 1, "RB": 2, "WR": 2, "DST": 1, "K": 1},
        flex_slots={"FLEX": 0, "SUPER_FLEX": 0, "REC_FLEX": 0},
        flex_eligible={},
        bench=6,
        ir=0,
        playoff_week_start=15,
        waiver_budget=None,
    )
    settings = Settings(
        data_root=Path("data"),
        sleeper_username="fixture_user",
        cache=CacheSettings(
            root=Path("data/raw"), offline_default=True, staleness_hours={}, warn_on_stale=True
        ),
    )

    result = board._streaming_replacement_overrides(
        league_format,
        {},
        board_positions={"QB", "RB", "WR"},  # DST/K already excluded upstream
        season=2026,
        offline=True,
        settings=settings,
    )

    assert result == {}


# --- draft_board_csv_path -----------------------------------------------------


def test_draft_board_csv_path_matches_spec_9_7() -> None:
    settings = Settings(
        data_root=Path("data"),
        sleeper_username="fixture_user",
        cache=CacheSettings(
            root=Path("data/raw"), offline_default=True, staleness_hours={}, warn_on_stale=True
        ),
    )

    result = board.draft_board_csv_path(settings, season=2026)

    assert result == Path("data") / "outputs" / "draft_board_2026.csv"


def test_source_rankings_csv_path_is_alongside_the_main_board() -> None:
    settings = Settings(
        data_root=Path("data"),
        sleeper_username="fixture_user",
        cache=CacheSettings(
            root=Path("data/raw"), offline_default=True, staleness_hours={}, warn_on_stale=True
        ),
    )

    result = board.source_rankings_csv_path(settings, season=2026)

    assert result == Path("data") / "outputs" / "source_rankings_2026.csv"


# --- POINT_SOURCE_NAMES / RANK_SOURCE_NAMES -------------------------------------


def test_point_and_rank_source_names_are_disjoint_and_cover_all_seven() -> None:
    assert board.POINT_SOURCE_NAMES == {"espn", "fantasysharks", "cbs", "fftoday"}
    assert board.RANK_SOURCE_NAMES == {"fantasypros", "footballguys", "draftsharks"}
    assert board.POINT_SOURCE_NAMES.isdisjoint(board.RANK_SOURCE_NAMES)


# --- _team_by_join_key ---------------------------------------------------------


def test_team_by_join_key_pulls_team_back_in_from_players_dim() -> None:
    players_dim = pl.DataFrame(
        {
            "normalized_name": ["jahmyr gibbs", "no team guy"],
            "position": ["RB", "WR"],
            "team": ["DET", None],
            "sleeper_id": ["1", "2"],
        }
    )

    result = board._team_by_join_key(players_dim)

    assert result.height == 1  # the null-team row is dropped, not carried as a fake join target
    row = result.row(0, named=True)
    assert row["join_key"] == "jahmyr gibbs|RB"
    assert row["team"] == "DET"


def test_team_by_join_key_prefers_the_row_with_a_real_sleeper_id() -> None:
    """Real case: an active player and a same-base-name retired relative
    both normalize to the same join key -- the crosswalk-only historical
    entry (no sleeper_id) must not win."""
    players_dim = pl.DataFrame(
        {
            "normalized_name": ["marvin harrison", "marvin harrison"],
            "position": ["WR", "WR"],
            "team": ["IND", "ARI"],
            "sleeper_id": [None, "11628"],
        }
    )

    result = board._team_by_join_key(players_dim)

    assert result.height == 1
    assert result.row(0, named=True)["team"] == "ARI"


# --- _per_source_rank_columns ---------------------------------------------------


def test_per_source_rank_columns_excludes_rank_sd() -> None:
    """Real bug: rank_sd (cross-source dispersion) matches the `rank_`
    prefix like a real per-source column would, but selecting it both
    explicitly and via this list raised a polars DuplicateError, caught
    only by running the real pipeline end to end."""
    df = pl.DataFrame(
        {
            "join_key": ["a|RB"],
            "avg_rank": [2.0],
            "rank_sd": [0.5],
            "rank_espn": [1.0],
            "rank_cbs": [3.0],
        }
    )

    result = board._per_source_rank_columns(df)

    assert result == ["rank_cbs", "rank_espn"]


# --- _resolve_cbs_manual_names ---------------------------------------------------


def _other_source(rows: list[tuple[str, str, float]]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "player_name": [name for name, _, _ in rows],
            "position": [position for _, position, _ in rows],
            "rank": [rank for _, _, rank in rows],
        },
        schema={"player_name": pl.Utf8, "position": pl.Utf8, "rank": pl.Float64},
    )


def _cbs_row(position: str, player_name: str, rank: float) -> pl.DataFrame:
    return pl.DataFrame({"position": [position], "player_name": [player_name], "rank": [rank]})


def test_resolve_cbs_manual_names_matches_the_abbreviated_form_exactly() -> None:
    other_sources = [_other_source([("Jahmyr Gibbs", "RB", 1.0)])]
    cbs_df = _cbs_row("RB", "J. Gibbs", 1.0)

    result = board._resolve_cbs_manual_names(cbs_df, other_sources)

    assert result["player_name"].to_list() == ["Jahmyr Gibbs"]


def test_resolve_cbs_manual_names_strips_a_suffix_via_normalize_name() -> None:
    other_sources = [_other_source([("Luther Burden III", "WR", 47.0)])]
    cbs_df = _cbs_row("WR", "L. Burden III", 47.0)

    result = board._resolve_cbs_manual_names(cbs_df, other_sources)

    assert result["player_name"].to_list() == ["Luther Burden III"]


def test_resolve_cbs_manual_names_is_scoped_to_the_same_position() -> None:
    """A same-initial player at a different position must never match --
    same precedent as every other cross-source join in this codebase."""
    other_sources = [_other_source([("Ja'Marr Chase", "WR", 3.0)])]
    cbs_df = _cbs_row("RB", "J. Chase", 3.0)

    result = board._resolve_cbs_manual_names(cbs_df, other_sources)

    assert result["player_name"].to_list() == ["J. Chase"]  # left abbreviated, not misassigned


def test_resolve_cbs_manual_names_leaves_an_unresolvable_row_abbreviated_not_dropped(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """CLAUDE.md rule 4: never silently drop a row in a join."""
    other_sources: list[pl.DataFrame] = [_other_source([])]
    cbs_df = _cbs_row("RB", "Z. Mystery", 150.0)

    with caplog.at_level("WARNING"):
        result = board._resolve_cbs_manual_names(cbs_df, other_sources)

    assert result.height == 1
    assert result["player_name"].to_list() == ["Z. Mystery"]
    assert any("could not be resolved" in record.message for record in caplog.records)


def test_resolve_cbs_manual_names_falls_back_to_fuzzy_matching_a_near_miss() -> None:
    other_sources = [_other_source([("Amon-Ra St. Brown", "WR", 6.0)])]
    # A one-character-off abbreviation of "Amon-Ra St. Brown" -- close
    # enough for the fuzzy fallback, not an exact match.
    cbs_df = _cbs_row("WR", "A. St Brown", 6.0)

    result = board._resolve_cbs_manual_names(cbs_df, other_sources)

    assert result["player_name"].to_list() == ["Amon-Ra St. Brown"]


def test_resolve_cbs_manual_names_resolves_a_genuine_ambiguity_by_rank_proximity() -> None:
    """Real bug, confirmed live: two different real RBs -- "Bijan Robinson"
    (a top-5 overall pick) and "Brian Robinson Jr." (a late-round flex) --
    both abbreviate to "B. Robinson". CBS's real "B. Robinson" row at
    overall rank 2 is obviously about the player every other source *also*
    has near rank 2, not the one near rank 90 -- resolved here by whichever
    candidate's own average rank across the other sources is closer to
    CBS's rank for this specific row, not guessed at or left ambiguous."""
    other_sources = [
        _other_source([("Bijan Robinson", "RB", 2.0), ("Brian Robinson Jr.", "RB", 91.0)])
    ]
    cbs_df = _cbs_row("RB", "B. Robinson", 2.0)

    result = board._resolve_cbs_manual_names(cbs_df, other_sources)

    assert result["player_name"].to_list() == ["Bijan Robinson"]


def test_resolve_cbs_manual_names_rank_proximity_goes_the_other_way_too() -> None:
    """Same two real players as above, but this time CBS's own "B.
    Robinson" row is deep (rank 88) -- proximity must correctly resolve to
    the *other* candidate, proving this isn't just "always pick the star"."""
    other_sources = [
        _other_source([("Bijan Robinson", "RB", 2.0), ("Brian Robinson Jr.", "RB", 91.0)])
    ]
    cbs_df = _cbs_row("RB", "B. Robinson", 88.0)

    result = board._resolve_cbs_manual_names(cbs_df, other_sources)

    assert result["player_name"].to_list() == ["Brian Robinson Jr."]


def test_resolve_cbs_manual_names_does_not_treat_spelling_variants_as_ambiguous() -> None:
    """Real bug in an earlier version of this fix: three different sources
    spell the *same* real player three different ways ("Kenneth Walker
    III"/"Kenneth Walker"/"Ken Walker III") -- these must collapse to one
    identity (via the same join_key/alias-table logic add_join_key already
    uses), not be flagged as three candidates and left unresolved."""
    other_sources = [
        _other_source([("Kenneth Walker III", "RB", 17.0)]),
        _other_source([("Kenneth Walker", "RB", 15.0)]),
        _other_source([("Ken Walker III", "RB", 16.0)]),
    ]
    cbs_df = _cbs_row("RB", "K. Walker III", 17.0)

    result = board._resolve_cbs_manual_names(cbs_df, other_sources)

    assert result["player_name"].to_list() == ["Kenneth Walker III"]


# --- finalize_draft_board ----------------------------------------------------


def _board_input() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "player_name": ["Elite RB", "Good WR", "Deep Bench TE"],
            "position": ["RB", "WR", "TE"],
            "team": ["DET", "CIN", "KC"],
            "bye_week": [6, 10, 12],
            "proj_points_adj": [280.0, 210.0, 90.0],
            "proj_ppg": [16.5, 12.4, 5.3],
            "expected_games": [17.0, 17.0, 17.0],
            "vor": [150.0, 90.0, 10.0],
            "dispersion": [5.0, 4.0, 2.0],
            "n_sources": [4, 4, 3],
            "adp": [2.0, 15.0, None],
            "adp_sd": [1.0, 2.0, None],
            "p_avail_next": [0.1, 0.6, 1.0],
            "opportunity_cost": [40.0, 5.0, 0.0],
            "tier": [1, 1, 3],
            "is_keeper": [True, False, False],
        }
    )


def test_finalize_draft_board_ranks_by_vor_descending_overall_and_per_position() -> None:
    result = board.finalize_draft_board(
        _board_input(), as_of_utc="2026-08-12T00:00:00+00:00", git_commit="abc123"
    )

    assert result["player"].to_list() == ["Elite RB", "Good WR", "Deep Bench TE"]
    assert result["overall_rank"].to_list() == [1, 2, 3]
    assert result["pos_rank"].to_list() == [1, 1, 1]  # each is the only player at its position


def test_finalize_draft_board_computes_value_vs_adp_as_adp_rank_minus_overall_rank() -> None:
    result = board.finalize_draft_board(_board_input(), as_of_utc="x", git_commit=None)

    elite_rb = result.filter(pl.col("player") == "Elite RB").row(0, named=True)
    good_wr = result.filter(pl.col("player") == "Good WR").row(0, named=True)
    deep_te = result.filter(pl.col("player") == "Deep Bench TE").row(0, named=True)
    assert elite_rb["value_vs_adp"] == 0  # adp_rank 1 - overall_rank 1
    assert good_wr["value_vs_adp"] == 0  # adp_rank 2 - overall_rank 2
    assert deep_te["value_vs_adp"] is None  # no adp coverage -- never fabricated


def test_finalize_draft_board_leaves_playoff_sos_null_and_carries_provenance() -> None:
    result = board.finalize_draft_board(
        _board_input(), as_of_utc="2026-08-12T00:00:00+00:00", git_commit="abc123"
    )

    assert result["playoff_sos"].is_null().all()  # SPEC §14.5 not built -- never faked
    assert result["as_of_utc"].to_list() == ["2026-08-12T00:00:00+00:00"] * 3
    # "g" prefix: a bare short hash like "abc123" would round-trip through a
    # naive CSV reader as a number in some cases -- see finalize_draft_board.
    assert result["git_commit"].to_list() == ["gabc123"] * 3


def test_finalize_draft_board_git_commit_stays_null_when_git_is_unavailable() -> None:
    result = board.finalize_draft_board(_board_input(), as_of_utc="x", git_commit=None)

    assert result["git_commit"].is_null().all()


def test_finalize_draft_board_column_order_matches_spec_9_7() -> None:
    result = board.finalize_draft_board(_board_input(), as_of_utc="x", git_commit=None)

    assert result.columns == board.BOARD_COLUMNS


def test_finalize_draft_board_carries_is_keeper_through_and_never_excludes_a_keeper() -> None:
    """Keepers stay on the board (a real, later request) -- `is_keeper` is
    purely informational, not a filter, so a keeper still gets ranked
    exactly like any other player."""
    result = board.finalize_draft_board(_board_input(), as_of_utc="x", git_commit=None)

    assert result.height == 3  # the keeper (Elite RB) was not dropped
    keeper_row = result.filter(pl.col("player") == "Elite RB").row(0, named=True)
    assert keeper_row["is_keeper"] is True
    assert keeper_row["overall_rank"] == 1  # ranked on VOR like everyone else
    non_keeper_rows = result.filter(pl.col("player") != "Elite RB")
    assert non_keeper_rows["is_keeper"].to_list() == [False, False]


# --- _fetch_point_sources: graceful per-source degradation ------------------


def _fake_source(height: int) -> pl.DataFrame:
    return pl.DataFrame({"player_name": [f"P{i}" for i in range(height)]})


def test_fetch_point_sources_skips_a_source_that_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(season, offline, settings):
        raise RuntimeError("403 Forbidden")

    monkeypatch.setattr(
        board,
        "_POINT_SOURCE_FETCHERS",
        {
            "broken": boom,
            "working": lambda season, offline, settings: _fake_source(5),
        },
    )

    result = board._fetch_point_sources(2026, offline=True, settings=None)

    assert len(result) == 1
    assert result[0].height == 5


def test_fetch_point_sources_skips_a_source_that_returns_zero_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        board,
        "_POINT_SOURCE_FETCHERS",
        {
            "empty": lambda season, offline, settings: _fake_source(0),
            "working": lambda season, offline, settings: _fake_source(5),
        },
    )

    result = board._fetch_point_sources(2026, offline=True, settings=None)

    assert len(result) == 1


def test_fetch_point_sources_returns_empty_list_if_every_source_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(season, offline, settings):
        raise RuntimeError("down")

    monkeypatch.setattr(board, "_POINT_SOURCE_FETCHERS", {"a": boom, "b": boom})

    result = board._fetch_point_sources(2026, offline=True, settings=None)

    assert result == []


# --- _fetch_rank_sources: graceful per-source degradation --------------------


def test_fetch_rank_sources_skips_a_source_that_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(season, offline, settings):
        raise RuntimeError("403 Forbidden")

    monkeypatch.setattr(
        board,
        "_RANK_SOURCE_FETCHERS",
        {
            "broken": boom,
            "working": lambda season, offline, settings: _fake_source(5),
        },
    )

    result = board._fetch_rank_sources(2026, offline=True, settings=None)

    assert len(result) == 1
    assert result[0].height == 5


def test_fetch_rank_sources_skips_a_source_that_returns_zero_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        board,
        "_RANK_SOURCE_FETCHERS",
        {
            "empty": lambda season, offline, settings: _fake_source(0),
            "working": lambda season, offline, settings: _fake_source(5),
        },
    )

    result = board._fetch_rank_sources(2026, offline=True, settings=None)

    assert len(result) == 1


def test_fetch_rank_sources_returns_empty_list_if_every_source_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(season, offline, settings):
        raise RuntimeError("down")

    monkeypatch.setattr(board, "_RANK_SOURCE_FETCHERS", {"a": boom, "b": boom})

    result = board._fetch_rank_sources(2026, offline=True, settings=None)

    assert result == []


# --- _current_git_commit -----------------------------------------------------


def test_current_git_commit_returns_a_short_hash_in_this_real_repo() -> None:
    result = board._current_git_commit()

    assert result is not None
    assert re.fullmatch(r"[0-9a-f]{7,}", result)


def test_current_git_commit_returns_none_if_git_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(*args, **kwargs):
        raise FileNotFoundError("git not found")

    monkeypatch.setattr(subprocess, "run", boom)

    assert board._current_git_commit() is None
