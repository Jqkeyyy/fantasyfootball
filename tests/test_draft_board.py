import re
import subprocess
from pathlib import Path

import polars as pl
import pytest

from ffapp.config import CacheSettings, Settings
from ffapp.draft import board

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


# --- _team_by_join_key ---------------------------------------------------------


def test_team_by_join_key_pulls_team_back_in_from_players_dim() -> None:
    players_dim = pl.DataFrame(
        {
            "normalized_name": ["jahmyr gibbs", "no team guy"],
            "position": ["RB", "WR"],
            "team": ["DET", None],
        }
    )

    result = board._team_by_join_key(players_dim)

    assert result.height == 1  # the null-team row is dropped, not carried as a fake join target
    row = result.row(0, named=True)
    assert row["join_key"] == "jahmyr gibbs|RB"
    assert row["team"] == "DET"


def test_team_by_join_key_keeps_first_team_on_a_duplicate_join_key() -> None:
    players_dim = pl.DataFrame(
        {
            "normalized_name": ["dup guy", "dup guy"],
            "position": ["RB", "RB"],
            "team": ["KC", "SF"],
        }
    )

    result = board._team_by_join_key(players_dim)

    assert result.height == 1
    assert result.row(0, named=True)["team"] == "KC"


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
