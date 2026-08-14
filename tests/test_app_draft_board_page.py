from pathlib import Path

import polars as pl
import pytest

from ffapp.app import draft_board_page

# --- load_board ---------------------------------------------------------------


def test_load_board_reads_the_csv(tmp_path: Path) -> None:
    path = tmp_path / "draft_board_2026.csv"
    pl.DataFrame({"player": ["A"], "position": ["RB"], "tier": [1]}).write_csv(path)

    result = draft_board_page.load_board(path)

    assert result["player"].to_list() == ["A"]


def test_load_board_raises_a_helpful_error_when_missing(tmp_path: Path) -> None:
    path = tmp_path / "draft_board_2026.csv"

    with pytest.raises(draft_board_page.DraftBoardNotBuiltError) as exc_info:
        draft_board_page.load_board(path)

    assert "ffapp draft board" in str(exc_info.value)


# --- filter_board ---------------------------------------------------------------


def _board() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "player": ["A", "B", "C", "D"],
            "position": ["RB", "WR", "RB", "QB"],
            "tier": [1, 1, 2, 3],
        }
    )


def test_filter_board_by_position() -> None:
    result = draft_board_page.filter_board(_board(), positions=["RB"])

    assert result["player"].to_list() == ["A", "C"]


def test_filter_board_by_tier() -> None:
    result = draft_board_page.filter_board(_board(), tiers=[1])

    assert result["player"].to_list() == ["A", "B"]


def test_filter_board_by_position_and_tier_combined() -> None:
    result = draft_board_page.filter_board(_board(), positions=["RB"], tiers=[2])

    assert result["player"].to_list() == ["C"]


def test_filter_board_with_no_filters_returns_everything() -> None:
    result = draft_board_page.filter_board(_board())

    assert result.height == 4


def test_filter_board_with_empty_list_returns_everything_not_nothing() -> None:
    """An empty multiselect in the UI means 'no filter applied', not
    'filter down to zero rows' -- is_in([]) would otherwise silently empty
    the board."""
    result = draft_board_page.filter_board(_board(), positions=[], tiers=[])

    assert result.height == 4


# --- cap_rows ----------------------------------------------------------------------


def _sized_board(n: int) -> pl.DataFrame:
    return pl.DataFrame({"player": [f"P{i}" for i in range(n)], "vor": list(range(n, 0, -1))})


def test_cap_rows_leaves_a_board_at_or_under_the_threshold_untouched() -> None:
    board = _sized_board(draft_board_page.ROW_CAP_THRESHOLD)

    result = draft_board_page.cap_rows(board)

    assert result.height == draft_board_page.ROW_CAP_THRESHOLD


def test_cap_rows_truncates_to_the_default_cap_once_over_threshold() -> None:
    board = _sized_board(draft_board_page.ROW_CAP_THRESHOLD + 1)

    result = draft_board_page.cap_rows(board)

    assert result.height == draft_board_page.DEFAULT_ROW_CAP
    expected = board.head(draft_board_page.DEFAULT_ROW_CAP)["player"].to_list()
    assert result["player"].to_list() == expected


def test_cap_rows_show_all_bypasses_the_cap() -> None:
    board = _sized_board(draft_board_page.ROW_CAP_THRESHOLD + 1)

    result = draft_board_page.cap_rows(board, show_all=True)

    assert result.height == board.height


def test_cap_rows_respects_custom_threshold_and_cap() -> None:
    board = _sized_board(10)

    result = draft_board_page.cap_rows(board, threshold=5, cap=3)

    assert result.height == 3


# --- tier_shade_groups -----------------------------------------------------------


def test_tier_shade_groups_stays_constant_within_a_tier() -> None:
    result = draft_board_page.tier_shade_groups([1, 1, 1])

    assert result == [0, 0, 0]


def test_tier_shade_groups_flips_on_every_tier_change() -> None:
    result = draft_board_page.tier_shade_groups([1, 1, 2, 2, 3, 4, 4])

    assert result == [0, 0, 1, 1, 0, 1, 1]


def test_tier_shade_groups_handles_empty_input() -> None:
    assert draft_board_page.tier_shade_groups([]) == []


# --- style_tier_breaks ------------------------------------------------------------


def test_style_tier_breaks_returns_a_styler_over_every_row() -> None:
    styler = draft_board_page.style_tier_breaks(_board())

    assert styler.data.shape[0] == 4
    assert list(styler.data["player"]) == ["A", "B", "C", "D"]


def test_style_tier_breaks_renders_alternating_background_colors_at_tier_boundaries() -> None:
    styler = draft_board_page.style_tier_breaks(_board())

    html = styler.to_html()
    assert "background-color: #f4f6fa" in html
    assert "background-color: #ffffff" in html


def _board_with_keeper() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "player": ["A", "B", "C"],
            "position": ["RB", "WR", "TE"],
            "tier": [1, 1, 2],
            "is_keeper": [True, False, False],
        }
    )


def test_style_tier_breaks_highlights_a_keeper_row_with_a_distinct_color() -> None:
    styler = draft_board_page.style_tier_breaks(_board_with_keeper())

    html = styler.to_html()
    assert f"background-color: {draft_board_page._KEEPER_HIGHLIGHT_COLOR}" in html


def test_style_tier_breaks_prefixes_a_keeper_players_displayed_name_with_a_lock_marker() -> None:
    styler = draft_board_page.style_tier_breaks(_board_with_keeper())

    assert list(styler.data["player"]) == ["\U0001f512 A", "B", "C"]


def test_style_tier_breaks_never_mutates_the_original_dataframes_player_column() -> None:
    """The lock-emoji prefix is a rendering-time-only change on the
    Styler's own throwaway pandas copy -- the real polars `df` (and so
    the CSV, draft.live's join-key matching, and every other consumer)
    must keep the clean player name."""
    original = _board_with_keeper()

    draft_board_page.style_tier_breaks(original)

    assert original["player"].to_list() == ["A", "B", "C"]


def test_style_tier_breaks_with_no_is_keeper_column_behaves_as_before() -> None:
    """A `df` with no `is_keeper` column (e.g. an older fixture) gets no
    keeper styling or name mutation -- backward compatible."""
    styler = draft_board_page.style_tier_breaks(_board())

    assert list(styler.data["player"]) == ["A", "B", "C", "D"]


# --- source rankings ("no model" board) -----------------------------------------


def _source_rankings() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "player": ["Star RB", "Deep WR"],
            "position": ["RB", "WR"],
            "team": ["KC", "CIN"],
            "avg_rank": [2.0, 30.0],
            "median_rank": [2.0, 30.0],
            "rank_sd": [0.5, 1.0],
            "n_sources": [3, 2],
            "rank_espn": [1.0, None],
            "rank_fantasypros": [3.0, 29.0],
        }
    )


def test_source_rank_columns_strips_the_rank_prefix_and_sorts() -> None:
    result = draft_board_page.source_rank_columns(_source_rankings())

    assert result == ["espn", "fantasypros"]


def test_consensus_rankings_keeps_only_the_cross_source_columns() -> None:
    result = draft_board_page.consensus_rankings(_source_rankings())

    assert result.columns == [
        "player",
        "position",
        "team",
        "avg_rank",
        "median_rank",
        "rank_sd",
        "n_sources",
    ]
    assert result["player"].to_list() == ["Star RB", "Deep WR"]


def test_single_source_rankings_drops_players_that_source_does_not_cover() -> None:
    result = draft_board_page.single_source_rankings(_source_rankings(), "espn")

    assert result["player"].to_list() == ["Star RB"]  # Deep WR has no rank_espn
    assert result.columns == ["player", "position", "team", "rank"]


def test_single_source_rankings_sorts_by_that_sources_own_rank() -> None:
    result = draft_board_page.single_source_rankings(_source_rankings(), "fantasypros")

    assert result["player"].to_list() == ["Star RB", "Deep WR"]  # rank 3 before rank 29
