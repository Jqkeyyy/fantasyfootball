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
