import polars as pl

from ffapp.app import mock_draft_page as page
from ffapp.draft.mock import GridCell


def _pool() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "join_key": ["a|WR", "b|RB", "c|WR"],
            "player_name": ["A", "B", "C"],
            "position": ["WR", "RB", "WR"],
            "team": ["CIN", "ATL", "DET"],
            "vor": [10.0, 30.0, 20.0],
            "tier": [2, 1, 1],
            "overall_rank": [3, 1, 2],
            "adp": [5.0, 1.0, 2.0],
        }
    )


def _pick(pick_no: int, roster_id: int, name: str, position: str, **extra: object) -> dict:
    return {
        "pick_no": pick_no,
        "roster_id": roster_id,
        "metadata": {"first_name": name, "last_name": "X", "position": position},
        "join_key": f"{name}|{position}",
        "player_name": name,
        "team": "XXX",
        "vor": 10.0,
        "tier": 1,
        "adp": 5.0,
        **extra,
    }


# --- available_pool_display ----------------------------------------------------------


def test_available_pool_display_sorts_by_vor_descending() -> None:
    result = page.available_pool_display(_pool())

    assert result["player_name"].to_list() == ["B", "C", "A"]


def test_available_pool_display_filters_by_position() -> None:
    result = page.available_pool_display(_pool(), position="WR")

    assert result["player_name"].to_list() == ["C", "A"]


def test_available_pool_display_caps_row_count() -> None:
    result = page.available_pool_display(_pool(), n=1)

    assert result.height == 1
    assert result["player_name"][0] == "B"


# --- roster_table -----------------------------------------------------------------------


def test_roster_table_empty_returns_expected_schema() -> None:
    result = page.roster_table([])

    assert result.height == 0
    assert set(result.columns) == {
        "pick_no",
        "player",
        "position",
        "team",
        "vor",
        "tier",
        "is_keeper",
    }


def test_roster_table_lists_picks_with_keeper_flag() -> None:
    picks = [
        _pick(None, 101, "Bijan Robinson", "RB", is_keeper=True),
        _pick(3, 101, "Puka Nacua", "WR"),
    ]

    result = page.roster_table(picks)

    assert result["player"].to_list() == ["Bijan Robinson", "Puka Nacua"]
    assert result["is_keeper"].to_list() == [True, False]


# --- render_draft_grid_html ----------------------------------------------------------------


def _cell(
    *,
    round: int = 1,
    slot: int = 1,
    pick_no: int = 1,
    original_roster_id: int = 101,
    owner_roster_id: int | None = None,
    player_name: str | None = None,
    position: str | None = None,
    is_mine: bool = False,
    is_current: bool = False,
    is_keeper: bool = False,
) -> GridCell:
    owner = owner_roster_id if owner_roster_id is not None else original_roster_id
    return GridCell(
        round=round,
        slot=slot,
        pick_no=pick_no,
        original_roster_id=original_roster_id,
        owner_roster_id=owner,
        player_name=player_name,
        position=position,
        is_mine=is_mine,
        is_traded=owner != original_roster_id,
        is_current=is_current,
        is_keeper=is_keeper,
    )


def test_render_draft_grid_html_empty_rows_returns_placeholder() -> None:
    assert "No draft in progress" in page.render_draft_grid_html([], {})


def test_render_draft_grid_html_includes_team_names_in_header() -> None:
    rows = [[_cell(original_roster_id=101), _cell(slot=2, original_roster_id=102, pick_no=2)]]

    result = page.render_draft_grid_html(rows, {101: "Me", 102: "Bot Two"})

    assert "<th>Me</th>" in result
    assert "<th>Bot Two</th>" in result


def test_render_draft_grid_html_shows_pending_pick_number() -> None:
    rows = [[_cell(pick_no=7)]]

    result = page.render_draft_grid_html(rows, {})

    assert "#7" in result
    assert "mdg-pending" in result


def test_render_draft_grid_html_shows_drafted_player_and_position() -> None:
    rows = [[_cell(player_name="Bijan Robinson", position="RB")]]

    result = page.render_draft_grid_html(rows, {})

    assert "Bijan Robinson" in result
    assert "RB" in result
    assert "mdg-pending" not in result


def test_render_draft_grid_html_marks_current_and_mine_classes() -> None:
    rows = [
        [
            _cell(slot=1, pick_no=1, is_current=True),
            _cell(slot=2, pick_no=2, is_mine=True),
            _cell(slot=3, pick_no=3),
        ]
    ]

    result = page.render_draft_grid_html(rows, {})

    assert 'class="mdg-cell mdg-current"' in result
    assert 'class="mdg-cell mdg-mine"' in result
    assert result.count('class="mdg-cell"') == 1  # the plain, untouched cell


def test_render_draft_grid_html_shows_trade_note_for_traded_cells() -> None:
    rows = [[_cell(original_roster_id=104, owner_roster_id=101, is_mine=True)]]

    result = page.render_draft_grid_html(rows, {101: "Me", 104: "Original Owner"})

    assert "→ Me" in result


def test_render_draft_grid_html_marks_keeper_class_and_lock_icon() -> None:
    rows = [[_cell(player_name="Jonathan Taylor", position="RB", is_keeper=True)]]

    result = page.render_draft_grid_html(rows, {})

    assert "mdg-keeper" in result
    assert "\U0001f512" in result
    assert "Jonathan Taylor" in result


def test_render_draft_grid_html_keeper_class_takes_priority_over_current_and_mine() -> None:
    rows = [[_cell(is_current=True, is_mine=True, is_keeper=True)]]

    result = page.render_draft_grid_html(rows, {})

    assert 'class="mdg-cell mdg-keeper"' in result
