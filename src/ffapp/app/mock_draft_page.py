"""Mock draft page logic (not a numbered SPEC/TASKS.md task -- see
`draft.mock`'s own module docstring for why this exists).

Pure, pytest-testable functions only, same split every other page in this
project already uses (`app.draft_board_page`'s own docstring) -- the actual
`pages/6_Mock_Draft.py` script is thin `st.*` glue plus `st.session_state`
bookkeeping, verified by running it, not by a unit test.
"""

from __future__ import annotations

import html
from typing import Any

import polars as pl

from ffapp.draft.mock import GridCell

DEFAULT_POOL_ROWS = 30


def available_pool_display(
    pool: pl.DataFrame, *, position: str | None = None, n: int = DEFAULT_POOL_ROWS
) -> pl.DataFrame:
    """`pool` (already VOR-descending, from `draft.mock`'s own state)
    optionally filtered to one position, capped to the top `n` rows --
    scanning 900+ rows one pick at a time defeats "mass practice" just as
    much as a slow full board render does (`app.draft_board_page.cap_rows`'s
    own reasoning, applied here at a fixed cap since a mock draft's pool
    never needs the "show everything" escape hatch a real draft-prep board
    does).
    """
    filtered = pool.filter(pl.col("position") == position) if position else pool
    return filtered.sort("vor", descending=True).head(n)


def roster_table(picks: list[dict[str, Any]]) -> pl.DataFrame:
    """One team's own drafted players so far, in draft order, with a
    keeper marker -- same `is_keeper` convention `app.draft_board_page`'s
    own `style_tier_breaks` already established, read here as a plain
    column rather than styling (this table is small; no tier-shading
    needed).
    """
    if not picks:
        return pl.DataFrame(
            schema={
                "pick_no": pl.Int64,
                "player": pl.Utf8,
                "position": pl.Utf8,
                "team": pl.Utf8,
                "vor": pl.Float64,
                "tier": pl.Int64,
                "is_keeper": pl.Boolean,
            }
        )
    return pl.DataFrame(
        [
            {
                "pick_no": pick.get("pick_no"),
                "player": pick["player_name"],
                "position": pick["metadata"]["position"],
                "team": pick.get("team"),
                "vor": pick.get("vor"),
                "tier": pick.get("tier"),
                "is_keeper": bool(pick.get("is_keeper", False)),
            }
            for pick in picks
        ]
    )


def _cell_label(cell: GridCell) -> str:
    if cell.player_name:
        position = html.escape(cell.position or "")
        lock = "\U0001f512 " if cell.is_keeper else ""
        return f"{lock}{html.escape(cell.player_name)}<br><span class='mdg-pos'>{position}</span>"
    return f"<span class='mdg-pending'>#{cell.pick_no}</span>"


def _cell_classes(cell: GridCell) -> str:
    classes = ["mdg-cell"]
    if cell.is_keeper:
        classes.append("mdg-keeper")
    elif cell.is_current:
        classes.append("mdg-current")
    elif cell.is_mine:
        classes.append("mdg-mine")
    return " ".join(classes)


def render_draft_grid_html(rows: list[list[GridCell]], team_names: dict[int, str]) -> str:
    """A snake-draft grid -- teams as fixed columns, rounds as rows, same
    layout the real Sleeper draft board uses -- so every pick made and
    every pick still to come is visible at a glance, not just the last few.
    A cell not yet picked shows its pick number, dimmed; the user's own
    cells (including picks traded to them, `cell.is_traded`) are tinted;
    the cell on the clock right now is highlighted solid. Pure string
    building -- no `st.*` calls -- so it's testable without a live app.
    """
    if not rows:
        return "<p>No draft in progress.</p>"

    header_cells = "".join(
        f"<th>{html.escape(team_names.get(cell.original_roster_id, str(cell.original_roster_id)))}"
        "</th>"
        for cell in rows[0]
    )
    body_rows = []
    for row in rows:
        cells = []
        for cell in row:
            trade_note = ""
            if cell.is_traded:
                owner_name = html.escape(
                    team_names.get(cell.owner_roster_id, str(cell.owner_roster_id))
                )
                trade_note = f"<div class='mdg-trade'>→ {owner_name}</div>"
            cells.append(f'<td class="{_cell_classes(cell)}">{_cell_label(cell)}{trade_note}</td>')
        body_rows.append(f"<tr>{''.join(cells)}</tr>")

    return (
        '<div class="mdg-scroll"><table class="mdg-table">'
        f"<thead><tr>{header_cells}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody>"
        "</table></div>"
    )


__all__ = [
    "DEFAULT_POOL_ROWS",
    "available_pool_display",
    "render_draft_grid_html",
    "roster_table",
]
