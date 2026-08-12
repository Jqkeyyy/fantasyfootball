"""Draft board page logic (SPEC.md §15, §9.7; task 0.13).

Pure, pytest-testable functions only -- filtering and tier-break styling.
`streamlit_app.py` is thin glue on top: read the CSV via `load_board`, call
these, render with `st.*`. A live Streamlit script isn't meaningfully
testable with pytest (it has no return value, it *is* the side effect), so
every real decision lives here instead, and the script is verified by
actually running it (CLAUDE.md's UI rule), not by a unit test.

SPEC §15's design constraint -- "fast to load, everything precomputed,
nothing trained on page load" -- means this reads the CSV `ffapp draft
board` (task 0.12) already wrote, rather than recomputing the pipeline on
every page load.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pandas as pd
import polars as pl
from pandas.io.formats.style import Styler


class DraftBoardNotBuiltError(Exception):
    """No draft board CSV exists yet for this season."""


def load_board(path: Path) -> pl.DataFrame:
    """Read the pre-built draft board CSV (`draft.board.draft_board_csv_path`).

    Raises `DraftBoardNotBuiltError` naming the fix (`ffapp draft board`)
    rather than a bare `FileNotFoundError` -- matches this project's
    existing `OfflineCacheMiss` convention of always naming the command
    that fixes a missing-artefact error.
    """
    if not path.exists():
        raise DraftBoardNotBuiltError(
            f"No draft board found at {path}. Run `ffapp draft board` to build one first."
        )
    return pl.read_csv(path)


def filter_board(
    df: pl.DataFrame, *, positions: list[str] | None = None, tiers: list[int] | None = None
) -> pl.DataFrame:
    """Filter by position and/or tier (SPEC §9.7's closing note: "provide
    filters by position and tier"). `None` or an empty list means "no
    filter" for that dimension -- an empty multiselect in the UI should
    show everything, not nothing, which is what an empty `is_in([])` would
    otherwise silently produce.
    """
    filtered = df
    if positions:
        filtered = filtered.filter(pl.col("position").is_in(positions))
    if tiers:
        filtered = filtered.filter(pl.col("tier").is_in(tiers))
    return filtered


_TIER_SHADE_COLORS = ("#f4f6fa", "#ffffff")


def tier_shade_groups(tiers: list[int]) -> list[int]:
    """Alternating shade-group index (0/1) per row, flipping every time
    `tiers` changes value from the previous row. Assumes `tiers` is already
    in the board's display order (grouped by tier, SPEC's own row order) --
    this does not sort or group, only detects boundaries as they occur.
    """
    groups: list[int] = []
    group = 0
    previous: int | None = None
    for tier in tiers:
        if previous is not None and tier != previous:
            group = 1 - group
        groups.append(group)
        previous = tier
    return groups


def style_tier_breaks(df: pl.DataFrame, *, tier_column: str = "tier") -> Styler:
    """A pandas Styler that alternates row background colour every tier
    boundary -- makes tier breaks a visible break on the board (SPEC §9.5's
    closing note) without a dedicated marker column. Converts to pandas at
    this exact rendering boundary only (CLAUDE.md: polars is the default;
    pandas only where a library boundary requires it -- here, Streamlit's
    `st.dataframe` styling is pandas-Styler-based, not polars-native).
    """
    pandas_df = df.to_pandas().reset_index(drop=True)
    groups = tier_shade_groups(pandas_df[tier_column].tolist())

    def _row_style(row: pd.Series) -> list[str]:
        color = _TIER_SHADE_COLORS[groups[cast(int, row.name)]]
        return [f"background-color: {color}"] * len(row)

    return pandas_df.style.apply(_row_style, axis=1)


__all__ = [
    "DraftBoardNotBuiltError",
    "filter_board",
    "load_board",
    "style_tier_breaks",
    "tier_shade_groups",
]
