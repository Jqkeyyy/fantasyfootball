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


ROW_CAP_THRESHOLD = 1000
DEFAULT_ROW_CAP = 400


def cap_rows(
    df: pl.DataFrame,
    *,
    threshold: int = ROW_CAP_THRESHOLD,
    cap: int = DEFAULT_ROW_CAP,
    show_all: bool = False,
) -> pl.DataFrame:
    """Truncate to the top `cap` rows once `df` exceeds `threshold` --
    styling a 1000+ row table is genuinely slow to paint in Streamlit's own
    dataframe grid (confirmed live, task 0.16's own JOURNAL entry), and the
    project owner never needs to draft that deep anyway. `show_all` is a
    real UI toggle (a button/checkbox in the page itself) that bypasses the
    cap entirely -- this function doesn't decide when that happens, only
    what to do once it has. Assumes `df` is already sorted the way it
    should be truncated (VOR descending, the board's own default order);
    never re-sorts.
    """
    if show_all or df.height <= threshold:
        return df
    return df.head(cap)


_TIER_SHADE_COLORS = ("#f4f6fa", "#ffffff")
_KEEPER_HIGHLIGHT_COLOR = "#fff3cd"


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

    Keeper rows (a real `is_keeper` boolean column, when present) get a
    distinct amber highlight overriding tier shading, plus a lock-emoji
    prefix on the displayed player name -- the project owner asked for
    keepers to be loudly visible on the board, not silently missing.
    Both are purely a rendering-time change on this function's own throw
    -away pandas copy: the underlying polars `df` (and so the real CSV,
    `draft.live`'s join-key matching, and every other consumer) keeps the
    player name clean. A `df` with no `is_keeper` column (e.g. an older
    fixture) behaves exactly as before this feature existed.
    """
    pandas_df = df.to_pandas().reset_index(drop=True)
    groups = tier_shade_groups(pandas_df[tier_column].tolist())
    has_keeper_column = "is_keeper" in pandas_df.columns
    is_keeper = pandas_df["is_keeper"].tolist() if has_keeper_column else [False] * len(pandas_df)

    if has_keeper_column and "player" in pandas_df.columns:
        pandas_df["player"] = [
            f"\U0001f512 {name}" if keeper else name
            for name, keeper in zip(pandas_df["player"], is_keeper, strict=True)
        ]

    def _row_style(row: pd.Series) -> list[str]:
        idx = cast(int, row.name)
        color = _KEEPER_HIGHLIGHT_COLOR if is_keeper[idx] else _TIER_SHADE_COLORS[groups[idx]]
        return [f"background-color: {color}"] * len(row)

    return pandas_df.style.apply(_row_style, axis=1)


def source_rank_columns(df: pl.DataFrame) -> list[str]:
    """Every source this source-rankings table covers (`rank_<source>`
    columns, stripped back down to just `<source>`), sorted alphabetically
    -- for building one Streamlit tab per source. `rank_sd` is the
    cross-source dispersion column, not a source, so it's excluded here.
    """
    return sorted(
        col.removeprefix("rank_")
        for col in df.columns
        if col.startswith("rank_") and col != "rank_sd"
    )


def consensus_rankings(df: pl.DataFrame) -> pl.DataFrame:
    """Player/position/team plus the cross-source consensus columns only --
    the individual `rank_<source>` columns are each source's own tab
    instead (`single_source_rankings`), not shown side by side here.
    `#` is a plain 1-indexed row count over the current (already
    avg_rank-sorted) order -- avg_rank/median_rank are cross-source
    statistics, not a clean "this player is Nth" position on their own,
    which is what's actually easy to scan at a glance.
    """
    return df.select(
        "player", "position", "team", "avg_rank", "median_rank", "rank_sd", "n_sources"
    ).with_row_index(name="#", offset=1)


def single_source_rankings(df: pl.DataFrame, source: str) -> pl.DataFrame:
    """One source's own positional rank, sorted by that source's rank
    ascending. Players that source doesn't cover are dropped -- a null
    rank there means no opinion, not a real ranking to display. `#` is a
    plain 1-indexed row count over the current (already rank-sorted, and
    possibly position-filtered) order -- distinct from `rank`, that
    source's own real published rank, which can have gaps once filtered
    to one position (e.g. RB7, RB9, RB11, ...).
    """
    column = f"rank_{source}"
    return (
        df.filter(pl.col(column).is_not_null())
        .select("player", "position", "team", pl.col(column).alias("rank"))
        .sort("rank")
        .with_row_index(name="#", offset=1)
    )


__all__ = [
    "DEFAULT_ROW_CAP",
    "ROW_CAP_THRESHOLD",
    "DraftBoardNotBuiltError",
    "cap_rows",
    "consensus_rankings",
    "filter_board",
    "load_board",
    "single_source_rankings",
    "source_rank_columns",
    "style_tier_breaks",
    "tier_shade_groups",
]
