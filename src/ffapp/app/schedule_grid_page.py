"""Schedule grid page composition (SPEC.md §14.5; task 2.8).

Pure, pytest-testable composition only -- the real page
(`app/pages/3_Schedule_Grid.py`) is thin `st.*` glue on top, matching
`weekly_rankings_page.py`'s own precedent (task 1.19). The opponent-
adjustment math itself (SOS sums, the grid pivot, per-position-group
matchup detail) lives in `tools.sos`; this module holds the two things
that are specific to *rendering* that math as a page: "which real NFL
teams are on my roster" (SPEC §14.5's own roster-highlight overlay) and
the heatmap's own cell styling.

**Heatmap colour, per the dataviz skill's own rule ("diverging = two
hues + a neutral gray midpoint, never a rainbow"):** blue/gray/red from
`references/palette.md`'s validated diverging pair (`node
scripts/validate_palette.js "#2a78d6,#e34948" --mode light` -- both
poles pass every check; the gray midpoint is intentionally near-surface
and exempt from the categorical chroma/lightness checks by the same
file's own design, "reads as nothing"). Implemented as a small manual hex
interpolator rather than pulling in `matplotlib` as a new dependency for
one three-stop gradient -- `pandas.Styler.background_gradient` needs it
internally, but a bespoke two-segment lerp needs nothing beyond the
stdlib.

**Blocked cells (byes and low-confidence matchups) get a flat,
desaturated colour distinct from any real diverging value** -- SPEC
§14.5's own "grey out grades... rather than displaying a confident-
looking colour" and "bye weeks rendered as blocked-out cells," applied
identically to both real reasons a cell shouldn't be trusted, so a
low-sample matchup never reads as a genuinely neutral (near-zero) one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import polars as pl

if TYPE_CHECKING:
    from pandas.io.formats.style import Styler

_EASY_COLOR = "#2a78d6"  # blue -- opponent allows more EPA, an easier real matchup
_NEUTRAL_COLOR = "#f0efec"  # gray midpoint -- an average real matchup
_HARD_COLOR = "#e34948"  # red -- opponent allows less EPA, a harder real matchup
_BLOCKED_COLOR = "#e1e0d9"  # flat, desaturated -- a real bye or a low-sample matchup
_TEXT_COLOR = "#0b0b0b"


def resolve_my_teams(my_roster_ids: set[str], players_dim: pl.DataFrame) -> set[str]:
    """Real current NFL teams for the players in `my_roster_ids`, from
    `players_dim`'s own `team` column (Sleeper's current team -- the same
    source `weekly_rankings_page`'s own roster resolution already
    trusts). A team-level heatmap has no player identity of its own to
    overlay a roster onto directly, so this maps roster -> team once, up
    front, rather than re-deriving it per grid cell."""
    if not my_roster_ids:
        return set()
    teams = (
        players_dim.filter(pl.col("player_id").is_in(list(my_roster_ids)))["team"]
        .drop_nulls()
        .unique()
        .to_list()
    )
    return set(teams)


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _interpolate(color_a: str, color_b: str, t: float) -> str:
    a, b = _hex_to_rgb(color_a), _hex_to_rgb(color_b)
    t = min(max(t, 0.0), 1.0)
    r, g, bl = (round(a[i] + (b[i] - a[i]) * t) for i in range(3))
    return f"#{r:02x}{g:02x}{bl:02x}"


def diverging_color(value: float, bound: float) -> str:
    """One cell's real colour on the blue/gray/red diverging scale.
    `bound` -- the grid's own real max |value|, computed once by the
    caller across every cell so each position group's own real scale
    (QB rates run much smaller than WR rates) maps to full saturation on
    its own terms rather than a single hardcoded scale that would wash
    out a narrower-range position group."""
    if bound <= 0:
        return _NEUTRAL_COLOR
    t = min(max(value / bound, -1.0), 1.0)
    if t >= 0:
        return _interpolate(_NEUTRAL_COLOR, _EASY_COLOR, t)
    return _interpolate(_NEUTRAL_COLOR, _HARD_COLOR, -t)


def _cell_text(value: float | None, is_confident: object) -> str:
    if value is None:
        return "bye"
    # a trailing "*" marks a real value that's blocked for low confidence
    # (distinct from a bye) without hiding the number itself -- the same
    # "show the real number, mark it, don't erase it" spirit as SPEC's own
    # "grey out... rather than displaying a confident-looking colour"
    # (greying the colour, not the underlying fact).
    return f"{value:.3f}" if is_confident is True else f"{value:.3f}*"


def style_schedule_grid(grid: pl.DataFrame, confidence: pl.DataFrame) -> Styler:
    """Pandas Styler for the schedule-grid heatmap (SPEC §14.5). Converts
    to pandas at the Streamlit-styling boundary only, matching
    `draft_board_page.style_tier_breaks`'s own precedent (task 0.13/0.14)
    -- `st.dataframe`'s cell styling is pandas-Styler-based, not
    polars-native. `grid` and `confidence` are `tools.sos.schedule_grid`/
    `.schedule_grid_confidence`'s own same-shaped pivots (same team rows,
    same week columns) -- combined here into one styled view rather than
    two separate tables the page would have to reconcile itself.

    Builds the *displayed text* as plain strings up front (`"bye"` /
    `"0.123"` / `"0.123*"`) rather than relying on `Styler.format`'s own
    `na_rep`/numeric formatting -- confirmed live that `st.dataframe`
    only honours a `Styler`'s per-cell background/text *colour*, not its
    `.format()` output, so a null cell rendered as Python's own `None`
    literal instead of the intended "bye" label. Pre-formatting the cell
    text itself sidesteps that Streamlit-specific rendering gap entirely.
    """
    week_columns = [c for c in grid.columns if c != "team"]
    values = [v for c in week_columns for v in grid[c].to_list() if v is not None]
    bound = max((abs(v) for v in values), default=0.0)

    confidence_by_team = {
        row["team"]: {c: row[c] for c in week_columns} for row in confidence.to_dicts()
    }

    css_by_team: dict[object, dict[str, str]] = {}
    text_by_team: dict[object, dict[str, str]] = {}
    for row in grid.to_dicts():
        team = row["team"]
        conf_row = confidence_by_team.get(team, {})
        css_row: dict[str, str] = {}
        text_row: dict[str, str] = {}
        for col in week_columns:
            value = row[col]
            is_confident = conf_row.get(col)
            if value is None or is_confident is not True:
                css_row[col] = f"background-color: {_BLOCKED_COLOR}; color: {_TEXT_COLOR}"
            else:
                css_row[col] = (
                    f"background-color: {diverging_color(value, bound)}; color: {_TEXT_COLOR}"
                )
            text_row[col] = _cell_text(value, is_confident)
        css_by_team[team] = css_row
        text_by_team[team] = text_row

    display_pd = pd.DataFrame.from_dict(text_by_team, orient="index", columns=week_columns)
    display_pd.index.name = "team"

    def _row_style(row: pd.Series) -> list[str]:
        team_css = css_by_team.get(row.name, {})
        blocked = f"background-color: {_BLOCKED_COLOR}; color: {_TEXT_COLOR}"
        return [team_css.get(col, blocked) for col in row.index]

    return display_pd.style.apply(_row_style, axis=1)


__all__ = ["diverging_color", "resolve_my_teams", "style_schedule_grid"]
