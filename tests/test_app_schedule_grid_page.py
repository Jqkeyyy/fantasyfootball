"""Schedule grid page composition (SPEC.md §14.5; task 2.8). Pure,
pytest-testable functions only -- the real Streamlit page
(`app/pages/3_Schedule_Grid.py`) is verified by actually running it
(CLAUDE.md's UI rule), documented in docs/JOURNAL.md, not here.
"""

from __future__ import annotations

import polars as pl

from ffapp.app.schedule_grid_page import diverging_color, resolve_my_teams, style_schedule_grid


def _players_dim() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "player_id": ["p1", "p2", "p3", "p4"],
            "team": ["KC", "BUF", "KC", None],
        }
    )


class TestResolveMyTeams:
    def test_maps_roster_players_to_their_real_current_teams(self) -> None:
        teams = resolve_my_teams({"p1", "p2"}, _players_dim())

        assert teams == {"KC", "BUF"}

    def test_dedupes_teammates_on_the_same_roster(self) -> None:
        teams = resolve_my_teams({"p1", "p3"}, _players_dim())

        assert teams == {"KC"}

    def test_a_player_with_no_real_team_contributes_nothing(self) -> None:
        teams = resolve_my_teams({"p4"}, _players_dim())

        assert teams == set()

    def test_an_empty_roster_returns_an_empty_set(self) -> None:
        teams = resolve_my_teams(set(), _players_dim())

        assert teams == set()


class TestDivergingColor:
    def test_zero_value_is_the_neutral_midpoint(self) -> None:
        assert diverging_color(0.0, bound=1.0) == "#f0efec"

    def test_full_positive_value_reaches_the_easy_pole(self) -> None:
        assert diverging_color(1.0, bound=1.0) == "#2a78d6"

    def test_full_negative_value_reaches_the_hard_pole(self) -> None:
        assert diverging_color(-1.0, bound=1.0) == "#e34948"

    def test_a_zero_bound_never_divides_by_zero(self) -> None:
        assert diverging_color(0.5, bound=0.0) == "#f0efec"

    def test_a_value_beyond_the_bound_clips_rather_than_overshoots(self) -> None:
        assert diverging_color(5.0, bound=1.0) == diverging_color(1.0, bound=1.0)


class TestStyleScheduleGrid:
    def _grid_and_confidence(self) -> tuple[pl.DataFrame, pl.DataFrame]:
        grid = pl.DataFrame({"team": ["KC", "BUF"], "1": [0.2, None], "2": [-0.1, 0.05]})
        confidence = pl.DataFrame({"team": ["KC", "BUF"], "1": [True, None], "2": [False, True]})
        return grid, confidence

    def test_returns_a_styler_over_every_team(self) -> None:
        grid, confidence = self._grid_and_confidence()

        styler = style_schedule_grid(grid, confidence)

        assert styler.data.shape[0] == 2

    def test_a_bye_cell_is_blocked(self) -> None:
        grid, confidence = self._grid_and_confidence()

        html = style_schedule_grid(grid, confidence).to_html()

        assert "#e1e0d9" in html  # BUF week 1: a real bye (null in grid)

    def test_a_low_confidence_cell_is_blocked_even_with_a_real_value(self) -> None:
        grid, confidence = self._grid_and_confidence()

        styler = style_schedule_grid(grid, confidence)

        # KC week 2 has a real value (-0.1) but confidence=False -- must
        # still render blocked, not a confident-looking diverging colour.
        # pandas dedupes identical rules into one shared CSS selector, so
        # inspect the computed per-cell styles directly rather than the
        # rendered HTML's own string layout.
        computed = styler._compute()
        kc_week2_css = computed.ctx[(0, 1)]  # row 0 = KC, col 1 = week "2"
        assert ("background-color", "#e1e0d9") in kc_week2_css

    def test_a_confident_real_value_uses_the_diverging_scale_not_blocked(self) -> None:
        grid, confidence = self._grid_and_confidence()

        html = style_schedule_grid(grid, confidence).to_html()

        # KC week 1 (0.2, confident=True) should use a real diverging colour
        expected = diverging_color(0.2, bound=0.2)
        assert expected in html
