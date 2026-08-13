"""Task 1.14's depth-chart feature pipeline (SPEC §11.2), the gap left by
task 0.x's own ingest-only `fetch_depth_charts`."""

from __future__ import annotations

import polars as pl

from ffapp.features import depth_chart
from ffapp.features.registry import FeatureSpec

# --- fixtures ---------------------------------------------------------------------


def _depth_chart_row(**kwargs: object) -> dict:
    row: dict[str, object] = {
        "season": 2025,
        "week": 1,
        "club_code": "KC",
        "gsis_id": "p1",
        "position": "RB",
        "depth_position": "RB",
        "depth_team": "1",
        "formation": "Offense",
    }
    row.update(kwargs)
    return row


# --- normalize_depth_charts ----------------------------------------------------------


def test_normalize_depth_charts_keeps_offense_rows_only() -> None:
    raw = pl.DataFrame(
        [
            _depth_chart_row(gsis_id="p1", depth_team="1", formation="Offense"),
            _depth_chart_row(gsis_id="p1", depth_team="1", formation="Special Teams"),
        ]
    )

    result = depth_chart.normalize_depth_charts(raw)

    assert result.height == 1
    assert result["depth_chart_rank"].to_list() == [1]


def test_normalize_depth_charts_takes_the_lowest_rank_when_a_player_has_two_offense_rows() -> None:
    """A player eligible at more than one offensive slot the same week
    (e.g. a WR also listed as an emergency RB) -- their most senior real
    role (lowest depth_team number) wins."""
    raw = pl.DataFrame(
        [
            _depth_chart_row(gsis_id="p1", position="WR", depth_position="WR", depth_team="1"),
            _depth_chart_row(gsis_id="p1", position="RB", depth_position="RB", depth_team="3"),
        ]
    )

    result = depth_chart.normalize_depth_charts(raw)

    assert result.height == 1
    assert result["depth_chart_rank"].to_list() == [1]


def test_normalize_depth_charts_drops_rows_with_no_real_gsis_id() -> None:
    raw = pl.DataFrame([_depth_chart_row(gsis_id=None)])

    result = depth_chart.normalize_depth_charts(raw)

    assert result.is_empty()


def test_normalize_depth_charts_is_one_row_per_player_season_week() -> None:
    raw = pl.DataFrame(
        [
            _depth_chart_row(gsis_id="p1", week=1, depth_team="2"),
            _depth_chart_row(gsis_id="p1", week=2, depth_team="1"),
        ]
    )

    result = depth_chart.normalize_depth_charts(raw).sort("week")

    assert result["week"].to_list() == [1, 2]
    assert result["depth_chart_rank"].to_list() == [2, 1]


# --- add_depth_chart_position ---------------------------------------------------------


def test_add_depth_chart_position_joins_directly_onto_the_target_week() -> None:
    """No lag shift -- week 2's own real depth chart, not week 1's."""
    grid = pl.DataFrame({"player_id": ["p1", "p1"], "season": [2025, 2025], "week": [1, 2]})
    depth_charts = pl.DataFrame(
        [
            _depth_chart_row(gsis_id="p1", week=1, depth_team="2"),
            _depth_chart_row(gsis_id="p1", week=2, depth_team="1"),
        ]
    )

    result = depth_chart.add_depth_chart_position(grid, depth_charts).sort("week")

    assert result["depth_chart_rank"].to_list() == [2, 1]


def test_add_depth_chart_position_is_honestly_null_when_the_player_is_absent_that_week() -> None:
    grid = pl.DataFrame({"player_id": ["p1"], "season": [2025], "week": [1]})
    depth_charts = pl.DataFrame([_depth_chart_row(gsis_id="p2", week=1)])

    result = depth_chart.add_depth_chart_position(grid, depth_charts)

    assert result["depth_chart_rank"].to_list() == [None]


# --- build_depth_chart_features --------------------------------------------------------


def test_build_depth_chart_features_registers_a_training_safe_spec() -> None:
    grid = pl.DataFrame({"player_id": ["p1"], "season": [2025], "week": [1]})
    depth_charts = pl.DataFrame([_depth_chart_row(gsis_id="p1")])
    registry: dict[str, FeatureSpec] = {}

    depth_chart.build_depth_chart_features(grid, depth_charts, registry=registry)

    spec = registry["depth_chart_rank"]
    assert spec.lag_weeks >= 1
    assert spec.available_at_inference is True
    assert spec.source_table == depth_chart.SOURCE_TABLE
