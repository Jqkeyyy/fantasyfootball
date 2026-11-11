from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest
from streamlit.testing.v1 import AppTest

from ffapp.app.performance_trends import performance_trends


def review_rows() -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "season": 2026,
                "week": week,
                "player_id": str(i),
                "position": position,
                "projected": float(110 - i),
                "actual": float(112 - i),
                "status": "Scored",
            }
            for week in [2, 10]
            for i, position in enumerate(["QB", "RB", "WR", "TE", "DEF", "K"] * 18)
        ]
    )


def test_top_100_is_ranked_before_missing_outcomes_and_injury_exclusions() -> None:
    review = review_rows().with_columns(
        pl.when(pl.col("player_id") == "0")
        .then(pl.lit("Awaiting actuals"))
        .when(pl.col("player_id") == "1")
        .then(pl.lit("Injury excluded"))
        .otherwise(pl.col("status"))
        .alias("status"),
        pl.when(pl.col("player_id") == "0").then(None).otherwise(pl.col("actual")).alias("actual"),
    )
    result = performance_trends(review, cohort="Top 100", include_injured=False)
    assert result["selected"].to_list() == [100, 100]
    assert result["matched"].to_list() == [98, 98]
    assert result["coverage"].to_list() == [0.98, 0.98]
    assert result["predicted_total"][0] == sum(float(110 - i) for i in range(2, 100))
    assert result["bias"].to_list() == [-2.0, -2.0]
    assert result["average_miss"].to_list() == [2.0, 2.0]
    assert result["Week"].to_list() == ["2026 W02", "2026 W10"]
    assert performance_trends(review, cohort="Top 100")["matched"].to_list() == [99, 99]


@pytest.mark.parametrize("position", ["QB", "RB", "WR", "TE", "DEFENSE", "K"])
def test_each_position_and_all_players(position) -> None:
    assert performance_trends(review_rows(), cohort=position)["matched"].to_list() == [18, 18]
    assert performance_trends(review_rows())["matched"].to_list() == [108, 108]


def test_unready_and_unverified_rows_are_excluded_and_missing_is_not_zero() -> None:
    rows = review_rows().with_columns(
        pl.when(pl.col("week") == 10)
        .then(pl.lit("No verified pregame snapshot"))
        .otherwise(pl.lit("Awaiting actuals"))
        .alias("status"),
        pl.lit(None, dtype=pl.Float64).alias("actual"),
    )
    result = performance_trends(rows)
    assert result["week"].to_list() == [2]
    assert result["matched"][0] == 0
    assert result["actual_total"][0] is None
    assert result["predicted_total"][0] is None
    assert performance_trends(pl.DataFrame()).is_empty()


def test_page_renders_all_groups_and_controls(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("ffapp.app.ui.apply_app_shell", lambda: None)
    monkeypatch.setattr(
        "ffapp.app.league_selector.select_league",
        lambda: SimpleNamespace(slug="test", display_name="Test League"),
    )
    monkeypatch.setattr("ffapp.config.load_settings", lambda: SimpleNamespace(data_root=tmp_path))
    monkeypatch.setattr(
        "ffapp.evaluation.inseason.load_prediction_history", lambda *a, **k: pl.DataFrame()
    )
    monkeypatch.setattr(
        "ffapp.evaluation.weekly_review.build_weekly_review", lambda *a, **k: review_rows()
    )
    page = Path(__file__).parents[1] / "src/ffapp/app/pages/4_Model_Lab.py"
    app = AppTest.from_file(str(page))
    app.query_params["section"] = "trends"
    app.run(timeout=30)
    assert not app.exception
    assert [tab.label for tab in app.tabs] == ["Top 100", "By position", "Everything"]
    assert len(app.get("vega_lite_chart")) == 16
    app.radio[0].set_value("Total points").run()
    assert not app.exception
    app.toggle[0].set_value(False).run()
    assert not app.exception
