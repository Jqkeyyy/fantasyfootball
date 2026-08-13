"""Task 1.17's own literal acceptance bar (SPEC §12.6): a markdown report
with every metric, baseline comparisons, feature importances, and
calibration -- exercised here with small synthetic fixtures, same
fixture-vs-live-run convention as `tests/test_evaluation_metrics.py`.

**Not exercised against real data this session.** This machine has no
`data/` at all (fresh checkout, see HANDOFF.md) and this network blocks
`api.sleeper.app` (Cisco Umbrella), so the real end-to-end run -- a real
`predictions.parquet` from `ffapp evaluate`, real fitted
`FittedPointsModels`/`FittedQuantileModels` boosters for feature
importances, real quantile/availability calibration curves -- is deferred
to a session with both `data/` materialised and Sleeper reachable. Noted
in HANDOFF.md rather than silently skipped.
"""

from __future__ import annotations

from pathlib import Path

from ffapp.evaluation import report
from ffapp.evaluation.metrics import MetricResult

# --- fixtures ---------------------------------------------------------------------


def _metric(
    metric: str = "mae",
    predictor: str = "model",
    position: str | None = "RB",
    scope: str = "all",
    value: float = 3.5,
    n_obs: int = 120,
    ci_low: float = 3.1,
    ci_high: float = 3.9,
) -> MetricResult:
    return MetricResult(
        metric=metric,
        predictor=predictor,
        position=position,
        scope=scope,
        value=value,
        n_obs=n_obs,
        ci_low=ci_low,
        ci_high=ci_high,
    )


class _StubBooster:
    """Stands in for a fitted `lgb.LGBMRegressor`/`LGBMClassifier` --
    both expose real `feature_name_`/`feature_importances_` attributes
    (confirmed live against the installed lightgbm this session), which
    is the only interface `extract_feature_importance` needs."""

    def __init__(self, names: list[str], importances: list[float]) -> None:
        self.feature_name_ = names
        self.feature_importances_ = importances


# --- extract_feature_importance ----------------------------------------------------


def test_extract_feature_importance_pairs_names_with_scores_sorted_descending() -> None:
    booster = _StubBooster(["a", "b", "c"], [1.0, 5.0, 3.0])

    result = report.extract_feature_importance(booster)

    assert result == [("b", 5.0), ("c", 3.0), ("a", 1.0)]


def test_extract_feature_importance_caps_at_top_n() -> None:
    booster = _StubBooster(["a", "b", "c", "d"], [4.0, 3.0, 2.0, 1.0])

    result = report.extract_feature_importance(booster, top_n=2)

    assert result == [("a", 4.0), ("b", 3.0)]


# --- render_report_markdown ---------------------------------------------------------


def test_render_report_markdown_includes_header_seasons_and_git_commit() -> None:
    text = report.render_report_markdown(
        seasons=[2023, 2024],
        generated_at="2026-08-13T00:00:00Z",
        git_commit="abc1234",
        metrics=[_metric()],
    )

    assert "2023" in text
    assert "2024" in text
    assert "2026-08-13T00:00:00Z" in text
    assert "abc1234" in text


def test_render_report_markdown_handles_a_missing_git_commit() -> None:
    text = report.render_report_markdown(
        seasons=[2024],
        generated_at="2026-08-13T00:00:00Z",
        git_commit=None,
        metrics=[_metric()],
    )

    assert "unknown" in text.lower()


def test_render_report_markdown_renders_one_row_per_metric_predictor_position() -> None:
    rows = [
        _metric(metric="mae", predictor="b0", position="RB", value=4.123),
        _metric(metric="mae", predictor="b1", position="RB", value=3.5),
        _metric(metric="mae", predictor="b0", position=None, scope="all", value=5.0),
    ]

    text = report.render_report_markdown(
        seasons=[2024], generated_at="2026-08-13T00:00:00Z", git_commit="abc1234", metrics=rows
    )

    assert "4.123" in text
    assert "3.500" in text
    assert "5.000" in text
    assert "ALL" in text  # position=None rendered as the pooled "ALL" row


def test_render_report_markdown_groups_by_metric_name() -> None:
    rows = [
        _metric(metric="mae"),
        _metric(metric="spearman", value=0.42),
    ]

    text = report.render_report_markdown(
        seasons=[2024], generated_at="2026-08-13T00:00:00Z", git_commit="abc1234", metrics=rows
    )

    assert "## mae" in text
    assert "## spearman" in text


def test_render_report_markdown_omits_feature_importance_section_when_none_given() -> None:
    text = report.render_report_markdown(
        seasons=[2024],
        generated_at="2026-08-13T00:00:00Z",
        git_commit="abc1234",
        metrics=[_metric()],
    )

    assert "Feature importance" not in text


def test_render_report_markdown_includes_feature_importances_when_given() -> None:
    text = report.render_report_markdown(
        seasons=[2024],
        generated_at="2026-08-13T00:00:00Z",
        git_commit="abc1234",
        metrics=[_metric()],
        feature_importances={"RB": [("target_share", 120.0), ("proe", 45.0)]},
    )

    assert "Feature importance" in text
    assert "target_share" in text
    assert "120.000" in text


def test_render_report_markdown_omits_calibration_section_when_none_given() -> None:
    text = report.render_report_markdown(
        seasons=[2024],
        generated_at="2026-08-13T00:00:00Z",
        git_commit="abc1234",
        metrics=[_metric()],
    )

    assert "Calibration" not in text


def test_render_report_markdown_includes_calibration_curve_when_given() -> None:
    text = report.render_report_markdown(
        seasons=[2024],
        generated_at="2026-08-13T00:00:00Z",
        git_commit="abc1234",
        metrics=[_metric()],
        calibration_curves={"availability": [(0.1, 0.12, 40), (0.9, 0.85, 55)]},
    )

    assert "Calibration" in text
    assert "availability" in text
    assert "0.120" in text


def test_render_report_markdown_is_stable_when_no_metrics_survive_filtering() -> None:
    text = report.render_report_markdown(
        seasons=[2024], generated_at="2026-08-13T00:00:00Z", git_commit="abc1234", metrics=[]
    )

    assert "No metrics" in text


# --- write_report ---------------------------------------------------------------------


def test_write_report_writes_report_md_into_the_given_directory(tmp_path: Path) -> None:
    output_dir = tmp_path / "20260813T000000Z"

    written_path = report.write_report(output_dir, "# hello\n")

    assert written_path == output_dir / "report.md"
    assert written_path.read_text(encoding="utf-8") == "# hello\n"


def test_write_report_creates_the_directory_if_missing(tmp_path: Path) -> None:
    output_dir = tmp_path / "nested" / "20260813T000000Z"

    report.write_report(output_dir, "# hello\n")

    assert output_dir.is_dir()
