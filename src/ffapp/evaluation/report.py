"""Evaluation report generator (SPEC.md §12.6; task 1.17).

Renders `evaluation.metrics`' `MetricResult` list (task 1.13) -- every
metric, per position, per predictor, with observation counts and CI --
into one markdown document, alongside optional feature importances (real
attributes off a fitted `lgb.LGBMRegressor`/`LGBMClassifier`, tasks
1.14-1.16) and calibration curves (`evaluation.metrics.calibration_curve`,
already built by task 1.16). Reports are archived under a timestamped
directory and never overwritten, the same "archive, don't overwrite"
precedent task 1.12's own `<timestamp>` output directories already
establish -- `write_report` takes that directory as a parameter rather
than inventing a second timestamping convention.

Calibration is rendered as a markdown table, not a real plotted image
(confirmed with you rather than guessed: SPEC §12.6 says "calibration
plots," but a markdown table of predicted-vs-actual-vs-n is an honest,
dependency-free stand-in, and `MetricResult`/`calibration_curve` already
carry exactly the numbers such a table needs).
"""

from __future__ import annotations

import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Protocol

from ffapp.config import REPO_ROOT
from ffapp.evaluation.metrics import MetricResult

_METRIC_ORDER_HINT = ["mae", "rmse", "spearman", "start_sit_accuracy"]


class SupportsFeatureImportance(Protocol):
    """The only real interface `extract_feature_importance` needs off a
    fitted booster -- both `lgb.LGBMRegressor` and `lgb.LGBMClassifier`
    satisfy this (confirmed live: `feature_name_`/`feature_importances_`
    are real post-fit attributes on the installed lightgbm version, and
    `feature_name_` carries the real pandas column names `to_feature_frame`
    fit with, not placeholder `Column_N` labels)."""

    feature_name_: Sequence[str]
    feature_importances_: Sequence[float]


def extract_feature_importance(
    booster: SupportsFeatureImportance, *, top_n: int = 20
) -> list[tuple[str, float]]:
    """Pairs a fitted booster's own feature names with its importances,
    sorted descending, capped at `top_n` -- the report only needs the
    features that actually moved the model, not the full column list."""
    pairs = list(zip(booster.feature_name_, booster.feature_importances_, strict=True))
    pairs.sort(key=lambda pair: pair[1], reverse=True)
    return [(name, float(score)) for name, score in pairs[:top_n]]


def current_git_commit() -> str | None:
    """Reused by the `evaluate` CLI command to stamp the report's own
    provenance -- same subprocess call `draft.board._current_git_commit`
    already makes for the draft board's output, duplicated rather than
    factored into a shared module for two call sites (CLAUDE.md's
    no-premature-abstraction rule)."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return result.stdout.strip()
    except Exception:
        return None


def _metric_sort_key(metric_name: str) -> tuple[int, str]:
    if metric_name in _METRIC_ORDER_HINT:
        return (_METRIC_ORDER_HINT.index(metric_name), metric_name)
    return (len(_METRIC_ORDER_HINT), metric_name)


def _fmt(value: float) -> str:
    if value != value:  # noqa: PLR0124 -- the standard nan check, no numpy import needed here
        return "nan"
    return f"{value:.3f}"


def _metrics_table(rows: Sequence[MetricResult]) -> list[str]:
    lines = [
        "| Position | Scope | Predictor | Value | N | 95% CI |",
        "|---|---|---|---|---|---|",
    ]
    ordered = sorted(
        rows,
        key=lambda r: (r.position or "ALL", r.scope, r.predictor),
    )
    for row in ordered:
        position = row.position or "ALL"
        ci = f"[{_fmt(row.ci_low)}, {_fmt(row.ci_high)}]"
        lines.append(
            f"| {position} | {row.scope} | {row.predictor} | {_fmt(row.value)} "
            f"| {row.n_obs} | {ci} |"
        )
    return lines


def _feature_importance_section(
    feature_importances: Mapping[str, Sequence[tuple[str, float]]],
) -> list[str]:
    lines = ["## Feature importances", ""]
    for name in sorted(feature_importances):
        lines.append(f"### {name}")
        lines.append("")
        lines.append("| Feature | Importance |")
        lines.append("|---|---|")
        for feature, importance in feature_importances[name]:
            lines.append(f"| {feature} | {_fmt(importance)} |")
        lines.append("")
    return lines


def _calibration_section(
    calibration_curves: Mapping[str, Sequence[tuple[float, float, int]]],
) -> list[str]:
    lines = ["## Calibration", ""]
    for name in sorted(calibration_curves):
        lines.append(f"### {name}")
        lines.append("")
        lines.append("| Predicted | Actual | N |")
        lines.append("|---|---|---|")
        for predicted, actual, n in calibration_curves[name]:
            lines.append(f"| {_fmt(predicted)} | {_fmt(actual)} | {n} |")
        lines.append("")
    return lines


def render_report_markdown(
    *,
    seasons: Sequence[int],
    generated_at: str,
    git_commit: str | None,
    metrics: Sequence[MetricResult],
    feature_importances: Mapping[str, Sequence[tuple[str, float]]] | None = None,
    calibration_curves: Mapping[str, Sequence[tuple[float, float, int]]] | None = None,
) -> str:
    """SPEC §12.6: "every metric... per position, versus every baseline...
    feature importances... calibration plots." `metrics` already carries
    every baseline/model comparison via its own `predictor` column
    (task 1.13) -- there is no separate "baseline comparison" section,
    the metrics table itself is the comparison, grouped by predictor.
    """
    lines = [
        "# Evaluation report",
        "",
        f"**Seasons:** {', '.join(str(s) for s in seasons)}",
        f"**Generated at:** {generated_at}",
        f"**Git commit:** {git_commit or 'unknown'}",
        "",
    ]

    if not metrics:
        lines.append("No metrics to report -- the predictions table was empty.")
    else:
        by_metric: dict[str, list[MetricResult]] = {}
        for row in metrics:
            by_metric.setdefault(row.metric, []).append(row)
        for metric_name in sorted(by_metric, key=_metric_sort_key):
            lines.append(f"## {metric_name}")
            lines.append("")
            lines.extend(_metrics_table(by_metric[metric_name]))
            lines.append("")

    if feature_importances:
        lines.extend(_feature_importance_section(feature_importances))

    if calibration_curves:
        lines.extend(_calibration_section(calibration_curves))

    return "\n".join(lines) + "\n"


def write_report(output_dir: Path, markdown: str) -> Path:
    """Writes `markdown` to `report.md` inside `output_dir`, creating it
    if needed. `output_dir` is the caller's own timestamped directory
    (e.g. the same one `ffapp evaluate` already wrote
    `predictions.parquet` into) -- this function has no timestamping
    opinion of its own."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "report.md"
    path.write_text(markdown, encoding="utf-8")
    return path


__all__ = [
    "SupportsFeatureImportance",
    "current_git_commit",
    "extract_feature_importance",
    "render_report_markdown",
    "write_report",
]
