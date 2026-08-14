"""Model health page logic (SPEC.md §12.6, §15; task 2.11). Pure,
pytest-testable functions only -- the real Streamlit page
(`app/pages/4_Model_Health.py`) is verified by actually running it
(CLAUDE.md's UI rule), documented in docs/JOURNAL.md, not here.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ffapp.app.model_health_page import (
    ModelHealthNotBuiltError,
    latest_report,
    list_reports,
    load_report_markdown,
)


def _make_report(eval_dir: Path, timestamp: str, *, content: str = "# report") -> Path:
    report_dir = eval_dir / timestamp
    report_dir.mkdir(parents=True)
    path = report_dir / "report.md"
    path.write_text(content, encoding="utf-8")
    return path


class TestListReports:
    def test_returns_every_real_report_dir_newest_first(self, tmp_path: Path) -> None:
        _make_report(tmp_path, "20260101T000000Z")
        _make_report(tmp_path, "20260813T183533Z")
        _make_report(tmp_path, "20260601T120000Z")

        reports = list_reports(tmp_path)

        assert [r.parent.name for r in reports] == [
            "20260813T183533Z",
            "20260601T120000Z",
            "20260101T000000Z",
        ]

    def test_a_directory_with_no_report_md_is_excluded(self, tmp_path: Path) -> None:
        _make_report(tmp_path, "20260813T183533Z")
        (tmp_path / "20260101T000000Z").mkdir(parents=True)  # no report.md inside

        reports = list_reports(tmp_path)

        assert len(reports) == 1

    def test_a_missing_eval_dir_returns_an_empty_list_not_a_crash(self, tmp_path: Path) -> None:
        reports = list_reports(tmp_path / "does_not_exist")

        assert reports == []

    def test_an_empty_eval_dir_returns_an_empty_list(self, tmp_path: Path) -> None:
        reports = list_reports(tmp_path)

        assert reports == []


class TestLatestReport:
    def test_returns_the_newest_real_report(self, tmp_path: Path) -> None:
        _make_report(tmp_path, "20260101T000000Z")
        newest = _make_report(tmp_path, "20260813T183533Z")

        result = latest_report(tmp_path)

        assert result == newest

    def test_returns_none_when_no_reports_exist(self, tmp_path: Path) -> None:
        result = latest_report(tmp_path)

        assert result is None


class TestLoadReportMarkdown:
    def test_loads_a_real_reports_content(self, tmp_path: Path) -> None:
        path = _make_report(
            tmp_path, "20260813T183533Z", content="# Evaluation report\n\nreal content"
        )

        content = load_report_markdown(path)

        assert content == "# Evaluation report\n\nreal content"

    def test_raises_a_named_error_when_missing(self, tmp_path: Path) -> None:
        with pytest.raises(ModelHealthNotBuiltError, match="ffapp evaluate"):
            load_report_markdown(tmp_path / "report.md")
