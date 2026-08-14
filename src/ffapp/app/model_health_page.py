"""Model health page logic (SPEC.md §12.6, §15; task 2.11).

Pure, pytest-testable functions only -- the real Streamlit page
(`app/pages/4_Model_Health.py`) is thin `st.*` glue on top, matching
every other page's own precedent (`draft_board_page.py`,
`weekly_rankings_page.py`, `schedule_grid_page.py`). Reads task 1.17's
own already-built `data/outputs/eval/<timestamp>/report.md` directly --
SPEC §15's own "fast to load... nothing trained on page load" design
constraint, applied here as "nothing re-rendered that's already
rendered."

**No new chart rendering here.** `report.md`'s own calibration section is
already a markdown table, not a plotted image -- a real, confirmed
decision from task 1.17 (see `evaluation.report`'s own module
docstring: "a markdown table of predicted-vs-actual-vs-n is an honest,
dependency-free stand-in"). SPEC §15's "calibration plots" and TASKS.md
2.11's own "calibration plots... visible in the UI" are both satisfied
by surfacing that same table in the browser, not by re-litigating
1.17's already-confirmed representation or building a second one.

**"The latest evaluation report" (SPEC's own words) plus real history,
not just the single newest one** -- SPEC §12.6 itself: "Reports are
kept, not overwritten -- the history of what you tried is the most
valuable artefact of the offseason." `list_reports` surfaces every real
report so the page's own selector can default to latest while still
letting a past run be reopened, directly serving that stated value
rather than only ever showing the newest.
"""

from __future__ import annotations

from pathlib import Path


class ModelHealthNotBuiltError(Exception):
    """No evaluation report exists yet for the given path."""


def list_reports(eval_dir: Path) -> list[Path]:
    """Every real `report.md` under `eval_dir`'s own timestamped
    subdirectories (task 1.17's own `<timestamp>/report.md` layout,
    never overwritten), newest first. Timestamp directory names are
    ISO-8601-basic (`20260813T183533Z`) and sort correctly as plain
    strings -- no date parsing needed. A missing `eval_dir`, an empty
    one, or a stray subdirectory with no real `report.md` inside all
    return an honestly empty/filtered list, never a crash."""
    if not eval_dir.exists():
        return []
    reports = sorted(
        (child / "report.md" for child in eval_dir.iterdir() if child.is_dir()),
        key=lambda p: p.parent.name,
        reverse=True,
    )
    return [r for r in reports if r.exists()]


def latest_report(eval_dir: Path) -> Path | None:
    """The single most recent real report, or `None` when none exist
    yet -- an honest absence, not a guessed path."""
    reports = list_reports(eval_dir)
    return reports[0] if reports else None


def load_report_markdown(path: Path) -> str:
    """Real report content, or a named error pointing at the real fix
    (`ffapp evaluate`) -- matching `weekly_rankings_page
    .ProjectionsNotBuiltError`/`draft_board_page.DraftBoardNotBuiltError`'s
    own convention for "the precomputed artefact this page reads doesn't
    exist yet."""
    if not path.exists():
        raise ModelHealthNotBuiltError(
            f"No evaluation report found at {path}. Run `ffapp evaluate --seasons ...` first."
        )
    return path.read_text(encoding="utf-8")


__all__ = [
    "ModelHealthNotBuiltError",
    "latest_report",
    "list_reports",
    "load_report_markdown",
]
