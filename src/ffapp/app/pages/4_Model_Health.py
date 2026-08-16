"""Model health Streamlit page (SPEC.md §12.6, §15; task 2.11).

Fourth page in SPEC §15's own build order, under `app/pages/` per
Streamlit's own multipage convention (matching `2_Weekly_Rankings.py`/
`3_Schedule_Grid.py`'s own precedent). Reads the pre-built
`data/outputs/eval/<timestamp>/report.md` (`ffapp evaluate`, task 1.17)
directly -- nothing computed on page load, SPEC §15's own "fast to
load" design constraint. Report selection and markdown loading live in
`model_health_page.py`, unit-tested there; this script is thin `st.*`
glue, verified by actually running it (CLAUDE.md's UI rule).

SPEC §15's own reason this page exists: "Being reminded weekly of how
your model is actually performing is the main defence against trusting
it too much." -- so the default view is always the latest real report,
with older ones (SPEC §12.6: "kept, not overwritten") reachable via the
sidebar, not the other way around.

**Live projection source** (SPEC-ADDENDUM-04.md §C, task 1.20): shown
above the report -- which `model.projection_source` is live
(`config/settings.yml`) and its real margin over B2, read from
`config/projection_source_evaluation.yml` (a small, human-curated
summary of `docs/JOURNAL.md`'s 2026-08-16 closing entry, not a live
computation).

Run with: `uv run streamlit run src/ffapp/app/streamlit_app.py`, then
open "Model Health" from the sidebar.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from ffapp.app.model_health_page import (
    ModelHealthNotBuiltError,
    ProjectionSourceEvaluationNotFoundError,
    list_reports,
    load_projection_source_evaluation,
    load_report_markdown,
    projection_source_summary,
)
from ffapp.config import CONFIG_DIR, load_settings

st.set_page_config(page_title="Model Health", layout="wide")


@st.cache_data(show_spinner="Loading evaluation report...")
def _load_report_cached(path_str: str, mtime: float) -> str:
    """`mtime` is only ever used as part of the cache key -- same
    file-mtime-busts-cache substitution `draft_board_page`/
    `weekly_rankings_page` already established (report.md is never
    edited in place after task 1.17's own `write_report`, but the cache
    key convention stays consistent with every other page's own
    precedent regardless)."""
    return load_report_markdown(Path(path_str))


settings = load_settings()
eval_dir = settings.data_root / "outputs" / "eval"

st.title("Model Health")

st.subheader("Live projection source")
live_source = settings.model.projection_source
st.markdown(f"**`{live_source}`** (`config/settings.yml`'s own `model.projection_source`)")
try:
    evaluation = load_projection_source_evaluation(CONFIG_DIR / "projection_source_evaluation.yml")
    summary = projection_source_summary(evaluation, live_source)
    st.markdown(f"**Status:** {summary['status']}")
    st.markdown(f"**Margin over B2:** {summary['margin_over_b2']}")
    st.caption(
        f"Real evaluation as of {evaluation.get('as_of', '?')} -- see "
        "docs/JOURNAL.md's 2026-08-16 closing entry for the full account."
    )
except (ModelHealthNotBuiltError, ProjectionSourceEvaluationNotFoundError) as exc:
    st.warning(str(exc))
st.divider()

reports = list_reports(eval_dir)
if not reports:
    st.error(
        f"No evaluation report found under `{eval_dir}`. Run `ffapp evaluate --seasons ...` first."
    )
    st.stop()

with st.sidebar:
    st.header("Report")
    labels = [r.parent.name for r in reports]
    selected_label = st.selectbox(
        "Evaluation run",
        options=labels,
        help="Reports are kept, never overwritten -- pick an older run to compare.",
    )
selected_path = reports[labels.index(selected_label)]

if selected_label == labels[0]:
    st.caption("Latest evaluation report.")
else:
    st.caption(f"Historical report ({selected_label}) -- not the latest run.")

try:
    content = _load_report_cached(str(selected_path), selected_path.stat().st_mtime)
except ModelHealthNotBuiltError as exc:
    st.error(str(exc))
    st.stop()

st.markdown(content)
