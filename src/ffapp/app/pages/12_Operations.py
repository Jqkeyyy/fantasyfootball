"""Automation health, schedules, run history, and manual controls."""

from datetime import UTC, datetime

import polars as pl
import streamlit as st

from ffapp.app.data_status import next_scheduled_refresh, run_weekly_refresh
from ffapp.app.league_selector import ordered_leagues
from ffapp.app.operations_page import league_operation_row, recent_run_rows
from ffapp.app.ui import apply_app_shell
from ffapp.config import load_all_leagues, load_settings
from ffapp.tools.discord_notifications import configured_webhook
from ffapp.tools.monitoring import host_status, source_status


def _number(value: object) -> float:
    return float(value) if isinstance(value, int | float) else 0.0


st.set_page_config(page_title="Operations", layout="wide")
apply_app_shell()
settings = load_settings()
leagues = ordered_leagues(load_all_leagues())
now = datetime.now(UTC)
next_run, next_label = next_scheduled_refresh(now)
next_run_label = next_run.strftime("%B %d at %I:%M %p").replace(" at 0", " at ")
rows = [league_operation_row(settings, league, now=now) for league in leagues]

st.title("Automation Control Center")
st.caption("Refresh health, data coverage, phone alerts, schedules, and manual recovery.")

healthy = sum(row["status"] == "healthy" for row in rows)
failed = sum(row["status"] == "failed" for row in rows)
oldest = max((_number(row["age_hours"]) for row in rows if row["age_hours"] is not None), default=0)
a, b, c, d = st.columns(4)
a.metric("Healthy leagues", f"{healthy}/{len(rows)}")
b.metric("Failed leagues", str(failed))
c.metric("Oldest full update", f"{oldest:.1f}h")
d.metric("Discord", "Connected" if configured_webhook() else "Not configured")
st.info(
    f"Next full refresh: **{next_label}, {next_run_label} Central**. "
    "Kickoff checks run every five minutes and refresh near 90 and 30 minutes before games."
)

st.subheader("Server and source monitoring")
host = host_status(settings, now=now)
host_a, host_b, host_c, host_d = st.columns(4)
host_a.metric("Dashboard", str(host["dashboard"]))
host_b.metric("Disk used", f"{_number(host['disk_used_pct']):.0%}")
host_c.metric("Disk free", f"{_number(host['disk_free_gb']):.1f} GB")
host_d.metric("GitHub access", str(host["github_token"]))
if host["github_token"] == "Anonymous":
    st.caption(
        "GitHub downloads are unauthenticated. Adding GITHUB_TOKEN on the server raises "
        "API limits and makes source refreshes more reliable."
    )
sources = source_status(settings, now=now)
if not sources.is_empty():
    st.dataframe(
        sources,
        width="stretch",
        hide_index=True,
        column_config={"age_hours": st.column_config.NumberColumn("Age", format="%.1f h")},
    )

st.subheader("League status")
status_table = pl.DataFrame(rows).select(
    "league",
    "status",
    "season",
    "week",
    "age_hours",
    "coverage",
    "decision_alerts",
    "gameday_status",
    "problems",
)
st.dataframe(
    status_table,
    width="stretch",
    hide_index=True,
    column_config={
        "league": "League",
        "status": "Full refresh",
        "season": "Season",
        "week": "Week",
        "age_hours": st.column_config.NumberColumn("Age", format="%.1f h"),
        "coverage": "Projection coverage",
        "decision_alerts": "Open alerts",
        "gameday_status": "Kickoff check",
        "problems": "Needs attention",
    },
)

st.subheader("Manual controls")
st.write("Use these after a source outage, roster change, or unexpected stale page.")
all_col, note_col = st.columns([1, 3])
if all_col.button("Refresh every league", type="primary", width="stretch"):
    with st.spinner(
        "Running the complete refresh for every league. This can take several minutes..."
    ):
        result = run_weekly_refresh(None, timeout_seconds=1200)
    if result.returncode == 0:
        st.cache_data.clear()
        st.success("Every league refreshed successfully.")
        st.rerun()
    else:
        detail = result.stderr.strip() or result.stdout.strip()
        st.error(f"Full refresh failed: {detail[-1600:]}")
note_col.caption(
    "The all-league run downloads shared football sources once, updates each Sleeper league, "
    "rebuilds weekly and rest-of-season projections, and sends actionable Discord alerts."
)

columns = st.columns(min(3, max(1, len(leagues))))
for index, league in enumerate(leagues):
    with columns[index % len(columns)]:
        if st.button(f"Refresh {league.display_name}", key=f"ops_refresh_{league.slug}"):
            with st.spinner(f"Refreshing {league.display_name}..."):
                result = run_weekly_refresh(league.slug)
            if result.returncode == 0:
                st.cache_data.clear()
                st.success("Refresh completed.")
                st.rerun()
            else:
                detail = result.stderr.strip() or result.stdout.strip()
                st.error(f"Refresh failed: {detail[-1200:]}")

st.subheader("Recent run history")
history = [row for league in leagues for row in recent_run_rows(settings, league)]
if history:
    st.dataframe(
        pl.DataFrame(history).sort("generated_at", descending=True),
        width="stretch",
        hide_index=True,
    )
else:
    st.info("No immutable refresh history has been recorded yet.")
