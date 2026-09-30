"""Live score, projected finish, win odds, and remaining-player command center."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import polars as pl
import streamlit as st

from ffapp.app.league_selector import select_league
from ffapp.app.live_matchup_page import matchup_projection
from ffapp.app.weekly_actions_page import recommended_lineup
from ffapp.app.weekly_rankings_page import build_weekly_rankings
from ffapp.config import load_settings
from ffapp.draft.pick_order import resolve_my_roster_id
from ffapp.ids import mapping
from ffapp.ingest import nflverse, sleeper
from ffapp.league_format import parse_league_format

st.set_page_config(page_title="Live Matchup", layout="wide")
settings = load_settings()
league = select_league()
st.title("Live Matchup Command Center")
st.caption(f"{league.display_name} · live Sleeper score plus projections for games not started")

if league.league_id is None or settings.sleeper_username is None:
    st.error("Sleeper league ID and username are required.")
    st.stop()

try:
    projections = pl.read_parquet(
        settings.data_root / "outputs" / league.slug / "projections.parquet"
    )
    season, week = (
        projections.select("season", "week")
        .unique()
        .sort(["season", "week"], descending=True)
        .row(0)
    )
    features = pl.read_parquet(settings.data_root / "features" / "player_week_features.parquet")
    schedule = pl.read_parquet(settings.data_root / "interim" / "schedule.parquet")
    players = mapping.build_players_dim(
        nflverse.fetch_player_ids(offline=True, settings=settings),
        sleeper.fetch_players(offline=True, settings=settings),
        mapping.ID_OVERRIDES_PATH,
    )
    user = json.loads(
        sleeper.fetch_user(settings.sleeper_username, offline=True, settings=settings).read_text()
    )
    rosters = json.loads(
        sleeper.fetch_rosters(league.league_id, offline=True, settings=settings).read_text()
    )
    my_roster_id = resolve_my_roster_id(str(user["user_id"]), rosters)
except (OSError, ValueError, StopIteration) as exc:
    st.error(f"Live matchup data is unavailable: {exc}")
    st.stop()

if st.button("Refresh live score", type="primary"):
    st.cache_data.clear()

try:
    matchups = json.loads(
        sleeper.fetch_matchups(league.league_id, week, offline=False, settings=settings).read_text()
    )
except Exception as exc:
    st.warning(f"Using the most recently cached score because Sleeper could not refresh: {exc}")
    matchups = json.loads(
        sleeper.fetch_matchups(league.league_id, week, offline=True, settings=settings).read_text()
    )

mine = next((row for row in matchups if row.get("roster_id") == my_roster_id), None)
opponent = next(
    (
        row
        for row in matchups
        if mine
        and row.get("matchup_id") == mine.get("matchup_id")
        and row.get("roster_id") != my_roster_id
    ),
    None,
)
if mine is None or opponent is None:
    st.info("Your opponent has not been assigned for this week yet.")
    st.stop()

sleeper_to_player = dict(players.select("sleeper_id", "player_id").drop_nulls().iter_rows())


def canonical(values: object) -> list[str]:
    return (
        [sleeper_to_player[str(value)] for value in values if str(value) in sleeper_to_player]
        if isinstance(values, list)
        else []
    )


all_rostered = {pid for roster in rosters for pid in canonical(roster.get("players"))}
my_roster = next(row for row in rosters if row.get("roster_id") == my_roster_id)
my_ids = set(canonical(my_roster.get("players")))
rankings = build_weekly_rankings(
    projections,
    features,
    schedule,
    players,
    season=season,
    week=week,
    my_roster_ids=my_ids,
    rostered_ids=all_rostered,
)
kickoffs: dict[str, datetime] = {}
for game in schedule.filter((pl.col("season") == season) & (pl.col("week") == week)).iter_rows(
    named=True
):
    raw = game.get("kickoff_utc")
    if isinstance(raw, str):
        kickoff = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        for key in ("home_team", "away_team"):
            if game.get(key):
                kickoffs[str(game[key])] = kickoff

points: dict[str, float] = {}
for payload in (mine, opponent):
    raw_points = payload.get("players_points") or {}
    if isinstance(raw_points, dict):
        points.update(
            {
                sleeper_to_player[str(pid)]: float(value or 0)
                for pid, value in raw_points.items()
                if str(pid) in sleeper_to_player
            }
        )

result = matchup_projection(
    rankings,
    canonical(mine.get("starters")),
    canonical(opponent.get("starters")),
    points,
    kickoffs,
    now=datetime.now(UTC),
)
ours, theirs = result["mine"], result["opponent"]
a, b, c, d = st.columns(4)
a.metric("Your live points", f"{float(ours['live']):.1f}")
b.metric("Opponent live points", f"{float(theirs['live']):.1f}")
c.metric(
    "Projected finish",
    f"{float(ours['projected_final']):.1f}–{float(theirs['projected_final']):.1f}",
)
d.metric("Win probability", f"{float(result['win_probability']):.0%}")
st.progress(
    float(result["win_probability"]),
    text=f"Projected margin {float(result['projected_margin']):+.1f}",
)
st.caption(
    "Projected final = current Sleeper points + full projections for starters whose games have "
    "not kicked off. In-progress players use their current points only."
)

left, right = st.columns(2)
for column, label, side in (
    (left, "Your players remaining", ours),
    (right, "Opponent players remaining", theirs),
):
    with column:
        st.subheader(label)
        remaining = side["remaining"]
        if remaining:
            st.dataframe(
                pl.DataFrame(remaining).select(
                    "player", "position", "team", "projection", "floor", "ceiling"
                ),
                width="stretch",
                hide_index=True,
            )
        else:
            st.info("No projected starters remain before kickoff.")

st.subheader("Swing players")
boom, bust = st.columns(2)
with boom:
    st.markdown("**Largest remaining ceilings**")
    upside = [dict(row, side="You") for row in ours["upside"]] + [
        dict(row, side="Opponent") for row in theirs["upside"]
    ]
    if upside:
        st.dataframe(
            pl.DataFrame(upside)
            .sort("ceiling", descending=True)
            .select("side", "player", "projection", "ceiling"),
            width="stretch",
            hide_index=True,
        )
with bust:
    st.markdown("**Largest remaining downside ranges**")
    downside = [dict(row, side="You") for row in ours["downside"]] + [
        dict(row, side="Opponent") for row in theirs["downside"]
    ]
    if downside:
        st.dataframe(
            pl.DataFrame(downside)
            .with_columns((pl.col("projection") - pl.col("floor")).alias("downside"))
            .sort("downside", descending=True)
            .select("side", "player", "floor", "projection", "downside"),
            width="stretch",
            hide_index=True,
        )

recommended = recommended_lineup(
    rankings, my_ids, set(canonical(my_roster.get("starters"))), parse_league_format(league)
)
changes = recommended.filter(~pl.col("currently_starting"))
st.subheader("Late lineup check")
if changes.is_empty():
    st.success("Your supported starters already match the recommended lineup.")
else:
    st.warning(
        "Review before kickoff: " + ", ".join(changes["player_name"].cast(pl.String).to_list())
    )
    st.link_button("Open lineup decisions", "/Weekly_Actions", icon=":material/swap_horiz:")
