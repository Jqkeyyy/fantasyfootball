"""Cached inputs for a compact, phone-first in-season dashboard."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import polars as pl
import streamlit as st

from ffapp.app.weekly_actions_page import waiver_recommendations
from ffapp.app.weekly_rankings_page import build_weekly_rankings
from ffapp.config import load_league, load_settings
from ffapp.draft.pick_order import resolve_my_roster_id
from ffapp.ids import mapping
from ffapp.ingest import nflverse, sleeper
from ffapp.league_format import parse_league_format
from ffapp.sim.lineup import slot_instances


@st.cache_data(ttl=300, show_spinner="Loading your week…")
def home_data(slug: str) -> dict[str, Any]:
    settings, league = load_settings(), load_league(slug)
    root = settings.data_root
    output = root / "outputs" / slug
    projections = pl.read_parquet(output / "projections.parquet")
    season, week = (
        projections.select("season", "week").unique().sort("season", "week", descending=True).row(0)
    )
    if not league.league_id or not settings.sleeper_username:
        raise ValueError("League or Sleeper user is not configured")
    players = mapping.build_players_dim(
        nflverse.fetch_player_ids(offline=True, settings=settings),
        sleeper.fetch_players(offline=True, settings=settings),
        mapping.ID_OVERRIDES_PATH,
    )
    ids = dict(players.select("sleeper_id", "player_id").drop_nulls().iter_rows())
    rosters = json.loads(
        sleeper.fetch_rosters(league.league_id, offline=True, settings=settings).read_text()
    )
    user = json.loads(
        sleeper.fetch_user(settings.sleeper_username, offline=True, settings=settings).read_text()
    )
    roster_id = resolve_my_roster_id(str(user["user_id"]), rosters)
    mine = next(r for r in rosters if r["roster_id"] == roster_id)
    my_ids = {ids[p] for p in mine.get("players", []) if p in ids}
    all_ids = {ids[p] for r in rosters for p in (r.get("players") or []) if p in ids}
    schedule = pl.read_parquet(root / "interim" / "schedule.parquet")
    rankings = build_weekly_rankings(
        projections,
        pl.read_parquet(root / "features" / "player_week_features.parquet"),
        schedule,
        players,
        season=season,
        week=week,
        my_roster_ids=my_ids,
        rostered_ids=all_ids,
    )
    by_id = {r["player_id"]: r for r in rankings.to_dicts()}
    kickoff = {}
    for r in schedule.filter((pl.col("season") == season) & (pl.col("week") == week)).to_dicts():
        if r.get("kickoff_utc"):
            for side in ["home_team", "away_team"]:
                kickoff[r[side]] = datetime.fromisoformat(r["kickoff_utc"].replace("Z", "+00:00"))
    fmt = parse_league_format(league)
    eligible = dict(slot_instances(fmt))
    starters = [ids.get(p) for p in (mine.get("starters") or [])]
    bench = my_ids - set(starters)
    positions = [
        p for p in league.league_cache.get("roster_positions", []) if p not in {"BN", "IR"}
    ]
    counts: dict[str, int] = {}
    swaps = []
    now = datetime.now(UTC)
    for raw_pos, player_id in zip(positions, starters, strict=False):
        pos = "DST" if raw_pos == "DEF" else raw_pos
        counts[pos] = counts.get(pos, 0) + 1
        # An empty slot (Sleeper's "0") scores nothing, so any eligible bench
        # player is an upgrade over it.
        current = (
            {"player_name": "your empty spot", "proj_mean": 0.0, "team": None}
            if player_id is None
            else by_id.get(player_id)
        )
        if not current or current.get("proj_mean") is None:
            continue
        if current.get("team") and kickoff.get(current.get("team"), now) <= now:
            continue
        for candidate_id in bench:
            candidate = by_id.get(candidate_id)
            if not candidate or candidate.get("proj_mean") is None:
                continue
            if (
                candidate["position"] not in eligible.get(f"{pos}_{counts[pos]}", [])
                or kickoff.get(candidate.get("team"), now) <= now
            ):
                continue
            gain = candidate["proj_mean"] - current["proj_mean"]
            if gain >= 1:
                swaps.append(
                    {
                        "start": candidate["player_name"],
                        "sit": current["player_name"],
                        "gain": gain,
                        "incoming": candidate_id,
                        "outgoing": player_id or f"empty:{pos}_{counts[pos]}",
                        "empty": player_id is None,
                    }
                )
    chosen = []
    used = set()
    for swap in sorted(swaps, key=lambda r: r["gain"], reverse=True):
        if swap["incoming"] not in used and swap["outgoing"] not in used:
            chosen.append(swap)
            used.update([swap["incoming"], swap["outgoing"]])
    remaining = max(
        0, (fmt.waiver_budget or 0) - mine.get("settings", {}).get("waiver_budget_used", 0)
    )
    waivers = waiver_recommendations(
        rankings,
        my_ids,
        fmt,
        current_week=week,
        remaining_budget=remaining,
        playoff_weight=settings.waivers.playoff_weight,
        aggressiveness=settings.waivers.aggressiveness,
        limit=1,
        candidates_per_position=5,
    )
    matchup: dict[str, Any] = {}
    try:
        matches = json.loads(
            sleeper.fetch_matchups(
                league.league_id, week, offline=True, settings=settings
            ).read_text()
        )
        ours = next(r for r in matches if r["roster_id"] == roster_id)
        opponents = [
            r
            for r in matches
            if ours.get("matchup_id") is not None
            and r.get("matchup_id") == ours["matchup_id"]
            and r["roster_id"] != roster_id
        ]
        matchup = {"ours": ours.get("points", 0), "opponent": None}
        if len(opponents) == 1:
            opponent = opponents[0]
            matchup.update(
                opponent=opponent.get("points", 0), opponent_name=f"Team {opponent['roster_id']}"
            )
            try:
                users = json.loads(
                    sleeper.fetch_users(
                        league.league_id, offline=True, settings=settings
                    ).read_text()
                )
                owner = next(r for r in rosters if r["roster_id"] == opponent["roster_id"]).get(
                    "owner_id"
                )
                other = next(u for u in users if u["user_id"] == owner)
                matchup["opponent_name"] = (other.get("metadata") or {}).get(
                    "team_name"
                ) or other.get("display_name")
            except (OSError, StopIteration):
                pass
    except (OSError, StopIteration):
        pass

    def read_json(name: str) -> dict[str, Any]:
        path = output / name
        return json.loads(path.read_text()) if path.exists() else {}

    return {
        "season": season,
        "week": week,
        "swaps": chosen[:3],
        "waiver": waivers.row(0, named=True) if not waivers.is_empty() else None,
        "matchup": matchup,
        "refresh": read_json("refresh_runs/latest.json"),
        "gameday": read_json("gameday.json"),
        "alerts": read_json("alerts/latest.json"),
        "projection_as_of": projections.filter(
            (pl.col("season") == season) & (pl.col("week") == week)
        )["as_of_utc"].max(),
    }


def render_home(slug: str) -> None:
    st.title("Your Week")
    try:
        data = home_data(slug)
    except (OSError, ValueError, StopIteration) as exc:
        st.warning(f"Your dashboard is waiting for refreshed league data: {exc}")
        st.page_link("pages/1_Weekly_Actions.py", label="Open Weekly Actions")
        return
    st.caption(f"Season {data['season']} · Week {data['week']}")
    refresh = data["refresh"]
    stamp = refresh.get("generated_at_utc")
    if stamp:
        age = (datetime.now(UTC) - datetime.fromisoformat(stamp)).total_seconds() / 3600
        status = refresh.get("status", "unknown")
        message = f"Last full update: {age:.0f} hours ago · {status.title()}"
        (st.success if status == "healthy" and age < 72 else st.warning)(message)
    else:
        st.warning("No successful full refresh has been recorded yet.")
    if data["gameday"]:
        st.caption(
            f"Kickoff check: {data['gameday'].get('checked_at', 'pending')} "
            f"· {data['gameday'].get('status', 'unknown')}"
        )
    st.subheader("Your matchup")
    matchup = data["matchup"]
    if matchup:
        left, right = st.columns(2)
        left.metric("Your points", f"{matchup['ours'] or 0:.1f}")
        if matchup.get("opponent") is not None:
            right.metric(matchup["opponent_name"], f"{matchup['opponent'] or 0:.1f}")
        st.caption("Last cached Sleeper scores; updated by full refreshes and kickoff checks.")
        st.page_link(
            "pages/13_Live_Matchup.py",
            label="Open live matchup",
            icon=":material/sports_football:",
        )
    else:
        st.info("No cached matchup is available for this week.")
    st.subheader("Lineup moves to consider")
    for swap in data["swaps"]:
        with st.container(border=True):
            st.markdown(
                f"**{swap['start']}** into {swap['sit']}"
                if swap.get("empty")
                else f"**{swap['start']}** over {swap['sit']}"
            )
            st.caption(f"Projected gain: +{swap['gain']:.1f} points · both games have not started")
    if not data["swaps"]:
        st.info("No direct swap worth at least one projected point among your unlocked players.")
    st.page_link(
        "pages/1_Weekly_Actions.py",
        label="Open Lineup Decision Center",
        icon=":material/checklist:",
    )
    st.subheader("Top waiver opportunity")
    waiver = data["waiver"]
    if waiver:
        with st.container(border=True):
            st.markdown(f"**{waiver['player_name']}**")
            st.write(f"+{waiver['value_added_per_week']:.1f} projected lineup points")
            if waiver.get("drop_player"):
                st.caption(f"Possible drop: {waiver['drop_player']}")
    else:
        st.info("No positive lineup upgrade among the current waiver candidates.")
    st.divider()
    left, middle, right, fourth = st.columns(4)
    left.page_link(
        "pages/9_Weekly_Report.py", label="Weekly Model Report", icon=":material/analytics:"
    )
    middle.page_link(
        "pages/10_Alerts.py", label="Set up phone alerts", icon=":material/notifications:"
    )
    right.page_link("pages/12_Operations.py", label="Automation status", icon=":material/settings:")
    fourth.page_link(
        "pages/14_Roster_Strategy.py", label="Roster strategy", icon=":material/account_tree:"
    )
    st.page_link(
        "pages/15_Decision_Learning.py",
        label="Review which recommendations helped",
        icon=":material/psychology:",
    )
    trade, postgame = st.columns(2)
    trade.page_link(
        "pages/18_Trade_Finder.py",
        label="Find an improving trade",
        icon=":material/swap_horiz:",
    )
    postgame.page_link(
        "pages/17_Postgame_Review.py",
        label="Review last week's misses",
        icon=":material/troubleshoot:",
    )
    with st.expander("Draft & offseason tools"):
        st.page_link("pages/11_Draft_Board.py", label="Draft Board")
        st.page_link("pages/5_Draft_Mobile.py", label="Draft Mobile")
        st.page_link("pages/6_Mock_Draft.py", label="Mock Draft")
