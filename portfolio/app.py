"""Self-contained public showcase. Run: streamlit run portfolio/app.py."""

from itertools import combinations
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Fantasy Football | Portfolio Demo", page_icon="🏈", layout="wide")


def best_lineup(players):
    """Enumerate a small sample lineup: QB, RB, WR, TE, FLEX (RB/WR/TE)."""
    best_score, best = 0.0, []
    for group in combinations(players.to_dict("records"), 5):
        positions = [p["position"] for p in group]
        if (
            positions.count("QB") == 1
            and all(p in positions for p in ("RB", "WR", "TE"))
            and all(p in ("QB", "RB", "WR", "TE") for p in positions)
        ):
            score = sum(p["projection"] for p in group)
            if score > best_score:
                best_score, best = score, list(group)
    return best_score, pd.DataFrame(best)


players = pd.read_csv(Path(__file__).with_name("players.csv"))
st.sidebar.title("🏈 FF Command Center")
st.sidebar.caption("Fantasy football decision support")
page = st.sidebar.radio("Explore", ["Overview", "Rankings", "Start / sit", "Trade explorer"])
st.sidebar.divider()
st.sidebar.info("Fictional sample league · Week 4\n\nAll names and numbers are illustrative.")
st.sidebar.link_button("View source on GitHub", "https://github.com/Jqkeyyy/fantasyfootball")
st.title("Make every roster decision count.")
st.caption("PORTFOLIO DEMO · Python · Data pipelines · Projections · Decision support")

if page == "Overview":
    st.write(
        "Explore how league-aware projections can turn football data into actionable "
        "rankings, lineup choices, and trade comparisons."
    )
    a, b, c = st.columns(3)
    a.metric("Sample players", len(players))
    b.metric("Sample teams", 2)
    c.metric("Scoring", "PPR")
    st.subheader("Try a decision")
    st.write(
        "Use **Rankings** to find positional value, compare a steady starter with a "
        "high-upside option in **Start / sit**, or exchange players in **Trade explorer** "
        "to see how roster depth affects each team's best lineup."
    )
    st.subheader("Behind the full project")
    st.markdown(
        "- **Data:** Sleeper league information, nflverse history, and projection sources.\n"
        "- **Modeling:** league-specific scoring, availability estimates, quantiles, "
        "and Monte Carlo simulations.\n"
        "- **Evaluation:** walk-forward testing and saved pregame predictions.\n"
        "- **Delivery:** precomputed artifacts served through a Streamlit dashboard."
    )
    st.info(
        "This lightweight demo uses bundled fictional projections and simplified calculations. "
        "It does not run the full project's trained models, matchup simulations, or live feeds. "
        "Sample ranges are illustrative, not measured model accuracy."
    )
elif page == "Rankings":
    st.subheader("Weekly rankings")
    positions = st.multiselect(
        "Positions", ["QB", "RB", "WR", "TE"], default=["QB", "RB", "WR", "TE"]
    )
    query = st.text_input("Find a player")
    shown = players[
        players.position.isin(positions) & players.name.str.contains(query, case=False, regex=False)
    ]
    shown = shown.sort_values("projection", ascending=False)
    st.dataframe(
        shown[["name", "position", "roster", "projection", "low", "high"]],
        hide_index=True,
        width="stretch",
    )
    st.caption("Points use a fictional PPR snapshot. Low/high values illustrate uncertainty.")
    st.download_button(
        "Download these rankings", shown.to_csv(index=False), "demo-rankings.csv", "text/csv"
    )
elif page == "Start / sit":
    st.subheader("Compare two flex options")
    options = players[players.position.isin(["RB", "WR", "TE"])].set_index("name")
    a, b = st.columns(2)
    first = a.selectbox("First player", options.index)
    second = b.selectbox("Second player", [n for n in options.index if n != first])
    for col, name in [(a, first), (b, second)]:
        row = options.loc[name]
        col.metric(name, f"{row.projection:.1f} pts")
        col.caption(f"{row.position} · Illustrative range: {row.low:.1f}–{row.high:.1f} pts")
    comparison = options.loc[[first, second], ["low", "projection", "high"]]
    st.bar_chart(comparison)
    delta = options.loc[first, "projection"] - options.loc[second, "projection"]
    if abs(delta) < 0.001:
        st.info("These options have equal projected points in the sample snapshot.")
    else:
        st.success(f"{first if delta > 0 else second} leads by {abs(delta):.1f} projected points.")
    st.caption(
        "This comparison ranks expected points. The full app also evaluates matchup win "
        "probability using lineup simulation and player correlations."
    )
else:
    st.subheader("Trade explorer")
    st.write(
        "Each sample lineup starts 1 QB, 1 RB, 1 WR, 1 TE, and 1 FLEX (RB/WR/TE). "
        "Compare the best projected starting lineups before and after a trade."
    )
    a, b = st.columns(2)
    give = a.multiselect(
        "Harbor Hawks send", players[players.roster == "Harbor Hawks"].name, default=["Theo Brooks"]
    )
    receive = b.multiselect(
        "Prairie Foxes send",
        players[players.roster == "Prairie Foxes"].name,
        default=["Miles Reed"],
    )
    if not give or not receive:
        st.info("Choose at least one player from each team to compare a trade.")
    else:
        traded = players.copy()
        traded.loc[traded.name.isin(give), "roster"] = "Prairie Foxes"
        traded.loc[traded.name.isin(receive), "roster"] = "Harbor Hawks"
        for col, team in [(a, "Harbor Hawks"), (b, "Prairie Foxes")]:
            before, _ = best_lineup(players[players.roster == team])
            after, lineup = best_lineup(traded[traded.roster == team])
            if lineup.empty:
                col.warning(f"{team} cannot fill all starting positions after this trade.")
            else:
                col.metric(f"{team}: after trade", f"{after:.1f} pts", f"{after - before:+.1f} pts")
                col.caption(f"Before trade: {before:.1f} projected points")
                col.dataframe(lineup[["name", "position", "projection"]], hide_index=True)
        st.caption(
            "One-week starting-lineup impact only; this excludes schedule, injuries, "
            "bench insurance, and playoff odds. Changes stay in your browser session."
        )
