"""Shared visual shell for the Streamlit decision dashboard."""

from __future__ import annotations

import streamlit as st


def _installable_app_metadata() -> None:
    """Attach install metadata to Streamlit's parent document from its same-origin component."""
    st.iframe("/app/static/pwa-bootstrap.html", height=1)


def apply_app_shell() -> None:
    """Apply the app's brand, spacing, cards, and mobile layout once per page."""
    _installable_app_metadata()
    st.markdown(
        """
        <style>
        :root {
            --ff-ink: #172033;
            --ff-muted: #687386;
            --ff-navy: #132238;
            --ff-panel: #ffffff;
            --ff-line: #dfe5ec;
            --ff-green: #13b981;
            --ff-green-soft: #e7f8f1;
        }

        [data-testid="stAppViewContainer"] {
            background:
                radial-gradient(circle at 86% 3%, rgba(19,185,129,.08), transparent 25rem),
                #f7f9fc;
        }
        [data-testid="stHeader"] { background: transparent; }
        [data-testid="stToolbar"] { display: none; }
        .ff-mobile-nav { display: none; }
        .block-container {
            max-width: 1260px;
            padding-top: 2.35rem;
            padding-bottom: 4rem;
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #132238 0%, #172a43 100%);
            border-right: 0;
        }
        [data-testid="stSidebarNav"]::before {
            content: "FF  COMMAND CENTER";
            display: block;
            color: #ffffff;
            font-size: .78rem;
            font-weight: 800;
            letter-spacing: .12em;
            padding: .8rem 1rem 1rem;
        }
        [data-testid="stSidebarNavLink"] p { color: #d8e1ec; font-weight: 550; }
        [data-testid="stSidebarNavLink"][aria-current="page"] {
            background: rgba(19,185,129,.18);
            border-left: 3px solid var(--ff-green);
        }
        [data-testid="stSidebarNavLink"][aria-current="page"] p { color: #ffffff; }
        [data-testid="stSidebarNavLink"] span[label="streamlit app"] p {
            font-size: 0;
        }
        [data-testid="stSidebarNavLink"] span[label="streamlit app"] p::after {
            content: "Your Week";
            font-size: .9rem;
        }
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
        [data-testid="stSidebar"] summary,
        [data-testid="stSidebar"] .stMarkdown p { color: #d8e1ec; }
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 { color: #ffffff !important; }
        [data-testid="stSidebar"] hr { border-color: rgba(255,255,255,.14); }
        [data-testid="stSidebar"] [data-testid="stExpander"] {
            background: rgba(255,255,255,.06) !important;
            border-color: rgba(255,255,255,.16) !important;
            box-shadow: none !important;
        }
        [data-testid="stSidebar"] [data-testid="stExpander"] summary {
            color: #d8e1ec;
        }

        h1, h2, h3 { color: var(--ff-ink); letter-spacing: -.025em; }
        h1 { font-weight: 800; }
        h2, h3 { font-weight: 720; }
        [data-testid="stCaptionContainer"] { color: var(--ff-muted); }

        [data-testid="stMetric"] {
            background: var(--ff-panel);
            border: 1px solid var(--ff-line);
            border-radius: 14px;
            box-shadow: 0 8px 24px rgba(20,35,55,.055);
            padding: .95rem 1rem;
        }
        [data-testid="stMetricLabel"] p {
            color: var(--ff-muted);
            font-size: .76rem;
            font-weight: 700;
            letter-spacing: .035em;
            text-transform: uppercase;
        }
        [data-testid="stMetricValue"] { color: var(--ff-ink); font-weight: 760; }

        [data-testid="stDataFrame"],
        [data-testid="stExpander"],
        [data-testid="stForm"] {
            border: 1px solid var(--ff-line);
            border-radius: 14px;
            overflow: hidden;
            box-shadow: 0 6px 20px rgba(20,35,55,.04);
            background: var(--ff-panel);
        }
        [data-testid="stAlert"] { border-radius: 12px; }
        .stButton > button {
            min-height: 2.6rem;
            border-radius: 10px;
            border-color: #cad3de;
            font-weight: 700;
            transition: transform .12s ease, box-shadow .12s ease;
        }
        .stButton > button:hover {
            border-color: var(--ff-green);
            color: #08795a;
            box-shadow: 0 7px 16px rgba(19,185,129,.14);
            transform: translateY(-1px);
        }
        [data-baseweb="tab-list"] { gap: .4rem; }
        [data-baseweb="tab"] {
            border-radius: 9px 9px 0 0;
            font-weight: 700;
        }

        @media (max-width: 700px) {
            .block-container { padding: 1.25rem 1rem 3rem; }
            h1 { font-size: 2rem !important; line-height: 1.08 !important; }
            h2 { font-size: 1.35rem !important; }
            [data-testid="stHorizontalBlock"] { gap: .65rem; }
            [data-testid="stMetric"] { padding: .75rem .8rem; }
            [data-testid="stDataFrame"] { border-radius: 10px; }
            .ff-mobile-nav {
                display: block;
                margin: 0 0 1.25rem;
                padding: .8rem .9rem;
                color: #ffffff;
                background: linear-gradient(135deg, #132238, #1b3453);
                border-radius: 14px;
                box-shadow: 0 9px 24px rgba(19,34,56,.18);
            }
            .ff-mobile-nav__brand {
                margin-bottom: .55rem;
                color: #ffffff;
                font-size: .7rem;
                font-weight: 800;
                letter-spacing: .12em;
            }
            .ff-mobile-nav summary {
                padding: .25rem 0;
                color: #ffffff;
                cursor: pointer;
                font-size: 1rem;
                font-weight: 750;
                list-style-position: inside;
            }
            .ff-mobile-nav__links {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: .45rem;
                margin-top: .7rem;
            }
            .ff-mobile-nav__links a {
                display: flex;
                align-items: center;
                min-height: 2.55rem;
                padding: .55rem .65rem;
                color: #eaf1f8 !important;
                background: rgba(255,255,255,.08);
                border: 1px solid rgba(255,255,255,.12);
                border-radius: 9px;
                font-size: .82rem;
                font-weight: 650;
                line-height: 1.15;
                text-decoration: none;
            }
            .ff-mobile-nav__links a:first-child {
                color: #ffffff !important;
                background: rgba(19,185,129,.28);
                border-color: rgba(19,185,129,.55);
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="ff-mobile-nav">
          <div class="ff-mobile-nav__brand">FF COMMAND CENTER</div>
          <details>
            <summary>Menu &amp; all tools</summary>
            <nav class="ff-mobile-nav__links" aria-label="Mobile navigation">
              <a href="/" target="_self">Your Week</a>
              <a href="/Weekly_Actions" target="_self">Weekly Actions</a>
              <a href="/Weekly_Rankings" target="_self">Weekly Rankings</a>
              <a href="/Weekly_Report" target="_self">Weekly Report</a>
              <a href="/Schedule_Grid" target="_self">Schedule Grid</a>
              <a href="/ROS_Rankings" target="_self">ROS Rankings</a>
              <a href="/Trade_Analyzer" target="_self">Trade Analyzer</a>
              <a href="/Trade_Finder" target="_self">Trade Finder</a>
              <a href="/Chopped_Bids" target="_self">Chopped Bids</a>
              <a href="/Alerts" target="_self">Phone Alerts</a>
              <a href="/Operations" target="_self">Operations</a>
              <a href="/Live_Matchup" target="_self">Live Matchup</a>
              <a href="/Roster_Strategy" target="_self">Roster Strategy</a>
              <a href="/Decision_Learning" target="_self">Decision Learning</a>
              <a href="/Postgame_Review" target="_self">Postgame Review</a>
              <a href="/Predictions_vs_Actuals" target="_self">Predictions vs Actuals</a>
              <a href="/Install_App" target="_self">Install App</a>
              <a href="/Model_Health" target="_self">Model Health</a>
              <a href="/Draft_Board" target="_self">Draft Board</a>
              <a href="/Draft_Mobile" target="_self">Draft Mobile</a>
              <a href="/Mock_Draft" target="_self">Mock Draft</a>
            </nav>
          </details>
        </div>
        """,
        unsafe_allow_html=True,
    )


__all__ = ["apply_app_shell"]
