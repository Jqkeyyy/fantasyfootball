"""In-season home page."""

import streamlit as st

from ffapp.app.home import render_home
from ffapp.app.league_selector import select_league
from ffapp.app.ui import apply_app_shell

st.set_page_config(page_title="FF Command Center", layout="wide")
apply_app_shell()
league = select_league()
render_home(league.slug)
