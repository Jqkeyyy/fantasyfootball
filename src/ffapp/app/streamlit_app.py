"""Dashboard entry point with always-expanded page navigation."""

from pathlib import Path

import streamlit as st

from ffapp.app.home import render_home
from ffapp.app.league_selector import select_league
from ffapp.app.ui import apply_app_shell


def your_week() -> None:
    apply_app_shell()
    league = select_league()
    render_home(league.slug)


st.set_page_config(page_title="FF Command Center", layout="wide")
page_files = sorted(
    Path(__file__).with_name("pages").glob("*.py"),
    key=lambda path: (int(path.stem.split("_", 1)[0]), path.stem.lower()),
)
pages = [st.Page(your_week, title="Your Week", default=True)]
for page_file in page_files:
    route = page_file.stem.split("_", 1)[1]
    pages.append(st.Page(page_file, title=route.replace("_", " "), url_path=route))

st.navigation(pages, expanded=True).run()
