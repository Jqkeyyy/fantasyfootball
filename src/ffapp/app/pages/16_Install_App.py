"""Phone installation and connectivity guide."""

import streamlit as st

from ffapp.app.ui import apply_app_shell
from ffapp.tools.discord_notifications import dashboard_url

st.set_page_config(page_title="Install App", layout="wide")
apply_app_shell()
st.title("Install FF Command Center")
st.caption("Put the private dashboard on your home screen so it opens like an app.")

st.success("You are connected to the home server through Tailscale.")
ios, android = st.tabs(["iPhone / iPad", "Android"])
with ios:
    st.markdown(
        """
        1. Open this page in **Safari**.
        2. Tap the **Share** button.
        3. Scroll down and tap **Add to Home Screen**.
        4. Name it **FF Command** and tap **Add**.
        """
    )
with android:
    st.markdown(
        """
        1. Open this page in **Chrome**.
        2. Tap the three-dot menu.
        3. Tap **Install app** or **Add to Home screen**.
        4. Confirm **FF Command**.
        """
    )

st.info("Tailscale must say Connected before the installed app can reach your private server.")
st.code(dashboard_url(), language=None)
st.link_button("Open Your Week", "/", icon=":material/home:")
