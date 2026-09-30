"""Private Discord setup and delivery status."""

import streamlit as st

from ffapp.app.ui import apply_app_shell
from ffapp.tools.discord_bot import configured_bot, save_bot_config
from ffapp.tools.discord_notifications import (
    configured_webhook,
    save_webhook,
    send_discord_message,
    with_dashboard_link,
)

st.set_page_config(page_title="Phone Alerts", layout="centered")
apply_app_shell()
st.title("Phone Alerts")
st.write(
    "Get Discord messages for starter availability changes, meaningful projection moves, "
    "waiver opportunities and refresh failures. Alerts notify @everyone in the configured "
    "channel, and repeated identical messages are suppressed."
)
st.markdown(
    "Alerts are sent when a decision crosses an action threshold:\n"
    "- an unlocked lineup swap gains at least **1.5 projected points**;\n"
    "- a starter is effectively ruled out or moves by at least **3 points**;\n"
    "- a free agent becomes at least a **3-point lineup upgrade**;\n"
    "- a scheduled refresh fails.\n\n"
    "Kickoff alerts identify the 90- or 30-minute decision window. Identical alerts are "
    "suppressed for six hours."
)
if configured_webhook():
    st.success("A Discord channel is configured.")
else:
    st.info("Connect a private Discord channel to finish phone alert setup.")
st.markdown(
    "1. In Discord, open your server's **Settings → Integrations → Webhooks**.\n"
    "2. Create a webhook for your private alerts channel and copy its URL.\n"
    "3. Paste it below, save, then send a test. Enable notifications for that channel "
    "in the Discord phone app."
)
with st.form("discord_setup", clear_on_submit=True):
    url = st.text_input("Discord webhook URL", type="password")
    if st.form_submit_button("Save Discord channel"):
        try:
            save_webhook(url)
            st.success("Saved securely on your home server. Send a test below.")
        except ValueError as exc:
            st.error(str(exc))
if st.button("Send test notification", disabled=not configured_webhook()):
    result = send_discord_message(
        with_dashboard_link("FF Command Center phone alerts are connected.")
    )
    if result.status == "sent":
        st.success(result.detail)
    else:
        st.error(result.detail)
st.caption(
    "The webhook is stored outside Git. It is never displayed back here. "
    "Only devices connected to your private Tailscale site can use this setup form."
)
with st.expander("If a dashboard link says it cannot be reached"):
    st.markdown(
        "1. Open **Tailscale** on your phone and turn it on so it says **Connected**.\n"
        "2. Return to Discord and choose **Open in Browser**, or paste the link into "
        "Safari/Chrome.\n"
        "3. Keep Tailscale connected while using the dashboard. The site is private and "
        "cannot be reached through a normal internet connection alone."
    )

st.divider()
st.subheader("Discord slash commands")
st.write(
    "Add `/lineup`, `/waivers`, `/matchup`, `/trades`, and `/health` to your server. "
    "Replies are private to the person who runs the command."
)
if configured_bot():
    st.success("The Discord command bot is configured. The server watchdog keeps it running.")
else:
    st.info("The existing webhook can send alerts, but a Discord bot token is needed for commands.")
st.markdown(
    "1. Open the **Discord Developer Portal**, create an application, then add a bot.\n"
    "2. Open **Installation**, enable **Guild Install**, and choose "
    "**Discord Provided Link**.\n"
    "3. Under **Default Install Settings > Guild Install**, add the `bot` and "
    "`applications.commands` scopes and give the bot **Send Messages** permission. Save, "
    "then copy and open the install link to add it to your server.\n"
    "4. Copy the bot token. Optionally enable Developer Mode in Discord, right-click your "
    "server, and copy its ID so commands appear there immediately."
)
with st.form("discord_bot_setup", clear_on_submit=True):
    bot_token = st.text_input("Discord bot token", type="password")
    guild_id = st.text_input("Discord server ID (optional)")
    if st.form_submit_button("Save command bot"):
        try:
            save_bot_config(bot_token, guild_id)
            st.success("Saved securely. The command bot will start within one minute.")
        except ValueError as exc:
            st.error(str(exc))
st.caption("The bot token is stored outside Git and is never displayed back here.")
