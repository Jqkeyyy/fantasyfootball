"""Private slash-command bot backed by the dashboard's current artifacts."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from ffapp.config import load_all_leagues, load_league, load_primary_league, load_settings
from ffapp.tools.artifacts import atomic_write_json
from ffapp.tools.discord_notifications import dashboard_url


def bot_config_path() -> Path:
    return load_settings().data_root / "private" / "discord_bot.json"


def load_bot_config() -> dict[str, str]:
    path = bot_config_path()
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {key: str(value) for key, value in raw.items() if value}


def configured_bot() -> bool:
    return bool(load_bot_config().get("bot_token"))


def save_bot_config(token: str, guild_id: str = "") -> None:
    clean_token = token.strip()
    clean_guild = guild_id.strip()
    if len(clean_token) < 30 or any(character.isspace() for character in clean_token):
        raise ValueError("Paste the bot token from the Discord Developer Portal.")
    if clean_guild and not clean_guild.isdigit():
        raise ValueError("The optional Discord server ID must contain only numbers.")
    path = bot_config_path()
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    payload = {"bot_token": clean_token}
    if clean_guild:
        payload["guild_id"] = clean_guild
    atomic_write_json(payload, path)
    path.chmod(0o600)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _league(slug: str | None) -> tuple[str, str]:
    league = load_league(slug) if slug else load_primary_league()
    return league.slug, league.display_name


def _alerts(slug: str) -> dict[str, Any]:
    return _read_json(load_settings().data_root / "outputs" / slug / "alerts" / "latest.json")


def _alert_lines(slug: str, kinds: set[str]) -> tuple[int | None, list[str]]:
    payload = _alerts(slug)
    rows = payload.get("alerts") or []
    alerts = [row for row in rows if isinstance(row, dict) and str(row.get("kind")) in kinds]
    return payload.get("week"), [
        f"• {str(row.get('message') or 'Action available')}" for row in alerts
    ]


def command_summary(command: str, league_slug: str | None = None) -> str:
    """Build a concise response without importing Discord, suitable for tests and the bot."""
    if command == "health":
        lines = ["**Fantasy Football system health**"]
        settings = load_settings()
        for league in load_all_leagues():
            manifest = _read_json(
                settings.data_root / "outputs" / league.slug / "refresh_runs" / "latest.json"
            )
            status = str(manifest.get("status") or "missing").upper()
            week = manifest.get("week", "?")
            generated = str(manifest.get("generated_at_utc") or "never")
            lines.append(f"• {league.display_name}: {status} · Week {week} · {generated[:16]}")
        return "\n".join(lines)[:1900]

    slug, display_name = _league(league_slug)
    if command == "waivers":
        week, lines = _alert_lines(slug, {"waiver_upgrade"})
        heading = f"**{display_name} · Week {week or '?'} waiver targets**"
        body = lines[:8] or ["• No waiver upgrade currently clears the action threshold."]
        return "\n".join([heading, *body, f"{dashboard_url()}Weekly_Actions"])[:1900]
    if command == "lineup":
        week, lines = _alert_lines(
            slug,
            {"lineup_swap", "starter_out", "starter_projection_drop", "starter_status"},
        )
        heading = f"**{display_name} · Week {week or '?'} lineup**"
        body = lines[:8] or ["• No unlocked lineup change currently clears the threshold."]
        return "\n".join([heading, *body, f"{dashboard_url()}Weekly_Actions"])[:1900]
    if command == "matchup":
        payload = _read_json(load_settings().data_root / "outputs" / slug / "gameday.json")
        week = payload.get("week", _alerts(slug).get("week", "?"))
        status = str(payload.get("status") or "No active kickoff-window alert")
        return (
            f"**{display_name} · Week {week} matchup**\n• {status}\n{dashboard_url()}Live_Matchup"
        )[:1900]
    if command == "trades":
        return (
            f"**{display_name} trade finder**\n"
            "Searches every roster for fair 1-for-1, 2-for-1, and 1-for-2 packages, then "
            "measures both teams' expected starting-lineup change.\n"
            f"{dashboard_url()}Trade_Finder"
        )[:1900]
    raise ValueError(f"Unknown Discord command: {command}")


def run() -> None:
    """Connect to Discord's Gateway and register private slash commands."""
    config = load_bot_config()
    token = config.get("bot_token") or os.getenv("DISCORD_BOT_TOKEN")
    if not token:
        print("Discord bot is not configured; use the Phone Alerts page.")
        return

    import discord
    from discord import app_commands

    intents = discord.Intents.default()

    class FantasyClient(discord.Client):
        def __init__(self) -> None:
            super().__init__(intents=intents)
            self.tree = app_commands.CommandTree(self)
            self.synced = False

        async def setup_hook(self) -> None:
            guild_id = config.get("guild_id")
            if guild_id:
                guild = discord.Object(id=int(guild_id))
                self.tree.copy_global_to(guild=guild)
                await self.tree.sync(guild=guild)
            else:
                await self.tree.sync()
            self.synced = True

    client = FantasyClient()

    async def respond(interaction: discord.Interaction, command: str, league: str | None) -> None:
        try:
            message = command_summary(command, league)
        except Exception as exc:
            message = f"Command data is unavailable: {exc}"
        await interaction.response.send_message(message, ephemeral=True)

    @client.tree.command(name="lineup", description="Show your current lineup actions")
    async def lineup(interaction: discord.Interaction, league: str | None = None) -> None:
        await respond(interaction, "lineup", league)

    @client.tree.command(name="waivers", description="Show the best current waiver upgrades")
    async def waivers(interaction: discord.Interaction, league: str | None = None) -> None:
        await respond(interaction, "waivers", league)

    @client.tree.command(name="matchup", description="Show your live matchup status")
    async def matchup(interaction: discord.Interaction, league: str | None = None) -> None:
        await respond(interaction, "matchup", league)

    @client.tree.command(name="trades", description="Open the lineup-aware trade finder")
    async def trades(interaction: discord.Interaction, league: str | None = None) -> None:
        await respond(interaction, "trades", league)

    @client.tree.command(name="health", description="Check refresh health for every league")
    async def health(interaction: discord.Interaction) -> None:
        await respond(interaction, "health", None)

    client.run(token, log_handler=None)


if __name__ == "__main__":
    run()


__all__ = [
    "bot_config_path",
    "command_summary",
    "configured_bot",
    "load_bot_config",
    "run",
    "save_bot_config",
]
