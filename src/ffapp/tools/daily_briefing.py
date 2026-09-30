"""Once-per-day Discord briefing assembled from materialized league artifacts."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from ffapp.config import LeagueConfig, Settings, load_all_leagues, load_settings
from ffapp.tools.artifacts import atomic_write_json
from ffapp.tools.discord_notifications import send_action_notification, with_dashboard_link

CENTRAL = ZoneInfo("America/Chicago")


def _read(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def league_briefing(settings: Settings, league: LeagueConfig) -> str:
    """Build one concise league line and its highest-priority current actions."""
    output = settings.data_root / "outputs" / league.slug
    refresh = _read(output / "refresh_runs" / "latest.json")
    alert_payload = _read(output / "alerts" / "latest.json")
    alerts = alert_payload.get("alerts")
    alert_rows = (
        [row for row in alerts if isinstance(row, dict)] if isinstance(alerts, list) else []
    )
    status = str(refresh.get("status") or "missing").upper()
    week = refresh.get("week") or alert_payload.get("week") or "?"
    lines = [f"**{league.display_name} · Week {week} · {status}**"]
    if alert_rows:
        for alert in alert_rows[:3]:
            confidence = alert.get("confidence")
            suffix = f" ({float(confidence):.0%})" if isinstance(confidence, int | float) else ""
            lines.append(f"• {str(alert.get('message') or 'Action needed')[:260]}{suffix}")
    else:
        lines.append("• No new lineup or waiver exception needs action.")
    return "\n".join(lines)


def build_briefing(settings: Settings, leagues: list[LeagueConfig], *, now: datetime) -> str:
    local = now.astimezone(CENTRAL)
    # Remove a leading zero without using platform-specific strftime flags.
    heading = local.strftime("%A, %B %d").replace(" 0", " ")
    sections = [f"🏈 **Daily fantasy briefing · {heading}**"]
    sections.extend(league_briefing(settings, league) for league in leagues)
    return with_dashboard_link("\n\n".join(sections))[:2000]


def run(*, now: datetime | None = None) -> str:
    settings = load_settings()
    current = now or datetime.now(UTC)
    day = current.astimezone(CENTRAL).date().isoformat()
    state_path = settings.data_root / "outputs" / "daily_briefing_state.json"
    state = _read(state_path)
    if state.get("sent_for") == day:
        return "skipped: already sent today"
    result = send_action_notification(
        build_briefing(settings, load_all_leagues(), now=current),
        key=f"daily-briefing-{day}",
        cooldown=25 * 60 * 60,
    )
    if result.status == "sent":
        atomic_write_json({"sent_for": day, "sent_at_utc": current.isoformat()}, state_path)
    return f"{result.status}: {result.detail}"


if __name__ == "__main__":
    print(run())


__all__ = ["build_briefing", "league_briefing", "run"]
