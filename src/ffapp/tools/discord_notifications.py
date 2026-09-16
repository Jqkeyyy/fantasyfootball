"""Optional Discord webhook delivery for refresh results and decision alerts."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import requests

DISCORD_WEBHOOK_ENV = "DISCORD_WEBHOOK_URL"


@dataclass(frozen=True)
class NotificationResult:
    status: str
    detail: str


def format_refresh_message(
    league_name: str,
    season: int,
    week: int,
    status: str,
    steps: list[dict[str, str]],
    alerts: list[dict[str, Any]] | None = None,
) -> str:
    icon = {"healthy": "✅", "degraded": "⚠️", "failed": "🚨"}.get(status, "ℹ️")
    lines = [f"{icon} **{league_name} — Week {week} refresh: {status.upper()}**"]
    problems = [step for step in steps if step.get("status") in {"degraded", "failed"}]
    for step in problems[:5]:
        lines.append(f"• {step['name']}: {step['detail'][:240]}")
    actionable = alerts or []
    if actionable:
        lines.append("**Decision alerts**")
        lines.extend(
            f"• {str(alert.get('message', 'Action needed'))[:300]}" for alert in actionable[:5]
        )
    lines.append(f"Season {season} · Week {week}")
    return "\n".join(lines)[:2000]


def send_discord_message(
    content: str,
    *,
    webhook_url: str | None = None,
    timeout_seconds: float = 15.0,
) -> NotificationResult:
    """Post one mention-free message; silently skip when no webhook is configured."""
    url = webhook_url or os.getenv(DISCORD_WEBHOOK_ENV)
    if not url:
        return NotificationResult("skipped", f"{DISCORD_WEBHOOK_ENV} is not configured")
    try:
        response = requests.post(
            url,
            json={"content": content[:2000], "allowed_mentions": {"parse": []}},
            timeout=timeout_seconds,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        status_code = getattr(getattr(exc, "response", None), "status_code", None)
        detail = f"Discord delivery failed{f' (HTTP {status_code})' if status_code else ''}"
        return NotificationResult("failed", detail)
    return NotificationResult("sent", "Discord notification delivered")


__all__ = [
    "DISCORD_WEBHOOK_ENV",
    "NotificationResult",
    "format_refresh_message",
    "send_discord_message",
]
