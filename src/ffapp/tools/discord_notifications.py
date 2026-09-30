"""Optional Discord webhook delivery for refresh results and decision alerts."""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

from ffapp.config import load_settings
from ffapp.env import load_env
from ffapp.tools.artifacts import atomic_write_json

DISCORD_WEBHOOK_ENV = "DISCORD_WEBHOOK_URL"
DASHBOARD_URL_ENV = "FFAPP_DASHBOARD_URL"
_DEFAULT_DASHBOARD_URL = "http://localhost:8501/"


def dashboard_url() -> str:
    """The private dashboard's address, kept in .env so it never ships in the repo."""
    load_env()
    url = os.getenv(DASHBOARD_URL_ENV, "").strip() or _DEFAULT_DASHBOARD_URL
    return url if url.endswith("/") else f"{url}/"


def with_dashboard_link(content: str) -> str:
    """Add the private dashboard link with the connection step it requires."""
    return (
        f"{content}\n\n"
        "Open Tailscale on your phone and confirm it says Connected, then open:\n"
        f"{dashboard_url()}\n"
        "If Discord cannot open it, choose Open in Browser or paste it into Safari/Chrome."
    )


def webhook_path() -> Path:
    return load_settings().data_root / "private" / "discord.json"


def configured_webhook() -> str | None:
    path = webhook_path()
    if path.exists():
        return str(json.loads(path.read_text()).get("webhook_url") or "") or None
    return os.getenv(DISCORD_WEBHOOK_ENV)


def save_webhook(url: str) -> None:
    parsed = urlparse(url.strip())
    if (
        parsed.scheme != "https"
        or parsed.netloc not in {"discord.com", "discordapp.com"}
        or not parsed.path.startswith("/api/webhooks/")
        or len(parsed.path.rstrip("/").split("/")) != 5
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("Paste a Discord channel webhook URL from Server Settings → Integrations.")
    path = webhook_path()
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    atomic_write_json({"webhook_url": url.strip()}, path)
    path.chmod(0o600)


def send_action_notification(
    content: str, *, key: str, cooldown: int = 21600
) -> NotificationResult:
    """Suppress identical alerts for six hours; failed deliveries can retry."""
    path = load_settings().data_root / "outputs" / "notification_state.json"
    state = json.loads(path.read_text()) if path.exists() else {}
    digest = hashlib.sha256((key + content).encode()).hexdigest()
    now = time.time()
    if now - state.get(digest, 0) < cooldown:
        return NotificationResult("skipped", "Duplicate notification suppressed")
    result = send_discord_message(content)
    if result.status == "sent":
        state = {k: v for k, v in state.items() if now - v < cooldown}
        state[digest] = now
        atomic_write_json(state, path)
    return result


@dataclass(frozen=True)
class NotificationResult:
    status: str
    detail: str


def format_decision_alert(alert: dict[str, Any]) -> str:
    """Render one actionable alert with decision evidence and risk."""
    priority = str(alert.get("priority") or "WATCH")
    message = str(alert.get("message") or "Action needed")
    confidence = alert.get("confidence")
    confidence_text = (
        f" · confidence {float(confidence):.0%}" if isinstance(confidence, int | float) else ""
    )
    lines = [f"• **{priority}:** {message[:300]}{confidence_text}"]
    why = alert.get("why")
    risk = alert.get("risk")
    if why:
        lines.append(f"  Why: {str(why)[:260]}")
    if risk:
        lines.append(f"  Risk: {str(risk)[:260]}")
    return "\n".join(lines)


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
        lines.extend(format_decision_alert(alert) for alert in actionable[:5])
    lines.append(f"Season {season} · Week {week}")
    return "\n".join(lines)[:2000]


def send_discord_message(
    content: str,
    *,
    webhook_url: str | None = None,
    timeout_seconds: float = 15.0,
    mention_everyone: bool = True,
) -> NotificationResult:
    """Post one message and optionally notify everyone in the Discord channel."""
    url = webhook_url or configured_webhook()
    if not url:
        return NotificationResult("skipped", f"{DISCORD_WEBHOOK_ENV} is not configured")
    try:
        message = f"@everyone {content}" if mention_everyone else content
        allowed_mentions = {"parse": ["everyone"] if mention_everyone else []}
        response = requests.post(
            url,
            json={"content": message[:2000], "allowed_mentions": allowed_mentions},
            timeout=timeout_seconds,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        status_code = getattr(getattr(exc, "response", None), "status_code", None)
        detail = f"Discord delivery failed{f' (HTTP {status_code})' if status_code else ''}"
        return NotificationResult("failed", detail)
    return NotificationResult("sent", "Discord notification delivered")


__all__ = [
    "DASHBOARD_URL_ENV",
    "DISCORD_WEBHOOK_ENV",
    "NotificationResult",
    "dashboard_url",
    "format_decision_alert",
    "format_refresh_message",
    "send_discord_message",
    "with_dashboard_link",
]
