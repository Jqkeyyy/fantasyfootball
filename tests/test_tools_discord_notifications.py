from __future__ import annotations

from typing import Any

from ffapp.tools import discord_notifications


class _Response:
    def raise_for_status(self) -> None:
        return None


def test_discord_skips_without_secret(monkeypatch, tmp_path) -> None:
    # A webhook saved from the Phone Alerts page must not leak in and post for real.
    monkeypatch.setattr(discord_notifications, "webhook_path", lambda: tmp_path / "none.json")
    monkeypatch.delenv(discord_notifications.DISCORD_WEBHOOK_ENV, raising=False)
    result = discord_notifications.send_discord_message("hello")
    assert result.status == "skipped"


def test_discord_posts_everyone_mention_payload(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    def fake_post(url: str, **kwargs: Any) -> _Response:
        captured.update({"url": url, **kwargs})
        return _Response()

    monkeypatch.setattr(discord_notifications.requests, "post", fake_post)
    result = discord_notifications.send_discord_message(
        "hello", webhook_url="https://discord.test/webhook"
    )
    assert result.status == "sent"
    assert captured["json"] == {
        "content": "@everyone hello",
        "allowed_mentions": {"parse": ["everyone"]},
    }


def test_discord_can_send_without_everyone_mention(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    def fake_post(url: str, **kwargs: Any) -> _Response:
        captured.update({"url": url, **kwargs})
        return _Response()

    monkeypatch.setattr(discord_notifications.requests, "post", fake_post)
    discord_notifications.send_discord_message(
        "hello",
        webhook_url="https://discord.test/webhook",
        mention_everyone=False,
    )
    assert captured["json"] == {"content": "hello", "allowed_mentions": {"parse": []}}


def test_refresh_message_prioritizes_problems_and_alerts() -> None:
    message = discord_notifications.format_refresh_message(
        "Chopped League",
        2026,
        3,
        "degraded",
        [{"name": "source", "status": "degraded", "detail": "cached fallback"}],
        [{"message": "Your starter is out"}],
    )
    assert "Chopped League" in message
    assert "cached fallback" in message
    assert "Your starter is out" in message


def test_decision_alert_includes_confidence_reason_and_risk() -> None:
    message = discord_notifications.format_decision_alert(
        {
            "priority": "NOW",
            "message": "Start Player A over Player B",
            "confidence": 0.78,
            "why": "Projected gain is 3.2 points.",
            "risk": "Player A is questionable.",
        }
    )

    assert "confidence 78%" in message
    assert "Why: Projected gain" in message
    assert "Risk: Player A" in message
