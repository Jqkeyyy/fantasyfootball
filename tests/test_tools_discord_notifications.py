from __future__ import annotations

from typing import Any

from ffapp.tools import discord_notifications


class _Response:
    def raise_for_status(self) -> None:
        return None


def test_discord_skips_without_secret(monkeypatch) -> None:
    monkeypatch.delenv(discord_notifications.DISCORD_WEBHOOK_ENV, raising=False)
    result = discord_notifications.send_discord_message("hello")
    assert result.status == "skipped"


def test_discord_posts_mention_free_payload(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    def fake_post(url: str, **kwargs: Any) -> _Response:
        captured.update({"url": url, **kwargs})
        return _Response()

    monkeypatch.setattr(discord_notifications.requests, "post", fake_post)
    result = discord_notifications.send_discord_message(
        "hello", webhook_url="https://discord.test/webhook"
    )
    assert result.status == "sent"
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
