import json

import pytest

from ffapp.tools import discord_bot


def test_save_bot_config_keeps_token_private(tmp_path, monkeypatch) -> None:
    path = tmp_path / "discord_bot.json"
    monkeypatch.setattr(discord_bot, "bot_config_path", lambda: path)

    discord_bot.save_bot_config("a" * 40, "123456789")

    assert json.loads(path.read_text()) == {
        "bot_token": "a" * 40,
        "guild_id": "123456789",
    }
    assert discord_bot.configured_bot()


@pytest.mark.parametrize(
    ("token", "guild"),
    [("short", ""), ("a" * 40, "not-a-number"), ("a" * 20 + " " + "b" * 20, "")],
)
def test_save_bot_config_rejects_invalid_values(tmp_path, monkeypatch, token, guild) -> None:
    monkeypatch.setattr(discord_bot, "bot_config_path", lambda: tmp_path / "bot.json")

    with pytest.raises(ValueError):
        discord_bot.save_bot_config(token, guild)
