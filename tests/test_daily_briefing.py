import json
from datetime import UTC, datetime

import pytest

from ffapp.config import CacheSettings, LeagueConfig, Settings
from ffapp.tools.daily_briefing import build_briefing


@pytest.mark.parametrize("alert_file", [None, "broken json", '{"alerts": []}'])
def test_degraded_briefing_explains_problem_without_claiming_all_clear(
    tmp_path, monkeypatch, alert_file
) -> None:
    monkeypatch.setenv("FFAPP_DASHBOARD_URL", "https://dashboard.example.ts.net")
    output = tmp_path / "outputs" / "test-league"
    (output / "refresh_runs").mkdir(parents=True)
    (output / "refresh_runs" / "latest.json").write_text(
        json.dumps(
            {
                "status": "degraded",
                "week": 6,
                "steps": [
                    {"name": "news", "status": "healthy", "detail": "Fetched news"},
                    {
                        "name": "Prior-week actuals",
                        "status": "degraded",
                        "detail": "No prior-week log found",
                    },
                ],
            }
        )
    )
    if alert_file is not None:
        (output / "alerts").mkdir()
        (output / "alerts" / "latest.json").write_text(alert_file)
    settings = Settings(
        data_root=tmp_path,
        sleeper_username=None,
        cache=CacheSettings(tmp_path / "raw", True, {}, True),
    )
    league = LeagueConfig(
        slug="test-league",
        display_name="Test League",
        is_primary=True,
        league_id=None,
        season=2026,
        league_cache={},
        overrides={},
    )
    message = build_briefing(settings, [league], now=datetime(2026, 10, 16, tzinfo=UTC))
    assert "Prior-week actuals: No prior-week log found" in message
    assert "Fetched news" not in message
    assert "No new lineup or waiver exception needs action" not in message
    if alert_file != '{"alerts": []}':
        assert "alerts missing or unreadable" in message
    else:
        assert "actions may be missing" in message


def test_long_briefing_preserves_all_leagues_and_dashboard(monkeypatch) -> None:
    monkeypatch.setenv("FFAPP_DASHBOARD_URL", "https://dashboard.example.ts.net")
    monkeypatch.setattr(
        "ffapp.tools.daily_briefing.league_briefing",
        lambda settings, league: f"**League {league}**\n• Missing: " + "details " * 1000,
    )
    message = build_briefing(None, [1, 2, 3], now=datetime(2026, 10, 16, tzinfo=UTC))
    assert len("@everyone " + message) <= 2000
    for league in [1, 2, 3]:
        assert f"**League {league}**" in message
    assert "https://dashboard.example.ts.net/" in message
    assert message.endswith("paste it into Safari/Chrome.")


def test_daily_briefing_uses_materialized_status_and_alerts(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FFAPP_DASHBOARD_URL", "https://dashboard.example.ts.net")
    output = tmp_path / "outputs" / "test-league"
    (output / "refresh_runs").mkdir(parents=True)
    (output / "alerts").mkdir()
    (output / "refresh_runs" / "latest.json").write_text(
        json.dumps({"status": "healthy", "week": 3})
    )
    (output / "alerts" / "latest.json").write_text(
        json.dumps({"alerts": [{"message": "Start A over B", "confidence": 0.8}]})
    )
    settings = Settings(
        data_root=tmp_path,
        sleeper_username=None,
        cache=CacheSettings(tmp_path / "raw", True, {}, True),
    )
    league = LeagueConfig(
        slug="test-league",
        display_name="Test League",
        is_primary=True,
        league_id=None,
        season=2026,
        league_cache={},
        overrides={},
    )

    message = build_briefing(settings, [league], now=datetime(2026, 9, 22, 13, tzinfo=UTC))

    assert "Test League · Week 3 · HEALTHY" in message
    assert "Start A over B (80%)" in message
    assert "https://dashboard.example.ts.net/" in message
