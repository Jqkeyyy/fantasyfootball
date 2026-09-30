import json
from datetime import UTC, datetime

import pytest

from ffapp.config import CacheSettings, LeagueConfig, Settings
from ffapp.tools.daily_briefing import build_briefing


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
