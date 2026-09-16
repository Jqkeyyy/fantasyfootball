from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest

from ffapp.config import CacheSettings, Settings
from ffapp.ingest import news
from ffapp.tools import news_refresh


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        data_root=tmp_path,
        sleeper_username="fixture",
        cache=CacheSettings(
            root=tmp_path / "raw",
            offline_default=True,
            staleness_hours={},
            warn_on_stale=True,
        ),
    )


def _players() -> pl.DataFrame:
    return pl.DataFrame({"player_id": ["p1"], "normalized_name": ["christian mccaffrey"]})


def _item() -> news.NewsItem:
    return news.NewsItem(
        source="espn",
        guid="story-1",
        title="Christian McCaffrey injury update",
        description="An update.",
        link="https://example.com/1",
        pub_date="Tue, 15 Sep 2026 10:00:00 GMT",
    )


def _event() -> news.StructuredNewsEvent:
    return news.StructuredNewsEvent(
        player_name="Christian McCaffrey",
        player_id="p1",
        resolution_score=100.0,
        team="SF",
        event_type="injury",
        severity="moderate",
        expected_usage_change="decrease",
        affected_teammates=["Jordan Mason"],
        confidence="high",
        effective_week=2,
        source_guid="story-1",
        source_title="Christian McCaffrey injury update",
    )


def test_missing_key_is_clean_noop(monkeypatch: pytest.MonkeyPatch, settings: Settings) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(
        news_refresh.news,
        "fetch_rss_feed",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not fetch")),
    )

    summary = news_refresh.refresh_news(settings)

    assert summary["status"] == "skipped"
    assert summary["new_items"] == 0


def test_refresh_persists_event_and_skips_known_guid(
    monkeypatch: pytest.MonkeyPatch, settings: Settings, tmp_path: Path
) -> None:
    rss_path = tmp_path / "espn.xml"
    rss_path.write_text("<rss />")
    monkeypatch.setattr(news_refresh.news, "fetch_rss_feed", lambda *args, **kwargs: rss_path)
    monkeypatch.setattr(news_refresh.news, "parse_rss_entries", lambda *args, **kwargs: [_item()])
    calls: list[str] = []
    monkeypatch.setattr(
        news_refresh.news,
        "structure_news_item",
        lambda item, players_dim, client: calls.append(item.guid) or _event(),
    )

    first = news_refresh.refresh_news(
        settings,
        sources=["espn"],
        client=object(),
        players_dim=_players(),
        now=datetime(2026, 9, 15, tzinfo=UTC),
    )
    second = news_refresh.refresh_news(
        settings,
        sources=["espn"],
        client=object(),
        players_dim=_players(),
    )

    assert first["structured"] == 1
    assert second["new_items"] == 0
    assert calls == ["story-1"]
    events = pl.read_parquet(settings.data_root / "outputs" / "news" / "events.parquet")
    assert events.height == 1
    assert events.to_dicts()[0]["source_link"] == "https://example.com/1"


def test_rejected_item_is_persisted_for_review(
    monkeypatch: pytest.MonkeyPatch, settings: Settings, tmp_path: Path
) -> None:
    rss_path = tmp_path / "espn.xml"
    rss_path.write_text("<rss />")
    monkeypatch.setattr(news_refresh.news, "fetch_rss_feed", lambda *args, **kwargs: rss_path)
    monkeypatch.setattr(news_refresh.news, "parse_rss_entries", lambda *args, **kwargs: [_item()])
    monkeypatch.setattr(news_refresh.news, "structure_news_item", lambda *args, **kwargs: None)

    summary = news_refresh.refresh_news(
        settings, sources=["espn"], client=object(), players_dim=_players()
    )

    assert summary["reviewed"] == 1
    queued = pl.read_parquet(settings.data_root / "outputs" / "news" / "manual_review.parquet")
    assert queued.to_dicts()[0]["reason"] == "structuring_rejected"


def test_unknown_source_rejected(settings: Settings) -> None:
    with pytest.raises(ValueError, match="Unknown news sources"):
        news_refresh.refresh_news(settings, sources=["bogus"])
