"""RSS ingestion + LLM structuring + manual review queue (SPEC.md §14.8; task 2.10).

No live network calls anywhere -- CLAUDE.md's own rule. The RSS fetch is
exercised via a fake `requests.Session`; the Anthropic call via a fake
client with the same `.messages.create(...)` surface, never a real key.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl
import pytest
from defusedxml.common import EntitiesForbidden

from ffapp.cache.offline import OfflineCacheMiss, sidecar_path, write_sidecar
from ffapp.config import CacheSettings, Settings
from ffapp.ingest import news

SAMPLE_RSS = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel>
<title>Fixture Feed</title>
<item>
  <title>Sources: RB1 done for the year with a real injury</title>
  <description>The team's RB1 will not return this season.</description>
  <link>https://example.com/story/1</link>
  <pubDate>Thu, 13 Aug 2026 20:25:34 EST</pubDate>
  <guid isPermaLink="false">story-1</guid>
</item>
<item>
  <title>No guid item</title>
  <description>Some text.</description>
  <link>https://example.com/story/2</link>
</item>
<item>
  <title></title>
  <description>An item with no real title, must be dropped.</description>
</item>
</channel></rss>
"""


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        data_root=tmp_path,
        sleeper_username="fixture_user",
        cache=CacheSettings(
            root=tmp_path / "raw",
            offline_default=True,
            staleness_hours={"espn": 2},
            warn_on_stale=True,
        ),
    )


# --- fetch_rss_feed ---------------------------------------------------------------


class _FakeResponse:
    def __init__(self, text: str) -> None:
        self.text = text

    def raise_for_status(self) -> None:
        return None


class _FakeSession:
    def __init__(self, text: str) -> None:
        self.text = text
        self.calls: list[str] = []

    def get(self, url: str, timeout: int) -> _FakeResponse:
        self.calls.append(url)
        return _FakeResponse(self.text)


def test_fetch_rss_feed_online_writes_raw_xml_and_sidecar(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    fake_session = _FakeSession(SAMPLE_RSS)
    monkeypatch.setattr(news, "_get_session", lambda: fake_session)

    path = news.fetch_rss_feed("espn", offline=False, settings=settings)

    assert path.exists()
    assert path.read_text() == SAMPLE_RSS
    assert fake_session.calls == [news.RSS_FEEDS["espn"]]
    meta = json.loads(sidecar_path(path).read_text())
    assert meta["source"] == "news"
    assert meta["cache_key"] == "espn"


def test_fetch_rss_feed_offline_with_fresh_cache_does_not_call_network(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    def _boom() -> None:
        raise AssertionError("network should not be called offline")

    monkeypatch.setattr(news, "_get_session", _boom)
    path = settings.cache.root / "news" / "espn.xml"
    path.parent.mkdir(parents=True)
    path.write_text(SAMPLE_RSS)
    write_sidecar(path, source="news", call="x", cache_key="espn")

    result = news.fetch_rss_feed("espn", offline=True, settings=settings)

    assert result == path


def test_fetch_rss_feed_offline_without_cache_raises_offline_cache_miss(
    settings: Settings,
) -> None:
    with pytest.raises(OfflineCacheMiss):
        news.fetch_rss_feed("espn", offline=True, settings=settings)


def test_fetch_rss_feed_unknown_source_raises_value_error(settings: Settings) -> None:
    with pytest.raises(ValueError, match="Unknown news source"):
        news.fetch_rss_feed("not-a-real-source", offline=True, settings=settings)


# --- parse_rss_entries -------------------------------------------------------------


class TestParseRssEntries:
    def test_parses_real_items(self) -> None:
        items = news.parse_rss_entries(SAMPLE_RSS, source="espn")

        titles = [i.title for i in items]
        assert "Sources: RB1 done for the year with a real injury" in titles
        assert "No guid item" in titles

    def test_an_item_with_no_title_is_dropped(self) -> None:
        items = news.parse_rss_entries(SAMPLE_RSS, source="espn")

        assert all(i.title for i in items)
        assert len(items) == 2

    def test_guid_falls_back_to_link_when_absent(self) -> None:
        items = news.parse_rss_entries(SAMPLE_RSS, source="espn")

        no_guid_item = next(i for i in items if i.title == "No guid item")
        assert no_guid_item.guid == "https://example.com/story/2"

    def test_real_guid_is_used_when_present(self) -> None:
        items = news.parse_rss_entries(SAMPLE_RSS, source="espn")

        real_item = next(i for i in items if "RB1" in i.title)
        assert real_item.guid == "story-1"

    def test_rejects_malicious_xml_entity_expansion(self) -> None:
        # A real billion-laughs payload -- defusedxml must reject this,
        # not the stdlib parser silently expanding it.
        bomb = """<?xml version="1.0"?>
<!DOCTYPE lolz [
 <!ENTITY lol "lol">
 <!ELEMENT lolz (#PCDATA)>
 <!ENTITY lol1 "&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;">
]>
<rss><channel><item><title>&lol1;</title></item></channel></rss>
"""
        with pytest.raises(EntitiesForbidden):
            news.parse_rss_entries(bomb, source="espn")


# --- structure_news_item -----------------------------------------------------------


def _players_dim() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "player_id": ["p1", "p2"],
            "normalized_name": ["christian mccaffrey", "jordan mason"],
        }
    )


def _item() -> news.NewsItem:
    return news.NewsItem(
        source="espn",
        guid="story-1",
        title="Sources: Christian McCaffrey done for the year",
        description="The 49ers RB1 will not return this season.",
        link="https://example.com/story/1",
        pub_date=None,
    )


class _FakeTextBlock:
    type = "text"

    def __init__(self, text: str) -> None:
        self.text = text


class _FakeMessage:
    def __init__(self, *, stop_reason: str, content: list[Any]) -> None:
        self.stop_reason = stop_reason
        self.content = content


class _FakeMessages:
    def __init__(self, response: _FakeMessage | Exception) -> None:
        self._response = response

    def create(self, **kwargs: Any) -> _FakeMessage:
        if isinstance(self._response, Exception):
            raise self._response
        return self._response


class _FakeClient:
    def __init__(self, response: _FakeMessage | Exception) -> None:
        self.messages = _FakeMessages(response)


def _structured_response(**overrides: Any) -> _FakeMessage:
    payload = {
        "player_name": "Christian McCaffrey",
        "team": "SF",
        "event_type": "injury",
        "severity": "season_ending",
        "expected_usage_change": "decrease",
        "affected_teammates": ["Jordan Mason"],
        "confidence": "high",
        "effective_week": 2,
    }
    payload.update(overrides)
    return _FakeMessage(stop_reason="end_turn", content=[_FakeTextBlock(json.dumps(payload))])


class TestStructureNewsItem:
    def test_a_real_high_confidence_item_resolves_and_structures(self) -> None:
        client = _FakeClient(_structured_response())

        event = news.structure_news_item(_item(), _players_dim(), client=client)  # type: ignore[arg-type]

        assert event is not None
        assert event.player_id == "p1"
        assert event.event_type == "injury"
        assert event.severity == "season_ending"
        assert event.confidence == "high"

    def test_low_confidence_routes_to_manual_review_none(self) -> None:
        client = _FakeClient(_structured_response(confidence="low"))

        event = news.structure_news_item(_item(), _players_dim(), client=client)  # type: ignore[arg-type]

        assert event is None

    def test_unresolvable_player_name_routes_to_manual_review_none(self) -> None:
        client = _FakeClient(_structured_response(player_name="Some Totally Unknown Guy"))

        event = news.structure_news_item(_item(), _players_dim(), client=client)  # type: ignore[arg-type]

        assert event is None

    def test_a_refusal_routes_to_manual_review_none(self) -> None:
        client = _FakeClient(_FakeMessage(stop_reason="refusal", content=[]))

        event = news.structure_news_item(_item(), _players_dim(), client=client)  # type: ignore[arg-type]

        assert event is None

    def test_non_json_response_routes_to_manual_review_none(self) -> None:
        client = _FakeClient(
            _FakeMessage(stop_reason="end_turn", content=[_FakeTextBlock("not json at all")])
        )

        event = news.structure_news_item(_item(), _players_dim(), client=client)  # type: ignore[arg-type]

        assert event is None

    def test_a_raised_api_error_routes_to_manual_review_none(self) -> None:
        import anthropic

        client = _FakeClient(
            anthropic.APIConnectionError(request=__import__("httpx").Request("POST", "https://x"))
        )

        event = news.structure_news_item(_item(), _players_dim(), client=client)  # type: ignore[arg-type]

        assert event is None


# --- manual review queue ------------------------------------------------------------


class TestReviewQueue:
    def test_build_review_row_has_expected_shape(self) -> None:
        row = news.build_review_row(_item(), reason="low_confidence")

        d = row.to_dicts()[0]
        assert d["guid"] == "story-1"
        assert d["reason"] == "low_confidence"
        assert d["reviewed"] is False

    def test_write_review_queue_creates_a_new_file(self, tmp_path: Path) -> None:
        row = news.build_review_row(_item(), reason="low_confidence")
        path = tmp_path / "manual_review.parquet"

        result = news.write_review_queue(row, path)

        assert path.exists()
        assert result.height == 1

    def test_write_review_queue_upserts_by_guid(self, tmp_path: Path) -> None:
        path = tmp_path / "manual_review.parquet"
        first = news.build_review_row(_item(), reason="low_confidence")
        news.write_review_queue(first, path)

        second = news.build_review_row(_item(), reason="unresolvable_player_name")
        result = news.write_review_queue(second, path)

        assert result.height == 1
        assert result.to_dicts()[0]["reason"] == "unresolvable_player_name"

    def test_write_review_queue_preserves_other_real_rows(self, tmp_path: Path) -> None:
        path = tmp_path / "manual_review.parquet"
        item_a = news.NewsItem(
            source="espn", guid="a", title="A", description="", link="", pub_date=None
        )
        item_b = news.NewsItem(
            source="espn", guid="b", title="B", description="", link="", pub_date=None
        )
        news.write_review_queue(news.build_review_row(item_a, reason="low_confidence"), path)
        result = news.write_review_queue(
            news.build_review_row(item_b, reason="low_confidence"), path
        )

        assert set(result["guid"].to_list()) == {"a", "b"}
