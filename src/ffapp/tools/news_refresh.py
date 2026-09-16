"""Scheduled RSS-to-structured-event refresh with durable deduplication."""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import anthropic
import polars as pl

from ffapp.config import Settings
from ffapp.ids import mapping
from ffapp.ingest import news, nflverse, sleeper
from ffapp.tools.artifacts import atomic_write_parquet

EVENTS_SCHEMA = pl.Schema(
    {  # type: ignore[arg-type]
        "player_name": pl.String,
        "player_id": pl.String,
        "resolution_score": pl.Float64,
        "team": pl.String,
        "event_type": pl.String,
        "severity": pl.String,
        "expected_usage_change": pl.String,
        "affected_teammates": pl.List(pl.String),
        "confidence": pl.String,
        "effective_week": pl.Int64,
        "source_guid": pl.String,
        "source_title": pl.String,
        "source": pl.String,
        "source_link": pl.String,
        "published_at": pl.String,
        "processed_at_utc": pl.String,
    }
)


def _known_guids(events_path: Path, review_path: Path) -> set[str]:
    known: set[str] = set()
    if events_path.exists():
        known.update(
            str(value)
            for value in pl.read_parquet(events_path, columns=["source_guid"])[
                "source_guid"
            ].drop_nulls()
        )
    if review_path.exists():
        known.update(
            str(value)
            for value in pl.read_parquet(review_path, columns=["guid"])["guid"].drop_nulls()
        )
    return known


def _write_events(rows: pl.DataFrame, output_path: Path) -> pl.DataFrame:
    if output_path.exists():
        existing = pl.read_parquet(output_path)
        incoming = rows.select("source_guid").unique()
        existing = existing.join(incoming, on="source_guid", how="anti")
        rows = pl.concat([existing, rows], how="vertical_relaxed")
    atomic_write_parquet(rows, output_path)
    return rows


def refresh_news(
    settings: Settings,
    *,
    offline: bool | None = None,
    sources: Iterable[str] | None = None,
    max_items_per_source: int = 20,
    api_key: str | None = None,
    client: Any | None = None,
    players_dim: pl.DataFrame | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Refresh and structure unseen news items.

    The optional Anthropic credential is read only from the environment. When it
    is absent, the operation is an intentional no-op so scheduled weekly runs
    remain healthy and never emit misleading review rows.
    """
    if max_items_per_source < 1:
        raise ValueError("max_items_per_source must be at least 1")

    selected_sources = list(sources or news.RSS_FEEDS)
    unknown = sorted(set(selected_sources) - set(news.RSS_FEEDS))
    if unknown:
        raise ValueError(f"Unknown news sources: {', '.join(unknown)}")

    resolved_key = api_key if api_key is not None else os.getenv("ANTHROPIC_API_KEY", "")
    if not resolved_key and client is None:
        return {
            "status": "skipped",
            "reason": "ANTHROPIC_API_KEY is not configured",
            "fetched": 0,
            "new_items": 0,
            "structured": 0,
            "reviewed": 0,
        }

    if players_dim is None:
        crosswalk = nflverse.fetch_player_ids(offline=offline, settings=settings)
        sleeper_players = sleeper.fetch_players(offline=offline, settings=settings)
        players_dim = mapping.build_players_dim(
            crosswalk, sleeper_players, mapping.ID_OVERRIDES_PATH
        )
    if client is None:
        client = anthropic.Anthropic(api_key=resolved_key)

    output_dir = settings.data_root / "outputs" / "news"
    events_path = output_dir / "events.parquet"
    review_path = output_dir / "manual_review.parquet"
    known = _known_guids(events_path, review_path)
    fetched = 0
    candidates: list[news.NewsItem] = []
    for source in selected_sources:
        path = news.fetch_rss_feed(source, offline=offline, settings=settings)
        items = news.parse_rss_entries(path.read_text(encoding="utf-8"), source=source)
        fetched += len(items)
        for item in items[:max_items_per_source]:
            if item.guid not in known:
                candidates.append(item)
                known.add(item.guid)

    processed_at = (now or datetime.now(UTC)).astimezone(UTC).isoformat()
    event_rows: list[dict[str, Any]] = []
    review_rows: list[pl.DataFrame] = []
    for item in candidates:
        event = news.structure_news_item(item, players_dim, client=client)
        if event is None:
            review_rows.append(news.build_review_row(item, reason="structuring_rejected"))
            continue
        row = asdict(event)
        row.update(
            {
                "source": item.source,
                "source_link": item.link,
                "published_at": item.pub_date,
                "processed_at_utc": processed_at,
            }
        )
        event_rows.append(row)

    if event_rows:
        _write_events(pl.DataFrame(event_rows, schema=EVENTS_SCHEMA), events_path)
    if review_rows:
        news.write_review_queue(pl.concat(review_rows), review_path)

    return {
        "status": "healthy",
        "reason": "",
        "fetched": fetched,
        "new_items": len(candidates),
        "structured": len(event_rows),
        "reviewed": len(review_rows),
        "events_path": str(events_path),
        "review_path": str(review_path),
    }


__all__ = ["EVENTS_SCHEMA", "refresh_news"]
