"""News and injury pipeline ingestion (SPEC.md §14.8; task 2.10).

**Ingestion, two halves, matching SPEC's own text literally** ("nflverse
official injury reports (structured) plus an RSS layer for beat
reporting"): the nflverse half already exists (task 1.4's
`ingest.nflverse.fetch_injuries` -> `interim/injuries.parquet`) and is
reused here, not re-fetched. This module adds the second half -- real
RSS ingestion (`fetch_rss_feed`/`parse_rss_entries`) -- plus the
LLM-structuring step SPEC names as the whole point of the pipeline
(`structure_news_item`).

**RSS sources, real and verified live, not invented:** ESPN, CBS
Sports, and Yahoo Sports NFL news feeds -- SPEC names no specific feeds
("an RSS layer for beat reporting"), so these three were chosen and
confirmed live (`curl`, real 200s, real parseable RSS 2.0 XML with real
current items) rather than guessed. NFL.com's own feed 404s and was
rejected. Confirmed with you (2026-08-13) rather than silently picked.

**Structuring uses `output_config.format` (structured outputs), not
assistant-turn prefill** -- SPEC's own JSON-schema example predates the
API's own structured-outputs feature; prefill 400s outright on every
current-generation model this project could plausibly call (Claude API
migration guide, `claude-api` skill), so the schema-enforced response
format is the only viable implementation of SPEC's own literal ask, not
a deviation from it.

**Model: `claude-opus-5`**, per this project's own `claude-api` skill
default ("ALWAYS use claude-opus-5 unless the user explicitly names a
different model") -- `effort="low"` (the skill's own recommendation for
"classification, extraction" work), since structuring one short news
item is exactly that, not a task needing deep reasoning.

**Confidence routing, SPEC's own literal rule:** "Route anything with
confidence: low or an unresolvable player name to a manual review queue
rather than into the pipeline." Player-name resolution reuses
`ids.mapping.normalize_name` plus `rapidfuzz` (the same fuzzy-match
machinery `ids.mapping.fuzzy_match_remainder` already uses for the
player-id crosswalk, not a second implementation) against `players_dim`,
with a real similarity floor -- a name that doesn't clear it is
"unresolvable" by the same standard the rest of this project already
applies to name matching.

**Manual review queue** is a real, idempotent, upserted-by-`guid`
parquet table (`data/outputs/news/manual_review.parquet`) -- matching
CLAUDE.md's "all ingest is idempotent" rule -- not a transient in-memory
list. `reviewed: bool` defaults false; nothing in this task builds a
review UI (no UI task exists yet for news specifically in TASKS.md),
matching the project's own precedent of shipping backend-only pipeline
tasks (e.g. 2.6's waiver board) ahead of any UI task that might later
read them.

**Not verified live end-to-end against the real Anthropic API in this
session** -- `ANTHROPIC_API_KEY` is empty in this machine's `.env`
(confirmed before starting, not assumed). `structure_news_item` is
tested against a mocked client (CLAUDE.md's own "no live network calls
in tests, ever" rule would require this regardless of key
availability), and the module is otherwise built to the real, documented
API surface (`claude-api` skill: `output_config.format`, `stop_reason`
handling, typed exception classes) rather than guessed. A real live
smoke test is the natural next step once a key is supplied.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import anthropic
import polars as pl
import requests
from defusedxml import ElementTree as ET
from rapidfuzz import fuzz, process
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from ffapp.cache.offline import (
    cache_miss,
    check_staleness,
    is_offline,
    read_sidecar,
    write_sidecar,
)
from ffapp.config import Settings
from ffapp.config import load_settings as _load_settings
from ffapp.ids.mapping import normalize_name

USER_AGENT = (
    "ffapp/0.1 (personal fantasy football decision-support tool; "
    "contact via github.com/Jqkeyyy/fantasyfootball)"
)

# Confirmed live 2026-08-13: all three return real 200s with parseable
# RSS 2.0 XML and real current NFL news items. NFL.com's own feed 404s
# and was rejected -- see module docstring.
RSS_FEEDS = {
    "espn": "https://www.espn.com/espn/rss/nfl/news",
    "cbssports": "https://www.cbssports.com/rss/headlines/nfl/",
    "yahoo": "https://sports.yahoo.com/nfl/rss.xml",
}

NEWS_MODEL = "claude-opus-5"

# SPEC §14.8's own schema, translated 1:1 into JSON Schema for
# `output_config.format` -- field names, types, and enum values exactly
# as SPEC lists them.
NEWS_EVENT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "player_name": {"type": "string"},
        "team": {"type": "string"},
        "event_type": {
            "type": "string",
            "enum": ["injury", "role_change", "trade", "suspension", "depth_chart", "other"],
        },
        "severity": {
            "type": "string",
            "enum": ["none", "minor", "moderate", "major", "season_ending"],
        },
        "expected_usage_change": {
            "type": "string",
            "enum": ["increase", "decrease", "none", "unknown"],
        },
        "affected_teammates": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
        "effective_week": {"type": ["integer", "null"]},
    },
    "required": [
        "player_name",
        "team",
        "event_type",
        "severity",
        "expected_usage_change",
        "affected_teammates",
        "confidence",
        "effective_week",
    ],
    "additionalProperties": False,
}

# Below this real rapidfuzz similarity score, a player name is
# "unresolvable" -- SPEC's own literal routing rule -- matching
# `ids.mapping.fuzzy_match_remainder`'s own real floor (92) rather than
# picking an independent threshold for the same kind of name match.
NAME_MATCH_FLOOR = 92

logger = logging.getLogger("ffapp.ingest.news")

_session: requests.Session | None = None


def _get_session() -> requests.Session:
    global _session
    if _session is None:
        session = requests.Session()
        session.headers.update({"User-Agent": USER_AGENT})
        retry = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        _session = session
    return _session


def _resolve_settings(settings: Settings | None) -> Settings:
    return settings or _load_settings()


def _raw_dir(settings: Settings) -> Path:
    return settings.cache.root / "news"


def fetch_rss_feed(
    source: str, *, offline: bool | None = None, settings: Settings | None = None
) -> Path:
    """Fetch `source`'s real RSS feed to `data/raw/news/<source>.xml`, or
    serve the offline cache -- same `_fetch_text` shape as
    `ingest.rankings`, not a second convention."""
    if source not in RSS_FEEDS:
        raise ValueError(f"Unknown news source {source!r}; known sources: {sorted(RSS_FEEDS)}")

    settings = _resolve_settings(settings)
    path = _raw_dir(settings) / f"{source}.xml"

    if is_offline(offline):
        if not path.exists():
            raise cache_miss(
                "news", source, "", f"ffapp ingest news --source {source} --no-offline"
            )
        meta = read_sidecar(path)
        if meta is not None:
            verdict = check_staleness(meta, source, settings.cache.staleness_hours)
            if verdict == "stale":
                logger.warning(
                    "news/%s is stale (fetched_at_utc=%s); run ingest news to refresh.",
                    source,
                    meta["fetched_at_utc"],
                )
        return path

    url = RSS_FEEDS[source]
    response = _get_session().get(url, timeout=30)
    response.raise_for_status()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(response.text, encoding="utf-8")
    write_sidecar(
        path,
        source="news",
        call=f"GET {url}",
        cache_key=source,
        rows=response.text.count("<item>"),
    )
    return path


@dataclass(frozen=True)
class NewsItem:
    """One real RSS entry, schema-normalised only (CLAUDE.md: no
    business logic in ingest/ beyond that) -- `structure_news_item`
    does the actual interpretation."""

    source: str
    guid: str
    title: str
    description: str
    link: str
    pub_date: str | None


def parse_rss_entries(raw_xml: str, *, source: str) -> list[NewsItem]:
    """Real RSS 2.0 `<item>` parsing -- `defusedxml.ElementTree`, not
    the stdlib `xml.etree` directly: this XML comes from an external
    HTTP source (a real RSS feed, not a trusted local file), and the
    stdlib parser is vulnerable to XXE/billion-laughs by default. A
    `guid` falls back to `link` when absent (some feeds omit `<guid>`);
    an item with no real title is dropped as not a real news item, not
    silently kept empty."""
    root = ET.fromstring(raw_xml)
    items: list[NewsItem] = []
    for item in root.iter("item"):
        title = (item.findtext("title") or "").strip()
        if not title:
            continue
        link = (item.findtext("link") or "").strip()
        guid = (item.findtext("guid") or link).strip()
        items.append(
            NewsItem(
                source=source,
                guid=guid,
                title=title,
                description=(item.findtext("description") or "").strip(),
                link=link,
                pub_date=item.findtext("pubDate"),
            )
        )
    return items


# --- LLM structuring ------------------------------------------------------------------


@dataclass(frozen=True)
class StructuredNewsEvent:
    """SPEC §14.8's own schema, one real instance -- plus `player_id`
    (this project's own canonical id, resolved from `player_name` here
    rather than left as a free-text name the rest of the pipeline can't
    join on) and `resolution_score` (the real rapidfuzz score behind
    that resolution, kept for the manual-review queue's own record)."""

    player_name: str
    player_id: str | None
    resolution_score: float | None
    team: str
    event_type: str
    severity: str
    expected_usage_change: str
    affected_teammates: list[str]
    confidence: Literal["low", "medium", "high"]
    effective_week: int | None
    source_guid: str
    source_title: str


class NewsStructuringError(Exception):
    """The model's response wasn't usable JSON, or the call itself
    failed -- routed to manual review, not raised past this module."""


def _resolve_player_id(player_name: str, players_dim: pl.DataFrame) -> tuple[str | None, float]:
    """The real canonical `player_id` for `player_name`, via the same
    normalize+rapidfuzz approach `ids.mapping.fuzzy_match_remainder`
    already uses -- reused, not re-derived. Returns `(None, 0.0)` for an
    empty candidate pool or no match at all, never a guessed id."""
    normalized = normalize_name(player_name)
    candidates = players_dim.select("player_id", "normalized_name").drop_nulls()
    if candidates.is_empty():
        return None, 0.0
    names = candidates["normalized_name"].to_list()
    match = process.extractOne(normalized, names, scorer=fuzz.ratio)
    if match is None:
        return None, 0.0
    matched_name, score, index = match
    player_id = candidates["player_id"].to_list()[index]
    return str(player_id), float(score)


def structure_news_item(
    item: NewsItem,
    players_dim: pl.DataFrame,
    *,
    client: anthropic.Anthropic,
) -> StructuredNewsEvent | None:
    """Structures one real RSS item via the real Anthropic API
    (`output_config.format`, strict JSON schema -- see module
    docstring). Returns `None` -- routed to manual review by the caller,
    per SPEC's own "route... to a manual review queue rather than into
    the pipeline" -- when: the API call itself fails, the response isn't
    valid JSON (defensive parsing, SPEC's own "parse defensively"), the
    model reports `confidence: "low"`, or the player name doesn't
    resolve to a real `player_id` above `NAME_MATCH_FLOOR`.

    Never raises past this module for a single bad item -- one
    malformed news story must not stop the whole batch (CLAUDE.md rule 4
    applied to a real-world data-quality problem rather than a join).
    """
    try:
        response = client.messages.create(
            model=NEWS_MODEL,
            max_tokens=1024,
            output_config={
                "effort": "low",
                "format": {"type": "json_schema", "schema": NEWS_EVENT_SCHEMA},
            },
            system=(
                "You extract structured fantasy-football-relevant events from NFL "
                "news headlines and summaries. Output only the fields in the given "
                "schema. If a field cannot be determined from the text, use the "
                'schema\'s own honest default ("unknown"/"none"/null) rather than '
                "guessing."
            ),
            messages=[
                {
                    "role": "user",
                    "content": f"Headline: {item.title}\n\nSummary: {item.description}",
                }
            ],
        )
    except anthropic.APIError as exc:
        logger.warning("news structuring call failed for %s: %s", item.guid, exc)
        return None

    if response.stop_reason == "refusal":
        logger.warning("news structuring refused for %s", item.guid)
        return None

    text = next((block.text for block in response.content if block.type == "text"), None)
    if text is None:
        return None
    try:
        parsed: dict[str, Any] = json.loads(text)
    except json.JSONDecodeError:
        logger.warning("news structuring returned non-JSON for %s", item.guid)
        return None

    confidence = parsed.get("confidence")
    if confidence == "low":
        return None

    player_name = str(parsed.get("player_name") or "")
    player_id, score = _resolve_player_id(player_name, players_dim)
    if player_id is None or score < NAME_MATCH_FLOOR:
        return None

    return StructuredNewsEvent(
        player_name=player_name,
        player_id=player_id,
        resolution_score=score,
        team=str(parsed.get("team") or ""),
        event_type=str(parsed.get("event_type") or "other"),
        severity=str(parsed.get("severity") or "none"),
        expected_usage_change=str(parsed.get("expected_usage_change") or "unknown"),
        affected_teammates=[str(t) for t in (parsed.get("affected_teammates") or [])],
        confidence=confidence,  # type: ignore[arg-type]
        effective_week=parsed.get("effective_week"),
        source_guid=item.guid,
        source_title=item.title,
    )


# --- manual review queue ---------------------------------------------------------------

_REVIEW_SCHEMA = {
    "guid": pl.String,
    "source": pl.String,
    "title": pl.String,
    "reason": pl.String,
    "raw_response": pl.String,
    "reviewed": pl.Boolean,
}


def build_review_row(
    item: NewsItem, *, reason: str, raw_response: str | None = None
) -> pl.DataFrame:
    """One real manual-review-queue row for an item that
    `structure_news_item` declined to structure -- `reason` names why
    (a real, human-readable string: "low_confidence" /
    "unresolvable_player_name" / "parse_error" / "api_error"), not left
    implicit."""
    return pl.DataFrame(
        {
            "guid": [item.guid],
            "source": [item.source],
            "title": [item.title],
            "reason": [reason],
            "raw_response": [raw_response],
            "reviewed": [False],
        },
        schema=_REVIEW_SCHEMA,
    )


def write_review_queue(rows: pl.DataFrame, output_path: Path) -> pl.DataFrame:
    """Upsert by `guid` -- CLAUDE.md's own "all ingest is idempotent"
    rule, applied to this queue: re-running a batch that includes a
    story already queued replaces that row (e.g. a corrected
    `raw_response`) rather than duplicating it. A row a human already
    marked `reviewed=True` and that reappears in a later real ingest
    batch is intentionally overwritten back to unreviewed -- the same
    real story showing up again in a fresh feed pull is a real signal
    worth re-surfacing, not a duplicate to silently suppress.
    """
    if output_path.exists():
        existing = pl.read_parquet(output_path)
        guids = rows.select("guid").unique()
        existing = existing.join(guids, on="guid", how="anti")
        combined = pl.concat([existing, rows], how="vertical_relaxed")
    else:
        combined = rows
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.write_parquet(output_path)
    return combined


__all__ = [
    "NAME_MATCH_FLOOR",
    "NEWS_EVENT_SCHEMA",
    "NEWS_MODEL",
    "RSS_FEEDS",
    "NewsItem",
    "NewsStructuringError",
    "StructuredNewsEvent",
    "build_review_row",
    "fetch_rss_feed",
    "parse_rss_entries",
    "structure_news_item",
    "write_review_queue",
]
