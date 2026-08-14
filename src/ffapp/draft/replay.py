"""Recorded-draft replay for local UI verification (SPEC-ADDENDUM-03.md §E).

Not a new data source and not part of the live draft assistant's own
public API (`draft.live`, task 0.14) -- this only picks a real, already
-completed draft out of the cache (or fetches one live) and reveals its
real picks over wall-clock time, so a page polling `picks_revealed_so_far`
can exercise its own auto-refresh behaviour the way a real live draft
would, without waiting for one. `ffapp draft live --replay` writes the
session; `picks_revealed_so_far` does all the actual "how far along are
we" math, purely from a timestamp -- nothing here runs continuously.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ffapp.config import LeagueConfig, Settings
from ffapp.ingest import sleeper


class NoCompletedDraftError(Exception):
    """No `status == "complete"` draft exists for this league to replay."""


@dataclass(frozen=True)
class ReplaySession:
    draft_id: str
    picks: list[dict[str, Any]]
    started_at_utc: str
    pace_seconds: float


def replay_session_path(settings: Settings, *, league_slug: str) -> Path:
    return settings.data_root / "outputs" / f"draft_replay_{league_slug}.json"


def _most_recently_completed_draft(drafts: list[dict[str, Any]]) -> dict[str, Any] | None:
    completed = [d for d in drafts if d.get("status") == "complete"]
    if not completed:
        return None
    return max(completed, key=lambda d: str(d.get("season", "")))


def _resolve_completed_draft(
    league: LeagueConfig, settings: Settings, *, offline: bool | None
) -> dict[str, Any]:
    """This season's own league object is pre-draft until the real draft
    happens, so it never has a completed draft of its own -- same problem
    `scoring.golden.run_golden_test` (SPEC §8.4) already solved for
    validating scoring against the most recently PLAYED season: resolve
    the league's own linked `previous_league_id` and look there instead.
    """
    assert league.league_id is not None
    drafts: list[dict[str, Any]] = json.loads(
        sleeper.fetch_drafts(league.league_id, offline=offline, settings=settings).read_text()
    )
    draft = _most_recently_completed_draft(drafts)
    if draft is not None:
        return draft

    current_raw = json.loads(
        sleeper.fetch_league(league.league_id, offline=offline, settings=settings).read_text()
    )
    previous_league_id = current_raw.get("previous_league_id")
    if previous_league_id:
        previous_drafts: list[dict[str, Any]] = json.loads(
            sleeper.fetch_drafts(previous_league_id, offline=offline, settings=settings).read_text()
        )
        draft = _most_recently_completed_draft(previous_drafts)
        if draft is not None:
            return draft

    raise NoCompletedDraftError(
        "No completed draft found for this league (or its previous season) to replay -- "
        "`ffapp cache warm` needs a real finished draft in this league's history."
    )


def start_replay(
    league: LeagueConfig, settings: Settings, *, pace_seconds: float, offline: bool | None = None
) -> ReplaySession:
    """Pick the most recent real completed draft for `league` (falling
    back to its previous season if the current one is still pre-draft) and
    record a fresh replay session starting now. `pace_seconds` sets how
    often another real pick becomes visible; `picks_revealed_so_far` does
    the actual reveal, computed fresh from wall-clock time on every call --
    this function only fetches the real picks once and writes them down.
    """
    draft = _resolve_completed_draft(league, settings, offline=offline)
    picks: list[dict[str, Any]] = json.loads(
        sleeper.fetch_draft_picks(draft["draft_id"], offline=offline, settings=settings).read_text()
    )
    picks_sorted = sorted(picks, key=lambda p: p.get("pick_no", 0))

    session = ReplaySession(
        draft_id=str(draft["draft_id"]),
        picks=picks_sorted,
        started_at_utc=datetime.now(UTC).isoformat(),
        pace_seconds=pace_seconds,
    )
    path = replay_session_path(settings, league_slug=league.slug)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "draft_id": session.draft_id,
                "picks": session.picks,
                "started_at_utc": session.started_at_utc,
                "pace_seconds": session.pace_seconds,
            }
        )
    )
    return session


def load_replay_session(settings: Settings, *, league_slug: str) -> ReplaySession | None:
    """`None` when no replay is active -- the caller's own signal to fall
    back to real live Sleeper polling instead."""
    path = replay_session_path(settings, league_slug=league_slug)
    if not path.exists():
        return None
    raw = json.loads(path.read_text())
    return ReplaySession(
        draft_id=raw["draft_id"],
        picks=raw["picks"],
        started_at_utc=raw["started_at_utc"],
        pace_seconds=raw["pace_seconds"],
    )


def picks_revealed_so_far(
    session: ReplaySession, *, now: datetime | None = None
) -> list[dict[str, Any]]:
    """How many of the session's real picks would have "happened" by `now`
    (defaults to the real current time), one every `pace_seconds`. Clamped
    to the real pick count both ways -- before the session starts (a clock
    skew, or `now` passed for a test) reveals nothing; once every pick is
    revealed, replay just holds at "draft complete" rather than erroring.
    """
    current = now if now is not None else datetime.now(UTC)
    started = datetime.fromisoformat(session.started_at_utc)
    elapsed = (current - started).total_seconds()
    if session.pace_seconds <= 0:
        revealed = len(session.picks)
    else:
        revealed = int(elapsed // session.pace_seconds)
    revealed = max(0, min(revealed, len(session.picks)))
    return session.picks[:revealed]


def stop_replay(settings: Settings, *, league_slug: str) -> None:
    replay_session_path(settings, league_slug=league_slug).unlink(missing_ok=True)


__all__ = [
    "NoCompletedDraftError",
    "ReplaySession",
    "load_replay_session",
    "picks_revealed_so_far",
    "replay_session_path",
    "start_replay",
    "stop_replay",
]
