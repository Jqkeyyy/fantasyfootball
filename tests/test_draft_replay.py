import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from ffapp.config import CacheSettings, LeagueConfig, Settings
from ffapp.draft import replay as draft_replay
from ffapp.ingest import sleeper

_LEAGUE = LeagueConfig(
    slug="test-league",
    display_name="Test League",
    is_primary=True,
    league_id="1",
    season=2026,
    league_cache={},
    overrides={},
)


@pytest.fixture
def fixture_settings(tmp_path: Path) -> Settings:
    return Settings(
        data_root=tmp_path,
        sleeper_username="fixture_user",
        cache=CacheSettings(
            root=tmp_path / "raw", offline_default=True, staleness_hours={}, warn_on_stale=True
        ),
    )


def _pick(pick_no: int) -> dict:
    return {
        "draft_id": "d1",
        "pick_no": pick_no,
        "metadata": {"first_name": "P", "last_name": str(pick_no), "position": "RB"},
    }


# --- replay_session_path -----------------------------------------------------------


def test_replay_session_path(fixture_settings: Settings) -> None:
    result = draft_replay.replay_session_path(fixture_settings, league_slug="test-league")

    assert result == fixture_settings.data_root / "outputs" / "draft_replay_test-league.json"


# --- start_replay --------------------------------------------------------------------


def test_start_replay_picks_the_completed_draft_and_sorts_picks_by_pick_no(
    monkeypatch: pytest.MonkeyPatch, fixture_settings: Settings
) -> None:
    drafts_path = fixture_settings.data_root / "drafts.json"
    drafts_path.write_text(
        json.dumps(
            [
                {"draft_id": "pre", "season": "2026", "status": "pre_draft"},
                {"draft_id": "d1", "season": "2025", "status": "complete"},
            ]
        )
    )
    picks_path = fixture_settings.data_root / "picks.json"
    picks_path.write_text(json.dumps([_pick(3), _pick(1), _pick(2)]))

    monkeypatch.setattr(sleeper, "fetch_drafts", lambda league_id, **kwargs: drafts_path)
    monkeypatch.setattr(sleeper, "fetch_draft_picks", lambda draft_id, **kwargs: picks_path)

    session = draft_replay.start_replay(_LEAGUE, fixture_settings, pace_seconds=5.0)

    assert session.draft_id == "d1"
    assert [p["pick_no"] for p in session.picks] == [1, 2, 3]
    assert session.pace_seconds == 5.0

    written = json.loads(
        draft_replay.replay_session_path(fixture_settings, league_slug="test-league").read_text()
    )
    assert written["draft_id"] == "d1"
    assert len(written["picks"]) == 3


def test_start_replay_falls_back_to_the_previous_season_when_current_is_pre_draft(
    monkeypatch: pytest.MonkeyPatch, fixture_settings: Settings
) -> None:
    """A league is pre-draft for its own current-season config until the
    real draft happens -- same real situation `scoring.golden
    .run_golden_test` already had to solve for validating scoring against
    the most recently PLAYED season."""
    current_drafts_path = fixture_settings.data_root / "current_drafts.json"
    current_drafts_path.write_text(
        json.dumps([{"draft_id": "pre", "season": "2026", "status": "pre_draft"}])
    )
    previous_drafts_path = fixture_settings.data_root / "previous_drafts.json"
    previous_drafts_path.write_text(
        json.dumps([{"draft_id": "d1", "season": "2025", "status": "complete"}])
    )
    league_path = fixture_settings.data_root / "league.json"
    league_path.write_text(json.dumps({"previous_league_id": "prev-1"}))
    picks_path = fixture_settings.data_root / "picks.json"
    picks_path.write_text(json.dumps([_pick(1)]))

    def fake_fetch_drafts(league_id: str, **kwargs: object) -> Path:
        return previous_drafts_path if league_id == "prev-1" else current_drafts_path

    monkeypatch.setattr(sleeper, "fetch_drafts", fake_fetch_drafts)
    monkeypatch.setattr(sleeper, "fetch_league", lambda league_id, **kwargs: league_path)
    monkeypatch.setattr(sleeper, "fetch_draft_picks", lambda draft_id, **kwargs: picks_path)

    session = draft_replay.start_replay(_LEAGUE, fixture_settings, pace_seconds=5.0)

    assert session.draft_id == "d1"


def test_start_replay_raises_when_neither_current_nor_previous_season_has_a_completed_draft(
    monkeypatch: pytest.MonkeyPatch, fixture_settings: Settings
) -> None:
    drafts_path = fixture_settings.data_root / "drafts.json"
    drafts_path.write_text(
        json.dumps([{"draft_id": "pre", "season": "2026", "status": "pre_draft"}])
    )
    league_path = fixture_settings.data_root / "league.json"
    league_path.write_text(json.dumps({}))  # no previous_league_id at all

    monkeypatch.setattr(sleeper, "fetch_drafts", lambda league_id, **kwargs: drafts_path)
    monkeypatch.setattr(sleeper, "fetch_league", lambda league_id, **kwargs: league_path)

    with pytest.raises(draft_replay.NoCompletedDraftError):
        draft_replay.start_replay(_LEAGUE, fixture_settings, pace_seconds=5.0)


# --- load_replay_session -------------------------------------------------------------


def test_load_replay_session_returns_none_when_no_session_exists(
    fixture_settings: Settings,
) -> None:
    assert draft_replay.load_replay_session(fixture_settings, league_slug="test-league") is None


def test_load_replay_session_round_trips_a_written_session(
    monkeypatch: pytest.MonkeyPatch, fixture_settings: Settings
) -> None:
    drafts_path = fixture_settings.data_root / "drafts.json"
    drafts_path.write_text(json.dumps([{"draft_id": "d1", "season": "2025", "status": "complete"}]))
    picks_path = fixture_settings.data_root / "picks.json"
    picks_path.write_text(json.dumps([_pick(1)]))
    monkeypatch.setattr(sleeper, "fetch_drafts", lambda league_id, **kwargs: drafts_path)
    monkeypatch.setattr(sleeper, "fetch_draft_picks", lambda draft_id, **kwargs: picks_path)
    draft_replay.start_replay(_LEAGUE, fixture_settings, pace_seconds=5.0)

    loaded = draft_replay.load_replay_session(fixture_settings, league_slug="test-league")

    assert loaded is not None
    assert loaded.draft_id == "d1"
    assert len(loaded.picks) == 1


# --- picks_revealed_so_far ------------------------------------------------------------


def _session(pace_seconds: float = 10.0, n_picks: int = 5) -> draft_replay.ReplaySession:
    return draft_replay.ReplaySession(
        draft_id="d1",
        picks=[_pick(i) for i in range(1, n_picks + 1)],
        started_at_utc=datetime(2026, 8, 13, 12, 0, 0, tzinfo=UTC).isoformat(),
        pace_seconds=pace_seconds,
    )


def test_picks_revealed_so_far_reveals_nothing_before_the_first_interval() -> None:
    session = _session(pace_seconds=10.0)
    now = datetime(2026, 8, 13, 12, 0, 5, tzinfo=UTC)

    assert draft_replay.picks_revealed_so_far(session, now=now) == []


def test_picks_revealed_so_far_reveals_one_pick_per_elapsed_interval() -> None:
    session = _session(pace_seconds=10.0)
    now = datetime(2026, 8, 13, 12, 0, 25, tzinfo=UTC)  # 2.5 intervals elapsed

    result = draft_replay.picks_revealed_so_far(session, now=now)

    assert [p["pick_no"] for p in result] == [1, 2]


def test_picks_revealed_so_far_clamps_to_the_real_pick_count() -> None:
    session = _session(pace_seconds=10.0, n_picks=3)
    now = datetime(2026, 8, 13, 13, 0, 0, tzinfo=UTC)  # far past every pick

    result = draft_replay.picks_revealed_so_far(session, now=now)

    assert len(result) == 3


def test_picks_revealed_so_far_reveals_everything_immediately_when_pace_is_zero() -> None:
    session = _session(pace_seconds=0.0, n_picks=4)
    now = datetime(2026, 8, 13, 12, 0, 0, tzinfo=UTC)

    result = draft_replay.picks_revealed_so_far(session, now=now)

    assert len(result) == 4


def test_picks_revealed_so_far_never_goes_negative_before_the_session_started() -> None:
    session = _session(pace_seconds=10.0)
    now = datetime(2026, 8, 13, 11, 59, 0, tzinfo=UTC)  # before started_at_utc

    assert draft_replay.picks_revealed_so_far(session, now=now) == []


# --- stop_replay -----------------------------------------------------------------------


def test_stop_replay_removes_the_session_file(
    monkeypatch: pytest.MonkeyPatch, fixture_settings: Settings
) -> None:
    drafts_path = fixture_settings.data_root / "drafts.json"
    drafts_path.write_text(json.dumps([{"draft_id": "d1", "season": "2025", "status": "complete"}]))
    picks_path = fixture_settings.data_root / "picks.json"
    picks_path.write_text(json.dumps([_pick(1)]))
    monkeypatch.setattr(sleeper, "fetch_drafts", lambda league_id, **kwargs: drafts_path)
    monkeypatch.setattr(sleeper, "fetch_draft_picks", lambda draft_id, **kwargs: picks_path)
    draft_replay.start_replay(_LEAGUE, fixture_settings, pace_seconds=5.0)

    draft_replay.stop_replay(fixture_settings, league_slug="test-league")

    assert draft_replay.load_replay_session(fixture_settings, league_slug="test-league") is None


def test_stop_replay_is_a_no_op_when_no_session_exists(fixture_settings: Settings) -> None:
    draft_replay.stop_replay(fixture_settings, league_slug="test-league")  # must not raise
