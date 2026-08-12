"""SPEC-ADDENDUM-02.md §C.2 — the single most important rule in that addendum.

An offline mode that returns an empty dataframe on a cache miss produces a model
trained on partial data, a draft board missing forty players, and a golden test
that passes because it compared zero rows. Never return empty; always raise.
"""

from pathlib import Path

import pytest

from ffapp.cache.offline import OfflineCacheMiss
from ffapp.config import CacheSettings, Settings
from ffapp.ingest import nflverse, sleeper


@pytest.fixture
def empty_cache_settings(tmp_path: Path) -> Settings:
    return Settings(
        data_root=tmp_path,
        sleeper_username="fixture_user",
        cache=CacheSettings(
            root=tmp_path / "raw",
            offline_default=True,
            staleness_hours={
                "sleeper_league": 168,
                "sleeper_rosters": 24,
                "nflverse_player_ids": 168,
            },
            warn_on_stale=True,
        ),
    )


@pytest.mark.parametrize(
    "call",
    [
        lambda s: sleeper.fetch_user("Maybe17", offline=True, settings=s),
        lambda s: sleeper.fetch_leagues("abc", 2026, offline=True, settings=s),
        lambda s: sleeper.fetch_league("111", offline=True, settings=s),
        lambda s: sleeper.fetch_rosters("111", offline=True, settings=s),
        lambda s: sleeper.fetch_users("111", offline=True, settings=s),
        lambda s: sleeper.fetch_matchups("111", 14, offline=True, settings=s),
        lambda s: sleeper.fetch_transactions("111", 14, offline=True, settings=s),
        lambda s: sleeper.fetch_drafts("111", offline=True, settings=s),
        lambda s: sleeper.fetch_draft_picks("d1", offline=True, settings=s),
        lambda s: sleeper.fetch_traded_picks("111", offline=True, settings=s),
        lambda s: sleeper.fetch_trending("add", offline=True, settings=s),
        lambda s: sleeper.fetch_players(offline=True, settings=s),
        lambda s: nflverse.fetch_player_ids(offline=True, settings=s),
        lambda s: nflverse.fetch_player_stats(2025, offline=True, settings=s),
        lambda s: nflverse.fetch_team_stats(2025, offline=True, settings=s),
        lambda s: nflverse.fetch_schedules(2025, offline=True, settings=s),
        lambda s: nflverse.fetch_pbp(2025, offline=True, settings=s),
    ],
    ids=[
        "fetch_user",
        "fetch_leagues",
        "fetch_league",
        "fetch_rosters",
        "fetch_users",
        "fetch_matchups",
        "fetch_transactions",
        "fetch_drafts",
        "fetch_draft_picks",
        "fetch_traded_picks",
        "fetch_trending",
        "fetch_players",
        "fetch_player_ids",
        "fetch_player_stats",
        "fetch_team_stats",
        "fetch_schedules",
        "fetch_pbp",
    ],
)
def test_cache_miss_under_offline_raises_not_returns_empty(
    call: object, empty_cache_settings: Settings
) -> None:
    with pytest.raises(OfflineCacheMiss):
        call(empty_cache_settings)  # type: ignore[operator]


def test_offline_cache_miss_message_names_the_fetch_command_to_run(
    empty_cache_settings: Settings,
) -> None:
    with pytest.raises(OfflineCacheMiss) as exc_info:
        sleeper.fetch_league("111", offline=True, settings=empty_cache_settings)

    message = str(exc_info.value)
    assert "not cached" in message
    assert "cache warm" in message
