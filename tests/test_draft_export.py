from pathlib import Path

import polars as pl
import pytest

from ffapp.cache import offline as offline_cache
from ffapp.config import CacheSettings, LeagueConfig, Settings
from ffapp.draft import board as draft_board
from ffapp.draft import export as draft_export

_LEAGUE = LeagueConfig(
    slug="test-league",
    display_name="Test League",
    is_primary=True,
    league_id="1",
    season=2026,
    league_cache={},
    overrides={},
)

_BOARD = pl.DataFrame(
    {
        "overall_rank": [1, 2, 3],
        "pos_rank": [1, 1, 1],
        "tier": [1, 1, 2],
        "player": ["Elite RB", "Good WR", "Deep Bench TE"],
        "is_keeper": [True, False, False],
        "position": ["RB", "WR", "TE"],
        "team": ["DET", "CIN", "KC"],
        "bye_week": [6, 10, 12],
        "proj_points_adj": [280.0, 210.0, 90.0],
        "proj_ppg": [16.5, 12.4, 5.3],
        "expected_games": [17.0, 17.0, 17.0],
        "vor": [150.0, 90.0, 10.0],
        "dispersion": [5.0, 4.0, 2.0],
        "n_sources": [4, 4, 3],
        "adp": [2.0, 15.0, None],
        "adp_sd": [1.0, 2.0, None],
        "value_vs_adp": [0, 0, None],
        "p_avail_next": [0.1, 0.6, 1.0],
        "opportunity_cost": [40.0, 5.0, 0.0],
        "playoff_sos": [None, None, None],
        "as_of_utc": ["2026-08-13T00:00:00+00:00"] * 3,
        "git_commit": ["abc123"] * 3,
    }
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


# --- build_export_bundle ------------------------------------------------------


def test_build_export_bundle_reuses_build_draft_board_for_the_board(
    monkeypatch: pytest.MonkeyPatch, fixture_settings: Settings
) -> None:
    monkeypatch.setattr(draft_board, "build_draft_board", lambda league, settings, **kwargs: _BOARD)
    monkeypatch.setattr(
        draft_board,
        "resolve_pick_context",
        lambda league, settings, **kwargs: draft_board.PickContext(
            my_roster_id=7, my_slot=3, my_picks=[3, 18, 27]
        ),
    )
    monkeypatch.setattr(
        draft_export.rankings,
        "fetch_adp",
        lambda *args, **kwargs: fixture_settings.data_root / "adp.json",
    )
    monkeypatch.setattr(draft_export, "_RANKINGS_SOURCE_FETCHERS", {})

    bundle = draft_export.build_export_bundle(_LEAGUE, fixture_settings, season=2026)

    assert bundle.board.equals(_BOARD)
    assert bundle.my_slot == 3
    assert bundle.my_picks == [3, 18, 27]
    assert bundle.rankings_ages == []


def test_build_export_bundle_skips_a_rankings_source_that_fails_to_fetch(
    monkeypatch: pytest.MonkeyPatch, fixture_settings: Settings
) -> None:
    monkeypatch.setattr(draft_board, "build_draft_board", lambda league, settings, **kwargs: _BOARD)
    monkeypatch.setattr(
        draft_board,
        "resolve_pick_context",
        lambda league, settings, **kwargs: draft_board.PickContext(
            my_roster_id=7, my_slot=3, my_picks=[3, 18]
        ),
    )
    monkeypatch.setattr(
        draft_export.rankings,
        "fetch_adp",
        lambda *args, **kwargs: fixture_settings.data_root / "adp.json",
    )

    def boom(season, teams, offline, settings):
        raise RuntimeError("403 Forbidden")

    working_path = fixture_settings.data_root / "fantasypros.csv"
    monkeypatch.setattr(
        draft_export,
        "_RANKINGS_SOURCE_FETCHERS",
        {
            "espn": boom,
            "fantasypros": lambda season, teams, offline, settings: working_path,
        },
    )

    bundle = draft_export.build_export_bundle(_LEAGUE, fixture_settings, season=2026)

    assert [age.name for age in bundle.rankings_ages] == ["fantasypros"]


# --- _input_age ----------------------------------------------------------------


def test_input_age_reads_the_real_sidecar(tmp_path: Path) -> None:
    raw_path = tmp_path / "adp_2026_12_ppr.json"
    raw_path.write_text("{}")
    offline_cache.write_sidecar(raw_path, source="adp", call="GET x", rows=100)

    result = draft_export._input_age("adp", raw_path)

    assert result.name == "adp"
    assert result.fetched_at_utc is not None
    assert result.age_hours is not None
    assert result.age_hours < 1.0  # just written


def test_input_age_handles_a_missing_sidecar(tmp_path: Path) -> None:
    result = draft_export._input_age("adp", tmp_path / "missing.json")

    assert result.fetched_at_utc is None
    assert result.age_hours is None


# --- render_html ----------------------------------------------------------------


def _bundle(**overrides: object) -> draft_export.ExportBundle:
    defaults: dict[str, object] = dict(
        board=_BOARD,
        generated_at_utc="2026-08-13T12:00:00+00:00",
        my_slot=3,
        my_picks=[3, 18, 27],
        adp_age=draft_export.InputAge(
            name="adp", fetched_at_utc="2026-08-13T10:00:00+00:00", age_hours=2.0
        ),
        rankings_ages=[
            draft_export.InputAge(
                name="fantasypros", fetched_at_utc="2026-08-13T09:00:00+00:00", age_hours=3.0
            ),
        ],
    )
    defaults.update(overrides)
    return draft_export.ExportBundle(**defaults)  # type: ignore[arg-type]


def test_render_html_embeds_every_player_as_inline_json() -> None:
    result = draft_export.render_html(_bundle(), league=_LEAGUE)

    assert "Elite RB" in result
    assert "Good WR" in result
    assert "Deep Bench TE" in result


def test_render_html_embeds_the_is_keeper_flag_in_the_inline_json() -> None:
    result = draft_export.render_html(_bundle(), league=_LEAGUE)

    assert '"is_keeper":true' in result.replace(" ", "")


def test_render_html_includes_keeper_row_styling_and_marker_logic() -> None:
    result = draft_export.render_html(_bundle(), league=_LEAGUE)

    assert "keeper-row" in result
    assert "is_keeper" in result
    assert "\U0001f512" in result  # lock emoji marker


def test_render_html_has_no_external_network_references() -> None:
    result = draft_export.render_html(_bundle(), league=_LEAGUE)

    lowered = result.lower()
    for forbidden in ("http://", "https://", "//cdn", "<link ", "googleapis"):
        assert forbidden not in lowered


def test_render_html_includes_position_filter_and_search_controls() -> None:
    result = draft_export.render_html(_bundle(), league=_LEAGUE)

    assert 'id="search"' in result
    assert 'data-pos="ALL"' in result
    assert 'data-pos="RB"' in result
    assert 'data-pos="WR"' in result
    assert 'data-pos="TE"' in result


def test_render_html_header_states_generation_time_and_input_ages() -> None:
    result = draft_export.render_html(_bundle(), league=_LEAGUE)

    assert "2026-08-13T12:00:00+00:00" in result
    assert "adp: 2.0h old" in result
    assert "fantasypros: 3.0h old" in result


def test_render_html_flags_stale_inputs_past_the_24h_policy() -> None:
    stale_bundle = _bundle(
        adp_age=draft_export.InputAge(
            name="adp", fetched_at_utc="2026-08-10T10:00:00+00:00", age_hours=48.0
        )
    )

    result = draft_export.render_html(stale_bundle, league=_LEAGUE)

    assert "adp: 48.0h old [STALE]" in result


def test_render_html_shows_draft_slot_and_pick_numbers() -> None:
    result = draft_export.render_html(_bundle(), league=_LEAGUE)

    assert "Slot 3" in result
    assert "3, 18, 27" in result


def test_render_html_handles_unresolved_slot_and_picks_gracefully() -> None:
    result = draft_export.render_html(_bundle(my_slot=None, my_picks=[]), league=_LEAGUE)

    assert "slot unknown" in result
    assert "none resolved" in result


def test_format_age_reports_unknown_for_a_missing_sidecar() -> None:
    result = draft_export._format_age(
        draft_export.InputAge(name="adp", fetched_at_utc=None, age_hours=None)
    )

    assert result == "adp: unknown (not cached)"


# --- path helpers ----------------------------------------------------------------


def test_export_html_and_csv_paths(fixture_settings: Settings) -> None:
    assert draft_export.export_html_path(fixture_settings, season=2026) == (
        fixture_settings.data_root / "outputs" / "draft_board_2026_export.html"
    )
    assert draft_export.export_csv_path(fixture_settings, season=2026) == (
        fixture_settings.data_root / "outputs" / "draft_board_2026_export.csv"
    )
