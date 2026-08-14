import polars as pl
import pytest

from ffapp.features import team_context
from ffapp.features.registry import FeatureSpec

# --- generic windowing primitives (team-grouped) -------------------------------------


def test_ewm_resets_at_a_season_boundary() -> None:
    df = pl.DataFrame(
        {
            "team": ["KC", "KC", "KC", "KC"],
            "season": [2025, 2025, 2025, 2026],
            "week": [1, 2, 3, 1],
            "x": [0.1, 0.3, 0.5, 0.9],
        }
    )

    result = team_context.ewm(df, "x", 3, "x_ewm_3")

    rows = result.sort(["season", "week"]).to_dicts()
    assert rows[0]["x_ewm_3"] == pytest.approx(0.1)
    assert rows[3]["x_ewm_3"] == pytest.approx(0.9)  # new season: reset


# --- add_opponent_pace ------------------------------------------------------------------


def test_add_opponent_pace_uses_the_opponents_own_lagged_pace_not_this_weeks() -> None:
    """Regression test for a real leakage bug found in task review: team T
    and its week-W opponent O play the same game, so O's own week-W
    neutral_pace_ewm_8 already reflects that shared game. The opponent's
    pace used must be their own value from *before* week W (their week
    W-1), never their own week-W value."""
    team_context_with_windows = pl.DataFrame(
        {
            "team": ["KC", "KC", "BAL", "DEN", "DEN"],
            "season": [2025, 2025, 2025, 2025, 2025],
            "week": [1, 2, 1, 1, 2],
            "neutral_pace_ewm_8": [28.0, 27.0, 31.5, 29.0, 30.0],
        }
    )
    schedule = pl.DataFrame(
        {
            "season": [2025, 2025],
            "week": [1, 2],
            "home_team": ["KC", "KC"],
            "away_team": ["BAL", "DEN"],
        }
    )

    result = team_context.add_opponent_pace(team_context_with_windows, schedule)

    kc_week2 = result.filter((pl.col("team") == "KC") & (pl.col("week") == 2)).row(0, named=True)
    # KC's week-2 opponent is DEN; must use DEN's own week-1 (lagged) value (29.0),
    # never DEN's own week-2 value (30.0) -- that would be the same-game leak.
    assert kc_week2["opponent_neutral_pace_ewm_8"] == pytest.approx(29.0)


def test_add_opponent_pace_is_null_for_a_teams_first_tracked_week() -> None:
    """KC's week-1 opponent (BAL) has no prior week to lag from -- honestly
    null, same convention as every other first-tracked-week trailing value
    in this project."""
    team_context_with_windows = pl.DataFrame(
        {
            "team": ["KC", "BAL"],
            "season": [2025, 2025],
            "week": [1, 1],
            "neutral_pace_ewm_8": [28.0, 31.5],
        }
    )
    schedule = pl.DataFrame(
        {"season": [2025], "week": [1], "home_team": ["KC"], "away_team": ["BAL"]}
    )

    result = team_context.add_opponent_pace(team_context_with_windows, schedule)

    kc_week1 = result.filter((pl.col("team") == "KC") & (pl.col("week") == 1)).row(0, named=True)
    assert kc_week1["opponent_neutral_pace_ewm_8"] is None


def test_add_opponent_pace_is_null_for_a_bye_week() -> None:
    """Distinct from the first-tracked-week case above: here the opponent
    resolution itself fails (no scheduled game at all for this team this
    week), rather than resolving to a real opponent who simply has no
    prior-week value to lag from."""
    team_context_with_windows = pl.DataFrame(
        {"team": ["KC"], "season": [2025], "week": [1], "neutral_pace_ewm_8": [28.0]}
    )
    schedule = pl.DataFrame(
        {"season": [], "week": [], "home_team": [], "away_team": []},
        schema={"season": pl.Int64, "week": pl.Int64, "home_team": pl.Utf8, "away_team": pl.Utf8},
    )  # no games scheduled at all -- KC's week 1 is unresolvable, same as a real bye

    result = team_context.add_opponent_pace(team_context_with_windows, schedule)

    assert result["opponent_neutral_pace_ewm_8"].to_list() == [None]


# --- _starting_ol_by_game --------------------------------------------------------------


def _snap_row(**kwargs: object) -> dict:
    row: dict[str, object] = {
        "season": 2025,
        "week": 1,
        "team": "KC",
        "pfr_player_id": "PlayerA",
        "position": "C",
        "offense_snaps": 50,
    }
    row.update(kwargs)
    return row


def test_starting_ol_by_game_picks_top_snap_player_per_slot() -> None:
    snap_counts = pl.DataFrame(
        [
            _snap_row(pfr_player_id="c1", position="C", offense_snaps=60),
            _snap_row(pfr_player_id="c2", position="C", offense_snaps=5),  # backup, excluded
            _snap_row(pfr_player_id="g1", position="G", offense_snaps=60),
            _snap_row(pfr_player_id="g2", position="G", offense_snaps=58),
            _snap_row(pfr_player_id="g3", position="G", offense_snaps=3),  # 3rd G, excluded
            _snap_row(pfr_player_id="t1", position="T", offense_snaps=60),
            _snap_row(pfr_player_id="t2", position="T", offense_snaps=57),
            _snap_row(pfr_player_id="wr1", position="WR", offense_snaps=60),  # not OL
        ]
    )

    result = team_context._starting_ol_by_game(snap_counts)

    ids = set(result["pfr_player_id"].to_list())
    assert ids == {"c1", "g1", "g2", "t1", "t2"}


# --- ol_continuity raw signal -----------------------------------------------------------


def test_ol_continuity_raw_is_full_when_all_five_starters_repeat() -> None:
    week1 = [
        _snap_row(week=1, pfr_player_id=pid, position=pos, offense_snaps=60)
        for pid, pos in [("c1", "C"), ("g1", "G"), ("g2", "G"), ("t1", "T"), ("t2", "T")]
    ]
    week2 = [
        _snap_row(week=2, pfr_player_id=pid, position=pos, offense_snaps=60)
        for pid, pos in [("c1", "C"), ("g1", "G"), ("g2", "G"), ("t1", "T"), ("t2", "T")]
    ]
    snap_counts = pl.DataFrame(week1 + week2)

    result = team_context.ol_continuity_raw(snap_counts)

    rows = {row["week"]: row for row in result.iter_rows(named=True)}
    assert rows[1]["ol_continuity_raw"] is None  # no prior week to compare to
    assert rows[2]["ol_continuity_raw"] == pytest.approx(1.0)


def test_ol_continuity_raw_reflects_a_partial_lineup_change() -> None:
    week1 = [
        _snap_row(week=1, pfr_player_id=pid, position=pos, offense_snaps=60)
        for pid, pos in [("c1", "C"), ("g1", "G"), ("g2", "G"), ("t1", "T"), ("t2", "T")]
    ]
    # week 2: one guard is swapped out (injury) -- 4 of 5 repeat.
    week2 = [
        _snap_row(week=2, pfr_player_id=pid, position=pos, offense_snaps=60)
        for pid, pos in [("c1", "C"), ("g1", "G"), ("g_new", "G"), ("t1", "T"), ("t2", "T")]
    ]
    snap_counts = pl.DataFrame(week1 + week2)

    result = team_context.ol_continuity_raw(snap_counts)

    row = result.filter(pl.col("week") == 2).row(0, named=True)
    assert row["ol_continuity_raw"] == pytest.approx(4 / 5)


def test_ol_continuity_raw_resets_at_a_season_boundary() -> None:
    """A new season's week 1 has no *in-season* prior week to compare to,
    even if the team's real week-17-of-last-season snap data is present in
    the same input -- consistent with every other window in this project
    being computed within-season only (see features/usage.py's own
    module docstring for that same convention)."""
    prior_season = [
        _snap_row(season=2024, week=17, pfr_player_id=pid, position=pos, offense_snaps=60)
        for pid, pos in [("c1", "C"), ("g1", "G"), ("g2", "G"), ("t1", "T"), ("t2", "T")]
    ]
    new_season = [
        _snap_row(season=2025, week=1, pfr_player_id=pid, position=pos, offense_snaps=60)
        for pid, pos in [("c1", "C"), ("g1", "G"), ("g2", "G"), ("t1", "T"), ("t2", "T")]
    ]
    snap_counts = pl.DataFrame(prior_season + new_season)

    result = team_context.ol_continuity_raw(snap_counts)

    row = result.filter((pl.col("season") == 2025) & (pl.col("week") == 1)).row(0, named=True)
    assert row["ol_continuity_raw"] is None


# --- add_vacated_shares -----------------------------------------------------------------


def _usage_features_row(**kwargs: object) -> dict:
    row: dict[str, object] = {
        "player_id": "wr1",
        "season": 2025,
        "week": 3,
        "team": "KC",
        "target_share_ewm_3": 0.30,
        "carry_share_ewm_3": 0.0,
    }
    row.update(kwargs)
    return row


def _injury_row(**kwargs: object) -> dict:
    row: dict[str, object] = {
        "player_id": "wr1",
        "season": 2025,
        "week": 3,
        "team": "KC",
        "report_status": "Out",
    }
    row.update(kwargs)
    return row


def _twc_row(**kwargs: object) -> dict:
    row: dict[str, object] = {"team": "KC", "season": 2025, "week": 3}
    row.update(kwargs)
    return row


def test_add_vacated_shares_looks_up_the_out_players_last_played_week() -> None:
    """A genuinely `Out` player has no usage row for the week they missed
    (they didn't play) -- their *established* role, from the last week
    they did play, is what "vacated" means."""
    twc = pl.DataFrame([_twc_row(team="KC", week=3), _twc_row(team="BAL", week=3)])
    injuries = pl.DataFrame(
        [
            _injury_row(player_id="wr1", team="KC", week=3, report_status="Out"),
            _injury_row(player_id="wr2", team="KC", week=3, report_status="Questionable"),
        ]
    )
    usage_features = pl.DataFrame(
        [
            # wr1's own week-3 row does NOT exist (Out, didn't play) --
            # only their week-2 established role does.
            _usage_features_row(player_id="wr1", team="KC", week=2, target_share_ewm_3=0.30),
            _usage_features_row(player_id="wr2", team="KC", week=3, target_share_ewm_3=0.15),
        ]
    )

    result = team_context.add_vacated_shares(twc, injuries, usage_features)

    rows = {row["team"]: row for row in result.iter_rows(named=True)}
    # Only wr1 (Out) counts, looked up from their week-2 role -- not wr2
    # (Questionable, still has their own week-3 row and isn't summed).
    assert rows["KC"]["teammate_vacated_target_share"] == pytest.approx(0.30)
    assert rows["BAL"]["teammate_vacated_target_share"] == pytest.approx(0.0)


def test_add_vacated_shares_sums_multiple_ruled_out_teammates() -> None:
    twc = pl.DataFrame([_twc_row(team="KC", week=3)])
    injuries = pl.DataFrame(
        [
            _injury_row(player_id="wr1", team="KC", week=3, report_status="Out"),
            _injury_row(player_id="rb1", team="KC", week=3, report_status="Out"),
        ]
    )
    usage_features = pl.DataFrame(
        [
            _usage_features_row(
                player_id="wr1",
                team="KC",
                week=2,
                target_share_ewm_3=0.30,
                carry_share_ewm_3=0.0,
            ),
            _usage_features_row(
                player_id="rb1",
                team="KC",
                week=2,
                target_share_ewm_3=0.05,
                carry_share_ewm_3=0.55,
            ),
        ]
    )

    result = team_context.add_vacated_shares(twc, injuries, usage_features)

    row = result.row(0, named=True)
    assert row["teammate_vacated_target_share"] == pytest.approx(0.35)
    assert row["teammate_vacated_carry_share"] == pytest.approx(0.55)


def test_add_vacated_shares_ignores_a_stray_same_week_usage_row() -> None:
    """Guards the `available_shares` anti-join: if an `Out` player somehow
    still has a stray usage row for the same week they were ruled out
    (shouldn't happen for real data, but not assumed), the asof lookup
    must skip it and fall back to their true prior week, not treat the
    stray row as their established role."""
    twc = pl.DataFrame([_twc_row(team="KC", week=3)])
    injuries = pl.DataFrame([_injury_row(player_id="wr1", team="KC", week=3, report_status="Out")])
    usage_features = pl.DataFrame(
        [
            _usage_features_row(player_id="wr1", team="KC", week=2, target_share_ewm_3=0.30),
            # Stray same-week row -- must be excluded, not used as-is.
            _usage_features_row(player_id="wr1", team="KC", week=3, target_share_ewm_3=0.0),
        ]
    )

    result = team_context.add_vacated_shares(twc, injuries, usage_features)

    assert result.row(0, named=True)["teammate_vacated_target_share"] == pytest.approx(0.30)


# --- build_team_context_features (integration) -------------------------------------------


def test_build_team_context_features_registers_every_feature() -> None:
    twc = pl.DataFrame(
        {
            "team": ["KC"],
            "season": [2025],
            "week": [1],
            "plays": [60],
            "neutral_pace_sec": [28.0],
            "proe": [0.02],
            "epa_per_play_off": [0.1],
            "success_rate_off": [0.48],
            "implied_total": [24.5],
            "spread": [-3.0],
        }
    )
    schedule = pl.DataFrame(
        schema={"season": pl.Int64, "week": pl.Int64, "home_team": pl.Utf8, "away_team": pl.Utf8}
    )
    snap_counts = pl.DataFrame(
        [_snap_row(pfr_player_id=pid, position=pos) for pid, pos in [("c1", "C")]]
    )
    injuries = pl.DataFrame([_injury_row()]).clear()  # no rows, correct empty schema
    usage_features = pl.DataFrame([_usage_features_row()]).clear()

    registry: dict[str, object] = {}
    team_context.build_team_context_features(
        twc, schedule, snap_counts, injuries, usage_features, registry=registry
    )

    expected = {
        "plays_per_game_ewm_5",
        "neutral_pace_ewm_8",
        "proe_ewm_5",
        "team_epa_off_ewm_8",
        "team_success_off_ewm_8",
        "ol_continuity_ewm_5",
        "implied_team_total",
        "spread",
        "teammate_vacated_target_share",
        "teammate_vacated_carry_share",
    }
    assert expected <= registry.keys()
    for name in expected:
        spec = registry[name]
        assert spec.lag_weeks == 1
        assert spec.available_at_inference is True


def test_build_team_context_features_windows_plays_per_game() -> None:
    twc = pl.DataFrame(
        {
            "team": ["KC", "KC"],
            "season": [2025, 2025],
            "week": [1, 2],
            "plays": [60.0, 64.0],
            "neutral_pace_sec": [28.0, 27.0],
            "proe": [0.0, 0.0],
            "epa_per_play_off": [0.1, 0.1],
            "success_rate_off": [0.48, 0.48],
            "implied_total": [24.5, 24.5],
            "spread": [-3.0, -3.0],
        }
    )
    schedule = pl.DataFrame(
        schema={"season": pl.Int64, "week": pl.Int64, "home_team": pl.Utf8, "away_team": pl.Utf8}
    )
    snap_counts = pl.DataFrame(
        [_snap_row(pfr_player_id=pid, position=pos) for pid, pos in [("c1", "C")]]
    ).clear()
    injuries = pl.DataFrame([_injury_row()]).clear()
    usage_features = pl.DataFrame([_usage_features_row()]).clear()

    result = team_context.build_team_context_features(
        twc, schedule, snap_counts, injuries, usage_features, registry={}
    )

    week1 = result.filter(pl.col("week") == 1).row(0, named=True)
    assert week1["plays_per_game_ewm_5"] == pytest.approx(60.0)  # first week: equals its own value


def test_build_team_context_features_registers_opponent_neutral_pace() -> None:
    team_week_context = pl.DataFrame(
        {
            "team": ["KC", "BAL"],
            "season": [2025, 2025],
            "week": [1, 1],
            "plays": [65, 60],
            "pass_rate": [0.6, 0.55],
            "neutral_pace_sec": [28.0, 31.5],
            "proe": [0.02, -0.01],
            "epa_per_play_off": [0.1, 0.05],
            "success_rate_off": [0.45, 0.42],
            "implied_total": [24.5, 20.0],
            "spread": [-3.0, 3.0],
        }
    )
    schedule = pl.DataFrame(
        {"season": [2025], "week": [1], "home_team": ["KC"], "away_team": ["BAL"]}
    )
    snap_counts = pl.DataFrame(
        schema={
            "season": pl.Int64,
            "week": pl.Int64,
            "team": pl.Utf8,
            "pfr_player_id": pl.Utf8,
            "position": pl.Utf8,
            "offense_snaps": pl.Int64,
        }
    )
    injuries = pl.DataFrame(
        schema={
            "player_id": pl.Utf8,
            "season": pl.Int64,
            "week": pl.Int64,
            "report_status": pl.Utf8,
        }
    )
    usage_features = pl.DataFrame(
        schema={
            "player_id": pl.Utf8,
            "season": pl.Int64,
            "week": pl.Int64,
            "team": pl.Utf8,
            "target_share_ewm_3": pl.Float64,
            "carry_share_ewm_3": pl.Float64,
        }
    )
    registry: dict[str, FeatureSpec] = {}

    result = team_context.build_team_context_features(
        team_week_context, schedule, snap_counts, injuries, usage_features, registry=registry
    )

    assert "opponent_neutral_pace_ewm_8" in result.columns
    assert "opponent_neutral_pace_ewm_8" in registry
    spec = registry["opponent_neutral_pace_ewm_8"]
    assert spec.source_table == team_context.SOURCE_TABLE
    assert spec.lag_weeks == 1
    assert spec.available_at_inference is True
