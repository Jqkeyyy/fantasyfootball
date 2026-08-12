import polars as pl
import pytest

from ffapp.interim import build

# --- build_team_week_context -----------------------------------------------------


def _pbp_row(**kwargs: object) -> dict:
    row: dict[str, object] = {
        "season": 2025,
        "week": 1,
        "posteam": "KC",
        "defteam": "BAL",
        "play_type": "pass",
        "epa": 0.0,
        "success": 0,
        "yardline_100": 50,
        "receiver_player_id": None,
        "rusher_player_id": None,
        "qb_scramble": 0,
    }
    row.update(kwargs)
    return row


def _pbp(rows: list[dict]) -> pl.DataFrame:
    """Real nflverse pbp always types receiver_player_id/rusher_player_id as
    Utf8 -- a small fixture where one of them is None in every row would
    otherwise get inferred as Null dtype, breaking the join against a real
    Utf8 column downstream. Force the real dtype explicitly."""
    return pl.DataFrame(
        rows, schema_overrides={"receiver_player_id": pl.Utf8, "rusher_player_id": pl.Utf8}
    )


def test_build_team_week_context_computes_plays_pass_rate_epa_success() -> None:
    pbp = _pbp(
        [
            _pbp_row(play_type="pass", epa=1.0, success=1),
            _pbp_row(play_type="pass", epa=-1.0, success=0),
            _pbp_row(play_type="run", epa=0.5, success=1),
            _pbp_row(play_type="punt", epa=0.0, success=0),  # not scrimmage -- excluded
        ]
    )

    result = build.build_team_week_context(pbp)

    kc = result.filter(pl.col("team") == "KC").row(0, named=True)
    assert kc["plays"] == 3  # punt excluded
    assert kc["pass_rate"] == pytest.approx(2 / 3)
    assert kc["epa_per_play_off"] == pytest.approx((1.0 - 1.0 + 0.5) / 3)
    assert kc["success_rate_off"] == pytest.approx(2 / 3)


def test_build_team_week_context_leaves_deferred_columns_null() -> None:
    pbp = _pbp([_pbp_row()])

    result = build.build_team_week_context(pbp)

    row = result.row(0, named=True)
    assert row["proe"] is None
    assert row["neutral_pace_sec"] is None
    assert row["implied_total"] is None
    assert row["spread"] is None


# --- add_schedule_context (task 1.3) -----------------------------------------------


def test_add_schedule_context_fills_home_and_away_from_the_teams_own_perspective() -> None:
    twc = pl.DataFrame(
        {
            "team": ["KC", "BAL"],
            "season": [2025, 2025],
            "week": [1, 1],
            "plays": [60, 55],
            "spread": [None, None],
            "implied_total": [None, None],
        },
        schema_overrides={"spread": pl.Float64, "implied_total": pl.Float64},
    )
    schedule = pl.DataFrame(
        {
            "season": [2025],
            "week": [1],
            "home_team": ["KC"],
            "away_team": ["BAL"],
            "spread_line": [-2.5],  # home (KC) is a 2.5-point underdog
            "home_implied_total": [23.0],
            "away_implied_total": [25.5],
        }
    )

    result = build.add_schedule_context(twc, schedule)

    rows = {row["team"]: row for row in result.iter_rows(named=True)}
    assert rows["KC"]["spread"] == pytest.approx(-2.5)  # home team's own spread, as-is
    assert rows["KC"]["implied_total"] == pytest.approx(23.0)
    assert rows["BAL"]["spread"] == pytest.approx(2.5)  # away team's own spread, mirrored
    assert rows["BAL"]["implied_total"] == pytest.approx(25.5)


def test_add_schedule_context_matches_a_relocated_franchises_pre_move_season() -> None:
    """schedule keeps the Rams as "STL" in 2015 (period-accurate); real
    team_week_context (built from pbp) only ever has "LA" rows, even for
    2015 -- confirmed live against the full 2015-2025 range. Without the
    alias, this join silently leaves spread/implied_total null for every
    relocated franchise's pre-move seasons (129 real rows, not a data
    gap)."""
    twc = pl.DataFrame(
        {
            "team": ["LA", "SEA"],
            "season": [2015, 2015],
            "week": [1, 1],
            "plays": [60, 55],
            "spread": [None, None],
            "implied_total": [None, None],
        },
        schema_overrides={"spread": pl.Float64, "implied_total": pl.Float64},
    )
    schedule = pl.DataFrame(
        {
            "season": [2015],
            "week": [1],
            "home_team": ["STL"],
            "away_team": ["SEA"],
            "spread_line": [3.0],
            "home_implied_total": [24.0],
            "away_implied_total": [21.0],
        }
    )

    result = build.add_schedule_context(twc, schedule)

    rows = {row["team"]: row for row in result.iter_rows(named=True)}
    assert rows["LA"]["spread"] == pytest.approx(3.0)
    assert rows["LA"]["implied_total"] == pytest.approx(24.0)
    assert rows["SEA"]["spread"] == pytest.approx(-3.0)


# --- add_kickoff_utc (task 1.3) -----------------------------------------------------


def _schedule_row(**kwargs: object) -> dict:
    row: dict[str, object] = {
        "game_id": "2025_01_KC_BAL",
        "season": 2025,
        "week": 1,
        "season_type": "REG",
        "home_team": "KC",
        "away_team": "BAL",
        "gameday": "2025-09-07",
        "gametime": "13:00",
        "kickoff_utc": None,
        "spread_line": -2.5,
        "total_line": 48.5,
        "home_implied_total": 23.0,
        "away_implied_total": 25.5,
        "roof": "outdoors",
        "surface": "grass",
        "stadium_id": "KAN00",
        "home_rest": 7,
        "away_rest": 7,
    }
    row.update(kwargs)
    return row


def _schedule(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(rows, schema_overrides={"kickoff_utc": pl.Utf8})


def _stadiums() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "stadium_id": ["KAN00", "PHI00"],
            "tz": ["America/Chicago", "America/New_York"],
        }
    )


def test_add_kickoff_utc_converts_local_kickoff_to_utc_in_the_venues_own_timezone() -> None:
    schedule = _schedule(
        [_schedule_row(stadium_id="KAN00", gameday="2025-09-07", gametime="13:00")]
    )

    result = build.add_kickoff_utc(schedule, _stadiums())

    # Kansas City is America/Chicago; September 7 is inside US DST (CDT, UTC-5).
    assert result.row(0, named=True)["kickoff_utc"] == "2025-09-07T18:00:00Z"


def test_add_kickoff_utc_handles_multiple_distinct_timezones_in_one_call() -> None:
    schedule = _schedule(
        [
            _schedule_row(
                game_id="2025_01_KC_BAL", stadium_id="KAN00", gameday="2025-09-07", gametime="13:00"
            ),
            _schedule_row(
                game_id="2025_01_PHI_DAL",
                stadium_id="PHI00",
                gameday="2025-09-07",
                gametime="13:00",
            ),
        ]
    )

    result = build.add_kickoff_utc(schedule, _stadiums())

    rows = {row["game_id"]: row for row in result.iter_rows(named=True)}
    # Same local wall-clock time, different real timezones -> different UTC hour.
    assert rows["2025_01_KC_BAL"]["kickoff_utc"] == "2025-09-07T18:00:00Z"  # Chicago, CDT (UTC-5)
    assert rows["2025_01_PHI_DAL"]["kickoff_utc"] == "2025-09-07T17:00:00Z"  # NY, EDT (UTC-4)


def test_add_kickoff_utc_applies_standard_time_outside_dst() -> None:
    schedule = _schedule(
        [_schedule_row(stadium_id="PHI00", gameday="2026-01-04", gametime="13:00")]
    )

    result = build.add_kickoff_utc(schedule, _stadiums())

    # January -> EST (UTC-5), not EDT (UTC-4) -- proves this isn't a fixed offset.
    assert result.row(0, named=True)["kickoff_utc"] == "2026-01-04T18:00:00Z"


def test_add_kickoff_utc_leaves_null_when_stadium_id_has_no_match() -> None:
    schedule = _schedule([_schedule_row(stadium_id="ZZZ00")])

    result = build.add_kickoff_utc(schedule, _stadiums())

    assert result.row(0, named=True)["kickoff_utc"] is None


def test_add_kickoff_utc_preserves_schedule_columns_and_other_values() -> None:
    schedule = _schedule([_schedule_row()])

    result = build.add_kickoff_utc(schedule, _stadiums())

    assert result.columns == schedule.columns
    row = result.row(0, named=True)
    assert row["spread_line"] == -2.5
    assert row["home_implied_total"] == 23.0
    assert row["stadium_id"] == "KAN00"


# --- backfill_injury_date_modified (task 1.4) --------------------------------------


def _injury_row(**kwargs: object) -> dict:
    row: dict[str, object] = {
        "player_id": "00-0031234",
        "season": 2025,
        "week": 1,
        "team": "KC",
        "report_status": "Questionable",
        "practice_status": "Limited Participation in Practice",
        "report_primary_injury": "Ankle",
        "date_modified": None,
    }
    row.update(kwargs)
    return row


def _injuries(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(rows, schema_overrides={"date_modified": pl.Datetime(time_zone="UTC")})


def _mini_schedule(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(rows)


def test_backfill_injury_date_modified_fills_from_the_teams_own_game_two_days_prior() -> None:
    injuries = _injuries([_injury_row(team="KC", season=2025, week=1, date_modified=None)])
    schedule = _mini_schedule(
        [
            {
                "season": 2025,
                "week": 1,
                "home_team": "KC",
                "away_team": "BAL",
                "gameday": "2025-09-07",  # Sunday
            }
        ]
    )

    result = build.backfill_injury_date_modified(injuries, schedule)

    row = result.row(0, named=True)
    assert row["date_modified"].isoformat() == "2025-09-05T12:00:00+00:00"  # Friday, 2 days prior
    assert row["date_modified_is_estimated"] is True


def test_backfill_injury_date_modified_generalises_to_a_thursday_game() -> None:
    injuries = _injuries([_injury_row(team="BAL", season=2025, week=1, date_modified=None)])
    schedule = _mini_schedule(
        [
            {
                "season": 2025,
                "week": 1,
                "home_team": "KC",
                "away_team": "BAL",
                "gameday": "2025-09-11",  # Thursday
            }
        ]
    )

    result = build.backfill_injury_date_modified(injuries, schedule)

    row = result.row(0, named=True)
    assert row["date_modified"].isoformat() == "2025-09-09T12:00:00+00:00"  # Tuesday, 2 days prior
    assert row["date_modified_is_estimated"] is True


def test_backfill_injury_date_modified_leaves_a_real_timestamp_untouched() -> None:
    from datetime import UTC, datetime

    real_ts = datetime(2025, 9, 5, 18, 0, tzinfo=UTC)
    injuries = _injuries([_injury_row(team="KC", season=2025, week=1, date_modified=real_ts)])
    schedule = _mini_schedule(
        [
            {
                "season": 2025,
                "week": 1,
                "home_team": "KC",
                "away_team": "BAL",
                "gameday": "2025-09-07",
            }
        ]
    )

    result = build.backfill_injury_date_modified(injuries, schedule)

    row = result.row(0, named=True)
    assert row["date_modified"] == real_ts
    assert row["date_modified_is_estimated"] is False


def test_backfill_injury_date_modified_keeps_null_when_no_matching_game() -> None:
    injuries = _injuries([_injury_row(team="ZZZ", season=2025, week=1, date_modified=None)])
    schedule = _mini_schedule(
        [
            {
                "season": 2025,
                "week": 1,
                "home_team": "KC",
                "away_team": "BAL",
                "gameday": "2025-09-07",
            }
        ]
    )

    result = build.backfill_injury_date_modified(injuries, schedule)

    row = result.row(0, named=True)
    assert row["date_modified"] is None
    assert row["date_modified_is_estimated"] is True


# --- _player_position_by_season ---------------------------------------------------


def test_player_position_by_season_takes_the_players_own_position() -> None:
    player_stats = pl.DataFrame(
        {
            "player_id": ["1", "1", "2"],
            "season": [2025, 2025, 2025],
            "week": [1, 2, 1],
            "position": ["WR", "WR", "RB"],
        }
    )

    result = build._player_position_by_season(player_stats)

    positions = {row["player_id"]: row["position"] for row in result.iter_rows(named=True)}
    assert positions == {"1": "WR", "2": "RB"}


# --- build_defense_position_allowed -----------------------------------------------


def test_build_defense_position_allowed_counts_plays_by_position_group() -> None:
    pbp = _pbp(
        [
            _pbp_row(play_type="pass", receiver_player_id="wr1", defteam="BAL"),
            _pbp_row(play_type="pass", receiver_player_id="te1", defteam="BAL"),
            _pbp_row(play_type="run", rusher_player_id="rb1", defteam="BAL"),
            _pbp_row(play_type="run", rusher_player_id="qb1", defteam="BAL"),
            _pbp_row(play_type="pass", receiver_player_id="rb1", defteam="BAL"),
        ]
    )
    player_stats = pl.DataFrame(
        {
            "player_id": ["wr1", "te1", "rb1", "qb1"],
            "season": [2025, 2025, 2025, 2025],
            "week": [1, 1, 1, 1],
            "position": ["WR", "TE", "RB", "QB"],
        }
    )

    result = build.build_defense_position_allowed(pbp, player_stats)

    groups = {row["position_group"]: row["n_plays"] for row in result.iter_rows(named=True)}
    assert groups == {"WR": 1, "TE": 1, "RB_rushing": 1, "QB_rushing": 1, "RB_receiving": 1}


def test_build_defense_position_allowed_leaves_adjusted_columns_null() -> None:
    pbp = _pbp([_pbp_row(play_type="pass", receiver_player_id="wr1")])
    player_stats = pl.DataFrame(
        {"player_id": ["wr1"], "season": [2025], "week": [1], "position": ["WR"]}
    )

    result = build.build_defense_position_allowed(pbp, player_stats)

    row = result.row(0, named=True)
    assert row["adj_epa_allowed"] is None
    assert row["adj_success_allowed"] is None
    assert row["adj_ypt_allowed"] is None
    assert row["adj_td_rate_allowed"] is None


def test_build_defense_position_allowed_drops_plays_with_no_position_match() -> None:
    """A play whose receiver/rusher isn't in player_stats (unresolvable
    position) contributes to no group rather than crashing or fabricating
    one -- the position_group-null filter drops it cleanly."""
    pbp = _pbp([_pbp_row(play_type="pass", receiver_player_id="unknown_guy")])
    player_stats = pl.DataFrame(
        {"player_id": ["wr1"], "season": [2025], "week": [1], "position": ["WR"]}
    )

    result = build.build_defense_position_allowed(pbp, player_stats)

    assert result.height == 0


# --- _snap_counts_by_player_id -----------------------------------------------------


def test_snap_counts_by_player_id_resolves_via_pfr_crosswalk() -> None:
    snap_counts = pl.DataFrame(
        {
            "pfr_player_id": ["MahoPa00"],
            "season": [2025],
            "week": [1],
            "offense_snaps": [65],
            "offense_pct": [1.0],
        }
    )
    players_dim = pl.DataFrame({"pfr_id": ["MahoPa00"], "player_id": ["00-0033873"]})

    result = build._snap_counts_by_player_id(snap_counts, players_dim)

    row = result.row(0, named=True)
    assert row["player_id"] == "00-0033873"
    assert row["offense_snaps"] == 65
    assert row["offense_snap_pct"] == 1.0


def test_snap_counts_by_player_id_leaves_unresolvable_rows_null() -> None:
    snap_counts = pl.DataFrame(
        {
            "pfr_player_id": ["NoMatch00"],
            "season": [2025],
            "week": [1],
            "offense_snaps": [10],
            "offense_pct": [0.2],
        }
    )
    players_dim = pl.DataFrame({"pfr_id": ["SomeoneElse00"], "player_id": ["00-1111111"]})

    result = build._snap_counts_by_player_id(snap_counts, players_dim)

    assert result.height == 1
    assert result.row(0, named=True)["player_id"] is None


# --- _red_zone_touch_counts ---------------------------------------------------------


def test_red_zone_touch_counts_filters_by_yardline() -> None:
    pbp = _pbp(
        [
            _pbp_row(play_type="pass", receiver_player_id="wr1", yardline_100=15),  # RZ
            _pbp_row(play_type="pass", receiver_player_id="wr1", yardline_100=50),  # not RZ
            _pbp_row(play_type="run", rusher_player_id="rb1", yardline_100=3),  # RZ + GZ
            _pbp_row(play_type="run", rusher_player_id="rb1", yardline_100=18),  # RZ only
        ]
    )

    result = build._red_zone_touch_counts(pbp)

    rows = {row["player_id"]: row for row in result.iter_rows(named=True)}
    assert rows["wr1"]["rz_targets"] == 1
    assert rows["rb1"]["rz_carries"] == 2
    assert rows["rb1"]["gz_carries"] == 1


# --- build_player_week_usage (integration) -----------------------------------------


def test_build_player_week_usage_pulls_share_columns_straight_from_player_stats() -> None:
    player_stats = pl.DataFrame(
        {
            "player_id": ["wr1"],
            "season": [2025],
            "week": [1],
            "team": ["KC"],
            "position": ["WR"],
            "targets": [8],
            "target_share": [0.3],
            "receiving_air_yards": [80],
            "air_yards_share": [0.4],
            "wopr": [0.55],
            "carries": [0],
        }
    )
    snap_counts = pl.DataFrame(
        {"pfr_player_id": [], "season": [], "week": [], "offense_snaps": [], "offense_pct": []},
        schema={
            "pfr_player_id": pl.Utf8,
            "season": pl.Int64,
            "week": pl.Int64,
            "offense_snaps": pl.Float64,
            "offense_pct": pl.Float64,
        },
    )
    players_dim = pl.DataFrame({"pfr_id": ["x"], "player_id": ["y"]})
    pbp = _pbp([_pbp_row(play_type="pass", receiver_player_id="wr1", yardline_100=50)])

    result = build.build_player_week_usage(player_stats, snap_counts, pbp, players_dim)

    row = result.row(0, named=True)
    assert row["target_share"] == 0.3
    assert row["air_yards_share"] == 0.4
    assert row["wopr"] == 0.55
    assert row["adot"] == pytest.approx(10.0)  # 80 / 8
    assert row["route_participation"] is None
    assert row["xfp"] is None


def test_build_player_week_usage_carry_share_guards_against_zero_team_carries() -> None:
    player_stats = pl.DataFrame(
        {
            "player_id": ["qb1"],
            "season": [2025],
            "week": [1],
            "team": ["KC"],
            "position": ["QB"],
            "targets": [0],
            "target_share": [0.0],
            "receiving_air_yards": [0],
            "air_yards_share": [0.0],
            "wopr": [0.0],
            "carries": [0],
        }
    )
    snap_counts = pl.DataFrame(
        {"pfr_player_id": [], "season": [], "week": [], "offense_snaps": [], "offense_pct": []},
        schema={
            "pfr_player_id": pl.Utf8,
            "season": pl.Int64,
            "week": pl.Int64,
            "offense_snaps": pl.Float64,
            "offense_pct": pl.Float64,
        },
    )
    players_dim = pl.DataFrame({"pfr_id": ["x"], "player_id": ["y"]})
    pbp = _pbp([_pbp_row(play_type="pass")])

    result = build.build_player_week_usage(player_stats, snap_counts, pbp, players_dim)

    row = result.row(0, named=True)
    assert row["carry_share"] is None
    assert row["adot"] is None  # 0 targets -- guarded, not 0/0


def test_build_player_week_usage_excludes_non_skill_positions() -> None:
    """Real bug found via task 1.2's xfp coverage check: nflreadpy's
    player_stats carries a row for every position that recorded any stat
    that week (26 distinct codes including LB/CB/DE), not just offensive
    skill positions -- confirmed live, this alone was responsible for xfp
    coverage coming out at 30% instead of the required >=95%. A
    linebacker's stray def_sacks value must not produce a usage row."""
    player_stats = pl.DataFrame(
        {
            "player_id": ["wr1", "lb1"],
            "season": [2025, 2025],
            "week": [1, 1],
            "team": ["KC", "KC"],
            "position": ["WR", "LB"],
            "targets": [5, 0],
            "target_share": [0.2, 0.0],
            "receiving_air_yards": [40, 0],
            "air_yards_share": [0.15, 0.0],
            "wopr": [0.3, 0.0],
            "carries": [0, 0],
        }
    )
    snap_counts = pl.DataFrame(
        {"pfr_player_id": [], "season": [], "week": [], "offense_snaps": [], "offense_pct": []},
        schema={
            "pfr_player_id": pl.Utf8,
            "season": pl.Int64,
            "week": pl.Int64,
            "offense_snaps": pl.Float64,
            "offense_pct": pl.Float64,
        },
    )
    players_dim = pl.DataFrame({"pfr_id": ["x"], "player_id": ["y"]})
    pbp = _pbp([_pbp_row(play_type="pass", receiver_player_id="wr1")])

    result = build.build_player_week_usage(player_stats, snap_counts, pbp, players_dim)

    assert result["player_id"].to_list() == ["wr1"]


def test_build_player_week_usage_keeps_team_and_computes_gz_carry_share() -> None:
    player_stats = pl.DataFrame(
        {
            "player_id": ["rb1", "rb2"],
            "season": [2025, 2025],
            "week": [1, 1],
            "team": ["KC", "KC"],
            "position": ["RB", "RB"],
            "targets": [0, 0],
            "target_share": [0.0, 0.0],
            "receiving_air_yards": [0, 0],
            "air_yards_share": [0.0, 0.0],
            "wopr": [0.0, 0.0],
            "carries": [10, 5],
        }
    )
    snap_counts = pl.DataFrame(
        {"pfr_player_id": [], "season": [], "week": [], "offense_snaps": [], "offense_pct": []},
        schema={
            "pfr_player_id": pl.Utf8,
            "season": pl.Int64,
            "week": pl.Int64,
            "offense_snaps": pl.Float64,
            "offense_pct": pl.Float64,
        },
    )
    players_dim = pl.DataFrame({"pfr_id": ["x"], "player_id": ["y"]})
    pbp = _pbp(
        [
            _pbp_row(play_type="run", rusher_player_id="rb1", yardline_100=3),  # GZ
            _pbp_row(play_type="run", rusher_player_id="rb1", yardline_100=3),  # GZ
            _pbp_row(play_type="run", rusher_player_id="rb2", yardline_100=3),  # GZ
        ]
    )

    result = build.build_player_week_usage(player_stats, snap_counts, pbp, players_dim)

    rows = {row["player_id"]: row for row in result.iter_rows(named=True)}
    assert rows["rb1"]["team"] == "KC"
    assert rows["rb1"]["gz_carries"] == 2
    assert rows["rb1"]["gz_carry_share"] == pytest.approx(
        2 / 3
    )  # 2 of the team's 3 real GZ carries
    assert rows["rb2"]["gz_carry_share"] == pytest.approx(1 / 3)


def test_build_player_week_usage_designed_rush_share_excludes_scrambles() -> None:
    """SPEC §10.2's designed_rush_share is specifically a called run, not
    a QB scramble -- qb_scramble=1 plays must not count toward it, even
    though they're still real carries counted in carry_share."""
    player_stats = pl.DataFrame(
        {
            "player_id": ["qb1"],
            "season": [2025],
            "week": [1],
            "team": ["KC"],
            "position": ["QB"],
            "targets": [0],
            "target_share": [0.0],
            "receiving_air_yards": [0],
            "air_yards_share": [0.0],
            "wopr": [0.0],
            "carries": [4],
        }
    )
    snap_counts = pl.DataFrame(
        {"pfr_player_id": [], "season": [], "week": [], "offense_snaps": [], "offense_pct": []},
        schema={
            "pfr_player_id": pl.Utf8,
            "season": pl.Int64,
            "week": pl.Int64,
            "offense_snaps": pl.Float64,
            "offense_pct": pl.Float64,
        },
    )
    players_dim = pl.DataFrame({"pfr_id": ["x"], "player_id": ["y"]})
    pbp = _pbp(
        [
            _pbp_row(play_type="run", rusher_player_id="qb1", qb_scramble=0),
            _pbp_row(play_type="run", rusher_player_id="qb1", qb_scramble=0),
            _pbp_row(play_type="run", rusher_player_id="qb1", qb_scramble=0),
            _pbp_row(play_type="run", rusher_player_id="qb1", qb_scramble=1),  # scramble, excluded
        ]
    )

    result = build.build_player_week_usage(player_stats, snap_counts, pbp, players_dim)

    row = result.row(0, named=True)
    assert row["carries"] == 4  # all 4 real carries, scramble included
    assert row["designed_rush_attempts"] == 3  # scramble excluded
    assert row["designed_rush_share"] == pytest.approx(0.75)  # 3 / 4 team carries


# --- add_xfp (task 1.2) -------------------------------------------------------------


def _usage_row(**kwargs: object) -> dict:
    row: dict[str, object] = {
        "player_id": "p1",
        "season": 2025,
        "week": 1,
        "offense_snaps": 50,
        "offense_snap_pct": 0.8,
        "targets": 5,
        "target_share": 0.2,
        "air_yards": 40,
        "air_yards_share": 0.15,
        "wopr": 0.35,
        "adot": 8.0,
        "carries": 0,
        "carry_share": None,
        "rz_targets": 1,
        "rz_carries": 0,
        "rz_touch_share": 0.1,
        "gz_carries": 0,
        "route_participation": None,
        "xfp": None,
    }
    row.update(kwargs)
    return row


def test_add_xfp_joins_by_player_season_week() -> None:
    usage = pl.DataFrame([_usage_row(player_id="p1", season=2025, week=1)])
    ff_opportunity = pl.DataFrame(
        {
            "player_id": ["p1"],
            "season": ["2025"],  # real nflreadpy dtype: String
            "week": [1.0],  # real nflreadpy dtype: Float64
            "total_fantasy_points_exp": [14.2],
        }
    )

    result = build.add_xfp(usage, ff_opportunity)

    assert result.row(0, named=True)["xfp"] == pytest.approx(14.2)


def test_add_xfp_leaves_unmatched_player_weeks_null() -> None:
    usage = pl.DataFrame(
        [
            _usage_row(player_id="p1", season=2025, week=1),
            _usage_row(player_id="p2", season=2025, week=1),
        ]
    )
    ff_opportunity = pl.DataFrame(
        {"player_id": ["p1"], "season": ["2025"], "week": [1.0], "total_fantasy_points_exp": [14.2]}
    )

    result = build.add_xfp(usage, ff_opportunity)

    rows = {row["player_id"]: row["xfp"] for row in result.iter_rows(named=True)}
    assert rows["p1"] == pytest.approx(14.2)
    assert rows["p2"] is None
