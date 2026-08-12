import polars as pl
import pytest

from ffapp.features import situation

# --- add_schedule_situation --------------------------------------------------------


def _schedule_row(**kwargs: object) -> dict:
    row: dict[str, object] = {
        "game_id": "2025_01_KC_BAL",
        "season": 2025,
        "week": 1,
        "home_team": "KC",
        "away_team": "BAL",
        "weekday": "Sunday",
        "gametime": "13:00",
        "home_rest": 7,
        "away_rest": 7,
    }
    row.update(kwargs)
    return row


def _grid_row(**kwargs: object) -> dict:
    row: dict[str, object] = {
        "player_id": "p1",
        "season": 2025,
        "week": 1,
        "team": "KC",
    }
    row.update(kwargs)
    return row


def test_add_schedule_situation_sets_is_home_and_rest_days_from_the_teams_own_side() -> None:
    schedule = pl.DataFrame([_schedule_row()])
    grid = pl.DataFrame(
        [_grid_row(player_id="p1", team="KC"), _grid_row(player_id="p2", team="BAL")]
    )

    result = situation.add_schedule_situation(grid, schedule)

    rows = {row["player_id"]: row for row in result.iter_rows(named=True)}
    assert rows["p1"]["is_home"] is True
    assert rows["p2"]["is_home"] is False
    assert rows["p1"]["rest_days"] == 7
    assert rows["p1"]["week_number"] == 1


def test_add_schedule_situation_leaves_a_bye_week_null_but_keeps_week_number() -> None:
    schedule = pl.DataFrame([_schedule_row(home_team="KC", away_team="BAL")])
    grid = pl.DataFrame([_grid_row(player_id="p1", team="DAL")])  # DAL has no game this week

    result = situation.add_schedule_situation(grid, schedule)

    row = result.row(0, named=True)
    assert row["is_home"] is None
    assert row["rest_days"] is None
    assert row["is_primetime"] is None
    assert row["week_number"] == 1  # still populated -- plain passthrough of `week`


@pytest.mark.parametrize(
    ("weekday", "gametime", "expected"),
    [
        ("Thursday", "20:15", True),
        ("Monday", "20:15", True),
        ("Sunday", "13:00", False),  # early Sunday -- not primetime
        ("Sunday", "20:20", True),  # Sunday Night Football
        ("Saturday", "20:00", False),
    ],
)
def test_add_schedule_situation_is_primetime(weekday: str, gametime: str, expected: bool) -> None:
    schedule = pl.DataFrame([_schedule_row(weekday=weekday, gametime=gametime)])
    grid = pl.DataFrame([_grid_row(team="KC")])

    result = situation.add_schedule_situation(grid, schedule)

    assert result.row(0, named=True)["is_primetime"] == expected


# --- add_weather --------------------------------------------------------------------


def test_add_weather_joins_via_the_teams_own_game_id() -> None:
    schedule = pl.DataFrame([_schedule_row()])
    weather = pl.DataFrame(
        {
            "game_id": ["2025_01_KC_BAL"],
            "wind_mph": [12.0],
            "precip_prob": [20.0],
            "temp_f": [55.0],
            "is_dome": [False],
        }
    )
    grid = pl.DataFrame([_grid_row(team="KC")])

    result = situation.add_weather(grid, schedule, weather)

    row = result.row(0, named=True)
    assert row["wind_mph"] == 12.0
    assert row["precip_prob"] == 20.0
    assert row["temp_f"] == 55.0
    assert row["is_dome"] is False


def test_add_weather_is_null_on_a_bye_week() -> None:
    schedule = pl.DataFrame([_schedule_row(home_team="KC", away_team="BAL")])
    weather = pl.DataFrame(
        {
            "game_id": ["2025_01_KC_BAL"],
            "wind_mph": [12.0],
            "precip_prob": [20.0],
            "temp_f": [55.0],
            "is_dome": [False],
        }
    )
    grid = pl.DataFrame([_grid_row(team="DAL")])

    result = situation.add_weather(grid, schedule, weather)

    row = result.row(0, named=True)
    assert row["wind_mph"] is None
    assert row["is_dome"] is None


# --- _latest_injury_report / add_injury_report --------------------------------------


def _injury_row(**kwargs: object) -> dict:
    row: dict[str, object] = {
        "player_id": "p1",
        "season": 2025,
        "week": 1,
        "team": "KC",
        "report_status": "Questionable",
        "practice_status": "Limited",  # real interim/injuries.parquet's own column name
        "date_modified": "2025-09-05T12:00:00Z",
    }
    row.update(kwargs)
    return row


def _injuries(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(rows, schema_overrides={"date_modified": pl.Utf8}).with_columns(
        pl.col("date_modified").str.to_datetime(time_zone="UTC")
    )


def test_latest_injury_report_keeps_the_most_recently_modified_row_on_a_real_duplicate() -> None:
    """Mirrors a real case found in production data: a mid-week trade
    produces two rows for the same (player, season, week) with different
    teams -- the later date_modified should win."""
    injuries = _injuries(
        [
            _injury_row(team="GB", report_status="Doubtful", date_modified="2025-09-05T10:00:00Z"),
            _injury_row(
                team="NE", report_status="Questionable", date_modified="2025-09-06T09:00:00Z"
            ),
        ]
    )

    result = situation._latest_injury_report(injuries)

    assert result.height == 1
    row = result.row(0, named=True)
    assert row["team"] == "NE"
    assert row["report_status"] == "Questionable"


def test_add_injury_report_fills_report_status_to_none_when_no_report_exists() -> None:
    injuries = _injuries([_injury_row(player_id="p2")])  # different player
    grid = pl.DataFrame([_grid_row(player_id="p1")])

    result = situation.add_injury_report(grid, injuries)

    row = result.row(0, named=True)
    assert row["report_status"] == "None"
    assert row["practice_participation"] is None


def test_add_injury_report_carries_a_real_designation_through() -> None:
    injuries = _injuries([_injury_row(player_id="p1", report_status="Out")])
    grid = pl.DataFrame([_grid_row(player_id="p1")])

    result = situation.add_injury_report(grid, injuries)

    assert result.row(0, named=True)["report_status"] == "Out"


# --- add_weeks_since_return ----------------------------------------------------------


def test_weeks_since_return_counts_real_games_since_the_last_out_week() -> None:
    grid = pl.DataFrame(
        [_grid_row(week=w) for w in [1, 2, 3, 4, 5]],
    )
    injuries = _injuries([_injury_row(week=3, report_status="Out")])

    result = situation.add_weeks_since_return(grid, injuries)

    rows = {row["week"]: row["weeks_since_return"] for row in result.iter_rows(named=True)}
    assert rows[1] is None  # no Out week has happened yet
    assert rows[2] is None
    assert rows[3] is None  # the Out week itself -- no *prior* episode
    assert rows[4] == pytest.approx(1.0)
    assert rows[5] == pytest.approx(2.0)


def test_weeks_since_return_is_null_when_never_ruled_out() -> None:
    grid = pl.DataFrame([_grid_row(week=w) for w in [1, 2, 3]])
    injuries = _injuries([_injury_row(week=1, report_status="Questionable")])

    result = situation.add_weeks_since_return(grid, injuries)

    assert result["weeks_since_return"].is_null().all()


def test_weeks_since_return_spans_a_season_boundary_correctly() -> None:
    """The naive season*100+week arithmetic would produce ~83 here; the
    real answer is 1 elapsed game (week 17 of 2024 -> week 1 of 2025)."""
    grid = pl.DataFrame(
        [_grid_row(season=2024, week=17), _grid_row(season=2025, week=1)],
    )
    injuries = _injuries([_injury_row(season=2024, week=17, report_status="Out")])

    result = situation.add_weeks_since_return(grid, injuries)

    row = result.filter((pl.col("season") == 2025) & (pl.col("week") == 1)).row(0, named=True)
    assert row["weeks_since_return"] == pytest.approx(1.0)


def test_weeks_since_return_ignores_a_second_ruled_out_episode_that_hasnt_happened_yet() -> None:
    """A player Out in week 3 and again in week 7, evaluated at week 5:
    only the week-3 episode is in the past -- 2 games since return, not
    confused by the *future* week-7 episode."""
    grid = pl.DataFrame([_grid_row(week=w) for w in [1, 2, 3, 4, 5, 6, 7, 8]])
    injuries = _injuries(
        [
            _injury_row(week=3, report_status="Out"),
            _injury_row(week=7, report_status="Out"),
        ]
    )

    result = situation.add_weeks_since_return(grid, injuries)

    week5 = result.filter(pl.col("week") == 5).row(0, named=True)
    week8 = result.filter(pl.col("week") == 8).row(0, named=True)
    assert week5["weeks_since_return"] == pytest.approx(2.0)
    assert week8["weeks_since_return"] == pytest.approx(1.0)  # since week 7's episode


# --- build_situation_features (integration) -------------------------------------------


def test_build_situation_features_registers_every_feature() -> None:
    schedule = pl.DataFrame([_schedule_row()])
    weather = pl.DataFrame(
        {
            "game_id": ["2025_01_KC_BAL"],
            "wind_mph": [5.0],
            "precip_prob": [0.0],
            "temp_f": [70.0],
            "is_dome": [False],
        }
    )
    injuries = _injuries([_injury_row()]).clear()
    grid = pl.DataFrame([_grid_row()])

    registry: dict[str, object] = {}
    situation.build_situation_features(grid, schedule, injuries, weather, registry=registry)

    expected = {
        "is_home",
        "rest_days",
        "is_primetime",
        "week_number",
        "wind_mph",
        "precip_prob",
        "temp_f",
        "is_dome",
        "report_status",
        "practice_participation",
        "weeks_since_return",
    }
    assert expected <= registry.keys()
    for name in expected:
        spec = registry[name]
        assert spec.lag_weeks == 1
        assert spec.available_at_inference is True
