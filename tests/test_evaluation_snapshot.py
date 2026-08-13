from datetime import UTC, datetime

import polars as pl
import pytest

from ffapp.evaluation import snapshot

# --- fixtures ---------------------------------------------------------------------


def _schedule_row(**kwargs: object) -> dict:
    row: dict[str, object] = {
        "season": 2025,
        "week": 1,
        "home_team": "KC",
        "away_team": "BAL",
        "kickoff_utc": "2025-09-07T17:00:00Z",
    }
    row.update(kwargs)
    return row


def _stats_row(**kwargs: object) -> dict:
    row: dict[str, object] = {"player_id": "p1", "season": 2025, "week": 1, "target": 10.0}
    row.update(kwargs)
    return row


def _injury_row(**kwargs: object) -> dict:
    row: dict[str, object] = {
        "player_id": "p1",
        "season": 2025,
        "week": 1,
        "report_status": "Questionable",
        "date_modified": "2025-09-05T12:00:00Z",
    }
    row.update(kwargs)
    return row


def _injuries(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(rows, schema_overrides={"date_modified": pl.Utf8}).with_columns(
        pl.col("date_modified").str.to_datetime(time_zone="UTC")
    )


# --- snapshot -----------------------------------------------------------------------


def test_snapshot_keeps_a_kickoff_gated_row_strictly_before_as_of() -> None:
    tables = {
        "schedule": pl.DataFrame([_schedule_row(week=1, kickoff_utc="2025-09-07T17:00:00Z")]),
        "player_week_stats": pl.DataFrame([_stats_row(week=1)]),
    }
    as_of = datetime(2025, 9, 14, 17, 0, tzinfo=UTC)  # week 2's own kickoff

    result = snapshot.snapshot(tables, as_of)

    assert result["player_week_stats"].height == 1


def test_snapshot_drops_a_kickoff_gated_row_not_yet_knowable() -> None:
    tables = {
        "schedule": pl.DataFrame(
            [
                _schedule_row(week=1, kickoff_utc="2025-09-07T17:00:00Z"),
                _schedule_row(week=2, kickoff_utc="2025-09-14T17:00:00Z"),
            ]
        ),
        "player_week_stats": pl.DataFrame([_stats_row(week=1), _stats_row(week=2)]),
    }
    as_of = datetime(2025, 9, 10, 0, 0, tzinfo=UTC)  # between week 1 and week 2 kickoffs

    result = snapshot.snapshot(tables, as_of)

    assert result["player_week_stats"]["week"].to_list() == [1]


def test_snapshot_excludes_a_row_exactly_at_as_of() -> None:
    """Strict `<`, not `<=` -- the conservative direction (SPEC's own
    as_of already has a safety margin baked in by the caller)."""
    tables = {
        "schedule": pl.DataFrame([_schedule_row(week=1, kickoff_utc="2025-09-07T17:00:00Z")]),
        "player_week_stats": pl.DataFrame([_stats_row(week=1)]),
    }
    as_of = datetime(2025, 9, 7, 17, 0, tzinfo=UTC)  # exactly the game's own kickoff

    result = snapshot.snapshot(tables, as_of)

    assert result["player_week_stats"].height == 0


def test_snapshot_gates_injuries_by_date_modified_not_kickoff() -> None:
    tables = {
        "schedule": pl.DataFrame([_schedule_row(week=1, kickoff_utc="2025-09-07T17:00:00Z")]),
        "injuries": _injuries(
            [
                _injury_row(date_modified="2025-09-05T12:00:00Z"),  # Friday report, before as_of
                _injury_row(player_id="p2", date_modified="2025-09-06T23:00:00Z"),  # after as_of
            ]
        ),
    }
    as_of = datetime(2025, 9, 6, 0, 0, tzinfo=UTC)

    result = snapshot.snapshot(tables, as_of)

    assert result["injuries"]["player_id"].to_list() == ["p1"]


def test_snapshot_passes_schedule_through_unfiltered() -> None:
    """schedule is a reference table a walk-forward prediction legitimately
    needs for its own target week (situation/team-context features join
    it directly, unshifted) -- it must never be gated by its own kickoff,
    even for a genuinely future week."""
    tables = {
        "schedule": pl.DataFrame(
            [
                _schedule_row(week=1, kickoff_utc="2025-09-07T17:00:00Z"),
                _schedule_row(week=2, kickoff_utc="2025-09-14T17:00:00Z"),
            ]
        ),
    }
    as_of = datetime(2025, 9, 8, tzinfo=UTC)  # before week 2's own kickoff

    result = snapshot.snapshot(tables, as_of)

    assert result["schedule"].height == 2  # both weeks kept, including the "future" one


def test_snapshot_raises_for_an_unrecognized_table_name() -> None:
    tables = {
        "schedule": pl.DataFrame([_schedule_row()]),
        "some_new_table_nobody_configured": pl.DataFrame({"season": [2025], "week": [1]}),
    }

    with pytest.raises(snapshot.LeakageError, match="some_new_table_nobody_configured"):
        snapshot.snapshot(tables, datetime(2025, 9, 10, tzinfo=UTC))


# --- assert_no_leakage ----------------------------------------------------------------


def test_assert_no_leakage_passes_when_every_row_is_already_knowable() -> None:
    tables = {
        "schedule": pl.DataFrame([_schedule_row(week=1, kickoff_utc="2025-09-07T17:00:00Z")]),
        "player_week_stats": pl.DataFrame([_stats_row(week=1)]),
    }
    as_of = datetime(2025, 9, 14, 17, 0, tzinfo=UTC)

    snapshot.assert_no_leakage(tables, as_of)  # should not raise


def test_assert_no_leakage_fails_when_a_deliberate_leak_is_introduced() -> None:
    """The literal task 1.11 acceptance bar: this test must fail when a
    real leak is introduced. Here, a future week's stats are included in
    the "known" tables passed to the assertion -- exactly the mistake a
    regression in a future feature-computation change could make."""
    tables = {
        "schedule": pl.DataFrame(
            [
                _schedule_row(week=1, kickoff_utc="2025-09-07T17:00:00Z"),
                _schedule_row(week=2, kickoff_utc="2025-09-14T17:00:00Z"),
            ]
        ),
        "player_week_stats": pl.DataFrame([_stats_row(week=1), _stats_row(week=2)]),
    }
    as_of = datetime(2025, 9, 10, 0, 0, tzinfo=UTC)  # before week 2's own kickoff

    with pytest.raises(snapshot.LeakageError, match="player_week_stats"):
        snapshot.assert_no_leakage(tables, as_of)


def test_assert_no_leakage_fails_on_a_leaked_injury_report() -> None:
    tables = {
        "schedule": pl.DataFrame([_schedule_row(week=1, kickoff_utc="2025-09-07T17:00:00Z")]),
        "injuries": _injuries([_injury_row(date_modified="2025-09-06T23:00:00Z")]),
    }
    as_of = datetime(2025, 9, 6, 0, 0, tzinfo=UTC)  # before the report was actually published

    with pytest.raises(snapshot.LeakageError, match="injuries"):
        snapshot.assert_no_leakage(tables, as_of)
