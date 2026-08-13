"""SPEC §12.1's own literal test: `evaluation/snapshot.py`'s as_of contract,
exercised over a sample of backtest weeks (task 1.11's acceptance bar).

Uses small committed fixtures, not `data/` (gitignored, not reproducible
on a fresh clone -- this project's established fixture-vs-live-run
convention: pure logic gets committed fixtures here; the real end-to-end
run against real cached data is documented in HANDOFF.md instead, same
pattern as every other task this session).
"""

from datetime import datetime

import polars as pl
import pytest

from ffapp.evaluation import snapshot

SEASON = 2025
BACKTEST_WEEKS = [1, 2, 3, 4]


def _schedule() -> pl.DataFrame:
    """Four real-shaped weeks of one season, one game each, kickoffs a
    week apart -- a small but genuine "sample of backtest weeks."""
    return pl.DataFrame(
        {
            "season": [SEASON] * len(BACKTEST_WEEKS),
            "week": BACKTEST_WEEKS,
            "home_team": ["KC", "BAL", "DAL", "PHI"],
            "away_team": ["BAL", "KC", "PHI", "DAL"],
            "kickoff_utc": [
                "2025-09-07T17:00:00Z",
                "2025-09-14T17:00:00Z",
                "2025-09-21T17:00:00Z",
                "2025-09-28T17:00:00Z",
            ],
        }
    )


def _player_week_stats() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "player_id": ["p1"] * len(BACKTEST_WEEKS),
            "season": [SEASON] * len(BACKTEST_WEEKS),
            "week": BACKTEST_WEEKS,
            "target": [12.0, 18.0, 6.0, 21.0],
        }
    )


def _injuries() -> pl.DataFrame:
    """Each week's real injury report, published the Friday before its
    own kickoff -- a real (season, week, date_modified) relationship."""
    return pl.DataFrame(
        {
            "player_id": ["p1"] * len(BACKTEST_WEEKS),
            "season": [SEASON] * len(BACKTEST_WEEKS),
            "week": BACKTEST_WEEKS,
            "report_status": ["Questionable", "None", "Out", "None"],
            "date_modified": [
                "2025-09-05T12:00:00Z",
                "2025-09-12T12:00:00Z",
                "2025-09-19T12:00:00Z",
                "2025-09-26T12:00:00Z",
            ],
        },
        schema_overrides={"date_modified": pl.Utf8},
    ).with_columns(pl.col("date_modified").str.to_datetime(time_zone="UTC"))


def _kickoff(week: int) -> datetime:
    row = _schedule().filter(pl.col("week") == week).row(0, named=True)
    return datetime.fromisoformat(row["kickoff_utc"].replace("Z", "+00:00"))


# --- passes over a sample of backtest weeks ------------------------------------------


def test_snapshot_respects_as_of_across_a_sample_of_backtest_weeks() -> None:
    """The walk-forward shape from SPEC §12.2's own pseudocode: for each
    backtest week, snapshot as_of that week's own kickoff and confirm no
    leakage -- run across *every* week in the sample, not just one."""
    tables = {
        "schedule": _schedule(),
        "player_week_stats": _player_week_stats(),
        "injuries": _injuries(),
    }

    for week in BACKTEST_WEEKS:
        as_of = _kickoff(week)
        result = snapshot.snapshot(tables, as_of)
        snapshot.assert_no_leakage(result, as_of)  # should never raise

        # the walk-forward property, checked directly: only strictly
        # prior weeks' stats survive.
        assert (
            result["player_week_stats"]["week"].max() is None
            or result["player_week_stats"]["week"].max() < week
        )


def test_assert_no_leakage_passes_on_the_full_unfiltered_history_as_of_the_last_week() -> None:
    """As of the *last* backtest week's own kickoff, every *prior* week's
    real data is fully knowable -- confirms the assertion doesn't
    over-trigger on legitimate, real historical data."""
    tables = {
        "schedule": _schedule(),
        "player_week_stats": _player_week_stats().filter(pl.col("week") < BACKTEST_WEEKS[-1]),
        "injuries": _injuries().filter(pl.col("week") < BACKTEST_WEEKS[-1]),
    }

    snapshot.assert_no_leakage(tables, _kickoff(BACKTEST_WEEKS[-1]))  # should not raise


# --- fails when a deliberate leak is introduced ---------------------------------------


def test_assert_no_leakage_fails_when_a_future_weeks_stats_leak_into_an_earlier_snapshot() -> None:
    """The literal task 1.11 acceptance bar. A real regression scenario:
    someone builds `train_rows` for week 2 but accidentally includes week
    4's own stats (already played by the time the bug was introduced,
    say, in a later re-run) -- must be caught."""
    tables = {
        "schedule": _schedule(),
        "player_week_stats": _player_week_stats(),  # includes all 4 weeks, not filtered
        "injuries": _injuries().filter(pl.col("week") < 2),
    }

    with pytest.raises(snapshot.LeakageError, match="player_week_stats"):
        snapshot.assert_no_leakage(tables, _kickoff(2))


def test_assert_no_leakage_fails_when_an_injury_report_leaks_from_a_later_week() -> None:
    tables = {
        "schedule": _schedule(),
        "player_week_stats": _player_week_stats().filter(pl.col("week") < 2),
        "injuries": _injuries(),  # includes week 3's report, published after week-2 kickoff
    }

    with pytest.raises(snapshot.LeakageError, match="injuries"):
        snapshot.assert_no_leakage(tables, _kickoff(2))


def test_a_correctly_snapshotted_week_never_fails_the_leakage_assertion() -> None:
    """The inverse of the two tests above: proves the assertion isn't
    simply always-raising -- it passes once the same tables are actually
    filtered through snapshot() first."""
    tables = {
        "schedule": _schedule(),
        "player_week_stats": _player_week_stats(),
        "injuries": _injuries(),
    }
    as_of = _kickoff(2)

    filtered = snapshot.snapshot(tables, as_of)

    snapshot.assert_no_leakage(filtered, as_of)  # should not raise
