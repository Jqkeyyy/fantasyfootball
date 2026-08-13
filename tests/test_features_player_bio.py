"""Task 1.14's player-age feature (SPEC §11.2)."""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from ffapp.features import player_bio
from ffapp.features.registry import FeatureSpec


def _roster_row(**kwargs: object) -> dict:
    row: dict[str, object] = {"gsis_id": "p1", "birth_date": date(1995, 1, 1)}
    row.update(kwargs)
    return row


# --- add_player_age -------------------------------------------------------------------


def test_add_player_age_computes_fractional_years_as_of_the_rows_own_kickoff() -> None:
    grid = pl.DataFrame(
        {"player_id": ["p1"], "season": [2025], "week": [1], "as_of_utc": ["2025-09-07T17:00:00Z"]}
    )
    rosters = pl.DataFrame([_roster_row(birth_date=date(1995, 9, 7))])

    result = player_bio.add_player_age(grid, rosters)

    assert result["age"].to_list() == pytest.approx([30.0], abs=0.01)


def test_add_player_age_uses_each_rows_own_as_of_utc_not_a_single_snapshot() -> None:
    """Two rows for the same player, different weeks -- age must differ,
    proving it's computed per-row, not from one fixed reference date."""
    grid = pl.DataFrame(
        {
            "player_id": ["p1", "p1"],
            "season": [2025, 2025],
            "week": [1, 10],
            "as_of_utc": ["2025-09-07T17:00:00Z", "2025-11-09T17:00:00Z"],
        }
    )
    rosters = pl.DataFrame([_roster_row(birth_date=date(1995, 1, 1))])

    result = player_bio.add_player_age(grid, rosters).sort("week")

    ages = result["age"].to_list()
    assert ages[1] > ages[0]


def test_add_player_age_is_honestly_null_when_the_player_has_no_real_birth_date() -> None:
    grid = pl.DataFrame(
        {"player_id": ["p1"], "season": [2025], "week": [1], "as_of_utc": ["2025-09-07T17:00:00Z"]}
    )
    rosters = pl.DataFrame(
        [
            _roster_row(gsis_id="p1", birth_date=None),
            _roster_row(gsis_id="p2", birth_date=date(1990, 1, 1)),
        ]
    )

    result = player_bio.add_player_age(grid, rosters)

    assert result["age"].to_list() == [None]


def test_add_player_age_dedupes_rosters_to_one_row_per_player_before_joining() -> None:
    """`rosters` is a real weekly table -- the same player appears many
    times with the identical real birth_date. Must not fan the grid out
    into duplicate rows."""
    grid = pl.DataFrame(
        {"player_id": ["p1"], "season": [2025], "week": [1], "as_of_utc": ["2025-09-07T17:00:00Z"]}
    )
    rosters = pl.DataFrame(
        [_roster_row(birth_date=date(1995, 1, 1)) for _ in range(5)]  # 5 real weekly rows
    )

    result = player_bio.add_player_age(grid, rosters)

    assert result.height == 1


# --- build_player_bio_features ----------------------------------------------------------


def test_build_player_bio_features_registers_a_training_safe_spec() -> None:
    grid = pl.DataFrame(
        {"player_id": ["p1"], "season": [2025], "week": [1], "as_of_utc": ["2025-09-07T17:00:00Z"]}
    )
    rosters = pl.DataFrame([_roster_row()])
    registry: dict[str, FeatureSpec] = {}

    player_bio.build_player_bio_features(grid, rosters, registry=registry)

    spec = registry["age"]
    assert spec.lag_weeks >= 1
    assert spec.available_at_inference is True
    assert spec.source_table == player_bio.SOURCE_TABLE
