"""Pure helpers for `app/pages/6_ROS_Rankings.py` (SPEC-ADDENDUM-04.md
§D.5). Matches `app.schedule_grid_page`'s own precedent -- real math and
data access live in `tools.ros_rankings`/`tools.ros_aggregate`; this
module is glue-support only, kept separately testable from Streamlit
itself.
"""

from __future__ import annotations

import polars as pl

from ffapp.tools import tiers
from ffapp.tools.ros_rankings import REQUIRED_ROS_BOARD_COLUMNS, ROS_BOARD_SCHEMA_VERSION

SORT_OPTIONS = {
    "Value over replacement": "vor_ros",
    "Projected ROS points": "ros_points",
    "Projected points per game": "ros_ppg",
    "Upside (P90)": "ros_p90",
    "Floor (P10)": "ros_p10",
    "Playoff value": "playoff_weeks_value",
}


class RosBoardSchemaError(ValueError):
    """The materialized ROS board predates the UI's required schema."""


def _number(value: object) -> float:
    if not isinstance(value, int | float):
        raise TypeError(f"Expected a numeric ROS value, got {value!r}")
    return float(value)


def validate_board_schema(board: pl.DataFrame) -> None:
    missing = REQUIRED_ROS_BOARD_COLUMNS - set(board.columns)
    if missing:
        raise RosBoardSchemaError(
            "ROS rankings are from an older artifact schema; missing "
            f"{', '.join(sorted(missing))}. Refresh this league's ROS rankings."
        )
    versions = set(board["artifact_schema_version"].drop_nulls().unique().to_list())
    if versions != {ROS_BOARD_SCHEMA_VERSION}:
        raise RosBoardSchemaError(
            f"ROS rankings schema is {sorted(versions)}, but this app requires "
            f"version {ROS_BOARD_SCHEMA_VERSION}. Refresh this league's ROS rankings."
        )


def style_rank_change(board: pl.DataFrame) -> pl.DataFrame:
    """Adds `rank_change_display`: `"+N"` for real upward movement,
    `"-N"` for real downward movement, an em dash for a genuinely new
    player or a first-ever real run (null `rank_change` -- see
    `tools.ros_rankings.rank_change`'s own docstring for why this is
    never guessed)."""
    return board.with_columns(
        pl.when(pl.col("rank_change").is_null())
        .then(pl.lit("—"))
        .when(pl.col("rank_change") > 0)
        .then(pl.lit("+") + pl.col("rank_change").cast(pl.String))
        .otherwise(pl.col("rank_change").cast(pl.String))
        .alias("rank_change_display")
    )


def filter_board(
    board: pl.DataFrame,
    *,
    position: str | None,
    availability: str | None,
    nfl_team: str | None = None,
    fantasy_team: str | None = None,
    search: str = "",
) -> pl.DataFrame:
    """Filter the materialized all-player board without changing its values."""
    result = board
    if position is not None:
        result = result.filter(pl.col("position") == position)
    if availability is not None:
        result = result.filter(pl.col("availability") == availability)
    if nfl_team is not None:
        result = result.filter(pl.col("nfl_team") == nfl_team)
    if fantasy_team is not None:
        result = result.filter(pl.col("fantasy_team") == fantasy_team)
    if search.strip():
        result = result.filter(
            pl.col("player_name")
            .str.to_lowercase()
            .str.contains(search.strip().lower(), literal=True)
        )
    return result


def prepare_board(board: pl.DataFrame, *, sort_label: str) -> pl.DataFrame:
    """Add PPG, metric-specific rank, and position tiers, then sort."""
    if sort_label not in SORT_OPTIONS:
        raise ValueError(f"Unknown ROS sort option {sort_label!r}")
    enriched = board.with_columns(
        pl.when(pl.col("expected_games") > 0)
        .then(pl.col("ros_points") / pl.col("expected_games"))
        .otherwise(None)
        .alias("ros_ppg")
    )
    if enriched.is_empty():
        return enriched.with_columns(
            pl.lit(None, dtype=pl.Int64).alias("position_tier"),
            pl.lit(None, dtype=pl.Int64).alias("view_rank"),
        )
    metric = SORT_OPTIONS[sort_label]
    tiered = tiers.assign_tiers(enriched, vor_column=metric).rename({"tier": "position_tier"})
    return tiered.sort(metric, descending=True, nulls_last=True).with_row_index(
        "view_rank", offset=1
    )


def player_week_schedule(projections_ros: pl.DataFrame, player_id: str) -> pl.DataFrame:
    """Weekly projection range and a projection-derived matchup signal."""
    player = projections_ros.filter(pl.col("player_id") == player_id).sort("week")
    if player.is_empty():
        return player
    raw_average = player["mean"].drop_nulls().mean()
    average = _number(raw_average) if raw_average is not None else 0.0
    if average <= 0:
        signal = pl.lit("Neutral")
    else:
        signal = (
            pl.when(pl.col("mean") >= average * 1.10)
            .then(pl.lit("Favorable"))
            .when(pl.col("mean") <= average * 0.90)
            .then(pl.lit("Tough"))
            .otherwise(pl.lit("Neutral"))
        )
    return player.with_columns(signal.alias("schedule_signal")).select(
        "week",
        "opponent_team",
        "mean",
        "q10",
        "q50",
        "q90",
        "schedule_signal",
        "is_current_week",
    )


def explain_ros_player(row: dict[str, object], *, remaining_weeks: int) -> list[str]:
    """Factual explanation of value, uncertainty, availability, and movement."""
    points = _number(row["ros_points"])
    vor = _number(row["vor_ros"])
    expected_games = _number(row["expected_games"])
    replacement = points - vor
    availability_rate = expected_games / remaining_weeks if remaining_weeks else 0.0
    movement = row.get("rank_change")
    if movement is None:
        movement_text = "No comparable prior all-player rank is available yet."
    elif _number(movement) > 0:
        movement_text = f"VOR rank moved up {int(_number(movement))} place(s) since the prior run."
    elif _number(movement) < 0:
        movement_text = (
            f"VOR rank moved down {abs(int(_number(movement)))} place(s) since the prior run."
        )
    else:
        movement_text = "VOR rank is unchanged since the prior run."
    return [
        f"Projects for {points:.1f} points over {expected_games:.1f} expected games "
        f"({availability_rate:.0%} of the remaining schedule).",
        f"Uncertainty range: {_number(row['ros_p10']):.1f} (P10) to "
        f"{_number(row['ros_p90']):.1f} (P90).",
        f"VOR is {vor:.1f}; the current {row['position']} free-agent replacement "
        f"baseline is {replacement:.1f} ROS points.",
        movement_text,
    ]


__all__ = [
    "RosBoardSchemaError",
    "SORT_OPTIONS",
    "explain_ros_player",
    "filter_board",
    "player_week_schedule",
    "prepare_board",
    "style_rank_change",
    "validate_board_schema",
]
