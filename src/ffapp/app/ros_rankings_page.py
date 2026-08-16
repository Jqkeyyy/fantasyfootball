"""Pure helpers for `app/pages/6_ROS_Rankings.py` (SPEC-ADDENDUM-04.md
§D.5). Matches `app.schedule_grid_page`'s own precedent -- real math and
data access live in `tools.ros_rankings`/`tools.ros_aggregate`; this
module is glue-support only, kept separately testable from Streamlit
itself.
"""

from __future__ import annotations

import polars as pl


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
    board: pl.DataFrame, *, position: str | None, available_ids: set[str] | None
) -> pl.DataFrame:
    """Position filter, plus an optional real-id restriction
    (`available_ids`) a future caller can reuse for a different scoping
    need -- the ROS Rankings page itself never passes anything but
    `None` for `available_ids`, since `rankings_ros/latest.parquet`
    (Task 11's own output) is already scoped to the current free-agent
    pool before it's ever written (`tools.ros_rankings
    .current_free_agent_projections`, Task 10) -- there is no rostered
    player on this board left to filter out."""
    result = board
    if position is not None:
        result = result.filter(pl.col("position") == position)
    if available_ids is not None:
        result = result.filter(pl.col("player_id").is_in(list(available_ids)))
    return result


__all__ = ["filter_board", "style_rank_change"]
