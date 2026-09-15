from __future__ import annotations

import polars as pl

from ffapp.tools.projection_coverage import build_projection_coverage


def test_coverage_distinguishes_rostered_relevant_and_deep_omissions() -> None:
    projections = pl.DataFrame(
        {
            "season": [2026] * 4,
            "week": [2] * 4,
            "player_id": ["star", "rostered", "deep", "unknown"],
            "mean": [20.0, None, None, None],
        }
    )
    players = pl.DataFrame(
        {
            "player_id": ["star", "rostered", "deep"],
            "full_name": ["Star", "Rostered", "Deep"],
            "position": ["WR", "RB", "TE"],
            "sleeper_id": ["s1", "s2", "s3"],
            "search_rank": [1, 400, 800],
        }
    )

    result = build_projection_coverage(projections, players, {"s2"})
    reasons = dict(zip(result["player_id"], result["coverage_reason"], strict=True))

    assert reasons == {
        "star": "projected",
        "rostered": "rostered_missing",
        "deep": "deep_source_omission",
        "unknown": "identity_unresolved",
    }
    relevant_missing = result.filter(~pl.col("projected") & pl.col("fantasy_relevant"))
    assert relevant_missing["player_id"].to_list() == ["rostered"]
