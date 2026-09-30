from datetime import UTC, datetime, timedelta

import polars as pl

from ffapp.app.live_matchup_page import matchup_projection


def test_matchup_projection_combines_live_points_and_unplayed_projections() -> None:
    now = datetime(2026, 9, 20, 12, tzinfo=UTC)
    rankings = pl.DataFrame(
        {
            "player_id": ["mine", "theirs"],
            "player_name": ["My Player", "Their Player"],
            "position": ["WR", "RB"],
            "team": ["A", "B"],
            "proj_mean": [12.0, 8.0],
            "floor": [6.0, 4.0],
            "ceiling": [18.0, 12.0],
        }
    )
    result = matchup_projection(
        rankings,
        ["mine"],
        ["theirs"],
        {"mine": 3.0, "theirs": 5.0},
        {"A": now + timedelta(hours=2), "B": now - timedelta(hours=2)},
        now=now,
    )

    assert result["mine"]["projected_final"] == 15.0
    assert result["opponent"]["projected_final"] == 5.0
    assert result["projected_margin"] == 10.0
    assert result["win_probability"] > 0.9

