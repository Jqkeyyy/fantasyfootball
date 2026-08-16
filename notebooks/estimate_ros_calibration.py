# notebooks/estimate_ros_calibration.py
"""Materializes `config/ros_calibration.yml` -- this project's real,
committed calibration constants for the rest-of-season pipeline
(`SPEC-ADDENDUM-04.md` §D, TASKS.md 1.21): each position's real
within-player week-to-week point correlation (`sim.persistence
.estimate_within_player_correlation`) and real injury-duration recovery
rate (`sim.injury.estimate_recovery_prob`). Not scratch -- a real,
permanent, re-runnable script, same status as
`materialize_b3_historical.py`. Re-run whenever more real seasons of
history accumulate (each offseason); overwrites the committed file
cleanly, matching this project's idempotent-materialization convention.

No live network calls -- `player_week_features.parquet` and the raw
nflverse `rosters` table are both already cached locally (the latter via
`ingest.nflverse.fetch_rosters(..., offline=True)`); this script itself
only reads already-interim/features/raw tables.

**Real bug found against real data, worked around here (not in
`sim.persistence`, out of this task's scope):**
`persistence.estimate_within_player_correlation`'s own docstring and its
own test fixture (`tests/test_sim_persistence.py`) both assume its input
carries a `season_type` column. The real, committed
`features/player_week_features.parquet` does NOT carry that column --
confirmed live (its real schema has no `season_type` at all, and its
real `week` values run through 22, matching `tools.sos`'s own module
docstring: "both `player_week_features.parquet` and
`defense_position_allowed.parquet` carry real rows for [postseason]
weeks too"). `season_type` is a real attribute of `(season, week)` alone
(confirmed live: every real `(season, week)` in `interim/schedule.parquet`
maps to exactly one `season_type`), so this script joins it in from the
real `interim/schedule.parquet` table before calling the estimator --
supplying the documented input contract, not changing the estimator's
own logic.
"""

from __future__ import annotations

import polars as pl
import yaml

from ffapp.config import ROS_CALIBRATION_PATH, load_settings
from ffapp.ingest import nflverse
from ffapp.sim import injury, persistence


def main() -> None:
    settings = load_settings()
    data_root = settings.data_root

    print("Loading real player_week_features.parquet for within-player correlation...")
    features_path = data_root / "features" / "player_week_features.parquet"
    features = pl.read_parquet(features_path)

    print("Joining real season_type from interim/schedule.parquet (see module docstring's")
    print("real-bug note -- player_week_features.parquet itself carries no season_type column)...")
    schedule = pl.read_parquet(data_root / "interim" / "schedule.parquet")
    season_type_by_week = schedule.select("season", "week", "season_type").unique(
        subset=["season", "week"], keep="first"
    )
    features = features.join(season_type_by_week, on=["season", "week"], how="left")

    rho_by_position = persistence.estimate_within_player_correlation(features)
    print("Real within-player week-to-week correlation (ICC), by position:")
    for position, rho in sorted(rho_by_position.items()):
        ratio = persistence.season_variance_ratio(n_weeks=10, rho=rho)
        print(
            f"  {position}: rho={rho:.4f}  "
            f"(10-week season total variance ratio vs independent: {ratio:.3f}x)"
        )

    print("\nBuilding real hazard grid for injury-duration recovery estimation...")
    rosters_path = nflverse.fetch_rosters(
        list(range(settings.seasons.train_start, settings.seasons.current)),
        offline=True,
        settings=settings,
    )
    rosters = pl.read_parquet(rosters_path)
    hazard_grid = injury.build_hazard_grid(rosters)
    recovery_by_position = injury.estimate_recovery_prob(hazard_grid)
    print("Real injury-duration recovery_prob, by position (1 / mean real run length):")
    for position, recovery_prob in sorted(recovery_by_position.items()):
        mean_duration = 1.0 / recovery_prob
        print(
            f"  {position}: recovery_prob={recovery_prob:.4f}  "
            f"(mean real duration {mean_duration:.2f} weeks)"
        )

    ROS_CALIBRATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    ROS_CALIBRATION_PATH.write_text(
        yaml.safe_dump(
            {
                "within_player_week_correlation": {
                    k: round(v, 4) for k, v in rho_by_position.items()
                },
                "recovery_prob": {k: round(v, 4) for k, v in recovery_by_position.items()},
            },
            sort_keys=True,
        )
    )
    print(f"\nWrote real calibration to {ROS_CALIBRATION_PATH}")


if __name__ == "__main__":
    main()
