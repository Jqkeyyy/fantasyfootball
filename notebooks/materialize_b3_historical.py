# notebooks/materialize_b3_historical.py
"""Materializes `data/interim/b3_predictions.parquet` -- this project's
real historical FantasyPros weekly-consensus archive. Not scratch: a
real, load-bearing production dependency, not just an evaluation aid.

`models.predict.project_week`'s `projection_source="consensus_b3"` (the
real shipped default, `SPEC-ADDENDUM-04.md` §C) needs this file's own
real historical rows to build the empirical error-quantile spread around
this week's own live B3 value (`models.baselines.empirical_error_quantiles`,
`docs/JOURNAL.md`'s 2026-08-16 entry) -- `ffapp project` exits with a
clear, named error if this file doesn't exist yet.

Real, unauthenticated GitHub API calls: one paginated commit-list fetch
(~13 requests, cached after the first run) plus one snapshot fetch per
distinct real commit selected (roughly one per real week in
`SEASONS`) -- GitHub's unauthenticated rate limit is 60/hour, so a wide
`SEASONS` range may hit it partway through. Every fetch is
independently cached to `data/raw/rankings/`
(`ingest.rankings.fetch_fp_weekly_snapshot`'s own idempotent design), so
a rate-limited run can simply be re-launched later and picks up exactly
where it left off, at no cost to already-fetched weeks.

Re-run whenever `SEASONS` should grow (e.g. adding the just-finished
season each offseason) -- upserts by full overwrite of this one file,
matching this project's "re-running for the same scope overwrites
cleanly" idempotent-ingest convention (CLAUDE.md).
"""

from __future__ import annotations

import json

import polars as pl

from ffapp.config import load_settings
from ffapp.ids import mapping
from ffapp.ingest import nflverse, rankings, sleeper
from ffapp.interim.build import SKILL_POSITIONS
from ffapp.models import baselines

# The real FantasyPros weekly archive starts 2021-08-29
# (`ingest.rankings.select_commit_before`'s own docstring) -- no real
# season before 2021 has anything to fetch.
SEASONS = [2021, 2022, 2023, 2024, 2025]


def main() -> None:
    settings = load_settings()

    print("Fetching real FantasyPros weekly-archive commit list (cached after first run)...")
    commits_path = rankings.fetch_fp_weekly_commits(offline=False, settings=settings)
    commits = json.loads(commits_path.read_text())["commits"]
    print(f"{len(commits)} real commits in the archive's full history.")

    schedule = pl.read_parquet(settings.data_root / "interim" / "schedule.parquet")
    week_cutoffs = (
        schedule.filter(pl.col("season").is_in(SEASONS) & (pl.col("season_type") == "REG"))
        .group_by(["season", "week"])
        .agg(pl.col("kickoff_utc").min().alias("cutoff_utc"))
        .sort(["season", "week"])
    )
    print(f"{week_cutoffs.height} real (season, week) cutoffs to resolve across {SEASONS}.")

    frames: list[pl.DataFrame] = []
    n_missing = 0
    for row in week_cutoffs.iter_rows(named=True):
        sha = rankings.select_commit_before(commits, row["cutoff_utc"])
        if sha is None:
            n_missing += 1
            continue
        snapshot_path = rankings.fetch_fp_weekly_snapshot(sha, offline=False, settings=settings)
        csv_text = snapshot_path.read_text()
        frame = rankings.normalize_fp_weekly(csv_text, season=row["season"], week=row["week"])
        frames.append(frame)
        print(f"  season={row['season']} week={row['week']}: sha={sha[:8]} rows={frame.height}")

    print(f"\n{n_missing} weeks had no real commit before cutoff (real archive-start gap).")
    fp_weekly = pl.concat(frames, how="vertical_relaxed")
    fp_weekly_path = settings.data_root / "interim" / "fp_weekly_consensus.parquet"
    fp_weekly.write_parquet(fp_weekly_path)
    print(f"Wrote {fp_weekly.height} real fp_weekly rows to {fp_weekly_path}")

    print("\nResolving to real player_id via players_dim...")
    crosswalk = nflverse.fetch_player_ids(offline=True, settings=settings)
    sleeper_players = sleeper.fetch_players(offline=False, settings=settings)
    players_dim = mapping.build_players_dim(crosswalk, sleeper_players, mapping.ID_OVERRIDES_PATH)

    b3 = baselines.add_b3_fp_weekly_consensus(fp_weekly, players_dim)
    b3_path = settings.data_root / "interim" / "b3_predictions.parquet"
    b3.write_parquet(b3_path)
    print(f"Wrote {b3.height} real resolved B3 rows to {b3_path}")
    print(f"Null b3_points: {b3['b3_points'].null_count()} / {b3.height}")
    for position in SKILL_POSITIONS:
        pos_players = b3.join(
            players_dim.select("player_id", "position"), on="player_id", how="left"
        ).filter(pl.col("position") == position)
        print(f"  {position}: {pos_players.height} resolved rows")


if __name__ == "__main__":
    main()
