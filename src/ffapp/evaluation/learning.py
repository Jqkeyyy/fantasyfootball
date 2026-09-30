"""Repeated error patterns and strictly forward-only experiments on saved forecasts."""

from __future__ import annotations

from typing import cast

import polars as pl

from ffapp.evaluation.weekly_review import score_summary


def recurring_patterns(review: pl.DataFrame) -> pl.DataFrame:
    if review.is_empty():
        return pl.DataFrame()
    weekly = score_summary(review, ["season", "week", "source", "position"])
    if weekly.is_empty():
        return pl.DataFrame()
    evidence = []
    for group in weekly.filter(pl.col("players") >= 10).partition_by(["source", "position"]):
        recent = group.sort("season", "week").tail(6)
        if recent.height < 3:
            continue
        for label, condition, action in [
            (
                "Consistently too high",
                pl.col("bias") >= 2,
                "Review usage assumptions; test a downward bias correction in the learning lab.",
            ),
            (
                "Consistently too low",
                pl.col("bias") <= -2,
                "Check rising roles and opportunities; test an upward bias correction.",
            ),
            (
                "Ranges too narrow",
                (pl.col("coverage") < 0.7) & (pl.col("interval_players") >= 20),
                "Recalibrate interval widths on earlier weeks, then validate on later weeks.",
            ),
        ]:
            hits = recent.filter(condition)
            if hits.height >= 3:
                evidence.append(
                    {
                        "Source": group["source"][0],
                        "Position": group["position"][0],
                        "Pattern": label,
                        "Weeks": hits.height,
                        "Weeks checked": recent.height,
                        "Next investigation": action,
                    }
                )
    return pl.DataFrame(evidence)


def replay_bias_correction(review: pl.DataFrame) -> pl.DataFrame:
    """Train on at least two earlier weeks; evaluate only the next unseen week.

    Candidate correction is capped at three points and uses at most four previous
    weeks for the same source, position and season. This never changes live forecasts.
    """
    if review.is_empty():
        return pl.DataFrame()
    scored = review.filter(pl.col("status") == "Scored")
    rows = []
    for group in scored.partition_by(["season", "source", "position"]):
        weeks = sorted(group["week"].unique().to_list())
        for index, week in enumerate(weeks):
            prior_weeks = weeks[max(0, index - 4) : index]
            training = group.filter(pl.col("week").is_in(prior_weeks))
            enough = training.group_by("week").len().filter(pl.col("len") >= 10)
            if enough.height < 2:
                continue
            training = training.filter(pl.col("week").is_in(enough["week"].to_list()))
            test = group.filter(pl.col("week") == week)
            if test.height < 10:
                continue
            correction = max(-3.0, min(3.0, float(cast(float, training["error"].mean()))))
            candidate_error = test.select(
                ((pl.col("projected") - correction).clip(lower_bound=0) - pl.col("actual"))
                .abs()
                .mean()
            ).item()
            original = float(cast(float, test["absolute_error"].mean()))
            rows.append(
                {
                    "Season": group["season"][0],
                    "Week tested": week,
                    "Source": group["source"][0],
                    "Position": group["position"][0],
                    "Training weeks": enough.height,
                    "Players tested": test.height,
                    "Correction": correction,
                    "Original miss": original,
                    "Trial miss": candidate_error,
                    "Improvement": original - candidate_error,
                }
            )
    return pl.DataFrame(rows)
