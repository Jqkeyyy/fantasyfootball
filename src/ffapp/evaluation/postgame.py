"""Explain completed-week model misses and recorded fantasy decisions."""

from __future__ import annotations

import polars as pl


def model_miss_autopsy(review: pl.DataFrame, *, limit: int = 15) -> pl.DataFrame:
    """Rank non-injury misses and attach cautious evidence-based explanations."""
    if review.is_empty():
        return pl.DataFrame()
    scored = review.filter(pl.col("status") == "Scored").sort("absolute_error", descending=True)
    rows = []
    for row in scored.head(limit).iter_rows(named=True):
        signal = str(row.get("signal") or "No clear signal")
        if "Usage changed" in signal:
            category = "Role / opportunity changed"
        elif "Efficiency changed" in signal:
            category = "Efficiency moved"
        elif "Multiple touchdowns" in signal:
            category = "Touchdown-heavy result"
        else:
            category = "Unexplained game variance"
        rows.append(
            {
                "player": row.get("player"),
                "position": row.get("position"),
                "projected": row.get("projected"),
                "actual": row.get("actual"),
                "miss": row.get("absolute_error"),
                "direction": "Too high" if float(row.get("error") or 0) > 0 else "Too low",
                "category": category,
                "evidence": row.get("evidence"),
            }
        )
    return pl.DataFrame(rows) if rows else pl.DataFrame()


def decision_autopsy(
    ledger: pl.DataFrame, review: pl.DataFrame, *, season: int, week: int
) -> pl.DataFrame:
    """Explain whether the user's recorded choice helped, excluding injury outcomes."""
    if ledger.is_empty():
        return pl.DataFrame()
    decisions = ledger.filter((pl.col("season") == season) & (pl.col("week") == week))
    if decisions.is_empty():
        return pl.DataFrame()
    evidence = {
        str(row["player_id"]): row
        for row in review.filter((pl.col("season") == season) & (pl.col("week") == week)).iter_rows(
            named=True
        )
    }
    rows = []
    for row in decisions.iter_rows(named=True):
        subject = evidence.get(str(row["subject_player_id"]), {})
        alternative = evidence.get(str(row.get("alternative_player_id")), {})
        injury = any(
            item.get("status") == "Injury excluded" for item in (subject, alternative) if item
        )
        realized = row.get("realized_delta")
        status = str(row.get("status") or "recommended")
        chosen_delta = None
        if isinstance(realized, int | float) and status in {"accepted", "rejected"}:
            chosen_delta = float(realized) if status == "accepted" else -float(realized)
        if injury:
            outcome = "Injury excluded"
        elif chosen_delta is None:
            outcome = "Awaiting choice or result"
        elif chosen_delta > 0:
            outcome = "Choice helped"
        elif chosen_delta < 0:
            outcome = "Choice cost points"
        else:
            outcome = "No difference"
        clues = []
        for label, item in (("Recommended player", subject), ("Alternative", alternative)):
            if item:
                clues.append(
                    f"{label}: {item.get('signal') or 'No clear signal'}"
                    + (f" ({item.get('evidence')})" if item.get("evidence") else "")
                )
        rows.append(
            {
                "decision": row.get("recommended_action"),
                "choice": "Followed"
                if status == "accepted"
                else "Declined"
                if status == "rejected"
                else "Not recorded",
                "projected_edge": row.get("expected_delta"),
                "realized_edge": realized,
                "your_choice_value": chosen_delta,
                "outcome": outcome,
                "counted": not injury and chosen_delta is not None,
                "evidence": " · ".join(clues) or "No game-level evidence is available.",
            }
        )
    return pl.DataFrame(rows)


__all__ = ["decision_autopsy", "model_miss_autopsy"]
