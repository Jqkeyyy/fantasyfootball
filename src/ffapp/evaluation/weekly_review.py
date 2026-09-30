"""Auditable weekly scores from saved pregame projections and observed outcomes."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from typing import Any

import polars as pl


def _time(value: Any) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed
    except (ValueError, TypeError):
        return None


def _number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(value):
        return float(value)
    return None


def _key(row: dict[str, Any]) -> tuple[int, int, str]:
    return int(row["season"]), int(row["week"]), str(row["player_id"])


def review_key(row: dict[str, Any]) -> str:
    return ":".join(map(str, _key(row)))


def build_weekly_review(
    history: pl.DataFrame,
    schedule: pl.DataFrame,
    stats: pl.DataFrame,
    usage: pl.DataFrame,
    injuries: pl.DataFrame,
    overrides: dict[str, Any] | None = None,
    *,
    now: datetime | None = None,
) -> pl.DataFrame:
    """Choose latest saved pre-kickoff snapshot; never reconstruct old projections.

    Injury designations alone don't prove an in-game limitation. Out/IR games and
    user-confirmed injury games are excluded. Other low-snap games are review flags,
    retained in the score unless the user confirms an injury. Outcome diagnostics
    compare with the prior four appearances, not an invented component forecast.
    """
    if history.is_empty():
        return pl.DataFrame()
    now = now or datetime.now(UTC)
    overrides = overrides or {}
    games: dict[tuple[int, int, str], datetime] = {}
    week_end: dict[tuple[int, int], datetime] = {}
    for row in schedule.to_dicts():
        kickoff = _time(row.get("kickoff_utc"))
        if kickoff is None:
            continue
        period = (int(row["season"]), int(row["week"]))
        week_end[period] = max(week_end.get(period, kickoff), kickoff)
        for side in ("home_team", "away_team"):
            games[(*period, str(row[side]))] = kickoff

    stat_map = {_key(r): r for r in stats.to_dicts()}
    player_names = {
        str(r["player_id"]): r.get("player_display_name") or r.get("player_name")
        for r in stats.to_dicts()
        if r.get("player_display_name") or r.get("player_name")
    }
    usage_map = {_key(r): r for r in usage.to_dicts()}
    injury_map: dict[tuple[int, int, str], dict[str, Any]] = {}
    for row in sorted(injuries.to_dicts(), key=lambda r: str(r.get("date_modified") or "")):
        injury_map[_key(row)] = row
    appearances: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for row in stat_map.values():
        appearances.setdefault((int(row["season"]), str(row["player_id"])), []).append(row)
    for rows in appearances.values():
        rows.sort(key=lambda r: int(r["week"]))

    candidates: dict[tuple[int, int, str], list[dict[str, Any]]] = {}
    for row in history.to_dicts():
        candidates.setdefault(_key(row), []).append(row)
    result = []
    for key, snapshots in candidates.items():
        season, week, player_id = key
        eligible = []
        for snapshot in snapshots:
            saved = _time(snapshot.get("as_of_utc"))
            kickoff = games.get((season, week, str(snapshot.get("team"))))
            if saved and kickoff and saved < kickoff:
                eligible.append(snapshot)
        row = max(
            eligible or snapshots,
            key=lambda r: _time(r.get("as_of_utc")) or datetime.min.replace(tzinfo=UTC),
        )
        source = row.get("projection_source") or "unknown"
        prediction = _number(row.get("live_mean"))
        prefix = "live"
        if prediction is None:
            legacy = {
                "consensus_b3": "b3_mean",
                "baseline_b2": "b2_mean",
                "direct": "model_mean",
            }.get(source)
            prediction = _number(row.get(legacy)) if legacy else None
            prefix = "b3" if source == "consensus_b3" else "unavailable"
        actual = _number(row.get("actual_points"))
        if actual is None:
            # Immutable kickoff files get outcomes from the normal weekly backfill.
            actual = next(
                (
                    _number(r.get("actual_points"))
                    for r in reversed(snapshots)
                    if _number(r.get("actual_points")) is not None
                ),
                None,
            )
        stat = stat_map.get(key, {})
        current_usage = usage_map.get(key, {})
        injury = injury_map.get(key, {})
        designation = str(row.get("gameday_injury_status") or injury.get("report_status") or "")
        manual = overrides.get(review_key(row), {})
        injury_reason = ""
        if manual.get("exclude"):
            injury_reason = "Confirmed injury: " + str(manual.get("note") or "user reviewed")
        elif designation.lower() in {"out", "ir", "injured reserve"}:
            injury_reason = f"Injury report: {designation}"

        prior = [r for r in appearances.get((season, player_id), []) if r["week"] < week][-4:]
        clues = []
        signals = []
        for column, label in (
            ("targets", "Targets"),
            ("carries", "Carries"),
            ("attempts", "Pass attempts"),
        ):
            values = [v for r in prior if (v := _number(r.get(column))) is not None]
            current = _number(stat.get(column))
            if len(values) >= 2 and current is not None:
                avg = sum(values) / len(values)
                if avg >= 3 and abs(current - avg) >= max(3, avg * 0.3):
                    clues.append(f"{label} {current:.0f} vs prior {len(values)}-game avg {avg:.1f}")
                    signals.append("Usage changed")
        for yards, opportunities, label in (
            ("receiving_yards", "targets", "Yards per target"),
            ("rushing_yards", "carries", "Yards per carry"),
            ("passing_yards", "attempts", "Yards per pass attempt"),
        ):
            current_yards, current_opps = _number(stat.get(yards)), _number(stat.get(opportunities))
            complete = [
                r
                for r in prior
                if _number(r.get(yards)) is not None and _number(r.get(opportunities)) is not None
            ]
            prior_opps = sum(float(r[opportunities]) for r in complete)
            if (
                len(complete) >= 2
                and prior_opps >= 10
                and current_opps is not None
                and current_opps >= 5
                and current_yards is not None
            ):
                old_rate = sum(float(r[yards]) for r in complete) / prior_opps
                rate = current_yards / current_opps
                if old_rate > 0 and abs(rate - old_rate) >= 0.35 * old_rate:
                    clues.append(f"{label} {rate:.1f} vs recent {old_rate:.1f}")
                    signals.append("Efficiency changed")
        snap = _number(current_usage.get("offense_snap_pct"))
        prior_snaps = [
            v
            for r in prior
            if (v := _number(usage_map.get(_key(r), {}).get("offense_snap_pct"))) is not None
        ]
        low_snaps = bool(
            snap is not None
            and len(prior_snaps) >= 2
            and sum(prior_snaps) / len(prior_snaps) >= 0.3
            and snap < 0.5 * sum(prior_snaps) / len(prior_snaps)
        )
        if low_snaps:
            clues.append(
                "Snaps below half of recent share; injury, benching or role change needs review"
            )
        if designation:
            clues.append(f"Injury designation: {designation}; does not establish an in-game injury")
        touchdowns = [_number(stat.get(c)) for c in ("passing_tds", "rushing_tds", "receiving_tds")]
        if all(v is not None for v in touchdowns):
            td = sum(v for v in touchdowns if v is not None)
            clues.append(f"{td:.0f} total touchdowns")
            if td >= 2:
                signals.append("Multiple touchdowns")
        if not clues:
            clues.append("Insufficient game-level evidence to explain this miss")

        end = week_end.get((season, week))
        status = "Scored"
        if not eligible:
            status = "No verified pregame snapshot"
        elif end is None or now < end + timedelta(hours=8):
            status = "Week not ready"
        elif prediction is None:
            status = "Missing saved live projection"
        elif actual is None:
            status = "Awaiting actuals"
        elif injury_reason:
            status = "Injury excluded"
        lower, upper = _number(row.get(f"{prefix}_q10")), _number(row.get(f"{prefix}_q90"))
        error = prediction - actual if prediction is not None and actual is not None else None
        result.append(
            {
                "season": season,
                "week": week,
                "player_id": player_id,
                "player": player_names.get(player_id) or player_id,
                "position": row.get("position"),
                "source": source,
                "projected": prediction,
                "actual": actual,
                "error": error,
                "absolute_error": abs(error) if error is not None else None,
                "status": status,
                "injury_reason": injury_reason,
                "injury_data_available": key in injury_map,
                "needs_review": low_snaps,
                "signal": ", ".join(dict.fromkeys(signals)) or "No clear signal",
                "evidence": "; ".join(clues),
                "snapshot": row.get("as_of_utc"),
                "baseline": _number(row.get("b2_mean")),
                "in_range": lower <= actual <= upper
                if lower is not None and upper is not None and actual is not None
                else None,
            }
        )
    frame = pl.DataFrame(result, infer_schema_length=None).with_columns(
        pl.col("projected", "actual", "error", "absolute_error", "baseline").cast(pl.Float64),
        pl.col("in_range").cast(pl.Boolean),
    )
    # An all-zero week is a known placeholder failure in the existing backfill path.
    valid = frame.group_by("season", "week").agg(pl.col("actual").abs().sum().alias("magnitude"))
    return (
        frame.join(valid, on=["season", "week"])
        .with_columns(
            pl.when(
                (pl.col("magnitude") == 0) & pl.col("status").is_in(["Scored", "Injury excluded"])
            )
            .then(pl.lit("Awaiting real actuals"))
            .otherwise(pl.col("status"))
            .alias("status")
        )
        .drop("magnitude")
    )


def score_summary(rows: pl.DataFrame, groups: list[str]) -> pl.DataFrame:
    """Same-sample score and baseline; positive bias means projections were too high."""
    scored = rows.filter(pl.col("status") == "Scored")
    if scored.is_empty():
        return pl.DataFrame()
    return (
        scored.group_by(groups)
        .agg(
            pl.len().alias("players"),
            pl.col("absolute_error").mean().alias("mae"),
            pl.col("error").mean().alias("bias"),
            (pl.col("error").pow(2).mean().sqrt()).alias("rmse"),
            (pl.col("absolute_error") <= 5).mean().alias("within_five"),
            pl.col("in_range").cast(pl.Float64).mean().alias("coverage"),
            pl.col("in_range").count().alias("interval_players"),
            (pl.col("baseline") - pl.col("actual")).abs().mean().alias("baseline_mae"),
            pl.col("absolute_error")
            .filter(pl.col("baseline").is_not_null())
            .mean()
            .alias("paired_model_mae"),
            pl.col("baseline").count().alias("baseline_players"),
        )
        .sort(groups)
    )
