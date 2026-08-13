"""The as_of snapshot (SPEC.md §12.1; task 1.11).

The rest of this project's pipeline (tasks 1.6-1.9) is already
walk-forward by construction -- `proe`'s per-season refit, opponent
adjustment's per-week refit, every lag-shift join in `features/build.py`.
`snapshot()` and `assert_no_leakage()` exist as an *independent* check on
top of that, not a replacement for it: a "trust but verify" mechanism
that audits the real *interim* tables directly against a real cutoff
timestamp, so a future regression in the feature-computation logic can
still be caught even if it doesn't touch the pipeline's own walk-forward
mechanics.

Every real interim table this project has falls into exactly one of
three knowability rules:

- `schedule` is a **passthrough**, not gated at all. It's a reference
  table a walk-forward prediction legitimately needs for its own target
  week -- `features.situation`/`features.team_context` already join it
  directly, unshifted, for exactly that reason (SPEC §12.1's own "Vegas
  lines" caveat is about *precision* -- this project only has one
  closing-line snapshot per game, not a real line-movement history, so
  "be explicit that closing lines are slightly optimistic relative to
  what you would have had on Thursday, and document the bias" *is* that
  documentation, not a hard gate to enforce here). Gating `schedule` by
  its own kickoff was tried and rejected during this task's own
  development: a real test scenario surfaced that a target week's own
  schedule row would then read as "leaked" by the very mechanism meant
  to let a model use that week's own situation features, which is
  exactly backwards.
- `injuries`: gated by its own real per-row `date_modified` column --
  the actual publication timestamp of that specific designation.
- Everything else (`player_week_stats`, `player_week_usage`,
  `team_week_context`, `defense_position_allowed`, `weather`, `rosters`,
  `snap_counts`): gated by that row's own `(season, week)`'s real
  kickoff -- these are all genuinely post-game facts (stats, usage,
  team context's own actual EPA/success/pass rate, opponent-allowed
  rates, actual weather, who actually played, snap counts), knowable
  only once that week's game has actually happened.

A table not in any of these three groups raises rather than silently
passing through unfiltered, which would be a real, silent leakage risk,
not a defensible default -- CLAUDE.md rule 4's "never silently..."
principle applied to the as_of contract instead of a join.

`as_of` must be a timezone-aware UTC `datetime` -- compared directly
against `schedule.kickoff_utc` (parsed) and `injuries.date_modified`
(already a real `Datetime(time_zone="UTC")` column). The comparison is
strict (`<`, not `<=`): a row whose own timestamp is *exactly* `as_of`
is treated as not yet knowable, the conservative direction -- SPEC's own
`as_of` is already "the kickoff time of the first game of that week,
minus a configurable safety margin," so a caller applying that margin
never sees this boundary in practice; a caller who doesn't still gets
the safe behaviour by default.
"""

from __future__ import annotations

from datetime import datetime

import polars as pl

PASSTHROUGH_TABLES = frozenset({"schedule"})
DATE_MODIFIED_GATED_TABLES = frozenset({"injuries"})
KICKOFF_GATED_TABLES = frozenset(
    {
        "player_week_stats",
        "player_week_usage",
        "team_week_context",
        "defense_position_allowed",
        "weather",
        "rosters",
        "snap_counts",
    }
)

_KICKOFF_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


class LeakageError(Exception):
    """A table row's source timestamp is not strictly before `as_of` --
    the as_of contract (SPEC §12.1) is violated."""


def _kickoff_lookup(schedule: pl.DataFrame) -> pl.DataFrame:
    """(season, week) -> that week's own earliest real kickoff, as a real
    `Datetime(time_zone="UTC")` -- the boundary every kickoff-gated
    table's own rows are compared against."""
    return schedule.group_by(["season", "week"]).agg(
        pl.col("kickoff_utc")
        .str.strptime(pl.Datetime(time_zone="UTC"), _KICKOFF_FORMAT)
        .min()
        .alias("_known_at")
    )


def _with_known_at(table_name: str, df: pl.DataFrame, kickoff_lookup: pl.DataFrame) -> pl.DataFrame:
    if table_name in DATE_MODIFIED_GATED_TABLES:
        return df.with_columns(pl.col("date_modified").alias("_known_at"))
    if table_name in KICKOFF_GATED_TABLES:
        return df.join(kickoff_lookup, on=["season", "week"], how="left")
    raise LeakageError(
        f"snapshot()/assert_no_leakage() has no as_of rule for table {table_name!r} -- "
        "add it to KICKOFF_GATED_TABLES, DATE_MODIFIED_GATED_TABLES, or PASSTHROUGH_TABLES "
        "explicitly rather than guessing (or silently passing it through unfiltered, "
        "which would be a real leakage risk)."
    )


def snapshot(tables: dict[str, pl.DataFrame], as_of: datetime) -> dict[str, pl.DataFrame]:
    """SPEC §12.1: "Return every table filtered to rows knowable at
    `as_of`." `tables` must include a real `"schedule"` entry -- every
    kickoff-gated table's own knowability is resolved through it.
    `"schedule"` itself is a passthrough (see module docstring for why).
    """
    kickoff_lookup = _kickoff_lookup(tables["schedule"])
    result: dict[str, pl.DataFrame] = {}
    for name, df in tables.items():
        if name in PASSTHROUGH_TABLES:
            result[name] = df
            continue
        with_known_at = _with_known_at(name, df, kickoff_lookup)
        result[name] = with_known_at.filter(pl.col("_known_at") < as_of).drop("_known_at")
    return result


def assert_no_leakage(tables: dict[str, pl.DataFrame], as_of: datetime) -> None:
    """SPEC §12.1's own literal test assertion: "no feature row has a
    source timestamp later than its `as_of`." Reuses `snapshot()`
    directly (comparing each table's row count before/after) rather than
    duplicating the per-table knowability dispatch -- the two can never
    silently drift apart on what "knowable" means for a given table.
    """
    filtered = snapshot(tables, as_of)
    for name, df in tables.items():
        dropped = df.height - filtered[name].height
        if dropped > 0:
            raise LeakageError(
                f"table {name!r}: {dropped} real row(s) would be dropped by "
                f"snapshot(as_of={as_of.isoformat()}) -- at least one row's source "
                "timestamp is not strictly before as_of. The as_of contract (SPEC "
                "§12.1) is violated for this input."
            )


__all__ = [
    "DATE_MODIFIED_GATED_TABLES",
    "KICKOFF_GATED_TABLES",
    "PASSTHROUGH_TABLES",
    "LeakageError",
    "assert_no_leakage",
    "snapshot",
]
