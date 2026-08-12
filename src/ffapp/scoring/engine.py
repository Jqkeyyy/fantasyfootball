"""Scoring engine (SPEC.md §8.3). Applies a league's `scoring_settings` to a
per-player-week stat frame using scoring/keymap.py's `STAT_KEY_MAP`.
"""

from __future__ import annotations

import polars as pl

from ffapp.scoring.keymap import FG_BUCKET_KEYS, FG_YARDAGE_KEY, STAT_KEY_MAP, DirectStat


class UnhandledScoringKeysError(Exception):
    """A non-zero scoring key has no STAT_KEY_MAP entry (CLAUDE.md rule 3: the
    scoring engine must be trusted before anything downstream uses it -- a silently
    ignored scoring rule is exactly the kind of bug that costs a season)."""


class ConflictingFieldGoalSchemeError(Exception):
    """A league sets both `fgm_yds` and a bucketed FG key non-zero (ADDENDUM-01
    §C.1) -- scoring both would double-count every made field goal."""


def unhandled_keys(scoring: dict[str, float]) -> list[str]:
    """Scoring keys present in `scoring` with a non-zero value and no STAT_KEY_MAP
    entry. MUST be empty before `score_stat_line` is trusted."""
    return sorted(key for key, value in scoring.items() if value != 0 and key not in STAT_KEY_MAP)


def _check_fg_scheme_conflict(scoring: dict[str, float]) -> None:
    yardage_active = scoring.get(FG_YARDAGE_KEY, 0) != 0
    bucket_active = any(scoring.get(key, 0) != 0 for key in FG_BUCKET_KEYS)
    if yardage_active and bucket_active:
        raise ConflictingFieldGoalSchemeError(
            f"League scoring sets both '{FG_YARDAGE_KEY}' and a bucketed FG key "
            "non-zero (ADDENDUM-01 §C.1) -- this would double-count made field goals."
        )


def score_stat_line(stats: pl.DataFrame, scoring: dict[str, float]) -> pl.Series:
    """Apply league scoring to a per-player-week stat frame. Returns points."""
    missing = unhandled_keys(scoring)
    if missing:
        raise UnhandledScoringKeysError(
            f"No STAT_KEY_MAP entry for non-zero scoring key(s): {missing}"
        )
    _check_fg_scheme_conflict(scoring)

    points = pl.Series("points", [0.0] * stats.height, dtype=pl.Float64)
    for key, value in scoring.items():
        if value == 0:
            continue
        spec = STAT_KEY_MAP[key]
        if isinstance(spec, DirectStat):
            contribution = stats[spec.column].fill_null(0).cast(pl.Float64) * value
        else:
            contribution = spec.compute(stats, value).fill_null(0.0)
        points = points + contribution
    return points


__all__ = [
    "ConflictingFieldGoalSchemeError",
    "UnhandledScoringKeysError",
    "score_stat_line",
    "unhandled_keys",
]
