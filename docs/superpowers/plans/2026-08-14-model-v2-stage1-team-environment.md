# Decomposed Model v2 — Stage 1 (Team Environment) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and evaluate Stage 1 of SPEC.md §11.4's decomposed model v2 — two small LightGBM regressors predicting a team's own `team_plays` and `pass_rate` for a given week from Vegas lines, pace, and PROE — and prove (against real 2021-2025 data) whether it beats a trailing-average baseline.

**Architecture:** Reuses existing, already-`as_of`-safe features (`implied_team_total`, `spread`, `proe_ewm_5`, `neutral_pace_ewm_8`) plus one new feature (`opponent_neutral_pace_ewm_8`). Reshapes team-week rows to fit the existing player-week-shaped walk-forward harness unmodified (the same trick `models/dst.py::build_dst_table` already uses), so `evaluation/backtest.py` itself is never touched. `pass_attempts`/`rush_attempts` are derived from the two predicted quantities, never modeled directly, so they always sum consistently.

**Tech Stack:** Python 3.11+, polars, LightGBM (`lightgbm.LGBMRegressor`), pytest, `uv run`.

**Spec:** `docs/design-model-v2-stage1-team-environment.md`

## Global Constraints

- No random train/test splits anywhere — everything is walk-forward in time (CLAUDE.md rule 1; SPEC §12.2).
- Every feature must have `lag_weeks >= 1` relative to its target week (SPEC §10.1; enforced by `features/build.py`'s own leakage assertion).
- Nothing hardcodes league format, team count, or scoring — this stage doesn't touch league-specific config at all, so this mostly doesn't apply here, but don't introduce any.
- Type hints on every public function; `ruff check`/`ruff format --check`/`mypy src/` must stay clean.
- Tests before implementation (TDD) — write the failing test, watch it fail for the right reason, then implement.
- No live network calls in tests — use small hand-built polars fixtures, matching every existing test file's own convention in this repo.
- `evaluation/backtest.py` must not be modified — Stage 1 reuses it exactly as-is (this is itself part of the approved design).

---

### Task 1: `opponent_neutral_pace_ewm_8` feature

**Files:**
- Modify: `src/ffapp/features/team_context.py`
- Modify: `src/ffapp/features/build.py:255-257` (the one call site of `build_team_context_features`)
- Test: `tests/test_features_team_context.py`

**Interfaces:**
- Consumes: `features.opponent.team_opponent(schedule: pl.DataFrame) -> pl.DataFrame` (already exists, returns one row per `(team, season, week)` with that team's real `opponent` for a scheduled game).
- Produces: `team_context.add_opponent_pace(team_context_with_windows: pl.DataFrame, schedule: pl.DataFrame) -> pl.DataFrame` — adds an `opponent_neutral_pace_ewm_8` column. `build_team_context_features` now takes an additional required `schedule: pl.DataFrame` parameter (after `team_week_context`, before `snap_counts`) and registers `opponent_neutral_pace_ewm_8` the same way it registers every other windowed feature (`source_table="team_week_context"`, `window="ewm_8"`, `lag_weeks=1`, `available_at_inference=True`).

- [ ] **Step 1: Write the failing test for `add_opponent_pace`**

```python
# tests/test_features_team_context.py — add near the top, after the existing "generic windowing primitives" section

def test_add_opponent_pace_looks_up_the_real_opponents_own_pace() -> None:
    team_context_with_windows = pl.DataFrame(
        {
            "team": ["KC", "BAL"],
            "season": [2025, 2025],
            "week": [1, 1],
            "neutral_pace_ewm_8": [28.0, 31.5],
        }
    )
    schedule = pl.DataFrame(
        {
            "season": [2025],
            "week": [1],
            "home_team": ["KC"],
            "away_team": ["BAL"],
        }
    )

    result = team_context.add_opponent_pace(team_context_with_windows, schedule)

    kc = result.filter(pl.col("team") == "KC").row(0, named=True)
    bal = result.filter(pl.col("team") == "BAL").row(0, named=True)
    assert kc["opponent_neutral_pace_ewm_8"] == pytest.approx(31.5)  # KC's opponent is BAL
    assert bal["opponent_neutral_pace_ewm_8"] == pytest.approx(28.0)  # BAL's opponent is KC


def test_add_opponent_pace_is_null_for_a_bye_week() -> None:
    team_context_with_windows = pl.DataFrame(
        {"team": ["KC"], "season": [2025], "week": [1], "neutral_pace_ewm_8": [28.0]}
    )
    schedule = pl.DataFrame(
        {"season": [], "week": [], "home_team": [], "away_team": []},
        schema={"season": pl.Int64, "week": pl.Int64, "home_team": pl.Utf8, "away_team": pl.Utf8},
    )  # no games scheduled at all -- KC's week 1 is unresolvable, same as a real bye

    result = team_context.add_opponent_pace(team_context_with_windows, schedule)

    assert result["opponent_neutral_pace_ewm_8"].to_list() == [None]
```

Add `from ffapp.features import team_context` import if not already present (it already is, at the top of this test file).

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_features_team_context.py -k add_opponent_pace -v`
Expected: FAIL with `AttributeError: module 'ffapp.features.team_context' has no attribute 'add_opponent_pace'`

- [ ] **Step 3: Implement `add_opponent_pace`**

In `src/ffapp/features/team_context.py`, add this import at the top alongside the existing ones:

```python
from ffapp.features.opponent import team_opponent
```

Add this function after `ol_continuity_raw` (before the `_WindowedFeature` dataclass):

```python
def add_opponent_pace(team_context_with_windows: pl.DataFrame, schedule: pl.DataFrame) -> pl.DataFrame:
    """`opponent_neutral_pace_ewm_8`: the opponent's own `neutral_pace_ewm_8`
    for this same (season, week), looked up via `team_opponent` and joined
    onto this team's row. Must run after `neutral_pace_ewm_8` itself is
    computed (self-referential lookup on the same table). A bye week (no
    scheduled opponent) is honestly null, same convention as every other
    unresolvable trailing value in this project."""
    opponent_pace = team_context_with_windows.select(
        pl.col("team").alias("opponent"),
        "season",
        "week",
        pl.col("neutral_pace_ewm_8").alias("opponent_neutral_pace_ewm_8"),
    )
    with_opponent = team_context_with_windows.join(
        team_opponent(schedule), on=["team", "season", "week"], how="left"
    )
    return with_opponent.join(
        opponent_pace, on=["opponent", "season", "week"], how="left"
    ).drop("opponent")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_features_team_context.py -k add_opponent_pace -v`
Expected: PASS

- [ ] **Step 5: Fix the two existing `build_team_context_features` tests for the new `schedule` parameter**

`tests/test_features_team_context.py` already has two tests calling
`build_team_context_features` with the *old* 4-positional-argument
signature — `test_build_team_context_features_registers_every_feature`
(around line 244) and `test_build_team_context_features_windows_plays_per_game`
(around line 289). Both will break once Step 7 adds the `schedule`
parameter. Fix both now, before that happens:

In `test_build_team_context_features_registers_every_feature`, add a
`schedule` fixture (an empty-but-correctly-typed frame is fine — this test
doesn't exercise opponent resolution) right after the existing `twc`
fixture, and insert it as the second positional argument:

```python
    schedule = pl.DataFrame(
        schema={"season": pl.Int64, "week": pl.Int64, "home_team": pl.Utf8, "away_team": pl.Utf8}
    )
    snap_counts = pl.DataFrame(
        [_snap_row(pfr_player_id=pid, position=pos) for pid, pos in [("c1", "C")]]
    )
    injuries = pl.DataFrame([_injury_row()]).clear()  # no rows, correct empty schema
    usage_features = pl.DataFrame([_usage_features_row()]).clear()

    registry: dict[str, object] = {}
    team_context.build_team_context_features(
        twc, schedule, snap_counts, injuries, usage_features, registry=registry
    )
```

Do the same in `test_build_team_context_features_windows_plays_per_game` —
add the identical `schedule` fixture and insert it as the second
positional argument:

```python
    schedule = pl.DataFrame(
        schema={"season": pl.Int64, "week": pl.Int64, "home_team": pl.Utf8, "away_team": pl.Utf8}
    )
    snap_counts = pl.DataFrame(
        [_snap_row(pfr_player_id=pid, position=pos) for pid, pos in [("c1", "C")]]
    ).clear()
    injuries = pl.DataFrame([_injury_row()]).clear()
    usage_features = pl.DataFrame([_usage_features_row()]).clear()

    result = team_context.build_team_context_features(
        twc, schedule, snap_counts, injuries, usage_features, registry={}
    )
```

Neither existing assertion needs to change — `registers_every_feature`
checks `expected <= registry.keys()` (a subset check, so the new
`opponent_neutral_pace_ewm_8` key being present too doesn't break it), and
`windows_plays_per_game` only inspects `plays_per_game_ewm_5`.

- [ ] **Step 6: Write the new failing test for `opponent_neutral_pace_ewm_8`'s own registration**

```python
# tests/test_features_team_context.py — add after the two tests just fixed above

from ffapp.features.registry import FeatureSpec


def test_build_team_context_features_registers_opponent_neutral_pace() -> None:
    team_week_context = pl.DataFrame(
        {
            "team": ["KC", "BAL"],
            "season": [2025, 2025],
            "week": [1, 1],
            "plays": [65, 60],
            "pass_rate": [0.6, 0.55],
            "neutral_pace_sec": [28.0, 31.5],
            "proe": [0.02, -0.01],
            "epa_per_play_off": [0.1, 0.05],
            "success_rate_off": [0.45, 0.42],
            "implied_total": [24.5, 20.0],
            "spread": [-3.0, 3.0],
        }
    )
    schedule = pl.DataFrame(
        {"season": [2025], "week": [1], "home_team": ["KC"], "away_team": ["BAL"]}
    )
    snap_counts = pl.DataFrame(
        schema={
            "season": pl.Int64,
            "week": pl.Int64,
            "team": pl.Utf8,
            "pfr_player_id": pl.Utf8,
            "position": pl.Utf8,
            "offense_snaps": pl.Int64,
        }
    )
    injuries = pl.DataFrame(
        schema={
            "player_id": pl.Utf8,
            "season": pl.Int64,
            "week": pl.Int64,
            "report_status": pl.Utf8,
        }
    )
    usage_features = pl.DataFrame(
        schema={
            "player_id": pl.Utf8,
            "season": pl.Int64,
            "week": pl.Int64,
            "team": pl.Utf8,
            "target_share_ewm_3": pl.Float64,
            "carry_share_ewm_3": pl.Float64,
        }
    )
    registry: dict[str, FeatureSpec] = {}

    result = team_context.build_team_context_features(
        team_week_context, schedule, snap_counts, injuries, usage_features, registry=registry
    )

    assert "opponent_neutral_pace_ewm_8" in result.columns
    assert "opponent_neutral_pace_ewm_8" in registry
    spec = registry["opponent_neutral_pace_ewm_8"]
    assert spec.source_table == team_context.SOURCE_TABLE
    assert spec.lag_weeks == 1
    assert spec.available_at_inference is True
```

- [ ] **Step 7: Run test to verify it fails**

Run: `uv run pytest tests/test_features_team_context.py -k registers_opponent_neutral_pace -v`
Expected: FAIL with a `TypeError` about `build_team_context_features`'s argument count/order (the `schedule` parameter doesn't exist yet)

- [ ] **Step 8: Add the `schedule` parameter and wire registration**

In `src/ffapp/features/team_context.py`, change `build_team_context_features`'s signature and body:

```python
def build_team_context_features(
    team_week_context: pl.DataFrame,
    schedule: pl.DataFrame,
    snap_counts: pl.DataFrame,
    injuries: pl.DataFrame,
    usage_features: pl.DataFrame,
    *,
    registry: dict[str, FeatureSpec] | None = None,
) -> pl.DataFrame:
    """Assemble every SPEC §10.2 "Team context" feature and register each
    one's `FeatureSpec` (task 1.5's registry). Every registered feature
    declares `lag_weeks=1` and `available_at_inference=True` -- none of
    these have an in-season availability gap the way route participation
    does (SPEC §10.5).
    """
    result = team_week_context.join(
        ol_continuity_raw(snap_counts), on=["team", "season", "week"], how="left"
    )

    for feature in _WINDOWED_FEATURES:
        for window in feature.windows:
            out_col = f"{feature.name_base}_{window}"
            result = ewm(result, feature.raw_column, int(window.removeprefix("ewm_")), out_col)
            register(
                FeatureSpec(
                    name=out_col,
                    description=feature.description,
                    positions=[],
                    window=window,
                    source_table=SOURCE_TABLE,
                    available_at_inference=True,
                    lag_weeks=1,
                ),
                registry=registry,
            )

    result = add_opponent_pace(result, schedule)
    register(
        FeatureSpec(
            name="opponent_neutral_pace_ewm_8",
            description="the opponent's own trailing neutral-pace, sec/play",
            positions=[],
            window="ewm_8",
            source_table=SOURCE_TABLE,
            available_at_inference=True,
            lag_weeks=1,
        ),
        registry=registry,
    )

    # implied_team_total/spread: "current week", no smoothing -- straight
    # passthroughs of team_week_context's own already-current-week values.
    result = result.rename({"implied_total": "implied_team_total"})
    for name, description in [
        ("implied_team_total", "from Vegas total and spread"),
        ("spread", "signed, team perspective"),
    ]:
        register(
            FeatureSpec(
                name=name,
                description=description,
                positions=[],
                window=None,
                source_table=SOURCE_TABLE,
                available_at_inference=True,
                lag_weeks=1,
            ),
            registry=registry,
        )

    result = add_vacated_shares(result, injuries, usage_features)
    for name, description in [
        ("teammate_vacated_target_share", "sum of target_share of teammates ruled Out"),
        ("teammate_vacated_carry_share", "as above for carries"),
    ]:
        register(
            FeatureSpec(
                name=name,
                description=description,
                positions=[],
                window=None,
                source_table=SOURCE_TABLE,
                available_at_inference=True,
                lag_weeks=1,
            ),
            registry=registry,
        )

    return result
```

Also add `"add_opponent_pace"` to this module's `__all__` list.

- [ ] **Step 9: Run all `build_team_context_features`-related tests to verify they pass**

Run: `uv run pytest tests/test_features_team_context.py -v`
Expected: all PASS, including the two fixed tests from Step 5 and the new one from Step 6

- [ ] **Step 10: Update the one call site in `features/build.py`**

In `src/ffapp/features/build.py`, change the existing call (around line 255-257):

```python
    team_context_features = team_context.build_team_context_features(
        team_week_context, schedule, snap_counts, injuries, usage_features, registry=effective_registry
    )
```

(`schedule` is already in scope at this point in the function — it's used a few lines earlier for `situation.build_situation_features`/`opponent.build_opponent_features`.)

- [ ] **Step 11: Run the full existing test suite for both touched files to confirm nothing regressed**

Run: `uv run pytest tests/test_features_team_context.py tests/test_features_build.py -v`
Expected: all PASS

- [ ] **Step 12: Lint and typecheck**

Run: `uv run ruff check src/ffapp/features/team_context.py src/ffapp/features/build.py tests/test_features_team_context.py && uv run ruff format --check src/ffapp/features/team_context.py src/ffapp/features/build.py && uv run mypy src/ffapp/features/team_context.py src/ffapp/features/build.py`
Expected: no errors

- [ ] **Step 13: Commit**

```bash
git add src/ffapp/features/team_context.py src/ffapp/features/build.py tests/test_features_team_context.py
git commit -m "feat: add opponent_neutral_pace_ewm_8 feature for model v2 Stage 1"
```

---

### Task 2: `build_team_environment_table` — DST-style row reshaping

**Files:**
- Create: `src/ffapp/models/team_environment.py`
- Modify: `src/ffapp/features/build.py` (promote `_lag_shift_join` to public `lag_shift_join`)
- Test: `tests/test_models_team_environment.py`

**Interfaces:**
- Consumes: `features.build.lag_shift_join(grid, feature_table, group_key, feature_columns, *, lag_weeks=1) -> pl.DataFrame` (promoted from private in this task); `features.team_context.build_team_context_features(...)`'s own output shape (one row per `(team, season, week)`, carries real `plays`/`pass_rate` alongside every feature this module registers).
- Produces: `models.team_environment.build_team_environment_table(team_context_features: pl.DataFrame) -> pl.DataFrame` — one row per `(team, season, week)` with `player_id` (= team abbreviation), `position` (= `"TEAM_ENV"`), `availability_flag` (= `True`), `team_plays` (real, this week's outcome), `pass_rate` (real, this week's outcome), and every input feature (`implied_team_total`, `spread`, `proe_ewm_5`, `neutral_pace_ewm_8`, `opponent_neutral_pace_ewm_8`) correctly lag-shifted one week. `TRAILING_FEATURE_COLUMNS` and `CURRENT_FEATURE_COLUMNS` are module-level constants later tasks (3, 4) import.

- [ ] **Step 1: Write the failing test for promoting `lag_shift_join`**

```python
# tests/test_features_build.py — add near any existing test of the private function,
# or at the end if none exists. Check the file first for an existing
# `_lag_shift_join` test to replace/rename; if one exists, just update its call
# to use the public name instead of adding a new test.

def test_lag_shift_join_is_public() -> None:
    from ffapp.features import build

    assert hasattr(build, "lag_shift_join")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_features_build.py -k lag_shift_join_is_public -v`
Expected: FAIL (`assert False`, `lag_shift_join` doesn't exist yet — only the private `_lag_shift_join`)

- [ ] **Step 3: Promote `_lag_shift_join`**

In `src/ffapp/features/build.py`: rename `_lag_shift_join` to `lag_shift_join` (drop the leading underscore) in its `def` line, and update every call site within this same file (there are two: inside `build_player_week_features`, once for `usage_cols` and once for `team_context_cols`'s `trailing_cols`) to use the new name. Add `"lag_shift_join"` to this module's `__all__` list if one exists; if this module has no `__all__`, skip that.

- [ ] **Step 4: Run test to verify it passes, plus the full file's existing suite**

Run: `uv run pytest tests/test_features_build.py -v`
Expected: all PASS (including the new test and every pre-existing one — the rename must not change behavior)

- [ ] **Step 5: Write the failing test for `build_team_environment_table`**

```python
# tests/test_models_team_environment.py (new file)

from __future__ import annotations

import polars as pl
import pytest

from ffapp.models import team_environment


def _team_context_features() -> pl.DataFrame:
    """Two teams, two consecutive weeks -- week 2's row is what exercises
    the lag shift (week 2's feature values must come from week 1's real
    numbers, not week 2's own)."""
    return pl.DataFrame(
        {
            "team": ["KC", "KC", "BAL", "BAL"],
            "season": [2025, 2025, 2025, 2025],
            "week": [1, 2, 1, 2],
            "plays": [65, 70, 60, 58],
            "pass_rate": [0.60, 0.65, 0.55, 0.50],
            "implied_team_total": [24.5, 27.0, 20.0, 21.5],
            "spread": [-3.0, -2.5, 3.0, 2.5],
            "proe_ewm_5": [0.02, 0.03, -0.01, -0.02],
            "neutral_pace_ewm_8": [28.0, 27.5, 31.5, 31.0],
            "opponent_neutral_pace_ewm_8": [31.5, 31.0, 28.0, 27.5],
        }
    )


def test_build_team_environment_table_reshapes_to_harness_contract() -> None:
    result = team_environment.build_team_environment_table(_team_context_features())

    row = result.filter((pl.col("team") == "KC") & (pl.col("week") == 2)).row(0, named=True)
    assert row["player_id"] == "KC"
    assert row["position"] == "TEAM_ENV"
    assert row["availability_flag"] is True


def test_build_team_environment_table_carries_real_targets_unshifted() -> None:
    result = team_environment.build_team_environment_table(_team_context_features())

    row = result.filter((pl.col("team") == "KC") & (pl.col("week") == 2)).row(0, named=True)
    assert row["team_plays"] == 70  # week 2's own real outcome, not week 1's
    assert row["pass_rate"] == pytest.approx(0.65)


def test_build_team_environment_table_lag_shifts_trailing_features_by_one_week() -> None:
    result = team_environment.build_team_environment_table(_team_context_features())

    row = result.filter((pl.col("team") == "KC") & (pl.col("week") == 2)).row(0, named=True)
    assert row["proe_ewm_5"] == pytest.approx(0.02)  # week 1's value, not week 2's 0.03
    assert row["neutral_pace_ewm_8"] == pytest.approx(28.0)  # week 1's value
    assert row["opponent_neutral_pace_ewm_8"] == pytest.approx(31.5)  # week 1's value


def test_build_team_environment_table_first_week_has_null_trailing_features() -> None:
    result = team_environment.build_team_environment_table(_team_context_features())

    row = result.filter((pl.col("team") == "KC") & (pl.col("week") == 1)).row(0, named=True)
    assert row["proe_ewm_5"] is None  # no week-0 data to shift from
    assert row["team_plays"] == 65  # the real target is still present even though features are null


def test_build_team_environment_table_current_week_features_are_not_shifted() -> None:
    result = team_environment.build_team_environment_table(_team_context_features())

    row = result.filter((pl.col("team") == "KC") & (pl.col("week") == 2)).row(0, named=True)
    assert row["implied_team_total"] == pytest.approx(27.0)  # week 2's own real Vegas line
    assert row["spread"] == pytest.approx(-2.5)
```

- [ ] **Step 6: Run tests to verify they fail**

Run: `uv run pytest tests/test_models_team_environment.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ffapp.models.team_environment'`

- [ ] **Step 7: Implement `models/team_environment.py`**

```python
"""Decomposed model v2, Stage 1: team environment (SPEC.md §11.4; not a
numbered TASKS.md task -- see docs/design-model-v2-stage1-team-environment.md
for the full design). Predicts a team's own `team_plays` and `pass_rate`
for a week from Vegas lines, pace, and PROE.

`build_team_environment_table` reshapes team-week rows into the shape
`evaluation.backtest.run_walk_forward_backtest` already expects
(`player_id`/`position`/`availability_flag`) -- the same trick
`models.dst.build_dst_table` already uses, so the harness itself is never
touched and nothing else that depends on it (points, dst, availability,
quantiles) can regress.

`pass_attempts`/`rush_attempts` are never modeled directly -- they're
derived (`team_plays * pass_rate` / `team_plays * (1 - pass_rate)`) so the
two always sum to the predicted total exactly, by construction.
"""

from __future__ import annotations

import polars as pl

from ffapp.features.build import lag_shift_join

TRAILING_FEATURE_COLUMNS = [
    "proe_ewm_5",
    "neutral_pace_ewm_8",
    "opponent_neutral_pace_ewm_8",
]
CURRENT_FEATURE_COLUMNS = [
    "implied_team_total",
    "spread",
]
FEATURE_COLUMNS = TRAILING_FEATURE_COLUMNS + CURRENT_FEATURE_COLUMNS

TARGET_COLUMNS = ["team_plays", "pass_rate"]


def build_team_environment_table(team_context_features: pl.DataFrame) -> pl.DataFrame:
    """One row per real `(team, season, week)` from `team_context_features`
    (`features.team_context.build_team_context_features`'s own output),
    reshaped for the walk-forward harness: `player_id`/`position`/
    `availability_flag` added (DST-style), `plays` renamed to
    `team_plays`, trailing features lag-shifted one week, current-week
    features (Vegas lines) joined directly.
    """
    targets = team_context_features.select(
        "team", "season", "week", pl.col("plays").alias("team_plays"), "pass_rate"
    )
    shifted = lag_shift_join(targets, team_context_features, "team", TRAILING_FEATURE_COLUMNS)
    with_current = shifted.join(
        team_context_features.select("team", "season", "week", *CURRENT_FEATURE_COLUMNS),
        on=["team", "season", "week"],
        how="left",
    )
    return with_current.with_columns(
        pl.col("team").alias("player_id"),
        pl.lit("TEAM_ENV").alias("position"),
        pl.lit(True).alias("availability_flag"),
    )


__all__ = [
    "CURRENT_FEATURE_COLUMNS",
    "FEATURE_COLUMNS",
    "TARGET_COLUMNS",
    "TRAILING_FEATURE_COLUMNS",
    "build_team_environment_table",
]
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `uv run pytest tests/test_models_team_environment.py -v`
Expected: all PASS

- [ ] **Step 9: Lint and typecheck**

Run: `uv run ruff check src/ffapp/models/team_environment.py src/ffapp/features/build.py tests/test_models_team_environment.py tests/test_features_build.py && uv run ruff format --check src/ffapp/models/team_environment.py && uv run mypy src/ffapp/models/team_environment.py src/ffapp/features/build.py`
Expected: no errors

- [ ] **Step 10: Commit**

```bash
git add src/ffapp/models/team_environment.py src/ffapp/features/build.py tests/test_models_team_environment.py tests/test_features_build.py
git commit -m "feat: add build_team_environment_table for model v2 Stage 1"
```

---

### Task 3: Baseline columns (league mean + trailing ewm_4)

**Files:**
- Modify: `src/ffapp/models/baselines.py` (promote `_positional_rolling_rate` to public `pooled_rolling_mean`)
- Modify: `src/ffapp/models/team_environment.py`
- Test: `tests/test_models_baselines.py`
- Test: `tests/test_models_team_environment.py`

**Interfaces:**
- Consumes: `models.baselines.pooled_rolling_mean(df: pl.DataFrame, group_column: str, target_column: str, output_column: str) -> pl.DataFrame` (promoted and generalized in this task — the existing `_positional_rolling_rate` hardcodes `"position"` as the pooling column; this task parameterizes it so Stage 1 can pool by `"position"` too, since every row already shares the literal value `"TEAM_ENV"`, giving "pooled across every team that week" for free with zero new grouping logic).
- Produces: `team_environment.add_team_environment_baselines(table: pl.DataFrame) -> pl.DataFrame` — adds `team_plays_league_mean`, `team_plays_b2_ewm_4`, `pass_rate_league_mean`, `pass_rate_b2_ewm_4` columns to `build_team_environment_table`'s own output.

- [ ] **Step 1: Write the failing test for generalizing `_positional_rolling_rate`**

First, find the existing tests for `_positional_rolling_rate`/`add_b0_positional_mean` in `tests/test_models_baselines.py` (read the file to find them) — they call the function under its current private name and hardcode `"position"`. Add one new test confirming the generalized public version works with the pooling column explicitly named:

```python
# tests/test_models_baselines.py — add near the existing add_b0_positional_mean tests

def test_pooled_rolling_mean_works_with_a_different_group_column() -> None:
    df = pl.DataFrame(
        {
            "position": ["TEAM_ENV", "TEAM_ENV", "TEAM_ENV", "TEAM_ENV"],
            "season": [2025, 2025, 2025, 2025],
            "week": [1, 1, 2, 2],
            "target": [60.0, 70.0, 65.0, 75.0],
        }
    )

    result = baselines.pooled_rolling_mean(df, "position", "target", "pooled_mean")

    week2 = result.filter(pl.col("week") == 2)
    # week 2's pooled mean is week 1's pooled average across both rows: (60+70)/2 = 65
    assert week2["pooled_mean"].to_list() == [65.0, 65.0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_models_baselines.py -k pooled_rolling_mean -v`
Expected: FAIL with `AttributeError: module 'ffapp.models.baselines' has no attribute 'pooled_rolling_mean'`

- [ ] **Step 3: Generalize and promote `_positional_rolling_rate`**

In `src/ffapp/models/baselines.py`, replace the existing `_positional_rolling_rate` function with a public, parameterized version:

```python
def pooled_rolling_mean(
    df: pl.DataFrame, group_column: str, target_column: str, output_column: str
) -> pl.DataFrame:
    """Shared machinery behind B0, the availability base-rate baseline
    (task 1.14), and model v2 Stage 1's own league-mean baseline: pooled
    (across every row sharing the same `group_column` value),
    season-to-date-through-week-W-1 mean of `target_column`. A season's
    own week 1 has no current-season trailing data, so it falls back to
    the *entire* prior season's own mean (same fallback convention as
    `features.usage.prior_season`); the very first tracked season's own
    week 1 has neither and stays honestly null, same precedent as task
    1.7's `proe` and task 1.8's opponent adjustment for the identical
    reason.
    """
    by_week = (
        df.group_by([group_column, "season", "week"])
        .agg(
            pl.col(target_column).cast(pl.Float64).sum().alias("_week_sum"),
            pl.len().alias("_week_n"),
        )
        .sort([group_column, "season", "week"])
    )
    prior_sum = pl.col("_week_sum").cum_sum().shift(1).over([group_column, "season"])
    prior_n = pl.col("_week_n").cum_sum().shift(1).over([group_column, "season"])
    with_current_season_mean = by_week.with_columns(
        (prior_sum / prior_n).alias("_current_season_mean")
    )

    prior_season_mean = (
        df.group_by([group_column, "season"])
        .agg(pl.col(target_column).cast(pl.Float64).mean().alias("_prior_season_mean"))
        .with_columns((pl.col("season") + 1).alias("season"))
    )

    combined = (
        with_current_season_mean.join(prior_season_mean, on=[group_column, "season"], how="left")
        .with_columns(
            pl.coalesce(["_current_season_mean", "_prior_season_mean"]).alias(output_column)
        )
        .select(group_column, "season", "week", output_column)
    )

    return df.join(combined, on=[group_column, "season", "week"], how="left")
```

Update the two existing callers in this same file to pass the group column explicitly:

```python
def add_b0_positional_mean(player_week_features: pl.DataFrame) -> pl.DataFrame:
    """B0: "positional weekly mean" -- SPEC calls this the "sanity floor."
    Pooled across *every* player at a position (not per-player, the
    distinguishing feature from B1/B2) -- see `pooled_rolling_mean` for
    the shared season-to-date/prior-season-fallback mechanics.
    """
    return pooled_rolling_mean(player_week_features, "position", "target", "b0_positional_mean")


def add_availability_base_rate(player_week_features: pl.DataFrame) -> pl.DataFrame:
    """The availability model's own comparison baseline (SPEC §11.2/task
    1.14's own acceptance bar: "Brier score beats a positional base-rate
    predictor") -- exactly B0's shape (`pooled_rolling_mean`), applied to
    `availability_flag` instead of `target`: "what fraction of this
    position's players were active, on average, through last week."
    """
    return pooled_rolling_mean(
        player_week_features, "position", "availability_flag", "availability_base_rate"
    )
```

Add `"pooled_rolling_mean"` to this module's `__all__` list.

- [ ] **Step 4: Run tests to verify they pass, plus the full file's existing suite**

Run: `uv run pytest tests/test_models_baselines.py -v`
Expected: all PASS (including the new test and every pre-existing B0/availability-base-rate test — the rename/generalization must not change their behavior)

- [ ] **Step 5: Write the failing test for `add_team_environment_baselines`**

```python
# tests/test_models_team_environment.py — add to the existing file from Task 2

def _reshaped_table() -> pl.DataFrame:
    base = team_environment.build_team_environment_table(_team_context_features())
    return base


def test_add_team_environment_baselines_adds_all_four_columns() -> None:
    result = team_environment.add_team_environment_baselines(_reshaped_table())

    assert "team_plays_league_mean" in result.columns
    assert "team_plays_b2_ewm_4" in result.columns
    assert "pass_rate_league_mean" in result.columns
    assert "pass_rate_b2_ewm_4" in result.columns


def test_add_team_environment_baselines_b2_never_leaks_the_target_week() -> None:
    result = team_environment.add_team_environment_baselines(_reshaped_table())

    row = result.filter((pl.col("team") == "KC") & (pl.col("week") == 2)).row(0, named=True)
    # week 2's b2 baseline must be built only from week 1's real outcome (65),
    # never week 2's own (70) -- with a single prior week, ewm_4 of one point
    # equals that point exactly.
    assert row["team_plays_b2_ewm_4"] == pytest.approx(65.0)
```

- [ ] **Step 6: Run tests to verify they fail**

Run: `uv run pytest tests/test_models_team_environment.py -k baselines -v`
Expected: FAIL with `AttributeError: module 'ffapp.models.team_environment' has no attribute 'add_team_environment_baselines'`

- [ ] **Step 7: Implement `add_team_environment_baselines`**

In `src/ffapp/models/team_environment.py`, add this import at the top:

```python
from ffapp.models.baselines import pooled_rolling_mean
```

Add this function after `build_team_environment_table`:

```python
def add_team_environment_baselines(table: pl.DataFrame) -> pl.DataFrame:
    """Two baselines per target, following this project's established B0/B2
    pattern (SPEC §12.3) at team grain instead of player grain:

    - `*_league_mean` (B0-equivalent, sanity floor): every team pooled
      together, via `models.baselines.pooled_rolling_mean`.
    - `*_b2_ewm_4` (the real bar, same span as every other B2 in this
      project -- see `models.dst.add_dst_b2_ewm_4`): this team's own
      trailing `ewm_4`, `.shift(1)`'d so the target week's own outcome
      never leaks in.

    Stage 1's model must beat `*_b2_ewm_4` on MAE to be considered
    working -- see the design doc.
    """
    with_league_means = table
    for target_column in TARGET_COLUMNS:
        with_league_means = pooled_rolling_mean(
            with_league_means, "position", target_column, f"{target_column}_league_mean"
        )

    sorted_table = with_league_means.sort(["team", "season", "week"])
    with_b2 = sorted_table
    for target_column in TARGET_COLUMNS:
        with_b2 = with_b2.with_columns(
            pl.col(target_column)
            .ewm_mean(span=4)
            .shift(1)
            .over(["team", "season"])
            .alias(f"{target_column}_b2_ewm_4")
        )
    return with_b2


__all__ = [
    "CURRENT_FEATURE_COLUMNS",
    "FEATURE_COLUMNS",
    "TARGET_COLUMNS",
    "TRAILING_FEATURE_COLUMNS",
    "add_team_environment_baselines",
    "build_team_environment_table",
]
```

(This replaces the earlier `__all__` from Task 2 — the full, final list.)

- [ ] **Step 8: Run tests to verify they pass**

Run: `uv run pytest tests/test_models_team_environment.py -v`
Expected: all PASS

- [ ] **Step 9: Lint and typecheck**

Run: `uv run ruff check src/ffapp/models/baselines.py src/ffapp/models/team_environment.py tests/test_models_baselines.py tests/test_models_team_environment.py && uv run ruff format --check src/ffapp/models/baselines.py src/ffapp/models/team_environment.py && uv run mypy src/ffapp/models/baselines.py src/ffapp/models/team_environment.py`
Expected: no errors

- [ ] **Step 10: Commit**

```bash
git add src/ffapp/models/baselines.py src/ffapp/models/team_environment.py tests/test_models_baselines.py tests/test_models_team_environment.py
git commit -m "feat: add team-grain baselines for model v2 Stage 1"
```

---

### Task 4: The two LightGBM regressors

**Files:**
- Modify: `src/ffapp/models/team_environment.py`
- Test: `tests/test_models_team_environment.py`

**Interfaces:**
- Consumes: `config.LightGBMSettings` (existing dataclass: `n_estimators`, `learning_rate`, `num_leaves`, `min_child_samples`, `subsample`, `colsample_bytree`, `reg_lambda`); `evaluation.backtest.Predictor` protocol (`name: str`, `fit(train_rows) -> Any`, `predict(fitted, target_rows) -> pl.Series`).
- Produces: `team_environment.fit_team_environment_model(train_rows: pl.DataFrame, *, target_column: str, lightgbm_params: LightGBMSettings) -> FittedTeamEnvironmentModel`; `team_environment.predict_team_environment(model, rows) -> pl.Series`; `team_environment.TeamEnvironmentPredictor` (a `Predictor` implementation, constructed with `name` and `target_column`, used twice — once for `"team_plays"`, once for `"pass_rate"` — passed into `run_walk_forward_backtest` alongside the two `BaselinePredictor`s for that same target).

- [ ] **Step 1: Write the failing test for fit/predict**

```python
# tests/test_models_team_environment.py — add to the existing file

from ffapp.config import DEFAULT_LIGHTGBM_SETTINGS


def _training_rows() -> pl.DataFrame:
    """20 rows, enough real variation in the feature columns for LightGBM
    to fit without every leaf collapsing to a single value."""
    rows = []
    for i in range(20):
        rows.append(
            {
                "team": "KC",
                "season": 2024,
                "week": (i % 17) + 1,
                "team_plays": 60.0 + i,
                "pass_rate": 0.5 + (i % 5) * 0.02,
                "implied_team_total": 20.0 + i * 0.3,
                "spread": -3.0 + i * 0.1,
                "proe_ewm_5": 0.01 * i,
                "neutral_pace_ewm_8": 28.0 - i * 0.1,
                "opponent_neutral_pace_ewm_8": 29.0 + i * 0.1,
            }
        )
    return pl.DataFrame(rows)


def test_fit_and_predict_team_plays_model() -> None:
    train_rows = _training_rows()

    model = team_environment.fit_team_environment_model(
        train_rows, target_column="team_plays", lightgbm_params=DEFAULT_LIGHTGBM_SETTINGS
    )
    predictions = team_environment.predict_team_environment(model, train_rows)

    assert predictions.len() == train_rows.height
    assert predictions.null_count() == 0


def test_fit_and_predict_pass_rate_model() -> None:
    train_rows = _training_rows()

    model = team_environment.fit_team_environment_model(
        train_rows, target_column="pass_rate", lightgbm_params=DEFAULT_LIGHTGBM_SETTINGS
    )
    predictions = team_environment.predict_team_environment(model, train_rows)

    assert predictions.len() == train_rows.height
    assert predictions.null_count() == 0


def test_team_environment_predictor_satisfies_the_harness_protocol() -> None:
    train_rows = _training_rows()
    predictor = team_environment.TeamEnvironmentPredictor(
        name="team_env_plays", target_column="team_plays", lightgbm_params=DEFAULT_LIGHTGBM_SETTINGS
    )

    fitted = predictor.fit(train_rows)
    predictions = predictor.predict(fitted, train_rows)

    assert predictor.name == "team_env_plays"
    assert predictions.len() == train_rows.height
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_models_team_environment.py -k "fit_and_predict or predictor_satisfies" -v`
Expected: FAIL with `AttributeError` (none of `fit_team_environment_model`/`predict_team_environment`/`TeamEnvironmentPredictor` exist yet)

- [ ] **Step 3: Implement the regressor and `Predictor` wrapper**

In `src/ffapp/models/team_environment.py`, add these imports at the top:

```python
from dataclasses import dataclass
from typing import Any

import lightgbm as lgb

from ffapp.config import LightGBMSettings
```

Add this near the top of the module, after the existing constants:

```python
_INCREASING_FEATURES = {
    "team_plays": {"neutral_pace_ewm_8", "opponent_neutral_pace_ewm_8"},
    "pass_rate": {"proe_ewm_5"},
}


def monotone_constraints(target_column: str) -> list[int]:
    """LightGBM's own `monotone_constraints` vector, aligned 1:1 with
    `FEATURE_COLUMNS`'s order -- `1` (increasing) for this target's own
    `_INCREASING_FEATURES` entry, `0` (unconstrained) everywhere else.
    Mirrors `models.points.monotone_constraints`'s own shape."""
    increasing = _INCREASING_FEATURES[target_column]
    return [1 if column in increasing else 0 for column in FEATURE_COLUMNS]
```

Add this after `add_team_environment_baselines`:

```python
def to_feature_frame(rows: pl.DataFrame) -> Any:
    """Polars -> pandas only at this fit/predict boundary (CLAUDE.md's own
    convention, same as `models.points`/`models.dst`). No categorical
    columns here -- every Stage 1 feature is numeric."""
    return rows.select(FEATURE_COLUMNS).to_pandas()


@dataclass
class FittedTeamEnvironmentModel:
    booster: lgb.LGBMRegressor
    target_column: str


def fit_team_environment_model(
    train_rows: pl.DataFrame, *, target_column: str, lightgbm_params: LightGBMSettings
) -> FittedTeamEnvironmentModel:
    booster = lgb.LGBMRegressor(
        n_estimators=lightgbm_params.n_estimators,
        learning_rate=lightgbm_params.learning_rate,
        num_leaves=lightgbm_params.num_leaves,
        min_child_samples=lightgbm_params.min_child_samples,
        subsample=lightgbm_params.subsample,
        colsample_bytree=lightgbm_params.colsample_bytree,
        reg_lambda=lightgbm_params.reg_lambda,
        monotone_constraints=monotone_constraints(target_column),
        verbosity=-1,
    )
    booster.fit(to_feature_frame(train_rows), train_rows[target_column].to_numpy())
    return FittedTeamEnvironmentModel(booster=booster, target_column=target_column)


def predict_team_environment(model: FittedTeamEnvironmentModel, rows: pl.DataFrame) -> pl.Series:
    predictions = model.booster.predict(to_feature_frame(rows))
    return pl.Series("prediction", predictions, dtype=pl.Float64)


class TeamEnvironmentPredictor:
    """A `evaluation.backtest.Predictor` wrapping
    `fit_team_environment_model`/`predict_team_environment`, exercised via
    the same `run_walk_forward_backtest` harness every other predictor in
    this project uses -- construct one per target (`team_plays`,
    `pass_rate`), each with its own `name`."""

    def __init__(self, *, name: str, target_column: str, lightgbm_params: LightGBMSettings) -> None:
        self.name = name
        self.target_column = target_column
        self.lightgbm_params = lightgbm_params

    def fit(self, train_rows: pl.DataFrame) -> Any:
        return fit_team_environment_model(
            train_rows, target_column=self.target_column, lightgbm_params=self.lightgbm_params
        )

    def predict(self, fitted: Any, target_rows: pl.DataFrame) -> pl.Series:
        return predict_team_environment(fitted, target_rows)
```

Update `__all__` to the full final list:

```python
__all__ = [
    "CURRENT_FEATURE_COLUMNS",
    "FEATURE_COLUMNS",
    "TARGET_COLUMNS",
    "TRAILING_FEATURE_COLUMNS",
    "FittedTeamEnvironmentModel",
    "TeamEnvironmentPredictor",
    "add_team_environment_baselines",
    "build_team_environment_table",
    "fit_team_environment_model",
    "monotone_constraints",
    "predict_team_environment",
    "to_feature_frame",
]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_models_team_environment.py -v`
Expected: all PASS

- [ ] **Step 5: Lint and typecheck**

Run: `uv run ruff check src/ffapp/models/team_environment.py tests/test_models_team_environment.py && uv run ruff format --check src/ffapp/models/team_environment.py && uv run mypy src/ffapp/models/team_environment.py`
Expected: no errors

- [ ] **Step 6: Commit**

```bash
git add src/ffapp/models/team_environment.py tests/test_models_team_environment.py
git commit -m "feat: add the two Stage 1 LightGBM regressors (team_plays, pass_rate)"
```

---

### Task 5: Derived `pass_attempts`/`rush_attempts`

**Files:**
- Modify: `src/ffapp/models/team_environment.py`
- Test: `tests/test_models_team_environment.py`

**Interfaces:**
- Produces: `team_environment.derive_attempts(team_plays: pl.Series, pass_rate: pl.Series) -> tuple[pl.Series, pl.Series]` — returns `(pass_attempts, rush_attempts)`, always summing to `team_plays` exactly.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_models_team_environment.py — add to the existing file

def test_derive_attempts_sums_to_team_plays_exactly() -> None:
    team_plays = pl.Series([70.0, 60.0])
    pass_rate = pl.Series([0.6, 0.55])

    pass_attempts, rush_attempts = team_environment.derive_attempts(team_plays, pass_rate)

    assert pass_attempts.to_list() == pytest.approx([42.0, 33.0])
    assert rush_attempts.to_list() == pytest.approx([28.0, 27.0])
    assert (pass_attempts + rush_attempts).to_list() == pytest.approx(team_plays.to_list())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_models_team_environment.py -k derive_attempts -v`
Expected: FAIL with `AttributeError: module 'ffapp.models.team_environment' has no attribute 'derive_attempts'`

- [ ] **Step 3: Implement `derive_attempts`**

Add this function to `src/ffapp/models/team_environment.py`, after `predict_team_environment`:

```python
def derive_attempts(team_plays: pl.Series, pass_rate: pl.Series) -> tuple[pl.Series, pl.Series]:
    """`pass_attempts`/`rush_attempts` are never modeled directly -- always
    derived from the two predicted quantities, so they sum to
    `team_plays` exactly by construction (see module docstring)."""
    pass_attempts = (team_plays * pass_rate).rename("pass_attempts")
    rush_attempts = (team_plays * (1 - pass_rate)).rename("rush_attempts")
    return pass_attempts, rush_attempts
```

Add `"derive_attempts"` to `__all__`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_models_team_environment.py -v`
Expected: all PASS

- [ ] **Step 5: Lint and typecheck**

Run: `uv run ruff check src/ffapp/models/team_environment.py tests/test_models_team_environment.py && uv run mypy src/ffapp/models/team_environment.py`
Expected: no errors

- [ ] **Step 6: Commit**

```bash
git add src/ffapp/models/team_environment.py tests/test_models_team_environment.py
git commit -m "feat: derive pass_attempts/rush_attempts for model v2 Stage 1"
```

---

### Task 6: Real walk-forward verification against 2021-2025 data

**Files:**
- Create: `notebooks/evaluate_team_environment_v2_stage1.py` (a one-off script, matching this project's own `notebooks/` = scratch convention — see CLAUDE.md: "`notebooks/` is scratch. No module under `src/` may import from it.")
- Modify: `docs/JOURNAL.md` (record the real result)
- Modify: `HANDOFF.md` (update current state)

**Interfaces:**
- Consumes everything built in Tasks 1-5, plus already-existing real pipeline pieces: `ingest.nflverse.fetch_team_stats`/`fetch_schedules`/`fetch_pbp`/`fetch_player_stats` (or whatever this project's own established loader for `interim/team_week_context.parquet`'s upstream sources is — check `HANDOFF.md` §6's rebuild sequence for the exact real call chain used to build `team_week_context.parquet` before this task, since that table must already exist locally), `evaluation.backtest.run_walk_forward_backtest`, `evaluation.metrics.accuracy_metrics`.
- Produces: no new library code — this is a verification step whose output is a real, documented result (same as every other model task's own "verified against real data" bar in this project).

- [ ] **Step 1: Write the verification script**

```python
# notebooks/evaluate_team_environment_v2_stage1.py
"""One-off script: real walk-forward evaluation of model v2 Stage 1 against
2021-2025 data. Scratch, per CLAUDE.md's notebooks/ convention -- not
imported by anything under src/."""

import polars as pl

from ffapp.config import DEFAULT_LIGHTGBM_SETTINGS, load_settings
from ffapp.evaluation.backtest import BaselinePredictor, run_walk_forward_backtest
from ffapp.evaluation.metrics import accuracy_metrics
from ffapp.features import team_context
from ffapp.models import team_environment

settings = load_settings()

team_week_context = pl.read_parquet(settings.data_root / "interim" / "team_week_context.parquet")
schedule = pl.read_parquet(settings.data_root / "interim" / "schedule.parquet")
snap_counts = pl.read_parquet(settings.data_root / "raw" / "nflverse" / "snap_counts_2015-2025.parquet")
injuries = pl.read_parquet(settings.data_root / "interim" / "injuries.parquet")
usage_features_path = settings.data_root / "features" / "player_week_features.parquet"
usage_features = pl.read_parquet(usage_features_path).select(
    "player_id", "season", "week", "team", "target_share_ewm_3", "carry_share_ewm_3"
)

team_context_features = team_context.build_team_context_features(
    team_week_context, schedule, snap_counts, injuries, usage_features
)
table = team_environment.build_team_environment_table(team_context_features)
table = team_environment.add_team_environment_baselines(table)

validation_seasons = [2021, 2022, 2023, 2024, 2025]

for target_column, baseline_league_col, baseline_b2_col in [
    ("team_plays", "team_plays_league_mean", "team_plays_b2_ewm_4"),
    ("pass_rate", "pass_rate_league_mean", "pass_rate_b2_ewm_4"),
]:
    predictors = [
        team_environment.TeamEnvironmentPredictor(
            name="team_env_model",
            target_column=target_column,
            lightgbm_params=DEFAULT_LIGHTGBM_SETTINGS,
        ),
        BaselinePredictor(name="league_mean", column=baseline_league_col),
        BaselinePredictor(name="trailing_ewm_4", column=baseline_b2_col),
    ]
    predictions = run_walk_forward_backtest(
        table,
        schedule,
        predictors,
        validation_seasons=validation_seasons,
        train_start=settings.seasons.train_start,
        min_train_rows=settings.model.min_train_rows,
        target_column=target_column,
    )
    print(f"\n=== {target_column} ===")
    for result in accuracy_metrics(predictions):
        if result.scope == "all" and result.position is None:
            print(f"{result.predictor}: {result.metric}={result.value:.4f}")
```

- [ ] **Step 2: Run it against real cached data**

Run: `uv run python notebooks/evaluate_team_environment_v2_stage1.py`

If this fails because `interim/team_week_context.parquet` (or any other input) doesn't exist locally yet, rebuild it first following `HANDOFF.md` §6's documented sequence — do not guess at a substitute path.

- [ ] **Step 3: Record the real result**

Read the printed MAE values for both targets. Add an entry to `docs/JOURNAL.md` §2 (Completed tasks), following this project's existing entry style (see any recent entry for the exact voice/format), stating:
- The real MAE for `team_env_model` vs `league_mean` vs `trailing_ewm_4`, for both `team_plays` and `pass_rate`.
- Whether the model beat `trailing_ewm_4` (the real bar, per the approved design) on both targets, one, or neither.
- If it did not beat the baseline on one or both targets, state that plainly as a real, documented result — same treatment as task 1.15's own outcome (CLAUDE.md: this is not a bug to silently fix). Do not tune parameters to force a passing number; if the honest result is "doesn't beat baseline," that itself is the deliverable, and the next step (whether to try again, adjust the design, or stop here) is a decision for the project owner, not something to resolve unilaterally in this task.

Update `HANDOFF.md`'s "Next task" line and current-session narrative to reflect this real outcome.

- [ ] **Step 4: Commit**

```bash
git add notebooks/evaluate_team_environment_v2_stage1.py docs/JOURNAL.md HANDOFF.md
git commit -m "test: real walk-forward evaluation of model v2 Stage 1 against 2021-2025 data"
```

---

## After this plan

Do not proceed to Stage 2 (Opportunity) automatically. Report the real result from Task 6 to the project owner and let them decide next steps — same "your call" pattern this project has used at every prior real decision point (see `docs/design-model-v2-stage1-team-environment.md`'s own scope section).
