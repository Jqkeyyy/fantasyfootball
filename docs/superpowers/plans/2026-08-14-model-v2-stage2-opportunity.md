# Decomposed Model v2 — Stage 2 (Opportunity) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and evaluate Stage 2 of SPEC.md §11.4's decomposed model v2 — arithmetic composition of a player's own trailing share of team volume with Stage 1's own predicted team volume, producing expected targets/carries/red-zone touches per player — and prove (against real 2021-2025 data) whether it beats a player's own trailing raw count.

**Architecture:** No trained model. `models/opportunity.py::build_opportunity_table` joins already-existing, already-lag-shifted player usage-share features to Stage 1's own out-of-sample walk-forward predictions (never ground truth) and multiplies. Position eligibility is gated explicitly (share columns aren't position-filtered by construction). Evaluation reuses `evaluation.metrics.accuracy_metrics` — the same accuracy/CI reporting every other model in this project uses — via a small reshape helper, rather than a second hand-rolled MAE calculation.

**Tech Stack:** Python 3.11+, polars, pytest, `uv run`. No LightGBM in this stage.

**Spec:** `docs/design-model-v2-stage2-opportunity.md`

## Global Constraints

- No random train/test splits — Stage 1's own predictions (which this stage consumes) are already walk-forward; this stage adds no new time-ordering risk itself, but must not introduce one (e.g. never join a player's own future-week outcome onto a past-week row).
- Every real evaluation must scope to `season_type == "REG"` from the start (`player_week_features.parquet` and `player_week_usage.parquet` both carry real postseason weeks 19-22, confirmed present for 2024; `tools/sos.py` already established this exact scoping convention) — apply proactively, not after a review catches its absence, per Stage 1's own final-review lesson.
- Every real evaluation must report `n_obs` and CI alongside MAE, per Stage 1's own final-review lesson (SPEC §12.5).
- Type hints on every public function; `ruff check`/`ruff format --check`/`mypy src/` must stay clean.
- Tests before implementation (TDD) — write the failing test, watch it fail for the right reason, then implement.
- No live network calls in tests — small hand-built polars fixtures only, matching this repo's existing test files.
- `evaluation/backtest.py` must not be modified — this stage's evaluation reuses `run_walk_forward_backtest` only to re-run Stage 1's own model (already built), never to fit anything new.
- This stage produces no trained model and adds no wiring into any player-facing consumer (the draft board, weekly rankings, etc.) — stays standalone, same precedent as Stage 1.

---

### Task 1: Promote `PASS_CATCHERS_AND_RB`/`RB_QB` to public in `features/usage.py`

**Files:**
- Modify: `src/ffapp/features/usage.py`
- Test: `tests/test_features_usage.py`

**Interfaces:**
- Produces: `usage.PASS_CATCHERS_AND_RB: list[str]` (`["WR", "TE", "RB"]`), `usage.RB_QB: list[str]` (`["RB", "QB"]`) — later tasks (2) import both directly for real position-eligibility gating. These are real position lists, not new data — no behavior change to `usage.py` itself, only visibility.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_features_usage.py — add near the top-level constants section
# (find the file's own existing import block and add to it if not already
# imported: `from ffapp.features import usage`)

def test_pass_catchers_and_rb_and_rb_qb_are_public_and_correct() -> None:
    assert usage.PASS_CATCHERS_AND_RB == ["WR", "TE", "RB"]
    assert usage.RB_QB == ["RB", "QB"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_features_usage.py -k pass_catchers_and_rb_and_rb_qb -v`
Expected: FAIL with `AttributeError: module 'ffapp.features.usage' has no attribute 'PASS_CATCHERS_AND_RB'`

- [ ] **Step 3: Rename the two constants**

In `src/ffapp/features/usage.py`, find the position-group constants near the
top of the file (`_ALL_OFFENSE`, `_PASS_CATCHERS`, `_PASS_CATCHERS_AND_RB`,
`_RB_ONLY`, `_RB_QB`, `_QB_ONLY`). Rename only these two (drop the leading
underscore):

```python
PASS_CATCHERS_AND_RB = ["WR", "TE", "RB"]
```
```python
RB_QB = ["RB", "QB"]
```

Then update every use of the old names within this same file — search for
`_PASS_CATCHERS_AND_RB` and `_RB_QB` (there are 3 more occurrences total,
all inside the `_WINDOWED_FEATURES` list literal) and drop the leading
underscore on each. Do not rename `_ALL_OFFENSE`, `_PASS_CATCHERS`,
`_RB_ONLY`, or `_QB_ONLY` — only these two are needed publicly.

Add both new names to this module's `__all__` list (alphabetical, matching
the existing list's own ordering convention).

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_features_usage.py -k pass_catchers_and_rb_and_rb_qb -v`
Expected: PASS

- [ ] **Step 5: Run the full file's existing suite to confirm the rename didn't break anything**

Run: `uv run pytest tests/test_features_usage.py -v`
Expected: all PASS

- [ ] **Step 6: Lint and typecheck**

Run: `uv run ruff check src/ffapp/features/usage.py tests/test_features_usage.py && uv run ruff format --check src/ffapp/features/usage.py && uv run mypy src/ffapp/features/usage.py`
Expected: no errors

- [ ] **Step 7: Commit**

```bash
git add src/ffapp/features/usage.py tests/test_features_usage.py
git commit -m "feat: promote PASS_CATCHERS_AND_RB/RB_QB to public in features/usage.py"
```

---

### Task 2: `build_opportunity_table` — the arithmetic composition

**Files:**
- Create: `src/ffapp/models/opportunity.py`
- Test: `tests/test_models_opportunity.py`

**Interfaces:**
- Consumes: `features.usage.PASS_CATCHERS_AND_RB`/`RB_QB` (Task 1). `player_week_features` (a DataFrame shaped like `data/features/player_week_features.parquet` — real columns `player_id`, `season`, `week`, `team`, `position`, `target_share_ewm_3`, `carry_share_ewm_3`, `rz_touch_share_ewm_6`, already lag-shifted by the existing pipeline, task 1.9). `player_week_usage` (shaped like `data/interim/player_week_usage.parquet` — real columns `player_id`, `season`, `week`, `targets`, `carries`, `rz_targets`, `rz_carries`, same-week real counts, not shifted). `stage1_predictions` (a plain DataFrame the caller builds — `team`, `season`, `week`, `predicted_team_plays`, `predicted_pass_attempts`, `predicted_rush_attempts` — Stage 1's own out-of-sample predictions, built in Task 5's evaluation script, not by this function).
- Produces: `opportunity.build_opportunity_table(player_week_features, player_week_usage, stage1_predictions) -> pl.DataFrame` and `opportunity.TARGET_COLUMNS = ["targets", "carries", "rz_touches"]`, both imported directly by Task 3. Output columns include `player_id`, `season`, `week`, `team`, `position`, the three real target columns (`targets`, `carries`, `rz_touches`), and the three composed columns (`expected_targets`, `expected_carries`, `expected_rz_touches`).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_models_opportunity.py (new file)

from __future__ import annotations

import polars as pl
import pytest

from ffapp.models import opportunity


def _player_week_features() -> pl.DataFrame:
    """One team (KC), one real week, three players at three different
    positions -- WR (in PASS_CATCHERS_AND_RB, not RB_QB), RB (in both),
    QB (in RB_QB, not PASS_CATCHERS_AND_RB). Share values are deliberately
    non-null/non-zero for every player regardless of position eligibility
    (matching the real data: features.usage's windowing computes shares for
    every row, not just eligible positions) -- this is what exercises the
    position-gating logic, not just null propagation."""
    return pl.DataFrame(
        {
            "player_id": ["wr1", "rb1", "qb1"],
            "season": [2025, 2025, 2025],
            "week": [1, 1, 1],
            "team": ["KC", "KC", "KC"],
            "position": ["WR", "RB", "QB"],
            "target_share_ewm_3": [0.25, 0.10, 0.01],
            "carry_share_ewm_3": [0.02, 0.60, 0.08],
            "rz_touch_share_ewm_6": [0.15, 0.30, 0.05],
        }
    )


def _player_week_usage() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "player_id": ["wr1", "rb1", "qb1"],
            "season": [2025, 2025, 2025],
            "week": [1, 1, 1],
            "targets": [8, 3, 0],
            "carries": [1, 15, 4],
            "rz_targets": [1, 0, 0],
            "rz_carries": [0, 3, 1],
        }
    )


def _stage1_predictions() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "team": ["KC"],
            "season": [2025],
            "week": [1],
            "predicted_team_plays": [55.0],
            "predicted_pass_attempts": [30.0],
            "predicted_rush_attempts": [25.0],
        }
    )


def test_build_opportunity_table_computes_expected_targets_for_eligible_positions() -> None:
    result = opportunity.build_opportunity_table(
        _player_week_features(), _player_week_usage(), _stage1_predictions()
    )

    wr = result.filter(pl.col("player_id") == "wr1").row(0, named=True)
    rb = result.filter(pl.col("player_id") == "rb1").row(0, named=True)
    assert wr["expected_targets"] == pytest.approx(0.25 * 30.0)  # WR is in PASS_CATCHERS_AND_RB
    assert rb["expected_targets"] == pytest.approx(0.10 * 30.0)  # RB is in PASS_CATCHERS_AND_RB


def test_build_opportunity_table_nulls_expected_targets_for_ineligible_position() -> None:
    result = opportunity.build_opportunity_table(
        _player_week_features(), _player_week_usage(), _stage1_predictions()
    )

    qb = result.filter(pl.col("player_id") == "qb1").row(0, named=True)
    # QB is NOT in PASS_CATCHERS_AND_RB -- must be null even though
    # target_share_ewm_3 itself has a real (non-null) value of 0.01.
    assert qb["expected_targets"] is None


def test_build_opportunity_table_computes_expected_carries_for_eligible_positions() -> None:
    result = opportunity.build_opportunity_table(
        _player_week_features(), _player_week_usage(), _stage1_predictions()
    )

    rb = result.filter(pl.col("player_id") == "rb1").row(0, named=True)
    qb = result.filter(pl.col("player_id") == "qb1").row(0, named=True)
    assert rb["expected_carries"] == pytest.approx(0.60 * 25.0)  # RB is in RB_QB
    assert qb["expected_carries"] == pytest.approx(0.08 * 25.0)  # QB is in RB_QB


def test_build_opportunity_table_nulls_expected_carries_for_ineligible_position() -> None:
    result = opportunity.build_opportunity_table(
        _player_week_features(), _player_week_usage(), _stage1_predictions()
    )

    wr = result.filter(pl.col("player_id") == "wr1").row(0, named=True)
    # WR is NOT in RB_QB -- must be null even though carry_share_ewm_3
    # itself has a real (non-null) value of 0.02.
    assert wr["expected_carries"] is None


def test_build_opportunity_table_computes_expected_rz_touches_for_eligible_positions() -> None:
    result = opportunity.build_opportunity_table(
        _player_week_features(), _player_week_usage(), _stage1_predictions()
    )

    wr = result.filter(pl.col("player_id") == "wr1").row(0, named=True)
    qb = result.filter(pl.col("player_id") == "qb1").row(0, named=True)
    assert wr["expected_rz_touches"] == pytest.approx(0.15 * 55.0)  # WR is in PASS_CATCHERS_AND_RB
    assert qb["expected_rz_touches"] is None  # QB is NOT in PASS_CATCHERS_AND_RB


def test_build_opportunity_table_derives_real_rz_touches_from_targets_and_carries() -> None:
    result = opportunity.build_opportunity_table(
        _player_week_features(), _player_week_usage(), _stage1_predictions()
    )

    rb = result.filter(pl.col("player_id") == "rb1").row(0, named=True)
    assert rb["rz_touches"] == 3  # rz_targets=0 + rz_carries=3, real counts carried through


def test_build_opportunity_table_carries_real_target_counts_unmodified() -> None:
    result = opportunity.build_opportunity_table(
        _player_week_features(), _player_week_usage(), _stage1_predictions()
    )

    wr = result.filter(pl.col("player_id") == "wr1").row(0, named=True)
    assert wr["targets"] == 8
    assert wr["carries"] == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_models_opportunity.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ffapp.models.opportunity'`

- [ ] **Step 3: Implement `models/opportunity.py`**

```python
"""Decomposed model v2, Stage 2: opportunity (SPEC.md §11.4; not a numbered
TASKS.md task -- see docs/design-model-v2-stage2-opportunity.md for the full
design). Predicts a player's own expected targets, carries, and red-zone
touches for a week, as a plain arithmetic composition of their own trailing
share of team volume and Stage 1's own predicted team volume -- no trained
model here, per that design's own reasoning: Stage 1's own trained model
added real complexity (and a real constraint-sign bug) without beating a
naive baseline for team-level volume, so there's no reason to expect a
second trained model would do better at this stage either.

Position eligibility is NOT automatic from `features.usage`'s own share
columns -- `target_share_ewm_3`/`carry_share_ewm_3`/`rz_touch_share_ewm_6`
are computed for every row regardless of position (the `_WindowedFeature
.positions` field there is metadata for the feature registry only, not a
row filter that nulls out ineligible rows), so a QB row can carry a real
but meaningless non-null `target_share_ewm_3` value. This module gates each
formula explicitly using `features.usage.PASS_CATCHERS_AND_RB`/`RB_QB`, the
same real position lists `features.usage`'s own share features are already
documented against.

`stage1_predictions` must be Stage 1's own real out-of-sample walk-forward
predictions (`evaluation.backtest.run_walk_forward_backtest`'s own output
for `team_environment.TeamEnvironmentPredictor`), never Stage 1's ground
truth -- using ground truth would hide Stage 1's real prediction error and
make this stage look better than it would actually perform live. Building
that predictions table is the evaluation script's job (not this module's),
matching the same separation Stage 1 itself keeps between its own table-
building functions and its evaluation script.
"""

from __future__ import annotations

import polars as pl

from ffapp.features.usage import PASS_CATCHERS_AND_RB, RB_QB

TARGET_COLUMNS = ["targets", "carries", "rz_touches"]


def build_opportunity_table(
    player_week_features: pl.DataFrame,
    player_week_usage: pl.DataFrame,
    stage1_predictions: pl.DataFrame,
) -> pl.DataFrame:
    """One row per real `(player_id, season, week)` from
    `player_week_features` (task 1.9's own assembled table -- already has
    real `position`/`team` and the already-lag-shifted trailing shares),
    joined to `player_week_usage`'s own real, same-week target/carry/
    red-zone-touch counts (the real outcomes this stage is trying to
    predict -- not shifted, matching how Stage 1's own `team_plays`/
    `pass_rate` targets came from `team_context`'s same-week real values)
    and `stage1_predictions` (see module docstring).

    `expected_targets`/`expected_carries`/`expected_rz_touches` are null
    for a position that share doesn't apply to (e.g. `expected_targets`
    for a QB row) -- an honest "not applicable," not a guessed zero.
    """
    features = player_week_features.select(
        "player_id",
        "season",
        "week",
        "team",
        "position",
        "target_share_ewm_3",
        "carry_share_ewm_3",
        "rz_touch_share_ewm_6",
    )
    with_predictions = features.join(
        stage1_predictions, on=["team", "season", "week"], how="left"
    )
    with_real_outcomes = with_predictions.join(
        player_week_usage.select(
            "player_id", "season", "week", "targets", "carries", "rz_targets", "rz_carries"
        ),
        on=["player_id", "season", "week"],
        how="left",
    )
    return with_real_outcomes.with_columns(
        (pl.col("rz_targets") + pl.col("rz_carries")).alias("rz_touches"),
        pl.when(pl.col("position").is_in(PASS_CATCHERS_AND_RB))
        .then(pl.col("target_share_ewm_3") * pl.col("predicted_pass_attempts"))
        .otherwise(None)
        .alias("expected_targets"),
        pl.when(pl.col("position").is_in(RB_QB))
        .then(pl.col("carry_share_ewm_3") * pl.col("predicted_rush_attempts"))
        .otherwise(None)
        .alias("expected_carries"),
        pl.when(pl.col("position").is_in(PASS_CATCHERS_AND_RB))
        .then(pl.col("rz_touch_share_ewm_6") * pl.col("predicted_team_plays"))
        .otherwise(None)
        .alias("expected_rz_touches"),
    )


__all__ = ["TARGET_COLUMNS", "build_opportunity_table"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_models_opportunity.py -v`
Expected: all PASS

- [ ] **Step 5: Lint and typecheck**

Run: `uv run ruff check src/ffapp/models/opportunity.py tests/test_models_opportunity.py && uv run ruff format --check src/ffapp/models/opportunity.py && uv run mypy src/ffapp/models/opportunity.py`
Expected: no errors

- [ ] **Step 6: Commit**

```bash
git add src/ffapp/models/opportunity.py tests/test_models_opportunity.py
git commit -m "feat: add build_opportunity_table for model v2 Stage 2"
```

---

### Task 3: `add_opportunity_baselines`

**Files:**
- Modify: `src/ffapp/models/opportunity.py`
- Test: `tests/test_models_opportunity.py`

**Interfaces:**
- Consumes: `models.baselines.pooled_rolling_mean(df, group_column, target_column, output_column) -> pl.DataFrame` (already exists, promoted during Stage 1). `opportunity.TARGET_COLUMNS` (Task 2).
- Produces: `opportunity.add_opportunity_baselines(table: pl.DataFrame) -> pl.DataFrame`, adding `targets_league_mean`, `targets_b2_ewm_4`, `carries_league_mean`, `carries_b2_ewm_4`, `rz_touches_league_mean`, `rz_touches_b2_ewm_4` to `build_opportunity_table`'s own output. Imported directly by Task 5's evaluation script.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_models_opportunity.py — add to the existing file

def _baseline_fixture_table() -> pl.DataFrame:
    """Two WRs, two weeks each -- enough to exercise both the pooled
    league-mean (two players at the same position, same week) and the
    per-player trailing ewm_4 (two real weeks for the same player, so the
    shift is provably not leaking week 2's own outcome)."""
    return pl.DataFrame(
        {
            "player_id": ["wrA", "wrA", "wrB", "wrB"],
            "season": [2025, 2025, 2025, 2025],
            "week": [1, 2, 1, 2],
            "position": ["WR", "WR", "WR", "WR"],
            "targets": [6, 10, 8, 9],
            "carries": [0, 0, 0, 0],
            "rz_touches": [1, 2, 0, 1],
        }
    )


def test_add_opportunity_baselines_adds_all_six_columns() -> None:
    result = opportunity.add_opportunity_baselines(_baseline_fixture_table())

    for target_column in opportunity.TARGET_COLUMNS:
        assert f"{target_column}_league_mean" in result.columns
        assert f"{target_column}_b2_ewm_4" in result.columns


def test_add_opportunity_baselines_b2_never_leaks_the_target_week() -> None:
    result = opportunity.add_opportunity_baselines(_baseline_fixture_table())

    week2 = result.filter((pl.col("player_id") == "wrA") & (pl.col("week") == 2)).row(
        0, named=True
    )
    # week 2's b2 baseline must be built only from week 1's real outcome (6),
    # never week 2's own (10) -- with a single prior week, ewm_4 of one
    # point equals that point exactly.
    assert week2["targets_b2_ewm_4"] == pytest.approx(6.0)


def test_add_opportunity_baselines_league_mean_pools_across_players_at_the_position() -> None:
    result = opportunity.add_opportunity_baselines(_baseline_fixture_table())

    week2_wrA = result.filter((pl.col("player_id") == "wrA") & (pl.col("week") == 2)).row(
        0, named=True
    )
    # week 2's pooled mean is week 1's real values across BOTH WRs: (6+8)/2 = 7
    assert week2_wrA["targets_league_mean"] == pytest.approx(7.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_models_opportunity.py -k baselines -v`
Expected: FAIL with `AttributeError: module 'ffapp.models.opportunity' has no attribute 'add_opportunity_baselines'`

- [ ] **Step 3: Implement `add_opportunity_baselines`**

In `src/ffapp/models/opportunity.py`, add this import at the top:

```python
from ffapp.models.baselines import pooled_rolling_mean
```

Add this function after `build_opportunity_table`:

```python
def add_opportunity_baselines(table: pl.DataFrame) -> pl.DataFrame:
    """Two baselines per target, following this project's established B0/B2
    pattern (SPEC §12.3) at player grain:

    - `*_league_mean` (B0-equivalent, sanity floor): every player pooled by
      `position`, via `models.baselines.pooled_rolling_mean` -- a position-
      blind pool (RB and WR carries averaged together) would be meaningless,
      unlike Stage 1's single "TEAM_ENV" pool.
    - `*_b2_ewm_4` (the real bar): this player's own trailing `ewm_4` of the
      real raw count, `.shift(1)`'d so the target week's own outcome never
      leaks in -- same shape as every other B2 in this project (see
      `models.dst.add_dst_b2_ewm_4`, `models.team_environment
      .add_team_environment_baselines`).
    """
    with_league_means = table
    for target_column in TARGET_COLUMNS:
        with_league_means = pooled_rolling_mean(
            with_league_means, "position", target_column, f"{target_column}_league_mean"
        )

    sorted_table = with_league_means.sort(["player_id", "season", "week"])
    with_b2 = sorted_table
    for target_column in TARGET_COLUMNS:
        with_b2 = with_b2.with_columns(
            pl.col(target_column)
            .ewm_mean(span=4)
            .shift(1)
            .over(["player_id", "season"])
            .alias(f"{target_column}_b2_ewm_4")
        )
    return with_b2
```

Update `__all__` to the full list:

```python
__all__ = ["TARGET_COLUMNS", "add_opportunity_baselines", "build_opportunity_table"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_models_opportunity.py -v`
Expected: all PASS

- [ ] **Step 5: Lint and typecheck**

Run: `uv run ruff check src/ffapp/models/opportunity.py tests/test_models_opportunity.py && uv run mypy src/ffapp/models/opportunity.py`
Expected: no errors

- [ ] **Step 6: Commit**

```bash
git add src/ffapp/models/opportunity.py tests/test_models_opportunity.py
git commit -m "feat: add player-grain baselines for model v2 Stage 2"
```

---

### Task 4: `to_predictions_frame` — reuse `accuracy_metrics` instead of a second MAE calculation

**Files:**
- Modify: `src/ffapp/models/opportunity.py`
- Test: `tests/test_models_opportunity.py`

**Interfaces:**
- Consumes: nothing new — pure reshape of `build_opportunity_table`/`add_opportunity_baselines`'s combined output.
- Produces: `opportunity.to_predictions_frame(table, *, real_column, composition_column, trailing_raw_column, league_mean_column) -> pl.DataFrame`, shaped for `evaluation.metrics.accuracy_metrics` (columns `player_id`, `season`, `week`, `position`, `team`, `target`, `predictor`, `prediction`). Used three times (once per output) by Task 5's evaluation script.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_models_opportunity.py — add to the existing file

def test_to_predictions_frame_reshapes_to_one_row_per_predictor() -> None:
    table = pl.DataFrame(
        {
            "player_id": ["wr1"],
            "season": [2025],
            "week": [1],
            "position": ["WR"],
            "team": ["KC"],
            "targets": [8],
            "expected_targets": [7.5],
            "targets_b2_ewm_4": [6.0],
            "targets_league_mean": [7.0],
        }
    )

    result = opportunity.to_predictions_frame(
        table,
        real_column="targets",
        composition_column="expected_targets",
        trailing_raw_column="targets_b2_ewm_4",
        league_mean_column="targets_league_mean",
    )

    assert result.height == 3  # one row per predictor
    by_predictor = {row["predictor"]: row for row in result.to_dicts()}
    assert by_predictor["opportunity_composition"]["prediction"] == pytest.approx(7.5)
    assert by_predictor["trailing_raw"]["prediction"] == pytest.approx(6.0)
    assert by_predictor["league_mean"]["prediction"] == pytest.approx(7.0)
    for row in result.to_dicts():
        assert row["target"] == 8  # the real outcome, same on every predictor's own row
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_models_opportunity.py -k to_predictions_frame -v`
Expected: FAIL with `AttributeError: module 'ffapp.models.opportunity' has no attribute 'to_predictions_frame'`

- [ ] **Step 3: Implement `to_predictions_frame`**

Add this function to `src/ffapp/models/opportunity.py`, after `add_opportunity_baselines`:

```python
def to_predictions_frame(
    table: pl.DataFrame,
    *,
    real_column: str,
    composition_column: str,
    trailing_raw_column: str,
    league_mean_column: str,
) -> pl.DataFrame:
    """Reshapes one opportunity output's wide columns into the long
    `player_id`/`season`/`week`/`position`/`team`/`predictor`/`prediction`/
    `target` shape `evaluation.metrics.accuracy_metrics` expects, so Stage
    2's evaluation reuses the exact same accuracy/CI reporting every other
    model in this project already uses (SPEC §12.5), rather than a second,
    hand-rolled MAE calculation.
    """
    base = table.select(
        "player_id",
        "season",
        "week",
        "position",
        "team",
        pl.col(real_column).alias("target"),
    )
    frames = []
    for predictor_name, column in [
        ("opportunity_composition", composition_column),
        ("trailing_raw", trailing_raw_column),
        ("league_mean", league_mean_column),
    ]:
        frames.append(
            base.with_columns(
                pl.lit(predictor_name).alias("predictor"),
                table[column].alias("prediction"),
            )
        )
    return pl.concat(frames, how="vertical_relaxed")
```

Update `__all__`:

```python
__all__ = [
    "TARGET_COLUMNS",
    "add_opportunity_baselines",
    "build_opportunity_table",
    "to_predictions_frame",
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_models_opportunity.py -v`
Expected: all PASS

- [ ] **Step 5: Lint and typecheck**

Run: `uv run ruff check src/ffapp/models/opportunity.py tests/test_models_opportunity.py && uv run mypy src/ffapp/models/opportunity.py`
Expected: no errors

- [ ] **Step 6: Commit**

```bash
git add src/ffapp/models/opportunity.py tests/test_models_opportunity.py
git commit -m "feat: reuse accuracy_metrics reporting for model v2 Stage 2"
```

---

### Task 5: Real evaluation against 2021-2025 data

**Files:**
- Create: `notebooks/evaluate_opportunity_v2_stage2.py` (scratch, per CLAUDE.md's `notebooks/` convention)
- Modify: `docs/JOURNAL.md`
- Modify: `HANDOFF.md`

**Interfaces:**
- Consumes everything built in Tasks 1-4, plus Stage 1's own already-built pieces (`features.team_context.build_team_context_features`, `models.team_environment.build_team_environment_table`, `models.team_environment.TeamEnvironmentPredictor`, `models.team_environment.derive_attempts`), plus already-existing real pipeline pieces (`evaluation.backtest.run_walk_forward_backtest`, `evaluation.metrics.accuracy_metrics`).
- Produces: no new library code — a real, documented result.

**This re-runs Stage 1's own model fitting** (to capture its real out-of-sample predictions, not just its accuracy summary) **and will take real wall-clock time** — Stage 1's own equivalent runs on this machine have taken anywhere from ~15 minutes to 80+ minutes depending on system load from other running applications. This is expected, not a sign something is broken; let it finish.

- [ ] **Step 1: Write the verification script**

```python
# notebooks/evaluate_opportunity_v2_stage2.py
"""One-off script: real evaluation of model v2 Stage 2 (opportunity)
against 2021-2025 data. Scratch, per CLAUDE.md's notebooks/ convention --
not imported by anything under src/."""

import polars as pl

from ffapp.config import DEFAULT_LIGHTGBM_SETTINGS, load_settings
from ffapp.evaluation.backtest import run_walk_forward_backtest
from ffapp.evaluation.metrics import accuracy_metrics
from ffapp.features import team_context
from ffapp.models import opportunity, team_environment

settings = load_settings()

# --- Load real data. ---
team_week_context = pl.read_parquet(settings.data_root / "interim" / "team_week_context.parquet")
schedule = pl.read_parquet(settings.data_root / "interim" / "schedule.parquet")
snap_counts = pl.read_parquet(settings.data_root / "raw" / "nflverse" / "snap_counts_2015-2025.parquet")
injuries = pl.read_parquet(settings.data_root / "interim" / "injuries.parquet")
player_week_features = pl.read_parquet(settings.data_root / "features" / "player_week_features.parquet")
player_week_usage = pl.read_parquet(settings.data_root / "interim" / "player_week_usage.parquet")

# Real regular season only, upstream of everything else -- both
# player_week_features.parquet and player_week_usage.parquet carry real
# postseason weeks (19-22), confirmed present for real 2024 data. Same
# scoping tools.sos and Stage 1's own (corrected) evaluation script already
# established, applied proactively here from the start.
schedule = schedule.filter(pl.col("season_type") == "REG")
_reg_weeks = schedule.select("season", "week").unique()
team_week_context = team_week_context.join(_reg_weeks, on=["season", "week"], how="inner")
player_week_features = player_week_features.join(_reg_weeks, on=["season", "week"], how="inner")
player_week_usage = player_week_usage.join(_reg_weeks, on=["season", "week"], how="inner")

# --- Rebuild Stage 1's own table (same real data, same real construction
# its own evaluation script already uses) so its backtest can be re-run to
# capture its own out-of-sample predictions this time, not just accuracy. ---
usage_features_for_stage1 = player_week_features.select(
    "player_id", "season", "week", "team", "target_share_ewm_3", "carry_share_ewm_3"
)
team_context_features = team_context.build_team_context_features(
    team_week_context, schedule, snap_counts, injuries, usage_features_for_stage1
)
stage1_table = team_environment.build_team_environment_table(team_context_features)

validation_seasons = [2021, 2022, 2023, 2024, 2025]

# --- Re-run Stage 1's own backtest, once per target, keeping the full
# predictions this time (not just accuracy_metrics's summary). ---
stage1_predicted_team_plays = run_walk_forward_backtest(
    stage1_table,
    schedule,
    [
        team_environment.TeamEnvironmentPredictor(
            name="team_env_model",
            target_column="team_plays",
            lightgbm_params=DEFAULT_LIGHTGBM_SETTINGS,
        )
    ],
    validation_seasons=validation_seasons,
    train_start=settings.seasons.train_start,
    min_train_rows=settings.model.min_train_rows,
    target_column="team_plays",
).select(
    pl.col("player_id").alias("team"), "season", "week", pl.col("prediction").alias("predicted_team_plays")
)

stage1_predicted_pass_rate = run_walk_forward_backtest(
    stage1_table,
    schedule,
    [
        team_environment.TeamEnvironmentPredictor(
            name="team_env_model",
            target_column="pass_rate",
            lightgbm_params=DEFAULT_LIGHTGBM_SETTINGS,
        )
    ],
    validation_seasons=validation_seasons,
    train_start=settings.seasons.train_start,
    min_train_rows=settings.model.min_train_rows,
    target_column="pass_rate",
).select(
    pl.col("player_id").alias("team"), "season", "week", pl.col("prediction").alias("predicted_pass_rate")
)

stage1_predictions = stage1_predicted_team_plays.join(
    stage1_predicted_pass_rate, on=["team", "season", "week"], how="inner"
)
predicted_pass_attempts, predicted_rush_attempts = team_environment.derive_attempts(
    stage1_predictions["predicted_team_plays"], stage1_predictions["predicted_pass_rate"]
)
stage1_predictions = stage1_predictions.with_columns(
    predicted_pass_attempts.alias("predicted_pass_attempts"),
    predicted_rush_attempts.alias("predicted_rush_attempts"),
)

# --- Build Stage 2's own table and baselines. ---
table = opportunity.build_opportunity_table(player_week_features, player_week_usage, stage1_predictions)
table = opportunity.add_opportunity_baselines(table)

for target_column, composition_column in [
    ("targets", "expected_targets"),
    ("carries", "expected_carries"),
    ("rz_touches", "expected_rz_touches"),
]:
    predictions = opportunity.to_predictions_frame(
        table,
        real_column=target_column,
        composition_column=composition_column,
        trailing_raw_column=f"{target_column}_b2_ewm_4",
        league_mean_column=f"{target_column}_league_mean",
    )
    print(f"\n=== {target_column} ===")
    for result in accuracy_metrics(predictions):
        if result.scope == "all" and result.position is None:
            print(
                f"{result.predictor}: {result.metric}={result.value:.4f} "
                f"(n={result.n_obs}, ci=[{result.ci_low:.4f}, {result.ci_high:.4f}])"
            )
```

- [ ] **Step 2: Run it against real cached data**

Run: `uv run python notebooks/evaluate_opportunity_v2_stage2.py`

If it fails because a specific input path doesn't exist locally yet, rebuild it first following `HANDOFF.md` §6's documented sequence — do not guess at a substitute path. This script needs the same real inputs Stage 1's own evaluation already used successfully, plus `data/interim/player_week_usage.parquet` (already required by task 1.6/1.9, should already exist alongside `player_week_features.parquet`).

- [ ] **Step 3: Record the real result**

Read the printed MAE/n_obs/CI values for all three outputs (`targets`, `carries`, `rz_touches`). Add an entry to `docs/JOURNAL.md` §2 (Completed tasks), following this project's existing entry style (see the Stage 1 entries immediately above for the exact voice/format — specific real numbers, not vague, and the honest-reporting discipline those entries already establish), stating:
- The real MAE/n_obs/CI for `opportunity_composition` vs `trailing_raw` (the real bar) vs `league_mean` (sanity floor, not the bar), for all three outputs.
- Whether the composition beat `trailing_raw` on each of the three outputs, stated plainly.
- If it did not beat the baseline on one or more outputs, state that plainly as a real, documented result — same treatment as Stage 1's own outcome and task 1.15's before it. Do not tune anything to force a passing number; there is nothing to tune here (no hyperparameters, no model) — if the result is unfavorable, the honest next step is a design conversation about Stage 2's own approach, not a code change made unilaterally in this task.

Update `HANDOFF.md`'s "Next task" line and current-session narrative to reflect this real outcome, matching the existing style of the Stage 1 entries directly above it.

- [ ] **Step 4: Commit**

```bash
git add notebooks/evaluate_opportunity_v2_stage2.py docs/JOURNAL.md HANDOFF.md
git commit -m "test: real evaluation of model v2 Stage 2 against 2021-2025 data"
```

---

## After this plan

Do not proceed to Stage 3 (efficiency priors) automatically. Report the real result from Task 5 to the project owner and let them decide next steps, same pattern this project has used at every prior real decision point (see `docs/design-model-v2-stage2-opportunity.md`'s own scope section, and Stage 1's own "After this plan" precedent).
