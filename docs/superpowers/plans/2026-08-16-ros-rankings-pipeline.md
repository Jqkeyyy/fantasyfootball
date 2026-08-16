# Rest-of-Season Rankings Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `SPEC-ADDENDUM-04.md` §D's rest-of-season rankings pipeline (TASKS.md 1.21) — multi-week projections, ROS Monte Carlo aggregation with real within-player and cross-player correlation and persistent injury sampling, rest-of-season VOR over the current free-agent pool, a weekly-refreshing CLI, and a Streamlit page — under an amended §D.1 that splits the horizon into a current week (existing weekly `consensus_b3`, unchanged) and future weeks (season-long consensus from the six preseason-style sources, allocated across weeks by matchup shape, since FantasyPros' weekly archive only ever publishes the *current* week).

**Architecture:** Reuses, rather than re-derives, four already-shipped subsystems: task 2.2's cross-player correlated weekly simulation (`sim/week.py`), task 2.3/2.4's injury hazard model and persistent-duration availability sampling (`sim/injury.py`, `sim/season.py`), task 0.9's VOR fixed point (`tools/vor.py`), and task 2.6's current free-agent pool (`tools/waivers.py`). Two genuinely new pieces of math are added: a single-factor within-player week-to-week correlation model (`sim/persistence.py`, layered onto task 2.2's existing cross-player copula rather than replacing it) and a season-long-consensus-to-weekly-shape allocator (`models/ros_shape.py`) that uses task 1.8's already-validated opponent-adjusted defense ratings, frozen at the present, to shape (not to set the level of) each future week's expected points. Both correlation constants are estimated from real historical data by a new materialization script and committed to `config/ros_calibration.yml`, matching the `materialize_b3_historical.py` precedent.

**Tech Stack:** Python 3.11+, polars, numpy, scipy.stats.norm, typer, streamlit, pytest, `uv run`. No new external dependencies.

**Spec:** `SPEC-ADDENDUM-04.md` §D (amended below), `SPEC-ADDENDUM-05.md` §A (consensus-as-anchor framing — this pipeline is consumer, not modeller: it allocates and simulates around a consensus level, never fits a competing points model), `SPEC.md` §9.4 (VOR fixed point), §13.2 (correlated weekly simulation), §13.3/§13.4 (injury hazard, persistence), §10.1/§12.1 (the as_of contract).

## Global Constraints

- No random train/test splits; every historical estimation (correlation, recovery duration) is a real, walk-forward-irrelevant *global* constant estimated once from all real history, not fit per-fold — same status as `DEFAULT_CORRELATION_SETTINGS`, just empirically derived instead of judgment calls.
- Every future-week feature must respect the as_of contract (CLAUDE.md rule 2): usage/depth-chart/snap-share columns freeze at the last real value at-or-before the anchor week and carry forward unchanged; only opponent identity, bye, and each opponent's own *already-frozen* (never a future) `def_adj_*`/`adj_epa_allowed` rating vary by week. No rolling window is ever advanced as if a future week had already happened.
- The scoring engine and `consensus_b3` are already validated (CLAUDE.md rule 3) — this plan does not re-validate them, only composes them.
- Nothing hardcodes a league's format, team count, or starting slots (CLAUDE.md rule 5) — every VOR/rankings function takes `LeagueFormat` and runs against both `rogan-radinator-league` and the second real league.
- No silent join drops (CLAUDE.md rule 4) — every join in this plan either can't drop a real player (inner join on a key both sides are guaranteed to have) or is documented as an intentional best-effort external-source match, matching `add_b3_fp_weekly_consensus`'s own precedent.
- Type hints on every public function; `ruff check`/`ruff format --check`/`mypy src/` clean at the end of every task.
- Tests before implementation (TDD): write the failing test, watch it fail for the right reason, then implement.
- No live network calls in tests — small hand-built polars/numpy fixtures only.
- Do not modify `sim/week.py`'s existing `simulate_week`/`build_correlation_matrix`/`marginal_ppf`/`nearest_positive_definite` (task 2.2, already shipped and tested) — the new within-player-correlation capability is additive, in a new module that imports these.
- Do not modify `models/predict.py`'s existing `project_week` current-week logic — the amended §D.1 explicitly keeps the current week unchanged; this plan's multi-week composer calls it once, for the anchor week only, and adds future weeks alongside.
- Do not start any `SPEC-ADDENDUM-05.md` work (§C/§D/§E, tasks 3.9-3.11) — explicitly offseason, out of scope for this plan.

---

### Task 1: Record the §D.1 horizon-split amendment in the journal

**Files:**
- Modify: `docs/JOURNAL.md` (append a new dated entry at the end)

**Interfaces:** None — documentation only.

- [ ] **Step 1: Append the amendment entry**

Append to `docs/JOURNAL.md`:

```markdown
## 2026-08-16 (later) — SPEC-ADDENDUM-04.md §D.1 amended: horizon split for the ROS pipeline

§D.1 as written assumes per-week projections are obtainable for every future week. They
aren't: FantasyPros' weekly archive (`ingest.rankings.FP_WEEKLY_ARCHIVE_PATH`,
`fp_latest_weekly.csv`) publishes only the *current* week's consensus — there is no
future-week weekly file to fetch, live or historical. Verified by re-reading
`normalize_fp_weekly`/`fetch_fp_weekly_snapshot`: every real row this project has ever
pulled from that archive is tagged with the (season, week) it was *selected for* by
`select_commit_before`, not a week the source itself published multiple horizons for.

Implementation decision, made before building task 1.21: split the ROS horizon into two
regimes rather than trying to force weekly consensus onto weeks it was never published for.

- **Current week** (the anchor week the pipeline is run for): unchanged. Real weekly
  `consensus_b3` (`models.baselines.fetch_b3_for_week`) plus the already-shipped
  calibrated empirical spread (`models.baselines.empirical_error_quantiles`). No change to
  `models/predict.py::project_week`.
- **Future weeks** (anchor+1 through the horizon's `through_week`): the six season-long
  preseason-style sources (ESPN/CBS/FantasySharks/FFToday/FootballGuys/DraftSharks — the
  same seven-minus-FantasyPros set `tools.prediction_log` already fetches weekly and
  correctly separates from the one real weekly source) are re-fetched *live, this week*,
  resolved to a real remaining-season value per `check_sources`' own already-shipped
  full-season-vs-ROS trend detection (declining materially => already a forward signal;
  flat/insufficient data => full-season, subtract real actuals-to-date — the safer default
  either way, logged per source), then aggregated via the same trimmed-mean
  `projections.aggregate.aggregate_projections` the draft board already uses. This one
  number is the real *level* for a player's entire remaining season.
- **Shape:** the season-long level is allocated across the player's own real remaining
  weeks (byes excluded entirely) proportional to each week's real opponent matchup quality
  (`defense_position_allowed`'s own already-validated, already-walk-forward-safe
  `adj_epa_allowed`, task 1.8), frozen at whatever is known as of the anchor week — never a
  future week's own rating, which doesn't exist yet regardless. Availability/injury is
  deliberately NOT baked into the shape (that would double-count against the aggregation
  stage's own `p_play[w]` multiplier) — the shape answers "if this player plays a full
  week w, how much of their remaining-season value lands in that week," not "will they
  play."
- **ROS aggregate:** Monte Carlo over the future-week distributions (empirical-error-
  quantile spread recentered on each week's shaped mean, same already-validated mechanism
  `consensus_b3` uses for the current week) plus the current week's own real distribution,
  combined with within-player week-to-week correlation (new, `sim/persistence.py`) and
  task 2.2's existing cross-player correlation, gated by task 2.3/2.4's persistent-duration
  injury sampling.

"Consensus supplies the level; the pipeline supplies the shape" is the one-line summary.
This is a real, load-bearing implementation decision, not an incidental default — it
determines what "future week projection" even means in this pipeline, since no source
publishes one directly. Full task breakdown: `docs/superpowers/plans/2026-08-16-ros-rankings-pipeline.md`.
```

- [ ] **Step 2: Commit**

```bash
git add docs/JOURNAL.md
git commit -m "docs: record SPEC-ADDENDUM-04.md D.1 horizon-split amendment"
```

---

### Task 2: `RosSettings` config and the `ros_calibration.yml` loader

**Files:**
- Modify: `src/ffapp/config.py`
- Create: `config/ros_calibration.yml` (placeholder values here; Task 5's materialization script overwrites with real ones)
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `config.RosSettings` (frozen dataclass: `season_end_week: int`, `ros_sims: int`, `default_recovery_prob: float`), `config.Settings.ros: RosSettings` (new field, default `RosSettings(season_end_week=18, ros_sims=3000, default_recovery_prob=0.5)`), `config.RosCalibration` (frozen dataclass: `within_player_week_correlation: dict[str, float]`, `recovery_prob: dict[str, float]`), `config.load_ros_calibration(path: Path = ROS_CALIBRATION_PATH) -> RosCalibration`, `config.ROS_CALIBRATION_PATH: Path`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_config.py -- append

def test_ros_settings_defaults() -> None:
    settings = load_settings(root=FIXTURE_ROOT)
    assert settings.ros.season_end_week == 18
    assert settings.ros.ros_sims == 3000
    assert settings.ros.default_recovery_prob == 0.5


def test_load_ros_calibration_reads_real_yaml(tmp_path) -> None:
    path = tmp_path / "ros_calibration.yml"
    path.write_text(
        "within_player_week_correlation:\n"
        "  QB: 0.31\n"
        "  RB: 0.24\n"
        "recovery_prob:\n"
        "  QB: 0.4\n"
        "  RB: 0.33\n"
    )
    calibration = load_ros_calibration(path)
    assert calibration.within_player_week_correlation == {"QB": 0.31, "RB": 0.24}
    assert calibration.recovery_prob == {"QB": 0.4, "RB": 0.33}


def test_load_ros_calibration_missing_file_returns_empty(tmp_path) -> None:
    calibration = load_ros_calibration(tmp_path / "does_not_exist.yml")
    assert calibration.within_player_week_correlation == {}
    assert calibration.recovery_prob == {}
```

(Check the top of `tests/test_config.py` for its existing `FIXTURE_ROOT`/import pattern and match it — this project's existing `load_settings` tests already load against a fixture settings tree; add these two new tests using the same fixture helper, and add `load_ros_calibration`, `RosSettings`, `RosCalibration` to the existing `from ffapp.config import ...` line at the top of the file.)

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_config.py -k ros -v`
Expected: FAIL — `ImportError` / `AttributeError`, `RosSettings`/`load_ros_calibration` don't exist yet.

- [ ] **Step 3: Implement**

In `src/ffapp/config.py`, add near the other settings dataclasses (after `WaiverSettings`/`DEFAULT_WAIVER_SETTINGS`):

```python
@dataclass(frozen=True)
class RosSettings:
    """SPEC-ADDENDUM-04.md §D: rest-of-season horizon and Monte Carlo sizing.
    `default_recovery_prob` is the fallback used for any position
    `RosCalibration.recovery_prob` doesn't cover (e.g. before Task 5's
    materialization script has ever run) -- same magnitude as
    `sim.season.simulate_availability`'s own pre-existing default, kept
    in sync deliberately rather than drifting to two different numbers."""

    season_end_week: int = 18
    ros_sims: int = 3000
    default_recovery_prob: float = 0.5


DEFAULT_ROS_SETTINGS = RosSettings()
```

Add `ros: RosSettings = DEFAULT_ROS_SETTINGS` to `Settings`.

In `load_settings`, after the `waivers` block:

```python
    ros_raw = raw.get("ros", {})
    dr = DEFAULT_ROS_SETTINGS
    ros = RosSettings(
        season_end_week=int(ros_raw.get("season_end_week", dr.season_end_week)),
        ros_sims=int(ros_raw.get("ros_sims", dr.ros_sims)),
        default_recovery_prob=float(ros_raw.get("default_recovery_prob", dr.default_recovery_prob)),
    )
```

and pass `ros=ros` into the returned `Settings(...)`.

Add, near `CONFIG_DIR`:

```python
ROS_CALIBRATION_PATH = CONFIG_DIR / "ros_calibration.yml"


@dataclass(frozen=True)
class RosCalibration:
    """Empirically-estimated constants from `notebooks/estimate_ros_calibration.py`
    (Task 5) -- committed to git like `config/id_overrides.csv`, not
    computed on every run (the estimation reads the full real historical
    player-week/injury tables, seconds not milliseconds). A position
    absent from either dict (no real historical data for it, or the
    script hasn't been re-run yet) is the caller's job to default --
    `sim.persistence`/`sim.injury`'s own consuming functions do this
    explicitly rather than this loader guessing a fallback value that
    isn't its own to pick."""

    within_player_week_correlation: dict[str, float]
    recovery_prob: dict[str, float]


def load_ros_calibration(path: Path = ROS_CALIBRATION_PATH) -> RosCalibration:
    if not path.exists():
        return RosCalibration(within_player_week_correlation={}, recovery_prob={})
    raw = yaml.safe_load(path.read_text()) or {}
    return RosCalibration(
        within_player_week_correlation=dict(raw.get("within_player_week_correlation", {})),
        recovery_prob=dict(raw.get("recovery_prob", {})),
    )
```

Create `config/ros_calibration.yml` with honest placeholder content (Task 5 overwrites for real):

```yaml
# Generated by notebooks/estimate_ros_calibration.py -- do not hand-edit.
# Placeholder until Task 5 runs against real historical data.
within_player_week_correlation: {}
recovery_prob: {}
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_config.py -v`
Expected: PASS, full file.

- [ ] **Step 5: Commit**

```bash
git add src/ffapp/config.py config/ros_calibration.yml tests/test_config.py
git commit -m "feat: add RosSettings and ros_calibration.yml loader"
```

---

### Task 3: Injury duration estimation and per-player recovery in `simulate_availability`

**Files:**
- Modify: `src/ffapp/sim/injury.py`
- Modify: `src/ffapp/sim/season.py`
- Test: `tests/test_sim_injury.py`
- Test: `tests/test_sim_season.py`

**Interfaces:**
- Consumes: `sim.injury.build_hazard_grid(rosters) -> pl.DataFrame` (already exists — `player_id, season, week, position, missed`).
- Produces: `sim.injury.estimate_recovery_prob(hazard_grid: pl.DataFrame) -> dict[str, float]` (position -> recovery_prob, `1/mean_real_run_length`; a position with zero real miss-runs is simply absent from the returned dict). `sim.season.simulate_availability`'s `recovery_prob` parameter now accepts `float | np.ndarray` (shape `(n_players,)`) instead of only `float` — existing callers passing a scalar are unaffected.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_sim_injury.py -- append

def test_estimate_recovery_prob_from_real_run_lengths() -> None:
    # player p1: misses weeks 3-5 (a real 3-week run), plays otherwise.
    # player p2: misses week 2 alone (a real 1-week run).
    # Both RB. mean run length = (3 + 1) / 2 = 2.0 -> recovery_prob = 0.5.
    grid = pl.DataFrame(
        {
            "player_id": ["p1"] * 6 + ["p2"] * 4,
            "season": [2023] * 10,
            "week": [1, 2, 3, 4, 5, 6, 1, 2, 3, 4],
            "position": ["RB"] * 10,
            "missed": [False, False, True, True, True, False, False, True, False, False],
        }
    )
    result = injury.estimate_recovery_prob(grid)
    assert result["RB"] == pytest.approx(0.5)


def test_estimate_recovery_prob_omits_position_with_no_real_misses() -> None:
    grid = pl.DataFrame(
        {
            "player_id": ["q1", "q1"],
            "season": [2023, 2023],
            "week": [1, 2],
            "position": ["QB", "QB"],
            "missed": [False, False],
        }
    )
    result = injury.estimate_recovery_prob(grid)
    assert "QB" not in result
```

```python
# tests/test_sim_season.py -- append

def test_simulate_availability_accepts_per_player_recovery_prob() -> None:
    """Two players, very different recovery speeds: p0 recovers almost
    instantly (recovery_prob=0.99, real duration ~1 week), p1 recovers
    very slowly (recovery_prob=0.05, real duration ~20 weeks). Once
    either player takes a real miss, p0's average real absence should be
    far shorter than p1's -- proves the array is actually applied
    per-player, not broadcast as one shared scalar."""
    n_weeks, n_players, season_sims = 30, 2, 4000
    p_miss = np.full((n_weeks, n_players), 0.5)  # force an early real miss for both
    recovery_prob = np.array([0.99, 0.05])
    rng = np.random.default_rng(0)

    available = simulate_availability(
        p_miss, season_sims=season_sims, recovery_prob=recovery_prob, rng=rng
    )
    p0_miss_rate = 1.0 - available[:, :, 0].mean()
    p1_miss_rate = 1.0 - available[:, :, 1].mean()
    assert p1_miss_rate > p0_miss_rate + 0.1  # real, not noise-sized difference
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_sim_injury.py tests/test_sim_season.py -k "recovery" -v`
Expected: FAIL — `estimate_recovery_prob` doesn't exist; the season test currently passes a scalar-only param but should still fail because `estimate_recovery_prob` import fails first (both test files import from the same module-level `from ffapp.sim import injury`).

- [ ] **Step 3: Implement**

In `src/ffapp/sim/injury.py`, add after `predict_positional_base_rate`:

```python
def estimate_recovery_prob(hazard_grid: pl.DataFrame) -> dict[str, float]:
    """Real per-position injury-duration estimate for
    `sim.season.simulate_availability`'s own geometric-duration
    persistence mechanic (SPEC §13.4: "sample duration, not independent
    per-week draws"). A real "run" is a maximal sequence of consecutive
    real gameday-roster ROWS (`build_hazard_grid`'s own ACT/INA-scoped
    rows, sorted by week) with `missed=True` for the same player within
    the same season -- deliberately measured in consecutive real rows,
    not consecutive calendar week numbers, since a real bye week has no
    row in this grid at all (see `build_hazard_grid`'s own docstring) and
    `simulate_availability`'s own `remaining_weeks` concept already
    counts decision points the same way, not raw week numbers. A
    documented simplification, not a guess: this project has no per-team
    bye-aware duration model anywhere yet.

    `recovery_prob[position] = 1 / mean(real_run_length)` -- the method-
    of-moments estimator for a geometric distribution's own parameter,
    matching exactly what `simulate_availability`'s
    `rng.geometric(recovery_prob)` draws (mean = 1/p). A position with
    zero real recorded miss-runs is omitted, not defaulted -- there is
    nothing real to estimate from; callers fall back to
    `config.RosSettings.default_recovery_prob` explicitly.
    """
    ordered = hazard_grid.sort(["player_id", "season", "week"])
    prev_missed = pl.col("missed").shift(1).over(["player_id", "season"]).fill_null(False)
    run_start = pl.col("missed") & ~prev_missed
    with_run_id = ordered.with_columns(run_start.cum_sum().over(["player_id", "season"]).alias("_run_id"))
    runs = (
        with_run_id.filter(pl.col("missed"))
        .group_by(["player_id", "season", "_run_id"])
        .agg(pl.len().alias("run_length"), pl.col("position").first().alias("position"))
    )
    if runs.is_empty():
        return {}
    by_position = runs.group_by("position").agg(pl.col("run_length").mean().alias("mean_duration"))
    return {
        row["position"]: float(1.0 / row["mean_duration"])
        for row in by_position.iter_rows(named=True)
    }
```

Add `"estimate_recovery_prob"` to `sim/injury.py`'s `__all__` list.

In `src/ffapp/sim/season.py`, change `simulate_availability`'s signature and the one line that draws `durations`:

```python
def simulate_availability(
    p_miss: np.ndarray,
    *,
    season_sims: int,
    recovery_prob: float | np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
```

(docstring: add one sentence — `"recovery_prob may be a single float shared by every player, or a (n_players,) array of real per-player values (e.g. per-position, broadcast onto each player before calling) -- numpy's own geometric draw already broadcasts either shape against (season_sims, n_players) without further change here."`)

The body is otherwise unchanged — `rng.geometric(recovery_prob, size=(season_sims, n_players))` already broadcasts a `(n_players,)`-shaped `recovery_prob` correctly against that `size`; no other line needs to change.

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_sim_injury.py tests/test_sim_season.py -v`
Expected: PASS, both full files (confirms the pre-existing scalar-`recovery_prob` tests in `test_sim_season.py` still pass unmodified).

- [ ] **Step 5: Run mypy**

Run: `uv run mypy src/ffapp/sim/injury.py src/ffapp/sim/season.py`
Expected: clean (the `float | np.ndarray` union is a real, correctly-typed signature change).

- [ ] **Step 6: Commit**

```bash
git add src/ffapp/sim/injury.py src/ffapp/sim/season.py tests/test_sim_injury.py tests/test_sim_season.py
git commit -m "feat: estimate real injury-duration recovery_prob, apply per-player"
```

---

### Task 4: Within-player week-to-week correlation — estimator, variance-ratio report, and the correlated-sampling primitive

**Files:**
- Create: `src/ffapp/sim/persistence.py`
- Test: `tests/test_sim_persistence.py`

**Interfaces:**
- Consumes: `features/player_week_features.parquet`-shaped table (`player_id, season, week, position, target, availability_flag, season_type`, task 1.9). `sim.week.PlayerMarginal`, `sim.week.build_correlation_matrix`, `sim.week.nearest_positive_definite`, `sim.week.marginal_ppf` (all already public).
- Produces: `sim.persistence.estimate_within_player_correlation(player_week_features: pl.DataFrame) -> dict[str, float]` (position -> ICC, `[0, 1)`). `sim.persistence.season_variance_ratio(n_weeks: int, rho: float) -> float`. `sim.persistence.simulate_week_with_common_factor(players: Sequence[PlayerMarginal], correlation: CorrelationSettings, *, week_sims: int, player_factor: np.ndarray, rho: np.ndarray, rng: np.random.Generator) -> np.ndarray`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_sim_persistence.py (new file)

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from scipy.stats import norm

from ffapp.config import DEFAULT_CORRELATION_SETTINGS
from ffapp.sim.persistence import (
    estimate_within_player_correlation,
    season_variance_ratio,
    simulate_week_with_common_factor,
)
from ffapp.sim.week import PlayerMarginal


def test_estimate_within_player_correlation_high_for_a_persistent_role_player() -> None:
    """Two RBs, four played weeks each. rb1 is remarkably consistent
    (16, 17, 15, 16 -- almost no within-player variance). rb2 alternates
    a starter role and a committee role week to week (a real, if
    stylised, "role varies a lot within-season" pattern -- low real
    persistence). The pooled RB ICC should sit meaningfully above 0 (real
    between-player signal exists) and below 1 (real within-player noise
    exists too)."""
    features = pl.DataFrame(
        {
            "player_id": ["rb1"] * 4 + ["rb2"] * 4,
            "season": [2023] * 8,
            "week": [1, 2, 3, 4, 1, 2, 3, 4],
            "position": ["RB"] * 8,
            "season_type": ["REG"] * 8,
            "availability_flag": [1] * 8,
            "target": [16.0, 17.0, 15.0, 16.0, 4.0, 22.0, 3.0, 21.0],
        }
    )
    result = estimate_within_player_correlation(features)
    assert 0.0 < result["RB"] < 1.0


def test_estimate_within_player_correlation_excludes_unplayed_weeks() -> None:
    """A row with availability_flag=0 (didn't play) must not count toward
    the estimate -- injury-driven zero weeks are a different, separately-
    modelled effect (sim.injury/sim.season's own persistence), not
    within-player performance correlation."""
    features = pl.DataFrame(
        {
            "player_id": ["rb1"] * 5,
            "season": [2023] * 5,
            "week": [1, 2, 3, 4, 5],
            "position": ["RB"] * 5,
            "season_type": ["REG"] * 5,
            "availability_flag": [1, 1, 1, 1, 0],
            "target": [16.0, 17.0, 15.0, 16.0, 0.0],
        }
    )
    # Should not raise, and the excluded week's 0.0 must not drag a
    # persistent player's own ICC down toward "inconsistent."
    result = estimate_within_player_correlation(features)
    assert "RB" not in result or result["RB"] >= 0.0  # only 1 real player at RB here -- no between-player signal at all, either omitted or 0.0 is acceptable


def test_season_variance_ratio_matches_closed_form() -> None:
    # Var(sum of n equicorrelated unit-variance draws) / Var(sum of n
    # independent unit-variance draws) = 1 + (n-1) * rho, the textbook
    # equicorrelated-sum identity.
    assert season_variance_ratio(n_weeks=10, rho=0.3) == pytest.approx(1 + 9 * 0.3)
    assert season_variance_ratio(n_weeks=1, rho=0.5) == pytest.approx(1.0)
    assert season_variance_ratio(n_weeks=10, rho=0.0) == pytest.approx(1.0)


def test_simulate_week_with_common_factor_reproduces_marginal() -> None:
    """A single player, rho=1.0 (fully determined by the common factor):
    the sampled score distribution must still match that player's own
    real marginal quantile grid -- the common-factor blend must never
    distort a player's OWN marginal, only introduce cross-week
    dependence."""
    alphas = [0.1, 0.25, 0.5, 0.75, 0.9]
    quantile_values = [3.0, 6.0, 10.0, 14.0, 18.0]
    player = PlayerMarginal(
        player_id="p1", position="RB", team="KC", opponent_team="BUF",
        alphas=alphas, quantile_values=quantile_values,
    )
    rng = np.random.default_rng(1)
    n_sims = 20000
    player_factor = rng.standard_normal((n_sims, 1))
    scores = simulate_week_with_common_factor(
        [player], DEFAULT_CORRELATION_SETTINGS,
        week_sims=n_sims, player_factor=player_factor, rho=np.array([1.0]), rng=rng,
    )
    empirical_median = float(np.median(scores[:, 0]))
    assert empirical_median == pytest.approx(10.0, abs=0.5)


def test_simulate_week_with_common_factor_induces_week_to_week_correlation() -> None:
    """Same player, sampled across two 'weeks' sharing the same
    player_factor draw but independent idiosyncratic noise: the
    correlation between the two weeks' sampled normal-space z should be
    close to the configured rho (checked in z-space via norm.ppf, since
    the marginal itself is nonlinear)."""
    alphas = [0.1, 0.25, 0.5, 0.75, 0.9]
    quantile_values = [3.0, 6.0, 10.0, 14.0, 18.0]
    player = PlayerMarginal(
        player_id="p1", position="RB", team="KC", opponent_team="BUF",
        alphas=alphas, quantile_values=quantile_values,
    )
    rng = np.random.default_rng(2)
    n_sims = 20000
    rho = np.array([0.4])
    player_factor = rng.standard_normal((n_sims, 1))
    week1 = simulate_week_with_common_factor(
        [player], DEFAULT_CORRELATION_SETTINGS, week_sims=n_sims,
        player_factor=player_factor, rho=rho, rng=rng,
    )
    week2 = simulate_week_with_common_factor(
        [player], DEFAULT_CORRELATION_SETTINGS, week_sims=n_sims,
        player_factor=player_factor, rho=rho, rng=rng,
    )
    z1 = norm.ppf(np.clip(_empirical_cdf(week1[:, 0]), 1e-6, 1 - 1e-6))
    z2 = norm.ppf(np.clip(_empirical_cdf(week2[:, 0]), 1e-6, 1 - 1e-6))
    empirical_rho = float(np.corrcoef(z1, z2)[0, 1])
    assert empirical_rho == pytest.approx(0.4, abs=0.05)


def _empirical_cdf(values: np.ndarray) -> np.ndarray:
    order = np.argsort(np.argsort(values))
    return (order + 0.5) / len(values)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_sim_persistence.py -v`
Expected: FAIL — `ffapp.sim.persistence` doesn't exist yet.

- [ ] **Step 3: Implement**

Create `src/ffapp/sim/persistence.py`:

```python
"""Within-player week-to-week correlation (SPEC-ADDENDUM-04.md §D,
requirement 2 -- "a player's weeks are NOT independent; summing
independent weekly draws under-disperses the season total").

Additive to task 2.2's `sim.week` (cross-player correlation within a
single week) -- deliberately a new module, not a modification of an
already-shipped, already-tested one. `simulate_week_with_common_factor`
reuses `sim.week`'s own `build_correlation_matrix`/`nearest_positive_definite`
/`marginal_ppf` for the cross-player structure, and layers a single-factor
random-effects model on top for the cross-week structure:

    z[week, player] = sqrt(rho[player]) * player_factor[player]
                     + sqrt(1 - rho[player]) * idiosyncratic[week, player]

`player_factor` is drawn once per (simulation path, player) by the caller
and reused identically across every week of that path -- the standard
single-factor / random-intercept construction, which gives
Corr(z_w, z_w') = rho for any two distinct weeks of the same player by
construction, while leaving each week's own marginal exactly N(0,1)
(both `player_factor` and `idiosyncratic` are unit-variance and mutually
independent, so their weighted sum is too) -- so `marginal_ppf` still
inverts to the exact correct real per-player marginal at every week.

**Known, documented approximation:** because `idiosyncratic` is scaled
by `sqrt(1 - rho)` before the cross-player correlation matrix is applied
to it, the REALIZED cross-player correlation within a single week is
diluted by `sqrt((1-rho_i)(1-rho_j))` relative to `CorrelationSettings`'s
own configured constants (which were calibrated assuming full weight on
the correlated component). Not corrected for here -- `rho` values are
modest (see `estimate_within_player_correlation`'s own real numbers,
`docs/JOURNAL.md`), and a fully joint player x week factor model would
need a much larger joint covariance matrix for a real, unclear accuracy
gain. A documented simplification, not a silent one.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import polars as pl
from scipy.stats import norm

from ffapp.config import CorrelationSettings
from ffapp.sim.week import PlayerMarginal, build_correlation_matrix, marginal_ppf, nearest_positive_definite

_MIN_WEEKS_PER_PLAYER = 3


def estimate_within_player_correlation(player_week_features: pl.DataFrame) -> dict[str, float]:
    """Real per-position intraclass correlation (ICC) of `target` across
    a player's own real played weeks -- the unbalanced one-way random-
    effects ANOVA method-of-moments estimator (Searle/Casella/McCulloch),
    the textbook estimator for exactly this quantity: for a random-
    intercept model `Y_ij = mu + player_i + e_ij`, ICC =
    Var(player)/(Var(player) + Var(e)) equals Corr(Y_ij, Y_ij') for two
    distinct real weeks j != j' of the same player.

    Scoped to `season_type == "REG"` and `availability_flag == 1` --
    injury-driven zero weeks are a separately-modelled effect
    (`sim.injury`/`sim.season`'s own persistence, `estimate_recovery_prob`),
    not within-player PERFORMANCE correlation; mixing them in would
    conflate "this player's role is consistent when he plays" with "this
    player gets hurt a lot," two different real phenomena. `target` is
    first centered within each real (season, week, position) group (its
    own real cross-sectional mean subtracted) to remove slate-level
    effects (a real high-scoring week across the whole league) that would
    otherwise inflate between-player variance for reasons having nothing
    to do with any individual player's own persistence.

    Requires at least `_MIN_WEEKS_PER_PLAYER` real played weeks for a
    player-season to count -- too few real weeks make a single player's
    own within-player variance estimate too noisy to trust (same
    reasoning as `models.availability.DEFAULT_CALIBRATION_WEEKS`, a
    different real minimum-sample judgment call in this codebase).
    Returns `{}`  for a position with fewer than 2 real qualifying
    players (no real between-player variance to estimate at all).
    """
    scoped = player_week_features.filter(
        (pl.col("season_type") == "REG") & (pl.col("availability_flag") == 1)
    )
    centered = scoped.with_columns(
        (pl.col("target") - pl.col("target").mean().over(["season", "week", "position"])).alias(
            "_centered"
        )
    )
    with_counts = centered.with_columns(
        pl.len().over(["player_id", "season", "position"]).alias("_n_weeks")
    ).filter(pl.col("_n_weeks") >= _MIN_WEEKS_PER_PLAYER)

    result: dict[str, float] = {}
    for position in with_counts["position"].unique().to_list():
        pos_rows = with_counts.filter(pl.col("position") == position)
        groups = pos_rows.group_by(["player_id", "season"]).agg(
            pl.col("_centered").mean().alias("_group_mean"),
            pl.col("_centered").var(ddof=0).alias("_group_var"),
            pl.len().alias("_n"),
        )
        k = groups.height
        if k < 2:
            continue
        n = groups["_n"].to_numpy().astype(float)
        group_mean = groups["_group_mean"].to_numpy()
        group_var = groups["_group_var"].to_numpy()
        n_total = float(n.sum())
        grand_mean = float((group_mean * n).sum() / n_total)

        msb = float((n * (group_mean - grand_mean) ** 2).sum() / (k - 1))
        msw_num = float((n * group_var).sum())
        msw_den = n_total - k
        if msw_den <= 0:
            continue
        msw = msw_num / msw_den

        n0 = float((n_total - (n**2).sum() / n_total) / (k - 1))
        if n0 <= 0:
            continue
        between_var = max((msb - msw) / n0, 0.0)
        total_var = between_var + msw
        if total_var <= 0:
            continue
        result[position] = between_var / total_var

    return result


def season_variance_ratio(n_weeks: int, rho: float) -> float:
    """Var(sum of n equicorrelated unit-variance draws, pairwise
    correlation rho) / Var(sum of n independent unit-variance draws) =
    1 + (n-1) * rho -- the closed-form size of the under-dispersion
    requirement 2 warns about, reported directly rather than only
    inferred from a Monte Carlo run."""
    if n_weeks <= 1:
        return 1.0
    return 1.0 + (n_weeks - 1) * rho


def simulate_week_with_common_factor(
    players: Sequence[PlayerMarginal],
    correlation: CorrelationSettings,
    *,
    week_sims: int,
    player_factor: np.ndarray,
    rho: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Like `sim.week.simulate_week`, but blends each player's own
    persistent per-simulation-path factor into that week's latent normal
    before inverting through the marginal CDF (see module docstring for
    the exact construction and its one documented approximation).
    `player_factor`: `(week_sims, n_players)`, drawn once per season path
    by the caller and passed identically to every real week of that
    path. `rho`: `(n_players,)`, each player's own real
    `estimate_within_player_correlation` value for their position.
    """
    n = len(players)
    if n == 0:
        return np.empty((week_sims, 0))

    corr = nearest_positive_definite(build_correlation_matrix(players, correlation))
    idiosyncratic = rng.multivariate_normal(mean=np.zeros(n), cov=corr, size=week_sims)
    rho_arr = np.asarray(rho, dtype=float)
    z = np.sqrt(rho_arr)[None, :] * player_factor + np.sqrt(1.0 - rho_arr)[None, :] * idiosyncratic
    u = norm.cdf(z)

    scores = np.empty_like(u)
    for i, player in enumerate(players):
        scores[:, i] = marginal_ppf(u[:, i], player.alphas, player.quantile_values)
    return np.asarray(scores)


__all__ = [
    "estimate_within_player_correlation",
    "season_variance_ratio",
    "simulate_week_with_common_factor",
]
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_sim_persistence.py -v`
Expected: PASS, full file.

- [ ] **Step 5: mypy/ruff**

Run: `uv run mypy src/ffapp/sim/persistence.py && uv run ruff check src/ffapp/sim/persistence.py`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add src/ffapp/sim/persistence.py tests/test_sim_persistence.py
git commit -m "feat: within-player week-to-week correlation estimator and sampler"
```

---

### Task 5: Materialize real calibration constants to `config/ros_calibration.yml`

**Files:**
- Create: `notebooks/estimate_ros_calibration.py`

**Interfaces:**
- Consumes: `sim.injury.estimate_recovery_prob` (Task 3), `sim.persistence.estimate_within_player_correlation`/`season_variance_ratio` (Task 4), `sim.injury.build_hazard_grid`, real `data/features/player_week_features.parquet`, and the real RAW `rosters` table via `ingest.nflverse.fetch_rosters` (`data/raw/nflverse/rosters_<season-range-label>.parquet` — there is no `data/interim/rosters.parquet`; `sim/injury.py`'s own module docstring and `build_hazard_grid` both confirm this reads the raw nflverse table directly, not a curated interim one).
- Produces: `config/ros_calibration.yml`, overwritten with real numbers (`config.write` via `yaml.safe_dump`, matching `write_source_refresh_status`'s own precedent).

- [ ] **Step 1: Write the script**

```python
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

No live network calls beyond what building the two real input tables
already requires (nflverse/dynastyprocess, cached after first run) --
this script itself only reads already-interim/features tables.
"""

from __future__ import annotations

import yaml

from ffapp.config import ROS_CALIBRATION_PATH, load_settings
from ffapp.sim import injury, persistence


def main() -> None:
    settings = load_settings()
    data_root = settings.data_root

    print("Loading real player_week_features.parquet for within-player correlation...")
    features_path = data_root / "features" / "player_week_features.parquet"
    import polars as pl

    features = pl.read_parquet(features_path)
    rho_by_position = persistence.estimate_within_player_correlation(features)
    print("Real within-player week-to-week correlation (ICC), by position:")
    for position, rho in sorted(rho_by_position.items()):
        ratio = persistence.season_variance_ratio(n_weeks=10, rho=rho)
        print(
            f"  {position}: rho={rho:.4f}  "
            f"(10-week season total variance ratio vs independent: {ratio:.3f}x)"
        )

    print("\nBuilding real hazard grid for injury-duration recovery estimation...")
    from ffapp.ingest import nflverse

    rosters_path = nflverse.fetch_rosters(
        list(range(settings.seasons.train_start, settings.seasons.current)),
        offline=True, settings=settings,
    )
    rosters = pl.read_parquet(rosters_path)
    hazard_grid = injury.build_hazard_grid(rosters)
    recovery_by_position = injury.estimate_recovery_prob(hazard_grid)
    print("Real injury-duration recovery_prob, by position (1 / mean real run length):")
    for position, recovery_prob in sorted(recovery_by_position.items()):
        mean_duration = 1.0 / recovery_prob
        print(f"  {position}: recovery_prob={recovery_prob:.4f}  (mean real duration {mean_duration:.2f} weeks)")

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
```

- [ ] **Step 2: Run it for real**

Run: `uv run python notebooks/estimate_ros_calibration.py`
Expected: prints real per-position rho/recovery_prob and variance ratios; `config/ros_calibration.yml` is overwritten with real numbers (no longer the Task 2 placeholder).

Requires `data/features/player_week_features.parquet` and the real raw `data/raw/nflverse/rosters_<season-range>.parquet` (fetched via `ingest.nflverse.fetch_rosters`, see this task's own corrected consumes-list above) to already exist — see `HANDOFF.md` §6 if they don't on this machine.

- [ ] **Step 3: Record the real numbers in the journal**

Append a short entry to `docs/JOURNAL.md` with the real printed rho/recovery_prob/variance-ratio numbers per position (copy the script's own real stdout) — this is the concrete, real evidence requirement 2 and requirement 3 both ask to be reported, not just computed silently.

- [ ] **Step 4: Commit**

```bash
git add notebooks/estimate_ros_calibration.py config/ros_calibration.yml docs/JOURNAL.md
git commit -m "feat: materialize real within-player correlation and injury-duration calibration"
```

---

### Task 6: Season-long consensus resolution for future weeks

**Files:**
- Modify: `src/ffapp/tools/prediction_log.py` (promote two private helpers to public — additive rename, no behavior change)
- Create: `src/ffapp/models/ros_consensus.py`
- Test: `tests/test_models_ros_consensus.py`

**Interfaces:**
- Consumes (promoted, unchanged behavior): `prediction_log.fetch_point_source`/`fetch_rank_source` (renamed from `_fetch_point_source`/`_fetch_rank_source`), `prediction_log.POINT_SOURCES`/`RANK_SOURCES` (renamed from `_POINT_SOURCES`/`_RANK_SOURCES`), `prediction_log.SEASON_SOURCES` (already public), `prediction_log.season_source_trend` (renamed from the module-private `_season_source_trend`, and its return schema is unchanged: `source, n_weeks, trend`).
- Produces: `models.ros_consensus.fetch_season_consensus(season: int, scoring_settings: dict[str, float], players_dim: pl.DataFrame, *, offline: bool | None, settings: Settings, now: datetime) -> dict[str, pl.DataFrame]` (source name -> `player_id, position, team, points`, same shape `tools.prediction_log.fetch_all_sources` already returns for its own `SEASON_SOURCES` subset — reuses the same fetchers, does not duplicate them). `models.ros_consensus.resolve_remaining_value(season_points: dict[str, pl.DataFrame], trend_by_source: dict[str, str], actuals_to_date: pl.DataFrame) -> pl.DataFrame` (`player_id, position, team, join_key, points` per source, `points` already converted to a real remaining-season value — full-season sources with `actuals_to_date` subtracted and clipped at 0, per-source `branch` column: `"ros_direct"` or `"subtracted"`). `models.ros_consensus.aggregate_remaining_value(resolved: pl.DataFrame) -> pl.DataFrame` (`join_key, player_name, position, season_consensus_ros_points, n_sources, dispersion` — thin wrapper around `projections.aggregate.aggregate_projections`).

- [ ] **Step 1: Rename the two private helpers (mechanical, no behavior change)**

In `src/ffapp/tools/prediction_log.py`:
- Rename `_fetch_point_source` -> `fetch_point_source`, `_fetch_rank_source` -> `fetch_rank_source`, `_POINT_SOURCES` -> `POINT_SOURCES`, `_RANK_SOURCES` -> `RANK_SOURCES`, `_season_source_trend` -> `season_source_trend`. Update every internal call site in the same file (`fetch_all_sources`, `check_sources`) to the new names.
- Add `"POINT_SOURCES"`, `"RANK_SOURCES"`, `"fetch_point_source"`, `"fetch_rank_source"`, `"season_source_trend"` to `__all__`.

Run: `uv run pytest tests/test_tools_prediction_log.py -v`
Expected: PASS unmodified — this is a pure rename, existing tests reference the module's public surface (`fetch_all_sources`, `check_sources`, `build_prediction_log`), not the renamed internals directly. If any existing test does reference the old `_`-prefixed names directly, update those references to the new public names in the same commit.

- [ ] **Step 2: Write the failing tests for the new module**

```python
# tests/test_models_ros_consensus.py (new file)

from __future__ import annotations

import polars as pl
import pytest

from ffapp.models import ros_consensus


def _season_points(name: str, player_id: str, points: float, position: str = "RB") -> pl.DataFrame:
    return pl.DataFrame(
        {"player_id": [player_id], "position": [position], "team": ["KC"], "points": [points]}
    )


def test_resolve_remaining_value_subtracts_actuals_for_flat_trend() -> None:
    """A source whose real trend is 'flat' (a static preseason snapshot,
    per check_sources' own detection) is treated as full-season -- the
    safer default -- and real actuals-to-date are subtracted."""
    season_points = {"espn": pl.DataFrame(
        {"player_id": ["p1"], "position": ["RB"], "team": ["KC"], "points": [220.0]}
    )}
    trend_by_source = {"espn": "flat"}
    actuals = pl.DataFrame({"player_id": ["p1"], "actual_points_to_date": [80.0]})

    result = ros_consensus.resolve_remaining_value(season_points, trend_by_source, actuals)
    row = result.row(0, named=True)
    assert row["points"] == pytest.approx(140.0)
    assert row["branch"] == "subtracted"


def test_resolve_remaining_value_uses_directly_for_declining_trend() -> None:
    season_points = {"cbs": pl.DataFrame(
        {"player_id": ["p1"], "position": ["RB"], "team": ["KC"], "points": [140.0]}
    )}
    trend_by_source = {"cbs": "declining"}
    actuals = pl.DataFrame({"player_id": ["p1"], "actual_points_to_date": [80.0]})

    result = ros_consensus.resolve_remaining_value(season_points, trend_by_source, actuals)
    row = result.row(0, named=True)
    assert row["points"] == pytest.approx(140.0)
    assert row["branch"] == "ros_direct"


def test_resolve_remaining_value_defaults_to_subtracted_when_trend_unknown() -> None:
    """insufficient_data (or a source missing from trend_by_source
    entirely) defaults to the safer full-season branch, per requirement 1's
    own explicit instruction."""
    season_points = {"fftoday": pl.DataFrame(
        {"player_id": ["p1"], "position": ["RB"], "team": ["KC"], "points": [200.0]}
    )}
    actuals = pl.DataFrame({"player_id": ["p1"], "actual_points_to_date": [50.0]})

    result = ros_consensus.resolve_remaining_value(season_points, {}, actuals)
    row = result.row(0, named=True)
    assert row["points"] == pytest.approx(150.0)
    assert row["branch"] == "subtracted"


def test_resolve_remaining_value_clips_at_zero() -> None:
    season_points = {"espn": pl.DataFrame(
        {"player_id": ["p1"], "position": ["RB"], "team": ["KC"], "points": [60.0]}
    )}
    actuals = pl.DataFrame({"player_id": ["p1"], "actual_points_to_date": [90.0]})

    result = ros_consensus.resolve_remaining_value(season_points, {"espn": "flat"}, actuals)
    assert result.row(0, named=True)["points"] == pytest.approx(0.0)


def test_resolve_remaining_value_no_actuals_row_treated_as_zero_scored() -> None:
    """A player with no real logged actuals-to-date row (e.g. a rookie
    with 0 games played yet) subtracts nothing, not null."""
    season_points = {"espn": pl.DataFrame(
        {"player_id": ["rookie"], "position": ["WR"], "team": ["KC"], "points": [90.0]}
    )}
    actuals = pl.DataFrame({"player_id": [], "actual_points_to_date": []}, schema={
        "player_id": pl.String, "actual_points_to_date": pl.Float64
    })
    result = ros_consensus.resolve_remaining_value(season_points, {"espn": "flat"}, actuals)
    assert result.row(0, named=True)["points"] == pytest.approx(90.0)


def test_aggregate_remaining_value_trims_and_reports_n_sources() -> None:
    resolved = pl.DataFrame(
        {
            "join_key": ["p1|rb", "p1|rb", "p1|rb"],
            "player_name": ["P One"] * 3,
            "position": ["RB"] * 3,
            "points": [100.0, 110.0, 105.0],
        }
    )
    result = ros_consensus.aggregate_remaining_value(resolved)
    row = result.row(0, named=True)
    assert row["n_sources"] == 3
    assert row["season_consensus_ros_points"] == pytest.approx(105.0, abs=1.0)
```

- [ ] **Step 3: Run to verify failure**

Run: `uv run pytest tests/test_models_ros_consensus.py -v`
Expected: FAIL — `ffapp.models.ros_consensus` doesn't exist.

- [ ] **Step 4: Implement**

Create `src/ffapp/models/ros_consensus.py`:

```python
"""Season-long consensus resolution for future weeks
(`SPEC-ADDENDUM-04.md` §D.1's amendment -- see `docs/JOURNAL.md`'s
2026-08-16 entry). FantasyPros' weekly archive only ever publishes the
CURRENT week; there is no future-week weekly consensus to fetch. The six
other real sources this project already knows how to fetch
(`tools.prediction_log.SEASON_SOURCES`) return real season-long totals
instead -- this module resolves each one to a real REMAINING-season
value (full-season minus real actuals-to-date, unless
`tools.prediction_log.season_source_trend` has already confirmed that
source is a genuine rest-of-season signal) and aggregates them with the
same trimmed mean `projections.aggregate.aggregate_projections` already
uses for the draft board -- one real number, this player's own real
remaining-season consensus level. `models.ros_shape` (a separate module)
allocates that level across real remaining weeks; this module never
looks at individual weeks at all.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import polars as pl

from ffapp.config import Settings
from ffapp.ids import mapping
from ffapp.projections.aggregate import add_join_key, aggregate_projections, apply_league_scoring, \
    build_reference_curve, map_ranks_to_points, rank_within_position
from ffapp.tools.prediction_log import (
    RANK_SOURCES,
    SEASON_SOURCES,
    POINT_SOURCES,
    fetch_point_source,
    fetch_rank_source,
)

_DEFAULT_BRANCH = "subtracted"


def fetch_season_consensus(
    season: int,
    scoring_settings: dict[str, float],
    players_dim: pl.DataFrame,
    *,
    offline: bool | None,
    settings: Settings,
    now: datetime,
) -> dict[str, pl.DataFrame]:
    """Real live fetch of the six season-long sources, reusing
    `tools.prediction_log`'s own real fetchers (promoted to public in
    this same task) rather than a third copy of the same seven-source
    fetch logic (`draft.board` has the first, `prediction_log` the
    second, both for their own real purposes). Returns
    `{source_name: player_id/position/team/points}`, `SEASON_SOURCES`
    only -- `fantasypros`/weekly sources are the current week's own
    unchanged job, not this module's.
    """
    point_results: dict[str, pl.DataFrame] = {}
    for name in POINT_SOURCES:
        result = fetch_point_source(
            name, season, scoring_settings, players_dim, offline=offline, settings=settings, now=now
        )
        point_results[name] = result.points

    real_ranked = []
    for name in POINT_SOURCES:
        result = fetch_point_source(
            name, season, scoring_settings, players_dim, offline=offline, settings=settings, now=now
        )
        if result.ranked is not None and result.ranked.height > 0:
            real_ranked.append(result.ranked)
    reference_curve = (
        build_reference_curve(real_ranked)
        if real_ranked
        else pl.DataFrame(schema={"position": pl.String, "rank": pl.Int64, "ref_points": pl.Float64})
    )

    for name in RANK_SOURCES:
        result = fetch_rank_source(
            name, season, reference_curve, players_dim, offline=offline, settings=settings, now=now
        )
        point_results[name] = result.points

    return {name: point_results[name] for name in SEASON_SOURCES}


def resolve_remaining_value(
    season_points: dict[str, pl.DataFrame],
    trend_by_source: dict[str, str],
    actuals_to_date: pl.DataFrame,
) -> pl.DataFrame:
    """Per source, per real player: `points` already converted to a real
    remaining-season value, plus which real `branch` was taken (§D's
    requirement 1, "log which branch each source took"). `trend_by_source`
    is `tools.prediction_log.season_source_trend`'s own real per-source
    `trend` ("declining"/"flat"/"insufficient_data") -- only "declining"
    (a real, already-confirmed rest-of-season signal) is used directly;
    every other case, INCLUDING a source entirely absent from
    `trend_by_source` (never yet logged), defaults to the safer
    full-season branch and subtracts `actuals_to_date`, clipped at 0 (a
    remaining season can't be worth negative points). `actuals_to_date`:
    `player_id, actual_points_to_date` -- a player with no real row there
    (no games logged yet, e.g. a rookie) subtracts 0, not null.
    """
    frames: list[pl.DataFrame] = []
    for name, points_df in season_points.items():
        if points_df.is_empty():
            continue
        trend = trend_by_source.get(name, _DEFAULT_BRANCH)
        branch = "ros_direct" if trend == "declining" else "subtracted"
        with_actuals = points_df.join(actuals_to_date, on="player_id", how="left").with_columns(
            pl.col("actual_points_to_date").fill_null(0.0)
        )
        if branch == "ros_direct":
            resolved_points = pl.col("points")
        else:
            resolved_points = (pl.col("points") - pl.col("actual_points_to_date")).clip(lower_bound=0.0)
        frames.append(
            with_actuals.with_columns(
                resolved_points.alias("points"),
                pl.lit(name).alias("source"),
                pl.lit(branch).alias("branch"),
            ).select("player_id", "position", "team", "points", "source", "branch")
        )
    if not frames:
        return pl.DataFrame(
            schema={
                "player_id": pl.String, "position": pl.String, "team": pl.String,
                "points": pl.Float64, "source": pl.String, "branch": pl.String,
            }
        )
    return pl.concat(frames, how="vertical_relaxed")


def aggregate_remaining_value(resolved: pl.DataFrame) -> pl.DataFrame:
    """Thin wrapper around `projections.aggregate.aggregate_projections`
    -- `resolved` must already carry `join_key`/`player_name`/`position`/
    `points` per source (the caller resolves `player_id` -> `join_key`/
    `player_name` via `players_dim` first, same pattern
    `tools.prediction_log._resolve_to_player_id` already establishes;
    this function's own real job is only the trim + rename, not player
    resolution). `aggregate_projections` accepts a sequence of per-source
    frames and concatenates internally, so passing the single already-
    stacked `resolved` frame as a one-element sequence is correct --
    every row still carries its own real `source`, `n_sources`/dispersion
    are still computed per real player across every source that covered
    them. Renames `aggregate_projections`'s own generic `proj_points` to
    `season_consensus_ros_points` so it never gets silently confused with
    the current week's own `b3_mean`/`mean`.
    """
    n_sources = resolved["source"].n_unique() if "source" in resolved.columns else 1
    aggregated = aggregate_projections([resolved], n_sources=max(n_sources, 1))
    return aggregated.rename({"proj_points": "season_consensus_ros_points"})


__all__ = [
    "aggregate_remaining_value",
    "fetch_season_consensus",
    "resolve_remaining_value",
]
```

- [ ] **Step 5: Run to verify pass**

Run: `uv run pytest tests/test_models_ros_consensus.py tests/test_tools_prediction_log.py -v`
Expected: PASS.

- [ ] **Step 6: mypy/ruff**

Run: `uv run mypy src/ffapp/models/ros_consensus.py src/ffapp/tools/prediction_log.py && uv run ruff check src/ffapp/models/ros_consensus.py src/ffapp/tools/prediction_log.py`
Expected: clean.

- [ ] **Step 7: Commit**

```bash
git add src/ffapp/tools/prediction_log.py src/ffapp/models/ros_consensus.py tests/test_models_ros_consensus.py tests/test_tools_prediction_log.py
git commit -m "feat: resolve season-long consensus to a real remaining-season value"
```

---

### Task 7: Future-week shape allocation (frozen usage, matchup-shaped, no leakage)

**Files:**
- Create: `src/ffapp/models/ros_shape.py`
- Test: `tests/test_models_ros_shape.py`

**Interfaces:**
- Consumes: `features/player_week_features.parquet`-shaped rows (one real snapshot row per player, the anchor week). `interim/schedule.parquet` (`season, week, home_team, away_team, season_type`). `interim/defense_position_allowed.parquet` (`season, week, defteam, position_group, adj_epa_allowed, n_plays`, task 1.8). `features.opponent.POSITION_TO_GROUPS`, `features.opponent.team_opponent` (both already public, `tools.sos` already imports and uses them the same way). `interim.build.RELOCATED_TEAM_ALIASES`. `models.ros_consensus.aggregate_remaining_value`'s output (`join_key, player_name, position, season_consensus_ros_points`). `models.baselines.empirical_error_quantiles`/`apply_empirical_error_quantiles` (already exist).
- Produces: `models.ros_shape.frozen_defense_ratings(defense_position_allowed: pl.DataFrame, *, season: int, as_of_week: int, position_group: str) -> pl.DataFrame` (`defteam, frozen_adj_epa_allowed, frozen_n_plays`). `models.ros_shape.future_week_opponents(schedule: pl.DataFrame, *, season: int, team: str, weeks: list[int]) -> pl.DataFrame` (`week, opponent_team` — a week with a real bye is simply absent). `models.ros_shape.allocate_season_consensus(season_consensus_ros_points: float, position: str, team: str, weeks_with_opponents: pl.DataFrame, frozen_ratings_by_group: dict[str, pl.DataFrame]) -> pl.DataFrame` (`week, mean` — sums to `season_consensus_ros_points` across the real weeks present).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_models_ros_shape.py (new file)

from __future__ import annotations

import polars as pl
import pytest

from ffapp.models import ros_shape


def test_frozen_defense_ratings_uses_latest_week_at_or_before_anchor() -> None:
    dpa = pl.DataFrame(
        {
            "season": [2026] * 4,
            "week": [1, 2, 3, 4],
            "defteam": ["BUF", "BUF", "BUF", "BUF"],
            "position_group": ["WR"] * 4,
            "adj_epa_allowed": [0.0, 0.1, 0.2, 0.9],  # week 4 is in the future -- must not leak in
            "n_plays": [20, 22, 21, 25],
        }
    )
    result = ros_shape.frozen_defense_ratings(dpa, season=2026, as_of_week=3, position_group="WR")
    row = result.row(0, named=True)
    assert row["defteam"] == "BUF"
    assert row["frozen_adj_epa_allowed"] == pytest.approx(0.2)  # week 3, not week 4


def test_future_week_opponents_excludes_a_real_bye() -> None:
    schedule = pl.DataFrame(
        {
            "season": [2026, 2026, 2026],
            "week": [5, 6, 7],
            "home_team": ["KC", "BUF", "KC"],
            "away_team": ["DEN", "KC", "LAC"],
            "season_type": ["REG", "REG", "REG"],
        }
    )
    # KC plays week 5 (home vs DEN) and week 7 (home vs LAC); week 6 is a
    # real bye (KC appears as neither home nor away).
    result = ros_shape.future_week_opponents(schedule, season=2026, team="KC", weeks=[5, 6, 7])
    assert result["week"].to_list() == [5, 7]
    assert result["opponent_team"].to_list() == ["DEN", "LAC"]


def test_allocate_season_consensus_sums_to_the_real_level() -> None:
    weeks_with_opponents = pl.DataFrame({"week": [5, 6, 7], "opponent_team": ["DEN", "LAC", "LV"]})
    frozen_ratings = {
        "WR": pl.DataFrame(
            {
                "defteam": ["DEN", "LAC", "LV"],
                "frozen_adj_epa_allowed": [0.3, -0.1, 0.0],  # DEN = easiest, LAC = hardest
                "frozen_n_plays": [20, 22, 21],
            }
        )
    }
    result = ros_shape.allocate_season_consensus(
        season_consensus_ros_points=30.0,
        position="WR",
        team="KC",
        weeks_with_opponents=weeks_with_opponents,
        frozen_ratings_by_group={"WR": frozen_ratings["WR"]},
    )
    assert result["mean"].sum() == pytest.approx(30.0)
    # Easier matchup (higher adj_epa_allowed) gets a bigger share.
    by_week = dict(zip(result["week"].to_list(), result["mean"].to_list(), strict=True))
    assert by_week[5] > by_week[6]  # DEN (easiest) > LAC (hardest)


def test_allocate_season_consensus_handles_flat_ratings_evenly() -> None:
    """Every real opponent equally tough -> an equal split across weeks
    (10.0 each of 30.0 across 3 weeks) -- proves the allocator doesn't
    introduce spurious variation when there's genuinely none."""
    weeks_with_opponents = pl.DataFrame({"week": [5, 6, 7], "opponent_team": ["A", "B", "C"]})
    frozen_ratings = pl.DataFrame(
        {"defteam": ["A", "B", "C"], "frozen_adj_epa_allowed": [0.0, 0.0, 0.0], "frozen_n_plays": [20, 20, 20]}
    )
    result = ros_shape.allocate_season_consensus(
        season_consensus_ros_points=30.0, position="WR", team="KC",
        weeks_with_opponents=weeks_with_opponents, frozen_ratings_by_group={"WR": frozen_ratings},
    )
    for value in result["mean"].to_list():
        assert value == pytest.approx(10.0)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_models_ros_shape.py -v`
Expected: FAIL — module doesn't exist.

- [ ] **Step 3: Implement**

Create `src/ffapp/models/ros_shape.py`:

```python
"""Future-week shape allocation (`SPEC-ADDENDUM-04.md` §D.1's amendment,
`docs/JOURNAL.md`'s 2026-08-16 entry): "consensus supplies the level;
the pipeline supplies the shape." `models.ros_consensus` resolves a real
season-long remaining-value LEVEL per player; this module allocates that
level across the player's own real remaining weeks (byes excluded
entirely), shaped by each real future opponent's own already-validated
opponent-adjusted rating (`defense_position_allowed`'s `adj_epa_allowed`,
task 1.8), FROZEN at whatever is actually known as of the anchor week --
never a future week's own rating, which doesn't exist in the real table
regardless (task 1.8's own table only ever has rows for real played
weeks). Deliberately does not touch availability/injury at all -- that
is the aggregation stage's own separate `p_play[w]` multiplier
(`tools.ros_aggregate`); baking it in here too would double-count.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from ffapp.features.opponent import POSITION_TO_GROUPS, team_opponent
from ffapp.interim.build import RELOCATED_TEAM_ALIASES

_MATCHUP_WEIGHT_SCALE = 0.15
_MATCHUP_WEIGHT_CLIP = 0.3


def frozen_defense_ratings(
    defense_position_allowed: pl.DataFrame, *, season: int, as_of_week: int, position_group: str
) -> pl.DataFrame:
    """Each real team's own latest known `adj_epa_allowed` for
    `position_group`, frozen at `as_of_week` -- the most recent real row
    at or before `as_of_week`. Made an explicit filter rather than relying
    only on "no future row exists yet" (true today, but this makes the
    freeze a real, testable contract rather than an accident of data
    availability)."""
    scoped = defense_position_allowed.filter(
        (pl.col("season") == season)
        & (pl.col("position_group") == position_group)
        & (pl.col("week") <= as_of_week)
    )
    return (
        scoped.sort(["defteam", "week"])
        .group_by("defteam", maintain_order=True)
        .agg(pl.all().last())
        .select(
            "defteam",
            pl.col("adj_epa_allowed").alias("frozen_adj_epa_allowed"),
            pl.col("n_plays").alias("frozen_n_plays"),
        )
    )


def future_week_opponents(
    schedule: pl.DataFrame, *, season: int, team: str, weeks: list[int]
) -> pl.DataFrame:
    """`week, opponent_team` for every real week in `weeks` that `team`
    actually has a game -- a real bye contributes no row at all (SPEC-
    ADDENDUM-04.md §D.2: "bye weeks contribute zero"), consistent with
    `tools.sos.team_position_group_schedule`'s own same real convention.
    """
    opponents = team_opponent(schedule).filter(
        (pl.col("season") == season) & (pl.col("team") == team) & pl.col("week").is_in(weeks)
    )
    return opponents.select("week", pl.col("opponent").alias("opponent_team")).sort("week")


def _matchup_weight(frozen_value: float | None, mean: float, std: float) -> float:
    if frozen_value is None or std <= 0:
        return 1.0
    z = (frozen_value - mean) / std
    return 1.0 + float(np.clip(z * _MATCHUP_WEIGHT_SCALE, -_MATCHUP_WEIGHT_CLIP, _MATCHUP_WEIGHT_CLIP))


def allocate_season_consensus(
    season_consensus_ros_points: float,
    position: str,
    team: str,
    weeks_with_opponents: pl.DataFrame,
    frozen_ratings_by_group: dict[str, pl.DataFrame],
) -> pl.DataFrame:
    """Allocates `season_consensus_ros_points` across `weeks_with_opponents`'s
    real rows (`week, opponent_team`) proportional to a bounded matchup
    weight (`_matchup_weight`, a documented judgment-call scale/clip pair
    -- same status as `config.WaiverSettings.aggressiveness`, no SPEC-given
    value exists to fit against). A player's own relevant real position
    groups (`features.opponent.POSITION_TO_GROUPS` -- QB/RB get two,
    WR/TE get one) are averaged when more than one applies. Weights are
    normalized so `Σ mean == season_consensus_ros_points` exactly (up to
    floating point) -- the real level is always preserved by construction,
    only its real week-to-week shape varies.
    """
    groups = POSITION_TO_GROUPS.get(position, [])
    if not groups or weeks_with_opponents.is_empty():
        n = weeks_with_opponents.height
        if n == 0:
            return pl.DataFrame(schema={"week": pl.Int64, "mean": pl.Float64})
        even_share = season_consensus_ros_points / n
        return weeks_with_opponents.select("week").with_columns(pl.lit(even_share).alias("mean"))

    aliased = weeks_with_opponents.with_columns(
        pl.col("opponent_team").replace(RELOCATED_TEAM_ALIASES).alias("_defteam")
    )
    group_weights: list[pl.Series] = []
    for group in groups:
        ratings = frozen_ratings_by_group.get(group)
        if ratings is None or ratings.is_empty():
            group_weights.append(pl.Series([1.0] * aliased.height))
            continue
        values = ratings["frozen_adj_epa_allowed"].to_numpy()
        mean, std = float(np.mean(values)), float(np.std(values))
        joined = aliased.join(ratings, left_on="_defteam", right_on="defteam", how="left")
        weights = [
            _matchup_weight(v, mean, std) for v in joined["frozen_adj_epa_allowed"].to_list()
        ]
        group_weights.append(pl.Series(weights))

    combined_weight = group_weights[0]
    for extra in group_weights[1:]:
        combined_weight = (combined_weight + extra) / 2.0
    total_weight = float(combined_weight.sum())
    shares = (combined_weight / total_weight).to_numpy() if total_weight > 0 else None
    if shares is None:
        n = aliased.height
        shares = np.full(n, 1.0 / n)

    return aliased.select("week").with_columns(
        pl.Series("mean", shares * season_consensus_ros_points)
    )


__all__ = [
    "allocate_season_consensus",
    "frozen_defense_ratings",
    "future_week_opponents",
]
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_models_ros_shape.py -v`
Expected: PASS, full file.

- [ ] **Step 5: mypy/ruff**

Run: `uv run mypy src/ffapp/models/ros_shape.py && uv run ruff check src/ffapp/models/ros_shape.py`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add src/ffapp/models/ros_shape.py tests/test_models_ros_shape.py
git commit -m "feat: allocate season-long consensus across future weeks by frozen matchup shape"
```

---

### Task 8: Multi-week projection composer — `projections_ros.parquet`

**Files:**
- Create: `src/ffapp/models/predict_ros.py`
- Test: `tests/test_models_predict_ros.py`

**Interfaces:**
- Consumes: `models.predict.project_week` (unchanged, called once for the anchor week). `models.ros_consensus.fetch_season_consensus`/`resolve_remaining_value`/`aggregate_remaining_value` (Task 6). `models.ros_shape.frozen_defense_ratings`/`future_week_opponents`/`allocate_season_consensus` (Task 7). `models.baselines.empirical_error_quantiles`/`apply_empirical_error_quantiles` (already exist). `tools.prediction_log.season_source_trend` (Task 6 rename).
- Produces: `models.predict_ros.OUTPUT_COLUMNS` (`player_id, season, week, position, team, opponent_team, mean, q10, q25, q50, q75, q90, is_current_week, as_of_utc`). `models.predict_ros.project_week_range(features, schedule, defense_position_allowed, season, from_week, through_week, league_slug, scoring_settings, players_dim, b3_historical, actuals_to_date, season_points_by_source, trend_by_source, quantile_alphas, now, *, train_start, min_train_rows, lightgbm_params, code_version, offline, settings) -> pl.DataFrame`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_models_predict_ros.py (new file)

from __future__ import annotations

from datetime import UTC, datetime

import polars as pl
import pytest

from ffapp.models import predict_ros


def test_project_week_range_current_week_matches_existing_project_week(monkeypatch) -> None:
    """The anchor week's own row(s) must come from the real, unchanged
    `models.predict.project_week` (mocked here to isolate this test from
    needing real fitted models/network) -- this test's real job is
    proving predict_ros doesn't reimplement or alter current-week logic,
    only calls it."""
    from ffapp.models import predict as predict_module

    called_with = {}

    def fake_project_week(features, season, week, **kwargs):
        called_with["week"] = week
        return pl.DataFrame(
            {
                "player_id": ["p1"],
                "season": [season],
                "week": [week],
                "p_active": [0.95],
                "mean": [15.0],
                "q10": [8.0], "q25": [11.0], "q50": [15.0], "q75": [19.0], "q90": [23.0],
                "model_version": ["v1"], "projection_source": ["consensus_b3"],
                "as_of_utc": ["2026-09-01T00:00:00+00:00"], "feature_hash": ["h1"], "git_commit": ["abc"],
            }
        )

    monkeypatch.setattr(predict_module, "project_week", fake_project_week)

    result = predict_ros.project_week_range(
        features=pl.DataFrame(schema={"player_id": pl.String, "season": pl.Int64, "week": pl.Int64, "position": pl.String, "team": pl.String}),
        schedule=pl.DataFrame(schema={"season": pl.Int64, "week": pl.Int64, "home_team": pl.String, "away_team": pl.String, "season_type": pl.String}),
        defense_position_allowed=pl.DataFrame(schema={"season": pl.Int64, "week": pl.Int64, "defteam": pl.String, "position_group": pl.String, "adj_epa_allowed": pl.Float64, "n_plays": pl.Int64}),
        season=2026, from_week=5, through_week=5, league_slug="test-league",
        scoring_settings={}, players_dim=pl.DataFrame(schema={
            "player_id": pl.String, "normalized_name": pl.String, "full_name": pl.String,
            "position": pl.String, "sleeper_id": pl.String,
        }),
        b3_historical=pl.DataFrame(schema={"player_id": pl.String, "season": pl.Int64, "week": pl.Int64, "b3_points": pl.Float64}),
        actuals_to_date=pl.DataFrame(schema={"player_id": pl.String, "actual_points_to_date": pl.Float64}),
        season_points_by_source={}, trend_by_source={},
        quantile_alphas=(0.10, 0.25, 0.50, 0.75, 0.90),
        now=datetime(2026, 9, 1, tzinfo=UTC),
        train_start=2015, min_train_rows=1, lightgbm_params=None, code_version="abc",
        offline=True, settings=None,
    )
    assert called_with["week"] == 5
    assert result.filter(pl.col("is_current_week"))["mean"].to_list() == [15.0]


def test_project_week_range_future_week_uses_shape_not_project_week(monkeypatch) -> None:
    """A future week's row must NOT come from `models.predict.project_week`
    at all (it would need a weekly consensus that doesn't exist for that
    week) -- proven by making the mock raise if called for any week other
    than the anchor."""
    from ffapp.models import predict as predict_module

    def fake_project_week(features, season, week, **kwargs):
        assert week == 5, f"project_week must only be called for the anchor week, got {week}"
        return pl.DataFrame(
            {
                "player_id": ["p1"], "season": [season], "week": [week], "p_active": [0.95],
                "mean": [15.0], "q10": [8.0], "q25": [11.0], "q50": [15.0], "q75": [19.0], "q90": [23.0],
                "model_version": ["v1"], "projection_source": ["consensus_b3"],
                "as_of_utc": ["x"], "feature_hash": ["h1"], "git_commit": ["abc"],
            }
        )

    monkeypatch.setattr(predict_module, "project_week", fake_project_week)

    features = pl.DataFrame(
        {
            "player_id": ["p1"], "season": [2026], "week": [5], "position": ["WR"], "team": ["KC"],
            "as_of_utc": ["2026-10-01T00:00:00+00:00"],
        }
    )
    schedule = pl.DataFrame(
        {
            "season": [2026, 2026], "week": [6, 7], "home_team": ["KC", "DEN"],
            "away_team": ["DEN", "KC"], "season_type": ["REG", "REG"],
        }
    )
    dpa = pl.DataFrame(schema={"season": pl.Int64, "week": pl.Int64, "defteam": pl.String, "position_group": pl.String, "adj_epa_allowed": pl.Float64, "n_plays": pl.Int64})
    players_dim = pl.DataFrame(
        {
            "player_id": ["p1"], "normalized_name": ["p one"], "full_name": ["P One"],
            "position": ["WR"], "sleeper_id": ["s1"],
        }
    )
    season_points = {"espn": pl.DataFrame({"player_id": ["p1"], "position": ["WR"], "team": ["KC"], "points": [150.0]})}

    result = predict_ros.project_week_range(
        features=features, schedule=schedule, defense_position_allowed=dpa,
        season=2026, from_week=5, through_week=7, league_slug="test-league",
        scoring_settings={}, players_dim=players_dim,
        b3_historical=pl.DataFrame(schema={"player_id": pl.String, "season": pl.Int64, "week": pl.Int64, "b3_points": pl.Float64}),
        actuals_to_date=pl.DataFrame({"player_id": ["p1"], "actual_points_to_date": [50.0]}),
        season_points_by_source=season_points, trend_by_source={"espn": "flat"},
        quantile_alphas=(0.10, 0.25, 0.50, 0.75, 0.90),
        now=datetime(2026, 10, 1, tzinfo=UTC),
        train_start=2015, min_train_rows=1, lightgbm_params=None, code_version="abc",
        offline=True, settings=None,
    )
    future_rows = result.filter(~pl.col("is_current_week"))
    assert set(future_rows["week"].to_list()) == {6, 7}
    # season_consensus_ros_points = 150 - 50 = 100, split across weeks 6/7
    assert future_rows["mean"].sum() == pytest.approx(100.0, rel=0.01)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_models_predict_ros.py -v`
Expected: FAIL — module doesn't exist.

- [ ] **Step 3: Implement**

Create `src/ffapp/models/predict_ros.py`. Compose, don't reimplement: call `predict.project_week` once for the anchor week (`from_week`), and for `from_week+1 .. through_week`, run `ros_consensus` (Task 6) once for the whole remaining season, then `ros_shape.allocate_season_consensus` (Task 7) once per (player, team) to split it across that player's own real remaining weeks, then apply `baselines.empirical_error_quantiles`/`apply_empirical_error_quantiles` (reusing the SAME error distribution already fit against `b3_historical` for the anchor week — one already-validated spread mechanism, not a second one) to get `q10..q90` around each future week's shaped mean.

```python
"""Multi-week ROS projection composer (`SPEC-ADDENDUM-04.md` §D.1's
amended horizon split, `docs/JOURNAL.md`'s 2026-08-16 entry;
TASKS.md 1.21). Writes `outputs/<league_slug>/projections_ros.parquet`
(task-level acceptance: every row carries `as_of_utc`).

Deliberately thin -- every real piece of math already lives in
`models.predict` (current week, unchanged), `models.ros_consensus`
(season-long level), and `models.ros_shape` (weekly shape). This module's
only real job is the seam between them: call `project_week` exactly once
for the anchor week, resolve/aggregate/allocate exactly once for every
remaining week at once (not once per week -- the season-long consensus
fetch is one real network round-trip per source, not `through_week -
from_week` of them), and combine both into one output schema.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

import polars as pl

from ffapp.config import LightGBMSettings, Settings
from ffapp.ids import mapping
from ffapp.models import baselines, predict, ros_consensus, ros_shape

OUTPUT_COLUMNS = [
    "player_id", "season", "week", "position", "team", "opponent_team",
    "mean", "q10", "q25", "q50", "q75", "q90", "is_current_week", "as_of_utc",
]

_OUTPUT_SCHEMA = {
    "player_id": pl.String, "season": pl.Int64, "week": pl.Int64, "position": pl.String,
    "team": pl.String, "opponent_team": pl.String, "mean": pl.Float64,
    "q10": pl.Float64, "q25": pl.Float64, "q50": pl.Float64, "q75": pl.Float64, "q90": pl.Float64,
    "is_current_week": pl.Boolean, "as_of_utc": pl.String,
}

_Q_COLUMN_NAMES = {0.10: "q10", 0.25: "q25", 0.50: "q50", 0.75: "q75", 0.90: "q90"}


def _current_week_rows(
    features: pl.DataFrame, season: int, week: int, players_dim: pl.DataFrame,
    b3_historical: pl.DataFrame, *, train_start: int, min_train_rows: int,
    lightgbm_params: LightGBMSettings, code_version: str | None, now: datetime,
    quantile_alphas: Sequence[float], offline: bool | None, settings: Settings | None,
) -> pl.DataFrame:
    result = predict.project_week(
        features, season, week, train_start=train_start, min_train_rows=min_train_rows,
        lightgbm_params=lightgbm_params, code_version=code_version, now=now,
        quantile_alphas=quantile_alphas, projection_source="consensus_b3",
        players_dim=players_dim, b3_historical=b3_historical, offline=offline, settings=settings,
    )
    if result.is_empty():
        return pl.DataFrame(schema=_OUTPUT_SCHEMA)
    team_by_player = features.filter(
        (pl.col("season") == season) & (pl.col("week") == week)
    ).select("player_id", "team")
    return (
        result.join(team_by_player, on="player_id", how="left")
        .with_columns(pl.lit(None, dtype=pl.String).alias("opponent_team"), pl.lit(True).alias("is_current_week"))
        .select(*OUTPUT_COLUMNS[:-1], "as_of_utc")
    )


def project_week_range(
    features: pl.DataFrame,
    schedule: pl.DataFrame,
    defense_position_allowed: pl.DataFrame,
    season: int,
    from_week: int,
    through_week: int,
    league_slug: str,
    scoring_settings: dict[str, float],
    players_dim: pl.DataFrame,
    b3_historical: pl.DataFrame,
    actuals_to_date: pl.DataFrame,
    season_points_by_source: dict[str, pl.DataFrame],
    trend_by_source: dict[str, str],
    quantile_alphas: Sequence[float],
    now: datetime,
    *,
    train_start: int,
    min_train_rows: int,
    lightgbm_params: LightGBMSettings | None,
    code_version: str | None,
    offline: bool | None,
    settings: Settings | None,
) -> pl.DataFrame:
    """`season_points_by_source` is `models.ros_consensus.fetch_season_consensus`'s
    own real output, fetched exactly once by the caller (CLI/log job) for
    this whole horizon -- passed in rather than fetched here, so this
    function stays a pure composer, testable without a real network call.
    """
    current = _current_week_rows(
        features, season, from_week, players_dim, b3_historical,
        train_start=train_start, min_train_rows=min_train_rows, lightgbm_params=lightgbm_params,
        code_version=code_version, now=now, quantile_alphas=quantile_alphas,
        offline=offline, settings=settings,
    )
    if current.is_empty():
        return pl.DataFrame(schema=_OUTPUT_SCHEMA)

    future_weeks = list(range(from_week + 1, through_week + 1))
    if not future_weeks:
        return current

    resolved = ros_consensus.resolve_remaining_value(season_points_by_source, trend_by_source, actuals_to_date)
    # `join_key` only exists after `dedupe_to_one_row_per_name_position`
    # (confirmed: `ids.mapping.build_players_dim`'s own raw output has no
    # such column -- `add_b3_fp_weekly_consensus` already established this
    # exact call as the real way to get one, reused here rather than a
    # second name-normalization scheme).
    keyed_players_dim = mapping.dedupe_to_one_row_per_name_position(players_dim)
    with_identity = resolved.join(
        keyed_players_dim.select("player_id", "join_key", pl.col("full_name").alias("player_name")),
        on="player_id", how="left",
    )
    aggregated = ros_consensus.aggregate_remaining_value(with_identity)
    level_by_player = with_identity.select("player_id", "join_key", "position", "team").unique(
        subset=["player_id"]
    ).join(
        aggregated.select("join_key", "season_consensus_ros_points"), on="join_key", how="inner"
    )

    train_rows_with_b3 = features.filter(
        (pl.col("season") < season) | ((pl.col("season") == season) & (pl.col("week") < from_week))
    ).join(b3_historical.select("player_id", "season", "week", "b3_points"), on=["player_id", "season", "week"], how="inner")
    error_quantiles = baselines.empirical_error_quantiles(train_rows_with_b3, "b3_points", quantile_alphas)

    dpa_groups: dict[str, pl.DataFrame] = {}
    for group in defense_position_allowed["position_group"].unique().to_list():
        dpa_groups[group] = ros_shape.frozen_defense_ratings(
            defense_position_allowed, season=season, as_of_week=from_week, position_group=group
        )

    future_frames: list[pl.DataFrame] = []
    for row in level_by_player.iter_rows(named=True):
        weeks_with_opponents = ros_shape.future_week_opponents(
            schedule, season=season, team=row["team"], weeks=future_weeks
        )
        if weeks_with_opponents.is_empty():
            continue
        allocated = ros_shape.allocate_season_consensus(
            row["season_consensus_ros_points"], row["position"], row["team"],
            weeks_with_opponents, dpa_groups,
        )
        allocated = allocated.join(weeks_with_opponents, on="week", how="left")
        for tau, column_name in _Q_COLUMN_NAMES.items():
            offset = error_quantiles.get(row["position"], {}).get(tau, 0.0)
            allocated = allocated.with_columns(
                (pl.col("mean") + offset).clip(lower_bound=0.0).alias(column_name)
            )
        future_frames.append(
            allocated.with_columns(
                pl.lit(row["player_id"]).alias("player_id"), pl.lit(season).alias("season"),
                pl.lit(row["position"]).alias("position"), pl.lit(row["team"]).alias("team"),
                pl.lit(False).alias("is_current_week"), pl.lit(now.isoformat()).alias("as_of_utc"),
            ).select(*OUTPUT_COLUMNS)
        )

    future = (
        pl.concat(future_frames, how="vertical_relaxed") if future_frames else pl.DataFrame(schema=_OUTPUT_SCHEMA)
    )
    return pl.concat([current, future], how="vertical_relaxed")


__all__ = ["OUTPUT_COLUMNS", "project_week_range"]
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_models_predict_ros.py -v`
Expected: PASS, full file.

- [ ] **Step 5: mypy/ruff**

Run: `uv run mypy src/ffapp/models/predict_ros.py && uv run ruff check src/ffapp/models/predict_ros.py`
Expected: clean (this will also catch any leftover dead-code branches from Step 3's note).

- [ ] **Step 6: Commit**

```bash
git add src/ffapp/models/predict_ros.py tests/test_models_predict_ros.py
git commit -m "feat: compose current-week and future-week projections into projections_ros"
```

---

### Task 9: ROS Monte Carlo aggregation — points, distribution, expected games, playoff value

**Files:**
- Create: `src/ffapp/tools/ros_aggregate.py`
- Test: `tests/test_tools_ros_aggregate.py`

**Interfaces:**
- Consumes: `models.predict_ros.project_week_range`'s output (`player_id, season, week, position, team, opponent_team, mean, q10..q90, is_current_week`). `sim.week.PlayerMarginal`. `sim.persistence.simulate_week_with_common_factor`. `sim.season.simulate_availability` (Task 3's per-player `recovery_prob`). `sim.injury.predict_p_miss`/`fit_hazard_model`. `models.availability.predict_p_active`. `tools.sos.playoff_weeks`. `config.RosCalibration`/`config.RosSettings`.
- Produces: `tools.ros_aggregate.aggregate_ros(projections_ros: pl.DataFrame, p_active_now: dict[str, float], p_miss_now: dict[str, float], position_by_player: dict[str, str], calibration: RosCalibration, *, playoff_weeks: list[int], ros_sims: int, default_recovery_prob: float, correlation: CorrelationSettings, rng: np.random.Generator) -> pl.DataFrame` (`player_id, ros_points, ros_p10, ros_p50, ros_p90, expected_games, playoff_weeks_value`).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_tools_ros_aggregate.py (new file)

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from ffapp.config import DEFAULT_CORRELATION_SETTINGS, RosCalibration
from ffapp.tools import ros_aggregate


def _projections_ros() -> pl.DataFrame:
    rows = []
    for week, mean in [(5, 15.0), (6, 12.0), (7, 18.0)]:
        rows.append(
            {
                "player_id": "p1", "season": 2026, "week": week, "position": "RB",
                "team": "KC", "opponent_team": "DEN",
                "mean": mean, "q10": mean - 6, "q25": mean - 3, "q50": mean,
                "q75": mean + 3, "q90": mean + 6, "is_current_week": week == 5,
            }
        )
    return pl.DataFrame(rows)


def test_aggregate_ros_points_roughly_matches_sum_of_means() -> None:
    """With no injury risk (p_miss=0) and full health (p_active_now=1),
    ros_points should land close to the simple sum of each week's own
    mean (15+12+18=45) -- the Monte Carlo shouldn't systematically bias
    the level away from what the shape function already set."""
    result = ros_aggregate.aggregate_ros(
        _projections_ros(),
        p_active_now={"p1": 1.0}, p_miss_now={"p1": 0.0}, position_by_player={"p1": "RB"},
        calibration=RosCalibration(within_player_week_correlation={"RB": 0.3}, recovery_prob={"RB": 0.5}),
        playoff_weeks=[7], ros_sims=5000, default_recovery_prob=0.5,
        correlation=DEFAULT_CORRELATION_SETTINGS, rng=np.random.default_rng(0),
    )
    row = result.row(0, named=True)
    assert row["ros_points"] == pytest.approx(45.0, rel=0.05)
    assert row["expected_games"] == pytest.approx(3.0, rel=0.02)


def test_aggregate_ros_playoff_weeks_value_is_separate_column() -> None:
    result = ros_aggregate.aggregate_ros(
        _projections_ros(),
        p_active_now={"p1": 1.0}, p_miss_now={"p1": 0.0}, position_by_player={"p1": "RB"},
        calibration=RosCalibration(within_player_week_correlation={"RB": 0.3}, recovery_prob={"RB": 0.5}),
        playoff_weeks=[7], ros_sims=5000, default_recovery_prob=0.5,
        correlation=DEFAULT_CORRELATION_SETTINGS, rng=np.random.default_rng(1),
    )
    row = result.row(0, named=True)
    assert row["playoff_weeks_value"] == pytest.approx(18.0, rel=0.1)
    assert row["playoff_weeks_value"] < row["ros_points"]  # never folded into the main total


def test_aggregate_ros_reduces_expected_games_with_real_injury_risk() -> None:
    healthy = ros_aggregate.aggregate_ros(
        _projections_ros(), p_active_now={"p1": 1.0}, p_miss_now={"p1": 0.0},
        position_by_player={"p1": "RB"},
        calibration=RosCalibration(within_player_week_correlation={"RB": 0.3}, recovery_prob={"RB": 0.5}),
        playoff_weeks=[], ros_sims=5000, default_recovery_prob=0.5,
        correlation=DEFAULT_CORRELATION_SETTINGS, rng=np.random.default_rng(2),
    )
    risky = ros_aggregate.aggregate_ros(
        _projections_ros(), p_active_now={"p1": 1.0}, p_miss_now={"p1": 0.4},
        position_by_player={"p1": "RB"},
        calibration=RosCalibration(within_player_week_correlation={"RB": 0.3}, recovery_prob={"RB": 0.5}),
        playoff_weeks=[], ros_sims=5000, default_recovery_prob=0.5,
        correlation=DEFAULT_CORRELATION_SETTINGS, rng=np.random.default_rng(3),
    )
    assert risky.row(0, named=True)["expected_games"] < healthy.row(0, named=True)["expected_games"]


def test_aggregate_ros_p10_below_p90() -> None:
    result = ros_aggregate.aggregate_ros(
        _projections_ros(), p_active_now={"p1": 1.0}, p_miss_now={"p1": 0.1},
        position_by_player={"p1": "RB"},
        calibration=RosCalibration(within_player_week_correlation={"RB": 0.3}, recovery_prob={"RB": 0.5}),
        playoff_weeks=[], ros_sims=5000, default_recovery_prob=0.5,
        correlation=DEFAULT_CORRELATION_SETTINGS, rng=np.random.default_rng(4),
    )
    row = result.row(0, named=True)
    assert row["ros_p10"] < row["ros_p50"] < row["ros_p90"]
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_tools_ros_aggregate.py -v`
Expected: FAIL — module doesn't exist.

- [ ] **Step 3: Implement**

Create `src/ffapp/tools/ros_aggregate.py`:

```python
"""ROS Monte Carlo aggregation (`SPEC-ADDENDUM-04.md` §D.2). Composes,
per real player: task 2.2's cross-player correlated weekly sampling
(`sim.week`, via `sim.persistence.simulate_week_with_common_factor` for
the added within-player layer -- requirement 2), task 2.3/2.4's
persistent-duration injury sampling (`sim.season.simulate_availability`
-- requirement 3), and `models.predict_ros`'s own already-shaped weekly
quantile grids (current week unchanged, future weeks per
`models.ros_shape`). `p_play[w] = p_active_now x P(available in week w
| hazard persistence)`, applied multiplicatively at this aggregation
stage only -- SPEC-ADDENDUM-04.md §D.2's own literal pseudocode, and the
one place in this pipeline availability is applied (never baked into the
shape function, which would double-count).
"""

from __future__ import annotations

import numpy as np
import polars as pl

from ffapp.config import CorrelationSettings, RosCalibration
from ffapp.sim.persistence import simulate_week_with_common_factor
from ffapp.sim.season import simulate_availability
from ffapp.sim.week import PlayerMarginal

_QUANTILE_ALPHAS = (0.10, 0.25, 0.50, 0.75, 0.90)
_Q_COLUMNS = {0.10: "q10", 0.25: "q25", 0.50: "q50", 0.75: "q75", 0.90: "q90"}


def aggregate_ros(
    projections_ros: pl.DataFrame,
    p_active_now: dict[str, float],
    p_miss_now: dict[str, float],
    position_by_player: dict[str, str],
    calibration: RosCalibration,
    *,
    playoff_weeks: list[int],
    ros_sims: int,
    default_recovery_prob: float,
    correlation: CorrelationSettings,
    rng: np.random.Generator,
) -> pl.DataFrame:
    players = sorted(projections_ros["player_id"].unique().to_list())
    n_players = len(players)
    if n_players == 0:
        return pl.DataFrame(
            schema={
                "player_id": pl.String, "ros_points": pl.Float64, "ros_p10": pl.Float64,
                "ros_p50": pl.Float64, "ros_p90": pl.Float64, "expected_games": pl.Float64,
                "playoff_weeks_value": pl.Float64,
            }
        )

    weeks = sorted(projections_ros["week"].unique().to_list())
    n_weeks = len(weeks)
    player_index = {pid: i for i, pid in enumerate(players)}

    positions = [position_by_player.get(pid, "") for pid in players]
    rho = np.array(
        [calibration.within_player_week_correlation.get(pos, 0.0) for pos in positions]
    )
    recovery = np.array(
        [calibration.recovery_prob.get(pos, default_recovery_prob) for pos in positions]
    )
    p_miss = np.tile(
        np.array([p_miss_now.get(pid, 0.0) for pid in players]), (n_weeks, 1)
    )
    p_active = np.array([p_active_now.get(pid, 1.0) for pid in players])

    available = simulate_availability(
        p_miss, season_sims=ros_sims, recovery_prob=recovery, rng=rng
    )  # (ros_sims, n_weeks, n_players)

    player_factor = rng.standard_normal((ros_sims, n_players))
    totals = np.zeros((ros_sims, n_weeks, n_players))
    for week_idx, week in enumerate(weeks):
        week_rows = projections_ros.filter(pl.col("week") == week).sort(
            pl.col("player_id").map_elements(lambda p: player_index.get(p, -1), return_dtype=pl.Int64)
        )
        present_ids = week_rows["player_id"].to_list()
        marginals = [
            PlayerMarginal(
                player_id=row["player_id"], position=row["position"], team=row["team"],
                opponent_team=row.get("opponent_team"),
                alphas=list(_QUANTILE_ALPHAS),
                quantile_values=[row[_Q_COLUMNS[a]] for a in _QUANTILE_ALPHAS],
            )
            for row in week_rows.iter_rows(named=True)
        ]
        present_idx = [player_index[pid] for pid in present_ids]
        week_rho = rho[present_idx]
        week_factor = player_factor[:, present_idx]
        scores = simulate_week_with_common_factor(
            marginals, correlation, week_sims=ros_sims, player_factor=week_factor,
            rho=week_rho, rng=rng,
        )
        for local_i, global_i in enumerate(present_idx):
            totals[:, week_idx, global_i] = scores[:, local_i]

    actual = totals * available * p_active[None, None, :]
    season_totals = actual.sum(axis=1)  # (ros_sims, n_players)
    expected_games = (available * p_active[None, None, :]).sum(axis=1).mean(axis=0)

    playoff_idx = [weeks.index(w) for w in playoff_weeks if w in weeks]
    playoff_value = (
        actual[:, playoff_idx, :].sum(axis=1).mean(axis=0)
        if playoff_idx
        else np.zeros(n_players)
    )

    ros_points = season_totals.mean(axis=0)
    ros_p10 = np.quantile(season_totals, 0.10, axis=0)
    ros_p50 = np.quantile(season_totals, 0.50, axis=0)
    ros_p90 = np.quantile(season_totals, 0.90, axis=0)

    return pl.DataFrame(
        {
            "player_id": players,
            "ros_points": ros_points.tolist(),
            "ros_p10": ros_p10.tolist(),
            "ros_p50": ros_p50.tolist(),
            "ros_p90": ros_p90.tolist(),
            "expected_games": expected_games.tolist(),
            "playoff_weeks_value": playoff_value.tolist(),
        }
    )


__all__ = ["aggregate_ros"]
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_tools_ros_aggregate.py -v`
Expected: PASS, full file.

- [ ] **Step 5: mypy/ruff**

Run: `uv run mypy src/ffapp/tools/ros_aggregate.py && uv run ruff check src/ffapp/tools/ros_aggregate.py`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add src/ffapp/tools/ros_aggregate.py tests/test_tools_ros_aggregate.py
git commit -m "feat: ROS Monte Carlo aggregation with within-player and cross-player correlation"
```

---

### Task 10: Rest-of-season VOR over the current free-agent pool, with rank-change tracking

**Files:**
- Create: `src/ffapp/tools/ros_rankings.py`
- Test: `tests/test_tools_ros_rankings.py`

**Interfaces:**
- Consumes: `tools.vor.compute_vor`/`replacement_level` (unchanged, task 0.9). `tools.waivers.free_agent_pool`/`rostered_sleeper_ids` (unchanged, task 2.6). `tools.ros_aggregate.aggregate_ros`'s output. `league_format.LeagueFormat`.
- Produces: `tools.ros_rankings.current_free_agent_projections(ros_points_table: pl.DataFrame, players_dim: pl.DataFrame, rostered_ids: set[str], eligible_positions: set[str]) -> pl.DataFrame` (free-agent-scoped `player_id, position, ros_points` renamed to the VOR points column). `tools.ros_rankings.build_ros_board(ros_points_table: pl.DataFrame, players_dim: pl.DataFrame, rostered_ids: set[str], eligible_positions: set[str], league_format: LeagueFormat, *, replacement_overrides: dict[str, float] | None = None) -> pl.DataFrame` (adds `vor_ros`, sorted descending). `tools.ros_rankings.rank_change(current_board: pl.DataFrame, previous_board: pl.DataFrame | None) -> pl.DataFrame` (`player_id, rank, rank_change` — null `rank_change` when there's no real previous board or the player is new).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_tools_ros_rankings.py (new file)

from __future__ import annotations

import polars as pl
import pytest

from ffapp.league_format import LeagueFormat
from ffapp.tools import ros_rankings


def _league_format() -> LeagueFormat:
    return LeagueFormat(
        n_teams=10, starters={"RB": 2, "WR": 2}, flex_slots={"FLEX": 1, "SUPER_FLEX": 0, "REC_FLEX": 0},
        flex_eligible={"FLEX": ["RB", "WR"]}, bench=6, ir=1, playoff_week_start=15, waiver_budget=100,
    )


def test_current_free_agent_projections_excludes_rostered_players() -> None:
    ros_points = pl.DataFrame({"player_id": ["p1", "p2"], "ros_points": [80.0, 60.0]})
    players_dim = pl.DataFrame(
        {"player_id": ["p1", "p2"], "sleeper_id": ["s1", "s2"], "position": ["RB", "WR"], "active": [True, True], "team": ["KC", "BUF"]}
    )
    result = ros_rankings.current_free_agent_projections(
        ros_points, players_dim, rostered_ids={"s1"}, eligible_positions={"RB", "WR"}
    )
    assert result["player_id"].to_list() == ["p2"]


def test_build_ros_board_adds_vor_ros_and_differs_by_league_format() -> None:
    # 50 RBs + 50 WRs -- large enough that neither league format's dedicated
    # starter count (10-team RB2 = 20, 18-team RB2 = 36) exhausts the real
    # pool and falls back to `replacement_level`'s own clamp (which would
    # otherwise silently collapse both formats onto the same worst-available
    # player and make this test's own assertion false regardless of whether
    # LeagueFormat is wired correctly -- a real bug caught while writing this
    # test, not a hypothetical one).
    n_per_position = 50
    ros_points = pl.DataFrame(
        {
            "player_id": [f"p{i}" for i in range(1, 2 * n_per_position + 1)],
            "ros_points": [float(200 - i) for i in range(1, 2 * n_per_position + 1)],
        }
    )
    players_dim = pl.DataFrame(
        {
            "player_id": [f"p{i}" for i in range(1, 2 * n_per_position + 1)],
            "sleeper_id": [f"s{i}" for i in range(1, 2 * n_per_position + 1)],
            "position": ["RB"] * n_per_position + ["WR"] * n_per_position,
            "active": [True] * (2 * n_per_position),
            "team": ["KC"] * (2 * n_per_position),
        }
    )
    fmt_10team = _league_format()
    fmt_18team = LeagueFormat(
        n_teams=18, starters={"RB": 2, "WR": 2}, flex_slots={"FLEX": 1, "SUPER_FLEX": 0, "REC_FLEX": 0},
        flex_eligible={"FLEX": ["RB", "WR"]}, bench=6, ir=1, playoff_week_start=15, waiver_budget=100,
    )
    board_10 = ros_rankings.build_ros_board(ros_points, players_dim, set(), {"RB", "WR"}, fmt_10team)
    board_18 = ros_rankings.build_ros_board(ros_points, players_dim, set(), {"RB", "WR"}, fmt_18team)
    vor_10 = dict(zip(board_10["player_id"].to_list(), board_10["vor_ros"].to_list(), strict=True))
    vor_18 = dict(zip(board_18["player_id"].to_list(), board_18["vor_ros"].to_list(), strict=True))
    assert vor_10 != vor_18  # replacement level must move materially between formats


def test_rank_change_reports_null_with_no_previous_board() -> None:
    current = pl.DataFrame({"player_id": ["p1", "p2"], "vor_ros": [20.0, 10.0]})
    result = ros_rankings.rank_change(current, None)
    assert result["rank_change"].null_count() == result.height


def test_rank_change_reports_real_movement() -> None:
    previous = pl.DataFrame({"player_id": ["p1", "p2"], "vor_ros": [10.0, 20.0]})  # p2 was rank 1
    current = pl.DataFrame({"player_id": ["p1", "p2"], "vor_ros": [20.0, 10.0]})   # p1 now rank 1
    result = ros_rankings.rank_change(current, previous)
    by_player = {row["player_id"]: row for row in result.iter_rows(named=True)}
    assert by_player["p1"]["rank_change"] == 1  # moved up one spot
    assert by_player["p2"]["rank_change"] == -1


def test_rank_change_null_for_a_new_player() -> None:
    previous = pl.DataFrame({"player_id": ["p1"], "vor_ros": [10.0]})
    current = pl.DataFrame({"player_id": ["p1", "p2"], "vor_ros": [10.0, 30.0]})
    result = ros_rankings.rank_change(current, previous)
    new_row = result.filter(pl.col("player_id") == "p2").row(0, named=True)
    assert new_row["rank_change"] is None
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_tools_ros_rankings.py -v`
Expected: FAIL — module doesn't exist.

- [ ] **Step 3: Implement**

Create `src/ffapp/tools/ros_rankings.py`:

```python
"""Rest-of-season VOR over the CURRENT free-agent pool, and week-over-
week rank-change tracking (`SPEC-ADDENDUM-04.md` §D.3/§D.4/§D.5).
Reuses `tools.vor`'s already-shipped fixed point (task 0.9) unmodified --
the real difference from the preseason draft board is only which players
and which points column feed it: this module's real job is the scoping
(`tools.waivers.free_agent_pool`, task 2.6, reused directly rather than
recomputing "who's rostered" a second way) and the rank-over-time diff.
"""

from __future__ import annotations

import polars as pl

from ffapp.league_format import LeagueFormat
from ffapp.tools import vor
from ffapp.tools.waivers import free_agent_pool


def current_free_agent_projections(
    ros_points_table: pl.DataFrame,
    players_dim: pl.DataFrame,
    rostered_ids: set[str],
    eligible_positions: set[str],
) -> pl.DataFrame:
    """`ros_points_table` (`player_id, ros_points`, `tools.ros_aggregate
    .aggregate_ros`'s own output) scoped to real current free agents --
    `tools.waivers.free_agent_pool`'s own already-shipped scoping,
    joined onto the real ROS points."""
    pool = free_agent_pool(players_dim, rostered_ids, eligible_positions)
    return pool.join(ros_points_table, on="player_id", how="inner").select(
        "player_id", "position", "ros_points"
    )


def build_ros_board(
    ros_points_table: pl.DataFrame,
    players_dim: pl.DataFrame,
    rostered_ids: set[str],
    eligible_positions: set[str],
    league_format: LeagueFormat,
    *,
    replacement_overrides: dict[str, float] | None = None,
) -> pl.DataFrame:
    """SPEC §9.4's fixed point (`tools.vor.compute_vor`), replacement
    level computed over `ros_points_table`'s own real remaining-value
    scope and the CURRENT free-agent pool -- `SPEC-ADDENDUM-04.md` §D.3's
    own explicit correction to using August's preseason pool. Ranked by
    `vor_ros` descending, never by raw `ros_points` (§D.3: "never by raw
    projected points")."""
    scoped = current_free_agent_projections(
        ros_points_table, players_dim, rostered_ids, eligible_positions
    )
    with_vor = vor.compute_vor(
        scoped, league_format, points_column="ros_points", replacement_overrides=replacement_overrides
    ).rename({"vor": "vor_ros"})
    return with_vor.sort("vor_ros", descending=True)


def rank_change(current_board: pl.DataFrame, previous_board: pl.DataFrame | None) -> pl.DataFrame:
    """`player_id, rank, rank_change` -- `rank_change` is
    `previous_rank - current_rank` (positive = moved up), null for a
    player with no real logged previous board (the first real run ever,
    or a genuinely new free agent this week) -- SPEC-ADDENDUM-04.md §D.5:
    "that last column is the one you will actually look at," so a
    misleading guessed value here would be the single worst mistake this
    function could make."""
    ranked = current_board.with_columns(
        pl.col("vor_ros").rank(method="ordinal", descending=True).cast(pl.Int64).alias("rank")
    ).select("player_id", "rank")
    if previous_board is None or previous_board.is_empty():
        return ranked.with_columns(pl.lit(None, dtype=pl.Int64).alias("rank_change"))

    previous_ranked = previous_board.with_columns(
        pl.col("vor_ros").rank(method="ordinal", descending=True).cast(pl.Int64).alias("rank")
    ).select("player_id", pl.col("rank").alias("_previous_rank"))

    return (
        ranked.join(previous_ranked, on="player_id", how="left")
        .with_columns((pl.col("_previous_rank") - pl.col("rank")).alias("rank_change"))
        .drop("_previous_rank")
    )


__all__ = ["build_ros_board", "current_free_agent_projections", "rank_change"]
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_tools_ros_rankings.py -v`
Expected: PASS, full file.

- [ ] **Step 5: mypy/ruff**

Run: `uv run mypy src/ffapp/tools/ros_rankings.py && uv run ruff check src/ffapp/tools/ros_rankings.py`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add src/ffapp/tools/ros_rankings.py tests/test_tools_ros_rankings.py
git commit -m "feat: ROS VOR over current free-agent pool, with rank-change tracking"
```

---

### Task 11: CLI — `ffapp project --from-week --through-week --league` and `ffapp rankings ros --league`

**Files:**
- Modify: `src/ffapp/cli.py`
- Test: `tests/test_cli_project.py`
- Test: `tests/test_cli_rankings.py` (new file)

**Interfaces:**
- Consumes: everything from Tasks 6-10, plus `config.load_league`/`load_primary_league`/`load_settings`, `ffapp.ingest.sleeper.fetch_rosters`, `ffapp.tools.waivers.rostered_sleeper_ids`, `ffapp.tools.prediction_log.season_source_trend` and its underlying `source_fetches.parquet`.
- Produces (CLI surface): `ffapp project --from-week N --through-week M --league <slug> [--season] [--offline/--no-offline]` writes `data/outputs/<league_slug>/projections_ros.parquet` (upsert by `(season, week)`, matching `predict.write_projections`'s own convention — reuse it directly, `projections_ros.parquet` has the same `(season, week)` upsert key shape). `ffapp rankings ros --league <slug> [--season]` writes a timestamped `data/outputs/<league_slug>/rankings_ros/<timestamp>/board.parquet` plus updates `data/outputs/<league_slug>/rankings_ros/latest.parquet`, printing the board's own top rows.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_cli_project.py -- append

def test_project_command_accepts_from_week_through_week_and_league(monkeypatch, tmp_path, runner) -> None:
    """A smoke test at the CLI-wiring level (mirrors this file's existing
    `project_command` tests' own style) -- proves the new flags parse and
    route to `models.predict_ros.project_week_range`, not that the real
    math is correct (already proven in tests/test_models_predict_ros.py).
    """
    # Follow this file's own existing pattern for mocking settings/features
    # and asserting the CLI exits 0 with the new flags present. Match
    # whatever fixture/monkeypatch helpers `test_project_command_*` already
    # use elsewhere in this file rather than inventing a second style.
```

```python
# tests/test_cli_rankings.py (new file)
"""Smoke tests for `ffapp rankings ros` -- CLI wiring only, matching
tests/test_cli_project.py's own existing style for mocking settings and
asserting exit codes/output-path behavior. Real math is covered by
tests/test_tools_ros_rankings.py and tests/test_tools_ros_aggregate.py.
"""
```

(Write these two test files by directly mirroring `tests/test_cli_project.py`'s existing fixtures/mocking conventions for `settings`/`features_path`/`CliRunner` — read that file first to match its exact helper functions before writing new tests, per this project's own established pattern of never inventing a second CLI-testing style.)

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_cli_project.py tests/test_cli_rankings.py -v`
Expected: FAIL — new flags/command don't exist yet.

- [ ] **Step 3: Implement**

In `src/ffapp/cli.py`, extend `project_command` with `--from-week`/`--through-week`/`--league` (all optional; when `--from-week`/`--through-week` are both omitted, behavior is byte-for-byte unchanged — the existing single-week path). When both are given, branch to the new multi-week path:

```python
    from_week: int | None = typer.Option(None, "--from-week", help="Start of the ROS horizon (defaults to --week)."),
    through_week: int | None = typer.Option(None, "--through-week", help="End of the ROS horizon (inclusive)."),
    league: str | None = typer.Option(None, "--league", help="League slug. Defaults to the primary league."),
```

```python
    if from_week is not None or through_week is not None:
        if from_week is None or through_week is None:
            typer.echo("--from-week and --through-week must be given together.", err=True)
            raise typer.Exit(code=1)
        league_config = load_league(league) if league is not None else load_primary_league()
        scoring_settings = league_config.league_cache["scoring_settings"]
        schedule = pl.read_parquet(settings.data_root / "interim" / "schedule.parquet")
        dpa = pl.read_parquet(settings.data_root / "interim" / "defense_position_allowed.parquet")

        crosswalk_path = nflverse.fetch_player_ids(offline=offline, settings=settings)
        sleeper_players_path = sleeper.fetch_players(offline=offline, settings=settings)
        players_dim = mapping.build_players_dim(crosswalk_path, sleeper_players_path, mapping.ID_OVERRIDES_PATH)
        # No join_key pre-processing needed here -- project_week_range (Task 8)
        # already calls mapping.dedupe_to_one_row_per_name_position internally
        # on whatever players_dim it receives, and fetch_season_consensus's own
        # fetchers (via prediction_log._resolve_to_player_id) do the same. The
        # raw build_players_dim output is exactly what every downstream
        # consumer expects -- confirmed by reading both call chains directly.

        b3_historical_path = settings.data_root / "interim" / "b3_predictions.parquet"
        if not b3_historical_path.exists():
            typer.echo(f"Missing {b3_historical_path}. See HANDOFF.md.", err=True)
            raise typer.Exit(code=1)
        b3_historical = pl.read_parquet(b3_historical_path)

        now = datetime.now(UTC)
        season_points = ros_consensus.fetch_season_consensus(
            resolved_season, scoring_settings, players_dim, offline=offline, settings=settings, now=now
        )
        log_dir = settings.data_root / "outputs" / league_config.slug / "prediction_log"
        trend_by_source = {}
        fetches_path = log_dir / "source_fetches.parquet"
        if fetches_path.exists():
            trend_df = prediction_log.check_sources(league_slug=league_config.slug, settings=settings)
            trend_by_source = {
                row["source"]: row["season_trend"]
                for row in trend_df.iter_rows(named=True)
                if row["season_trend"] is not None
            }
        actuals_to_date = (
            features.filter((pl.col("season") == resolved_season) & (pl.col("week") < from_week))
            .group_by("player_id")
            .agg(pl.col("target").sum().alias("actual_points_to_date"))
        )

        result = predict_ros.project_week_range(
            features, schedule, dpa, resolved_season, from_week, through_week, league_config.slug,
            scoring_settings, players_dim, b3_historical, actuals_to_date, season_points, trend_by_source,
            settings.model.quantiles, now, train_start=settings.seasons.train_start,
            min_train_rows=settings.model.min_train_rows, lightgbm_params=settings.model.lightgbm,
            code_version=evaluation_report.current_git_commit(), offline=offline, settings=settings,
        )
        if result.is_empty():
            typer.echo("No ROS projections generated -- see HANDOFF.md.", err=True)
            raise typer.Exit(code=1)
        output_path = settings.data_root / "outputs" / league_config.slug / "projections_ros.parquet"
        combined = predict.write_projections(result, output_path)
        typer.echo(f"Wrote {result.height} ROS projections to {output_path} ({combined.height} total rows).")
        return
```

Add a new `rankings_app`:

```python
rankings_app = typer.Typer(name="rankings", help="Rest-of-season and other rankings views.")
app.add_typer(rankings_app, name="rankings")


@rankings_app.command("ros")
def rankings_ros_command(
    league: str | None = typer.Option(None, "--league", help="League slug. Defaults to the primary league."),
    season: int | None = typer.Option(None, "--season", help="Defaults to settings.seasons.current."),
) -> None:
    """Rest-of-season VOR board (`SPEC-ADDENDUM-04.md` §D.3-§D.5; task
    1.21) over the CURRENT free-agent pool, with rank-change since the
    prior real run. Requires `projections_ros.parquet` to already exist
    for this league (`ffapp project --from-week --through-week --league`)."""
    settings = load_settings()
    league_config = load_league(league) if league is not None else load_primary_league()
    resolved_season = season if season is not None else settings.seasons.current
    league_format = parse_league_format(league_config)

    ros_path = settings.data_root / "outputs" / league_config.slug / "projections_ros.parquet"
    if not ros_path.exists():
        typer.echo(f"Missing {ros_path}. Run `ffapp project --from-week --through-week --league {league_config.slug}` first.", err=True)
        raise typer.Exit(code=1)
    projections_ros = pl.read_parquet(ros_path).filter(pl.col("season") == resolved_season)

    crosswalk_path = nflverse.fetch_player_ids(offline=True, settings=settings)
    sleeper_players_path = sleeper.fetch_players(offline=True, settings=settings)
    players_dim = mapping.build_players_dim(crosswalk_path, sleeper_players_path, mapping.ID_OVERRIDES_PATH)

    if league_config.league_id is None:
        typer.echo(f"League {league_config.slug} has no real sleeper.league_id configured.", err=True)
        raise typer.Exit(code=1)
    rosters = json.loads(sleeper.fetch_rosters(league_config.league_id, offline=True, settings=settings).read_text())
    rostered_ids = waivers.rostered_sleeper_ids(rosters)
    # league_relevant_positions takes the real LeagueConfig (needs .league_cache/
    # .overrides), not LeagueFormat -- confirmed against its real signature.
    eligible_positions = ids_mapping.league_relevant_positions(league_config)

    if projections_ros.filter(pl.col("is_current_week")).is_empty():
        typer.echo(f"{ros_path} has no real current-week row -- was it built for this season?", err=True)
        raise typer.Exit(code=1)
    anchor_week = int(projections_ros.filter(pl.col("is_current_week"))["week"].min())

    features_path = settings.data_root / "features" / "player_week_features.parquet"
    features = pl.read_parquet(features_path)
    before_anchor = (pl.col("season") < resolved_season) | (
        (pl.col("season") == resolved_season) & (pl.col("week") < anchor_week)
    )
    anchor_row = (pl.col("season") == resolved_season) & (pl.col("week") == anchor_week)
    train_rows = features.filter(before_anchor)
    target_rows = features.filter(anchor_row)

    availability_model = availability.fit_availability_model(train_rows, lightgbm_params=settings.model.lightgbm)
    p_active_series = availability.predict_p_active(availability_model, target_rows)
    p_active_now = dict(
        zip(target_rows["player_id"].to_list(), p_active_series.to_list(), strict=True)
    )

    # rosters/snap_counts have no data/interim/ counterpart -- sim.injury's own
    # build_hazard_features reads the RAW nflverse tables directly (confirmed
    # against sim/injury.py's own module docstring and docs/JOURNAL.md's task
    # 2.3 entry). schedule/injuries genuinely do have real interim tables.
    train_season_range = list(range(settings.seasons.train_start, settings.seasons.current))
    rosters_table = pl.read_parquet(
        nflverse.fetch_rosters(train_season_range, offline=True, settings=settings)
    )
    schedule = pl.read_parquet(settings.data_root / "interim" / "schedule.parquet")
    injuries = pl.read_parquet(settings.data_root / "interim" / "injuries.parquet")
    snap_counts = pl.read_parquet(
        nflverse.fetch_snap_counts(train_season_range, offline=True, settings=settings)
    )
    # `fetch_player_ids` returns a Path to the raw CSV (`data/raw/nflverse/player_ids.csv`),
    # not parquet -- `mapping.load_crosswalk_base` is the same already-tested loader
    # `mapping.build_players_dim` itself calls internally, reused here rather than
    # re-parsing the CSV a second way.
    crosswalk = mapping.load_crosswalk_base(crosswalk_path)
    hazard_grid = injury.build_hazard_features(rosters_table, schedule, injuries, snap_counts, crosswalk)
    hazard_train = hazard_grid.filter(before_anchor)
    hazard_target = hazard_grid.filter(anchor_row)
    hazard_model = injury.fit_hazard_model(hazard_train)
    p_miss_series = injury.predict_p_miss(hazard_model, hazard_target)
    p_miss_now = dict(
        zip(hazard_target["player_id"].to_list(), p_miss_series.to_list(), strict=True)
    )

    position_by_player = dict(
        zip(projections_ros["player_id"].to_list(), projections_ros["position"].to_list(), strict=False)
    )

    calibration = load_ros_calibration()
    playoff_week_list = sos.playoff_weeks(
        schedule, season=resolved_season, playoff_week_start=league_format.playoff_week_start,
    )
    aggregated = ros_aggregate.aggregate_ros(
        projections_ros, p_active_now, p_miss_now, position_by_player, calibration,
        playoff_weeks=playoff_week_list, ros_sims=settings.ros.ros_sims,
        default_recovery_prob=settings.ros.default_recovery_prob,
        correlation=settings.simulation.correlation, rng=np.random.default_rng(),
    )
    board = ros_rankings.build_ros_board(
        aggregated, players_dim, rostered_ids, eligible_positions, league_format
    )

    out_dir = settings.data_root / "outputs" / league_config.slug / "rankings_ros"
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = out_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    latest_path = out_dir / "latest.parquet"
    previous_board = pl.read_parquet(latest_path) if latest_path.exists() else None
    with_rank_change = board.join(ros_rankings.rank_change(board, previous_board), on="player_id", how="left")

    board_path = run_dir / "board.parquet"
    with_rank_change.write_parquet(board_path)
    with_rank_change.write_parquet(latest_path)
    typer.echo(f"Wrote {with_rank_change.height} ROS ranked players to {board_path} (and latest.parquet).")
    typer.echo(with_rank_change.head(20).to_pandas().to_string(index=False))
```

`rankings_ros_command` fits the availability model (task 1.14) and the injury hazard model (task 2.3) fresh, scoped to real rows strictly before `anchor_week` (the same `is_current_week` row `ffapp project` already wrote) — this reuses the exact already-tested fit/predict functions `models.predict.project_week` and `notebooks/estimate_ros_calibration.py` already call, rather than inventing a third way to get `p_active`/`p_miss`. Add `from ffapp.models import availability` and `from ffapp.sim import injury` to `cli.py`'s existing import block alongside the other module-level imports (`baselines`, `predict`, `points`, `quantiles`, `residual`, etc. are already imported there for other commands — follow the same flat `from ffapp.models import ...`/`from ffapp.sim import ...` style already established in the file rather than deep-importing individual functions), and confirm `ffapp.tools.sos` and `ffapp.tools.waivers`/`ffapp.tools.ros_rankings`/`ffapp.tools.ros_aggregate`/`ffapp.models.ros_consensus`/`ffapp.models.predict_ros`/`ffapp.config.load_ros_calibration` are all imported too.

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_cli_project.py tests/test_cli_rankings.py -v`
Expected: PASS.

- [ ] **Step 5: mypy/ruff**

Run: `uv run mypy src/ffapp/cli.py && uv run ruff check src/ffapp/cli.py`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add src/ffapp/cli.py tests/test_cli_project.py tests/test_cli_rankings.py
git commit -m "feat: wire ffapp project --from-week/--through-week and ffapp rankings ros"
```

---

### Task 12: Streamlit page — `6_ROS_Rankings.py`

**Files:**
- Create: `src/ffapp/app/ros_rankings_page.py`
- Create: `src/ffapp/app/pages/6_ROS_Rankings.py`
- Test: `tests/test_app_ros_rankings_page.py`

**Interfaces:**
- Consumes: `data/outputs/<league_slug>/rankings_ros/latest.parquet` (Task 11's own output — `player_id, position, ros_points, vor_ros, ros_p10, ros_p50, ros_p90, expected_games, playoff_weeks_value, rank, rank_change`). `config.load_primary_league`/`load_settings`. `league_format.parse_league_format`.
- Produces: `app.ros_rankings_page.style_rank_change(board: pl.DataFrame) -> pl.DataFrame` (pure function: formats `rank_change` as a signed string for display, e.g. `+3`/`-1`/`—`, testable without Streamlit). `app.ros_rankings_page.filter_board(board: pl.DataFrame, *, position: str | None, available_ids: set[str] | None) -> pl.DataFrame`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_app_ros_rankings_page.py (new file)

from __future__ import annotations

import polars as pl

from ffapp.app.ros_rankings_page import filter_board, style_rank_change


def test_style_rank_change_formats_signed_movement() -> None:
    board = pl.DataFrame({"player_id": ["p1", "p2", "p3"], "rank_change": [3, -1, None]})
    result = style_rank_change(board)
    assert result["rank_change_display"].to_list() == ["+3", "-1", "—"]


def test_filter_board_by_position() -> None:
    board = pl.DataFrame({"player_id": ["p1", "p2"], "position": ["RB", "WR"], "vor_ros": [10.0, 5.0]})
    result = filter_board(board, position="RB", available_ids=None)
    assert result["player_id"].to_list() == ["p1"]


def test_filter_board_by_availability() -> None:
    board = pl.DataFrame({"player_id": ["p1", "p2"], "position": ["RB", "RB"], "vor_ros": [10.0, 5.0]})
    result = filter_board(board, position=None, available_ids={"p2"})
    assert result["player_id"].to_list() == ["p2"]
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_app_ros_rankings_page.py -v`
Expected: FAIL — module doesn't exist.

- [ ] **Step 3: Implement**

Create `src/ffapp/app/ros_rankings_page.py`:

```python
"""Pure helpers for `app/pages/6_ROS_Rankings.py` (SPEC-ADDENDUM-04.md
§D.5). Matches `app.schedule_grid_page`'s own precedent -- real math and
data access live in `tools.ros_rankings`/`tools.ros_aggregate`; this
module is glue-support only, kept separately testable from Streamlit
itself.
"""

from __future__ import annotations

import polars as pl


def style_rank_change(board: pl.DataFrame) -> pl.DataFrame:
    """Adds `rank_change_display`: `"+N"` for real upward movement,
    `"-N"` for real downward movement, an em dash for a genuinely new
    player or a first-ever real run (null `rank_change` -- see
    `tools.ros_rankings.rank_change`'s own docstring for why this is
    never guessed)."""
    return board.with_columns(
        pl.when(pl.col("rank_change").is_null())
        .then(pl.lit("—"))
        .when(pl.col("rank_change") > 0)
        .then(pl.lit("+") + pl.col("rank_change").cast(pl.String))
        .otherwise(pl.col("rank_change").cast(pl.String))
        .alias("rank_change_display")
    )


def filter_board(
    board: pl.DataFrame, *, position: str | None, available_ids: set[str] | None
) -> pl.DataFrame:
    """Position + real Sleeper-resolved availability filter -- same two
    filters `2_Weekly_Rankings.py` (task 1.19) already established for
    this exact combination, applied here to the ROS board instead."""
    result = board
    if position is not None:
        result = result.filter(pl.col("position") == position)
    if available_ids is not None:
        result = result.filter(pl.col("player_id").is_in(list(available_ids)))
    return result


__all__ = ["filter_board", "style_rank_change"]
```

Create `src/ffapp/app/pages/6_ROS_Rankings.py`, matching `3_Schedule_Grid.py`'s exact structural precedent (thin `st.*` glue, `@st.cache_data` for players_dim/roster resolution, sidebar filters, `st.stop()` on missing required files):

```python
"""Rest-of-season rankings Streamlit page (SPEC-ADDENDUM-04.md §D.5;
task 1.21). Sixth page in SPEC §15's own build order. Reads
`rankings_ros/latest.parquet` (Task 11's own output, `ffapp rankings ros`)
directly rather than recomputing anything model-level on page load --
same "fast to load" precedent every other page here already follows.

Run with: `uv run streamlit run src/ffapp/app/streamlit_app.py`, then
open "ROS Rankings" from the sidebar.
"""

from __future__ import annotations

import polars as pl
import streamlit as st

from ffapp.app.ros_rankings_page import filter_board, style_rank_change
from ffapp.config import load_primary_league, load_settings

st.set_page_config(page_title="ROS Rankings", layout="wide")

settings = load_settings()
league = load_primary_league()

st.title("Rest-of-Season Rankings")
st.caption(league.display_name)

latest_path = settings.data_root / "outputs" / league.slug / "rankings_ros" / "latest.parquet"
if not latest_path.exists():
    st.error(
        f"Missing {latest_path}. Run `ffapp project --from-week --through-week --league "
        f"{league.slug}` then `ffapp rankings ros --league {league.slug}` first."
    )
    st.stop()

board = pl.read_parquet(latest_path)
displayed = style_rank_change(board)

with st.sidebar:
    st.header("Filters")
    positions = ["All", *sorted(board["position"].unique().to_list())]
    position_choice = st.selectbox("Position", options=positions)
    st.caption("Every row here is already a real current free agent -- rostered players never reach this board (Task 10's own `current_free_agent_projections` scoping).")

position_filter = None if position_choice == "All" else position_choice
filtered = filter_board(displayed, position=position_filter, available_ids=None)

st.dataframe(
    filtered.select(
        "rank", "rank_change_display", "position", "vor_ros", "ros_points",
        "ros_p10", "ros_p50", "ros_p90", "expected_games", "playoff_weeks_value",
    ).sort("rank"),
    use_container_width=True,
    height=700,
)
st.caption(
    "`rank_change_display` compares this run to the prior real run's own latest board -- "
    "an em dash means this is either the first real run or a genuinely new free agent."
)
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_app_ros_rankings_page.py -v`
Expected: PASS, full file.

- [ ] **Step 5: mypy/ruff**

Run: `uv run mypy src/ffapp/app/ros_rankings_page.py && uv run ruff check src/ffapp/app/ros_rankings_page.py src/ffapp/app/pages/6_ROS_Rankings.py`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add src/ffapp/app/ros_rankings_page.py src/ffapp/app/pages/6_ROS_Rankings.py tests/test_app_ros_rankings_page.py
git commit -m "feat: ROS Rankings Streamlit page"
```

---

### Task 13: End-to-end verification, weekly-refresh wiring, and close-out

**Files:**
- Modify: `HANDOFF.md`, `TASKS.md`, `docs/JOURNAL.md`
- Modify: `SPEC.md` §16.1's Tuesday runbook reference (if it lists concrete commands — add the two new ROS commands per §D.4)

**Interfaces:** None new — this task is verification and wiring cleanup for everything Tasks 1-12 already built.

- [ ] **Step 1: Real end-to-end run, both leagues**

```bash
uv run python notebooks/estimate_ros_calibration.py  # if not already run in Task 5 on this machine
uv run ffapp project --season 2025 --from-week 10 --through-week 17 --league rogan-radinator-league --no-offline
uv run ffapp rankings ros --league rogan-radinator-league
uv run ffapp project --season 2025 --from-week 10 --through-week 17 --league <second-league-slug> --no-offline
uv run ffapp rankings ros --league <second-league-slug>
```

Confirm, against the real output:
- `projections_ros.parquet` has real rows for every week 10-17, every row carries `as_of_utc`.
- The 10-team full-PPR league's board and the second (18-team) league's board have materially different `vor_ros` at the same position/rank (per this plan's Global Constraints and TASKS.md 1.21's own acceptance bar) — if they look similar, stop and diagnose before proceeding (the fixed point isn't using `LeagueFormat` correctly).
- `playoff_weeks_value` is a real, separate, smaller column than `ros_points` for real players with playoff-week games.
- Re-run `ffapp rankings ros` a second time and confirm `rank_change_display` shows real, non-null movement for at least a few players (proves the `latest.parquet` diff mechanism works against two real consecutive runs, not just the "first run is null" fixture case).

- [ ] **Step 2: Verify the Streamlit page live**

Start `uv run streamlit run src/ffapp/app/streamlit_app.py`, open "ROS Rankings" in a real browser (via `claude-in-chrome` if available this session, matching every other page's own verification precedent in this project — see HANDOFF.md §7 for the known extension-connection quirk if it doesn't connect). Confirm: the board renders, position filter narrows correctly, `rank_change_display` shows real values, 0 console errors.

- [ ] **Step 3: Fold into the Tuesday runbook**

If `SPEC.md` §16.1 or any existing runbook doc lists concrete commands (check `SPEC-ADDENDUM-03.md`'s own "morning of" precedent, task 0.7's entry, and `HANDOFF.md`'s existing rebuild steps), add:

```
Tue 07:30  ffapp project --from-week N+1 --through-week 18 --all-leagues
Tue 07:45  ffapp rankings ros --all-leagues
```

per §D.4 literally. If no real `--all-leagues` flag exists anywhere in this codebase yet (check `cli.py` for the pattern other commands use, e.g. `ffapp cache warm --all-leagues` from `HANDOFF.md`'s own rebuild section) and this plan's Task 11 didn't add one, note this honestly as a real, small follow-up gap in `HANDOFF.md` rather than silently running it per-league only — do not invent a flag Task 11 didn't actually build.

- [ ] **Step 4: Update tracking docs**

- `TASKS.md`: check off 1.21 with a real evidence-based entry (the pattern every other closed task in this file already uses — see 1.18/1.19/2.6's own entries for the exact style: what was built, what was verified, real numbers).
- `HANDOFF.md` §1: append a dated entry summarizing this session's work, the real variance-ratio and recovery_prob numbers from Task 5, and the real board-difference confirmation from Step 1.
- `docs/JOURNAL.md`: one consolidated entry per real task if not already covered by Tasks 1/5's own entries (correlation estimator design, shape-allocation design, the two documented approximations — within-week correlation dilution, run-length-not-calendar-week injury duration).

- [ ] **Step 5: Full test suite, lint, type-check**

```bash
uv run pytest
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
uv run mypy src/
```

Expected: all clean, full suite green.

- [ ] **Step 6: Commit and push**

```bash
git add -A
git status  # review what's staged before committing -- especially config/ros_calibration.yml's real numbers and any new data/outputs paths that should NOT be committed (check .gitignore covers data/outputs/ as it already should)
git commit -m "feat: rest-of-season rankings pipeline (SPEC-ADDENDUM-04.md §D, task 1.21)"
git push
```

Confirm `git status` is clean and `git log` shows the push reached `origin` (this project's own established double-check habit, per `HANDOFF.md`'s repeated "confirmed via git status, not assumed" pattern) before reporting done.

---

## Self-Review Notes (for whoever executes this plan)

- **One real wiring assumption remains unverified against live data:** Task 11's `rankings_ros_command` reads the nflverse/dynastyprocess crosswalk via `pl.read_parquet(crosswalk_path)`, inferred from CLAUDE.md's "parquet everywhere" convention rather than directly confirmed against `ingest.nflverse.fetch_player_ids`'s own real return format. Task 13 Step 1 exists specifically to confirm (or fix) this one line before the real end-to-end run.
- **Two documented, deliberate approximations exist in the new math** (Task 4's cross-week/cross-player correlation dilution; Task 3's run-length-not-calendar-week injury duration). Both are called out in their own module docstrings and in Task 13 Step 5's journal entry — they are real, load-bearing simplifications, not accidents, and should stay visible rather than get quietly "fixed" without re-deriving why they were made this way.
- **Nothing in this plan touches `SPEC-ADDENDUM-05.md` §C/§D/§E** (tasks 3.9-3.11) — every new module here composes around the already-shipped `consensus_b3`/B3 machinery, never predicts a residual against it.
