# TASKS.md — ordered work queue

Work top to bottom. The ordering encodes real dependencies; skipping ahead will cost more time than it saves.

Each task lists the `SPEC.md` section that defines it and the acceptance criteria that close it. Tick a box only when the criteria are demonstrably met. Time estimates assume solo work with agent assistance.

**Legend:** 🔴 blocking (nothing downstream is trustworthy until this passes) · ⏱ estimate

---

# Phase 0 — draft board

**Deadline: before the league draft.** No trained model is involved. Everything in this phase is aggregation, rescaling, and arithmetic. Total ≈ 15–25 hours.

If you are running short on time, the minimum viable path is 0.1 → 0.6 → 0.7 → 0.9 → 0.12. Tiers, survival probabilities, and the live assistant are improvements on a board that already works.

- [x] **0.1 — Repo scaffold** ⏱ 1h
  SPEC §4, §5
  Create the directory structure, `pyproject.toml` with `uv`, `.gitignore` (must exclude `data/` and `.env`), `.env.example`, ruff + mypy config, and a Typer app in `cli.py` with a `--version` command that runs.
  **Done when:** `uv run ffapp --version` prints a version; `ruff check` and `mypy src/` both pass on the empty scaffold.

- [x] **0.2 — Config loading and Sleeper league pull** ⏱ 2h (+ Addendum-01 multi-league, + Addendum-02 offline cache: ~5h actual)
  SPEC §5, §6.3, §18
  Implement `config.py` (YAML loading, path resolution) and `ingest/sleeper.py` with `fetch_user`, `fetch_leagues`, `fetch_league`, `fetch_rosters`, `fetch_users`, `fetch_matchups`, `fetch_transactions`, `fetch_drafts`, `fetch_draft_picks`, `fetch_players` (day-cached), `fetch_trending`. Shared HTTP session with retry/backoff and a descriptive User-Agent.
  Add `ffapp ingest sleeper --season 2026` which resolves username → user_id → league, and writes the league object into `config/league.yml` under `league_cache`.
  **Done when:** `config/league.yml` is populated with real `roster_positions` and `scoring_settings`; `/players/nfl` is cached to disk and a second invocation within 24h does not re-fetch it.

- [x] **0.3 — Player ID mapping** 🔴 ⏱ 3h
  SPEC §7
  Build `ids/mapping.py`: load the published ffverse player-id crosswalk as the base, layer Sleeper's cross-reference fields, fuzzy-match the remainder with `rapidfuzz` (floor 92), apply `config/id_overrides.csv` last. Implement `unmatched_report()` and the `ffapp ids check` command.
  **Done when:** `ffapp ids check` reports zero unmatched players inside the top 300 by ADP, and `tests/test_ids.py` asserts this.

- [x] **0.4 — Scoring keymap and engine** 🔴 ⏱ 4h
  SPEC §8.1–8.3
  Implement `scoring/keymap.py` (`STAT_KEY_MAP`, covering direct stats, bonuses, FG distance buckets, and DST points-allowed buckets) and `scoring/engine.py` (`score_stat_line`, `unhandled_keys`). The engine raises if any non-zero scoring key is unmapped.
  **Done when:** every non-zero key in the league's actual `scoring_settings` is mapped; `unhandled_keys()` returns empty for the real league.

- [x] **0.5 — Scoring golden test** 🔴 ⏱ 2h
  SPEC §8.4
  Pull matchups for every completed week of the league's most recent season, extract `players_points`, independently recompute from nflverse stats, compare.
  **Done when:** ≥99% of player-weeks agree within 0.01 points, every disagreement is logged and explained, and `ffapp scoring validate` exits zero. **Do not proceed past this task with it failing.**

- [ ] **0.6 — LeagueFormat parser** ⏱ 1.5h
  SPEC §8.5
  Parse `roster_positions` into the `LeagueFormat` dataclass, handling FLEX, SUPER_FLEX, REC_FLEX, BN, IR.
  **Done when:** `tests/test_league_format.py` covers standard 12-team, superflex, and a two-flex format with correct `starters`/`flex_slots`/`flex_eligible` output.

- [ ] **0.7 — Consensus projection ingestion and aggregation** ⏱ 4h
  SPEC §9.1, §9.2
  Implement `ingest/rankings.py` for each chosen source (see Open Decision #2). Apply league scoring to per-stat projections *per source* before aggregating. Rank-only sources are mapped onto the value scale via the reference curve. Aggregate with a 20% trimmed mean; retain `dispersion` and `n_sources`.
  **Done when:** a per-player table exists with `proj_points`, `dispersion`, `n_sources`, `coverage`, sourced from at least four providers, with league scoring applied before aggregation (assert this in a test using a non-PPR scoring fixture).

- [ ] **0.8 — Games-played prior** ⏱ 1h
  SPEC §9.3
  Simple positional/age prior for expected games played. Convert source projections to points-per-game, multiply by expected games.
  **Done when:** `expected_games` is populated for every projected player and is visibly below 17 for the positions and age bands where it should be.

- [ ] **0.9 — Value over replacement** ⏱ 3h
  SPEC §9.4
  Implement the fixed-point replacement-level algorithm in `tools/vor.py`, driven entirely by `LeagueFormat`.
  **Done when:** the iteration converges in under 10 passes; `tests/test_vor.py` verifies known replacement ranks for a standard 12-team 1QB league and confirms the QB baseline shifts correctly under a superflex fixture.

- [ ] **0.10 — Tiers** ⏱ 2h
  SPEC §9.5
  Gap-based tiering with the k-means and GMM alternatives behind a common interface, selected by `settings.draft.tier_method`.
  **Done when:** tiers are assigned per position, no tier has fewer than 2 players, and the count is capped at 12.

- [ ] **0.11 — ADP and survival probability** ⏱ 2h
  SPEC §9.6
  Ingest ADP with spread, compute `p_avail_next`, `p_avail_after_next`, and `opportunity_cost` given draft slot and league size.
  **Done when:** given a draft slot, the board shows sensible survival probabilities (near 1.0 for late-ADP players at an early pick, near 0 for the reverse) and `opportunity_cost` is populated.

- [ ] **0.12 — Draft board output** ⏱ 2h
  SPEC §9.7
  Assemble every column from §9.7 into `data/outputs/draft_board_<season>.csv`. Include `bye_week` and, if §14.5 is not yet built, leave `playoff_sos` null rather than faking it.
  **Done when:** `ffapp draft board` produces a sorted, complete CSV you would actually draft from.

- [ ] **0.13 — Streamlit draft board page** ⏱ 3h
  SPEC §15
  Sortable, filterable table with visible tier breaks.
  **Done when:** the page loads in under two seconds and supports filtering by position and tier.

- [ ] **0.14 — Live draft assistant** ⏱ 3h
  SPEC §9.8
  Poll `/draft/{draft_id}/picks`, maintain the available pool, display best-available-by-VOR, tier depth remaining per position, positional run detection, and your starting-lineup gaps.
  **Done when:** tested end to end against a completed mock draft, replaying its picks; the available pool stays correct throughout.

---

# Phase 1 — projections pipeline (season weeks 1–6)

Goal: automated weekly projections that beat baseline B2, with a working evaluation harness. Total ≈ 40–60 hours.

- [ ] **1.1 — nflverse ingestion** ⏱ 3h — SPEC §6.1, §6.2, §6.3
  `nflreadpy` pulls for play-by-play, player weekly stats, snap counts, depth charts, rosters, injuries, schedules. Normalise to the canonical schemas.
  **Done when:** all six canonical interim tables materialise for seasons 2015–2026 and row counts are sane per season.

- [ ] **1.2 — ffopportunity ingestion** ⏱ 1.5h — SPEC §6.1
  Pull precomputed expected fantasy points releases; join `xfp` onto `player_week_usage`. Record the CC-BY-SA licence in the raw directory.
  **Done when:** `xfp` is populated for ≥95% of played player-weeks in the training range.

- [ ] **1.3 — Schedule, betting lines, weather** ⏱ 3h — SPEC §6.2, §10.3
  Derive `home_implied_total` / `away_implied_total` from spread and total; **verify and document the sign convention of `spread_line`**. Build `config/stadiums.csv`. Open-Meteo forecast plus historical archive, with the dome override.
  **Done when:** implied totals are correct on a hand-checked sample of five games; dome games show wind 0.

- [ ] **1.4 — Injury ingestion** ⏱ 2h — SPEC §6.2
  Weekly report status with `date_modified` preserved for as_of logic.
  **Done when:** designations are available per player-week with publication timestamps.

- [ ] **1.5 — Feature registry and as_of contract** 🔴 ⏱ 3h — SPEC §10.1
  `FeatureSpec` dataclass, registry, and the build-time assertions on `lag_weeks` and `available_at_inference`.
  **Done when:** the assertions are active and a deliberately mis-specified feature fails the build in a test.

- [ ] **1.6 — Usage features** ⏱ 5h — SPEC §10.2
  Every feature in the player usage block, with the specified windows.
  **Done when:** spot-checked against known player-seasons (pick a WR1 and confirm target share matches published figures).

- [ ] **1.7 — Team context features** ⏱ 4h — SPEC §10.2
  Including PROE, neutral pace, OL continuity, and the two `vacated_*` features.
  **Done when:** `teammate_vacated_target_share` is non-zero in a week where a known WR1 was ruled out.

- [ ] **1.8 — Opponent adjustment** ⏱ 6h — SPEC §10.4
  Ridge two-way adjustment on rate outcomes per position group, with empirical-Bayes shrinkage and exponential recency weighting. Emit `n_plays` alongside each estimate.
  **Done when:** adjusted values differ materially from raw fantasy-points-allowed rankings, and early-season estimates are visibly shrunk toward the prior season.

- [ ] **1.9 — Feature table build** ⏱ 3h — SPEC §6.2, §10.1
  Assemble the wide table including zero-target rows for players who did not play.
  **Done when:** `features/player_week_features.parquet` exists with `as_of_utc` on every row, and non-played rows are present.

- [ ] **1.10 — Baselines** 🔴 ⏱ 2h — SPEC §12.3
  B0–B3. B3 requires ingesting weekly consensus projections.
  **Done when:** all four produce weekly predictions over the validation range. These are the yardstick; a buggy baseline flatters the model.

- [ ] **1.11 — Snapshot and leakage test** 🔴 ⏱ 3h — SPEC §12.1
  `snapshot()` plus `tests/test_leakage.py`.
  **Done when:** the test passes over a sample of backtest weeks and fails when a deliberate leak is introduced.

- [ ] **1.12 — Walk-forward backtest harness** 🔴 ⏱ 4h — SPEC §12.2
  **Done when:** `ffapp evaluate --seasons 2021 2022 2023 2024 2025` runs end to end and no code path anywhere performs a random split.

- [ ] **1.13 — Metrics module** ⏱ 4h — SPEC §12.4
  Accuracy, ranking, distribution, and both decision-quality metrics (start/sit accuracy and lineup regret). Bootstrap CIs resampled by week.
  **Done when:** every metric is computed per position with observation counts and confidence intervals reported.

- [ ] **1.14 — Availability model** ⏱ 4h — SPEC §11.2
  LightGBM classifier plus isotonic calibration.
  **Done when:** calibration curve is near-diagonal on held-out weeks; Brier score beats a positional base-rate predictor.

- [ ] **1.15 — Conditional points model v1** ⏱ 5h — SPEC §11.3
  Per-position LightGBM with `xfp` as a feature and monotonic constraints where the direction is certain.
  **Done when:** beats B2 on MAE and on Spearman-within-position-week, across at least four validation seasons, with CIs reported.

- [ ] **1.16 — Quantile models** ⏱ 4h — SPEC §11.5
  Five quantiles per position, crossing fix, coverage recalibration, mixture with `p_active`.
  **Done when:** 80% interval empirical coverage is within 5 percentage points of nominal, per position.

- [ ] **1.17 — Evaluation report** ⏱ 3h — SPEC §12.6
  Markdown report with all metrics, baseline comparisons, feature importances, calibration plots. Reports are kept, never overwritten.
  **Done when:** a report is generated and archived under a timestamped directory.

- [ ] **1.18 — Projection output pipeline** ⏱ 2h — SPEC §6.2, §11.8
  `ffapp project --week N` writing `outputs/projections.parquet` with full provenance.
  **Done when:** every row carries `model_version`, `as_of_utc`, `feature_hash`, and git commit.

- [ ] **1.19 — Streamlit weekly rankings page** ⏱ 3h — SPEC §14.1, §15
  **Done when:** the floor/median/ceiling range is visible by default, not hidden in a column.

---

# Phase 2 — decision tools (season weeks 7–18)

Total ≈ 40–55 hours.

- [ ] **2.1 — Lineup optimiser** ⏱ 3h — SPEC §13.1. Done when the ILP produces known-correct lineups on FLEX and superflex fixtures.
- [ ] **2.2 — Correlated weekly simulation** ⏱ 5h — SPEC §13.2. Done when simulated team-total variance is materially lower than the independent-sampling equivalent and the correlation matrix is positive definite after correction.
- [ ] **2.3 — Injury hazard model** ⏱ 4h — SPEC §13.3. Done when `p_miss` is produced per player-week and beats a positional base rate.
- [ ] **2.4 — Season simulator** ⏱ 6h — SPEC §13.4. Done when lineups are set on *projections* and results drawn from *samples* (assert this in a test — it is the most commonly botched detail), and playoff odds sum sensibly across the league.
- [ ] **2.5 — Start/sit assistant** ⏱ 4h — SPEC §14.3. Done when a constructed heavy-underdog scenario recommends the higher-variance option and a heavy-favourite scenario recommends the floor.
- [ ] **2.6 — Waiver wire** ⏱ 5h — SPEC §14.4. Done when value is computed relative to your roster (verify: a high-projection player at a position where you are already deep ranks low), and FAAB guidance is calibrated against your league's transaction history.
- [ ] **2.7 — DST model** ⏱ 4h — SPEC §11.6. Done when it beats B2 for DST and produces a weekly streamer list.
- [ ] **2.8 — SOS and schedule grid** ⏱ 5h — SPEC §14.5. Done when full-season, rest-of-season, and playoff-weeks SOS are all available, low-confidence grades are greyed out, and matchup grade is never the largest element on a card.
- [ ] **2.9 — Trade analyzer** ⏱ 5h — SPEC §14.6. Done when it uses common random numbers across the before/after runs and reports both sides' deltas.
- [ ] **2.10 — News pipeline** ⏱ 6h — SPEC §14.8. Done when a ruled-out RB1 automatically propagates to the backup's projection and the waiver board, and low-confidence items route to manual review.
- [ ] **2.11 — Model health page** ⏱ 2h — SPEC §15. Done when the latest evaluation report, calibration plots, and baseline comparison are visible in the UI.

---

# Phase 3 — offseason (January–August 2027)

- [ ] **3.1 — Decomposed model v2** — SPEC §11.4. Only after v1 is fully evaluated. Compare against v1 on the same harness; keep whichever wins.
- [ ] **3.2 — Trade finder** — SPEC §14.7. Two-stage surrogate-then-simulate filter.
- [ ] **3.3 — Empirical correlation estimation** — SPEC §13.2. Replace the configured correlation constants with values estimated from historical data.
- [ ] **3.4 — Season-long rankings via simulation** — SPEC §14.2. Replaces the Phase 0 static board for 2027.
- [ ] **3.5 — Route/coverage data evaluation** — SPEC §10.5, Open Decision #4. With a full season of logged predictions, quantify what the missing charting data actually costs and decide whether to buy it.
- [ ] **3.6 — Public-readiness audit (optional)** — SPEC §16.5. Licence re-audit, storage layer swap, multi-user `LeagueFormat` handling.

---

## Before you start

Resolve these from `SPEC.md` §17. Several block Phase 0.

- [ ] League format confirmed from Sleeper (Open Decision #1) — blocks 0.6
- [ ] Ranking sources chosen, and which publish per-stat projections (#2) — blocks 0.7
- [ ] Draft slot and date known (#5) — blocks 0.11
- [ ] Waiver type and budget read from Sleeper (#6) — blocks 2.6
