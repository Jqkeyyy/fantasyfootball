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

- [x] **0.6 — LeagueFormat parser** ⏱ 1.5h
  SPEC §8.5
  Parse `roster_positions` into the `LeagueFormat` dataclass, handling FLEX, SUPER_FLEX, REC_FLEX, BN, IR.
  **Done when:** `tests/test_league_format.py` covers standard 12-team, superflex, and a two-flex format with correct `starters`/`flex_slots`/`flex_eligible` output.

- [x] **0.7 — Consensus projection ingestion and aggregation** ⏱ 4h
  SPEC §9.1, §9.2
  Implement `ingest/rankings.py` for each chosen source (see Open Decision #2). Apply league scoring to per-stat projections *per source* before aggregating. Rank-only sources are mapped onto the value scale via the reference curve. Aggregate with a 20% trimmed mean; retain `dispersion` and `n_sources`.
  **Done when:** a per-player table exists with `proj_points`, `dispersion`, `n_sources`, `coverage`, sourced from at least four providers, with league scoring applied before aggregation (assert this in a test using a non-PPR scoring fixture).

- [x] **0.8 — Games-played prior** ⏱ 1h
  SPEC §9.3
  Simple positional/age prior for expected games played. Convert source projections to points-per-game, multiply by expected games.
  **Done when:** `expected_games` is populated for every projected player and is visibly below 17 for the positions and age bands where it should be.

- [x] **0.9 — Value over replacement** ⏱ 3h
  SPEC §9.4
  Implement the fixed-point replacement-level algorithm in `tools/vor.py`, driven entirely by `LeagueFormat`.
  **Done when:** the iteration converges in under 10 passes; `tests/test_vor.py` verifies known replacement ranks for a standard 12-team 1QB league and confirms the QB baseline shifts correctly under a superflex fixture.

- [x] **0.10 — Tiers** ⏱ 2h
  SPEC §9.5
  Gap-based tiering with the k-means and GMM alternatives behind a common interface, selected by `settings.draft.tier_method`.
  **Done when:** tiers are assigned per position, no tier has fewer than 2 players, and the count is capped at 12.

- [x] **0.11 — ADP and survival probability** ⏱ 2h
  SPEC §9.6
  Ingest ADP with spread, compute `p_avail_next`, `p_avail_after_next`, and `opportunity_cost` given draft slot and league size.
  **Done when:** given a draft slot, the board shows sensible survival probabilities (near 1.0 for late-ADP players at an early pick, near 0 for the reverse) and `opportunity_cost` is populated.

- [x] **0.12 — Draft board output** ⏱ 2h
  SPEC §9.7
  Assemble every column from §9.7 into `data/outputs/draft_board_<season>.csv`. Include `bye_week` and, if §14.5 is not yet built, leave `playoff_sos` null rather than faking it.
  **Done when:** `ffapp draft board` produces a sorted, complete CSV you would actually draft from.

- [x] **0.13 — Streamlit draft board page** ⏱ 3h
  SPEC §15
  Sortable, filterable table with visible tier breaks.
  **Done when:** the page loads in under two seconds and supports filtering by position and tier.

- [x] **0.14 — Live draft assistant** ⏱ 3h
  SPEC §9.8
  Poll `/draft/{draft_id}/picks`, maintain the available pool, display best-available-by-VOR, tier depth remaining per position, positional run detection, and your starting-lineup gaps.
  **Done when:** tested end to end against a completed mock draft, replaying its picks; the available pool stays correct throughout.

- [x] **0.15 — Mobile draft page** 🔴 ⏱ 3h
  ADDENDUM-03 §C
  New `src/ffapp/app/pages/5_Draft_Mobile.py`: card-per-player layout (name/position/team/bye, tier/VOR, one-line "why"), best available filtered to the current pool (20–30 players, not 300), three numbers above the fold (best available by VOR, tier depth remaining per position, survival probability to your next pick), position filter as a big-tap-target button row, 10-second auto-refresh with a visible "last updated" timestamp. No hover, tooltips, or side-by-side comparison. Minimum 16px body text.
  **Done when:** the page is usable one-handed on a real phone, shows tier depth remaining, and auto-refreshes during a replayed draft.

- [x] **0.16 — Static HTML export** 🔴 ⏱ 2h
  ADDENDUM-03 §D
  New `ffapp draft export --league <slug> [--out <path>]`: writes a single self-contained `.html` file — full board sorted by VOR with tier breaks, board data embedded as inline JSON, all CSS/JS inline, no CDN links, no external fonts, no network calls. Client-side position filter and text search. Header states the generation timestamp and the age of the ADP and rankings inputs (per ADDENDUM-02 §C.3). Includes draft slot and computed pick numbers. Also export the same board as CSV as a second fallback.
  **Done when:** the file opens on the phone with WiFi and cellular disabled, filters work, and the input-age header is present and accurate.

---

# Phase 1 — projections pipeline (season weeks 1–6)

Goal: automated weekly projections that beat baseline B2, with a working evaluation harness. Total ≈ 40–60 hours.

- [x] **1.1 — nflverse ingestion** ⏱ 3h — SPEC §6.1, §6.2, §6.3
  `nflreadpy` pulls for play-by-play, player weekly stats, snap counts, depth charts, rosters, injuries, schedules. Normalise to the canonical schemas.
  **Done when:** all six canonical interim tables materialise for seasons 2015–2026 and row counts are sane per season. (2015–2025 — 2026 has no nflverse release yet, no games played; re-run once the season starts.)

- [x] **1.2 — ffopportunity ingestion** ⏱ 1.5h — SPEC §6.1
  Pull precomputed expected fantasy points releases; join `xfp` onto `player_week_usage`. Record the CC-BY-SA licence in the raw directory.
  **Done when:** `xfp` is populated for ≥95% of played player-weeks in the training range. (100% for player-weeks with real target/carry opportunity — see HANDOFF for how "played" was scoped.)

- [x] **1.3 — Schedule, betting lines, weather** ⏱ 3h — SPEC §6.2, §10.3
  Derive `home_implied_total` / `away_implied_total` from spread and total; **verify and document the sign convention of `spread_line`**. Build `config/stadiums.csv`. Open-Meteo forecast plus historical archive, with the dome override.
  **Done when:** implied totals are correct on a hand-checked sample of five games; dome games show wind 0.

- [x] **1.4 — Injury ingestion** ⏱ 2h — SPEC §6.2
  Weekly report status with `date_modified` preserved for as_of logic.
  **Done when:** designations are available per player-week with publication timestamps.

- [x] **1.5 — Feature registry and as_of contract** 🔴 ⏱ 3h — SPEC §10.1
  `FeatureSpec` dataclass, registry, and the build-time assertions on `lag_weeks` and `available_at_inference`.
  **Done when:** the assertions are active and a deliberately mis-specified feature fails the build in a test.

- [x] **1.6 — Usage features** ⏱ 5h — SPEC §10.2
  Every feature in the player usage block, with the specified windows.
  **Done when:** spot-checked against known player-seasons (pick a WR1 and confirm target share matches published figures).

- [x] **1.7 — Team context features** ⏱ 4h — SPEC §10.2
  Including PROE, neutral pace, OL continuity, and the two `vacated_*` features.
  **Done when:** `teammate_vacated_target_share` is non-zero in a week where a known WR1 was ruled out.

- [x] **1.8 — Opponent adjustment** ⏱ 6h — SPEC §10.4
  Ridge two-way adjustment on rate outcomes per position group, with empirical-Bayes shrinkage and exponential recency weighting. Emit `n_plays` alongside each estimate.
  **Done when:** adjusted values differ materially from raw fantasy-points-allowed rankings, and early-season estimates are visibly shrunk toward the prior season.

- [x] **1.9 — Feature table build** ⏱ 3h — SPEC §6.2, §10.1
  Assemble the wide table including zero-target rows for players who did not play.
  **Done when:** `features/player_week_features.parquet` exists with `as_of_utc` on every row, and non-played rows are present.

- [x] **1.10 — Baselines** 🔴 ⏱ 2h — SPEC §12.3
  B0–B3. B3 requires ingesting weekly consensus projections.
  **Done when:** all four produce weekly predictions over the validation range. These are the yardstick; a buggy baseline flatters the model.

- [x] **1.11 — Snapshot and leakage test** 🔴 ⏱ 3h — SPEC §12.1
  `snapshot()` plus `tests/test_leakage.py`.
  **Done when:** the test passes over a sample of backtest weeks and fails when a deliberate leak is introduced.

- [x] **1.12 — Walk-forward backtest harness** 🔴 ⏱ 4h — SPEC §12.2
  **Done when:** `ffapp evaluate --seasons 2021 2022 2023 2024 2025` runs end to end and no code path anywhere performs a random split.

- [x] **1.13 — Metrics module** ⏱ 4h — SPEC §12.4
  Accuracy, ranking, distribution, and both decision-quality metrics (start/sit accuracy and lineup regret). Bootstrap CIs resampled by week.
  **Done when:** every metric is computed per position with observation counts and confidence intervals reported.
  Lineup regret was deferred pending task 2.1 (`sim/lineup.py`); 2.1 landed and `lineup_regret()` is now wired into `evaluation/metrics.py`, verified against a real 2021-2025 walk-forward run (`data/outputs/eval/20260813T183533Z/report.md`).

- [x] **1.14 — Availability model** ⏱ 4h — SPEC §11.2
  LightGBM classifier plus isotonic calibration.
  **Done when:** calibration curve is near-diagonal on held-out weeks; Brier score beats a positional base-rate predictor.

- [ ] **1.15 — Conditional points model v1** ⏱ 5h — SPEC §11.3
  Per-position LightGBM with `xfp` as a feature and monotonic constraints where the direction is certain.
  **Done when:** beats B2 on MAE and on Spearman-within-position-week, across at least four validation seasons, with CIs reported.
  **Status:** built, tested, and verified end to end against real 2015-2025 data (four validation seasons, 2021-2024, per SPEC §12.5) — but the real result does **not** clear this task's own bar: MAE is statistically tied with B2 overall (4.883 vs 4.856) and Spearman-within-position-week is worse than B2 at every position (QB 0.496 vs 0.512, RB 0.572 vs 0.610, TE 0.487 vs 0.538, WR 0.576 vs 0.594). Not a bug — feature importances are sane, monotonic constraints apply correctly, no leakage — this is SPEC §11.4's own anticipated v1 outcome ("modelling their sum lets the noise swamp the signal," which is *why* v2 exists). Confirmed with you: ship as documented rather than tune against the validation seasons (CLAUDE.md's own overfitting warning) or silently mark this done. Revisit once v2 (SPEC §11.4, not yet in TASKS.md as its own task) or a real hyperparameter search is in scope.
  **Hyperparameter-search follow-up (2026-08-13):** ran a real 25-trial random search (`notebooks/tune_points_v1.py`/`.log`) over the same shared architecture — n_estimators/learning_rate/num_leaves/min_child_samples/subsample/colsample_bytree/reg_lambda — scored on dev seasons 2018-2020 (strictly before the 2021-2024 validation range and the fully-held-out 2025 season, so the search itself doesn't touch the seasons this task is actually graded on). Result: the best config beat B2 on MAE (5.005 vs B2's 5.107 on the dev set), but **no config out of 25 beat B2 on Spearman-within-position-week** (best 0.550 vs B2's 0.556) — confirming this is an architecture ceiling, not an undertuned model, exactly as SPEC §11.4 anticipated. Stopped here per your instruction rather than spending the full-fidelity `ffapp evaluate` confirmation run on a result the dev-season search already predicts will still miss the bar. Still open; next step if resumed is either that confirmation run (low expectation of clearing the bar) or scoping v2's decomposed pipeline as its own task.
  **Reopened by `SPEC-ADDENDUM-04.md` §A (2026-08-15):** the "architectural ceiling" conclusion above is disputed there as premature — a 25-trial search over the same features/target/loss can only show those specific choices aren't undertuned; it can't distinguish an architecture problem from a feature, target, or measurement problem. Five cheap diagnostics (§A.1–A.5: wrong population for Spearman, B2's own signal missing as a feature, compressed predictions, wrong training objective, non-uniform failure across positions) must be run before 1.15 can honestly be closed as architectural.
  **§A diagnostics run for real (2026-08-15, later):** A.1 confirmed Spearman was being measured on the wrong population (all rows, not startable) — restricting to startable narrowed the gap (all-rows Δ −0.20, startable-only Δ −0.12) but did not close it. A.2 confirmed B2's own value was absent as a feature — adding it plus two more points-history features closed about a third of the startable-scope gap (−0.123→−0.081) but not all of it. A.3 confirmed real prediction compression (std(pred)/std(actual) ≈ 0.57 at every position, under the addendum's own 0.6 threshold). A.4 (a `lambdarank` run) ruled out the training objective as the fix — statistically indistinguishable from the plain model. A.5 found the failure isn't uniform: RB/WR sit close to B2, QB/TE clearly behind (QB Δ −0.19, TE Δ −0.20). None of the five cleared the bar outright, which is why 1.20 (below) was built next per the addendum's own §B fallback.
  **Closed 2026-08-16, per `SPEC-ADDENDUM-04.md` §C's shipping rule: does not clear this task's own bar, B3/consensus ships instead.** Three independent real tests (this task's own Spearman/MAE result above, 1.20's anchored-residual Spearman/regret result below, and 1.20's own regret-optimized blend weight — tuned directly on the decisive metric) all converge on the same finding: nothing in this line of work beats B2 robustly, and B3 beats both B2 and every v1/v2-line model tried on real lineup regret, the metric that actually matters for a shipping decision. Not revisited further inside Phase 1 — the decomposed v2 pipeline (SPEC §11.4) remains the real long-term answer, scoped to Phase 3 per `SPEC-ADDENDUM-04.md` §F. Full account in `docs/JOURNAL.md`.

- [x] **1.16 — Quantile models** ⏱ 4h — SPEC §11.5
  Five quantiles per position, crossing fix, coverage recalibration, mixture with `p_active`.
  **Done when:** 80% interval empirical coverage is within 5 percentage points of nominal, per position.

- [x] **1.17 — Evaluation report** ⏱ 3h — SPEC §12.6
  Markdown report with all metrics, baseline comparisons, feature importances, calibration plots. Reports are kept, never overwritten.
  **Done when:** a report is generated and archived under a timestamped directory.
  Verified for real: `ffapp evaluate --seasons 2021 --seasons 2022 --seasons 2023 --seasons 2024 --seasons 2025` (repeated-flag syntax — see TASKS.md/HANDOFF gotcha on `--seasons`) wrote `data/outputs/eval/20260813T183533Z/report.md` with populated MAE/RMSE/Spearman/start-sit/Brier/lineup-regret tables, real LightGBM feature importances, and real calibration curves.

- [x] **1.18 — Projection output pipeline** ⏱ 2h — SPEC §6.2, §11.8
  `ffapp project --week N` writing `outputs/projections.parquet` with full provenance.
  **Done when:** every row carries `model_version`, `as_of_utc`, `feature_hash`, and git commit.
  Verified for real against an already-played week: `ffapp project --season 2025 --week 10` wrote 445 real projections (0 nulls anywhere, all 4 provenance columns populated, real git commit). Upsert-by-`(season, week)` confirmed against real data too — a second week appended (445+477=922 total), re-running the first week overwrote in place (still 922, not 1367).

- [x] **1.19 — Streamlit weekly rankings page** ⏱ 3h — SPEC §14.1, §15
  **Done when:** the floor/median/ceiling range is visible by default, not hidden in a column.
  Verified for real in a live browser session (Chrome, via claude-in-chrome): real 2025 week 10/11 projections rendered across position tabs, floor-to-ceiling shown as one combined visible range column (not hidden/separate), position + availability filters both confirmed narrowing correctly against real Sleeper-resolved roster data (e.g. Kittle/Bowers/Pitts correctly excluded from the TE free-agent view), switching weeks reloaded real different data, 0 console errors.

- [ ] **1.20 — Anchored residual architecture** ⏱ 6h — `SPEC-ADDENDUM-04.md` §B (supersedes `SPEC.md` §11.3 alongside 1.15)
  Only if §A's five diagnostics don't clear 1.15's bar. Predict the residual against B2 (`target = actual_points − B2(player, week)`, `predict = B2 + model(features)`), not points from scratch — the floor is the baseline by construction, so this can't ship materially worse than B2. Blend weight `w` fitted per position on held-out seasons (`final = w × (B2 + residual_model) + (1 − w) × B2`); `w → 0` automatically for a position the model can't help.
  **Done when:** beats B2 on Spearman-within-position-week over startable rows, across at least four validation seasons, with bootstrap CIs excluding zero — same bar as 1.15, no softening. Report the fitted `w` per position in the evaluation report.
  **Built and evaluated for real (2026-08-15/16), does not clear the bar — closed per `SPEC-ADDENDUM-04.md` §C, B3 ships instead.** `models/residual.py`: `fit_residual_model`/`predict_residual_points` (composes `B2 + model(features)`, floor = B2 by construction, confirmed: QB/TE reproduce B2 exactly since their fitted `w=0`), `fit_blend_weight`/`apply_blend_weight` (grid search maximizing mean weekly Spearman, per position, on dev seasons 2018-2020 — never touching the reported 2021-2024 range). Real fitted weight: `{QB: 0.0, RB: 0.1, WR: 0.3, TE: 0.0}`. Real result on 2021-2024, startable rows only: the blend edges B2 at RB (0.336 vs 0.333) and WR (0.318 vs 0.304), ties it exactly at QB/TE — **no CI excludes zero anywhere**, so this task's own literal bar isn't cleared. Real lineup regret (SPEC §12.4, the "what's it worth in points" question): a single-draw run first showed the blend clearly worse than B2 (37.46 vs 33.79 pts/week); a 20-seed robustness check (`lineup_regret`'s own synthetic-roster draw is unseeded) found that was partly noise — the honest multi-seed picture is a near-wash (mean +1.28 pts worse, wins in 4/20 seeds; narrows to +0.41, a coin flip, under a fair B3-common-support restriction). **The decisive test: fit the blend weight to directly minimize dev-season lineup regret instead of maximizing Spearman.** Grid search over `(w_RB, w_WR)` (QB/TE pinned to 0, already established as real-zero-skill positions) found `w_RB=0.6, w_WR=0.7` with dev regret 28.02 — a dramatic apparent win over both the baseline (39.02) and the Spearman-fit weight (33.49). **That result did not replicate on the separate, held-out 2021-2024 predictions** (mean 33.20 vs B2's 32.41 across 10 seeds, wins in only 4/10) — a clean, textbook overfitting/multiple-comparisons failure (121 combos evaluated against one fixed roster draw will always find something that looks good by exploiting that draw's own noise). This is the most reusable finding in this line of work: tuning directly on the target metric found nothing that survives a real held-out check. Full account, every real number, in `docs/JOURNAL.md`.
  **Shipping B3 made real (2026-08-16), `SPEC-ADDENDUM-04.md` §C.** New `config.ModelSettings.projection_source` (`anchored | direct | baseline_b2 | consensus_b3`, real default `consensus_b3`), `models.predict.project_week` rewritten to dispatch on it, `ffapp project` wired end to end (real `players_dim`/`b3_historical` resolution), Model Health page shows the live source and its real margin over B2 (`config/projection_source_evaluation.yml`). A real, separate blocking question this surfaced — §D's own "B3 gives point estimates, distributions need a spread" — was resolved by a real coverage check (not assumed): an empirical B3-error-based spread cleared task 1.16's own 5pp bar at every position; v1's own quantile spread recentered on B3 (the first interim implementation) badly failed (up to 24pp off on the 80% interval) and was replaced before shipping. Verified live twice against real 2025 week 10 data. Full account in `docs/JOURNAL.md`.

- [x] **1.21 — Rest-of-season rankings pipeline** ⏱ 8h — `SPEC-ADDENDUM-04.md` §D (supersedes `SPEC.md` §14.2)
  Multi-week projection (`ffapp project --from-week N --through-week 18`, features frozen at the current week and carried forward — never rolled forward as if future weeks already happened, that's leakage-by-construction). Aggregation to `ros_points`/`ros_dist`/`expected_games` via `p_play[w] × E[points[w]]` summed across remaining weeks, with fantasy-playoff weeks (`playoff_week_start`..17) reported as a separate column, not folded into the season total. Rest-of-season VOR (§9.4's fixed point, replacement level over remaining value and the current free-agent pool, not preseason). Weekly refresh folded into the Tuesday run (`SPEC.md` §16.1). New Streamlit page `6_ROS_Rankings.py` (`vor_ros`, `p10/p50/p90` season total, playoff-weeks value, rank change since last week).
  **Done when:** a real rest-of-season ranking run completes end to end against real data, `ros_dist` uses task 2.2's own correlation structure (not independent per-week sampling), and the UI page renders live with 0 console errors.
  **Built across a 13-task plan (`.superpowers/sdd/2026-08-16-ros-rankings-pipeline/`), closed out 2026-08-16 with a real end-to-end run against both real leagues.** `ffapp project --from-week N --through-week 18 --league <slug> --no-offline` (`models/predict_ros.py`, `models/ros_shape.py`, `models/ros_consensus.py`) + `ffapp rankings ros --league <slug>` (`tools/ros_aggregate.py`'s Monte Carlo season aggregation over task 2.2's own correlated-week machinery via `sim/persistence.py`, `tools/ros_rankings.py`'s VOR-over-free-agents board), Streamlit page `app/pages/6_ROS_Rankings.py`.
  **Real e2e run, `rogan-radinator-league` (10-team) season 2025 weeks 10-17:** `projections_ros.parquet` — 5242 real rows, all 8 weeks (10-17) present, 0 nulls in `as_of_utc`. `rankings ros` — 622 real free-agent players ranked. Rank 1: RB `00-0041027`, `ros_points=224.71`, `vor_ros=185.24`, `expected_games=8.0`, `playoff_weeks_value=98.68` (a real, separate, smaller column than `ros_points`, as required — playoff weeks 15-17 of 8 total remaining weeks). Second consecutive real run confirmed real non-null `rank_change` movement (e.g. +1/-1 at several ranks 10-20), proving the `latest.parquet` diff mechanism against two real runs, not just the fixture's "first run is null" case.
  **Real e2e run, `bdff-chopped` (18-team) season 2025 weeks 10-17:** same real 8-week/5242-row coverage. `rankings ros` — 766 real free-agent players ranked (a materially larger pool than the 10-team league, as expected — more real rostered players elsewhere). Rank 1: QB `00-0036442`, `vor_ros=203.41`. `playoff_weeks_value=0.0` for every real row — **not a bug**: this league's real `playoff_week_start=0` (not yet configured for the 2026 season on Sleeper), and `tools.sos.playoff_weeks`'s own docstring defines `playoff_week_start <= 0` as an honestly empty playoff-week list. `vor_ros` confirmed materially different by league format for the same real player: `00-0041027` (RB) scored `vor_ros=185.24` (rank 1) in the 10-team league vs `vor_ros=165.93` (rank 2) in the 18-team league — different replacement levels/free-agent pools from `LeagueFormat`, not a fixed-point bug.
  **Three real bugs found live during this task's own e2e verification, none caught by any unit test (each test's own fixtures matched the pre-fix behavior), all fixed:** (1) `cli.py`'s `rankings_ros_command` read `interim/injuries.parquet` (already renamed `gsis_id`→`player_id` by `ingest.nflverse.normalize_injuries`, task 1.4) into `sim.injury.build_hazard_features`, which needs the **raw** nflverse schema (still `gsis_id`) — same raw-table pattern already used two lines below for `rosters_table`/`snap_counts`; fixed to call `nflverse.fetch_injuries` like its neighbors. (2) `sim.injury.fit_hazard_model` crashed on a real `ValueError: Input X contains NaN` — 157 of 96,081 real 2015-2025 hazard-grid rows have a null `age` (a real missing nflverse `birth_date` or an off-`schedule` roster week), and scikit-learn's `LogisticRegression` refuses NaN natively; fixed with `SimpleImputer(strategy="median")` ahead of `StandardScaler` in the numeric pipeline, preserving `predict_p_miss`'s one-row-in/one-row-out contract. (3) The real, more consequential one: `models.predict_ros.project_week_range` (tasks 7/8, already committed) deliberately leaves a real `(player, week)`'s `mean`/quantile grid null when no empirical error-quantile bucket exists yet for that position/tau ("never guess, leave it null" — that module's own comment) — but `tools.ros_aggregate.aggregate_ros` (task 9) let that single null propagate into a `NaN` season total for the whole player once summed across weeks, and `tools.ros_rankings.build_ros_board`'s sort then put every `NaN`-`vor_ros` player at the very top of the board (confirmed live: 49 of 622 real `rogan-radinator-league` players occupied ranks 1-49 ahead of every real, well-formed projection before the fix). Fixed by excluding a null-`mean` `(player, week)` row from that week's own marginals in `aggregate_ros` rather than fabricating a point value for it — that player's `totals` array simply keeps its already-zero-initialised default for that one week, the same numeric treatment a bye/unavailable week already gets elsewhere in the same aggregation, with every other real week for that player untouched. All three fixes covered by real/updated tests (`tests/test_cli_rankings.py`, `tests/test_sim_injury.py`, `tests/test_tools_ros_aggregate.py`), full suite green.
  **Streamlit page (`6_ROS_Rankings.py`) not verified in a live browser this session** — `claude-in-chrome` had no connected browser on this machine (`list_connected_browsers` returned `[]`, the same real quirk `HANDOFF.md` §7 already documents). Fallback per this task's own brief: confirmed a real `HTTP 200` on the page's own route (`/ROS_Rankings`) while `uv run streamlit run src/ffapp/app/streamlit_app.py` was live, plus the page's own pure functions carry real test coverage (`tests/test_app_ros_rankings_page.py`). Not the same as an actual rendered-in-browser check — owed next time this machine has Chrome properly signed in, matching the Model Health page's own still-open item from the prior session.
  **Real, small follow-up gap, not built this session:** no `--all-leagues` flag exists on `ffapp project`/`ffapp rankings ros` (only `ffapp cache warm --all-leagues`/`ffapp scoring validate --all-leagues` have it) — `SPEC-ADDENDUM-04.md` §D.4's own literal Tuesday-runbook text names it for both new rows; `SPEC.md` §16.1 was updated with the real per-league invocation instead of inventing a flag that doesn't exist, see `HANDOFF.md`.
  Full account, every real number, in `docs/JOURNAL.md`.

- [ ] **1.22 — Cold-start source schedule** ⏱ 2h — `SPEC-ADDENDUM-04.md` §E
  Weeks 1–3 default `projection_source` to `consensus_b3` (public projections have real information — beat reporting, camp news — this project's feature set has no access to yet). Weeks 4–6 blend toward the model as sample accumulates (`w = min(1, (week − 3) / 4)`, tune on historical early-season data). Week 7+ the model source proper. Configured schedule, not a manual switch to remember on a Tuesday in September. Verify `prior_season` windows are actually populated for rookies/role-changers early on, not silently null.
  **Done when:** the schedule is configured (not hardcoded per CLAUDE.md rule 5) and a real early-season run resolves to the correct source for a given week without manual intervention.

---

# Phase 2 — decision tools (season weeks 7–18)

Total ≈ 40–55 hours.

- [x] **2.1 — Lineup optimiser** ⏱ 3h — SPEC §13.1. Done when the ILP produces known-correct lineups on FLEX and superflex fixtures.
- [x] **2.2 — Correlated weekly simulation** ⏱ 5h — SPEC §13.2. Done when simulated team-total variance is materially lower than the independent-sampling equivalent and the correlation matrix is positive definite after correction.
- [x] **2.3 — Injury hazard model** ⏱ 4h — SPEC §13.3. Done when `p_miss` is produced per player-week and beats a positional base rate.
- [x] **2.4 — Season simulator** ⏱ 6h — SPEC §13.4. Done when lineups are set on *projections* and results drawn from *samples* (assert this in a test — it is the most commonly botched detail), and playoff odds sum sensibly across the league.
- [x] **2.5 — Start/sit assistant** ⏱ 4h — SPEC §14.3. Done when a constructed heavy-underdog scenario recommends the higher-variance option and a heavy-favourite scenario recommends the floor.
- [x] **2.6 — Waiver wire** ⏱ 5h — SPEC §14.4. Done when value is computed relative to your roster (verify: a high-projection player at a position where you are already deep ranks low), and FAAB guidance is calibrated against your league's transaction history.
  Verified against real data: `tests/test_tools_waivers.py::TestBuildWaiverBoard::test_ranks_by_roster_relative_value_not_raw_projection` proves the acceptance bar literally with a hand-built fixture; a real end-to-end run against the completed 2025 season (`rogan-radinator-league`, real rosters/players_dim/player_week_features) produced a sane 1366-candidate board with proportional FAAB bids. Real FAAB calibration data pulled from `bdff-chopped`'s actual 2025 transaction history (160 real bids, median $10/$1000 budget) — see `docs/JOURNAL.md`'s task 2.6 entry.
- [x] **2.7 — DST model** ⏱ 4h — SPEC §11.6. Done when it beats B2 for DST and produces a weekly streamer list.
- [x] **2.8 — SOS and schedule grid** ⏱ 5h — SPEC §14.5. Done when full-season, rest-of-season, and playoff-weeks SOS are all available, low-confidence grades are greyed out, and matchup grade is never the largest element on a card.
  New `tools/sos.py` (positional SOS sums, the team x week grid, per-position-group matchup detail) + `app/schedule_grid_page.py` (roster-highlight resolution, heatmap styling) + `app/pages/3_Schedule_Grid.py` (three tabs: Positional SOS, Schedule Grid, Matchup Detail). Verified live in a real Chrome session (Streamlit, `rogan-radinator-league`, real 2025 data): full/rest-of-season/fantasy-playoff SOS all populated with real confidence flags, the heatmap renders a real blue/gray/red diverging scale with real byes shown as blocked "bye" cells and low-confidence real values marked (not hidden), the roster-highlight toggle correctly resolved real Sleeper-rostered teams, matchup detail shows plain-text usage-trend/matchup breakdown side by side (never a colour badge, per SPEC's "required honesty"), 0 console errors throughout.
  **A real design bug caught live, not in a fixture, and fixed before shipping:** the first confidence-threshold design reused SPEC §10.4's `k=250` shrinkage constant directly as the UI's "grey out low n_plays" cutoff — wrong, since that `k` is the ridge model's own *cumulative trailing* play count, a different, much larger quantity than the single-week `n_plays` this module actually reports (task 1.8's own JOURNAL entry already names them as "a different, legitimate, unrelated quantity"). A flat 250 threshold greyed out every real cell, every position, every week (confirmed live: WR's own real single-week `n_plays` tops out at 37 in 2025). Fixed with a per-`position_group` threshold computed from that group's own real bottom-quartile `n_plays` distribution — full account in `docs/JOURNAL.md`.
- [x] **2.9 — Trade analyzer** ⏱ 5h — SPEC §14.6. Done when it uses common random numbers across the before/after runs and reports both sides' deltas.
- [x] **2.10 — News pipeline** ⏱ 6h — SPEC §14.8. Done when a ruled-out RB1 automatically propagates to the backup's projection and the waiver board, and low-confidence items route to manual review.
  New `ingest/news.py` (real RSS ingestion — ESPN/CBS Sports/Yahoo NFL feeds, all three confirmed live; the real Anthropic API structuring call, `output_config.format` structured outputs per SPEC's own JSON schema literally; confidence/name-resolution routing to a real upserted-by-guid manual review queue) + `tools/news_propagation.py` (the real second-order cascade: `add_vacated_shares`, task 1.7's own already-validated mechanism, fed one synthetic `Out` row instead of an official injury report; `models.predict.project_week` re-run on the patched table; the real handcuff surfaced in exactly the shape `tools.waivers.build_waiver_board`'s own `projection_by_player` already expects).
  **Not verified live against the real Anthropic API this session** — `ANTHROPIC_API_KEY` is empty in this machine's `.env` (checked before starting, confirmed with you). The structuring call is tested against a mocked client (CLAUDE.md's own "no live network calls in tests" rule requires this anyway) and built to the real, documented API surface, not guessed. A real live smoke test is the natural next step once a key is supplied.
  **Two real bugs caught only by running the propagation cascade against real 2015-2025 data, not by any fixture test:** (1) `pl.concat(..., how="vertical_relaxed")` against the real `interim/injuries.parquet` (9 real columns) failed outright — my own test fixture's `injuries` table happened to share the synthetic row's exact 4-column shape, masking a real schema mismatch; fixed with `diagonal_relaxed`. (2) A bare Python `int` for `season`/`week` in the synthetic row infers `Int64`, but every real nflverse-derived table in this project uses `Int32` — this broke `add_vacated_shares`'s own `join_asof` with a dtype-mismatch error a same-`Int64` fixture never exercised; fixed with an explicit `Int32` schema. Both are now covered by dedicated regression tests using realistically-shaped fixtures, not just the original narrow ones.
  **Real historical verification against 2025 week 5 (Bucky Irving ruled out, Tampa Bay):** rewound `player_week_features`/`injuries` to simulate "before the official injury report," fed a synthetic `Out` event through the real pipeline. Recomputed vacated shares (0.438/0.669) closely matched the real official values task 1.7's own mechanism separately produced that week (0.439/0.620). The real handcuff, Rachaad White, was correctly identified, and his recomputed projection jumped from a real 6.66 (nobody-out baseline) to 11.03 (post-event) — a genuine, substantial, correctly-directed move, purely from the synthetic pre-report signal. (Honestly undershoots his real actual output that week, 23.1 points — a known limitation of the underlying conditional-points model already documented in task 1.15's own entry, not a bug in this propagation mechanism.) The recomputed value was then fed through the real, already-shipped `tools.waivers.build_waiver_board` (task 2.6) unmodified and produced a real, sane suggested FAAB bid — confirming the full real cascade end to end.
  **Not built:** "adjust the team's projected pass rate if the change is at QB" (SPEC's own fourth cascade step) — no existing mechanism in this project computes that, and this task's own literal TASKS.md acceptance bar never names it either; a documented scope boundary, not an oversight (see `docs/JOURNAL.md`). No manual-review UI (no UI task exists yet for news specifically).
- [x] **2.11 — Model health page** ⏱ 2h — SPEC §15. Done when the latest evaluation report, calibration plots, and baseline comparison are visible in the UI.
  New `app/model_health_page.py` (`list_reports`/`latest_report`/`load_report_markdown`) + `app/pages/4_Model_Health.py`, SPEC §15's fourth page. Reads task 1.17's own already-built `data/outputs/eval/<timestamp>/report.md` directly and renders it via `st.markdown()` — no new chart rendering, since 1.17's own already-confirmed decision renders "calibration plots" as a markdown table, not a plotted image (see `evaluation/report.py`'s own module docstring); re-litigating that here would be scope creep, not this task's own job. A sidebar selector defaults to the latest real report while surfacing older ones too (SPEC §12.6: "Reports are kept, not overwritten — the history... is the most valuable artefact of the offseason").
  Verified live in a real Chrome session: the real report (seasons 2021-2025, git commit `c134529`) rendered in full — every real metric table (MAE/RMSE/Spearman/start-sit-accuracy/Brier/lineup-regret/top-k-precision) with real baseline-vs-model rows, real per-position LightGBM feature importances, and the real calibration tables for both availability predictors — 0 console errors. The report selector correctly showed only the one real directory that actually has a `report.md` (two other `data/outputs/eval/` directories are leftover partial runs with only `predictions.parquet`, correctly excluded — proving the "no report.md inside" filter works against real, not just fixture, data).

---

# Phase 3 — offseason (January–August 2027)

- [ ] **3.1 — Decomposed model v2** — SPEC §11.4. **Reopened (2026-08-15) — the prior "stopped" entry below was a real misread of the project owner's own instruction, corrected the same session, not a re-litigated decision.** Stage 1 (team environment) and Stage 2 (opportunity) are both built and evaluated for real against 2021-2025 data, every diagnosed bug fixed (a real row-set-mismatch, a null-target convention gap, a real `rz_touches` units mismatch), and a blend probe tried on top. Stage 1 beats its real bar (`trailing_ewm_4`) on both `team_plays`/`pass_rate`, but loses to a plain `league_mean` on both. Stage 2 loses to its real bar (`trailing_raw`) on all three outputs (`targets`/`carries`/`rz_touches`), and an equal-weight blend of composition + `trailing_raw` still lost to `trailing_raw` alone by ~25-36% even after confirming composition carries *some* real signal. **Proceeding to Stage 3 (efficiency priors) anyway, on the project owner's explicit instruction to continue building the full 4-stage pipeline** despite the mixed Stage 1/2 evidence — the real numbers/CIs and the honest case against continuing are preserved in `docs/JOURNAL.md`'s Stage 1/Stage 2 entries and the (superseded) closing entry, not erased.
  **Stage 3 (efficiency priors) built, evaluated, and simplified — real evidence removed its opponent-adjustment offset, and the corrected result clears CLAUDE.md rule 6 cleanly.** SPEC's own literal RULE (Empirical Bayes shrinkage, `prior_weight`=50 targets/80 carries, no trained model) — split by touch type into four outputs. The design originally also applied an additive opponent-adjustment offset; a real, position-broken-out paired-bootstrap ablation against 2021-2025 data (re-verified on fixed code after a real null-handling bug in the same step was caught and fixed) showed it measurably hurt `yards_per_target` (RB, WR) and `yards_per_carry` (QB — the single largest harm measured, RB), and was neutral everywhere else including TE (whose own point estimate sat between two confirmed-harmful positions, not evidence of a real different effect) — removed for every position on that evidence, not kept as a TE-specific exception. **Real result after removal: `shrunk_model` beats `trailing_raw` (the real bar) on 3 of 4 outputs** — `yards_per_target` (4.3203 vs 4.7230), `td_rate_per_target` (0.0740 vs 0.0747), `yards_per_carry` (2.1573 vs 2.2653) — **and loses on the fourth, `td_rate_per_carry` (0.0515 vs 0.0487, unchanged by the offset removal). It now also beats `league_mean` (the sanity floor) on all 4 of 4 outputs** (up from 2 of 4 with the offset still applied) — `yards_per_target` (4.3203 vs 4.3312) and `yards_per_carry` (2.1573 vs 2.1801) both flipped from narrow losses to real wins. Full numbers/CIs, the full ablation account, and the design doc's own low-touch-count noise caveat are in `docs/JOURNAL.md`'s newest entries. Stage 4 (Monte Carlo recombination) is the one remaining stage — not started, a real decision point for the project owner per this plan's own "after this plan" instruction, see `HANDOFF.md` §1.
- [ ] **3.2 — Trade finder** — SPEC §14.7. Two-stage surrogate-then-simulate filter.
- [ ] **3.3 — Empirical correlation estimation** — SPEC §13.2. Replace the configured correlation constants with values estimated from historical data.
- [ ] **3.4 — Season-long rankings via simulation** — SPEC §14.2. Replaces the Phase 0 static board for 2027.
- [ ] **3.5 — Route/coverage data evaluation** — SPEC §10.5, Open Decision #4. With a full season of logged predictions, quantify what the missing charting data actually costs and decide whether to buy it.
- [ ] **3.6 — Public-readiness audit (optional)** — SPEC §16.5. Licence re-audit, storage layer swap, multi-user `LeagueFormat` handling.
- [ ] **3.7 — Weekly DST/K streamer tab** — not in SPEC §15's original page list; requested 2026-08-14, explicitly deferred ("not now but later"). The project owner streams both positions week to week rather than drafting for season-long value (see task 0.9's VOR follow-up, same session, for the real historical data confirming why) and wants a page/tab ranking DST and K by *that week's* projection, not a season total. `models.dst.weekly_streamer_list` (task 2.7) already produces exactly this for DST from the walk-forward predictor's own output; K has no model yet (SPEC §11.7 — deliberately minimal, implied team total/dome/opponent red-zone-to-TD rate, not built). Natural home is SPEC §15 page 5 (Waivers), which also isn't built yet as a Streamlit page (only the `tools.waivers` backend, task 2.6, exists) — likely makes sense to build together rather than as two separate pages.

- [x] **3.8 (new, ⏱ 3h) — In-season prediction logging** — `SPEC-ADDENDUM-05.md` §B — **urgent, before Week 1, the one item in that addendum that is not offseason work.** *(Renumbered from the addendum's own literal "3.7" — that number was already taken by the DST/K streamer tab above when the addendum was added 2026-08-16; the mapping is by section reference, §B/§C/§D/§E below, not by number.)* Append-only log, one row per (league, season, week, player, source), written by the Tuesday/Sunday pipeline runs: keys + `as_of_utc` + `run_label` (tuesday/thursday/sunday) + `projection_source` + `b3_mean`/`b3_q10..q90` + **per-source `per_source_points`** (not just the trimmed mean — §E depends on it) + `n_sources`/`dispersion` + `model_mean` (logged even though the model isn't shipping — pairs with `b3_mean`/`actual_points` for §C's future comparison) + `b2_mean` + `p_active` + `actual_points` (backfilled) + `model_version`/`feature_hash`/`git_sha`. Lives at `data/outputs/<league_slug>/prediction_log/season=2026/week=NN.parquet` plus a `latest` pointer — **committed to git**, same non-reproducibility reasoning as the rankings gitignore exception (weekly consensus projections are not re-fetchable; nflverse data is). `ffapp log backfill --week N` fills `actual_points` after games complete; a missing backfill warns loudly, never silent nulls that look like zeros.
  **Done when:** the Tuesday and Sunday pipeline runs write per-player, per-source rows with `as_of_utc`, backfill populates `actual_points` for a completed week, the log is committed to git, and a missing backfill warns loudly.
  **Built and verified live, 2026-08-16.** New `tools/prediction_log.py` + `ffapp log week`/`log backfill`/`log check-sources` CLI commands. **A real bug was found during live verification, not caught by unit tests:** the six non-`fantasypros` sources (`espn`/`cbs`/`fantasysharks`/`fftoday`/`footballguys`/`draftsharks`) return real **season-long** point totals through `draft.board`'s own scoring path (correct for the draft board's own preseason use), not weekly ones — a real 2025 week-10 QB showed `espn`=243.7/`cbs`=213.3/`fantasysharks`=299.9 (season-total order of magnitude) alongside `fantasypros`/`model_mean`/`b3_mean` all agreeing ~15-18 (a real single-week number), which had been silently averaged into one `dispersion` figure. **Fixed by splitting into two real namespaces** — `weekly_source_points`/`season_source_points`, with `n_sources`/`dispersion` computed from `WEEKLY_SOURCES = ("fantasypros",)` only and a new `n_season_sources`/`season_dispersion` from `SEASON_SOURCES` (the other six) — per-source `payload_sha256`/`fetched_at_utc`/`refresh_status`, none dropped (the season set is a real, separately-perishable ROS signal for 3.9's own future work, not noise). `check_sources` also reports each season source's `season_trend` (`declining`/`flat`/`insufficient_data`, needs ≥3 real logged weeks) — resolves whether a season number is a genuine forward-looking signal or a stale preseason snapshot. A second real bug surfaced only by a live `--no-offline` run: `_fetch_fantasypros` originally called `models.baselines.fetch_b3_for_week` (which internally re-fetches the FantasyPros commit list) *in addition to* fetching that same commit list a second time just to hash it — a real `403 rate limit exceeded` from `api.github.com` was hit live from the doubled paginated call. Fixed by fetching the commit list/snapshot exactly once and building b3 via `baselines.add_b3_fp_weekly_consensus` directly. Verified end to end against real 2025 week-10 data: 445 rows, 0/7 sources failed, all 7 sources (including `fantasypros`, previously null) carry a real `payload_sha256`, `dispersion` is honestly 0.0 across all 445 rows (only one real confirmed-weekly source exists today — not a bug), `season_dispersion` shows real spread (mean 18.3, max 139.5). Two quick verification items from the same session: the weather re-ingest (`is_dome`/`wind_mph`/`precip_prob`/`temp_f`) is confirmed live at 0% null across all 94,422 rows, seasons 2015-2025 (previously 100% null per `SPEC-ADDENDUM-04.md` §A); task 0.16 (offline HTML draft export) was already verified live in an earlier session, no action needed. **Also discovered, not fixed (out of scope tonight):** the "rankings gitignore exception" this task's own commit-to-git reasoning cites as precedent does not actually exist — `data/raw/rankings/` (194 real files) is still caught by the blanket `data/` ignore rule; only the prediction-log path itself was given a real `.gitignore` exception this session. 17 tests, `mypy`/`ruff` clean, full suite (1111 tests) green. Full account in `docs/JOURNAL.md`.

- [ ] **3.9 (new, ⏱ 6h) — B3-anchored restricted residual model** — `SPEC-ADDENDUM-05.md` §C — offseason (January). *(Addendum's own "3.8".)* Same structure as task 1.20 but anchored on B3 instead of B2 (`target = actual_points − b3_mean`, `predict = b3_mean + residual_model(features)`, blended with a fitted `w`). **Feature-restricted by design** — only signals plausibly absent from consensus (`def_adj_*` opponent rates, weather, `teammate_vacated_*_share`, `implied_team_total`/`spread`, `rest_days`/`is_primetime`/`week_number`, `dispersion`/`n_sources`) — explicitly excludes season-long usage/snap share/target share/talent proxies/prior-season production, since consensus already encodes those and re-deriving them just adds variance (the same failure mode that sank Stage 2 and the Stage 3 opponent-adjustment offset).
  **Done when:** beats B3 (not B2) on **both** startable-rows Spearman-within-position-week **and** lineup regret, under all six of §F's gates. Beating one and not the other does not ship.

- [ ] **3.10 (new, ⏱ 8h) — Gated residual** — `SPEC-ADDENDUM-05.md` §D — offseason (January, after 3.9). *(Addendum's own "3.9".)* Stage 1: predict `|actual − b3_mean|` (expected consensus-error magnitude); Stage 2: apply §C's residual only where predicted error is in the top decile, ship B3 unchanged elsewhere. `dispersion` across sources is the most promising single Stage-1 input (free — real source disagreement already logged by 3.8). Report the real gate fire rate and lineup regret *restricted to gated rows* separately from the aggregate — an aggregate metric would hide a gate that meaningfully improves a small fraction of decisions.

- [ ] **3.11 (new, ⏱ 4h) — Learned per-position source weighting** — `SPEC-ADDENDUM-05.md` §E — cheap, can run before the offseason (Nov-Dec) if the itch is unbearable; otherwise January. *(Addendum's own "3.10".)* The aggregator's 20% trimmed mean is untested — fit non-negative, sum-to-one weights per position on held-out seasons (needs 3.8's real per-source log), compare against the trimmed mean under §F's gates. Constrained fit only — an unconstrained one produces negative weights that look great in-sample and generalize terribly. Highest expected-value item in the addendum after 3.8 (logging) — low risk, improves what's actually shipping.

---

## Before you start

Resolve these from `SPEC.md` §17. Several block Phase 0.

- [x] League format confirmed from Sleeper (Open Decision #1) — blocks 0.6
- [x] Ranking sources chosen, and which publish per-stat projections (#2) — blocks 0.7
- [x] Draft slot and date known (#5) — blocks 0.11. Slot 3 of 10, Aug 22 2026 (confirmed live: `draft.start_time` = 2026-08-23 01:00 UTC). This league trades draft picks and is a 1-keeper league — neither is in SPEC §9.6's plain-redraft model; see task 0.11's HANDOFF entry.
- [ ] Waiver type and budget read from Sleeper (#6) — blocks 2.6
