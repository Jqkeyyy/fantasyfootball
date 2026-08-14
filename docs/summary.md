# Project summary

**Last updated:** 2026-08-13. For live state and next steps, see `HANDOFF.md`. For per-task evidence and gotchas, see `docs/JOURNAL.md`. This file is neither — it's a standing narrative snapshot of where the project actually is, meant to be read cover to cover.

---

## What this is

A personal fantasy football decision-support system for one Sleeper league (`rogan-radinator-league`), built around a self-trained player projection model. Single user, local-first, no accounts, no hosting, no Sleeper writes — the system recommends, the human executes. Everything runs under this league's exact scoring rules and roster format, never a generic PPR/standard assumption.

The real 2026 draft is **Aug 22, 2026** — 9 days from today. Phase 0 (the draft board) has been done for a long time specifically so it wouldn't be at risk from later phases running long.

---

## Current state: Phase 2 complete, Phase 1 has one open item

| Phase | Scope | Status |
|---|---|---|
| **0 — Draft board** | Tiers, VOR, pick-survival, live draft mode | ✅ Complete |
| **1 — Projections pipeline** | Ingestion → features → hurdle model → evaluation harness | 18/19 tasks done; 1 open (see below) |
| **2 — Decision tools** | Sim layer, start/sit, waivers, DST, schedule/SOS, trades, news, model health | ✅ Complete (all 11 tasks) |
| **3 — Offseason work** | Decomposed model v2, trade finder, empirical correlations, sim-based season rankings, route-data cost/benefit, public-readiness audit | Not started |

Phase 1's one open item is **1.15, the conditional points model**: built, tested, and verified against five real seasons of walk-forward backtesting — but it doesn't clear its own bar. A follow-up 25-trial hyperparameter search (this session) confirmed the gap isn't a tuning problem: MAE can be nudged past the baseline, but ranking quality (Spearman-within-position-week, the metric that actually matters for a rankings product) never beats the simple baseline no matter how the model is tuned. That's a real, documented finding about the model's architecture — not a bug, and not silently patched by tuning against validation data. The fix, if pursued, is the decomposed v2 architecture already scoped in Phase 3 (§11.4), not more tuning.

---

## What it actually does today

**Draft prep**
- A ranked, tiered draft board (VOR-based) blending consensus sources, refreshed live during the draft with pick-survival probabilities and positional-run detection.

**Weekly decisions**
- Full-distribution projections (floor / median / ceiling, not just a point estimate) for every rosterable skill-position player and DST, refreshed weekly.
- A start/sit assistant that compares realistic win-probability, not just raw projected points.
- Waiver-wire rankings scored as *value added to your specific roster* (a 15-point free agent is worth nothing if your bench is already that deep), with FAAB bid guidance calibrated against this league's own real bidding history.
- A schedule/strength-of-schedule heatmap — full-season, rest-of-season, and fantasy-playoff-only views, opponent-adjusted (not the "fantasy points allowed" metric public sites use, which is confounded by garbage time and offense quality) — with confidence-greyed cells so an early-season three-game sample doesn't look as trustworthy as a full one.
- A trade analyzer that runs the season simulator before and after a proposed trade (same random seed both times) and reports both sides' real change in win probability and playoff odds — not just a summed-value comparison, which misses that lineups have fixed slots.

**Automated news reaction**
- Real RSS ingestion (ESPN, CBS Sports, Yahoo) feeds an LLM structuring step that extracts injury/role-change events into a strict schema, with low-confidence or unresolvable items routed to manual review rather than silently trusted.
- A ruled-out player's news event automatically recomputes the vacated target/carry share for their real teammates and re-projects them — the actual mechanism a Wednesday injury report would trigger days later, just triggered by the earlier signal instead. Verified against a real 2025 case (Bucky Irving's real season-ending injury): the recomputed values closely matched what the official injury-report pipeline separately produced, and the correctly-identified handcuff's recomputed projection flowed all the way through to a real waiver-wire bid.

**Self-monitoring**
- A model health page surfacing the latest evaluation report — every metric per position, versus every baseline, with confidence intervals, feature importances, and calibration — so the system's own accuracy stays visible rather than quietly assumed.

---

## How it's going

**The projection engine beats its baselines where it matters.** Evaluated over five real seasons (2021–2025) of strict walk-forward backtesting — never a random split — against three baselines (positional average, season-to-date average, trailing 4-week average):

- The availability model (will this player play?) is well-calibrated: Brier score 0.148 vs. 0.220 for a naive positional base rate.
- The quantile/floor-ceiling model hits its stated coverage target on every position, within 2.1 percentage points.
- The DST model beats its baseline by a real ~9–10% margin on both MAE and RMSE.
- The conditional points model (task 1.15, above) is the one real exception — a documented, not hidden, shortfall.

**The discipline that makes those numbers trustworthy:** no random train/test splits anywhere in the codebase; every feature respects an `as_of` cutoff so nothing trains on data it wouldn't have had at inference time; the scoring engine is validated against Sleeper's own computed points before anything downstream trusts it. Several real, load-bearing bugs (relocated-franchise team-code mismatches, a schema/dtype gap in a live-data join, a UI confidence threshold that was silently greying out every cell) were caught specifically because verification happened against real historical data, not synthetic fixtures alone, and are documented rather than quietly fixed and forgotten.

**Built substantially out of strict task order, deliberately.** Several Phase 2 tools were built before Phase 1 fully closed, once it was confirmed each one's own real prerequisites didn't actually depend on the unfinished piece (e.g., the simulation layer and trade analyzer needed the lineup optimizer and Sleeper data, not the conditional points model). Each of those calls was confirmed rather than assumed — the project's own standing rule is that ambiguity resolved by guessing is more expensive than asking.

**What's genuinely unverified right now:** the LLM news-structuring call has never been exercised against the real Anthropic API — no key is configured on this machine — so it's tested against a mocked client only, built to the real documented API surface but not yet proven end-to-end live.

---

## What's left

- **1.15's resolution** — either accept the direct-regression architecture's ceiling and move on, or scope the decomposed v2 pipeline (Phase 3) as real work.
- **Phase 3**, entirely: a decomposed points model, a trade finder (candidate generation + full-sim evaluation), empirically-estimated correlation constants for the season simulator, simulation-based season-long rankings (replacing the static Phase 0 board for the 2027 offseason), a real cost/benefit read on buying route-participation charting data, and an optional public-readiness audit. None of this blocks the 2026 season — it's explicitly offseason-scoped work.
