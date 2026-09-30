# Current project state

**Authoritative as of:** 2026-09-26

This is the single source of truth for what is shipping, what is trusted, and what comes next.
`TASKS.md` remains the detailed implementation backlog. `docs/JOURNAL.md` and `HANDOFF.md` are
historical evidence, not current-status documents.

## Shipping product

- League-aware weekly and rest-of-season projections for the configured Sleeper leagues.
- A Weekly Actions cockpit for lineup optimization, start/sit simulation, waiver upgrades,
  K/DST streaming, data health, and changes since the previous refresh.
- Draft, trade, schedule/SOS, model-health, and chopped-league decision tools.
- Tuesday, Thursday, and Sunday refresh automation with manifests and Discord alerts.
- In-season source prediction logging and actual-points backfill.

## What is trusted

- Consensus/B3 is the shipping projection anchor. Custom models do not replace it unless they
  beat the documented walk-forward baseline.
- Scoring validation, point-in-time feature rules, identity gates, and walk-forward evaluation
  remain blocking requirements.
- Healthy refreshes preserve a checksum-verified `last_known_good/` copy of decision artifacts.
- Weekly recommendations can be recorded, marked followed or rejected, and settled against
  actual points so decision regret can be measured instead of model error alone.

## Immediate operational state

- The always-on Ubuntu home server is the production host. Native cron keeps Streamlit and the
  Discord bot running, performs kickoff and transaction checks every five minutes, sends a daily
  briefing, and runs full Tuesday, Thursday, and Sunday refreshes in America/Chicago time.
- Tailscale Serve publishes the loopback-only dashboard privately at
  the tailnet URL saved as `FFAPP_DASHBOARD_URL` in the server `.env`; the app is not exposed to the public internet.
- A manual production refresh on 2026-09-26 completed all three leagues as healthy for week 3,
  with all seven prediction sources available, Sunday movement snapshots recorded, and weekly,
  ROS, role, injury, and source-movement artifacts rebuilt.
- GitHub API rate limiting remains recoverable: weekly and ROS projections retry from validated
  cache and report a degraded run rather than discarding usable recommendations.
- Manifests live under `data/outputs/<league>/refresh_runs/`. Recovery artifacts and hashes live
  under `data/outputs/<league>/last_known_good/`.
- Large raw/interim/feature artifacts remain local. An empty clone still needs the documented
  historical-data rebuild.

## Model evidence

- Consensus/B3 remains the strongest production choice.
- Model v2 Stage 1 beats its trailing baseline but not a league mean.
- Stage 2 and its simple blend lose to the trailing baseline.
- Stage 3 empirical-Bayes estimates are useful on three of four outputs.
- Stage 4 is parked because its prototype intervals were materially miscalibrated.

Do not resume Stage 4 before reading its journal entry and designing a calibration-first
experiment. The next high-value experiment is constrained, per-position source weighting after
enough real logged weeks exist.

## Model improvement roadmap

1. **Injury-duration projections — shipping.** Live Sleeper status and injury notes now create
   explained weekly availability curves. Explicit recovery ranges are parsed when present;
   otherwise the model uses a conservative status-based curve and clearly says the return date is
   unknown. The adjustment affects weekly and ROS points, intervals, lineups, waivers, and trades.
2. **Role-change detection — shipping.** Two recent games are compared with three to six earlier
   games using position-relevant snap and opportunity shares. At least two signals must agree,
   every input predates the projected week, changes are capped at 15%, and ROS effects fade as new
   usage arrives.
3. **Projection movement signals — shipping.** Saved Tuesday, Thursday, and Sunday snapshots now
   distinguish meaningful changes from refresh noise, identify the first-moving source, measure
   cross-source confirmation, explain pipeline-only and availability moves, and strengthen only
   consequential starter alerts.
4. **Opponent and game-environment improvements — queued.** Add pace, play volume, neutral pass
   rate, weather, and materially changed defensive personnel with point-in-time validation.
5. **Decision-based training — queued.** Optimize and evaluate lineup, waiver, and trade choices
   directly once enough settled decision-ledger examples exist.
6. **Early-season safeguards — shipping.** Role adjustments compare current usage with prior role
   evidence, run at 50% strength after two games, ramp to full strength after four, and label the
   active guard in Weekly Actions.
7. **Confidence-based recommendations — queued.** Require larger projected edges when uncertainty,
   source disagreement, or lineup regret history says a decision is fragile.

## Current priorities

1. Accumulate and settle real decision-ledger rows; report follow rate, realized advantage, and
   regret by decision type.
2. Refine the Weekly Actions inbox around consequential exceptions, not more tables.
3. Complete the cold-start source schedule (`TASKS.md` 1.22).
4. Test constrained source weights (`TASKS.md` 3.11) when sample size permits.
5. Add a reproducible demo/bootstrap dataset for fresh-clone onboarding.
6. Work through queued model improvements 3–7 in the roadmap above as evidence accumulates.

## Known risks

- A fresh clone cannot render every page without rebuilding or restoring local data.
- Some operational modules have less test coverage than core model/scoring code.
- Decision outcomes require the user to record whether advice was followed.
- League configuration and prediction logs require review before public release.

## Documentation map

- `docs/CURRENT_STATE.md` — current truth and priority order.
- `README.md` — setup, commands, workflows, and system overview.
- `TASKS.md` — detailed backlog and acceptance criteria.
- `SPEC.md` plus addenda — design contracts and methodology.
- `docs/JOURNAL.md` — append-only implementation evidence.
- `HANDOFF.md` and `docs/summary.md` — legacy historical snapshots.
