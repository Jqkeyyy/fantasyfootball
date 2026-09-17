# Current project state

**Authoritative as of:** 2026-09-16

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

- The three production Windows tasks use S4U and no longer depend on an interactive desktop
  session. A detached Task Scheduler invocation completed the full three-league pipeline on
  2026-09-16, confirming the former `0xC000013A` session-termination failure is resolved.
- A live verification also exposed GitHub API rate limiting. Weekly and ROS projections now retry
  from validated cache and report a degraded run rather than discarding usable recommendations.
- The obsolete duplicate `\FantasyFootball\* Refresh` task set was removed; only the three
  `FFApp Weekly *` production jobs remain enabled.
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

## Current priorities

1. Confirm the September 17 clock-triggered run remains healthy after the live S4U verification.
2. Accumulate and settle real decision-ledger rows; report follow rate, realized advantage, and
   regret by decision type.
3. Refine the Weekly Actions inbox around consequential exceptions, not more tables.
4. Complete the cold-start source schedule (`TASKS.md` 1.22).
5. Test constrained source weights (`TASKS.md` 3.11) when sample size permits.
6. Add a reproducible demo/bootstrap dataset for fresh-clone onboarding.

## Known risks

- Automation has not passed its next real unattended fire after the S4U change.
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
