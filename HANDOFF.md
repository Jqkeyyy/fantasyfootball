# HANDOFF.md — current state

**Last updated:** 2026-08-13
**Last machine:** `Maybe` (Windows). Sleeper reachable here — confirmed live, `api.sleeper.app` returns 200.
**Last commit:** `a75554a` (pushed) — task 1.18 (Projection output pipeline) + task 1.19 (Streamlit weekly rankings page). Working tree is clean, nothing uncommitted.

This file is **state and pointers**, not design or history. `SPEC.md` and the addenda hold the design. Per-task evidence, implementation decisions, and gotchas live in `docs/JOURNAL.md` — read that when investigating a specific past task or bug, not every session. Maintenance rules are at the bottom.

---

## 1. Where things stand

Current phase: Phase 1, projections pipeline. Phase 0 is complete (0.1–0.14).

Tasks 1.13 (Metrics module) and 1.17 (Evaluation report) are complete. Verified this session with a real 5-season walk-forward run: `uv run ffapp evaluate --seasons 2021 --seasons 2022 --seasons 2023 --seasons 2024 --seasons 2025` wrote `data/outputs/eval/20260813T183533Z/report.md` with populated metrics, real LightGBM feature importances, and real calibration curves. Both tasks' evidence is in `docs/JOURNAL.md`.

Phase 2 tasks 2.1–2.5, 2.7, and 2.9 were already complete — the simulation layer, start/sit assistant, DST model, and trade analyzer — built out of TASKS.md order on a prior work machine where Sleeper was network-blocked, confirmed with you first (see 2.1's entry in `docs/JOURNAL.md`).

Task 2.6 (Waiver wire) is complete — built this session, the first task on this machine (Sleeper reachable here) to actually need Sleeper's free-agent pool and transaction history. New `src/ffapp/tools/waivers.py` + `config.WaiverSettings`. Verified against the completed 2025 season (2026's real rosters are pre-draft leftovers, not usable — draft is Aug 22) and against real historical FAAB bidding data pulled from `bdff-chopped`'s actual transactions. Committed as `ea5cd70`, pushed.

Task 1.18 (Projection output pipeline) is complete — new `src/ffapp/models/predict.py` + `ffapp project --week N` CLI command, composing the availability/points/quantile models (tasks 1.14/1.15/1.16) into SPEC §6.2's `outputs/projections.parquet` schema. Verified for real against already-played weeks (2025 weeks 10/11 — 2026 has no nflverse release yet): 445/477 real projections, 0 nulls, full provenance, real upsert-by-`(season, week)` behavior confirmed with a second real run.

Task 1.19 (Streamlit weekly rankings page) is also complete — new `src/ffapp/app/weekly_rankings_page.py` + `src/ffapp/app/pages/2_Weekly_Rankings.py`, SPEC §15's second page. Verified live in a real Chrome session via `claude-in-chrome` (see §7 for the connection hiccup and how it resolved) against the real weeks 1.18 generated: position tabs, the floor/ceiling visible range, and both filters (position + real Sleeper-resolved availability) all confirmed working against real data, 0 console errors. Both 1.18 and 1.19 committed as `a75554a`, pushed. Full evidence for both tasks is in `docs/JOURNAL.md`.

**Next task: undecided.** 1.18/1.19 close out everything else in Phase 1 — the one remaining open item there is 1.15 (Conditional points model v1), a known, already-explained gap from an earlier session (the real model doesn't beat B2, documented as a genuine v1 result, not something to silently fix by tuning against validation seasons — revisit only once v2/a real hyperparameter search is in scope). Remaining open Phase 2 work: 2.8 (SOS/schedule grid — needs the full skill-position feature pipeline plus a real Streamlit heatmap page, neither built, deliberately not attempted), 2.10 (News pipeline, untouched), 2.11 (Model health page, untouched).

**Blocking on me (the human), not the agent:**

- [x] GitHub remote created and pushed
- [x] Sleeper username supplied (`Maybe17`)
- [x] Primary league chosen — `rogan-radinator-league`
- [x] Ranking sources chosen — FantasyPros, ESPN, FantasySharks, CBS Sports, FFToday. **FFToday now 403s live; ESPN's bulk endpoint now returns 0 rows.** Board runs on 3 of 5 sources today; re-check both before the real Aug 22 draft.
- [x] Draft date and slot — slot 3 of 10, Aug 22 2026. League trades draft picks and is a 1-keeper league — see task 0.11's entry.
- [ ] Waiver type/budget — captured automatically via `--discover` (`rogan-radinator-league`: type 1, $100 FAAB). Resolved unless you want to override it.

---

## 3. In progress

Nothing is mid-implementation and nothing is uncommitted. Resuming cold next session means picking one of §1's named next-task options (1.15 revisit, or Phase 2's 2.8/2.10/2.11) — there is no half-finished work to recover first.

This session's own work, in order: (1) reconciled `TASKS.md` against an orphaned commit (message `"g"`) from a prior session that had landed real, tested work for 1.13/1.17 without updating either tracking doc; (2) restructured this file — moved §2/§4/§5 to `docs/JOURNAL.md` verbatim, this file keeps only state and the rebuild/environment reference below; (3) built, verified, committed, and pushed task 2.6 (Waiver wire); (4) built, verified, committed, and pushed tasks 1.18 (Projection output pipeline) and 1.19 (Streamlit weekly rankings page) together, the latter verified live in a real browser session after resolving a real Chrome/Vivaldi extension-connection issue (see §7).

---

## 6. Rebuilding `data/` from empty

Full rationale for each step is in `docs/JOURNAL.md`'s per-task entries (§2). Run in this order on an unrestricted network — see ADDENDUM-02 §B.

```bash
uv run ffapp ingest sleeper --season 2026 --discover --no-offline
uv run ffapp cache warm --season 2026 --all-leagues --no-offline
uv run ffapp cache status
```

Player-id crosswalk (task 0.3 — no CLI command yet, only the function):

```bash
uv run python -c "from ffapp.ingest import nflverse; from ffapp.config import load_settings; nflverse.fetch_player_ids(offline=False, settings=load_settings())"
uv run ffapp ids check --season 2026
```

The six canonical interim tables (task 1.1, 2015–2025 — 2026 has no nflverse release until games are played) through the full feature pipeline (tasks 1.2–1.9) are built by a Python script, not a CLI command. `docs/JOURNAL.md` task 1.9's entry has the exact real call sequence ending in `features_build.build_player_week_features(...)` → `data/features/player_week_features.parquet`; task 1.1's entry lists the six interim tables and task 1.3/1.4's entries cover schedule/weather/injuries. **On a network where Sleeper is blocked**, `ids.mapping.build_players_dim` can skip `layer_sleeper_ids` — see JOURNAL task 1.1's entry for the exact workaround and its real coverage numbers.

Baselines B0–B3 (task 1.10): B0–B2 are pure functions over `player_week_features.parquet`. B3 needs a live FantasyPros weekly-archive git-history mining pass (`fetch_fp_weekly_commits`/`select_commit_before`/`fetch_fp_weekly_snapshot`, ~90+ GitHub requests for a 5-season range) — see JOURNAL task 1.10's entry for the exact sequence.

Rankings/ADP/traded-picks (tasks 0.7, 0.11) — no CLI command; see JOURNAL for the exact fetch calls, then:

```bash
uv run ffapp draft board --season 2026
```

Evaluation report (tasks 1.12–1.17):

```bash
uv run ffapp evaluate --seasons 2021 --seasons 2022 --seasons 2023 --seasons 2024 --seasons 2025
```

`--seasons` must repeat per value — typer's `Option` doesn't support `nargs=-1` (space-separated values after one flag), unlike SPEC's own literal example invocation. See JOURNAL's gotchas for the full explanation.

DST model (task 2.7): a narrower real materialisation, no Sleeper needed — only `pbp`/`team_stats`/`player_stats`/`schedules`/`snap_counts` (raw) and `schedule`/`team_week_context`/`weather` (interim). See JOURNAL task 2.7's entry for the exact script.

Weekly projections (task 1.18) — writes `data/outputs/projections.parquet`, upserted by `(season, week)`:

```bash
uv run ffapp project --season 2025 --week 10
```

Needs that `(season, week)`'s own row universe to already exist in `player_week_features.parquet` — for a genuinely live/current week this means whatever nflverse has published so far (2026 has none yet).

**Not reproducible by re-running the above:** `data/raw/rankings/` (time-sensitive projections/ADP — committed to git as CLAUDE.md's stated exception; do not delete), `data/raw/sleeper/traded_picks_*.json`/`rosters_*.json` (keeper locks, time-sensitive), odds snapshots if the paid API is enabled.

**2026 nflverse data doesn't exist yet.** Every `fetch_*` call for season 2026 404s until real games are played.

**Must be supplied by hand on a new machine:** `.env` (copy from `.env.example`; `FFAPP_OFFLINE=1`/`FFAPP_CACHE_STRICT=1` are sane defaults).

**Windows console gotcha, not a code bug:** `print()`-ing a polars DataFrame crashes with `UnicodeEncodeError` under `cp1252` stdout encoding. Use `.to_dicts()`/plain prints in one-off scripts run here.

---

## 7. Environment notes

**Python:** 3.13.5 (3.11+ satisfied), managed with `uv`. `uv.lock` is committed.
**Offline default:** `FFAPP_OFFLINE=1` in `.env`. Cache warming and league discovery need `--no-offline` explicitly.
**Dependencies:** `lxml`(+stubs), `scikit-learn` (`numpy` pinned `<2.5` — see JOURNAL task 0.10's gotcha), `scipy`, `streamlit`(+`pandas-stubs`), `lightgbm`, `pandas`, `pulp`. Full per-task rationale for each addition is in `docs/JOURNAL.md`.

**Network reachability, most recent confirmation per location:**

| Location | Sleeper | nflverse / GitHub | PyPI | Rankings sites |
|---|---|---|---|---|
| `Maybe` (this session, 2026-08-13) | ✓ (`api.sleeper.app` 200) | not re-checked this session, previously ✓ | not re-checked, previously ✓ | not re-checked this session |
| Work, `ITHQ-172-26-LT1` (2026-08-13) | ✗ — Cisco Umbrella content filter blocks Sleeper specifically (confirmed: real `block.sse.cisco.com` redirect, not a TLS/cert issue) | ✓ | ✓ | ✓ (FantasyPros, FantasyFootballCalculator, CBS Sports confirmed) |
| Home | ? | ? | ? | ? |

If a new machine shows up, add a row rather than overwriting existing ones — don't assume `Maybe` again without checking the actual hostname.

**`claude-in-chrome` browser verification on this machine (`Maybe`) needs the actual Chrome browser, not Vivaldi (the machine's apparent default) or any other Chromium-based browser, and the Claude extension must be installed *and signed in* there specifically.** `list_connected_browsers`/`tabs_context_mcp` both return empty until that's true — closing Vivaldi and opening Chrome alone isn't sufficient if the extension isn't logged into the same claude.ai account this session uses. Once signed in, connection works immediately (confirmed live, task 1.19's own real browser verification).

---

## Maintenance rules

At the end of every session, before committing:

1. Update the header (date, machine, commit).
2. Move newly completed tasks' evidence into `docs/JOURNAL.md` §2, and tick the matching box in `TASKS.md`.
3. Update §1 with the next task and its first concrete step.
4. Record anything half-finished in §3, precisely enough to resume cold.
5. Add any implementation decision to `docs/JOURNAL.md` §4 and any gotcha to §5.
6. Keep §6 and §7 accurate as commands and reachability change.

Delete sections that have gone stale rather than letting them rot. A handoff file nobody trusts is worse than none.
