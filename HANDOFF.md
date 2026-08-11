# HANDOFF.md — current state

**Last updated:** 2026-08-11
**Last machine:** Maybe (Windows)
**Last commit:** `ff59183` — feat: repo scaffold and Sleeper ingestion with offline-first cache layer

This file is **state and decisions**, not design. `SPEC.md` and the addenda hold the design; never restate them here. If a line here could have been written before any code existed, it does not belong in this file.

Maintenance rules are at the bottom. Update this file at the end of every session.

---

## 1. Where things stand

**Current phase:** Phase 0 — draft board
**Next task:** 0.3 (Player ID mapping) — 🔴 blocking
**First concrete step:** build `ids/mapping.py`, starting with loading the ffverse player-id crosswalk as the base table (SPEC §7).

**Blocking on me (the human), not the agent:**

- [x] GitHub remote created and pushed
- [x] Sleeper username supplied for task 0.2 (`Maybe17`)
- [x] Primary league chosen for model development (ADDENDUM-01 §F, decision #9) — `rogan-radinator-league`
- [ ] Ranking sources chosen (SPEC §17 decision #2) — blocks task 0.7
- [ ] Draft date and slot per league (decision #5) — blocks task 0.11
- [ ] Waiver type/budget — already captured automatically per league via `--discover` (`rogan-radinator-league`: type 1, $100 FAAB; `bdff-chopped`: type 2, $1000 FAAB), so decision #6 is effectively resolved unless you want to override it.

---

## 2. Completed tasks

- **0.1** — repo scaffold. `uv run ffapp --version` prints `ffapp 0.1.0`; `ruff check` and `mypy src/` both pass.
- **0.2** — config loading + Sleeper ingestion, folded together with ADDENDUM-01 multi-league support and ADDENDUM-02 §H's offline cache layer (all three were the same piece of work in practice). Evidence:
  - `ffapp ingest sleeper --season 2026 --discover --no-offline` ran live against the real account and discovered 2 leagues: `rogan-radinator-league` (now primary) and `bdff-chopped`.
  - `ffapp cache warm --season 2026 --all-leagues --no-offline` pulled league objects, rosters, users, drafts, draft picks, the full player dictionary (15.9MB), and trending add/drop lists.
  - `ffapp cache status` lists all 15 cached artefacts with correct freshness verdicts.
  - 70 tests pass (`uv run pytest -q`), including the ADDENDUM-02 §H-mandated `tests/test_offline_raises.py` (parametrized across all 11 `fetch_*` functions — every one raises `OfflineCacheMiss`, none silently returns empty).

---

## 3. Work in progress

None. 0.1 and 0.2 are complete and verified.

---

## 4. Decisions made during implementation

Places where the spec/addenda were silent, ambiguous, or (in one case) internally contradictory:

- **`_primary.txt` vs `is_primary: true`** (ADDENDUM-01 §A.2 proposes both, never reconciles them): treated `is_primary: true` in the league YAML as the sole source of truth and did not build `_primary.txt` — it would just be a second thing to keep in sync. Revisit if you actually want the separate pointer file.
- **`sleeper.username` location**: SPEC §5 nests it per-league; moved it to `config/settings.yml` instead since it's account-level (one Sleeper login, many leagues), not league-level. Duplicating it into every `config/leagues/<slug>.yml` would just invite drift.
- **`cache warm`/`cache verify` scope**: built as an extensible registry (`ffapp/cache/registry.py`) but only the Sleeper source is wired in — nflverse/rankings/odds/weather don't exist yet (Phase 1). `cache_verify()` raises a clear "not registered yet" error for any task id it can't check, rather than silently reporting OK.
- **Re-run semantics for `write_league_stub`**: re-running `--discover` refreshes `league_cache` from live data (matches SPEC's ingest-idempotency rule) but preserves hand-set `is_primary` and `overrides` — otherwise every re-warm would silently reset your primary-league choice.
- **Primary league**: `rogan-radinator-league` — its settings (10 teams, `fgm_yds: 0.1` per-yard FG scoring, $100 FAAB, playoffs starting week 15) match ADDENDUM-01 §B's "main-ppr" description exactly. Confirmed with you before setting `is_primary: true`.
- **ADDENDUM-01 open decision #10** (bucketed vs per-yard FG scoring per league) is now resolved from real data: `rogan-radinator-league` uses `fgm_yds` (per-yard); `bdff-chopped` uses bucketed `fgm_0_19`/`fgm_20_29`/etc. Task 0.4's keymap needs to handle both, as ADDENDUM-01 §C.1 anticipated.

---

## 5. Gotchas discovered

- **Path resolution bug, caught by actually running the live discovery.** `config.load_settings()` originally resolved `paths.data_root` (`"./data"`) relative to `settings.yml`'s own directory (`config/`), landing raw Sleeper payloads in `config/data/raw/` instead of the intended repo-root `data/raw/`. All the unit tests passed anyway, because the fixture settings.yml lived in a directory that also had a sibling `data/` — the bug was invisible until the real `config/settings.yml` (which lives one level under the repo root) exposed it. Fixed by resolving relative to `REPO_ROOT` explicitly (`config.py`'s `load_settings` now takes an explicit `root` parameter, decoupled from where the settings file itself sits). **Lesson: a fixture that happens to share the bug's blind spot will pass a test that a real config layout fails — this is exactly why ADDENDUM-02 pushed for testing against something closer to real usage before trusting green tests.**
- **`/players/nfl` is bigger than SPEC's estimate.** SPEC §6.1 says "~5MB"; the real payload today is 15.9MB. Not a problem, just don't be surprised.
- **Both real leagues already have a completed draft** (draft objects and draft picks were present during `cache warm`), even though the season hasn't started. Useful for task 0.14's replay-mode development later.

---

## 6. Rebuilding `data/` from empty

Run in this order on an unrestricted network. See ADDENDUM-02 §B.

```bash
uv run ffapp ingest sleeper --season 2026 --discover --no-offline
uv run ffapp cache warm --season 2026 --all-leagues --no-offline
uv run ffapp cache status
```

`--seasons 2015-2025` historical nflverse warming doesn't exist yet — that lands with task 1.1.

**Not reproducible by re-running the above:**

- `data/raw/rankings/` — consensus projections and ADP are time-sensitive; re-pulling on a different day gives different numbers. These are committed to git as the exception in `CLAUDE.md`. Do not delete them. (Not yet populated — task 0.7.)
- Odds snapshots, if the paid API is enabled — same reasoning, but cheap to lose.

**Must be supplied by hand on a new machine:** `.env` (copy from `.env.example`; `FFAPP_OFFLINE=1` and `FFAPP_CACHE_STRICT=1` are sane defaults, both already used here).

---

## 7. Environment notes

**Python:** 3.13.5 (3.11+ satisfied), managed with `uv` 0.12.3. `uv.lock` is committed.
**Offline default:** `FFAPP_OFFLINE=1` in `.env`. Cache warming and league discovery need `--no-offline` explicitly — both commands refuse to run offline with a clear message rather than failing deep inside an HTTP call.
**Vendored wheels:** not currently needed. If PyPI becomes unreachable, see ADDENDUM-02 §F.

**Network reachability by location:**

| Location | Sleeper | nflverse / GitHub | PyPI | Rankings sites |
|---|---|---|---|---|
| This session (2026-08-11) | ✓ | ✓ (raw.githubusercontent.com) | ✓ | not tested |
| Home | ? | ? | ? | ? |
| Work | ✗ | ? | ? | ✗ |

> This session's location wasn't identified as "home" or "work" — fold the confirmed row into whichever one applies once you know.

---

## Maintenance rules

At the end of every session, before committing:

1. Update the header (date, machine, commit).
2. Move newly completed tasks into §2 **with their evidence**, and tick the matching box in `TASKS.md`.
3. Update §1 with the next task and its first concrete step.
4. Record anything half-finished in §3, precisely enough to resume cold.
5. Add any implementation decision to §4 and any gotcha to §5.
6. Keep §6 and §7 accurate as commands and reachability change.

Delete sections that have gone stale rather than letting them rot. A handoff file nobody trusts is worse than none.
