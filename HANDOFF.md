# HANDOFF.md — current state

**Last updated:** 2026-08-11
**Last machine:** Maybe (Windows)
**Last commit:** `92d770c` — feat: player id mapping with fuzzy matching and league-scoped relevance gate

This file is **state and decisions**, not design. `SPEC.md` and the addenda hold the design; never restate them here. If a line here could have been written before any code existed, it does not belong in this file.

Maintenance rules are at the bottom. Update this file at the end of every session.

---

## 1. Where things stand

**Current phase:** Phase 0 — draft board
**Next task:** 0.4 (Scoring keymap and engine) — 🔴 blocking
**First concrete step:** build `scoring/keymap.py`'s `STAT_KEY_MAP` (direct stats, bonuses, FG distance buckets, DST points-allowed buckets — SPEC §8.1), then `scoring/engine.py`'s `score_stat_line`/`unhandled_keys` (§8.2–8.3). Both real leagues' `scoring_settings` are already cached in `config/leagues/*.yml` from task 0.2, so the keymap can be built and checked against real data immediately — no new ingestion needed.

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
- **0.3** — player id mapping (SPEC §7). New `ingest/nflverse.py` (`fetch_player_ids`, offline-cache pattern matching `sleeper.py`) pulls the dynastyprocess/ffverse crosswalk (`https://raw.githubusercontent.com/dynastyprocess/data/master/files/db_playerids.csv`). New `ids/mapping.py` implements the full pipeline: `load_crosswalk_base` → `layer_sleeper_ids` (fills crosswalk gaps via Sleeper's own `gsis_id` field, adds sleeper-only rows) → `fuzzy_match_remainder` (exact-name tiers 1–2, `rapidfuzz` floor-92 tier 3) → `assign_canonical_id` (`gsis_id` else `synthetic_<blake2b hash>`) → `apply_overrides` (`config/id_overrides.csv`, applied last). `ffapp ids check --season <year>` is live. Evidence:
  - 113 tests pass (`uv run pytest -q`, up from 70), `ruff check`/`ruff format --check`/`mypy src/` all clean.
  - Ran for real against the live cache: `uv run ffapp ids check --season 2026` → **0 unmatched players within top 300** (by the search_rank proxy, scoped to `rogan-radinator-league`'s roster positions). 8,936 total unmatched out of ~12.5k in the crosswalk, all outside the relevance cutoff (deep practice-squad/UDFA names, `search_rank=None`).
  - One real override applied (`config/id_overrides.csv`): sleeper_id 8008 (Josh Imatorbhebhe) — the crosswalk itself carries no gsis_id for him (never appeared in an NFL regular-season box score).

---

## 3. Work in progress

None. 0.1, 0.2, and 0.3 are complete and verified. This session's 0.3 work is uncommitted — see header.

---

## 4. Decisions made during implementation

Places where the spec/addenda were silent, ambiguous, or (in one case) internally contradictory:

- **`_primary.txt` vs `is_primary: true`** (ADDENDUM-01 §A.2 proposes both, never reconciles them): treated `is_primary: true` in the league YAML as the sole source of truth and did not build `_primary.txt` — it would just be a second thing to keep in sync. Revisit if you actually want the separate pointer file.
- **`sleeper.username` location**: SPEC §5 nests it per-league; moved it to `config/settings.yml` instead since it's account-level (one Sleeper login, many leagues), not league-level. Duplicating it into every `config/leagues/<slug>.yml` would just invite drift.
- **`cache warm`/`cache verify` scope**: built as an extensible registry (`ffapp/cache/registry.py`) but only the Sleeper source is wired in — nflverse/rankings/odds/weather don't exist yet (Phase 1). `cache_verify()` raises a clear "not registered yet" error for any task id it can't check, rather than silently reporting OK.
- **Re-run semantics for `write_league_stub`**: re-running `--discover` refreshes `league_cache` from live data (matches SPEC's ingest-idempotency rule) but preserves hand-set `is_primary` and `overrides` — otherwise every re-warm would silently reset your primary-league choice.
- **Primary league**: `rogan-radinator-league` — its settings (10 teams, `fgm_yds: 0.1` per-yard FG scoring, $100 FAAB, playoffs starting week 15) match ADDENDUM-01 §B's "main-ppr" description exactly. Confirmed with you before setting `is_primary: true`.
- **ADDENDUM-01 open decision #10** (bucketed vs per-yard FG scoring per league) is now resolved from real data: `rogan-radinator-league` uses `fgm_yds` (per-yard); `bdff-chopped` uses bucketed `fgm_0_19`/`fgm_20_29`/etc. Task 0.4's keymap needs to handle both, as ADDENDUM-01 §C.1 anticipated.
- **0.3's "top 300 by ADP" acceptance criterion, resolved against `search_rank` instead.** ADP itself doesn't exist yet (task 0.7, blocked on the ranking-sources decision below). Confirmed with you: `ffapp ids check`'s blocking gate ranks by Sleeper's own `search_rank` (already cached, no new ingestion) as an interim stand-in. **Revisit once 0.7 lands real ADP** — swap the gate's ranking source in `mapping.unmatched_report`/`ffapp ids check`.
- **`ingest/nflverse.py` created ahead of Phase 1.** CLAUDE.md forbids network calls outside `ingest/`, and the crosswalk fetch is a network call, so it needed a home there even though full nflverse ingestion (stats, snaps, schedules) is still Phase 1 work. Currently holds exactly one function, `fetch_player_ids`; the rest of the module fills in as Phase 1 tasks land.
- **"Unmatched" is defined as `player_id` starting with `synthetic_`, not "`gsis_id` is null."** An override can resolve a player without ever finding a real `gsis_id` (see Josh Imatorbhebhe below) — checking `gsis_id` directly would keep reporting him as unmatched even after a human explicitly resolved him. Checking the final `player_id` (post-overrides) is the correct "did a human or the pipeline resolve this" signal.
- **`ffapp ids check`'s blocking gate is scoped to the primary league's own roster positions and active, rostered players** (`mapping.league_relevant_positions` / `mapping.league_relevant`), derived from `config/leagues/rogan-radinator-league.yml`'s `roster_positions` + `overrides.flex_eligible` (CLAUDE.md rule 5 — never hardcode league format). This wasn't in the original plan; the raw `search_rank` universe turned out to include retired players and IDP positions this league doesn't roster (see gotchas below), which produced a lot of false-positive "build failures" until scoped.
- **Fuzzy-match tiers 1 and 2 use exact normalised-name equality, not `rapidfuzz`.** SPEC §7 says "Use rapidfuzz... for the last tier" — read literally, only the bare-name tier is fuzzy; the `(name, position, team)` and `(name, position)` tiers require an exact match. If this turns out too strict in practice, loosening tiers 1–2 to fuzzy-with-a-higher-floor is the likely next step.

---

## 5. Gotchas discovered

- **Path resolution bug, caught by actually running the live discovery.** `config.load_settings()` originally resolved `paths.data_root` (`"./data"`) relative to `settings.yml`'s own directory (`config/`), landing raw Sleeper payloads in `config/data/raw/` instead of the intended repo-root `data/raw/`. All the unit tests passed anyway, because the fixture settings.yml lived in a directory that also had a sibling `data/` — the bug was invisible until the real `config/settings.yml` (which lives one level under the repo root) exposed it. Fixed by resolving relative to `REPO_ROOT` explicitly (`config.py`'s `load_settings` now takes an explicit `root` parameter, decoupled from where the settings file itself sits). **Lesson: a fixture that happens to share the bug's blind spot will pass a test that a real config layout fails — this is exactly why ADDENDUM-02 pushed for testing against something closer to real usage before trusting green tests.**
- **`/players/nfl` is bigger than SPEC's estimate.** SPEC §6.1 says "~5MB"; the real payload today is 15.9MB. Not a problem, just don't be surprised.
- **Both real leagues already have a completed draft** (draft objects and draft picks were present during `cache warm`), even though the season hasn't started. Useful for task 0.14's replay-mode development later.
- **Sleeper's `search_rank` spans every player it has ever tracked, not just current fantasy-relevant ones.** Retired players (Tom Brady, Drew Brees, Antonio Brown) show up with real, low `search_rank` values, and IDP positions (LB/DL/DB/DT/CB) are ranked in the same pool as offense. Any future use of `search_rank` as a relevance proxy needs the same league-scoping + active-player filtering `ids/mapping.py` applies (see decision above), not raw rank alone.
- **Sleeper's `active` field means "not purged from Sleeper's system," not "on an NFL roster."** Cut/unsigned players can be `active: true` with `team: null` — and inconsistently, at least one real player (`sleeper_id 8008`, Josh Imatorbhebhe) has `team: null` in Sleeper's own payload but `team: "FA"` in the *dynastyprocess crosswalk's* own linked record for the same `sleeper_id`. `ids/mapping.py`'s `league_relevant()` requires both `active` and a non-null `team` before treating a player as relevant.
- **The dynastyprocess crosswalk can link a `sleeper_id` to a row with no `gsis_id` at all.** This isn't a matching-algorithm failure — nflverse itself never assigned that player a `gsis_id` (no NFL regular-season box score). No amount of better fuzzy matching fixes this; the only fix is `config/id_overrides.csv`. One instance found and overridden this session (`sleeper_id 8008`).
- **The real dynastyprocess crosswalk CSV (`db_playerids.csv`) uses the literal string `"NA"` for missing values**, not empty cells — `load_crosswalk_base` reads it with `null_values=["NA"]`. Team codes in the crosswalk can also differ from Sleeper's own (e.g. `LVR`/`KCC` vs Sleeper's `LV`/`KC`), which is part of why the exact `(name, position, team)` tier misses more than you'd expect and players fall through to the bare-name fuzzy tier.

---

## 6. Rebuilding `data/` from empty

Run in this order on an unrestricted network. See ADDENDUM-02 §B.

```bash
uv run ffapp ingest sleeper --season 2026 --discover --no-offline
uv run ffapp cache warm --season 2026 --all-leagues --no-offline
uv run ffapp cache status
```

Then warm the player-id crosswalk (task 0.3) — there's no `ffapp ingest nflverse` CLI command yet, only the underlying function; a real subcommand + `cache warm`/`cache status` wiring is a reasonable follow-up but wasn't required by 0.3's acceptance criteria:

```bash
uv run python -c "from ffapp.ingest import nflverse; from ffapp.config import load_settings; nflverse.fetch_player_ids(offline=False, settings=load_settings())"
uv run ffapp ids check --season 2026
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
| This session (2026-08-11, hostname `Maybe`) | ✓ | ✓ (raw.githubusercontent.com, reconfirmed fetching the dynastyprocess crosswalk) | ✓ | not tested |
| Home | ? | ? | ? | ? |
| Work | ✗ | ? | ? | ✗ |

> This session's location wasn't identified as "home" or "work" — fold the confirmed row into whichever one applies once you know. Hostname matched the prior session's (`Maybe`), so this was likely the same machine despite being described as "a different machine" at session start.

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
