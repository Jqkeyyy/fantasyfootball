# Fantasy Football Manager — Technical Specification

**Version:** 1.0
**Written:** 2026-08-11
**Target season:** NFL 2026
**Audience:** Claude Code (implementation) and the project owner
**Deployment intent:** Single-user, local-first. Multi-user is a possible future, not a v1 requirement.

---

## 0. How to use this document

This spec is written to be handed to Claude Code as the source of truth for implementation. It is deliberately opinionated: where a decision has been made, it is stated as a decision, not an option. Where a decision is genuinely open, it is listed in §17 (Open Decisions) and must be resolved before the affected module is built.

Companion files:

- `CLAUDE.md` — coding conventions, environment rules, and standing instructions for the agent.
- `TASKS.md` — the ordered, PR-sized task list with acceptance criteria.

Read order for implementation: `CLAUDE.md` → `TASKS.md` → the relevant `SPEC.md` section for the task at hand.

**Non-negotiable rules that appear throughout and are repeated here because violating them silently invalidates everything downstream:**

1. No random train/test splits. Ever. All validation is walk-forward in time.
2. Every feature row carries an `as_of` timestamp. A feature computed for (season, week) may only use data that existed before that week's first kickoff.
3. The scoring engine is validated against Sleeper's own computed points before it is trusted.
4. Beat the documented baselines before believing any model.

---

## 1. Project goals

### Primary goal

A personal fantasy football decision-support system for one Sleeper league, built around a self-trained player projection model that produces:

- **Weekly projections** — mean plus a calibrated distribution (floor/median/ceiling), per player, sortable and filterable by position.
- **Season-long rankings** — expected value over replacement, derived from simulated season outcomes rather than a single summed point total.

Both are computed under *this league's exact scoring rules*, not generic PPR or standard.

### Secondary goals (in build order)

1. Draft board with tiers, value over replacement, and pick-survival probabilities.
2. Start/sit assistant driven by win probability, not raw projected points.
3. Waiver wire ranking with FAAB bid guidance, scored as value added to *this specific roster*.
4. Weekly DST streaming recommendations.
5. Bye week and strength-of-schedule visualisation, including positional matchup difficulty.
6. Trade analyzer (evaluate a proposed trade).
7. Trade finder (generate mutually beneficial trade candidates).
8. Injury and news ingestion with automatic second-order effect propagation.

### Explicit non-goals for v1

- Multi-user accounts, authentication, hosted deployment.
- Any write action to Sleeper. The Sleeper API is read-only; the system recommends, the human executes.
- Mobile app, push notifications.
- DFS optimisation, betting, or any wagering feature.
- Dynasty or keeper league valuation (contract/rookie-pick value). Redraft only.
- Support for leagues on platforms other than Sleeper.

---

## 2. The timing constraint

Week 1 of the 2026 season is in early September. The league draft is expected in the next two to four weeks from the date of this spec.

This produces a hard sequencing rule: **Phase 0 (the draft board) must be complete and usable before the draft, and it must not depend on any trained model.** Phase 0 uses aggregated public consensus projections re-scored under league settings. The model is trained during the season on live data and does not gate the draft.

Do not begin Phase 1 model work until Phase 0 is shipped and the draft board has been used.

---

## 3. Core design decisions

| Decision | Choice | Rationale |
|---|---|---|
| Language | Python 3.11+ | Whole NFL data ecosystem is Python/R; nflreadpy is the official Python port |
| Dependency manager | `uv` | Fast, lockfile-based, single tool for venv + deps |
| Dataframe library | Polars | nflreadpy returns Polars natively; avoids a conversion layer. Convert to pandas only at model-fitting boundaries if needed |
| Analytics engine | DuckDB over Parquet | No server, SQL over files, excellent Parquet support. Postgres only if/when multi-user |
| Storage format | Parquet on local disk, partitioned by season | Simple, portable, versionable, fast |
| ML library | LightGBM | Right size for the data; native quantile objective; fast retraining |
| Optimisation | `pulp` (CBC solver) for lineup ILP | Exact optimal lineups including FLEX; small problems solve instantly |
| CLI | Typer | Every pipeline stage runs as a subcommand; scriptable and cron-friendly |
| UI (v1) | Streamlit | Weeks faster than a real frontend. UI is the seductive distraction; keep it cheap until the numbers are good |
| UI (v2, optional) | FastAPI + Next.js | Only if going public |
| Scheduling | Local cron initially; GitHub Actions if hosted | Free, sufficient |
| Config | YAML for league/settings, `.env` for secrets | Secrets never in the repo |
| Testing | pytest | Golden tests for the scoring engine are mandatory |

### Why not deep learning

Data volume: ~272 regular season games per year × ~40–50 fantasy-relevant players per game ≈ 12,000 usable player-weeks per season. Even fifteen seasons is low hundreds of thousands of rows, and pre-2015 football is arguably a different sport for modelling purposes. With ~60 engineered features, this is gradient-boosting territory. A neural network will overfit and will consume the implementation time that should go into feature engineering and evaluation, which is where the actual edge lives.

Revisit only if a sequence model over usage trajectories becomes interesting in the 2027 offseason. It is a v3 idea, not a v1 idea.

---

## 4. Repository layout

```
fantasy-app/
├── CLAUDE.md
├── SPEC.md
├── TASKS.md
├── pyproject.toml
├── .env.example
├── .gitignore
├── config/
│   ├── league.yml              # league id, username, overrides
│   ├── settings.yml            # paths, model hyperparams, sim counts
│   └── stadiums.csv            # team → lat/lon, dome flag, surface
├── data/                       # gitignored
│   ├── raw/                    # unmodified source pulls
│   │   ├── nflverse/
│   │   ├── sleeper/
│   │   ├── odds/
│   │   ├── weather/
│   │   └── rankings/
│   ├── interim/                # cleaned, joined, id-mapped
│   ├── features/               # wide feature tables, as_of stamped
│   ├── models/                 # serialised models + metadata
│   └── outputs/                # projections, draft boards, reports
├── src/ffapp/
│   ├── __init__.py
│   ├── config.py               # config loading, path resolution
│   ├── cli.py                  # Typer app, all subcommands
│   ├── ingest/
│   │   ├── nflverse.py
│   │   ├── sleeper.py
│   │   ├── odds.py
│   │   ├── weather.py
│   │   ├── rankings.py         # consensus projections + ADP
│   │   └── news.py
│   ├── ids/
│   │   └── mapping.py          # cross-source player id resolution
│   ├── scoring/
│   │   ├── engine.py           # stat line → league points
│   │   └── keymap.py           # Sleeper scoring key → stat column
│   ├── features/
│   │   ├── registry.py         # feature definitions + as_of contracts
│   │   ├── usage.py
│   │   ├── team_context.py
│   │   ├── opponent.py         # positional defence adjustment
│   │   └── build.py            # assembles the wide table
│   ├── models/
│   │   ├── baselines.py
│   │   ├── availability.py     # P(player is active and plays)
│   │   ├── points.py           # conditional points model
│   │   ├── quantile.py
│   │   ├── dst.py
│   │   ├── train.py
│   │   └── predict.py
│   ├── evaluation/
│   │   ├── snapshot.py         # as_of enforcement
│   │   ├── backtest.py         # walk-forward harness
│   │   ├── metrics.py
│   │   └── report.py
│   ├── sim/
│   │   ├── lineup.py           # optimal lineup solver
│   │   ├── week.py             # single-week matchup simulation
│   │   ├── season.py           # rest-of-season Monte Carlo
│   │   └── injury.py           # games-played hazard model
│   ├── tools/
│   │   ├── vor.py
│   │   ├── tiers.py
│   │   ├── draft.py
│   │   ├── rankings.py         # weekly + seasonal ranking assembly
│   │   ├── startsit.py
│   │   ├── waivers.py
│   │   ├── trades.py
│   │   ├── sos.py
│   │   └── schedule.py
│   └── app/
│       ├── streamlit_app.py
│       └── pages/
├── tests/
│   ├── test_scoring.py         # golden tests vs Sleeper
│   ├── test_leakage.py         # as_of contract enforcement
│   ├── test_lineup.py
│   └── ...
└── notebooks/                  # exploration only, never imported
```

**Rule:** `notebooks/` is scratch. No module in `src/` may import from it. Anything that matters gets promoted into `src/`.

---

## 5. Configuration

### `config/league.yml`

```yaml
sleeper:
  username: "REPLACE_ME"
  league_id: "REPLACE_ME"        # resolved via CLI on first run
  season: 2026

# Populated automatically from the Sleeper league object.
# Committed so the draft board is reproducible offline.
league_cache:
  total_rosters: null
  roster_positions: []           # e.g. [QB, RB, RB, WR, WR, TE, FLEX, K, DEF, BN, BN, ...]
  scoring_settings: {}
  waiver_type: null
  waiver_budget: null
  playoff_week_start: null

overrides:
  # Optional manual overrides if Sleeper's settings are ambiguous
  flex_eligible: [RB, WR, TE]
  superflex_eligible: [QB, RB, WR, TE]
```

### `config/settings.yml`

```yaml
paths:
  data_root: "./data"

seasons:
  train_start: 2015              # earliest season used for model training
  current: 2026

model:
  positions: [QB, RB, WR, TE, K, DST]
  quantiles: [0.10, 0.25, 0.50, 0.75, 0.90]
  retrain_cadence_weeks: 1
  min_train_rows: 2000
  lightgbm:
    n_estimators: 800
    learning_rate: 0.03
    num_leaves: 31
    min_child_samples: 40
    subsample: 0.8
    colsample_bytree: 0.8
    reg_lambda: 1.0

simulation:
  season_sims: 3000
  week_sims: 20000
  correlation:
    qb_pass_catcher: 0.35        # same-team QB↔WR/TE weekly correlation
    same_team_rb_rb: -0.25
    player_vs_opposing_dst: -0.30

draft:
  adp_sd_fallback: 8.0           # picks, when source gives no spread
  tier_method: "gap"             # gap | kmeans | gmm

ingest:
  sleeper_players_cache_hours: 24
  odds_provider: "the_odds_api"  # or "none" — falls back to nflverse schedule lines
```

### `.env.example`

```
ODDS_API_KEY=
ANTHROPIC_API_KEY=
```

Secrets are read via `os.environ` only. Never logged, never written to `data/`.

---

## 6. Data layer

### 6.1 Sources

| Source | Package / endpoint | What it provides | Cadence | Cost |
|---|---|---|---|---|
| nflverse | `nflreadpy` (Python, Polars) | Play-by-play (1999–), player weekly stats, snap counts, depth charts, rosters, injuries, schedules incl. spread/total, PFR advanced stats | PBP nightly on game days; snaps 4×/day; depth charts + PFR daily | Free (CC-BY) |
| ffopportunity | ffverse GitHub releases (Parquet/CSV) | Precomputed expected fantasy points from an xgboost model over nflverse PBP | Automated releases | Free (CC-BY-SA, attribution required) |
| Sleeper | `https://api.sleeper.app/v1` | League settings, scoring, rosters, users, matchups, transactions, drafts, picks, trending adds/drops, player dictionary | On demand | Free, read-only, no auth |
| FantasyPros | `ffpros` (R) or direct fetch | Expert consensus rankings, per-stat projections, ADP | Daily in preseason | Free tier; check ToS before any redistribution |
| The Odds API | `api.the-odds-api.com/v4` | Live spreads and totals | Multiple times weekly | NFL requires paid tier (~$29/mo). Optional |
| Open-Meteo | `api.open-meteo.com` | Forecast and historical weather by lat/lon | Weekly | Free |
| nflverse schedules | `load_schedules()` | Historical `spread_line`, `total_line` | With schedule updates | Free |

**Rate limits and etiquette:**

- Sleeper: stay well under 1000 requests/minute. In practice this project needs a few dozen calls per run.
- Sleeper `/players/nfl`: this is a very large payload (several MB) containing the full player dictionary. Fetch **at most once per day** and cache to `data/raw/sleeper/players_YYYYMMDD.parquet`. Never call it inside a loop.
- Open-Meteo: free tier is generous but batch stadium requests rather than one per player.
- All HTTP clients use a shared session with retry/backoff and a descriptive User-Agent.

**Licensing note for a possible public future:** nflverse data is CC-BY; ffopportunity is CC-BY-SA (share-alike propagates to derived data). FantasyPros scraping for personal use is a different matter from redistribution. If the app ever goes public, re-audit every source and remove or license anything that cannot be redistributed. Record the licence of each source in `data/raw/<source>/LICENSE.txt` at ingest time.

### 6.2 Canonical schemas

All tables are Parquet, partitioned by `season` where the table spans seasons. Column types are stated where ambiguity is likely.

#### `interim/players_dim.parquet`
The identity crosswalk. This table is the single most important piece of plumbing in the project — get it wrong and every join downstream silently drops players.

| Column | Type | Notes |
|---|---|---|
| `player_id` | str | Canonical internal id. Use nflverse `gsis_id` when present, else `synthetic_<hash>` |
| `gsis_id` | str? | nflverse primary key |
| `sleeper_id` | str? | Sleeper player id |
| `pfr_id` | str? | Pro Football Reference |
| `espn_id` | str? | |
| `fantasypros_name_key` | str? | normalised name + position + team, for rankings joins |
| `full_name` | str | |
| `position` | str | QB/RB/WR/TE/K/DST |
| `team` | str? | current team abbreviation |
| `birth_date` | date? | |
| `rookie_season` | int? | |
| `status` | str? | active/injured-reserve/etc |

#### `interim/schedule.parquet`

| Column | Notes |
|---|---|
| `game_id`, `season`, `week`, `season_type` | |
| `home_team`, `away_team` | nflverse abbreviations, canonical throughout |
| `gameday`, `gametime`, `kickoff_utc` | `kickoff_utc` is derived and is the `as_of` boundary |
| `spread_line`, `total_line` | from nflverse; positive spread = home favoured (verify sign at ingest and document) |
| `roof`, `surface`, `stadium_id` | |
| `home_implied_total`, `away_implied_total` | derived: `total/2 ± spread/2` |
| `home_rest`, `away_rest` | days since previous game |

#### `interim/player_week_stats.parquet`
Raw counting stats per player-week, one row per player per game played. Sourced from nflverse player stats. Includes at minimum: pass attempts/completions/yards/TD/INT/sacks, rush attempts/yards/TD, targets/receptions/receiving yards/TD, fumbles lost, two-point conversions, return yards/TD, kicking stats (FG made by distance bucket, XP), and DST stats (sacks, INT, fumble recoveries, safeties, TD, points allowed, yards allowed).

#### `interim/player_week_usage.parquet`

| Column | Definition |
|---|---|
| `offense_snaps`, `offense_snap_pct` | from nflverse snap counts (PFR-sourced) |
| `targets`, `target_share` | targets / team pass attempts |
| `air_yards`, `air_yards_share` | |
| `wopr` | 1.5 × target_share + 0.7 × air_yards_share |
| `adot` | air yards / targets |
| `carries`, `carry_share` | |
| `rz_targets`, `rz_carries`, `rz_touch_share` | inside opponent 20 |
| `gz_carries` | inside opponent 5 |
| `route_participation` | **may be null in-season** — see gap note below |
| `xfp` | expected fantasy points from ffopportunity |

**Route participation gap:** NFL Next Gen Stats participation data ceased mid-2023. The FTN replacement is published only after the postseason completes and does not update in-season. Therefore `route_participation` will be populated for historical training seasons but **null for the current season**. The feature registry must handle this: either exclude route features from in-season inference models, or train a second model variant without them. Do not let a feature that is available in training and missing at inference silently become a null-imputed constant — that is a leakage-adjacent failure that will quietly degrade live projections. See §10.5.

#### `interim/team_week_context.parquet`

| Column | Definition |
|---|---|
| `plays`, `neutral_pace_sec` | seconds per play in neutral game script (score within 7, Q1–Q3) |
| `pass_rate`, `proe` | pass rate over expectation vs down/distance/score baseline |
| `epa_per_play_off`, `success_rate_off` | |
| `implied_total`, `spread` | joined from schedule |

#### `interim/defense_position_allowed.parquet`
Opponent-adjusted defensive rates, by defence-team × week × position group. See §10.4 for the adjustment method.

| Column | Notes |
|---|---|
| `defteam`, `season`, `week`, `position_group` | groups: `WR_perimeter`, `WR_slot`, `TE`, `RB_receiving`, `RB_rushing`, `QB_passing`, `QB_rushing` |
| `adj_epa_allowed`, `adj_success_allowed`, `adj_ypt_allowed`, `adj_td_rate_allowed` | ridge-adjusted, shrunk |
| `n_plays` | sample size behind the estimate |

#### `interim/injuries.parquet`
Weekly official injury report: `player_id`, `season`, `week`, `report_status` (Out/Doubtful/Questionable/None), `practice_status`, `report_primary_injury`, `date_modified`.

`date_modified` is essential — the as_of snapshot logic needs to know when each designation was published.

#### `features/player_week_features.parquet`
The wide modelling table. One row per (player_id, season, week), including weeks the player did not play (target = 0, availability flag = 0). Carries `as_of_utc`.

#### `outputs/projections.parquet`

| Column | Notes |
|---|---|
| `player_id`, `season`, `week` | |
| `p_active` | probability the player plays |
| `mean` | expected league-scored points, unconditional |
| `q10`, `q25`, `q50`, `q75`, `q90` | unconditional quantiles |
| `model_version`, `as_of_utc`, `feature_hash` | reproducibility |

### 6.3 Ingestion contract

Every ingest module exposes:

```python
def fetch(season: int, weeks: list[int] | None = None, force: bool = False) -> Path:
    """Write raw source data to data/raw/<source>/. Return the path written."""

def normalise(raw_path: Path) -> pl.DataFrame:
    """Convert raw payload to the canonical schema in §6.2. No business logic."""
```

Rules:

- Ingest is idempotent. Re-running for the same season/week overwrites cleanly.
- Raw payloads are stored unmodified. If a source changes shape, the raw archive lets you diagnose it.
- Every write records a small sidecar JSON: `{source, fetched_at_utc, rows, url_or_call, package_version}`.
- No transformation logic in `fetch`. No network calls in `normalise`.

---

## 7. Player ID mapping

This is where projects like this die. Budget real time for it and build it early.

The problem: Sleeper ids, nflverse `gsis_id`, PFR ids, and FantasyPros name strings are all different, and they disagree about rookies, recently traded players, suffixes (Jr./III), apostrophes, and hyphenated names.

### Approach

1. **Start from published crosswalks.** `nflreadr::load_ff_playerids()` (also available via the ffverse data releases as a plain Parquet/CSV, so it is reachable from Python without R) is a maintained mapping covering gsis, sleeper, pfr, espn, and several other id systems. Load it as the base table.
2. **Layer Sleeper's own dictionary.** The `/players/nfl` payload contains cross-reference fields for several external id systems. Use them to fill gaps in the base crosswalk.
3. **Fuzzy-match the remainder by name.** Only for players still unmatched after steps 1 and 2. Normalise: lowercase, strip punctuation and suffixes, collapse whitespace. Match on `(normalised_name, position, team)` first, then `(normalised_name, position)`, then `(normalised_name)`. Use `rapidfuzz` with a similarity floor of 92 for the last tier.
4. **Persist manual overrides.** `config/id_overrides.csv` with columns `source, source_id, player_id, note`. Hand-resolved cases go here and are applied last, overriding everything. This file is committed.

### Mandatory guardrail

`ids/mapping.py` must expose:

```python
def unmatched_report(season: int) -> pl.DataFrame:
    """Players present in a source but not resolved to a canonical player_id,
    ranked by fantasy relevance (projected points or ADP)."""
```

The CLI command `ffapp ids check` prints this report. **Any unmatched player inside the top 300 by ADP is a build failure.** Fail loudly rather than dropping rows in a join — a silently dropped WR2 is a projection you will never notice is missing.

Add a test that asserts zero unmatched players in the top 300.

---

## 8. Scoring engine

### 8.1 Why this is critical

Every ranking, projection, VOR calculation, and trade valuation in the system is denominated in *this league's points*. If the scoring engine is wrong by half a point per reception, everything downstream is wrong and nothing will look obviously broken.

### 8.2 Sleeper scoring settings

The league object returned by `GET /league/{league_id}` contains `scoring_settings`: a flat dictionary mapping scoring keys to point values. Typical keys include (verify against the actual dump — do not assume):

```
pass_yd, pass_td, pass_int, pass_2pt, pass_sack
rush_yd, rush_td, rush_2pt
rec, rec_yd, rec_td, rec_2pt
bonus_rec_te, bonus_rush_yd_100, bonus_pass_yd_300, ...
fum, fum_lost, fum_rec_td
xpm, xpmiss, fgm_0_19, fgm_20_29, fgm_30_39, fgm_40_49, fgm_50p, fgmiss
def_td, sack, int, ff, fum_rec, safe, blk_kick
pts_allow_0, pts_allow_1_6, pts_allow_7_13, pts_allow_14_20,
pts_allow_21_27, pts_allow_28_34, pts_allow_35p
```

### 8.3 Implementation

`scoring/keymap.py` defines `STAT_KEY_MAP: dict[str, StatSpec]` where each Sleeper key maps to either:

- a direct stat column (`pass_yd` → `passing_yards`, multiplied by the setting value), or
- a **derived rule** (bonuses, per-bucket field goals, points-allowed buckets), implemented as a small callable.

`scoring/engine.py` exposes:

```python
def score_stat_line(stats: pl.DataFrame, scoring: dict[str, float]) -> pl.Series:
    """Apply league scoring to a per-player-week stat frame. Returns points."""

def unhandled_keys(scoring: dict[str, float]) -> list[str]:
    """Scoring keys present in the league settings with no mapping. MUST be empty."""
```

**Rule:** if `unhandled_keys()` is non-empty and any of those keys has a non-zero value, the engine raises. A silently ignored scoring rule is exactly the kind of bug that costs a season.

### 8.4 Golden test (mandatory)

Sleeper's matchups endpoint (`GET /league/{league_id}/matchups/{week}`) returns `players_points` — Sleeper's own computed score for every rostered player that week. This is a free, exact oracle.

The test:

1. Pull matchups for every completed week of the most recent season the league played.
2. Independently compute points for the same players from nflverse stats using `score_stat_line`.
3. Assert agreement within 0.01 points for at least 99% of player-weeks, and log every disagreement.

Any systematic disagreement means either the keymap is wrong or the nflverse stat definition differs from Sleeper's. Both are worth knowing before the season, not during it. Do not proceed past this test.

### 8.5 Roster and lineup settings

`roster_positions` is an ordered list like `[QB, RB, RB, WR, WR, WR, TE, FLEX, K, DEF, BN, BN, BN, BN, BN, BN, IR]`.

Parse into:

```python
@dataclass
class LeagueFormat:
    n_teams: int
    starters: dict[str, int]          # {"QB": 1, "RB": 2, "WR": 3, "TE": 1, "K": 1, "DST": 1}
    flex_slots: dict[str, int]        # {"FLEX": 1, "SUPER_FLEX": 0, "REC_FLEX": 0}
    flex_eligible: dict[str, list[str]]
    bench: int
    ir: int
    playoff_week_start: int
    waiver_budget: int | None
```

`LeagueFormat` is passed to the VOR calculator, the lineup optimiser, and the season simulator. Nothing hardcodes "2 RB, 3 WR".

---

## 9. Phase 0 — the draft board (build this first)

This phase must be complete before the draft and must not depend on any trained model. It formalises and improves the manual process already in use (average several published ranking sets, rescale for league scoring).

### 9.1 What is actually being improved

Averaging several sources is genuinely sound — it is variance reduction on a noisy signal, and it is hard to beat with a homemade model. Three concrete improvements over doing it by hand:

1. **Average projections, not ranks, wherever possible.** Ranks are ordinal: the gap between RB1 and RB2 is not the gap between RB25 and RB26. Averaging ranks throws away the magnitude information that VOR depends on. Average *projected points* (after league rescaling); fall back to rank aggregation only for sources that publish nothing else.
2. **Rescale before aggregating, not after.** Apply league scoring to each source's per-stat projections individually, then aggregate. Aggregating generic-PPR points and then applying a correction factor is not the same operation and is wrong for any league whose scoring is not a scalar multiple of PPR.
3. **Keep the disagreement.** The standard deviation across sources is a free, useful uncertainty estimate. Wide disagreement on a player is information — it flags boom/bust profiles and unsettled situations, and it should be visible on the draft board.

### 9.2 Aggregation algorithm

```
INPUT: k sources, each providing either
       (a) per-stat season projections, or
       (b) positional ranks / ECR only

FOR each source s:
    IF source provides per-stat projections:
        points[s][p] = score_stat_line(projected_stats[s][p], league_scoring)
    ELSE:
        # map rank to a value scale using the sources that do have points
        reference_curve = median over point-providing sources of
                          (points at each positional rank)
        points[s][p] = reference_curve[position(p)][rank[s][p]]

FOR each player p:
    n_sources[p]   = count of sources covering p
    proj_points[p] = trimmed_mean(points[*][p], trim=0.2)   # drops high+low
    dispersion[p]  = stdev(points[*][p])
    coverage[p]    = n_sources[p] / k
```

Use a **trimmed mean, not a plain mean**. One source with an outlier opinion should not move the consensus much; that is the entire point of aggregating.

Players covered by fewer than half the sources get flagged, not dropped — thin coverage in August usually means an unsettled situation, which is exactly where late-round value lives.

### 9.3 Games-played adjustment

Season projections from public sources are usually implicitly "if healthy" or lightly discounted. Convert to expected value:

```
expected_games[p] = 17 × p_available_baseline(position, age, injury_history)
proj_points_adj[p] = points_per_game[p] × expected_games[p]
```

For Phase 0 use a simple positional/age prior (see §13.3 for the full hazard model, which is a Phase 2 deliverable). A crude but honest prior beats pretending everyone plays 17 games — that assumption systematically overvalues injury-prone players and RBs generally.

### 9.4 Value over replacement

Replacement level depends on league format, and the FLEX slot makes it circular: which positions fill FLEX depends on the values, which depend on replacement level. Resolve with a short fixed-point iteration.

```
ALGORITHM: replacement_level(projections, league_format)

1. baseline[pos] = n_teams × starters[pos]         # dedicated slots only
2. REPEAT until baseline stops changing (max 10 iterations):
     a. For each flex type (FLEX, SUPER_FLEX, REC_FLEX):
          candidates = all players of flex-eligible positions
                       ranked BELOW their positional baseline
          fillers    = top (n_teams × flex_slots[type]) candidates by proj_points
     b. flex_count[pos] = number of fillers at each position
     c. baseline[pos] = n_teams × starters[pos] + flex_count[pos]
3. replacement_points[pos] = proj_points of the player at rank baseline[pos]
4. vor[p] = proj_points_adj[p] − replacement_points[position(p)]
```

Notes:

- K and DST get a replacement level too, and it is almost always the case that their VOR is tiny. That is the correct result, and it is the mathematical justification for not drafting them early.
- In a superflex league the QB baseline moves dramatically. This algorithm handles it automatically. Do not special-case superflex by hand.
- Rank the board by VOR, never by raw projected points.

### 9.5 Tiers

Tiers matter more than ranks at the draft. The decision that costs value is reaching past a tier break, not choosing #14 over #15.

Default method (`tier_method: gap`):

```
Within each position, sorted by VOR descending:
    gaps[i] = vor[i] − vor[i+1]
    threshold = 1.4 × rolling_median(gaps, window=9)
    cut where gaps[i] > threshold
    merge any tier smaller than 2 players into its neighbour
    cap at 12 tiers per position
```

Alternatives (`kmeans`, `gmm`) implemented behind the same interface for comparison. Whichever is used, the board shows tier boundaries as visual breaks, and the live assistant shows *how many players remain in the current tier at each position* — that number is what should drive positional decisions on the clock.

### 9.6 ADP and pick-survival probability

For each player, obtain ADP and a spread. If the source gives high/low, estimate `sd ≈ (high − low) / 4`. If not, use `draft.adp_sd_fallback`.

```
P(player available at pick k) = 1 − Φ((k − adp_mean) / adp_sd)
```

Given your draft slot and league size, compute your pick numbers, then for each player:

- `p_avail_next` — probability they last until your next pick
- `p_avail_after_next` — probability they last two picks

Derive the decision-relevant quantity:

```
opportunity_cost(p) = vor[p] − E[ best available VOR at position(p) at my next pick ]
```

This is what actually answers "do I take him now or wait." Surface it as a column.

### 9.7 Draft board output

`data/outputs/draft_board_<season>.csv` and a Streamlit page. Columns:

| Column | Purpose |
|---|---|
| `overall_rank`, `pos_rank`, `tier` | ordering |
| `player`, `position`, `team`, `bye_week` | identity |
| `proj_points_adj`, `proj_ppg`, `expected_games` | the projection |
| `vor` | the ranking metric |
| `dispersion`, `n_sources` | consensus confidence |
| `adp`, `adp_sd`, `value_vs_adp` | `adp_rank − overall_rank`; positive = falls to you |
| `p_avail_next`, `opportunity_cost` | on-the-clock decision support |
| `playoff_sos` | see §14.5 — weight weeks 15–17 separately |

Sort default: VOR descending. Provide filters by position and tier.

### 9.8 Live draft assistant

Sleeper exposes the draft and its picks:

```
GET /league/{league_id}/drafts
GET /draft/{draft_id}
GET /draft/{draft_id}/picks
```

Poll `/picks` every 5–10 seconds during the draft (well within rate limits). Maintain the available pool by removing drafted `player_id`s.

Display, in priority order:

1. **Best available by VOR**, with tier and opportunity cost.
2. **Tier depth remaining** per position — "3 left in RB tier 4" is the single most actionable number on the screen.
3. **Positional run detection** — compare the position mix of the last 8 picks to the expected mix; flag when a position is going at 2× its baseline rate.
4. **Your roster's current starting-lineup gaps** given `LeagueFormat`.

Keep this page dead simple and fast. It will be used under time pressure.

---

## 10. Feature engineering

### 10.1 The as_of contract

Every feature is defined with an explicit availability rule. The `features/registry.py` module holds, for each feature, a record:

```python
@dataclass
class FeatureSpec:
    name: str
    description: str
    positions: list[str]          # which positions it applies to
    window: str | None            # e.g. "ewm_4", "season_to_date", "prior_season"
    source_table: str
    available_at_inference: bool  # False = training-only (e.g. route data in-season)
    lag_weeks: int                # minimum lag; 1 means "uses data through week W−1"
```

`features/build.py` asserts that every feature in a training matrix has `lag_weeks >= 1` relative to the target week, and that every feature used by an inference model has `available_at_inference=True`. This assertion is a test, not a comment.

### 10.2 Feature catalogue

Windows: `ewm_k` = exponentially weighted mean with span k weeks; `std_k` = rolling standard deviation.

**Player usage (highest value block)**

| Feature | Definition | Windows | Positions |
|---|---|---|---|
| `snap_pct` | offensive snaps / team offensive snaps | ewm_3, ewm_8, prior_season | all off |
| `snap_pct_trend` | ewm_3 − ewm_8 | — | all off |
| `target_share` | targets / team pass attempts | ewm_3, ewm_8, season_to_date | WR/TE/RB |
| `air_yards_share` | player air yards / team air yards | ewm_4 | WR/TE |
| `wopr` | 1.5·target_share + 0.7·air_yards_share | ewm_4 | WR/TE/RB |
| `adot` | air yards / targets | ewm_8 | WR/TE |
| `carry_share` | carries / team rush attempts | ewm_3, ewm_8 | RB/QB |
| `rz_touch_share` | (rz targets + rz carries) / team rz touches | ewm_6 | RB/WR/TE |
| `gz_carry_share` | carries inside 5 / team carries inside 5 | ewm_6 | RB |
| `xfp_per_game` | ffopportunity expected fantasy points | ewm_4, season_to_date | all off |
| `xfp_minus_actual` | efficiency residual — regresses hard, use as a *negative* indicator of sustainability | ewm_6 | all off |
| `weeks_in_current_role` | weeks since snap_pct changed by >15pp | — | all off |
| `pass_attempts`, `cpoe`, `sack_rate_taken` | QB volume and efficiency | ewm_4 | QB |
| `designed_rush_share`, `rush_yards_per_game` | QB rushing floor — large fantasy signal | ewm_6 | QB |
| `points_std` | volatility of own league-scored points | std_8 | all |

**Team context**

| Feature | Definition | Windows |
|---|---|---|
| `implied_team_total` | from Vegas total and spread | current week |
| `spread` | signed, team perspective | current week |
| `plays_per_game` | offensive plays | ewm_5 |
| `neutral_pace` | sec/play, score within 7, Q1–Q3 | ewm_8 |
| `proe` | pass rate over expectation | ewm_5 |
| `team_epa_off`, `team_success_off` | offensive quality | ewm_8 |
| `ol_continuity` | starting OL snaps with same 5 | ewm_5 |
| `teammate_vacated_target_share` | sum of target_share of teammates ruled Out | current week |
| `teammate_vacated_carry_share` | as above for carries | current week |

The two `vacated_*` features are where injury news becomes a number. They are frequently worth more than the injured player's own designation.

**Opponent**

| Feature | Definition |
|---|---|
| `def_adj_epa_allowed_<group>` | ridge-adjusted EPA allowed to the player's position group |
| `def_adj_ypt_allowed_<group>` | adjusted yards per target allowed |
| `def_adj_td_rate_allowed_<group>` | adjusted TD rate allowed |
| `def_adj_rush_epa_allowed` | for RB/QB rushing |
| `def_pressure_rate`, `def_sack_rate` | matters for QB and for DST projections |
| `def_pace_faced` | opponent's own pace, affects total plays |

**Situation**

| Feature | Definition |
|---|---|
| `is_home` | |
| `rest_days` | |
| `wind_mph`, `precip_prob`, `temp_f`, `is_dome` | dome forces wind to 0 |
| `report_status` | Out / Doubtful / Questionable / None, as of the Friday report |
| `practice_participation` | DNP / Limited / Full |
| `weeks_since_return` | games since last missed game due to injury |
| `is_primetime` | |
| `week_number` | captures late-season rest/tanking effects |

### 10.3 Weather

Only wind has a large, reliable effect, and mainly on passing and kicking. Do not over-engineer this. Rules:

- If `roof` is `dome` or `closed`, set `wind_mph = 0`, `precip_prob = 0`, `temp_f = 70`.
- Otherwise fetch from Open-Meteo at the stadium coordinates for the kickoff hour.
- For backtesting use the Open-Meteo historical archive (actual conditions). For live projections use the forecast. Record which was used — a model trained on actuals and served forecasts has a subtle distribution shift, and it is worth measuring how large it is.

### 10.4 Opponent adjustment (positional strength of schedule)

**Do not use "fantasy points allowed to position."** It is confounded two ways: by the quality of offences that defence happened to face, and by garbage-time volume inflating totals for defences that led a lot. It is the single most common analytical error in public fantasy content.

Correct method — a ridge-regularised two-way adjustment on *rate* outcomes:

```
For each position group g and each rate outcome y (EPA/play, yards/target, TD rate):

  Fit over the trailing window of player-week (or play-level) observations:
      y = μ + offense_team + defense_team + home + ε
  using ridge regression on one-hot encoded team factors.

  The fitted defense_team coefficients are the opponent-adjusted values.
```

Details that matter:

- **Shrinkage.** Early in the season the estimates are noise. Blend with the prior-season estimate: `w = n_plays / (n_plays + k)` with `k ≈ 250` plays for the group. Report `n_plays` alongside every estimate so the UI can grey out low-confidence matchup grades.
- **Recency weighting.** Exponentially weight the trailing window (span ≈ 8 games). Defences change through a season — injuries to a secondary, coordinator adjustments.
- **Position groups** (this is where the Broncos/coverage intuition lives): `WR_perimeter`, `WR_slot`, `TE`, `RB_receiving`, `RB_rushing`, `QB_passing`, `QB_rushing`. A defence can be genuinely elite against outside receivers and poor against tight ends; a single "pass defence" number erases exactly the distinction you want.
- **Alignment classification.** Perimeter vs slot requires alignment data. nflverse PBP does not carry it directly for all seasons. Approximate from average depth of target and receiver position, or accept the limitation and use `WR` as one group in v1, splitting it in v2 if charting data is acquired.

**Honest calibration — read this before building a big matchup UI.** Opponent effects are real but much smaller than fantasy media implies, especially for RB and WR, where week-to-week variance is dominated by usage and touchdown luck. Matchup carries the most weight for DST, kickers, tight ends, and rushing quarterbacks. Build the feature, let the model weight it, and resist any UI design that lets a green "great matchup" badge override a declining usage trend. If the model assigns these features low importance, that is a finding, not a bug.

### 10.5 The route-data availability problem

`route_participation` and any derived features are available for historical seasons but **not in-season**. Handle explicitly:

- Tag those features `available_at_inference=False`.
- Train two model variants per position: `full` (with route features, for backtesting research and offseason analysis) and `live` (without, for in-season inference).
- Report both in the evaluation report so the cost of the missing data is quantified rather than assumed.
- Never impute in-season route participation from a training-set mean. A constant-valued feature at inference is worse than an absent one because the model still spends splits on it.

---

## 11. Modelling

### 11.1 Target definition

The target is **league-scored fantasy points for (player, season, week)**, computed by the scoring engine from actual stats.

Rows are generated for every player on an active roster that week, including those who did not play. A player who was inactive has target 0. Excluding them creates survivorship bias and produces a model that systematically overprojects, because it has only ever seen players who were healthy enough to play.

### 11.2 Architecture: hurdle model

Fantasy point distributions are zero-inflated (inactive, healthy scratch, third-string) and right-skewed conditional on playing. A single regressor handles this badly. Use a two-part (hurdle) structure:

```
E[points]        = P(plays) × E[points | plays]
Quantiles        = quantiles of the mixture:
                     mass (1 − P(plays)) at 0,
                     plus P(plays) × conditional quantile distribution
```

**Part A — availability model.** Binary classifier, LightGBM, target = "recorded ≥1 offensive snap." Features: injury report status, practice participation, weeks since return, depth chart position, snap trend, position, age. Output `p_active`. Calibrate with isotonic regression on a held-out validation period — raw GBM probabilities are not well calibrated, and this probability is multiplied through everything downstream.

**Part B — conditional points model.** Trained only on rows where the player played. One model per position.

### 11.3 v1: direct conditional model

Ship this first. LightGBM regression on league points, conditional on playing, with `xfp` (ffopportunity expected fantasy points) as an input feature alongside the usage, team, opponent, and situation blocks from §10.2.

Using someone else's expected-points model as a feature rather than rebuilding it is not cheating; it is the correct engineering decision. It gives you a strong opportunity signal on day one and lets you spend your effort on the parts nobody has done for you.

Monotonic constraints (LightGBM `monotone_constraints`) are worth setting where the direction is not in doubt — it costs a little fit and buys robustness and sane extrapolation:

- increasing: `target_share`, `carry_share`, `snap_pct`, `rz_touch_share`, `implied_team_total`, `xfp`
- decreasing: `def_adj_epa_allowed_*` where lower means a tougher defence (verify sign at implementation)

### 11.4 v2: decomposed pipeline

Build after v1 is evaluated and beaten baselines. The decomposition exists because opportunity is stable and efficiency is close to noise; modelling their sum lets the noise swamp the signal.

```
Stage 1  Team environment
         inputs : Vegas total & spread, pace, PROE, opponent pace
         outputs: team_plays, team_pass_attempts, team_rush_attempts

Stage 2  Opportunity
         inputs : Stage 1 outputs, player usage features, vacated shares
         outputs: expected targets, carries, red-zone touches per player

Stage 3  Efficiency priors
         inputs : player efficiency history, opponent adjusted rates
         outputs: yards per target, yards per carry, TD probability per touch
         RULE   : shrink hard toward positional means. Empirical Bayes with a
                  prior weight equivalent to ~50 targets or ~80 carries.

Stage 4  Recombination
         Monte Carlo: sample opportunity, sample efficiency, apply league
         scoring, produce the full outcome distribution.
```

Do not skip straight to v2. Without a v1 baseline and a working evaluation harness you will have no way to tell whether the decomposition helped.

### 11.5 Quantile models

Train separate LightGBM models with `objective="quantile"` at each alpha in `model.quantiles`, per position, on the conditional (played) rows.

- **Quantile crossing** is expected and must be fixed: after prediction, sort the quantile vector per row ascending. Log the frequency of crossings; a high rate signals an underfit or unstable model.
- **Recalibration:** on the validation period, check empirical coverage of each nominal quantile. If the 80% interval covers 71%, apply a per-position scalar width correction and re-check. Coverage is reported in every evaluation run.
- The mixture with `p_active` is applied afterwards to produce unconditional quantiles, since a player with `p_active = 0.5` has a genuine floor of 0.

### 11.6 DST model

Separate model, different feature set, and one of the easier wins in the project because most managers set and forget their defence.

Features: opponent implied team total, opponent sack rate allowed, opponent pressure rate allowed, opponent turnover-worthy play rate, opponent interception rate, opponent offensive line continuity, own defensive pressure rate, own takeaway rate, home/away, weather.

Do not attempt to project individual defensive players. Project team defence scoring directly from the above.

### 11.7 Kicker model

Be honest about this: kicker scoring is close to irreducible noise beyond team implied total and dome status. Implement a minimal model (implied total, dome, opponent red-zone-to-touchdown rate allowed) and spend no further time on it. Any effort spent improving kicker projections is effort not spent on the RB/WR models where the leverage is.

### 11.8 Model versioning

Every trained model is written to `data/models/<position>/<model_version>/` containing the serialised booster, the feature list, hyperparameters, the training cutoff, and the evaluation report.

```
model_version = sha256(feature_names + hyperparams + train_cutoff + code_version)[:12]
```

Predictions record `model_version` and `feature_hash`. Any projection in `outputs/` must be traceable to the exact model and feature set that produced it.

---

## 12. Evaluation and backtesting

This section is the difference between a system that works and one that only appears to. Build the harness before building v2 of the model.

### 12.1 The as_of snapshot

`evaluation/snapshot.py` provides:

```python
def snapshot(tables: dict[str, pl.DataFrame], as_of: datetime) -> dict[str, pl.DataFrame]:
    """Return every table filtered to rows knowable at `as_of`."""
```

For a projection of (season, week), `as_of` is the **kickoff time of the first game of that week**, minus a configurable safety margin. Injury designations use `date_modified`; stats use game completion time; Vegas lines use the line timestamp (or, for historical backtests using closing lines, be explicit that closing lines are slightly optimistic relative to what you would have had on Thursday, and document the bias).

Add a test (`tests/test_leakage.py`) that, for a sample of backtest weeks, asserts no feature row has a source timestamp later than its `as_of`.

### 12.2 Walk-forward protocol

```
FOR season S in [validation seasons]:
  FOR week W in 1..18:
      train_rows = all rows with (season, week) strictly before (S, W)
                   and season >= settings.seasons.train_start
      IF len(train_rows) < min_train_rows: skip
      fit model on train_rows
      predict week (S, W) using snapshot(as_of = kickoff(S, W))
      store predictions
```

Retrain cadence is configurable. Weekly refit is affordable at this data size and is the default; a 4-week cadence is available for faster experimentation.

**There is no random splitting anywhere in this project.** k-fold cross-validation over player-weeks leaks the same game into train and test through team-level features and produces validation metrics that are simply false.

### 12.3 Baselines

Implemented in `models/baselines.py`. Every evaluation report compares against all of them.

| ID | Baseline | Purpose |
|---|---|---|
| B0 | Positional weekly mean | Sanity floor |
| B1 | Player's season-to-date mean | Naive but surprisingly hard |
| B2 | Player's `ewm_4` of own points | The real bar for "did my features do anything" |
| B3 | Public consensus weekly projections | The bar for "should I use my model at all" |

Rule: if the model does not beat **B2**, the features are not working. If it does not beat **B3**, ship B3 and keep working on the model in the background. There is no shame in this; beating a consensus of full-time analysts with paid charting data is genuinely hard.

### 12.4 Metrics

**Accuracy**
- MAE and RMSE, overall and per position, on all rows and on "startable" rows only (players rostered and above a projection threshold — accuracy on the deep bench does not matter).

**Ranking**
- Spearman rank correlation within each (position, week). This is what a ranking product is actually measured on.
- Top-k precision: of the true top 12 RBs in a week, how many were in your projected top 12?

**Distribution**
- Pinball loss per quantile.
- Empirical coverage of the 50% and 80% intervals, per position.
- Calibration plot: predicted quantile vs realised frequency.

**Decision quality (the metrics that actually matter)**
- **Start/sit accuracy:** over historical weeks, construct realistic pairwise choices (two flex-eligible players on the same roster) and measure how often the model's pick outscored the alternative, versus the same measure for each baseline.
- **Lineup regret:** points left on the bench per week. `optimal_lineup_points − model_recommended_lineup_points`, averaged. This is the single number that best summarises "how much did this system cost or save me."

### 12.5 Statistical honesty

A season is ~17 weeks; a single league is 12 rosters. Differences of 0.1 MAE between two models over one season are noise.

- Evaluate over at least four validation seasons.
- Report bootstrap confidence intervals (resample by week, not by row — rows within a week are correlated).
- State the number of observations behind every reported metric.
- Resist the temptation to iterate on the validation seasons repeatedly; that is slow-motion overfitting. Hold out the most recent season entirely until you are ready to make a final go/no-go call.

### 12.6 Evaluation report

`ffapp evaluate --seasons 2021 2022 2023 2024 2025` writes `data/outputs/eval/<timestamp>/report.md` containing every metric above, per position, versus every baseline, with feature importances and calibration plots. Reports are kept, not overwritten — the history of what you tried is the most valuable artefact of the offseason.

---

## 13. Simulation layer

### 13.1 Lineup optimiser (`sim/lineup.py`)

Given a roster, per-player projected points, and `LeagueFormat`, return the optimal starting lineup.

Formulate as a small ILP with `pulp`: binary variable per (player, slot), constraints that each slot is filled exactly once, each player is used at most once, and slot eligibility is respected. This handles FLEX, SUPER_FLEX, and multi-flex formats exactly. Problems of this size solve in milliseconds.

```python
def optimal_lineup(
    players: list[PlayerProjection],
    fmt: LeagueFormat,
    objective: Literal["mean", "median", "ceiling"] = "mean",
) -> Lineup
```

Also expose `optimal_lineup_points(actual_points, fmt)` for computing lineup regret in evaluation.

### 13.2 Weekly simulation with correlation (`sim/week.py`)

Independent sampling of player scores is wrong and will make your win probabilities badly calibrated. A QB and his WR1 rise and fall together; two RBs on the same team split a fixed pool of carries; your player scoring against the DST you started hurts that DST.

Method — Gaussian copula:

1. For each player, build an empirical marginal CDF by interpolating the predicted quantiles (with a mass point at 0 of size `1 − p_active`), using a fitted tail beyond q90.
2. Build a correlation matrix from the pairwise rules in `settings.simulation.correlation`. Start with the configured constants; estimate them empirically from historical data in Phase 3.
3. Sample multivariate normal, transform through the normal CDF to uniforms, then through each player's inverse marginal CDF.
4. Nearest-positive-definite correction on the correlation matrix before sampling.

Output: `week_sims × n_players` matrix of sampled scores. Everything else (win probability, start/sit, DST value) reads from this.

### 13.3 Injury hazard model (`sim/injury.py`)

Discrete-time hazard: `P(misses game w | played through w−1, covariates)`.

Covariates: position, age, games missed in the prior two seasons, current injury designation, weeks since return from injury, snap load trend. Fit a logistic model or a small GBM on historical nflverse injury and snap data.

Outputs `p_miss[player, week]`, consumed by the season simulator and by the games-played adjustment in §9.3.

Keep this simple. It exists to stop the system pretending everyone plays 17 games; precision beyond that is not where the value is.

### 13.4 Season simulator (`sim/season.py`)

Monte Carlo over the remainder of the season.

```
FOR sim in 1..season_sims:
    FOR week in remaining_weeks:
        sample availability per player from injury hazard (with persistence:
            a multi-week injury keeps the player out for a sampled duration,
            not resampled independently each week)
        sample correlated weekly scores for all rostered players (§13.2)
        set each team's optimal lineup from its *projected* points
            (NOT the sampled actuals — managers do not know outcomes in advance;
             using actuals here inflates every team's simulated performance and
             is a subtle but serious bug)
        record head-to-head result against the league schedule
    determine playoff seeding, simulate the bracket
    record: wins, playoff berth, title

OUTPUT per team: E[wins], P(playoffs), P(title)
```

That parenthetical is the most commonly botched detail in fantasy season simulators. Lineups are set on expectation; results are drawn from the sample.

The season simulator is the engine behind the trade analyzer, the trade finder, and the season-long rankings. Build it once, correctly.

---

## 14. Application features

Each subsection specifies inputs, algorithm, and output. All of them read from `outputs/projections.parquet` and the simulation layer; none of them re-implement modelling logic.

### 14.1 Weekly rankings

**Inputs:** projections for the target week, `LeagueFormat`, optional roster filter.

**Output:** table with `player, position, team, opponent, p_active, proj_mean, floor (q10), median (q50), ceiling (q90), matchup_grade, n_plays_behind_matchup_grade`.

Sortable by any column, filterable by position and by availability (all / my roster / free agents / rostered elsewhere). Default sort is `proj_mean` descending within the selected position.

Show floor and ceiling as a visible range, not a hidden column. The range is the differentiating output of this system; burying it defeats the purpose.

### 14.2 Season-long rankings

**Inputs:** rest-of-season weekly projections, injury hazard, `LeagueFormat`.

**Algorithm:** run the season simulator at the player level. For each player, aggregate across sims: expected total points, distribution of season totals, expected games played. Then compute VOR using the §9.4 replacement-level algorithm applied to rest-of-season values.

**Output:** `player, position, expected_points_ros, p10/p50/p90 season total, expected_games, vor_ros, tier`.

Rank by `vor_ros`. Never by raw projected points — a QB who outscores every RB is not more valuable in a single-QB league, and ranking by points implies otherwise.

### 14.3 Start/sit assistant

The feature most tools get wrong. The correct objective is **probability of winning your matchup**, not expected points.

**Inputs:** your roster, opponent's roster, projections, `LeagueFormat`.

**Algorithm:**

```
1. Determine opponent's likely lineup (optimal by projection).
2. Simulate opponent's total score distribution (§13.2).
3. Enumerate candidate lineups for your roster:
     - the projection-optimal lineup
     - all single-swap variants at each flex-eligible slot
4. For each candidate lineup:
     simulate your total, jointly with the opponent's
       (shared correlation matrix — same-game players on both rosters matter)
     P(win) = fraction of sims where your total > opponent total
5. Rank candidates by P(win).
```

**Why this matters:** if you are a heavy underdog, the highest-P(win) lineup is not the highest-projected one — it is the higher-variance one, because you need an outcome in the tail. If you are a heavy favourite, the correct play is the safe floor. Expected points cannot express either of these.

**Output:** recommended lineup with `P(win)`, plus a table of each considered swap showing `Δ projected points` and `Δ P(win)`. Display both, because they will sometimes disagree, and the disagreement is the insight.

### 14.4 Waiver wire

**Inputs:** free agent pool (rosters from Sleeper, minus all rostered players), rest-of-season projections, your roster, `LeagueFormat`, FAAB budget, Sleeper trending adds.

**Algorithm:**

```
FOR each free agent p:
    baseline = expected weekly starting-lineup points of my roster
    with_p    = expected weekly starting-lineup points of (my roster + p,
                worst bench player dropped)
    value_added_per_week = with_p − baseline
    ros_value = value_added_per_week × weeks_remaining
    (weight later weeks slightly higher if they fall in the fantasy playoffs)
```

This is the key idea: **value is relative to your roster, not absolute.** A free agent projected for 11 points a week is worth nothing to you if your three worst starters already project for 13. Every public waiver list ignores this because it cannot know your roster; yours can.

**FAAB guidance:**

```
suggested_bid = clamp(
    ros_value / total_ros_value_of_remaining_budget_opportunities × remaining_budget,
    floor=1, ceiling=remaining_budget
)
```

Calibrate the aggressiveness parameter against your league's actual bidding history, which is available in full from the Sleeper transactions endpoint. Learning that your league systematically overbids on running backs is worth real money.

**Market signal:** join Sleeper's trending adds counts. High trend + high value-to-you means bid aggressively; high value-to-you + low trend means you may win them cheaply.

**Output:** ranked list with `player, position, value_added_per_week, ros_value, suggested_bid, trend_rank, drop_candidate`.

### 14.5 Strength of schedule and matchup views

**Positional SOS.** For each team and position group, sum the opponent-adjusted difficulty (§10.4) across a week range. Provide three ranges: full season, rest of season, and **fantasy playoffs (weeks `playoff_week_start` through 17)**.

Playoff SOS is the one most people ignore and it is worth more at the draft than full-season SOS, because full-season SOS is largely averaged away by week 14 while the playoff weeks are the ones that decide your season.

**Schedule grid.** Heatmap: teams (rows) × weeks (columns). Cell colour = matchup difficulty for the currently selected position group. Bye weeks rendered as blocked-out cells. Overlay a toggle to highlight only players on your roster.

**Matchup detail view.** For a given player-week, show the components: opponent's adjusted rates for that player's position group, the sample size behind the estimate, the implied team total, and the projected game script. Grey out grades with `n_plays` below a confidence threshold rather than displaying a confident-looking colour derived from three games of data.

**Required honesty in the UI:** display matchup grade alongside usage trend, never in isolation, and never as the largest element on the card. See the calibration note in §10.4.

### 14.6 Trade analyzer

**Inputs:** proposed trade (players from team A, players from team B), both rosters, rest-of-season projections, league schedule.

**Algorithm:**

```
1. Run season simulator for the current league state.
   Record P(playoffs), P(title), E[wins] for both teams.
2. Apply the trade. Re-run the simulator with identical random seeds
   (common random numbers — this dramatically reduces the variance of the
   difference and is essential at 3000 sims).
3. Report Δ for both teams.
4. Also report the naive value delta (sum of ROS VOR) for reference.
```

**Why simulate rather than sum values:** trade value is not additive because lineups have fixed slots. Trading two startable WRs for one elite WR reduces your points if you start three WRs; it increases them if you start two and your WR3 was replacement level anyway. Only a lineup-aware simulation captures this.

**Output:** for each side, `Δ E[wins]`, `Δ P(playoffs)`, `Δ P(title)`, plus the naive VOR delta and a plain-language summary of which positions each side gained and lost.

Show the other team's deltas too. A trade that helps you and visibly helps them is a trade that gets accepted.

### 14.7 Trade finder

Combinatorially expensive if done naively; use a two-stage filter.

```
STAGE 1 — candidate generation (cheap)
  FOR each other roster R:
      compute positional surplus/deficit for R and for me
        surplus = players above replacement who cannot crack the starting lineup
      generate candidate packages:
        1-for-1 and 2-for-1 among the top ~10 tradeable players per roster
      score with a fast surrogate:
        Δ expected weekly optimal-lineup points, for both sides
      keep candidates where BOTH sides improve on the surrogate

STAGE 2 — evaluation (expensive)
  Take the top ~25 candidates by combined surrogate gain.
  Run the full season simulator with common random numbers.
  Rank by my Δ P(playoffs), filtered to candidates where the other side's
  Δ P(playoffs) is also positive.

STAGE 3 — acceptance likelihood
  Compute the naive VOR delta as the other manager will perceive it.
  Flag trades that are mutually beneficial but *look* lopsided by conventional
  value charts — these need framing, and the UI should say so.
```

**Output:** ranked list of proposals with both sides' simulated deltas, the perceived-value delta, and a one-line rationale ("you are deep at WR and thin at TE; they are the reverse").

### 14.8 News and injury pipeline

**Ingestion:** nflverse official injury reports (structured) plus an RSS layer for beat reporting.

**Structuring:** send each unstructured item to the Anthropic API with a strict JSON schema:

```json
{
  "player_name": "string",
  "team": "string",
  "event_type": "injury|role_change|trade|suspension|depth_chart|other",
  "severity": "none|minor|moderate|major|season_ending",
  "expected_usage_change": "increase|decrease|none|unknown",
  "affected_teammates": ["string"],
  "confidence": "low|medium|high",
  "effective_week": "int|null"
}
```

Require JSON-only output, parse defensively, and route anything with `confidence: low` or an unresolvable player name to a manual review queue rather than into the pipeline.

**Second-order propagation — the valuable part.** When a player is ruled out:

1. Recompute `teammate_vacated_target_share` and `teammate_vacated_carry_share` for the team.
2. Re-run projections for all affected teammates.
3. Adjust the team's projected pass rate if the change is at QB.
4. Surface the handcuff or next-man-up on the waiver board with the recomputed value.

The headline "RB1 is out" is on every fantasy site within minutes. The automatically recomputed projection for RB2, the team's revised run rate, and the FAAB bid for the third-string back are not. That cascade is the feature.

**Boundary:** the LLM converts text to structure and writes explanations. It never produces a number that feeds a ranking. Projections come from the gradient boosters.

---

## 15. User interface (v1, Streamlit)

Pages, in build order:

1. **Draft board** — the §9.7 table with filters; live draft mode (§9.8) as a sub-tab.
2. **Weekly rankings** — §14.1, with position tabs and the floor/ceiling range visible.
3. **My team** — roster with projections, recommended lineup, and P(win) for the week.
4. **Start/sit** — §14.3, swap comparison table.
5. **Waivers** — §14.4 ranked list with bid guidance.
6. **Schedule & SOS** — §14.5 heatmap and bye-week grid.
7. **Trades** — analyzer and finder.
8. **Season outlook** — playoff odds, season-long rankings, simulated standings.
9. **Model health** — latest evaluation report, calibration plots, baseline comparison. Keep this visible. Being reminded weekly of how your model is actually performing is the main defence against trusting it too much.

Design constraints: fast to load (everything precomputed, nothing trained on page load), works on a laptop during a draft, no login. Cache aggressively with `st.cache_data` keyed on `model_version` and `as_of`.

---

## 16. Operations

### 16.1 Pipeline schedule (in-season)

| When | Command | Purpose |
|---|---|---|
| Tue 06:00 | `ffapp ingest all --season 2026` | Post-Monday-night stats, snaps, transactions |
| Tue 07:00 | `ffapp features build && ffapp train --incremental` | Refresh features, retrain |
| Tue 07:30 | `ffapp project --week N+1 && ffapp waivers` | Waiver-day projections and bids |
| Thu 09:00 | `ffapp ingest injuries odds && ffapp project --week N` | Final injury report, opening lines |
| Sun 10:00 | `ffapp ingest injuries && ffapp project --week N && ffapp startsit` | Inactives, final lineup call |

Run under local cron initially. Every command is idempotent and logs to `data/outputs/logs/`.

### 16.2 CLI surface

```
ffapp ingest <source|all>        --season --weeks --force
ffapp ids check
ffapp scoring validate           # the golden test against Sleeper
ffapp features build             --season --weeks
ffapp train                      --position --through-week --incremental
ffapp evaluate                   --seasons --model-version
ffapp project                    --week
ffapp draft board                --sources
ffapp draft live                 --draft-id
ffapp startsit                   --week
ffapp waivers                    --week
ffapp trade analyze              --give --get --with-roster
ffapp trade find                 --max-candidates
ffapp sim season                 --sims
ffapp ui
```

### 16.3 Testing

Mandatory tests, in order of importance:

1. `test_scoring.py` — the §8.4 golden test against Sleeper's own computed points. **Blocking.**
2. `test_leakage.py` — as_of contract; no feature sources a value later than its snapshot time.
3. `test_ids.py` — zero unmatched players in the top 300 by ADP.
4. `test_lineup.py` — the ILP produces known-correct lineups on hand-built fixtures, including FLEX and superflex edge cases.
5. `test_vor.py` — replacement-level fixed point converges and gives known values for standard 12-team formats.
6. `test_baselines.py` — baselines are implemented correctly (they are the yardstick; a buggy baseline flatters the model).
7. `test_no_network.py` — Addendum 02 §A; every module outside `ingest/` runs with sockets monkeypatched to raise, proving no accidental network dependency leaked past the ingest layer.
8. `test_offline_raises.py` — Addendum 02 §C.2; a cache miss under `FFAPP_OFFLINE=1` raises rather than silently returning an empty frame.

Fixtures use small committed sample data under `tests/fixtures/`, never live network calls.

### 16.4 Logging and reproducibility

- Structured logging (`structlog` or stdlib JSON) to `data/outputs/logs/`.
- Every output artefact carries `model_version`, `as_of_utc`, and the git commit hash.
- `ffapp project` refuses to run if the working tree is dirty and `--allow-dirty` is not passed. Reproducing "what did the model say on Sunday morning" is worth this small friction.

### 16.5 If this ever goes public

Not a v1 concern, but decisions made now that would be expensive to reverse:

- Keep all league-specific logic behind `LeagueFormat`. Nothing hardcodes this league.
- Keep the scoring engine general over Sleeper's full scoring key space.
- Keep data source licences recorded at ingest (§6.1). Re-audit before any redistribution; CC-BY-SA on ffopportunity propagates to derived data.
- Storage swap from Parquet to Postgres is the main migration; keep all data access behind a thin repository layer so it is a single-module change.

---

## 17. Open decisions

These must be resolved before the affected module is built. They are genuinely open — do not guess.

| # | Decision | Blocks | Default if unresolved |
|---|---|---|---|
| 1 | League format specifics: team count, PPR value, superflex?, starting slots, bench size | §8.5, §9.4, everything | Pull from Sleeper on first run; no default |
| 2 | Which 5+ ranking sources to aggregate, and which publish per-stat projections vs ranks only | §9.2 | FantasyPros consensus + ranks-only fallback |
| 3 | Paid odds API, or nflverse schedule lines only? | §10.2 team context | nflverse lines; live Vegas is a nice-to-have |
| 4 | Purchase charting data (PFF / Fantasy Points) for route participation and coverage scheme? | §10.5, §10.4 WR splits | No; use snap+target share proxies |
| 5 | Draft slot and league draft date | §9.6 survival probabilities | Unknown until draft order is set |
| 6 | Waiver type (FAAB vs rolling priority) and budget | §14.4 | Read from Sleeper league settings |
| 7 | Training start season — 2015 default; earlier data is a different NFL | §11 | 2015 |
| 8 | Keep a manual review queue for LLM-parsed news, or auto-apply high-confidence items? | §14.8 | Manual review for v1 |

---

## 18. Appendix A — Sleeper endpoint reference

Base URL: `https://api.sleeper.app/v1`. No authentication. Read-only. Stay under 1000 requests/minute.

```
GET /user/{username}                          → user object (capture user_id)
GET /user/{user_id}/leagues/nfl/{season}      → list of leagues
GET /league/{league_id}                       → settings, scoring_settings, roster_positions
GET /league/{league_id}/rosters               → roster composition + owner_id
GET /league/{league_id}/users                 → display names, team names
GET /league/{league_id}/matchups/{week}       → starters, points, players_points  ← scoring oracle
GET /league/{league_id}/transactions/{week}   → waivers, trades, FAAB bids
GET /league/{league_id}/drafts                → draft ids
GET /draft/{draft_id}                         → draft settings, slot→roster mapping
GET /draft/{draft_id}/picks                   → picks (poll during live draft)
GET /players/nfl                              → full player dictionary  ← ONCE PER DAY MAX
GET /players/nfl/trending/add?lookback_hours=24&limit=25   → market signal
GET /players/nfl/trending/drop?lookback_hours=24&limit=25
```

Notes:

- Usernames change; store `user_id`.
- `/players/nfl` is a multi-megabyte payload. Cache it. Never call it in a loop.
- `players_points` in the matchups response is Sleeper's own scoring output and is the oracle for §8.4.
- There is no write access. No endpoint sets a lineup or submits a waiver claim. The system recommends; you execute in the app.

## 19. Appendix B — glossary

| Term | Meaning |
|---|---|
| ADP | Average draft position |
| aDOT | Average depth of target |
| ECR | Expert consensus ranking |
| EPA | Expected points added, per play |
| FAAB | Free agent acquisition budget |
| PROE | Pass rate over expectation |
| SOS | Strength of schedule |
| VOR / VORP | Value over replacement (player) |
| WOPR | Weighted opportunity rating: 1.5×target share + 0.7×air yards share |
| xFP | Expected fantasy points, from opportunity alone |
| Hurdle model | Two-part model: P(event occurs) × E[magnitude \| occurs] |
| Common random numbers | Using identical random seeds across simulation scenarios so their difference has lower variance |
| Pinball loss | The loss function for quantile regression |
