# CLAUDE.md — working instructions for this repository

## What this project is

A personal fantasy football decision-support system for one Sleeper league, built around a self-trained player projection model. Single user, local-first. `SPEC.md` is the source of truth for design; `TASKS.md` is the ordered work queue.

Read `SPEC.md` §0 before doing anything else.

## Session start

Read `HANDOFF.md` at the start of every session — it's short by design and holds current state, the next task, and anything left mid-implementation. Read `docs/JOURNAL.md` only when investigating a specific past task's evidence, an implementation decision, or a gotcha — it's long, append-only, and not meant to be read cover to cover. Read `docs/summary.md` when you want the whole project in one pass — what it is, current state, what it does today, how it's performing — without reconstructing it from `HANDOFF.md`'s pointers or `JOURNAL.md`'s task-by-task log; it's a standing narrative snapshot, not append-only, so it can go stale and should be refreshed when it does.

## Standing rules

These are not style preferences. Violating any of them silently invalidates work downstream.

1. **No random train/test splits.** All model validation is walk-forward in time. If you find yourself typing `train_test_split` or `KFold`, stop — see SPEC §12.2.
2. **Every feature respects the as_of contract.** A feature for week W uses only data that existed before week W's first kickoff. See SPEC §10.1 and §12.1.
3. **The scoring engine is validated before it is trusted.** `ffapp scoring validate` must pass (SPEC §8.4) before any projection, ranking, or valuation is generated.
4. **Never silently drop rows in a join.** Unmatched player IDs fail loudly (SPEC §7). A dropped WR2 is a bug you will never notice.
5. **Nothing hardcodes this league's format.** Team count, starting slots, and scoring always come from `LeagueFormat` and the league's `scoring_settings`.
6. **Beat the baselines before believing the model.** SPEC §12.3.

## Conventions

**Environment**
- Python 3.11+, managed with `uv`. `uv sync` to install, `uv run` to execute.
- Never `pip install` into the system environment.
- Secrets from `os.environ` only, loaded from `.env`. Never logged, never committed, never written into `data/`.

**Code style**
- Type hints on every public function. `ruff` for lint and format, `mypy` on `src/`.
- Polars is the default dataframe library. Convert to pandas only at a model-fitting boundary that requires it, and convert back immediately.
- Dataclasses for structured config objects (`LeagueFormat`, `FeatureSpec`, `PlayerProjection`).
- No business logic in `ingest/` beyond schema normalisation. No network calls outside `ingest/`.
- `notebooks/` is scratch. No module under `src/` may import from it.

**Data**
- Parquet everywhere, partitioned by season where the table spans seasons.
- Raw source payloads are archived unmodified under `data/raw/<source>/` with a sidecar JSON recording `{source, fetched_at_utc, rows, call, package_version}`.
- All ingest is idempotent — re-running for the same season/week overwrites cleanly.
- `data/` is gitignored. `config/` is committed, including `config/id_overrides.csv`.

**Testing**
- pytest. Fixtures live in `tests/fixtures/` as small committed files.
- No live network calls in tests, ever. Mock or fixture.
- The six mandatory tests in SPEC §16.3 are blocking; do not mark a task complete with any of them failing.

**Commits and reproducibility**
- Conventional commits (`feat:`, `fix:`, `test:`, `refactor:`, `docs:`).
- Every output artefact records `model_version`, `as_of_utc`, and the git commit hash.
- One task from `TASKS.md` per branch, one PR-sized change per commit series.

## How to work through tasks

1. Pick the next unblocked task from `TASKS.md` in order. Do not skip ahead — the ordering encodes real dependencies.
2. Re-read the referenced `SPEC.md` section before writing code.
3. If the spec is ambiguous or appears wrong, **stop and ask** rather than guessing. Ambiguity resolved by guessing produces code that looks right and is silently wrong, which is the most expensive failure mode in this project.
4. Write the test alongside the implementation, not after.
5. Update the task's checkbox in `TASKS.md` only when its acceptance criteria are demonstrably met.

## Things to push back on

Say so directly rather than complying if you notice any of the following. These are known failure modes for this specific project.

- A request that would train or evaluate on shuffled data.
- A request to skip the scoring validation "for now."
- A request to add a feature whose value at inference time will differ in kind from its value in training (see the route-participation problem, SPEC §10.5).
- Scope creep during Phase 0. The draft board must ship before the draft; model work does not gate it and must not delay it.
- A UI request that would display a matchup grade more prominently than usage trend (SPEC §10.4, §14.5) — the numbers do not support that hierarchy.
- Any suggestion that the model is ready to trust after a single season of validation (SPEC §12.5).

## Current phase

**Phase 2 is complete; Phase 1 has one open item.** Phase 0 (draft board) shipped long ago. Phase 1's only open item is 1.15 (conditional points model v1 — built, but doesn't beat the B2 baseline; a documented, genuine result, not a bug to silently fix — a real hyperparameter-search follow-up also confirmed the gap is architectural, not undertuned). Every Phase 2 task (2.1–2.11) is done — several were built ahead of strict order once their own prerequisites turned out not to need Sleeper/unbuilt pipelines, each confirmed with the user first, see `docs/JOURNAL.md`. Everything past this point is Phase 3, not yet started. Check `HANDOFF.md` §1 for the live, current picture; this line is a coarse pointer, not the source of truth.
