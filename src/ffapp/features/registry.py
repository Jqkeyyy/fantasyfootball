"""The feature registry (SPEC.md §10.1; task 1.5).

`FeatureSpec` is the as_of contract for one feature, stated as data rather
than left implicit in the code that computes it: which positions it applies
to, what window it's aggregated over, which interim table it reads from,
and -- the two properties `features/build.py`'s assertions actually
enforce -- whether it's usable at live inference time and how many weeks
of lag it carries relative to its target week.

This module is deliberately infrastructure only: the `FeatureSpec`
dataclass, a shared `FEATURE_REGISTRY`, and `register()` to add to it.
Confirmed as this task's scope (not pre-declaring SPEC §10.2's full
catalogue): most of those features have no computed values yet -- the
windowed usage/team-context/opponent features are explicitly tasks
1.6-1.8's own deliverables, several building on interim tables that are
themselves still partly null (`team_week_context.proe`, `.neutral_pace_sec`;
`defense_position_allowed.adj_*`). Each of those tasks registers its own
features here as it builds the real computation logic, rather than this
task guessing at a `lag_weeks`/`window` for a feature nobody has computed
yet.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureSpec:
    """SPEC §10.1's own record, verbatim field-for-field."""

    name: str
    description: str
    positions: list[str]
    window: str | None
    source_table: str
    available_at_inference: bool
    lag_weeks: int


class DuplicateFeatureError(Exception):
    """A feature name was registered twice with `register()`."""


FEATURE_REGISTRY: dict[str, FeatureSpec] = {}


def register(spec: FeatureSpec, *, registry: dict[str, FeatureSpec] | None = None) -> FeatureSpec:
    """Add `spec` to `registry` (the shared `FEATURE_REGISTRY` by default).

    Raises `DuplicateFeatureError` on a name collision rather than silently
    overwriting -- a feature getting silently redefined partway through a
    build is exactly the kind of quiet correctness bug CLAUDE.md rule 4
    warns about for joins, and the same principle applies here: two
    different `FeatureSpec`s claiming the same name is a real conflict,
    not a no-op.
    """
    target = FEATURE_REGISTRY if registry is None else registry
    if spec.name in target:
        raise DuplicateFeatureError(
            f"Feature {spec.name!r} is already registered "
            f"(existing source_table={target[spec.name].source_table!r})."
        )
    target[spec.name] = spec
    return spec


__all__ = [
    "FEATURE_REGISTRY",
    "DuplicateFeatureError",
    "FeatureSpec",
    "register",
]
