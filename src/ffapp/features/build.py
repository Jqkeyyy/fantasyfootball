"""Build-time as_of assertions (SPEC.md §10.1; task 1.5).

Assembling the actual wide `features/player_week_features.parquet` table
is task 1.9's own deliverable and lives in this module too once it exists.
This task's job is narrower: the two build-time assertions SPEC §10.1
names explicitly -- every feature in a training matrix must carry at
least one week of lag, and every feature an inference model actually uses
must be marked usable at live inference time. Both raise immediately on
the first violation rather than collecting every one -- this is a
leakage gate, not a lint report; a single mis-specified feature is enough
to invalidate a training run (CLAUDE.md rule 2).
"""

from __future__ import annotations

from collections.abc import Iterable

from ffapp.features.registry import FeatureSpec


class LeakageError(Exception):
    """A feature's `lag_weeks` or `available_at_inference` violates the as_of contract."""


def assert_training_lag(specs: Iterable[FeatureSpec]) -> None:
    """SPEC §10.1: "every feature in a training matrix has `lag_weeks >= 1`
    relative to the target week." `lag_weeks` is the feature's own
    declared minimum lag (SPEC's own field description: "1 means uses
    data through week W-1") -- `lag_weeks < 1` means the feature can see
    the target week's own data, the textbook leakage case.
    """
    for spec in specs:
        if spec.lag_weeks < 1:
            raise LeakageError(
                f"Feature {spec.name!r} has lag_weeks={spec.lag_weeks}, but every "
                "feature in a training matrix must have lag_weeks >= 1 (SPEC §10.1) "
                "-- it would see data from the target week itself."
            )


def assert_inference_availability(specs: Iterable[FeatureSpec]) -> None:
    """SPEC §10.1: "every feature used by an inference model has
    `available_at_inference=True`." A training-only feature (e.g. route
    participation, in-season -- SPEC §10.5) reaching a live inference
    model is the specific failure mode this guards against: the model
    would have learned to rely on a signal that silently isn't there when
    it's actually asked to predict.
    """
    for spec in specs:
        if not spec.available_at_inference:
            raise LeakageError(
                f"Feature {spec.name!r} has available_at_inference=False, but it "
                "is used by an inference model (SPEC §10.1) -- it may be training-only "
                "(e.g. route participation in-season, SPEC §10.5)."
            )


__all__ = ["LeakageError", "assert_inference_availability", "assert_training_lag"]
