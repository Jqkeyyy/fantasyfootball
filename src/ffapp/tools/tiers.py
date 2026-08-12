"""Draft tiers (SPEC.md §9.5; task 0.10).

Tiers matter more than ranks at the draft -- the decision that costs value
is reaching past a tier break, not choosing #14 over #15. Three interchangeable
methods (`gap`, `kmeans`, `gmm`, selected by `config.DraftSettings.tier_method`)
each produce a raw, possibly-noncompliant tier-label sequence for one
position's players sorted by VOR descending; a single shared normalization
step then enforces SPEC's two hard constraints -- no tier smaller than
`min_tier_size`, no more than `max_tiers` -- uniformly across all three, so
that constraint isn't duplicated per method.

`gap` follows SPEC's own pseudocode literally: cut wherever a gap between
consecutive players' VOR exceeds `1.4 x` the rolling median of gaps in a
window of 9. "Rolling median" doesn't specify a windowing convention
(centered vs. trailing) -- implemented here as a centered window (shrinking
at the sequence's edges), since a local-density threshold for detecting an
anomalous gap should look both ways, not be biased by an already-passed
trend (e.g. gaps widening steadily further down a position). This is a
judgment call, not specified by SPEC; it doesn't change the task's
acceptance bar (tiers assigned per position, none smaller than 2, capped at
12), which any reasonable windowing convention satisfies.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from typing import Protocol

import numpy as np
import polars as pl
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

DEFAULT_MAX_TIERS = 12
DEFAULT_MIN_TIER_SIZE = 2
_GAP_WINDOW = 9
_GAP_MULTIPLIER = 1.4


class _RawTierMethod(Protocol):
    def __call__(self, values: Sequence[float], max_tiers: int) -> list[int]: ...


def _rolling_median_centered(values: Sequence[float], window: int) -> list[float]:
    """Centered rolling median, shrinking the window at the sequence's edges
    (pandas' `rolling(window, center=True, min_periods=1)` semantics)."""
    n = len(values)
    half = window // 2
    return [statistics.median(values[max(0, i - half) : min(n, i + half + 1)]) for i in range(n)]


def _gap_raw_tiers(values: Sequence[float], max_tiers: int) -> list[int]:
    """SPEC §9.5's `gap` method: cut wherever a VOR gap exceeds 1.4x the
    local rolling median of gaps. Raw output may violate min-size/max-count
    -- `_normalize_tiers` enforces both afterward."""
    n = len(values)
    if n <= 1:
        return [0] * n

    gaps = [values[i] - values[i + 1] for i in range(n - 1)]
    thresholds = _rolling_median_centered(gaps, _GAP_WINDOW)

    labels = [0]
    for gap, threshold in zip(gaps, thresholds, strict=True):
        labels.append(labels[-1] + 1 if gap > _GAP_MULTIPLIER * threshold else labels[-1])
    return labels


def _kmeans_raw_tiers(values: Sequence[float], max_tiers: int) -> list[int]:
    return _clustered_raw_tiers(values, max_tiers, _fit_kmeans)


def _gmm_raw_tiers(values: Sequence[float], max_tiers: int) -> list[int]:
    return _clustered_raw_tiers(values, max_tiers, _fit_gmm)


def _fit_kmeans(arr: np.ndarray, k: int) -> np.ndarray:
    model = KMeans(n_clusters=k, n_init=10, random_state=0)
    labels: np.ndarray = model.fit_predict(arr)
    return labels


def _fit_gmm(arr: np.ndarray, k: int) -> np.ndarray:
    model = GaussianMixture(n_components=k, random_state=0)
    labels: np.ndarray = model.fit(arr).predict(arr)
    return labels


def _clustered_raw_tiers(values: Sequence[float], max_tiers: int, fit) -> list[int]:  # type: ignore[no-untyped-def]
    """Shared k-means/GMM scaffolding: cluster VOR as a 1D scalar (k =
    min(max_tiers, n_players)), then remap sklearn's arbitrary cluster ids
    to tier ids ordered by descending cluster mean -- so tier 0 is always
    the best cluster, matching `_gap_raw_tiers`' convention, and the
    resulting per-player label sequence is contiguous when read in VOR-
    descending order (the well-known optimal-1D-clustering property).
    """
    n = len(values)
    if n <= 1:
        return [0] * n

    k = min(max_tiers, n)
    arr = np.array(values, dtype=float).reshape(-1, 1)
    raw_labels = fit(arr, k)

    order = sorted(set(raw_labels.tolist()), key=lambda c: -arr[raw_labels == c].mean())
    remap = {cluster: rank for rank, cluster in enumerate(order)}
    return [remap[label] for label in raw_labels.tolist()]


_RAW_METHODS: dict[str, _RawTierMethod] = {
    "gap": _gap_raw_tiers,
    "kmeans": _kmeans_raw_tiers,
    "gmm": _gmm_raw_tiers,
}


def _segments_from_labels(labels: Sequence[int]) -> list[tuple[int, int]]:
    """Contiguous (start, end_exclusive) index ranges wherever `labels`
    changes value -- works regardless of whether the raw labels themselves
    are contiguous integers, since only "did the label change" matters."""
    if not labels:
        return []
    segments = []
    start = 0
    for i in range(1, len(labels)):
        if labels[i] != labels[i - 1]:
            segments.append((start, i))
            start = i
    segments.append((start, len(labels)))
    return segments


def _merge_small_segments(
    segments: list[tuple[int, int]], min_tier_size: int
) -> list[tuple[int, int]]:
    """SPEC §9.5: merge any tier smaller than `min_tier_size` into its
    neighbour -- the following tier, or the preceding one if it's last."""
    segments = list(segments)
    while True:
        undersized = next((i for i, (s, e) in enumerate(segments) if e - s < min_tier_size), None)
        if undersized is None or len(segments) == 1:
            return segments
        if undersized < len(segments) - 1:
            (s, _e), (_s2, e2) = segments[undersized], segments[undersized + 1]
            segments[undersized : undersized + 2] = [(s, e2)]
        else:
            (s1, _e1), (_s, e) = segments[undersized - 1], segments[undersized]
            segments[undersized - 1 : undersized + 1] = [(s1, e)]


def _cap_segment_count(
    segments: list[tuple[int, int]], values: Sequence[float], max_tiers: int
) -> list[tuple[int, int]]:
    """SPEC §9.5: cap at `max_tiers` tiers, collapsing the least-significant
    boundary (the adjacent pair separated by the smallest VOR gap) first."""
    segments = list(segments)
    while len(segments) > max_tiers:
        gap_after = [
            values[segments[i][1] - 1] - values[segments[i + 1][0]]
            for i in range(len(segments) - 1)
        ]
        smallest = min(range(len(gap_after)), key=lambda i: gap_after[i])
        (s, _e), (_s2, e2) = segments[smallest], segments[smallest + 1]
        segments[smallest : smallest + 2] = [(s, e2)]
    return segments


def _normalize_tiers(
    values: Sequence[float], raw_labels: Sequence[int], *, max_tiers: int, min_tier_size: int
) -> list[int]:
    segments = _segments_from_labels(raw_labels)
    segments = _merge_small_segments(segments, min_tier_size)
    segments = _cap_segment_count(segments, values, max_tiers)

    tier_ids = [0] * len(values)
    for tier_number, (start, end) in enumerate(segments, start=1):
        for i in range(start, end):
            tier_ids[i] = tier_number
    return tier_ids


def _tiers_for_position(
    values: Sequence[float], *, method: str, max_tiers: int, min_tier_size: int
) -> list[int]:
    if not values:
        return []
    if len(values) == 1:
        return [1]
    raw = _RAW_METHODS[method](values, max_tiers)
    return _normalize_tiers(values, raw, max_tiers=max_tiers, min_tier_size=min_tier_size)


def assign_tiers(
    projections: pl.DataFrame,
    *,
    method: str = "gap",
    vor_column: str = "vor",
    position_column: str = "position",
    max_tiers: int = DEFAULT_MAX_TIERS,
    min_tier_size: int = DEFAULT_MIN_TIER_SIZE,
) -> pl.DataFrame:
    """Adds a `tier` column (1 = best), computed independently per position,
    sorted by `vor_column` descending within each position. Output row order
    is grouped by position then VOR descending, not the input's own order --
    the natural order for a tiered draft board.

    Raises ValueError for an unrecognised `method`, or for a null value in
    `vor_column` (a real gap upstream -- `tools/vor.compute_vor` should
    already guarantee every row has one; a null here means that contract was
    violated, not a case to silently skip).
    """
    if method not in _RAW_METHODS:
        raise ValueError(f"unknown tier method {method!r} (expected one of {sorted(_RAW_METHODS)})")

    tiered_positions = []
    for position in projections[position_column].unique(maintain_order=True).to_list():
        pos_df = projections.filter(pl.col(position_column) == position).sort(
            vor_column, descending=True
        )
        if pos_df[vor_column].null_count() > 0:
            raise ValueError(f"{vor_column} has null value(s) for position {position!r}")

        values = pos_df[vor_column].to_list()
        tier_ids = _tiers_for_position(
            values, method=method, max_tiers=max_tiers, min_tier_size=min_tier_size
        )
        tiered_positions.append(pos_df.with_columns(pl.Series("tier", tier_ids)))

    return pl.concat(tiered_positions, how="vertical")


__all__ = [
    "DEFAULT_MAX_TIERS",
    "DEFAULT_MIN_TIER_SIZE",
    "assign_tiers",
]
