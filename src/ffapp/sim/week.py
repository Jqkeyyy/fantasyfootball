"""Weekly simulation with correlation (SPEC.md §13.2; task 2.2).

Gaussian copula, SPEC's own four steps:

1. Marginal CDF per player (`marginal_ppf`) -- built directly from
   `models.quantiles.mixture_with_p_active`'s own output shape (a grid of
   `unconditional_q_<tau>` values at the configured `model.quantiles`
   alphas). That mixture step already bakes the "mass point at 0 of size
   1 - p_active" into the grid itself -- any alpha at or below the mass
   point comes back exactly `0.0` -- so `marginal_ppf` only needs to
   interpolate through the grid faithfully, not re-derive `p_active`.
   Below the lowest fitted alpha it interpolates down to the anchor
   `(0.0, 0.0)`; beyond the highest fitted alpha ("a fitted tail beyond
   q90") it switches to a smooth, strictly-increasing exponential tail
   matched to the grid's own local slope at that point, rather than
   flat-lining a real, unbounded upper tail at the last fitted quantile.
2. Correlation matrix from `settings.simulation.correlation`'s pairwise
   rules (`build_correlation_matrix`) -- the configured constants, not
   empirically estimated (SPEC: "estimate them empirically from
   historical data in Phase 3" -- out of scope for this task).
3. Sample multivariate normal, transform through the normal CDF to
   uniforms, then through each player's own inverse marginal CDF
   (`simulate_week`).
4. Nearest-positive-definite correction (`nearest_positive_definite`) on
   the correlation matrix before sampling -- eigenvalue clipping plus a
   rescale back to a unit-diagonal correlation matrix, applied
   unconditionally (a no-op on an already-PD matrix, a real correction
   on an inconsistent one).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from ffapp.config import CorrelationSettings

_MIN_TAIL_SLOPE = 1e-6


@dataclass(frozen=True)
class PlayerMarginal:
    """One player's own marginal distribution for a single week, plus
    the metadata `build_correlation_matrix`'s pairwise rules need.
    `alphas`/`quantile_values` are `models.quantiles.mixture_with_p_active`'s
    own `unconditional_q_<tau>` grid (ascending, same length)."""

    player_id: str
    position: str
    team: str
    opponent_team: str | None
    alphas: Sequence[float]
    quantile_values: Sequence[float]


def marginal_ppf(
    u: np.ndarray, alphas: Sequence[float], quantile_values: Sequence[float]
) -> np.ndarray:
    """Inverts one player's marginal CDF at each uniform level in `u`
    (SPEC §13.2 step 1). Piecewise-linear through the fitted grid
    (extended down to the `(0.0, 0.0)` anchor), then a smooth tail beyond
    the highest fitted alpha."""
    alpha_arr = np.asarray(alphas, dtype=float)
    value_arr = np.asarray(quantile_values, dtype=float)
    alpha_max = alpha_arr[-1]
    value_max = value_arr[-1]

    # np.interp clamps outside its own x-range, which is correct below
    # alpha_arr[0] once the (0.0, 0.0) anchor is prepended, but wrong
    # above alpha_max (it would flatten a real, unbounded tail) -- the
    # tail branch below overwrites that region explicitly.
    extended_alphas = np.concatenate(([0.0], alpha_arr))
    extended_values = np.concatenate(([0.0], value_arr))
    body = np.interp(u, extended_alphas, extended_values)

    if len(alpha_arr) >= 2:
        slope = (value_arr[-1] - value_arr[-2]) / (alpha_arr[-1] - alpha_arr[-2])
    else:
        slope = 0.0
    slope = max(slope, _MIN_TAIL_SLOPE)
    rate = 1.0 / (slope * max(1.0 - alpha_max, _MIN_TAIL_SLOPE))

    tail_mask = u > alpha_max
    if np.any(tail_mask):
        t = (u[tail_mask] - alpha_max) / (1.0 - alpha_max)
        t = np.clip(t, 0.0, 1.0 - 1e-12)  # keep log finite as u -> 1
        body[tail_mask] = value_max - np.log(1.0 - t) / rate

    return np.asarray(body)


def _pairwise_correlation(
    a: PlayerMarginal, b: PlayerMarginal, correlation: CorrelationSettings
) -> float:
    if a.team == b.team:
        positions = {a.position, b.position}
        if positions == {"QB", "WR"} or positions == {"QB", "TE"}:
            return correlation.qb_pass_catcher
        if a.position == "RB" and b.position == "RB":
            return correlation.same_team_rb_rb
    if a.position == "DST" and b.opponent_team == a.team:
        return correlation.player_vs_opposing_dst
    if b.position == "DST" and a.opponent_team == b.team:
        return correlation.player_vs_opposing_dst
    return 0.0


def build_correlation_matrix(
    players: Sequence[PlayerMarginal], correlation: CorrelationSettings
) -> np.ndarray:
    """SPEC §13.2 step 2: identity diagonal, each off-diagonal pair set
    by whichever pairwise rule (if any) matches, `0.0` otherwise."""
    n = len(players)
    corr = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            rho = _pairwise_correlation(players[i], players[j], correlation)
            corr[i, j] = rho
            corr[j, i] = rho
    return corr


def nearest_positive_definite(corr: np.ndarray, *, epsilon: float = 1e-8) -> np.ndarray:
    """SPEC §13.2 step 4: eigenvalue clipping (any eigenvalue below
    `epsilon` is raised to it), then rescaled back to a unit-diagonal
    correlation matrix. A no-op (up to floating-point noise) on a matrix
    that's already positive definite."""
    if corr.shape[0] == 0:
        return corr
    eigvals, eigvecs = np.linalg.eigh(corr)
    clipped = np.clip(eigvals, epsilon, None)
    reconstructed = eigvecs @ np.diag(clipped) @ eigvecs.T
    d = np.sqrt(np.diag(reconstructed))
    rescaled = reconstructed / np.outer(d, d)
    np.fill_diagonal(rescaled, 1.0)
    return np.asarray((rescaled + rescaled.T) / 2)  # symmetrize away floating-point drift


def simulate_week(
    players: Sequence[PlayerMarginal],
    correlation: CorrelationSettings,
    *,
    week_sims: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """SPEC §13.2 steps 2-4, applied: `week_sims x n_players` matrix of
    sampled scores. Everything else (win probability, start/sit, DST
    value) reads from this."""
    n = len(players)
    if n == 0:
        return np.empty((week_sims, 0))

    corr = nearest_positive_definite(build_correlation_matrix(players, correlation))
    z = rng.multivariate_normal(mean=np.zeros(n), cov=corr, size=week_sims)
    u = norm.cdf(z)

    scores = np.empty_like(u)
    for i, player in enumerate(players):
        scores[:, i] = marginal_ppf(u[:, i], player.alphas, player.quantile_values)
    return np.asarray(scores)


__all__ = [
    "PlayerMarginal",
    "build_correlation_matrix",
    "marginal_ppf",
    "nearest_positive_definite",
    "simulate_week",
]
