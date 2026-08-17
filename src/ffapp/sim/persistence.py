"""Within-player week-to-week correlation (SPEC-ADDENDUM-04.md §D,
requirement 2 -- "a player's weeks are NOT independent; summing
independent weekly draws under-disperses the season total").

Additive to task 2.2's `sim.week` (cross-player correlation within a
single week) -- deliberately a new module, not a modification of an
already-shipped, already-tested one. `simulate_week_with_common_factor`
reuses `sim.week`'s own `build_correlation_matrix`/`nearest_positive_definite`
/`marginal_ppf` for the cross-player structure, and layers a single-factor
random-effects model on top for the cross-week structure:

    z[week, player] = sqrt(rho[player]) * player_factor[player]
                     + sqrt(1 - rho[player]) * idiosyncratic[week, player]

`player_factor` is drawn once per (simulation path, player) by the caller
and reused identically across every week of that path -- the standard
single-factor / random-intercept construction, which gives
Corr(z_w, z_w') = rho for any two distinct weeks of the same player by
construction, while leaving each week's own marginal exactly N(0,1)
(both `player_factor` and `idiosyncratic` are unit-variance and mutually
independent, so their weighted sum is too) -- so `marginal_ppf` still
inverts to the exact correct real per-player marginal at every week.

**Known, documented approximation:** because `idiosyncratic` is scaled
by `sqrt(1 - rho)` before the cross-player correlation matrix is applied
to it, the REALIZED cross-player correlation within a single week is
diluted by `sqrt((1-rho_i)(1-rho_j))` relative to `CorrelationSettings`'s
own configured constants (which were calibrated assuming full weight on
the correlated component). Not corrected for here -- `rho` values are
modest (see `estimate_within_player_correlation`'s own real numbers,
`docs/JOURNAL.md`), and a fully joint player x week factor model would
need a much larger joint covariance matrix for a real, unclear accuracy
gain. A documented simplification, not a silent one.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import polars as pl
from scipy.stats import norm

from ffapp.config import CorrelationSettings
from ffapp.sim.week import (
    PlayerMarginal,
    build_correlation_matrix,
    marginal_ppf,
    nearest_positive_definite,
)

_MIN_WEEKS_PER_PLAYER = 3


def estimate_within_player_correlation(player_week_features: pl.DataFrame) -> dict[str, float]:
    """Real per-position intraclass correlation (ICC) of `target` across
    a player's own real played weeks -- the unbalanced one-way random-
    effects ANOVA method-of-moments estimator (Searle/Casella/McCulloch),
    the textbook estimator for exactly this quantity: for a random-
    intercept model `Y_ij = mu + player_i + e_ij`, ICC =
    Var(player)/(Var(player) + Var(e)) equals Corr(Y_ij, Y_ij') for two
    distinct real weeks j != j' of the same player.

    Scoped to `season_type == "REG"` and `availability_flag == 1` --
    injury-driven zero weeks are a separately-modelled effect
    (`sim.injury`/`sim.season`'s own persistence, `estimate_recovery_prob`),
    not within-player PERFORMANCE correlation; mixing them in would
    conflate "this player's role is consistent when he plays" with "this
    player gets hurt a lot," two different real phenomena. `target` is
    first centered within each real (season, week, position) group (its
    own real cross-sectional mean subtracted) to remove slate-level
    effects (a real high-scoring week across the whole league) that would
    otherwise inflate between-player variance for reasons having nothing
    to do with any individual player's own persistence.

    Requires at least `_MIN_WEEKS_PER_PLAYER` real played weeks for a
    player-season to count -- too few real weeks make a single player's
    own within-player variance estimate too noisy to trust (same
    reasoning as `models.availability.DEFAULT_CALIBRATION_WEEKS`, a
    different real minimum-sample judgment call in this codebase).
    Returns `{}`  for a position with fewer than 2 real qualifying
    players (no real between-player variance to estimate at all).
    """
    scoped = player_week_features.filter(
        (pl.col("season_type") == "REG") & (pl.col("availability_flag") == 1)
    )
    centered = scoped.with_columns(
        (pl.col("target") - pl.col("target").mean().over(["season", "week", "position"])).alias(
            "_centered"
        )
    )
    with_counts = centered.with_columns(
        pl.len().over(["player_id", "season", "position"]).alias("_n_weeks")
    ).filter(pl.col("_n_weeks") >= _MIN_WEEKS_PER_PLAYER)

    result: dict[str, float] = {}
    for position in with_counts["position"].unique().to_list():
        pos_rows = with_counts.filter(pl.col("position") == position)
        groups = pos_rows.group_by(["player_id", "season"]).agg(
            pl.col("_centered").mean().alias("_group_mean"),
            pl.col("_centered").var(ddof=0).alias("_group_var"),
            pl.len().alias("_n"),
        )
        k = groups.height
        if k < 2:
            continue
        n = groups["_n"].to_numpy().astype(float)
        group_mean = groups["_group_mean"].to_numpy()
        group_var = groups["_group_var"].to_numpy()
        n_total = float(n.sum())
        grand_mean = float((group_mean * n).sum() / n_total)

        msb = float((n * (group_mean - grand_mean) ** 2).sum() / (k - 1))
        msw_num = float((n * group_var).sum())
        msw_den = n_total - k
        if msw_den <= 0:
            continue
        msw = msw_num / msw_den

        n0 = float((n_total - (n**2).sum() / n_total) / (k - 1))
        if n0 <= 0:
            continue
        between_var = max((msb - msw) / n0, 0.0)
        total_var = between_var + msw
        if total_var <= 0:
            continue
        result[position] = between_var / total_var

    return result


def season_variance_ratio(n_weeks: int, rho: float) -> float:
    """Var(sum of n equicorrelated unit-variance draws, pairwise
    correlation rho) / Var(sum of n independent unit-variance draws) =
    1 + (n-1) * rho -- the closed-form size of the under-dispersion
    requirement 2 warns about, reported directly rather than only
    inferred from a Monte Carlo run."""
    if n_weeks <= 1:
        return 1.0
    return 1.0 + (n_weeks - 1) * rho


def simulate_week_with_common_factor(
    players: Sequence[PlayerMarginal],
    correlation: CorrelationSettings,
    *,
    week_sims: int,
    player_factor: np.ndarray,
    rho: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Like `sim.week.simulate_week`, but blends each player's own
    persistent per-simulation-path factor into that week's latent normal
    before inverting through the marginal CDF (see module docstring for
    the exact construction and its one documented approximation).
    `player_factor`: `(week_sims, n_players)`, drawn once per season path
    by the caller and passed identically to every real week of that
    path. `rho`: `(n_players,)`, each player's own real
    `estimate_within_player_correlation` value for their position.
    """
    n = len(players)
    if n == 0:
        return np.empty((week_sims, 0))

    corr = nearest_positive_definite(build_correlation_matrix(players, correlation))
    idiosyncratic = rng.multivariate_normal(mean=np.zeros(n), cov=corr, size=week_sims)
    rho_arr = np.asarray(rho, dtype=float)
    z = np.sqrt(rho_arr)[None, :] * player_factor + np.sqrt(1.0 - rho_arr)[None, :] * idiosyncratic
    u = norm.cdf(z)

    scores = np.empty_like(u)
    for i, player in enumerate(players):
        scores[:, i] = marginal_ppf(u[:, i], player.alphas, player.quantile_values)
    return np.asarray(scores)


__all__ = [
    "estimate_within_player_correlation",
    "season_variance_ratio",
    "simulate_week_with_common_factor",
]
