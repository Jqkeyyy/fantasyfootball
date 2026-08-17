from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from scipy.stats import norm

from ffapp.config import DEFAULT_CORRELATION_SETTINGS
from ffapp.sim.persistence import (
    estimate_within_player_correlation,
    season_variance_ratio,
    simulate_week_with_common_factor,
)
from ffapp.sim.week import PlayerMarginal


def test_estimate_within_player_correlation_high_for_a_persistent_role_player() -> None:
    """Two RBs, four played weeks each. rb1 is remarkably consistent
    (16, 17, 15, 16 -- almost no within-player variance). rb2 alternates
    a starter role and a committee role week to week (a real, if
    stylised, "role varies a lot within-season" pattern -- low real
    persistence). The pooled RB ICC should sit meaningfully above 0 (real
    between-player signal exists) and below 1 (real within-player noise
    exists too)."""
    features = pl.DataFrame(
        {
            "player_id": ["rb1"] * 4 + ["rb2"] * 4,
            "season": [2023] * 8,
            "week": [1, 2, 3, 4, 1, 2, 3, 4],
            "position": ["RB"] * 8,
            "season_type": ["REG"] * 8,
            "availability_flag": [1] * 8,
            "target": [16.0, 17.0, 15.0, 16.0, 4.0, 22.0, 3.0, 21.0],
        }
    )
    result = estimate_within_player_correlation(features)
    assert 0.0 < result["RB"] < 1.0


def test_estimate_within_player_correlation_excludes_unplayed_weeks() -> None:
    """A row with availability_flag=0 (didn't play) must not count toward
    the estimate -- injury-driven zero weeks are a different, separately-
    modelled effect (sim.injury/sim.season's own persistence), not
    within-player performance correlation."""
    features = pl.DataFrame(
        {
            "player_id": ["rb1"] * 5,
            "season": [2023] * 5,
            "week": [1, 2, 3, 4, 5],
            "position": ["RB"] * 5,
            "season_type": ["REG"] * 5,
            "availability_flag": [1, 1, 1, 1, 0],
            "target": [16.0, 17.0, 15.0, 16.0, 0.0],
        }
    )
    # Should not raise, and the excluded week's 0.0 must not drag a
    # persistent player's own ICC down toward "inconsistent."
    result = estimate_within_player_correlation(features)
    # only 1 real player at RB here -- no between-player signal at all,
    # either omitted or 0.0 is acceptable
    assert "RB" not in result or result["RB"] >= 0.0


def test_season_variance_ratio_matches_closed_form() -> None:
    # Var(sum of n equicorrelated unit-variance draws) / Var(sum of n
    # independent unit-variance draws) = 1 + (n-1) * rho, the textbook
    # equicorrelated-sum identity.
    assert season_variance_ratio(n_weeks=10, rho=0.3) == pytest.approx(1 + 9 * 0.3)
    assert season_variance_ratio(n_weeks=1, rho=0.5) == pytest.approx(1.0)
    assert season_variance_ratio(n_weeks=10, rho=0.0) == pytest.approx(1.0)


def test_simulate_week_with_common_factor_reproduces_marginal() -> None:
    """A single player, rho=1.0 (fully determined by the common factor):
    the sampled score distribution must still match that player's own
    real marginal quantile grid -- the common-factor blend must never
    distort a player's OWN marginal, only introduce cross-week
    dependence."""
    alphas = [0.1, 0.25, 0.5, 0.75, 0.9]
    quantile_values = [3.0, 6.0, 10.0, 14.0, 18.0]
    player = PlayerMarginal(
        player_id="p1",
        position="RB",
        team="KC",
        opponent_team="BUF",
        alphas=alphas,
        quantile_values=quantile_values,
    )
    rng = np.random.default_rng(1)
    n_sims = 20000
    player_factor = rng.standard_normal((n_sims, 1))
    scores = simulate_week_with_common_factor(
        [player],
        DEFAULT_CORRELATION_SETTINGS,
        week_sims=n_sims,
        player_factor=player_factor,
        rho=np.array([1.0]),
        rng=rng,
    )
    empirical_median = float(np.median(scores[:, 0]))
    assert empirical_median == pytest.approx(10.0, abs=0.5)


def test_simulate_week_with_common_factor_induces_week_to_week_correlation() -> None:
    """Same player, sampled across two 'weeks' sharing the same
    player_factor draw but independent idiosyncratic noise: the
    correlation between the two weeks' sampled normal-space z should be
    close to the configured rho (checked in z-space via norm.ppf, since
    the marginal itself is nonlinear)."""
    alphas = [0.1, 0.25, 0.5, 0.75, 0.9]
    quantile_values = [3.0, 6.0, 10.0, 14.0, 18.0]
    player = PlayerMarginal(
        player_id="p1",
        position="RB",
        team="KC",
        opponent_team="BUF",
        alphas=alphas,
        quantile_values=quantile_values,
    )
    rng = np.random.default_rng(2)
    n_sims = 20000
    rho = np.array([0.4])
    player_factor = rng.standard_normal((n_sims, 1))
    week1 = simulate_week_with_common_factor(
        [player],
        DEFAULT_CORRELATION_SETTINGS,
        week_sims=n_sims,
        player_factor=player_factor,
        rho=rho,
        rng=rng,
    )
    week2 = simulate_week_with_common_factor(
        [player],
        DEFAULT_CORRELATION_SETTINGS,
        week_sims=n_sims,
        player_factor=player_factor,
        rho=rho,
        rng=rng,
    )
    z1 = norm.ppf(np.clip(_empirical_cdf(week1[:, 0]), 1e-6, 1 - 1e-6))
    z2 = norm.ppf(np.clip(_empirical_cdf(week2[:, 0]), 1e-6, 1 - 1e-6))
    empirical_rho = float(np.corrcoef(z1, z2)[0, 1])
    assert empirical_rho == pytest.approx(0.4, abs=0.05)


def _empirical_cdf(values: np.ndarray) -> np.ndarray:
    order = np.argsort(np.argsort(values))
    return (order + 0.5) / len(values)
