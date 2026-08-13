"""Task 2.2's own literal acceptance bar (SPEC §13.2): "simulated
team-total variance is materially lower than the independent-sampling
equivalent and the correlation matrix is positive definite after
correction." All fixtures here are small and hand-verifiable -- no live
data needed (SPEC §13.2 itself says empirical correlation estimation is
Phase 3 work; Phase 2 uses the configured constants as-is).
"""

from __future__ import annotations

import numpy as np
import pytest

from ffapp.config import CorrelationSettings
from ffapp.sim.week import (
    PlayerMarginal,
    build_correlation_matrix,
    marginal_ppf,
    nearest_positive_definite,
    simulate_week,
)

_CORRELATION = CorrelationSettings(
    qb_pass_catcher=0.35, same_team_rb_rb=-0.25, player_vs_opposing_dst=-0.30
)

_ALPHAS = (0.10, 0.25, 0.50, 0.75, 0.90)


def _player(
    player_id: str,
    position: str,
    team: str,
    *,
    opponent_team: str | None = None,
    values: tuple[float, ...] = (5.0, 8.0, 10.0, 12.0, 15.0),
) -> PlayerMarginal:
    return PlayerMarginal(
        player_id=player_id,
        position=position,
        team=team,
        opponent_team=opponent_team,
        alphas=_ALPHAS,
        quantile_values=values,
    )


# --- marginal_ppf ---------------------------------------------------------------------


def test_marginal_ppf_returns_zero_at_u_equals_zero() -> None:
    result = marginal_ppf(np.array([0.0]), _ALPHAS, (5.0, 8.0, 10.0, 12.0, 15.0))

    assert result[0] == pytest.approx(0.0)


def test_marginal_ppf_returns_the_exact_grid_value_at_a_fitted_alpha() -> None:
    result = marginal_ppf(np.array([0.5]), _ALPHAS, (5.0, 8.0, 10.0, 12.0, 15.0))

    assert result[0] == pytest.approx(10.0)


def test_marginal_ppf_interpolates_linearly_between_grid_points() -> None:
    # Halfway between alpha=0.10 (value 5.0) and alpha=0.25 (value 8.0).
    result = marginal_ppf(np.array([0.175]), _ALPHAS, (5.0, 8.0, 10.0, 12.0, 15.0))

    assert result[0] == pytest.approx(6.5)


def test_marginal_ppf_tail_beyond_q90_is_smooth_and_strictly_increasing() -> None:
    u = np.array([0.90, 0.95, 0.99, 0.999])

    result = marginal_ppf(u, _ALPHAS, (5.0, 8.0, 10.0, 12.0, 15.0))

    assert result[0] == pytest.approx(15.0)  # continuous at the q90 anchor
    assert np.all(np.diff(result) > 0)  # strictly increasing into the tail
    assert result[-1] > result[0] + 5  # a real tail, not a flat clamp at q90


def test_marginal_ppf_is_nondecreasing_across_the_full_unit_interval() -> None:
    u = np.linspace(0.0, 0.999, 200)

    result = marginal_ppf(u, _ALPHAS, (5.0, 8.0, 10.0, 12.0, 15.0))

    assert np.all(np.diff(result) >= -1e-9)


def test_marginal_ppf_reflects_a_mass_point_at_zero_from_a_zeroed_low_quantile() -> None:
    """`models.quantiles.mixture_with_p_active` already bakes a low-`p_active`
    player's mass-at-zero into its own low `unconditional_q_<tau>` columns
    (any tau below the mass point comes back exactly 0.0) -- `marginal_ppf`
    just needs to interpolate through that grid faithfully, not re-derive
    `p_active` itself."""
    # A player whose alpha=0.10 and 0.25 grid points are already 0.0 (baked
    # in by the mixture upstream, e.g. p_active around 0.4).
    values = (0.0, 0.0, 3.0, 9.0, 14.0)

    result = marginal_ppf(np.array([0.05, 0.20, 0.50]), _ALPHAS, values)

    assert result[0] == pytest.approx(0.0)
    assert result[1] == pytest.approx(0.0)
    assert result[2] == pytest.approx(3.0)


# --- build_correlation_matrix ----------------------------------------------------------


def test_build_correlation_matrix_applies_qb_pass_catcher_rule() -> None:
    players = [
        _player("qb1", "QB", "KC"),
        _player("wr1", "WR", "KC"),
    ]

    corr = build_correlation_matrix(players, _CORRELATION)

    assert corr[0, 1] == pytest.approx(0.35)
    assert corr[1, 0] == pytest.approx(0.35)
    assert corr[0, 0] == pytest.approx(1.0)


def test_build_correlation_matrix_applies_same_team_rb_rb_rule() -> None:
    players = [
        _player("rb1", "RB", "SF"),
        _player("rb2", "RB", "SF"),
    ]

    corr = build_correlation_matrix(players, _CORRELATION)

    assert corr[0, 1] == pytest.approx(-0.25)


def test_build_correlation_matrix_applies_player_vs_opposing_dst_rule() -> None:
    players = [
        _player("wr1", "WR", "KC", opponent_team="BUF"),
        _player("BUF", "DST", "BUF"),
    ]

    corr = build_correlation_matrix(players, _CORRELATION)

    assert corr[0, 1] == pytest.approx(-0.30)


def test_build_correlation_matrix_leaves_unrelated_pairs_uncorrelated() -> None:
    players = [
        _player("wr1", "WR", "KC"),
        _player("rb1", "RB", "SF"),
    ]

    corr = build_correlation_matrix(players, _CORRELATION)

    assert corr[0, 1] == pytest.approx(0.0)


def test_build_correlation_matrix_does_not_apply_rb_rb_rule_across_different_teams() -> None:
    players = [
        _player("rb1", "RB", "SF"),
        _player("rb2", "RB", "KC"),
    ]

    corr = build_correlation_matrix(players, _CORRELATION)

    assert corr[0, 1] == pytest.approx(0.0)


# --- nearest_positive_definite ---------------------------------------------------------


def test_nearest_positive_definite_is_a_noop_on_an_already_pd_matrix() -> None:
    corr = np.array([[1.0, 0.3], [0.3, 1.0]])

    result = nearest_positive_definite(corr)

    assert result == pytest.approx(corr, abs=1e-6)


def test_nearest_positive_definite_fixes_an_inconsistent_matrix() -> None:
    # A classic non-PD pairwise-correlation set: high positive AB and BC,
    # high negative AC -- mutually inconsistent for 3 real random variables.
    corr = np.array(
        [
            [1.0, 0.9, -0.9],
            [0.9, 1.0, 0.9],
            [-0.9, 0.9, 1.0],
        ]
    )
    eigvals_before = np.linalg.eigvalsh(corr)
    assert eigvals_before.min() < 0  # confirms the fixture really is non-PD

    result = nearest_positive_definite(corr)

    eigvals_after = np.linalg.eigvalsh(result)
    assert eigvals_after.min() > 0
    assert np.allclose(np.diag(result), 1.0)
    assert np.allclose(result, result.T)  # still symmetric


# --- simulate_week: acceptance bar -----------------------------------------------------


def test_simulate_week_output_shape_is_week_sims_by_n_players() -> None:
    players = [_player("rb1", "RB", "SF"), _player("wr1", "WR", "KC")]
    rng = np.random.default_rng(0)

    result = simulate_week(players, _CORRELATION, week_sims=500, rng=rng)

    assert result.shape == (500, 2)


def test_simulate_week_team_total_variance_is_materially_lower_than_independent_sampling() -> None:
    """SPEC §13.2's own literal acceptance bar. Two same-team RBs
    (configured `same_team_rb_rb = -0.25`) -- a Gaussian copula with
    negative correlation preserves the sign of dependence through any
    monotone marginal transform, so the correlated roster total should
    come out with materially lower variance than summing the same two
    marginals sampled independently (rho=0)."""
    players = [_player("rb1", "RB", "SF"), _player("rb2", "RB", "SF")]
    independent = CorrelationSettings(
        qb_pass_catcher=0.0, same_team_rb_rb=0.0, player_vs_opposing_dst=0.0
    )

    correlated_totals = simulate_week(
        players, _CORRELATION, week_sims=20000, rng=np.random.default_rng(1)
    ).sum(axis=1)
    independent_totals = simulate_week(
        players, independent, week_sims=20000, rng=np.random.default_rng(2)
    ).sum(axis=1)

    correlated_var = correlated_totals.var()
    independent_var = independent_totals.var()
    assert correlated_var < independent_var * 0.85  # materially, not marginally, lower


def test_simulate_week_handles_zero_players() -> None:
    result = simulate_week([], _CORRELATION, week_sims=10, rng=np.random.default_rng(0))

    assert result.shape == (10, 0)
