"""
Tests for parameter and objective scaling utilities.

Tests cover:
- ParameterScaler roundtrip and edge cases
- ObjectiveNormaliser weight calculation and bias reduction
- adaptive_mcmc_proposal with/without covariance
- scaled_minimize correctness for well- and poorly-scaled problems
"""

import numpy as np
import pytest

from src.fitting.scaling import (
    ParameterScaler,
    ObjectiveNormaliser,
    adaptive_mcmc_proposal,
    scaled_minimize,
)


# ══════════════════════════════════════════════════════════════════════
#  ParameterScaler
# ══════════════════════════════════════════════════════════════════════

class TestParameterScaler:
    """Test parameter space normalisation."""

    def test_roundtrip(self):
        lower = np.array([0.01, 100, 8000, 0.1])
        upper = np.array([0.1, 500, 12000, 0.8])
        scaler = ParameterScaler(lower, upper)

        x = np.array([0.05, 300, 10000, 0.4])
        x_norm = scaler.to_normalised(x)
        x_back = scaler.to_raw(x_norm)

        np.testing.assert_allclose(x_back, x, rtol=1e-10)

    def test_bounds_map_to_zero_one(self):
        lower = np.array([1.0, 2.0])
        upper = np.array([10.0, 20.0])
        scaler = ParameterScaler(lower, upper)

        np.testing.assert_allclose(scaler.to_normalised(lower), [0.0, 0.0])
        np.testing.assert_allclose(scaler.to_normalised(upper), [1.0, 1.0])

    def test_normalised_bounds_list(self):
        scaler = ParameterScaler(np.array([0, 0, 0]), np.array([1, 1, 1]))
        nb = scaler.normalised_bounds()
        assert len(nb) == 3
        assert all(b == (0.0, 1.0) for b in nb)

    def test_zero_range_protection(self):
        """Zero-width bounds should not cause division by zero."""
        lower = np.array([5.0, 10.0])
        upper = np.array([5.0, 20.0])  # first param has zero range
        scaler = ParameterScaler(lower, upper)

        x_norm = scaler.to_normalised(np.array([5.0, 15.0]))
        assert np.isfinite(x_norm).all()

    def test_midpoint_maps_to_half(self):
        lower = np.array([0.0, 100.0])
        upper = np.array([10.0, 1000.0])
        scaler = ParameterScaler(lower, upper)

        mid = (lower + upper) / 2
        np.testing.assert_allclose(scaler.to_normalised(mid), [0.5, 0.5])


# ══════════════════════════════════════════════════════════════════════
#  ObjectiveNormaliser
# ══════════════════════════════════════════════════════════════════════

class TestObjectiveNormaliser:
    """Test objective function normalisation."""

    def test_weights_inversely_proportional_to_range(self):
        S = np.array([1000, 500, 100])     # range = 900
        X = np.array([10, 50, 100])        # range = 90

        norm = ObjectiveNormaliser(S, X)

        assert norm.S_weight < norm.X_weight  # X gets more weight per unit
        assert abs(norm.S_weight - 1 / 900) < 1e-10
        assert abs(norm.X_weight - 1 / 90) < 1e-10

    def test_equal_ranges_equal_weights(self):
        S = np.array([0, 100])
        X = np.array([0, 100])

        norm = ObjectiveNormaliser(S, X)
        assert abs(norm.S_weight - norm.X_weight) < 1e-10

    def test_normalisation_reduces_substrate_dominance(self):
        """Same absolute residual → after normalisation X contributes more."""
        S = np.array([1000, 500, 100])
        X = np.array([10, 50, 100])

        norm = ObjectiveNormaliser(S, X)
        res = np.array([10.0, 10.0, 10.0])

        norm_S = np.sum(norm.normalise_S(res) ** 2)
        norm_X = np.sum(norm.normalise_X(res) ** 2)
        # X has smaller range → bigger weight → larger normalised contribution
        assert norm_X > norm_S

    def test_zero_range_fallback(self):
        """Constant data should fall back to mean, not crash."""
        S = np.array([100, 100, 100])  # zero range
        X = np.array([50, 50, 50])

        norm = ObjectiveNormaliser(S, X)
        assert norm.S_range > 0
        assert norm.X_range > 0

    def test_info_dict_keys(self):
        norm = ObjectiveNormaliser(np.array([0, 100]), np.array([0, 50]))
        info = norm.info()
        assert set(info.keys()) == {"S_range", "X_range", "S_weight", "X_weight"}

    def test_proportional_residuals_become_balanced(self):
        """5% relative residuals on S and X should contribute equally
        after normalisation."""
        S_data = np.array([2000, 1500, 1000, 500, 100])
        X_data = np.array([10, 20, 40, 70, 90])

        S_res = 0.05 * S_data
        X_res = 0.05 * X_data

        # Without normalisation: S dominates
        raw_ratio = np.sum(S_res ** 2) / np.sum(X_res ** 2)

        # With normalisation: closer to 1
        norm = ObjectiveNormaliser(S_data, X_data)
        norm_S = np.sum(norm.normalise_S(S_res) ** 2)
        norm_X = np.sum(norm.normalise_X(X_res) ** 2)
        norm_ratio = norm_S / norm_X

        assert abs(norm_ratio - 1.0) < abs(raw_ratio - 1.0)


# ══════════════════════════════════════════════════════════════════════
#  Adaptive MCMC Proposal
# ══════════════════════════════════════════════════════════════════════

class TestAdaptiveMCMCProposal:
    """Test adaptive MCMC proposal generation."""

    def test_without_covariance_uses_range_scaling(self):
        rng = np.random.default_rng(42)
        current = np.array([1.0, 100.0])
        lower = np.array([0.0, 0.0])
        upper = np.array([10.0, 1000.0])

        proposal = adaptive_mcmc_proposal(
            current, None, lower, upper,
            fallback_scale=0.03, rng=rng,
        )
        assert proposal.shape == current.shape
        assert not np.allclose(proposal, current)

    def test_with_covariance_respects_shape(self):
        """Spread should track the covariance diagonal."""
        rng = np.random.default_rng(42)
        current = np.array([5.0, 500.0])
        lower = np.array([0.0, 0.0])
        upper = np.array([10.0, 1000.0])
        cov = np.array([[1.0, 0.0], [0.0, 100.0]])

        proposals = np.array([
            adaptive_mcmc_proposal(current, cov, lower, upper, rng=rng)
            for _ in range(200)
        ])
        std_0 = np.std(proposals[:, 0])
        std_1 = np.std(proposals[:, 1])
        assert std_1 > std_0

    def test_bad_covariance_falls_back_gracefully(self):
        rng = np.random.default_rng(42)
        current = np.array([5.0])
        bad_cov = np.array([[-1.0]])  # not PSD

        proposal = adaptive_mcmc_proposal(
            current, bad_cov,
            np.array([0.0]), np.array([10.0]),
            rng=rng,
        )
        assert np.isfinite(proposal).all()

    def test_mix_weight_zero_uses_fallback_only(self):
        rng = np.random.default_rng(42)
        current = np.array([5.0])
        cov = np.array([[1.0]])

        proposal = adaptive_mcmc_proposal(
            current, cov,
            np.array([0.0]), np.array([10.0]),
            mix_weight=0.0, rng=rng,
        )
        assert np.isfinite(proposal).all()

    def test_deterministic_with_seed(self):
        """Same seed should give same proposal."""
        args = dict(
            current=np.array([5.0, 500.0]),
            covariance=np.diag([1.0, 100.0]),
            lower=np.array([0.0, 0.0]),
            upper=np.array([10.0, 1000.0]),
            fallback_scale=0.03,
            mix_weight=0.8,
        )
        p1 = adaptive_mcmc_proposal(**args, rng=np.random.default_rng(99))
        p2 = adaptive_mcmc_proposal(**args, rng=np.random.default_rng(99))
        np.testing.assert_array_equal(p1, p2)


# ══════════════════════════════════════════════════════════════════════
#  Scaled Minimize
# ══════════════════════════════════════════════════════════════════════

class TestScaledMinimize:
    """Test scaled minimize wrapper."""

    def test_unscaled_passthrough(self):
        """Without scaling, should find standard minimum."""
        def quadratic(x):
            return (x[0] - 3.0) ** 2 + (x[1] - 7.0) ** 2

        result = scaled_minimize(
            quadratic,
            x0=np.array([0.0, 0.0]),
            bounds=[(-10, 10), (-10, 10)],
            use_scaling=False,
        )
        np.testing.assert_allclose(result.x, [3.0, 7.0], atol=1e-4)

    def test_scaled_finds_same_minimum(self):
        """With scaling enabled, should find the same result."""
        def quadratic(x):
            return (x[0] - 3.0) ** 2 + (x[1] - 7.0) ** 2

        result = scaled_minimize(
            quadratic,
            x0=np.array([0.0, 0.0]),
            bounds=[(-10, 10), (-10, 10)],
            use_scaling=True,
        )
        np.testing.assert_allclose(result.x, [3.0, 7.0], atol=1e-4)

    def test_poorly_scaled_problem(self):
        """Scaling should help when parameter scales differ hugely."""
        # Minimum at (0.001, 10000)
        def poorly_scaled(x):
            return (x[0] - 0.001) ** 2 / 1e-6 + (x[1] - 10000) ** 2 / 1e8

        result_scaled = scaled_minimize(
            poorly_scaled,
            x0=np.array([0.01, 5000]),
            bounds=[(0.0001, 0.1), (1000, 50000)],
            use_scaling=True,
        )

        result_raw = scaled_minimize(
            poorly_scaled,
            x0=np.array([0.01, 5000]),
            bounds=[(0.0001, 0.1), (1000, 50000)],
            use_scaling=False,
        )

        # Scaled should be at least as good
        assert result_scaled.fun <= result_raw.fun * 1.1 or result_scaled.fun < 1e-6

    def test_result_is_in_raw_space(self):
        """Result.x should always be in the original parameter space."""
        def obj(x):
            return (x[0] - 500) ** 2

        result = scaled_minimize(
            obj,
            x0=np.array([100.0]),
            bounds=[(0.0, 1000.0)],
            use_scaling=True,
        )
        # Should be near 500, not near 0.5
        assert result.x[0] > 100


# ══════════════════════════════════════════════════════════════════════
#  Integration
# ══════════════════════════════════════════════════════════════════════

class TestIntegration:
    """Integration tests combining multiple scaling utilities."""

    def test_scaler_with_minimize(self):
        """ParameterScaler should be usable directly in an optimization loop."""
        lower = np.array([0.0, 0.0])
        upper = np.array([10.0, 100.0])
        scaler = ParameterScaler(lower, upper)

        target = np.array([3.0, 70.0])

        def obj_raw(x):
            return (x[0] - target[0]) ** 2 + (x[1] - target[1]) ** 2

        def obj_norm(x_norm):
            return obj_raw(scaler.to_raw(x_norm))

        from scipy.optimize import minimize
        x0 = scaler.to_normalised(np.array([5.0, 50.0]))
        res = minimize(
            obj_norm, x0,
            bounds=scaler.normalised_bounds(),
            method="L-BFGS-B",
        )
        result_raw = scaler.to_raw(res.x)
        np.testing.assert_allclose(result_raw, target, rtol=1e-3)

    def test_normaliser_with_objective(self):
        """ObjectiveNormaliser should make S and X contribute similarly."""
        S_data = np.array([2000, 1500, 1000, 500, 100])
        X_data = np.array([10, 20, 40, 70, 90])

        norm = ObjectiveNormaliser(S_data, X_data)

        # A model that is 10% off on both S and X
        S_pred = S_data * 1.1
        X_pred = X_data * 1.1

        raw_sse_S = np.sum((S_data - S_pred) ** 2)
        raw_sse_X = np.sum((X_data - X_pred) ** 2)
        raw_ratio = raw_sse_S / raw_sse_X
        assert raw_ratio > 100  # S dominates in raw space

        norm_sse_S = np.sum(norm.normalise_S(S_data - S_pred) ** 2)
        norm_sse_X = np.sum(norm.normalise_X(X_data - X_pred) ** 2)
        norm_ratio = norm_sse_S / norm_sse_X
        assert norm_ratio < 10  # Much more balanced
