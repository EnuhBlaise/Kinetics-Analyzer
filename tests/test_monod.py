"""
Tests for Monod kinetic functions.
"""

import pytest
import numpy as np

from src.core.monod import (
    single_monod_term,
    dual_monod_term,
    lag_phase_factor,
    step_lag_factor
)


class TestSingleMonodTerm:
    """Tests for single_monod_term function."""

    def test_basic_monod_no_inhibition(self):
        """Test basic Monod without substrate inhibition."""
        result = single_monod_term(substrate=500, qmax=2.5, Ks=400, Ki=None)
        expected = 2.5 * 500 / (400 + 500)
        assert abs(result - expected) < 1e-10

    def test_monod_with_inhibition(self):
        """Test Monod with substrate inhibition (Haldane model)."""
        result = single_monod_term(substrate=500, qmax=2.5, Ks=400, Ki=25000)
        # Haldane: q = qmax * S / (Ks + S + S^2/Ki)
        expected = 2.5 * 500 / (400 + 500 + 500**2 / 25000)
        assert abs(result - expected) < 1e-10

    def test_zero_substrate(self):
        """Test with zero substrate concentration."""
        result = single_monod_term(substrate=0, qmax=2.5, Ks=400, Ki=25000)
        assert result == 0.0

    def test_negative_substrate_clipped(self):
        """Test that negative substrate is clipped to zero."""
        result = single_monod_term(substrate=-10, qmax=2.5, Ks=400, Ki=25000)
        assert result == 0.0

    def test_high_inhibition(self):
        """Test high substrate causing strong inhibition."""
        result = single_monod_term(substrate=20000, qmax=2.5, Ks=400, Ki=25000)
        # Inhibition factor should be 1 - 20000/25000 = 0.2
        assert result > 0
        assert result < 2.5  # Less than qmax due to inhibition

    def test_array_input(self):
        """Test with numpy array input."""
        substrates = np.array([100, 500, 1000])
        result = single_monod_term(substrates, qmax=2.5, Ks=400, Ki=25000)
        assert len(result) == 3
        assert all(result >= 0)


class TestDualMonodTerm:
    """Tests for dual_monod_term function."""

    def test_dual_monod_basic(self):
        """Test dual Monod with substrate and oxygen."""
        result = dual_monod_term(
            substrate=500, oxygen=6.0,
            qmax=2.5, Ks=400, Ki=25000, K_o2=0.15
        )
        assert result > 0
        assert result < 2.5

    def test_zero_oxygen(self):
        """Test with zero oxygen - should be zero."""
        result = dual_monod_term(
            substrate=500, oxygen=0,
            qmax=2.5, Ks=400, Ki=25000, K_o2=0.15
        )
        assert result == 0.0

    def test_high_oxygen(self):
        """Test with saturating oxygen."""
        result_high = dual_monod_term(
            substrate=500, oxygen=100,
            qmax=2.5, Ks=400, Ki=25000, K_o2=0.15
        )
        result_low = dual_monod_term(
            substrate=500, oxygen=1.0,
            qmax=2.5, Ks=400, Ki=25000, K_o2=0.15
        )
        assert result_high > result_low  # Higher O2 = higher rate


class TestLagPhaseFactor:
    """Tests for lag_phase_factor function."""

    def test_before_lag(self):
        """Test factor is low before lag time."""
        factor = lag_phase_factor(time=0.5, lag_time=3.0)
        assert factor < 0.1

    def test_at_lag_time(self):
        """Test factor is approximately 1 at lag time."""
        factor = lag_phase_factor(time=3.0, lag_time=3.0)
        assert factor > 0.95

    def test_after_lag_time(self):
        """Test factor is close to 1 well after lag time."""
        factor = lag_phase_factor(time=5.0, lag_time=3.0)
        assert factor > 0.99

    def test_zero_time(self):
        """Test factor is close to 0 at time zero."""
        factor = lag_phase_factor(time=0, lag_time=3.0)
        assert factor < 0.01

    def test_array_input(self):
        """Test with array of times."""
        times = np.array([0, 1.5, 3.0, 5.0])
        factors = lag_phase_factor(times, lag_time=3.0)
        assert len(factors) == 4
        assert factors[0] < 0.01
        assert factors[-1] > 0.99

    def test_monotonic_increase(self):
        """Test that factor increases monotonically."""
        times = np.linspace(0.1, 5, 50)
        factors = lag_phase_factor(times, lag_time=3.0)
        for i in range(1, len(factors)):
            assert factors[i] >= factors[i-1]


class TestStepLagFactor:
    """Tests for step_lag_factor function."""

    def test_before_lag(self):
        """Test step function before lag."""
        factor = step_lag_factor(time=2.0, lag_time=3.0)
        assert factor == 0.0

    def test_at_lag(self):
        """Test step function at lag time."""
        factor = step_lag_factor(time=3.0, lag_time=3.0)
        assert factor == 1.0

    def test_after_lag(self):
        """Test step function after lag."""
        factor = step_lag_factor(time=4.0, lag_time=3.0)
        assert factor == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
