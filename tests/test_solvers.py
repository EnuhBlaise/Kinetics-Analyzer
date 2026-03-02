"""
Tests for ODE solvers and simulation utilities.
"""

import pytest
import numpy as np

from src.core.ode_systems import SingleMonodODE, DualMonodODE, DualMonodLagODE
from src.core.solvers import solve_ode, create_time_grid, run_simulation_batch
from src.core.oxygen import OxygenModel


class TestSingleMonodODE:
    """Tests for SingleMonodODE class."""

    @pytest.fixture
    def ode_system(self):
        """Create a standard single Monod ODE system."""
        return SingleMonodODE(
            qmax=2.5,
            Ks=400,
            Ki=25000,
            Y=0.35,
            b_decay=0.01
        )

    def test_n_states(self, ode_system):
        """Test correct number of states."""
        assert ode_system.n_states == 2

    def test_state_names(self, ode_system):
        """Test state variable names."""
        assert ode_system.state_names == ["Substrate", "Biomass"]

    def test_parameter_names(self, ode_system):
        """Test parameter names."""
        assert "qmax" in ode_system.parameter_names
        assert "Ks" in ode_system.parameter_names
        assert "Y" in ode_system.parameter_names

    def test_get_parameters(self, ode_system):
        """Test parameter retrieval."""
        params = ode_system.get_parameters()
        assert params["qmax"] == 2.5
        assert params["Ks"] == 400
        assert params["Y"] == 0.35

    def test_derivatives(self, ode_system):
        """Test derivative calculation."""
        y = np.array([500.0, 10.0])  # [S, X]
        dydt = ode_system.derivatives(t=0, y=y)
        assert len(dydt) == 2
        assert dydt[0] < 0  # Substrate decreasing
        assert dydt[1] > 0  # Biomass increasing (growth > decay)


class TestDualMonodODE:
    """Tests for DualMonodODE class."""

    @pytest.fixture
    def ode_system(self):
        """Create a dual Monod ODE system."""
        return DualMonodODE(
            qmax=2.5,
            Ks=400,
            Ki=25000,
            Y=0.35,
            b_decay=0.01,
            K_o2=0.15,
            Y_o2=0.8,
            oxygen_model=OxygenModel()
        )

    def test_n_states(self, ode_system):
        """Test correct number of states."""
        assert ode_system.n_states == 3

    def test_state_names(self, ode_system):
        """Test state variable names."""
        assert "Oxygen" in ode_system.state_names

    def test_derivatives_with_oxygen(self, ode_system):
        """Test derivative calculation with oxygen."""
        y = np.array([500.0, 10.0, 6.0])  # [S, X, O2]
        dydt = ode_system.derivatives(t=0, y=y)
        assert len(dydt) == 3


class TestDualMonodLagODE:
    """Tests for DualMonodLagODE class."""

    @pytest.fixture
    def ode_system(self):
        """Create a dual Monod with lag ODE system."""
        return DualMonodLagODE(
            qmax=2.5,
            Ks=400,
            Ki=25000,
            Y=0.35,
            b_decay=0.01,
            K_o2=0.15,
            Y_o2=0.8,
            lag_time=3.0,
            oxygen_model=OxygenModel()
        )

    def test_lag_time_parameter(self, ode_system):
        """Test lag time is stored correctly."""
        params = ode_system.get_parameters()
        assert params["lag_time"] == 3.0

    def test_derivatives_during_lag(self, ode_system):
        """Test derivatives are reduced during lag phase."""
        y = np.array([500.0, 10.0, 6.0])

        # During lag (t=1)
        dydt_lag = ode_system.derivatives(t=1.0, y=y)

        # After lag (t=5)
        dydt_active = ode_system.derivatives(t=5.0, y=y)

        # Growth should be higher after lag
        assert abs(dydt_active[0]) > abs(dydt_lag[0])


class TestSolveODE:
    """Tests for solve_ode function."""

    def test_solve_single_monod(self):
        """Test solving single Monod ODE."""
        ode = SingleMonodODE(qmax=2.5, Ks=400, Ki=25000, Y=0.35, b_decay=0.01)
        result = solve_ode(
            ode_system=ode,
            initial_conditions=[500.0, 1.0],
            t_span=(0, 5),
            t_eval=np.linspace(0, 5, 100)
        )

        assert result.success
        assert len(result.time) == 100
        assert "Substrate" in result.states
        assert "Biomass" in result.states

    def test_substrate_decreases(self):
        """Test that substrate decreases over time."""
        ode = SingleMonodODE(qmax=2.5, Ks=400, Ki=25000, Y=0.35, b_decay=0.01)
        result = solve_ode(
            ode_system=ode,
            initial_conditions=[500.0, 1.0],
            t_span=(0, 5),
            t_eval=np.linspace(0, 5, 100)
        )

        substrate = result.get_state("Substrate")
        assert substrate[0] > substrate[-1]

    def test_biomass_increases(self):
        """Test that biomass increases over time."""
        ode = SingleMonodODE(qmax=2.5, Ks=400, Ki=25000, Y=0.35, b_decay=0.01)
        result = solve_ode(
            ode_system=ode,
            initial_conditions=[500.0, 1.0],
            t_span=(0, 5),
            t_eval=np.linspace(0, 5, 100)
        )

        biomass = result.get_state("Biomass")
        assert biomass[-1] > biomass[0]

    def test_to_dataframe(self):
        """Test conversion to DataFrame."""
        ode = SingleMonodODE(qmax=2.5, Ks=400, Ki=25000, Y=0.35, b_decay=0.01)
        result = solve_ode(
            ode_system=ode,
            initial_conditions=[500.0, 1.0],
            t_span=(0, 5),
            t_eval=np.linspace(0, 5, 100)
        )

        df = result.to_dataframe()
        assert "Time" in df.columns
        assert "Substrate" in df.columns
        assert "Biomass" in df.columns
        assert len(df) == 100


class TestCreateTimeGrid:
    """Tests for create_time_grid function."""

    def test_linear_spacing(self):
        """Test linear time grid."""
        grid = create_time_grid(0, 5, num_points=100, log_spacing=False)
        assert len(grid) == 100
        assert grid[0] == 0
        assert grid[-1] == 5

    def test_log_spacing(self):
        """Test logarithmic time grid."""
        grid = create_time_grid(0.01, 5, num_points=100, log_spacing=True)
        assert len(grid) == 100
        # Log spacing should have denser points at start
        early_spacing = grid[1] - grid[0]
        late_spacing = grid[-1] - grid[-2]
        assert early_spacing < late_spacing


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
