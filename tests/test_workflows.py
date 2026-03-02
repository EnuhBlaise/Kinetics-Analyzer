"""
Tests for workflow classes.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile

from src.io.data_loader import ExperimentalData
from src.io.config_loader import SubstrateConfig
from workflows.single_monod import SingleMonodWorkflow
from workflows.dual_monod import DualMonodWorkflow
from workflows.dual_monod_lag import DualMonodLagWorkflow


@pytest.fixture
def sample_config():
    """Create a sample substrate configuration."""
    return SubstrateConfig(
        name="TestSubstrate",
        molecular_weight=180.0,
        initial_guesses={
            "qmax": 2.5,
            "Ks": 400,
            "Ki": 25000,
            "Y": 0.35,
            "b_decay": 0.01,
            "K_o2": 0.15,
            "Y_o2": 0.8,
            "lag_time": 3.0
        },
        bounds={
            "qmax": (0.1, 10),
            "Ks": (10, 2000),
            "Ki": (50, 50000),
            "Y": (0.1, 1.0),
            "b_decay": (0.001, 0.2),
            "K_o2": (0.05, 1.0),
            "Y_o2": (0.1, 2.0),
            "lag_time": (0, 10)
        },
        oxygen={"o2_max": 8.0, "o2_min": 0.01, "reaeration_rate": 5.0},
        simulation={"t_final": 5.0, "num_points": 1000}
    )


@pytest.fixture
def sample_data():
    """Create sample experimental data."""
    time = np.array([0, 1, 2, 3, 4, 5])
    substrate_5mM = np.array([900, 800, 500, 200, 50, 10])
    biomass_5mM = np.array([1, 5, 20, 50, 60, 62])

    df = pd.DataFrame({
        "Time (days)": time,
        "5mM_TestSubstrate (mg/L)": substrate_5mM,
        "5mM_Biomass (mgCells/L)": biomass_5mM
    })

    return ExperimentalData(
        data=df,
        time_column="Time (days)",
        substrate_columns=["5mM_TestSubstrate (mg/L)"],
        biomass_columns=["5mM_Biomass (mgCells/L)"],
        conditions=["5mM"],
        substrate_name="TestSubstrate"
    )


class TestSingleMonodWorkflow:
    """Tests for SingleMonodWorkflow."""

    def test_model_type(self, sample_config, sample_data):
        """Test model type string."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workflow = SingleMonodWorkflow(sample_config, sample_data, tmpdir)
            assert workflow.model_type == "single_monod"

    def test_parameter_names(self, sample_config, sample_data):
        """Test parameter names for single Monod."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workflow = SingleMonodWorkflow(sample_config, sample_data, tmpdir)
            assert "qmax" in workflow.parameter_names
            assert "Ks" in workflow.parameter_names
            assert "lag_time" not in workflow.parameter_names
            assert len(workflow.parameter_names) == 5

    def test_create_ode_system(self, sample_config, sample_data):
        """Test ODE system creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workflow = SingleMonodWorkflow(sample_config, sample_data, tmpdir)
            params = {"qmax": 2.5, "Ks": 400, "Ki": 25000, "Y": 0.35, "b_decay": 0.01}
            ode = workflow.create_ode_system(params)
            assert ode.n_states == 2


class TestDualMonodWorkflow:
    """Tests for DualMonodWorkflow."""

    def test_model_type(self, sample_config, sample_data):
        """Test model type string."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workflow = DualMonodWorkflow(sample_config, sample_data, tmpdir)
            assert workflow.model_type == "dual_monod"

    def test_parameter_names(self, sample_config, sample_data):
        """Test parameter names for dual Monod."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workflow = DualMonodWorkflow(sample_config, sample_data, tmpdir)
            assert "K_o2" in workflow.parameter_names
            assert "Y_o2" in workflow.parameter_names
            assert "lag_time" not in workflow.parameter_names
            assert len(workflow.parameter_names) == 7

    def test_initial_conditions(self, sample_config, sample_data):
        """Test initial conditions include oxygen."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workflow = DualMonodWorkflow(sample_config, sample_data, tmpdir)
            ic = workflow._get_initial_conditions(S0=900, X0=1)
            assert len(ic) == 3  # S, X, O2
            assert ic[2] == 8.0  # O2 max


class TestDualMonodLagWorkflow:
    """Tests for DualMonodLagWorkflow."""

    def test_model_type(self, sample_config, sample_data):
        """Test model type string."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workflow = DualMonodLagWorkflow(sample_config, sample_data, tmpdir)
            assert workflow.model_type == "dual_monod_lag"

    def test_parameter_names(self, sample_config, sample_data):
        """Test parameter names include lag_time."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workflow = DualMonodLagWorkflow(sample_config, sample_data, tmpdir)
            assert "lag_time" in workflow.parameter_names
            assert len(workflow.parameter_names) == 8


class TestWorkflowResult:
    """Tests for WorkflowResult class."""

    def test_summary(self, sample_config, sample_data):
        """Test result summary generation."""
        from workflows.base_workflow import WorkflowResult
        from src.fitting.optimizer import OptimizationResult

        opt_result = OptimizationResult(
            parameters={"qmax": 2.5, "Ks": 400},
            statistics={"R_squared": 0.95, "RMSE": 10.5},
            success=True,
            message="Success",
            n_iterations=100,
            n_function_evals=500,
            initial_guess={"qmax": 2.0, "Ks": 300},
            bounds={"qmax": (0.1, 10), "Ks": (10, 2000)},
            method="L-BFGS-B"
        )

        result = WorkflowResult(
            model_type="single_monod",
            optimization_result=opt_result,
            predictions=pd.DataFrame(),
            experimental_data=sample_data,
            statistics={"R_squared": 0.95, "RMSE": 10.5},
            conditions=["5mM"]
        )

        summary = result.summary()
        assert "single_monod" in summary
        assert "0.95" in summary  # R² value


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
