"""
Tests for optimizer diagnostics module.
"""

import pytest
import numpy as np
import json
import os
from pathlib import Path

from src.fitting.diagnostics import (
    OptimizerDiagnostics,
    DiagnosticsReport,
    MultiStartResult,
    ProfileResult,
    ContourResult,
    HessianDiagnosticResult,
    ConvergenceTrace,
)


# ---------------------------------------------------------------------------
# Helpers: simple quadratic objective for fast, predictable tests
# ---------------------------------------------------------------------------

def make_quadratic_objective(true_params, scales=None):
    """
    Create a simple quadratic objective:
        f(x) = sum( ((x_i - true_i) / scale_i)^2 )
    This has a unique minimum at true_params with a well-conditioned Hessian.
    """
    true = np.array(true_params)
    if scales is None:
        scales = np.ones_like(true)
    else:
        scales = np.array(scales)

    def objective(x):
        return float(np.sum(((np.asarray(x) - true) / scales) ** 2))

    return objective


def make_ridge_objective(true_params):
    """
    Create an objective with a ridge/correlation between param 0 and param 1:
        f(x) = (x0 + x1 - (t0+t1))^2 + 0.01*(x0 - t0)^2 + sum(rest)
    The combination x0+x1 is well-determined, but individual values are sloppy.
    """
    true = np.array(true_params)

    def objective(x):
        x = np.asarray(x)
        # Strong constraint on x0 + x1
        err = (x[0] + x[1] - true[0] - true[1]) ** 2
        # Weak constraint on x0 individually
        err += 0.01 * (x[0] - true[0]) ** 2
        # Normal constraints on other params
        for i in range(2, len(x)):
            err += (x[i] - true[i]) ** 2
        return float(err)

    return objective


PARAM_NAMES = ["qmax", "Ks", "Ki"]
BOUNDS = {"qmax": (0.5, 10.0), "Ks": (10.0, 500.0), "Ki": (100.0, 50000.0)}
TRUE = {"qmax": 2.5, "Ks": 100.0, "Ki": 25000.0}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestMultiStart:
    """Tests for multi-start optimization."""

    def test_single_minimum_quadratic(self):
        """Quadratic has one minimum — all starts should converge there."""
        obj = make_quadratic_objective([2.5, 100.0, 25000.0], [1.0, 100.0, 10000.0])
        diag = OptimizerDiagnostics(obj, PARAM_NAMES, BOUNDS, verbose=False)

        result = diag.multi_start(TRUE, n_starts=10, seed=42)

        assert isinstance(result, MultiStartResult)
        assert result.n_unique_minima == 1
        assert result.best_objective < 1e-6
        assert len(result.objectives) == 10
        assert sum(result.success_flags) >= 8  # Most should converge

    def test_best_params_close_to_true(self):
        """Best multi-start solution should match the true optimum."""
        obj = make_quadratic_objective([2.5, 100.0, 25000.0], [1.0, 100.0, 10000.0])
        diag = OptimizerDiagnostics(obj, PARAM_NAMES, BOUNDS, verbose=False)

        result = diag.multi_start(TRUE, n_starts=20, seed=42)

        for p in PARAM_NAMES:
            assert abs(result.best_params[p] - TRUE[p]) < 1e-2 * abs(TRUE[p])

    def test_to_dict(self):
        """MultiStartResult.to_dict should produce valid JSON."""
        obj = make_quadratic_objective([2.5, 100.0, 25000.0], [1.0, 100.0, 10000.0])
        diag = OptimizerDiagnostics(obj, PARAM_NAMES, BOUNDS, verbose=False)
        result = diag.multi_start(TRUE, n_starts=5, seed=42)

        d = result.to_dict()
        assert "best_objective" in d
        assert "n_unique_minima" in d
        # Should be JSON-serializable
        json.dumps(d)


class TestParameterProfiles:
    """Tests for 1-D parameter profiles."""

    def test_profiles_identifiable(self):
        """Well-scaled quadratic should yield identifiable profiles."""
        obj = make_quadratic_objective([2.5, 100.0, 25000.0], [1.0, 100.0, 10000.0])
        diag = OptimizerDiagnostics(obj, PARAM_NAMES, BOUNDS, verbose=False)

        profiles = diag.parameter_profiles(TRUE, n_points=15)

        assert len(profiles) == 3
        for pname in PARAM_NAMES:
            assert pname in profiles
            p = profiles[pname]
            assert isinstance(p, ProfileResult)
            assert p.is_identifiable
            assert len(p.fixed_values) == 15
            assert len(p.profile_objectives) == 15

    def test_sloppy_parameter_detected(self):
        """Ridge objective should show qmax has weaker curvature than Ki."""
        obj = make_ridge_objective([2.5, 100.0, 25000.0])
        diag = OptimizerDiagnostics(obj, PARAM_NAMES, BOUNDS, verbose=False)

        profiles = diag.parameter_profiles(TRUE, n_points=15)

        # qmax profile curvature should be much smaller than Ki
        # because qmax is involved in the ridge correlation
        assert profiles["qmax"].curvature < profiles["Ki"].curvature

    def test_profile_subset(self):
        """Should profile only requested parameters."""
        obj = make_quadratic_objective([2.5, 100.0, 25000.0], [1.0, 100.0, 10000.0])
        diag = OptimizerDiagnostics(obj, PARAM_NAMES, BOUNDS, verbose=False)

        profiles = diag.parameter_profiles(TRUE, n_points=10, parameters=["qmax"])

        assert len(profiles) == 1
        assert "qmax" in profiles


class TestContourAnalysis:
    """Tests for 2-D contour analysis."""

    def test_contour_shape(self):
        """Grid should have correct dimensions."""
        obj = make_quadratic_objective([2.5, 100.0, 25000.0], [1.0, 100.0, 10000.0])
        diag = OptimizerDiagnostics(obj, PARAM_NAMES, BOUNDS, verbose=False)

        contours = diag.contour_analysis(
            TRUE,
            param_pairs=[("qmax", "Ks")],
            n_grid=10
        )

        assert len(contours) == 1
        key = "qmax__Ks"
        assert key in contours
        cr = contours[key]
        assert isinstance(cr, ContourResult)
        assert cr.objective_grid.shape == (10, 10)

    def test_uncorrelated_shows_none(self):
        """Quadratic with independent params should show no correlation."""
        obj = make_quadratic_objective([2.5, 100.0, 25000.0], [1.0, 100.0, 10000.0])
        diag = OptimizerDiagnostics(obj, PARAM_NAMES, BOUNDS, verbose=False)

        contours = diag.contour_analysis(
            TRUE,
            param_pairs=[("qmax", "Ki")],
            n_grid=15
        )

        cr = contours["qmax__Ki"]
        assert cr.correlation_direction == "none"

    def test_correlated_pair_detected(self):
        """Ridge objective should show correlation via Hessian.
        
        Note: contour_analysis holds non-grid params fixed, so the 
        ridge doesn't tilt the contours. The Hessian correlation matrix
        is the proper tool for detecting this.
        """
        obj = make_ridge_objective([2.5, 100.0, 25000.0])
        diag = OptimizerDiagnostics(obj, PARAM_NAMES, BOUNDS, verbose=False)

        result = diag.hessian_analysis(TRUE)

        # qmax-Ks correlation should be strong (negative because of ridge shape)
        rho = abs(result.correlation_matrix[0, 1])  # qmax vs Ks
        assert rho > 0.5, f"Expected strong correlation, got rho={rho:.3f}"


class TestHessianAnalysis:
    """Tests for Hessian eigenvalue analysis."""

    def test_well_conditioned_quadratic(self):
        """Quadratic should have a finite, computable condition number."""
        # Use equal scales to get a truly well-conditioned Hessian
        obj = make_quadratic_objective([2.5, 100.0, 25000.0], [1.0, 1.0, 1.0])
        diag = OptimizerDiagnostics(obj, PARAM_NAMES, BOUNDS, verbose=False)

        result = diag.hessian_analysis(TRUE)

        assert isinstance(result, HessianDiagnosticResult)
        assert len(result.eigenvalues) == 3
        assert all(e > 0 for e in result.eigenvalues)
        # Equal scales: condition should be exactly 1
        assert result.condition_number < 10
        assert len(result.sloppy_directions) == 0

    def test_correlation_matrix_diagonal(self):
        """Independent quadratic should have near-identity correlation."""
        obj = make_quadratic_objective([2.5, 100.0, 25000.0], [1.0, 100.0, 10000.0])
        diag = OptimizerDiagnostics(obj, PARAM_NAMES, BOUNDS, verbose=False)

        result = diag.hessian_analysis(TRUE)

        # Diagonal should be 1.0
        for i in range(3):
            assert abs(result.correlation_matrix[i, i] - 1.0) < 0.01

        # Off-diagonal should be near zero
        for i in range(3):
            for j in range(3):
                if i != j:
                    assert abs(result.correlation_matrix[i, j]) < 0.1

    def test_sloppy_direction_detected(self):
        """Ridge objective should produce a sloppy eigenvalue."""
        obj = make_ridge_objective([2.5, 100.0, 25000.0])
        diag = OptimizerDiagnostics(obj, PARAM_NAMES, BOUNDS, verbose=False)

        result = diag.hessian_analysis(TRUE)

        # Should have at least one sloppy direction
        assert len(result.sloppy_directions) >= 1

    def test_standard_errors_finite(self):
        """Standard errors should be finite for well-posed problems."""
        obj = make_quadratic_objective([2.5, 100.0, 25000.0], [1.0, 100.0, 10000.0])
        diag = OptimizerDiagnostics(obj, PARAM_NAMES, BOUNDS, verbose=False)

        result = diag.hessian_analysis(TRUE)

        assert all(np.isfinite(result.standard_errors))
        assert all(se > 0 for se in result.standard_errors)

    def test_to_dict(self):
        """HessianDiagnosticResult.to_dict should be JSON-serializable."""
        obj = make_quadratic_objective([2.5, 100.0, 25000.0], [1.0, 100.0, 10000.0])
        diag = OptimizerDiagnostics(obj, PARAM_NAMES, BOUNDS, verbose=False)
        result = diag.hessian_analysis(TRUE)

        d = result.to_dict()
        json.dumps(d)


class TestConvergenceTrace:
    """Tests for convergence tracing."""

    def test_trace_records_iterations(self):
        """Trace should capture the full optimization trajectory."""
        obj = make_quadratic_objective([2.5, 100.0, 25000.0], [1.0, 100.0, 10000.0])
        diag = OptimizerDiagnostics(obj, PARAM_NAMES, BOUNDS, verbose=False)

        # Start away from optimum
        start = {"qmax": 5.0, "Ks": 200.0, "Ki": 10000.0}
        trace = diag.trace_convergence(start)

        assert isinstance(trace, ConvergenceTrace)
        assert len(trace.iterations) >= 2  # At least start + finish
        assert len(trace.objective_history) == len(trace.iterations)
        for p in PARAM_NAMES:
            assert p in trace.parameter_history
            assert len(trace.parameter_history[p]) == len(trace.iterations)

    def test_objective_decreases(self):
        """Objective should decrease (or stay flat) over iterations."""
        obj = make_quadratic_objective([2.5, 100.0, 25000.0], [1.0, 100.0, 10000.0])
        diag = OptimizerDiagnostics(obj, PARAM_NAMES, BOUNDS, verbose=False)

        start = {"qmax": 8.0, "Ks": 400.0, "Ki": 5000.0}
        trace = diag.trace_convergence(start)

        # Final should be <= initial
        assert trace.objective_history[-1] <= trace.objective_history[0] + 1e-10


class TestDiagnosticsReport:
    """Tests for the report container and serialization."""

    def test_save_creates_json(self, tmp_path):
        """Report.save should create diagnostics_summary.json."""
        report = DiagnosticsReport()
        saved = report.save(str(tmp_path / "output"))

        json_path = tmp_path / "output" / "diagnostics_summary.json"
        assert json_path.exists()
        with open(json_path) as f:
            data = json.load(f)
        assert isinstance(data, dict)

    def test_full_run_all(self):
        """run_all should produce a complete report without errors."""
        obj = make_quadratic_objective([2.5, 100.0, 25000.0], [1.0, 100.0, 10000.0])
        diag = OptimizerDiagnostics(obj, PARAM_NAMES, BOUNDS, verbose=False)

        report = diag.run_all(
            TRUE,
            n_starts=5,
            n_profile_points=8,
            n_contour_grid=8,
        )

        assert report.multi_start is not None
        assert report.profiles is not None
        assert report.contours is not None
        assert report.hessian is not None
        assert report.convergence is not None


class TestPlotting:
    """Tests that plot methods run without errors (visual correctness not tested)."""

    @pytest.fixture
    def diag_and_report(self):
        obj = make_quadratic_objective([2.5, 100.0, 25000.0], [1.0, 100.0, 10000.0])
        diag = OptimizerDiagnostics(obj, PARAM_NAMES, BOUNDS, verbose=False)
        report = diag.run_all(TRUE, n_starts=5, n_profile_points=8, n_contour_grid=8)
        return diag, report

    def test_plot_multi_start(self, diag_and_report, tmp_path):
        diag, report = diag_and_report
        fig = diag.plot_multi_start(report.multi_start, str(tmp_path / "ms.png"))
        assert (tmp_path / "ms.png").exists()
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_profiles(self, diag_and_report, tmp_path):
        diag, report = diag_and_report
        fig = diag.plot_profiles(report.profiles, TRUE, str(tmp_path / "prof.png"))
        assert (tmp_path / "prof.png").exists()
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_contours(self, diag_and_report, tmp_path):
        diag, report = diag_and_report
        fig = diag.plot_contours(report.contours, TRUE, str(tmp_path / "cont.png"))
        assert (tmp_path / "cont.png").exists()
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_hessian(self, diag_and_report, tmp_path):
        diag, report = diag_and_report
        fig = diag.plot_hessian(report.hessian, str(tmp_path / "hess.png"))
        assert (tmp_path / "hess.png").exists()
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_convergence(self, diag_and_report, tmp_path):
        diag, report = diag_and_report
        fig = diag.plot_convergence(report.convergence, str(tmp_path / "conv.png"))
        assert (tmp_path / "conv.png").exists()
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_save_all_plots(self, diag_and_report, tmp_path):
        diag, report = diag_and_report
        saved = diag.save_all_plots(report, TRUE, str(tmp_path / "all"))
        assert len(saved) == 5
        for p in saved:
            assert Path(p).exists()
