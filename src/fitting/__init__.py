"""
Fitting module for parameter optimization and statistical analysis.

Provides:
- Basic optimization: ObjectiveFunction, ParameterOptimizer
- Robust fitting: RobustFitter (combines weighting, two-stage, bootstrap)
- Weighting strategies: Handle heteroscedastic data
- Two-stage estimation: Intelligent initialization
- Bootstrap: Uncertainty quantification
"""

from .objective import ObjectiveFunction, GlobalObjectiveFunction
from .optimizer import ParameterOptimizer, OptimizationResult
from .statistics import calculate_r_squared, calculate_rmse, calculate_aic, calculate_bic
from .weighting import (
    WeightingStrategy, UniformWeighting, MaxValueWeighting,
    VarianceWeighting, RangeWeighting, get_weighting_strategy
)
from .two_stage import TwoStageEstimator, TwoStageResult, estimate_initial_parameters
from .bootstrap import BootstrapEngine, BootstrapResult, run_bootstrap
from .robust_fitter import RobustFitter, RobustFitResult
from .diagnostics import OptimizerDiagnostics, DiagnosticsReport
from .scaling import (
    ParameterScaler, ObjectiveNormaliser,
    adaptive_mcmc_proposal, scaled_minimize,
)

__all__ = [
    # Basic fitting
    "ObjectiveFunction",
    "GlobalObjectiveFunction",
    "ParameterOptimizer",
    "OptimizationResult",
    # Statistics
    "calculate_r_squared",
    "calculate_rmse",
    "calculate_aic",
    "calculate_bic",
    # Weighting strategies
    "WeightingStrategy",
    "UniformWeighting",
    "MaxValueWeighting",
    "VarianceWeighting",
    "RangeWeighting",
    "get_weighting_strategy",
    # Two-stage estimation
    "TwoStageEstimator",
    "TwoStageResult",
    "estimate_initial_parameters",
    # Bootstrap
    "BootstrapEngine",
    "BootstrapResult",
    "run_bootstrap",
    # High-level API
    "RobustFitter",
    "RobustFitResult",
    # Diagnostics
    "OptimizerDiagnostics",
    "DiagnosticsReport",
    # Scaling utilities
    "ParameterScaler",
    "ObjectiveNormaliser",
    "adaptive_mcmc_proposal",
    "scaled_minimize",
]
