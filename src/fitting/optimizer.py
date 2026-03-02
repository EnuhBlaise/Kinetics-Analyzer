"""
Parameter optimization module.

This module provides optimization routines for fitting kinetic
parameters to experimental data, with support for parallel processing.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional, Callable
from scipy.optimize import minimize, differential_evolution
import multiprocessing as mp
from functools import partial

from .objective import ObjectiveFunction, GlobalObjectiveFunction
from .statistics import calculate_r_squared, calculate_rmse, calculate_aic, calculate_bic


@dataclass
class OptimizationResult:
    """
    Container for optimization results.

    Attributes:
        parameters: Dictionary of fitted parameter values
        statistics: Dictionary of fit statistics
        success: Whether optimization converged
        message: Optimizer status message
        n_iterations: Number of optimizer iterations
        n_function_evals: Number of objective function evaluations
        initial_guess: Initial parameter values
        bounds: Parameter bounds used
        method: Optimization method used
    """
    parameters: Dict[str, float]
    statistics: Dict[str, float]
    success: bool
    message: str
    n_iterations: int
    n_function_evals: int
    initial_guess: Dict[str, float]
    bounds: Dict[str, Tuple[float, float]]
    method: str
    raw_result: Any = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excluding raw_result)."""
        return {
            "parameters": self.parameters,
            "statistics": self.statistics,
            "success": self.success,
            "message": self.message,
            "n_iterations": self.n_iterations,
            "n_function_evals": self.n_function_evals,
            "initial_guess": self.initial_guess,
            "bounds": self.bounds,
            "method": self.method
        }


class ParameterOptimizer:
    """
    Main optimizer class for kinetic parameter estimation.

    Supports both local (L-BFGS-B) and global (differential evolution)
    optimization methods, with optional parallel processing.
    """

    def __init__(
        self,
        parameter_names: List[str],
        bounds: Dict[str, Tuple[float, float]],
        initial_guess: Dict[str, float],
        method: str = "L-BFGS-B",
        max_iterations: int = 10000,
        tolerance: float = 1e-8,
        n_workers: int = 1,
        verbose: bool = False
    ):
        """
        Initialize the optimizer.

        Args:
            parameter_names: Names of parameters to optimize
            bounds: Dictionary of parameter bounds
            initial_guess: Dictionary of initial parameter values
            method: Optimization method ("L-BFGS-B" or "differential_evolution")
            max_iterations: Maximum optimizer iterations
            tolerance: Convergence tolerance
            n_workers: Number of parallel workers (for differential_evolution)
            verbose: Whether to print progress
        """
        self.parameter_names = parameter_names
        self.bounds = bounds
        self.initial_guess = initial_guess
        self.method = method
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.n_workers = n_workers
        self.verbose = verbose

        # Prepare bounds as list for scipy
        self.bounds_list = [bounds[p] for p in parameter_names]
        self.x0 = np.array([initial_guess[p] for p in parameter_names])

    def optimize(
        self,
        objective_function: Callable,
        callback: Optional[Callable] = None
    ) -> OptimizationResult:
        """
        Run optimization to find best-fit parameters.

        Args:
            objective_function: Callable that takes parameter array and returns error
            callback: Optional callback function called after each iteration

        Returns:
            OptimizationResult with fitted parameters and statistics
        """
        if self.verbose:
            print(f"Starting optimization with {self.method}...")
            print(f"Parameters: {self.parameter_names}")
            print(f"Initial guess: {self.initial_guess}")

        if self.method.lower() == "l-bfgs-b":
            result = self._optimize_lbfgsb(objective_function, callback)
        elif self.method.lower() == "differential_evolution":
            result = self._optimize_de(objective_function, callback)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        if self.verbose:
            print(f"Optimization {'succeeded' if result.success else 'failed'}")
            print(f"Final parameters: {result.parameters}")

        return result

    def _optimize_lbfgsb(
        self,
        objective: Callable,
        callback: Optional[Callable]
    ) -> OptimizationResult:
        """Run L-BFGS-B optimization."""
        result = minimize(
            fun=objective,
            x0=self.x0,
            method='L-BFGS-B',
            bounds=self.bounds_list,
            options={
                'maxiter': self.max_iterations,
                'ftol': self.tolerance,
                'disp': self.verbose
            },
            callback=callback
        )

        parameters = dict(zip(self.parameter_names, result.x))

        return OptimizationResult(
            parameters=parameters,
            statistics={},  # To be filled later
            success=result.success,
            message=result.message,
            n_iterations=result.nit,
            n_function_evals=result.nfev,
            initial_guess=self.initial_guess,
            bounds=self.bounds,
            method="L-BFGS-B",
            raw_result=result
        )

    def _optimize_de(
        self,
        objective: Callable,
        callback: Optional[Callable]
    ) -> OptimizationResult:
        """Run differential evolution optimization."""
        workers = self.n_workers if self.n_workers > 1 else 1

        result = differential_evolution(
            func=objective,
            bounds=self.bounds_list,
            maxiter=self.max_iterations,
            tol=self.tolerance,
            workers=workers,
            seed=42,  # For reproducibility
            disp=self.verbose,
            callback=callback,
            init='latinhypercube'
        )

        parameters = dict(zip(self.parameter_names, result.x))

        return OptimizationResult(
            parameters=parameters,
            statistics={},
            success=result.success,
            message=result.message,
            n_iterations=result.nit,
            n_function_evals=result.nfev,
            initial_guess=self.initial_guess,
            bounds=self.bounds,
            method="differential_evolution",
            raw_result=result
        )


def fit_single_condition(
    objective_function: ObjectiveFunction,
    config,
    parameter_names: List[str],
    method: str = "L-BFGS-B"
) -> OptimizationResult:
    """
    Fit parameters to a single experimental condition.

    Args:
        objective_function: Configured ObjectiveFunction
        config: SubstrateConfig with initial guesses and bounds
        parameter_names: Parameters to optimize
        method: Optimization method

    Returns:
        OptimizationResult
    """
    optimizer = ParameterOptimizer(
        parameter_names=parameter_names,
        bounds={p: config.bounds[p] for p in parameter_names},
        initial_guess={p: config.initial_guesses[p] for p in parameter_names},
        method=method
    )

    return optimizer.optimize(objective_function)


def fit_global(
    global_objective: GlobalObjectiveFunction,
    config,
    parameter_names: List[str],
    method: str = "L-BFGS-B",
    n_workers: int = 1
) -> OptimizationResult:
    """
    Fit parameters globally across multiple conditions.

    Args:
        global_objective: GlobalObjectiveFunction for all conditions
        config: SubstrateConfig with initial guesses and bounds
        parameter_names: Parameters to optimize
        method: Optimization method
        n_workers: Number of parallel workers

    Returns:
        OptimizationResult with globally fitted parameters
    """
    optimizer = ParameterOptimizer(
        parameter_names=parameter_names,
        bounds={p: config.bounds[p] for p in parameter_names},
        initial_guess={p: config.initial_guesses[p] for p in parameter_names},
        method=method,
        n_workers=n_workers
    )

    result = optimizer.optimize(global_objective)

    # Calculate statistics
    result.statistics["n_conditions"] = len(global_objective.conditions)
    result.statistics["total_error"] = global_objective.best_error

    return result


def fit_parallel_conditions(
    conditions: List[Dict[str, Any]],
    model_type: str,
    config,
    parameter_names: List[str],
    n_workers: int = None # Use all CPUs by default to maximize speed. 
) -> Dict[str, OptimizationResult]:
    """
    Fit parameters independently for multiple conditions in parallel.

    Args:
        conditions: List of condition dictionaries
        model_type: Type of model
        config: SubstrateConfig
        parameter_names: Parameters to optimize
        n_workers: Number of parallel workers (None = all CPUs)

    Returns:
        Dictionary mapping condition labels to OptimizationResults
    """
    # Maximize CPU usage if not specified. #usually won't take that much anyway.
    if n_workers is None:
        n_workers = mp.cpu_count()

    # Create fitting function for single condition
    def fit_condition(condition):
        from .objective import ObjectiveFunction
        from ..core.oxygen import OxygenModel

        oxygen_model = OxygenModel(
            o2_max=config.oxygen.get("o2_max", 8.0),
            o2_min=config.oxygen.get("o2_min", 0.1),
            reaeration_rate=config.oxygen.get("reaeration_rate", 15.0)
        )

        objective = ObjectiveFunction(
            experimental_time=condition['time'],
            experimental_substrate=condition['substrate'],
            experimental_biomass=condition['biomass'],
            model_type=model_type,
            initial_conditions=condition['initial_conditions'],
            t_span=condition['t_span'],
            parameter_names=parameter_names,
            oxygen_model=oxygen_model
        )

        optimizer = ParameterOptimizer(
            parameter_names=parameter_names,
            bounds={p: config.bounds[p] for p in parameter_names},
            initial_guess={p: config.initial_guesses[p] for p in parameter_names},
            method="L-BFGS-B"
        )

        return condition.get('label', 'unknown'), optimizer.optimize(objective)

    # Run in parallel
    if n_workers > 1 and len(conditions) > 1:
        with mp.Pool(n_workers) as pool:
            results = pool.map(fit_condition, conditions)
    else:
        results = [fit_condition(c) for c in conditions]

    return dict(results)


def compute_fit_statistics(
    result: OptimizationResult,
    objective: ObjectiveFunction,
    n_data_points: int
) -> OptimizationResult:
    """
    Compute fit statistics for an optimization result.

    Args:
        result: OptimizationResult from optimizer
        objective: ObjectiveFunction used for fitting
        n_data_points: Number of experimental data points

    Returns:
        OptimizationResult with updated statistics
    """
    from scipy.integrate import solve_ivp
    from ..core.ode_systems import SingleMonodODE, DualMonodODE, DualMonodLagODE
    from ..core.oxygen import OxygenModel

    params = result.parameters
    param_array = np.array([params[p] for p in objective.parameter_names])

    # Create ODE system
    ode_system = objective._create_ode_system(param_array)

    # Solve ODE
    solution = solve_ivp(
        fun=ode_system.derivatives,
        t_span=objective.t_span,
        y0=objective.initial_conditions,
        method='RK45',
        t_eval=objective.t_eval
    )

    # Interpolate to experimental times
    pred_substrate = np.interp(
        objective.experimental_time,
        solution.t,
        solution.y[0]
    )
    pred_biomass = np.interp(
        objective.experimental_time,
        solution.t,
        solution.y[1]
    )

    # Calculate statistics
    r2_substrate = calculate_r_squared(objective.experimental_substrate, pred_substrate)
    r2_biomass = calculate_r_squared(objective.experimental_biomass, pred_biomass)
    rmse_substrate = calculate_rmse(objective.experimental_substrate, pred_substrate)
    rmse_biomass = calculate_rmse(objective.experimental_biomass, pred_biomass)

    # Overall R² (weighted average)
    all_obs = np.concatenate([objective.experimental_substrate, objective.experimental_biomass])
    all_pred = np.concatenate([pred_substrate, pred_biomass])
    r2_overall = calculate_r_squared(all_obs, all_pred)

    # Total sum of squared residuals
    sse = np.sum((all_obs - all_pred) ** 2)

    n_params = len(objective.parameter_names)
    aic = calculate_aic(sse, n_params, n_data_points * 2)
    bic = calculate_bic(sse, n_params, n_data_points * 2)

    result.statistics = {
        "R_squared": r2_overall,
        "R_squared_substrate": r2_substrate,
        "R_squared_biomass": r2_biomass,
        "RMSE_substrate": rmse_substrate,
        "RMSE_biomass": rmse_biomass,
        "AIC": aic,
        "BIC": bic,
        "SSE": sse,
        "n_parameters": n_params,
        "n_data_points": n_data_points * 2
    }

    return result
