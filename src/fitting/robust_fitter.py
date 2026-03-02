"""
RobustFitter: High-Level API for Parameter Estimation.

This module provides a unified interface that combines:
1. Condition-specific weighting (handles heteroscedasticity)
2. Two-stage estimation (intelligent initialization)
3. Bootstrap aggregation (uncertainty quantification)

The scientist's interface:
    fitter = RobustFitter(model_type="dual_monod_lag")
    result = fitter.fit(conditions, config)
    print(f"qmax = {result.parameters['qmax']:.3f} "
          f"95% CI: {result.confidence_intervals['qmax']}")
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import time
import numpy as np
from scipy.integrate import solve_ivp

from .weighting import get_weighting_strategy, WeightingStrategy
from .two_stage import TwoStageEstimator, TwoStageResult
from .bootstrap import BootstrapEngine, BootstrapResult
from .optimizer import ParameterOptimizer
from .objective import GlobalObjectiveFunction
from ..core.ode_systems import SingleMonodODE, DualMonodODE, DualMonodLagODE
from ..core.oxygen import OxygenModel


class PicklablePredictor:
    """Picklable wrapper for prediction function used in bootstrap."""
    
    def __init__(self, model_type: str, oxygen_model):
        self.model_type = model_type
        self.oxygen_model = oxygen_model
    
    def __call__(self, cond, params):
        """Predict substrate and biomass for a condition."""
        from ..core.solvers import solve_ode
        
        # Get initial conditions - support both dict formats
        if 'initial_conditions' in cond:
            ic = cond['initial_conditions']
            S0, X0 = ic[0], ic[1]
            O0 = ic[2] if len(ic) > 2 else 8.0
        else:
            S0 = cond.get('S0', cond['substrate'][0])
            X0 = cond.get('X0', cond['biomass'][0])
            O0 = cond.get('O0', 8.0)
        
        # Create model
        if self.model_type == "single_monod":
            model = SingleMonodODE(
                qmax=params['qmax'], Ks=params['Ks'], Ki=params['Ki'],
                Y=params['Y'], b_decay=params['b_decay']
            )
            y0 = np.array([S0, X0])
        elif self.model_type == "dual_monod":
            model = DualMonodODE(
                qmax=params['qmax'], Ks=params['Ks'], Ki=params['Ki'],
                Y=params['Y'], b_decay=params['b_decay'],
                K_o2=params['K_o2'], Y_o2=params['Y_o2'],
                oxygen_model=self.oxygen_model
            )
            y0 = np.array([S0, X0, O0])
        elif self.model_type == "dual_monod_lag":
            model = DualMonodLagODE(
                qmax=params['qmax'], Ks=params['Ks'], Ki=params['Ki'],
                Y=params['Y'], b_decay=params['b_decay'],
                K_o2=params['K_o2'], Y_o2=params['Y_o2'],
                lag_time=params['lag_time'],
                oxygen_model=self.oxygen_model
            )
            y0 = np.array([S0, X0, O0])
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Solve ODE
        times = np.array(cond['time'])
        result = solve_ode(model, y0, (times[0], times[-1]), t_eval=times)
        
        # Extract predictions
        substrate_pred = result.states['Substrate']
        biomass_pred = result.states['Biomass']
        
        return substrate_pred, biomass_pred


class PicklableFitter:
    """Picklable wrapper for fitter function used in bootstrap."""
    
    def __init__(self, model_type: str, param_names: List[str], 
                 original_params: Dict[str, float], bounds: Dict[str, Tuple[float, float]],
                 weights: np.ndarray, oxygen_model):
        self.model_type = model_type
        self.param_names = param_names
        self.original_params = original_params
        self.bounds = bounds
        self.weights = weights
        self.oxygen_model = oxygen_model
    
    def __call__(self, conditions):
        """Fit parameters to conditions."""
        from scipy.optimize import minimize
        
        # Create objective function
        objective = WeightedGlobalObjective(
            conditions=conditions,
            model_type=self.model_type,
            parameter_names=self.param_names,
            oxygen_model=self.oxygen_model,
            weights=self.weights
        )
        
        # Extract bounds in correct order
        bounds_list = [self.bounds[name] for name in self.param_names]
        initial = [self.original_params[name] for name in self.param_names]
        
        # Run optimization (simplified for bootstrap)
        result = minimize(
            objective,
            x0=initial,
            method='L-BFGS-B',
            bounds=bounds_list,
            options={'maxiter': 500}
        )
        
        return dict(zip(self.param_names, result.x))


@dataclass
class RobustFitResult:
    """
    Complete results from robust parameter fitting.

    Provides everything needed for publication:
    - Point estimates with confidence intervals
    - Per-condition fit statistics
    - Diagnostic information for quality assessment
    """
    parameters: Dict[str, float]
    confidence_intervals: Dict[str, Tuple[float, float]]
    statistics: Dict[str, Dict[str, float]]  # {condition: {R2, RMSE, ...}}
    diagnostics: Dict[str, Any]
    fit_time_seconds: float
    model_type: str
    param_names: List[str]

    # Optional detailed results
    bootstrap_result: Optional[BootstrapResult] = None
    two_stage_result: Optional[TwoStageResult] = None

    def summary(self) -> str:
        """Generate publication-ready summary."""
        lines = [
            f"Robust Fit Results ({self.model_type})",
            "=" * 60,
            f"Fitting time: {self.fit_time_seconds:.2f} seconds",
            "",
            "Parameter Estimates (with 95% CI):",
        ]

        for param in self.param_names:
            val = self.parameters[param]
            if param in self.confidence_intervals:
                lo, hi = self.confidence_intervals[param]
                lines.append(f"  {param:12s}: {val:10.4f}  ({lo:.4f}, {hi:.4f})")
            else:
                lines.append(f"  {param:12s}: {val:10.4f}")

        lines.extend(["", "Fit Statistics by Condition:"])
        for cond_label, stats in self.statistics.items():
            r2 = stats.get('r_squared', 0)
            rmse = stats.get('rmse', 0)
            sse_sub = stats.get('sse_substrate', 0)
            sse_bio = stats.get('sse_biomass', 0)
            lines.append(
                f"  {cond_label:12s}: R² = {r2:.4f}, "
                f"SSE(S) = {sse_sub:.4f}, SSE(X) = {sse_bio:.4f}, RMSE = {rmse:.4f}"
            )

        if self.diagnostics.get('bootstrap_success_rate'):
            lines.extend([
                "",
                f"Bootstrap: {self.diagnostics['bootstrap_success_rate']:.1%} success rate"
            ])

        if self.diagnostics.get('two_stage_r_squared'):
            lines.extend([
                f"Two-Stage Init: R² = {self.diagnostics['two_stage_r_squared']:.3f}"
            ])

        return "\n".join(lines)


class RobustFitter:
    """
    High-level API for robust kinetic parameter estimation.

    Combines three complementary approaches:
    1. Weighted global fitting (handles heteroscedastic data)
    2. Two-stage initialization (avoids local minima)
    3. Bootstrap uncertainty quantification (confidence intervals)

    Example:
        >>> from src.fitting.robust_fitter import RobustFitter
        >>>
        >>> fitter = RobustFitter(
        >>>     model_type="dual_monod_lag",
        >>>     weighting="max_value",
        >>>     use_two_stage=True,
        >>>     bootstrap_iterations=500
        >>> )
        >>>
        >>> result = fitter.fit(conditions, config)
        >>> print(result.summary())
    """

    def __init__(
        self,
        model_type: str = "single_monod",
        weighting: str = "max_value",
        use_two_stage: bool = True,
        bootstrap_iterations: int = 500,
        bootstrap_workers: Optional[int] = None,
        optimizer_method: str = "L-BFGS-B",
        random_seed: Optional[int] = None
    ):
        """
        Initialize robust fitter.

        Args:
            model_type: ODE model type - "single_monod", "dual_monod", "dual_monod_lag"
            weighting: Weighting strategy - "uniform", "max_value", "variance", "range"
            use_two_stage: Whether to use two-stage initialization
            bootstrap_iterations: Number of bootstrap iterations (0 to disable)
            bootstrap_workers: Parallel workers for bootstrap (None = auto)
            optimizer_method: Optimization method - "L-BFGS-B" or "differential_evolution"
            random_seed: Seed for reproducibility
        """
        self.model_type = model_type.lower()
        self.weighting_strategy = get_weighting_strategy(weighting)
        self.use_two_stage = use_two_stage
        self.bootstrap_iterations = bootstrap_iterations
        self.bootstrap_workers = bootstrap_workers
        self.optimizer_method = optimizer_method
        self.random_seed = random_seed

        # Set parameter names based on model type
        self.param_names = self._get_param_names()

    def _get_param_names(self) -> List[str]:
        """Get parameter names for the model type."""
        base_params = ["qmax", "Ks", "Ki", "Y", "b_decay"]

        if self.model_type == "single_monod":
            return base_params
        elif self.model_type == "dual_monod":
            return base_params + ["K_o2", "Y_o2"]
        elif self.model_type == "dual_monod_lag":
            return base_params + ["K_o2", "Y_o2", "lag_time"]
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def fit(
        self,
        conditions: List[Dict[str, Any]],
        config: Any,
        verbose: bool = True
    ) -> RobustFitResult:
        """
        Fit parameters to experimental data.

        Args:
            conditions: List of condition dictionaries with keys:
                - 'time': Time points (np.ndarray)
                - 'substrate': Substrate measurements (np.ndarray)
                - 'biomass': Biomass measurements (np.ndarray)
                - 'initial_conditions': [S0, X0] or [S0, X0, O2_0]
                - 't_span': (t_start, t_end)
                - 'label': Condition identifier (optional)
            config: SubstrateConfig or dict with initial_guesses and bounds
            verbose: Whether to print progress

        Returns:
            RobustFitResult with parameters, confidence intervals, and diagnostics
        """
        start_time = time.time()
        diagnostics = {}

        if verbose:
            print(f"Starting robust fit: {self.model_type}")
            print(f"  Weighting: {self.weighting_strategy.get_name()}")
            print(f"  Two-stage init: {self.use_two_stage}")
            print(f"  Bootstrap: {self.bootstrap_iterations} iterations")

        # Extract configuration
        initial_guesses = self._extract_initial_guesses(config)
        bounds = self._extract_bounds(config)
        oxygen_model = self._create_oxygen_model(config)

        # Step 1: Two-stage initialization (if enabled)
        two_stage_result = None
        if self.use_two_stage:
            if verbose:
                print("\nPhase 1: Two-stage initialization...")
            two_stage_result = self._run_two_stage(conditions, initial_guesses)
            initial_guesses = two_stage_result.initial_guesses
            diagnostics['two_stage_r_squared'] = two_stage_result.stage1_quality.get('r_squared', 0)
            diagnostics['two_stage_warnings'] = two_stage_result.warnings

            if verbose and two_stage_result.warnings:
                for warn in two_stage_result.warnings:
                    print(f"    Warning: {warn}")

        # Step 2: Compute condition weights
        weights = self.weighting_strategy.compute_weights(conditions)
        diagnostics['condition_weights'] = dict(zip(
            [c.get('label', f'Cond_{i}') for i, c in enumerate(conditions)],
            weights.tolist()
        ))

        if verbose:
            print("\nCondition weights:")
            for label, w in diagnostics['condition_weights'].items():
                print(f"    {label}: {w:.3f}")

        # Step 3: Global fit with weighting
        if verbose:
            print("\nPhase 2: Weighted global optimization...")

        fitted_params = self._run_weighted_fit(
            conditions, initial_guesses, bounds, weights, oxygen_model
        )

        if verbose:
            print("  Optimization complete")
            for p, v in fitted_params.items():
                print(f"    {p}: {v:.4f}")

        # Step 4: Compute per-condition statistics
        statistics = self._compute_statistics(conditions, fitted_params, oxygen_model)

        # Step 5: Bootstrap (if enabled)
        bootstrap_result = None
        confidence_intervals = {}

        if self.bootstrap_iterations > 0:
            if verbose:
                print(f"\nPhase 3: Bootstrap ({self.bootstrap_iterations} iterations)...")

            bootstrap_result = self._run_bootstrap(
                conditions, fitted_params, bounds, weights, oxygen_model
            )

            confidence_intervals = bootstrap_result.confidence_intervals
            diagnostics['bootstrap_success_rate'] = (
                bootstrap_result.n_successful /
                (bootstrap_result.n_successful + bootstrap_result.n_failed)
            )

            if verbose:
                print(f"  Success rate: {diagnostics['bootstrap_success_rate']:.1%}")

        fit_time = time.time() - start_time

        if verbose:
            print(f"\nFitting complete in {fit_time:.2f} seconds")

        return RobustFitResult(
            parameters=fitted_params,
            confidence_intervals=confidence_intervals,
            statistics=statistics,
            diagnostics=diagnostics,
            fit_time_seconds=fit_time,
            model_type=self.model_type,
            param_names=self.param_names,
            bootstrap_result=bootstrap_result,
            two_stage_result=two_stage_result
        )

    def _extract_initial_guesses(self, config) -> Dict[str, float]:
        """Extract initial guesses from config."""
        if hasattr(config, 'initial_guesses'):
            return dict(config.initial_guesses)
        elif isinstance(config, dict) and 'initial_guesses' in config:
            return dict(config['initial_guesses'])
        else:
            # Defaults
            return {
                'qmax': 2.0, 'Ks': 500.0, 'Ki': 25000.0,
                'Y': 0.35, 'b_decay': 0.01, 'K_o2': 0.15,
                'Y_o2': 0.8, 'lag_time': 1.0
            }

    def _extract_bounds(self, config) -> Dict[str, Tuple[float, float]]:
        """Extract parameter bounds from config."""
        if hasattr(config, 'bounds'):
            return {k: tuple(v) for k, v in config.bounds.items()}
        elif isinstance(config, dict) and 'bounds' in config:
            return {k: tuple(v) for k, v in config['bounds'].items()}
        else:
            # Defaults
            return {
                'qmax': (0.1, 10.0), 'Ks': (10.0, 5000.0),
                'Ki': (100.0, 100000.0), 'Y': (0.05, 1.0),
                'b_decay': (0.001, 0.5), 'K_o2': (0.01, 2.0),
                'Y_o2': (0.1, 3.0), 'lag_time': (0.0, 20.0)
            }

    def _create_oxygen_model(self, config) -> OxygenModel:
        """Create oxygen model from config."""
        if hasattr(config, 'oxygen'):
            oxy = config.oxygen
        elif isinstance(config, dict) and 'oxygen' in config:
            oxy = config['oxygen']
        else:
            oxy = {}

        return OxygenModel(
            o2_max=oxy.get('o2_max', 8.0),
            o2_min=oxy.get('o2_min', 0.01),
            reaeration_rate=oxy.get('reaeration_rate', 5.0),
            o2_range=oxy.get('o2_range', 8.0)
        )

    def _run_two_stage(
        self,
        conditions: List[Dict[str, Any]],
        default_params: Dict[str, float]
    ) -> TwoStageResult:
        """Run two-stage estimation."""
        estimator = TwoStageEstimator()
        return estimator.estimate_initial_params(
            conditions, self.param_names, default_params
        )

    def _run_weighted_fit(
        self,
        conditions: List[Dict[str, Any]],
        initial_guesses: Dict[str, float],
        bounds: Dict[str, Tuple[float, float]],
        weights: np.ndarray,
        oxygen_model: OxygenModel
    ) -> Dict[str, float]:
        """Run weighted global optimization."""
        # Create weighted objective
        objective = WeightedGlobalObjective(
            conditions=conditions,
            model_type=self.model_type,
            parameter_names=self.param_names,
            oxygen_model=oxygen_model,
            weights=weights
        )

        # Build initial guess and bounds dicts
        bounds_dict = {p: bounds.get(p, (0.01, 1000.0)) for p in self.param_names}
        initial_dict = {p: initial_guesses.get(p, 1.0) for p in self.param_names}

        # Create optimizer and run
        optimizer = ParameterOptimizer(
            parameter_names=self.param_names,
            bounds=bounds_dict,
            initial_guess=initial_dict,
            method=self.optimizer_method,
            verbose=False
        )

        result = optimizer.optimize(objective)

        # Convert to dict
        return result.parameters

    def _compute_statistics(
        self,
        conditions: List[Dict[str, Any]],
        params: Dict[str, float],
        oxygen_model: OxygenModel
    ) -> Dict[str, Dict[str, float]]:
        """Compute fit statistics for each condition."""
        from ..fitting.statistics import calculate_r_squared, calculate_rmse

        statistics = {}

        for cond in conditions:
            label = cond.get('label', 'unknown')

            # Predict
            substrate_pred, biomass_pred = self._predict(cond, params, oxygen_model)

            # Calculate stats
            substrate_obs = np.array(cond['substrate'])
            biomass_obs = np.array(cond['biomass'])

            r2_sub = calculate_r_squared(substrate_obs, substrate_pred)
            r2_bio = calculate_r_squared(biomass_obs, biomass_pred)
            rmse_sub = calculate_rmse(substrate_obs, substrate_pred)
            rmse_bio = calculate_rmse(biomass_obs, biomass_pred)
            sse_sub = float(np.sum((substrate_obs - substrate_pred) ** 2))
            sse_bio = float(np.sum((biomass_obs - biomass_pred) ** 2))

            statistics[label] = {
                'r_squared': (r2_sub + r2_bio) / 2,
                'r_squared_substrate': r2_sub,
                'r_squared_biomass': r2_bio,
                'sse_substrate': sse_sub,
                'sse_biomass': sse_bio,
                'sse_total': sse_sub + sse_bio,
                'rmse': (rmse_sub + rmse_bio) / 2,
                'rmse_substrate': rmse_sub,
                'rmse_biomass': rmse_bio
            }

        return statistics

    def _predict(
        self,
        condition: Dict[str, Any],
        params: Dict[str, float],
        oxygen_model: OxygenModel
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict substrate and biomass for a condition."""
        time = np.array(condition['time'])
        t_span = condition.get('t_span', (time[0], time[-1]))
        initial_conditions = condition['initial_conditions']

        # Create ODE system
        ode = self._create_ode_system(params, oxygen_model)

        # Solve
        solution = solve_ivp(
            fun=ode.derivatives,
            t_span=t_span,
            y0=initial_conditions,
            method='RK45',
            t_eval=np.linspace(t_span[0], t_span[1], 1000),
            rtol=1e-6,
            atol=1e-9
        )

        # Interpolate to experimental time points
        substrate_pred = np.interp(time, solution.t, solution.y[0])
        biomass_pred = np.interp(time, solution.t, solution.y[1])

        return substrate_pred, biomass_pred

    def _create_ode_system(self, params: Dict[str, float], oxygen_model: OxygenModel):
        """Create ODE system from parameters."""
        if self.model_type == "single_monod":
            return SingleMonodODE(
                qmax=params['qmax'], Ks=params['Ks'], Ki=params['Ki'],
                Y=params['Y'], b_decay=params['b_decay']
            )
        elif self.model_type == "dual_monod":
            return DualMonodODE(
                qmax=params['qmax'], Ks=params['Ks'], Ki=params['Ki'],
                Y=params['Y'], b_decay=params['b_decay'],
                K_o2=params['K_o2'], Y_o2=params['Y_o2'],
                oxygen_model=oxygen_model
            )
        elif self.model_type == "dual_monod_lag":
            return DualMonodLagODE(
                qmax=params['qmax'], Ks=params['Ks'], Ki=params['Ki'],
                Y=params['Y'], b_decay=params['b_decay'],
                K_o2=params['K_o2'], Y_o2=params['Y_o2'],
                lag_time=params['lag_time'],
                oxygen_model=oxygen_model
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def _run_bootstrap(
        self,
        conditions: List[Dict[str, Any]],
        original_params: Dict[str, float],
        bounds: Dict[str, Tuple[float, float]],
        weights: np.ndarray,
        oxygen_model: OxygenModel
    ) -> BootstrapResult:
        """Run bootstrap uncertainty quantification."""
        # Use picklable wrappers instead of closures (required for multiprocessing)
        fitter_func = PicklableFitter(
            model_type=self.model_type,
            param_names=self.param_names,
            original_params=original_params,
            bounds=bounds,
            weights=weights,
            oxygen_model=oxygen_model
        )
        
        predict_func = PicklablePredictor(
            model_type=self.model_type,
            oxygen_model=oxygen_model
        )

        # Run bootstrap
        engine = BootstrapEngine(
            n_iterations=self.bootstrap_iterations,
            n_workers=self.bootstrap_workers,
            random_seed=self.random_seed
        )

        return engine.run(
            fitter_func=fitter_func,
            conditions=conditions,
            original_params=original_params,
            param_names=self.param_names,
            predict_func=predict_func
        )


class WeightedGlobalObjective:
    """
    Weighted objective function for global fitting.

    Applies condition-specific weights to handle heteroscedasticity.
    """

    def __init__(
        self,
        conditions: List[Dict[str, Any]],
        model_type: str,
        parameter_names: List[str],
        oxygen_model: OxygenModel,
        weights: np.ndarray,
        num_eval_points: int = 1000
    ):
        """Initialize weighted objective."""
        self.conditions = conditions
        self.model_type = model_type
        self.parameter_names = parameter_names
        self.oxygen_model = oxygen_model
        self.weights = weights
        self.num_eval_points = num_eval_points
        self.n_evaluations = 0

    def __call__(self, params: np.ndarray) -> float:
        """Evaluate weighted objective."""
        self.n_evaluations += 1
        total_error = 0.0

        param_dict = dict(zip(self.parameter_names, params))

        for i, cond in enumerate(self.conditions):
            try:
                error = self._compute_condition_error(cond, param_dict)
                weighted_error = error * self.weights[i]
                total_error += weighted_error
            except Exception:
                return 1e10

        return total_error

    def _compute_condition_error(
        self,
        condition: Dict[str, Any],
        params: Dict[str, float]
    ) -> float:
        """Compute SSE for one condition."""
        time = np.array(condition['time'])
        substrate_obs = np.array(condition['substrate'])
        biomass_obs = np.array(condition['biomass'])
        t_span = condition.get('t_span', (time[0], time[-1]))
        initial_conditions = condition['initial_conditions']

        # Create ODE
        ode = self._create_ode_system(params)

        # Solve
        solution = solve_ivp(
            fun=ode.derivatives,
            t_span=t_span,
            y0=initial_conditions,
            method='RK45',
            t_eval=np.linspace(t_span[0], t_span[1], self.num_eval_points),
            rtol=1e-6,
            atol=1e-9
        )

        if not solution.success:
            return 1e10

        # Interpolate
        substrate_pred = np.interp(time, solution.t, solution.y[0])
        biomass_pred = np.interp(time, solution.t, solution.y[1])

        # Normalized SSE (by max value to handle scale differences)
        sub_norm = max(np.max(substrate_obs), 1e-6)
        bio_norm = max(np.max(biomass_obs), 1e-6)

        error_sub = np.sum(((substrate_obs - substrate_pred) / sub_norm) ** 2)
        error_bio = np.sum(((biomass_obs - biomass_pred) / bio_norm) ** 2)

        return error_sub + error_bio

    def _create_ode_system(self, params: Dict[str, float]):
        """Create ODE system from parameters."""
        if self.model_type == "single_monod":
            return SingleMonodODE(
                qmax=params['qmax'], Ks=params['Ks'], Ki=params['Ki'],
                Y=params['Y'], b_decay=params['b_decay']
            )
        elif self.model_type == "dual_monod":
            return DualMonodODE(
                qmax=params['qmax'], Ks=params['Ks'], Ki=params['Ki'],
                Y=params['Y'], b_decay=params['b_decay'],
                K_o2=params['K_o2'], Y_o2=params['Y_o2'],
                oxygen_model=self.oxygen_model
            )
        elif self.model_type == "dual_monod_lag":
            return DualMonodLagODE(
                qmax=params['qmax'], Ks=params['Ks'], Ki=params['Ki'],
                Y=params['Y'], b_decay=params['b_decay'],
                K_o2=params['K_o2'], Y_o2=params['Y_o2'],
                lag_time=params['lag_time'],
                oxygen_model=self.oxygen_model
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
