"""
Individual Condition Workflow for separate fitting of each experimental condition.

This module provides a workflow that fits parameters independently for each
concentration condition, providing detailed statistics, confidence intervals,
and diagnostic visualizations for each fit before optionally combining them
for global parameter estimation.

Approach based on Two-Stage Estimation with individual condition analysis.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import json
from datetime import datetime

from src.core.ode_systems import SingleMonodODE, SingleMonodLagODE, DualMonodODE, DualMonodLagODE, BaseODE
from src.core.solvers import solve_ode
from src.core.oxygen import OxygenModel
from src.fitting.objective import ObjectiveFunction, GlobalObjectiveFunction
from src.fitting.optimizer import ParameterOptimizer, OptimizationResult
from src.fitting.statistics import (
    calculate_separate_statistics,
    calculate_parameter_confidence_intervals_with_diagnostics,
    residual_analysis,
    calculate_r_squared,
    calculate_rmse,
    calculate_nrmse,
    calculate_mae
)
from src.io.data_loader import ExperimentalData
from src.io.config_loader import SubstrateConfig
from src.io.results_writer import ResultsWriter
from src.io.pdf_report import generate_individual_condition_report
from src.utils.plotting import setup_figure, style_axis, COLORS, COLOR_LIST, save_figure


@dataclass
class ConditionResult:
    """
    Results for a single condition fit.
    
    Attributes:
        condition: Condition label (e.g., '5mM')
        parameters: Fitted parameter values
        confidence_intervals: 95% CI for each parameter
        statistics: Fit statistics (R², RMSE, NRMSE, etc.)
        residual_diagnostics: Residual analysis results
        predictions: DataFrame with time, substrate, biomass predictions
        experimental_data: Original experimental data for this condition
        optimization_result: Full optimization result object
        success: Whether optimization succeeded
    """
    condition: str
    parameters: Dict[str, float]
    confidence_intervals: Dict[str, Dict[str, float]]
    statistics: Dict[str, Any]
    residual_diagnostics: Dict[str, Any]
    predictions: pd.DataFrame
    experimental_time: np.ndarray
    experimental_substrate: np.ndarray
    experimental_biomass: np.ndarray
    optimization_result: OptimizationResult
    success: bool
    ci_diagnostics: Dict[str, Any] = field(default_factory=dict)
    
    def summary(self) -> str:
        """Generate text summary for this condition."""
        lines = [
            f"\n{'='*50}",
            f"Condition: {self.condition}",
            f"Optimization: {'Success' if self.success else 'Failed'}",
            f"{'='*50}",
            "",
            "Fitted Parameters (± Std Error, 95% CI):",
        ]
        
        for name, value in self.parameters.items():
            ci = self.confidence_intervals.get(name, {})
            if ci and not np.isnan(ci.get('std_error', np.nan)):
                lines.append(
                    f"  {name:12s}: {value:10.6f} ± {ci['std_error']:.4f} "
                    f"[{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}]"
                )
            else:
                lines.append(f"  {name:12s}: {value:10.6f}")
        
        lines.extend([
            "",
            "Goodness of Fit:",
            f"  Substrate: R²={self.statistics['substrate']['R_squared']:.4f}, "
            f"SSE={self.statistics['substrate']['SSE']:.2f}, "
            f"RMSE={self.statistics['substrate']['RMSE']:.2f}, "
            f"NRMSE={self.statistics['substrate']['NRMSE']:.4f}",
            f"  Biomass:   R²={self.statistics['biomass']['R_squared']:.4f}, "
            f"SSE={self.statistics['biomass']['SSE']:.4f}, "
            f"RMSE={self.statistics['biomass']['RMSE']:.4f}, "
            f"NRMSE={self.statistics['biomass']['NRMSE']:.4f}",
            "",
            "Residual Analysis:",
            f"  Residual Mean: {self.residual_diagnostics.get('mean', 0):.4f}",
            f"  Residual Std:  {self.residual_diagnostics.get('std', 0):.4f}",
            f"  Autocorrelation (lag-1): {self.residual_diagnostics.get('autocorrelation_lag1', 0):.4f}",
        ])
        
        return "\n".join(lines)


@dataclass
class IndividualConditionResult:
    """
    Container for all individual condition fitting results.

    Attributes:
        model_type: Type of model used
        condition_results: Results for each condition
        parameter_summary: Summary statistics across conditions
        global_parameters: Optional global parameter estimates
        global_optimization_result: Full optimization result from global cost function
        global_loss: Final global cost function value
        individual_losses: Per-condition loss values at their individual optima
        figures: Generated figure paths
        config: Configuration used
    """
    model_type: str
    condition_results: Dict[str, ConditionResult]
    parameter_summary: Dict[str, Dict[str, float]]
    global_parameters: Optional[Dict[str, float]]
    global_optimization_result: Optional[OptimizationResult] = None
    global_confidence_intervals: Optional[Dict[str, Dict[str, float]]] = None
    global_ci_diagnostics: Optional[Dict[str, Any]] = None
    global_loss: Optional[float] = None
    individual_losses: Optional[Dict[str, float]] = None
    figures: List[Path] = field(default_factory=list)
    config: SubstrateConfig = None
    display_name: str = ''
    
    def summary(self) -> str:
        """Generate comprehensive summary."""
        header_name = self.display_name if self.display_name else self.model_type.upper()
        lines = [
            "",
            "=" * 70,
            f"INDIVIDUAL CONDITION FITTING RESULTS - {header_name}",
            "=" * 70,
            "",
        ]
        
        # Individual condition summaries
        for cond, result in self.condition_results.items():
            lines.append(result.summary())
            if result.ci_diagnostics:
                method = result.ci_diagnostics.get('method', 'unknown')
                if method == 'mcmc':
                    acc = result.ci_diagnostics.get('acceptance_rate', np.nan)
                    if np.isfinite(acc):
                        lines.append(f"  CI Diagnostics (MCMC): acceptance={acc:.3f}")
                elif method in {'hessian', 'hessian_log'}:
                    cn = result.ci_diagnostics.get('hessian_condition_number', np.nan)
                    if np.isfinite(cn):
                        lines.append(f"  CI Diagnostics ({method}): cond(H)={cn:.2e}")
        
        # Parameter summary table
        lines.extend([
            "",
            "=" * 70,
            "PARAMETER SUMMARY ACROSS CONDITIONS",
            "=" * 70,
            "",
            f"{'Parameter':<12} {'Mean':>12} {'Std':>12} {'CV (%)':>10} {'Min':>12} {'Max':>12}",
            "-" * 70,
        ])
        
        for param, stats in self.parameter_summary.items():
            lines.append(
                f"{param:<12} {stats['mean']:>12.4f} {stats['std']:>12.4f} "
                f"{stats['cv']:>10.2f} {stats['min']:>12.4f} {stats['max']:>12.4f}"
            )
        
        if self.individual_losses:
            lines.extend([
                "",
                "=" * 70,
                "INDIVIDUAL CONDITION LOSSES (Local Objective Values)",
                "=" * 70,
                "",
            ])
            for cond, loss in self.individual_losses.items():
                lines.append(f"  {cond}: {loss:.6f}")

        if self.global_parameters:
            lines.extend([
                "",
                "=" * 70,
                "RECOMMENDED GLOBAL PARAMETERS",
                "(Obtained by minimizing sum of per-condition normalized losses)",
                "=" * 70,
                "",
                "Strategy:",
                "  1. Each condition was fitted independently (local loss minimization)",
                "  2. Median of individual fits used as initial guess for global optimization",
                "  3. A global cost function J = sum_i L_i(theta) was constructed",
                "     where L_i(theta) is the normalized SSE for condition i",
                "  4. L-BFGS-B optimization minimized J over all conditions simultaneously",
                "",
            ])
            if self.global_confidence_intervals:
                for param, value in self.global_parameters.items():
                    ci = self.global_confidence_intervals.get(param, {})
                    se = ci.get('std_error', float('nan'))
                    ci_lo = ci.get('ci_lower', float('nan'))
                    ci_hi = ci.get('ci_upper', float('nan'))
                    if not np.isnan(se):
                        lines.append(
                            f"  {param:12s}: {value:10.6f} +/- {se:.4f}  "
                            f"95% CI [{ci_lo:.4f}, {ci_hi:.4f}]"
                        )
                    else:
                        lines.append(
                            f"  {param:12s}: {value:10.6f}  "
                            f"(CI unavailable — check identifiability)"
                        )
            else:
                for param, value in self.global_parameters.items():
                    lines.append(f"  {param}: {value:.6f}")

            if self.global_loss is not None:
                lines.extend([
                    "",
                    f"  Global Cost Function Value: {self.global_loss:.6f}",
                ])

            if self.global_ci_diagnostics:
                g_method = self.global_ci_diagnostics.get('method', 'unknown')
                if g_method == 'mcmc':
                    g_acc = self.global_ci_diagnostics.get('acceptance_rate', np.nan)
                    lines.append(
                        f"  Global CI Diagnostics (MCMC): acceptance={g_acc:.3f}"
                        if np.isfinite(g_acc) else
                        "  Global CI Diagnostics (MCMC): unavailable"
                    )
                elif g_method in {'hessian', 'hessian_log'}:
                    g_cn = self.global_ci_diagnostics.get('hessian_condition_number', np.nan)
                    if np.isfinite(g_cn):
                        lines.append(f"  Global CI Diagnostics ({g_method}): cond(H)={g_cn:.2e}")

            if self.global_optimization_result is not None:
                lines.extend([
                    f"  Optimization Converged: {self.global_optimization_result.success}",
                    f"  Function Evaluations: {self.global_optimization_result.n_function_evals}",
                ])

        return "\n".join(lines)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to a DataFrame for easy export."""
        rows = []
        for cond, result in self.condition_results.items():
            row = {'Condition': cond}
            for param, value in result.parameters.items():
                row[param] = value
                ci = result.confidence_intervals.get(param, {})
                row[f'{param}_std_error'] = ci.get('std_error', np.nan)
                row[f'{param}_ci_lower'] = ci.get('ci_lower', np.nan)
                row[f'{param}_ci_upper'] = ci.get('ci_upper', np.nan)
                row[f'{param}_r_hat'] = ci.get('r_hat', np.nan)
                row[f'{param}_ess'] = ci.get('effective_sample_size', np.nan)
            row['ci_method'] = result.ci_diagnostics.get('method', np.nan)
            row['ci_acceptance_rate'] = result.ci_diagnostics.get('acceptance_rate', np.nan)
            row['ci_hessian_condition_number'] = result.ci_diagnostics.get('hessian_condition_number', np.nan)
            row['R2_substrate'] = result.statistics['substrate']['R_squared']
            row['R2_biomass'] = result.statistics['biomass']['R_squared']
            row['NRMSE_substrate'] = result.statistics['substrate']['NRMSE']
            row['NRMSE_biomass'] = result.statistics['biomass']['NRMSE']
            rows.append(row)
        
        return pd.DataFrame(rows)


def _fit_condition_worker(init_kwargs: dict, condition: str,
                          optimization_method: str) -> 'ConditionResult':
    """
    Module-level worker for ProcessPoolExecutor.

    Creates a temporary IndividualConditionWorkflow in the child process
    and fits a single condition.  Must be at module level so that
    ProcessPoolExecutor can pickle it.
    """
    workflow = IndividualConditionWorkflow(**init_kwargs)
    return workflow.fit_condition(condition, optimization_method, verbose=False)


class IndividualConditionWorkflow:
    """
    Workflow for fitting parameters individually for each experimental condition.

    This approach provides:
    1. Separate parameter estimates for each concentration
    2. Confidence intervals using Jacobian-based standard errors
    3. Residual diagnostics for model adequacy assessment
    4. Visualization at each stage
    5. Summary statistics across conditions
    6. Optional weighted global parameter estimation

    Supports: single_monod, single_monod_lag, dual_monod, dual_monod_lag models.
    """
    
    # Human-readable display names keyed by (model_type, no_inhibition)
    DISPLAY_NAMES = {
        ('single_monod', True): 'Single Monod',
        ('single_monod', False): 'Single Monod (Haldane)',
        ('single_monod_lag', True): 'Single Monod + Lag',
        ('single_monod_lag', False): 'Single Monod + Lag (Haldane)',
        ('dual_monod', True): 'Dual Monod',
        ('dual_monod', False): 'Dual Monod (Haldane)',
        ('dual_monod_lag', True): 'Dual Monod + Lag',
        ('dual_monod_lag', False): 'Dual Monod + Lag (Haldane)',
    }

    # Model configurations
    MODEL_CONFIGS = {
        'single_monod': {
            'parameters': ['qmax', 'Ks', 'Ki', 'Y', 'b_decay'],
            'ode_class': SingleMonodODE,
            'n_states': 2,  # S, X
        },
        'single_monod_lag': {
            'parameters': ['qmax', 'Ks', 'Ki', 'Y', 'b_decay', 'lag_time'],
            'ode_class': SingleMonodLagODE,
            'n_states': 2,  # S, X
        },
        'dual_monod': {
            'parameters': ['qmax', 'Ks', 'Ki', 'Y', 'b_decay', 'K_o2', 'Y_o2'],
            'ode_class': DualMonodODE,
            'n_states': 3,  # S, X, O2
        },
        'dual_monod_lag': {
            'parameters': ['qmax', 'Ks', 'Ki', 'Y', 'b_decay', 'K_o2', 'Y_o2', 'lag_time'],
            'ode_class': DualMonodLagODE,
            'n_states': 3,  # S, X, O2
        }
    }
    
    def __init__(
        self,
        config: SubstrateConfig,
        experimental_data: ExperimentalData,
        model_type: str = 'single_monod',
        output_dir: str = 'results',
        ci_method: str = 'hessian',
        ci_confidence_level: float = 0.95,
        ci_mcmc_samples: int = 4000,
        ci_mcmc_burn_in: int = 1000,
        ci_mcmc_step_scale: float = 0.05,
        ci_mcmc_seed: Optional[int] = None,
        config_path: Optional[str] = None,
        data_path: Optional[str] = None,
        no_inhibition: bool = False,
        n_workers: int = 1,
        mcmc_adaptive: bool = False,
        normalize_objective: bool = False,
        global_guess_strategy: str = 'median',
    ):
        """
        Initialize the individual condition workflow.

        Args:
            config: Substrate configuration with parameters and bounds
            experimental_data: Loaded experimental data
            model_type: 'single_monod', 'dual_monod', or 'dual_monod_lag'
            output_dir: Base directory for output files
            ci_method: Confidence interval method: 'hessian', 'hessian_log', 'mcmc'
            ci_confidence_level: Confidence level for intervals (default 0.95)
            ci_mcmc_samples: Number of MCMC posterior samples
            ci_mcmc_burn_in: Number of MCMC burn-in iterations
            ci_mcmc_step_scale: Proposal step scale as fraction of parameter range
            ci_mcmc_seed: Optional random seed for MCMC
            no_inhibition: If True, exclude Ki (use basic Monod instead of Haldane)
            n_workers: Number of parallel workers for per-condition fitting (1 = sequential)
            mcmc_adaptive: Use Hessian-informed adaptive MCMC proposals (default False)
            normalize_objective: Weight S and X residuals by 1/range (default False)
            global_guess_strategy: Strategy for global optimisation initial guess.
                'median' — median of individual fits (robust to outliers, default).
                'best_r2' — parameters from the condition with highest mean R²(S,X).
        """
        if model_type not in self.MODEL_CONFIGS:
            raise ValueError(f"Unknown model type: {model_type}. "
                           f"Choose from: {list(self.MODEL_CONFIGS.keys())}")
        
        self.config = config
        self.experimental_data = experimental_data
        self.model_type = model_type
        self.model_config = self.MODEL_CONFIGS[model_type]

        valid_ci_methods = {'hessian', 'hessian_log', 'mcmc'}
        if ci_method not in valid_ci_methods:
            raise ValueError(
                f"Unknown ci_method: {ci_method}. Choose from: {sorted(valid_ci_methods)}"
            )
        self.ci_method = ci_method
        self.ci_confidence_level = float(ci_confidence_level)
        self.ci_mcmc_samples = int(ci_mcmc_samples)
        self.ci_mcmc_burn_in = int(ci_mcmc_burn_in)
        self.ci_mcmc_step_scale = float(ci_mcmc_step_scale)
        self.ci_mcmc_seed = ci_mcmc_seed
        self.no_inhibition = no_inhibition
        self.config_path = config_path
        self.data_path = data_path
        self.mcmc_adaptive = mcmc_adaptive
        self.normalize_objective = normalize_objective

        valid_guess_strategies = {'median', 'best_r2'}
        if global_guess_strategy not in valid_guess_strategies:
            raise ValueError(
                f"Unknown global_guess_strategy: {global_guess_strategy}. "
                f"Choose from: {sorted(valid_guess_strategies)}"
            )
        self.global_guess_strategy = global_guess_strategy

        self.results_writer = ResultsWriter(
            base_dir=output_dir,
            substrate_name=config.name
        )
        
        self.n_workers = max(1, int(n_workers))

        # Oxygen model for dual Monod workflows
        self.oxygen_model = OxygenModel(
            o2_max=config.oxygen.get("o2_max", 8.0),
            o2_min=config.oxygen.get("o2_min", 0.1),
            reaeration_rate=config.oxygen.get("reaeration_rate", 15.0),
            o2_range=config.oxygen.get("o2_range", 8.0)
        )

        # Store init kwargs for parallel workers (they recreate the workflow)
        self._init_kwargs = dict(
            config=config,
            experimental_data=experimental_data,
            model_type=model_type,
            output_dir=output_dir,
            ci_method=ci_method,
            ci_confidence_level=ci_confidence_level,
            ci_mcmc_samples=ci_mcmc_samples,
            ci_mcmc_burn_in=ci_mcmc_burn_in,
            ci_mcmc_step_scale=ci_mcmc_step_scale,
            ci_mcmc_seed=ci_mcmc_seed,
            config_path=config_path,
            data_path=data_path,
            no_inhibition=no_inhibition,
            n_workers=1,  # workers themselves run sequentially
            mcmc_adaptive=mcmc_adaptive,
            normalize_objective=normalize_objective,
            global_guess_strategy=global_guess_strategy,
        )
    
    @property
    def display_name(self) -> str:
        """Human-readable model name based on (model_type, no_inhibition)."""
        return self.DISPLAY_NAMES.get(
            (self.model_type, self.no_inhibition),
            self.model_type.replace('_', ' ').title()
        )

    @property
    def parameter_names(self) -> List[str]:
        """Get parameter names for the current model."""
        names = self.model_config['parameters']
        if self.no_inhibition:
            names = [p for p in names if p != 'Ki']
        return names
    
    def create_ode_system(self, parameters: Dict[str, float]) -> BaseODE:
        """Create ODE system with given parameters."""
        ode_class = self.model_config['ode_class']
        
        ki_value = parameters.get('Ki', None)

        if self.model_type == 'single_monod':
            return ode_class(
                qmax=parameters['qmax'],
                Ks=parameters['Ks'],
                Ki=ki_value,
                Y=parameters['Y'],
                b_decay=parameters['b_decay']
            )
        elif self.model_type == 'single_monod_lag':
            return ode_class(
                qmax=parameters['qmax'],
                Ks=parameters['Ks'],
                Ki=ki_value,
                Y=parameters['Y'],
                b_decay=parameters['b_decay'],
                lag_time=parameters['lag_time']
            )
        elif self.model_type == 'dual_monod':
            return ode_class(
                qmax=parameters['qmax'],
                Ks=parameters['Ks'],
                Ki=ki_value,
                Y=parameters['Y'],
                b_decay=parameters['b_decay'],
                K_o2=parameters['K_o2'],
                Y_o2=parameters['Y_o2'],
                oxygen_model=self.oxygen_model
            )
        else:  # dual_monod_lag
            return ode_class(
                qmax=parameters['qmax'],
                Ks=parameters['Ks'],
                Ki=ki_value,
                Y=parameters['Y'],
                b_decay=parameters['b_decay'],
                K_o2=parameters['K_o2'],
                Y_o2=parameters['Y_o2'],
                lag_time=parameters['lag_time'],
                oxygen_model=self.oxygen_model
            )
    
    def _get_initial_conditions(self, S0: float, X0: float) -> List[float]:
        """Get initial conditions based on model type."""
        if self.model_type in ('single_monod', 'single_monod_lag'):
            return [S0, X0]
        else:
            # Dual Monod: include oxygen at saturation
            O2_0 = self.oxygen_model.o2_max
            return [S0, X0, O2_0]
    
    def fit_condition(
        self,
        condition: str,
        optimization_method: str = 'L-BFGS-B',
        verbose: bool = True
    ) -> ConditionResult:
        """
        Fit parameters for a single condition.
        
        Args:
            condition: Condition label (e.g., '5mM')
            optimization_method: Optimization algorithm
            verbose: Print progress
            
        Returns:
            ConditionResult with fitted parameters and diagnostics
        """
        if verbose:
            print(f"\nFitting condition: {condition}")
        
        # Get data for this condition
        time, substrate, biomass = self.experimental_data.get_condition_data(condition)
        S0, X0 = substrate[0], biomass[0]
        initial_conditions = self._get_initial_conditions(S0, X0)
        
        # Create objective function
        objective = ObjectiveFunction(
            experimental_time=time,
            experimental_substrate=substrate,
            experimental_biomass=biomass,
            model_type=self.model_type,
            initial_conditions=initial_conditions,
            t_span=(time[0], time[-1]),
            parameter_names=self.parameter_names,
            oxygen_model=self.oxygen_model,
            normalize_errors=True  # Use normalized errors
        )
        
        # Create optimizer
        optimizer = ParameterOptimizer(
            parameter_names=self.parameter_names,
            bounds={p: self.config.bounds[p] for p in self.parameter_names},
            initial_guess={p: self.config.initial_guesses[p] for p in self.parameter_names},
            method=optimization_method,
            verbose=verbose
        )
        
        # Run optimization
        opt_result = optimizer.optimize(objective)
        
        # Generate predictions
        predictions = self._generate_condition_predictions(
            opt_result.parameters, time, initial_conditions
        )
        
        # Calculate statistics
        pred_substrate = predictions['Substrate'].values
        pred_biomass = predictions['Biomass'].values
        
        # Interpolate predictions to experimental time points
        pred_substrate_interp = np.interp(time, predictions['Time'].values, pred_substrate)
        pred_biomass_interp = np.interp(time, predictions['Time'].values, pred_biomass)
        
        stats = calculate_separate_statistics(
            observed_substrate=substrate,
            predicted_substrate=pred_substrate_interp,
            observed_biomass=biomass,
            predicted_biomass=pred_biomass_interp,
            n_parameters=len(self.parameter_names)
        )
        
        # Calculate confidence intervals
        confidence_intervals, ci_diagnostics = calculate_parameter_confidence_intervals_with_diagnostics(
            objective_function=objective,
            optimal_params=np.array([opt_result.parameters[p] for p in self.parameter_names]),
            parameter_names=self.parameter_names,
            n_observations=len(substrate) + len(biomass),
            confidence_level=self.ci_confidence_level,
            method=self.ci_method,
            bounds={p: self.config.bounds[p] for p in self.parameter_names},
            mcmc_samples=self.ci_mcmc_samples,
            mcmc_burn_in=self.ci_mcmc_burn_in,
            mcmc_step_scale=self.ci_mcmc_step_scale,
            mcmc_random_seed=self.ci_mcmc_seed,
            mcmc_adaptive=self.mcmc_adaptive,
        )
        
        # Residual analysis (combined normalized residuals)
        sub_range = np.ptp(substrate) or 1.0
        bio_range = np.ptp(biomass) or 1.0
        normalized_residuals = np.concatenate([
            (substrate - pred_substrate_interp) / sub_range,
            (biomass - pred_biomass_interp) / bio_range
        ])
        residual_diag = residual_analysis(
            np.zeros_like(normalized_residuals),
            -normalized_residuals
        )
        
        result = ConditionResult(
            condition=condition,
            parameters=opt_result.parameters,
            confidence_intervals=confidence_intervals,
            statistics=stats,
            residual_diagnostics=residual_diag,
            predictions=predictions,
            experimental_time=time,
            experimental_substrate=substrate,
            experimental_biomass=biomass,
            optimization_result=opt_result,
            success=opt_result.success,
            ci_diagnostics=ci_diagnostics,
        )
        
        if verbose:
            print(f"  R² (Substrate): {stats['substrate']['R_squared']:.4f}")
            print(f"  R² (Biomass):   {stats['biomass']['R_squared']:.4f}")
            print(f"  CI method:      {self.ci_method}")
        
        return result
    
    def _generate_condition_predictions(
        self,
        parameters: Dict[str, float],
        exp_time: np.ndarray,
        initial_conditions: List[float]
    ) -> pd.DataFrame:
        """Generate model predictions for a condition."""
        t_span = (exp_time[0], exp_time[-1])
        n_points = max(1000, len(exp_time) * 10)
        t_eval = np.linspace(t_span[0], t_span[1], n_points)
        
        ode_system = self.create_ode_system(parameters)
        result = solve_ode(
            ode_system=ode_system,
            initial_conditions=np.array(initial_conditions),
            t_span=t_span,
            t_eval=t_eval
        )
        
        if self.model_type in ('single_monod', 'single_monod_lag'):
            return pd.DataFrame({
                'Time': result.time,
                'Substrate': result.states.get('S', result.states.get('Substrate', np.zeros_like(result.time))),
                'Biomass': result.states.get('X', result.states.get('Biomass', np.zeros_like(result.time)))
            })
        else:
            return pd.DataFrame({
                'Time': result.time,
                'Substrate': result.states.get('S', result.states.get('Substrate', np.zeros_like(result.time))),
                'Biomass': result.states.get('X', result.states.get('Biomass', np.zeros_like(result.time))),
                'Oxygen': result.states.get('O2', result.states.get('Oxygen', np.zeros_like(result.time)))
            })
    
    def run(
        self,
        optimization_method: str = 'L-BFGS-B',
        save_results: bool = True,
        generate_plots: bool = True,
        verbose: bool = True
    ) -> IndividualConditionResult:
        """
        Run the individual condition fitting workflow.
        
        Args:
            optimization_method: 'L-BFGS-B' or 'differential_evolution'
            save_results: Whether to save results to disk
            generate_plots: Whether to generate figures
            verbose: Print progress
            
        Returns:
            IndividualConditionResult with all condition fits and summaries
        """
        conditions = self.experimental_data.conditions
        n_conditions = len(conditions)
        use_parallel = self.n_workers > 1 and n_conditions > 1
        effective_workers = min(self.n_workers, n_conditions) if use_parallel else 1

        if verbose:
            print("\n" + "=" * 70)
            print(f"INDIVIDUAL CONDITION FITTING - {self.display_name}")
            print(f"Substrate: {self.config.name}")
            print(f"Conditions: {conditions}")
            print(f"CI method: {self.ci_method} ({int(self.ci_confidence_level*100)}% interval)")
            if self.ci_method == 'mcmc':
                print(
                    "MCMC settings: "
                    f"samples={self.ci_mcmc_samples}, burn_in={self.ci_mcmc_burn_in}, "
                    f"step_scale={self.ci_mcmc_step_scale}"
                )
            print(f"Workers: {effective_workers}"
                  f" {'(parallel)' if use_parallel else '(sequential)'}")
            print("=" * 70)

        # Step 1: Fit each condition
        condition_results = {}
        if use_parallel:
            # Parallel execution via ProcessPoolExecutor
            future_to_cond = {}
            with ProcessPoolExecutor(max_workers=effective_workers) as executor:
                for condition in conditions:
                    future = executor.submit(
                        _fit_condition_worker,
                        self._init_kwargs,
                        condition,
                        optimization_method,
                    )
                    future_to_cond[future] = condition

                for i, future in enumerate(as_completed(future_to_cond), 1):
                    condition = future_to_cond[future]
                    cond_result = future.result()
                    condition_results[condition] = cond_result
                    if verbose:
                        stats = cond_result.statistics
                        status = "+" if cond_result.success else "x"
                        print(
                            f"  [{i}/{n_conditions}] {status} {condition}: "
                            f"R²(S)={stats['substrate']['R_squared']:.4f}, "
                            f"R²(X)={stats['biomass']['R_squared']:.4f}"
                        )
        else:
            # Sequential execution (original path)
            for condition in conditions:
                result = self.fit_condition(condition, optimization_method, verbose)
                condition_results[condition] = result
        
        # Step 2: Calculate parameter summary statistics
        parameter_summary = self._calculate_parameter_summary(condition_results)

        # Step 3: Calculate individual losses at each condition's optimum
        individual_losses = self._calculate_individual_losses(condition_results)

        # Step 4: Optimize global cost function across all conditions
        global_parameters, global_opt_result, global_loss, global_cis, global_ci_diagnostics = (
            self._calculate_global_parameters(condition_results, verbose=verbose)
        )

        # Step 5: Generate plots
        figures = []
        if generate_plots:
            if verbose:
                print("\nGenerating diagnostic plots...")
            figures = self._generate_all_plots(
                condition_results,
                parameter_summary,
                global_ci_diagnostics,
            )

        # Create result object
        result = IndividualConditionResult(
            model_type=self.model_type,
            condition_results=condition_results,
            parameter_summary=parameter_summary,
            global_parameters=global_parameters,
            global_optimization_result=global_opt_result,
            global_confidence_intervals=global_cis,
            global_ci_diagnostics=global_ci_diagnostics,
            global_loss=global_loss,
            individual_losses=individual_losses,
            figures=figures,
            config=self.config,
            display_name=self.display_name,
        )
        
        # Step 5: Save results
        if save_results:
            if verbose:
                print("\nSaving results...")
            self._save_results(result)
        
        if verbose:
            print(result.summary())
        
        return result
    
    def _calculate_parameter_summary(
        self,
        condition_results: Dict[str, ConditionResult]
    ) -> Dict[str, Dict[str, float]]:
        """Calculate summary statistics for each parameter across conditions."""
        summary = {}
        
        for param in self.parameter_names:
            values = [
                r.parameters[param] 
                for r in condition_results.values() 
                if r.success
            ]
            
            if len(values) > 0:
                mean_val = np.mean(values)
                std_val = np.std(values) if len(values) > 1 else 0.0
                cv = (std_val / mean_val * 100) if mean_val != 0 else 0.0
                
                summary[param] = {
                    'mean': mean_val,
                    'std': std_val,
                    'cv': cv,
                    'min': np.min(values),
                    'max': np.max(values),
                    'median': np.median(values),
                    'n': len(values)
                }
            else:
                summary[param] = {
                    'mean': np.nan,
                    'std': np.nan,
                    'cv': np.nan,
                    'min': np.nan,
                    'max': np.nan,
                    'median': np.nan,
                    'n': 0
                }
        
        return summary
    
    def _calculate_individual_losses(
        self,
        condition_results: Dict[str, ConditionResult]
    ) -> Dict[str, float]:
        """
        Calculate the local loss (objective function value) for each condition
        at its individual optimum.

        Returns:
            Dictionary mapping condition labels to their local loss values.
        """
        losses = {}
        for cond, result in condition_results.items():
            if result.success:
                # Reconstruct the objective and evaluate at the optimum
                time, substrate, biomass = self.experimental_data.get_condition_data(cond)
                S0, X0 = substrate[0], biomass[0]
                initial_conditions = self._get_initial_conditions(S0, X0)

                objective = ObjectiveFunction(
                    experimental_time=time,
                    experimental_substrate=substrate,
                    experimental_biomass=biomass,
                    model_type=self.model_type,
                    initial_conditions=initial_conditions,
                    t_span=(time[0], time[-1]),
                    parameter_names=self.parameter_names,
                    oxygen_model=self.oxygen_model,
                    normalize_errors=True
                )

                param_array = np.array([result.parameters[p] for p in self.parameter_names])
                losses[cond] = float(objective(param_array))
            else:
                losses[cond] = float('inf')
        return losses

    def _calculate_global_parameters(
        self,
        condition_results: Dict[str, ConditionResult],
        verbose: bool = True
    ) -> Tuple[Dict[str, float], OptimizationResult, float,
               Optional[Dict[str, Dict[str, float]]], Optional[Dict[str, Any]]]:
        """
        Calculate global parameters by minimizing a global cost function,
        then compute Hessian-based 95% confidence intervals on the result.

        Strategy (Two-Stage Global Estimation):
            Stage 1 (already complete): Each condition was fitted independently,
                producing per-condition parameter estimates theta_i*.

            Stage 2 (this method): A global cost function is constructed:

                J(theta) = sum_{i=1}^{N} L_i(theta)

            where L_i(theta) is the normalized SSE for condition i evaluated
            at the shared parameter vector theta. "Normalized" means substrate
            and biomass residuals are divided by their respective data ranges,
            making the loss dimensionless and ensuring equal contribution.

            The median of individual estimates {theta_i*} is used as the
            initial guess for global optimization (median is robust to outliers
            from poorly identified conditions). Alternatively, the parameters
            from the best-fitting individual condition (by mean R²) can be used
            — controlled by self.global_guess_strategy ('median' or 'best_r2').
            The optimizer then finds the
            single parameter set that best explains ALL conditions simultaneously.

        Confidence Intervals:
            After optimization, the Hessian of J(theta) is computed at the global
            optimum via central finite differences. The covariance matrix is:

                Cov(theta) = sigma^2 * inv(H/2)

            where sigma^2 = J(theta*) / (n_total - p), n_total is the total
            number of (substrate + biomass) observations across all conditions,
            and p is the number of parameters. CIs use the t-distribution with
            (n_total - p) degrees of freedom.

            This is valid because J(theta) is a sum of squared normalized
            residuals — a standard nonlinear least squares objective — so the
            Hessian at the minimum approximates the Fisher information matrix.

        Returns:
            Tuple of (global_parameters_dict, optimization_result, global_loss,
                      global_confidence_intervals, global_ci_diagnostics)
        """
        # Build condition data list for GlobalObjectiveFunction
        conditions_data = []
        n_total_observations = 0
        for cond in self.experimental_data.conditions:
            result = condition_results.get(cond)
            if result is None or not result.success:
                continue

            time, substrate, biomass = self.experimental_data.get_condition_data(cond)
            S0, X0 = substrate[0], biomass[0]
            initial_conditions = self._get_initial_conditions(S0, X0)

            conditions_data.append({
                'time': time,
                'substrate': substrate,
                'biomass': biomass,
                'initial_conditions': initial_conditions,
                't_span': (time[0], time[-1]),
                'label': cond
            })
            # Each condition contributes len(substrate) + len(biomass) observations
            n_total_observations += len(substrate) + len(biomass)

        if not conditions_data:
            # Fallback: no successful fits, return simple average
            global_params = {}
            for param in self.parameter_names:
                values = [r.parameters[param] for r in condition_results.values()]
                global_params[param] = float(np.mean(values))
            return global_params, None, float('inf'), None, None

        # Create global objective function
        global_objective = GlobalObjectiveFunction(
            conditions=conditions_data,
            model_type=self.model_type,
            parameter_names=self.parameter_names,
            oxygen_model=self.oxygen_model,
            normalize_errors=True
        )

        # --- Build initial guess from individual fits ---
        successful = {
            cond: cr for cond, cr in condition_results.items() if cr.success
        }

        if self.global_guess_strategy == 'best_r2':
            # Pick parameters from the condition with the highest mean R²
            best_cond, best_r2 = None, -np.inf
            for cond, cr in successful.items():
                r2_s = cr.statistics.get('substrate', {}).get('R_squared', -np.inf)
                r2_x = cr.statistics.get('biomass', {}).get('R_squared', -np.inf)
                mean_r2 = (r2_s + r2_x) / 2.0
                if mean_r2 > best_r2:
                    best_r2, best_cond = mean_r2, cond

            initial_guess = dict(successful[best_cond].parameters)
            guess_label = f"best individual fit (condition={best_cond}, mean R²={best_r2:.4f})"
        else:
            # Default: median — robust to outliers from poorly identified conditions
            initial_guess = {}
            for param in self.parameter_names:
                values = [successful[c].parameters[param] for c in successful]
                initial_guess[param] = float(np.median(values))
            guess_label = "median of individual fits"

        if verbose:
            print("\nGlobal optimization: minimizing sum of per-condition losses...")
            print(f"  Strategy: {guess_label}")
            print(f"  Initial guess: {initial_guess}")

        # Optimize the global cost function
        optimizer = ParameterOptimizer(
            parameter_names=self.parameter_names,
            bounds={p: self.config.bounds[p] for p in self.parameter_names},
            initial_guess=initial_guess,
            method='L-BFGS-B',
            verbose=verbose
        )

        opt_result = optimizer.optimize(global_objective)
        global_loss = float(global_objective.best_error)

        if verbose:
            print(f"  Global optimization converged: {opt_result.success}")
            print(f"  Global cost function value: {global_loss:.6f}")

        # Compute selected confidence intervals on the global parameters
        optimal_params = np.array([
            opt_result.parameters[p] for p in self.parameter_names
        ])
        global_cis, global_ci_diagnostics = calculate_parameter_confidence_intervals_with_diagnostics(
            objective_function=global_objective,
            optimal_params=optimal_params,
            parameter_names=self.parameter_names,
            n_observations=n_total_observations,
            confidence_level=self.ci_confidence_level,
            method=self.ci_method,
            bounds={p: self.config.bounds[p] for p in self.parameter_names},
            mcmc_samples=self.ci_mcmc_samples,
            mcmc_burn_in=self.ci_mcmc_burn_in,
            mcmc_step_scale=self.ci_mcmc_step_scale,
            mcmc_random_seed=self.ci_mcmc_seed,
            mcmc_adaptive=self.mcmc_adaptive,
        )

        if verbose:
            print(
                f"  Global parameter {int(self.ci_confidence_level*100)}% intervals "
                f"computed ({self.ci_method})"
            )

        return opt_result.parameters, opt_result, global_loss, global_cis, global_ci_diagnostics
    
    def _generate_all_plots(
        self,
        condition_results: Dict[str, ConditionResult],
        parameter_summary: Dict[str, Dict[str, float]],
        global_ci_diagnostics: Optional[Dict[str, Any]] = None,
    ) -> List[Path]:
        """Generate all diagnostic plots."""
        figures = []
        output_dir = self.results_writer.output_dir

        # 1. Substrate summary (all conditions on one graph)
        fig_sub = self._plot_substrate_summary(condition_results)
        path = output_dir / 'substrate_summary.png'
        fig_sub.savefig(path, dpi=300, bbox_inches='tight')
        fig_sub.savefig(path.with_suffix('.pdf'), bbox_inches='tight')
        figures.append(path)
        plt.close(fig_sub)

        # 2. Biomass summary (all conditions on one graph)
        fig_bio = self._plot_biomass_summary(condition_results)
        path = output_dir / 'biomass_summary.png'
        fig_bio.savefig(path, dpi=300, bbox_inches='tight')
        fig_bio.savefig(path.with_suffix('.pdf'), bbox_inches='tight')
        figures.append(path)
        plt.close(fig_bio)

        # 3. Parameter comparison across conditions
        fig_params = self._plot_parameter_comparison(condition_results)
        path = output_dir / 'parameter_comparison.png'
        fig_params.savefig(path, dpi=300, bbox_inches='tight')
        fig_params.savefig(path.with_suffix('.pdf'), bbox_inches='tight')
        figures.append(path)
        plt.close(fig_params)

        # 4. Residual diagnostics
        fig_resid = self._plot_residual_diagnostics(condition_results)
        path = output_dir / 'residual_diagnostics.png'
        fig_resid.savefig(path, dpi=300, bbox_inches='tight')
        fig_resid.savefig(path.with_suffix('.pdf'), bbox_inches='tight')
        figures.append(path)
        plt.close(fig_resid)

        # 5. Confidence interval plot
        fig_ci = self._plot_confidence_intervals(condition_results)
        path = output_dir / 'confidence_intervals.png'
        fig_ci.savefig(path, dpi=300, bbox_inches='tight')
        fig_ci.savefig(path.with_suffix('.pdf'), bbox_inches='tight')
        figures.append(path)
        plt.close(fig_ci)

        # 6. Goodness of fit summary
        fig_gof = self._plot_goodness_of_fit(condition_results)
        path = output_dir / 'goodness_of_fit.png'
        fig_gof.savefig(path, dpi=300, bbox_inches='tight')
        fig_gof.savefig(path.with_suffix('.pdf'), bbox_inches='tight')
        figures.append(path)
        plt.close(fig_gof)

        # 7. CI diagnostics (method-specific)
        fig_ci_diag = self._plot_ci_diagnostics(condition_results, global_ci_diagnostics)
        path = output_dir / 'ci_diagnostics.png'
        fig_ci_diag.savefig(path, dpi=300, bbox_inches='tight')
        fig_ci_diag.savefig(path.with_suffix('.pdf'), bbox_inches='tight')
        figures.append(path)
        plt.close(fig_ci_diag)

        if self.ci_method == 'mcmc':
            fig_trace = self._plot_mcmc_trace(global_ci_diagnostics)
            if fig_trace is not None:
                path = output_dir / 'mcmc_trace_plots.png'
                fig_trace.savefig(path, dpi=300, bbox_inches='tight')
                fig_trace.savefig(path.with_suffix('.pdf'), bbox_inches='tight')
                figures.append(path)
                plt.close(fig_trace)

        return figures
    
    def _plot_substrate_summary(
        self,
        condition_results: Dict[str, ConditionResult]
    ) -> plt.Figure:
        """
        Plot substrate consumption summary with all conditions on one graph.

        Shows experimental data (scatter) and model predictions (lines)
        for every concentration overlaid on a single axis.
        """
        fig, ax = plt.subplots(figsize=(10, 7), dpi=300)

        for idx, (condition, result) in enumerate(condition_results.items()):
            color = COLOR_LIST[idx % len(COLOR_LIST)]
            r2 = result.statistics['substrate']['R_squared']

            # Experimental data
            ax.scatter(
                result.experimental_time,
                result.experimental_substrate,
                color=color, s=60, alpha=0.7,
                marker='o', edgecolors='white', linewidths=0.8,
                label=f'{condition} (exp)', zorder=3
            )
            # Model prediction
            ax.plot(
                result.predictions['Time'],
                result.predictions['Substrate'],
                color=color, linewidth=2, linestyle='-',
                label=f'{condition} (model, R²={r2:.3f})', zorder=2
            )

        ax.set_xlabel('Time (days)', fontsize=12)
        ax.set_ylabel(f'{self.config.name} Concentration (mg/L)', fontsize=12)
        ax.set_title(
            f'Substrate Consumption - {self.config.name}\n'
            f'{self.display_name} Model',
            fontsize=14, fontweight='bold'
        )
        ax.legend(loc='best', fontsize=9, frameon=True, fancybox=True, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        fig.tight_layout()

        return fig

    def _plot_biomass_summary(
        self,
        condition_results: Dict[str, ConditionResult]
    ) -> plt.Figure:
        """
        Plot biomass growth summary with all conditions on one graph.

        Shows experimental data (scatter) and model predictions (lines)
        for every concentration overlaid on a single axis.
        """
        fig, ax = plt.subplots(figsize=(10, 7), dpi=300)

        for idx, (condition, result) in enumerate(condition_results.items()):
            color = COLOR_LIST[idx % len(COLOR_LIST)]
            r2 = result.statistics['biomass']['R_squared']

            # Experimental data
            ax.scatter(
                result.experimental_time,
                result.experimental_biomass,
                color=color, s=60, alpha=0.7,
                marker='o', edgecolors='white', linewidths=0.8,
                label=f'{condition} (exp)', zorder=3
            )
            # Model prediction
            ax.plot(
                result.predictions['Time'],
                result.predictions['Biomass'],
                color=color, linewidth=2, linestyle='-',
                label=f'{condition} (model, R²={r2:.3f})', zorder=2
            )

        ax.set_xlabel('Time (days)', fontsize=12)
        ax.set_ylabel('Biomass Concentration (mg cells/L)', fontsize=12)
        ax.set_title(
            f'Biomass Growth - {self.config.name}\n'
            f'{self.display_name} Model',
            fontsize=14, fontweight='bold'
        )
        ax.legend(loc='best', fontsize=9, frameon=True, fancybox=True, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        fig.tight_layout()

        return fig
    
    def _plot_parameter_comparison(
        self,
        condition_results: Dict[str, ConditionResult]
    ) -> plt.Figure:
        """Plot parameter values across conditions with error bars."""
        
        n_params = len(self.parameter_names)
        conditions = list(condition_results.keys())
        n_conds = len(conditions)
        
        # Calculate grid dimensions
        ncols = min(3, n_params)
        nrows = (n_params + ncols - 1) // ncols
        
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
        axes = np.atleast_2d(axes).flatten()
        
        x = np.arange(n_conds)
        
        for idx, param in enumerate(self.parameter_names):
            ax = axes[idx]
            
            values = [condition_results[c].parameters[param] for c in conditions]
            errors = [
                condition_results[c].confidence_intervals.get(param, {}).get('std_error', 0)
                for c in conditions
            ]
            
            colors = [COLOR_LIST[i % len(COLOR_LIST)] for i in range(n_conds)]
            
            bars = ax.bar(x, values, yerr=errors, capsize=5, color=colors, alpha=0.8,
                         edgecolor='black', linewidth=1)
            
            ax.set_xlabel('Condition')
            ax.set_ylabel(param)
            ax.set_title(f'{param}')
            ax.set_xticks(x)
            ax.set_xticklabels(conditions, rotation=45, ha='right')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add mean line
            mean_val = np.mean(values)
            ax.axhline(mean_val, color='red', linestyle='--', linewidth=2,
                      label=f'Mean: {mean_val:.4f}')
            ax.legend(fontsize=8)
        
        # Hide unused axes
        for idx in range(n_params, len(axes)):
            axes[idx].set_visible(False)
        
        fig.suptitle('Parameter Comparison Across Conditions',
                    fontsize=14, fontweight='bold', y=1.02)
        fig.tight_layout()
        
        return fig
    
    def _plot_residual_diagnostics(
        self,
        condition_results: Dict[str, ConditionResult]
    ) -> plt.Figure:
        """Plot residual diagnostics for each condition."""
        
        n_conditions = len(condition_results)
        fig, axes = plt.subplots(n_conditions, 2, figsize=(12, 3 * n_conditions))
        
        if n_conditions == 1:
            axes = axes.reshape(1, -1)
        
        for idx, (condition, result) in enumerate(condition_results.items()):
            color = COLOR_LIST[idx % len(COLOR_LIST)]
            
            # Get predictions interpolated to experimental times
            pred_sub = np.interp(
                result.experimental_time,
                result.predictions['Time'].values,
                result.predictions['Substrate'].values
            )
            pred_bio = np.interp(
                result.experimental_time,
                result.predictions['Time'].values,
                result.predictions['Biomass'].values
            )
            
            # Residuals
            resid_sub = result.experimental_substrate - pred_sub
            resid_bio = result.experimental_biomass - pred_bio
            
            # Subplot 1: Residuals vs Time
            ax1 = axes[idx, 0]
            ax1.scatter(result.experimental_time, resid_sub, 
                       color=COLORS['blue'], s=40, alpha=0.7, label='Substrate')
            ax1.scatter(result.experimental_time, resid_bio * (np.ptp(resid_sub) / np.ptp(resid_bio) if np.ptp(resid_bio) > 0 else 1),
                       color=COLORS['orange'], s=40, alpha=0.7, label='Biomass (scaled)')
            ax1.axhline(0, color='black', linestyle='--', linewidth=1)
            ax1.set_xlabel('Time (days)')
            ax1.set_ylabel('Residual')
            ax1.set_title(f'{condition} - Residuals vs Time')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Subplot 2: Residual histogram
            ax2 = axes[idx, 1]
            # Normalize residuals
            sub_norm = resid_sub / (np.ptp(result.experimental_substrate) or 1)
            bio_norm = resid_bio / (np.ptp(result.experimental_biomass) or 1)
            combined = np.concatenate([sub_norm, bio_norm])
            
            ax2.hist(combined, bins=15, color=color, alpha=0.7, edgecolor='black')
            ax2.axvline(0, color='red', linestyle='--', linewidth=2)
            ax2.set_xlabel('Normalized Residual')
            ax2.set_ylabel('Frequency')
            ax2.set_title(f'{condition} - Residual Distribution\n'
                         f'Mean: {np.mean(combined):.4f}, Std: {np.std(combined):.4f}')
            ax2.grid(True, alpha=0.3)
        
        fig.suptitle('Residual Diagnostics',
                    fontsize=14, fontweight='bold', y=1.02)
        fig.tight_layout()
        
        return fig
    
    def _plot_confidence_intervals(
        self,
        condition_results: Dict[str, ConditionResult]
    ) -> plt.Figure:
        """Plot 95% confidence intervals for each parameter."""
        
        conditions = list(condition_results.keys())
        n_params = len(self.parameter_names)
        
        fig, axes = plt.subplots(1, n_params, figsize=(4 * n_params, 5))
        if n_params == 1:
            axes = [axes]
        
        for idx, param in enumerate(self.parameter_names):
            ax = axes[idx]
            
            y_positions = np.arange(len(conditions))
            
            for i, cond in enumerate(conditions):
                result = condition_results[cond]
                value = result.parameters[param]
                ci = result.confidence_intervals.get(param, {})
                
                color = COLOR_LIST[i % len(COLOR_LIST)]
                
                if ci and not np.isnan(ci.get('ci_lower', np.nan)):
                    ci_lower = float(ci['ci_lower'])
                    ci_upper = float(ci['ci_upper'])

                    # Guard against inverted intervals from numerical issues
                    if ci_lower > ci_upper:
                        ci_lower, ci_upper = ci_upper, ci_lower

                    # For MCMC percentile intervals, the optimizer point estimate can
                    # lie outside [ci_lower, ci_upper]. Matplotlib requires nonnegative
                    # xerr arms, so anchor the error bar at a point inside the interval.
                    center = min(max(float(value), ci_lower), ci_upper)
                    xerr = [[center - ci_lower], [ci_upper - center]]

                    ax.errorbar(center, i, xerr=xerr, fmt='o', color=color,
                              capsize=5, markersize=8, capthick=2)
                else:
                    ax.scatter(value, i, color=color, s=80, zorder=3)
            
            ax.set_yticks(y_positions)
            ax.set_yticklabels(conditions)
            ax.set_xlabel(param)
            ax.set_title(f'{param}\n95% Confidence Intervals')
            ax.grid(True, alpha=0.3, axis='x')
            ax.invert_yaxis()
        
        fig.suptitle('Parameter Confidence Intervals by Condition',
                    fontsize=14, fontweight='bold', y=1.02)
        fig.tight_layout()
        
        return fig

    def _plot_ci_diagnostics(
        self,
        condition_results: Dict[str, ConditionResult],
        global_ci_diagnostics: Optional[Dict[str, Any]] = None,
    ) -> plt.Figure:
        """Plot diagnostics specific to the chosen CI estimation method."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        conditions = list(condition_results.keys())

        if self.ci_method in {'hessian', 'hessian_log'}:
            cond_numbers = []
            rel_errors = []

            for cond in conditions:
                diag = condition_results[cond].ci_diagnostics or {}
                cond_numbers.append(diag.get('hessian_condition_number', np.nan))

                per_param_rel = [
                    ci.get('relative_error_pct', np.nan)
                    for ci in condition_results[cond].confidence_intervals.values()
                ]
                rel_errors.append(np.nanmedian(per_param_rel) if len(per_param_rel) else np.nan)

            # Condition number panel
            ax = axes[0]
            x = np.arange(len(conditions))
            ax.bar(x, cond_numbers, color=COLORS['purple'], alpha=0.8)
            ax.set_yscale('log')
            ax.set_xticks(x)
            ax.set_xticklabels(conditions, rotation=45, ha='right')
            ax.set_ylabel('Hessian Condition Number (log scale)')
            ax.set_title('Numerical Conditioning of CI Estimation')
            ax.grid(True, alpha=0.3, axis='y')

            if global_ci_diagnostics is not None:
                g_cn = global_ci_diagnostics.get('hessian_condition_number', np.nan)
                if np.isfinite(g_cn):
                    ax.axhline(g_cn, color='red', linestyle='--', linewidth=1.5,
                               label=f'Global: {g_cn:.2e}')
                    ax.legend()

            # Relative error panel
            ax = axes[1]
            ax.bar(x, rel_errors, color=COLORS['blue'], alpha=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(conditions, rotation=45, ha='right')
            ax.set_ylabel('Median Relative SE (%)')
            ax.set_title('CI Width Diagnostic')
            ax.grid(True, alpha=0.3, axis='y')

            method_name = 'Hessian (log-transform)' if self.ci_method == 'hessian_log' else 'Hessian'
            fig.suptitle(f'CI Diagnostics — {method_name}', fontsize=14, fontweight='bold', y=1.02)

        else:
            # MCMC diagnostics panel: acceptance, R-hat, ESS
            ax = axes[0]
            if global_ci_diagnostics:
                acc_chain = global_ci_diagnostics.get('acceptance_rate_per_chain', [])
                if acc_chain:
                    ax.bar(np.arange(len(acc_chain)), acc_chain,
                           color=COLORS['green'], alpha=0.8)
                    ax.axhline(0.2, color='orange', linestyle='--', linewidth=1, label='0.2')
                    ax.axhline(0.5, color='green', linestyle='--', linewidth=1, label='0.5')
                    ax.axhline(0.8, color='orange', linestyle='--', linewidth=1, label='0.8')
                    ax.set_ylim(0, 1)
                    ax.set_xlabel('Chain index')
                    ax.set_ylabel('Acceptance rate')
                    ax.set_title('MCMC Acceptance by Chain')
                    ax.legend(loc='lower right', fontsize=8)
                else:
                    ax.text(0.5, 0.5, 'No chain acceptance diagnostics available',
                            ha='center', va='center', transform=ax.transAxes)
            else:
                ax.text(0.5, 0.5, 'No global MCMC diagnostics available',
                        ha='center', va='center', transform=ax.transAxes)

            ax.grid(True, alpha=0.3, axis='y')

            ax = axes[1]
            if global_ci_diagnostics:
                rhat = global_ci_diagnostics.get('r_hat', {})
                ess = global_ci_diagnostics.get('effective_sample_size', {})
                params = [p for p in self.parameter_names if p in rhat]

                if params:
                    x = np.arange(len(params))
                    rvals = [rhat.get(p, np.nan) for p in params]
                    evals = [ess.get(p, np.nan) for p in params]

                    ax.plot(x, rvals, 'o-', color=COLORS['red'], label='R-hat')
                    ax.axhline(1.01, color='black', linestyle='--', linewidth=1, label='R-hat=1.01')
                    ax.set_xticks(x)
                    ax.set_xticklabels(params, rotation=45, ha='right')
                    ax.set_ylabel('R-hat')
                    ax.set_title('MCMC Convergence (Global)')

                    ax2 = ax.twinx()
                    ax2.bar(x, evals, alpha=0.25, color=COLORS['blue'], label='ESS')
                    ax2.set_ylabel('Effective Sample Size')

                    lines1, labels1 = ax.get_legend_handles_labels()
                    lines2, labels2 = ax2.get_legend_handles_labels()
                    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=8)
                else:
                    ax.text(0.5, 0.5, 'No R-hat / ESS diagnostics available',
                            ha='center', va='center', transform=ax.transAxes)
            else:
                ax.text(0.5, 0.5, 'No R-hat / ESS diagnostics available',
                        ha='center', va='center', transform=ax.transAxes)

            ax.grid(True, alpha=0.3, axis='y')
            fig.suptitle('CI Diagnostics — MCMC', fontsize=14, fontweight='bold', y=1.02)

        fig.tight_layout()
        return fig

    def _plot_mcmc_trace(
        self,
        global_ci_diagnostics: Optional[Dict[str, Any]] = None,
    ) -> Optional[plt.Figure]:
        """Plot MCMC trace plots for global parameter inference."""
        if not global_ci_diagnostics:
            return None

        trace_samples = global_ci_diagnostics.get('trace_samples', {})
        if not trace_samples:
            return None

        params = [p for p in self.parameter_names if p in trace_samples]
        if not params:
            return None

        n_params = len(params)
        fig, axes = plt.subplots(n_params, 1, figsize=(10, 2.5 * n_params), sharex=True)
        if n_params == 1:
            axes = [axes]

        for idx, p in enumerate(params):
            ax = axes[idx]
            chains = np.asarray(trace_samples[p], dtype=float)
            if chains.ndim != 2:
                ax.text(0.5, 0.5, f'No trace for {p}', ha='center', va='center', transform=ax.transAxes)
                continue

            for c in range(chains.shape[0]):
                ax.plot(chains[c], linewidth=0.8, alpha=0.8, label=f'Chain {c+1}')

            ax.set_ylabel(p)
            if idx == 0:
                ax.legend(loc='upper right', ncol=min(chains.shape[0], 4), fontsize=8)
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel('Iteration (post burn-in)')
        fig.suptitle('MCMC Trace Plots (Global Parameters)', fontsize=14, fontweight='bold', y=1.02)
        fig.tight_layout()
        return fig
    
    def _plot_goodness_of_fit(
        self,
        condition_results: Dict[str, ConditionResult]
    ) -> plt.Figure:
        """Plot goodness of fit metrics summary."""
        
        conditions = list(condition_results.keys())
        n_conds = len(conditions)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        x = np.arange(n_conds)
        width = 0.35
        
        # R² values
        ax1 = axes[0]
        r2_sub = [condition_results[c].statistics['substrate']['R_squared'] for c in conditions]
        r2_bio = [condition_results[c].statistics['biomass']['R_squared'] for c in conditions]
        
        bars1 = ax1.bar(x - width/2, r2_sub, width, label='Substrate', color=COLORS['blue'], alpha=0.8)
        bars2 = ax1.bar(x + width/2, r2_bio, width, label='Biomass', color=COLORS['orange'], alpha=0.8)
        
        ax1.axhline(0.9, color='green', linestyle='--', linewidth=1.5, label='Good fit (0.9)')
        ax1.set_xlabel('Condition')
        ax1.set_ylabel('R²')
        ax1.set_title('Coefficient of Determination (R²)')
        ax1.set_xticks(x)
        ax1.set_xticklabels(conditions, rotation=45, ha='right')
        ax1.legend()
        ax1.set_ylim(0, 1.05)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # NRMSE values
        ax2 = axes[1]
        nrmse_sub = [condition_results[c].statistics['substrate']['NRMSE'] for c in conditions]
        nrmse_bio = [condition_results[c].statistics['biomass']['NRMSE'] for c in conditions]
        
        bars3 = ax2.bar(x - width/2, nrmse_sub, width, label='Substrate', color=COLORS['blue'], alpha=0.8)
        bars4 = ax2.bar(x + width/2, nrmse_bio, width, label='Biomass', color=COLORS['orange'], alpha=0.8)
        
        ax2.axhline(0.1, color='green', linestyle='--', linewidth=1.5, label='Good fit (0.1)')
        ax2.set_xlabel('Condition')
        ax2.set_ylabel('NRMSE')
        ax2.set_title('Normalized RMSE')
        ax2.set_xticks(x)
        ax2.set_xticklabels(conditions, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Combined heatmap
        ax3 = axes[2]
        data = np.array([
            [condition_results[c].statistics['substrate']['R_squared'] for c in conditions],
            [condition_results[c].statistics['biomass']['R_squared'] for c in conditions],
            [1 - condition_results[c].statistics['substrate']['NRMSE'] for c in conditions],
            [1 - condition_results[c].statistics['biomass']['NRMSE'] for c in conditions],
        ])
        
        im = ax3.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        ax3.set_xticks(np.arange(n_conds))
        ax3.set_yticks(np.arange(4))
        ax3.set_xticklabels(conditions, rotation=45, ha='right')
        ax3.set_yticklabels(['R² Sub', 'R² Bio', '1-NRMSE Sub', '1-NRMSE Bio'])
        ax3.set_title('Fit Quality Heatmap\n(Green = Good)')
        
        # Add text annotations
        for i in range(4):
            for j in range(n_conds):
                ax3.text(j, i, f'{data[i, j]:.2f}', ha='center', va='center',
                        color='white' if data[i, j] < 0.5 else 'black', fontsize=9)
        
        fig.colorbar(im, ax=ax3, label='Quality Score')
        
        fig.suptitle('Goodness of Fit Summary',
                    fontsize=14, fontweight='bold', y=1.02)
        fig.tight_layout()
        
        return fig
    
    def _save_results(self, result: IndividualConditionResult):
        """Save all results to files."""
        output_dir = self.results_writer.output_dir

        def _serialize_ci_diag(diag: Optional[Dict[str, Any]], include_trace: bool = False) -> Optional[Dict[str, Any]]:
            if diag is None:
                return None
            out = {}
            for k, v in diag.items():
                if k == 'trace_samples' and not include_trace:
                    continue
                if isinstance(v, np.ndarray):
                    out[k] = v.tolist()
                elif isinstance(v, dict):
                    out[k] = {
                        kk: (float(vv) if isinstance(vv, (int, float, np.floating)) and np.isfinite(vv) else vv)
                        for kk, vv in v.items()
                    }
                elif isinstance(v, (int, float, np.floating)):
                    out[k] = float(v) if np.isfinite(v) else None
                else:
                    out[k] = v
            return out
        
        # Save DataFrame
        df = result.to_dataframe()
        df.to_csv(output_dir / 'individual_condition_results.csv', index=False)
        
        # Save detailed JSON
        results_dict = {
            'model_type': result.model_type,
            'display_name': result.display_name or self.display_name,
            'substrate': self.config.name,
            'config_path': self.config_path,
            'data_path': self.data_path,
            'ci_method': self.ci_method,
            'timestamp': datetime.now().isoformat(),
            'conditions': {},
            'individual_losses': result.individual_losses,
            'parameter_summary': result.parameter_summary,
            'global_parameters': result.global_parameters,
            'global_confidence_intervals': (
                {
                    k: {kk: float(vv) if not np.isnan(vv) else None
                        for kk, vv in v.items()}
                    for k, v in result.global_confidence_intervals.items()
                } if result.global_confidence_intervals else None
            ),
            'global_loss': result.global_loss,
            'global_ci_diagnostics': _serialize_ci_diag(result.global_ci_diagnostics, include_trace=False),
            'global_optimization_converged': (
                result.global_optimization_result.success
                if result.global_optimization_result else None
            ),
            'global_fitting_strategy': (
                'Two-stage: (1) independent per-condition fitting with local loss '
                'functions, (2) global cost function J=sum(L_i) minimized using '
                'median of individual fits as initial guess via L-BFGS-B. '
                'CIs computed via Hessian of J at global optimum.'
            )
        }
        
        for cond, cond_result in result.condition_results.items():
            results_dict['conditions'][cond] = {
                'parameters': cond_result.parameters,
                'confidence_intervals': {
                    k: {kk: float(vv) if not np.isnan(vv) else None for kk, vv in v.items()}
                    for k, v in cond_result.confidence_intervals.items()
                },
                'statistics': {
                    'substrate': cond_result.statistics['substrate'],
                    'biomass': cond_result.statistics['biomass'],
                    'combined': cond_result.statistics['combined']
                },
                'residual_diagnostics': {
                    k: float(v) if isinstance(v, (int, float, np.floating)) else v
                    for k, v in cond_result.residual_diagnostics.items()
                },
                'ci_diagnostics': _serialize_ci_diag(cond_result.ci_diagnostics, include_trace=False),
                'success': cond_result.success
            }
        
        with open(output_dir / 'individual_condition_results.json', 'w') as f:
            json.dump(results_dict, f, indent=2, default=str)
        
        # Save summary text
        with open(output_dir / 'individual_condition_summary.txt', 'w') as f:
            f.write(result.summary())

        # Save full global MCMC traces separately when available
        if result.global_ci_diagnostics and 'trace_samples' in result.global_ci_diagnostics:
            with open(output_dir / 'global_mcmc_traces.json', 'w') as f:
                json.dump(
                    _serialize_ci_diag(result.global_ci_diagnostics, include_trace=True),
                    f,
                    indent=2,
                    default=str,
                )

        # Generate PDF report
        try:
            figure_paths = result.figures or []
            # Also collect any PNG files saved directly to output_dir
            figure_paths = list(set(
                list(figure_paths) + list(output_dir.glob('*.png'))
            ))
            pdf_path = generate_individual_condition_report(
                output_dir=output_dir,
                summary_text=result.summary(),
                model_type=result.display_name or self.display_name,
                substrate_name=self.config.name,
                condition_results=result.condition_results,
                parameter_summary=result.parameter_summary,
                global_parameters=result.global_parameters,
                global_loss=result.global_loss,
                figure_paths=figure_paths,
            )
            print(f"  PDF report saved to: {pdf_path}")
        except Exception as e:
            print(f"  Warning: PDF report generation failed: {e}")


def run_individual_condition_fit(
    config: SubstrateConfig,
    experimental_data: ExperimentalData,
    model_type: str = 'single_monod',
    output_dir: str = 'results',
    optimization_method: str = 'L-BFGS-B',
    ci_method: str = 'hessian',
    ci_confidence_level: float = 0.95,
    ci_mcmc_samples: int = 4000,
    ci_mcmc_burn_in: int = 1000,
    ci_mcmc_step_scale: float = 0.05,
    ci_mcmc_seed: Optional[int] = None,
    config_path: Optional[str] = None,
    data_path: Optional[str] = None,
    no_inhibition: bool = False,
    verbose: bool = True
) -> IndividualConditionResult:
    """
    Convenience function to run individual condition fitting.

    Args:
        config: Substrate configuration
        experimental_data: Loaded experimental data
        model_type: 'single_monod', 'dual_monod', or 'dual_monod_lag'
        output_dir: Directory for output files
        optimization_method: Optimization algorithm
        ci_method: Confidence interval method ('hessian', 'hessian_log', 'mcmc')
        ci_confidence_level: Confidence level for intervals
        ci_mcmc_samples: Number of MCMC samples
        ci_mcmc_burn_in: Number of MCMC burn-in iterations
        ci_mcmc_step_scale: MCMC proposal scale
        ci_mcmc_seed: Optional MCMC random seed
        no_inhibition: If True, exclude Ki (basic Monod instead of Haldane)
        verbose: Print progress

    Returns:
        IndividualConditionResult with all fits and summaries
    """
    workflow = IndividualConditionWorkflow(
        config=config,
        experimental_data=experimental_data,
        model_type=model_type,
        output_dir=output_dir,
        ci_method=ci_method,
        ci_confidence_level=ci_confidence_level,
        ci_mcmc_samples=ci_mcmc_samples,
        ci_mcmc_burn_in=ci_mcmc_burn_in,
        ci_mcmc_step_scale=ci_mcmc_step_scale,
        ci_mcmc_seed=ci_mcmc_seed,
        config_path=config_path,
        data_path=data_path,
        no_inhibition=no_inhibition,
    )
    
    return workflow.run(
        optimization_method=optimization_method,
        save_results=True,
        generate_plots=True,
        verbose=verbose
    )
