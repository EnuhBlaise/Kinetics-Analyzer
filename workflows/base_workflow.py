"""
Base workflow class for kinetic parameter estimation.

This module defines the abstract base class that all workflow
implementations inherit from, ensuring consistent interfaces
and behavior across different model types.
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path

from src.core.ode_systems import BaseODE
from src.core.solvers import solve_ode, SimulationResult
from src.core.oxygen import OxygenModel
from src.fitting.objective import ObjectiveFunction, GlobalObjectiveFunction
from src.fitting.optimizer import ParameterOptimizer, OptimizationResult, compute_fit_statistics
from src.fitting.statistics import calculate_all_statistics, calculate_separate_statistics, compare_models, calculate_parameter_confidence_intervals, format_confidence_intervals
from src.io.data_loader import ExperimentalData
from src.io.config_loader import SubstrateConfig
from src.io.results_writer import ResultsWriter, FittedParameters
from src.io.pdf_report import generate_workflow_report
from src.utils.plotting import plot_fit_results, save_figure


@dataclass
class WorkflowResult:
    """
    Container for workflow execution results.

    Attributes:
        model_type: Type of model used
        optimization_result: Results from parameter optimization
        predictions: Model predictions as DataFrame
        experimental_data: Original experimental data
        statistics: Fit statistics
        figures: List of generated figure paths
        conditions: Experimental conditions fitted
        config: Configuration used
    """
    model_type: str
    optimization_result: OptimizationResult
    predictions: pd.DataFrame
    experimental_data: ExperimentalData
    statistics: Dict[str, float]
    figures: List[Path] = field(default_factory=list)
    conditions: List[str] = field(default_factory=list)
    config: SubstrateConfig = None
    confidence_intervals: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def get_parameters(self) -> Dict[str, float]:
        """Get fitted parameters."""
        return self.optimization_result.parameters

    def get_r_squared(self) -> float:
        """Get overall R² value."""
        return self.statistics.get("R_squared", 0.0)

    def summary(self) -> str:
        """Generate a text summary of results."""
        def _fmt(value, precision: int = 4) -> str:
            """Format numeric values safely for summary output."""
            if value is None:
                return "N/A"
            try:
                return f"{float(value):.{precision}f}"
            except (TypeError, ValueError):
                return "N/A"

        lines = [
            f"Model Type: {self.model_type}",
            f"Conditions: {', '.join(self.conditions)}",
            f"Optimization: {'Success' if self.optimization_result.success else 'Failed'}",
            "",
            "Fitted Parameters:",
        ]

        for name, value in self.optimization_result.parameters.items():
            ci_info = self.confidence_intervals.get(name, {})
            if ci_info and not np.isnan(ci_info.get('std_error', np.nan)):
                lines.append(f"  {name}: {value:.6f} ± {ci_info['std_error']:.4f} (95% CI: [{ci_info['ci_lower']:.4f}, {ci_info['ci_upper']:.4f}])")
            else:
                lines.append(f"  {name}: {value:.6f}")

        lines.extend([
            "",
            "Fit Statistics (Separate):",
            f"  Substrate: R²={_fmt(self.statistics.get('R_squared_substrate'))}, "
            f"Total Error (SSE)={_fmt(self.statistics.get('SSE_substrate'), precision=2)}, "
            f"RMSE={_fmt(self.statistics.get('RMSE_substrate'))}, NRMSE={_fmt(self.statistics.get('NRMSE_substrate'))}",
            f"  Biomass:   R²={_fmt(self.statistics.get('R_squared_biomass'))}, "
            f"Total Error (SSE)={_fmt(self.statistics.get('SSE_biomass'), precision=2)}, "
            f"RMSE={_fmt(self.statistics.get('RMSE_biomass'))}, NRMSE={_fmt(self.statistics.get('NRMSE_biomass'))}",
            "",
            "Combined Statistics (Weighted Average):",
            f"  R²: {_fmt(self.statistics.get('R_squared'))}",
            f"  AIC: {_fmt(self.statistics.get('AIC'), precision=2)}",
            f"  BIC: {_fmt(self.statistics.get('BIC'), precision=2)}",
        ])

        return "\n".join(lines)


class BaseWorkflow(ABC):
    """
    Abstract base class for all kinetic estimation workflows.

    Each workflow implementation must define:
    - model_type: String identifier for the model
    - parameter_names: List of parameters to optimize
    - create_ode_system: Method to create the ODE system
    """

    def __init__(
        self,
        config: SubstrateConfig,
        experimental_data: ExperimentalData,
        output_dir: str = "results"
    ):
        """
        Initialize the workflow.

        Args:
            config: Substrate configuration with parameters and bounds
            experimental_data: Loaded experimental data
            output_dir: Base directory for output files
        """
        self.config = config
        self.experimental_data = experimental_data
        self.results_writer = ResultsWriter(
            base_dir=output_dir,
            substrate_name=config.name
        )

        # Oxygen model for dual Monod workflows
        self.oxygen_model = OxygenModel(
            o2_max=config.oxygen.get("o2_max", 8.0),
            o2_min=config.oxygen.get("o2_min", 0.1),
            reaeration_rate=config.oxygen.get("reaeration_rate", 15.0),
            o2_range=config.oxygen.get("o2_range", 8.0)
        )

    @property
    @abstractmethod
    def model_type(self) -> str:
        """String identifier for the model type."""
        pass

    @property
    @abstractmethod
    def parameter_names(self) -> List[str]:
        """List of parameter names to optimize."""
        pass

    @abstractmethod
    def create_ode_system(self, parameters: Dict[str, float]) -> BaseODE:
        """
        Create an ODE system with the given parameters.

        Args:
            parameters: Dictionary of parameter values

        Returns:
            Configured ODE system
        """
        pass

    def run(
        self,
        fit_method: str = "global",
        optimization_method: str = "L-BFGS-B",
        save_results: bool = True,
        generate_plots: bool = True,
        verbose: bool = True
    ) -> WorkflowResult:
        """
        Execute the workflow: fit parameters and generate results.

        Args:
            fit_method: "global" for single parameter set across conditions,
                       "individual" for separate fits per condition
            optimization_method: "L-BFGS-B" or "differential_evolution"
            save_results: Whether to save results to disk
            generate_plots: Whether to generate figures
            verbose: Whether to print progress

        Returns:
            WorkflowResult with all outputs
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"Running {self.model_type} workflow")
            print(f"Substrate: {self.config.name}")
            print(f"Conditions: {self.experimental_data.conditions}")
            print(f"{'='*60}\n")

        # Step 1: Fit parameters
        if verbose:
            print("Step 1: Fitting parameters...")

        if fit_method == "global":
            opt_result = self._fit_global(optimization_method, verbose)
        else:
            opt_result = self._fit_individual(optimization_method, verbose)

        # Step 2: Generate predictions
        if verbose:
            print("\nStep 2: Generating predictions...")

        predictions = self._generate_predictions(opt_result.parameters)

        # Step 3: Calculate statistics
        if verbose:
            print("Step 3: Calculating statistics...")

        statistics = self._calculate_statistics(opt_result.parameters, predictions)
        opt_result.statistics = statistics

        # Step 3b: Calculate confidence intervals
        if verbose:
            print("Step 3b: Calculating confidence intervals...")
        
        confidence_intervals = self._calculate_confidence_intervals(opt_result)

        # Step 4: Generate plots
        figures = []
        if generate_plots:
            if verbose:
                print("Step 4: Generating plots...")
            figures = self._generate_plots(predictions)

        # Step 5: Save results
        if save_results:
            if verbose:
                print("Step 5: Saving results...")
            self._save_results(opt_result, predictions, statistics, figures, confidence_intervals)

        # Create result object
        result = WorkflowResult(
            model_type=self.model_type,
            optimization_result=opt_result,
            predictions=predictions,
            experimental_data=self.experimental_data,
            statistics=statistics,
            figures=figures,
            conditions=self.experimental_data.conditions,
            config=self.config,
            confidence_intervals=confidence_intervals
        )

        # Step 6: Generate PDF report
        if save_results:
            if verbose:
                print("Step 6: Generating PDF report...")
            try:
                pdf_path = generate_workflow_report(
                    output_dir=self.results_writer.output_dir,
                    summary_text=result.summary(),
                    model_type=self.model_type,
                    substrate_name=self.config.name,
                    statistics=statistics,
                    parameters=opt_result.parameters,
                    confidence_intervals=confidence_intervals,
                    figure_paths=figures,
                )
                if verbose:
                    print(f"  PDF report saved to: {pdf_path}")
            except Exception as e:
                if verbose:
                    print(f"  Warning: PDF report generation failed: {e}")

        if verbose:
            print("\n" + result.summary())
            print(f"\nResults saved to: {self.results_writer.output_dir}")

        return result

    def _fit_global(
        self,
        method: str,
        verbose: bool
    ) -> OptimizationResult:
        """Fit a single parameter set across all conditions."""
        # Build condition data for global objective
        conditions = []
        for cond in self.experimental_data.conditions:
            time, substrate, biomass = self.experimental_data.get_condition_data(cond)
            S0, X0 = substrate[0], biomass[0]

            initial_conditions = self._get_initial_conditions(S0, X0)

            conditions.append({
                'time': time,
                'substrate': substrate,
                'biomass': biomass,
                'initial_conditions': initial_conditions,
                't_span': (time[0], time[-1]),
                'label': cond
            })

        # Create global objective
        global_obj = GlobalObjectiveFunction(
            conditions=conditions,
            model_type=self.model_type,
            parameter_names=self.parameter_names,
            oxygen_model=self.oxygen_model
        )

        # Create optimizer
        optimizer = ParameterOptimizer(
            parameter_names=self.parameter_names,
            bounds={p: self.config.bounds[p] for p in self.parameter_names},
            initial_guess={p: self.config.initial_guesses[p] for p in self.parameter_names},
            method=method,
            verbose=verbose
        )

        return optimizer.optimize(global_obj)

    def _fit_individual(
        self,
        method: str,
        verbose: bool
    ) -> OptimizationResult:
        """Fit parameters separately for each condition (returns average)."""
        all_params = {p: [] for p in self.parameter_names}

        for cond in self.experimental_data.conditions:
            time, substrate, biomass = self.experimental_data.get_condition_data(cond)
            S0, X0 = substrate[0], biomass[0]

            objective = ObjectiveFunction(
                experimental_time=time,
                experimental_substrate=substrate,
                experimental_biomass=biomass,
                model_type=self.model_type,
                initial_conditions=self._get_initial_conditions(S0, X0),
                t_span=(time[0], time[-1]),
                parameter_names=self.parameter_names,
                oxygen_model=self.oxygen_model
            )

            optimizer = ParameterOptimizer(
                parameter_names=self.parameter_names,
                bounds={p: self.config.bounds[p] for p in self.parameter_names},
                initial_guess={p: self.config.initial_guesses[p] for p in self.parameter_names},
                method=method,
                verbose=False
            )

            result = optimizer.optimize(objective)

            for p in self.parameter_names:
                all_params[p].append(result.parameters[p])

            if verbose:
                print(f"  {cond}: R² = {result.statistics.get('R_squared', 'N/A')}")

        # Average parameters across conditions
        avg_params = {p: np.mean(vals) for p, vals in all_params.items()}

        return OptimizationResult(
            parameters=avg_params,
            statistics={},
            success=True,
            message="Averaged from individual fits",
            n_iterations=0,
            n_function_evals=0,
            initial_guess={p: self.config.initial_guesses[p] for p in self.parameter_names},
            bounds={p: self.config.bounds[p] for p in self.parameter_names},
            method=f"individual_{method}"
        )

    def _get_initial_conditions(self, S0: float, X0: float) -> List[float]:
        """Get initial conditions for ODE (overridden by subclasses if needed)."""
        return [S0, X0]

    def _generate_predictions(
        self,
        parameters: Dict[str, float]
    ) -> pd.DataFrame:
        """Generate model predictions for all conditions."""
        all_predictions = []

        for cond in self.experimental_data.conditions:
            time, substrate, biomass = self.experimental_data.get_condition_data(cond)
            S0, X0 = substrate[0], biomass[0]

            # Create ODE system with fitted parameters
            ode_system = self.create_ode_system(parameters)

            # Solve ODE
            t_eval = np.linspace(time[0], time[-1], 10000)
            result = solve_ode(
                ode_system=ode_system,
                initial_conditions=self._get_initial_conditions(S0, X0),
                t_span=(time[0], time[-1]),
                t_eval=t_eval
            )

            # Create DataFrame for this condition
            df = result.to_dataframe()
            df['Condition'] = cond
            df['Substrate_Name'] = self.config.name

            all_predictions.append(df)

        return pd.concat(all_predictions, ignore_index=True)

    def _calculate_statistics(
        self,
        parameters: Dict[str, float],
        predictions: pd.DataFrame
    ) -> Dict[str, float]:
        """Calculate fit statistics across all conditions.
        
        Uses separate statistics for substrate and biomass since they have
        different magnitudes (e.g., substrate ~1000 mg/L, biomass ~0.5 OD).
        """
        all_obs_substrate = []
        all_pred_substrate = []
        all_obs_biomass = []
        all_pred_biomass = []

        for cond in self.experimental_data.conditions:
            time, substrate, biomass = self.experimental_data.get_condition_data(cond)

            # Get predictions for this condition
            cond_pred = predictions[predictions['Condition'] == cond]

            # Interpolate to experimental time points
            pred_substrate = np.interp(time, cond_pred['Time'].values, cond_pred['Substrate'].values)
            pred_biomass = np.interp(time, cond_pred['Time'].values, cond_pred['Biomass'].values)

            all_obs_substrate.extend(substrate)
            all_pred_substrate.extend(pred_substrate)
            all_obs_biomass.extend(biomass)
            all_pred_biomass.extend(pred_biomass)

        # Calculate separate statistics for substrate and biomass
        separate_stats = calculate_separate_statistics(
            observed_substrate=np.array(all_obs_substrate),
            predicted_substrate=np.array(all_pred_substrate),
            observed_biomass=np.array(all_obs_biomass),
            predicted_biomass=np.array(all_pred_biomass),
            n_parameters=len(self.parameter_names)
        )
        
        # Build combined statistics dict with both overall and separate metrics
        stats = {
            # Overall combined metrics (weighted average)
            "R_squared": separate_stats["combined"]["R_squared"],
            "NRMSE": separate_stats["combined"]["NRMSE"],
            "AIC": separate_stats["combined"]["AIC"],
            "BIC": separate_stats["combined"]["BIC"],
            "n_observations": separate_stats["combined"]["n_observations"],
            "n_parameters": separate_stats["combined"]["n_parameters"],
            
            # Substrate-specific metrics
            "R_squared_substrate": separate_stats["substrate"]["R_squared"],
            "SSE_substrate": separate_stats["substrate"]["SSE"],
            "RMSE_substrate": separate_stats["substrate"]["RMSE"],
            "NRMSE_substrate": separate_stats["substrate"]["NRMSE"],
            
            # Biomass-specific metrics
            "R_squared_biomass": separate_stats["biomass"]["R_squared"],
            "SSE_biomass": separate_stats["biomass"]["SSE"],
            "RMSE_biomass": separate_stats["biomass"]["RMSE"],
            "NRMSE_biomass": separate_stats["biomass"]["NRMSE"],
        }
        
        # Also compute legacy combined RMSE for backwards compatibility
        all_obs = np.array(all_obs_substrate + all_obs_biomass)
        all_pred = np.array(all_pred_substrate + all_pred_biomass)
        legacy_stats = calculate_all_statistics(all_obs, all_pred, len(self.parameter_names))
        stats["RMSE"] = legacy_stats["RMSE"]  # Keep legacy RMSE for compatibility
        stats["SSE"] = legacy_stats["SSE"]
        
        return stats

    def _calculate_confidence_intervals(
        self,
        opt_result: OptimizationResult,
        confidence_level: float = 0.95
    ) -> Dict[str, Dict[str, float]]:
        """Calculate confidence intervals for fitted parameters.
        
        Uses Hessian-based approximation to estimate parameter uncertainties.
        
        Args:
            opt_result: Optimization result with fitted parameters
            confidence_level: Confidence level for intervals (default 0.95 = 95%)
            
        Returns:
            Dictionary mapping parameter names to their confidence intervals
            with 'value', 'lower', 'upper', and 'std_error' keys
        """
        try:
            # Build condition data for objective function
            conditions = []
            for cond in self.experimental_data.conditions:
                time, substrate, biomass = self.experimental_data.get_condition_data(cond)
                S0, X0 = substrate[0], biomass[0]
                
                conditions.append({
                    'time': time,
                    'substrate': substrate,
                    'biomass': biomass,
                    'S0': S0,
                    'X0': X0
                })
            
            # Create objective function for CI calculation
            def objective_for_ci(params):
                param_dict = dict(zip(self.parameter_names, params))
                total_error = 0.0
                
                for cond_data in conditions:
                    try:
                        # Create ODE system with these parameters
                        ode_system = self.create_ode_system(param_dict)
                        
                        # Solve ODE
                        result = solve_ode(
                            ode_system=ode_system,
                            initial_conditions=self._get_initial_conditions(cond_data['S0'], cond_data['X0']),
                            t_span=(cond_data['time'].min(), cond_data['time'].max()),
                            t_eval=cond_data['time']
                        )
                        
                        # Get predictions from states dict
                        pred_substrate = result.states.get('Substrate', result.states.get('S', np.zeros_like(cond_data['time'])))
                        pred_biomass = result.states.get('Biomass', result.states.get('X', np.zeros_like(cond_data['time'])))
                        
                        # Calculate SSE
                        substrate_err = np.sum((pred_substrate - cond_data['substrate'])**2)
                        biomass_err = np.sum((pred_biomass - cond_data['biomass'])**2)
                        total_error += substrate_err + biomass_err
                        
                    except Exception:
                        total_error += 1e10
                
                return total_error
            
            # Get parameter array
            param_values = np.array([opt_result.parameters[name] for name in self.parameter_names])
            
            # Calculate confidence intervals
            ci_results = calculate_parameter_confidence_intervals(
                objective_function=objective_for_ci,
                optimal_params=param_values,
                parameter_names=self.parameter_names,
                n_observations=sum(len(c['time']) * 2 for c in conditions),  # 2 measurements per time point
                confidence_level=confidence_level
            )
            
            return ci_results
            
        except Exception as e:
            # Return dict with NaN values if CI calculation fails
            print(f"Warning: Could not calculate confidence intervals: {e}")
            return {
                name: {
                    "value": float(opt_result.parameters[name]),
                    "std_error": np.nan,
                    "ci_lower": np.nan,
                    "ci_upper": np.nan,
                    "relative_error_pct": np.nan
                }
                for name in self.parameter_names
            }

    def _generate_plots(self, predictions: pd.DataFrame) -> List[Path]:
        """Generate visualization plots."""
        import matplotlib.pyplot as plt

        figures = []

        # Plot fit results for each condition
        fig, axes = plt.subplots(
            nrows=2,
            ncols=len(self.experimental_data.conditions),
            figsize=(5 * len(self.experimental_data.conditions), 8),
            dpi=300
        )

        if len(self.experimental_data.conditions) == 1:
            axes = axes.reshape(-1, 1)

        for i, cond in enumerate(self.experimental_data.conditions):
            time, substrate, biomass = self.experimental_data.get_condition_data(cond)
            cond_pred = predictions[predictions['Condition'] == cond]

            # Substrate plot
            axes[0, i].scatter(time, substrate, color='#EE6677', s=60, label='Experimental', zorder=5)
            axes[0, i].plot(cond_pred['Time'], cond_pred['Substrate'], color='#4477AA', linewidth=2, label='Model')
            axes[0, i].set_xlabel('Time (days)')
            axes[0, i].set_ylabel(f'{self.config.name} (mg/L)')
            axes[0, i].set_title(f'{self.config.name} - {cond}')
            axes[0, i].legend()
            axes[0, i].grid(True, alpha=0.3)

            # Biomass plot
            axes[1, i].scatter(time, biomass, color='#EE6677', s=60, label='Experimental', zorder=5)
            axes[1, i].plot(cond_pred['Time'], cond_pred['Biomass'], color='#228833', linewidth=2, label='Model')
            axes[1, i].set_xlabel('Time (days)')
            axes[1, i].set_ylabel('Biomass (mg cells/L)')
            axes[1, i].set_title(f'Biomass - {cond}')
            axes[1, i].legend()
            axes[1, i].grid(True, alpha=0.3)

        plt.suptitle(f'{self.model_type} Model Fit', fontsize=14, fontweight='bold')
        plt.tight_layout()

        paths = self.results_writer.save_figure(fig, f"fit_results_{self.model_type}")
        figures.extend(paths)
        plt.close(fig)

        return figures

    def _save_results(
        self,
        opt_result: OptimizationResult,
        predictions: pd.DataFrame,
        statistics: Dict[str, float],
        figures: List[Path],
        confidence_intervals: Dict[str, Dict[str, float]] = None
    ) -> None:
        """Save all results to disk."""
        # Parameter units
        units = {
            "qmax": f"mg{self.config.name}/(mgCells·day)",
            "Ks": f"mg{self.config.name}/L",
            "Ki": f"mg{self.config.name}/L",
            "Y": f"mgCells/mg{self.config.name}",
            "b_decay": "day^-1",
            "K_o2": "mgO2/L",
            "Y_o2": f"mgO2/mg{self.config.name}",
            "lag_time": "days"
        }

        fitted = FittedParameters(
            parameters=opt_result.parameters,
            units={p: units.get(p, "") for p in opt_result.parameters},
            statistics=statistics,
            conditions=self.experimental_data.conditions,
            model_type=self.model_type,
            confidence_intervals=confidence_intervals or {}
        )

        self.results_writer.save_fitted_parameters(fitted)
        self.results_writer.save_predictions(predictions)
        self.results_writer.save_statistics(statistics)
        self.results_writer.save_run_info(
            config=self.config.__dict__,
            data_path=self.experimental_data.metadata.get("source_file", "unknown")
        )
