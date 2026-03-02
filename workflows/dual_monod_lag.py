"""
Dual Monod workflow with lag phase - the complete kinetic model.

This workflow implements the full model with substrate limitation,
oxygen dynamics, reaeration, and a lag phase to account for
microbial adaptation before exponential growth.

Model equations:
    dS/dt = -(1/Y) * q * X * lag_factor
    dX/dt = q * lag_factor * X - b_decay * X
    dO2/dt = -r_O2 * lag_factor + reaeration

where:
    q = qmax * S/(Ks+S) * (1-S/Ki) * O2/(K_o2+O2)
    lag_factor = 1 / (1 + exp(-k * (t - lag_time/2) / lag_time))

Use case:
- Cultures with observable lag phase
- Inoculum adaptation studies
- Accurate modeling of initial growth dynamics
"""

from typing import Dict, List
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.core.ode_systems import DualMonodLagODE
from src.core.monod import lag_phase_factor
from src.io.data_loader import ExperimentalData
from src.io.config_loader import SubstrateConfig
from .base_workflow import BaseWorkflow


class DualMonodLagWorkflow(BaseWorkflow):
    """
    Workflow for dual Monod kinetics with lag phase.

    This is the most comprehensive model, accounting for:
    - Substrate limitation with inhibition
    - Oxygen limitation with reaeration
    - Lag phase for microbial adaptation

    Parameters optimized:
        - qmax: Maximum specific uptake rate
        - Ks: Half-saturation constant for substrate
        - Ki: Substrate inhibition constant
        - Y: Yield coefficient (biomass/substrate)
        - b_decay: Decay/maintenance coefficient
        - K_o2: Half-saturation constant for oxygen
        - Y_o2: Oxygen yield coefficient
        - lag_time: Duration of the lag phase
    """

    @property
    def model_type(self) -> str:
        return "dual_monod_lag"

    @property
    def parameter_names(self) -> List[str]:
        return ["qmax", "Ks", "Ki", "Y", "b_decay", "K_o2", "Y_o2", "lag_time"]

    def create_ode_system(self, parameters: Dict[str, float]) -> DualMonodLagODE:
        """
        Create a DualMonodLagODE system with given parameters.

        Args:
            parameters: Dictionary containing all kinetic parameters

        Returns:
            Configured DualMonodLagODE instance
        """
        return DualMonodLagODE(
            qmax=parameters["qmax"],
            Ks=parameters["Ks"],
            Ki=parameters["Ki"],
            Y=parameters["Y"],
            b_decay=parameters["b_decay"],
            K_o2=parameters["K_o2"],
            Y_o2=parameters["Y_o2"],
            lag_time=parameters["lag_time"],
            oxygen_model=self.oxygen_model
        )

    def _get_initial_conditions(self, S0: float, X0: float) -> List[float]:
        """
        Get initial conditions for dual Monod with lag (3 states).

        Args:
            S0: Initial substrate concentration
            X0: Initial biomass concentration

        Returns:
            List [S0, X0, O2_0] where O2_0 is saturation
        """
        return [S0, X0, self.oxygen_model.o2_max]

    def _generate_plots(self, predictions) -> List[Path]:
        """
        Generate plots including lag phase visualization.

        Extends base class to add a lag phase factor plot.
        """
        # Generate standard fit plots
        figures = super()._generate_plots(predictions)

        # Add lag phase visualization
        lag_time = self.config.initial_guesses.get("lag_time", 3.0)
        t_max = predictions['Time'].max()

        fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

        time = np.linspace(0, t_max, 10000)
        lag_factors = np.array([lag_phase_factor(t, lag_time) for t in time])

        ax.plot(time, lag_factors, color='#4477AA', linewidth=2.5)
        ax.axvline(x=lag_time, color='#EE6677', linestyle='--', linewidth=1.5,
                   label=f'Lag time = {lag_time:.2f} days')
        ax.axhline(y=0.5, color='#BBBBBB', linestyle=':', linewidth=1,
                   label='50% activity')

        ax.fill_between(time, 0, lag_factors, alpha=0.2, color='#4477AA')

        ax.set_xlabel('Time (days)', fontsize=11)
        ax.set_ylabel('Growth Activity Factor', fontsize=11)
        ax.set_title('Lag Phase Factor', fontsize=12, fontweight='bold')
        ax.legend(frameon=True, fancybox=True)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.1)

        plt.tight_layout()

        paths = self.results_writer.save_figure(fig, "lag_phase_factor")
        figures.extend(paths)
        plt.close(fig)

        return figures


def run_dual_monod_lag(
    config: SubstrateConfig,
    experimental_data: ExperimentalData,
    output_dir: str = "results",
    fit_method: str = "global",
    verbose: bool = True
):
    """
    Convenience function to run the dual Monod with lag workflow.

    Args:
        config: Substrate configuration
        experimental_data: Loaded experimental data
        output_dir: Directory for output files
        fit_method: "global" or "individual"
        verbose: Print progress

    Returns:
        WorkflowResult with fitted parameters and predictions

    Example:
        >>> from src.io.data_loader import load_experimental_data
        >>> from src.io.config_loader import load_config
        >>>
        >>> config = load_config("config/substrates/xylose.json")
        >>> data = load_experimental_data("data/xylose_data.csv")
        >>> result = run_dual_monod_lag(config, data)
        >>> print(f"Lag time = {result.get_parameters()['lag_time']:.2f} days")
    """
    workflow = DualMonodLagWorkflow(config, experimental_data, output_dir)
    return workflow.run(
        fit_method=fit_method,
        verbose=verbose
    )


def compare_with_without_lag(
    config: SubstrateConfig,
    experimental_data: ExperimentalData,
    output_dir: str = "results"
) -> Dict[str, any]:
    """
    Compare dual Monod model with and without lag phase.

    This function runs both models and provides a direct comparison
    of their performance.

    Args:
        config: Substrate configuration
        experimental_data: Experimental data
        output_dir: Output directory

    Returns:
        Dictionary with comparison results
    """
    from .dual_monod import DualMonodWorkflow
    from src.fitting.statistics import compare_models

    # Run dual Monod (no lag)
    workflow_no_lag = DualMonodWorkflow(config, experimental_data, output_dir)
    result_no_lag = workflow_no_lag.run(verbose=False)

    # Run dual Monod with lag
    workflow_lag = DualMonodLagWorkflow(config, experimental_data, output_dir)
    result_lag = workflow_lag.run(verbose=False)

    # Compare statistics
    model_stats = {
        "Dual Monod": result_no_lag.statistics,
        "Dual Monod + Lag": result_lag.statistics
    }

    comparison = compare_models(model_stats)

    return {
        "result_no_lag": result_no_lag,
        "result_with_lag": result_lag,
        "comparison": comparison,
        "recommendation": comparison["best_model"]["by_AIC"]
    }
