"""
Single Monod workflow with lag phase - substrate limitation with lag.

This workflow implements a model with substrate limitation, biomass growth,
and a lag phase to account for microbial adaptation, without oxygen dynamics.

Model equations:
    dS/dt = -(1/Y) * q * X * lag_factor
    dX/dt = q * lag_factor * X - b_decay * X

where:
    q = qmax * S/(Ks+S) * (1-S/Ki)
    lag_factor = 1 / (1 + exp(-k * (t - lag_time/2) / lag_time))

Use case:
- Anaerobic systems with observable lag phase
- Batch cultures with adaptation period but excess oxygen
- Simpler lag-phase models when oxygen is not limiting
"""

from typing import Dict, List
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.core.ode_systems import SingleMonodLagODE
from src.core.monod import lag_phase_factor
from src.io.data_loader import ExperimentalData
from src.io.config_loader import SubstrateConfig
from .base_workflow import BaseWorkflow


class SingleMonodLagWorkflow(BaseWorkflow):
    """
    Workflow for single Monod kinetics with lag phase (no oxygen dynamics).

    This model accounts for:
    - Substrate limitation with optional inhibition
    - Lag phase for microbial adaptation
    - No oxygen dynamics (2-state system)

    Parameters optimized:
        - qmax: Maximum specific uptake rate
        - Ks: Half-saturation constant
        - Ki: Substrate inhibition constant
        - Y: Yield coefficient
        - b_decay: Decay/maintenance coefficient
        - lag_time: Duration of the lag phase
    """

    @property
    def model_type(self) -> str:
        return "single_monod_lag"

    @property
    def parameter_names(self) -> List[str]:
        return ["qmax", "Ks", "Ki", "Y", "b_decay", "lag_time"]

    def create_ode_system(self, parameters: Dict[str, float]) -> SingleMonodLagODE:
        """
        Create a SingleMonodLagODE system with given parameters.

        Args:
            parameters: Dictionary containing qmax, Ks, Ki, Y, b_decay, lag_time

        Returns:
            Configured SingleMonodLagODE instance
        """
        return SingleMonodLagODE(
            qmax=parameters["qmax"],
            Ks=parameters["Ks"],
            Ki=parameters["Ki"],
            Y=parameters["Y"],
            b_decay=parameters["b_decay"],
            lag_time=parameters["lag_time"],
        )

    def _get_initial_conditions(self, S0: float, X0: float) -> List[float]:
        """
        Get initial conditions for single Monod with lag (2 states).

        Args:
            S0: Initial substrate concentration
            X0: Initial biomass concentration

        Returns:
            List [S0, X0]
        """
        return [S0, X0]

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


def run_single_monod_lag(
    config: SubstrateConfig,
    experimental_data: ExperimentalData,
    output_dir: str = "results",
    fit_method: str = "global",
    verbose: bool = True
):
    """
    Convenience function to run the single Monod with lag workflow.

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
        >>> config = load_config("config/substrates/glucose.json")
        >>> data = load_experimental_data("data/glucose_data.csv")
        >>> result = run_single_monod_lag(config, data)
        >>> print(f"Lag time = {result.get_parameters()['lag_time']:.2f} days")
    """
    workflow = SingleMonodLagWorkflow(config, experimental_data, output_dir)
    return workflow.run(
        fit_method=fit_method,
        verbose=verbose
    )
