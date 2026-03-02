"""
Single Monod workflow - substrate limitation only.

This workflow implements the simplest kinetic model with only
substrate limitation and biomass growth, without oxygen dynamics.

Model equations:
    dS/dt = -(1/Y) * q * X
    dX/dt = (q - b_decay) * X

where q = qmax * S/(Ks + S) * (1 - S/Ki)

Use case:
- Anaerobic systems
- Systems with excess oxygen
- Simple batch cultures where O2 is not limiting
"""

from typing import Dict, List

from src.core.ode_systems import SingleMonodODE
from src.io.data_loader import ExperimentalData
from src.io.config_loader import SubstrateConfig
from .base_workflow import BaseWorkflow


class SingleMonodWorkflow(BaseWorkflow):
    """
    Workflow for single Monod kinetics (no oxygen dynamics).

    This is the simplest model suitable for cases where oxygen
    is not limiting or for anaerobic cultures.

    Parameters optimized:
        - qmax: Maximum specific uptake rate
        - Ks: Half-saturation constant
        - Ki: Substrate inhibition constant
        - Y: Yield coefficient
        - b_decay: Decay/maintenance coefficient
    """

    @property
    def model_type(self) -> str:
        return "single_monod"

    @property
    def parameter_names(self) -> List[str]:
        return ["qmax", "Ks", "Ki", "Y", "b_decay"]

    def create_ode_system(self, parameters: Dict[str, float]) -> SingleMonodODE:
        """
        Create a SingleMonodODE system with given parameters.

        Args:
            parameters: Dictionary containing qmax, Ks, Ki, Y, b_decay

        Returns:
            Configured SingleMonodODE instance
        """
        return SingleMonodODE(
            qmax=parameters["qmax"],
            Ks=parameters["Ks"],
            Ki=parameters["Ki"],
            Y=parameters["Y"],
            b_decay=parameters["b_decay"]
        )

    def _get_initial_conditions(self, S0: float, X0: float) -> List[float]:
        """
        Get initial conditions for single Monod (2 states).

        Args:
            S0: Initial substrate concentration
            X0: Initial biomass concentration

        Returns:
            List [S0, X0]
        """
        return [S0, X0]


def run_single_monod(
    config: SubstrateConfig,
    experimental_data: ExperimentalData,
    output_dir: str = "results",
    fit_method: str = "global",
    verbose: bool = True
):
    """
    Convenience function to run the single Monod workflow.

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
        >>> result = run_single_monod(config, data)
        >>> print(result.get_r_squared())
    """
    workflow = SingleMonodWorkflow(config, experimental_data, output_dir)
    return workflow.run(
        fit_method=fit_method,
        verbose=verbose
    )
