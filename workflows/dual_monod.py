"""
Dual Monod workflow - substrate and oxygen limitation with reaeration.

This workflow extends the single Monod model by adding oxygen dynamics,
suitable for aerobic systems where dissolved oxygen can become limiting.

Model equations:
    dS/dt = -(1/Y) * q * X
    dX/dt = (q - b_decay) * X
    dO2/dt = -r_O2 + reaeration

where q = qmax * S/(Ks+S) * (1-S/Ki) * O2/(K_o2+O2)

Use case:
- Aerobic batch or continuous cultures
- Systems with variable oxygen levels
- Wastewater treatment applications
"""

from typing import Dict, List

from src.core.ode_systems import DualMonodODE
from src.core.oxygen import OxygenModel
from src.io.data_loader import ExperimentalData
from src.io.config_loader import SubstrateConfig
from .base_workflow import BaseWorkflow


class DualMonodWorkflow(BaseWorkflow):
    """
    Workflow for dual Monod kinetics with oxygen dynamics.

    This model accounts for both substrate and oxygen limitation,
    with reaeration to replenish dissolved oxygen.

    Parameters optimized:
        - qmax: Maximum specific uptake rate
        - Ks: Half-saturation constant for substrate
        - Ki: Substrate inhibition constant
        - Y: Yield coefficient (biomass/substrate)
        - b_decay: Decay/maintenance coefficient
        - K_o2: Half-saturation constant for oxygen
        - Y_o2: Oxygen yield coefficient
    """

    @property
    def model_type(self) -> str:
        return "dual_monod"

    @property
    def parameter_names(self) -> List[str]:
        return ["qmax", "Ks", "Ki", "Y", "b_decay", "K_o2", "Y_o2"]

    def create_ode_system(self, parameters: Dict[str, float]) -> DualMonodODE:
        """
        Create a DualMonodODE system with given parameters.

        Args:
            parameters: Dictionary containing all kinetic parameters

        Returns:
            Configured DualMonodODE instance
        """
        return DualMonodODE(
            qmax=parameters["qmax"],
            Ks=parameters["Ks"],
            Ki=parameters["Ki"],
            Y=parameters["Y"],
            b_decay=parameters["b_decay"],
            K_o2=parameters["K_o2"],
            Y_o2=parameters["Y_o2"],
            oxygen_model=self.oxygen_model
        )

    def _get_initial_conditions(self, S0: float, X0: float) -> List[float]:
        """
        Get initial conditions for dual Monod (3 states).

        Args:
            S0: Initial substrate concentration
            X0: Initial biomass concentration

        Returns:
            List [S0, X0, O2_0] where O2_0 is saturation
        """
        return [S0, X0, self.oxygen_model.o2_max]


def run_dual_monod(
    config: SubstrateConfig,
    experimental_data: ExperimentalData,
    output_dir: str = "results",
    fit_method: str = "global",
    verbose: bool = True
):
    """
    Convenience function to run the dual Monod workflow.

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
        >>> result = run_dual_monod(config, data)
        >>> print(f"R² = {result.get_r_squared():.4f}")
    """
    workflow = DualMonodWorkflow(config, experimental_data, output_dir)
    return workflow.run(
        fit_method=fit_method,
        verbose=verbose
    )
