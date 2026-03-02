"""
Objective functions for parameter optimization.

This module defines objective functions that measure the discrepancy
between model predictions and experimental data, used by optimizers
to find the best-fit parameters.
"""

import numpy as np
from typing import Callable, List, Tuple, Dict, Any, Optional
from scipy.integrate import solve_ivp

from ..core.ode_systems import BaseODE, SingleMonodODE, SingleMonodLagODE, DualMonodODE, DualMonodLagODE
from ..core.oxygen import OxygenModel


class ObjectiveFunction:
    """
    Objective function for parameter fitting.

    This class wraps the ODE solver and computes the sum of squared
    errors between model predictions and experimental data.

    Attributes:
        experimental_time: Time points from experimental data
        experimental_substrate: Substrate measurements
        experimental_biomass: Biomass measurements
        model_type: Type of ODE model to use
        initial_conditions: Initial state values
        t_span: Time span for simulation
        parameter_names: Names of parameters being optimized
    """

    def __init__(
        self,
        experimental_time: np.ndarray,
        experimental_substrate: np.ndarray,
        experimental_biomass: np.ndarray,
        model_type: str,
        initial_conditions: List[float],
        t_span: Tuple[float, float],
        parameter_names: List[str],
        oxygen_model: Optional[OxygenModel] = None,
        weight_substrate: float = 1.0,
        weight_biomass: float = 1.0,
        num_eval_points: int = 10000,
        normalize_errors: bool = True
    ):
        """
        Initialize the objective function.

        Args:
            experimental_time: Time points of experimental measurements
            experimental_substrate: Substrate concentration measurements
            experimental_biomass: Biomass concentration measurements
            model_type: "single_monod", "dual_monod", or "dual_monod_lag"
            initial_conditions: [S0, X0] or [S0, X0, O2_0]
            t_span: (t_start, t_end) for simulation
            parameter_names: List of parameter names in optimization order
            oxygen_model: OxygenModel instance for dual Monod models
            weight_substrate: Weight for substrate error (default 1.0)
            weight_biomass: Weight for biomass error (default 1.0)
            num_eval_points: Number of points for ODE evaluation
            normalize_errors: Whether to normalize errors by data range (recommended)
        """
        self.experimental_time = experimental_time
        self.experimental_substrate = experimental_substrate
        self.experimental_biomass = experimental_biomass
        self.model_type = model_type.lower()
        self.initial_conditions = initial_conditions
        self.t_span = t_span
        self.parameter_names = parameter_names
        self.oxygen_model = oxygen_model or OxygenModel()
        self.weight_substrate = weight_substrate
        self.weight_biomass = weight_biomass
        self.normalize_errors = normalize_errors

        # Create evaluation time grid
        self.t_eval = np.linspace(t_span[0], t_span[1], num_eval_points)

        # Calculate normalization factors (data ranges) to make errors comparable
        # This ensures substrate (e.g., 1000 mg/L) and biomass (e.g., 0.5 OD) contribute equally
        substrate_range = np.ptp(experimental_substrate)  # peak-to-peak (max - min)
        biomass_range = np.ptp(experimental_biomass)
        
        # Avoid division by zero; use mean as fallback if range is 0
        self.substrate_norm = substrate_range if substrate_range > 0 else np.mean(experimental_substrate) or 1.0
        self.biomass_norm = biomass_range if biomass_range > 0 else np.mean(experimental_biomass) or 1.0

        # Track function evaluations
        self.n_evaluations = 0
        self.best_error = np.inf 
        self.best_params = None

    def __call__(self, params: np.ndarray) -> float:
        """
        Evaluate the objective function.

        Args:
            params: Array of parameter values in the order of parameter_names

        Returns:
            Sum of squared errors (lower is better)
        """
        self.n_evaluations += 1

        try:
            # Create ODE system with current parameters
            ode_system = self._create_ode_system(params)

            # Solve ODE
            solution = solve_ivp(
                fun=ode_system.derivatives,
                t_span=self.t_span,
                y0=self.initial_conditions,
                method='RK45',
                t_eval=self.t_eval,
                rtol=1e-6,
                atol=1e-9
            )

            if not solution.success:
                return 1e10  # Return large error for failed integration

            # Extract model predictions
            model_substrate = solution.y[0]
            model_biomass = solution.y[1]

            # Interpolate model to experimental time points
            interp_substrate = np.interp(
                self.experimental_time,
                solution.t,
                model_substrate
            )
            interp_biomass = np.interp(
                self.experimental_time,
                solution.t,
                model_biomass
            )

            # Compute errors with optional normalization
            if self.normalize_errors:
                # Normalized errors: divide by data range so both contribute equally
                error_substrate = np.sum(((self.experimental_substrate - interp_substrate) / self.substrate_norm) ** 2)
                error_biomass = np.sum(((self.experimental_biomass - interp_biomass) / self.biomass_norm) ** 2)
            else:
                # Raw errors (original behavior)
                error_substrate = np.sum((self.experimental_substrate - interp_substrate) ** 2)
                error_biomass = np.sum((self.experimental_biomass - interp_biomass) ** 2)

            total_error = (
                self.weight_substrate * error_substrate +
                self.weight_biomass * error_biomass
            )

            # Track best result
            if total_error < self.best_error:
                self.best_error = total_error
                self.best_params = params.copy()

            return total_error
        
        except Exception:
            return 1e10  # Return large error for any exception. In case of failure.

    def _create_ode_system(self, params: np.ndarray) -> BaseODE:
        """Create an ODE system with the given parameters."""
        param_dict = dict(zip(self.parameter_names, params))

        ki_value = param_dict.get("Ki", None)

        if self.model_type == "single_monod":
            return SingleMonodODE(
                qmax=param_dict["qmax"],
                Ks=param_dict["Ks"],
                Ki=ki_value,
                Y=param_dict["Y"],
                b_decay=param_dict["b_decay"]
            )
        elif self.model_type == "single_monod_lag":
            return SingleMonodLagODE(
                qmax=param_dict["qmax"],
                Ks=param_dict["Ks"],
                Ki=ki_value,
                Y=param_dict["Y"],
                b_decay=param_dict["b_decay"],
                lag_time=param_dict["lag_time"]
            )
        elif self.model_type == "dual_monod":
            return DualMonodODE(
                qmax=param_dict["qmax"],
                Ks=param_dict["Ks"],
                Ki=ki_value,
                Y=param_dict["Y"],
                b_decay=param_dict["b_decay"],
                K_o2=param_dict["K_o2"],
                Y_o2=param_dict["Y_o2"],
                oxygen_model=self.oxygen_model
            )
        elif self.model_type == "dual_monod_lag":
            return DualMonodLagODE(
                qmax=param_dict["qmax"],
                Ks=param_dict["Ks"],
                Ki=ki_value,
                Y=param_dict["Y"],
                b_decay=param_dict["b_decay"],
                K_o2=param_dict["K_o2"],
                Y_o2=param_dict["Y_o2"],
                lag_time=param_dict["lag_time"],
                oxygen_model=self.oxygen_model
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")


class GlobalObjectiveFunction:
    """
    Objective function for global fitting across multiple conditions.

    This fits a single parameter set to multiple experimental conditions
    (e.g., different starting substrate concentrations).
    """

    def __init__(
        self,
        conditions: List[Dict[str, Any]],
        model_type: str,
        parameter_names: List[str],
        oxygen_model: Optional[OxygenModel] = None,
        weight_substrate: float = 1.0,
        weight_biomass: float = 1.0,
        num_eval_points: int = 10000,
        normalize_errors: bool = True
    ):
        """
        Initialize global objective function.

        Args:
            conditions: List of condition dictionaries, each containing:
                - 'time': experimental time array
                - 'substrate': substrate measurements
                - 'biomass': biomass measurements
                - 'initial_conditions': [S0, X0] or [S0, X0, O2_0]
                - 't_span': (t_start, t_end)
                - 'label': optional condition label
            model_type: Type of ODE model
            parameter_names: Parameter names in optimization order
            oxygen_model: OxygenModel instance
            weight_substrate: Weight for substrate error
            weight_biomass: Weight for biomass error
            num_eval_points: Points for ODE evaluation
            normalize_errors: Whether to normalize errors by data range (recommended)
        """
        self.conditions = conditions
        self.model_type = model_type.lower()
        self.parameter_names = parameter_names
        self.oxygen_model = oxygen_model or OxygenModel()
        self.weight_substrate = weight_substrate
        self.weight_biomass = weight_biomass
        self.num_eval_points = num_eval_points
        self.normalize_errors = normalize_errors

        # Track evaluations
        self.n_evaluations = 0
        self.best_error = np.inf
        self.best_params = None

    def __call__(self, params: np.ndarray) -> float:
        """
        Evaluate global objective across all conditions.

        Args:
            params: Array of parameter values

        Returns:
            Total sum of squared errors across all conditions
        """
        self.n_evaluations += 1
        total_error = 0.0

        for condition in self.conditions:
            # Create objective for this condition
            obj = ObjectiveFunction(
                experimental_time=condition['time'],
                experimental_substrate=condition['substrate'],
                experimental_biomass=condition['biomass'],
                model_type=self.model_type,
                initial_conditions=condition['initial_conditions'],
                t_span=condition['t_span'],
                parameter_names=self.parameter_names,
                oxygen_model=self.oxygen_model,
                weight_substrate=self.weight_substrate,
                weight_biomass=self.weight_biomass,
                num_eval_points=self.num_eval_points,
                normalize_errors=self.normalize_errors
            )

            error = obj(params)
            total_error += error

        # Track best result
        if total_error < self.best_error:
            self.best_error = total_error
            self.best_params = params.copy()

        return total_error


def create_objective_from_data(
    experimental_data,
    condition: str,
    model_type: str,
    config,
    parameter_names: List[str]
) -> ObjectiveFunction:
    """
    Create an objective function from ExperimentalData and config.

    Args:
        experimental_data: ExperimentalData object
        condition: Condition label (e.g., '5mM')
        model_type: Type of model
        config: SubstrateConfig object
        parameter_names: List of parameter names

    Returns:
        Configured ObjectiveFunction
    """
    time, substrate, biomass = experimental_data.get_condition_data(condition)

    # Initial conditions
    S0, X0 = substrate[0], biomass[0]

    if model_type in ("single_monod", "single_monod_lag"):
        initial_conditions = [S0, X0]
    else:
        o2_max = config.oxygen.get("o2_max", 8.0)
        initial_conditions = [S0, X0, o2_max]

    t_span = (time[0], time[-1])

    # Create oxygen model if needed
    oxygen_model = None
    if model_type in ("dual_monod", "dual_monod_lag"):
        oxygen_model = OxygenModel(
            o2_max=config.oxygen.get("o2_max", 8.0),
            o2_min=config.oxygen.get("o2_min", 0.1),
            reaeration_rate=config.oxygen.get("reaeration_rate", 15.0),
            o2_range=config.oxygen.get("o2_range", 8.0)
        )

    return ObjectiveFunction(
        experimental_time=time,
        experimental_substrate=substrate,
        experimental_biomass=biomass,
        model_type=model_type,
        initial_conditions=initial_conditions,
        t_span=t_span,
        parameter_names=parameter_names,
        oxygen_model=oxygen_model
    )
