"""
ODE solvers and simulation utilities.

This module provides wrappers around scipy's ODE solvers optimized
for kinetic parameter estimation problems.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union
from scipy.integrate import solve_ivp

from .ode_systems import BaseODE


@dataclass
class SimulationResult:
    """
    Container for ODE simulation results.

    Attributes:
        time: Array of time points
        states: Dictionary mapping state names to value arrays
        success: Whether the integration succeeded
        message: Solver status message
        n_evaluations: Number of ODE evaluations
        parameters: Parameters used for simulation
    """
    time: np.ndarray
    states: Dict[str, np.ndarray]
    success: bool
    message: str
    n_evaluations: int
    parameters: Dict[str, float] = field(default_factory=dict)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to a pandas DataFrame."""
        data = {"Time": self.time}
        data.update(self.states)
        return pd.DataFrame(data)

    def get_state(self, name: str) -> np.ndarray:
        """Get a specific state variable by name."""
        if name not in self.states:
            raise KeyError(f"State '{name}' not found. Available: {list(self.states.keys())}")
        return self.states[name]

    def interpolate_at(self, times: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Interpolate state values at specific time points.

        Args:
            times: Array of time points for interpolation

        Returns:
            Dictionary of interpolated state values
        """
        result = {}
        for name, values in self.states.items():
            result[name] = np.interp(times, self.time, values)
        return result


def solve_ode(
    ode_system: BaseODE,
    initial_conditions: np.ndarray,
    t_span: tuple,
    t_eval: Optional[np.ndarray] = None,
    method: str = "RK45",
    max_step: float = np.inf,
    rtol: float = 1e-6,
    atol: float = 1e-9,
    **kwargs
) -> SimulationResult:
    """
    Solve an ODE system using scipy's solve_ivp.

    This is the main entry point for running simulations with any
    of the kinetic models (SingleMonod, DualMonod, DualMonodLag).

    Args:
        ode_system: An instance of a BaseODE subclass
        initial_conditions: Array of initial state values
        t_span: Tuple of (t_start, t_end)
        t_eval: Optional array of time points for output.
                If None, solver chooses automatically.
        method: Integration method. Options include:
                - "RK45": Explicit Runge-Kutta 4(5) (default, recommended)
                - "RK23": Explicit Runge-Kutta 2(3)
                - "DOP853": Explicit Runge-Kutta 8(5,3)
                - "Radau": Implicit Runge-Kutta (for stiff problems)
                - "BDF": Backward differentiation (for stiff problems)
        max_step: Maximum allowed step size
        rtol: Relative tolerance
        atol: Absolute tolerance
        **kwargs: Additional arguments passed to solve_ivp

    Returns:
        SimulationResult containing time, states, and metadata

    Example:
        >>> from src.core.ode_systems import SingleMonodODE
        >>> ode = SingleMonodODE(qmax=2.5, Ks=400, Ki=25000, Y=0.35, b_decay=0.01)
        >>> result = solve_ode(
        ...     ode_system=ode,
        ...     initial_conditions=[750.0, 1.0],  # [S0, X0]
        ...     t_span=(0, 5),
        ...     t_eval=np.linspace(0, 5, 100)
        ... )
        >>> df = result.to_dataframe()
    """
    # Validate initial conditions
    if len(initial_conditions) != ode_system.n_states:
        raise ValueError(
            f"Expected {ode_system.n_states} initial conditions, "
            f"got {len(initial_conditions)}"
        )

    # Run the solver
    solution = solve_ivp(
        fun=ode_system.derivatives,
        t_span=t_span,
        y0=initial_conditions,
        method=method,
        t_eval=t_eval,
        max_step=max_step,
        rtol=rtol,
        atol=atol,
        **kwargs
    )

    # Package results
    states = {}
    for i, name in enumerate(ode_system.state_names):
        states[name] = solution.y[i]

    return SimulationResult(
        time=solution.t,
        states=states,
        success=solution.success,
        message=solution.message,
        n_evaluations=solution.nfev,
        parameters=ode_system.get_parameters()
    )


def run_simulation_batch(
    ode_system: BaseODE,
    conditions: List[Dict[str, Any]],
    t_span: tuple,
    t_eval: Optional[np.ndarray] = None,
    **solver_kwargs
) -> List[SimulationResult]:
    """
    Run simulations for multiple initial conditions.

    Useful for fitting to experiments with different starting
    substrate concentrations.

    Args:
        ode_system: ODE system to use
        conditions: List of dictionaries, each containing:
                   - 'initial_conditions': array of initial values
                   - 'label': optional identifier for this condition
        t_span: Time span for all simulations
        t_eval: Time points for output
        **solver_kwargs: Additional arguments for solve_ode

    Returns:
        List of SimulationResult objects

    Example:
        >>> conditions = [
        ...     {'initial_conditions': [750.0, 1.0], 'label': '5 mM'},
        ...     {'initial_conditions': [1500.0, 1.0], 'label': '10 mM'},
        ... ]
        >>> results = run_simulation_batch(ode, conditions, (0, 5))
    """
    results = []

    for condition in conditions:
        y0 = condition['initial_conditions']
        result = solve_ode(
            ode_system=ode_system,
            initial_conditions=y0,
            t_span=t_span,
            t_eval=t_eval,
            **solver_kwargs
        )

        # Add label if provided
        if 'label' in condition:
            result.parameters['condition_label'] = condition['label']

        results.append(result)

    return results

# Utility function to create time grids if needed.
def create_time_grid(
    t_start: float,
    t_end: float,
    num_points: int = 10000,
    log_spacing: bool = False
) -> np.ndarray:
    """
    Create a time grid for simulation output.

    Args:
        t_start: Start time (must be >= 0 for log spacing)
        t_end: End time
        num_points: Number of time points
        log_spacing: If True, use logarithmic spacing (good for
                     capturing fast initial dynamics)

    Returns:
        Array of time points
    """
    if log_spacing:
        if t_start <= 0:
            # Start slightly above zero for log spacing
            t_start = t_end / num_points / 10
        return np.logspace(np.log10(t_start), np.log10(t_end), num_points)
    else:
        return np.linspace(t_start, t_end, num_points)
