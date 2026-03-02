"""
Core module containing Monod kinetics, ODE systems, and solvers.
"""

from .monod import single_monod_term, dual_monod_term, lag_phase_factor
from .oxygen import oxygen_utilization_rate, update_oxygen, OxygenModel
from .ode_systems import SingleMonodODE, DualMonodODE, DualMonodLagODE
from .solvers import solve_ode, SimulationResult

__all__ = [
    "single_monod_term",
    "dual_monod_term",
    "lag_phase_factor",
    "oxygen_utilization_rate",
    "update_oxygen",
    "OxygenModel",
    "SingleMonodODE",
    "DualMonodODE",
    "DualMonodLagODE",
    "solve_ode",
    "SimulationResult",
]
