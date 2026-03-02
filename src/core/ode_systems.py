"""
ODE system definitions for different kinetic models.

This module defines ordinary differential equation systems for three
model configurations:
1. Single Monod: Substrate and biomass only (no oxygen dynamics)
2. Dual Monod: Substrate, biomass, and oxygen with reaeration
3. Dual Monod with Lag: Full model including lag phase

Each ODE class provides:
- Parameter storage
- Derivative calculation (for use with scipy.integrate.solve_ivp)
- State variable names and units
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any
from abc import ABC, abstractmethod

from .monod import single_monod_term, dual_monod_term, lag_phase_factor
from .oxygen import oxygen_utilization_rate, OxygenModel

# ABC helps for defining abstract base classes.
class BaseODE(ABC):
    """
    Abstract base class for ODE systems.

    All kinetic models inherit from this class and must implement
    the derivatives method and provide state variable information.
    """

    @property
    @abstractmethod
    def state_names(self) -> List[str]:
        """Names of state variables (e.g., ['Substrate', 'Biomass'])."""
        pass

    @property
    @abstractmethod
    def state_units(self) -> List[str]:
        """Units of state variables (e.g., ['mg/L', 'mg cells/L'])."""
        pass

    @property
    @abstractmethod
    def n_states(self) -> int:
        """Number of state variables."""
        pass

    @abstractmethod
    def derivatives(self, t: float, y: np.ndarray) -> np.ndarray:
        """
        Calculate time derivatives of state variables.

        Args:
            t: Current time
            y: Array of current state values

        Returns:
            Array of derivatives [dy1/dt, dy2/dt, ...]
        """
        pass

    def __call__(self, t: float, y: np.ndarray) -> np.ndarray:
        """Allow ODE object to be called directly by solvers."""
        return self.derivatives(t, y)


@dataclass
class SingleMonodODE(BaseODE):
    """
    Single Monod ODE system: substrate limitation only.

    This is the simplest model with just substrate consumption and
    biomass growth, without oxygen dynamics.

    State Variables:
        y[0] = S: Substrate concentration (mg/L)
        y[1] = X: Biomass concentration (mg cells/L)

    Equations:
        dS/dt = -(1/Y) * q * X
        dX/dt = (q - b_decay) * X

    where q = qmax * S / (Ks + S) * (1 - S/Ki)

    Attributes:
        qmax: Maximum specific uptake rate (1/day)
        Ks: Half-saturation constant (mg/L)
        Ki: Substrate inhibition constant (mg/L)
        Y: Yield coefficient (mg cells / mg substrate)
        b_decay: Decay/maintenance coefficient (1/day)
    """
    qmax: float
    Ks: float
    Ki: float
    Y: float
    b_decay: float

    @property
    def state_names(self) -> List[str]:
        return ["Substrate", "Biomass"]

    @property
    def state_units(self) -> List[str]:
        return ["mg/L", "mg cells/L"]

    @property
    def n_states(self) -> int:
        return 2

    @property
    def parameter_names(self) -> List[str]:
        return ["qmax", "Ks", "Ki", "Y", "b_decay"]

    def get_parameters(self) -> Dict[str, float]:
        """Return current parameter values as dictionary."""
        return {
            "qmax": self.qmax,
            "Ks": self.Ks,
            "Ki": self.Ki,
            "Y": self.Y,
            "b_decay": self.b_decay
        }

    def derivatives(self, t: float, y: np.ndarray) -> np.ndarray:
        """Calculate derivatives for single Monod system."""
        S, X = y

        # Ensure non-negative values
        S = max(S, 0.0)
        X = max(X, 0.0)

        # Specific uptake rate (Monod with inhibition)
        q = single_monod_term(S, self.qmax, self.Ks, self.Ki)

        # Substrate consumption
        dSdt = -(1.0 / self.Y) * q * X

        # Biomass growth (growth - decay)
        dXdt = (q - self.b_decay) * X

        return np.array([dSdt, dXdt])


@dataclass
class DualMonodODE(BaseODE):
    """
    Dual Monod ODE system: substrate and oxygen limitation with reaeration.

    This model extends single Monod by adding oxygen dynamics,
    suitable for aerobic systems where oxygen can become limiting.

    State Variables:
        y[0] = S: Substrate concentration (mg/L)
        y[1] = X: Biomass concentration (mg cells/L)
        y[2] = O2: Dissolved oxygen concentration (mg/L)

    Equations:
        dS/dt = -(1/Y) * q * X
        dX/dt = (q - b_decay) * X
        dO2/dt = -r_O2 + relaxation_to_equilibrium

    where q = qmax * S/(Ks+S) * (1-S/Ki) * O2/(K_o2+O2)
    """
    qmax: float
    Ks: float
    Ki: float
    Y: float
    b_decay: float
    K_o2: float
    Y_o2: float
    oxygen_model: OxygenModel = field(default_factory=OxygenModel)
    #relaxation_rate: float = 0.1  # Rate of O2 equilibration .

    @property
    def state_names(self) -> List[str]:
        return ["Substrate", "Biomass", "Oxygen"]

    @property
    def state_units(self) -> List[str]:
        return ["mg/L", "mg cells/L", "mg O2/L"]

    @property
    def n_states(self) -> int:
        return 3

    @property
    def parameter_names(self) -> List[str]:
        return ["qmax", "Ks", "Ki", "Y", "b_decay", "K_o2", "Y_o2"]

    def get_parameters(self) -> Dict[str, float]:
        """Return current parameter values as dictionary."""
        return {
            "qmax": self.qmax,
            "Ks": self.Ks,
            "Ki": self.Ki,
            "Y": self.Y,
            "b_decay": self.b_decay,
            "K_o2": self.K_o2,
            "Y_o2": self.Y_o2
        }

    def derivatives(self, t: float, y: np.ndarray) -> np.ndarray:
        """Calculate derivatives for dual Monod system with oxygen."""
        S, X, O2 = y

        # Ensure non-negative values and enforce o2_min floor from oxygen model
        S = max(S, 0.0)
        X = max(X, 0.0)
        O2 = max(O2, self.oxygen_model.o2_min)

        # Specific uptake rate (dual Monod)
        q = dual_monod_term(S, O2, self.qmax, self.Ks, self.Ki, self.K_o2)

        # Substrate consumption
        dSdt = -(1.0 / self.Y) * q * X

        # Biomass growth
        dXdt = (q - self.b_decay) * X

        # Oxygen dynamics with reaeration
        r_o2 = oxygen_utilization_rate(q, X, self.Y_o2)
        target_o2 = self.oxygen_model.get_equilibrium_oxygen(r_o2)
        dO2dt = -r_o2 + (target_o2 - O2) * 0.1 # relaxation term


        return np.array([dSdt, dXdt, dO2dt])


@dataclass
class DualMonodLagODE(BaseODE):
    """
    Dual Monod ODE system with lag phase.

    This is the full model including substrate limitation, oxygen dynamics,
    reaeration, and a lag phase to account for microbial adaptation.

    The lag phase factor multiplies growth-related terms but NOT decay,
    representing that cells adapt before active growth but still have
    maintenance requirements.

    State Variables:
        y[0] = S: Substrate concentration (mg/L)
        y[1] = X: Biomass concentration (mg cells/L)
        y[2] = O2: Dissolved oxygen concentration (mg/L)

    Equations:
        dS/dt = -(1/Y) * q * X * lag_factor
        dX/dt = q * lag_factor * X - b_decay * X
        dO2/dt = -r_O2 * lag_factor + relaxation
    """
    qmax: float
    Ks: float
    Ki: float
    Y: float
    b_decay: float
    K_o2: float
    Y_o2: float
    lag_time: float
    oxygen_model: OxygenModel = field(default_factory=OxygenModel)
    #relaxation_rate: float = 0.1
    lag_steepness: float = 14.0

    @property
    def state_names(self) -> List[str]:
        return ["Substrate", "Biomass", "Oxygen"]

    @property
    def state_units(self) -> List[str]:
        return ["mg/L", "mg cells/L", "mg O2/L"]

    @property
    def n_states(self) -> int:
        return 3

    @property
    def parameter_names(self) -> List[str]:
        return ["qmax", "Ks", "Ki", "Y", "b_decay", "K_o2", "Y_o2", "lag_time"]

    def get_parameters(self) -> Dict[str, float]:
        """Return current parameter values as dictionary."""
        return {
            "qmax": self.qmax,
            "Ks": self.Ks,
            "Ki": self.Ki,
            "Y": self.Y,
            "b_decay": self.b_decay,
            "K_o2": self.K_o2,
            "Y_o2": self.Y_o2,
            "lag_time": self.lag_time
        }

    def derivatives(self, t: float, y: np.ndarray) -> np.ndarray:
        """Calculate derivatives for dual Monod system with lag phase."""
        S, X, O2 = y

        # Ensure non-negative values and enforce o2_min floor from oxygen model
        S = max(S, 0.0)
        X = max(X, 0.0)
        O2 = max(O2, self.oxygen_model.o2_min)

        # Calculate lag phase factor
        lag_factor = lag_phase_factor(t, self.lag_time, self.lag_steepness)

        # Specific uptake rate (dual Monod)
        q = dual_monod_term(S, O2, self.qmax, self.Ks, self.Ki, self.K_o2)

        # Substrate consumption (affected by lag)
        dSdt = -(1.0 / self.Y) * q * X * lag_factor

        # Biomass: growth affected by lag, decay is not
        dXdt = q * X - self.b_decay * X * lag_factor

        # Oxygen dynamics (consumption affected by lag)
        r_o2 = oxygen_utilization_rate(q, X, self.Y_o2) * lag_factor
        target_o2 = self.oxygen_model.get_equilibrium_oxygen(r_o2)
        dO2dt = -r_o2 + (target_o2 - O2)

        # Prevent O2 from going below o2_min: if at floor and derivative is negative, clamp to 0
        if O2 <= self.oxygen_model.o2_min and dO2dt < 0:
            dO2dt = 0.0

        return np.array([dSdt, dXdt, dO2dt])

    def get_lag_factor(self, t: float) -> float:
        """Get the lag phase factor at a given time (useful for plotting)."""
        return lag_phase_factor(t, self.lag_time, self.lag_steepness)


@dataclass
class SingleMonodLagODE(BaseODE):
    """
    Single Monod ODE system with lag phase.

    This model extends the single Monod model by adding a lag phase to
    account for microbial adaptation, without oxygen dynamics.

    State Variables:
        y[0] = S: Substrate concentration (mg/L)
        y[1] = X: Biomass concentration (mg cells/L)

    Equations:
        dS/dt = -(1/Y) * q * X * lag_factor
        dX/dt = q * X - b_decay * X * lag_factor

    where q = qmax * S / (Ks + S) * (1 - S/Ki)
          lag_factor = sigmoid(t, lag_time)

    Attributes:
        qmax: Maximum specific uptake rate (1/day)
        Ks: Half-saturation constant (mg/L)
        Ki: Substrate inhibition constant (mg/L)
        Y: Yield coefficient (mg cells / mg substrate)
        b_decay: Decay/maintenance coefficient (1/day)
        lag_time: Duration of the lag phase (days)
        lag_steepness: Steepness of the lag sigmoid transition
    """
    qmax: float
    Ks: float
    Ki: float
    Y: float
    b_decay: float
    lag_time: float
    lag_steepness: float = 14.0

    @property
    def state_names(self) -> List[str]:
        return ["Substrate", "Biomass"]

    @property
    def state_units(self) -> List[str]:
        return ["mg/L", "mg cells/L"]

    @property
    def n_states(self) -> int:
        return 2

    @property
    def parameter_names(self) -> List[str]:
        return ["qmax", "Ks", "Ki", "Y", "b_decay", "lag_time"]

    def get_parameters(self) -> Dict[str, float]:
        """Return current parameter values as dictionary."""
        return {
            "qmax": self.qmax,
            "Ks": self.Ks,
            "Ki": self.Ki,
            "Y": self.Y,
            "b_decay": self.b_decay,
            "lag_time": self.lag_time,
        }

    def derivatives(self, t: float, y: np.ndarray) -> np.ndarray:
        """Calculate derivatives for single Monod system with lag phase."""
        S, X = y

        # Ensure non-negative values
        S = max(S, 0.0)
        X = max(X, 0.0)

        # Calculate lag phase factor
        lag_factor = lag_phase_factor(t, self.lag_time, self.lag_steepness)

        # Specific uptake rate (Monod with inhibition)
        q = single_monod_term(S, self.qmax, self.Ks, self.Ki)

        # Substrate consumption (affected by lag)
        dSdt = -(1.0 / self.Y) * q * X * lag_factor

        # Biomass: growth affected by lag, decay is not
        dXdt = q * X - self.b_decay * X * lag_factor

        return np.array([dSdt, dXdt])

    def get_lag_factor(self, t: float) -> float:
        """Get the lag phase factor at a given time (useful for plotting)."""
        return lag_phase_factor(t, self.lag_time, self.lag_steepness)
