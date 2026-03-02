"""
Oxygen dynamics and reaeration model for aerobic microbial systems.

This module handles dissolved oxygen (DO) dynamics including:
- Oxygen utilization rate calculation
- Reaeration from atmosphere
- Linear oxygen model with min/max bounds

The oxygen model assumes a linear relationship between oxygen consumption
and reaeration, bounded by saturation (o2_max) and minimum (o2_min) values.
"""

import numpy as np
from dataclasses import dataclass
from typing import Union

Numeric = Union[int, float, np.ndarray] # Accept int, float or arrays from the user. Consider them as numeric values in each case.


@dataclass
class OxygenModel:
    """
    Configuration for the oxygen dynamics model.

    Attributes:
        o2_max: Maximum (saturation) dissolved oxygen (mg/L), typically ~8 mg/L at 25C
        o2_min: Minimum dissolved oxygen (mg/L), typically near 0
        reaeration_rate: Base reaeration rate (mg O2/L/day)
        o2_range: Multiplier for reaeration range (dimensionless)

    The linear model calculates equilibrium DO based on:
        - If consumption < reaeration: DO stays at o2_max
        - If consumption > reaeration * o2_range: DO drops to o2_min
        - Otherwise: linear interpolation between these bounds
    """
    o2_max: float = 8.0
    o2_min: float = 0.1
    reaeration_rate: float = 15.0
    o2_range: float = 8.0

    @classmethod
    def from_config(cls, config: dict) -> 'OxygenModel':
        """Create OxygenModel from configuration dictionary."""
        return cls(
            o2_max=config.get('o2_max', 8.0),
            o2_min=config.get('o2_min', 0.1),
            reaeration_rate=config.get('reaeration_rate', 15.0),
            o2_range=config.get('o2_range', 8.0)
        )

    def __post_init__(self):
        """Calculate derived parameters for the linear oxygen model."""
        # Linear model coefficients: O2 = a * ro2_rate + b
        # At ro2 = reaeration_rate: O2 = o2_max
        # At ro2 = reaeration_rate * o2_range: O2 = o2_min

        self.o2_a = (self.o2_min - self.o2_max) / (
            self.reaeration_rate * self.o2_range - self.reaeration_rate
        )
        self.o2_b = self.o2_max - self.o2_a * self.reaeration_rate

    def get_equilibrium_oxygen(self, ro2_rate: Numeric) -> Numeric:
        """
        Calculate equilibrium dissolved oxygen based on consumption rate.

        Args:
            ro2_rate: Oxygen utilization rate (mg O2/L/day)

        Returns:
            Equilibrium dissolved oxygen concentration (mg/L)
        """
        return update_oxygen(
            ro2_rate,
            self.o2_max,
            self.o2_min,
            self.o2_a,
            self.o2_b,
            self.reaeration_rate,
            self.o2_range
        )


def oxygen_utilization_rate(
    monod_term: Numeric,
    biomass: Numeric,
    Y_o2: float
) -> Numeric:
    """
    Calculate the volumetric oxygen utilization rate.

    The oxygen consumption is proportional to the substrate uptake rate
    and biomass concentration, scaled by the oxygen yield coefficient.

    Args:
        monod_term: Specific uptake rate from Monod kinetics
        biomass: Biomass concentration (mg cells/L)
        Y_o2: Oxygen yield coefficient (mg O2 consumed per mg substrate consumed)

    Returns:
        Volumetric oxygen utilization rate (mg O2/L/time)

    Mathematical Form:
        r_O2 = Y_O2 * q * X
        where q is the specific uptake rate and X is biomass
    """
    return Y_o2 * monod_term * biomass


def update_oxygen(
    ro2_rate: Numeric,
    o2_max: float,
    o2_min: float,
    o2_a: float,
    o2_b: float,
    reaeration_rate: float,
    o2_range: float
) -> Numeric:
    """
    Update dissolved oxygen concentration using the linear reaeration model.

    This function implements a piecewise linear model for DO based on
    oxygen consumption rate, assuming quasi-steady-state reaeration.

    Args:
        ro2_rate: Oxygen utilization rate (mg O2/L/day)
        o2_max: Maximum (saturation) DO (mg/L)
        o2_min: Minimum DO (mg/L)
        o2_a: Linear model slope coefficient
        o2_b: Linear model intercept coefficient
        reaeration_rate: Base reaeration rate (mg O2/L/day)
        o2_range: Multiplier for max consumption range

    Returns:
        Equilibrium dissolved oxygen concentration (mg/L)

    Model Regions:
        1. Low consumption (ro2 < reaeration): O2 = o2_max (fully aerated)
        2. High consumption (ro2 > reaeration * range): O2 = o2_min (oxygen limited)
        3. Intermediate: O2 = a * ro2 + b (linear interpolation)
    """
    #vectorized input if there are multiple rates to process. # may not be necessary.
    #allocate output array for downstream use.

    if isinstance(ro2_rate, np.ndarray): 
        result = np.zeros_like(ro2_rate) 

        # Region 1: Low consumption
        low_mask = ro2_rate < reaeration_rate
        result[low_mask] = o2_max

        # Region 2: High consumption
        high_threshold = reaeration_rate * o2_range
        high_mask = ro2_rate > high_threshold
        result[high_mask] = o2_min

        # Region 3: Linear region
        linear_mask = ~low_mask & ~high_mask
        result[linear_mask] = o2_a * ro2_rate[linear_mask] + o2_b

        # Enforce minimum O2 floor - reset to o2_min if below
        result = np.maximum(result, o2_min)

        return result
    else:
        # Scalar input. This is what would be used in most cases.
        if ro2_rate < reaeration_rate:
            return o2_max
        elif ro2_rate > reaeration_rate * o2_range:
            return o2_min
        else:
            result = o2_a * ro2_rate + o2_b
            # Enforce minimum O2 floor - reset to o2_min if below
            return max(result, o2_min)

#Using this alternative reaeration model function is optional. Mostly for experimentation. Use it only if you want to try a different approach.
def calculate_reaeration_flux(             
    current_o2: float,
    o2_saturation: float,
    kla: float
) -> float:
    """
    Calculate oxygen reaeration flux using mass transfer model.

    This is an alternative to the linear model, using standard
    gas-liquid mass transfer principles.

    Args:
        current_o2: Current dissolved oxygen (mg/L)
        o2_saturation: Saturation concentration (mg/L)
        kla: Volumetric mass transfer coefficient (1/time)

    Returns:
        Oxygen flux into liquid (mg O2/L/time)

    Mathematical Form:
        flux = kla * (O2_sat - O2)
    """
    return kla * (o2_saturation - current_o2)
