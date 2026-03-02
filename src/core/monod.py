"""
Monod kinetic functions for microbial growth modeling.

This module provides the core kinetic terms used in microbial growth models:
- Single Monod term: substrate limitation with optional inhibition
- Dual Monod term: combined substrate and oxygen limitation
- Lag phase factor: sigmoid function for delayed growth onset

Mathematical Background:
    The Monod equation describes microbial growth rate as a function of
    substrate concentration, analogous to Michaelis-Menten enzyme kinetics.

    Basic Monod: mu = mu_max * S / (Ks + S)

    With substrate inhibition (Haldane model):
    mu = mu_max * S / (Ks + S + S²/Ki)
"""

import numpy as np
from typing import Union

# Type alias for numeric inputs (scalars or arrays)
Numeric = Union[float, np.ndarray]


def single_monod_term(
    substrate: Numeric,
    qmax: float,
    Ks: float,
    Ki: float = None
) -> Numeric:
    """
    Calculate the single Monod kinetic term with optional substrate inhibition.

    This function computes the specific uptake rate based on substrate
    concentration using Monod kinetics with optional Haldane inhibition.

    Args:
        substrate: Substrate concentration (mg/L or mM)
        qmax: Maximum specific uptake rate (substrate units per biomass per time)
        Ks: Half-saturation constant (same units as substrate)
        Ki: Substrate inhibition constant (same units as substrate).
            If None, no inhibition term is applied.

    Returns:
        Specific uptake rate (same time units as qmax)

    Mathematical Form:
        Without inhibition: q = qmax * S / (Ks + S)
        With inhibition: q = qmax * S / (Ks + S + (S² / Ki))

    Example:
        >>> rate = single_monod_term(substrate=500, qmax=2.5, Ks=400, Ki=25000)
        >>> print(f"Uptake rate: {rate:.4f}")
    """
    # Ensure substrate is non-negative
    substrate = np.maximum(substrate, 0.0)

    # Apply Haldane model if Ki is specified, otherwise basic Monod
    if Ki is not None and Ki > 0:
        # Haldane equation: q = qmax * S / (Ks + S + S²/Ki)
        monod = qmax * substrate / (Ks + substrate + substrate**2 / Ki)
    else:
        # Basic Monod term: q = qmax * S / (Ks + S)
        monod = qmax * substrate / (Ks + substrate)

    return monod


def dual_monod_term(
    substrate: Numeric,
    oxygen: Numeric,
    qmax: float,
    Ks: float,
    Ki: float,
    K_o2: float
) -> Numeric:
    """
    Calculate the dual Monod term combining substrate and oxygen limitation.

    This function extends the single Monod term by multiplying with an
    oxygen limitation factor, modeling aerobic microbial growth where
    both substrate and oxygen can be limiting.

    Args:
        substrate: Substrate concentration (mg/L)
        oxygen: Dissolved oxygen concentration (mg/L)
        qmax: Maximum specific uptake rate
        Ks: Half-saturation constant for substrate
        Ki: Substrate inhibition constant
        K_o2: Half-saturation constant for oxygen (mg/L)

    Returns:
        Specific uptake rate accounting for both substrate and O2 limitation

    Mathematical Form:
        q = single_monod(S) * O2 / (K_o2 + O2)

    Example:
        >>> rate = dual_monod_term(
        ...     substrate=500, oxygen=6.0, qmax=2.5,
        ...     Ks=400, Ki=25000, K_o2=0.15
        ... )
    """
    # Get single Monod term for substrate
    substrate_term = single_monod_term(substrate, qmax, Ks, Ki)

    # Oxygen limitation term (simple Monod, no inhibition)
    oxygen = np.maximum(oxygen, 0.0)
    oxygen_term = oxygen / (K_o2 + oxygen)

    return substrate_term * oxygen_term


def lag_phase_factor(
    time: Numeric,
    lag_time: float,
    steepness: float = 10.0
) -> Numeric:
    """
    Calculate the lag phase factor for delayed microbial growth.

    The lag phase represents the adaptation period before exponential growth
    begins. This function uses a sigmoid (logistic) function to create a
    smooth transition from no growth to full growth activity.

    Args:
        time: Current time (days or other time unit)
        lag_time: Duration of the lag phase (same units as time)
        steepness: Controls how sharp the transition is (default=14).
                   Higher values = sharper transition.

    Returns:
        Factor between 0 and 1:
        - 0 at time=0 (no growth)
        - ~0.5 at time=lag_time/2 (half growth)
        - ~1 at time>=lag_time (full growth)

    Mathematical Form:
        f(t) = 1 / (1 + exp(-k * (t - lag_time/2) / lag_time))
        where k is the steepness parameter

    Example:
        >>> factor = lag_phase_factor(time=2.0, lag_time=3.2)
        >>> print(f"Growth factor at day 2: {factor:.2%}")
    """
    x = steepness * (time - lag_time / 2) / lag_time
    return 1.0 / (1.0 + np.exp(-x))

# Alternative lag phase function: step function. I Used this mostly for experimentation and comparison.
def step_lag_factor(time: Numeric, lag_time: float) -> Numeric:
    """
    Simple step function for lag phase (alternative to sigmoid).

    This provides a sharp transition at the lag time, useful for
    comparison or when smooth transitions are not needed.

    Args:
        time: Current time
        lag_time: Duration of lag phase

    Returns:
        0 if time < lag_time, 1 otherwise
    """
    if isinstance(time, np.ndarray):
        return np.where(time >= lag_time, 1.0, 0.0)
    return 1.0 if time >= lag_time else 0.0
