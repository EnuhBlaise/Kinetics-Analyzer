"""
Unit conversion utilities for kinetic parameters.

This module provides functions to convert kinetic parameters between
different time units (days, hours, minutes) and concentration units
(mg/L, mM, g/L).
"""

from typing import Dict, Union, Tuple

# Time conversion factors (relative to days)
TIME_FACTORS = {
    "days": 1.0,
    "day": 1.0,
    "d": 1.0,
    "hours": 24.0,
    "hour": 24.0,
    "h": 24.0,
    "hr": 24.0,
    "minutes": 1440.0,
    "minute": 1440.0,
    "min": 1440.0,
    "m": 1440.0,
    "seconds": 86400.0,
    "second": 86400.0,
    "sec": 86400.0,
    "s": 86400.0,
}

# Concentration factors (relative to mg/L)
CONC_FACTORS = {
    "mg/L": 1.0,
    "mg/l": 1.0,
    "ppm": 1.0,
    "g/L": 0.001,
    "g/l": 0.001,
    "ug/L": 1000.0,
    "ug/l": 1000.0,
    "ppb": 1000.0,
}


def convert_time_units(
    value: float,
    from_unit: str,
    to_unit: str
) -> float:
    """
    Convert a value between time units.

    Supports conversion between days, hours, minutes, and seconds.

    Args:
        value: The numeric value to convert
        from_unit: Source time unit (e.g., "days", "hours", "min")
        to_unit: Target time unit

    Returns:
        Converted value

    Example:
        >>> convert_time_units(2.5, "days", "hours")
        60.0
        >>> convert_time_units(1440, "minutes", "days")
        1.0
    """
    from_unit = from_unit.lower().strip()
    to_unit = to_unit.lower().strip()

    if from_unit not in TIME_FACTORS:
        raise ValueError(f"Unknown time unit: {from_unit}. Supported: {list(TIME_FACTORS.keys())}")
    if to_unit not in TIME_FACTORS:
        raise ValueError(f"Unknown time unit: {to_unit}. Supported: {list(TIME_FACTORS.keys())}")

    # Convert to days first, then to target unit
    value_in_days = value / TIME_FACTORS[from_unit]
    return value_in_days * TIME_FACTORS[to_unit]


def convert_concentration_units(
    value: float,
    from_unit: str,
    to_unit: str,
    molecular_weight: float = None
) -> float:
    """
    Convert a value between concentration units.

    Supports mg/L, g/L, ug/L, and mM (requires molecular weight).

    Args:
        value: The numeric value to convert
        from_unit: Source concentration unit
        to_unit: Target concentration unit
        molecular_weight: Molecular weight (g/mol), required for mM conversions

    Returns:
        Converted value

    Example:
        >>> convert_concentration_units(180.16, "mg/L", "mM", molecular_weight=180.16)
        1.0
    """
    from_unit = from_unit.strip()
    to_unit = to_unit.strip()

    # Handle mM (millimolar) conversions
    if "mM" in from_unit or "mM" in to_unit:
        if molecular_weight is None:
            raise ValueError("Molecular weight required for mM conversions")

        if "mM" in from_unit:
            # mM to mg/L: multiply by MW
            value_mg_L = value * molecular_weight
        else:
            # mg/L already
            from_lower = from_unit.lower()
            if from_lower not in CONC_FACTORS:
                raise ValueError(f"Unknown concentration unit: {from_unit}")
            value_mg_L = value / CONC_FACTORS[from_lower]

        if "mM" in to_unit:
            # Convert mg/L to mM
            return value_mg_L / molecular_weight
        else:
            to_lower = to_unit.lower()
            if to_lower not in CONC_FACTORS:
                raise ValueError(f"Unknown concentration unit: {to_unit}")
            return value_mg_L * CONC_FACTORS[to_lower]

    # Standard conversions (not involving mM)
    from_lower = from_unit.lower()
    to_lower = to_unit.lower()

    if from_lower not in CONC_FACTORS:
        raise ValueError(f"Unknown concentration unit: {from_unit}")
    if to_lower not in CONC_FACTORS:
        raise ValueError(f"Unknown concentration unit: {to_unit}")

    # Convert to mg/L first, then to target
    value_mg_L = value / CONC_FACTORS[from_lower]
    return value_mg_L * CONC_FACTORS[to_lower]


def convert_kinetic_parameters(
    parameters: Dict[str, float],
    from_time_unit: str,
    to_time_unit: str
) -> Dict[str, float]:
    """
    Convert all time-dependent kinetic parameters between time units.

    This function identifies which parameters have time dependencies
    and converts them appropriately:
    - qmax: rate, converts directly with time
    - b_decay: rate, converts directly with time
    - Ks, Ki, K_o2, Y, Y_o2: not time-dependent, unchanged

    Args:
        parameters: Dictionary of parameter names and values
        from_time_unit: Source time unit
        to_time_unit: Target time unit

    Returns:
        Dictionary with converted parameters

    Example:
        >>> params = {"qmax": 2.5, "Ks": 400, "b_decay": 0.01}
        >>> convert_kinetic_parameters(params, "days", "hours")
        {"qmax": 0.104..., "Ks": 400, "b_decay": 0.000416...}
    """
    # Parameters that have time in denominator (per time)
    time_dependent = {"qmax", "b_decay"}

    conversion_factor = convert_time_units(1.0, from_time_unit, to_time_unit)

    converted = {}
    for name, value in parameters.items():
        if name.lower() in time_dependent:
            # Rate parameters: divide by conversion factor
            # (e.g., 2.5/day = 2.5/24 per hour)
            converted[name] = value / conversion_factor
        else:
            # Non-time-dependent parameters
            converted[name] = value

    return converted


def mM_to_mgL(concentration_mM: float, molecular_weight: float) -> float:
    """
    Convert concentration from millimolar to mg/L.

    Args:
        concentration_mM: Concentration in mM
        molecular_weight: Molecular weight in g/mol

    Returns:
        Concentration in mg/L

    Example:
        >>> mM_to_mgL(5, 150.13)  # 5 mM xylose
        750.65
    """
    return concentration_mM * molecular_weight


def mgL_to_mM(concentration_mgL: float, molecular_weight: float) -> float:
    """
    Convert concentration from mg/L to millimolar.

    Args:
        concentration_mgL: Concentration in mg/L
        molecular_weight: Molecular weight in g/mol

    Returns:
        Concentration in mM

    Example:
        >>> mgL_to_mM(750.65, 150.13)  # mg/L xylose
        5.0
    """
    return concentration_mgL / molecular_weight


def get_time_unit_info(unit: str) -> Tuple[str, float]:
    """
    Get standardized name and conversion factor for a time unit.

    Args:
        unit: Time unit string (can be abbreviated)

    Returns:
        Tuple of (standardized name, factor relative to days)
    """
    unit_lower = unit.lower().strip()

    if unit_lower not in TIME_FACTORS:
        raise ValueError(f"Unknown time unit: {unit}")

    # Standardize names
    if unit_lower in ("days", "day", "d"):
        return "days", TIME_FACTORS[unit_lower]
    elif unit_lower in ("hours", "hour", "h", "hr"):
        return "hours", TIME_FACTORS[unit_lower]
    elif unit_lower in ("minutes", "minute", "min", "m"):
        return "minutes", TIME_FACTORS[unit_lower]
    else:
        return "seconds", TIME_FACTORS[unit_lower]


# Common molecular weights for reference
MOLECULAR_WEIGHTS = {
    "glucose": 180.16,
    "xylose": 150.13,
    "fructose": 180.16,
    "sucrose": 342.30,
    "lactose": 342.30,
    "maltose": 342.30,
    "acetate": 59.04,
    "ethanol": 46.07,
    "glycerol": 92.09,
}


def get_molecular_weight(substrate_name: str) -> float:
    """
    Get molecular weight for common substrates.

    Args:
        substrate_name: Name of substrate (case-insensitive)

    Returns:
        Molecular weight in g/mol

    Raises:
        ValueError: If substrate not in database
    """
    name = substrate_name.lower().strip()
    if name not in MOLECULAR_WEIGHTS:
        available = ", ".join(MOLECULAR_WEIGHTS.keys())
        raise ValueError(f"Unknown substrate: {substrate_name}. Available: {available}")
    return MOLECULAR_WEIGHTS[name]
