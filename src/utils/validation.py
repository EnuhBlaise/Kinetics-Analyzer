"""
Input validation utilities for kinetic parameter estimation.

This module provides validation functions for:
- Parameter values (positivity, bounds)
- Data columns and format
- Configuration files
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


def validate_positive(
    value: float,
    name: str,
    allow_zero: bool = False
) -> float:
    """
    Validate that a value is positive (or non-negative).

    Args:
        value: The value to check
        name: Parameter name (for error messages)
        allow_zero: If True, zero is allowed

    Returns:
        The validated value

    Raises:
        ValidationError: If value is negative (or zero when not allowed)
    """
    if allow_zero:
        if value < 0:
            raise ValidationError(f"{name} must be non-negative, got {value}")
    else:
        if value <= 0:
            raise ValidationError(f"{name} must be positive, got {value}")
    return value


def validate_bounds(
    value: float,
    name: str,
    bounds: Tuple[float, float]
) -> float:
    """
    Validate that a value is within specified bounds.

    Args:
        value: The value to check
        name: Parameter name (for error messages)
        bounds: Tuple of (lower_bound, upper_bound)

    Returns:
        The validated value

    Raises:
        ValidationError: If value is outside bounds
    """
    lower, upper = bounds
    if value < lower or value > upper:
        raise ValidationError(
            f"{name} must be in [{lower}, {upper}], got {value}"
        )
    return value


def validate_parameter_set(
    parameters: Dict[str, float],
    bounds: Dict[str, Tuple[float, float]],
    required: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Validate a complete set of kinetic parameters.

    Args:
        parameters: Dictionary of parameter names and values
        bounds: Dictionary of parameter bounds
        required: List of required parameter names. If None, all params
                  in bounds are required.

    Returns:
        Validated parameters dictionary

    Raises:
        ValidationError: If any parameter is missing or out of bounds
    """
    if required is None:
        required = list(bounds.keys())

    # Check for missing parameters
    missing = set(required) - set(parameters.keys())
    if missing:
        raise ValidationError(f"Missing required parameters: {missing}")

    # Validate each parameter
    validated = {}
    for name, value in parameters.items():
        if name in bounds:
            validate_bounds(value, name, bounds[name])
        validated[name] = value

    return validated


def validate_data_columns(
    data: pd.DataFrame,
    required_columns: List[str],
    substrate_name: str = "Substrate"
) -> None:
    """
    Validate that a DataFrame has required columns.

    Args:
        data: The DataFrame to validate
        required_columns: List of column names that must be present
        substrate_name: Name of substrate (for error messages)

    Raises:
        ValidationError: If required columns are missing
    """
    missing = set(required_columns) - set(data.columns)
    if missing:
        raise ValidationError(
            f"Missing columns for {substrate_name}: {missing}\n"
            f"Available columns: {list(data.columns)}"
        )


def validate_experimental_data(
    data: pd.DataFrame,
    time_column: str = "Time (days)",
    substrate_pattern: str = "_Substrate",
    biomass_pattern: str = "_Biomass"
) -> Tuple[List[str], List[str]]:
    """
    Validate experimental data format and return column names.

    Args:
        data: Experimental data DataFrame
        time_column: Name of time column
        substrate_pattern: Pattern to identify substrate columns
        biomass_pattern: Pattern to identify biomass columns

    Returns:
        Tuple of (substrate_columns, biomass_columns)

    Raises:
        ValidationError: If data format is invalid
    """
    # Check time column
    if time_column not in data.columns:
        # Try to find a similar column
        time_candidates = [c for c in data.columns if "time" in c.lower()]
        if time_candidates:
            raise ValidationError(
                f"Time column '{time_column}' not found. "
                f"Did you mean: {time_candidates}?"
            )
        raise ValidationError(f"Time column '{time_column}' not found")

    # Find substrate and biomass columns
    substrate_cols = [c for c in data.columns if substrate_pattern in c or "Glucose" in c or "Xylose" in c]
    biomass_cols = [c for c in data.columns if biomass_pattern in c]

    if not substrate_cols:
        raise ValidationError(
            f"No substrate columns found (looking for pattern: {substrate_pattern})"
        )
    if not biomass_cols:
        raise ValidationError(
            f"No biomass columns found (looking for pattern: {biomass_pattern})"
        )

    # Check for NaN values
    for col in [time_column] + substrate_cols + biomass_cols:
        if data[col].isna().any():
            nan_count = data[col].isna().sum()
            raise ValidationError(
                f"Column '{col}' contains {nan_count} NaN values"
            )

    return substrate_cols, biomass_cols


def validate_initial_conditions(
    conditions: np.ndarray,
    n_states: int,
    state_names: List[str]
) -> np.ndarray:
    """
    Validate initial conditions array.

    Args:
        conditions: Array of initial values
        n_states: Expected number of states
        state_names: Names of state variables

    Returns:
        Validated conditions array

    Raises:
        ValidationError: If conditions are invalid
    """
    conditions = np.asarray(conditions)

    if len(conditions) != n_states:
        raise ValidationError(
            f"Expected {n_states} initial conditions for {state_names}, "
            f"got {len(conditions)}"
        )

    # Check for negative values
    for i, (val, name) in enumerate(zip(conditions, state_names)):
        if val < 0:
            raise ValidationError(
                f"Initial {name} cannot be negative (got {val})"
            )

    return conditions


def validate_time_span(
    t_span: Tuple[float, float]
) -> Tuple[float, float]:
    """
    Validate time span for simulation.

    Args:
        t_span: Tuple of (start_time, end_time)

    Returns:
        Validated time span

    Raises:
        ValidationError: If time span is invalid
    """
    t_start, t_end = t_span

    if t_start < 0:
        raise ValidationError(f"Start time cannot be negative (got {t_start})")
    if t_end <= t_start:
        raise ValidationError(
            f"End time must be greater than start time "
            f"(start={t_start}, end={t_end})"
        )

    return t_span


def validate_config_structure(config: Dict[str, Any]) -> None:
    """
    Validate the structure of a configuration dictionary.

    Args:
        config: Configuration dictionary

    Raises:
        ValidationError: If config structure is invalid
    """
    required_sections = ["substrate", "initial_guesses", "bounds"]

    for section in required_sections:
        if section not in config:
            raise ValidationError(f"Config missing required section: {section}")

    # Validate substrate section
    substrate = config["substrate"]
    if "name" not in substrate:
        raise ValidationError("Config substrate section must have 'name'")
    if "molecular_weight" not in substrate:
        raise ValidationError("Config substrate section must have 'molecular_weight'")

    # Validate that bounds have corresponding initial guesses
    guesses = set(config["initial_guesses"].keys())
    bounds = set(config["bounds"].keys())

    missing_bounds = guesses - bounds
    if missing_bounds:
        raise ValidationError(
            f"Parameters have initial guesses but no bounds: {missing_bounds}"
        )
