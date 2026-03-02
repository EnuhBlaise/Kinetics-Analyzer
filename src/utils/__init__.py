"""
Utility functions for conversions, validation, plotting, and theoretical bounds.
"""

from .conversions import convert_time_units, convert_concentration_units
from .validation import validate_positive, validate_bounds, validate_data_columns
from .plotting import plot_fit_results, plot_model_comparison, save_figure
from .theoretical_bounds import (
    parse_formula,
    degree_of_reduction,
    theoretical_yield_max,
    theoretical_oxygen_demand,
    compute_bounds_report,
    compute_from_config,
    TheoreticalBoundsReport,
)

__all__ = [
    "convert_time_units",
    "convert_concentration_units",
    "validate_positive",
    "validate_bounds",
    "validate_data_columns",
    "plot_fit_results",
    "plot_model_comparison",
    "save_figure",
    "parse_formula",
    "degree_of_reduction",
    "theoretical_yield_max",
    "theoretical_oxygen_demand",
    "compute_bounds_report",
    "compute_from_config",
    "TheoreticalBoundsReport",
]
