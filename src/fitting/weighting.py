"""
Weighting strategies for handling heteroscedasticity in parameter estimation.

The Problem:
When fitting Monod kinetics across multiple substrate concentrations (e.g., 5mM vs 20mM),
the high-concentration flasks produce larger absolute biomass values. Standard SSE
optimization will over-weight these large values, effectively ignoring the low-concentration
data that is crucial for determining Ks (half-saturation constant).

The Solution:
Weight each condition's contribution to the total error by normalizing against
the scale of that condition's data. This ensures all conditions contribute
meaningfully to the parameter estimates.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import numpy as np


class WeightingStrategy(ABC):
    """
    Abstract base class for weighting strategies.

    Weighting strategies determine how errors from different experimental
    conditions are combined during global parameter fitting.
    """

    @abstractmethod
    def compute_weights(self, conditions: List[Dict[str, Any]]) -> np.ndarray:
        """
        Compute weights for each condition.

        Args:
            conditions: List of condition dictionaries, each containing:
                - 'biomass': np.ndarray of biomass measurements
                - 'substrate': np.ndarray of substrate measurements
                - 'label': optional condition identifier

        Returns:
            Array of weights, one per condition. Weights are normalized
            so they sum to the number of conditions (mean weight = 1).
        """
        pass

    def apply_weight(self, error: float, weight: float) -> float:
        """
        Apply a weight to an error term.

        Default implementation is simple multiplication.
        Subclasses may override for more complex weighting schemes.

        Args:
            error: The unweighted error (e.g., SSE) for one condition
            weight: The weight for this condition

        Returns:
            Weighted error
        """
        return error * weight

    @abstractmethod
    def get_name(self) -> str:
        """Return the name of this weighting strategy."""
        pass


class UniformWeighting(WeightingStrategy):
    """
    All conditions contribute equally regardless of data scale.

    This is the naive approach that ignores heteroscedasticity.
    Use only when all conditions have similar data ranges.
    """

    def compute_weights(self, conditions: List[Dict[str, Any]]) -> np.ndarray:
        """Return uniform weights of 1.0 for all conditions."""
        return np.ones(len(conditions))

    def get_name(self) -> str:
        return "uniform"


class MaxValueWeighting(WeightingStrategy):
    """
    Weight each condition inversely proportional to max biomass squared.

    This is the recommended approach for Monod kinetics fitting.
    It ensures low-substrate conditions (which define Ks) are not
    dominated by high-substrate conditions (which define μmax).

    Mathematical basis:
        weight_i = 1 / max(X_i)²

    After normalization:
        Low-concentration flasks with small biomass → higher weight
        High-concentration flasks with large biomass → lower weight
    """

    def __init__(self, use_substrate: bool = False, combine_method: str = "biomass_only"):
        """
        Initialize weighting strategy.

        Args:
            use_substrate: If True, also consider substrate range in weighting
            combine_method: How to combine biomass and substrate weights
                - "biomass_only": Only use biomass (default, recommended)
                - "substrate_only": Only use substrate
                - "geometric_mean": sqrt(w_biomass * w_substrate)
                - "arithmetic_mean": (w_biomass + w_substrate) / 2
        """
        self.use_substrate = use_substrate
        self.combine_method = combine_method

    def compute_weights(self, conditions: List[Dict[str, Any]]) -> np.ndarray:
        """
        Compute weights based on maximum biomass values.

        The intuition: If flask A reaches max biomass of 1.0 g/L and
        flask B reaches 4.0 g/L, the squared errors in B will naturally
        be ~16x larger. To compensate, weight B by 1/16 relative to A.
        """
        n_conditions = len(conditions)
        weights = np.zeros(n_conditions)

        for i, cond in enumerate(conditions):
            biomass = cond.get('biomass', np.array([1.0]))
            substrate = cond.get('substrate', np.array([1.0]))

            # Compute max values (use peak-to-peak range as alternative)
            max_biomass = np.max(np.abs(biomass))
            max_substrate = np.max(np.abs(substrate))

            # Avoid division by zero
            max_biomass = max_biomass if max_biomass > 1e-10 else 1.0
            max_substrate = max_substrate if max_substrate > 1e-10 else 1.0

            # Compute weight based on method
            if self.combine_method == "biomass_only":
                weights[i] = 1.0 / (max_biomass ** 2)
            elif self.combine_method == "substrate_only":
                weights[i] = 1.0 / (max_substrate ** 2)
            elif self.combine_method == "geometric_mean":
                w_bio = 1.0 / (max_biomass ** 2)
                w_sub = 1.0 / (max_substrate ** 2)
                weights[i] = np.sqrt(w_bio * w_sub)
            elif self.combine_method == "arithmetic_mean":
                w_bio = 1.0 / (max_biomass ** 2)
                w_sub = 1.0 / (max_substrate ** 2)
                weights[i] = (w_bio + w_sub) / 2
            else:
                raise ValueError(f"Unknown combine_method: {self.combine_method}")

        # Normalize so weights sum to n_conditions (mean = 1)
        weights = weights * n_conditions / np.sum(weights)

        return weights

    def get_name(self) -> str:
        return "max_value"


class VarianceWeighting(WeightingStrategy):
    """
    Weight each condition inversely proportional to data variance.

    This approach estimates the noise level in each condition and
    down-weights noisier measurements. Requires sufficient data
    points to estimate variance reliably.

    Mathematical basis:
        weight_i = 1 / var(residuals_i)

    Note: This requires an initial fit to compute residuals,
    so it's typically used in iteratively reweighted least squares (IRLS).
    For initial fitting, use MaxValueWeighting instead.
    """

    def __init__(self, min_points: int = 5):
        """
        Args:
            min_points: Minimum data points required to estimate variance.
                        Conditions with fewer points use uniform weighting.
        """
        self.min_points = min_points

    def compute_weights(self, conditions: List[Dict[str, Any]]) -> np.ndarray:
        """
        Compute weights based on data variance.

        For initial fitting without residuals, uses variance of the data itself.
        For iterative refinement, pass residuals in condition dict.
        """
        n_conditions = len(conditions)
        weights = np.zeros(n_conditions)

        for i, cond in enumerate(conditions):
            # Check for residuals (from previous fit iteration)
            if 'residuals' in cond:
                data = cond['residuals']
            else:
                # Use biomass data variance as proxy
                data = cond.get('biomass', np.array([1.0]))

            if len(data) < self.min_points:
                # Insufficient data for variance estimate
                weights[i] = 1.0
            else:
                variance = np.var(data)
                # Avoid division by zero or near-zero variance
                weights[i] = 1.0 / max(variance, 1e-10)

        # Normalize
        weights = weights * n_conditions / np.sum(weights)

        return weights

    def get_name(self) -> str:
        return "variance"


class RangeWeighting(WeightingStrategy):
    """
    Weight each condition inversely proportional to data range (max - min).

    Similar to MaxValueWeighting but uses the dynamic range of the data.
    This can be more appropriate when initial values vary between conditions.
    """

    def compute_weights(self, conditions: List[Dict[str, Any]]) -> np.ndarray:
        """Compute weights based on biomass range."""
        n_conditions = len(conditions)
        weights = np.zeros(n_conditions)

        for i, cond in enumerate(conditions):
            biomass = cond.get('biomass', np.array([1.0]))

            data_range = np.ptp(biomass)  # peak-to-peak (max - min)
            data_range = data_range if data_range > 1e-10 else 1.0

            weights[i] = 1.0 / (data_range ** 2)

        # Normalize
        weights = weights * n_conditions / np.sum(weights)

        return weights

    def get_name(self) -> str:
        return "range"


def get_weighting_strategy(
    name: str,
    **kwargs
) -> WeightingStrategy:
    """
    Factory function to create weighting strategy by name.

    Args:
        name: Strategy name - one of:
            - "uniform": Equal weights (naive)
            - "max_value": Inverse max squared (recommended)
            - "variance": Inverse variance (for IRLS)
            - "range": Inverse range squared
        **kwargs: Additional arguments passed to strategy constructor

    Returns:
        Configured WeightingStrategy instance

    Example:
        >>> strategy = get_weighting_strategy("max_value", combine_method="biomass_only")
        >>> weights = strategy.compute_weights(conditions)
    """
    strategies = {
        "uniform": UniformWeighting,
        "max_value": MaxValueWeighting,
        "variance": VarianceWeighting,
        "range": RangeWeighting
    }

    name_lower = name.lower().replace("-", "_")

    if name_lower not in strategies:
        available = ", ".join(strategies.keys())
        raise ValueError(f"Unknown weighting strategy: {name}. Available: {available}")

    return strategies[name_lower](**kwargs)


def compute_condition_weights(
    conditions: List[Dict[str, Any]],
    strategy: str = "max_value"
) -> Tuple[np.ndarray, WeightingStrategy]:
    """
    Convenience function to compute weights for conditions.

    Args:
        conditions: List of condition dictionaries
        strategy: Name of weighting strategy

    Returns:
        Tuple of (weights array, strategy instance)

    Example:
        >>> weights, strategy = compute_condition_weights(conditions, "max_value")
        >>> print(f"Condition weights: {weights}")
    """
    strat = get_weighting_strategy(strategy)
    weights = strat.compute_weights(conditions)
    return weights, strat
