"""
Two-Stage Parameter Estimation for Monod Kinetics.

The Problem:
Differential equation solvers (like L-BFGS-B with ODE objectives) are
sensitive to initial parameter guesses. Poor guesses can lead to:
1. Solver crashes (integration failure)
2. Local minima (suboptimal parameters)
3. Slow convergence

The Solution (Two-Stage Estimation):
Stage 1: Smooth the data, compute derivatives, fit the ALGEBRAIC Monod equation
         to estimate parameters without solving ODEs.
Stage 2: Use Stage 1 results as initial guesses for the full ODE-based fit.

Mathematical Basis:
The Monod growth equation can be written differentially:
    dX/dt = μ * X = μmax * S/(Ks + S) * X

Rearranging:
    (1/X) * dX/dt = μmax * S/(Ks + S)

The left side is the specific growth rate, which can be estimated from data.
This allows algebraic fitting without ODE integration.
"""

from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import numpy as np
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
import warnings


@dataclass
class TwoStageResult:
    """Results from two-stage parameter estimation."""
    initial_guesses: Dict[str, float]
    stage1_quality: Dict[str, float]  # R², RMSE of Stage 1 fit
    derivatives: Dict[str, np.ndarray]  # Computed derivatives per condition
    warnings: List[str]  # Any quality warnings


class SavitzkyGolayDifferentiator:
    """
    Numerical differentiation using Savitzky-Golay filter.

    The Savitzky-Golay filter fits successive polynomials to data windows
    and uses the polynomial coefficients to estimate derivatives. This
    approach:
    - Smooths noise while preserving curve shape
    - Provides continuous derivative estimates
    - Is more robust than finite differences

    Default parameters (window=5, polyorder=2) are suitable for typical
    microbial growth data with 10-20 time points.
    """

    def __init__(
        self,
        window_length: int = 5,
        polyorder: int = 2,
        mode: str = 'interp'
    ):
        """
        Args:
            window_length: Number of points in smoothing window (must be odd)
            polyorder: Polynomial order for fitting (must be < window_length)
            mode: How to handle edges - 'interp', 'nearest', or 'mirror'
        """
        # Ensure window is odd
        if window_length % 2 == 0:
            window_length += 1
            warnings.warn(f"Window length must be odd. Adjusted to {window_length}")

        if polyorder >= window_length:
            polyorder = window_length - 1
            warnings.warn(f"Polyorder must be < window_length. Adjusted to {polyorder}")

        self.window_length = window_length
        self.polyorder = polyorder
        self.mode = mode

    def differentiate(
        self,
        time: np.ndarray,
        data: np.ndarray
    ) -> np.ndarray:
        """
        Compute derivative of data with respect to time.

        Args:
            time: Time points (must be sorted, monotonically increasing)
            data: Data values at each time point

        Returns:
            Derivative estimates at each time point (same length as input)
        """
        n_points = len(data)

        # Handle small datasets
        if n_points < self.window_length:
            # Fall back to simple finite differences
            return self._finite_difference(time, data)

        # Compute time step (handle non-uniform spacing)
        dt = np.mean(np.diff(time))

        # Apply Savitzky-Golay filter with derivative
        # deriv=1 computes first derivative
        derivative = savgol_filter(
            data,
            window_length=min(self.window_length, n_points if n_points % 2 == 1 else n_points - 1),
            polyorder=min(self.polyorder, min(self.window_length, n_points) - 1),
            deriv=1,
            delta=dt,
            mode=self.mode
        )

        return derivative

    def _finite_difference(self, time: np.ndarray, data: np.ndarray) -> np.ndarray:
        """Fallback to central finite differences for small datasets."""
        n = len(data)
        derivative = np.zeros(n)

        # Forward difference for first point
        derivative[0] = (data[1] - data[0]) / (time[1] - time[0])

        # Central differences for middle points
        for i in range(1, n - 1):
            derivative[i] = (data[i + 1] - data[i - 1]) / (time[i + 1] - time[i - 1])

        # Backward difference for last point
        derivative[-1] = (data[-1] - data[-2]) / (time[-1] - time[-2])

        return derivative

    def smooth(self, data: np.ndarray) -> np.ndarray:
        """Smooth data without differentiating."""
        n_points = len(data)

        if n_points < self.window_length:
            return data  # Can't smooth small datasets

        return savgol_filter(
            data,
            window_length=min(self.window_length, n_points if n_points % 2 == 1 else n_points - 1),
            polyorder=min(self.polyorder, min(self.window_length, n_points) - 1),
            deriv=0,
            mode=self.mode
        )


class TwoStageEstimator:
    """
    Two-stage parameter estimation for Monod kinetics.

    Stage 1: Derivative-based algebraic fitting
    - Smooth biomass data
    - Compute dX/dt numerically
    - Fit: dX/dt = μmax * S/(Ks + S) * X to estimate μmax and Ks

    Stage 2 (external): Use Stage 1 results as initial guesses for ODE fit
    """

    def __init__(
        self,
        window_length: int = 5,
        polyorder: int = 2,
        quality_threshold: float = 0.5,
        min_growth_rate: float = 0.01
    ):
        """
        Args:
            window_length: Savitzky-Golay window for smoothing
            polyorder: Polynomial order for smoothing
            quality_threshold: Minimum R² for Stage 1 to be considered valid
            min_growth_rate: Minimum positive growth rate to include in fit
        """
        self.differentiator = SavitzkyGolayDifferentiator(window_length, polyorder)
        self.quality_threshold = quality_threshold
        self.min_growth_rate = min_growth_rate

    def estimate_initial_params(
        self,
        conditions: List[Dict[str, Any]],
        param_names: List[str],
        default_params: Optional[Dict[str, float]] = None
    ) -> TwoStageResult:
        """
        Estimate initial parameters from multiple experimental conditions.

        Args:
            conditions: List of condition dictionaries with keys:
                - 'time': time points
                - 'substrate': substrate measurements
                - 'biomass': biomass measurements
            param_names: Names of parameters to estimate
            default_params: Fallback values if Stage 1 fails

        Returns:
            TwoStageResult with initial guesses and quality metrics
        """
        if default_params is None:
            default_params = {
                'qmax': 2.0, 'Ks': 500.0, 'Ki': 25000.0,
                'Y': 0.35, 'b_decay': 0.01, 'K_o2': 0.15,
                'Y_o2': 0.8, 'lag_time': 0.0
            }

        warnings_list = []
        derivatives_dict = {}

        # Collect data from all conditions
        all_substrate = []
        all_biomass = []
        all_specific_rate = []

        for i, cond in enumerate(conditions):
            time = np.array(cond['time'])
            substrate = np.array(cond['substrate'])
            biomass = np.array(cond['biomass'])
            label = cond.get('label', f'Condition_{i}')

            # Smooth data
            biomass_smooth = self.differentiator.smooth(biomass)
            substrate_smooth = self.differentiator.smooth(substrate)

            # Compute derivatives
            dXdt = self.differentiator.differentiate(time, biomass_smooth)
            derivatives_dict[label] = dXdt

            # Compute specific growth rate: μ = (1/X) * dX/dt
            # Avoid division by zero
            safe_biomass = np.maximum(biomass_smooth, 1e-6)
            specific_rate = dXdt / safe_biomass

            # Filter out negative or very low rates (lag/death phase)
            mask = specific_rate > self.min_growth_rate
            if np.sum(mask) < 3:
                warnings_list.append(f"{label}: Insufficient positive growth data")
                continue

            all_substrate.extend(substrate_smooth[mask])
            all_biomass.extend(biomass_smooth[mask])
            all_specific_rate.extend(specific_rate[mask])

        # Convert to arrays
        S = np.array(all_substrate)
        X = np.array(all_biomass)
        mu = np.array(all_specific_rate)

        # Fit Monod equation: μ = μmax * S / (Ks + S)
        stage1_quality = {}
        estimated_params = {}

        if len(S) < 3:
            warnings_list.append("Insufficient data for Stage 1 fit. Using defaults.")
            return TwoStageResult(
                initial_guesses={k: default_params.get(k, 1.0) for k in param_names},
                stage1_quality={'r_squared': 0.0, 'rmse': np.inf},
                derivatives=derivatives_dict,
                warnings=warnings_list
            )

        try:
            # Fit μ = μmax * S / (Ks + S)
            popt, pcov = curve_fit(
                self._monod_rate,
                S,
                mu,
                p0=[np.max(mu) * 1.2, np.median(S)],  # Initial guesses
                bounds=([0.01, 0.1], [10.0, 10000.0]),  # Reasonable bounds
                maxfev=5000
            )

            mu_max_est, Ks_est = popt

            # Compute R² for quality assessment
            mu_pred = self._monod_rate(S, mu_max_est, Ks_est)
            ss_res = np.sum((mu - mu_pred) ** 2)
            ss_tot = np.sum((mu - np.mean(mu)) ** 2)
            r_squared = 1 - ss_res / max(ss_tot, 1e-10)
            rmse = np.sqrt(ss_res / len(mu))

            stage1_quality['r_squared'] = r_squared
            stage1_quality['rmse'] = rmse

            # Map to parameter names
            # μmax relates to qmax via yield: μmax = qmax * Y
            # We estimate qmax assuming Y ≈ 0.35 as typical
            Y_assumed = default_params.get('Y', 0.35)
            qmax_est = mu_max_est / Y_assumed

            estimated_params['qmax'] = qmax_est
            estimated_params['Ks'] = Ks_est

            if r_squared < self.quality_threshold:
                warnings_list.append(
                    f"Stage 1 R² = {r_squared:.3f} below threshold {self.quality_threshold}. "
                    "Consider using default values."
                )

        except Exception as e:
            warnings_list.append(f"Stage 1 fitting failed: {str(e)}. Using defaults.")
            stage1_quality = {'r_squared': 0.0, 'rmse': np.inf}

        # Build initial guesses, using estimates where available, defaults otherwise
        initial_guesses = {}
        for param in param_names:
            if param in estimated_params:
                initial_guesses[param] = estimated_params[param]
            elif param in default_params:
                initial_guesses[param] = default_params[param]
            else:
                initial_guesses[param] = 1.0
                warnings_list.append(f"No estimate or default for {param}, using 1.0")

        # Estimate yield from mass balance if possible
        if 'Y' in param_names and len(conditions) > 0:
            Y_est = self._estimate_yield(conditions)
            if Y_est is not None:
                initial_guesses['Y'] = Y_est
                # Update qmax estimate with better Y
                if 'qmax' in initial_guesses and 'qmax' in estimated_params:
                    initial_guesses['qmax'] = mu_max_est / Y_est if 'mu_max_est' in dir() else initial_guesses['qmax']

        return TwoStageResult(
            initial_guesses=initial_guesses,
            stage1_quality=stage1_quality,
            derivatives=derivatives_dict,
            warnings=warnings_list
        )

    def _monod_rate(self, S: np.ndarray, mu_max: float, Ks: float) -> np.ndarray:
        """Monod rate equation for curve fitting."""
        return mu_max * S / (Ks + S)

    def _estimate_yield(self, conditions: List[Dict[str, Any]]) -> Optional[float]:
        """
        Estimate yield coefficient from mass balance.

        Y = ΔX / ΔS (change in biomass / change in substrate consumed)
        """
        yields = []

        for cond in conditions:
            substrate = np.array(cond['substrate'])
            biomass = np.array(cond['biomass'])

            delta_X = biomass[-1] - biomass[0]
            delta_S = substrate[0] - substrate[-1]

            if delta_S > 0 and delta_X > 0:
                Y = delta_X / delta_S
                if 0.05 < Y < 2.0:  # Sanity check
                    yields.append(Y)

        if len(yields) > 0:
            return np.median(yields)
        return None


def estimate_initial_parameters(
    conditions: List[Dict[str, Any]],
    param_names: List[str],
    config: Optional[Dict[str, Any]] = None
) -> TwoStageResult:
    """
    Convenience function for two-stage estimation.

    Args:
        conditions: Experimental condition data
        param_names: Parameters to estimate
        config: Optional configuration with keys:
            - 'window_length': Smoothing window (default 5)
            - 'polyorder': Polynomial order (default 2)
            - 'quality_threshold': Min R² (default 0.5)
            - 'defaults': Default parameter values

    Returns:
        TwoStageResult with estimates

    Example:
        >>> result = estimate_initial_parameters(conditions, ['qmax', 'Ks', 'Y'])
        >>> print(f"qmax estimate: {result.initial_guesses['qmax']:.3f}")
        >>> if result.warnings:
        >>>     print(f"Warnings: {result.warnings}")
    """
    config = config or {}

    estimator = TwoStageEstimator(
        window_length=config.get('window_length', 5),
        polyorder=config.get('polyorder', 2),
        quality_threshold=config.get('quality_threshold', 0.5)
    )

    return estimator.estimate_initial_params(
        conditions,
        param_names,
        default_params=config.get('defaults')
    )
