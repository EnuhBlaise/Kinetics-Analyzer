"""
Bootstrap Aggregation for Parameter Uncertainty Quantification.

The Problem:
Point estimates of kinetic parameters tell you nothing about confidence.
When you report μmax = 0.5 h⁻¹, is the true value between 0.4-0.6 or 0.1-0.9?
This matters critically for:
- Bioreactor design safety factors
- Model comparison (do two models really differ?)
- Publication credibility

The Solution (Bootstrap):
1. Fit model to original data → get residuals
2. Resample residuals with replacement
3. Add resampled residuals to fitted values → synthetic dataset
4. Fit model to synthetic dataset → one bootstrap estimate
5. Repeat B times → get distribution of parameter estimates
6. Compute percentile-based confidence intervals

Why Residual Resampling (not Case Resampling):
- Case resampling randomly selects (time, data) pairs
- This disrupts the time-series structure of kinetic data
- Residual resampling preserves the experimental design
- More appropriate for regression problems with fixed X values
"""

from typing import Dict, List, Callable, Optional, Any, Tuple
from dataclasses import dataclass, field
import numpy as np
from multiprocessing import Pool, cpu_count
import warnings
from functools import partial


@dataclass
class BootstrapResult:
    """
    Results from bootstrap parameter estimation.

    Attributes:
        point_estimates: Median parameter values (robust to outliers)
        confidence_intervals: Dict of {param: (lower, upper)} at given confidence level
        all_estimates: Array of shape (n_successful, n_params) with all bootstrap estimates
        param_names: Ordered list of parameter names
        n_successful: Number of successful bootstrap iterations
        n_failed: Number of failed iterations
        confidence_level: Confidence level used (e.g., 0.95 for 95% CI)
    """
    point_estimates: Dict[str, float]
    confidence_intervals: Dict[str, Tuple[float, float]]
    all_estimates: np.ndarray
    param_names: List[str]
    n_successful: int
    n_failed: int
    confidence_level: float = 0.95

    def summary(self) -> str:
        """Generate human-readable summary of results."""
        lines = [
            "Bootstrap Results",
            "=" * 50,
            f"Iterations: {self.n_successful}/{self.n_successful + self.n_failed} successful",
            f"Confidence Level: {self.confidence_level * 100:.0f}%",
            "",
            "Parameter Estimates:",
        ]

        for param in self.param_names:
            est = self.point_estimates[param]
            lo, hi = self.confidence_intervals[param]
            lines.append(f"  {param}: {est:.4f} ({lo:.4f}, {hi:.4f})")

        return "\n".join(lines)


class BootstrapEngine:
    """
    Bootstrap engine for parameter uncertainty quantification.

    Supports:
    - Residual resampling (preserves time structure)
    - Parallel execution via multiprocessing
    - Robust handling of failed iterations
    - Percentile-based confidence intervals
    """

    def __init__(
        self,
        n_iterations: int = 500,
        n_workers: Optional[int] = None,
        confidence_level: float = 0.95,
        random_seed: Optional[int] = None,
        min_success_rate: float = 0.5
    ):
        """
        Initialize bootstrap engine.

        Args:
            n_iterations: Number of bootstrap iterations (default 500)
            n_workers: Number of parallel workers (default: CPU count - 1)
            confidence_level: Confidence level for intervals (default 0.95)
            random_seed: Seed for reproducibility (default: None)
            min_success_rate: Minimum fraction of successful iterations required
        """
        self.n_iterations = n_iterations
        self.n_workers = n_workers or max(1, cpu_count() - 1)
        self.confidence_level = confidence_level
        self.random_seed = random_seed
        self.min_success_rate = min_success_rate

        if random_seed is not None:
            np.random.seed(random_seed)

    def run(
        self,
        fitter_func: Callable,
        conditions: List[Dict[str, Any]],
        original_params: Dict[str, float],
        param_names: List[str],
        predict_func: Callable
    ) -> BootstrapResult:
        """
        Run bootstrap analysis.

        Args:
            fitter_func: Function that fits parameters to data.
                Signature: fitter_func(conditions) -> Dict[str, float]
            conditions: Original experimental conditions
            original_params: Parameters from fitting original data
            param_names: Ordered list of parameter names
            predict_func: Function to predict data given params.
                Signature: predict_func(condition, params) -> (substrate_pred, biomass_pred)

        Returns:
            BootstrapResult with confidence intervals
        """
        # Compute residuals from original fit
        residuals = self._compute_residuals(conditions, original_params, predict_func)

        # Generate bootstrap seeds for reproducibility
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
        seeds = np.random.randint(0, 2**31, size=self.n_iterations)

        # Prepare arguments for parallel execution
        worker_args = [
            (i, conditions, residuals, original_params, predict_func, fitter_func, seed, param_names)
            for i, seed in enumerate(seeds)
        ]

        # Run bootstrap iterations
        if self.n_workers > 1:
            # Parallel execution
            with Pool(processes=self.n_workers) as pool:
                results = pool.map(_bootstrap_single_iteration, worker_args)
        else:
            # Sequential execution
            results = [_bootstrap_single_iteration(args) for args in worker_args]

        # Collect successful results
        successful_estimates = []
        n_failed = 0

        for result in results:
            if result is not None:
                successful_estimates.append(result)
            else:
                n_failed += 1

        n_successful = len(successful_estimates)

        # Check success rate
        success_rate = n_successful / self.n_iterations
        if success_rate < self.min_success_rate:
            warnings.warn(
                f"Bootstrap success rate ({success_rate:.1%}) below threshold "
                f"({self.min_success_rate:.1%}). Results may be unreliable."
            )

        if n_successful == 0:
            raise RuntimeError("All bootstrap iterations failed. Check fitter function.")

        # Convert to array
        all_estimates = np.array(successful_estimates)

        # Compute point estimates (median for robustness)
        point_estimates = {}
        for i, param in enumerate(param_names):
            point_estimates[param] = np.median(all_estimates[:, i])

        # Compute confidence intervals (percentile method)
        alpha = 1 - self.confidence_level
        lower_pct = alpha / 2 * 100
        upper_pct = (1 - alpha / 2) * 100

        confidence_intervals = {}
        for i, param in enumerate(param_names):
            lower = np.percentile(all_estimates[:, i], lower_pct)
            upper = np.percentile(all_estimates[:, i], upper_pct)
            confidence_intervals[param] = (lower, upper)

        return BootstrapResult(
            point_estimates=point_estimates,
            confidence_intervals=confidence_intervals,
            all_estimates=all_estimates,
            param_names=param_names,
            n_successful=n_successful,
            n_failed=n_failed,
            confidence_level=self.confidence_level
        )

    def _compute_residuals(
        self,
        conditions: List[Dict[str, Any]],
        params: Dict[str, float],
        predict_func: Callable
    ) -> List[Dict[str, np.ndarray]]:
        """
        Compute residuals between observed and predicted values.

        Returns list of dicts with 'substrate_residuals' and 'biomass_residuals'.
        """
        residuals = []

        for cond in conditions:
            substrate_pred, biomass_pred = predict_func(cond, params)

            substrate_obs = np.array(cond['substrate'])
            biomass_obs = np.array(cond['biomass'])

            residuals.append({
                'substrate_residuals': substrate_obs - substrate_pred,
                'biomass_residuals': biomass_obs - biomass_pred
            })

        return residuals


def _bootstrap_single_iteration(args: tuple) -> Optional[List[float]]:
    """
    Execute a single bootstrap iteration.

    This is a module-level function to support multiprocessing.

    Args:
        args: Tuple of (iteration_idx, conditions, residuals, original_params,
                        predict_func, fitter_func, seed, param_names)

    Returns:
        List of parameter values if successful, None if failed
    """
    (idx, conditions, residuals, original_params, predict_func,
     fitter_func, seed, param_names) = args

    np.random.seed(seed)

    try:
        # Create synthetic dataset via residual resampling
        synthetic_conditions = _resample_residuals(
            conditions, residuals, original_params, predict_func
        )

        # Fit model to synthetic data
        fitted_params = fitter_func(synthetic_conditions)

        # Return parameters in correct order
        return [fitted_params[name] for name in param_names]

    except Exception:
        # Silently handle failures (tracked in main results)
        return None


def _resample_residuals(
    conditions: List[Dict[str, Any]],
    residuals: List[Dict[str, np.ndarray]],
    params: Dict[str, float],
    predict_func: Callable
) -> List[Dict[str, Any]]:
    """
    Create synthetic dataset by resampling residuals.

    For each condition:
    1. Get predicted values from original fit
    2. Resample residuals with replacement
    3. Add resampled residuals to predictions
    4. Create new condition dict with synthetic data
    """
    synthetic_conditions = []

    for cond, resid in zip(conditions, residuals):
        # Get predictions
        substrate_pred, biomass_pred = predict_func(cond, params)

        # Resample residuals (with replacement)
        n_points = len(cond['time'])
        substrate_resid_resampled = np.random.choice(
            resid['substrate_residuals'], size=n_points, replace=True
        )
        biomass_resid_resampled = np.random.choice(
            resid['biomass_residuals'], size=n_points, replace=True
        )

        # Create synthetic data
        synthetic_substrate = substrate_pred + substrate_resid_resampled
        synthetic_biomass = biomass_pred + biomass_resid_resampled

        # Ensure non-negative (physical constraint)
        synthetic_substrate = np.maximum(synthetic_substrate, 0)
        synthetic_biomass = np.maximum(synthetic_biomass, 0)

        # Copy condition with synthetic data
        synthetic_cond = cond.copy()
        synthetic_cond['substrate'] = synthetic_substrate
        synthetic_cond['biomass'] = synthetic_biomass

        synthetic_conditions.append(synthetic_cond)

    return synthetic_conditions


def run_bootstrap(
    fitter_func: Callable,
    predict_func: Callable,
    conditions: List[Dict[str, Any]],
    original_params: Dict[str, float],
    param_names: List[str],
    n_iterations: int = 500,
    confidence_level: float = 0.95,
    n_workers: Optional[int] = None,
    random_seed: Optional[int] = None
) -> BootstrapResult:
    """
    Convenience function to run bootstrap analysis.

    Args:
        fitter_func: Function that fits parameters to conditions data
        predict_func: Function that predicts data given params
        conditions: Original experimental data
        original_params: Parameters from fitting original data
        param_names: Ordered parameter names
        n_iterations: Number of bootstrap iterations
        confidence_level: Confidence level (e.g., 0.95)
        n_workers: Parallel workers (None = auto)
        random_seed: Seed for reproducibility

    Returns:
        BootstrapResult with confidence intervals

    Example:
        >>> def my_fitter(conditions):
        >>>     # Your fitting logic
        >>>     return {'qmax': 2.5, 'Ks': 400, ...}
        >>>
        >>> def my_predictor(condition, params):
        >>>     # Your prediction logic
        >>>     return substrate_pred, biomass_pred
        >>>
        >>> result = run_bootstrap(my_fitter, my_predictor, conditions,
        >>>                        original_params, param_names)
        >>> print(result.summary())
    """
    engine = BootstrapEngine(
        n_iterations=n_iterations,
        n_workers=n_workers,
        confidence_level=confidence_level,
        random_seed=random_seed
    )

    return engine.run(
        fitter_func=fitter_func,
        conditions=conditions,
        original_params=original_params,
        param_names=param_names,
        predict_func=predict_func
    )
