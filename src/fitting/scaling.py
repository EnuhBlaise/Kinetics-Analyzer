"""
Parameter and objective scaling utilities.

All transformations are optional — when disabled, raw values pass through
unchanged.  This ensures backward compatibility: existing pipelines produce
identical results unless the user explicitly enables a scaling option.

Three independent concerns are addressed:

1. **ParameterScaler** — maps parameters ↔ [0, 1] normalised space so that
   gradient-based re-optimisations (profiles, multi-start) see uniform
   step sizes regardless of the 6-order-of-magnitude spread in raw values.

2. **ObjectiveNormaliser** — weights substrate and biomass residuals by
   1/range so that both observables contribute equally to the loss
   function, preventing substrate values (50–2000 mg/L) from dominating
   biomass values (10–500 mg/L).

3. **adaptive_mcmc_proposal** — shapes Metropolis random-walk proposals
   using the Hessian covariance matrix when available, dramatically
   improving mixing for correlated parameters (qmax↔Ks, Y↔b_decay).
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Tuple, Dict, List


# ══════════════════════════════════════════════════════════════════════
#  Parameter Scaler
# ══════════════════════════════════════════════════════════════════════

class ParameterScaler:
    """Transform parameters between raw space and [0, 1] normalised space.

    Usage::

        scaler = ParameterScaler(lower_bounds, upper_bounds)
        x_norm = scaler.to_normalised(x_raw)
        x_raw  = scaler.to_raw(x_norm)

    Parameters
    ----------
    lower, upper : array-like
        Lower and upper bounds for each parameter.
    """

    def __init__(self, lower: np.ndarray, upper: np.ndarray):
        self.lower = np.asarray(lower, dtype=float)
        self.upper = np.asarray(upper, dtype=float)
        self.range = self.upper - self.lower
        # Protect against zero-width bounds
        self.range[self.range == 0] = 1.0

    def to_normalised(self, x: np.ndarray) -> np.ndarray:
        """Raw → [0, 1]."""
        return (np.asarray(x) - self.lower) / self.range

    def to_raw(self, x_norm: np.ndarray) -> np.ndarray:
        """[0, 1] → raw."""
        return np.asarray(x_norm) * self.range + self.lower

    def normalised_bounds(self) -> List[Tuple[float, float]]:
        """Return bounds as list of (0, 1) tuples for ``scipy.optimize``."""
        return [(0.0, 1.0)] * len(self.lower)


# ══════════════════════════════════════════════════════════════════════
#  Objective Normaliser
# ══════════════════════════════════════════════════════════════════════

class ObjectiveNormaliser:
    """Normalise residuals by observable range so substrate and biomass
    contribute equally regardless of their numerical scale.

    Usage::

        normaliser = ObjectiveNormaliser(S_data, X_data)
        weighted_S = normaliser.normalise_S(residuals_S)
        weighted_X = normaliser.normalise_X(residuals_X)

    Parameters
    ----------
    S_data, X_data : array-like
        Experimental substrate and biomass vectors.
    """

    def __init__(self, S_data: np.ndarray, X_data: np.ndarray):
        S_data = np.asarray(S_data, dtype=float)
        X_data = np.asarray(X_data, dtype=float)

        self.S_range = float(np.ptp(S_data))  # max − min
        self.X_range = float(np.ptp(X_data))

        # Fallback to mean of absolute values if range is zero
        if self.S_range == 0:
            self.S_range = float(np.mean(np.abs(S_data))) if np.any(S_data) else 1.0
        if self.X_range == 0:
            self.X_range = float(np.mean(np.abs(X_data))) if np.any(X_data) else 1.0

        self.S_weight = 1.0 / self.S_range
        self.X_weight = 1.0 / self.X_range

    def normalise_S(self, residuals: np.ndarray) -> np.ndarray:
        """Scale substrate residuals."""
        return np.asarray(residuals) * self.S_weight

    def normalise_X(self, residuals: np.ndarray) -> np.ndarray:
        """Scale biomass residuals."""
        return np.asarray(residuals) * self.X_weight

    def info(self) -> Dict[str, float]:
        """Return normalisation metadata."""
        return {
            "S_range": self.S_range,
            "X_range": self.X_range,
            "S_weight": self.S_weight,
            "X_weight": self.X_weight,
        }


# ══════════════════════════════════════════════════════════════════════
#  Adaptive MCMC Proposal
# ══════════════════════════════════════════════════════════════════════

def adaptive_mcmc_proposal(
    current: np.ndarray,
    covariance: Optional[np.ndarray],
    lower: np.ndarray,
    upper: np.ndarray,
    fallback_scale: float = 0.03,
    mix_weight: float = 0.8,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Generate an MCMC proposal using Hessian-adapted covariance when available.

    Uses the Haario *et al.* (2001) adaptive strategy::

        proposal = mix_weight × N(0, 2.38²/d × Σ)
                 + (1 − mix_weight) × N(0, small × I_range)

    The second component ensures ergodicity (the chain can always escape).

    Parameters
    ----------
    current : array
        Current parameter values.
    covariance : (d, d) array or None
        Estimated covariance matrix (e.g. from Hessian inverse).
        If *None*, falls back to range-scaled identity.
    lower, upper : arrays
        Parameter bounds.
    fallback_scale : float
        Scale for the range-based fallback proposal.
    mix_weight : float
        Probability of drawing from the adapted component (0–1).
    rng : numpy Generator or None
        Random number generator.

    Returns
    -------
    proposal : array
        Proposed parameter values (**not** clipped to bounds — caller
        handles that so the prior remains correct).
    """
    if rng is None:
        rng = np.random.default_rng()

    d = len(current)
    param_range = np.asarray(upper) - np.asarray(lower)
    param_range[param_range == 0] = 1.0

    if covariance is not None and mix_weight > 0:
        try:
            # Optimal scaling for random-walk MH:  2.38² / d
            scale_factor = (2.38 ** 2) / d
            scaled_cov = scale_factor * covariance

            # Ensure positive-definite
            eigvals = np.linalg.eigvalsh(scaled_cov)
            if np.any(eigvals <= 0):
                min_eig = np.min(eigvals)
                scaled_cov += (abs(min_eig) + 1e-10) * np.eye(d)

            L = np.linalg.cholesky(scaled_cov)
            adapted_step = L @ rng.standard_normal(d)
        except np.linalg.LinAlgError:
            # Cholesky failed — fall back entirely
            adapted_step = fallback_scale * param_range * rng.standard_normal(d)
            mix_weight = 0.0

        # Small identity component for ergodicity
        identity_step = (fallback_scale * 0.1) * param_range * rng.standard_normal(d)

        # Stochastic mixture
        if rng.random() < mix_weight:
            step = adapted_step
        else:
            step = identity_step
    else:
        # Pure range-scaled fallback
        step = fallback_scale * param_range * rng.standard_normal(d)

    return current + step


# ══════════════════════════════════════════════════════════════════════
#  Scaled Minimize
# ══════════════════════════════════════════════════════════════════════

def scaled_minimize(
    objective_fn,
    x0: np.ndarray,
    bounds: List[Tuple[float, float]],
    use_scaling: bool = True,
    method: str = "L-BFGS-B",
    **kwargs,
):
    """Wrapper around ``scipy.optimize.minimize`` with optional [0, 1]
    parameter scaling.

    When ``use_scaling=True``, the optimisation runs in normalised space
    so that the gradient-based solver sees unit-scale parameters.  The
    result is transparently converted back to raw space.

    Parameters
    ----------
    objective_fn : callable
        Objective function in **raw** parameter space.
    x0 : array
        Initial guess (raw space).
    bounds : list of (lower, upper)
        Parameter bounds (raw space).
    use_scaling : bool
        If False, passes straight through to ``minimize``.
    method : str
        Optimisation algorithm (default ``'L-BFGS-B'``).
    **kwargs
        Extra keyword arguments forwarded to ``minimize``.

    Returns
    -------
    OptimizeResult
        ``result.x`` is always in **raw** space.
    """
    from scipy.optimize import minimize

    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])

    if not use_scaling:
        return minimize(objective_fn, x0, bounds=bounds, method=method, **kwargs)

    scaler = ParameterScaler(lower, upper)

    def scaled_obj(x_norm):
        return objective_fn(scaler.to_raw(x_norm))

    x0_norm = scaler.to_normalised(np.asarray(x0))
    x0_norm = np.clip(x0_norm, 0.001, 0.999)

    result = minimize(
        scaled_obj,
        x0_norm,
        bounds=scaler.normalised_bounds(),
        method=method,
        **kwargs,
    )

    # Map result back to raw space
    result.x = scaler.to_raw(result.x)
    return result
