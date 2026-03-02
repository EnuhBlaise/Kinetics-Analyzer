"""
Statistical metrics for model evaluation.

This module provides functions to calculate common fit statistics
including R², RMSE, AIC, and BIC for comparing model performance.
"""

import numpy as np
from typing import Dict, Tuple, List, Optional, Any

from src.fitting.scaling import adaptive_mcmc_proposal


def calculate_r_squared(
    observed: np.ndarray,
    predicted: np.ndarray
) -> float:
    """
    Calculate the coefficient of determination (R²).

    R² measures how well the model predictions match the observed data.
    Values range from -inf to 1, where 1 indicates perfect fit.

    R² = 1 - SS_res / SS_tot

    Args:
        observed: Array of observed (experimental) values
        predicted: Array of model predicted values

    Returns:
        R² value (can be negative for poor fits)

    Example:
        >>> obs = np.array([1, 2, 3, 4, 5])
        >>> pred = np.array([1.1, 2.0, 2.9, 4.1, 5.0])
        >>> r2 = calculate_r_squared(obs, pred)
        >>> print(f"R² = {r2:.4f}")
    """
    observed = np.asarray(observed)
    predicted = np.asarray(predicted)

    # Sum of squared residuals
    ss_res = np.sum((observed - predicted) ** 2)

    # Total sum of squares
    ss_tot = np.sum((observed - np.mean(observed)) ** 2)

    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0

    return 1.0 - (ss_res / ss_tot)


def calculate_rmse(
    observed: np.ndarray,
    predicted: np.ndarray
) -> float:
    """
    Calculate Root Mean Square Error (RMSE).

    RMSE measures the average magnitude of prediction errors.
    Lower values indicate better fit. RMSE is in the same units
    as the measured variable.

    Args:
        observed: Array of observed values
        predicted: Array of predicted values

    Returns:
        RMSE value (always >= 0)
    """
    observed = np.asarray(observed)
    predicted = np.asarray(predicted)

    mse = np.mean((observed - predicted) ** 2)
    return np.sqrt(mse)


def calculate_mae(
    observed: np.ndarray,
    predicted: np.ndarray
) -> float:
    """
    Calculate Mean Absolute Error (MAE).

    MAE is less sensitive to outliers than RMSE.

    Args:
        observed: Array of observed values
        predicted: Array of predicted values

    Returns:
        MAE value (always >= 0)
    """
    observed = np.asarray(observed)
    predicted = np.asarray(predicted)

    return np.mean(np.abs(observed - predicted))


def calculate_aic(
    sse: float,
    n_parameters: int,
    n_observations: int
) -> float:
    """
    Calculate Akaike Information Criterion (AIC).

    AIC balances model fit against complexity. Lower AIC indicates
    a better model. Used for comparing models with different numbers
    of parameters.

    AIC = n * ln(SSE/n) + 2k

    where n is number of observations and k is number of parameters.

    Args:
        sse: Sum of squared errors (residuals)
        n_parameters: Number of model parameters (k)
        n_observations: Number of data points (n)

    Returns:
        AIC value (lower is better)

    Note:
        For small sample sizes (n/k < 40), use AICc (corrected AIC)
        which is returned when applicable.
    """
    n = n_observations
    k = n_parameters

    if n <= 0 or sse <= 0:
        return np.inf

    # Log-likelihood based AIC
    aic = n * np.log(sse / n) + 2 * k

    # Apply small-sample correction (AICc) when needed
    if n / k < 40:
        if n > k + 1:
            aic += (2 * k * (k + 1)) / (n - k - 1)
        else:
            return np.inf

    return aic


def calculate_bic(
    sse: float,
    n_parameters: int,
    n_observations: int
) -> float:
    """
    Calculate Bayesian Information Criterion (BIC).

    BIC penalizes model complexity more strongly than AIC,
    especially for larger datasets. Lower BIC indicates better model.

    BIC = n * ln(SSE/n) + k * ln(n)

    Args:
        sse: Sum of squared errors
        n_parameters: Number of model parameters (k)
        n_observations: Number of data points (n)

    Returns:
        BIC value (lower is better)
    """
    n = n_observations
    k = n_parameters

    if n <= 0 or sse <= 0:
        return np.inf

    return n * np.log(sse / n) + k * np.log(n)


def calculate_all_statistics(
    observed: np.ndarray,
    predicted: np.ndarray,
    n_parameters: int
) -> Dict[str, float]:
    """
    Calculate all fit statistics at once.

    Args:
        observed: Observed values
        predicted: Predicted values
        n_parameters: Number of model parameters

    Returns:
        Dictionary with all statistics
    """
    observed = np.asarray(observed)
    predicted = np.asarray(predicted)

    n = len(observed)
    sse = np.sum((observed - predicted) ** 2)

    return {
        "R_squared": calculate_r_squared(observed, predicted),
        "RMSE": calculate_rmse(observed, predicted),
        "MAE": calculate_mae(observed, predicted),
        "AIC": calculate_aic(sse, n_parameters, n),
        "BIC": calculate_bic(sse, n_parameters, n),
        "SSE": sse,
        "n_observations": n,
        "n_parameters": n_parameters
    }


def calculate_nrmse(
    observed: np.ndarray,
    predicted: np.ndarray
) -> float:
    """
    Calculate Normalized Root Mean Square Error (NRMSE).
    
    NRMSE = RMSE / (max - min) of observed values.
    This allows comparison across variables with different scales.
    
    Args:
        observed: Array of observed values
        predicted: Array of predicted values
        
    Returns:
        NRMSE value (0-1 range for good fits, can exceed 1 for poor fits)
    """
    observed = np.asarray(observed)
    predicted = np.asarray(predicted)
    
    rmse = calculate_rmse(observed, predicted)
    data_range = np.ptp(observed)  # max - min
    
    if data_range == 0:
        return 0.0 if rmse == 0 else np.inf
    
    return rmse / data_range


def calculate_separate_statistics(
    observed_substrate: np.ndarray,
    predicted_substrate: np.ndarray,
    observed_biomass: np.ndarray,
    predicted_biomass: np.ndarray,
    n_parameters: int,
    weight_substrate: float = 0.5,
    weight_biomass: float = 0.5
) -> Dict[str, Dict[str, float]]:
    """
    Calculate separate statistics for substrate and biomass.
    
    This is the recommended approach since substrate (e.g., 1000 mg/L) and 
    biomass (e.g., 0.5 OD) have very different magnitudes and their errors
    cannot be meaningfully combined without normalization.
    
    Args:
        observed_substrate: Observed substrate concentrations
        predicted_substrate: Predicted substrate concentrations
        observed_biomass: Observed biomass concentrations
        predicted_biomass: Predicted biomass concentrations
        n_parameters: Number of model parameters
        weight_substrate: Weight for substrate in combined metrics (default 0.5)
        weight_biomass: Weight for biomass in combined metrics (default 0.5)
        
    Returns:
        Dictionary with 'substrate', 'biomass', and 'combined' statistics
    """
    observed_substrate = np.asarray(observed_substrate)
    predicted_substrate = np.asarray(predicted_substrate)
    observed_biomass = np.asarray(observed_biomass)
    predicted_biomass = np.asarray(predicted_biomass)
    
    # Substrate statistics
    n_sub = len(observed_substrate)
    sse_sub = np.sum((observed_substrate - predicted_substrate) ** 2)
    r2_sub = calculate_r_squared(observed_substrate, predicted_substrate)
    rmse_sub = calculate_rmse(observed_substrate, predicted_substrate)
    nrmse_sub = calculate_nrmse(observed_substrate, predicted_substrate)
    mae_sub = calculate_mae(observed_substrate, predicted_substrate)
    
    substrate_stats = {
        "R_squared": r2_sub,
        "RMSE": rmse_sub,
        "NRMSE": nrmse_sub,
        "MAE": mae_sub,
        "SSE": sse_sub,
        "n_points": n_sub
    }
    
    # Biomass statistics
    n_bio = len(observed_biomass)
    sse_bio = np.sum((observed_biomass - predicted_biomass) ** 2)
    r2_bio = calculate_r_squared(observed_biomass, predicted_biomass)
    rmse_bio = calculate_rmse(observed_biomass, predicted_biomass)
    nrmse_bio = calculate_nrmse(observed_biomass, predicted_biomass)
    mae_bio = calculate_mae(observed_biomass, predicted_biomass)
    
    biomass_stats = {
        "R_squared": r2_bio,
        "RMSE": rmse_bio,
        "NRMSE": nrmse_bio,
        "MAE": mae_bio,
        "SSE": sse_bio,
        "n_points": n_bio
    }
    
    # Combined statistics (weighted averages where appropriate)
    # Normalize weights
    total_weight = weight_substrate + weight_biomass
    w_sub = weight_substrate / total_weight
    w_bio = weight_biomass / total_weight
    
    # Weighted average R² (more meaningful than combined)
    combined_r2 = w_sub * r2_sub + w_bio * r2_bio
    
    # Average NRMSE (normalized, so can be averaged)
    combined_nrmse = w_sub * nrmse_sub + w_bio * nrmse_bio
    
    # Total observations
    n_total = n_sub + n_bio
    
    # Combined SSE (normalized for AIC/BIC calculation)
    # Use normalized SSE so substrate and biomass contribute equally
    sub_range = np.ptp(observed_substrate) or 1.0
    bio_range = np.ptp(observed_biomass) or 1.0
    normalized_sse = (
        w_sub * sse_sub / (sub_range ** 2) + 
        w_bio * sse_bio / (bio_range ** 2)
    ) * ((sub_range ** 2 + bio_range ** 2) / 2)  # Scale back
    
    combined_stats = {
        "R_squared": combined_r2,
        "NRMSE": combined_nrmse,
        "AIC": calculate_aic(normalized_sse, n_parameters, n_total),
        "BIC": calculate_bic(normalized_sse, n_parameters, n_total),
        "n_observations": n_total,
        "n_parameters": n_parameters,
        "weight_substrate": weight_substrate,
        "weight_biomass": weight_biomass
    }
    
    return {
        "substrate": substrate_stats,
        "biomass": biomass_stats,
        "combined": combined_stats
    }


def compare_models(
    model_statistics: Dict[str, Dict[str, float]]
) -> Dict[str, Dict[str, any]]:
    """
    Compare multiple models based on their statistics.

    Args:
        model_statistics: Dictionary mapping model names to their statistics

    Returns:
        Dictionary with comparison results including rankings
    """
    model_names = list(model_statistics.keys())

    # Extract values for comparison
    r2_values = {m: model_statistics[m].get("R_squared", -np.inf) for m in model_names}
    aic_values = {m: model_statistics[m].get("AIC", np.inf) for m in model_names}
    bic_values = {m: model_statistics[m].get("BIC", np.inf) for m in model_names}
    rmse_values = {m: model_statistics[m].get("RMSE", np.inf) for m in model_names}

    # Rank models (higher R² is better, lower AIC/BIC/RMSE is better)
    r2_ranking = sorted(model_names, key=lambda m: r2_values[m], reverse=True)
    aic_ranking = sorted(model_names, key=lambda m: aic_values[m])
    bic_ranking = sorted(model_names, key=lambda m: bic_values[m])
    rmse_ranking = sorted(model_names, key=lambda m: rmse_values[m])

    # Calculate delta AIC (relative to best model)
    min_aic = min(aic_values.values())
    delta_aic = {m: aic_values[m] - min_aic for m in model_names}

    # Calculate Akaike weights
    akaike_weights = calculate_akaike_weights(aic_values)

    return {
        "rankings": {
            "by_R_squared": r2_ranking,
            "by_AIC": aic_ranking,
            "by_BIC": bic_ranking,
            "by_RMSE": rmse_ranking
        },
        "best_model": {
            "by_R_squared": r2_ranking[0],
            "by_AIC": aic_ranking[0],
            "by_BIC": bic_ranking[0],
            "by_RMSE": rmse_ranking[0]
        },
        "delta_AIC": delta_aic,
        "akaike_weights": akaike_weights
    }


def calculate_akaike_weights(
    aic_values: Dict[str, float]
) -> Dict[str, float]:
    """
    Calculate Akaike weights for model comparison.

    Akaike weights represent the probability that each model is the
    best model given the data and set of candidate models.

    Args:
        aic_values: Dictionary mapping model names to AIC values

    Returns:
        Dictionary of Akaike weights (sum to 1.0)
    """
    # Find minimum AIC
    min_aic = min(aic_values.values())

    # Calculate delta AIC and weights
    delta = {m: aic - min_aic for m, aic in aic_values.items()}

    # exp(-0.5 * delta)
    exp_delta = {m: np.exp(-0.5 * d) for m, d in delta.items()}

    # Normalize
    total = sum(exp_delta.values())
    if total == 0:
        return {m: 1.0 / len(aic_values) for m in aic_values}

    return {m: w / total for m, w in exp_delta.items()}


def residual_analysis(
    observed: np.ndarray,
    predicted: np.ndarray
) -> Dict[str, any]:
    """
    Perform residual analysis for model diagnostics.

    Args:
        observed: Observed values
        predicted: Predicted values

    Returns:
        Dictionary with residual statistics and diagnostics
    """
    residuals = observed - predicted

    return {
        "mean": float(np.mean(residuals)),
        "std": float(np.std(residuals)),
        "min": float(np.min(residuals)),
        "max": float(np.max(residuals)),
        "median": float(np.median(residuals)),
        "skewness": float(_calculate_skewness(residuals)),
        "kurtosis": float(_calculate_kurtosis(residuals)),
        "autocorrelation_lag1": float(_autocorrelation(residuals, 1)),
        "normality_test_passed": _check_normality(residuals)
    }


def _calculate_skewness(data: np.ndarray) -> float:
    """Calculate skewness of data."""
    n = len(data)
    if n < 3:
        return 0.0
    mean = np.mean(data)
    std = np.std(data)
    if std == 0:
        return 0.0
    return np.mean(((data - mean) / std) ** 3)


def _calculate_kurtosis(data: np.ndarray) -> float:
    """Calculate excess kurtosis of data."""
    n = len(data)
    if n < 4:
        return 0.0
    mean = np.mean(data)
    std = np.std(data)
    if std == 0:
        return 0.0
    return np.mean(((data - mean) / std) ** 4) - 3


def _autocorrelation(data: np.ndarray, lag: int) -> float:
    """Calculate autocorrelation at given lag."""
    n = len(data)
    if n <= lag:
        return 0.0
    mean = np.mean(data)
    var = np.var(data)
    if var == 0:
        return 0.0
    return np.mean((data[:-lag] - mean) * (data[lag:] - mean)) / var


def _check_normality(residuals: np.ndarray, alpha: float = 0.05) -> bool:
    """
    Simple normality check based on skewness and kurtosis.

    For more rigorous testing, use scipy.stats.shapiro or similar.
    """
    skew = abs(_calculate_skewness(residuals))
    kurt = abs(_calculate_kurtosis(residuals))

    # Rule of thumb: skewness < 2 and kurtosis < 7 suggests normality
    return skew < 2 and kurt < 7


def calculate_parameter_confidence_intervals(
    objective_function,
    optimal_params: np.ndarray,
    parameter_names: list,
    n_observations: int,
    confidence_level: float = 0.95,
    method: str = "hessian",
    bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    mcmc_samples: int = 4000,
    mcmc_burn_in: int = 1000,
    mcmc_step_scale: float = 0.05,
    mcmc_random_seed: Optional[int] = None,
    mcmc_chains: int = 4,
    mcmc_adaptive: bool = False,
) -> Dict[str, Dict[str, float]]:
    """
    Calculate confidence intervals for fitted parameters using Hessian approximation.
    
    This uses the inverse of the Hessian matrix (curvature of the objective function)
    to estimate the covariance matrix of parameters, then computes CIs.
    
    Args:
        objective_function: The objective function used for fitting
        optimal_params: Array of optimal parameter values
        parameter_names: List of parameter names
        n_observations: Number of data points
        confidence_level: Confidence level (default 0.95 for 95% CI)
    
    Returns:
        Dictionary mapping parameter names to their CI info:
        {param_name: {"value": x, "std_error": se, "ci_lower": lo, "ci_upper": hi}}
    """
    ci_results, _ = calculate_parameter_confidence_intervals_with_diagnostics(
        objective_function=objective_function,
        optimal_params=optimal_params,
        parameter_names=parameter_names,
        n_observations=n_observations,
        confidence_level=confidence_level,
        method=method,
        bounds=bounds,
        mcmc_samples=mcmc_samples,
        mcmc_burn_in=mcmc_burn_in,
        mcmc_step_scale=mcmc_step_scale,
        mcmc_random_seed=mcmc_random_seed,
        mcmc_chains=mcmc_chains,
        mcmc_adaptive=mcmc_adaptive,
    )
    return ci_results


def calculate_parameter_confidence_intervals_with_diagnostics(
    objective_function,
    optimal_params: np.ndarray,
    parameter_names: list,
    n_observations: int,
    confidence_level: float = 0.95,
    method: str = "hessian",
    bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    mcmc_samples: int = 4000,
    mcmc_burn_in: int = 1000,
    mcmc_step_scale: float = 0.05,
    mcmc_random_seed: Optional[int] = None,
    mcmc_chains: int = 5,
    mcmc_adaptive: bool = False,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Any]]:
    """Calculate confidence intervals and return diagnostics for plotting/reporting."""
    method = (method or "hessian").lower()

    if method == "hessian":
        return _calculate_ci_hessian(
            objective_function=objective_function,
            optimal_params=optimal_params,
            parameter_names=parameter_names,
            n_observations=n_observations,
            confidence_level=confidence_level,
        )
    elif method == "hessian_log":
        return _calculate_ci_hessian_log_transform(
            objective_function=objective_function,
            optimal_params=optimal_params,
            parameter_names=parameter_names,
            n_observations=n_observations,
            confidence_level=confidence_level,
            bounds=bounds,
        )
    elif method == "mcmc":
        return _calculate_ci_mcmc(
            objective_function=objective_function,
            optimal_params=optimal_params,
            parameter_names=parameter_names,
            confidence_level=confidence_level,
            bounds=bounds,
            n_samples=mcmc_samples,
            burn_in=mcmc_burn_in,
            step_scale=mcmc_step_scale,
            mcmc_adaptive=mcmc_adaptive,
            random_seed=mcmc_random_seed,
            n_chains=mcmc_chains,
        )
    else:
        raise ValueError(
            f"Unknown CI method: {method}. Choose from: hessian, hessian_log, mcmc"
        )


def _empty_ci(parameter_names: list, optimal_params: np.ndarray) -> Dict[str, Dict[str, float]]:
    """Return NaN confidence intervals for all parameters."""
    return {
        name: {
            "value": float(val),
            "std_error": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "relative_error_pct": np.nan,
            "acceptance_rate": np.nan,
        }
        for name, val in zip(parameter_names, optimal_params)
    }


def _empty_ci_diagnostics(method: str) -> Dict[str, Any]:
    """Return empty diagnostics payload."""
    return {
        "method": method,
        "status": "unavailable",
        "message": "Diagnostics unavailable",
    }


def _calculate_ci_hessian(
    objective_function,
    optimal_params: np.ndarray,
    parameter_names: list,
    n_observations: int,
    confidence_level: float,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Any]]:
    """Classic Hessian-based confidence intervals in original parameter space."""
    from scipy.stats import t as t_dist

    n_params = len(optimal_params)
    degrees_of_freedom = n_observations - n_params

    if degrees_of_freedom <= 0:
        return _empty_ci(parameter_names, optimal_params), {
            "method": "hessian",
            "status": "unavailable",
            "message": "Insufficient degrees of freedom",
            "degrees_of_freedom": int(degrees_of_freedom),
        }

    hessian = _compute_hessian(objective_function, optimal_params)
    sse = objective_function(optimal_params)
    residual_variance = sse / degrees_of_freedom

    used_pinv = False
    try:
        covariance_matrix = np.linalg.inv(hessian / 2) * residual_variance
        variances = np.diag(covariance_matrix)
        if np.any(variances < 0):
            raise np.linalg.LinAlgError("Negative variance detected")
        std_errors = np.sqrt(variances)
    except np.linalg.LinAlgError:
        try:
            covariance_matrix = np.linalg.pinv(hessian / 2) * residual_variance
            variances = np.diag(covariance_matrix)
            std_errors = np.sqrt(np.abs(variances))
            used_pinv = True
        except Exception:
            std_errors = np.full(n_params, np.nan)

    alpha = 1 - confidence_level
    t_value = t_dist.ppf(1 - alpha / 2, degrees_of_freedom)

    results = {}
    for i, name in enumerate(parameter_names):
        value = float(optimal_params[i])
        se = float(std_errors[i]) if not np.isnan(std_errors[i]) else np.nan

        if np.isnan(se) or se <= 0:
            ci_lower, ci_upper, rel_err = np.nan, np.nan, np.nan
        else:
            ci_lower = value - t_value * se
            ci_upper = value + t_value * se
            rel_err = 100 * se / abs(value) if value != 0 else np.nan

        results[name] = {
            "value": value,
            "std_error": se,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "relative_error_pct": rel_err,
            "acceptance_rate": np.nan,
        }

    diagnostics = {
        "method": "hessian",
        "status": "ok",
        "degrees_of_freedom": int(degrees_of_freedom),
        "sse": float(sse),
        "residual_variance": float(residual_variance),
        "hessian_condition_number": float(np.linalg.cond(hessian)) if np.all(np.isfinite(hessian)) else np.nan,
        "used_pseudoinverse": bool(used_pinv),
    }

    return results, diagnostics


def _calculate_ci_hessian_log_transform(
    objective_function,
    optimal_params: np.ndarray,
    parameter_names: list,
    n_observations: int,
    confidence_level: float,
    bounds: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Any]]:
    """
    Hessian CIs using a mixed transform:
    - Positive/bounded-positive parameters are estimated in log-space
    - Others remain in linear space
    """
    from scipy.stats import t as t_dist

    n_params = len(optimal_params)
    degrees_of_freedom = n_observations - n_params

    if degrees_of_freedom <= 0:
        return _empty_ci(parameter_names, optimal_params), {
            "method": "hessian_log",
            "status": "unavailable",
            "message": "Insufficient degrees of freedom",
            "degrees_of_freedom": int(degrees_of_freedom),
        }

    can_log = np.array([val > 0 for val in optimal_params], dtype=bool)
    if bounds is not None:
        for i, name in enumerate(parameter_names):
            b = bounds.get(name)
            if b is not None and b[0] is not None and b[0] <= 0:
                can_log[i] = False

    z_opt = np.array(optimal_params, dtype=float)
    z_opt[can_log] = np.log(z_opt[can_log])

    def objective_in_z(z):
        theta = np.array(z, dtype=float)
        theta[can_log] = np.exp(theta[can_log])
        return objective_function(theta)

    hessian_z = _compute_hessian(objective_in_z, z_opt)
    sse = objective_function(optimal_params)
    residual_variance = sse / degrees_of_freedom

    used_pinv = False
    try:
        covariance_z = np.linalg.inv(hessian_z / 2) * residual_variance
    except np.linalg.LinAlgError:
        covariance_z = np.linalg.pinv(hessian_z / 2) * residual_variance
        used_pinv = True

    variances_z = np.clip(np.diag(covariance_z), a_min=0.0, a_max=None)
    std_errors_z = np.sqrt(variances_z)

    alpha = 1 - confidence_level
    t_value = t_dist.ppf(1 - alpha / 2, degrees_of_freedom)

    results = {}
    for i, name in enumerate(parameter_names):
        value = float(optimal_params[i])
        se_z = float(std_errors_z[i]) if not np.isnan(std_errors_z[i]) else np.nan

        if np.isnan(se_z) or se_z <= 0:
            ci_lower, ci_upper, se_theta, rel_err = np.nan, np.nan, np.nan, np.nan
        elif can_log[i]:
            ci_lower = float(np.exp(z_opt[i] - t_value * se_z))
            ci_upper = float(np.exp(z_opt[i] + t_value * se_z))
            se_theta = abs(value) * se_z  # delta method
            rel_err = 100 * se_theta / abs(value) if value != 0 else np.nan
        else:
            se_theta = se_z
            ci_lower = value - t_value * se_theta
            ci_upper = value + t_value * se_theta
            rel_err = 100 * se_theta / abs(value) if value != 0 else np.nan

        results[name] = {
            "value": value,
            "std_error": se_theta,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "relative_error_pct": rel_err,
            "acceptance_rate": np.nan,
        }

    diagnostics = {
        "method": "hessian_log",
        "status": "ok",
        "degrees_of_freedom": int(degrees_of_freedom),
        "sse": float(sse),
        "residual_variance": float(residual_variance),
        "n_log_transformed": int(np.sum(can_log)),
        "log_transformed_parameters": [parameter_names[i] for i in range(len(parameter_names)) if can_log[i]],
        "hessian_condition_number": float(np.linalg.cond(hessian_z)) if np.all(np.isfinite(hessian_z)) else np.nan,
        "used_pseudoinverse": bool(used_pinv),
    }

    return results, diagnostics


def _autocorrelation_1d(x: np.ndarray, lag: int) -> float:
    """Autocorrelation helper for ESS estimation."""
    n = len(x)
    if lag >= n:
        return 0.0
    x0 = x - np.mean(x)
    denom = np.dot(x0, x0)
    if denom <= 0:
        return 0.0
    num = np.dot(x0[:-lag], x0[lag:]) if lag > 0 else denom
    return float(num / denom)


def _rhat(chains: np.ndarray) -> float:
    """Gelman-Rubin R-hat for chains with shape (m, n)."""
    m, n = chains.shape
    if m < 2 or n < 4:
        return np.nan
    chain_means = np.mean(chains, axis=1)
    chain_vars = np.var(chains, axis=1, ddof=1)
    W = np.mean(chain_vars)
    B = n * np.var(chain_means, ddof=1)
    if W <= 0:
        return np.nan
    var_hat = ((n - 1) / n) * W + (B / n)
    return float(np.sqrt(var_hat / W))


def _effective_sample_size(chains: np.ndarray, max_lag: int = 100) -> float:
    """Approximate ESS using mean within-chain autocorrelation."""
    m, n = chains.shape
    if n < 4:
        return float(m * n)
    max_lag = min(max_lag, n - 1)
    rho_sum = 0.0
    for lag in range(1, max_lag + 1):
        rhos = [_autocorrelation_1d(chains[i], lag) for i in range(m)]
        rho = float(np.mean(rhos))
        if rho <= 0:
            break
        rho_sum += rho
    ess = (m * n) / (1 + 2 * rho_sum)
    return float(max(1.0, min(ess, float(m * n))))


def _calculate_ci_mcmc(
    objective_function,
    optimal_params: np.ndarray,
    parameter_names: list,
    confidence_level: float,
    bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    n_samples: int = 4000,
    burn_in: int = 1000,
    step_scale: float = 0.05,
    random_seed: Optional[int] = None,
    n_chains: int = 5,
    mcmc_adaptive: bool = False,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Any]]:
    """Random-walk Metropolis MCMC for posterior interval estimation."""
    n_params = len(optimal_params)
    n_samples = max(int(n_samples), 200)
    burn_in = max(int(burn_in), 50)
    step_scale = float(max(step_scale, 1e-6))
    n_chains = max(int(n_chains), 2)

    rng = np.random.default_rng(random_seed)

    # Build numeric bounds vectors
    lb = np.zeros(n_params)
    ub = np.zeros(n_params)
    for i, name in enumerate(parameter_names):
        if bounds and name in bounds and bounds[name] is not None:
            b0, b1 = bounds[name]
            lb[i] = float(b0)
            ub[i] = float(b1)
        else:
            val = float(optimal_params[i])
            span = max(abs(val), 1.0)
            lb[i] = val - 10.0 * span
            ub[i] = val + 10.0 * span

    base = np.clip(np.array(optimal_params, dtype=float), lb, ub)
    proposal_sigma = step_scale * np.maximum(ub - lb, 1e-8)

    # ── Adaptive covariance (Haario et al. 2001) ──────────────────
    hessian_cov = None
    if mcmc_adaptive:
        try:
            H = _compute_hessian(objective_function, base)
            if np.isfinite(H).all():
                info_matrix = H / 2.0  # approx Fisher information
                cov = np.linalg.inv(info_matrix)
                # Sanity: require positive diagonal
                if np.all(np.diag(cov) > 0):
                    hessian_cov = cov
        except Exception:
            pass  # fallback to range-based proposal

    def log_posterior(theta: np.ndarray) -> float:
        if np.any(theta < lb) or np.any(theta > ub):
            return -np.inf
        sse = objective_function(theta)
        if not np.isfinite(sse):
            return -np.inf
        # Gaussian error model with unknown variance marginalized up to a constant.
        return -0.5 * float(sse)

    chains_samples = np.zeros((n_chains, n_samples, n_params), dtype=float)
    accepted_per_chain = np.zeros(n_chains, dtype=int)

    for c in range(n_chains):
        jitter = rng.normal(0.0, 0.01 * np.maximum(ub - lb, 1e-8), size=n_params)
        current = np.clip(base + jitter, lb, ub)
        current_lp = log_posterior(current)
        stored = 0

        for it in range(n_samples + burn_in):
            if mcmc_adaptive:
                proposal = adaptive_mcmc_proposal(
                    current, hessian_cov, lb, ub,
                    fallback_scale=step_scale, rng=rng,
                )
            else:
                proposal = current + rng.normal(0.0, proposal_sigma, size=n_params)
            proposal_lp = log_posterior(proposal)

            if np.isfinite(proposal_lp):
                log_alpha = proposal_lp - current_lp
                if np.log(rng.uniform()) < log_alpha:
                    current = proposal
                    current_lp = proposal_lp
                    accepted_per_chain[c] += 1

            if it >= burn_in and stored < n_samples:
                chains_samples[c, stored, :] = current
                stored += 1

    if chains_samples.size == 0:
        return _empty_ci(parameter_names, optimal_params), _empty_ci_diagnostics("mcmc")

    alpha = 1 - confidence_level
    q_low = 100 * (alpha / 2)
    q_high = 100 * (1 - alpha / 2)
    total_draws = float(n_chains * (n_samples + burn_in))
    acceptance_rate = float(np.sum(accepted_per_chain) / total_draws)
    flat_samples = chains_samples.reshape(n_chains * n_samples, n_params)

    results = {}
    rhat_by_param = {}
    ess_by_param = {}
    for i, name in enumerate(parameter_names):
        value = float(optimal_params[i])
        chain = flat_samples[:, i]
        chain_matrix = chains_samples[:, :, i]
        se = float(np.std(chain, ddof=1)) if len(chain) > 1 else np.nan
        ci_lower = float(np.percentile(chain, q_low))
        ci_upper = float(np.percentile(chain, q_high))
        rel_err = 100 * se / abs(value) if value != 0 and not np.isnan(se) else np.nan
        rhat_val = _rhat(chain_matrix)
        ess_val = _effective_sample_size(chain_matrix)

        rhat_by_param[name] = rhat_val
        ess_by_param[name] = ess_val

        results[name] = {
            "value": value,
            "std_error": se,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "relative_error_pct": rel_err,
            "acceptance_rate": acceptance_rate,
            "r_hat": rhat_val,
            "effective_sample_size": ess_val,
        }

    diagnostics = {
        "method": "mcmc",
        "status": "ok",
        "n_chains": int(n_chains),
        "n_samples": int(n_samples),
        "burn_in": int(burn_in),
        "step_scale": float(step_scale),
        "mcmc_adaptive": mcmc_adaptive,
        "adaptive_cov_available": hessian_cov is not None if mcmc_adaptive else False,
        "acceptance_rate": acceptance_rate,
        "acceptance_rate_per_chain": [float(v / (n_samples + burn_in)) for v in accepted_per_chain],
        "r_hat": {k: float(v) if np.isfinite(v) else np.nan for k, v in rhat_by_param.items()},
        "effective_sample_size": {k: float(v) for k, v in ess_by_param.items()},
        "trace_samples": {
            name: chains_samples[:, :, i].tolist()
            for i, name in enumerate(parameter_names)
        },
    }

    return results, diagnostics


def _compute_hessian(func, x, epsilon=None):
    """
    Compute the Hessian matrix using central finite differences.

    Uses simple central differences requiring O(n²) function evaluations,
    making it practical for expensive objective functions (e.g., those
    involving ODE solves across multiple experimental conditions).

    Args:
        func: Objective function
        x: Point at which to compute Hessian
        epsilon: Fixed step size. If None, adaptive step sizes are used
                 based on parameter magnitudes (h_i = max(|x_i|, 1) * eps^(1/3)).

    Returns:
        Hessian matrix (n x n)
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    f0 = func(x)

    # Adaptive step size: eps^(1/3) is optimal for central differences
    eps_third = np.finfo(float).eps ** (1.0 / 3.0)
    if epsilon is not None:
        h = np.full(n, float(epsilon))
    else:
        h = eps_third * np.maximum(np.abs(x), 1.0)

    hessian = np.zeros((n, n))

    # Evaluate f(x ± h_i) for each parameter — used for diagonal terms
    f_plus = np.empty(n)
    f_minus = np.empty(n)
    for i in range(n):
        x_p = x.copy()
        x_m = x.copy()
        x_p[i] += h[i]
        x_m[i] -= h[i]
        f_plus[i] = func(x_p)
        f_minus[i] = func(x_m)
        # Diagonal: H[i,i] = (f(x+h) - 2f(x) + f(x-h)) / h²
        hessian[i, i] = (f_plus[i] - 2.0 * f0 + f_minus[i]) / (h[i] ** 2)

    # Off-diagonal: H[i,j] = (f(x+h_i+h_j) - f(x+h_i-h_j)
    #                        - f(x-h_i+h_j) + f(x-h_i-h_j)) / (4 h_i h_j)
    for i in range(n):
        for j in range(i + 1, n):
            x_pp = x.copy()
            x_pm = x.copy()
            x_mp = x.copy()
            x_mm = x.copy()

            x_pp[i] += h[i]; x_pp[j] += h[j]
            x_pm[i] += h[i]; x_pm[j] -= h[j]
            x_mp[i] -= h[i]; x_mp[j] += h[j]
            x_mm[i] -= h[i]; x_mm[j] -= h[j]

            val = (func(x_pp) - func(x_pm) - func(x_mp) + func(x_mm)) / (4.0 * h[i] * h[j])
            hessian[i, j] = val
            hessian[j, i] = val

    return hessian


def format_confidence_intervals(ci_results: Dict[str, Dict[str, float]]) -> str:
    """Format confidence interval results as a readable string."""
    lines = ["Parameter Confidence Intervals (95%):", "-" * 60]
    
    for name, info in ci_results.items():
        if np.isnan(info["std_error"]):
            lines.append(f"  {name}: {info['value']:.4f} (CI unavailable - check identifiability)")
        else:
            lines.append(
                f"  {name}: {info['value']:.4f} ± {info['std_error']:.4f} "
                f"[{info['ci_lower']:.4f}, {info['ci_upper']:.4f}] "
                f"({info['relative_error_pct']:.1f}% rel. error)"
            )
    
    return "\n".join(lines)
