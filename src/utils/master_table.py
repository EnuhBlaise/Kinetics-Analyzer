"""
Master results table utility for aggregating individual condition fitting results.

Provides functions to extract global parameters and fit quality metrics from
individual condition results and append them to a master CSV for systematic
comparison across substrates and models.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, List

from src.io.config_loader import load_config
from src.io.data_loader import load_experimental_data
from src.core.ode_systems import SingleMonodODE, SingleMonodLagODE, DualMonodODE, DualMonodLagODE
from src.core.oxygen import OxygenModel
from src.core.solvers import solve_ode
from src.fitting.statistics import calculate_separate_statistics


# All possible parameter columns in canonical order
PARAMETER_COLUMNS = [
    'qmax', 'Ks', 'Ki', 'Y', 'K_O2', 'Y_O2', 'b_decay', 'lag_time',
]

# Parameters used by each model type
MODEL_PARAMETERS = {
    'single_monod': ['qmax', 'Ks', 'Ki', 'Y', 'b_decay'],
    'single_monod_lag': ['qmax', 'Ks', 'Ki', 'Y', 'b_decay', 'lag_time'],
    'dual_monod': ['qmax', 'Ks', 'Ki', 'Y', 'b_decay', 'K_o2', 'Y_o2'],
    'dual_monod_lag': ['qmax', 'Ks', 'Ki', 'Y', 'b_decay', 'K_o2', 'Y_o2', 'lag_time'],
}

# Mapping from JSON parameter keys to master table column names
_PARAM_TO_COLUMN = {
    'qmax': 'qmax',
    'Ks': 'Ks',
    'Ki': 'Ki',
    'Y': 'Y',
    'K_o2': 'K_O2',
    'Y_o2': 'Y_O2',
    'b_decay': 'b_decay',
    'lag_time': 'lag_time',
}

# Full ordered list of master table columns
MASTER_COLUMNS = [
    'Substrate', 'Model',
    'qmax', 'qmax_95CI',
    'Ks', 'Ks_95CI',
    'Ki', 'Ki_95CI',
    'Y', 'Y_95CI',
    'K_O2', 'K_O2_95CI',
    'Y_O2', 'Y_O2_95CI',
    'b_decay', 'b_decay_95CI',
    'lag_time', 'lag_time_95CI',
    'Total_Error', 'R2', 'AIC', 'AIC_weight',
]


def append_to_master_table(
    results_dir: str,
    master_csv: str = "output/master_results.csv",
    config_path: Optional[str] = None,
    data_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Extract global parameters and fit metrics from a results directory
    and append them as a row to the master CSV.

    Args:
        results_dir: Path to the individual condition results directory
            (containing individual_condition_results.json).
        master_csv: Path to the master CSV file. Created if it doesn't exist.
        config_path: Path to substrate config JSON. If None, read from the
            results JSON (requires the run to have stored it).
        data_path: Path to experimental data CSV. If None, read from the
            results JSON.

    Returns:
        The updated master DataFrame.
    """
    results_dir = Path(results_dir)
    json_path = results_dir / 'individual_condition_results.json'
    if not json_path.exists():
        raise FileNotFoundError(
            f"Results JSON not found: {json_path}"
        )

    with open(json_path, 'r') as f:
        results = json.load(f)

    substrate = results['substrate']
    model_type = results['model_type']
    display_name = results.get('display_name', model_type)
    global_params = results.get('global_parameters')
    global_cis = results.get('global_confidence_intervals')

    if global_params is None:
        raise ValueError(
            "No global_parameters found in results JSON. "
            "Re-run fit_individual.py to generate global parameters."
        )

    # Resolve config and data paths
    config_path = config_path or results.get('config_path')
    data_path = data_path or results.get('data_path')

    # Build the row — use display_name for human-readable Model column
    row: Dict[str, object] = {
        'Substrate': substrate,
        'Model': display_name,
    }

    # Parameters and CIs — use actual keys present in global_params
    # (handles both Haldane models with Ki and basic models without)
    model_params = MODEL_PARAMETERS.get(model_type, [])
    for json_key in model_params:
        col = _PARAM_TO_COLUMN.get(json_key, json_key)
        value = global_params.get(json_key)
        row[col] = value

        ci_col = f'{col}_95CI'
        if global_cis and json_key in global_cis:
            ci = global_cis[json_key]
            ci_lo = ci.get('ci_lower')
            ci_hi = ci.get('ci_upper')
            if ci_lo is not None and ci_hi is not None:
                row[ci_col] = f'[{ci_lo:.6f}, {ci_hi:.6f}]'
            else:
                row[ci_col] = None
        else:
            row[ci_col] = None

    # Re-evaluate Total_Error (raw SSE), R², and AIC using global parameters
    total_sse, r2, aic = _reevaluate_fit_metrics(
        results=results,
        global_params=global_params,
        model_type=model_type,
        config_path=config_path,
        data_path=data_path,
    )
    row['Total_Error'] = total_sse
    row['R2'] = r2
    row['AIC'] = aic
    row['AIC_weight'] = np.nan  # will be computed after all rows are present

    # Load or create master table
    master_csv = Path(master_csv)
    master_csv.parent.mkdir(parents=True, exist_ok=True)

    if master_csv.exists():
        df = pd.read_csv(master_csv)
        # Ensure all expected columns exist
        for col in MASTER_COLUMNS:
            if col not in df.columns:
                df[col] = np.nan
    else:
        df = pd.DataFrame(columns=MASTER_COLUMNS)

    # Duplicate handling: replace row if (Substrate, Model) already exists
    mask = (df['Substrate'] == substrate) & (df['Model'] == display_name)
    if mask.any():
        df = df[~mask]

    new_row = pd.DataFrame([row])
    df = pd.concat([df, new_row], ignore_index=True)

    # Recompute AIC weights per substrate group
    df = update_aic_weights(df)

    # Ensure column order
    ordered_cols = [c for c in MASTER_COLUMNS if c in df.columns]
    extra_cols = [c for c in df.columns if c not in MASTER_COLUMNS]
    df = df[ordered_cols + extra_cols]

    df.to_csv(master_csv, index=False)
    return df


def update_aic_weights(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recompute AIC weights per substrate group using the delta-AIC method.

    w_i = exp(-0.5 * delta_i) / sum(exp(-0.5 * delta_j))
    where delta_i = AIC_i - AIC_min within the substrate group.

    Args:
        df: Master results DataFrame (modified in place and returned).

    Returns:
        DataFrame with updated AIC_weight column.
    """
    if 'AIC_weight' not in df.columns:
        df['AIC_weight'] = np.nan

    for substrate, group in df.groupby('Substrate'):
        aic_values = pd.to_numeric(group['AIC'], errors='coerce')
        valid = aic_values.notna() & np.isfinite(aic_values)

        if valid.sum() == 0:
            df.loc[group.index, 'AIC_weight'] = np.nan
            continue

        aic_min = aic_values[valid].min()
        delta = aic_values - aic_min
        weights = np.exp(-0.5 * delta)
        weights[~valid] = 0.0
        total = weights.sum()

        if total > 0:
            df.loc[group.index, 'AIC_weight'] = weights / total
        else:
            df.loc[group.index, 'AIC_weight'] = np.nan

    return df


def load_master_table(master_csv: str = "output/master_results.csv") -> pd.DataFrame:
    """
    Load and return the master results table.

    Args:
        master_csv: Path to the master CSV file.

    Returns:
        Master results DataFrame.
    """
    master_csv = Path(master_csv)
    if not master_csv.exists():
        raise FileNotFoundError(f"Master table not found: {master_csv}")
    return pd.read_csv(master_csv)


def _reevaluate_fit_metrics(
    results: dict,
    global_params: dict,
    model_type: str,
    config_path: Optional[str],
    data_path: Optional[str],
) -> tuple:
    """
    Re-evaluate fit metrics using global parameters against all conditions.

    Returns:
        Tuple of (total_raw_sse, combined_r2, summed_aic).
        total_raw_sse is the sum of substrate SSE + biomass SSE across
        all conditions (not normalized).
    """
    if config_path is None or data_path is None:
        # Cannot re-evaluate without config and data paths; fall back to
        # per-condition stats from JSON if available.
        return _fallback_metrics(results)

    try:
        config = load_config(config_path)
        substrate_name = config.name
        experimental_data = load_experimental_data(data_path, substrate_name=substrate_name)
    except Exception:
        return _fallback_metrics(results)

    # Setup oxygen model
    oxygen_model = OxygenModel(
        o2_max=config.oxygen.get("o2_max", 8.0),
        o2_min=config.oxygen.get("o2_min", 0.1),
        reaeration_rate=config.oxygen.get("reaeration_rate", 15.0),
        o2_range=config.oxygen.get("o2_range", 8.0),
    )

    param_names = MODEL_PARAMETERS.get(model_type, [])
    n_params = len(param_names)

    r2_list = []
    aic_list = []
    total_sse = 0.0

    for condition in experimental_data.conditions:
        try:
            time, substrate, biomass = experimental_data.get_condition_data(condition)
        except ValueError:
            continue

        S0, X0 = substrate[0], biomass[0]
        if model_type in ('single_monod', 'single_monod_lag'):
            initial_conditions = [S0, X0]
        else:
            initial_conditions = [S0, X0, oxygen_model.o2_max]

        # Create ODE system
        ode_system = _create_ode_system(
            global_params, model_type, oxygen_model
        )

        t_span = (time[0], time[-1])
        n_points = max(1000, len(time) * 10)
        t_eval = np.linspace(t_span[0], t_span[1], n_points)

        try:
            result = solve_ode(
                ode_system=ode_system,
                initial_conditions=np.array(initial_conditions),
                t_span=t_span,
                t_eval=t_eval,
            )
        except Exception:
            continue

        pred_sub = result.states.get('S', result.states.get('Substrate', np.zeros_like(result.time)))
        pred_bio = result.states.get('X', result.states.get('Biomass', np.zeros_like(result.time)))

        pred_sub_interp = np.interp(time, result.time, pred_sub)
        pred_bio_interp = np.interp(time, result.time, pred_bio)

        stats = calculate_separate_statistics(
            observed_substrate=substrate,
            predicted_substrate=pred_sub_interp,
            observed_biomass=biomass,
            predicted_biomass=pred_bio_interp,
            n_parameters=n_params,
        )

        r2_list.append(stats['combined']['R_squared'])
        aic_list.append(stats['combined']['AIC'])
        total_sse += stats['substrate']['SSE'] + stats['biomass']['SSE']

    if not r2_list:
        return _fallback_metrics(results)

    combined_r2 = float(np.mean(r2_list))
    summed_aic = float(np.sum(aic_list))

    return float(total_sse), combined_r2, summed_aic


def _fallback_metrics(results: dict) -> tuple:
    """
    Extract total SSE, average R², and summed AIC from per-condition stats
    in the JSON when re-evaluation is not possible.
    """
    conditions = results.get('conditions', {})
    if not conditions:
        return np.nan, np.nan, np.nan

    r2_vals = []
    aic_vals = []
    total_sse = 0.0
    has_sse = False
    for cond_data in conditions.values():
        stats = cond_data.get('statistics', {})
        combined = stats.get('combined', {})
        sub_stats = stats.get('substrate', {})
        bio_stats = stats.get('biomass', {})
        r2 = combined.get('R_squared')
        aic = combined.get('AIC')
        sse_sub = sub_stats.get('SSE')
        sse_bio = bio_stats.get('SSE')
        if r2 is not None:
            r2_vals.append(r2)
        if aic is not None:
            aic_vals.append(aic)
        if sse_sub is not None and sse_bio is not None:
            total_sse += sse_sub + sse_bio
            has_sse = True

    r2 = float(np.mean(r2_vals)) if r2_vals else np.nan
    aic = float(np.sum(aic_vals)) if aic_vals else np.nan
    sse = float(total_sse) if has_sse else np.nan
    return sse, r2, aic


def _create_ode_system(params: dict, model_type: str, oxygen_model: OxygenModel):
    """Create an ODE system from a parameter dictionary."""
    ki_value = params.get('Ki', None)

    if model_type == 'single_monod':
        return SingleMonodODE(
            qmax=params['qmax'],
            Ks=params['Ks'],
            Ki=ki_value,
            Y=params['Y'],
            b_decay=params['b_decay'],
        )
    elif model_type == 'single_monod_lag':
        return SingleMonodLagODE(
            qmax=params['qmax'],
            Ks=params['Ks'],
            Ki=ki_value,
            Y=params['Y'],
            b_decay=params['b_decay'],
            lag_time=params['lag_time'],
        )
    elif model_type == 'dual_monod':
        return DualMonodODE(
            qmax=params['qmax'],
            Ks=params['Ks'],
            Ki=ki_value,
            Y=params['Y'],
            b_decay=params['b_decay'],
            K_o2=params['K_o2'],
            Y_o2=params['Y_o2'],
            oxygen_model=oxygen_model,
        )
    elif model_type == 'dual_monod_lag':
        return DualMonodLagODE(
            qmax=params['qmax'],
            Ks=params['Ks'],
            Ki=ki_value,
            Y=params['Y'],
            b_decay=params['b_decay'],
            K_o2=params['K_o2'],
            Y_o2=params['Y_o2'],
            lag_time=params['lag_time'],
            oxygen_model=oxygen_model,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
