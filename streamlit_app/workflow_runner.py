"""
Workflow runner utilities for Streamlit integration.

This module provides helper functions to run kinetic parameter
estimation workflows from the Streamlit interface.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import tempfile
import json
import signal
import threading
from typing import Dict, Any, Optional, Tuple
from dataclasses import asdict

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TimeoutError(Exception):
    """Custom timeout error for fitting."""
    pass


def run_fitting_workflow(
    experimental_df: pd.DataFrame,
    config_dict: dict,
    model_type: str = "dual_monod_lag",
    fit_method: str = "global",
    optimizer: str = "L-BFGS-B",
    verbose: bool = False,
    timeout: int = 120,
    weighting_strategy: str = "max_value",
    use_two_stage: bool = True,
    bootstrap_iterations: int = 500,
    bootstrap_workers: int = 4
) -> Dict[str, Any]:
    """
    Run the parameter fitting workflow.

    Args:
        experimental_df: Experimental data DataFrame
        config_dict: Configuration dictionary
        model_type: Type of model to use
        fit_method: Fitting method ('global', 'individual', or 'robust')
        optimizer: Optimizer to use
        verbose: Whether to print progress
        timeout: Maximum time in seconds for fitting
        weighting_strategy: Weighting strategy for robust fitting
        use_two_stage: Whether to use two-stage initialization (robust fitting)
        bootstrap_iterations: Number of bootstrap iterations (robust fitting)
        bootstrap_workers: Number of parallel workers for bootstrap

    Returns:
        Results dictionary
    """
    from src.io.config_loader import SubstrateConfig
    from src.io.data_loader import load_experimental_data
    from workflows.single_monod import SingleMonodWorkflow
    from workflows.single_monod_lag import SingleMonodLagWorkflow
    from workflows.dual_monod import DualMonodWorkflow
    from workflows.dual_monod_lag import DualMonodLagWorkflow

    # Save data to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        experimental_df.to_csv(f, index=False)
        data_path = f.name

    # Create SubstrateConfig
    # Convert bounds from list to tuple if needed
    bounds = {}
    for param, bound_vals in config_dict.get('bounds', {}).items():
        if isinstance(bound_vals, list):
            bounds[param] = tuple(bound_vals)
        else:
            bounds[param] = bound_vals

    substrate_config = SubstrateConfig(
        name=config_dict.get('substrate', {}).get('name', 'Unknown'),
        molecular_weight=config_dict.get('substrate', {}).get('molecular_weight', 180.0),
        unit=config_dict.get('substrate', {}).get('unit', 'mg/L'),
        initial_guesses=config_dict.get('initial_guesses', {}),
        bounds=bounds,
        oxygen=config_dict.get('oxygen', {
            'o2_max': 8.0,
            'o2_min': 0.1,
            'reaeration_rate': 15.0,
            'o2_range': 8.0
        }),
        simulation=config_dict.get('simulation', {
            't_final': 0.5,
            'num_points': 1000,
            'time_unit': 'days'
        })
    )

    # Load experimental data
    exp_data = load_experimental_data(data_path, substrate_config.name)

    # Robust fitting pathway
    if fit_method == "robust":
        return _run_robust_fitting(
            exp_data, substrate_config, model_type,
            weighting_strategy, use_two_stage,
            bootstrap_iterations, bootstrap_workers, verbose
        )

    # Individual fitting pathway
    if fit_method == "individual":
        return _run_individual_fitting(
            exp_data, substrate_config, model_type, optimizer, verbose
        )

    # Standard fitting pathway
    # Map all 6 CLI names to base workflow classes.
    # Haldane variants use the same workflow (Ki is included by default).
    workflow_map = {
        'single_monod': SingleMonodWorkflow,
        'single_haldane': SingleMonodWorkflow,
        'single_monod_lag': SingleMonodLagWorkflow,
        'single_haldane_lag': SingleMonodLagWorkflow,
        'dual_monod': DualMonodWorkflow,
        'dual_haldane': DualMonodWorkflow,
        'dual_monod_lag': DualMonodLagWorkflow,
        'dual_haldane_lag': DualMonodLagWorkflow,
    }

    WorkflowClass = workflow_map.get(model_type, DualMonodLagWorkflow)

    # Create output directory
    output_dir = PROJECT_ROOT / "results" / "streamlit"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create and run workflow
    workflow = WorkflowClass(
        config=substrate_config,
        experimental_data=exp_data,
        output_dir=str(output_dir)
    )

    # Run fitting with L-BFGS-B (fast) optimizer
    # Use the specified optimizer, default to L-BFGS-B for speed
    opt_method = optimizer if optimizer != "Differential Evolution" else "L-BFGS-B"

    result = workflow.run(
        fit_method=fit_method,
        optimization_method=opt_method,
        generate_plots=False,
        save_results=False,  # Don't save to disk in GUI mode for speed
        verbose=verbose
    )

    # Build predictions DataFrame
    predictions_df = _build_predictions_df(result, workflow, exp_data)

    # Extract and return results
    return {
        'success': result.optimization_result.success,
        'model_type': model_type,
        'conditions': result.conditions,
        'parameters': result.optimization_result.parameters,
        'confidence_intervals': result.confidence_intervals,
        'statistics': result.statistics,
        'units': _get_parameter_units(substrate_config.name, result.optimization_result.parameters),
        'predictions': predictions_df,
        'message': result.optimization_result.message if hasattr(result.optimization_result, 'message') else '',
        'output_dir': str(output_dir)
    }


def _run_robust_fitting(
    exp_data, config, model_type: str,
    weighting_strategy: str, use_two_stage: bool,
    bootstrap_iterations: int, bootstrap_workers: int,
    verbose: bool
) -> Dict[str, Any]:
    """Run robust fitting with weighting, two-stage init, and bootstrap."""
    from src.fitting.robust_fitter import RobustFitter

    # Build conditions list from experimental data
    conditions = []
    for label in exp_data.conditions:
        time, substrate, biomass = exp_data.get_condition_data(label)
        S0 = substrate[0]
        X0 = biomass[0]

        if model_type in ["dual_monod", "dual_haldane", "dual_monod_lag", "dual_haldane_lag"]:
            o2_max = config.oxygen.get('o2_max', 8.0)
            initial_conditions = [S0, X0, o2_max]
        else:
            initial_conditions = [S0, X0]

        conditions.append({
            'time': time,
            'substrate': substrate,
            'biomass': biomass,
            'initial_conditions': initial_conditions,
            't_span': (time[0], time[-1]),
            'label': label
        })

    # Create and run robust fitter
    fitter = RobustFitter(
        model_type=model_type,
        weighting=weighting_strategy,
        use_two_stage=use_two_stage,
        bootstrap_iterations=bootstrap_iterations,
        bootstrap_workers=bootstrap_workers
    )

    result = fitter.fit(conditions, config, verbose=verbose)

    # Convert bootstrap CIs to the dict format expected by results_display
    ci_display = {}
    for param, (lo, hi) in result.confidence_intervals.items():
        val = result.parameters[param]
        ci_display[param] = {
            'ci_lower': lo,
            'ci_upper': hi,
            'std_error': (hi - lo) / (2 * 1.96),
            'relative_error_pct': ((hi - lo) / (2 * abs(val)) * 100) if val != 0 else 0
        }

    # Build a combined statistics dict
    combined_stats = {}
    for cond_label, stats in result.statistics.items():
        combined_stats.update({
            f'R_squared_{cond_label}': stats.get('r_squared', 0),
        })
    # Add overall averages
    all_r2 = [s.get('r_squared', 0) for s in result.statistics.values()]
    combined_stats['R_squared'] = np.mean(all_r2) if all_r2 else 0

    return {
        'success': True,
        'model_type': model_type,
        'conditions': [c['label'] for c in conditions],
        'parameters': result.parameters,
        'confidence_intervals': ci_display,
        'statistics': combined_stats,
        'units': _get_parameter_units(config.name, result.parameters),
        'predictions': pd.DataFrame(),
        'message': f'Robust fit complete. Bootstrap: {bootstrap_iterations} iterations.',
        'output_dir': str(PROJECT_ROOT / "results" / "streamlit"),
        'diagnostics': result.diagnostics,
        'per_condition_statistics': result.statistics
    }


def _run_individual_fitting(
    exp_data, config, model_type: str,
    optimizer: str, verbose: bool
) -> Dict[str, Any]:
    """Run individual condition fitting with per-condition parameters and global estimation."""
    from workflows.individual_condition import IndividualConditionWorkflow

    output_dir = PROJECT_ROOT / "results" / "streamlit"
    output_dir.mkdir(parents=True, exist_ok=True)

    opt_method = optimizer if optimizer != "Differential Evolution" else "L-BFGS-B"

    # Decompose the 6-name model_type into base type + no_inhibition flag
    _model_mapping = {
        'single_monod':       ('single_monod', True),
        'single_haldane':     ('single_monod', False),
        'single_monod_lag':   ('single_monod_lag', True),
        'single_haldane_lag': ('single_monod_lag', False),
        'dual_monod':         ('dual_monod', True),
        'dual_haldane':       ('dual_monod', False),
        'dual_monod_lag':     ('dual_monod_lag', True),
        'dual_haldane_lag':   ('dual_monod_lag', False),
    }
    base_type, no_inhibition = _model_mapping.get(model_type, (model_type, False))

    workflow = IndividualConditionWorkflow(
        config=config,
        experimental_data=exp_data,
        model_type=base_type,
        output_dir=str(output_dir),
        no_inhibition=no_inhibition,
    )

    result = workflow.run(
        optimization_method=opt_method,
        generate_plots=False,
        save_results=False,
        verbose=verbose
    )

    # Build per-condition predictions DataFrame
    all_predictions = []
    for cond, cond_result in result.condition_results.items():
        pred_df = cond_result.predictions.copy()
        pred_df['Condition'] = cond
        all_predictions.append(pred_df)

    predictions_df = pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()

    # Build per-condition results for display
    per_condition_results = {}
    for cond, cond_result in result.condition_results.items():
        per_condition_results[cond] = {
            'parameters': cond_result.parameters,
            'confidence_intervals': cond_result.confidence_intervals,
            'statistics': cond_result.statistics,
            'residual_diagnostics': cond_result.residual_diagnostics,
            'success': cond_result.success
        }

    # Build global CIs in standard format
    global_ci_display = {}
    if result.global_confidence_intervals:
        for param, ci in result.global_confidence_intervals.items():
            global_ci_display[param] = {
                'std_error': ci.get('std_error', np.nan),
                'ci_lower': ci.get('ci_lower', np.nan),
                'ci_upper': ci.get('ci_upper', np.nan),
                'relative_error_pct': ci.get('relative_error_pct', np.nan)
            }

    return {
        'success': True,
        'fit_method': 'individual',
        'model_type': model_type,
        'conditions': list(result.condition_results.keys()),
        'parameters': result.global_parameters or {},
        'confidence_intervals': global_ci_display,
        'statistics': {},
        'units': _get_parameter_units(config.name, result.global_parameters or {}),
        'predictions': predictions_df,
        'message': 'Individual condition fitting complete.',
        'output_dir': str(output_dir),
        # Individual-specific data
        'per_condition_results': per_condition_results,
        'individual_losses': result.individual_losses,
        'parameter_summary': result.parameter_summary,
        'global_loss': result.global_loss,
        'global_optimization_converged': (
            result.global_optimization_result.success
            if result.global_optimization_result else None
        ),
        'global_n_function_evals': (
            result.global_optimization_result.n_function_evals
            if result.global_optimization_result else None
        ),
    }


def _build_predictions_df(result, workflow, exp_data) -> pd.DataFrame:
    """Build predictions DataFrame from workflow results."""
    
    if result.predictions is not None and not result.predictions.empty:
        return result.predictions
    
    # Generate predictions manually
    all_predictions = []
    
    for condition in result.conditions:
        try:
            time_exp, _, _ = exp_data.get_condition_data(condition)
            t_span = [0, max(time_exp) * 1.1]
            
            # Simulate with fitted parameters
            sim_result = workflow.simulate(
                result.optimization_result.parameters,
                condition,
                t_span=t_span,
                num_points=1000
            )
            
            pred_df = pd.DataFrame({
                'Time': sim_result.time,
                'Substrate': sim_result.states.get('Substrate', sim_result.states.get('S', np.zeros_like(sim_result.time))),
                'Biomass': sim_result.states.get('Biomass', sim_result.states.get('X', np.zeros_like(sim_result.time))),
                'Condition': condition
            })
            
            all_predictions.append(pred_df)
            
        except Exception as e:
            print(f"Could not generate predictions for {condition}: {e}")
    
    if all_predictions:
        return pd.concat(all_predictions, ignore_index=True)
    
    return pd.DataFrame()


def _get_parameter_units(substrate_name: str, parameters: dict) -> dict:
    """Get units for each parameter."""
    units = {
        "qmax": f"mg{substrate_name}/(mgCells·day)",
        "Ks": f"mg{substrate_name}/L",
        "Ki": f"mg{substrate_name}/L",
        "Y": f"mgCells/mg{substrate_name}",
        "b_decay": "day⁻¹",
        "K_o2": "mgO₂/L",
        "Y_o2": f"mgO₂/mg{substrate_name}",
        "lag_time": "days"
    }
    return {k: units.get(k, '') for k in parameters.keys()}


def validate_data_format(df: pd.DataFrame) -> Tuple[bool, str, list]:
    """
    Validate that a DataFrame has the expected format for fitting.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Tuple of (is_valid, message, detected_conditions)
    """
    issues = []
    conditions = []
    
    # Check for time column
    time_cols = [c for c in df.columns if 'time' in c.lower()]
    if not time_cols:
        issues.append("No time column found")
    
    # Check for substrate columns
    substrate_cols = [c for c in df.columns if '(mg/L)' in c.lower() and 'biomass' not in c.lower()]
    if not substrate_cols:
        # Also check for mM format
        substrate_cols = [c for c in df.columns if '(mM)' in c.lower() and 'biomass' not in c.lower()]
    
    if not substrate_cols:
        issues.append("No substrate concentration columns found")
    
    # Check for biomass columns
    biomass_cols = [c for c in df.columns if 'biomass' in c.lower()]
    if not biomass_cols:
        issues.append("No biomass columns found")
    
    # Detect conditions
    for col in df.columns:
        parts = col.split('_')
        if len(parts) >= 2:
            if 'mM' in parts[0] or 'mg' in parts[0].lower():
                if parts[0] not in conditions:
                    conditions.append(parts[0])
    
    conditions.sort(key=lambda x: float(''.join(c for c in x if c.isdigit()) or 0))
    
    is_valid = len(issues) == 0
    message = "Valid format" if is_valid else "; ".join(issues)
    
    return is_valid, message, conditions


def convert_numpy_types(obj):
    """Recursively convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(i) for i in obj]
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict()
    return obj
