#!/usr/bin/env python3
"""
Robust Parameter Fitting CLI

Combines three approaches for robust kinetic parameter estimation:
1. Condition-specific weighting (handles heteroscedasticity)
2. Two-stage initialization (avoids local minima)
3. Bootstrap uncertainty quantification (confidence intervals)

Usage:
    python scripts/robust_fit.py \
        --config config/substrates/xylose.json \
        --data data/xylose_data.csv \
        --workflow dual_monod_lag \
        --bootstrap 500 \
        --workers 4 \
        --output results/xylose/
"""

import argparse
import json
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from src.io.config_loader import load_config
from src.io.data_loader import load_experimental_data as load_exp_data
from src.fitting.robust_fitter import RobustFitter, RobustFitResult
from src.io.pdf_report import generate_robust_fit_report


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Robust parameter fitting with weighting, two-stage init, and bootstrap",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic robust fit with defaults
  python scripts/robust_fit.py --config config/substrates/glucose.json --data data/glucose.csv

  # Full robust fit with bootstrap
  python scripts/robust_fit.py \\
      --config config/substrates/xylose.json \\
      --data data/xylose.csv \\
      --workflow dual_monod_lag \\
      --weighting max_value \\
      --two-stage \\
      --bootstrap 500 \\
      --workers 4 \\
      --output results/xylose/
        """
    )

    # Required arguments
    parser.add_argument(
        "--config", "-c",
        required=True,
        help="Path to substrate config JSON file"
    )
    parser.add_argument(
        "--data", "-d",
        required=True,
        help="Path to experimental data CSV file"
    )

    # Model selection
    parser.add_argument(
        "--workflow", "-w",
        choices=["single_monod", "dual_monod", "dual_monod_lag"],
        default="single_monod",
        help="Model type to fit (default: single_monod)"
    )

    # Robust fitting options
    parser.add_argument(
        "--weighting",
        choices=["uniform", "max_value", "variance", "range"],
        default="max_value",
        help="Weighting strategy for heteroscedasticity (default: max_value)"
    )
    parser.add_argument(
        "--two-stage",
        action="store_true",
        default=True,
        help="Use two-stage initialization (default: enabled)"
    )
    parser.add_argument(
        "--no-two-stage",
        action="store_true",
        help="Disable two-stage initialization"
    )
    parser.add_argument(
        "--bootstrap", "-b",
        type=int,
        default=500,
        help="Number of bootstrap iterations (0 to disable, default: 500)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers for bootstrap (default: auto)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )

    # Output options
    parser.add_argument(
        "--output", "-o",
        default="results/",
        help="Output directory for results (default: results/)"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output"
    )

    return parser.parse_args()


def load_experimental_data(data_path: str, config, model_type: str = "single_monod") -> list:
    """
    Load experimental data and format as conditions list.

    Returns list of condition dictionaries for RobustFitter.
    """
    # Load data using the proper loader function
    exp_data = load_exp_data(data_path, substrate_name=config.name)

    conditions = []
    for label in exp_data.conditions:
        time, substrate, biomass = exp_data.get_condition_data(label)

        # Determine initial conditions based on model type
        S0 = substrate[0]
        X0 = biomass[0]

        # Only include oxygen for dual_monod and dual_monod_lag models
        if model_type in ["dual_monod", "dual_monod_lag"]:
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

    return conditions


def save_results(result: RobustFitResult, output_dir: str, config_name: str):
    """Save fitting results to files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save parameters JSON
    params_file = output_path / f"{config_name}_robust_params.json"
    params_data = {
        "model_type": result.model_type,
        "parameters": result.parameters,
        "confidence_intervals": {
            k: list(v) for k, v in result.confidence_intervals.items()
        },
        "statistics": result.statistics,
        "diagnostics": result.diagnostics,
        "fit_time_seconds": result.fit_time_seconds
    }
    with open(params_file, 'w') as f:
        json.dump(params_data, f, indent=2, default=str)
    print(f"Parameters saved to: {params_file}")

    # Save bootstrap distribution if available
    if result.bootstrap_result is not None:
        bootstrap_file = output_path / f"{config_name}_bootstrap_distribution.csv"
        bootstrap_df = pd.DataFrame(
            result.bootstrap_result.all_estimates,
            columns=result.param_names
        )
        bootstrap_df.to_csv(bootstrap_file, index=False)
        print(f"Bootstrap distribution saved to: {bootstrap_file}")

    # Save summary text
    summary_file = output_path / f"{config_name}_robust_summary.txt"
    with open(summary_file, 'w') as f:
        f.write(result.summary())
    print(f"Summary saved to: {summary_file}")

    # Generate PDF report
    try:
        figure_paths = list(output_path.glob('*.png')) + list(output_path.glob('figures/*.png'))
        bootstrap_info = {}
        if result.bootstrap_result is not None:
            bootstrap_info = {
                'n_iterations': result.bootstrap_result.n_iterations,
                'success_rate': result.bootstrap_result.success_rate,
            }
        pdf_path = generate_robust_fit_report(
            output_dir=output_path,
            summary_text=result.summary(),
            substrate_name=config_name,
            figure_paths=figure_paths,
            bootstrap_info=bootstrap_info,
            filename=f"{config_name}_results_report.pdf",
        )
        print(f"PDF report saved to: {pdf_path}")
    except Exception as e:
        print(f"Warning: PDF report generation failed: {e}")


def main():
    """Main entry point."""
    args = parse_args()

    # Determine two-stage setting
    use_two_stage = args.two_stage and not args.no_two_stage

    verbose = not args.quiet

    if verbose:
        print("=" * 60)
        print("ROBUST PARAMETER FITTING")
        print("=" * 60)
        print(f"Config: {args.config}")
        print(f"Data: {args.data}")
        print(f"Model: {args.workflow}")
        print(f"Weighting: {args.weighting}")
        print(f"Two-stage: {use_two_stage}")
        print(f"Bootstrap: {args.bootstrap} iterations")
        print("=" * 60)

    # Load configuration
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)

    # Load experimental data
    try:
        conditions = load_experimental_data(args.data, config, model_type=args.workflow)
        if verbose:
            print(f"\nLoaded {len(conditions)} conditions:")
            for cond in conditions:
                print(f"  - {cond['label']}: {len(cond['time'])} time points")
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

    # Create fitter
    fitter = RobustFitter(
        model_type=args.workflow,
        weighting=args.weighting,
        use_two_stage=use_two_stage,
        bootstrap_iterations=args.bootstrap,
        bootstrap_workers=args.workers,
        random_seed=args.seed
    )

    # Run fitting
    try:
        result = fitter.fit(conditions, config, verbose=verbose)
    except Exception as e:
        print(f"Error during fitting: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Print summary
    if verbose:
        print("\n" + "=" * 60)
        print(result.summary())
        print("=" * 60)

    # Save results
    config_name = Path(args.config).stem
    save_results(result, args.output, config_name)

    if verbose:
        print("\nFitting complete!")


if __name__ == "__main__":
    main()
