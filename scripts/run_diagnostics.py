#!/usr/bin/env python
"""
Command-line interface for optimizer diagnostics.

Runs multi-start analysis, parameter profiles, 2-D contours,
Hessian eigenvalue analysis, and convergence tracing to investigate
how the optimizer behaves in the parameter landscape.

Usage:
    python scripts/run_diagnostics.py --config config/substrates/glucose.json \\
        --data data/example/Experimental_data_glucose.csv --model single_haldane

    python scripts/run_diagnostics.py -c config/substrates/xylose.json \\
        -d data/example/Experimental_data_xylose.csv -m dual_monod_lag \\
        --n-starts 100 --profile-points 40 -v
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.io.data_loader import load_experimental_data
from src.io.config_loader import load_config
from src.fitting.objective import ObjectiveFunction
from src.fitting.diagnostics import OptimizerDiagnostics
from src.core.oxygen import OxygenModel


MODEL_CHOICES = [
    'single_monod', 'single_haldane',
    'single_monod_lag', 'single_haldane_lag',
    'dual_monod', 'dual_haldane',
    'dual_monod_lag', 'dual_haldane_lag',
]

# Maps each CLI model choice to (base_model_type, no_inhibition).
MODEL_MAPPING = {
    'single_monod':       ('single_monod', True),
    'single_haldane':     ('single_monod', False),
    'single_monod_lag':   ('single_monod_lag', True),
    'single_haldane_lag': ('single_monod_lag', False),
    'dual_monod':         ('dual_monod', True),
    'dual_haldane':       ('dual_monod', False),
    'dual_monod_lag':     ('dual_monod_lag', True),
    'dual_haldane_lag':   ('dual_monod_lag', False),
}

# Parameter sets for each base model type
PARAMETER_SETS = {
    'single_monod': {
        True:  ['qmax', 'Ks', 'Y', 'b_decay'],              # Monod
        False: ['qmax', 'Ks', 'Ki', 'Y', 'b_decay'],        # Haldane
    },
    'single_monod_lag': {
        True:  ['qmax', 'Ks', 'Y', 'b_decay', 'lag_time'],             # Monod + Lag
        False: ['qmax', 'Ks', 'Ki', 'Y', 'b_decay', 'lag_time'],       # Haldane + Lag
    },
    'dual_monod': {
        True:  ['qmax', 'Ks', 'Y', 'b_decay', 'K_o2', 'Y_o2'],
        False: ['qmax', 'Ks', 'Ki', 'Y', 'b_decay', 'K_o2', 'Y_o2'],
    },
    'dual_monod_lag': {
        True:  ['qmax', 'Ks', 'Y', 'b_decay', 'K_o2', 'Y_o2', 'lag_time'],
        False: ['qmax', 'Ks', 'Ki', 'Y', 'b_decay', 'K_o2', 'Y_o2', 'lag_time'],
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run optimizer diagnostics on a kinetic model fit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Diagnostics performed:
  1. Multi-start optimization (sensitivity to initial guesses)
  2. 1-D parameter profiles (identifiability)
  3. 2-D contour surfaces (parameter correlations)
  4. Hessian eigenvalue analysis (conditioning)
  5. Convergence trace (optimization trajectory)

Output:
  diagnostics_summary.json   - Complete results in JSON
  diag_multi_start.png       - Multi-start objective histogram
  diag_profiles.png          - 1-D profile likelihoods
  diag_contours.png          - 2-D objective contours
  diag_hessian.png           - Eigenvalue spectrum & correlation matrix
  diag_convergence.png       - Optimization trajectory
        """
    )

    parser.add_argument(
        "-c", "--config", required=True,
        help="Path to substrate configuration JSON file"
    )
    parser.add_argument(
        "-d", "--data", required=True,
        help="Path to experimental data CSV file"
    )
    parser.add_argument(
        "-m", "--model", choices=MODEL_CHOICES, default="single_haldane",
        help="Model type (default: single_haldane)"
    )
    parser.add_argument(
        "-o", "--output", default=None,
        help="Output directory (default: results/<Substrate>/diagnostics)"
    )
    parser.add_argument(
        "--condition", default=None,
        help="Specific condition to analyse (default: first condition)"
    )
    parser.add_argument(
        "--n-starts", type=int, default=50,
        help="Number of multi-start trials (default: 50)"
    )
    parser.add_argument(
        "--profile-points", type=int, default=30,
        help="Grid points per parameter profile (default: 30)"
    )
    parser.add_argument(
        "--contour-grid", type=int, default=25,
        help="Grid points per contour axis (default: 25)"
    )
    parser.add_argument(
        "--skip-multi-start", action="store_true",
        help="Skip multi-start analysis (saves time)"
    )
    parser.add_argument(
        "--skip-profiles", action="store_true",
        help="Skip 1-D parameter profiles"
    )
    parser.add_argument(
        "--skip-contours", action="store_true",
        help="Skip 2-D contour analysis"
    )
    parser.add_argument(
        "--skip-hessian", action="store_true",
        help="Skip Hessian analysis"
    )
    parser.add_argument(
        "--skip-convergence", action="store_true",
        help="Skip convergence trace"
    )
    parser.add_argument(
        "--profile-params", nargs="+", default=None,
        help="Only profile these parameters (default: all)"
    )
    parser.add_argument(
        "--contour-pairs", nargs="+", default=None,
        help="Parameter pairs for contours, e.g. 'qmax,Ks qmax,b_decay'"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--scale-params", action="store_true", default=False,
        help="Run diagnostic re-optimisations in [0,1]-normalised parameter "
             "space for better conditioning (default: off)"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Load config and data
    print(f"Loading configuration: {args.config}")
    config = load_config(args.config)

    print(f"Loading experimental data: {args.data}")
    experimental_data = load_experimental_data(args.data, substrate_name=config.name)

    # Resolve model
    base_model_type, no_inhibition = MODEL_MAPPING[args.model]
    parameter_names = PARAMETER_SETS[base_model_type][no_inhibition]

    # Pick condition
    conditions = experimental_data.conditions
    if args.condition:
        if args.condition not in conditions:
            print(f"Error: condition '{args.condition}' not found. "
                  f"Available: {conditions}")
            return 1
        condition = args.condition
    else:
        condition = conditions[0]

    print(f"Condition: {condition}")
    print(f"Model: {args.model} ({len(parameter_names)} params: {parameter_names})")

    # Build objective function
    time, substrate, biomass = experimental_data.get_condition_data(condition)
    S0, X0 = substrate[0], biomass[0]

    if base_model_type in ('single_monod', 'single_monod_lag'):
        initial_conditions = [S0, X0]
    else:
        o2_max = config.oxygen.get("o2_max", 8.0)
        initial_conditions = [S0, X0, o2_max]

    oxygen_model = None
    if base_model_type in ('dual_monod', 'dual_monod_lag'):
        oxygen_model = OxygenModel(
            o2_max=config.oxygen.get("o2_max", 8.0),
            o2_min=config.oxygen.get("o2_min", 0.1),
            reaeration_rate=config.oxygen.get("reaeration_rate", 15.0),
            o2_range=config.oxygen.get("o2_range", 8.0),
        )

    objective = ObjectiveFunction(
        experimental_time=time,
        experimental_substrate=substrate,
        experimental_biomass=biomass,
        model_type=base_model_type,
        initial_conditions=initial_conditions,
        t_span=(time[0], time[-1]),
        parameter_names=parameter_names,
        oxygen_model=oxygen_model,
        normalize_errors=True,
    )

    # Bounds and initial guesses
    bounds = {p: config.bounds[p] for p in parameter_names}
    initial_guesses = {p: config.initial_guesses[p] for p in parameter_names}

    # Output directory
    if args.output:
        output_dir = args.output
    else:
        output_dir = os.path.join("results", config.name, "diagnostics")

    # ------------------------------------------------------------------
    # First do a quick fit to get best_params
    # ------------------------------------------------------------------
    from scipy.optimize import minimize as sp_minimize
    import numpy as np

    print(f"\nRunning initial fit to find best parameters...")
    x0 = np.array([initial_guesses[p] for p in parameter_names])
    bounds_list = [bounds[p] for p in parameter_names]

    res = sp_minimize(
        objective, x0,
        method='L-BFGS-B',
        bounds=bounds_list,
        options={'maxiter': 10000, 'ftol': 1e-12},
    )
    best_params = dict(zip(parameter_names, res.x))
    print(f"  Best objective: {res.fun:.6e}")
    print(f"  Best params: {best_params}")

    # ------------------------------------------------------------------
    # Run diagnostics
    # ------------------------------------------------------------------
    diag = OptimizerDiagnostics(
        objective_fn=objective,
        parameter_names=parameter_names,
        bounds=bounds,
        verbose=args.verbose or True,  # Always verbose for CLI
        use_scaled_optimize=args.scale_params,
    )

    from src.fitting.diagnostics import DiagnosticsReport
    report = DiagnosticsReport()

    if not args.skip_multi_start:
        report.multi_start = diag.multi_start(
            best_params, n_starts=args.n_starts
        )

    if not args.skip_profiles:
        report.profiles = diag.parameter_profiles(
            best_params,
            n_points=args.profile_points,
            parameters=args.profile_params,
        )

    if not args.skip_contours:
        contour_pairs = None
        if args.contour_pairs:
            contour_pairs = [tuple(p.split(',')) for p in args.contour_pairs]
        report.contours = diag.contour_analysis(
            best_params,
            param_pairs=contour_pairs,
            n_grid=args.contour_grid,
        )

    if not args.skip_hessian:
        report.hessian = diag.hessian_analysis(best_params)

    if not args.skip_convergence:
        report.convergence = diag.trace_convergence(initial_guesses)

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  SAVING RESULTS")
    print(f"{'='*60}")

    saved = report.save(output_dir)
    plot_files = diag.save_all_plots(report, best_params, output_dir)
    saved.extend(plot_files)

    print(f"\nSaved {len(saved)} files to {output_dir}:")
    for f in saved:
        print(f"  {f}")

    # ------------------------------------------------------------------
    # Print interpretation summary
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  INTERPRETATION SUMMARY")
    print(f"{'='*60}")

    if report.multi_start is not None:
        ms = report.multi_start
        if ms.n_unique_minima == 1:
            print(f"\n  ✓ Multi-start: Single minimum found ({ms.n_unique_minima} cluster)")
            print(f"    The landscape is well-behaved — starting point doesn't matter.")
        else:
            print(f"\n  ⚠ Multi-start: {ms.n_unique_minima} distinct minima found!")
            print(f"    The objective landscape has multiple basins.")
            print(f"    Use differential_evolution or run from multiple starts.")

    if report.profiles is not None:
        poorly_id = [p for p, r in report.profiles.items() if not r.is_identifiable]
        well_id = [p for p, r in report.profiles.items() if r.is_identifiable]
        if well_id:
            print(f"\n  ✓ Well-identified parameters: {', '.join(well_id)}")
        if poorly_id:
            print(f"\n  ⚠ Poorly identified parameters: {', '.join(poorly_id)}")
            print(f"    These parameters have flat profiles — the data cannot")
            print(f"    constrain their values. Consider fixing them or using")
            print(f"    informative priors.")

    if report.hessian is not None:
        h = report.hessian
        if h.condition_number > 1e6:
            print(f"\n  ⚠ Hessian condition number: {h.condition_number:.1e} (ILL-CONDITIONED)")
            print(f"    The optimizer may struggle with numerical precision.")
        elif h.condition_number > 1e4:
            print(f"\n  △ Hessian condition number: {h.condition_number:.1e} (moderate)")
        else:
            print(f"\n  ✓ Hessian condition number: {h.condition_number:.1e} (well-conditioned)")

        # Flag high correlations
        high_corr = []
        for i in range(len(parameter_names)):
            for j in range(i + 1, len(parameter_names)):
                rho = h.correlation_matrix[i, j]
                if abs(rho) > 0.9:
                    high_corr.append((parameter_names[i], parameter_names[j], rho))
        if high_corr:
            print(f"\n  ⚠ Highly correlated parameter pairs:")
            for pi, pj, rho in high_corr:
                print(f"    {pi} ↔ {pj}: ρ = {rho:.3f}")

        if h.sloppy_directions:
            print(f"\n  ⚠ {len(h.sloppy_directions)} sloppy direction(s) in parameter space")

    print(f"\n{'='*60}")
    print(f"  DIAGNOSTICS COMPLETE")
    print(f"{'='*60}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
