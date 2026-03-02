#!/usr/bin/env python
"""
Command-line interface for individual condition fitting.

This script fits kinetic parameters separately for each experimental condition,
providing detailed statistics, confidence intervals, and diagnostic plots.

Usage:
    python scripts/fit_individual.py --config config/substrates/glucose.json \\
        --data data/example/Experimental_data_glucose.csv --model single_monod

    python scripts/fit_individual.py -c config/substrates/xylose.json \\
        -d data/example/Experimental_data_xylose.csv -m dual_monod_lag -v
"""

import argparse
import math
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.io.data_loader import load_experimental_data
from src.io.config_loader import load_config
from workflows.individual_condition import IndividualConditionWorkflow
from monitoring.logger import setup_logger, get_logger
from monitoring.performance import PerformanceMonitor
from monitoring.version_info import print_version_info


MODEL_CHOICES = [
    'single_monod', 'single_haldane',
    'single_monod_lag', 'single_haldane_lag',
    'dual_monod', 'dual_haldane',
    'dual_monod_lag', 'dual_haldane_lag',
]
CI_METHOD_CHOICES = ['hessian', 'hessian_log', 'mcmc']

# Maps each CLI model choice to (base_model_type, no_inhibition).
# no_inhibition=True means basic Monod (no Ki); False means Haldane (with Ki).
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


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Fit kinetic parameters individually for each condition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Fit glucose data with single Monod model:
    python scripts/fit_individual.py -c config/substrates/glucose.json \\
        -d data/example/Experimental_data_glucose.csv -m single_monod

  Fit xylose data with dual Monod + lag model:
    python scripts/fit_individual.py -c config/substrates/xylose.json \\
        -d data/example/Experimental_data_xylose.csv -m dual_monod_lag -v

Available models:
  single_monod       - Single Monod: basic Monod (4 params: qmax, Ks, Y, b_decay)
  single_haldane     - Single Monod (Haldane): + substrate inhibition (5 params: + Ki)
  single_monod_lag   - Single Monod + Lag: + lag phase (5 params: + lag_time)
  single_haldane_lag - Single Monod + Lag (Haldane): + substrate inhibition (6 params: + Ki)
  dual_monod         - Dual Monod: + oxygen dynamics (6 params: + K_o2, Y_o2)
  dual_haldane       - Dual Monod (Haldane): + substrate inhibition (7 params: + Ki)
  dual_monod_lag     - Dual Monod + Lag: + lag phase (7 params: + lag_time)
  dual_haldane_lag   - Dual Monod + Lag (Haldane): + substrate inhibition (8 params: + Ki)

Output:
  - individual_fits.png: Fitted curves for each condition
  - parameter_comparison.png: Parameter values across conditions
  - residual_diagnostics.png: Residual analysis plots
  - confidence_intervals.png: Parameter CIs by condition
  - goodness_of_fit.png: R² and NRMSE summary
  - individual_condition_results.csv: Tabular results
  - individual_condition_results.json: Detailed results with CIs
        """
    )

    parser.add_argument(
        "-c", "--config",
        required=True,
        help="Path to substrate configuration JSON file"
    )

    parser.add_argument(
        "-d", "--data",
        required=True,
        help="Path to experimental data CSV file"
    )

    parser.add_argument(
        "-m", "--model",
        choices=MODEL_CHOICES,
        default="single_monod",
        help="Model type to use (default: single_monod)"
    )

    parser.add_argument(
        "-o", "--output",
        default="results",
        help="Output directory for results (default: results)"
    )

    parser.add_argument(
        "--optimizer",
        choices=["L-BFGS-B", "differential_evolution"],
        default="L-BFGS-B",
        help="Optimization algorithm (default: L-BFGS-B)"
    )

    parser.add_argument(
        "--ci-method",
        choices=CI_METHOD_CHOICES,
        default="hessian",
        help="Confidence interval method: hessian, hessian_log, or mcmc (default: hessian)"
    )

    parser.add_argument(
        "--ci-level",
        type=float,
        default=0.95,
        help="Confidence level for intervals (default: 0.95)"
    )

    parser.add_argument(
        "--mcmc-samples",
        type=int,
        default=4000,
        help="MCMC posterior samples (used only if --ci-method mcmc)"
    )

    parser.add_argument(
        "--mcmc-burn-in",
        type=int,
        default=1000,
        help="MCMC burn-in iterations (used only if --ci-method mcmc)"
    )

    parser.add_argument(
        "--mcmc-step-scale",
        type=float,
        default=0.05,
        help="MCMC proposal step scale as fraction of parameter range (default: 0.05)"
    )

    parser.add_argument(
        "--mcmc-seed",
        type=int,
        default=None,
        help="Random seed for MCMC reproducibility"
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers for per-condition fitting "
             "(default: 1 = sequential). Use 0 for CPU count - 1."
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output"
    )

    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip plot generation"
    )

    # --- Global optimisation strategy ---
    parser.add_argument(
        "--global-guess",
        choices=["median", "best_r2"],
        default="median",
        help="Strategy for the global optimisation initial guess. "
             "'median' uses the median of per-condition fits (robust to outliers). "
             "'best_r2' uses the parameters from the condition with the highest "
             "mean R²(S,X). (default: median)"
    )

    # --- Scaling options (defaults OFF) ---
    parser.add_argument(
        "--normalize-objective",
        action="store_true",
        default=False,
        help="Weight S and X residuals by 1/range so both contribute equally "
             "(default: off)"
    )

    parser.add_argument(
        "--mcmc-adaptive",
        action="store_true",
        default=False,
        help="Use Hessian-informed adaptive MCMC proposals "
             "(only affects --ci-method mcmc, default: off)"
    )

    parser.add_argument(
        "--scale-diagnostics",
        action="store_true",
        default=False,
        help="Run diagnostic re-optimisations in [0,1]-normalised parameter space "
             "(default: off)"
    )

    parser.add_argument(
        "--version",
        action="store_true",
        help="Print version information and exit"
    )

    return parser.parse_args()


def main():
    """Main entry point for individual condition fitting."""
    args = parse_args()

    if args.version:
        print_version_info()
        return 0

    # Setup logging
    setup_logger(verbose=args.verbose)
    logger = get_logger(__name__)

    # Start performance monitoring
    monitor = PerformanceMonitor()
    monitor.start()

    try:
        logger.info(f"Loading configuration from: {args.config}")
        config = load_config(args.config)

        logger.info(f"Loading experimental data from: {args.data}")
        experimental_data = load_experimental_data(args.data, substrate_name=config.name)

        # Decompose CLI model choice into (base_model_type, no_inhibition)
        base_model_type, no_inhibition = MODEL_MAPPING[args.model]

        # Resolve worker count
        n_workers = args.workers
        if n_workers == 0:
            n_workers = max(1, os.cpu_count() - 1)
        n_workers = max(1, n_workers)

        logger.info(f"Running individual condition fitting with {args.model} model...")
        logger.info(f"Confidence interval method: {args.ci_method}")
        if n_workers > 1:
            logger.info(f"Parallel workers: {n_workers}")

        # Create and run workflow
        workflow = IndividualConditionWorkflow(
            config=config,
            experimental_data=experimental_data,
            model_type=base_model_type,
            output_dir=args.output,
            ci_method=args.ci_method,
            ci_confidence_level=args.ci_level,
            ci_mcmc_samples=args.mcmc_samples,
            ci_mcmc_burn_in=args.mcmc_burn_in,
            ci_mcmc_step_scale=args.mcmc_step_scale,
            ci_mcmc_seed=args.mcmc_seed,
            config_path=args.config,
            data_path=args.data,
            no_inhibition=no_inhibition,
            n_workers=n_workers,
            mcmc_adaptive=args.mcmc_adaptive,
            normalize_objective=args.normalize_objective,
            global_guess_strategy=args.global_guess,
        )

        result = workflow.run(
            optimization_method=args.optimizer,
            save_results=True,
            generate_plots=not args.no_plots,
            verbose=args.verbose
        )

        # Stop monitoring and get stats
        monitor.stop()
        perf_stats = monitor.get_summary()

        # Print summary
        print("\n" + "=" * 70)
        print("INDIVIDUAL CONDITION FITTING COMPLETE")
        print("=" * 70)
        print("Run Settings:")
        print(f"  Model: {workflow.display_name}")
        print(f"  Optimizer: {args.optimizer}")
        print(f"  CI method: {args.ci_method}")
        print(f"  Global guess: {args.global_guess}")
        print(f"  Workers: {n_workers}"
              f" {'(parallel)' if n_workers > 1 else '(sequential)'}")
        print(f"  CI level: {int(args.ci_level * 100)}%")
        if args.ci_method == 'mcmc':
            print(f"  MCMC samples: {args.mcmc_samples}")
            print(f"  MCMC burn-in: {args.mcmc_burn_in}")
            print(f"  MCMC step scale: {args.mcmc_step_scale}")
            print(f"  MCMC seed: {args.mcmc_seed}")
            if args.mcmc_adaptive:
                print(f"  MCMC adaptive proposals: ON")
        if args.normalize_objective:
            print(f"  Objective normalisation: ON")
        if args.scale_diagnostics:
            print(f"  Scaled diagnostics: ON")
        
        # Print abbreviated summary
        print(f"\nModel: {result.display_name or workflow.display_name}")
        print(f"Substrate: {config.name}")
        print(f"\nConditions fitted: {len(result.condition_results)}")
        
        print("\nFit Quality Summary (Individual Conditions):")
        print("-" * 50)
        for cond, cond_result in result.condition_results.items():
            r2_sub = cond_result.statistics['substrate']['R_squared']
            r2_bio = cond_result.statistics['biomass']['R_squared']
            sse_sub = cond_result.statistics['substrate']['SSE']
            sse_bio = cond_result.statistics['biomass']['SSE']
            loss = result.individual_losses.get(cond, float('nan'))
            status = "+" if cond_result.success else "x"
            print(
                f"  {status} {cond}: R2(S)={r2_sub:.4f}, R2(X)={r2_bio:.4f}, "
                f"SSE(S)={sse_sub:.2f}, SSE(X)={sse_bio:.4f}, Loss={loss:.4f}"
            )

        # Individual condition parameters with 95% CIs
        params = list(result.parameter_summary.keys())
        ci_label = "95% CI" if args.ci_method != 'mcmc' else "95% CrI"
        print(f"\nIndividual Condition Parameters (+/- Std Error, [{ci_label}]):")
        print("-" * 70)
        for cond, cond_result in result.condition_results.items():
            print(f"\n  {cond}:")
            for param in params:
                val = cond_result.parameters[param]
                ci = cond_result.confidence_intervals.get(param, {})
                se = ci.get('std_error', float('nan'))
                ci_lo = ci.get('ci_lower', float('nan'))
                ci_hi = ci.get('ci_upper', float('nan'))
                if not math.isnan(se):
                    print(f"    {param:<12s}: {val:>12.6f} +/- {se:<10.4f}  "
                          f"[{ci_lo:.4f}, {ci_hi:.4f}]")
                else:
                    print(f"    {param:<12s}: {val:>12.6f}  (CI unavailable)")

        print("\nGlobal Parameter Estimation Strategy:")
        print("-" * 50)
        print("  1. Each condition fitted independently (local loss)")
        print("  2. Median of individual fits -> initial guess")
        print("  3. Global cost J = sum(L_i) minimized via L-BFGS-B")
        if args.ci_method == 'hessian':
            print("  4. 95% CIs from Hessian of J at global optimum")
        elif args.ci_method == 'hessian_log':
            print("  4. 95% CIs from Hessian in mixed log/linear parameter space")
        else:
            print("  4. 95% CrIs from Metropolis MCMC posterior sampling")

        print(f"\nRecommended Global Parameters (+/- Std Error, [{ci_label}]):")
        print("-" * 70)
        for param, value in result.global_parameters.items():
            summary = result.parameter_summary[param]
            if result.global_confidence_intervals:
                ci = result.global_confidence_intervals.get(param, {})
                se = ci.get('std_error', float('nan'))
                ci_lo = ci.get('ci_lower', float('nan'))
                ci_hi = ci.get('ci_upper', float('nan'))
                if not math.isnan(se):
                    print(f"  {param:<12s}: {value:>12.6f} +/- {se:<10.4f}  "
                          f"[{ci_lo:.4f}, {ci_hi:.4f}]  "
                          f"(CV: {summary['cv']:.1f}%)")
                else:
                    print(f"  {param:<12s}: {value:>12.6f}  "
                          f"(CI unavailable)  (CV: {summary['cv']:.1f}%)")
            else:
                print(f"  {param}: {value:.6f} (CV across individuals: {summary['cv']:.1f}%)")

        if result.global_loss is not None:
            print(f"\n  Global Cost Function Value: {result.global_loss:.6f}")
        if args.ci_method == 'mcmc' and result.global_confidence_intervals:
            any_param = next(iter(result.global_confidence_intervals.values()))
            acc = any_param.get('acceptance_rate', float('nan'))
            if not math.isnan(acc):
                print(f"  MCMC acceptance rate: {acc:.3f}")

            # Summarize convergence diagnostics
            rhat_values = [
                ci.get('r_hat', float('nan'))
                for ci in result.global_confidence_intervals.values()
                if not math.isnan(ci.get('r_hat', float('nan')))
            ]
            ess_values = [
                ci.get('effective_sample_size', float('nan'))
                for ci in result.global_confidence_intervals.values()
                if not math.isnan(ci.get('effective_sample_size', float('nan')))
            ]
            if rhat_values:
                print(f"  R-hat (max): {max(rhat_values):.4f}")
                print(f"  R-hat (median): {sorted(rhat_values)[len(rhat_values)//2]:.4f}")
            if ess_values:
                print(f"  ESS (min): {min(ess_values):.1f}")
                print(f"  ESS (median): {sorted(ess_values)[len(ess_values)//2]:.1f}")
        if result.global_optimization_result is not None:
            print(f"  Optimization Converged: {result.global_optimization_result.success}")
            print(f"  Function Evaluations: {result.global_optimization_result.n_function_evals}")

        print(f"\nPerformance:")
        print(f"  CPU Time: {perf_stats['cpu_time']:.2f} seconds")
        print(f"  Peak Memory: {perf_stats['peak_memory_mb']:.1f} MB")
        print(f"\nResults saved to: {workflow.results_writer.output_dir}")

        return 0

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except Exception as e:
        logger.error(f"Error during fitting: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
