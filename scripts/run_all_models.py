#!/usr/bin/env python
"""
Run all 8 model variants across all substrates and store results in the master CSV.

For each (substrate, model) combination this script:
  1. Runs fit_individual.py via the workflow API (not subprocess)
  2. Appends global parameters, CIs, R², AIC, and Total_Error to the master CSV

Combinations run in parallel across CPU cores using ProcessPoolExecutor.

Usage:
    python scripts/run_all_models.py                    # all substrates, all models
    python scripts/run_all_models.py --substrates glucose xylose
    python scripts/run_all_models.py --models single_monod single_haldane
    python scripts/run_all_models.py --master output/my_results.csv
    python scripts/run_all_models.py --ci-method hessian_log
    python scripts/run_all_models.py --workers 4        # limit parallelism
    python scripts/run_all_models.py --workers 1        # sequential (no parallelism)
"""

import argparse
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.io.config_loader import load_config
from src.io.data_loader import load_experimental_data
from src.utils.master_table import append_to_master_table
from workflows.individual_condition import IndividualConditionWorkflow


# ── Substrate registry ──────────────────────────────────────────────
# Each entry: (config_path, data_path) relative to project root.
SUBSTRATE_REGISTRY = {
    'glucose': (
        'config/substrates/glucose.json',
        'data/example/Experimental_data_glucose.csv',
    ),
    'xylose': (
        'config/substrates/xylose.json',
        'data/example/Experimental_data_xylose.csv',
    ),
    'vanillic_acid': (
        'config/substrates/vanillic_acid.json',
        'data/example/VanillicAcid_experimental_data.csv',
    ),
    'p_coumaric_acid': (
        'config/substrates/p_coumaric_acid.json',
        'data/example/Experimental_data_pCoumaricAcid.csv',
    ),
    'p_hydroxybenzoic_acid': (
        'config/substrates/p_hydroxybenzoic_acid.json',
        'data/example/pHydroxybenzoicAcid_experimental_data.csv',
    ),
    'syringic_acid': (
        'config/substrates/syringic_acid.json',
        'data/example/SyringicAcid_experimental_data.csv',
    ),
}

# ── Model registry ──────────────────────────────────────────────────
# Maps CLI model name to (base_model_type, no_inhibition).
MODEL_REGISTRY = {
    'single_monod':       ('single_monod', True),
    'single_haldane':     ('single_monod', False),
    'single_monod_lag':   ('single_monod_lag', True),
    'single_haldane_lag': ('single_monod_lag', False),
    'dual_monod':         ('dual_monod', True),
    'dual_haldane':       ('dual_monod', False),
    'dual_monod_lag':     ('dual_monod_lag', True),
    'dual_haldane_lag':   ('dual_monod_lag', False),
}

ALL_SUBSTRATES = list(SUBSTRATE_REGISTRY.keys())
ALL_MODELS = list(MODEL_REGISTRY.keys())


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run all model families across all substrates and build master CSV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available substrates: {', '.join(ALL_SUBSTRATES)}
Available models:     {', '.join(ALL_MODELS)}

Examples:
  # Run everything (6 substrates x 6 models = 36 fits)
  python scripts/run_all_models.py

  # Only glucose, two models
  python scripts/run_all_models.py --substrates glucose --models single_monod single_haldane

  # Custom output path
  python scripts/run_all_models.py --master output/comparison.csv

  # Limit to 4 parallel workers
  python scripts/run_all_models.py --workers 4

  # Run sequentially (no parallelism)
  python scripts/run_all_models.py --workers 1
        """,
    )
    parser.add_argument(
        '--substrates', nargs='+', default=ALL_SUBSTRATES,
        choices=ALL_SUBSTRATES, metavar='NAME',
        help=f"Substrates to fit (default: all). Choices: {', '.join(ALL_SUBSTRATES)}",
    )
    parser.add_argument(
        '--models', nargs='+', default=ALL_MODELS,
        choices=ALL_MODELS, metavar='MODEL',
        help=f"Models to fit (default: all). Choices: {', '.join(ALL_MODELS)}",
    )
    parser.add_argument(
        '--master', default='output/master_results.csv',
        help="Path to master CSV (default: output/master_results.csv)",
    )
    parser.add_argument(
        '--output', default='results',
        help="Base output directory for per-run results (default: results)",
    )
    parser.add_argument(
        '--ci-method', default='mcmc',
        choices=['hessian', 'hessian_log', 'mcmc'],
        help="Confidence interval method (default: mcmc)",
    )
    parser.add_argument(
        '--optimizer', default='L-BFGS-B',
        choices=['L-BFGS-B', 'differential_evolution'],
        help="Optimization algorithm (default: L-BFGS-B)",
    )
    parser.add_argument(
        '--global-guess', default='median',
        choices=['median', 'best_r2'],
        help="Strategy for global optimisation initial guess (default: median)",
    )
    parser.add_argument(
        '--normalize-objective', action='store_true',
        help="Weight S and X residuals by 1/range so both contribute equally",
    )
    parser.add_argument(
        '--mcmc-adaptive', action='store_true',
        help="Use Hessian-informed adaptive MCMC proposals",
    )
    parser.add_argument(
        '--scale-diagnostics', action='store_true',
        help="Run diagnostic re-optimisations in [0,1]-normalised parameter space",
    )
    parser.add_argument(
        '--workers', type=int, default=None,
        help="Number of parallel workers (default: CPU count - 1). Use 1 for sequential.",
    )
    parser.add_argument(
        '--no-plots', action='store_true',
        help="Skip plot generation (faster)",
    )
    parser.add_argument(
        '-v', '--verbose', action='store_true',
        help="Enable verbose output for each fit",
    )
    return parser.parse_args()


def _worker_fit(
    substrate_key: str,
    model_key: str,
    output_base: str,
    ci_method: str,
    optimizer: str,
    global_guess_strategy: str,
    normalize_objective: bool,
    mcmc_adaptive: bool,
    scale_diagnostics: bool,
    generate_plots: bool,
    verbose: bool,
) -> dict:
    """
    Worker function executed in a child process.

    Runs one (substrate, model) fit and returns a result dict.
    Does NOT write to the master CSV — that is serialized in the main process.
    """
    # Each worker must set up its own project root path
    root = Path(__file__).parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    config_rel, data_rel = SUBSTRATE_REGISTRY[substrate_key]
    config_path = str(root / config_rel)
    data_path = str(root / data_rel)
    base_model_type, no_inhibition = MODEL_REGISTRY[model_key]

    config = load_config(config_path)
    experimental_data = load_experimental_data(data_path, substrate_name=config.name)

    # Each (substrate, model) gets its own output subdirectory to avoid
    # timestamp collisions when workers start within the same second.
    run_output_dir = str(Path(output_base) / model_key)

    workflow = IndividualConditionWorkflow(
        config=config,
        experimental_data=experimental_data,
        model_type=base_model_type,
        output_dir=run_output_dir,
        ci_method=ci_method,
        config_path=config_path,
        data_path=data_path,
        no_inhibition=no_inhibition,
        global_guess_strategy=global_guess_strategy,
        normalize_objective=normalize_objective,
        mcmc_adaptive=mcmc_adaptive,
    )

    workflow.run(
        optimization_method=optimizer,
        save_results=True,
        generate_plots=generate_plots,
        verbose=verbose,
    )

    return {
        'substrate_key': substrate_key,
        'model_key': model_key,
        'results_dir': str(workflow.results_writer.output_dir),
        'config_path': config_path,
        'data_path': data_path,
        'success': True,
        'error': None,
    }


def main():
    args = parse_args()

    substrates = args.substrates
    models = args.models
    total = len(substrates) * len(models)

    max_workers = args.workers
    if max_workers is None:
        max_workers = max(1, os.cpu_count() - 1)
    max_workers = max(1, min(max_workers, total))

    print("=" * 70)
    print("BATCH MODEL FITTING — ALL SUBSTRATES × ALL MODELS")
    print("=" * 70)
    print(f"  Substrates: {', '.join(substrates)} ({len(substrates)})")
    print(f"  Models:     {', '.join(models)} ({len(models)})")
    print(f"  Total runs: {total}")
    print(f"  Workers:    {max_workers} {'(sequential)' if max_workers == 1 else '(parallel)'}")
    print(f"  Master CSV: {args.master}")
    print(f"  CI method:  {args.ci_method}")
    print(f"  Optimizer:  {args.optimizer}")
    print(f"  Global guess: {args.global_guess}")
    if args.normalize_objective:
        print(f"  Normalize objective: ON")
    if args.mcmc_adaptive:
        print(f"  MCMC adaptive: ON")
    if args.scale_diagnostics:
        print(f"  Scale diagnostics: ON")
    print("=" * 70)

    # Build list of all (substrate, model) jobs
    jobs = [
        (sub, mod)
        for sub in substrates
        for mod in models
    ]

    succeeded = 0
    failed = 0
    failures = []
    completed_results = []
    t_start = time.time()

    if max_workers == 1:
        # Sequential mode — run directly, no subprocesses
        for run_num, (substrate_key, model_key) in enumerate(jobs, 1):
            display_name = IndividualConditionWorkflow.DISPLAY_NAMES.get(
                MODEL_REGISTRY[model_key], model_key
            )
            config_rel, _ = SUBSTRATE_REGISTRY[substrate_key]
            substrate_name = load_config(str(project_root / config_rel)).name

            print(f"\n[{run_num}/{total}] {substrate_name} — {display_name}")
            print("-" * 50)

            try:
                result = _worker_fit(
                    substrate_key=substrate_key,
                    model_key=model_key,
                    output_base=args.output,
                    ci_method=args.ci_method,
                    optimizer=args.optimizer,
                    global_guess_strategy=args.global_guess,
                    normalize_objective=args.normalize_objective,
                    mcmc_adaptive=args.mcmc_adaptive,
                    scale_diagnostics=args.scale_diagnostics,
                    generate_plots=not args.no_plots,
                    verbose=args.verbose,
                )
                completed_results.append(result)
                succeeded += 1
                print(f"  -> OK")
            except Exception as e:
                failed += 1
                failures.append((substrate_key, model_key, str(e)))
                print(f"  -> FAILED: {e}")
                if args.verbose:
                    traceback.print_exc()
    else:
        # Parallel mode
        future_to_job = {}
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for substrate_key, model_key in jobs:
                future = executor.submit(
                    _worker_fit,
                    substrate_key=substrate_key,
                    model_key=model_key,
                    output_base=args.output,
                    ci_method=args.ci_method,
                    optimizer=args.optimizer,
                    global_guess_strategy=args.global_guess,
                    normalize_objective=args.normalize_objective,
                    mcmc_adaptive=args.mcmc_adaptive,
                    scale_diagnostics=args.scale_diagnostics,
                    generate_plots=not args.no_plots,
                    verbose=args.verbose,
                )
                future_to_job[future] = (substrate_key, model_key)

            for i, future in enumerate(as_completed(future_to_job), 1):
                substrate_key, model_key = future_to_job[future]
                display_name = IndividualConditionWorkflow.DISPLAY_NAMES.get(
                    MODEL_REGISTRY[model_key], model_key
                )
                config_rel, _ = SUBSTRATE_REGISTRY[substrate_key]
                substrate_name = load_config(str(project_root / config_rel)).name

                try:
                    result = future.result()
                    completed_results.append(result)
                    succeeded += 1
                    print(f"  [{i}/{total}] {substrate_name} — {display_name} -> OK")
                except Exception as e:
                    failed += 1
                    failures.append((substrate_key, model_key, str(e)))
                    print(f"  [{i}/{total}] {substrate_name} — {display_name} -> FAILED: {e}")
                    if args.verbose:
                        traceback.print_exc()

    # Serialize master CSV writes (safe — single process, sequential)
    print(f"\nWriting {len(completed_results)} results to master CSV...")
    for result in completed_results:
        try:
            append_to_master_table(
                results_dir=result['results_dir'],
                master_csv=args.master,
                config_path=result['config_path'],
                data_path=result['data_path'],
            )
        except Exception as e:
            print(f"  Warning: failed to append {result['substrate_key']}+"
                  f"{result['model_key']} to master CSV: {e}")

    elapsed = time.time() - t_start

    # Final summary
    print("\n" + "=" * 70)
    print("BATCH FITTING COMPLETE")
    print("=" * 70)
    print(f"  Succeeded: {succeeded}/{total}")
    print(f"  Failed:    {failed}/{total}")
    print(f"  Workers:   {max_workers}")
    print(f"  Elapsed:   {elapsed:.1f} seconds ({elapsed/60:.1f} min)")
    print(f"  Master CSV: {args.master}")

    if failures:
        print("\nFailed runs:")
        for sub, mod, err in failures:
            print(f"  {sub} + {mod}: {err}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
