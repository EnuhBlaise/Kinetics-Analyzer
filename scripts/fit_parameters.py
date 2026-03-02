#!/usr/bin/env python
"""
Command-line interface for fitting kinetic parameters to experimental data.

Usage:
    python scripts/fit_parameters.py --config config/substrates/xylose.json \\
        --data data/xylose_data.csv --workflow dual_monod_lag

    python scripts/fit_parameters.py -c config/substrates/glucose.json \\
        -d data/glucose_data.csv -w single_monod --output results/glucose
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.io.data_loader import load_experimental_data
from src.io.config_loader import load_config
from workflows.single_monod import SingleMonodWorkflow
from workflows.single_monod_lag import SingleMonodLagWorkflow
from workflows.dual_monod import DualMonodWorkflow
from workflows.dual_monod_lag import DualMonodLagWorkflow
from monitoring.logger import setup_logger, get_logger
from monitoring.performance import PerformanceMonitor
from monitoring.version_info import print_version_info


WORKFLOW_MAP = {
    "single_monod": SingleMonodWorkflow,
    "single_haldane": SingleMonodWorkflow,       # alias (default includes Ki)
    "single_monod_lag": SingleMonodLagWorkflow,
    "single_haldane_lag": SingleMonodLagWorkflow, # alias
    "dual_monod": DualMonodWorkflow,
    "dual_haldane": DualMonodWorkflow,            # alias
    "dual_monod_lag": DualMonodLagWorkflow,
    "dual_haldane_lag": DualMonodLagWorkflow,     # alias
}


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Fit kinetic parameters to experimental data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Fit xylose data with dual Monod + lag model:
    python scripts/fit_parameters.py -c config/substrates/xylose.json \\
        -d data/xylose_data.csv -w dual_monod_lag

  Fit glucose data with simple Monod model:
    python scripts/fit_parameters.py -c config/substrates/glucose.json \\
        -d data/glucose_data.csv -w single_monod

Available workflows:
  single_monod       - Single Monod (5 params, with Ki / Haldane inhibition)
  single_haldane     - Alias for single_monod
  single_monod_lag   - Single Monod + Lag (6 params, with Ki)
  single_haldane_lag - Alias for single_monod_lag
  dual_monod         - Dual Monod with O2 dynamics (7 params, with Ki)
  dual_haldane       - Alias for dual_monod
  dual_monod_lag     - Dual Monod + Lag phase (8 params, with Ki)
  dual_haldane_lag   - Alias for dual_monod_lag
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
        "-w", "--workflow",
        choices=list(WORKFLOW_MAP.keys()),
        default="dual_monod_lag",
        help="Workflow/model type to use (default: dual_monod_lag)"
    )

    parser.add_argument(
        "-o", "--output",
        default="results",
        help="Output directory for results (default: results)"
    )

    parser.add_argument(
        "-m", "--method",
        choices=["global", "individual"],
        default="global",
        help="Fitting method: global (single parameter set) or individual (per condition)"
    )

    parser.add_argument(
        "--optimizer",
        choices=["L-BFGS-B", "differential_evolution"],
        default="L-BFGS-B",
        help="Optimization algorithm (default: L-BFGS-B)"
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

    parser.add_argument(
        "--version",
        action="store_true",
        help="Print version information and exit"
    )

    return parser.parse_args()


def main():
    """Main entry point for parameter fitting."""
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

        logger.info(f"Running {args.workflow} workflow...")

        # Create and run workflow
        WorkflowClass = WORKFLOW_MAP[args.workflow]
        workflow = WorkflowClass(
            config=config,
            experimental_data=experimental_data,
            output_dir=args.output
        )

        result = workflow.run(
            fit_method=args.method,
            optimization_method=args.optimizer,
            save_results=True,
            generate_plots=not args.no_plots,
            verbose=args.verbose
        )

        # Stop monitoring and get stats
        monitor.stop()
        perf_stats = monitor.get_summary()

        # Print summary
        print("\n" + "=" * 60)
        print("PARAMETER FITTING COMPLETE")
        print("=" * 60)
        print(result.summary())
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
