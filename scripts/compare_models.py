#!/usr/bin/env python
"""
Command-line interface for comparing different kinetic models.

This script runs all three model types (single_monod, dual_monod, dual_monod_lag)
and provides a comparison of their performance.

Usage:
    python scripts/compare_models.py --config config/substrates/xylose.json \\
        --data data/xylose_data.csv --output results/comparison
"""

import argparse
import json
import sys
from pathlib import Path
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.io.data_loader import load_experimental_data
from src.io.config_loader import load_config
from src.fitting.statistics import compare_models
from workflows.single_monod import SingleMonodWorkflow
from workflows.single_monod_lag import SingleMonodLagWorkflow
from workflows.dual_monod import DualMonodWorkflow
from workflows.dual_monod_lag import DualMonodLagWorkflow
from monitoring.logger import setup_logger, get_logger
from monitoring.performance import PerformanceMonitor


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Compare different kinetic models on experimental data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Compare all models on xylose data:
    python scripts/compare_models.py -c config/substrates/xylose.json \\
        -d data/xylose_data.csv

  Compare with specific output directory:
    python scripts/compare_models.py -c config/substrates/glucose.json \\
        -d data/glucose_data.csv -o results/glucose_comparison
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
        "-o", "--output",
        default="results/comparison",
        help="Output directory for comparison results"
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

    return parser.parse_args()


def main():
    """Main entry point for model comparison."""
    args = parse_args()

    setup_logger(verbose=args.verbose)
    logger = get_logger(__name__)

    # Start performance monitoring
    monitor = PerformanceMonitor()
    monitor.start()

    try:
        # Load data
        logger.info(f"Loading configuration from: {args.config}")
        config = load_config(args.config)

        logger.info(f"Loading experimental data from: {args.data}")
        experimental_data = load_experimental_data(args.data, substrate_name=config.name)

        # Define models to compare
        workflows = {
            "Single Monod": SingleMonodWorkflow,
            "Single Monod + Lag": SingleMonodLagWorkflow,
            "Dual Monod": DualMonodWorkflow,
            "Dual Monod + Lag": DualMonodLagWorkflow
        }

        # Run each model
        results = {}
        model_stats = {}

        print("\n" + "=" * 60)
        print("KINETIC MODEL COMPARISON")
        print("=" * 60)
        print(f"Substrate: {config.name}")
        print(f"Conditions: {experimental_data.conditions}")
        print("=" * 60 + "\n")

        for name, WorkflowClass in workflows.items():
            print(f"\nRunning {name}...")

            workflow = WorkflowClass(
                config=config,
                experimental_data=experimental_data,
                output_dir=args.output
            )

            result = workflow.run(
                fit_method="global",
                save_results=True,
                generate_plots=not args.no_plots,
                verbose=False
            )

            results[name] = result
            model_stats[name] = result.statistics

            print(f"  R² = {result.statistics.get('R_squared', 0):.4f}")
            print(f"  AIC = {result.statistics.get('AIC', 0):.2f}")

        # Compare models
        comparison = compare_models(model_stats)

        # Print comparison results
        print("\n" + "=" * 60)
        print("COMPARISON RESULTS")
        print("=" * 60)

        # Create comparison table
        table_data = []
        for name in workflows.keys():
            stats = model_stats[name]
            table_data.append({
                "Model": name,
                "R²": f"{stats.get('R_squared', 0):.4f}",
                "RMSE": f"{stats.get('RMSE', 0):.4f}",
                "AIC": f"{stats.get('AIC', 0):.2f}",
                "BIC": f"{stats.get('BIC', 0):.2f}",
                "Parameters": stats.get('n_parameters', 0)
            })

        comparison_df = pd.DataFrame(table_data)
        print("\n" + comparison_df.to_string(index=False))

        print("\n" + "-" * 60)
        print("Model Rankings:")
        print(f"  By R²:  {' > '.join(comparison['rankings']['by_R_squared'])}")
        print(f"  By AIC: {' > '.join(comparison['rankings']['by_AIC'])}")
        print(f"  By BIC: {' > '.join(comparison['rankings']['by_BIC'])}")

        print("\n" + "-" * 60)
        print("Akaike Weights (probability of being best model):")
        for model, weight in comparison['akaike_weights'].items():
            print(f"  {model}: {weight:.2%}")

        print("\n" + "-" * 60)
        best_model = comparison['best_model']['by_AIC']
        print(f"RECOMMENDED MODEL (by AIC): {best_model}")
        print("-" * 60)

        # Save comparison results
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save comparison table
        comparison_df.to_csv(output_dir / "model_comparison.csv", index=False)

        # Save detailed comparison
        comparison_report = {
            "substrate": config.name,
            "conditions": experimental_data.conditions,
            "models": {
                name: {
                    "parameters": result.optimization_result.parameters,
                    "statistics": result.statistics
                }
                for name, result in results.items()
            },
            "comparison": {
                "rankings": comparison['rankings'],
                "best_model": comparison['best_model'],
                "akaike_weights": comparison['akaike_weights'],
                "delta_AIC": comparison['delta_AIC']
            },
            "recommendation": best_model
        }

        with open(output_dir / "comparison_report.json", 'w') as f:
            json.dump(comparison_report, f, indent=2, default=str)

        # Generate comparison plot
        if not args.no_plots:
            _generate_comparison_plot(results, output_dir)

        # Stop monitoring
        monitor.stop()
        perf_stats = monitor.get_summary()

        print(f"\nPerformance:")
        print(f"  Total CPU Time: {perf_stats['cpu_time']:.2f} seconds")
        print(f"  Peak Memory: {perf_stats['peak_memory_mb']:.1f} MB")

        print(f"\nResults saved to: {output_dir}")

        return 0

    except Exception as e:
        logger.error(f"Error during comparison: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def _generate_comparison_plot(results, output_dir):
    """Generate a comparison plot of all models."""
    import matplotlib.pyplot as plt
    import numpy as np

    model_names = list(results.keys())
    n_models = len(model_names)

    # Create bar plot for statistics
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=300)

    # Colors
    colors = ['#4477AA', '#228833', '#EE6677']

    # R² comparison
    r2_values = [results[m].statistics.get('R_squared', 0) for m in model_names]
    axes[0].bar(model_names, r2_values, color=colors)
    axes[0].set_ylabel('R²')
    axes[0].set_title('Coefficient of Determination')
    axes[0].set_ylim(0, 1.1)
    for i, v in enumerate(r2_values):
        axes[0].text(i, v + 0.02, f'{v:.4f}', ha='center', fontsize=9)

    # RMSE comparison
    rmse_values = [results[m].statistics.get('RMSE', 0) for m in model_names]
    axes[1].bar(model_names, rmse_values, color=colors)
    axes[1].set_ylabel('RMSE')
    axes[1].set_title('Root Mean Square Error')
    for i, v in enumerate(rmse_values):
        axes[1].text(i, v + max(rmse_values)*0.02, f'{v:.2f}', ha='center', fontsize=9)

    # AIC comparison
    aic_values = [results[m].statistics.get('AIC', 0) for m in model_names]
    axes[2].bar(model_names, aic_values, color=colors)
    axes[2].set_ylabel('AIC')
    axes[2].set_title('Akaike Information Criterion')
    for i, v in enumerate(aic_values):
        axes[2].text(i, v + max(aic_values)*0.02, f'{v:.1f}', ha='center', fontsize=9)

    for ax in axes:
        ax.set_xticklabels(model_names, rotation=15, ha='right')
        ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Model Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()

    fig.savefig(output_dir / "model_comparison.png", dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / "model_comparison.pdf", bbox_inches='tight')
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
