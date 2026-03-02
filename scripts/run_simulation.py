#!/usr/bin/env python
"""
Command-line interface for running simulations with fitted parameters.

Usage:
    python scripts/run_simulation.py --params results/xylose/fitted_parameters.json \\
        --config config/substrates/xylose.json --output results/xylose/simulation

    python scripts/run_simulation.py -p fitted_params.json -c config.json \\
        --conditions 5,10,15,20 --t-final 10
"""

import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.ode_systems import SingleMonodODE, DualMonodODE, DualMonodLagODE
from src.core.solvers import solve_ode, create_time_grid
from src.core.oxygen import OxygenModel
from src.io.config_loader import load_config
from src.io.results_writer import ResultsWriter, load_fitted_parameters
from src.utils.plotting import setup_figure, style_axis, save_figure
from monitoring.logger import setup_logger, get_logger


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run simulations with fitted kinetic parameters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Run simulation with previously fitted parameters:
    python scripts/run_simulation.py -p results/xylose/fitted_parameters.json \\
        -c config/substrates/xylose.json

  Run simulation with custom conditions:
    python scripts/run_simulation.py -p fitted_params.json -c config.json \\
        --conditions 5,10,15,20 --t-final 10 --initial-biomass 1.0
        """
    )

    parser.add_argument(
        "-p", "--params",
        required=True,
        help="Path to fitted parameters JSON file"
    )

    parser.add_argument(
        "-c", "--config",
        required=True,
        help="Path to substrate configuration JSON file"
    )

    parser.add_argument(
        "-o", "--output",
        default="results/simulation",
        help="Output directory for simulation results"
    )

    parser.add_argument(
        "--conditions",
        type=str,
        help="Comma-separated list of substrate concentrations in mM (e.g., '5,10,15,20')"
    )

    parser.add_argument(
        "--t-final",
        type=float,
        default=0.5,
        help="Final simulation time in days (default: 0.5)"
    )

    parser.add_argument(
        "--num-points",
        type=int,
        default=10000,
        help="Number of time points (default: 10000)"
    )

    parser.add_argument(
        "--initial-biomass",
        type=float,
        default=1.0,
        help="Initial biomass concentration in mg/L (default: 1.0)"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output"
    )

    return parser.parse_args()


def main():
    """Main entry point for simulation."""
    args = parse_args()

    setup_logger(verbose=args.verbose)
    logger = get_logger(__name__)

    try:
        # Load configuration and parameters
        logger.info(f"Loading configuration from: {args.config}")
        config = load_config(args.config)

        logger.info(f"Loading fitted parameters from: {args.params}")
        fitted = load_fitted_parameters(args.params)

        # Determine conditions
        if args.conditions:
            conditions_mM = [float(c.strip()) for c in args.conditions.split(",")]
        else:
            conditions_mM = [5, 10, 15, 20]  # Default conditions

        logger.info(f"Simulating conditions: {conditions_mM} mM")

        # Create oxygen model
        oxygen_model = OxygenModel(
            o2_max=config.oxygen.get("o2_max", 8.0),
            o2_min=config.oxygen.get("o2_min", 0.1),
            reaeration_rate=config.oxygen.get("reaeration_rate", 15.0),
            o2_range=config.oxygen.get("o2_range", 8.0)
        )

        # Create ODE system based on model type
        model_type = fitted.model_type
        params = fitted.parameters

        if model_type == "single_monod":
            ode_system = SingleMonodODE(
                qmax=params["qmax"],
                Ks=params["Ks"],
                Ki=params["Ki"],
                Y=params["Y"],
                b_decay=params["b_decay"]
            )
        elif model_type == "dual_monod":
            ode_system = DualMonodODE(
                qmax=params["qmax"],
                Ks=params["Ks"],
                Ki=params["Ki"],
                Y=params["Y"],
                b_decay=params["b_decay"],
                K_o2=params["K_o2"],
                Y_o2=params["Y_o2"],
                oxygen_model=oxygen_model
            )
        else:  # dual_monod_lag
            ode_system = DualMonodLagODE(
                qmax=params["qmax"],
                Ks=params["Ks"],
                Ki=params["Ki"],
                Y=params["Y"],
                b_decay=params["b_decay"],
                K_o2=params["K_o2"],
                Y_o2=params["Y_o2"],
                lag_time=params["lag_time"],
                oxygen_model=oxygen_model
            )

        # Run simulations
        t_eval = create_time_grid(0, args.t_final, args.num_points)
        all_results = []

        for conc_mM in conditions_mM:
            S0 = conc_mM * config.molecular_weight  # Convert mM to mg/L
            X0 = args.initial_biomass

            if model_type == "single_monod":
                initial_conditions = [S0, X0]
            else:
                initial_conditions = [S0, X0, oxygen_model.o2_max]

            result = solve_ode(
                ode_system=ode_system,
                initial_conditions=initial_conditions,
                t_span=(0, args.t_final),
                t_eval=t_eval
            )

            df = result.to_dataframe()
            df['Condition'] = f"{conc_mM}mM"
            df['Condition_mM'] = conc_mM
            all_results.append(df)

        # Combine results
        combined = pd.concat(all_results, ignore_index=True)

        # Save results
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / "simulation_results.csv"
        combined.to_csv(output_file, index=False)
        logger.info(f"Saved simulation results to: {output_file}")

        # Generate plot
        import matplotlib.pyplot as plt

        n_conditions = len(conditions_mM)
        fig, axes = plt.subplots(2, n_conditions, figsize=(5*n_conditions, 8), dpi=300)

        if n_conditions == 1:
            axes = axes.reshape(-1, 1)

        for i, conc_mM in enumerate(conditions_mM):
            cond_data = combined[combined['Condition_mM'] == conc_mM]

            # Substrate
            axes[0, i].plot(cond_data['Time'], cond_data['Substrate'], color='#4477AA', linewidth=2)
            axes[0, i].set_xlabel('Time (days)')
            axes[0, i].set_ylabel(f'{config.name} (mg/L)')
            axes[0, i].set_title(f'{conc_mM} mM')
            axes[0, i].grid(True, alpha=0.3)

            # Biomass
            axes[1, i].plot(cond_data['Time'], cond_data['Biomass'], color='#228833', linewidth=2)
            axes[1, i].set_xlabel('Time (days)')
            axes[1, i].set_ylabel('Biomass (mg/L)')
            axes[1, i].grid(True, alpha=0.3)

        plt.suptitle(f'Simulation Results - {model_type}', fontsize=14, fontweight='bold')
        plt.tight_layout()

        plot_file = output_dir / "simulation_plot.png"
        fig.savefig(plot_file, dpi=300, bbox_inches='tight')
        fig.savefig(output_dir / "simulation_plot.pdf", bbox_inches='tight')
        plt.close(fig)

        logger.info(f"Saved plot to: {plot_file}")

        print(f"\nSimulation complete!")
        print(f"Results saved to: {output_dir}")

        return 0

    except Exception as e:
        logger.error(f"Error during simulation: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
