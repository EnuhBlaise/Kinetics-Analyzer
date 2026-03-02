#!/usr/bin/env python3
"""
Estimate theoretical parameter ceilings from substrate stoichiometry.

Usage
-----
  # Single substrate from config file
  python scripts/estimate_bounds.py --config config/substrates/glucose.json

  # All substrates in the config directory
  python scripts/estimate_bounds.py --all

  # Direct formula input (no config file needed)
  python scripts/estimate_bounds.py --name "Glucose" --formula C6H12O6 --mw 180.16

  # Save results to JSON
  python scripts/estimate_bounds.py --all --output results/theoretical_bounds.json

  # Compare theoretical vs current bounds in config
  python scripts/estimate_bounds.py --config config/substrates/glucose.json --compare
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.theoretical_bounds import (
    compute_bounds_report,
    compute_from_config,
    compare_with_current_bounds,
)


def find_all_configs(config_dir: str = "config/substrates") -> list:
    """Find all substrate config JSON files."""
    config_path = Path(config_dir)
    if not config_path.exists():
        print(f"Config directory not found: {config_dir}")
        return []
    return sorted(config_path.glob("*.json"))


def main():
    parser = argparse.ArgumentParser(
        description="Estimate theoretical parameter ceilings from "
        "substrate stoichiometry.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Input options (mutually exclusive groups)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--config",
        type=str,
        help="Path to a substrate config JSON file.",
    )
    input_group.add_argument(
        "--all",
        action="store_true",
        help="Process all config files in config/substrates/.",
    )
    input_group.add_argument(
        "--formula",
        type=str,
        help="Molecular formula (e.g. C6H12O6). Requires --name and --mw.",
    )

    # Additional options for --formula mode
    parser.add_argument("--name", type=str, help="Substrate name.")
    parser.add_argument(
        "--mw", type=float, help="Molecular weight (g/mol)."
    )

    # Output options
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Save results to a JSON file.",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare theoretical bounds against current config bounds.",
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default="config/substrates",
        help="Directory containing config files (for --all mode).",
    )

    args = parser.parse_args()

    # ── Validate formula-mode arguments ───────────────────────────
    if args.formula:
        if not args.name or not args.mw:
            parser.error("--formula requires --name and --mw.")

    # ── Collect reports ───────────────────────────────────────────
    reports = []

    if args.config:
        report = compute_from_config(args.config)
        reports.append((report, args.config))

    elif args.all:
        configs = find_all_configs(args.config_dir)
        if not configs:
            print("No config files found.")
            sys.exit(1)
        for cfg_path in configs:
            try:
                report = compute_from_config(str(cfg_path))
                reports.append((report, str(cfg_path)))
            except ValueError as e:
                print(f"⚠ Skipping {cfg_path.name}: {e}")

    elif args.formula:
        report = compute_bounds_report(args.name, args.formula, args.mw)
        reports.append((report, None))

    # ── Print results ─────────────────────────────────────────────
    for report, cfg_path in reports:
        print(report.summary_text())

        if args.compare and cfg_path:
            comparison = compare_with_current_bounds(report, cfg_path)
            print(comparison)

        print()

    # ── Save JSON ─────────────────────────────────────────────────
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        all_results = {}
        for report, _ in reports:
            all_results[report.substrate_name] = report.to_dict()

        with open(output_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\n✓ Results saved to {output_path}")

    # ── Summary table (when --all) ────────────────────────────────
    if args.all and len(reports) > 1:
        print(f"\n{'═' * 80}")
        print(f"  SUMMARY TABLE — All Substrates")
        print(f"{'═' * 80}")
        header = (
            f"  {'Substrate':20s}  {'Formula':12s}  {'γ_s':>6s}  "
            f"{'Y_max':>8s}  {'Y_prac':>8s}  {'ThOD':>8s}  "
            f"{'Y_o2':>8s}  {'qmax':>6s}  {'Class':>10s}"
        )
        print(header)
        print(f"  {'─' * 76}")
        for report, _ in reports:
            line = (
                f"  {report.substrate_name:20s}  "
                f"{report.molecular_formula:12s}  "
                f"{report.gamma_s:6.3f}  "
                f"{report.Y_max_mg:8.4f}  "
                f"{report.Y_practical_mg:8.4f}  "
                f"{report.ThOD_mg_mg:8.4f}  "
                f"{report.Y_o2_max:8.4f}  "
                f"{report.qmax_ceiling:6.1f}  "
                f"{report.substrate_class:>10s}"
            )
            print(line)
        print(f"{'═' * 80}")
        print()
        print("  Units: Y [mg cells/mg sub], ThOD [mg O₂/mg sub], "
              "Y_o2 [mg cells/mg O₂], qmax [1/day]")
        print(f"  Biomass formula: CH₁.₈O₀.₅N₀.₂  "
              f"(γ_bio = 4.2, MW = 24.6 g/C-mol)")
        print(f"  Practical ceiling assumes "
              f"ε = 60% of electrons → anabolism")


if __name__ == "__main__":
    main()
