#!/usr/bin/env python
"""
Command-line utility for converting kinetic parameter units.

Converts time-dependent parameters between days, hours, and minutes.
Converts concentrations between mg/L, mM, and g/L.

Usage:
    python scripts/convert_units.py --params fitted_params.json --from days --to hours
    python scripts/convert_units.py --value 2.5 --param qmax --from days --to minutes
    python scripts/convert_units.py --conc 500 --mw 150.13 --from mg/L --to mM
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.conversions import (
    convert_time_units,
    convert_kinetic_parameters,
    convert_concentration_units,
    mM_to_mgL,
    mgL_to_mM,
    MOLECULAR_WEIGHTS
)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert kinetic parameter units",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Convert all parameters in a file from days to hours:
    python scripts/convert_units.py --params fitted_params.json --from days --to hours

  Convert a single qmax value:
    python scripts/convert_units.py --value 2.5 --param qmax --from days --to minutes

  Convert concentration from mg/L to mM:
    python scripts/convert_units.py --conc 750.65 --mw 150.13 --from mg/L --to mM

  Convert concentration using substrate name:
    python scripts/convert_units.py --conc 900 --substrate glucose --from mg/L --to mM
        """
    )

    # File-based conversion
    parser.add_argument(
        "--params",
        help="Path to fitted parameters JSON file for batch conversion"
    )

    # Single value conversion
    parser.add_argument(
        "--value",
        type=float,
        help="Single value to convert"
    )

    parser.add_argument(
        "--param",
        help="Parameter name (for determining if time-dependent)"
    )

    # Concentration conversion
    parser.add_argument(
        "--conc",
        type=float,
        help="Concentration value to convert"
    )

    parser.add_argument(
        "--mw",
        type=float,
        help="Molecular weight in g/mol (for mM conversions)"
    )

    parser.add_argument(
        "--substrate",
        help="Substrate name for automatic MW lookup"
    )

    # Unit specification
    parser.add_argument(
        "--from",
        dest="from_unit",
        required=True,
        help="Source unit (days, hours, minutes, mg/L, mM, g/L)"
    )

    parser.add_argument(
        "--to",
        dest="to_unit",
        required=True,
        help="Target unit"
    )

    parser.add_argument(
        "-o", "--output",
        help="Output file for converted parameters (optional)"
    )

    return parser.parse_args()


def main():
    """Main entry point for unit conversion."""
    args = parse_args()

    try:
        # Concentration conversion
        if args.conc is not None:
            mw = args.mw
            if mw is None and args.substrate:
                mw = MOLECULAR_WEIGHTS.get(args.substrate.lower())
                if mw is None:
                    print(f"Error: Unknown substrate '{args.substrate}'")
                    print(f"Known substrates: {list(MOLECULAR_WEIGHTS.keys())}")
                    return 1

            result = convert_concentration_units(
                args.conc,
                args.from_unit,
                args.to_unit,
                molecular_weight=mw
            )

            print(f"\nConcentration Conversion:")
            print(f"  {args.conc} {args.from_unit} = {result:.6f} {args.to_unit}")
            if mw:
                print(f"  (Molecular weight: {mw} g/mol)")

            return 0

        # Single parameter value conversion
        elif args.value is not None:
            time_dependent = {"qmax", "b_decay"}

            if args.param and args.param.lower() in time_dependent:
                factor = convert_time_units(1.0, args.from_unit, args.to_unit)
                result = args.value / factor
            else:
                result = convert_time_units(args.value, args.from_unit, args.to_unit)

            print(f"\nUnit Conversion:")
            print(f"  {args.value} per {args.from_unit} = {result:.8f} per {args.to_unit}")

            return 0

        # File-based parameter conversion
        elif args.params:
            with open(args.params, 'r') as f:
                data = json.load(f)

            if "parameters" in data:
                params = data["parameters"]
            else:
                params = data

            converted = convert_kinetic_parameters(
                params,
                args.from_unit,
                args.to_unit
            )

            print(f"\nParameter Conversion ({args.from_unit} -> {args.to_unit}):")
            print("-" * 50)
            print(f"{'Parameter':<15} {'Original':<15} {'Converted':<15}")
            print("-" * 50)

            for name in params:
                orig = params[name]
                conv = converted[name]
                if orig != conv:
                    print(f"{name:<15} {orig:<15.6f} {conv:<15.8f}")
                else:
                    print(f"{name:<15} {orig:<15.6f} {conv:<15.6f} (unchanged)")

            # Save if output specified
            if args.output:
                if "parameters" in data:
                    data["parameters"] = converted
                    data["units_note"] = f"Converted from {args.from_unit} to {args.to_unit}"
                else:
                    data = converted

                with open(args.output, 'w') as f:
                    json.dump(data, f, indent=2)

                print(f"\nConverted parameters saved to: {args.output}")

            return 0

        else:
            print("Error: Must specify --params, --value, or --conc")
            return 1

    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
