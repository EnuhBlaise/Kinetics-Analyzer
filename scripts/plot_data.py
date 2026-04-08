#!/usr/bin/env python
"""
Quick-look utility for plotting raw experimental data from a CSV file.

Produces a two-panel figure (Substrate & Biomass vs Time) coloured by
condition, with no fitting or model evaluation — just the raw data.

Usage:
    python scripts/plot_data.py data/example/Experimental_data_glucose.csv
    python scripts/plot_data.py data/example/pHydroxybenzoicAcid_experimental_data.csv --save
    python scripts/plot_data.py data/example/Experimental_data_xylose.csv --title "Xylose Experiment"
"""

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.plotting import COLORS, COLOR_LIST, setup_figure, style_axis


# ── helpers ──────────────────────────────────────────────────────────

def _detect_columns(df: pd.DataFrame):
    """
    Auto-detect time, substrate, and biomass columns from a CSV DataFrame.

    Returns:
        (time_col, condition_pairs)
        where condition_pairs is a list of (label, substrate_col, biomass_col).
    """
    cols = list(df.columns)

    # Time column: prefer "Time (days)", fall back to "Time (hours)", then first col
    time_col = None
    for candidate in cols:
        if re.search(r"time.*day", candidate, re.IGNORECASE):
            time_col = candidate
            break
    if time_col is None:
        for candidate in cols:
            if re.search(r"time.*hour", candidate, re.IGNORECASE):
                time_col = candidate
                break
    if time_col is None:
        time_col = cols[0]

    # Detect condition pairs: columns like "5mM_Biomass (OD)" / "5mM_Glucose (mg/L)"
    biomass_cols = [c for c in cols if re.search(r"biomass", c, re.IGNORECASE)]
    substrate_cols = [c for c in cols if c not in biomass_cols
                      and c != time_col
                      and not re.search(r"^time|^period$", c, re.IGNORECASE)]

    # Pair them by shared condition prefix (e.g. "5mM")
    condition_pairs = []
    for bcol in biomass_cols:
        prefix = bcol.split("_")[0]  # e.g. "5mM"
        # find matching substrate column
        matching = [s for s in substrate_cols if s.startswith(f"{prefix}_")]
        if matching:
            condition_pairs.append((prefix, matching[0], bcol))

    if not condition_pairs:
        raise RuntimeError(
            "Could not auto-detect condition columns. "
            "Expected columns like '<conc>_Biomass …' and '<conc>_Substrate …'."
        )

    return time_col, condition_pairs


def plot_experimental_data(
    csv_path: str,
    title: str | None = None,
    figsize: tuple[float, float] = (12, 5),
    dpi: int = 200,
    marker_size: float = 40,
    save_path: str | None = None,
    show: bool = True,
) -> plt.Figure:
    """
    Load an experimental CSV and produce a two-panel scatter plot.

    Args:
        csv_path: Path to the experimental data CSV.
        title: Optional super-title (defaults to the file stem).
        figsize: Figure size in inches.
        dpi: Resolution.
        marker_size: Scatter marker size.
        save_path: If given, save the figure to this path.
        show: Whether to call plt.show().

    Returns:
        The matplotlib Figure object.
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    time_col, condition_pairs = _detect_columns(df)
    time = df[time_col].values

    # Infer time unit from column name
    if "day" in time_col.lower():
        time_unit = "days"
    elif "hour" in time_col.lower():
        time_unit = "hours"
    else:
        time_unit = ""

    fig, (ax_sub, ax_bio) = plt.subplots(1, 2, figsize=figsize, dpi=dpi)

    for i, (label, scol, bcol) in enumerate(condition_pairs):
        colour = COLOR_LIST[i % len(COLOR_LIST)]
        substrate = pd.to_numeric(df[scol], errors="coerce").values
        biomass = pd.to_numeric(df[bcol], errors="coerce").values

        # Substrate panel
        mask_s = np.isfinite(substrate)
        ax_sub.scatter(
            time[mask_s], substrate[mask_s],
            s=marker_size, color=colour, edgecolors="white", linewidths=0.5,
            label=label, zorder=3,
        )
        ax_sub.plot(time[mask_s], substrate[mask_s], color=colour, alpha=0.35, linewidth=1)

        # Biomass panel
        mask_b = np.isfinite(biomass)
        ax_bio.scatter(
            time[mask_b], biomass[mask_b],
            s=marker_size, color=colour, edgecolors="white", linewidths=0.5,
            label=label, zorder=3,
        )
        ax_bio.plot(time[mask_b], biomass[mask_b], color=colour, alpha=0.35, linewidth=1)

    # Substrate axis
    substrate_name = _infer_substrate_name(condition_pairs[0][1])
    ax_sub.set_xlabel(f"Time ({time_unit})" if time_unit else "Time", fontsize=11)
    ax_sub.set_ylabel(f"{substrate_name} (mg/L)", fontsize=11)
    ax_sub.set_title("Substrate", fontsize=12, fontweight="bold")
    ax_sub.legend(fontsize=9, frameon=True, fancybox=True)
    ax_sub.grid(True, alpha=0.3)

    # Biomass axis
    ax_bio.set_xlabel(f"Time ({time_unit})" if time_unit else "Time", fontsize=11)
    ax_bio.set_ylabel("Biomass (OD)", fontsize=11)
    ax_bio.set_title("Biomass", fontsize=12, fontweight="bold")
    ax_bio.legend(fontsize=9, frameon=True, fancybox=True)
    ax_bio.grid(True, alpha=0.3)

    sup = title or csv_path.stem.replace("_", " ")
    fig.suptitle(sup, fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=dpi)
        pdf_path = Path(save_path).with_suffix(".pdf")
        if str(pdf_path) != str(save_path):
            fig.savefig(pdf_path, bbox_inches="tight")
            print(f"Saved → {save_path}")
            print(f"Saved → {pdf_path}")
        else:
            print(f"Saved → {save_path}")

    if show:
        plt.show()

    return fig


def _infer_substrate_name(col_name: str) -> str:
    """Extract substrate name from a column like '5mM_Glucose (mg/L)'."""
    # Remove the condition prefix and unit suffix
    parts = col_name.split("_", 1)
    if len(parts) > 1:
        name = parts[1]
        name = re.sub(r"\s*\(.*\)", "", name).strip()
        return name
    return "Substrate"


# ── CLI ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Quick-look plot of raw experimental data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/plot_data.py data/example/Experimental_data_glucose.csv
  python scripts/plot_data.py data/example/pHydroxybenzoicAcid_experimental_data.csv --save
  python scripts/plot_data.py data/example/Experimental_data_xylose.csv --title "Xylose"
        """,
    )
    parser.add_argument("csv", help="Path to experimental data CSV")
    parser.add_argument("--title", default=None, help="Figure title (default: filename)")
    parser.add_argument("--save", action="store_true", help="Save figure as PNG next to CSV")
    parser.add_argument("--out", default=None, help="Explicit save path (overrides --save)")
    parser.add_argument("--no-show", action="store_true", help="Don't call plt.show()")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"Error: file not found: {csv_path}")
        return 1

    save_path = args.out
    if save_path is None and args.save:
        save_path = csv_path.with_suffix(".png")

    plot_experimental_data(
        csv_path=str(csv_path),
        title=args.title,
        save_path=save_path,
        show=not args.no_show,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
