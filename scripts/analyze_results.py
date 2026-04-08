#!/usr/bin/env python
"""
Generate publication-quality analysis figures from batch model fitting results.

Produces seven figures:
  A) R² heatmap (substrate vs biomass, per model × substrate)
  B) ΔAIC grouped bar chart
  C) Parameter identifiability (CV%) dot plot for best models
  D) Residual diagnostics panels for best models
  E) Total Error grouped bar chart (log scale) across model families
  F) Total Error heatmap (log10 scale) — compact substrate × model overview
  G) Parameter dot plots per substrate (parameter values across models, best starred)

Usage:
    python scripts/analyze_results.py
    python scripts/analyze_results.py --master output/master_results.csv
    python scripts/analyze_results.py --results-dir results --output-dir output/figures
    python scripts/analyze_results.py --formats png pdf --dpi 300
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.plotting import COLOR_LIST, style_axis, save_figure
from src.utils.master_table import _create_ode_system, MODEL_PARAMETERS
from src.io.config_loader import load_config
from src.io.data_loader import load_experimental_data
from src.core.oxygen import OxygenModel
from src.core.solvers import solve_ode

# Suppress matplotlib warnings for clean output
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# ── Global font settings: Arial 12pt ─────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 12,
    "axes.titlesize": 12,
    "axes.labelsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
})

# ── Canonical substrate display order ────────────────────────────────
SUBSTRATE_ORDER = [
    "Glucose",
    "Xylose",
    "SyringicAcid",
    "VanillicAcid",
    "pCoumaricAcid",
    "pHydroxybenzoicAcid",
]

SUBSTRATE_DISPLAY = {
    "Glucose": "Glucose",
    "Xylose": "Xylose",
    "SyringicAcid": "Syringic Acid",
    "VanillicAcid": "Vanillic Acid",
    "pCoumaricAcid": "p-Coumaric Acid",
    "pHydroxybenzoicAcid": "p-Hydroxybenzoic Acid",
}


def _ordered_substrates(substrates):
    """Return substrates in canonical order, appending any unknowns at the end."""
    ordered = [s for s in SUBSTRATE_ORDER if s in substrates]
    extras = [s for s in substrates if s not in SUBSTRATE_ORDER]
    return ordered + sorted(extras)


def _display(substrate):
    """Return a display-friendly substrate name."""
    return SUBSTRATE_DISPLAY.get(substrate, substrate)


# ── Canonical model complexity order (simple → complex) ─────────────
MODEL_ORDER = [
    "Single Monod",
    "Single Monod (Haldane)",
    "Single Monod + Lag",
    "Single Monod + Lag (Haldane)",
    "Dual Monod",
    "Dual Monod (Haldane)",
    "Dual Monod + Lag",
    "Dual Monod + Lag (Haldane)",
]


def _ordered_models(models):
    """Return models in complexity order, appending any unknowns at the end."""
    ordered = [m for m in MODEL_ORDER if m in models]
    extras = [m for m in models if m not in MODEL_ORDER]
    return ordered + sorted(extras)


# ── CLI ──────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate publication analysis figures from batch fitting results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--master-csv", default="output/master_results.csv",
        help="Path to master results CSV (default: output/master_results.csv)",
    )
    parser.add_argument(
        "--results-dir", default="results",
        help="Base directory containing per-run results (default: results)",
    )
    parser.add_argument(
        "--output-dir", default="output/figures",
        help="Directory for saved figures (default: output/figures)",
    )
    parser.add_argument(
        "--formats", nargs="+", default=["png", "pdf"],
        help="Output formats (default: png pdf)",
    )
    parser.add_argument(
        "--dpi", type=int, default=300,
        help="Resolution for raster formats (default: 300)",
    )
    return parser.parse_args()


# ── JSON Discovery ───────────────────────────────────────────────────

def discover_results_jsons(results_dir: str) -> dict:
    """
    Scan results_dir recursively for individual_condition_results.json files.

    When multiple timestamps exist for the same (substrate, display_name) combo,
    keep only the most recent (lexicographic sort on parent directory name).

    Returns:
        dict mapping (substrate, display_name) -> parsed JSON dict
    """
    results_dir = Path(results_dir)
    if not results_dir.exists():
        print(f"Warning: results directory not found: {results_dir}")
        return {}

    # Collect all JSONs with their directory paths for timestamp sorting
    candidates: dict = {}  # (substrate, display_name) -> [(dir_path, json_data)]

    for json_path in sorted(results_dir.rglob("individual_condition_results.json")):
        try:
            with open(json_path, "r") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        substrate = data.get("substrate")
        display_name = data.get("display_name")
        if not substrate or not display_name:
            continue

        key = (substrate, display_name)
        if key not in candidates:
            candidates[key] = []
        candidates[key].append((str(json_path.parent), data))

    # Keep only the most recent timestamp directory for each combo
    result = {}
    for key, entries in candidates.items():
        # Sort by directory path (timestamps are embedded, lexicographic = chronological)
        entries.sort(key=lambda x: x[0])
        result[key] = entries[-1][1]  # most recent

    return result


# ── Figure A: R² Heatmap ─────────────────────────────────────────────

def figure_a_r2_heatmap(
    json_data: dict,
    substrates: list,
    models: list,
    output_dir: Path,
    formats: list,
    dpi: int,
):
    """Two-panel heatmap: substrate R² (left) and biomass R² (right)."""
    n_sub = len(substrates)
    n_mod = len(models)

    # Build matrices
    r2_substrate = np.full((n_sub, n_mod), np.nan)
    r2_biomass = np.full((n_sub, n_mod), np.nan)

    for i, sub in enumerate(substrates):
        for j, mod in enumerate(models):
            jd = json_data.get((sub, mod))
            if jd is None:
                continue
            conditions = jd.get("conditions", {})
            sub_r2s = []
            bio_r2s = []
            for cond_data in conditions.values():
                stats = cond_data.get("statistics", {})
                sr2 = stats.get("substrate", {}).get("R_squared")
                br2 = stats.get("biomass", {}).get("R_squared")
                if sr2 is not None:
                    sub_r2s.append(sr2)
                if br2 is not None:
                    bio_r2s.append(br2)
            if sub_r2s:
                r2_substrate[i, j] = np.mean(sub_r2s)
            if bio_r2s:
                r2_biomass[i, j] = np.mean(bio_r2s)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=dpi)

    for ax, matrix, title in [
        (ax1, r2_substrate, "Substrate R²"),
        (ax2, r2_biomass, "Biomass R²"),
    ]:
        cmap = plt.cm.RdYlGn.copy()
        cmap.set_bad(color="#dddddd")
        masked = np.ma.masked_invalid(matrix)

        im = ax.imshow(masked, cmap=cmap, vmin=0, vmax=1, aspect="auto")

        # Annotate cells
        for i in range(n_sub):
            for j in range(n_mod):
                val = matrix[i, j]
                if np.isnan(val):
                    ax.text(j, i, "N/A", ha="center", va="center",
                            fontsize=12, color="#888888", fontweight="bold")
                else:
                    # White text on dark cells, black on light
                    text_color = "white" if val < 0.5 else "black"
                    ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                            fontsize=12, color=text_color, fontweight="bold")

        ax.set_xticks(range(n_mod))
        ax.set_xticklabels(models, rotation=45, ha="right")
        ax.set_yticks(range(n_sub))
        ax.set_yticklabels([_display(s) for s in substrates])
        ax.set_title(title, fontweight="bold", pad=12)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=12)

    fig.suptitle("Model Performance: R² by Component", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    out = output_dir / "figure_A_r2_heatmap"
    saved = save_figure(fig, out, formats=formats, dpi=dpi)
    plt.close(fig)
    return saved


# ── Figure B: ΔAIC Grouped Bar Chart ─────────────────────────────────

def figure_b_delta_aic(
    master_df: pd.DataFrame,
    substrates: list,
    models: list,
    output_dir: Path,
    formats: list,
    dpi: int,
):
    """Grouped bar chart of ΔAIC per substrate, bars sorted within each group."""
    fig, ax = plt.subplots(figsize=(14, 6), dpi=dpi)

    n_models = len(models)
    bar_width = 0.8 / n_models
    x = np.arange(len(substrates))

    # Pre-compute ΔAIC for every (substrate, model) pair
    delta_dict = {}  # {(sub, model): float | nan}
    for sub in substrates:
        group = master_df[master_df["Substrate"] == sub]
        aic_vals = pd.to_numeric(group["AIC"], errors="coerce")
        aic_min = aic_vals.min()
        for model in models:
            row = group[group["Model"] == model]
            if row.empty or pd.isna(row["AIC"].values[0]):
                delta_dict[(sub, model)] = np.nan
            else:
                delta_dict[(sub, model)] = float(row["AIC"].values[0]) - aic_min

    # Build a stable color map so each model always gets the same color
    model_color = {m: COLOR_LIST[j % len(COLOR_LIST)] for j, m in enumerate(models)}
    legend_added = set()

    for i, sub in enumerate(substrates):
        # Sort models within this substrate by ΔAIC ascending
        pairs = [(model, delta_dict[(sub, model)]) for model in models]
        pairs.sort(key=lambda p: p[1] if not np.isnan(p[1]) else np.inf)

        for slot, (model, val) in enumerate(pairs):
            offset = (slot - n_models / 2 + 0.5) * bar_width
            label = model if model not in legend_added else None
            ax.bar(
                x[i] + offset, val, bar_width,
                label=label,
                color=model_color[model],
                edgecolor="white", linewidth=0.5,
            )
            if label:
                legend_added.add(model)

            # Mark best model (ΔAIC = 0) with a star
            if val is not None and not np.isnan(val) and val == 0:
                ax.annotate(
                    "*", xy=(x[i] + offset, 0),
                    xytext=(0, -14), textcoords="offset points",
                    ha="center", va="top", fontsize=18, color=model_color[model],
                    fontweight="bold",
                )

    ax.set_xticks(x)
    ax.set_xticklabels([_display(s) for s in substrates], rotation=30, ha="right")
    ax.set_ylabel("\u0394AIC (relative to best model)")
    ax.set_title("Model Selection: \u0394AIC by Substrate", fontweight="bold")
    style_axis(ax, legend=False, grid=True)
    ax.legend(
        bbox_to_anchor=(1.05, 1), loc="upper left",
        frameon=True, fontsize=10,
    )
    plt.tight_layout()

    out = output_dir / "figure_B_delta_aic"
    saved = save_figure(fig, out, formats=formats, dpi=dpi)
    plt.close(fig)
    return saved


# ── Figure C: Parameter Identifiability (CV% Dot Plot) ───────────────

PARAM_ORDER = ["qmax", "Ks", "Ki", "Y", "K_o2", "Y_o2", "b_decay", "lag_time"]


def figure_c_cv_dotplot(
    json_data: dict,
    best_models: dict,
    output_dir: Path,
    formats: list,
    dpi: int,
):
    """
    Dot plot of parameter CV% for the best model per substrate.

    Args:
        best_models: dict mapping substrate -> (model_display_name, aic_value)
    """
    # Collect data
    rows = []  # (substrate, model, param, cv)
    for substrate, (model_name, _) in sorted(best_models.items()):
        jd = json_data.get((substrate, model_name))
        if jd is None:
            continue
        param_summary = jd.get("parameter_summary", {})
        for param, info in param_summary.items():
            cv = info.get("cv")
            if cv is not None:
                rows.append((substrate, model_name, param, cv))

    if not rows:
        print("  Warning: No CV% data available for Figure C")
        return []

    df = pd.DataFrame(rows, columns=["substrate", "model", "param", "cv"])

    # Get ordered list of parameters actually present
    params_present = [p for p in PARAM_ORDER if p in df["param"].values]
    if not params_present:
        params_present = sorted(df["param"].unique())

    substrates_sorted = _ordered_substrates(list(best_models.keys()))

    fig, ax = plt.subplots(figsize=(12, 7), dpi=dpi)

    # CV colormap: green = low/good, red = high/poor
    cv_max = min(df["cv"].max(), 200)  # cap for color scaling
    norm = mcolors.Normalize(vmin=0, vmax=cv_max)
    cmap = plt.cm.RdYlGn_r

    for i, substrate in enumerate(substrates_sorted):
        model_name = best_models[substrate][0]
        sub_df = df[(df["substrate"] == substrate) & (df["model"] == model_name)]

        for j, param in enumerate(params_present):
            row = sub_df[sub_df["param"] == param]
            if row.empty:
                continue
            cv = row["cv"].values[0]
            size = max(20, min(cv * 3, 400))  # dot size proportional to CV%
            color = cmap(norm(min(cv, cv_max)))

            ax.scatter(j, i, s=size, c=[color], edgecolors="black",
                       linewidths=0.5, zorder=5)
            ax.text(j, i - 0.35, f"{cv:.0f}%", ha="center", va="top",
                    fontsize=10, color="#444444")

    # Y-axis labels: substrate + model name
    y_labels = [
        f"{_display(sub)}\n({best_models[sub][0]})" for sub in substrates_sorted
    ]
    ax.set_yticks(range(len(substrates_sorted)))
    ax.set_yticklabels(y_labels)

    ax.set_xticks(range(len(params_present)))
    ax.set_xticklabels(params_present, rotation=45, ha="right")

    ax.set_xlim(-0.5, len(params_present) - 0.5)
    ax.set_ylim(-0.5, len(substrates_sorted) - 0.5)

    ax.set_title("Parameter Identifiability: CV% (Best Model per Substrate)",
                 fontweight="bold")

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label("CV (%)")
    cbar.ax.tick_params(labelsize=12)

    style_axis(ax, legend=False, grid=True)
    plt.tight_layout()

    out = output_dir / "figure_C_cv_dotplot"
    saved = save_figure(fig, out, formats=formats, dpi=dpi)
    plt.close(fig)
    return saved


# ── Figure D: Residual Diagnostics ───────────────────────────────────

def _simulate_condition(global_params, model_type, oxygen_model, time, S0, X0):
    """Re-simulate a single condition and return interpolated predictions."""
    if model_type == "single_monod":
        initial_conditions = [S0, X0]
    else:
        initial_conditions = [S0, X0, oxygen_model.o2_max]

    ode_system = _create_ode_system(global_params, model_type, oxygen_model)
    t_span = (time[0], time[-1])
    t_eval = np.linspace(t_span[0], t_span[1], max(1000, len(time) * 10))

    result = solve_ode(
        ode_system=ode_system,
        initial_conditions=np.array(initial_conditions, dtype=float),
        t_span=t_span,
        t_eval=t_eval,
    )

    pred_sub = result.states.get("Substrate", np.zeros_like(result.time))
    pred_bio = result.states.get("Biomass", np.zeros_like(result.time))

    return (
        np.interp(time, result.time, pred_sub),
        np.interp(time, result.time, pred_bio),
    )


def figure_d_residuals(
    json_data: dict,
    best_models: dict,
    output_dir: Path,
    formats: list,
    dpi: int,
):
    """Residual scatter plots for the best model per substrate."""
    substrates_sorted = _ordered_substrates(list(best_models.keys()))
    n = len(substrates_sorted)

    # Determine grid layout
    ncols = min(n, 3)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), dpi=dpi)
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, -1)
    elif ncols == 1:
        axes = axes.reshape(-1, 1)

    for idx, substrate in enumerate(substrates_sorted):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]
        model_name = best_models[substrate][0]
        jd = json_data.get((substrate, model_name))

        if jd is None:
            ax.set_visible(False)
            continue

        model_type = jd.get("model_type")
        global_params = jd.get("global_parameters")
        config_path = jd.get("config_path")
        data_path = jd.get("data_path")

        if not all([model_type, global_params, config_path, data_path]):
            ax.text(0.5, 0.5, "Data unavailable", transform=ax.transAxes,
                    ha="center", va="center")
            ax.set_title(f"{_display(substrate)}\n({model_name})", fontweight="bold")
            continue

        try:
            config = load_config(config_path)
            experimental_data = load_experimental_data(data_path, substrate_name=config.name)
            oxygen_model = OxygenModel(
                o2_max=config.oxygen.get("o2_max", 8.0),
                o2_min=config.oxygen.get("o2_min", 0.1),
                reaeration_rate=config.oxygen.get("reaeration_rate", 15.0),
                o2_range=config.oxygen.get("o2_range", 8.0),
            )
        except Exception as e:
            ax.text(0.5, 0.5, f"Load error:\n{e}", transform=ax.transAxes,
                    ha="center", va="center", fontsize=10)
            ax.set_title(f"{_display(substrate)}\n({model_name})", fontweight="bold")
            continue

        autocorrs = []
        for c_idx, condition in enumerate(experimental_data.conditions):
            color = COLOR_LIST[c_idx % len(COLOR_LIST)]
            try:
                time, obs_sub, obs_bio = experimental_data.get_condition_data(condition)
                pred_sub, pred_bio = _simulate_condition(
                    global_params, model_type, oxygen_model, time,
                    obs_sub[0], obs_bio[0],
                )
            except Exception:
                continue

            # Normalized residuals
            sub_range = np.ptp(obs_sub) or 1.0
            bio_range = np.ptp(obs_bio) or 1.0
            res_sub = (obs_sub - pred_sub) / sub_range
            res_bio = (obs_bio - pred_bio) / bio_range

            ax.scatter(time, res_sub, marker="o", color=color, s=30,
                       alpha=0.7, label=f"{condition} (S)" if c_idx == 0 else None)
            ax.scatter(time, res_bio, marker="^", color=color, s=30,
                       alpha=0.7, label=f"{condition} (X)" if c_idx == 0 else None)

            # Autocorrelation of combined residuals
            combined = np.concatenate([res_sub, res_bio])
            if len(combined) > 2:
                r = np.corrcoef(combined[:-1], combined[1:])[0, 1]
                if np.isfinite(r):
                    autocorrs.append(r)

        ax.axhline(y=0, color="black", linestyle="--", linewidth=1, alpha=0.5)
        ax.set_title(f"{_display(substrate)}\n({model_name})", fontweight="bold")
        ax.set_xlabel("Time (days)")
        ax.set_ylabel("Normalized Residual")

        # Autocorrelation annotation
        if autocorrs:
            mean_ac = np.mean(autocorrs)
            ax.text(
                0.97, 0.97, f"Mean lag-1\nautocorr: {mean_ac:.2f}",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=10, bbox=dict(boxstyle="round,pad=0.3",
                                      facecolor="wheat", alpha=0.8),
            )

        # Add legend for marker types
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker="o", color="gray", linestyle="None",
                   markersize=5, label="Substrate"),
            Line2D([0], [0], marker="^", color="gray", linestyle="None",
                   markersize=5, label="Biomass"),
        ]
        ax.legend(handles=legend_elements, loc="lower right", fontsize=10,
                  framealpha=0.8)

    # Hide unused panels
    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    fig.suptitle("Residual Diagnostics (Best Model per Substrate)",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    out = output_dir / "figure_D_residuals"
    saved = save_figure(fig, out, formats=formats, dpi=dpi)
    plt.close(fig)
    return saved


# ── Figure E: Total Error Grouped Bar Chart (log scale) ──────────────

def figure_e_total_error_bars(
    master_df: pd.DataFrame,
    substrates: list,
    models: list,
    output_dir: Path,
    formats: list,
    dpi: int,
):
    """Grouped bar chart of Total_Error (log scale) per substrate, bars sorted within each group."""
    fig, ax = plt.subplots(figsize=(14, 6), dpi=dpi)

    n_models = len(models)
    bar_width = 0.8 / n_models
    x = np.arange(len(substrates))

    # Pre-compute Total_Error for every (substrate, model) pair
    err_dict = {}  # {(sub, model): float | nan}
    for sub in substrates:
        for model in models:
            row = master_df[
                (master_df["Substrate"] == sub) & (master_df["Model"] == model)
            ]
            if row.empty or pd.isna(row["Total_Error"].values[0]):
                err_dict[(sub, model)] = np.nan
            else:
                err_dict[(sub, model)] = float(row["Total_Error"].values[0])

    # Build a stable color map so each model always gets the same color
    model_color = {m: COLOR_LIST[j % len(COLOR_LIST)] for j, m in enumerate(models)}
    legend_added = set()

    for i, sub in enumerate(substrates):
        # Sort models within this substrate by Total_Error ascending
        pairs = [(model, err_dict[(sub, model)]) for model in models]
        pairs.sort(key=lambda p: p[1] if not np.isnan(p[1]) else np.inf)

        for slot, (model, val) in enumerate(pairs):
            offset = (slot - n_models / 2 + 0.5) * bar_width
            label = model if model not in legend_added else None
            ax.bar(
                x[i] + offset, val, bar_width,
                label=label,
                color=model_color[model],
                edgecolor="white", linewidth=0.5,
            )
            if label:
                legend_added.add(model)

            # Mark lowest error (first in sorted order) with a star
            if slot == 0 and not np.isnan(val):
                ax.annotate(
                    "*", xy=(x[i] + offset, val),
                    xytext=(0, 5), textcoords="offset points",
                    ha="center", va="bottom", fontsize=18,
                    color=model_color[model], fontweight="bold",
                )

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([_display(s) for s in substrates], rotation=30, ha="right")
    ax.set_ylabel("Total Error (SSE, log scale)")
    ax.set_title("Total Error Across Model Families by Substrate",
                 fontweight="bold")
    style_axis(ax, legend=False, grid=True)
    ax.legend(
        bbox_to_anchor=(1.05, 1), loc="upper left",
        frameon=True, fontsize=10,
    )
    plt.tight_layout()

    out = output_dir / "figure_E_total_error_bars"
    saved = save_figure(fig, out, formats=formats, dpi=dpi)
    plt.close(fig)
    return saved


# ── Figure F: Total Error Heatmap ────────────────────────────────────

def figure_f_total_error_heatmap(
    master_df: pd.DataFrame,
    substrates: list,
    models: list,
    output_dir: Path,
    formats: list,
    dpi: int,
):
    """Heatmap of log10(Total_Error) by substrate × model."""
    n_sub = len(substrates)
    n_mod = len(models)

    matrix = np.full((n_sub, n_mod), np.nan)
    for i, sub in enumerate(substrates):
        for j, mod in enumerate(models):
            row = master_df[
                (master_df["Substrate"] == sub) & (master_df["Model"] == mod)
            ]
            if not row.empty:
                val = pd.to_numeric(row["Total_Error"], errors="coerce").values[0]
                if np.isfinite(val) and val > 0:
                    matrix[i, j] = np.log10(val)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=dpi)

    cmap = plt.cm.RdYlGn_r.copy()  # green = low error, red = high
    cmap.set_bad(color="#dddddd")
    masked = np.ma.masked_invalid(matrix)

    # Symmetric range around data for better contrast
    vmin = np.nanmin(matrix)
    vmax = np.nanmax(matrix)
    im = ax.imshow(masked, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")

    # Annotate cells with raw Total_Error (scientific notation) and highlight min per row
    for i in range(n_sub):
        row_vals = matrix[i, :]
        row_min = np.nanmin(row_vals) if np.any(np.isfinite(row_vals)) else None
        for j in range(n_mod):
            val = matrix[i, j]
            if np.isnan(val):
                ax.text(j, i, "N/A", ha="center", va="center",
                        fontsize=12, color="#888888")
            else:
                raw = 10 ** val
                # Format: e.g., "8.8e5" or "5.6e7"
                txt = f"{raw:.1e}"
                # Highlight minimum in row with bold + box
                norm_pos = (val - vmin) / (vmax - vmin) if vmax > vmin else 0.5
                text_color = "white" if norm_pos > 0.6 else "black"
                fw = "bold"
                if row_min is not None and val == row_min:
                    ax.add_patch(plt.Rectangle(
                        (j - 0.45, i - 0.45), 0.9, 0.9,
                        fill=False, edgecolor="black", linewidth=2.5,
                    ))
                ax.text(j, i, txt, ha="center", va="center",
                        fontsize=12, color=text_color, fontweight=fw)

    ax.set_xticks(range(n_mod))
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.set_yticks(range(n_sub))
    ax.set_yticklabels([_display(s) for s in substrates])
    ax.set_title("Total Error: log10(SSE) by Substrate x Model",
                 fontweight="bold", pad=12)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("log10(Total Error)")
    cbar.ax.tick_params(labelsize=12)

    plt.tight_layout()

    out = output_dir / "figure_F_total_error_heatmap"
    saved = save_figure(fig, out, formats=formats, dpi=dpi)
    plt.close(fig)
    return saved


# ── Figure G: Parameter Dot Plots per Substrate ─────────────────────

# Parameters in the master CSV and their display labels
_MASTER_PARAMS = [
    ('qmax', 'qmax'),
    ('Ks', 'Ks'),
    ('Ki', 'Ki'),
    ('Y', 'Y'),
    ('K_O2', 'K_O2'),
    ('Y_O2', 'Y_O2'),
    ('b_decay', 'b_decay'),
    ('lag_time', 'lag_time'),
]


def _parse_ci_string(ci_str):
    """Parse a '[lower, upper]' CI string into (lower, upper) floats."""
    if not isinstance(ci_str, str):
        return None, None
    ci_str = ci_str.strip()
    if not ci_str.startswith('['):
        return None, None
    try:
        parts = ci_str.strip('[]').split(',')
        return float(parts[0].strip()), float(parts[1].strip())
    except (ValueError, IndexError):
        return None, None


def figure_g_parameter_dotplots(
    master_df: pd.DataFrame,
    substrates: list,
    models: list,
    best_models: dict,
    output_dir: Path,
    formats: list,
    dpi: int,
):
    """
    One subplot per substrate showing parameter values (dots) across models.

    Each parameter gets its own y-position; models are distinguished by colour.
    Error bars show 95% CIs when available. The best model (lowest AIC) is
    marked with a star.
    """
    n_sub = len(substrates)
    ncols = min(n_sub, 3)
    nrows = int(np.ceil(n_sub / ncols))
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(7 * ncols, 5 * nrows),
        dpi=dpi,
        squeeze=False,
    )

    # Build colour map for models (consistent across subplots)
    model_colors = {mod: COLOR_LIST[j % len(COLOR_LIST)] for j, mod in enumerate(models)}

    for idx, substrate in enumerate(substrates):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]

        sub_df = master_df[master_df['Substrate'] == substrate]
        best_model = best_models.get(substrate, (None, None))[0]

        # Determine which parameters have at least one non-NaN value
        params_present = []
        for col_name, label in _MASTER_PARAMS:
            vals = pd.to_numeric(sub_df[col_name], errors='coerce')
            if vals.notna().any():
                params_present.append((col_name, label))

        if not params_present:
            ax.set_visible(False)
            continue

        n_params = len(params_present)
        n_models_here = len(sub_df)

        # Vertical positions for parameters (bottom to top)
        y_positions = np.arange(n_params)

        # Slight vertical jitter so overlapping models are distinguishable
        n_mod = len(models)
        jitter_total = 0.6  # total vertical span within one parameter row
        jitter_offsets = np.linspace(-jitter_total / 2, jitter_total / 2, n_mod)

        legend_handles = []

        for j, model in enumerate(models):
            model_row = sub_df[sub_df['Model'] == model]
            if model_row.empty:
                continue

            color = model_colors[model]
            is_best = (model == best_model)
            marker = '*' if is_best else 'o'
            ms = 18 if is_best else 8
            zorder = 10 if is_best else 5

            x_vals = []
            y_vals = []
            xerr_lo = []
            xerr_hi = []

            for p_idx, (col_name, label) in enumerate(params_present):
                val = pd.to_numeric(model_row[col_name], errors='coerce').values[0]
                if np.isnan(val):
                    continue

                x_vals.append(val)
                y_vals.append(y_positions[p_idx] + jitter_offsets[j])

                # Parse CI
                ci_col = f"{col_name}_95CI"
                lo, hi = None, None
                if ci_col in model_row.columns:
                    lo, hi = _parse_ci_string(model_row[ci_col].values[0])

                if lo is not None and hi is not None:
                    xerr_lo.append(max(0, val - lo))
                    xerr_hi.append(max(0, hi - val))
                else:
                    xerr_lo.append(0)
                    xerr_hi.append(0)

            if not x_vals:
                continue

            x_vals = np.array(x_vals)
            y_vals = np.array(y_vals)
            xerr = np.array([xerr_lo, xerr_hi])

            # Plot error bars first (behind markers)
            ax.errorbar(
                x_vals, y_vals, xerr=xerr,
                fmt='none', ecolor=color, elinewidth=1.2, capsize=3,
                alpha=0.6, zorder=zorder - 1,
            )
            # Plot markers
            handle = ax.scatter(
                x_vals, y_vals,
                c=[color], marker=marker, s=ms ** 2 if is_best else ms ** 2,
                edgecolors='black' if is_best else color,
                linewidths=1.0 if is_best else 0.5,
                zorder=zorder,
                label=f"{model}{' *' if is_best else ''}",
            )
            legend_handles.append(handle)

        ax.set_yticks(y_positions)
        ax.set_yticklabels([label for _, label in params_present])
        ax.set_xlabel('Parameter Value')
        ax.set_title(_display(substrate), fontweight='bold')
        ax.set_xscale('symlog', linthresh=1e-2)  # handle wide range of param magnitudes

        # Light horizontal gridlines for readability
        for yp in y_positions:
            ax.axhline(y=yp, color='#eeeeee', linewidth=0.8, zorder=0)

        ax.set_ylim(-0.8, n_params - 0.2)
        ax.invert_yaxis()

        # Legend in each subplot
        ax.legend(
            fontsize=9, loc='lower right',
            framealpha=0.9, handletextpad=0.5,
            markerscale=0.6,
        )

    # Hide unused panels
    for idx in range(n_sub, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    fig.suptitle(
        'Parameter Comparison Across Models (* = Best AIC)',
        fontsize=14, fontweight='bold', y=1.02,
    )
    plt.tight_layout()

    out = output_dir / 'figure_G_parameter_dotplots'
    saved = save_figure(fig, out, formats=formats, dpi=dpi)
    plt.close(fig)
    return saved


# ── Main ─────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load master CSV
    master_path = Path(args.master_csv)
    if not master_path.exists():
        print(f"Error: Master CSV not found: {master_path}")
        return 1

    master_df = pd.read_csv(master_path)
    substrates = _ordered_substrates(master_df["Substrate"].unique().tolist())
    models = _ordered_models(master_df["Model"].unique().tolist())

    print("=" * 60)
    print("PUBLICATION ANALYSIS FIGURES")
    print("=" * 60)
    print(f"  Master CSV:   {master_path}")
    print(f"  Results dir:  {args.results_dir}")
    print(f"  Output dir:   {output_dir}")
    print(f"  Substrates:   {len(substrates)}")
    print(f"  Models:       {len(models)}")
    print(f"  Formats:      {', '.join(args.formats)}")
    print("=" * 60)

    # 2. Discover results JSONs
    print("\nDiscovering results JSONs...")
    json_data = discover_results_jsons(args.results_dir)
    print(f"  Found {len(json_data)} unique (substrate, model) results")

    # 3. Determine best model per substrate (lowest AIC)
    best_models = {}
    for sub in substrates:
        group = master_df[master_df["Substrate"] == sub]
        aic_vals = pd.to_numeric(group["AIC"], errors="coerce")
        valid = aic_vals.dropna()
        if not valid.empty:
            best_idx = valid.idxmin()
            best_model = group.loc[best_idx, "Model"]
            best_aic = float(valid.loc[best_idx])
            best_models[sub] = (best_model, best_aic)

    print(f"\nBest models per substrate:")
    for sub, (mod, aic) in sorted(best_models.items()):
        print(f"  {sub}: {mod} (AIC={aic:.1f})")

    # 4. Generate figures
    all_saved = []

    print("\nGenerating Figure A: R² Heatmap...")
    saved = figure_a_r2_heatmap(json_data, substrates, models, output_dir,
                                args.formats, args.dpi)
    all_saved.extend(saved)
    print(f"  Saved: {', '.join(str(p) for p in saved)}")

    print("\nGenerating Figure B: \u0394AIC Grouped Bars...")
    saved = figure_b_delta_aic(master_df, substrates, models, output_dir,
                               args.formats, args.dpi)
    all_saved.extend(saved)
    print(f"  Saved: {', '.join(str(p) for p in saved)}")

    print("\nGenerating Figure C: CV% Dot Plot...")
    saved = figure_c_cv_dotplot(json_data, best_models, output_dir,
                                args.formats, args.dpi)
    all_saved.extend(saved)
    if saved:
        print(f"  Saved: {', '.join(str(p) for p in saved)}")

    print("\nGenerating Figure D: Residual Diagnostics...")
    saved = figure_d_residuals(json_data, best_models, output_dir,
                               args.formats, args.dpi)
    all_saved.extend(saved)
    print(f"  Saved: {', '.join(str(p) for p in saved)}")

    print("\nGenerating Figure E: Total Error Bars (log scale)...")
    saved = figure_e_total_error_bars(master_df, substrates, models, output_dir,
                                      args.formats, args.dpi)
    all_saved.extend(saved)
    print(f"  Saved: {', '.join(str(p) for p in saved)}")

    print("\nGenerating Figure F: Total Error Heatmap...")
    saved = figure_f_total_error_heatmap(master_df, substrates, models, output_dir,
                                          args.formats, args.dpi)
    all_saved.extend(saved)
    print(f"  Saved: {', '.join(str(p) for p in saved)}")

    print("\nGenerating Figure G: Parameter Dot Plots per Substrate...")
    saved = figure_g_parameter_dotplots(master_df, substrates, models, best_models,
                                         output_dir, args.formats, args.dpi)
    all_saved.extend(saved)
    if saved:
        print(f"  Saved: {', '.join(str(p) for p in saved)}")

    print(f"\nDone. {len(all_saved)} files saved to {output_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
