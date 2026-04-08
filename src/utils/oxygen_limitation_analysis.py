#!/usr/bin/env python
"""
Oxygen Limitation Analysis across Substrates — All Dual Monod Variants.

For every substrate, evaluates **all four** dual Monod model variants:
  - Dual Monod (DM)
  - Dual Monod + Haldane (DH)
  - Dual Monod + Lag (DM+L)
  - Dual Monod + Lag + Haldane (DH+L)

Key diagnostic quantities per (substrate, model):
  1. O2 saturation factor at O2_min — how throttled the rate is at the
     minimum DO the system can reach (0.1 mg/L by default).
  2. O2 saturation factor at O2_max — how saturated the rate is at full
     aeration (8.0 mg/L by default).
  3. Effective O2 swing ratio — sat@max / sat@min.
  4. O2 demand intensity — Y_O2 x qmax.

Produces a 3-row figure:
  Row 1 — Small-multiple O2 Monod saturation curves (one subplot per
           substrate, each showing all 4 model variants).
  Row 2 — Grouped bar charts: sat@O2_min (left) and O2 demand (right).
  Row 3 — Full summary table (all substrate x model combinations).

Usage:
    python src/utils/oxygen_limitation_analysis.py
    python src/utils/oxygen_limitation_analysis.py --csv master_results_all_models_DE_4.3.csv
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

# ── Global font settings ─────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 12,
    "axes.titlesize": 12,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 9,
    "figure.titlesize": 14,
})

# ── Canonical ordering ───────────────────────────────────────────────────────
SUBSTRATE_ORDER = [
    "Glucose", "Xylose", "SyringicAcid",
    "VanillicAcid", "pCoumaricAcid", "pHydroxybenzoicAcid",
]
SUBSTRATE_DISPLAY = {
    "Glucose": "Glucose",
    "Xylose": "Xylose",
    "SyringicAcid": "Syringic Acid",
    "VanillicAcid": "Vanillic Acid",
    "pCoumaricAcid": "p-Coumaric Acid",
    "pHydroxybenzoicAcid": "p-Hydroxybenzoic Acid",
}
MODEL_ORDER = [
    "Dual Monod",
    "Dual Monod (Haldane)",
    "Dual Monod + Lag",
    "Dual Monod + Lag (Haldane)",
]
MODEL_SHORT = {
    "Dual Monod": "DM",
    "Dual Monod (Haldane)": "DH",
    "Dual Monod + Lag": "DM+L",
    "Dual Monod + Lag (Haldane)": "DH+L",
}

# Colours per model variant — matches analyze_results.py / COLOR_LIST
# (indices 4–7 in MODEL_ORDER correspond to the four dual models)
MODEL_COLORS = {
    "Dual Monod":                 "#66CCEE",   # cyan   (COLOR_LIST[4])
    "Dual Monod (Haldane)":       "#AA3377",   # purple (COLOR_LIST[5])
    "Dual Monod + Lag":           "#BBBBBB",   # grey   (COLOR_LIST[6])
    "Dual Monod + Lag (Haldane)": "#EE7733",   # orange (COLOR_LIST[7])
}
MODEL_LINESTYLES = {
    "Dual Monod":                 "-",
    "Dual Monod (Haldane)":       "--",
    "Dual Monod + Lag":           "-.",
    "Dual Monod + Lag (Haldane)": ":",
}

# Colours per substrate (Okabe-Ito inspired)
SUB_COLORS = {
    "Glucose":              "#0072B2",
    "Xylose":               "#56B4E9",
    "SyringicAcid":         "#009E73",
    "VanillicAcid":         "#E69F00",
    "pCoumaricAcid":        "#D55E00",
    "pHydroxybenzoicAcid":  "#CC79A7",
}

# ── Default oxygen bounds from config ────────────────────────────────────────
O2_MIN = 0.1   # mg/L
O2_MAX = 8.0   # mg/L


def _display(s):
    return SUBSTRATE_DISPLAY.get(s, s)


def _ordered(items, order):
    ordered = [x for x in order if x in items]
    extras = [x for x in items if x not in order]
    return ordered + sorted(extras)


def o2_monod(o2, k_o2):
    """Oxygen Monod saturation factor: O2 / (K_O2 + O2)."""
    o2 = np.maximum(np.asarray(o2, dtype=float), 0.0)
    return o2 / (k_o2 + o2)


# ══════════════════════════════════════════════════════════════════════════════
# Main analysis
# ══════════════════════════════════════════════════════════════════════════════
def analyse(csv_path: str, output_dir: str = "output"):
    """Run the full oxygen limitation analysis and produce figures."""
    df = pd.read_csv(csv_path)

    # Keep only dual models
    dual = df[df["Model"].isin(MODEL_ORDER)].copy()
    if dual.empty:
        print("Error: no Dual Monod rows found in the CSV.")
        return

    # Coerce numeric columns
    for col in ["K_O2", "Y_O2", "qmax", "Total_Error", "R2", "AIC"]:
        if col in dual.columns:
            dual[col] = pd.to_numeric(dual[col], errors="coerce")

    substrates = _ordered(dual["Substrate"].unique().tolist(), SUBSTRATE_ORDER)
    models = _ordered(dual["Model"].unique().tolist(), MODEL_ORDER)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Compute per-row diagnostics ──────────────────────────────────────────
    o2_sweep = np.linspace(0, O2_MAX, 500)
    rows = []

    for _, r in dual.iterrows():
        k = r["K_O2"]
        sub = r["Substrate"]
        mod = r["Model"]
        if np.isnan(k):
            continue
        sat_min = float(o2_monod(O2_MIN, k))
        sat_max = float(o2_monod(O2_MAX, k))
        swing = sat_max / sat_min if sat_min > 0 else np.inf
        demand = r["Y_O2"] * r["qmax"]

        rows.append({
            "Substrate":     sub,
            "Model":         mod,
            "K_O2":          k,
            "Y_O2":          r["Y_O2"],
            "qmax":          r["qmax"],
            "sat_at_O2min":  sat_min,
            "sat_at_O2max":  sat_max,
            "swing":         swing,
            "O2_demand":     demand,
            "R2":            r.get("R2", np.nan),
            "AIC":           r.get("AIC", np.nan),
        })

    info = pd.DataFrame(rows)

    # Sort into canonical order
    info["_sub_idx"] = info["Substrate"].map(
        {s: i for i, s in enumerate(SUBSTRATE_ORDER)}
    ).fillna(99).astype(int)
    info["_mod_idx"] = info["Model"].map(
        {m: i for i, m in enumerate(MODEL_ORDER)}
    ).fillna(99).astype(int)
    info = info.sort_values(["_sub_idx", "_mod_idx"]).reset_index(drop=True)

    n_sub = len(substrates)
    n_mod = len(models)

    # ═══════════════════════════════════════════════════════════════════════════
    # FIGURE — 3-row layout
    #   Row 0: small-multiple saturation curves (one per substrate)
    #   Row 1: grouped bar charts (sat@O2min left, O2 demand right)
    #   Row 2: full summary table
    # ═══════════════════════════════════════════════════════════════════════════
    n_cols_top = min(n_sub, 3)
    n_rows_top = int(np.ceil(n_sub / n_cols_top))

    fig = plt.figure(figsize=(20, 8 + 5 * n_rows_top + 6), dpi=200,
                     facecolor="white")
    outer = fig.add_gridspec(
        3, 1,
        height_ratios=[n_rows_top * 4, 5, 6],
        hspace=0.30,
    )

    # ── Row 0: Small-multiple saturation curves ──────────────────────────────
    inner_curves = outer[0].subgridspec(n_rows_top, n_cols_top,
                                        hspace=0.40, wspace=0.28)

    panel_letter = ord("A")
    for idx, sub in enumerate(substrates):
        row_i = idx // n_cols_top
        col_i = idx % n_cols_top
        ax = fig.add_subplot(inner_curves[row_i, col_i])
        letter = chr(panel_letter + idx)
        ax.set_title(
            f"{letter}.  {_display(sub)}",
            fontweight="bold", fontsize=12,
        )

        sub_data = info[info["Substrate"] == sub]
        for _, r in sub_data.iterrows():
            mod = r["Model"]
            k = r["K_O2"]
            curve = o2_monod(o2_sweep, k)
            ax.plot(
                o2_sweep, curve,
                color=MODEL_COLORS.get(mod, "#888"),
                linestyle=MODEL_LINESTYLES.get(mod, "-"),
                linewidth=2.2,
                label=f"{MODEL_SHORT[mod]}  K_O2={k:.3f}",
            )

        # Reference lines
        ax.axvline(O2_MIN, color="#CC0000", linestyle="--", linewidth=0.9,
                   alpha=0.6)
        ax.axhline(1.0, color="#888", linestyle=":", linewidth=0.5, alpha=0.4)
        ax.axhline(0.5, color="#CC0000", linestyle=":", linewidth=0.5,
                   alpha=0.3)
        ax.set_xlim(0, O2_MAX + 0.3)
        ax.set_ylim(0, 1.08)
        ax.set_xlabel("Dissolved O2 (mg/L)")
        ax.set_ylabel("O2 / (K_O2 + O2)")
        ax.legend(fontsize=8, loc="lower right", framealpha=0.85)
        ax.grid(True, alpha=0.20)

    # Hide unused subplots
    for idx in range(n_sub, n_rows_top * n_cols_top):
        row_i = idx // n_cols_top
        col_i = idx % n_cols_top
        ax = fig.add_subplot(inner_curves[row_i, col_i])
        ax.axis("off")

    # ── Row 1: Grouped bar charts ────────────────────────────────────────────
    inner_bars = outer[1].subgridspec(1, 2, wspace=0.30)

    bar_width = 0.18
    x_base = np.arange(n_sub)
    offsets = np.linspace(
        -(n_mod - 1) * bar_width / 2,
         (n_mod - 1) * bar_width / 2,
        n_mod,
    )

    # ---- Panel: Sat @ O2_min ----
    panel_sat_letter = chr(panel_letter + n_sub)
    ax_sat = fig.add_subplot(inner_bars[0, 0])
    ax_sat.set_title(
        f"{panel_sat_letter}.  O2 Saturation at Minimum DO ({O2_MIN} mg/L)",
        fontweight="bold",
    )

    for j, mod in enumerate(models):
        vals = []
        for sub in substrates:
            row = info[(info["Substrate"] == sub) & (info["Model"] == mod)]
            vals.append(row["sat_at_O2min"].values[0] if len(row) else 0)
        ax_sat.bar(
            x_base + offsets[j], vals, width=bar_width,
            color=MODEL_COLORS.get(mod, "#888"),
            edgecolor="white", linewidth=0.6,
            label=mod,
        )
        for xi, val in zip(x_base + offsets[j], vals):
            ax_sat.text(xi, val + 0.01, f"{val:.2f}",
                        ha="center", va="bottom", fontsize=7.5,
                        fontweight="bold", rotation=90)

    ax_sat.set_xticks(x_base)
    ax_sat.set_xticklabels([_display(s) for s in substrates],
                           rotation=30, ha="right")
    ax_sat.set_ylabel("O2 / (K_O2 + O2)  at O2 = 0.1")
    ax_sat.set_ylim(0, 1.15)
    ax_sat.axhline(0.5, color="#CC0000", linestyle="--", linewidth=0.8,
                   alpha=0.5, label="50% threshold")
    ax_sat.legend(fontsize=8, loc="upper right", ncol=1, framealpha=0.9)
    ax_sat.grid(axis="y", alpha=0.20)

    # ---- Panel: O2 demand intensity ----
    panel_dem_letter = chr(panel_letter + n_sub + 1)
    ax_dem = fig.add_subplot(inner_bars[0, 1])
    ax_dem.set_title(
        f"{panel_dem_letter}.  O2 Demand Intensity  (Y_O2 x qmax)",
        fontweight="bold",
    )

    for j, mod in enumerate(models):
        vals = []
        for sub in substrates:
            row = info[(info["Substrate"] == sub) & (info["Model"] == mod)]
            vals.append(row["O2_demand"].values[0] if len(row) else 0)
        ax_dem.bar(
            x_base + offsets[j], vals, width=bar_width,
            color=MODEL_COLORS.get(mod, "#888"),
            edgecolor="white", linewidth=0.6,
            label=mod,
        )
        for xi, val in zip(x_base + offsets[j], vals):
            ax_dem.text(xi, val + 0.15, f"{val:.1f}",
                        ha="center", va="bottom", fontsize=7.5,
                        fontweight="bold", rotation=90)

    ax_dem.set_xticks(x_base)
    ax_dem.set_xticklabels([_display(s) for s in substrates],
                           rotation=30, ha="right")
    ax_dem.set_ylabel("Y_O2 x qmax  (mg O2 / mg cells / day)")
    ax_dem.legend(fontsize=8, loc="upper right", ncol=1, framealpha=0.9)
    ax_dem.grid(axis="y", alpha=0.20)

    # ── Row 2: Full summary table ────────────────────────────────────────────
    panel_tbl_letter = chr(panel_letter + n_sub + 2)
    ax_tbl = fig.add_subplot(outer[2])
    ax_tbl.set_title(
        f"{panel_tbl_letter}.  Full Summary — All Dual Monod Variants",
        fontweight="bold", fontsize=13,
    )
    ax_tbl.axis("off")

    col_labels = [
        "Substrate", "Model", "K_O2", "Y_O2", "qmax",
        "Sat@0.1", "Sat@8.0", "Swing", "O2 Demand", "R2", "AIC",
    ]
    table_data = []
    for _, r in info.iterrows():
        table_data.append([
            _display(r["Substrate"]),
            MODEL_SHORT.get(r["Model"], r["Model"]),
            f"{r['K_O2']:.4f}",
            f"{r['Y_O2']:.4f}",
            f"{r['qmax']:.2f}",
            f"{r['sat_at_O2min']:.3f}",
            f"{r['sat_at_O2max']:.4f}",
            f"{r['swing']:.1f}x",
            f"{r['O2_demand']:.2f}",
            f"{r['R2']:.3f}",
            f"{r['AIC']:.1f}",
        ])

    tbl = ax_tbl.table(
        cellText=table_data, colLabels=col_labels,
        loc="center", cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.45)

    # Style header
    for j in range(len(col_labels)):
        tbl[0, j].set_text_props(fontweight="bold", fontsize=9)
        tbl[0, j].set_facecolor("#D8D8D8")

    # Colour-code cells
    for i, (_, r) in enumerate(info.iterrows(), start=1):
        # Sat@0.1 column (index 5)
        val = r["sat_at_O2min"]
        if val < 0.3:
            tbl[i, 5].set_facecolor("#FFCCCC")     # heavily limited — red
        elif val < 0.5:
            tbl[i, 5].set_facecolor("#FFEEBB")     # moderate — amber
        else:
            tbl[i, 5].set_facecolor("#CCFFCC")     # negligible — green

        # Model column (index 1) — tint by model colour
        mod = r["Model"]
        mc = MODEL_COLORS.get(mod, "#FFFFFF")
        tbl[i, 1].set_facecolor(mc + "33")  # low-alpha hex suffix

        # Alternate row shading by substrate
        sub_idx = SUBSTRATE_ORDER.index(r["Substrate"]) if r["Substrate"] in SUBSTRATE_ORDER else 0
        if sub_idx % 2 == 1:
            for j in [0, 2, 3, 4, 6, 7, 8, 9, 10]:
                tbl[i, j].set_facecolor("#F4F4F4")

    # Mark best AIC per substrate with bold text
    for sub in substrates:
        sub_rows = info[info["Substrate"] == sub]
        if sub_rows.empty:
            continue
        best_idx = sub_rows["AIC"].idxmin()
        row_pos = info.index.get_loc(best_idx) + 1  # +1 for header
        for j in range(len(col_labels)):
            tbl[row_pos, j].set_text_props(fontweight="bold")

    fig.suptitle(
        "Oxygen Limitation Analysis — All Dual Monod Variants per Substrate",
        fontsize=15, fontweight="bold", y=0.995,
    )

    # ── Save ─────────────────────────────────────────────────────────────────
    stem = out / "oxygen_limitation_analysis"
    for fmt in ("png", "pdf"):
        p = stem.with_suffix(f".{fmt}")
        fig.savefig(p, bbox_inches="tight", facecolor="white",
                    dpi=300 if fmt == "png" else 150)
        print(f"Saved -> {p}")
    plt.close(fig)

    # ── Print full text summary ──────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("OXYGEN LIMITATION SUMMARY  —  All Dual Monod Variants")
    print("=" * 100)
    print(
        f"{'Substrate':<24} {'Model':<7} {'K_O2':>8} {'Y_O2':>8} "
        f"{'qmax':>8} {'Sat@0.1':>8} {'Sat@8.0':>8} {'Swing':>7} "
        f"{'O2 Dem':>8} {'R2':>7} {'AIC':>9}"
    )
    print("-" * 100)
    prev_sub = None
    for _, r in info.iterrows():
        sub = r["Substrate"]
        if prev_sub is not None and sub != prev_sub:
            print("-" * 100)
        prev_sub = sub

        flag = " **" if r["sat_at_O2min"] < 0.3 else ""
        sub_rows = info[info["Substrate"] == sub]
        is_best = r["AIC"] == sub_rows["AIC"].min()
        marker = " <-- best" if is_best else ""

        print(
            f"{_display(sub):<24} "
            f"{MODEL_SHORT.get(r['Model'], '?'):<7} "
            f"{r['K_O2']:8.4f} "
            f"{r['Y_O2']:8.4f} "
            f"{r['qmax']:8.2f} "
            f"{r['sat_at_O2min']:8.3f} "
            f"{r['sat_at_O2max']:8.4f} "
            f"{r['swing']:6.1f}x "
            f"{r['O2_demand']:8.2f}"
            f"{flag}"
            f"  R2={r['R2']:.3f}  AIC={r['AIC']:.1f}{marker}"
        )
    print("-" * 100)
    print("Sat@0.1 = O2/(K_O2+O2) evaluated at O2 = 0.1 mg/L")
    print("Swing   = Sat@8.0 / Sat@0.1")
    print("O2 Dem  = Y_O2 * qmax  (oxygen consumption intensity)")
    print("**  = heavily O2-limited (Sat@0.1 < 0.30)")
    print("<-- best = lowest AIC among dual variants for that substrate")


def main():
    parser = argparse.ArgumentParser(
        description="Analyse oxygen limitation across substrates — all dual Monod variants",
    )
    parser.add_argument(
        "--csv", default="master_results_all_models_DE_4.3.csv",
        help="Path to master results CSV (default: master_results_all_models_DE_4.3.csv)",
    )
    parser.add_argument(
        "--output-dir", default="output",
        help="Directory for saved figures (default: output)",
    )
    args = parser.parse_args()
    analyse(args.csv, args.output_dir)


if __name__ == "__main__":
    main()
