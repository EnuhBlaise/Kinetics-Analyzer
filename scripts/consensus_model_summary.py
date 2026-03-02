"""
consensus_publication.py
────────────────────────
Dot matrix: X = substrate, Y = model (simple → complex)
  • Dot appears where a model received ≥1 metric vote
  • Dot SIZE   = number of metrics that voted for it (1, 2, or 3)
  • Dot COLOR  = model identity (Okabe-Ito palette)
  • Dot SHAPE  = ★ if consensus winner, ○ otherwise
  • Orbit markers = which metric(s) voted (▲ AIC, ■ R², ● SSE)

Usage:
    python consensus_publication.py master_results.csv [output.pdf]
"""

import sys
import warnings
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Okabe-Ito colorblind-safe palette ───────────────────────────────────────
MODEL_COLORS = {
    "Single Monod":                   "#0072B2",
    "Single Monod (Haldane)":         "#56B4E9",
    "Single Monod + Lag":             "#009E73",
    "Single Monod + Lag (Haldane)":   "#2ECC71",
    "Dual Monod":                     "#E69F00",
    "Dual Monod (Haldane)":           "#D55E00",
    "Dual Monod + Lag":               "#CC79A7",
    "Dual Monod + Lag (Haldane)":     "#8B4513",
}

# ordered simple → complex (bottom → top on y-axis)
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

METRICS     = ["AIC", "R\u00b2", "SSE"]
METRIC_KEYS = ["AIC", "R2",      "Total_Error"]
METRIC_DIR  = ["min", "max",     "min"]

# orbit offsets for metric sub-markers (angle in degrees)
METRIC_ANGLES  = [130, 270, 50]     # AIC top-left, R² bottom, SSE top-right
METRIC_MARKERS = ["^", "s", "o"]    # ▲ ■ ●
METRIC_COLORS  = ["#333333", "#333333", "#333333"]

plt.rcParams.update({
    "font.family":  "serif",
    "font.serif":   ["Times New Roman", "DejaVu Serif"],
    "font.size":    8,
    "pdf.fonttype": 42,
    "ps.fonttype":  42,
    "axes.linewidth": 0.6,
})


def model_color(m):
    return MODEL_COLORS.get(m, "#888")


def best_per_metric(df_sub):
    out = {}
    for lab, key, direction in zip(METRICS, METRIC_KEYS, METRIC_DIR):
        fn = df_sub[key].idxmin if direction == "min" else df_sub[key].idxmax
        out[lab] = df_sub.loc[fn(), "Model"]
    return out


def get_consensus(votes):
    c = Counter(votes.values())
    winner, n = c.most_common(1)[0]
    return winner, n


def make_figure(csv_path,
                out_pdf="consensus_publication.pdf",
                out_png="consensus_publication.png"):

    df = pd.read_csv(csv_path)
    substrates = sorted(df["Substrate"].unique())
    n_sub = len(substrates)
    n_mod = len(MODEL_ORDER)

    # ── pre-compute all votes ────────────────────────────────────────────────
    # vote_map[(substrate, model)] = list of metric labels that voted for it
    vote_map   = defaultdict(list)
    winners    = {}   # substrate → winning model

    for sub in substrates:
        votes  = best_per_metric(df[df["Substrate"] == sub])
        winner, n_votes = get_consensus(votes)
        winners[sub] = winner
        for metric, model in votes.items():
            vote_map[(sub, model)].append(metric)

    # ── figure setup ─────────────────────────────────────────────────────────
    fig_w = max(7.0, n_sub * 1.15 + 2.8)
    fig_h = n_mod * 0.95 + 2.0
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor="white")
    ax.set_facecolor("#FAFAFA")

    # x = substrate index, y = model index
    sub_pos = {s: i for i, s in enumerate(substrates)}
    mod_pos = {m: i for i, m in enumerate(MODEL_ORDER)}

    ax.set_xlim(-0.7, n_sub - 0.3)
    ax.set_ylim(-0.7, n_mod - 0.3)

    # ── light grid ───────────────────────────────────────────────────────────
    for xi in range(n_sub):
        ax.axvline(xi, color="#E0E0E0", lw=0.5, zorder=0)
    for yi in range(n_mod):
        ax.axhline(yi, color="#E0E0E0", lw=0.5, zorder=0)

    # ── size scale: votes → marker area ─────────────────────────────────────
    size_map = {1: 220, 2: 520, 3: 900}
    orbit_r  = 0.28   # radius of orbit sub-markers in data units

    # ── draw dots ────────────────────────────────────────────────────────────
    for (sub, model), metric_votes in vote_map.items():
        x = sub_pos[sub]
        y = mod_pos.get(model)
        if y is None:
            continue

        n_votes    = len(metric_votes)
        is_winner  = (model == winners[sub])
        col        = model_color(model)
        sz         = size_map[n_votes]

        # ── outer glow ring for winner ────────────────────────────────────
        if is_winner:
            ax.scatter(x, y, s=sz * 2.6, color=col, alpha=0.18,
                       zorder=2, linewidths=0)
            ax.scatter(x, y, s=sz * 1.55, color=col, alpha=0.30,
                       zorder=2, linewidths=0)

        # ── main dot ─────────────────────────────────────────────────────
        marker = "*" if is_winner else "o"
        ms     = sz * 1.6 if is_winner else sz   # stars need slightly more area
        ax.scatter(x, y, s=ms, color=col, marker=marker,
                   edgecolors="white" if not is_winner else "#222222",
                   linewidths=1.0 if not is_winner else 0.8,
                   zorder=4, alpha=0.95)

        # ── orbit sub-markers (which metrics voted) ───────────────────────
        for metric, angle_deg, mk in zip(METRICS, METRIC_ANGLES, METRIC_MARKERS):
            if metric in metric_votes:
                angle = np.radians(angle_deg)
                ox = x + orbit_r * np.cos(angle)
                oy = y + orbit_r * np.sin(angle)
                ax.scatter(ox, oy, s=38, marker=mk,
                           color="white", edgecolors="#444444",
                           linewidths=0.8, zorder=5, alpha=0.95)

    # ── axes labels ──────────────────────────────────────────────────────────
    ax.set_xticks(range(n_sub))
    ax.set_xticklabels(substrates, fontsize=8, fontweight="bold", rotation=30,
                       ha="right", rotation_mode="anchor")

    ax.set_yticks(range(n_mod))
    ax.set_yticklabels(MODEL_ORDER, fontsize=8)

    ax.set_xlabel("Substrate", fontsize=9, fontweight="bold", labelpad=6)
    ax.set_ylabel("Model  (simple \u2192 complex)", fontsize=9,
                  fontweight="bold", labelpad=6)
    ax.set_title("Consensus Model Selection",
                 fontsize=11, fontweight="bold", pad=10)

    # subtle shading alternating model rows
    for yi in range(n_mod):
        if yi % 2 == 0:
            ax.axhspan(yi - 0.5, yi + 0.5, color="#F0F0F0", zorder=0, alpha=0.6)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_linewidth(0.6)
        ax.spines[spine].set_color("#AAAAAA")

    # ── legends ──────────────────────────────────────────────────────────────
    # 1. model color swatches
    swatch_handles = [
        mpatches.Patch(facecolor=MODEL_COLORS[m], edgecolor="none",
                       label=m, alpha=0.90)
        for m in MODEL_ORDER
    ]
    leg1 = ax.legend(handles=swatch_handles,
                     title="Model", title_fontsize=7.5,
                     loc="upper left",
                     bbox_to_anchor=(1.01, 1.0),
                     fontsize=7, frameon=True, framealpha=1,
                     edgecolor="#CCCCCC",
                     handlelength=1.0, handletextpad=0.5,
                     borderpad=0.6)
    ax.add_artist(leg1)

    # 2. dot size = number of votes
    size_handles = [
        ax.scatter([], [], s=size_map[n], color="#888888",
                   marker="o", edgecolors="white", linewidths=0.8,
                   label=f"{n} metric{'s' if n > 1 else ''} voted", alpha=0.90)
        for n in [1, 2, 3]
    ]
    leg2 = ax.legend(handles=size_handles,
                     title="Votes (dot size)", title_fontsize=7.5,
                     loc="upper left",
                     bbox_to_anchor=(1.01, 0.52),
                     fontsize=7, frameon=True, framealpha=1,
                     edgecolor="#CCCCCC",
                     handletextpad=0.5, borderpad=0.6)
    ax.add_artist(leg2)

    # 3. shape + orbit markers
    shape_handles = [
        ax.scatter([], [], s=220, color="#888", marker="*",
                   edgecolors="#222", linewidths=0.8,
                   label="★  Consensus winner"),
        ax.scatter([], [], s=220, color="#888", marker="o",
                   edgecolors="white", linewidths=0.8,
                   label="●  Non-winning vote"),
    ]
    orbit_handles = [
        ax.scatter([], [], s=38, marker=mk, color="white",
                   edgecolors="#444", linewidths=0.8,
                   label=f"  {mk_label}")
        for mk, mk_label in zip(METRIC_MARKERS,
                                ["▲  AIC voted", "■  R² voted", "●  SSE voted"])
    ]
    leg3 = ax.legend(handles=shape_handles + orbit_handles,
                     title="Shape / orbit key", title_fontsize=7.5,
                     loc="upper left",
                     bbox_to_anchor=(1.01, 0.22),
                     fontsize=7, frameon=True, framealpha=1,
                     edgecolor="#CCCCCC",
                     handletextpad=0.5, borderpad=0.6)

    plt.tight_layout(rect=[0, 0, 0.78, 1])

    fig.savefig(out_pdf, format="pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(out_png, format="png", dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved -> {out_pdf}")
    print(f"Saved -> {out_png}")


if __name__ == "__main__":
    csv_in  = sys.argv[1] if len(sys.argv) > 1 else "master_results_all_models_LBFGS.csv"
    out_pdf = sys.argv[2] if len(sys.argv) > 2 else "consensus_publication.pdf"
    out_png = out_pdf.replace(".pdf", ".png")
    make_figure(csv_in, out_pdf, out_png)
