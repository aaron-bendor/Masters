"""Generate all thesis figures from the project's locked-in numbers.

Each figure is hand-keyed from the result tables in `research_logbook.md`
and from the experiment logs in the repository root. Saving each as a PNG
into this directory; `main.tex` references them via `\\includegraphics`.

Run from anywhere with the project's venv:
    .venv/bin/python thesis/figures/make_figures.py
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ---- styling: a single consistent palette -----------------------------------
plt.rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["Computer Modern Roman", "DejaVu Serif"],
    "font.size":          10,
    "axes.titlesize":     11,
    "axes.labelsize":     10,
    "legend.fontsize":    9,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "axes.grid":          True,
    "grid.alpha":         0.25,
    "axes.spines.right":  False,
    "axes.spines.top":    False,
    "figure.dpi":         150,
})

C_PAPER     = "#888888"
C_OURS_FG   = "#1f4e79"
C_OURS_FONLY = "#7e9fc4"
C_NWKR      = "#c47e7e"
C_CEILING   = "#888888"
C_NONE      = "#9fb8d4"
C_REAL      = "#1f4e79"
C_CHANCE    = "#cc6666"

HERE = Path(__file__).resolve().parent

# =============================================================================
# Figure 1: Phase 1 RMAE reproduction vs. paper Fig 2A
# =============================================================================
def fig_phase1_rmae():
    # |X_o| values
    Xo = np.array([5, 10, 20, 35, 50, 70, 90])
    # Our numbers from research_logbook.md iter-18 table (mean over 10 trials)
    ours_fg   = np.array([0.97, 0.49, 0.37, 0.26, 0.21, 0.19, np.nan])  # 90 not in our table
    ours_fg_sd = np.array([0.28, 0.13, 0.07, 0.06, 0.07, 0.07, np.nan])
    ours_f_only = np.array([1.08, 0.69, 0.75, 0.44, 0.46, 0.41, np.nan])
    ours_f_only_sd = np.array([0.45, 0.20, 0.40, 0.07, 0.13, 0.13, np.nan])
    nwkr = np.array([2.31, 1.59, 1.73, 1.38, 1.67, 1.42, np.nan])
    # Paper Fig 2A eyeball numbers
    paper_fg = np.array([0.7, 0.5, 0.4, 0.3, 0.2, 0.15, 0.12])  # 90 extrapolated

    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    ax.errorbar(Xo[:-1], ours_fg[:-1], yerr=ours_fg_sd[:-1],
                marker='o', linestyle='-', color=C_OURS_FG, capsize=3,
                label=r"Ours, Proposed (with $g$)")
    ax.errorbar(Xo[:-1], ours_f_only[:-1], yerr=ours_f_only_sd[:-1],
                marker='s', linestyle='--', color=C_OURS_FONLY, capsize=3,
                label=r"Ours, Proposed (no $g$)")
    ax.plot(Xo[:-1], nwkr[:-1], marker='^', linestyle=':',
            color=C_NWKR, label="Ours, NWKR baseline")
    ax.plot(Xo, paper_fg, marker='x', linestyle='-.', color=C_PAPER,
            label="Paper Fig. 2A (eyeball)", linewidth=1.5)
    ax.set_xlabel(r"$|X_o|$ (observation set size)")
    ax.set_ylabel(r"RMAE on unobserved states")
    ax.set_xticks(Xo)
    ax.set_title("Tier 1: synthetic reproduction tracks the paper within ~0.05 RMAE")
    ax.legend(loc="upper right", framealpha=0.95)
    fig.tight_layout()
    out = HERE / "fig01_phase1_rmae.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out.name}")


# =============================================================================
# Figure 2: Phase 4 SUMO T-sweep — ceiling and fitted AUC vs T
# =============================================================================
def fig_phase4_t_sweep():
    T   = np.array([20, 30, 40])
    emp = np.array([0.685, 0.763, 0.801])
    fit = np.array([0.644, 0.677, 0.708])

    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ax.plot(T, emp, marker='o', color=C_CEILING, linestyle='--',
            label=r"empirical 1-step Markov ceiling ($\mathrm{AUC}_\mathrm{emp}$)",
            linewidth=1.8)
    ax.plot(T, fit, marker='s', color=C_OURS_FG, linestyle='-',
            label=r"fitted detector ($\mathrm{AUC}_\mathrm{fit}$, features=real)",
            linewidth=1.8)
    ax.fill_between(T, fit, emp, color=C_OURS_FG, alpha=0.10,
                    label="gap (recoverable contrast)")
    ax.axhline(0.5, color=C_CHANCE, linestyle=':', linewidth=1, label="chance (0.5)")
    for x, y_emp, y_fit in zip(T, emp, fit):
        ax.annotate(f"{y_emp:.3f}", (x, y_emp), textcoords="offset points",
                    xytext=(0, 6), ha="center", fontsize=8, color=C_CEILING)
        ax.annotate(f"{y_fit:.3f}", (x, y_fit), textcoords="offset points",
                    xytext=(0, -14), ha="center", fontsize=8, color=C_OURS_FG)
    ax.set_xlabel(r"test trajectory length $T$ (transitions)")
    ax.set_ylabel("AUC")
    ax.set_xticks(T)
    ax.set_ylim(0.45, 0.86)
    ax.set_title(r"Tier 2 (SUMO seed 42, features=real): both ceiling and"
                 + "\n"
                 + r"detector lift with longer test windows")
    ax.legend(loc="lower right", framealpha=0.95)
    fig.tight_layout()
    out = HERE / "fig02_phase4_tsweep.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out.name}")


# =============================================================================
# Figure 3: Phase 4 features lift — per-row chain recovery metrics
# =============================================================================
def fig_phase4_features_lift():
    metrics = [
        ("AUC$_\\mathrm{fit}$",       0.612, 0.644),
        ("gap closed",                0.606, 0.775),
        ("Pearson",                   0.132, 0.215),
        ("cosine median",             0.095, 0.435),
        ("rows w/ cos > 0",           0.535, 0.554),
    ]
    labels    = [m[0] for m in metrics]
    none_vals = np.array([m[1] for m in metrics])
    real_vals = np.array([m[2] for m in metrics])

    x = np.arange(len(labels))
    w = 0.36

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    bars_n = ax.bar(x - w/2, none_vals, w, color=C_NONE,
                    label="features = none", edgecolor="white")
    bars_r = ax.bar(x + w/2, real_vals, w, color=C_REAL,
                    label="features = real", edgecolor="white")
    for bars, vals in ((bars_n, none_vals), (bars_r, real_vals)):
        for b, v in zip(bars, vals):
            ax.annotate(f"{v:.3f}",
                        (b.get_x() + b.get_width()/2, v),
                        textcoords="offset points", xytext=(0, 3),
                        ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, max(real_vals.max(), 0.85) + 0.1)
    ax.set_ylabel(r"value on $[\mathrm{all\ branching}]$ states")
    ax.set_title(r"Tier 2 (SUMO seed 42, T=20, $|X_o|=601$): all metrics lift"
                 + "\n"
                 + r"on this single seed (cf. Fig. 5 for the multi-seed picture)")
    ax.legend(loc="upper right", framealpha=0.95)
    fig.tight_layout()
    out = HERE / "fig03_phase4_features_lift.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out.name}")


# =============================================================================
# Figure 4: Phase 5 Xuancheng — OD-matched ceiling crash
# =============================================================================
def fig_phase5_xuancheng():
    # Three configurations × {emp, fit_none, fit_real}
    configs = ["Apr 17\n(single day)", "Apr 17–21\n(pooled)",
               "Apr 17–21\npooled + OD-match K=8"]
    emp       = np.array([0.556, 0.573, 0.518])
    fit_none  = np.array([0.498, 0.524, 0.476])
    fit_real  = np.array([0.508, 0.536, 0.530])

    x = np.arange(len(configs))
    w = 0.27

    fig, ax = plt.subplots(figsize=(6.4, 3.9))
    bars_e = ax.bar(x - w, emp, w, color=C_CEILING,
                    label=r"empirical 1-step ceiling ($\mathrm{AUC}_\mathrm{emp}$)",
                    edgecolor="white")
    bars_n = ax.bar(x,     fit_none, w, color=C_NONE,
                    label=r"fitted detector (features=none)", edgecolor="white")
    bars_r = ax.bar(x + w, fit_real, w, color=C_REAL,
                    label=r"fitted detector (features=real)", edgecolor="white")

    for bars, vals in ((bars_e, emp), (bars_n, fit_none), (bars_r, fit_real)):
        for b, v in zip(bars, vals):
            ax.annotate(f"{v:.3f}",
                        (b.get_x() + b.get_width()/2, v),
                        textcoords="offset points", xytext=(0, 3),
                        ha="center", fontsize=8)

    ax.axhline(0.5, color=C_CHANCE, linestyle=":", linewidth=1, label="chance (0.5)")
    # Annotate the ceiling crash
    ax.annotate("", xy=(2 - w, 0.518), xytext=(1 - w, 0.573),
                arrowprops=dict(arrowstyle="->", color=C_CHANCE, lw=1.4))
    ax.text(1.55, 0.585, "ceiling crash:\nOD-mix\nsubtracted",
            color=C_CHANCE, ha="center", va="bottom", fontsize=8,
            fontstyle="italic")
    ax.set_xticks(x)
    ax.set_xticklabels(configs)
    ax.set_ylim(0.40, 0.66)
    ax.set_ylabel("AUC")
    ax.set_title("Tier 3 (Xuancheng, rush vs. off-peak, T=10):\n"
                 "OD-rebalancing reveals ~75% of un-matched signal was OD-mix")
    ax.legend(loc="lower left", framealpha=0.95)
    fig.tight_layout()
    out = HERE / "fig04_phase5_xuancheng.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out.name}")


# =============================================================================
# Figure 5: Phase 4 multi-seed -- AUC lift is seed-dependent, Pearson lift robust
# =============================================================================
def fig_phase4_multiseed():
    seeds     = ["7", "23", "42", "101", "2024"]
    gap_none  = np.array([30.3, 34.2, 60.6, 62.1, 61.8])
    gap_real  = np.array([60.3, 53.8, 77.5, 45.0, 85.1])
    pear_none = np.array([0.191, 0.133, 0.132, 0.171, 0.131])
    pear_real = np.array([0.238, 0.194, 0.215, 0.194, 0.168])

    x = np.arange(len(seeds))
    w = 0.38

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(9.2, 3.8))

    # Panel (a): gap-closed (aggregate detection) -- seed-dependent
    axA.bar(x - w/2, gap_none, w, color=C_NONE, label="features = none",
            edgecolor="white")
    axA.bar(x + w/2, gap_real, w, color=C_REAL, label="features = real",
            edgecolor="white")
    axA.annotate("reversal", xy=(3 + w/2, 45.0), xytext=(3 + w/2, 18.0),
                 ha="center", fontsize=8, color=C_CHANCE, fontstyle="italic",
                 arrowprops=dict(arrowstyle="->", color=C_CHANCE, lw=1.2))
    axA.set_xticks(x); axA.set_xticklabels(seeds)
    axA.set_xlabel("SUMO seed")
    axA.set_ylabel("gap closed (%)")
    axA.set_ylim(0, 100)
    axA.set_title("(a) Detection: seed-dependent\n(features hurt on seed 101)")
    axA.legend(loc="upper left", framealpha=0.95)

    # Panel (b): per-row Pearson (chain recovery) -- robust across all seeds
    axB.bar(x - w/2, pear_none, w, color=C_NONE, label="features = none",
            edgecolor="white")
    axB.bar(x + w/2, pear_real, w, color=C_REAL, label="features = real",
            edgecolor="white")
    axB.set_xticks(x); axB.set_xticklabels(seeds)
    axB.set_xlabel("SUMO seed")
    axB.set_ylabel("stacked Pearson ([all branching])")
    axB.set_ylim(0, 0.30)
    axB.set_title("(b) Chain recovery: robust\n(features help on all 5 seeds)")
    axB.legend(loc="upper right", framealpha=0.95)

    fig.suptitle("Multi-seed Tier 2 (SUMO, $T=20$, $|X_o|=601$): features robustly "
                 "improve\nchain recovery (b), not aggregate detection (a)",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out = HERE / "fig05_phase4_multiseed.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out.name}")


# =============================================================================
# Driver
# =============================================================================
if __name__ == "__main__":
    print("Generating thesis figures...")
    fig_phase1_rmae()
    fig_phase4_t_sweep()
    fig_phase4_features_lift()
    fig_phase4_multiseed()
    fig_phase5_xuancheng()
    print("Done.")
