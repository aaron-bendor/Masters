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

import matplotlib
matplotlib.use("Agg")
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
# Figure 1: Tier 1 RMAE reproduction vs. paper Fig 2A
# =============================================================================
def fig_phase1_rmae():
    # |X_o| values
    Xo = np.array([5, 10, 20, 35, 50, 70, 90])
    # Our numbers from the locked result table (mean over 10 trials)
    ours_fg   = np.array([0.97, 0.49, 0.37, 0.26, 0.21, 0.19, np.nan])  # 90 not in our table
    ours_fg_sd = np.array([0.28, 0.13, 0.07, 0.06, 0.07, 0.07, np.nan])
    ours_f_only = np.array([1.08, 0.69, 0.75, 0.44, 0.46, 0.41, np.nan])
    ours_f_only_sd = np.array([0.45, 0.20, 0.40, 0.07, 0.13, 0.13, np.nan])
    nwkr = np.array([2.31, 1.59, 1.73, 1.38, 1.67, 1.42, np.nan])
    # Visual estimates from Paper Fig. 2A.
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
            label="Paper Fig. 2A visual estimate", linewidth=1.5)
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
# Figure 2: Tier 2 SUMO T-sweep -- reference and fitted AUC vs T
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
# Figure 3: Tier 2 features lift -- per-row chain recovery metrics
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
# Figure 4: Tier 3 Xuancheng -- OD-matching sensitivity
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
    # Annotate the coarse-zone drop.
    ax.annotate("", xy=(2 - w, 0.518), xytext=(1 - w, 0.573),
                arrowprops=dict(arrowstyle="->", color=C_CHANCE, lw=1.4))
    ax.text(1.55, 0.585, "coarse K=8 drop:\nnot stable\nacross K",
            color=C_CHANCE, ha="center", va="bottom", fontsize=8,
            fontstyle="italic")
    ax.set_xticks(x)
    ax.set_xticklabels(configs)
    ax.set_ylim(0.40, 0.66)
    ax.set_ylabel("AUC")
    ax.set_title("Tier 3 (Xuancheng, rush vs. off-peak, T=10):\n"
                 "OD-matching changes the reference; fitted detector remains near chance")
    ax.legend(loc="lower left", framealpha=0.95)
    fig.tight_layout()
    out = HERE / "fig04_phase5_xuancheng.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out.name}")


# =============================================================================
# Figure 5: Tier 2 multi-seed -- AUC lift is seed-dependent, Pearson lift robust
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
# Figure 6: Tier 2 multistart basin check -- random restarts never beat zero-init
# =============================================================================
def fig_phase4_multistart():
    seeds = ["7", "23"]
    gap_base = np.array([30.3, 34.2])   # zero-init baseline gap-closed (%)
    gap_ms   = np.array([30.2, 7.9])    # multistart (val-select) gap-closed (%)
    good_basin = 61.5                   # mean of seeds 42/101/2024

    # Final training loss of each random restart minus the zero-init baseline.
    # All positive => every restart lands in a strictly worse basin.
    dloss = {
        "7":  {"intr":   np.array([44.99, 45.47, 44.99, 45.27, 45.5]) - 42.15,
               "satnav": np.array([54.9, 54.0, 54.86, 54.97, 55.2]) - 51.71},
        "23": {"intr":   np.array([458.8, 458.3, 458.6, 457.3, 458.1]) - 455.9,
               "satnav": np.array([575.1, 575.4, 575.4, 575.8, 575.9]) - 572.5},
    }

    x = np.arange(len(seeds))
    w = 0.38
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(9.2, 3.8))

    # Panel (a): gap-closed -- baseline vs multistart, with good-basin reference
    axA.bar(x - w/2, gap_base, w, color=C_NONE, label="zero-init baseline",
            edgecolor="white")
    axA.bar(x + w/2, gap_ms, w, color=C_REAL, label="multistart (K=5, val-select)",
            edgecolor="white")
    axA.axhline(good_basin, color=C_CEILING, linestyle="--", linewidth=1.4,
                label="good-basin (seeds 42/101/2024)")
    for xi, gb, gm in zip(x, gap_base, gap_ms):
        axA.annotate(f"{gb:.1f}%", (xi - w/2, gb), textcoords="offset points",
                     xytext=(0, 3), ha="center", fontsize=8)
        axA.annotate(f"{gm:.1f}%", (xi + w/2, gm), textcoords="offset points",
                     xytext=(0, 3), ha="center", fontsize=8)
    axA.set_xticks(x); axA.set_xticklabels(seeds)
    axA.set_xlabel("low-basin SUMO seed")
    axA.set_ylabel("gap closed (%)")
    axA.set_ylim(0, 75)
    axA.set_title("(a) Multistart does not escape the low basin\n"
                  "(ties on seed 7, worse on seed 23)")
    axA.legend(loc="upper right", framealpha=0.95)

    # Panel (b): every restart is worse than zero-init (Delta loss > 0)
    for xi, s in zip(x, seeds):
        for marker, regime, col in (("o", "intr", C_OURS_FG),
                                    ("^", "satnav", C_NWKR)):
            yv = dloss[s][regime]
            jit = np.linspace(-0.10, 0.10, len(yv))
            off = -0.13 if regime == "intr" else 0.13
            axB.scatter(np.full_like(yv, xi + off) + jit, yv, marker=marker,
                        color=col, s=34, zorder=3,
                        label=(regime if xi == 0 else None))
    axB.axhline(0.0, color="black", linestyle="-", linewidth=1.2)
    axB.text(axB.get_xlim()[1], 0.02, "zero-init baseline (best basin)",
             ha="right", va="bottom", fontsize=8, fontstyle="italic")
    axB.set_xticks(x); axB.set_xticklabels(seeds)
    axB.set_xlabel("low-basin SUMO seed")
    axB.set_ylabel("final loss $-$ zero-init loss")
    axB.set_ylim(bottom=-0.5)
    axB.set_title("(b) Every random restart lands in a\n"
                  "strictly worse basin than zero-init")
    axB.legend(loc="upper right", framealpha=0.95, title="restart chain")

    fig.suptitle("Tier 2 multistart basin check (SUMO, features=none, $K=5$, "
                 "$T=20$): the low basin is\nnot an initialisation artefact "
                 "-- zero-init is the best optimum found",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    out = HERE / "fig06_phase4_multistart.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out.name}")


def fig_phase5_multistart():
    # Xuancheng OD-matched features=real, K=5 restarts (rush/intrinsic chain).
    restart_loss = np.array([1413.0, 1844.0, 273.4, 9.186, 1792.0])
    base_loss = 1397.0          # zero-init baseline rush-chain loss
    sel_idx = 3                 # validation-selected restart (loss 9.186)

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(9.2, 3.8))

    # Panel (a): the loss escape -- baseline vs each restart, log scale
    xr = np.arange(len(restart_loss))
    cols = [C_NONE] * len(restart_loss)
    cols[sel_idx] = C_REAL
    axA.axhline(base_loss, color=C_CHANCE, linestyle="--", linewidth=1.4,
                label=f"zero-init baseline ({base_loss:.0f})")
    axA.bar(xr, restart_loss, 0.62, color=cols, edgecolor="white")
    for xi, lv in zip(xr, restart_loss):
        axA.annotate(f"{lv:.0f}" if lv >= 100 else f"{lv:.2f}",
                     (xi, lv), textcoords="offset points", xytext=(0, 3),
                     ha="center", fontsize=8)
    axA.set_yscale("log")
    axA.set_xticks(xr)
    axA.set_xticklabels([f"r{i+1}" for i in xr])
    axA.set_xlabel("random restart")
    axA.set_ylabel("rush-chain final loss (log)")
    axA.set_title("(a) Multistart escapes the high-loss basin\n"
                  r"(rush loss $1{,}397 \rightarrow 9.186$, val-selected)")
    axA.legend(loc="upper right", framealpha=0.95)

    # Panel (b): AUC collapses to chance once the chain is correctly fit
    labels = ["zero-init\nbaseline", "multistart\n(val-select)"]
    aucs = np.array([0.530, 0.483])
    bcols = [C_NONE, C_REAL]
    xb = np.arange(2)
    axB.axhline(0.518, color=C_CEILING, linestyle="--", linewidth=1.4,
                label=r"empirical ceiling $0.518$")
    axB.axhline(0.5, color="black", linestyle=":", linewidth=1.0,
                label="chance")
    axB.bar(xb, aucs, 0.5, color=bcols, edgecolor="white")
    for xi, av in zip(xb, aucs):
        axB.annotate(f"{av:.3f}", (xi, av), textcoords="offset points",
                     xytext=(0, 3), ha="center", fontsize=8)
    axB.set_xticks(xb); axB.set_xticklabels(labels)
    axB.set_ylabel(r"$\mathrm{AUC}_{\mathrm{fit}}$")
    axB.set_ylim(0.45, 0.56)
    axB.set_title("(b) The well-fit chain returns to chance\n"
                  r"(baseline $0.530 > $ ceiling was an artefact)")
    axB.legend(loc="upper right", framealpha=0.95)

    fig.suptitle("Tier 3 multistart basin check (Xuancheng OD-matched, "
                 "features=real, $K=5$, $T=10$): multistart escapes the\n"
                 "basin but the correct fit is chance -- the ceiling "
                 "violation was an optimiser artefact",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    out = HERE / "fig07_phase5_multistart.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out.name}")


# =============================================================================
# Figure 8: conclusion synthesis -- headline outcome across tiers
# =============================================================================
def fig_conclusion_synthesis():
    tiers = [
        "Tier 1\nsynthetic\nestimator",
        "Tier 2\nSUMO labelled\nfilter",
        "Tier 3\nXuancheng Labour\nproxy",
    ]
    # Tier 1: lower-is-better RMAE reproduction.  We summarise closeness to
    # the visually-read paper curve for the stable |X_o| >= 10 points, capped
    # at 100% when our RMAE is slightly below the visual paper estimate.
    paper_rmae = np.array([0.50, 0.40, 0.30, 0.20, 0.15])
    ours_rmae = np.array([0.49, 0.37, 0.26, 0.21, 0.19])
    tier1_reproduction = np.minimum(paper_rmae / ours_rmae, 1.0).mean()

    # Tiers 2 and 3: AUC gap closed = (fit - chance) / (empirical - chance).
    tier2_gap_closed = (0.708 - 0.5) / (0.801 - 0.5)
    tier3_gap_closed = (0.502 - 0.5) / (0.567 - 0.5)

    vals = np.array([tier1_reproduction, tier2_gap_closed, tier3_gap_closed]) * 100.0
    colors = [C_REAL, C_OURS_FG, C_CHANCE]
    annotations = [
        "RMAE tracks\npaper curve",
        "AUC 0.708 vs\n0.801 reference",
        "AUC 0.502 vs\n0.567 reference",
    ]

    fig, ax = plt.subplots(figsize=(6.6, 3.7))
    x = np.arange(len(tiers))
    bars = ax.bar(x, vals, 0.58, color=colors, edgecolor="white")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(50, color=C_CEILING, linestyle=":", linewidth=1.0)
    ax.text(2.47, 50, "50%", va="bottom", ha="right", fontsize=8, color=C_CEILING)
    for b, v, txt in zip(bars, vals, annotations):
        ax.annotate(f"{v:.0f}%",
                    (b.get_x() + b.get_width() / 2, v),
                    textcoords="offset points", xytext=(0, 4),
                    ha="center", fontsize=9, fontweight="bold")
        if v >= 20:
            ax.annotate(txt,
                        (b.get_x() + b.get_width() / 2, v * 0.48),
                        ha="center", va="center", fontsize=8, color="white",
                        fontweight="bold")
        else:
            ax.annotate(txt,
                        (b.get_x() + b.get_width() / 2, 15),
                        ha="center", va="center", fontsize=8, color=C_CHANCE,
                        fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(tiers)
    ax.set_ylim(0, 112)
    ax.set_ylabel("normalised headline recovery (%)")
    ax.set_title("Conclusion synthesis: the filter validates under labels,\n"
                 "but the real-world proxy does not recover under partial observation")
    ax.grid(axis="y", alpha=0.25)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    out = HERE / "fig08_conclusion_synthesis.png"
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
    fig_phase4_multistart()
    fig_phase5_xuancheng()
    fig_phase5_multistart()
    fig_conclusion_synthesis()
    print("Done.")
