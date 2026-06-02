"""Time the sparse f+g fit on SUMO bigger-bbox seed 42.

Reference points (existing thesis §6.7):
  - Dense maxiter=1500 at |X_o|=601: 9.5 hours wall, no L-BFGS termination.
  - Empirical g_obs is 84-89% at floor (data-sparsity issue, independent of solver).

This script fits the intrinsic chain at small maxiter steps to characterise
per-iteration cost under sparse-LU, then attempts a longer fit if feasible.
Reports timing, final loss, convergence status, and AUC.

Usage:
  ../.venv/bin/python time_fg_sparse_sumo.py
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
os.chdir(HERE)

from sumo_to_phase3 import (
    build_adj, read_edge_counts, read_trajectories, empirical_chain,
    empirical_g,
)
from per_car_detector import fit_chain, _scores_for_chains
from congestion_filter import roc


def main(seed=42, T=20, gamma_fg=0.5, x_o_seed=13):
    print(f"=== Sparse f+g timing on SUMO bigger-bbox (seed={seed}, T={T}) ===")

    edges, adj_out, idx = build_adj("net.net.xml")
    n = len(edges)
    f_intr = read_edge_counts(f"edgedata.intr.big.seed{seed}.xml", idx)
    f_satnav = read_edge_counts(f"edgedata.satnav.big.seed{seed}.xml", idx)
    trajs_intr, _ = read_trajectories(f"vehroutes.intr.big.seed{seed}.xml", idx)
    trajs_satnav, _ = read_trajectories(f"vehroutes.satnav.big.seed{seed}.xml", idx)

    mean_len = float(np.mean([len(t) for t in trajs_intr + trajs_satnav]))
    beta = float(max(0.5, min(0.99, 1.0 - 1.0 / mean_len)))
    print(f"  n={n}  beta={beta:.3f}")

    # Same test/train split as score_seed_big.py
    long_intr = [t for t in trajs_intr if len(t) >= T + 1]
    long_satnav = [t for t in trajs_satnav if len(t) >= T + 1]
    rng_test = np.random.default_rng(42)
    rng_test.shuffle(long_intr)
    rng_test.shuffle(long_satnav)
    n_t = min(300, len(long_intr), len(long_satnav))
    test_intr_full = long_intr[:n_t]
    test_satnav_full = long_satnav[:n_t]
    test_intr = [t[: T + 1] for t in test_intr_full]
    test_satnav = [t[: T + 1] for t in test_satnav_full]
    test_ids = ({id(t) for t in test_intr_full},
                {id(t) for t in test_satnav_full})
    train_intr = [t for t in trajs_intr if id(t) not in test_ids[0]]
    train_satnav = [t for t in trajs_satnav if id(t) not in test_ids[1]]

    cands = np.where((f_intr > 0) & (f_satnav > 0))[0]
    n_obs = min(max(20, int(0.25 * n)), len(cands))
    rng_xo = np.random.default_rng(x_o_seed)
    X_o = np.sort(rng_xo.choice(cands, size=n_obs, replace=False))
    print(f"  |X_o|={len(X_o)}  ({100 * n_obs / n:.1f}% of n)")

    g_intr_emp = empirical_g(train_intr, X_o, beta, floor=1e-3)
    g_satnav_emp = empirical_g(train_satnav, X_o, beta, floor=1e-3)
    frac_floor_i = float((g_intr_emp <= 1e-3 + 1e-12).mean())
    frac_floor_s = float((g_satnav_emp <= 1e-3 + 1e-12).mean())
    print(f"  g_intr   {100*frac_floor_i:.0f}% at floor")
    print(f"  g_satnav {100*frac_floor_s:.0f}% at floor")

    # Empirical 1-step ceiling (for AUC reference)
    PT_intr_emp = empirical_chain(train_intr, adj_out)
    PT_satnav_emp = empirical_chain(train_satnav, adj_out)
    test_all = test_intr + test_satnav
    labels = np.concatenate([np.zeros(n_t), np.ones(n_t)])
    _, _, auc_emp = roc(_scores_for_chains(test_all, PT_intr_emp,
                                            PT_satnav_emp), labels)
    print(f"\n  empirical 1-step AUC = {auc_emp:.3f}\n")

    # Sparse f+g fit at maxiter=10 first, to characterise per-iter cost.
    for maxit in (10, 100, 1500):
        print(f"  --- intrinsic f+g fit, sparse, maxiter={maxit}, "
              f"gamma={gamma_fg} ---")
        t0 = time.perf_counter()
        _, PT_intr_fg, _ = fit_chain(adj_out, beta, X_o, f_intr[X_o],
                                     g_intr_emp, gamma=gamma_fg,
                                     maxiter=maxit, solver="sparse")
        t_intr = time.perf_counter() - t0
        print(f"    intr fit wall = {t_intr:.1f}s")

        t0 = time.perf_counter()
        _, PT_satnav_fg, _ = fit_chain(adj_out, beta, X_o, f_satnav[X_o],
                                       g_satnav_emp, gamma=gamma_fg,
                                       maxiter=maxit, solver="sparse")
        t_satnav = time.perf_counter() - t0
        print(f"    satnav fit wall = {t_satnav:.1f}s")

        scores_fg = _scores_for_chains(test_all, PT_intr_fg, PT_satnav_fg)
        _, _, auc_fg = roc(scores_fg, labels)
        print(f"    auc_fg = {auc_fg:.3f}  (auc_emp = {auc_emp:.3f})\n")

        if t_intr + t_satnav > 3600:  # 1 hour total budget
            print("    Wall time exceeds 1 hr budget; stopping sweep.")
            break


if __name__ == "__main__":
    main()
