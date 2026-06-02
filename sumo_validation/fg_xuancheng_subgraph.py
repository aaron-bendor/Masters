"""Test sparse f+g on a dense Xuancheng subgraph.

Motivation: separate the two limitations conflated in the old §6.7 claim:
  (a) framework/solver issue — dense LU is O(n^3) and unbearable at scale.
      Addressed by the sparse-LU implementation in morimura.py.
  (b) data issue — empirical g_obs is 84-89% at floor across X_o because
      most (i, j) pairs in X_o are never traversed in the observation window.

This script picks a dense central subgraph of Xuancheng (one K-means zone
out of K_macro zones; configurable, default K_macro=8 — uses the same
clustering machinery as the OD-rebalancing), filters trajectories to those
that stay within the zone, builds f and empirical g on that subgraph,
and fits f-only vs f+g chains under the new sparse solver. Two possible
readings:

  - f+g lifts AUC on the dense subgraph but not on full Xuancheng: data
    sparsity in g is the real limit; framework works when g is observable.
  - f+g does not lift AUC even on the dense subgraph: the parametric form
    does not carry routing-mechanism information beyond what f alone
    carries here.

Either is a publishable finding.

Usage:
  ../.venv/bin/python fg_xuancheng_subgraph.py --K_macro 8 --pick_zone 0
"""
import argparse
import os
import sys
import time
import datetime

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
_REAL_DATA = os.path.join(os.path.dirname(HERE), "real_data")
sys.path.insert(0, _REAL_DATA)
os.chdir(HERE)

from sumo_to_phase3 import empirical_chain, empirical_g
from per_car_detector import fit_chain, _scores_for_chains
from congestion_filter import roc
from load_xuancheng import (  # noqa: E402
    load_xuancheng_regime, day_spec, split_rush_vs_offpeak,
    compute_edge_endpoint_zones, NET_PATH_DEFAULT,
)


def restrict_to_zone(edges_full, adj_out_full, from_zones, to_zones,
                     zone_id):
    """Pick the edges whose BOTH endpoints lie in ``zone_id``, then restrict
    further to the largest strongly-connected component of the resulting
    subgraph (so the Inverter's softmax forward pass never sees a sink row).

    Returns:
      keep_mask  -- bool array of length len(edges_full)
      idx_map    -- old-index -> new-index dict (only for kept edges)
      adj_out_sub -- adjacency in the new index frame
    """
    keep = (from_zones == zone_id) & (to_zones == zone_id)
    keep_idx = np.where(keep)[0]
    old_to_new0 = {int(o): i for i, o in enumerate(keep_idx)}
    adj_zone = []
    for old_i in keep_idx:
        adj_zone.append([old_to_new0[old_j]
                         for old_j in adj_out_full[old_i]
                         if old_j in old_to_new0])
    # Largest SCC of the zone subgraph
    import scipy.sparse as sp
    import scipy.sparse.csgraph as csgraph
    n_zone = len(keep_idx)
    rows = [i for i, nbrs in enumerate(adj_zone) for _ in nbrs]
    cols = [j for nbrs in adj_zone for j in nbrs]
    A = sp.csr_matrix((np.ones(len(rows)), (rows, cols)),
                      shape=(n_zone, n_zone))
    n_comp, labels = csgraph.connected_components(A, connection="strong")
    sizes = np.bincount(labels)
    largest = int(np.argmax(sizes))
    in_scc = labels == largest
    print(f"  zone {zone_id}: zone edges={n_zone}, SCC count={n_comp}, "
          f"largest SCC size={int(sizes[largest])}")
    # Map zone-index -> final-index
    scc_idx = np.where(in_scc)[0]
    zone_to_scc = {int(z): i for i, z in enumerate(scc_idx)}
    # Build adjacency in the final index frame
    adj_out_sub = []
    for old_zone_i in scc_idx:
        new_nbrs = []
        for old_zone_j in adj_zone[old_zone_i]:
            if old_zone_j in zone_to_scc:
                new_nbrs.append(zone_to_scc[old_zone_j])
        adj_out_sub.append(sorted(new_nbrs))
    # old_to_new in the loader's original index frame -> final
    old_to_new = {int(keep_idx[z]): zone_to_scc[int(z)] for z in scc_idx}
    return keep, old_to_new, adj_out_sub


def filter_trajectories_to_subgraph(trajs, old_to_new):
    """Keep only trajectories that stay entirely within the subgraph.
    Re-index each kept trajectory to the new index frame."""
    out = []
    n_orig = len(trajs)
    for t in trajs:
        if all(int(e) in old_to_new for e in t):
            out.append(np.array([old_to_new[int(e)] for e in t],
                                dtype=np.int64))
    print(f"    kept {len(out):,}/{n_orig:,} trajectories "
          f"({100 * len(out) / max(n_orig, 1):.1f}%)")
    return out


def main(K_macro=8, pick_zone=0, day="2023-04-17", T=10, gamma_fg=0.5,
         x_o_frac=0.25, x_o_seed=13, maxiter=300):
    print(f"=== sparse f+g on Xuancheng subgraph (K_macro={K_macro}, "
          f"pick_zone={pick_zone}, day={day}, T={T}) ===")

    date_strs = [s.strip() for s in day.split(",") if s.strip()]
    dates = [datetime.date.fromisoformat(s) for s in date_strs]

    edges, adj_out, idx, trajs_intr, trajs_satnav, f_intr, f_satnav = (
        load_xuancheng_regime(
            net_path=NET_PATH_DEFAULT,
            day_specs=[day_spec(d) for d in dates],
            regime_split=lambda st, d: split_rush_vs_offpeak(st["start_time"]),
            label_a="rush", label_b="offpeak",
            min_segment_len=5, verbose=True,
        )
    )
    n_full = len(edges)
    print(f"\n  full graph: n={n_full}  |E|_dir = "
          f"{sum(len(ns) for ns in adj_out):,}")
    print(f"  trajs: rush={len(trajs_intr):,}  offpeak={len(trajs_satnav):,}")

    from_zones, to_zones = compute_edge_endpoint_zones(
        NET_PATH_DEFAULT, edges, k_zones=K_macro, random_state=42)

    # Pick zone: by default the first (zone 0). Surface zone sizes so the
    # user can pick the densest.
    zone_sizes = np.bincount(from_zones, minlength=K_macro)
    print(f"  zone sizes (by edge from-endpoint): {zone_sizes.tolist()}")

    keep, old_to_new, adj_out_sub = restrict_to_zone(
        edges, adj_out, from_zones, to_zones, pick_zone)
    n_sub = int(keep.sum())
    E_sub = sum(len(ns) for ns in adj_out_sub)
    print(f"\n  subgraph (zone {pick_zone}): n={n_sub}  |E|_dir = {E_sub}")

    if n_sub < 50:
        print(f"  zone {pick_zone} too small (n={n_sub}); pick a denser zone.")
        sys.exit(1)

    # Filter trajectories
    print("  filtering trajectories (rush) ...")
    trajs_intr_sub = filter_trajectories_to_subgraph(trajs_intr, old_to_new)
    print("  filtering trajectories (offpeak) ...")
    trajs_satnav_sub = filter_trajectories_to_subgraph(trajs_satnav, old_to_new)

    if len(trajs_intr_sub) < 200 or len(trajs_satnav_sub) < 200:
        print("  not enough subgraph-internal trajectories; bail.")
        sys.exit(1)

    # Build f on the subgraph
    f_intr_sub = np.zeros(n_sub, dtype=int)
    f_satnav_sub = np.zeros(n_sub, dtype=int)
    for t in trajs_intr_sub:
        for e in t:
            f_intr_sub[e] += 1
    for t in trajs_satnav_sub:
        for e in t:
            f_satnav_sub[e] += 1

    # Test/train split on subgraph trajectories
    long_intr = [t for t in trajs_intr_sub if len(t) >= T + 1]
    long_satnav = [t for t in trajs_satnav_sub if len(t) >= T + 1]
    print(f"  long-enough subgraph trajectories (>= T+1={T+1}): "
          f"rush={len(long_intr)}  offpeak={len(long_satnav)}")
    if len(long_intr) < 50 or len(long_satnav) < 50:
        print("  too few long subgraph trajectories; consider lowering T.")

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
    train_intr = [t for t in trajs_intr_sub if id(t) not in test_ids[0]]
    train_satnav = [t for t in trajs_satnav_sub if id(t) not in test_ids[1]]

    mean_len = float(np.mean([len(t) for t in trajs_intr_sub + trajs_satnav_sub]))
    beta = float(max(0.5, min(0.99, 1.0 - 1.0 / mean_len)))
    print(f"  beta (subgraph) = {beta:.3f}  N_test = {n_t}/class")

    # X_o on subgraph
    cands = np.where((f_intr_sub > 0) & (f_satnav_sub > 0))[0]
    n_obs = min(max(20, int(x_o_frac * n_sub)), len(cands))
    rng_xo = np.random.default_rng(x_o_seed)
    X_o = np.sort(rng_xo.choice(cands, size=n_obs, replace=False))
    print(f"  |X_o| = {len(X_o)}  ({100 * n_obs / n_sub:.1f}% of subgraph)")

    # Empirical g (key: floor fraction matters for the data-sparsity claim)
    g_intr = empirical_g(train_intr, X_o, beta, floor=1e-3)
    g_satnav = empirical_g(train_satnav, X_o, beta, floor=1e-3)
    frac_floor_i = float((g_intr <= 1e-3 + 1e-12).mean())
    frac_floor_s = float((g_satnav <= 1e-3 + 1e-12).mean())
    print(f"  g_rush    {100*frac_floor_i:.1f}% at floor")
    print(f"  g_offpeak {100*frac_floor_s:.1f}% at floor")

    # Empirical 1-step ceiling
    PT_intr_emp = empirical_chain(train_intr, adj_out_sub)
    PT_satnav_emp = empirical_chain(train_satnav, adj_out_sub)
    test_all = test_intr + test_satnav
    labels = np.concatenate([np.zeros(n_t), np.ones(n_t)])
    _, _, auc_emp = roc(_scores_for_chains(test_all, PT_intr_emp,
                                            PT_satnav_emp), labels)
    print(f"\n  empirical 1-step AUC on subgraph = {auc_emp:.3f}\n")

    # f-only fit
    print(f"  --- f-only fit, sparse, maxiter={maxiter} ---")
    t0 = time.perf_counter()
    _, PT_intr_f, _ = fit_chain(adj_out_sub, beta, X_o, f_intr_sub[X_o],
                                maxiter=maxiter, solver="sparse")
    _, PT_satnav_f, _ = fit_chain(adj_out_sub, beta, X_o,
                                  f_satnav_sub[X_o], maxiter=maxiter,
                                  solver="sparse")
    t_f = time.perf_counter() - t0
    scores_f = _scores_for_chains(test_all, PT_intr_f, PT_satnav_f)
    _, _, auc_f = roc(scores_f, labels)
    print(f"    f-only wall = {t_f:.1f}s  auc_f = {auc_f:.3f}\n")

    # f+g fit
    print(f"  --- f+g fit, sparse, gamma={gamma_fg}, maxiter={maxiter} ---")
    t0 = time.perf_counter()
    _, PT_intr_fg, _ = fit_chain(adj_out_sub, beta, X_o, f_intr_sub[X_o],
                                 g_intr, gamma=gamma_fg, maxiter=maxiter,
                                 solver="sparse")
    _, PT_satnav_fg, _ = fit_chain(adj_out_sub, beta, X_o,
                                   f_satnav_sub[X_o], g_satnav,
                                   gamma=gamma_fg, maxiter=maxiter,
                                   solver="sparse")
    t_fg = time.perf_counter() - t0
    scores_fg = _scores_for_chains(test_all, PT_intr_fg, PT_satnav_fg)
    _, _, auc_fg = roc(scores_fg, labels)
    print(f"    f+g wall    = {t_fg:.1f}s  auc_fg = {auc_fg:.3f}\n")

    delta = auc_fg - auc_f
    print(f"RESULT  subgraph_zone={pick_zone}  n_sub={n_sub}  "
          f"|X_o|={len(X_o)}  N_test={n_t}  "
          f"g_floor_rush={100*frac_floor_i:.1f}%  "
          f"g_floor_off={100*frac_floor_s:.1f}%  "
          f"auc_emp={auc_emp:.3f}  auc_f={auc_f:.3f}  auc_fg={auc_fg:.3f}  "
          f"delta_fg={delta:+.3f}  t_f={t_f:.1f}s  t_fg={t_fg:.1f}s")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--K_macro", type=int, default=8)
    p.add_argument("--pick_zone", type=int, default=0)
    p.add_argument("--day", type=str, default="2023-04-17")
    p.add_argument("--T", type=int, default=10)
    p.add_argument("--gamma_fg", type=float, default=0.5)
    p.add_argument("--x_o_frac", type=float, default=0.25)
    p.add_argument("--x_o_seed", type=int, default=13)
    p.add_argument("--maxiter", type=int, default=300)
    a = p.parse_args()
    main(K_macro=a.K_macro, pick_zone=a.pick_zone, day=a.day, T=a.T,
         gamma_fg=a.gamma_fg, x_o_frac=a.x_o_frac, x_o_seed=a.x_o_seed,
         maxiter=a.maxiter)
