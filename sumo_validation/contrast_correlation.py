"""Contrast-correlation diagnostic for the LR detector.

The detector classifies trajectories using
  Lambda(tau) = sum_t [log PT_sat_h(y|x) - log PT_intr_h(y|x)],
so the only object that matters is the *contrast*
  Delta_fit(x, :) = PT_sat_h(x, :) - PT_intr_h(x, :).
If Delta_fit doesn't track the contrast we'd see under full observation
  Delta_emp(x, :) = PT_sat_emp(x, :) - PT_intr_emp(x, :),
then any AUC the detector reports is coming from the wrong place.

This script replicates score_seed_big.py's fit, then computes:
  - per-row cosine similarity between Delta_fit and Delta_emp at branching
    states (out_deg > 1), restricted to legal out-neighbours.
  - stacked Pearson and stacked cosine over all legal off-diagonal entries.
  - the same diagnostics restricted to branching states in X_o (states the
    fit directly observed) vs. all branching states (generalisation).

Usage:
  ../.venv/bin/python contrast_correlation.py --seed 23
  for S in 23 7 42 101 2024; do ../.venv/bin/python contrast_correlation.py --seed $S; done
"""
import argparse, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.chdir(HERE)

from sumo_to_phase3 import (
    build_adj, read_edge_counts, read_trajectories, empirical_chain,
)
from per_car_detector import fit_chain, _scores_for_chains
from congestion_filter import roc


def contrast_rows(PT_a_h, PT_b_h, PT_a_emp, PT_b_emp, adj_out, states):
    """Return (delta_fit, delta_emp) stacked over legal (x -> y) pairs for
    each state x in `states` that is a branching state. Each is a 1-D
    vector of length sum(out_deg(x)) over the included states."""
    fit_parts, emp_parts = [], []
    n_branching_included = 0
    for x in states:
        nbrs = adj_out[x]
        if len(nbrs) <= 1:
            continue
        nbrs = np.array(nbrs, dtype=int)
        d_fit = PT_b_h[x, nbrs] - PT_a_h[x, nbrs]
        d_emp = PT_b_emp[x, nbrs] - PT_a_emp[x, nbrs]
        fit_parts.append(d_fit)
        emp_parts.append(d_emp)
        n_branching_included += 1
    if not fit_parts:
        return np.array([]), np.array([]), 0
    return (np.concatenate(fit_parts),
            np.concatenate(emp_parts),
            n_branching_included)


def per_row_cosines(PT_a_h, PT_b_h, PT_a_emp, PT_b_emp, adj_out, states,
                    eps=1e-12):
    """Cosine similarity of Delta_fit(x, :) vs Delta_emp(x, :) for each
    branching state x in `states`. Returns array of cosines and the matching
    state ids."""
    cosines, kept = [], []
    for x in states:
        nbrs = adj_out[x]
        if len(nbrs) <= 1:
            continue
        nbrs = np.array(nbrs, dtype=int)
        d_fit = PT_b_h[x, nbrs] - PT_a_h[x, nbrs]
        d_emp = PT_b_emp[x, nbrs] - PT_a_emp[x, nbrs]
        nf, ne = np.linalg.norm(d_fit), np.linalg.norm(d_emp)
        if nf < eps or ne < eps:
            continue
        cosines.append(float(d_fit @ d_emp / (nf * ne)))
        kept.append(int(x))
    return np.array(cosines), np.array(kept, dtype=int)


def pearson(a, b, eps=1e-12):
    if a.size < 2:
        return float("nan")
    a = a - a.mean()
    b = b - b.mean()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < eps or nb < eps:
        return float("nan")
    return float(a @ b / (na * nb))


def cosine(a, b, eps=1e-12):
    if a.size == 0:
        return float("nan")
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < eps or nb < eps:
        return float("nan")
    return float(a @ b / (na * nb))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--x_o_seed", type=int, default=13)
    p.add_argument("--T", type=int, default=20)
    p.add_argument("--maxiter", type=int, default=1500)
    args = p.parse_args()

    S = args.seed
    print(f"=== contrast-correlation diagnostic (big bbox, seed={S}) ===")

    edges, adj_out, idx = build_adj("net.net.xml")
    n = len(edges)
    out_deg = np.array([len(a) for a in adj_out])
    f_intr = read_edge_counts(f"edgedata.intr.big.seed{S}.xml", idx)
    f_satnav = read_edge_counts(f"edgedata.satnav.big.seed{S}.xml", idx)
    trajs_intr, _ = read_trajectories(f"vehroutes.intr.big.seed{S}.xml", idx)
    trajs_satnav, _ = read_trajectories(f"vehroutes.satnav.big.seed{S}.xml", idx)

    mean_len = float(np.mean([len(t) for t in trajs_intr + trajs_satnav]))
    beta = float(max(0.5, min(0.99, 1.0 - 1.0 / mean_len)))

    T = args.T
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
    test_ids = ({id(t) for t in test_intr_full}, {id(t) for t in test_satnav_full})
    train_intr = [t for t in trajs_intr if id(t) not in test_ids[0]]
    train_satnav = [t for t in trajs_satnav if id(t) not in test_ids[1]]

    cands = np.where((f_intr > 0) & (f_satnav > 0))[0]
    n_obs = min(max(20, int(0.25 * n)), len(cands))
    rng_xo = np.random.default_rng(args.x_o_seed)
    X_o = np.sort(rng_xo.choice(cands, size=n_obs, replace=False))

    print(f"  n={n}  |X_o|={len(X_o)}  beta={beta:.3f}  T={T}  N_test={n_t}")

    print("  fitting chains ...")
    _, PT_intr_h, _ = fit_chain(adj_out, beta, X_o, f_intr[X_o], maxiter=args.maxiter)
    _, PT_satnav_h, _ = fit_chain(adj_out, beta, X_o, f_satnav[X_o], maxiter=args.maxiter)
    PT_intr_emp = empirical_chain(train_intr, adj_out)
    PT_satnav_emp = empirical_chain(train_satnav, adj_out)

    test_all = test_intr + test_satnav
    labels = np.concatenate([np.zeros(n_t), np.ones(n_t)])
    _, _, auc_fit = roc(_scores_for_chains(test_all, PT_intr_h, PT_satnav_h), labels)
    _, _, auc_emp = roc(_scores_for_chains(test_all, PT_intr_emp, PT_satnav_emp), labels)
    gap = (auc_fit - 0.5) / (auc_emp - 0.5) if auc_emp > 0.5 else float("nan")

    print(f"\n  AUC reference: empirical={auc_emp:.3f}  fitted_f={auc_fit:.3f}  "
          f"gap_closed={100*gap:5.1f}%")

    # State subsets
    all_states = np.arange(n)
    branching_all = all_states[out_deg > 1]
    branching_xo = np.array([x for x in X_o if out_deg[x] > 1], dtype=int)

    print(f"\n  branching states total: {len(branching_all)} / {n}")
    print(f"  branching states in X_o: {len(branching_xo)} / {len(X_o)}")

    # ---- stacked diagnostics ----
    for name, states in [("X_o branching", branching_xo),
                         ("all branching", branching_all)]:
        d_fit, d_emp, n_b = contrast_rows(
            PT_intr_h, PT_satnav_h, PT_intr_emp, PT_satnav_emp, adj_out, states
        )
        if d_fit.size == 0:
            print(f"\n  [{name}] no branching states with data.")
            continue
        r_pearson = pearson(d_fit, d_emp)
        r_cosine = cosine(d_fit, d_emp)
        cosines, _ = per_row_cosines(
            PT_intr_h, PT_satnav_h, PT_intr_emp, PT_satnav_emp, adj_out, states
        )
        frac_pos = float((cosines > 0).mean()) if cosines.size else float("nan")
        median_cos = float(np.median(cosines)) if cosines.size else float("nan")
        mean_cos = float(cosines.mean()) if cosines.size else float("nan")
        print(f"\n  [{name}]  ({n_b} branching states, "
              f"{d_fit.size} legal-edge entries)")
        print(f"    stacked Pearson   = {r_pearson:+.3f}")
        print(f"    stacked cosine    = {r_cosine:+.3f}")
        print(f"    per-row cosine    mean = {mean_cos:+.3f}  "
              f"median = {median_cos:+.3f}")
        print(f"    rows with cosine > 0: {frac_pos*100:5.1f}%  "
              f"(50% = chance)")

    print(f"\nRESULT  seed={S}  auc_emp={auc_emp:.3f}  auc_fit={auc_fit:.3f}  "
          f"gap_closed={100*gap:.1f}%")


if __name__ == "__main__":
    main()
