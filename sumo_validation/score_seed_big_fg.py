"""Tier-1 experiment: f+g fitting on Phase 4 SUMO data with vs. without
empirical-Bayes Beta-Binomial shrinkage on the empirical hitting-rate g.

Runs three variants on the bigger-bbox seed=42 dataset (T=20, x_o_seed=13,
maxiter=1500):
  - fitted_f       : f-only baseline (gamma=1.0, no g term).
  - fitted_fg_floor: f+g with empirical g floored at 1e-3 (current default).
  - fitted_fg_shrink: f+g with empirical-Bayes Beta-Binomial shrinkage on g.

Background: ~80 % of empirical-g cells at this configuration sit at the
1e-3 floor; the floor contributes a constant pseudo-likelihood that
dominates L_h and pushes the fitted chains toward each other, washing
out the LR signal. Beta-Binomial shrinkage replaces the hard floor with
a smooth posterior mean (alpha + s_ij) / (alpha + beta + n_i), where the
Beta prior (alpha, beta) is fit by method of moments across all cells.

Usage:
  ../.venv/bin/python score_seed_big_fg.py
"""

import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.chdir(HERE)

from sumo_to_phase3 import (
    build_adj, read_edge_counts, read_trajectories, empirical_chain,
)
from per_car_detector import fit_chain, _scores_for_chains
from congestion_filter import roc


# ===============================================================
#  Empirical g with raw counts (numerator and denominators)
# ===============================================================

def empirical_g_counts(trajs, X_o, beta):
    """Return the same hit_sum and visit_count that empirical_g averages.

    For each (i, j) in X_o x X_o:
      visit_count[ki] = number of visits to X_o[ki] across all trajectories
      hit_sum[ki, kj] = sum over visits to ki of beta^(t_j - t_i) for the
                       FIRST hit of j after that visit (0 if never hit).

    Hits are discounted: each visit contributes at most 1 to the sum.
    """
    no = len(X_o)
    Xset = {int(x): k for k, x in enumerate(X_o)}
    hit_sum = np.zeros((no, no), dtype=float)
    visit_count = np.zeros(no, dtype=float)

    for traj in trajs:
        T = len(traj)
        for t_i in range(T):
            ki = Xset.get(int(traj[t_i]))
            if ki is None:
                continue
            visit_count[ki] += 1
            seen = set()
            for t_j in range(t_i + 1, T):
                kj = Xset.get(int(traj[t_j]))
                if kj is None or kj in seen:
                    continue
                hit_sum[ki, kj] += beta ** (t_j - t_i)
                seen.add(kj)
                if len(seen) == no:
                    break
    return hit_sum, visit_count


# ===============================================================
#  Beta prior fit (method of moments) and posterior shrinkage
# ===============================================================

def fit_beta_mom(values, weights=None):
    """Fit Beta(alpha, beta) by method of moments to the cell-wise raw
    g_ij = hit_sum[ki, kj] / visit_count[ki] (clipped to (0,1)).

    If `weights` is given, use a weighted mean/variance with weights
    equal to visit_count[ki] (cells from rows with more visits carry
    more information). Falls back to an uninformative prior when the
    estimated variance is too small or non-positive.
    """
    v = np.asarray(values, dtype=float).ravel()
    if weights is None:
        w = np.ones_like(v)
    else:
        w = np.asarray(weights, dtype=float).ravel()
    # Drop cells with no information (n_i == 0 means g_ij is undefined).
    mask = w > 0
    v, w = v[mask], w[mask]
    if v.size == 0:
        return 1.0, 1.0
    W = w.sum()
    mu = float((w * v).sum() / W)
    var = float((w * (v - mu) ** 2).sum() / W)
    # Need 0 < mu < 1 and var < mu(1-mu) for valid Beta MoM.
    eps = 1e-6
    mu = float(np.clip(mu, eps, 1.0 - eps))
    max_var = mu * (1.0 - mu)
    if var <= 0.0 or var >= max_var:
        # Either no spread (use uniform-like prior) or overdispersed
        # (no valid Beta - fall back to a weakly-informative prior at mu).
        # Use kappa = 2 (effective sample size 2) so the prior is weak.
        kappa = 2.0
    else:
        kappa = max_var / var - 1.0  # = alpha + beta
        kappa = max(kappa, 1e-3)
    alpha = mu * kappa
    beta_ = (1.0 - mu) * kappa
    return float(alpha), float(beta_)


def shrink_g_beta_binomial(hit_sum, visit_count, alpha, beta_):
    """Posterior mean under Beta(alpha, beta) prior, Binomial-like
    likelihood with effective trials n_i = visit_count[ki] and
    effective successes s_ij = hit_sum[ki, kj]:

      g_ij_post = (alpha + s_ij) / (alpha + beta + n_i)

    For rows with n_i = 0, the posterior reduces to the prior mean
    alpha / (alpha + beta). Output is clipped to (1e-15, 1.0) so log(g)
    is finite (no hard 1e-3 floor).
    """
    no = hit_sum.shape[0]
    n = visit_count[:, None]                      # (no, 1) broadcast
    post = (alpha + hit_sum) / (alpha + beta_ + n)
    return np.clip(post, 1e-15, 1.0)


# ===============================================================
#  Driver
# ===============================================================

def main(seed=42, x_o_seed=13, T=20, maxiter=1500, gamma_fg=0.1):
    S = seed
    print(f"=== Loading seed={S}, x_o_seed={x_o_seed}, T={T}, "
          f"maxiter={maxiter}, gamma_fg={gamma_fg} ===")

    edges, adj_out, idx = build_adj("net.net.xml")
    n = len(edges)
    f_intr = read_edge_counts(f"edgedata.intr.big.seed{S}.xml", idx)
    f_satnav = read_edge_counts(f"edgedata.satnav.big.seed{S}.xml", idx)
    trajs_intr, stats_i = read_trajectories(f"vehroutes.intr.big.seed{S}.xml", idx)
    trajs_satnav, stats_s = read_trajectories(f"vehroutes.satnav.big.seed{S}.xml", idx)

    mean_len = float(np.mean([len(t) for t in trajs_intr + trajs_satnav]))
    beta = float(max(0.5, min(0.99, 1.0 - 1.0 / mean_len)))
    print(f"  n={n}, mean traj len = {mean_len:.1f}, beta = {beta:.3f}")
    print(f"  trajs: intr={len(trajs_intr)}, satnav={len(trajs_satnav)}")

    # Test/train split (same rule as score_seed_big.py).
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
    print(f"  test: {n_t}/class at T={T}; train: "
          f"intr={len(train_intr)}, satnav={len(train_satnav)}")

    # X_o selection (same rule as score_seed_big.py).
    cands = np.where((f_intr > 0) & (f_satnav > 0))[0]
    n_obs = min(max(20, int(0.25 * n)), len(cands))
    rng_xo = np.random.default_rng(x_o_seed)
    X_o = np.sort(rng_xo.choice(cands, size=n_obs, replace=False))
    print(f"  |X_o| = {len(X_o)} ({100 * n_obs / n:.1f}% of n)")

    # ---- 1. f-only baseline ----
    print("\n=== Variant 1: fitted_f (f-only, gamma=1.0) ===")
    _, PT_intr_f, _ = fit_chain(adj_out, beta, X_o, f_intr[X_o],
                                maxiter=maxiter)
    _, PT_satnav_f, _ = fit_chain(adj_out, beta, X_o, f_satnav[X_o],
                                  maxiter=maxiter)

    # ---- 2. Empirical g + sparsity diagnostics ----
    print("\n=== Computing empirical g ===")
    hit_intr, vis_intr = empirical_g_counts(train_intr, X_o, beta)
    hit_satnav, vis_satnav = empirical_g_counts(train_satnav, X_o, beta)

    def _stats(hit, vis, label):
        no = hit.shape[0]
        # Cells with zero hits (sparsity rate).
        zero_frac = float((hit == 0).mean())
        # Per-cell raw count (the hit_sum is fractional due to discount;
        # report counts of "any hit" by counting cells > 0).
        nonzero = hit[hit > 0]
        if nonzero.size > 0:
            median_nz = float(np.median(nonzero))
            max_nz = float(nonzero.max())
        else:
            median_nz = 0.0
            max_nz = 0.0
        # Row visit counts (n_i).
        row_zero = int((vis == 0).sum())
        print(f"  {label}: zero-cell fraction = {100*zero_frac:.1f}% "
              f"({no*no} cells), nonzero median = {median_nz:.4f}, "
              f"max = {max_nz:.4f}, rows with n_i=0: {row_zero}/{no}")
        # Also empirical floor rate to compare with floor variant.
        return zero_frac, median_nz, max_nz

    s_intr = _stats(hit_intr, vis_intr, "g_intr  ")
    s_satnav = _stats(hit_satnav, vis_satnav, "g_satnav")

    # ---- 3. fitted_fg_floor: empirical g with 1e-3 floor ----
    print("\n=== Variant 2: fitted_fg_floor (gamma={:.2f}, floor=1e-3) ===".format(gamma_fg))
    # Build floored g matrices (matches empirical_g(... floor=1e-3) exactly).
    g_intr_floor = np.clip(
        hit_intr / np.maximum(vis_intr[:, None], 1.0), 1e-3, 1.0)
    g_satnav_floor = np.clip(
        hit_satnav / np.maximum(vis_satnav[:, None], 1.0), 1e-3, 1.0)
    frac_at_floor_i = float((g_intr_floor <= 1e-3 + 1e-12).mean())
    frac_at_floor_s = float((g_satnav_floor <= 1e-3 + 1e-12).mean())
    print(f"  at-floor fraction: intr {100*frac_at_floor_i:.1f}%, "
          f"satnav {100*frac_at_floor_s:.1f}%")

    _, PT_intr_fg_floor, _ = fit_chain(
        adj_out, beta, X_o, f_intr[X_o], g_intr_floor,
        gamma=gamma_fg, maxiter=maxiter)
    _, PT_satnav_fg_floor, _ = fit_chain(
        adj_out, beta, X_o, f_satnav[X_o], g_satnav_floor,
        gamma=gamma_fg, maxiter=maxiter)

    # ---- 4. fitted_fg_shrink: empirical-Bayes Beta-Binomial ----
    print("\n=== Variant 3: fitted_fg_shrink (gamma={:.2f}, Beta-Binomial MoM) ===".format(gamma_fg))
    # Fit the Beta prior on each regime's cell-wise g separately.
    # Use n_i as weights (more visits => more reliable g_ij estimate).
    raw_g_intr = hit_intr / np.maximum(vis_intr[:, None], 1.0)
    raw_g_satnav = hit_satnav / np.maximum(vis_satnav[:, None], 1.0)
    # Weights: broadcast n_i across columns so each cell is weighted by
    # the row's visit count.
    w_intr = np.broadcast_to(vis_intr[:, None], raw_g_intr.shape)
    w_satnav = np.broadcast_to(vis_satnav[:, None], raw_g_satnav.shape)
    a_i, b_i = fit_beta_mom(raw_g_intr, weights=w_intr)
    a_s, b_s = fit_beta_mom(raw_g_satnav, weights=w_satnav)
    print(f"  Beta prior (intr  ): alpha = {a_i:.4f}, beta = {b_i:.4f}, "
          f"prior mean = {a_i/(a_i+b_i):.4f}, kappa = {a_i+b_i:.2f}")
    print(f"  Beta prior (satnav): alpha = {a_s:.4f}, beta = {b_s:.4f}, "
          f"prior mean = {a_s/(a_s+b_s):.4f}, kappa = {a_s+b_s:.2f}")

    g_intr_shrink = shrink_g_beta_binomial(hit_intr, vis_intr, a_i, b_i)
    g_satnav_shrink = shrink_g_beta_binomial(hit_satnav, vis_satnav, a_s, b_s)
    print(f"  g_intr_shrink   range: {g_intr_shrink.min():.4f} .. "
          f"{g_intr_shrink.max():.4f}, mean = {g_intr_shrink.mean():.4f}")
    print(f"  g_satnav_shrink range: {g_satnav_shrink.min():.4f} .. "
          f"{g_satnav_shrink.max():.4f}, mean = {g_satnav_shrink.mean():.4f}")

    _, PT_intr_fg_shrink, _ = fit_chain(
        adj_out, beta, X_o, f_intr[X_o], g_intr_shrink,
        gamma=gamma_fg, maxiter=maxiter)
    _, PT_satnav_fg_shrink, _ = fit_chain(
        adj_out, beta, X_o, f_satnav[X_o], g_satnav_shrink,
        gamma=gamma_fg, maxiter=maxiter)

    # ---- 5. Scoring ----
    print("\n=== Scoring ===")
    all_test = test_intr + test_satnav
    labels = np.concatenate([np.zeros(n_t, dtype=int),
                             np.ones(n_t, dtype=int)])
    _, _, auc_f = roc(_scores_for_chains(all_test, PT_intr_f, PT_satnav_f),
                      labels)
    _, _, auc_fg_floor = roc(
        _scores_for_chains(all_test, PT_intr_fg_floor, PT_satnav_fg_floor),
        labels)
    _, _, auc_fg_shrink = roc(
        _scores_for_chains(all_test, PT_intr_fg_shrink, PT_satnav_fg_shrink),
        labels)
    # Empirical-chain ceiling for context.
    PT_intr_emp = empirical_chain(train_intr, adj_out)
    PT_satnav_emp = empirical_chain(train_satnav, adj_out)
    _, _, auc_emp = roc(
        _scores_for_chains(all_test, PT_intr_emp, PT_satnav_emp),
        labels)

    def _gap(a):
        return (a - 0.5) / (auc_emp - 0.5) if auc_emp > 0.5 else float("nan")

    print(f"\n  empirical (ceiling)      AUC = {auc_emp:.3f}")
    print(f"  fitted_f                 AUC = {auc_f:.3f}    "
          f"gap_closed = {100*_gap(auc_f):6.1f}%")
    print(f"  fitted_fg (floor 1e-3)   AUC = {auc_fg_floor:.3f}    "
          f"gap_closed = {100*_gap(auc_fg_floor):6.1f}%")
    print(f"  fitted_fg (shrinkage)    AUC = {auc_fg_shrink:.3f}    "
          f"gap_closed = {100*_gap(auc_fg_shrink):6.1f}%")

    print(f"\nRESULT_TABLE  seed={S}  T={T}  |X_o|={len(X_o)}  "
          f"n={n}  beta={beta:.3f}  N_test={n_t}  gamma_fg={gamma_fg}")
    print(f"RESULT  f={auc_f:.3f}  fg_floor={auc_fg_floor:.3f}  "
          f"fg_shrink={auc_fg_shrink:.3f}  emp_ceiling={auc_emp:.3f}")
    print(f"BETA_PRIOR  intr  (alpha,beta) = ({a_i:.4f}, {b_i:.4f})  "
          f"satnav (alpha,beta) = ({a_s:.4f}, {b_s:.4f})")
    print(f"SPARSITY  intr  zero_frac={s_intr[0]:.3f} "
          f"nonzero_median={s_intr[1]:.4f} max={s_intr[2]:.4f}")
    print(f"SPARSITY  satnav zero_frac={s_satnav[0]:.3f} "
          f"nonzero_median={s_satnav[1]:.4f} max={s_satnav[2]:.4f}")


if __name__ == "__main__":
    main()
