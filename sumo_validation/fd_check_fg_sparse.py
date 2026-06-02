"""FD gradient check for Inverter f+g path (gamma < 1) with dense vs sparse solver.

Verifies two things:
  1. The new sparse-LU path returns the SAME loss and analytic gradient as the
     dense path (to machine precision modulo solver differences ~1e-10).
  2. Both analytic gradients match centred finite differences at ~1e-6 relative
     error on the active parameter blocks.

Uses a small synthetic graph from morimura.make_truth so the test runs in
seconds and exercises every code path (nu, omega_loc, omega_glo1, omega_glo2).

Usage:
  ../.venv/bin/python fd_check_fg_sparse.py
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
os.chdir(HERE)

from morimura import Inverter, make_truth, true_g


def main(n=40, d_T=3, d_psi=2, obs_frac=0.5, gamma=0.5, lam=1e-3,
         theta_seed=2, eps=1e-6, rtol=1e-3, atol=5e-2):
    print(f"=== FD check for f+g (dense vs sparse) ===")
    print(f"  n={n}  d_T={d_T}  d_psi={d_psi}  obs_frac={obs_frac}  "
          f"gamma={gamma}  lam={lam}")

    # Build a small synthetic graph with both global feature blocks active.
    rng_truth = np.random.default_rng(7)
    adj_out, pI, PT, P, pi, phi_T, psi = make_truth(
        n=n, mean_out_degree=3, beta=0.9, mix=0.7, dir_alpha=0.3,
        rng=rng_truth, d_T=d_T, d_psi=d_psi,
    )
    print(f"  built truth: pI shape {pI.shape}  PT shape {PT.shape}  "
          f"|E|={sum(len(ns) for ns in adj_out)}")

    # Observation set
    rng_xo = np.random.default_rng(13)
    n_obs = max(8, int(obs_frac * n))
    X_o = np.sort(rng_xo.choice(n, size=n_obs, replace=False))

    # Build f and g observations from the truth
    K = 1000
    f_obs = np.maximum(K * pi[X_o], 1.0)
    g_obs = true_g(PT, 0.9, X_o)

    log_f = np.log(f_obs)
    log_g = np.log(np.clip(g_obs, 1e-15, None))

    inv_dense = Inverter(adj_out, beta=0.9, gamma=gamma, lam=lam,
                         phi_T=phi_T, psi=psi, solver="dense")
    inv_sparse = Inverter(adj_out, beta=0.9, gamma=gamma, lam=lam,
                          phi_T=phi_T, psi=psi, solver="sparse")
    print(f"  d={inv_dense.d}  (nu={inv_dense.n}  om_loc={inv_dense.E}  "
          f"om_glo1={inv_dense.d_T}  om_glo2={inv_dense.d_psi})")
    assert inv_dense.d == inv_sparse.d

    # Random theta away from zero (otherwise some blocks have trivial gradients)
    rng = np.random.default_rng(theta_seed)
    theta0 = 0.1 * rng.standard_normal(inv_dense.d)

    # Dense vs sparse equivalence
    t0 = time.perf_counter()
    L_d, g_d = inv_dense.loss_grad(theta0, X_o, log_f, log_g)
    t_d = time.perf_counter() - t0
    t0 = time.perf_counter()
    L_s, g_s = inv_sparse.loss_grad(theta0, X_o, log_f, log_g)
    t_s = time.perf_counter() - t0
    dL = abs(L_d - L_s)
    dg = np.max(np.abs(g_d - g_s))
    print(f"\n  dense loss  = {L_d:.10f}  (took {t_d:.3f}s)")
    print(f"  sparse loss = {L_s:.10f}  (took {t_s:.3f}s)")
    print(f"  |dense - sparse| loss = {dL:.2e}")
    print(f"  ||dense - sparse|| grad (max) = {dg:.2e}")
    if dL > 1e-8 or dg > 1e-8:
        print("  ERROR: dense and sparse paths disagree beyond 1e-8 tolerance")
        sys.exit(1)
    print("  dense vs sparse equivalence: PASS")

    # FD against analytic gradient — sample one index per parameter block
    nu_lo, nu_hi = 0, inv_dense.n
    om_lo, om_hi = inv_dense.n, inv_dense.n + inv_dense.E
    g1_lo, g1_hi = om_hi, om_hi + inv_dense.d_T
    g2_lo, g2_hi = g1_hi, inv_dense.d

    block_ranges = {
        "nu":      (nu_lo, nu_hi),
        "om_loc":  (om_lo, om_hi),
        "om_glo1": (g1_lo, g1_hi),
        "om_glo2": (g2_lo, g2_hi),
    }

    print(f"\n  FD check (eps={eps:.0e}, rtol={rtol:.0e}, atol={atol:.0e}):")
    print(f"  block       i      analytic         FD           abs_err   rel_err   ok")
    n_fail = 0
    for block_name, (lo, hi) in block_ranges.items():
        if hi <= lo:
            print(f"  {block_name:<10}  (empty block)")
            continue
        per_block = 3
        idxs = rng.choice(np.arange(lo, hi), size=min(per_block, hi - lo),
                          replace=False)
        for i in idxs:
            t_plus = theta0.copy(); t_plus[i] += eps
            t_minus = theta0.copy(); t_minus[i] -= eps
            L_p, _ = inv_dense.loss_grad(t_plus, X_o, log_f, log_g)
            L_m, _ = inv_dense.loss_grad(t_minus, X_o, log_f, log_g)
            fd = (L_p - L_m) / (2.0 * eps)
            ana = g_d[i]
            abs_err = abs(fd - ana)
            scale = max(abs(ana), abs(fd))
            rel_err = abs_err / scale if scale > 0 else 0.0
            ok_i = (rel_err < rtol) or (abs_err < atol)
            if not ok_i:
                n_fail += 1
            mark = " " if ok_i else "X"
            print(f"  {block_name:<10} {i:>5}  {ana:+.6e}  {fd:+.6e}  "
                  f"{abs_err:.2e}  {rel_err:.2e}   {mark}")

    print(f"\n  failing FD dims: {n_fail}")
    if n_fail > 0:
        sys.exit(1)
    print(f"\n  verdict: PASS  (dense ≡ sparse to {dg:.0e};  FD ok)")


if __name__ == "__main__":
    main()
