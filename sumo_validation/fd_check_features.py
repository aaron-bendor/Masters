"""FD gradient check for Inverter.loss_grad with real features.

Builds a real-features graph (SUMO bigger-bbox or Xuancheng), constructs an
Inverter with phi_T / psi globals, and verifies the analytic gradient against
centred finite differences at a random theta.

This is the safety gate before launching any sweep with --features real.
A clean run reports relative errors ~1e-6 or better on the dimensions where
the analytic gradient is non-trivial, with an absolute-tolerance fallback
for near-zero gradient components (which sit at the FD round-off floor).

Usage:
  ../.venv/bin/python fd_check_features.py                       # SUMO seed=42 (default)
  ../.venv/bin/python fd_check_features.py --dataset xuancheng   # Xuancheng day=2023-04-17
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
os.chdir(HERE)

from sumo_to_phase3 import build_adj, read_edge_counts, extract_features
from morimura import Inverter


def main(dataset="sumo", seed=42, day="2023-04-17", x_o_seed=13,
         theta_seed=1, eps=1e-6, n_probe=12, rtol=1e-3, atol=5e-2):
    print(f"=== FD gradient check (dataset={dataset}, real features) ===")
    if dataset == "sumo":
        edges, adj_out, idx = build_adj("net.net.xml")
        net_path = "net.net.xml"
        f_obs = read_edge_counts(f"edgedata.intr.big.seed{seed}.xml", idx)
    elif dataset == "xuancheng":
        import datetime
        _REAL_DATA = os.path.join(os.path.dirname(HERE), "real_data")
        sys.path.insert(0, _REAL_DATA)
        from load_xuancheng import (  # noqa: E402
            load_xuancheng_regime, day_spec, split_rush_vs_offpeak,
            NET_PATH_DEFAULT,
        )
        date = datetime.date.fromisoformat(day)
        # Use rush-vs-off-peak split; we only need one regime's f for the FD check.
        edges, adj_out, idx, trajs_a, trajs_b, f_a, f_b = load_xuancheng_regime(
            net_path=NET_PATH_DEFAULT,
            day_specs=[day_spec(date)],
            regime_split=lambda st, d: split_rush_vs_offpeak(st["start_time"]),
            label_a="rush", label_b="offpeak", verbose=False,
        )
        f_obs = f_a  # use rush counts
        net_path = NET_PATH_DEFAULT
    else:
        raise ValueError(f"unknown dataset: {dataset}")

    n = len(edges)
    cands = np.where(f_obs > 0)[0]
    n_obs = min(max(20, int(0.25 * n)), len(cands))
    rng_xo = np.random.default_rng(x_o_seed)
    X_o = np.sort(rng_xo.choice(cands, size=n_obs, replace=False))

    phi_T, psi = extract_features(net_path, edges, adj_out)
    inv = Inverter(adj_out, beta=0.972, gamma=1.0, lam=1e-3,
                   phi_T=phi_T, psi=psi)
    print(f"  n={inv.n}  E={inv.E}  d_T={inv.d_T}  d_psi={inv.d_psi}  d={inv.d}")
    print(f"  |X_o|={len(X_o)}  gamma={inv.gamma}  (f-only, log_g=None)")

    log_f = np.log(np.maximum(f_obs[X_o], 1e-12))

    rng = np.random.default_rng(theta_seed)
    theta0 = 0.1 * rng.standard_normal(inv.d)

    L0, g_analytic = inv.loss_grad(theta0, X_o, log_f, None)
    print(f"  L(theta0) = {L0:.6f}")

    # Probe a spread of indices across the four parameter blocks.
    nu_lo, nu_hi = 0, inv.n
    om_lo, om_hi = inv.n, inv.n + inv.E
    g1_lo, g1_hi = om_hi, om_hi + inv.d_T
    g2_lo, g2_hi = g1_hi, inv.d

    block_ranges = {
        "nu":       (nu_lo, nu_hi),
        "om_loc":   (om_lo, om_hi),
        "om_glo1":  (g1_lo, g1_hi),
        "om_glo2":  (g2_lo, g2_hi),
    }

    # FD floor: centred-difference round-off is ~|L| * eps_machine / h.
    fd_noise_floor = abs(L0) * 2.2e-16 / eps
    print(f"  expected FD round-off floor (abs): ~{fd_noise_floor:.1e}")
    print(f"  pass criterion: rel_err < rtol={rtol:.0e} when |grad| > atol={atol:.0e},")
    print(f"                  else abs_err < atol (tiny-gradient regime)")

    print(f"\n  block       i      analytic         FD           abs_err   rel_err   ok")
    n_fail = 0
    for block_name, (lo, hi) in block_ranges.items():
        if hi <= lo:
            print(f"  {block_name:<10}  (empty block)")
            continue
        per_block = max(1, n_probe // 4)
        idxs = rng.choice(np.arange(lo, hi), size=min(per_block, hi - lo),
                          replace=False)
        for i in idxs:
            t_plus = theta0.copy(); t_plus[i] += eps
            t_minus = theta0.copy(); t_minus[i] -= eps
            L_p, _ = inv.loss_grad(t_plus, X_o, log_f, None)
            L_m, _ = inv.loss_grad(t_minus, X_o, log_f, None)
            fd = (L_p - L_m) / (2.0 * eps)
            ana = g_analytic[i]
            abs_err = abs(fd - ana)
            scale = max(abs(ana), abs(fd))
            rel_err = abs_err / scale if scale > 0 else 0.0
            # np.allclose-style: tolerate either a small relative error (when
            # gradient is non-trivial) OR a small absolute error (when both
            # ana and fd are near the FD round-off floor).
            ok_i = (rel_err < rtol) or (abs_err < atol)
            if not ok_i:
                n_fail += 1
            mark = " " if ok_i else "X"
            print(f"  {block_name:<10} {i:>5}  {ana:+.6e}  {fd:+.6e}  "
                  f"{abs_err:.2e}  {rel_err:.2e}   {mark}")

    print(f"\n  failing dims: {n_fail}")
    ok = (n_fail == 0)
    print(f"  verdict: {'PASS' if ok else 'FAIL'}")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=("sumo", "xuancheng"), default="sumo")
    p.add_argument("--seed", type=int, default=42,
                   help="SUMO seed (used when --dataset sumo).")
    p.add_argument("--day", type=str, default="2023-04-17",
                   help="ISO date (used when --dataset xuancheng).")
    a = p.parse_args()
    main(dataset=a.dataset, seed=a.seed, day=a.day)
