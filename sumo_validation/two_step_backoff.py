"""Interpolation-backoff 2-step empirical detector.

Companion to two_step_empirical.py. Where that script shrinks the 2-step
estimate toward uniform-over-neighbors via Dirichlet smoothing, this one
shrinks it toward the 1-step empirical chain via linear interpolation
(Jelinek-Mercer):

    P_back(y | xp, x) = lambda * P_2(y | xp, x) + (1 - lambda) * P_1(y | x)

where P_2 is the MLE from 2-step counts (no Dirichlet) and P_1 is the
existing 1-step Laplace-smoothed empirical chain. The backoff itself
handles sparsity: when (xp, x) is rarely seen, low lambda makes the
estimate rely on P_1.

Two lambda modes:
  --lam_mode fixed   lambda = lam_fixed (constant)
  --lam_mode count   lambda = N(xp, x) / (N(xp, x) + kappa)

The count-conditional mode (Witten-Bell-ish) trusts the 2-step estimate
proportionally to how many observations support it.

Usage:
  ../.venv/bin/python two_step_backoff.py --seed 42 --lam_mode fixed --lam_fixed 0.5
  ../.venv/bin/python two_step_backoff.py --seed 42 --lam_mode count --kappa 10
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
os.chdir(HERE)

from sumo_to_phase3 import (
    build_adj, read_trajectories, empirical_chain,
)
from per_car_detector import _scores_for_chains
from congestion_filter import roc
from two_step_empirical import empirical_2step_counts


def score_2step_backoff_lr(test_trajs, counts_a, counts_b,
                           PT_a_1step, PT_b_1step,
                           lam_mode="fixed", lam_fixed=0.5, kappa=10.0,
                           eps=1e-15):
    """LR score under interpolation backoff. 1-step fallback for i=0 (no x_prev).

    For (xp, x) never seen in a regime, lambda effectively collapses to 0
    (the 2-step MLE is undefined; we fall back to 1-step entirely).
    """
    scores = np.zeros(len(test_trajs), dtype=float)
    n_fallback = 0
    n_total = 0

    for ti, t in enumerate(test_trajs):
        s = 0.0
        T = len(t)
        for i in range(T - 1):
            x = int(t[i])
            y = int(t[i + 1])
            n_total += 1

            p_a_1 = float(PT_a_1step[x, y])
            p_b_1 = float(PT_b_1step[x, y])

            if i == 0:
                p_a, p_b = p_a_1, p_b_1
                n_fallback += 1
            else:
                xp = int(t[i - 1])
                key = (xp, x)

                sub_a = counts_a.get(key)
                if sub_a is None:
                    p_a = p_a_1
                    n_fallback += 1
                else:
                    N_a = sum(sub_a.values())
                    p_a_2 = sub_a.get(y, 0) / N_a if N_a > 0 else 0.0
                    lam_a = lam_fixed if lam_mode == "fixed" else N_a / (N_a + kappa)
                    p_a = lam_a * p_a_2 + (1 - lam_a) * p_a_1

                sub_b = counts_b.get(key)
                if sub_b is None:
                    p_b = p_b_1
                else:
                    N_b = sum(sub_b.values())
                    p_b_2 = sub_b.get(y, 0) / N_b if N_b > 0 else 0.0
                    lam_b = lam_fixed if lam_mode == "fixed" else N_b / (N_b + kappa)
                    p_b = lam_b * p_b_2 + (1 - lam_b) * p_b_1

            s += np.log(max(p_b, eps)) - np.log(max(p_a, eps))
        scores[ti] = s

    fallback_rate = n_fallback / max(n_total, 1)
    return scores, fallback_rate


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--T", type=int, default=20)
    p.add_argument("--lam_mode", choices=("fixed", "count"), default="fixed")
    p.add_argument("--lam_fixed", type=float, default=0.5)
    p.add_argument("--kappa", type=float, default=10.0)
    p.add_argument("--smoothing", type=float, default=1e-3,
                   help="Laplace smoothing for the 1-step empirical_chain.")
    args = p.parse_args()

    S = args.seed
    edges, adj_out, idx = build_adj("net.net.xml")
    n = len(edges)

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
    test_ids = ({id(t) for t in test_intr_full},
                {id(t) for t in test_satnav_full})
    train_intr = [t for t in trajs_intr if id(t) not in test_ids[0]]
    train_satnav = [t for t in trajs_satnav if id(t) not in test_ids[1]]

    PT_intr_1 = empirical_chain(train_intr, adj_out, smoothing=args.smoothing)
    PT_satnav_1 = empirical_chain(train_satnav, adj_out, smoothing=args.smoothing)

    counts_intr, _ = empirical_2step_counts(train_intr)
    counts_satnav, _ = empirical_2step_counts(train_satnav)

    test_all = test_intr + test_satnav
    labels = np.concatenate([np.zeros(n_t, dtype=int),
                             np.ones(n_t, dtype=int)])

    s_1 = _scores_for_chains(test_all, PT_intr_1, PT_satnav_1)
    _, _, auc_1 = roc(s_1, labels)

    s_back, fb_rate = score_2step_backoff_lr(
        test_all, counts_intr, counts_satnav,
        PT_intr_1, PT_satnav_1,
        lam_mode=args.lam_mode, lam_fixed=args.lam_fixed, kappa=args.kappa,
    )
    _, _, auc_back = roc(s_back, labels)
    delta = auc_back - auc_1

    if args.lam_mode == "fixed":
        cfg = f"lam_fixed={args.lam_fixed}"
    else:
        cfg = f"kappa={args.kappa}"
    print(f"RESULT  seed={S}  T={T}  N_test={n_t}  beta={beta:.3f}  "
          f"backoff_mode={args.lam_mode}  {cfg}  "
          f"auc_1step={auc_1:.3f}  auc_2step_backoff={auc_back:.3f}  "
          f"delta={delta:+.3f}  fallback_rate={fb_rate:.3f}")


if __name__ == "__main__":
    main()
