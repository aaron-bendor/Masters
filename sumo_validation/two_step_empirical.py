"""Model-misspecification diagnostic: 2-step empirical detector.

The logbook's iter-16 / iter-17 conclusion is that the 1-step Markov empirical
ceiling at T=20 caps around 0.685 on the bigger bbox, and no fitter (more
iterations, multistart, LM, lambda tuning) can exceed it. The open question:
is that ceiling a fitter limit or a *data* limit?

This script settles it without fitting. It builds 2-step empirical conditional
distributions
    P_2(y | x, x_prev) = (count(x_prev, x, y) + alpha) / (sum_y' count + alpha * |adj(x)|)
from training trajectories under each regime (intr, satnav), then scores test
trajectories with the LR detector. Comparing the resulting AUC to the 1-step
empirical ceiling tells us:

  AUC_2step > AUC_1step + noise band:
      Higher-order memory exists in the data. The local-only / softmax
      1-step parametric family is genuinely misspecified, and a 2-step
      parametric inverter is worth building.

  AUC_2step ~= AUC_1step:
      No extra 2-step memory. The 1-step Markov ceiling IS the data limit.
      No optimisation, multistart, or feature tweak can push past it; the
      project should reframe Phase 4 around what is and is not recoverable
      with 1-step Markov machinery.

This is purely diagnostic: no inverter, no optimisation, no parameter sweep.
Runs in seconds-to-minutes (dominated by counting transitions).

Usage:
  ../.venv/bin/python two_step_empirical.py --seed 42
  for S in 23 7 42 101 2024; do
    ../.venv/bin/python two_step_empirical.py --seed $S
  done
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
    build_adj, read_edge_counts, read_trajectories, empirical_chain,
)
from per_car_detector import _scores_for_chains
from congestion_filter import roc


def empirical_2step_counts(trajs):
    """Count (x_prev, x, y) triples across all training trajectories.

    Returns a dict keyed by (x_prev, x) whose value is another dict y -> count.
    Memory-light: only (x_prev, x) pairs that actually appear in the data are
    stored, not the full n*n*n tensor.
    """
    counts = {}
    n_triples = 0
    for t in trajs:
        T = len(t)
        if T < 3:
            continue
        for i in range(T - 2):
            xp = int(t[i])
            x  = int(t[i + 1])
            y  = int(t[i + 2])
            key = (xp, x)
            sub = counts.get(key)
            if sub is None:
                counts[key] = {y: 1}
            else:
                sub[y] = sub.get(y, 0) + 1
            n_triples += 1
    return counts, n_triples


def score_2step_lr(test_trajs, counts_a, counts_b,
                   PT_a_1step, PT_b_1step, adj_out,
                   smoothing=1e-3, eps=1e-15):
    """LR score under 2-step empirical, with 1-step fallback for:
        (a) the first transition of each trajectory (no x_prev), and
        (b) any (x_prev, x) pair unseen in that regime's training data.

    Mirrors per_car_detector._scores_for_chains' contract: returns an array
    of Lambda(tau) = sum_t [log P_b - log P_a] of length len(test_trajs).
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

            if i == 0:
                p_a = float(PT_a_1step[x, y])
                p_b = float(PT_b_1step[x, y])
                n_fallback += 1
            else:
                xp = int(t[i - 1])
                key = (xp, x)

                sub_a = counts_a.get(key)
                if sub_a is None:
                    p_a = float(PT_a_1step[x, y])
                    n_fallback += 1
                else:
                    nbrs = adj_out[x]
                    if not nbrs:
                        p_a = float(PT_a_1step[x, y])
                    else:
                        denom = sum(sub_a.values()) + smoothing * len(nbrs)
                        p_a = (sub_a.get(y, 0) + smoothing) / denom

                sub_b = counts_b.get(key)
                if sub_b is None:
                    p_b = float(PT_b_1step[x, y])
                else:
                    nbrs = adj_out[x]
                    if not nbrs:
                        p_b = float(PT_b_1step[x, y])
                    else:
                        denom = sum(sub_b.values()) + smoothing * len(nbrs)
                        p_b = (sub_b.get(y, 0) + smoothing) / denom

            s += np.log(max(p_b, eps)) - np.log(max(p_a, eps))
        scores[ti] = s

    fallback_rate = n_fallback / max(n_total, 1)
    return scores, fallback_rate


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--T", type=int, default=20)
    p.add_argument("--smoothing", type=float, default=1e-3,
                   help="Laplace alpha for 2-step (and 1-step) conditional.")
    args = p.parse_args()

    S = args.seed
    print(f"=== 2-step empirical detector (seed={S}, T={args.T}) ===")

    edges, adj_out, idx = build_adj("net.net.xml")
    n = len(edges)
    f_intr = read_edge_counts(f"edgedata.intr.big.seed{S}.xml", idx)
    f_satnav = read_edge_counts(f"edgedata.satnav.big.seed{S}.xml", idx)
    trajs_intr, _ = read_trajectories(f"vehroutes.intr.big.seed{S}.xml", idx)
    trajs_satnav, _ = read_trajectories(f"vehroutes.satnav.big.seed{S}.xml", idx)

    mean_len = float(np.mean([len(t) for t in trajs_intr + trajs_satnav]))
    beta = float(max(0.5, min(0.99, 1.0 - 1.0 / mean_len)))

    # Same test/train split as contrast_correlation.py / score_seed_big.py.
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

    print(f"  n={n}  beta={beta:.3f}  T={T}  N_test={n_t}/class")
    print(f"  train trajs: intr={len(train_intr)}  satnav={len(train_satnav)}")

    print("\n  building 1-step empirical chains (smoothing=1e-3) ...")
    PT_intr_1 = empirical_chain(train_intr, adj_out, smoothing=args.smoothing)
    PT_satnav_1 = empirical_chain(train_satnav, adj_out,
                                  smoothing=args.smoothing)

    print("  counting (x_prev, x, y) triples ...")
    counts_intr, n_trip_i = empirical_2step_counts(train_intr)
    counts_satnav, n_trip_s = empirical_2step_counts(train_satnav)
    n_pairs_i = len(counts_intr)
    n_pairs_s = len(counts_satnav)
    print(f"    intr  : {n_trip_i:>9} triples across {n_pairs_i:>7} "
          f"(x_prev, x) pairs")
    print(f"    satnav: {n_trip_s:>9} triples across {n_pairs_s:>7} "
          f"(x_prev, x) pairs")

    test_all = test_intr + test_satnav
    labels = np.concatenate([np.zeros(n_t, dtype=int),
                             np.ones(n_t, dtype=int)])

    print("\n  scoring with 1-step empirical (ceiling reference) ...")
    s_1 = _scores_for_chains(test_all, PT_intr_1, PT_satnav_1)
    _, _, auc_1 = roc(s_1, labels)

    print("  scoring with 2-step empirical ...")
    s_2, fb_rate = score_2step_lr(
        test_all, counts_intr, counts_satnav,
        PT_intr_1, PT_satnav_1, adj_out,
        smoothing=args.smoothing,
    )
    _, _, auc_2 = roc(s_2, labels)

    delta = auc_2 - auc_1

    print(f"\n  1-step fallback rate on test transitions: "
          f"{100 * fb_rate:.1f}%")
    print(f"\n  AUC_1step (ceiling)  = {auc_1:.4f}")
    print(f"  AUC_2step            = {auc_2:.4f}")
    print(f"  delta (2step-1step)  = {delta:+.4f}")

    if delta > 0.03:
        verdict = ("HIGHER-ORDER MEMORY DETECTED -> "
                   "2-step parametric inverter is justified")
    elif delta > 0.01:
        verdict = ("marginal 2-step lift; investigate further before "
                   "committing to a 2-step parametric build")
    elif delta > -0.01:
        verdict = ("NO EXTRA 2-STEP SIGNAL -> 1-step Markov ceiling IS "
                   "the data limit; reframe Phase 4 accordingly")
    else:
        verdict = ("2-step AUC < 1-step AUC -> overfitting / sparsity "
                   "harms LR; not a useful direction")
    print(f"\n  verdict: {verdict}")

    print(f"\nRESULT  seed={S}  T={T}  N_test={n_t}  beta={beta:.3f}  "
          f"auc_1step={auc_1:.3f}  auc_2step={auc_2:.3f}  "
          f"delta={delta:+.3f}  fallback_rate={fb_rate:.3f}")


if __name__ == "__main__":
    main()
