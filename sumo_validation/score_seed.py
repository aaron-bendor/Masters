"""Score the detector on one SUMO-seed-tagged dataset.

Reads vehroutes/edgedata.{intr,satnav}.small.seed<N>.xml, fits f-only chains
on a single X_o draw, prints one CSV line with empirical / fitted_f / gap closed.

Usage:
  ../.venv/bin/python score_seed.py --seed 23
  ../.venv/bin/python score_seed.py --seed 42 --x_o_seed 13 --T 20

Loop over multiple seeds:
  for S in 23 7 42 101 2024; do ../.venv/bin/python score_seed.py --seed $S; done
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

p = argparse.ArgumentParser()
p.add_argument("--seed", type=int, required=True, help="SUMO/randomTrips seed used in run_seed.sh")
p.add_argument("--x_o_seed", type=int, default=13)
p.add_argument("--T", type=int, default=20)
p.add_argument("--maxiter", type=int, default=1500)
args = p.parse_args()

S = args.seed
edges, adj_out, idx = build_adj("net.small.net.xml")
n = len(edges)
f_intr = read_edge_counts(f"edgedata.intr.small.seed{S}.xml", idx)
f_satnav = read_edge_counts(f"edgedata.satnav.small.seed{S}.xml", idx)
trajs_intr, stats_i = read_trajectories(f"vehroutes.intr.small.seed{S}.xml", idx)
trajs_satnav, stats_s = read_trajectories(f"vehroutes.satnav.small.seed{S}.xml", idx)

mean_len = float(np.mean([len(t) for t in trajs_intr + trajs_satnav]))
beta = float(max(0.5, min(0.99, 1.0 - 1.0 / mean_len)))

# Build test set with len >= T+1.
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

# X_o.
cands = np.where((f_intr > 0) & (f_satnav > 0))[0]
n_obs = min(max(20, int(0.25 * n)), len(cands))
rng_xo = np.random.default_rng(args.x_o_seed)
X_o = np.sort(rng_xo.choice(cands, size=n_obs, replace=False))

# Fit + score.
_, PT_intr_h, _ = fit_chain(adj_out, beta, X_o, f_intr[X_o], maxiter=args.maxiter)
_, PT_satnav_h, _ = fit_chain(adj_out, beta, X_o, f_satnav[X_o], maxiter=args.maxiter)
PT_intr_emp = empirical_chain(train_intr, adj_out)
PT_satnav_emp = empirical_chain(train_satnav, adj_out)

test_all = test_intr + test_satnav
labels = np.concatenate([np.zeros(n_t), np.ones(n_t)])
_, _, auc_fit = roc(_scores_for_chains(test_all, PT_intr_h, PT_satnav_h), labels)
_, _, auc_emp = roc(_scores_for_chains(test_all, PT_intr_emp, PT_satnav_emp), labels)
gap = (auc_fit - 0.5) / (auc_emp - 0.5) if auc_emp > 0.5 else float("nan")

# Compact result line.
print(
    f"RESULT  seed={S:>5}  T={T}  N_test={n_t:>3}  "
    f"|X_o|={len(X_o)}  beta={beta:.3f}  "
    f"empirical={auc_emp:.3f}  fitted_f={auc_fit:.3f}  gap_closed={100*gap:6.1f}%"
)
