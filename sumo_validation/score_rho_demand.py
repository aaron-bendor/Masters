"""Score the detector on a pair of (demand, rho) SUMO populations.

Reads vehroutes/edgedata for two tags A and B (each tag = "d<P>.r<RHO>.seed<SEED>"),
fits chains on each, and computes empirical + fitted AUC for distinguishing
trajectories drawn from A vs B. Mirrors the protocol in score_seed_big.py
(f-only, local-only fit by default; pass --features real to enable globals).

Usage:
  ../.venv/bin/python score_rho_demand.py \
      --tag_a d0.3.r0.0.seed42 --tag_b d0.8.r1.0.seed42
"""
import argparse, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.chdir(HERE)

from sumo_to_phase3 import (
    build_adj, read_edge_counts, read_trajectories, empirical_chain,
    extract_features,
)
from per_car_detector import fit_chain, _scores_for_chains
from congestion_filter import roc

p = argparse.ArgumentParser()
p.add_argument("--tag_a", required=True,
               help="First population tag, e.g. d0.3.r0.0.seed42")
p.add_argument("--tag_b", required=True,
               help="Second population tag, e.g. d0.8.r1.0.seed42")
p.add_argument("--net", default="net.net.xml")
p.add_argument("--x_o_seed", type=int, default=13)
p.add_argument("--T", type=int, default=20)
p.add_argument("--maxiter", type=int, default=1500)
p.add_argument("--features", choices=("none", "real"), default="none")
args = p.parse_args()

edges, adj_out, idx = build_adj(args.net)
n = len(edges)

f_a = read_edge_counts(f"edgedata.{args.tag_a}.xml", idx)
f_b = read_edge_counts(f"edgedata.{args.tag_b}.xml", idx)
trajs_a, _ = read_trajectories(f"vehroutes.{args.tag_a}.xml", idx)
trajs_b, _ = read_trajectories(f"vehroutes.{args.tag_b}.xml", idx)

mean_len = float(np.mean([len(t) for t in trajs_a + trajs_b]))
beta = float(max(0.5, min(0.99, 1.0 - 1.0 / mean_len)))

T = args.T
long_a = [t for t in trajs_a if len(t) >= T + 1]
long_b = [t for t in trajs_b if len(t) >= T + 1]
rng_test = np.random.default_rng(42)
rng_test.shuffle(long_a)
rng_test.shuffle(long_b)
n_t = min(300, len(long_a), len(long_b))
test_a_full = long_a[:n_t]
test_b_full = long_b[:n_t]
test_a = [t[: T + 1] for t in test_a_full]
test_b = [t[: T + 1] for t in test_b_full]
test_ids = ({id(t) for t in test_a_full}, {id(t) for t in test_b_full})
train_a = [t for t in trajs_a if id(t) not in test_ids[0]]
train_b = [t for t in trajs_b if id(t) not in test_ids[1]]

cands = np.where((f_a > 0) & (f_b > 0))[0]
n_obs = min(max(20, int(0.25 * n)), len(cands))
rng_xo = np.random.default_rng(args.x_o_seed)
X_o = np.sort(rng_xo.choice(cands, size=n_obs, replace=False))

if args.features == "real":
    phi_T, psi = extract_features(args.net, edges, adj_out)
    d_T, d_psi = phi_T.shape[1], psi.shape[1]
else:
    phi_T, psi = None, None
    d_T, d_psi = 0, 0

_, PT_a_h, _ = fit_chain(adj_out, beta, X_o, f_a[X_o], maxiter=args.maxiter,
                         phi_T=phi_T, psi=psi)
_, PT_b_h, _ = fit_chain(adj_out, beta, X_o, f_b[X_o], maxiter=args.maxiter,
                         phi_T=phi_T, psi=psi)
PT_a_emp = empirical_chain(train_a, adj_out)
PT_b_emp = empirical_chain(train_b, adj_out)

test_all = test_a + test_b
labels = np.concatenate([np.zeros(n_t), np.ones(n_t)])
_, _, auc_fit = roc(_scores_for_chains(test_all, PT_a_h, PT_b_h), labels)
_, _, auc_emp = roc(_scores_for_chains(test_all, PT_a_emp, PT_b_emp), labels)
gap = (auc_fit - 0.5) / (auc_emp - 0.5) if auc_emp > 0.5 else float("nan")

print(
    f"RESULT  tag_a={args.tag_a}  tag_b={args.tag_b}  "
    f"T={T}  N_test={n_t:>3}  |X_o|={len(X_o)}  n={n}  beta={beta:.3f}  "
    f"features={args.features}  d_T={d_T}  d_psi={d_psi}  "
    f"empirical={auc_emp:.3f}  fitted_f={auc_fit:.3f}  gap_closed={100*gap:6.1f}%"
)
