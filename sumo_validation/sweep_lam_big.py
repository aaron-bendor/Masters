"""Tikhonov lambda sweep on bigger-bbox SUMO seed=42 — Tier-1 experiment.

Same data and pipeline as score_seed_big.py, but loops over lam in a log-spaced
grid and reports fitted_f AUC + training loss per lam. The baseline at default
lam=1e-3 is the first row.
"""
import os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.chdir(HERE)

from sumo_to_phase3 import build_adj, read_edge_counts, read_trajectories, empirical_chain
from per_car_detector import fit_chain, _scores_for_chains
from congestion_filter import roc

SEED = 42
X_O_SEED = 13
T = 20
MAXITER = 1500
LAMS = [1e-4, 1e-3, 1e-2, 1e-1]

edges, adj_out, idx = build_adj("net.net.xml")
n = len(edges)
f_intr = read_edge_counts(f"edgedata.intr.big.seed{SEED}.xml", idx)
f_satnav = read_edge_counts(f"edgedata.satnav.big.seed{SEED}.xml", idx)
trajs_intr, _ = read_trajectories(f"vehroutes.intr.big.seed{SEED}.xml", idx)
trajs_satnav, _ = read_trajectories(f"vehroutes.satnav.big.seed{SEED}.xml", idx)

mean_len = float(np.mean([len(t) for t in trajs_intr + trajs_satnav]))
beta = float(max(0.5, min(0.99, 1.0 - 1.0 / mean_len)))

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

cands = np.where((f_intr > 0) & (f_satnav > 0))[0]
n_obs = min(max(20, int(0.25 * n)), len(cands))
rng_xo = np.random.default_rng(X_O_SEED)
X_o = np.sort(rng_xo.choice(cands, size=n_obs, replace=False))

test_all = test_intr + test_satnav
labels = np.concatenate([np.zeros(n_t), np.ones(n_t)])

print(f"# seed={SEED} x_o_seed={X_O_SEED} T={T} maxiter={MAXITER} n={n} |X_o|={len(X_o)} beta={beta:.3f}")
print(f"# n_test_per_class={n_t}")
print()
print(f"{'lam':>10s} {'auc_fit':>8s} {'loss_intr':>10s} {'loss_satnav':>11s} {'sec':>6s}")

for lam in LAMS:
    t0 = time.time()
    _, PT_intr_h, _ = fit_chain(adj_out, beta, X_o, f_intr[X_o], lam=lam, maxiter=MAXITER)
    _, PT_satnav_h, _ = fit_chain(adj_out, beta, X_o, f_satnav[X_o], lam=lam, maxiter=MAXITER)
    elapsed = time.time() - t0
    _, _, auc_fit = roc(_scores_for_chains(test_all, PT_intr_h, PT_satnav_h), labels)
    # Re-fit just to grab the final loss in a clean way — fit_chain prints it,
    # but we want it captured. Cheaper: parse the printed line, but simplest is
    # to call inverter.loss(theta_final). Skipping for now; rely on printed.
    print(f"{lam:10.0e} {auc_fit:8.3f} {'see-print':>10s} {'see-print':>11s} {elapsed:6.1f}")
