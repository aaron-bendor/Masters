"""Small-bbox T-sweep mirroring iter 13's bigger-bbox protocol.

Fits f-only chains ONCE on the small-bbox SUMO data (n=686, -p 2.0).
Then scores per-car LR at T_target in {20, 30, 40} against a fixed
held-out test set (300 per class, length >= 41). The same X_o is used
for the fit and the same trajectories are used for scoring at every T.

Reports empirical 1-step Markov ceiling AUC, fitted_f AUC, and AUC gap
closed at each T.
"""
import os, sys
import numpy as np

HERE = "/Users/aaronbendor/DESENG/Masters/sumo_validation"
sys.path.insert(0, HERE)
os.chdir(HERE)

from sumo_to_phase3 import (
    build_adj, read_edge_counts, read_trajectories,
    empirical_chain, _scores_branching,
)
from per_car_detector import fit_chain, _scores_for_chains
from congestion_filter import roc

NET = "net.small.net.xml"
EDGE_INTR = "edgedata.intr.small.p2.0.xml"
EDGE_SATNAV = "edgedata.satnav.small.p2.0.xml"
VEH_INTR = "vehroutes.intr.small.p2.0.xml"
VEH_SATNAV = "vehroutes.satnav.small.p2.0.xml"

OBS_FRAC = 0.25
X_O_SEEDS = (13, 42, 7, 101, 2024)  # multi-seed to bound X_o variance
TEST_SEED = 42    # used to shuffle and pick the held-out test set
N_TEST = 300
T_MAX = 40        # all test trajectories need len >= T_MAX + 1


print("=== Loading small-bbox SUMO data ===")
edges, adj_out, idx = build_adj(NET)
n = len(edges)
out_deg = float(np.mean([len(a) for a in adj_out]))
print(f"  n = {n}, mean out-degree = {out_deg:.2f}")

f_intr = read_edge_counts(EDGE_INTR, idx)
f_satnav = read_edge_counts(EDGE_SATNAV, idx)
print(f"  f_intr   total = {f_intr.sum():.0f}, nonzero = {int((f_intr > 0).sum())}/{n}")
print(f"  f_satnav total = {f_satnav.sum():.0f}, nonzero = {int((f_satnav > 0).sum())}/{n}")

trajs_intr, stats_i = read_trajectories(VEH_INTR, idx)
trajs_satnav, stats_s = read_trajectories(VEH_SATNAV, idx)
print(f"  intr   trajs kept = {stats_i['kept']}, mean len = {np.mean([len(t) for t in trajs_intr]):.1f}")
print(f"  satnav trajs kept = {stats_s['kept']}, mean len = {np.mean([len(t) for t in trajs_satnav]):.1f}")

mean_len_all = float(np.mean([len(t) for t in trajs_intr + trajs_satnav]))
beta = float(max(0.5, min(0.99, 1.0 - 1.0 / mean_len_all)))
print(f"  combined mean len = {mean_len_all:.1f}, beta = {beta:.3f}")

# Per-T test sets: small bbox has only ~75 intrinsic trajectories with len>=41,
# too few to fix a single test set across T. Instead, for each T_target build a
# test set of min(300, available) per class with len >= T_target+1. Reports the
# N_test used at each row. The training pool excludes the union of all per-T
# test trajectories so train/test contamination cannot bias any single row.
print(f"\n=== Building per-T test sets ===")
per_T_test = {}
for T_target in (20, 30, 40):
    long_intr = [t for t in trajs_intr if len(t) >= T_target + 1]
    long_satnav = [t for t in trajs_satnav if len(t) >= T_target + 1]
    rng_t = np.random.default_rng(TEST_SEED + T_target)
    rng_t.shuffle(long_intr)
    rng_t.shuffle(long_satnav)
    nt = min(N_TEST, len(long_intr), len(long_satnav))
    per_T_test[T_target] = (long_intr[:nt], long_satnav[:nt], nt)
    print(f"  T={T_target}: available intr={len(long_intr)}, satnav={len(long_satnav)} "
          f"=> N_test={nt} per class")

# Aggregate test-set IDs across all T to remove from training pool.
all_test_ids_intr = set()
all_test_ids_satnav = set()
for ti, ts, _ in per_T_test.values():
    all_test_ids_intr.update(id(t) for t in ti)
    all_test_ids_satnav.update(id(t) for t in ts)
train_intr = [t for t in trajs_intr if id(t) not in all_test_ids_intr]
train_satnav = [t for t in trajs_satnav if id(t) not in all_test_ids_satnav]
print(f"  train pool: intr={len(train_intr)}, satnav={len(train_satnav)}")

# Empirical chains (X_o-independent — used for ceiling).
print(f"\n=== Empirical chains (1-step Markov ceiling, X_o-independent) ===")
PT_intr_emp = empirical_chain(train_intr, adj_out)
PT_satnav_emp = empirical_chain(train_satnav, adj_out)

# Multi-seed fitted_f sweep.
cands = np.where((f_intr > 0) & (f_satnav > 0))[0]
n_obs = min(max(20, int(OBS_FRAC * n)), len(cands))

print(f"\n=== Multi-seed fitted_f sweep ({len(X_O_SEEDS)} X_o draws) ===")
all_aucs_fit = {T: [] for T in (20, 30, 40)}
all_aucs_emp = {T: None for T in (20, 30, 40)}

for seed_idx, x_o_seed in enumerate(X_O_SEEDS):
    rng_xo = np.random.default_rng(x_o_seed)
    X_o = np.sort(rng_xo.choice(cands, size=n_obs, replace=False))
    print(f"\n--- X_O_SEED={x_o_seed}  |X_o|={len(X_o)} ---")
    print("  fit intrinsic ...")
    _, PT_intr_h, _ = fit_chain(adj_out, beta, X_o, f_intr[X_o], maxiter=5000)
    print("  fit satnav ...")
    _, PT_satnav_h, _ = fit_chain(adj_out, beta, X_o, f_satnav[X_o], maxiter=5000)

    for T_target in (20, 30, 40):
        test_intr_full_T, test_satnav_full_T, n_t = per_T_test[T_target]
        test_intr_T = [t[: T_target + 1] for t in test_intr_full_T]
        test_satnav_T = [t[: T_target + 1] for t in test_satnav_full_T]
        test_all = test_intr_T + test_satnav_T
        labels = np.concatenate([np.zeros(n_t), np.ones(n_t)])
        s_fit = _scores_for_chains(test_all, PT_intr_h, PT_satnav_h)
        _, _, auc_fit = roc(s_fit, labels)
        all_aucs_fit[T_target].append(auc_fit)
        if all_aucs_emp[T_target] is None:
            s_emp = _scores_for_chains(test_all, PT_intr_emp, PT_satnav_emp)
            _, _, auc_emp = roc(s_emp, labels)
            all_aucs_emp[T_target] = auc_emp

# Summary.
print(f"\n=== Summary across {len(X_O_SEEDS)} X_o seeds ===")
print(f"  {'T':>3}  {'N_test':>7}  {'empirical':>10}  {'fitted_f (mean ± std)':>22}  {'gap closed (mean)':>18}")
print(f"  {'-'*3}  {'-'*7}  {'-'*10}  {'-'*22}  {'-'*18}")
for T_target in (20, 30, 40):
    n_t = per_T_test[T_target][2]
    emp = all_aucs_emp[T_target]
    fits = np.array(all_aucs_fit[T_target])
    gap_mean = (fits.mean() - 0.5) / (emp - 0.5) if emp > 0.5 else float("nan")
    print(f"  {T_target:>3}  {n_t:>7}  {emp:>10.3f}  "
          f"{fits.mean():>10.3f} ± {fits.std():>5.3f}      {100*gap_mean:>14.1f}%")
    print(f"          per-seed fitted_f: {['%.3f' % v for v in fits]}")
