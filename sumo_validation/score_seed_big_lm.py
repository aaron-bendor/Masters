"""LM variant of score_seed_big.py.

Same pipeline (seed=42, x_o_seed=13, T=20, bigger-bbox SUMO files) but
uses scipy.optimize.least_squares (trust-region-reflective) on the
analytic residual vector + Jacobian (lm_fit.py) instead of L-BFGS-B.

Also rebuilds the L-BFGS-B baseline at the same seed so the two are
compared on identical X_o, beta, train/test splits.
"""
import argparse, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.chdir(HERE)

from sumo_to_phase3 import (
    build_adj, read_edge_counts, read_trajectories, empirical_chain,
)
from per_car_detector import fit_chain, _scores_for_chains
from congestion_filter import roc

from lm_fit import fit_lm, fd_check_jacobian, _residuals_and_J
from morimura import Inverter

p = argparse.ArgumentParser()
p.add_argument("--seed", type=int, default=42)
p.add_argument("--x_o_seed", type=int, default=13)
p.add_argument("--T", type=int, default=20)
p.add_argument("--maxiter", type=int, default=1500)
p.add_argument("--skip_baseline", action="store_true")
args = p.parse_args()

S = args.seed
print(f"=== score_seed_big_lm  seed={S} ===")
edges, adj_out, idx = build_adj("net.net.xml")
n = len(edges)
print(f"n_edges={n}")
f_intr = read_edge_counts(f"edgedata.intr.big.seed{S}.xml", idx)
f_satnav = read_edge_counts(f"edgedata.satnav.big.seed{S}.xml", idx)
trajs_intr, stats_i = read_trajectories(f"vehroutes.intr.big.seed{S}.xml", idx)
trajs_satnav, stats_s = read_trajectories(f"vehroutes.satnav.big.seed{S}.xml", idx)

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
print(f"|X_o|={len(X_o)}  beta={beta:.3f}")

# --------------------------------------------------------------
# 1) FD-check the residual Jacobian on this exact problem
# --------------------------------------------------------------
# Use eps=1e-3 - n=2405 dense stationary solves contribute ~O(n * eps_mach
# / eps) round-off noise to central differences. Empirically the sweet
# spot for this network is eps in [1e-4, 1e-3]; eps=1e-6 (the small-n
# default) is dominated by round-off and gives misleading rel_err ~ 1e-4.
print("\n[FD check on real SUMO instance, f_intr, eps=1e-3]")
rel_err, abs_err, _, _ = fd_check_jacobian(
    adj_out, beta, X_o, f_intr[X_o], lam=1e-3, eps=1e-3, seed=1,
    max_cols=10,
)
print(f"  rel_err={rel_err:.3e}   abs_err={abs_err:.3e}")
if rel_err > 1e-5:
    print(f"  !!! FD check FAILED (rel_err={rel_err:.2e}); aborting. !!!")
    sys.exit(1)
print(f"  PASS")

# --------------------------------------------------------------
# 2) Baseline L-BFGS-B run (matches the original score_seed_big.py)
# --------------------------------------------------------------
if not args.skip_baseline:
    print("\n[Baseline L-BFGS-B]")
    t0 = time.time()
    inv_b = Inverter(adj_out, beta, gamma=1.0, lam=1e-3)
    x0 = np.zeros(inv_b.d)
    log_f_b = np.log(np.asarray(f_intr[X_o], dtype=float))
    import scipy.optimize as sopt
    res_b_intr = sopt.minimize(
        inv_b.loss_grad, x0, args=(X_o, log_f_b, None),
        jac=True, method="L-BFGS-B",
        options=dict(maxiter=args.maxiter, ftol=1e-9, gtol=1e-7),
    )
    pI_b_i, PT_b_i = inv_b.forward(res_b_intr.x)
    t_intr_b = time.time() - t0

    t1 = time.time()
    log_f_s = np.log(np.asarray(f_satnav[X_o], dtype=float))
    res_b_sat = sopt.minimize(
        inv_b.loss_grad, x0, args=(X_o, log_f_s, None),
        jac=True, method="L-BFGS-B",
        options=dict(maxiter=args.maxiter, ftol=1e-9, gtol=1e-7),
    )
    pI_b_s, PT_b_s = inv_b.forward(res_b_sat.x)
    t_sat_b = time.time() - t1

    print(f"  intr:   loss={res_b_intr.fun:.6g}  nit={res_b_intr.nit}  "
          f"nfev={res_b_intr.nfev}  success={res_b_intr.success}  "
          f"t={t_intr_b:.1f}s")
    print(f"  satnav: loss={res_b_sat.fun:.6g}  nit={res_b_sat.nit}  "
          f"nfev={res_b_sat.nfev}  success={res_b_sat.success}  "
          f"t={t_sat_b:.1f}s")
    baseline_intr_loss = res_b_intr.fun
    baseline_sat_loss = res_b_sat.fun
    baseline_nit_i = res_b_intr.nit
    baseline_nit_s = res_b_sat.nit
    baseline_nfev_i = res_b_intr.nfev
    baseline_nfev_s = res_b_sat.nfev
    baseline_t_i = t_intr_b
    baseline_t_s = t_sat_b
    PT_intr_h_b = PT_b_i
    PT_satnav_h_b = PT_b_s
    theta_b_intr = res_b_intr.x
    theta_b_sat = res_b_sat.x
else:
    PT_intr_h_b = None
    PT_satnav_h_b = None

# --------------------------------------------------------------
# 3) LM fit
# --------------------------------------------------------------
print("\n[LM via least_squares(method='trf')]")
t0 = time.time()
theta_lm_i, PT_intr_h_lm, _, info_i = fit_lm(
    adj_out, beta, X_o, f_intr[X_o], lam=1e-3, verbose=2,
    ftol=1e-9, xtol=1e-9, gtol=1e-7,
)
t_intr_lm = time.time() - t0
print(f"  intr:   loss={info_i['final_loss']:.6g}  "
      f"nfev={info_i['nfev']}  njev={info_i['njev']}  "
      f"status={info_i['status']}  t={t_intr_lm:.1f}s")
print(f"          msg={info_i['message']!r}")

t1 = time.time()
theta_lm_s, PT_satnav_h_lm, _, info_s = fit_lm(
    adj_out, beta, X_o, f_satnav[X_o], lam=1e-3, verbose=2,
    ftol=1e-9, xtol=1e-9, gtol=1e-7,
)
t_sat_lm = time.time() - t1
print(f"  satnav: loss={info_s['final_loss']:.6g}  "
      f"nfev={info_s['nfev']}  njev={info_s['njev']}  "
      f"status={info_s['status']}  t={t_sat_lm:.1f}s")
print(f"          msg={info_s['message']!r}")

# --------------------------------------------------------------
# 4) Score AUCs
# --------------------------------------------------------------
PT_intr_emp = empirical_chain(train_intr, adj_out)
PT_satnav_emp = empirical_chain(train_satnav, adj_out)

test_all = test_intr + test_satnav
labels = np.concatenate([np.zeros(n_t), np.ones(n_t)])
_, _, auc_emp = roc(_scores_for_chains(test_all, PT_intr_emp, PT_satnav_emp), labels)
_, _, auc_lm = roc(_scores_for_chains(test_all, PT_intr_h_lm, PT_satnav_h_lm), labels)
auc_b = None
if not args.skip_baseline:
    _, _, auc_b = roc(_scores_for_chains(test_all, PT_intr_h_b, PT_satnav_h_b), labels)

gap_lm = (auc_lm - 0.5) / (auc_emp - 0.5) if auc_emp > 0.5 else float("nan")
print(f"\nempirical    AUC = {auc_emp:.4f}")
if auc_b is not None:
    gap_b = (auc_b - 0.5) / (auc_emp - 0.5) if auc_emp > 0.5 else float("nan")
    print(f"L-BFGS-B     AUC = {auc_b:.4f}   gap_closed={100*gap_b:6.2f}%")
    print(f"                  intr_loss={baseline_intr_loss:.6g}   "
          f"sat_loss={baseline_sat_loss:.6g}")
    print(f"                  nit (intr/sat) = {baseline_nit_i}/{baseline_nit_s}   "
          f"nfev = {baseline_nfev_i}/{baseline_nfev_s}   "
          f"time = {baseline_t_i:.1f}s/{baseline_t_s:.1f}s")
print(f"LM (trf)     AUC = {auc_lm:.4f}   gap_closed={100*gap_lm:6.2f}%")
print(f"                  intr_loss={info_i['final_loss']:.6g}   "
      f"sat_loss={info_s['final_loss']:.6g}")
print(f"                  nfev (intr/sat) = {info_i['nfev']}/{info_s['nfev']}   "
      f"njev = {info_i['njev']}/{info_s['njev']}   "
      f"time = {t_intr_lm:.1f}s/{t_sat_lm:.1f}s")

# Diagnostic: theta L2 distance and PT distance
if not args.skip_baseline:
    dtheta_i = float(np.linalg.norm(theta_lm_i - theta_b_intr))
    dtheta_s = float(np.linalg.norm(theta_lm_s - theta_b_sat))
    dPT_i = float(np.linalg.norm(PT_intr_h_lm - PT_intr_h_b))
    dPT_s = float(np.linalg.norm(PT_satnav_h_lm - PT_satnav_h_b))
    print(f"\nTheta L2 dist (LM vs L-BFGS-B):  intr={dtheta_i:.3e}   "
          f"sat={dtheta_s:.3e}")
    print(f"PT Frobenius dist     :  intr={dPT_i:.3e}   sat={dPT_s:.3e}")
