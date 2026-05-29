"""Multi-start L-BFGS-B variant of score_seed_big.py.

Runs K independent fits per chain (intr and satnav) from different random
initial theta_0, picks the best by validation log-likelihood on a held-out
subset of training trajectories, then scores the test set as usual.

Compares the multi-start AUC to the single-init (theta_0=zeros) baseline.

Identical pipeline to score_seed_big.py: seed=42, x_o_seed=13, T=20,
maxiter=1500, bigger bbox; SUMO files pre-built.
"""
import argparse
import os
import sys
import time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.chdir(HERE)

from sumo_to_phase3 import (
    build_adj, read_edge_counts, read_trajectories, empirical_chain,
    extract_features,
)
from per_car_detector import _scores_for_chains, log_lik
from congestion_filter import roc
from morimura import Inverter, stationary


# ---------------------------------------------------------------
# Multi-start fitter
# ---------------------------------------------------------------

def fit_chain_with_init(adj, beta, X_o, f_obs, x0,
                        gamma=1.0, lam=1e-3, maxiter=1500,
                        phi_T=None, psi=None):
    """Single Inverter fit from a specific x0. Mirrors per_car_detector.fit_chain
    but exposes x0 and silences printing. Returns (pI_h, PT_h, pi_h, res, d)."""
    inv = Inverter(adj, beta, gamma=gamma, lam=lam, phi_T=phi_T, psi=psi)
    theta, res = inv.fit(X_o, f_obs, g_obs=None, x0=x0, maxiter=maxiter)
    pI_h, PT_h = inv.forward(theta)
    pi_h = stationary(beta * PT_h + (1.0 - beta) * pI_h[None, :])
    return pI_h, PT_h, pi_h, res, inv.d


def val_loglik(trajs, PT, eps=1e-15):
    """Sum of per-step log-likelihoods over a list of trajectories."""
    return float(sum(log_lik(t, PT, eps=eps) for t in trajs))


def multi_start_fit(adj, beta, X_o, f_obs, train_trajs, val_frac=0.2,
                    K=10, base_seed=0, scale=1.0, maxiter=1500,
                    gamma=1.0, lam=1e-3, label="",
                    phi_T=None, psi=None):
    """K independent Inverter fits, each from a different random theta_0.
    Selects the fit with the highest validation log-likelihood on a
    held-out val_frac slice of train_trajs.

    Returns dict with selected fit, all per-init diagnostics.
    """
    # Held-out validation split (deterministic given base_seed)
    rng_split = np.random.default_rng(base_seed * 100 + 17)
    perm = rng_split.permutation(len(train_trajs))
    n_val = max(1, int(round(val_frac * len(train_trajs))))
    val_trajs = [train_trajs[i] for i in perm[:n_val]]

    # Probe d via a zero-init Inverter (no fit)
    inv_probe = Inverter(adj, beta, gamma=gamma, lam=lam,
                         phi_T=phi_T, psi=psi)
    d = inv_probe.d

    results = []
    rng_inits = np.random.default_rng(base_seed)
    init_seeds = rng_inits.integers(0, 10**9, size=K)
    for k, s in enumerate(init_seeds):
        rng_k = np.random.default_rng(int(s))
        # Random normal init scaled by `scale`.
        x0 = scale * rng_k.standard_normal(d)
        t0 = time.time()
        pI_h, PT_h, pi_h, res, _ = fit_chain_with_init(
            adj, beta, X_o, f_obs, x0,
            gamma=gamma, lam=lam, maxiter=maxiter,
            phi_T=phi_T, psi=psi,
        )
        elapsed = time.time() - t0
        vloglik = val_loglik(val_trajs, PT_h)
        results.append({
            "init_seed": int(s),
            "x0_norm": float(np.linalg.norm(x0)),
            "final_loss": float(res.fun),
            "nit": int(res.nit),
            "success": bool(res.success),
            "val_loglik": vloglik,
            "elapsed": elapsed,
            "PT_h": PT_h,
            "pI_h": pI_h,
            "pi_h": pi_h,
        })
        print(f"  [{label} init {k:2d}/{K-1}] seed={s:>10} "
              f"loss={res.fun:.4g} nit={res.nit:>4} "
              f"val_loglik={vloglik:.3f} t={elapsed:.1f}s",
              flush=True)

    # Pick best by validation log-likelihood (higher is better)
    best_idx = int(np.argmax([r["val_loglik"] for r in results]))
    return {
        "best_idx": best_idx,
        "results": results,
        "n_val": n_val,
        "d": d,
    }


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

p = argparse.ArgumentParser()
p.add_argument("--dataset", choices=("sumo", "xuancheng"), default="sumo",
               help="'sumo' = bigger-bbox SUMO with --seed (historical default); "
                    "'xuancheng' = real-world Xuancheng AVI data with --day / --regime_split.")
p.add_argument("--seed", type=int, default=None,
               help="Required when --dataset sumo. SUMO/randomTrips seed.")
p.add_argument("--day", type=str, default=None,
               help="Required when --dataset xuancheng. ISO date e.g. 2023-04-17. "
                    "Comma-separated list for multi-day pooling.")
p.add_argument("--regime_split", choices=("rush_offpeak", "weekday_weekend",
                                          "holiday_normal"),
               default="rush_offpeak",
               help="Xuancheng only. How to partition trips into two regimes.")
p.add_argument("--od_match", type=int, default=None,
               help="Xuancheng only. If set, OD-match the two regime pools by "
                    "k-means-clustering junctions into K zones then subsampling "
                    "per (zone_o, zone_d) cell. Removes OD-mix confound. "
                    "Recommended: K=8.")
p.add_argument("--x_o_seed", type=int, default=13)
p.add_argument("--T", type=int, default=20)
p.add_argument("--maxiter", type=int, default=1500)
p.add_argument("--K", type=int, default=10, help="Number of restarts")
p.add_argument("--val_frac", type=float, default=0.2)
p.add_argument("--init_scale", type=float, default=1.0,
               help="Std-dev of random normal init for theta_0")
p.add_argument("--ms_base_seed", type=int, default=20260526)
p.add_argument("--features", choices=("none", "real"), default="none",
               help="'none' = local-only fit; "
                    "'real' = enable omega-globals via SUMO edge attrs.")
args = p.parse_args()

print(f"\n=== Multi-start L-BFGS-B variant (K={args.K}, "
      f"val_frac={args.val_frac}, init_scale={args.init_scale}) ===")

if args.dataset == "sumo":
    if args.seed is None:
        p.error("--seed is required when --dataset sumo")
    S = args.seed
    print(f"dataset=sumo  seed={S}  x_o_seed={args.x_o_seed}  T={args.T}  "
          f"maxiter={args.maxiter}")
    edges, adj_out, idx = build_adj("net.net.xml")
    f_intr = read_edge_counts(f"edgedata.intr.big.seed{S}.xml", idx)
    f_satnav = read_edge_counts(f"edgedata.satnav.big.seed{S}.xml", idx)
    trajs_intr, _ = read_trajectories(f"vehroutes.intr.big.seed{S}.xml", idx)
    trajs_satnav, _ = read_trajectories(f"vehroutes.satnav.big.seed{S}.xml", idx)
    net_path_for_features = "net.net.xml"
    regime_a, regime_b = "intr", "satnav"
else:  # xuancheng
    if args.day is None:
        p.error("--day is required when --dataset xuancheng (e.g. 2023-04-17)")
    import datetime
    import sys as _sys
    _REAL_DATA = os.path.join(os.path.dirname(HERE), "real_data")
    _sys.path.insert(0, _REAL_DATA)
    from load_xuancheng import (  # noqa: E402
        load_xuancheng_regime, day_spec,
        split_rush_vs_offpeak, split_weekday_vs_weekend,
        split_holiday_vs_normal, NET_PATH_DEFAULT,
    )
    date_strs = [s.strip() for s in args.day.split(",") if s.strip()]
    dates = [datetime.date.fromisoformat(s) for s in date_strs]
    split_map = {
        "rush_offpeak":    (lambda st, d: split_rush_vs_offpeak(st["start_time"]),
                            "rush", "offpeak"),
        "weekday_weekend": (lambda st, d: split_weekday_vs_weekend(d),
                            "weekday", "weekend"),
        "holiday_normal":  (lambda st, d: split_holiday_vs_normal(d),
                            "holiday", "normal"),
    }
    rs, label_a, label_b = split_map[args.regime_split]
    S = args.day  # used only for log printing
    print(f"dataset=xuancheng  days={','.join(date_strs)}  "
          f"split={args.regime_split}  od_match={args.od_match}  "
          f"x_o_seed={args.x_o_seed}  T={args.T}  maxiter={args.maxiter}")
    edges, adj_out, idx, trajs_intr, trajs_satnav, f_intr, f_satnav = (
        load_xuancheng_regime(
            net_path=NET_PATH_DEFAULT,
            day_specs=[day_spec(d) for d in dates],
            regime_split=rs,
            label_a=label_a, label_b=label_b,
            min_segment_len=5,
            verbose=True,
            od_match_zones=args.od_match,
        )
    )
    net_path_for_features = NET_PATH_DEFAULT
    regime_a, regime_b = label_a, label_b

n = len(edges)

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

print(f"n={n}  |X_o|={len(X_o)}  beta={beta:.3f}  "
      f"#train_intr={len(train_intr)}  #train_satnav={len(train_satnav)}  "
      f"#test_per_class={n_t}")

if args.features == "real":
    phi_T, psi = extract_features(net_path_for_features, edges, adj_out)
    d_T, d_psi = phi_T.shape[1], psi.shape[1]
else:
    phi_T, psi = None, None
    d_T, d_psi = 0, 0
print(f"features={args.features}  d_T={d_T}  d_psi={d_psi}")

# Probe d
inv_probe = Inverter(adj_out, beta, gamma=1.0, lam=1e-3,
                     phi_T=phi_T, psi=psi)
d_full = inv_probe.d
print(f"theta dimension d = n + E + d_T + d_psi = {d_full}")

# ---------------------------------------------------------------
# 1) Baseline: single zero-init fit (identical to score_seed_big.py)
# ---------------------------------------------------------------
print("\n--- Baseline single-init fits (theta_0 = zeros) ---")
t0 = time.time()
_, PT_intr_b, _, res_b_i, _ = fit_chain_with_init(
    adj_out, beta, X_o, f_intr[X_o],
    x0=np.zeros(d_full), maxiter=args.maxiter,
    phi_T=phi_T, psi=psi,
)
_, PT_satnav_b, _, res_b_s, _ = fit_chain_with_init(
    adj_out, beta, X_o, f_satnav[X_o],
    x0=np.zeros(d_full), maxiter=args.maxiter,
    phi_T=phi_T, psi=psi,
)
t_single = time.time() - t0
print(f"  baseline intr   loss={res_b_i.fun:.4g} nit={res_b_i.nit}")
print(f"  baseline satnav loss={res_b_s.fun:.4g} nit={res_b_s.nit}")
print(f"  baseline wall-clock (2 fits): {t_single:.1f}s")

PT_intr_emp = empirical_chain(train_intr, adj_out)
PT_satnav_emp = empirical_chain(train_satnav, adj_out)

test_all = test_intr + test_satnav
labels = np.concatenate([np.zeros(n_t), np.ones(n_t)])
_, _, auc_emp = roc(_scores_for_chains(test_all, PT_intr_emp, PT_satnav_emp), labels)
_, _, auc_baseline = roc(
    _scores_for_chains(test_all, PT_intr_b, PT_satnav_b), labels,
)
gap_base = (auc_baseline - 0.5) / (auc_emp - 0.5) if auc_emp > 0.5 else float("nan")

print(f"\n  BASELINE  empirical={auc_emp:.4f}  "
      f"fitted_f={auc_baseline:.4f}  gap_closed={100*gap_base:.1f}%")

# Validation log-likelihood of the baseline zero-init fit (for comparison)
rng_split_intr = np.random.default_rng(args.ms_base_seed * 100 + 17)
perm_i = rng_split_intr.permutation(len(train_intr))
n_val_i = max(1, int(round(args.val_frac * len(train_intr))))
val_trajs_i = [train_intr[k] for k in perm_i[:n_val_i]]
val_ll_baseline_intr = val_loglik(val_trajs_i, PT_intr_b)

rng_split_sat = np.random.default_rng((args.ms_base_seed + 1) * 100 + 17)
perm_s = rng_split_sat.permutation(len(train_satnav))
n_val_s = max(1, int(round(args.val_frac * len(train_satnav))))
val_trajs_s = [train_satnav[k] for k in perm_s[:n_val_s]]
val_ll_baseline_satnav = val_loglik(val_trajs_s, PT_satnav_b)
print(f"  baseline val_loglik:  intr={val_ll_baseline_intr:.3f}  "
      f"satnav={val_ll_baseline_satnav:.3f}")

# ---------------------------------------------------------------
# 2) Multi-start fits per chain
# ---------------------------------------------------------------
print(f"\n--- Multi-start intr (K={args.K}) ---")
t0 = time.time()
ms_intr = multi_start_fit(
    adj_out, beta, X_o, f_intr[X_o], train_intr,
    val_frac=args.val_frac, K=args.K,
    base_seed=args.ms_base_seed,
    scale=args.init_scale,
    maxiter=args.maxiter,
    label="intr",
    phi_T=phi_T, psi=psi,
)
t_ms_intr = time.time() - t0
print(f"  multi-start intr wall-clock: {t_ms_intr:.1f}s "
      f"({t_ms_intr/args.K:.1f}s per fit avg)")

print(f"\n--- Multi-start satnav (K={args.K}) ---")
t0 = time.time()
ms_satnav = multi_start_fit(
    adj_out, beta, X_o, f_satnav[X_o], train_satnav,
    val_frac=args.val_frac, K=args.K,
    base_seed=args.ms_base_seed + 1,
    scale=args.init_scale,
    maxiter=args.maxiter,
    label="satnav",
    phi_T=phi_T, psi=psi,
)
t_ms_satnav = time.time() - t0
print(f"  multi-start satnav wall-clock: {t_ms_satnav:.1f}s "
      f"({t_ms_satnav/args.K:.1f}s per fit avg)")

# ---------------------------------------------------------------
# 3) Score test AUC at selected, and across inits to measure basin variance
# ---------------------------------------------------------------
best_intr = ms_intr["results"][ms_intr["best_idx"]]
best_satnav = ms_satnav["results"][ms_satnav["best_idx"]]

auc_ms = roc(
    _scores_for_chains(test_all, best_intr["PT_h"], best_satnav["PT_h"]),
    labels,
)[2]
gap_ms = (auc_ms - 0.5) / (auc_emp - 0.5) if auc_emp > 0.5 else float("nan")

# All pairwise test AUCs (10 intr inits crossed with the selected satnav)
auc_per_intr_init = []
for r in ms_intr["results"]:
    a = roc(_scores_for_chains(test_all, r["PT_h"], best_satnav["PT_h"]),
            labels)[2]
    auc_per_intr_init.append(a)
auc_per_satnav_init = []
for r in ms_satnav["results"]:
    a = roc(_scores_for_chains(test_all, best_intr["PT_h"], r["PT_h"]),
            labels)[2]
    auc_per_satnav_init.append(a)

# Per-init AUC using each init paired with its *own* counterpart by index
auc_per_paired = []
for r_i, r_s in zip(ms_intr["results"], ms_satnav["results"]):
    a = roc(_scores_for_chains(test_all, r_i["PT_h"], r_s["PT_h"]),
            labels)[2]
    auc_per_paired.append(a)

# ---------------------------------------------------------------
# 4) Summary print
# ---------------------------------------------------------------
print("\n" + "=" * 72)
print("SUMMARY")
print("=" * 72)
print(f"empirical AUC        : {auc_emp:.4f}")
print(f"baseline (zeros init): fitted_f AUC = {auc_baseline:.4f}  "
      f"gap_closed = {100*gap_base:5.1f}%")
print(f"multi-start (K={args.K}, val-select): "
      f"fitted_f AUC = {auc_ms:.4f}  gap_closed = {100*gap_ms:5.1f}%")
print()
print("Per-init validation log-likelihoods (sorted desc):")
val_lls_intr = [r["val_loglik"] for r in ms_intr["results"]]
val_lls_satnav = [r["val_loglik"] for r in ms_satnav["results"]]
final_loss_intr = [r["final_loss"] for r in ms_intr["results"]]
final_loss_satnav = [r["final_loss"] for r in ms_satnav["results"]]
order_i = np.argsort(val_lls_intr)[::-1]
order_s = np.argsort(val_lls_satnav)[::-1]
print(f"  intr   (selected idx={ms_intr['best_idx']}, n_val={ms_intr['n_val']}, "
      f"baseline val_ll={val_ll_baseline_intr:.3f}):")
for rank, k in enumerate(order_i):
    mark = " *" if k == ms_intr["best_idx"] else "  "
    print(f"   {mark} rank {rank:2d}  init {k:2d}  "
          f"val_ll={val_lls_intr[k]:>12.3f}  "
          f"final_loss={final_loss_intr[k]:.4g}")
print(f"  satnav (selected idx={ms_satnav['best_idx']}, "
      f"n_val={ms_satnav['n_val']}, "
      f"baseline val_ll={val_ll_baseline_satnav:.3f}):")
for rank, k in enumerate(order_s):
    mark = " *" if k == ms_satnav["best_idx"] else "  "
    print(f"   {mark} rank {rank:2d}  init {k:2d}  "
          f"val_ll={val_lls_satnav[k]:>12.3f}  "
          f"final_loss={final_loss_satnav[k]:.4g}")

print()
print("Per-init test AUCs (intr init varied, satnav at selected best):")
print("  " + ", ".join(f"{a:.3f}" for a in auc_per_intr_init))
print(f"  range = [{min(auc_per_intr_init):.3f}, "
      f"{max(auc_per_intr_init):.3f}]  "
      f"mean={np.mean(auc_per_intr_init):.3f}  "
      f"std={np.std(auc_per_intr_init):.3f}")

print("Per-init test AUCs (satnav init varied, intr at selected best):")
print("  " + ", ".join(f"{a:.3f}" for a in auc_per_satnav_init))
print(f"  range = [{min(auc_per_satnav_init):.3f}, "
      f"{max(auc_per_satnav_init):.3f}]  "
      f"mean={np.mean(auc_per_satnav_init):.3f}  "
      f"std={np.std(auc_per_satnav_init):.3f}")

print("Per-init test AUCs (intr[k] paired with satnav[k]):")
print("  " + ", ".join(f"{a:.3f}" for a in auc_per_paired))
print(f"  range = [{min(auc_per_paired):.3f}, "
      f"{max(auc_per_paired):.3f}]  "
      f"mean={np.mean(auc_per_paired):.3f}  "
      f"std={np.std(auc_per_paired):.3f}")

print()
print(f"Wall-clock: single (2 fits) = {t_single:.1f}s, "
      f"multi-start total = {t_ms_intr + t_ms_satnav:.1f}s "
      f"(intr {t_ms_intr:.1f}s + satnav {t_ms_satnav:.1f}s) "
      f"-- ratio {(t_ms_intr + t_ms_satnav)/max(t_single, 1e-6):.1f}x")
print("=" * 72)
