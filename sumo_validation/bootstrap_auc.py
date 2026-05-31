"""Bootstrap confidence intervals for the Tier-3 (and Tier-2) detection AUC.

Reuses the EXACT data-loading and chain-fitting protocol of score_seed_big.py,
then puts a stratified vehicle-level bootstrap CI around both the empirical
("full-observation 1-step reference") AUC and the fitted-chain AUC.

The headline Tier-3 question is: the OD-matched empirical AUC is 0.518, only
0.018 above chance (0.5). Is that margin distinguishable from sampling noise?
This script answers it by resampling the test trajectories with replacement
(stratified within each regime to preserve class balance) B times and
recomputing the AUC each time. The 2.5/97.5 percentiles give a 95% CI; if the
CI excludes 0.500 the margin is a real (if tiny) signal, if it straddles 0.500
it is indistinguishable from chance.

Usage (matches the headline OD-matched pooled row of Table 3):
  ../.venv/bin/python bootstrap_auc.py --dataset xuancheng \
      --day 2023-04-17,2023-04-18,2023-04-19,2023-04-20,2023-04-21 \
      --regime_split rush_offpeak --od_match 8 --T 10 --B 2000

Drop --od_match to get the un-matched pooled CI (ceiling 0.573).
Use --dataset sumo --seed 42 to get a Tier-2 reference CI.
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
p.add_argument("--dataset", choices=("sumo", "xuancheng"), default="xuancheng")
p.add_argument("--seed", type=int, default=None)
p.add_argument("--day", type=str, default=None)
p.add_argument("--regime_split", choices=("rush_offpeak", "weekday_weekend",
                                          "holiday_normal"),
               default="rush_offpeak")
p.add_argument("--od_match", type=int, default=None)
p.add_argument("--x_o_seed", type=int, default=13)
p.add_argument("--T", type=int, default=10)
p.add_argument("--maxiter", type=int, default=1500)
p.add_argument("--features", choices=("none", "real"), default="none")
p.add_argument("--B", type=int, default=2000, help="number of bootstrap resamples")
p.add_argument("--boot_seed", type=int, default=7, help="bootstrap RNG seed")
args = p.parse_args()

# ---- identical setup to score_seed_big.py --------------------------------
if args.dataset == "sumo":
    if args.seed is None:
        p.error("--seed is required when --dataset sumo")
    S = args.seed
    edges, adj_out, idx = build_adj("net.net.xml")
    f_intr = read_edge_counts(f"edgedata.intr.big.seed{S}.xml", idx)
    f_satnav = read_edge_counts(f"edgedata.satnav.big.seed{S}.xml", idx)
    trajs_intr, _ = read_trajectories(f"vehroutes.intr.big.seed{S}.xml", idx)
    trajs_satnav, _ = read_trajectories(f"vehroutes.satnav.big.seed{S}.xml", idx)
    net_path_for_features = "net.net.xml"
    regime_a, regime_b = "intr", "satnav"
else:
    if args.day is None:
        p.error("--day is required when --dataset xuancheng (e.g. 2023-04-17)")
    import datetime
    _REAL_DATA = os.path.join(os.path.dirname(HERE), "real_data")
    sys.path.insert(0, _REAL_DATA)
    from load_xuancheng import (
        load_xuancheng_regime, day_spec,
        split_rush_vs_offpeak, split_weekday_vs_weekend,
        split_holiday_vs_normal, NET_PATH_DEFAULT,
    )
    split_map = {
        "rush_offpeak":    (lambda st, d: split_rush_vs_offpeak(st["start_time"]),
                            "rush", "offpeak"),
        "weekday_weekend": (lambda st, d: split_weekday_vs_weekend(d),
                            "weekday", "weekend"),
        "holiday_normal":  (lambda st, d: split_holiday_vs_normal(d),
                            "holiday", "normal"),
    }
    rs, regime_a, regime_b = split_map[args.regime_split]
    S = args.day
    date_strs = [s.strip() for s in args.day.split(",") if s.strip()]
    dates = [datetime.date.fromisoformat(s) for s in date_strs]
    edges, adj_out, idx, trajs_intr, trajs_satnav, f_intr, f_satnav = (
        load_xuancheng_regime(
            net_path=NET_PATH_DEFAULT,
            day_specs=[day_spec(d) for d in dates],
            regime_split=rs,
            label_a=regime_a, label_b=regime_b,
            min_segment_len=5,
            verbose=True,
            od_match_zones=args.od_match,
        )
    )
    net_path_for_features = NET_PATH_DEFAULT
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

if args.features == "real":
    phi_T, psi = extract_features(net_path_for_features, edges, adj_out)
else:
    phi_T, psi = None, None

_, PT_intr_h, _ = fit_chain(adj_out, beta, X_o, f_intr[X_o], maxiter=args.maxiter,
                            phi_T=phi_T, psi=psi)
_, PT_satnav_h, _ = fit_chain(adj_out, beta, X_o, f_satnav[X_o], maxiter=args.maxiter,
                              phi_T=phi_T, psi=psi)
PT_intr_emp = empirical_chain(train_intr, adj_out)
PT_satnav_emp = empirical_chain(train_satnav, adj_out)

# ---- per-trajectory scores computed ONCE ---------------------------------
test_all = test_intr + test_satnav
labels = np.concatenate([np.zeros(n_t), np.ones(n_t)])
s_fit = _scores_for_chains(test_all, PT_intr_h, PT_satnav_h)
s_emp = _scores_for_chains(test_all, PT_intr_emp, PT_satnav_emp)

auc_fit_pt = roc(s_fit, labels)[2]
auc_emp_pt = roc(s_emp, labels)[2]

# ---- stratified vehicle-level bootstrap ----------------------------------
rng_b = np.random.default_rng(args.boot_seed)
idx_a = np.arange(n_t)            # regime A (label 0) positions
idx_b = np.arange(n_t, 2 * n_t)   # regime B (label 1) positions
boot_emp = np.empty(args.B)
boot_fit = np.empty(args.B)
for b in range(args.B):
    ra = rng_b.choice(idx_a, size=n_t, replace=True)
    rb = rng_b.choice(idx_b, size=n_t, replace=True)
    sel = np.concatenate([ra, rb])
    lab = labels[sel]
    boot_emp[b] = roc(s_emp[sel], lab)[2]
    boot_fit[b] = roc(s_fit[sel], lab)[2]


def report(name, point, boot):
    lo, hi = np.percentile(boot, [2.5, 97.5])
    p_above = float(np.mean(boot > 0.5))
    excl = "EXCLUDES 0.5" if lo > 0.5 or hi < 0.5 else "straddles 0.5"
    print(f"  {name:9s} point={point:.3f}  "
          f"95% CI=[{lo:.3f}, {hi:.3f}]  ({excl})  "
          f"P(AUC>0.5)={p_above:.3f}")


print(
    f"\nBOOTSTRAP  dataset={args.dataset}  seed_or_day={S}  "
    f"regimes={regime_a}/{regime_b}  T={T}  N_test_per_class={n_t}  "
    f"|X_o|={len(X_o)}  features={args.features}  od_match={args.od_match}  "
    f"B={args.B}"
)
report("empirical", auc_emp_pt, boot_emp)
report("fitted_f", auc_fit_pt, boot_fit)
