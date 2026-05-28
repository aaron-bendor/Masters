"""Score the detector on one bigger-bbox SUMO-seed-tagged dataset.

Reads vehroutes/edgedata.{intr,satnav}.big.seed<N>.xml from net.net.xml.
Fits f-only chains (local-only, no globals — matches iter-11/iter-13 protocol)
at maxiter=1500 (matches iter-15 small-bbox protocol).

Usage:
  ../.venv/bin/python score_seed_big.py --seed 23
  for S in 23 7 42 101 2024; do ../.venv/bin/python score_seed_big.py --seed $S; done
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
p.add_argument("--dataset", choices=("sumo", "xuancheng"), default="sumo",
               help="'sumo' = bigger-bbox SUMO with --seed; "
                    "'xuancheng' = real-world Xuancheng AVI data with --day / --regime_split.")
p.add_argument("--seed", type=int, default=None,
               help="Required when --dataset sumo. SUMO/randomTrips seed.")
p.add_argument("--day", type=str, default=None,
               help="Required when --dataset xuancheng. ISO date e.g. 2023-04-17. "
                    "Multi-day pool: comma-separated list e.g. 2023-04-17,2023-04-18.")
p.add_argument("--regime_split", choices=("rush_offpeak", "weekday_weekend",
                                          "holiday_normal"),
               default="rush_offpeak",
               help="Xuancheng only. How to partition trips into two regimes.")
p.add_argument("--od_match", type=int, default=None,
               help="Xuancheng only. If set, OD-match the two regime pools by "
                    "k-means-clustering junctions into K zones then "
                    "subsampling per (zone_o, zone_d) cell to equalise counts.")
p.add_argument("--x_o_seed", type=int, default=13)
p.add_argument("--T", type=int, default=20)
p.add_argument("--maxiter", type=int, default=1500)
p.add_argument("--features", choices=("none", "real"), default="none",
               help="'none' = local-only fit (historical default); "
                    "'real' = enable omega-globals using SUMO edge attributes "
                    "(road class, log speed, log lanes, turn cosine, log speed ratio).")
args = p.parse_args()

if args.dataset == "sumo":
    if args.seed is None:
        p.error("--seed is required when --dataset sumo")
    S = args.seed
    edges, adj_out, idx = build_adj("net.net.xml")
    f_intr = read_edge_counts(f"edgedata.intr.big.seed{S}.xml", idx)
    f_satnav = read_edge_counts(f"edgedata.satnav.big.seed{S}.xml", idx)
    trajs_intr, stats_i = read_trajectories(f"vehroutes.intr.big.seed{S}.xml", idx)
    trajs_satnav, stats_s = read_trajectories(f"vehroutes.satnav.big.seed{S}.xml", idx)
    net_path_for_features = "net.net.xml"
    regime_a, regime_b = "intr", "satnav"
else:  # xuancheng
    if args.day is None:
        p.error("--day is required when --dataset xuancheng (e.g. 2023-04-17)")
    import datetime
    _REAL_DATA = os.path.join(os.path.dirname(HERE), "real_data")
    sys.path.insert(0, _REAL_DATA)
    from load_xuancheng import (  # noqa: E402
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
    S = args.day  # for log printing only
    # Multi-day support: --day can be a comma-separated list of ISO dates.
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
    d_T, d_psi = phi_T.shape[1], psi.shape[1]
else:
    phi_T, psi = None, None
    d_T, d_psi = 0, 0

_, PT_intr_h, _ = fit_chain(adj_out, beta, X_o, f_intr[X_o], maxiter=args.maxiter,
                            phi_T=phi_T, psi=psi)
_, PT_satnav_h, _ = fit_chain(adj_out, beta, X_o, f_satnav[X_o], maxiter=args.maxiter,
                              phi_T=phi_T, psi=psi)
PT_intr_emp = empirical_chain(train_intr, adj_out)
PT_satnav_emp = empirical_chain(train_satnav, adj_out)

test_all = test_intr + test_satnav
labels = np.concatenate([np.zeros(n_t), np.ones(n_t)])
_, _, auc_fit = roc(_scores_for_chains(test_all, PT_intr_h, PT_satnav_h), labels)
_, _, auc_emp = roc(_scores_for_chains(test_all, PT_intr_emp, PT_satnav_emp), labels)
gap = (auc_fit - 0.5) / (auc_emp - 0.5) if auc_emp > 0.5 else float("nan")

print(
    f"RESULT  dataset={args.dataset}  seed_or_day={S}  "
    f"regimes={regime_a}/{regime_b}  T={T}  N_test={n_t:>3}  "
    f"|X_o|={len(X_o)}  n={n}  beta={beta:.3f}  "
    f"features={args.features}  d_T={d_T}  d_psi={d_psi}  "
    f"od_match={args.od_match}  "
    f"empirical={auc_emp:.3f}  fitted_f={auc_fit:.3f}  gap_closed={100*gap:6.1f}%"
)
