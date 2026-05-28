"""Contrast-correlation diagnostic for the LR detector.

The detector classifies trajectories using
  Lambda(tau) = sum_t [log PT_sat_h(y|x) - log PT_intr_h(y|x)],
so the only object that matters is the *contrast*
  Delta_fit(x, :) = PT_sat_h(x, :) - PT_intr_h(x, :).
If Delta_fit doesn't track the contrast we'd see under full observation
  Delta_emp(x, :) = PT_sat_emp(x, :) - PT_intr_emp(x, :),
then any AUC the detector reports is coming from the wrong place.

This script replicates score_seed_big.py's fit, then computes:
  - per-row cosine similarity between Delta_fit and Delta_emp at branching
    states (out_deg > 1), restricted to legal out-neighbours.
  - stacked Pearson and stacked cosine over all legal off-diagonal entries.
  - the same diagnostics restricted to branching states in X_o (states the
    fit directly observed) vs. all branching states (generalisation).

Usage:
  ../.venv/bin/python contrast_correlation.py --seed 23
  for S in 23 7 42 101 2024; do ../.venv/bin/python contrast_correlation.py --seed $S; done
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
from score_seed_big_fg import (
    empirical_g_counts, fit_beta_mom, shrink_g_beta_binomial,
)


def contrast_rows(PT_a_h, PT_b_h, PT_a_emp, PT_b_emp, adj_out, states):
    """Return (delta_fit, delta_emp) stacked over legal (x -> y) pairs for
    each state x in `states` that is a branching state. Each is a 1-D
    vector of length sum(out_deg(x)) over the included states."""
    fit_parts, emp_parts = [], []
    n_branching_included = 0
    for x in states:
        nbrs = adj_out[x]
        if len(nbrs) <= 1:
            continue
        nbrs = np.array(nbrs, dtype=int)
        d_fit = PT_b_h[x, nbrs] - PT_a_h[x, nbrs]
        d_emp = PT_b_emp[x, nbrs] - PT_a_emp[x, nbrs]
        fit_parts.append(d_fit)
        emp_parts.append(d_emp)
        n_branching_included += 1
    if not fit_parts:
        return np.array([]), np.array([]), 0
    return (np.concatenate(fit_parts),
            np.concatenate(emp_parts),
            n_branching_included)


def per_row_cosines(PT_a_h, PT_b_h, PT_a_emp, PT_b_emp, adj_out, states,
                    eps=1e-12):
    """Cosine similarity of Delta_fit(x, :) vs Delta_emp(x, :) for each
    branching state x in `states`. Returns array of cosines and the matching
    state ids."""
    cosines, kept = [], []
    for x in states:
        nbrs = adj_out[x]
        if len(nbrs) <= 1:
            continue
        nbrs = np.array(nbrs, dtype=int)
        d_fit = PT_b_h[x, nbrs] - PT_a_h[x, nbrs]
        d_emp = PT_b_emp[x, nbrs] - PT_a_emp[x, nbrs]
        nf, ne = np.linalg.norm(d_fit), np.linalg.norm(d_emp)
        if nf < eps or ne < eps:
            continue
        cosines.append(float(d_fit @ d_emp / (nf * ne)))
        kept.append(int(x))
    return np.array(cosines), np.array(kept, dtype=int)


def pearson(a, b, eps=1e-12):
    if a.size < 2:
        return float("nan")
    a = a - a.mean()
    b = b - b.mean()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < eps or nb < eps:
        return float("nan")
    return float(a @ b / (na * nb))


def cosine(a, b, eps=1e-12):
    if a.size == 0:
        return float("nan")
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < eps or nb < eps:
        return float("nan")
    return float(a @ b / (na * nb))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=("sumo", "xuancheng"), default="sumo",
                   help="'sumo' = bigger-bbox SUMO with --seed (historical default); "
                        "'xuancheng' = real-world Xuancheng AVI data with --day / --regime_split.")
    p.add_argument("--seed", type=int, default=None,
                   help="Required when --dataset sumo. SUMO/randomTrips seed.")
    p.add_argument("--day", type=str, default=None,
                   help="Required when --dataset xuancheng. ISO date e.g. 2023-04-17. "
                        "For multi-day pooling, pass a comma-separated list "
                        "(e.g. 2023-04-17,2023-04-18,2023-04-19).")
    p.add_argument("--regime_split", choices=("rush_offpeak", "weekday_weekend",
                                              "holiday_normal"),
                   default="rush_offpeak",
                   help="Xuancheng only. How to partition trips into two regimes.")
    p.add_argument("--od_match", type=int, default=None,
                   help="Xuancheng only. If set, OD-match the two regime pools "
                        "by k-means-clustering junctions into K zones then "
                        "subsampling per (zone_o, zone_d) cell to equalise "
                        "counts. Removes OD-mix confound. Recommended: K=8.")
    p.add_argument("--x_o_seed", type=int, default=13)
    p.add_argument("--T", type=int, default=20)
    p.add_argument("--maxiter", type=int, default=1500)
    p.add_argument("--features", choices=("none", "real"), default="none",
                   help="'none' = local-only fit (historical default); "
                        "'real' = enable omega-globals using SUMO edge attributes.")
    p.add_argument("--fit_mode",
                   choices=("f", "fg_floor", "fg_shrink"), default="f",
                   help="'f' = f-only fit (gamma=1.0, default); "
                        "'fg_floor' = f+g with empirical g floored at 1e-3; "
                        "'fg_shrink' = f+g with Beta-Binomial shrinkage on g.")
    p.add_argument("--gamma_fg", type=float, default=0.1,
                   help="Mix weight on stationary loss when fit_mode includes g "
                        "(0.1 = paper default).")
    args = p.parse_args()

    if args.dataset == "sumo":
        if args.seed is None:
            p.error("--seed is required when --dataset sumo")
        S = args.seed
        print(f"=== contrast-correlation diagnostic (SUMO big bbox, seed={S}) ===")
        edges, adj_out, idx = build_adj("net.net.xml")
        f_intr = read_edge_counts(f"edgedata.intr.big.seed{S}.xml", idx)
        f_satnav = read_edge_counts(f"edgedata.satnav.big.seed{S}.xml", idx)
        trajs_intr, _ = read_trajectories(f"vehroutes.intr.big.seed{S}.xml", idx)
        trajs_satnav, _ = read_trajectories(f"vehroutes.satnav.big.seed{S}.xml", idx)
        # Path used by extract_features below
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
        print(f"=== contrast-correlation diagnostic (Xuancheng, "
              f"days={','.join(date_strs)}, "
              f"split={args.regime_split}) ===")
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
        # extract_features uses sumolib to re-read the network for feature
        # construction; point it at the Xuancheng net.
        net_path_for_features = NET_PATH_DEFAULT
        regime_a, regime_b = label_a, label_b

    n = len(edges)
    out_deg = np.array([len(a) for a in adj_out])

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

    print(f"  dataset={args.dataset}  n={n}  |X_o|={len(X_o)}  beta={beta:.3f}  "
          f"T={T}  N_test={n_t}  features={args.features}  "
          f"d_T={d_T}  d_psi={d_psi}  fit_mode={args.fit_mode}  "
          f"gamma_fg={args.gamma_fg}  regimes={regime_a}/{regime_b}  "
          f"od_match={args.od_match}")

    # Compute the g matrices (only if fit_mode requests them).
    if args.fit_mode == "f":
        g_intr, g_satnav = None, None
        gamma_to_use = None  # let fit_chain pick its default (1.0 when g=None)
    else:
        print("  computing empirical g (hit_sum / visit_count) ...")
        hit_intr, vis_intr = empirical_g_counts(train_intr, X_o, beta)
        hit_satnav, vis_satnav = empirical_g_counts(train_satnav, X_o, beta)

        if args.fit_mode == "fg_floor":
            g_intr = np.clip(
                hit_intr / np.maximum(vis_intr[:, None], 1.0), 1e-3, 1.0)
            g_satnav = np.clip(
                hit_satnav / np.maximum(vis_satnav[:, None], 1.0), 1e-3, 1.0)
            print(f"    floor 1e-3: intr at-floor={100*(g_intr<=1e-3+1e-12).mean():.1f}%  "
                  f"satnav at-floor={100*(g_satnav<=1e-3+1e-12).mean():.1f}%")
        else:  # fg_shrink
            raw_g_i = hit_intr / np.maximum(vis_intr[:, None], 1.0)
            raw_g_s = hit_satnav / np.maximum(vis_satnav[:, None], 1.0)
            w_i = np.broadcast_to(vis_intr[:, None], raw_g_i.shape)
            w_s = np.broadcast_to(vis_satnav[:, None], raw_g_s.shape)
            a_i, b_i = fit_beta_mom(raw_g_i, weights=w_i)
            a_s, b_s = fit_beta_mom(raw_g_s, weights=w_s)
            g_intr = shrink_g_beta_binomial(hit_intr, vis_intr, a_i, b_i)
            g_satnav = shrink_g_beta_binomial(hit_satnav, vis_satnav, a_s, b_s)
            print(f"    Beta(intr  ): alpha={a_i:.3f} beta={b_i:.3f} "
                  f"prior_mean={a_i/(a_i+b_i):.3f}")
            print(f"    Beta(satnav): alpha={a_s:.3f} beta={b_s:.3f} "
                  f"prior_mean={a_s/(a_s+b_s):.3f}")
        gamma_to_use = args.gamma_fg

    print("  fitting chains ...")
    _, PT_intr_h, _ = fit_chain(
        adj_out, beta, X_o, f_intr[X_o], g_obs=g_intr, gamma=gamma_to_use,
        maxiter=args.maxiter, phi_T=phi_T, psi=psi,
    )
    _, PT_satnav_h, _ = fit_chain(
        adj_out, beta, X_o, f_satnav[X_o], g_obs=g_satnav, gamma=gamma_to_use,
        maxiter=args.maxiter, phi_T=phi_T, psi=psi,
    )
    PT_intr_emp = empirical_chain(train_intr, adj_out)
    PT_satnav_emp = empirical_chain(train_satnav, adj_out)

    test_all = test_intr + test_satnav
    labels = np.concatenate([np.zeros(n_t), np.ones(n_t)])
    _, _, auc_fit = roc(_scores_for_chains(test_all, PT_intr_h, PT_satnav_h), labels)
    _, _, auc_emp = roc(_scores_for_chains(test_all, PT_intr_emp, PT_satnav_emp), labels)
    gap = (auc_fit - 0.5) / (auc_emp - 0.5) if auc_emp > 0.5 else float("nan")

    print(f"\n  AUC reference: empirical={auc_emp:.3f}  fitted_f={auc_fit:.3f}  "
          f"gap_closed={100*gap:5.1f}%")

    # State subsets
    all_states = np.arange(n)
    branching_all = all_states[out_deg > 1]
    branching_xo = np.array([x for x in X_o if out_deg[x] > 1], dtype=int)

    print(f"\n  branching states total: {len(branching_all)} / {n}")
    print(f"  branching states in X_o: {len(branching_xo)} / {len(X_o)}")

    # ---- stacked diagnostics ----
    for name, states in [("X_o branching", branching_xo),
                         ("all branching", branching_all)]:
        d_fit, d_emp, n_b = contrast_rows(
            PT_intr_h, PT_satnav_h, PT_intr_emp, PT_satnav_emp, adj_out, states
        )
        if d_fit.size == 0:
            print(f"\n  [{name}] no branching states with data.")
            continue
        r_pearson = pearson(d_fit, d_emp)
        r_cosine = cosine(d_fit, d_emp)
        cosines, _ = per_row_cosines(
            PT_intr_h, PT_satnav_h, PT_intr_emp, PT_satnav_emp, adj_out, states
        )
        frac_pos = float((cosines > 0).mean()) if cosines.size else float("nan")
        median_cos = float(np.median(cosines)) if cosines.size else float("nan")
        mean_cos = float(cosines.mean()) if cosines.size else float("nan")
        print(f"\n  [{name}]  ({n_b} branching states, "
              f"{d_fit.size} legal-edge entries)")
        print(f"    stacked Pearson   = {r_pearson:+.3f}")
        print(f"    stacked cosine    = {r_cosine:+.3f}")
        print(f"    per-row cosine    mean = {mean_cos:+.3f}  "
              f"median = {median_cos:+.3f}")
        print(f"    rows with cosine > 0: {frac_pos*100:5.1f}%  "
              f"(50% = chance)")

    print(f"\nRESULT  dataset={args.dataset}  seed_or_day={S}  "
          f"regimes={regime_a}/{regime_b}  features={args.features}  "
          f"d_T={d_T}  d_psi={d_psi}  fit_mode={args.fit_mode}  "
          f"gamma_fg={args.gamma_fg}  od_match={args.od_match}  "
          f"auc_emp={auc_emp:.3f}  auc_fit={auc_fit:.3f}  "
          f"gap_closed={100*gap:.1f}%")


if __name__ == "__main__":
    main()
