"""Xuancheng empirical 2-step backoff diagnostic.

This is the real-data counterpart to ``two_step_backoff.py``. It is deliberately
empirical/full-observation: it asks whether a second-order route-memory score
contains more holiday-vs-normal signal than the first-order empirical reference.
It does not fit a partial-observation inverse model.

Typical Labour-Day stress-test run:

  ../.venv/bin/python xuancheng_2step_backoff.py \
      --day "2023-04-28,2023-04-29,2023-04-30,2023-04-11,2023-04-12,2023-04-13,2023-04-14,2023-04-17,2023-04-18,2023-04-19,2023-04-20,2023-04-21" \
      --regime_split holiday_normal --od_match 64 --T 10 --B 2000
"""
import argparse
import datetime
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "real_data"))
os.chdir(HERE)

from congestion_filter import roc  # noqa: E402
from per_car_detector import _scores_for_chains  # noqa: E402
from sumo_to_phase3 import empirical_chain  # noqa: E402
from two_step_backoff import score_2step_backoff_lr  # noqa: E402
from two_step_empirical import empirical_2step_counts  # noqa: E402
from load_xuancheng import (  # noqa: E402
    NET_PATH_DEFAULT,
    day_spec,
    load_xuancheng_regime,
    split_holiday_vs_normal,
    split_rush_vs_offpeak,
    split_weekday_vs_weekend,
)


LABOUR_DAYS = (
    "2023-04-28,2023-04-29,2023-04-30,"
    "2023-04-11,2023-04-12,2023-04-13,2023-04-14,"
    "2023-04-17,2023-04-18,2023-04-19,2023-04-20,2023-04-21"
)


def parse_days(day_arg):
    return [datetime.date.fromisoformat(s.strip())
            for s in day_arg.split(",") if s.strip()]


def split_config(name):
    if name == "rush_offpeak":
        return (lambda st, d: split_rush_vs_offpeak(st["start_time"]),
                "rush", "offpeak")
    if name == "weekday_weekend":
        return (lambda st, d: split_weekday_vs_weekend(d),
                "weekday", "weekend")
    if name == "holiday_normal":
        return (lambda st, d: split_holiday_vs_normal(d),
                "holiday", "normal")
    raise ValueError(name)


def stratified_bootstrap(scores, labels, n_per_class, B, seed):
    rng = np.random.default_rng(seed)
    idx_a = np.arange(n_per_class)
    idx_b = np.arange(n_per_class, 2 * n_per_class)
    out = np.empty(B, dtype=float)
    for b in range(B):
        sel = np.concatenate([
            rng.choice(idx_a, size=n_per_class, replace=True),
            rng.choice(idx_b, size=n_per_class, replace=True),
        ])
        out[b] = roc(scores[sel], labels[sel])[2]
    return out


def ci_summary(samples):
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return lo, hi, float(np.mean(samples > 0.5))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--day", type=str, default=LABOUR_DAYS,
                   help="Comma-separated ISO dates. Default is Labour-vs-normal pool.")
    p.add_argument("--regime_split",
                   choices=("rush_offpeak", "weekday_weekend", "holiday_normal"),
                   default="holiday_normal")
    p.add_argument("--od_match", type=int, default=64)
    p.add_argument("--T", type=int, default=10)
    p.add_argument("--N_test", type=int, default=300)
    p.add_argument("--smoothing", type=float, default=1e-3,
                   help="Laplace smoothing for the 1-step empirical chains.")
    p.add_argument("--B", type=int, default=2000)
    p.add_argument("--boot_seed", type=int, default=7)
    p.add_argument("--x_split_seed", type=int, default=42)
    args = p.parse_args()

    dates = parse_days(args.day)
    regime_split, label_a, label_b = split_config(args.regime_split)

    print("=== Xuancheng empirical 2-step backoff diagnostic ===")
    print(f"days={','.join(str(d) for d in dates)}")
    print(f"split={args.regime_split}  regimes={label_a}/{label_b}  "
          f"od_match={args.od_match}  T={args.T}  B={args.B}")

    edges, adj_out, idx, trajs_a, trajs_b, f_a, f_b = load_xuancheng_regime(
        net_path=NET_PATH_DEFAULT,
        day_specs=[day_spec(d) for d in dates],
        regime_split=regime_split,
        label_a=label_a,
        label_b=label_b,
        min_segment_len=5,
        verbose=True,
        od_match_zones=args.od_match,
    )
    n = len(edges)
    mean_len = float(np.mean([len(t) for t in trajs_a + trajs_b]))
    beta = float(max(0.5, min(0.99, 1.0 - 1.0 / mean_len)))

    long_a = [t for t in trajs_a if len(t) >= args.T + 1]
    long_b = [t for t in trajs_b if len(t) >= args.T + 1]
    rng_test = np.random.default_rng(args.x_split_seed)
    rng_test.shuffle(long_a)
    rng_test.shuffle(long_b)
    n_t = min(args.N_test, len(long_a), len(long_b))
    test_a_full = long_a[:n_t]
    test_b_full = long_b[:n_t]
    test_a = [t[: args.T + 1] for t in test_a_full]
    test_b = [t[: args.T + 1] for t in test_b_full]
    test_ids = ({id(t) for t in test_a_full}, {id(t) for t in test_b_full})
    train_a = [t for t in trajs_a if id(t) not in test_ids[0]]
    train_b = [t for t in trajs_b if id(t) not in test_ids[1]]

    print(f"n={n}  beta={beta:.3f}  N_test_per_class={n_t}")
    print(f"train trajs: {label_a}={len(train_a):,}  {label_b}={len(train_b):,}")

    print("\nbuilding 1-step empirical chains ...")
    PT_a_1 = empirical_chain(train_a, adj_out, smoothing=args.smoothing)
    PT_b_1 = empirical_chain(train_b, adj_out, smoothing=args.smoothing)

    print("counting 2-step triples ...")
    counts_a, n_trip_a = empirical_2step_counts(train_a)
    counts_b, n_trip_b = empirical_2step_counts(train_b)
    print(f"  {label_a:>8s}: {n_trip_a:>10,} triples across "
          f"{len(counts_a):>6,} (x_prev, x) pairs")
    print(f"  {label_b:>8s}: {n_trip_b:>10,} triples across "
          f"{len(counts_b):>6,} (x_prev, x) pairs")

    test_all = test_a + test_b
    labels = np.concatenate([
        np.zeros(n_t, dtype=int),
        np.ones(n_t, dtype=int),
    ])

    s_1 = _scores_for_chains(test_all, PT_a_1, PT_b_1)
    auc_1 = roc(s_1, labels)[2]
    boot_1 = stratified_bootstrap(s_1, labels, n_t, args.B, args.boot_seed)
    lo_1, hi_1, p_1 = ci_summary(boot_1)

    configs = (
        [("fixed", v) for v in (0.0, 0.25, 0.5, 0.75, 0.9, 1.0)]
        + [("count", v) for v in (1.0, 10.0, 100.0, 1000.0)]
    )

    print("\nAUC summary")
    print(f"  1-step empirical    point={auc_1:.3f}  "
          f"95% CI=[{lo_1:.3f}, {hi_1:.3f}]  P(AUC>0.5)={p_1:.3f}")

    rows = []
    for mode, value in configs:
        if mode == "fixed":
            scores, fb_rate = score_2step_backoff_lr(
                test_all, counts_a, counts_b, PT_a_1, PT_b_1,
                lam_mode="fixed", lam_fixed=value,
            )
            cfg = f"fixed lam={value:g}"
        else:
            scores, fb_rate = score_2step_backoff_lr(
                test_all, counts_a, counts_b, PT_a_1, PT_b_1,
                lam_mode="count", kappa=value,
            )
            cfg = f"count kappa={value:g}"
        auc = roc(scores, labels)[2]
        boot = stratified_bootstrap(scores, labels, n_t, args.B, args.boot_seed)
        lo, hi, p_above = ci_summary(boot)
        rows.append((auc, mode, value, cfg, fb_rate, lo, hi, p_above))
        print(f"  2-step {cfg:16s} point={auc:.3f}  "
              f"delta={auc - auc_1:+.3f}  "
              f"95% CI=[{lo:.3f}, {hi:.3f}]  "
              f"P(AUC>0.5)={p_above:.3f}  fallback={fb_rate:.3f}")

    best = max(rows, key=lambda r: r[0])
    print("\nRESULT  dataset=xuancheng"
          f"  days={','.join(str(d) for d in dates)}"
          f"  regimes={label_a}/{label_b}"
          f"  od_match={args.od_match}"
          f"  T={args.T}"
          f"  auc_1step={auc_1:.3f}"
          f"  auc_2step_best={best[0]:.3f}"
          f"  delta_best={best[0] - auc_1:+.3f}"
          f"  best_mode={best[1]}"
          f"  best_value={best[2]:g}"
          f"  fallback={best[4]:.3f}")


if __name__ == "__main__":
    main()
