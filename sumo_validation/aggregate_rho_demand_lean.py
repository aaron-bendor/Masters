"""Parse the lean multi-seed (rho, demand) dispatcher log and report
per-pair mean/std plus the decomposition of the proxy-diagonal signal.

Usage:
  ../.venv/bin/python aggregate_rho_demand_lean.py rho_demand_lean.log
"""
import argparse, re, statistics
from collections import defaultdict

p = argparse.ArgumentParser()
p.add_argument("log")
args = p.parse_args()

# Section header line: "==== <pair_type>_seed<N>  (<tag_a> vs <tag_b>) ===="
hdr_re = re.compile(r"^====\s+([A-Za-z_]+?)_seed(\d+)\s")
# RESULT line containing empirical=... and fitted_f=...
res_re = re.compile(r"empirical=([0-9.]+)\s+fitted_f=([0-9.]+)")

groups = defaultdict(list)  # pair_type -> list of (seed, empirical, fitted)
current_label = None
current_seed = None

with open(args.log) as f:
    for line in f:
        m = hdr_re.match(line)
        if m:
            current_label = m.group(1)
            current_seed = int(m.group(2))
            continue
        m = res_re.search(line)
        if m and current_label is not None:
            emp, fit = float(m.group(1)), float(m.group(2))
            groups[current_label].append((current_seed, emp, fit))
            current_label = None
            current_seed = None

print(f"{'Pair':<22}  {'n':>3}  {'emp_mean':>9}  {'emp_std':>8}  {'fit_mean':>9}  {'fit_std':>8}")
print("-" * 70)
order = ["pure_congestion", "satnav_at_peak", "proxy_diagonal"]
ordered_keys = [k for k in order if k in groups] + [k for k in groups if k not in order]
for label in ordered_keys:
    rows = groups[label]
    emp = [r[1] for r in rows]
    fit = [r[2] for r in rows]
    em = statistics.mean(emp)
    es = statistics.stdev(emp) if len(emp) > 1 else 0.0
    fm = statistics.mean(fit)
    fs = statistics.stdev(fit) if len(fit) > 1 else 0.0
    print(f"{label:<22}  {len(rows):>3}  {em:>9.3f}  {es:>8.4f}  {fm:>9.3f}  {fs:>8.4f}")
    per_seed = "  ".join(f"s{r[0]}:{r[1]:.3f}" for r in sorted(rows))
    print(f"    per-seed empirical: {per_seed}")

needed = {"pure_congestion", "satnav_at_peak", "proxy_diagonal"}
if needed.issubset(groups):
    pc = statistics.mean(r[1] for r in groups["pure_congestion"])
    sn = statistics.mean(r[1] for r in groups["satnav_at_peak"])
    pd = statistics.mean(r[1] for r in groups["proxy_diagonal"])
    pc_s = pc - 0.5
    sn_s = sn - 0.5
    pd_s = pd - 0.5
    print()
    print("=== DECOMPOSITION (empirical AUC, mean across seeds) ===")
    print(f"  pure-congestion signal: {pc_s:+.3f}  ({100 * pc_s / pd_s:5.1f}% of proxy)")
    print(f"  sat-nav-at-peak signal: {sn_s:+.3f}  ({100 * sn_s / pd_s:5.1f}% of proxy)")
    print(f"  proxy-diagonal signal:  {pd_s:+.3f}  (total observed)")
    interaction = pd_s - pc_s - sn_s
    print(f"  residual / interaction: {interaction:+.3f}  ({100 * interaction / pd_s:5.1f}% of proxy)")

    print()
    print("=== PER-SEED DECOMPOSITION (sat-nav % of proxy) ===")
    by_seed = defaultdict(dict)
    for label in needed:
        for seed, emp, _ in groups[label]:
            by_seed[seed][label] = emp - 0.5
    for seed in sorted(by_seed):
        d = by_seed[seed]
        if needed.issubset(d) and d["proxy_diagonal"] > 0:
            pct = 100 * d["satnav_at_peak"] / d["proxy_diagonal"]
            pct_pc = 100 * d["pure_congestion"] / d["proxy_diagonal"]
            print(f"  seed {seed:>5}:  sat-nav {pct:5.1f}%  pure-congestion {pct_pc:5.1f}%  "
                  f"(proxy +{d['proxy_diagonal']:.3f})")
