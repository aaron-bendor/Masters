"""Parse a 2-step sweep log (alpha or backoff) and aggregate by parameter
across seeds. Prints mean +- std AUC per parameter, plus the best parameter
and its multi-seed delta vs 1-step.

Usage:
  ../.venv/bin/python aggregate_2step_sweep.py 2step_alpha_sweep.log
  ../.venv/bin/python aggregate_2step_sweep.py 2step_backoff_sweep.log
"""
import argparse, re, statistics
from collections import defaultdict

p = argparse.ArgumentParser()
p.add_argument("log")
args = p.parse_args()

# RESULT line. Both scripts produce auc_1step and one of auc_2step / auc_2step_backoff.
res_re = re.compile(
    r"seed=(\d+).*?"
    r"(?:smoothing|lam_fixed|kappa)?"
    r".*?auc_1step=([0-9.]+).*?"
    r"auc_2step(?:_backoff)?=([0-9.]+)"
)
# Try to pick up the parameter (alpha or lam or kappa) from the run line.
alpha_re = re.compile(r"alpha=([0-9.eE+-]+)")
lam_re   = re.compile(r"lam_fixed=([0-9.]+)")
kappa_re = re.compile(r"kappa=([0-9.]+)")

groups = defaultdict(list)  # param_value -> list of (seed, auc_1step, auc_2step)
current_param = None

with open(args.log) as f:
    for line in f:
        if line.startswith("===="):
            # config header — pull the parameter out
            m = alpha_re.search(line)
            if m:
                current_param = ("alpha", float(m.group(1)))
                continue
            m = lam_re.search(line)
            if m:
                current_param = ("lam_fixed", float(m.group(1)))
                continue
            m = kappa_re.search(line)
            if m:
                current_param = ("kappa", float(m.group(1)))
                continue
        m_r = res_re.search(line)
        if m_r and current_param is not None:
            seed = int(m_r.group(1))
            a1 = float(m_r.group(2))
            a2 = float(m_r.group(3))
            groups[current_param].append((seed, a1, a2))

if not groups:
    print("No RESULT lines parsed; check log path and format.")
    raise SystemExit(0)

# Order by parameter value
ordered = sorted(groups.keys(), key=lambda k: k[1])

# Compute 1-step mean across all rows (should be similar across configs since it's the same chains)
all_1step = []
for k in ordered:
    for _, a1, _ in groups[k]:
        all_1step.append(a1)
one_step_mean = statistics.mean(all_1step)
one_step_std = statistics.stdev(all_1step) if len(all_1step) > 1 else 0.0

print(f"1-step empirical (over all rows in log):  mean {one_step_mean:.3f}  std {one_step_std:.3f}")
print()

label = ordered[0][0]
print(f"{label:>10}  {'n':>3}  {'2step mean':>11}  {'2step std':>10}  "
      f"{'delta mean':>11}  {'delta std':>10}  per-seed deltas")
print("-" * 100)

best_key = None
best_delta = -1.0
for k in ordered:
    rows = groups[k]
    a2s = [r[2] for r in rows]
    deltas = [r[2] - r[1] for r in rows]
    m = statistics.mean(a2s)
    s = statistics.stdev(a2s) if len(a2s) > 1 else 0.0
    dm = statistics.mean(deltas)
    ds = statistics.stdev(deltas) if len(deltas) > 1 else 0.0
    per_seed = "  ".join(f"s{r[0]}:{r[2]-r[1]:+.3f}" for r in sorted(rows))
    print(f"{k[1]:>10}  {len(rows):>3}  {m:>11.3f}  {s:>10.4f}  "
          f"{dm:>+11.3f}  {ds:>10.4f}  {per_seed}")
    if dm > best_delta:
        best_delta = dm
        best_key = k

print()
print(f"Best: {best_key[0]}={best_key[1]}  multi-seed delta = {best_delta:+.3f}")

# Highlight which seeds the best config wins on
if best_key is not None:
    rows = groups[best_key]
    wins = sum(1 for _, a1, a2 in rows if a2 > a1)
    print(f"  wins on {wins}/{len(rows)} seeds")
