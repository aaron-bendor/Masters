#!/usr/bin/env bash
# Sweep Dirichlet smoothing alpha for the 2-step empirical detector.
# 5 SUMO seeds x 6 alpha values = 30 runs, each ~30s. Total ~15 min.
#
# Output: one RESULT line per (seed, alpha), tagged so aggregation is trivial.
# The existing two_step_empirical.py already implements graph-aware Dirichlet:
#   P_2(y | xp, x) = (count(xp, x, y) + alpha) / (sum_y count + alpha * |adj(x)|)
# so we only need to sweep its --smoothing flag.

set -euo pipefail
cd "$(dirname "$0")"

SEEDS=(7 23 42 101 2024)
ALPHAS=(0.001 0.01 0.1 1.0 10.0 100.0)

PY=../.venv/bin/python

t_start=$(date +%s)
for SEED in "${SEEDS[@]}"; do
    for A in "${ALPHAS[@]}"; do
        echo "==== seed=$SEED  alpha=$A ===="
        "$PY" two_step_empirical.py --seed "$SEED" --smoothing "$A" 2>&1 | \
            grep -E "^RESULT|fallback rate" | \
            sed "s/^/  /"
        echo
    done
done
t_end=$(date +%s)
echo "==== ALL DONE  total=$((t_end - t_start))s ===="
