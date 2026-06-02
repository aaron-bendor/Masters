#!/usr/bin/env bash
# Interpolation-backoff 2-step sweep across 5 seeds.
# Two modes:
#   fixed lambda in {0.0, 0.25, 0.5, 0.75, 0.9, 1.0}
#   count-conditional lambda kappa in {1, 10, 100, 1000}
# Total: 5 seeds x (6 fixed + 4 count) = 50 runs, each <2s. ~3 min total.

set -euo pipefail
cd "$(dirname "$0")"

SEEDS=(7 23 42 101 2024)
LAM_FIXED=(0.0 0.25 0.5 0.75 0.9 1.0)
KAPPAS=(1 10 100 1000)

PY=../.venv/bin/python

t_start=$(date +%s)

echo "=== FIXED LAMBDA SWEEP ==="
for SEED in "${SEEDS[@]}"; do
    for L in "${LAM_FIXED[@]}"; do
        echo "==== seed=$SEED  lam_fixed=$L ===="
        "$PY" two_step_backoff.py --seed "$SEED" --lam_mode fixed \
            --lam_fixed "$L" 2>&1 | grep "^RESULT" | sed "s/^/  /"
        echo
    done
done

echo "=== COUNT-CONDITIONAL KAPPA SWEEP ==="
for SEED in "${SEEDS[@]}"; do
    for K in "${KAPPAS[@]}"; do
        echo "==== seed=$SEED  kappa=$K ===="
        "$PY" two_step_backoff.py --seed "$SEED" --lam_mode count \
            --kappa "$K" 2>&1 | grep "^RESULT" | sed "s/^/  /"
        echo
    done
done

t_end=$(date +%s)
echo "=== ALL DONE  total=$((t_end - t_start))s ==="
