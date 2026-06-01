#!/usr/bin/env bash
# Lean multi-seed (rho, demand) sweep: 5 seeds x 3 cells = 15 SUMO runs,
# then 5 seeds x 3 pair-scorings = 15 scoring calls.
#
# Cells per seed:
#   (d=0.3, rho=0.0)  -- peak intrinsic
#   (d=0.3, rho=1.0)  -- peak sat-nav
#   (d=0.8, rho=0.0)  -- off-peak intrinsic
#
# Pair scorings per seed:
#   pure_congestion : d0.3 r0.0  vs  d0.8 r0.0   (demand axis, no sat-nav)
#   satnav_at_peak  : d0.3 r0.0  vs  d0.3 r1.0   (rho axis, at peak)
#   proxy_diagonal  : d0.8 r0.0  vs  d0.3 r1.0   (mimics Xuancheng comparison)
#
# Skips cells whose outputs already exist (seed=42 should be cached from the scope run).
# Total expected wall time: ~1.5-2 hours.

set -euo pipefail
cd "$(dirname "$0")"

SEEDS=(7 23 42 101 2024)
CELLS=("0.3 0.0" "0.3 1.0" "0.8 0.0")

PY=../.venv/bin/python

echo "=== STAGE 1: SUMO RUNS ==="
t_sumo_start=$(date +%s)
for SEED in "${SEEDS[@]}"; do
    for CELL in "${CELLS[@]}"; do
        D=$(echo "$CELL" | awk '{print $1}')
        R=$(echo "$CELL" | awk '{print $2}')
        TAG="d${D}.r${R}.seed${SEED}"
        if [ -f "vehroutes.${TAG}.xml" ] && [ -f "edgedata.${TAG}.xml" ]; then
            echo "[SKIP] $TAG (outputs already exist)"
            continue
        fi
        t0=$(date +%s)
        echo "==== START cell $TAG at $(date) ===="
        bash run_rho_demand.sh "$SEED" "$R" "$D"
        t1=$(date +%s)
        echo "==== END   cell $TAG  ($((t1 - t0))s) ===="
    done
done
t_sumo_end=$(date +%s)
echo "=== STAGE 1 DONE  total=$((t_sumo_end - t_sumo_start))s ==="
echo

echo "=== STAGE 2: SCORING ==="
t_score_start=$(date +%s)
score_pair() {
    local label="$1" tag_a="$2" tag_b="$3"
    echo "==== $label  ($tag_a  vs  $tag_b) ===="
    "$PY" score_rho_demand.py --tag_a "$tag_a" --tag_b "$tag_b"
    echo
}
for SEED in "${SEEDS[@]}"; do
    score_pair "pure_congestion_seed${SEED}"  "d0.3.r0.0.seed${SEED}"  "d0.8.r0.0.seed${SEED}"
    score_pair "satnav_at_peak_seed${SEED}"   "d0.3.r0.0.seed${SEED}"  "d0.3.r1.0.seed${SEED}"
    score_pair "proxy_diagonal_seed${SEED}"   "d0.8.r0.0.seed${SEED}"  "d0.3.r1.0.seed${SEED}"
done
t_score_end=$(date +%s)
echo "=== STAGE 2 DONE  total=$((t_score_end - t_score_start))s ==="
echo
echo "=== ALL DONE  grand_total=$((t_score_end - t_sumo_start))s ==="
