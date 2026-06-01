#!/usr/bin/env bash
# Scoping dispatcher: 1 seed x 3 rho x 2 demand = 6 SUMO runs.
# Loops demand on the outer axis so each demand's routes (randomTrips + duarouter)
# are generated once, then reused across rho values within that demand.

set -euo pipefail
cd "$(dirname "$0")"

SEED=42
DEMANDS=(0.3 0.8)
RHOS=(0.0 0.5 1.0)

t_total_start=$(date +%s)

for D in "${DEMANDS[@]}"; do
    for R in "${RHOS[@]}"; do
        t0=$(date +%s)
        echo "==== START cell d=$D rho=$R seed=$SEED at $(date) ===="
        bash run_rho_demand.sh "$SEED" "$R" "$D"
        t1=$(date +%s)
        echo "==== END   cell d=$D rho=$R seed=$SEED  ($((t1 - t0))s) ===="
    done
done

t_total_end=$(date +%s)
echo "==== ALL 6 CELLS DONE  total=$((t_total_end - t_total_start))s ===="
