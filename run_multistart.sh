#!/usr/bin/env bash
# Sim #1 - SUMO multistart basin check on LOW-BASIN seeds (7, 23), K=5.
# Decisive test: can multi-start L-BFGS escape the ~30% gap-closed "bad basin"
# (seeds 7/23) and reach the ~60% "good basin" (seeds 42/101/2024)?
# The script prints multi-start (val-selected) AUC vs the single-init baseline,
# so each run directly shows whether the bad basin is escapable.
# features=none is the decisive run; add 'real' to the F loop if you also want
# to test whether the seed-101-style AUC reversal is basin-related.
cd /Users/aaronbendor/DESENG/Masters || exit 1
for S in 7 23; do
  for F in none; do
    echo ""
    echo "############ multistart seed=$S features=$F  start=$(date '+%F %T') ############"
    .venv/bin/python sumo_validation/score_seed_big_multistart.py \
        --seed "$S" --features "$F" --K 5 --T 20 --maxiter 1500
  done
done
echo "ALL DONE $(date '+%F %T')"
