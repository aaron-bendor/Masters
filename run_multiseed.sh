#!/usr/bin/env bash
# Sim #2 - multi-seed Tier-2 features sweep (5 SUMO seeds x {none,real}).
cd /Users/aaronbendor/DESENG/Masters || exit 1
for S in 7 23 42 101 2024; do
  for F in none real; do
    echo ""
    echo "############ seed=$S features=$F  start=$(date '+%F %T') ############"
    .venv/bin/python sumo_validation/contrast_correlation.py \
        --dataset sumo --seed "$S" --features "$F" --T 20 --maxiter 1500
  done
done
echo "ALL DONE $(date '+%F %T')"
