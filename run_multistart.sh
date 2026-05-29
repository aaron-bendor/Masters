#!/usr/bin/env bash
# Sim #1 - SUMO multistart basin check on LOW-BASIN seeds (7, 23), K=5.
# Decisive test: can multi-start L-BFGS escape the ~30% gap-closed "bad basin"
# (seeds 7/23) and reach the ~60% "good basin" (seeds 42/101/2024)?
# The script prints multi-start (val-selected) AUC vs the single-init baseline,
# so each run directly shows whether the bad basin is escapable.
# features=none is the decisive run; add 'real' to the F loop if you also want
# to test whether the seed-101-style AUC reversal is basin-related.
cd /Users/aaronbendor/DESENG/Masters || exit 1

# Single-init baselines from the iter-23 multi-seed run (features=none, T=20),
# for direct comparison against the multistart (val-selected) numbers below.
# Good-basin reference: seeds 42/101/2024 = gap_closed 60.6/62.1/61.8%, Pearson 0.132/0.171/0.131.
echo "=== single-init baselines (features=none, T=20) ==="
echo "  seed  7 : gap_closed=30.3%  stacked_Pearson=+0.191"
echo "  seed 23 : gap_closed=34.2%  stacked_Pearson=+0.133"
echo "  target  : good-basin gap_closed ~60-62%  (seeds 42/101/2024)"
echo "==================================================="

for S in 7 23; do
  case "$S" in
    7)  BASE_GAP="30.3%"; BASE_PEAR="+0.191" ;;
    23) BASE_GAP="34.2%"; BASE_PEAR="+0.133" ;;
    *)  BASE_GAP="?";     BASE_PEAR="?" ;;
  esac
  for F in none; do
    echo ""
    echo "############ multistart seed=$S features=$F  start=$(date '+%F %T') ############"
    echo "#### single-init baseline: gap_closed=$BASE_GAP  stacked_Pearson=$BASE_PEAR ####"
    .venv/bin/python sumo_validation/score_seed_big_multistart.py \
        --seed "$S" --features "$F" --K 5 --T 20 --maxiter 1500
  done
done
echo "ALL DONE $(date '+%F %T')"
