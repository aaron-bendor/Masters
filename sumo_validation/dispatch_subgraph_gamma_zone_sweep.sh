#!/usr/bin/env bash
# Sweep gamma_fg and pick_zone on the Xuancheng subgraph f+g test.
# At zone 0 (default) we saw f+g hurt at gamma=0.5. Two open questions:
#   (a) does a lighter g weight (gamma >= 0.8) let f+g help, even with g 88% floor?
#   (b) does the densest zone (6, ~363 edges) have less floor and more signal?

set -euo pipefail
cd "$(dirname "$0")"

PY=../.venv/bin/python

# Gamma sweep on zone 0 (the one we already ran at gamma=0.5)
for G in 0.5 0.7 0.9 0.95 0.99; do
    echo "==== zone=0  gamma_fg=$G ===="
    "$PY" fg_xuancheng_subgraph.py --K_macro 8 --pick_zone 0 --T 10 \
        --maxiter 300 --gamma_fg "$G" 2>&1 | grep "^RESULT\|g_floor\|auc_" | \
        sed "s/^/  /"
    echo
done

# Zone sweep at gamma_fg=0.9 (a likely sweet spot for noisy g)
for Z in 1 2 3 4 5 6 7; do
    echo "==== zone=$Z  gamma_fg=0.9 ===="
    "$PY" fg_xuancheng_subgraph.py --K_macro 8 --pick_zone "$Z" --T 10 \
        --maxiter 300 --gamma_fg 0.9 2>&1 | grep "^RESULT\|g_floor\|auc_" | \
        sed "s/^/  /"
    echo
done

echo "==== ALL DONE ===="
