#!/usr/bin/env bash
# Scoring dispatcher for the 6-cell rho x demand scope at seed=42.
# Runs the five key pair comparisons and tags each with a label.

set -euo pipefail
cd "$(dirname "$0")"

PY=../.venv/bin/python

run_pair() {
    local label="$1" tag_a="$2" tag_b="$3"
    echo "==== $label  ($tag_a  vs  $tag_b) ===="
    "$PY" score_rho_demand.py --tag_a "$tag_a" --tag_b "$tag_b"
    echo
}

# Pure congestion effect (no sat-nav anywhere; only demand changes)
run_pair "pure_congestion"           d0.3.r0.0.seed42 d0.8.r0.0.seed42

# Pure sat-nav effect at off-peak demand
run_pair "satnav_only_offpeak"       d0.8.r0.0.seed42 d0.8.r1.0.seed42

# Pure sat-nav effect at peak demand
run_pair "satnav_only_peak"          d0.3.r0.0.seed42 d0.3.r1.0.seed42

# Full proxy diagonal (off-peak no-nav vs peak full-nav) — mimics Xuancheng comparison
run_pair "proxy_diagonal"            d0.8.r0.0.seed42 d0.3.r1.0.seed42

# Mixed populations at fixed peak demand (intrinsic vs half-mixed)
run_pair "mixed_at_peak"             d0.3.r0.0.seed42 d0.3.r0.5.seed42

echo "==== ALL 5 PAIRS SCORED ===="
