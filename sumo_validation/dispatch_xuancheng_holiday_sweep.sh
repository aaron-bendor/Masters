#!/usr/bin/env bash
# Xuancheng holiday-vs-normal stress test.
#
# Primary question:
#   If the behavioural contrast is stronger than rush/off-peak, does the
#   partial-observation detector recover it once OD mix is controlled?
#
# Primary pool:
#   holiday = Apr 5 (Tomb-Sweeping) + Apr 28-30 (Labour Day period)
#   normal  = Apr 11-14 + Apr 17-21 (midweek workdays; Apr 10 excluded)
#
# Usage:
#   ./dispatch_xuancheng_holiday_sweep.sh
#   POOL=labour ./dispatch_xuancheng_holiday_sweep.sh
#   POOL=tomb ./dispatch_xuancheng_holiday_sweep.sh
#
# Capture with:
#   ./dispatch_xuancheng_holiday_sweep.sh | tee logs/xuancheng_holiday_Ksweep.log

set -euo pipefail
cd "$(dirname "$0")"

PY=../.venv/bin/python
POOL="${POOL:-all}"
T="${T:-10}"
B="${B:-2000}"
MAXITER="${MAXITER:-1500}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/private/tmp/matplotlib-cache}"

NORMAL_DAYS="2023-04-11,2023-04-12,2023-04-13,2023-04-14,2023-04-17,2023-04-18,2023-04-19,2023-04-20,2023-04-21"

case "$POOL" in
    all)
        DAYS="2023-04-05,2023-04-28,2023-04-29,2023-04-30,$NORMAL_DAYS"
        ;;
    labour)
        DAYS="2023-04-28,2023-04-29,2023-04-30,$NORMAL_DAYS"
        ;;
    tomb)
        DAYS="2023-04-05,$NORMAL_DAYS"
        ;;
    *)
        echo "Unknown POOL='$POOL' (expected all, labour, or tomb)" >&2
        exit 2
        ;;
esac

echo "=== Xuancheng holiday-vs-normal sweep ==="
echo "POOL=$POOL"
echo "DAYS=$DAYS"
echo "T=$T  B=$B  MAXITER=$MAXITER"

echo
echo "######## unmatched ########"
"$PY" bootstrap_auc.py --dataset xuancheng --day "$DAYS" \
    --regime_split holiday_normal --features none --T "$T" \
    --maxiter "$MAXITER" --B "$B"

for K in 8 16 32 64; do
    echo
    echo "######## OD-matched K=$K ########"
    "$PY" bootstrap_auc.py --dataset xuancheng --day "$DAYS" \
        --regime_split holiday_normal --od_match "$K" \
        --features none --T "$T" --maxiter "$MAXITER" --B "$B"
done

echo
echo "=== Optional diagnostics ==="
echo "Run the strongest empirical K through contrast_correlation.py, e.g.:"
echo "  $PY contrast_correlation.py --dataset xuancheng --day \"$DAYS\" --regime_split holiday_normal --od_match 32 --features none --T $T --maxiter $MAXITER"
echo "If fitted_f beats the rush/off-peak result, follow with features=real + multistart as a basin check."

echo "=== ALL DONE ==="
