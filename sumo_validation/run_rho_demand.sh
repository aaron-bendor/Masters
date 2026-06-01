#!/usr/bin/env bash
# Parameterized SUMO runner for the (rho x demand) sweep.
# Generates one SUMO scenario at the given (seed, demand_p, rho) combination.
# Routes are cached per (seed, demand_p): randomTrips + duarouter are independent
# of rho, so we reuse routes.d<P>.seed<SEED>.xml across rho values.
#
# Usage:   bash run_rho_demand.sh <seed> <rho> <demand_p>
# Example: bash run_rho_demand.sh 42 0.5 0.3
# Output tag: d<P>.r<RHO>.seed<SEED>

set -euo pipefail

SEED="${1:?usage: bash run_rho_demand.sh <seed> <rho> <demand_p>}"
RHO="${2:?usage: bash run_rho_demand.sh <seed> <rho> <demand_p>}"
P="${3:?usage: bash run_rho_demand.sh <seed> <rho> <demand_p>}"

cd "$(dirname "$0")"

SUMO_HOME="${SUMO_HOME:-/Library/Frameworks/EclipseSUMO.framework/Versions/1.26.0/EclipseSUMO/share/sumo}"
if [ ! -x "$SUMO_HOME/tools/randomTrips.py" ]; then
    echo "ERROR: cannot find randomTrips.py under SUMO_HOME=$SUMO_HOME" >&2
    exit 1
fi

TAG="d${P}.r${RHO}.seed${SEED}"
ROUTES="routes.d${P}.seed${SEED}.xml"
TRIPS="trips.d${P}.seed${SEED}.xml"
ADD_FILE="edgedata.${TAG}.add.xml"

echo "[$TAG] SUMO_HOME=$SUMO_HOME"

cat > "$ADD_FILE" <<EOF
<additional><edgeData id="agg" file="edgedata.${TAG}.xml" period="14400" excludeEmpty="true"/></additional>
EOF

if [ ! -f "$ROUTES" ]; then
    echo "[$TAG] randomTrips -p $P ..."
    python3 "$SUMO_HOME/tools/randomTrips.py" -n net.net.xml -e 14400 -p "$P" \
        --vehicle-class passenger --seed "$SEED" \
        -o "$TRIPS" --validate

    echo "[$TAG] duarouter ..."
    duarouter --net-file net.net.xml \
        --route-files "$TRIPS" \
        --output-file "$ROUTES" \
        --remove-loops --ignore-errors --seed "$SEED"
else
    echo "[$TAG] reusing cached $ROUTES"
fi

echo "[$TAG] sumo (rho=$RHO) ..."
sumo --net-file net.net.xml \
    --route-files "$ROUTES" \
    --additional-files "$ADD_FILE" \
    --device.rerouting.probability "$RHO" \
    --device.rerouting.period 60 \
    --device.rerouting.adaptation-interval 30 \
    --device.rerouting.adaptation-weight 0.5 \
    --begin 0 --end 14400 --seed "$SEED" \
    --vehroute-output "vehroutes.${TAG}.xml" \
    --vehroute-output.exit-times true \
    --statistic-output "stats.${TAG}.xml" \
    --no-warnings true

echo "[$TAG] done"
