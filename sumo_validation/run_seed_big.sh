#!/usr/bin/env bash
# Run the BIGGER-bbox -p 0.5 SUMO pipeline (intr + satnav) for one randomTrips seed.
# Outputs: vehroutes/edgedata/stats tagged "big.seed<N>" to avoid clobbering
# the existing bigbox.p0.5.xml files (iter 11/13 outputs from an unknown seed).
# Usage:   bash run_seed_big.sh <seed>
set -euo pipefail
SEED="${1:?usage: bash run_seed_big.sh <seed>}"
cd "$(dirname "$0")"

# Fall back to the known SUMO install path if $SUMO_HOME isn't inherited.
SUMO_HOME="${SUMO_HOME:-/Library/Frameworks/EclipseSUMO.framework/Versions/1.26.0/EclipseSUMO/share/sumo}"
if [ ! -x "$SUMO_HOME/tools/randomTrips.py" ]; then
    echo "ERROR: cannot find randomTrips.py under SUMO_HOME=$SUMO_HOME" >&2
    exit 1
fi
echo "[seed=$SEED big] SUMO_HOME=$SUMO_HOME"

ADD_INTR="edgedata.intr.bigseed${SEED}.add.xml"
ADD_SATNAV="edgedata.satnav.bigseed${SEED}.add.xml"
cat > "$ADD_INTR" <<EOF
<additional><edgeData id="agg" file="edgedata.intr.big.seed${SEED}.xml" period="14400" excludeEmpty="true"/></additional>
EOF
cat > "$ADD_SATNAV" <<EOF
<additional><edgeData id="agg" file="edgedata.satnav.big.seed${SEED}.xml" period="14400" excludeEmpty="true"/></additional>
EOF

echo "[seed=$SEED big] randomTrips ..."
python3 "$SUMO_HOME/tools/randomTrips.py" -n net.net.xml -e 14400 -p 0.5 \
    --vehicle-class passenger --seed "$SEED" \
    -o "trips.big.seed${SEED}.xml" --validate

echo "[seed=$SEED big] duarouter ..."
duarouter --net-file net.net.xml \
    --route-files "trips.big.seed${SEED}.xml" \
    --output-file "routes.big.seed${SEED}.xml" \
    --remove-loops --ignore-errors --seed "$SEED"

echo "[seed=$SEED big] sumo intr ..."
sumo --net-file net.net.xml \
    --route-files "routes.big.seed${SEED}.xml" \
    --additional-files "$ADD_INTR" \
    --begin 0 --end 14400 --seed "$SEED" \
    --vehroute-output "vehroutes.intr.big.seed${SEED}.xml" \
    --vehroute-output.exit-times true \
    --statistic-output "stats.intr.big.seed${SEED}.xml" \
    --no-warnings true

echo "[seed=$SEED big] sumo satnav ..."
sumo --net-file net.net.xml \
    --route-files "routes.big.seed${SEED}.xml" \
    --additional-files "$ADD_SATNAV" \
    --device.rerouting.probability 1.0 \
    --device.rerouting.period 60 \
    --device.rerouting.adaptation-interval 30 \
    --device.rerouting.adaptation-weight 0.5 \
    --begin 0 --end 14400 --seed "$SEED" \
    --vehroute-output "vehroutes.satnav.big.seed${SEED}.xml" \
    --vehroute-output.exit-times true \
    --statistic-output "stats.satnav.big.seed${SEED}.xml" \
    --no-warnings true

echo "[seed=$SEED big] done"
