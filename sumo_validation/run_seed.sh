#!/usr/bin/env bash
# Run the small-bbox -p 2.0 SUMO pipeline (intr + satnav) for one randomTrips seed.
# Outputs: vehroutes/edgedata/stats tagged with the seed.
# Usage:   bash run_seed.sh <seed>
set -euo pipefail
SEED="${1:?usage: bash run_seed.sh <seed>}"
cd "$(dirname "$0")"

# Fall back to the known SUMO install path if $SUMO_HOME isn't inherited
# from the parent shell (common with zsh -> bash subshell on macOS).
SUMO_HOME="${SUMO_HOME:-/Library/Frameworks/EclipseSUMO.framework/Versions/1.26.0/EclipseSUMO/share/sumo}"
if [ ! -x "$SUMO_HOME/tools/randomTrips.py" ]; then
    echo "ERROR: cannot find randomTrips.py under SUMO_HOME=$SUMO_HOME" >&2
    exit 1
fi
echo "[seed=$SEED] SUMO_HOME=$SUMO_HOME"

ADD_INTR="edgedata.intr.seed${SEED}.add.xml"
ADD_SATNAV="edgedata.satnav.seed${SEED}.add.xml"
cat > "$ADD_INTR" <<EOF
<additional><edgeData id="agg" file="edgedata.intr.small.seed${SEED}.xml" period="14400" excludeEmpty="true"/></additional>
EOF
cat > "$ADD_SATNAV" <<EOF
<additional><edgeData id="agg" file="edgedata.satnav.small.seed${SEED}.xml" period="14400" excludeEmpty="true"/></additional>
EOF

echo "[seed=$SEED] randomTrips ..."
# Invoke via python3 explicitly — randomTrips.py's shebang is `#!/usr/bin/env python`
# which fails on macOS where only `python3` is on PATH.
python3 "$SUMO_HOME/tools/randomTrips.py" -n net.small.net.xml -e 14400 -p 2.0 \
    --vehicle-class passenger --seed "$SEED" \
    -o "trips.small.seed${SEED}.xml" --validate

echo "[seed=$SEED] duarouter ..."
duarouter --net-file net.small.net.xml \
    --route-files "trips.small.seed${SEED}.xml" \
    --output-file "routes.small.seed${SEED}.xml" \
    --remove-loops --ignore-errors --seed "$SEED"

echo "[seed=$SEED] sumo intr ..."
sumo --net-file net.small.net.xml \
    --route-files "routes.small.seed${SEED}.xml" \
    --additional-files "$ADD_INTR" \
    --begin 0 --end 14400 --seed "$SEED" \
    --vehroute-output "vehroutes.intr.small.seed${SEED}.xml" \
    --vehroute-output.exit-times true \
    --statistic-output "stats.intr.small.seed${SEED}.xml" \
    --no-warnings true

echo "[seed=$SEED] sumo satnav ..."
sumo --net-file net.small.net.xml \
    --route-files "routes.small.seed${SEED}.xml" \
    --additional-files "$ADD_SATNAV" \
    --device.rerouting.probability 1.0 \
    --device.rerouting.period 60 \
    --device.rerouting.adaptation-interval 30 \
    --device.rerouting.adaptation-weight 0.5 \
    --begin 0 --end 14400 --seed "$SEED" \
    --vehroute-output "vehroutes.satnav.small.seed${SEED}.xml" \
    --vehroute-output.exit-times true \
    --statistic-output "stats.satnav.small.seed${SEED}.xml" \
    --no-warnings true

echo "[seed=$SEED] done"
