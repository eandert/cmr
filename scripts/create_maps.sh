#!/bin/bash
# Create SUMO maps for highway and rural scenarios from OSM data.
# Requires: netconvert, randomTrips.py (from SUMO), wget
#
# Usage: bash scripts/create_maps.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
SUMO_TOOLS="${SUMO_HOME:-/usr/share/sumo}/tools"
TYPEMAP="${SUMO_HOME:-/usr/share/sumo}/data/typemap/osmNetconvert.typ.xml"

echo "=== Creating SUMO maps ==="
echo "SUMO_TOOLS: $SUMO_TOOLS"
echo "TYPEMAP: $TYPEMAP"

# -------------------------------------------------------
# Highway map: I-10 near Tempe/Chandler, AZ
# Straight ~3km stretch of freeway with on/off ramps
# -------------------------------------------------------
HIGHWAY_DIR="$REPO_ROOT/maps/highway"
mkdir -p "$HIGHWAY_DIR"
echo ""
echo "--- Highway Map (I-10 near Chandler) ---"

# Bounding box: west,south,east,north
# I-10 between Chandler Blvd and Ray Rd exits
HWY_BBOX="-111.8650,33.2950,-111.8200,33.3150"

echo "Downloading OSM data..."
wget -q -O "$HIGHWAY_DIR/osm_bbox.osm.xml" \
  "https://overpass-api.de/api/map?bbox=$HWY_BBOX"

echo "Running netconvert..."
netconvert \
  --type-files "$TYPEMAP" \
  --osm-files "$HIGHWAY_DIR/osm_bbox.osm.xml" \
  --output-file "$HIGHWAY_DIR/osm.net.xml" \
  --output.street-names true \
  --output.original-names true \
  --geometry.remove true \
  --roundabouts.guess true \
  --ramps.guess true \
  --tls.discard-simple true \
  --tls.join true \
  --tls.guess-signals true \
  --tls.default-type actuated \
  --junctions.join true \
  --junctions.corner-detail 5 \
  --verbose true 2>&1 | tail -5

echo "Generating trips..."
python3 "$SUMO_TOOLS/randomTrips.py" \
  -n "$HIGHWAY_DIR/osm.net.xml" \
  --seed 42 \
  --fringe-factor 5 \
  -p 1.5 \
  -o "$HIGHWAY_DIR/osm.passenger.trips.xml" \
  -e 3600 \
  --vehicle-class passenger \
  --vclass passenger \
  --prefix veh \
  --min-distance 300 \
  --trip-attributes 'departLane="best"' \
  --fringe-start-attributes 'departSpeed="max"' \
  --allow-fringe.min-length 1000 \
  --lanes \
  --validate

echo "Generating polygons..."
python3 "$SUMO_TOOLS/polyconvert.py" \
  --osm-files "$HIGHWAY_DIR/osm_bbox.osm.xml" \
  --net-file "$HIGHWAY_DIR/osm.net.xml" \
  -o "$HIGHWAY_DIR/osm.poly.xml" 2>/dev/null || \
  polyconvert \
    --osm-files "$HIGHWAY_DIR/osm_bbox.osm.xml" \
    --net-file "$HIGHWAY_DIR/osm.net.xml" \
    -o "$HIGHWAY_DIR/osm.poly.xml" 2>/dev/null || \
  echo '<?xml version="1.0" encoding="UTF-8"?><additional/>' > "$HIGHWAY_DIR/osm.poly.xml"

# Write sumocfg
cat > "$HIGHWAY_DIR/osm.sumocfg" << 'SUMOCFG'
<?xml version="1.0" encoding="UTF-8"?>
<configuration>
    <input>
        <net-file value="osm.net.xml"/>
        <route-files value="osm.passenger.trips.xml"/>
        <additional-files value="osm.poly.xml"/>
    </input>
    <processing>
        <ignore-route-errors value="true"/>
    </processing>
    <routing>
        <device.rerouting.adaptation-steps value="180"/>
    </routing>
    <report>
        <verbose value="true"/>
        <duration-log.statistics value="true"/>
        <no-step-log value="true"/>
    </report>
</configuration>
SUMOCFG

echo "Highway map created: $HIGHWAY_DIR"

# -------------------------------------------------------
# Rural map: area south of Queen Creek, AZ
# Sparse desert roads with few intersections
# -------------------------------------------------------
RURAL_DIR="$REPO_ROOT/maps/rural"
mkdir -p "$RURAL_DIR"
echo ""
echo "--- Rural Map (south of Queen Creek) ---"

# Bounding box: sparse area south of Queen Creek
RURAL_BBOX="-111.6500,33.1800,-111.5900,33.2200"

echo "Downloading OSM data..."
wget -q -O "$RURAL_DIR/osm_bbox.osm.xml" \
  "https://overpass-api.de/api/map?bbox=$RURAL_BBOX"

echo "Running netconvert..."
netconvert \
  --type-files "$TYPEMAP" \
  --osm-files "$RURAL_DIR/osm_bbox.osm.xml" \
  --output-file "$RURAL_DIR/osm.net.xml" \
  --output.street-names true \
  --output.original-names true \
  --geometry.remove true \
  --roundabouts.guess true \
  --ramps.guess true \
  --tls.discard-simple true \
  --tls.join true \
  --tls.guess-signals true \
  --tls.default-type actuated \
  --junctions.join true \
  --junctions.corner-detail 5 \
  --verbose true 2>&1 | tail -5

echo "Generating trips..."
python3 "$SUMO_TOOLS/randomTrips.py" \
  -n "$RURAL_DIR/osm.net.xml" \
  --seed 42 \
  --fringe-factor 5 \
  -p 3.0 \
  -o "$RURAL_DIR/osm.passenger.trips.xml" \
  -e 3600 \
  --vehicle-class passenger \
  --vclass passenger \
  --prefix veh \
  --min-distance 200 \
  --trip-attributes 'departLane="best"' \
  --fringe-start-attributes 'departSpeed="max"' \
  --allow-fringe.min-length 500 \
  --lanes \
  --validate

echo "Generating polygons..."
python3 "$SUMO_TOOLS/polyconvert.py" \
  --osm-files "$RURAL_DIR/osm_bbox.osm.xml" \
  --net-file "$RURAL_DIR/osm.net.xml" \
  -o "$RURAL_DIR/osm.poly.xml" 2>/dev/null || \
  polyconvert \
    --osm-files "$RURAL_DIR/osm_bbox.osm.xml" \
    --net-file "$RURAL_DIR/osm.net.xml" \
    -o "$RURAL_DIR/osm.poly.xml" 2>/dev/null || \
  echo '<?xml version="1.0" encoding="UTF-8"?><additional/>' > "$RURAL_DIR/osm.poly.xml"

# Write sumocfg
cat > "$RURAL_DIR/osm.sumocfg" << 'SUMOCFG'
<?xml version="1.0" encoding="UTF-8"?>
<configuration>
    <input>
        <net-file value="osm.net.xml"/>
        <route-files value="osm.passenger.trips.xml"/>
        <additional-files value="osm.poly.xml"/>
    </input>
    <processing>
        <ignore-route-errors value="true"/>
    </processing>
    <routing>
        <device.rerouting.adaptation-steps value="180"/>
    </routing>
    <report>
        <verbose value="true"/>
        <duration-log.statistics value="true"/>
        <no-step-log value="true"/>
    </report>
</configuration>
SUMOCFG

echo "Rural map created: $RURAL_DIR"

echo ""
echo "=== Done! ==="
echo "Maps created:"
echo "  maps/highway/ - I-10 freeway near Chandler, AZ"
echo "  maps/rural/   - Sparse roads south of Queen Creek, AZ"
echo ""
echo "To use: python3 src/experiment_runner.py --suite ... --map highway"
echo "        python3 src/experiment_runner.py --suite ... --map rural"
