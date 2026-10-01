#!/bin/bash
# Boundary-aligned /proc/stat snapshots for the port-boundedness whole-pass measurement.
# A read taken part-way through an object-pass makes the interval miss a specific slice of the
# 3820-key sweep, which reintroduces exactly the phase selection the two-read design removes.
# Triggered on the stage log's done-line COUNT, never on a clock, because boundary times are
# estimates. Bounded lifetime: a deadline and at most four snapshots.
set -u
S=/scratch/bobbyy/c210_stage.log
O=/scratch/bobbyy/portbound_boundaries.txt
DEADLINE=$(( $(date +%s) + 5700 ))
last=$(grep -cE '^    --- p[1-6] (aa|bb) done ' "$S")
echo "START last=$last deadline=$DEADLINE hms=$(date +%H:%M:%S)" >> "$O"
n=0
while [ "$(date +%s)" -lt "$DEADLINE" ] && [ "$n" -lt 4 ]; do
  sleep 15
  cur=$(grep -cE '^    --- p[1-6] (aa|bb) done ' "$S")
  if [ "$cur" -gt "$last" ]; then
    n=$((n+1)); last=$cur
    { echo "SNAP done=$cur epoch_ns=$(date +%s%N) hms=$(date +%H:%M:%S)"
      grep -E '^cpu[0-9]+ ' /proc/stat | awk '{print $1, $2+$3+$4+$7+$8+$9}'
    } >> "$O"
  fi
done
echo "EXIT n=$n hms=$(date +%H:%M:%S)" >> "$O"
