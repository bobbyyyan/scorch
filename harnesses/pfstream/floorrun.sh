#!/bin/bash
# Time the floor members with the member order permuted per slice.
#
# WHY PERMUTE. The members are separate processes, so running them as sequential blocks would
# measure later members later in wall-clock time, and any drift -- thermal, a co-tenant arriving,
# page-cache state -- would appear as a difference between members. That difference is exactly
# what the floor exists to measure, so the schedule has to spread each member over the run.
#
# THE SCHEDULE IS NOT COMPUTED HERE. floor_design.py builds it, audits it on both balance axes,
# refuses an unbalanced one, and emits the member order per slice. This script executes that file.
# An earlier version computed the order inline with its own arithmetic and printed a caption
# derived separately from it; the caption said "rotated by 1" for a schedule rotated by 4.
#
# A BALANCED SCHEDULE DOES NOT SURVIVE WINDOWING: analysing a subset of slices can reintroduce
# the imbalance. The slice label is in every output row so the analysis re-certifies balance on
# the window it uses.
set -u

ROOT=${1:-}
[ -n "$ROOT" ] || { echo "usage: floorrun.sh <root> [slices] [reps] [ks]"; exit 1; }
[ -d "$ROOT" ] || { echo "REFUSE: root $ROOT is not a directory"; exit 1; }
SLICES=${2:-4}
REPS=${3:-7}
KS=${4:-2,4,8,16,32}

# Every path derives from the root, and each is overridable for a host that lays things out
# differently. Nothing below is a literal host path.
TREE=${FLOOR_TREE:-$ROOT/scorch-beat-mkl}
RUN=${FLOOR_RUN:-$ROOT/pfs/floorrun}
KEYS=${FLOOR_KEYS:-$ROOT/floor_keys.txt}
OUT=${FLOOR_OUT:-$ROOT/floorcsv}
CACHE=${FLOOR_CACHE:-$ROOT/ss-npz}
ACTIVATE=${FLOOR_ACTIVATE:-$ROOT/pfs-activate.sh}

for spec in "TREE=$TREE" "RUN=$RUN" "CACHE=$CACHE"; do
  name=${spec%%=*}; path=${spec#*=}
  [ -d "$path" ] || { echo "REFUSE: $name=$path is not a directory"; exit 1; }
done
for spec in "KEYS=$KEYS" "ACTIVATE=$ACTIVATE"; do
  name=${spec%%=*}; path=${spec#*=}
  [ -f "$path" ] || { echo "REFUSE: $name=$path is not a file"; exit 1; }
done
HARNESS=$TREE/harnesses/pfstream
for f in mrgate.py floor_design.py; do
  [ -f "$HARNESS/$f" ] || { echo "REFUSE: $HARNESS/$f absent"; exit 1; }
done

# Probe the activation in a subshell first: `source` runs in the current shell, so a shim that
# ends in `exit 1` would terminate this script and the `||` branch would never print.
( source "$ACTIVATE" >/dev/null 2>&1 ) \
  || { echo "REFUSE: $ACTIVATE fails when sourced (probed in a subshell, which contains both a"
       echo "        nonzero return and an outright exit)"; exit 1; }
source "$ACTIVATE" || { echo "REFUSE: $ACTIVATE failed on the real source"; exit 1; }
python -c "import torch, sys; sys.stderr.write('torch %s at %s\n' % (torch.__version__, sys.executable))" \
  || { echo "REFUSE: the activated environment cannot import torch"; exit 1; }

cd "$HARNESS" || exit 1
mkdir -p "$OUT" || exit 1

# Members are the pNNN directories floorsetup.sh built, in sorted order. Reading them rather than
# restating the pad list means the two scripts cannot disagree about how many members exist.
MEMBERS=()
for d in "$RUN"/p*/; do
  [ -d "$d" ] || continue
  t=$(basename "$d")
  [ -f "$d/scorch_ops.so" ] || { echo "REFUSE: member $t has no scorch_ops.so"; exit 1; }
  MEMBERS+=("$t")
done
M=${#MEMBERS[@]}
[ "$M" -gt 0 ] || { echo "REFUSE: no member directories under $RUN -- run floorsetup.sh first"; exit 1; }

SCHED=$OUT/schedule.txt
python floor_design.py "$M" "$SLICES" --emit "$SCHED" || exit 1
nsched=$(wc -l < "$SCHED")
[ "$nsched" -eq "$SLICES" ] || { echo "REFUSE: schedule has $nsched lines for $SLICES slices"; exit 1; }

NK=$(wc -l < "$KEYS")
PER=$(( (NK + SLICES - 1) / SLICES ))
echo "members $M  keys $NK  slices $SLICES  per-slice $PER  reps $REPS  ks $KS"
echo "members in index order: ${MEMBERS[*]}"

ran=0
for s in $(seq 1 "$SLICES"); do
  start=$(( (s - 1) * PER + 1 ))
  sed -n "${start},$((start + PER - 1))p" "$KEYS" > "$OUT/keys_s$s.txt"
  nk=$(wc -l < "$OUT/keys_s$s.txt")
  [ "$nk" -gt 0 ] || { echo "slice $s EMPTY -- stopping"; break; }
  # The order comes from the audited file, so the caption below cannot drift from what runs.
  order=$(awk -v want="$s" '$1 == want { $1 = ""; print; exit }' "$SCHED")
  [ -n "$order" ] || { echo "REFUSE: schedule has no line for slice $s"; exit 1; }
  nord=$(echo $order | wc -w)
  [ "$nord" -eq "$M" ] || { echo "REFUSE: slice $s lists $nord members, expected $M"; exit 1; }
  names=""
  for j in $order; do names="$names ${MEMBERS[$j]}"; done
  echo "=== slice $s: $nk keys, member order$names"
  for j in $order; do
    tag=${MEMBERS[$j]}
    PYTHONPATH="$RUN/$tag" python mrgate.py \
      --cache "$CACHE" --keys "$OUT/keys_s$s.txt" \
      --csv "$OUT/floor_${tag}_s${s}.csv" \
      --ks "$KS" --reps "$REPS" --target-ms 15 \
      --build "$tag" --slice "s$s" --pass-tag "s$s" >> "$OUT/run.log" 2>&1
    rc=$?
    printf "  %s rc=%s rows=%s\n" "$tag" "$rc" "$(wc -l < "$OUT/floor_${tag}_s${s}.csv" 2>/dev/null || echo 0)"
    ran=$(( ran + 1 ))
  done
done
echo "DONE $(date +%H:%M:%S)  invocations $ran of $(( M * SLICES ))"
# The sentinel a comparator requires, written only after every CSV is closed. It carries the
# count, so a reader can tell a complete run from one that stopped on an empty slice.
ls "$OUT"/floor_*.csv 2>/dev/null | wc -l > "$OUT/floorcsv.done"
echo "csv files: $(cat "$OUT/floorcsv.done")"
