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

# RUN IDENTITY WITHOUT DELETING ANYTHING. The CSV names carry only (tag, slice), and this
# directory is never cleaned, so a run that stops early leaves earlier attempts' files in place
# for every (tag, slice) it did not reach -- and a completion marker built from `ls | wc -l` would
# then count a mixture of two schedules as one complete run. Two attempts were abandoned in this
# directory on 2026-10-01, so that is a live hazard rather than a hypothetical one.
# Clearing floor_*.csv at startup would destroy the evidence of the abandoned run, which is its
# own hazard beside a live measurement. So identity is added instead of deletion: this run names
# itself, keeps its own schedule file, and writes a manifest of exactly the files it wrote with
# their digests. A reader partitions the directory by manifest rather than by glob.
RUNID=$(date +%Y%m%dT%H%M%S)-$$
SCHED=$OUT/schedule_$RUNID.txt
MANIFEST=$OUT/run_$RUNID.manifest
python floor_design.py "$M" "$SLICES" --emit "$SCHED" || exit 1
SCHED_SHA=$(shasum -a 256 "$SCHED" | cut -c1-16)
nsched=$(wc -l < "$SCHED")
[ "$nsched" -eq "$SLICES" ] || { echo "REFUSE: schedule has $nsched lines for $SLICES slices"; exit 1; }

NK=$(wc -l < "$KEYS")
PER=$(( (NK + SLICES - 1) / SLICES ))
echo "members $M  keys $NK  slices $SLICES  per-slice $PER  reps $REPS  ks $KS"
echo "members in index order: ${MEMBERS[*]}"
echo "run id $RUNID  schedule $SCHED  schedule sha256 $SCHED_SHA"
{
  echo "# runid $RUNID"
  echo "# started $(date +%Y-%m-%dT%H:%M:%S)"
  echo "# schedule $SCHED sha256 $SCHED_SHA"
  echo "# members $M slices $SLICES reps $REPS ks $KS keys $NK"
  echo "# columns: tag slice rc rows sha256_16 path"
} > "$MANIFEST"

ran=0
failed=0
for s in $(seq 1 "$SLICES"); do
  start=$(( (s - 1) * PER + 1 ))
  sed -n "${start},$((start + PER - 1))p" "$KEYS" > "$OUT/keys_s$s.txt"
  nk=$(wc -l < "$OUT/keys_s$s.txt")
  [ "$nk" -gt 0 ] || { echo "slice $s EMPTY -- stopping"; break; }
  # The order comes from the audited file, so the caption below cannot drift from what runs.
  order=$(awk -v want="$s" '$1 == want { $1 = ""; sub(/^ /, ""); print; exit }' "$SCHED")
  [ -n "$order" ] || { echo "REFUSE: schedule has no line for slice $s"; exit 1; }
  # SET EQUALITY, not a count. The property the design audit establishes is that each slice is a
  # PERMUTATION of the members; a cardinality check passes a line naming one member M times, and
  # this seam is exactly what the claim "bash runs the order that was audited" rests on.
  got_set=$(printf "%s\n" $order | sort -n | tr "\n" " ")
  want_set=$(seq 0 $(( M - 1 )) | tr "\n" " ")
  [ "$got_set" = "$want_set" ] || { echo "REFUSE: slice $s is not a permutation of the $M members."
    echo "        got  [$got_set]"; echo "        want [$want_set]"; exit 1; }
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
    csv=$OUT/floor_${tag}_s${s}.csv
    rows=$(wc -l < "$csv" 2>/dev/null || echo 0)
    printf "  %s rc=%s rows=%s\n" "$tag" "$rc" "$rows"
    ran=$(( ran + 1 ))
    [ "$rc" -eq 0 ] || failed=$(( failed + 1 ))
    printf "%s %s %s %s %s %s\n" "$tag" "s$s" "$rc" "$rows" \
      "$(shasum -a 256 "$csv" 2>/dev/null | cut -c1-16 || echo none)" "$csv" >> "$MANIFEST"
  done
done
want=$(( M * SLICES ))
echo "DONE $(date +%H:%M:%S)  invocations $ran of $want, nonzero rc $failed"
# The sentinel is written only after every CSV is closed. It reports what THIS run wrote, counted
# from its own manifest rather than from a glob over a directory it does not own, and it carries
# the nonzero-rc count -- so "64 attempted" and "64 succeeded" are distinguishable, which two
# markers built from a file count and an unconditional counter were not.
wrote=$(grep -c "^p" "$MANIFEST" 2>/dev/null || echo 0)
{
  echo "runid $RUNID"
  echo "schedule_sha256 $SCHED_SHA"
  echo "manifest $MANIFEST"
  echo "wrote $wrote of $want"
  echo "nonzero_rc $failed"
  echo "glob_count_in_dir $(ls "$OUT"/floor_*.csv 2>/dev/null | wc -l | tr -d " ")"
  if [ "$failed" -ne 0 ] || [ "$wrote" -ne "$want" ]; then
    echo "status INCOMPLETE"
  else
    echo "status COMPLETE"
  fi
} > "$OUT/floorcsv.done"
cat "$OUT/floorcsv.done"
# The glob count is reported beside the manifest count ON PURPOSE: if they differ, files from an
# earlier attempt are present in the directory and a reader must partition by manifest.
if [ "$wrote" -ne "$(ls "$OUT"/floor_*.csv 2>/dev/null | wc -l | tr -d " ")" ]; then
  echo "NOTE: the directory holds more floor CSVs than this run wrote, so it contains at least"
  echo "      one earlier attempt. Partition by $MANIFEST; nothing has been deleted."
fi
