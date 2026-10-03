#!/bin/bash
#SBATCH -A mkt
#SBATCH -p mkt
#SBATCH -c 32
#SBATCH --time=5:00:00
#SBATCH -J k4dip-dl
#SBATCH -o /scr/bobbyy/k4dip/slurm-%j.out
set -u
ROOT=/scr/bobbyy/k4dip
BUILD=/scr/bobbyy/pfs/ship
OUT=$ROOT/dlmc_float32_ship_p1.csv
KEYS=$ROOT/dlmc_keys.txt
WANTKEYS=3820   # and 9 rows each: 3 widths x 3 arms

# SELF-SKIPPING BACKSTOP. Job 17701502 runs ss then dlmc inside one 6h limit; at the observed
# 4.45 rows/s the dlmc phase must hold above 2.79 rows/s to fit. If it already finished, this
# must NOT re-measure and must NOT overwrite a complete file.
#
# THE GATE COUNTS KEYS WITH EXACTLY 9 ROWS, NOT TOTAL ROWS. A 34,380-row total is satisfied by
# one key with 6 rows and another with 12, which is the defect already on record in this
# project's sentinel -- and it is WORSE here than there, because there the total was a reported
# FIELD (a wrong value at least appears in an artefact) while here it is a control-flow decision
# whose passing branch writes nothing. Partial data would become complete data with no trace
# that a check was made. So the passing branch is LOUD: the per-key census prints either way.
echo "backstop for 17701502's dlmc phase  $(date '+%F %T')  job ${SLURM_JOB_ID:-none}"
if [ -f "$OUT" ]; then
  HAVE=$(( $(wc -l < "$OUT") - 1 ))
  CENSUS=$(awk -F, 'NR>1{c[$3]++}END{e=0;s=0;o=0;for(k in c){if(c[k]==9)e++;else if(c[k]<9)s++;else o++};
           printf "%d %d %d %d", e, s, o, length(c)}' "$OUT")
  set -- $CENSUS
  echo "per-key census: $1 keys with exactly 9 rows, $2 short, $3 over, $4 keys present"
  echo "  (total rows $HAVE; a total is NOT the check -- 9 rows per key is)"
  if [ "$1" -ge "$WANTKEYS" ] && [ "$2" = "0" ] && [ "$3" = "0" ]; then
    echo "SKIP: $1 of $WANTKEYS keys complete at exactly 9 rows, none short or over."
    echo "      Leaving the existing file untouched."
    exit 0
  fi
  echo "NOT COMPLETE by the per-key gate, so re-measuring."
  echo "incomplete -- re-measuring to a SEPARATE path so the partial file is preserved as evidence"
  OUT=$ROOT/dlmc_float32_ship_p1_backstop.csv
fi
echo "object sha256 $(sha256sum "$BUILD"/src/scorch_ops*.so | cut -c1-16)   (expect 4c11296ce2a68c3d)"
echo "HOOKS: SCORCH_SPMM_* strings = $(strings -n 6 "$BUILD"/src/scorch_ops*.so | grep -c '^SCORCH_SPMM_')"
env | grep '^SCORCH_' || echo "ambient SCORCH_*: none"
[ -f "$KEYS" ] || { echo "REFUSE: $KEYS absent"; exit 1; }
echo "keys $(wc -l < "$KEYS")  sha $(sha256sum "$KEYS" | cut -c1-16)   (expect 3820, a8132f50d2c44fe4)"
. /scr/bobbyy/venv/bin/activate || exit 1
POOL=$(OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32} python -c 'import torch;print(torch.get_num_threads())')
echo "torch pool $POOL"
[ "$POOL" -ge 8 ] || { echo "REFUSE: pool $POOL"; exit 1; }
echo "--- dlmc start $(date +%H:%M:%S) -> $OUT"
OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32} PYTHONPATH="$BUILD/src" \
  python "$ROOT/mrgate.py" --cache /scr/bobbyy/mklsweep/csrcache/dlmc --keys "$KEYS" \
    --csv "$OUT" --dtype float32 --ks 2,4,8 --reps 9 --build ship --slice all --pass-tag p1 \
    > "$ROOT/dlmc_backstop.log" 2>&1
RC=$?
R=0; [ -f "$OUT" ] && R=$(( $(wc -l < "$OUT") - 1 ))
echo "--- dlmc done $(date +%H:%M:%S) rc=$RC rows=$R"
if [ -f "$OUT" ]; then
  awk -F, 'NR>1{c[$3]++}END{e=0;s=0;o=0;for(k in c){if(c[k]==9)e++;else if(c[k]<9)s++;else o++};
    printf "final per-key census: %d exactly-9, %d short, %d over, %d keys\n", e, s, o, length(c)}' "$OUT"
fi
exit $RC
