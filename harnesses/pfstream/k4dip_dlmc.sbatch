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
  # SET IDENTITY, not cardinality. Counting keys-with-9-rows is corpus-INDEPENDENT (9 = 3 widths
  # x 3 arms), so a file holding 4594 SuiteSparse keys plus 1000 DLMC keys gives 5594 exactly-9,
  # 0 short, 0 over and passes any count-based gate. This harness measures both corpora in one
  # job and this project has already recorded "a float32 DLMC screen wrote files named for a
  # float64 SuiteSparse one", so a dlmc-named file carrying ss keys is the documented failure.
  # Comparing the CSV's key SET against $KEYS catches it by identity. awk membership is used
  # rather than sort+comm because a merge is order-sensitive and a locale mismatch between the
  # sort and the merge silently fabricates set differences.
  # AND THE SAME DEFECT RECURS ONE LEVEL DOWN. A key whose nine rows are nine copies of a single
  # (k, arm) cell has no k=4 and no k=8 data, yet reads identically to a complete key on every
  # count above: present, exactly 9 rows, not short, not over, not unknown, not missing. A count
  # at one level is invariant to a permutation at the next, so the gate must name the population
  # at EVERY level of the key it aggregates over -- corpus, key, cell. `trip` counts distinct
  # (key, k, arm) triples per key and requires 9, which subsumes the short case too.
  # `nw` is length(want), not a line count: a duplicated or blank line in $KEYS would otherwise
  # make `wanted` disagree with the set actually built.
  # IDENTITY AT BOTH LEVELS, AND A VALUE CHECK BELOW THEM. The previous revision checked keys by
  # identity (!(k in want)) and cells by a NUMBER (nt[k] != 9) -- two standards in one script,
  # with the weaker one at the level the rule was meant to strengthen. A count is invariant to a
  # permutation of WHICH members, so a file with all 3820 keys at 9 distinct cells and ZERO k=8
  # rows anywhere passed. That is reachable rather than hypothetical: mrgate.py:27 defaults --ks
  # to "1,2,4,8,16,64", matching neither this study's {2,4,8} nor any screen's set, and the gate
  # judges a file written by a job whose --ks it cannot see.
  #
  # So the wanted (k, arm) pairs are ENUMERATED here. Counting to 9 over exactly 9 enumerated
  # members while requiring zero non-members forces set EQUALITY, which is identity rather than
  # cardinality.
  #
  # And `badms` is the level below both: csv.DictWriter emits a MISSING key as an EMPTY field, so
  # a dropped value is a complete-looking row. The gate previously verified a cell EXISTS and
  # never that it holds a usable number.
  CENSUS=$(awk -F, '
    BEGIN{ nk=split("2,4,8",KS,","); na=split("ours,oursb,mkl",AR,",")
           for(i=1;i<=nk;i++) for(j=1;j<=na;j++) wantcell[KS[i] SUBSEP AR[j]]=1
           NCELL=length(wantcell) }
    NR==FNR{ if($0!="") want[$0]=1; next }
    FNR>1{ c[$3]++
           cell=$4 SUBSEP $8
           if(!(cell in wantcell)) badcell++; else seen[$3 SUBSEP cell]=1
           if($9=="" || $9+0<=0) badms++ }
    END{ e=0;s=0;o=0;unk=0;miss=0;cellbad=0
         for(x in seen){ split(x,q,SUBSEP); have[q[1]]++ }
         for(k in c){ if(!(k in want)) unk++
                      if(c[k]==9) e++; else if(c[k]<9) s++; else o++
                      if(have[k]!=NCELL) cellbad++ }
         for(k in want) if(!(k in c)) miss++
         printf "%d %d %d %d %d %d %d %d %d %d", e, s, o, length(c), unk, miss,
                length(want), cellbad, badcell+0, badms+0 }' \
    "$KEYS" "$OUT")
  set -- $CENSUS
  echo "per-key census: exactly9=$1 short=$2 over=$3 keys=$4 unknown=$5 missing=$6 wanted=$7"
  echo "  cell-level: keys-missing-a-wanted-cell=$8 rows-outside-the-wanted-cells=$9 unusable-ms=${10}"
  echo "  (total rows $HAVE is NOT the check: identity of the key set is)"
  if [ "$1" = "$7" ] && [ "$2" = "0" ] && [ "$3" = "0" ] && [ "$4" = "$7" ] \
     && [ "$5" = "0" ] && [ "$6" = "0" ] && [ "$8" = "0" ] && [ "$9" = "0" ] \
     && [ "${10}" = "0" ]; then
    echo "SKIP: all $7 wanted keys present with 9 DISTINCT (k,arm) cells each; none short, over,"
    echo "      unknown, missing, or holding a repeated cell."
    echo "      Leaving the existing file untouched."
    exit 0
  fi
  echo "NOT COMPLETE by the key-set gate, so re-measuring."
  echo "incomplete -- re-measuring to a SEPARATE path so the partial file is preserved as evidence"
  OUT=$ROOT/dlmc_float32_ship_p1_backstop.csv
fi
# COMPARE, do not print beside. A line reading "expect X" next to a measured value reads as
# provenance to anyone auditing the log afterwards, which is worse than no check.
OBJSHA=$(sha256sum "$BUILD"/src/scorch_ops*.so | cut -c1-16)
echo "object sha256 $OBJSHA"
[ "$OBJSHA" = "4c11296ce2a68c3d" ] || { echo "REFUSE: object is $OBJSHA, expected 4c11296ce2a68c3d"; exit 1; }
echo "HOOKS: SCORCH_SPMM_* strings = $(strings -n 6 "$BUILD"/src/scorch_ops*.so | grep -c '^SCORCH_SPMM_')"
env | grep '^SCORCH_' || echo "ambient SCORCH_*: none"
[ -f "$KEYS" ] || { echo "REFUSE: $KEYS absent"; exit 1; }
KN=$(wc -l < "$KEYS" | tr -d ' '); KSHA=$(sha256sum "$KEYS" | cut -c1-16)
echo "keys $KN  sha $KSHA"
[ "$KN" = "3820" ] || { echo "REFUSE: key list has $KN lines, expected 3820"; exit 1; }
[ "$KSHA" = "a8132f50d2c44fe4" ] || { echo "REFUSE: key list sha $KSHA, expected a8132f50d2c44fe4"; exit 1; }
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
