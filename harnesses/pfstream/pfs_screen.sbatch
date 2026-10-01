#!/bin/bash
#SBATCH -A mkt
#SBATCH -p mkt
#SBATCH -c 32
#SBATCH -t 06:00:00
#SBATCH -J pfs-screen
#SBATCH -o /scr/bobbyy/pfs/screen.log
# The PFSTREAM screen: four BUILDS x five widths x a strided SuiteSparse sample, with ours/oursb/mkl
# inside each process.
#
# WHY THE ARMS ARE BUILDS AND MKL IS TIMED IN EVERY PROCESS. The three configurations cannot coexist
# in one object, so drift between builds is the threat; timing MKL in the same process as ours in
# every cell makes ours/MKL immune to it even where ours-here against ours-there is not.
#
# WHY shpb EXISTS. It is ship's configuration built a second time at an equal-length path. That pair
# is the floor every arm must clear, and it has to be a second build: the build floor's whole
# increment over the arm floor is reproduced by two byte-identical objects, so a floor taken inside
# one binary understates what a cross-build comparison must beat.
#
# WHY THE BUILD ORDER ROTATES PER PASS AND IS BALANCED. Four builds and four passes, each build
# leading exactly once, so no build systematically occupies the warm or the cold end of a pass. An
# unbalanced rotation reads as a build effect. The order is recorded as it LANDS, not as scheduled.
set -u
cd /scr/bobbyy/pfs || exit 1
source /scr/bobbyy/venv/bin/activate || exit 1
CACHE=/scr/bobbyy/mklsweep/csrcache/ss
KEYS=/scr/bobbyy/pfs/ss_sample.txt
mkdir -p /scr/bobbyy/pfs/csv
echo "host $(hostname)  pool $(python -c 'import torch;print(torch.get_num_threads())')  $(date +%H:%M:%S)"
echo "corpus $(wc -l < $KEYS) matrices, widths 2,4,8,16,32"
# A LATIN SQUARE over four passes: every build leads once and occupies each position once.
ORDERS=( "ship shpb strm nopf" "shpb strm nopf ship" "strm nopf ship shpb" "nopf ship shpb strm" )
for p in 1 2 3 4; do
  order=${ORDERS[$((p-1))]}
  echo "== pass p$p order: $order"
  for tag in $order; do
    out=/scr/bobbyy/pfs/csv/ss_${tag}_p${p}.csv
    echo "  --- p$p $tag start $(date +%H:%M:%S)"
    OMP_NUM_THREADS=32 PYTHONPATH=/scr/bobbyy/pfs/$tag/src \
      python /scr/bobbyy/pfs/mrgate.py --cache "$CACHE" --keys "$KEYS" --csv "$out" \
        --dtype float32 --ks 2,4,8,16,32 --reps 9 --target-ms 15 \
        --pass-tag p$p --build "$tag" --slice all > /scr/bobbyy/pfs/csv/ss_${tag}_p${p}.log 2>&1
    rc=$?
    rows=$(( $(wc -l < "$out" 2>/dev/null || echo 1) - 1 ))
    echo "  --- p$p $tag done  $(date +%H:%M:%S)  rc=$rc  ($rows rows)"
    [ "$rc" = "0" ] || { echo "REFUSE: p$p $tag exited $rc"; tail -4 /scr/bobbyy/pfs/csv/ss_${tag}_p${p}.log; exit 1; }
  done
done
echo "MKT_PFS_ALL_PASSES_DONE $(date +%H:%M:%S)"
