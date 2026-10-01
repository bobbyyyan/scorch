#!/bin/bash
#SBATCH -A mkt
#SBATCH -p mkt
#SBATCH -c 32
#SBATCH -t 08:00:00
#SBATCH -J pfs-screen-f64
#SBATCH -o /scr/bobbyy/pfs/screen_f64.log
# The PFSTREAM screen, float64 cell. Same four builds, same Latin square, same corpus, same
# calibration as pfs_screen.sbatch -- only the dtype and the width set differ, and both differ for
# reasons read out of spmm.h rather than chosen.
#
# WHY THIS CELL IS NOT OPTIONAL. PREREG amendment 8: the screen covers one of three reachable
# (entry point, dtype) cells. float64 reaches scorch_spmm_row_regblock through the drop-in SpMM and
# carries a live result -- 2.273x geomean against MKL with 0 of 129 cells below parity -- so a
# PFSTREAM regression there would be a regression in the strongest number this kernel has. All six
# release call sites pass SCORCH_SPMM_PFLIM, verified per-STATEMENT rather than per-line (two sites
# wrap, and a per-line test reported NO BOUND for them -- a false negative pointing at absence).
#
# WHY IT RUNS AFTER THE float32 SCREEN AND NOT BESIDE IT. --dependency=afterok, because mkt1's
# memory bandwidth is not cgroup-partitioned: a second timing job on the same node is a perturbation
# of the first, not a neighbour. afterok rather than afterany so a failed f32 screen does not hand
# this one a build set that is already in question.
#
# WHY THESE WIDTHS, AND WHAT THE float32 SCREEN DOES NOT REACH. float64 has scorch_simd<double>
# ::lanes == 4 (spmm.h:2178), so narrow_k is B1_size <= 4*SL = 16, nvec = ceil(k/4), and
# full_last is k % 4 == 0 (spmm.h:3979-3982). Every width below names the kernel branch it exists to
# reach; none is a workload shape, and k=32 is absent because at float64 it is not the narrow-k path
# at all, so PFSTREAM is unreachable there:
#
#     k=2    exact_width routes k in {1,2,3} to scorch_spmm_row_narrow_exact, so regblock is
#            BYPASSED -- the NEGATIVE CONTROL. At float64 halfvec_exact would also be true here
#            (lanes == 2), held off only by SCORCH_SPMM_HALFVEC_F64 shipping at 0, so two
#            independent mechanisms keep this width inert and any movement in it is floor.
#     k=4    nvec 1, full tail
#     k=6    nvec 2, MASKED tail (6 % 4 = 2)
#     k=8    nvec 2, full tail
#     k=10   nvec 3, MASKED tail
#     k=12   nvec 3, full tail
#     k=16   nvec 4, full tail
#
# That set covers nvec 1..4 and both tail states. The float32 screen's width set (2,4,8,16,32) with
# SL=8 covers nvec 1,2,4 and full_last only -- it reaches NEITHER nvec 3 (k=17..24) NOR a masked
# tail (any k not a multiple of 8). Those are two of the kernel's own branch axes left unexercised
# by the live screen, recorded as PREREG amendment 10 rather than quietly fixed here.
set -u
cd /scr/bobbyy/pfs || exit 1
source /scr/bobbyy/venv/bin/activate || exit 1
CACHE=/scr/bobbyy/mklsweep/csrcache/ss
KEYS=/scr/bobbyy/pfs/ss_sample.txt
KS=2,4,6,8,10,12,16
mkdir -p /scr/bobbyy/pfs/csv64
echo "host $(hostname)  pool $(python -c 'import torch;print(torch.get_num_threads())')  $(date +%H:%M:%S)"
echo "corpus $(wc -l < $KEYS) matrices, dtype float64, widths $KS"
# The identical Latin square: every build leads once and occupies each position once.
ORDERS=( "ship shpb strm nopf" "shpb strm nopf ship" "strm nopf ship shpb" "nopf ship shpb strm" )
for p in 1 2 3 4; do
  order=${ORDERS[$((p-1))]}
  echo "== pass p$p order: $order"
  for tag in $order; do
    out=/scr/bobbyy/pfs/csv64/ss64_${tag}_p${p}.csv
    echo "  --- p$p $tag start $(date +%H:%M:%S)"
    OMP_NUM_THREADS=32 PYTHONPATH=/scr/bobbyy/pfs/$tag/src \
      python /scr/bobbyy/pfs/mrgate.py --cache "$CACHE" --keys "$KEYS" --csv "$out" \
        --dtype float64 --ks "$KS" --reps 9 --target-ms 15 \
        --pass-tag p$p --build "$tag" --slice all > /scr/bobbyy/pfs/csv64/ss64_${tag}_p${p}.log 2>&1
    rc=$?
    rows=$(( $(wc -l < "$out" 2>/dev/null || echo 1) - 1 ))
    echo "  --- p$p $tag done  $(date +%H:%M:%S)  rc=$rc  ($rows rows)"
    [ "$rc" = "0" ] || { echo "REFUSE: p$p $tag exited $rc"; tail -4 /scr/bobbyy/pfs/csv64/ss64_${tag}_p${p}.log; exit 1; }
  done
done
echo "MKT_PFS64_ALL_PASSES_DONE $(date +%H:%M:%S)"
