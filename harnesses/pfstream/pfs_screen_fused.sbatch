#!/bin/bash
#SBATCH -A mkt
#SBATCH -p mkt
#SBATCH -c 32
#SBATCH -t 10:00:00
#SBATCH -J pfs-screen-fused
#SBATCH -o /scr/bobbyy/pfs/screen_fused.log
# The PFSTREAM screen's THIRD cell: the fused feature-major Linear path.
#
# WHY CORRECTNESS RUNS INSIDE THIS JOB AND THE TIMING IS GATED ON IT. The drop-in cell ran
# pfs_bits.py and pfs_bitcmp.py as a separate earlier job, and that worked only because a human
# read the verdict in between. Here the two are one job with an explicit gate, so a timing run
# cannot land on an object whose output was never compared. Bit-identity is the claim a prefetch
# must satisfy -- it changes no arithmetic and no reduction order -- so anything short of
# byte-identical output across the four builds makes every timing number below unreadable.
#
# WHY IT IS THIRD AND DEPENDENCY-GATED. mkt1's memory bandwidth is not cgroup-partitioned, so a
# second timing job on the node perturbs the first rather than neighbouring it. This waits for the
# float64 cell, which waits for the float32 one.
#
# WHAT THIS CELL REACHES THAT NEITHER OTHER CELL CAN. spmm_csr_linear_fused_float has no
# exact-width branch and no half-vector branch (zero references to narrow_exact, exact_width,
# halfvec, HALFVEC or NARROWK inside spmm.h:6119-6450), so the register block runs at EVERY width
# 1..32 here. Widths 1-4 therefore reach (nvec 1, masked tail), which the drop-in path cannot reach
# at any width because narrow_exact owns 1-3 and the half-vector kernel owns 4. k=2 is an ARM on
# this path and a control on the drop-in -- the same width, opposite roles.
#
# THE CONTROL HERE IS k=64, where narrow_k is false and the row takes scorch_spmm_row_regtile, so
# PFSTREAM is unreachable. It is NOT a substitute for the per-width shpb-against-ship floor: its
# magnitude is measured on a different kernel, so it bounds instrument false positives and nothing
# about a given width.
set -u
cd /scr/bobbyy/pfs || exit 1
source /scr/bobbyy/venv/bin/activate || exit 1
CACHE=/scr/bobbyy/mklsweep/csrcache/ss
KEYS=/scr/bobbyy/pfs/ss_sample.txt
KS=2,4,8,12,16,20,32,64
# Correctness runs on a SUBSET, because bit-identity is a property of the code path and not of the
# corpus: 10 widths x 3 activations x 40 matrices is already 1200 real cells per build plus 210
# degenerate ones. The timing run then covers the full sample.
BITS_KS=1,2,3,4,8,12,16,20,32,33
BITS_LIMIT=40
mkdir -p /scr/bobbyy/pfs/csvfused
echo "host $(hostname)  pool $(python -c 'import torch;print(torch.get_num_threads())')  $(date +%H:%M:%S)"

# THE JOB STAGES ITS OWN INPUTS FROM DURABLE STORAGE AND VERIFIES THEM.
# /scr is node-local, so a login node cannot write it and copying from an interactive
# allocation would mean running a command on mkt1 beside a live timing job. Staging here
# removes that entirely: the files arrive from /matx when this job starts, and each digest
# is checked, so a stale or partial copy refuses instead of producing numbers.
STAGE=/matx/u/bobbyy/pfsstage
declare -A WANT=(
  ["csrresolve.py"]="1872f707045dc14e149fd53e845fee4b60e206b48132b144ee8217c227bfba78"
  ["fmgate.py"]="99e20fee682c205300e4410ea77352cb6a05f6516b1711191146c92296ac60b0"
  ["pfs_bitcmp.py"]="f4b09ad2488b3d1ad7ed71c9a5861edc051d3cc0e75407927dac6e3625a27b41"
  ["pfs_bits_fused.py"]="fde751c65001b78a6da3b986671dc8340ddc952c00d3a2534e3c6fdcfcbfa4cc"
)
for f in "${!WANT[@]}"; do
  cp "$STAGE/$f" /scr/bobbyy/pfs/$f || { echo "REFUSE: cannot stage $f"; exit 1; }
  got=$(shasum -a 256 /scr/bobbyy/pfs/$f | cut -d" " -f1)
  [ "$got" = "${WANT[$f]}" ] || { echo "REFUSE: $f sha256 $got, expected ${WANT[$f]}"; exit 1; }
  echo "  staged $f sha256 ${got:0:16}"
done


echo "== STAGE 1: bit-identity across the four builds on the fused symbol"
for tag in ship shpb strm nopf; do
  echo "  --- bits $tag start $(date +%H:%M:%S)"
  PYTHONPATH=/scr/bobbyy/pfs/$tag/src OMP_NUM_THREADS=32 \
    python /scr/bobbyy/pfs/pfs_bits_fused.py --cache "$CACHE" --keys "$KEYS" \
      --csv /scr/bobbyy/pfs/bitsf_${tag}.csv --build "$tag" --ks "$BITS_KS" --acts 0,1,2 \
      --limit "$BITS_LIMIT" > /scr/bobbyy/pfs/bitsf_${tag}.log 2>&1
  rc=$?
  echo "  --- bits $tag done  $(date +%H:%M:%S)  rc=$rc"
  [ "$rc" = "0" ] || { echo "REFUSE: bits $tag exited $rc"; tail -6 /scr/bobbyy/pfs/bitsf_${tag}.log; exit 1; }
done
echo "== STAGE 2: compare the digests. The timing below does not run unless this passes."
python /scr/bobbyy/pfs/pfs_bitcmp.py /scr/bobbyy/pfs "$BITS_LIMIT" bitsf
rc=$?
[ "$rc" = "0" ] || { echo "REFUSE: fused bit-identity failed (rc=$rc); a timing number on objects "\
"whose output differs is not a measurement of a prefetch"; exit 1; }

echo "== STAGE 3: the timing screen, four passes of a Latin square"
ORDERS=( "ship shpb strm nopf" "shpb strm nopf ship" "strm nopf ship shpb" "nopf ship shpb strm" )
for p in 1 2 3 4; do
  order=${ORDERS[$((p-1))]}
  echo "== pass p$p order: $order"
  for tag in $order; do
    out=/scr/bobbyy/pfs/csvfused/fm_${tag}_p${p}.csv
    echo "  --- p$p $tag start $(date +%H:%M:%S)"
    OMP_NUM_THREADS=32 PYTHONPATH=/scr/bobbyy/pfs/$tag/src \
      python /scr/bobbyy/pfs/fmgate.py --cache "$CACHE" --keys "$KEYS" --csv "$out" \
        --dtype float32 --ks "$KS" --act 1 --reps 9 --target-ms 15 \
        --pass-tag p$p --build "$tag" --slice all > /scr/bobbyy/pfs/csvfused/fm_${tag}_p${p}.log 2>&1
    rc=$?
    rows=$(( $(wc -l < "$out" 2>/dev/null || echo 1) - 1 ))
    echo "  --- p$p $tag done  $(date +%H:%M:%S)  rc=$rc  ($rows rows)"
    [ "$rc" = "0" ] || { echo "REFUSE: p$p $tag exited $rc"; tail -4 /scr/bobbyy/pfs/csvfused/fm_${tag}_p${p}.log; exit 1; }
  done
done
echo "MKT_PFSFM_ALL_PASSES_DONE $(date +%H:%M:%S)"
