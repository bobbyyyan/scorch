#!/bin/bash
#SBATCH -A mkt
#SBATCH -p mkt
#SBATCH -J nt-all392
#SBATCH -t 8:00:00
# -c 32 IS LOAD-BEARING and for the same reason as the ADOPT_GRAIN screen: without it Slurm gives
# the default allocation, torch.get_num_threads() reads 2, and the cgroup caps OMP_NUM_THREADS so
# the run completes normally while measuring a pool-2 machine. /scr/bobbyy/pfs holds a directory
# named csvdlmc_pool2_abandoned from exactly that. Here the pool is the x-axis of the thing being
# measured -- the worker-count curve -- so a pool-2 reading is a different experiment, not a
# noisier one.
#SBATCH -c 32
#SBATCH -o /scr/bobbyy/an/nt_all392.log
#
# PREREG_thread_sizing.md amendments 21(c) and 22: the thread-count oracle has only ever run on
# adg_keys.txt (72 keys), while the population this project's goal is stated over is the 328
# float32 DLMC keys behind MKL in all 16 readings at k=2. The two lists share 20 keys, so
# amendment 20's <=2-worker result is measured on a corpus that barely intersects the deficit.
#
# WHY ALL 392 AND NOT THE 328 (amendment 22(c)). The 328 alone yields a <=2 share with nothing to
# compare it against. The 64 keys that are NOT behind are the clean control -- same corpus, same
# run, same instrument, and OUTSIDE the population being certified. Sweeping all 392 and splitting
# in analysis puts the arm and its control in ONE process on ONE object; two jobs would separate
# them by a process boundary, which this project has repeatedly measured as larger than the
# effects it looks for.
#
# THE FOUR REFUTATION CONDITIONS, registered BEFORE this script was written. s328 and s64 are the
# <=2-worker shares among the live k=2 cells of the two halves.
#  1. pooled k=2 excess <= 1.00 -> thread sizing closes for the deficit on its own population
#  2. s328 >= 0.50 licenses a causal claim; 0.25 <= s328 < 0.50 is a share with NO causal claim;
#     s328 < 0.25 is a BOUND ON REACH and not a pass. In units of the explained quantity, because
#     the deficit's own prevalence is 83.7% and the earlier 5% threshold passed at a value (17.0%)
#     that refutes the claim.
#  3. <=2 excess not exceeding >=8's -> the element-2 direction does not transfer from the 72 keys
#  4. a bootstrap CI on s328 - s64 containing zero -> the mechanism does not track the deficit
#     whatever its per-cell excess; the share is then a corpus property, not a deficit property.
# No rule or gate is proposed under any outcome.
set -u
export HOME=/scr/bobbyy/fakehome
export TORCH_EXTENSIONS_DIR=/scr/bobbyy/fakehome/.cache/torch_extensions
mkdir -p "$TORCH_EXTENSIONS_DIR"
AN=/scr/bobbyy/an
KEYS=$AN/all392.txt
OUT=$AN/oracle_all392.csv

# REFUSE IF ANY PRODUCER IS LIVE. Counted by lines rather than `pgrep -c`, which returns 1 for
# "no match". The pattern list cannot match this script's own command line: Slurm invokes it by
# its spooled path, and none of these words appears in that path.
for pat in mrgate arm_screen adggate ntoracle; do
  n=$(pgrep -f "$pat" 2>/dev/null | grep -c . || true)
  [ "${n:-0}" -eq 0 ] || { echo "REFUSE: $n process(es) matching '$pat' live; timing next to them perturbs both"; exit 1; }
done
echo "guard: no live producer"

# THE KEY FILE IS THE EXPERIMENT'S POPULATION, so its size is asserted against the published
# board rather than trusted. an_f64_vs_mkl.py's per-width table reads {2: 328, ...} on this host's
# float32 DLMC screen; a file of any other length is a different population and the refutation
# conditions above would be scored against the wrong denominator.
[ -s "$KEYS" ] || { echo "REFUSE: $KEYS missing or empty"; exit 1; }
nk=$(grep -c . "$KEYS")
[ "$nk" -eq 392 ] || { echo "REFUSE: $KEYS holds $nk keys, not the 392 the float32 DLMC board covers"; exit 1; }
nb=$(grep -c . "$AN/deficit328.txt" 2>/dev/null || echo 0)
[ "$nb" -eq 328 ] || { echo "REFUSE: deficit328.txt holds $nb keys, not 328; the analysis split would be wrong"; exit 1; }
nshared=$(comm -12 <(sort -u "$KEYS") <(sort -u "$AN/deficit328.txt") | grep -c . || true)
[ "$nshared" -eq 328 ] || { echo "REFUSE: only $nshared of the 328 deficit keys are inside the 392 swept; the control split is not a partition"; exit 1; }
echo "keys $nk = 328 deficit + $((nk - 328)) control, and the 328 are a SUBSET of the 392 (asserted, not assumed)"

source /scr/bobbyy/venv/bin/activate
ls -d /scr/bobbyy/pfs/ship/src >/dev/null || { echo "REFUSE: no staged ship tree"; exit 1; }
echo "host $(hostname)  nproc $(nproc)  $(date +%H:%M:%S)"
PYTHONPATH=/scr/bobbyy/pfs/ship/src python3 -c "import torch, scorch; print('pool', torch.get_num_threads())"
PYTHONPATH=/scr/bobbyy/pfs/ship/src python3 "$AN/ntoracle.py" \
  --root /scr/bobbyy/mklsweep/csrcache/dlmc \
  --keys "$KEYS" \
  --csv "$OUT" \
  --ks 2,4,8,12,16,20,24,32 --reps 2 --target-ms 10
rc=$?
echo "ntoracle rc=$rc  $(date +%H:%M:%S)"
# A post-close line count on the artefact itself, not a loop counter: chain208's own run log
# asserted "wrote ... (267 rows)" for a file holding only a header.
echo "rows in $OUT: $(wc -l < "$OUT" 2>/dev/null || echo MISSING)"
echo "distinct so_sha: $(awk -F, 'NR>1{print $NF}' "$OUT" 2>/dev/null | sort -u | head -3 | tr '\n' ' ')"
echo "NT_ALL392_DONE rc=$rc"
