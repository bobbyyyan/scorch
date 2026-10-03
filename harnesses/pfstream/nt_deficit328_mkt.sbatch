#!/bin/bash
#SBATCH -A mkt
#SBATCH -p mkt
#SBATCH -J nt-deficit328
#SBATCH -t 8:00:00
# -c 32 IS LOAD-BEARING and for the same reason as the ADOPT_GRAIN screen: without it Slurm gives
# the default allocation, torch.get_num_threads() reads 2, and the cgroup caps OMP_NUM_THREADS so
# the run completes normally while measuring a pool-2 machine. /scr/bobbyy/pfs holds a directory
# named csvdlmc_pool2_abandoned from exactly that. Here the pool is the x-axis of the thing being
# measured -- the worker-count curve -- so a pool-2 reading is a different experiment, not a
# noisier one.
#SBATCH -c 32
#SBATCH -o /scr/bobbyy/an/nt_deficit328.log
#
# PREREG_thread_sizing.md amendment 21(c): the thread-count oracle has only ever run on
# adg_keys.txt (72 keys), while the population this project's goal is stated over is the 328
# float32 DLMC cells that are behind MKL in all 16 readings at k=2. The two lists share ONE key,
# so amendment 20's <=2-worker result is measured on a corpus that barely intersects the deficit.
# This runs the registered statistic on the deficit itself.
#
# THE THREE REFUTATION CONDITIONS, registered in amendment 21(c) BEFORE this script was written:
#  1. pooled k=2 excess <= 1.00  -> thread sizing closes for the deficit on its own population
#  2. the <=2-worker group < 5% of the set -> cannot account for the deficit at any per-cell size
#  3. <=2 excess not exceeding >=8's -> the element-2 direction does not transfer from the 72 keys
# No rule or gate is proposed under any outcome.
set -u
export HOME=/scr/bobbyy/fakehome
export TORCH_EXTENSIONS_DIR=/scr/bobbyy/fakehome/.cache/torch_extensions
mkdir -p "$TORCH_EXTENSIONS_DIR"
AN=/scr/bobbyy/an
KEYS=$AN/deficit328.txt
OUT=$AN/oracle_deficit328.csv

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
[ "$nk" -eq 328 ] || { echo "REFUSE: $KEYS holds $nk keys, not the 328 the board publishes at k=2"; exit 1; }
echo "keys $nk  (asserted against the board's per-width count at k=2)"

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
echo "NT_DEFICIT328_DONE rc=$rc"
