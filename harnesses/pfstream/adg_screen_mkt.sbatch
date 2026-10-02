#!/bin/bash
#SBATCH -A mkt
#SBATCH -p mkt
#SBATCH -J adg-dlmc-f32
#SBATCH -t 8:00:00
# -c 32 IS LOAD-BEARING, for the reason already recorded next to the float32 prefetch screen:
# without a cpus-per-task request Slurm gives the default allocation and torch.get_num_threads()
# reads 2 on this node, so OMP_NUM_THREADS cannot help because the cgroup caps it. A pool-2 run
# completes normally and produces timings comparable to nothing -- /scr/bobbyy/pfs holds a
# directory literally named csvdlmc_pool2_abandoned from exactly that mistake. And here the pool
# is not incidental: the lever under test is the ADOPTION of the caller's pool, the binding
# predicate is nnz*k < grain * min(pool, rows/16), so the pool scales the region the knob reaches.
# A pool-2 reading would be a different knob, not a noisier one.
#SBATCH -c 32
#SBATCH -o /scr/bobbyy/pfs/adg_dlmc.log
#
# Prices SCORCH_SPMM_ADOPT_GRAIN on MKT, and runs the thread-count oracle sweep.
#
# WHAT IS REGISTERED FOR THIS HOST, so the reading can fail. MKT's measured per-worker team cost
# is 0.06 microseconds, 79x cheaper than the M5's 4.75. PREREG_thread_sizing.md amendment 1
# registers the consequence: n* = sqrt(T1/c) reaches the whole 32-wide pool once T1 >= 61us,
# which is about five cycles a nonzero, so the rule should be close to INERT here and the
# ordering of how much it bites is registered as M5 > redwood > MKT. If this host shows the
# M5's effect sizes, the model is wrong.
#
# AND THE COMPETING MECHANISM, also registered in advance. If MKT's k=2 time is FLAT in worker
# count rather than rising, the narrow-k deficit here is bandwidth saturation -- at k=2 each
# nonzero fetches a 64-byte line to use 8, and 32 workers demand 32x that traffic where 6 do
# not -- and a rule expressed in worker count is the wrong shape for it. The oracle sweep
# measures that shape directly, so this script answers both questions or neither.
set -u
export HOME=/scr/bobbyy/fakehome
export TORCH_EXTENSIONS_DIR=/scr/bobbyy/fakehome/.cache/torch_extensions
mkdir -p "$TORCH_EXTENSIONS_DIR"
H=/scr/bobbyy/scorch-beat-mkl/harnesses/pfstream
export ADG_STAGE=/scr/bobbyy/pfs/ship
export ADG_ROOT=/scr/bobbyy/adg
export ADG_ACTIVATE=/scr/bobbyy/adg-activate.sh
export ADG_CACHE=/scr/bobbyy/mklsweep/csrcache/dlmc
export ADG_OUT=/scr/bobbyy/adg/csv
export ADG_PERCORE=0.06
echo "host $(hostname)  nproc $(nproc)  $(date +%H:%M:%S)"
echo "stage $ADG_STAGE  root $ADG_ROOT  cache $ADG_CACHE  percore $ADG_PERCORE"
bash "$H/adg_host_run.sh"
