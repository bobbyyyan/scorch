#!/usr/bin/env bash
# Drives the ADOPT_GRAIN measurement on redwood, which has no Slurm -- the screens there run as
# plain background processes. Never AFS or $HOME: the shim exports HOME and TORCH_EXTENSIONS_DIR
# under /scratch, which is what every other harness on this host does.
set -u
export ADG_STAGE=/scratch/bobbyy/scorch-beat-mkl
export ADG_ROOT=/scratch/bobbyy/adg
export ADG_ACTIVATE=/scratch/bobbyy/pfs-activate.sh
export ADG_CACHE=/scratch/bobbyy/mklsweep/csrcache/dlmc
export ADG_OUT=/scratch/bobbyy/adg/csv
export ADG_PERCORE=0.26
echo "host $(hostname)  nproc $(nproc)  $(date +%H:%M:%S)  percore $ADG_PERCORE"
bash /scratch/bobbyy/scorch-beat-mkl/harnesses/pfstream/adg_host_run.sh
