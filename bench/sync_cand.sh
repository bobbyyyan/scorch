#!/usr/bin/env bash
# Push the local candidate worktree to redwood and build it in place.
#
# In-place build + PYTHONPATH, never `pip install`: the shared conda env must keep
# pointing at its own tree so a concurrent session is unaffected.
#
# Exclude-based rather than include-based: an include list silently omitted tools/,
# which one test imports, and the whole suite then failed at COLLECTION on the
# candidate while passing on the base -- a missing-file artifact that looked like a
# real regression. Exclude only what must not travel.
set -euo pipefail
LOCAL="${1:-/Users/bobby/scorch-beat-mkl}"
REMOTE="${2:-/scratch/bobbyy/spmm-beat-mkl/cand}"

ssh redwood "mkdir -p $REMOTE"
rsync -az --delete --copy-unsafe-links \
  --exclude='.git/' --exclude='build/' --exclude='__pycache__/' \
  --exclude='*.so' --exclude='*.o' --exclude='data/' --exclude='weights/' \
  --exclude='*.csv' --exclude='.pytest_cache/' \
  "$LOCAL/" "redwood:$REMOTE/"

ssh redwood "export PATH=/scratch/bobbyy/miniconda3/envs/scorch/bin:\$PATH
cd $REMOTE
python -c 'import setuptools; setuptools.setup()' build_ext --inplace 2>&1 | tail -3
ls -la src/scorch_ops*.so"
