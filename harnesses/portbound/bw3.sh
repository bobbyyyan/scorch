#!/bin/bash
export HOME=/scratch/bobbyy/fakehome
source /scratch/bobbyy/miniconda3/etc/profile.d/conda.sh
conda activate scorch
cd /scratch/bobbyy/synchk || exit 1
echo "host            $(hostname)"
echo "cpu             $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^ *//')"
echo "g++             $(g++ --version | head -1)"
echo "flags           -c -O3 -std=c++17 -march=native -funroll-loops -DSCORCH_NARROWK_DEEP_UNROLL=4"
echo "spmm.h          $(sha256sum spmm.h | cut -c1-16)"
echo "scorch_policy.h $(sha256sum scorch_policy.h | cut -c1-16)"
echo "deep.o          $(sha256sum deep.o | cut -c1-16)  $(stat -c %s deep.o) bytes"
echo "deep.asm        $(sha256sum deep.asm | cut -c1-16)  $(wc -l < deep.asm) lines"
