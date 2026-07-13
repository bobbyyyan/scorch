# Compiler IR Phase 0 benchmarks

`benchmark_compiler_ir.py` checks in the corpus and method, not benchmark results.
Write results outside the repository (the examples use `/tmp/scorch-phase0`) and
never commit JSON, CSV, generated extensions, or benchmark output.

## Python compiler latency

The latency corpus has four deliberately small operations:

| category | operation | formats |
|---|---|---|
| small dense | elementwise multiply, `ij,ij->ij` | `dd`, `dd` -> `dd` |
| reduction | matrix-vector reduction, `ij,j->i` | `dd`, `d` -> `d` |
| CSR intersection | elementwise multiply/intersection, `ij,ij->ij` | `ds`, `ds` -> `ds` |
| sparse union | elementwise `STensor.__add__` | `ds`, `ds` -> `ds` |

The harness starts at the public operation boundary, so validation and frontend CIN
construction are included. It intercepts `_load_kernel` after C++ and the build
arguments have been produced. Native C++ compilation, dynamic loading, and execution
are therefore excluded. This is the current pipeline's closest predecessor to the
planned validated-operation-to-`KernelBuildSpec` interval.

Run the production configuration from the activated `scorch` environment:

```bash
conda activate scorch
mkdir -p /tmp/scorch-phase0
python tools/benchmark_compiler_ir.py latency \
  --warmup 5 --samples 30 \
  --output /tmp/scorch-phase0/latency-m5.json
```

Each case records raw samples, p50, p95, the emitted-source digest, flags, platform,
commit, and dirty status. Repeated samples must emit the same source digest. Full
debug-verifier and stage-dump measurements are separate and do not replace this
production baseline.

Repeat the same command after a migration step, then enforce the design's 1.10x
per-category p50 and p95 budget:

```bash
python tools/benchmark_compiler_ir.py compare-latency \
  /tmp/scorch-phase0/latency-legacy-m5.json \
  /tmp/scorch-phase0/latency-candidate-m5.json
```

## Generated-kernel baseline and A/A control

The full generated-kernel corpus matches the local `codegen-parity` SpMM grid:
`M={512,4096,20000}`, `N={1,3,4,8,16,64,256}`, and
`density={0.02,0.1}`. It always takes the generic generated path
(`matmul(..., use_cache=False)`); handwritten prebuilt kernels are not a cutover
reference.

For each cell, lanes A and B call the same loaded generated module. Lane order
alternates each round. The tool retains every native-evaluation sample and reports a
symmetric per-cell band from the observed B/A and A/B ratios. It also records the
round-wise geomean band across the entire grid. These observed bands, not a fixed
percentage, calibrate a later generated-candidate/generated-legacy comparison.

Smoke-test the harness before a long run:

```bash
python tools/benchmark_compiler_ir.py kernel-aa \
  --quick --warmup 1 --rounds 2 --calls 1 --threads 2
```

Record the M5 baseline from the harness-only revision before compiler behavior
changes. Do not use a dirty compiler tree as the archived legacy baseline:

```bash
python tools/benchmark_compiler_ir.py kernel-aa \
  --warmup 3 --rounds 5 --calls 3 \
  --output /tmp/scorch-phase0/kernel-legacy-m5.json
```

On redwood, use the same commit, settings, and activated environment. The repository
is `/scratch/bobbyy/scorch` and the environment is
`/scratch/bobbyy/miniconda3/envs/scorch`:

```bash
ssh redwood
cd /scratch/bobbyy/scorch
source /scratch/bobbyy/miniconda3/etc/profile.d/conda.sh
conda activate scorch
mkdir -p /tmp/scorch-phase0
python tools/benchmark_compiler_ir.py kernel-aa \
  --warmup 3 --rounds 5 --calls 3 \
  --output /tmp/scorch-phase0/kernel-legacy-redwood.json
```

After a change, repeat the full run on the same machine and compare the two local
files:

```bash
python tools/benchmark_compiler_ir.py compare \
  /tmp/scorch-phase0/kernel-legacy-m5.json \
  /tmp/scorch-phase0/kernel-candidate-m5.json
```

The comparison checks each candidate/legacy median and the grid geomean against the
bands observed by the candidate run's accompanying same-binary control. It fails on
any out-of-band cell or machine geomean. If every cell's emitted source and build
inputs are byte identical, it waives the runtime comparison; structural activation
tests are still required.

Run one complete grid per declared measurement. Do not rerun individual cells until
they pass. A rerun is valid only for a diagnosed measurement defect and replaces the
entire affected machine run. Keep the existing local `codegen-parity/harness.py` as
the handwritten-prebuilt oracle; it is supplementary and must not replace the
legacy-generated comparison or its same-binary control.
