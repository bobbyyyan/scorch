// Port-boundedness probe for the k=4 float32 SpMM row kernel.
// Pre-registered in harnesses/prereg_portbound.md at digest c5ec8dec37925c79. Seven arms.
//
// V1, V2, V3 and V5 COMPUTE THE WRONG ANSWER BY CONSTRUCTION. They are throughput probes, not
// candidate kernels, and are named here so no later reader mistakes one for a proposed
// implementation.
//
// The shipped loop this replicates (spmm.h:2309-2321, half-width K=4 so L=4, NVEC=1,
// FULL_LAST=true) is 2-way unrolled over nonzeros and costs THREE load-port operations per
// nonzero: the column-index load, the A-value broadcast, and the B-row load folded into the FMA.
#include <immintrin.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <vector>

#define K 4
#define NSINK 4            // rotating sinks, so a sink's own latency chain cannot bind
#define NOINLINE __attribute__((noinline))
#define SCORCH_RESTRICT __restrict__

// Sinks live at file scope and are printed, so nothing in any variant is dead.
alignas(32) static float g_sink[NSINK][K];

struct Args {
  const int* idx; const float* aval; const float* bval; const float* dval; int n;
};

// ---- V0: the shipped inner loop. 3 load-port ops per nonzero.
NOINLINE float v0(const Args& A) {
  __m128 acc0 = _mm_setzero_ps(), acc1 = _mm_setzero_ps();
  int p = 0;
  // BODY WIDTH IS PINNED ON EVERY ARM, so it is a CONTROLLED CONSTANT rather than a compiler
  // choice. It had become a dependent variable: removing the index load lowered register
  // pressure and gcc unrolled V8 FOURfold to a 16-nonzero body, while ADDING a sink raised
  // pressure and it unrolled V4/V4c only twofold to 4 -- opposite directions, both caused by
  // the change under test. Then V8-vs-V6 and V4c-vs-V4b each measured their intended delta
  // PLUS a halved or doubled per-nonzero loop control, and the gate could not see it: all three
  // of its metrics are per-nonzero ratios, so doubling a body doubles numerator and denominator
  // and leaves them invariant. Every arm now gets 8 nonzeros per body. Some arms are slower in
  // absolute terms than their preferred width would give, which costs nothing: the probe only
  // ever reads differences, and matched bodies are what make a difference attributable.
  #pragma GCC unroll 4
  for (; p + 2 <= A.n; p += 2) {
    const float* B0 = A.bval + (size_t)A.idx[p] * K;
    const float* B1 = A.bval + (size_t)A.idx[p + 1] * K;
    const __m128 a0 = _mm_set1_ps(A.aval[p]);
    const __m128 a1 = _mm_set1_ps(A.aval[p + 1]);
    acc0 = _mm_fmadd_ps(a0, _mm_loadu_ps(B0), acc0);
    acc1 = _mm_fmadd_ps(a1, _mm_loadu_ps(B1), acc1);
  }
  const __m128 r = _mm_add_ps(acc0, acc1);
  alignas(16) float o[K]; _mm_storeu_ps(o, r);
  return o[0] + o[1] + o[2] + o[3];
}

// ---- V0b: byte-identical source under a second symbol. The same-binary A/A floor.
NOINLINE float v0b(const Args& A) {
  __m128 acc0 = _mm_setzero_ps(), acc1 = _mm_setzero_ps();
  int p = 0;
  #pragma GCC unroll 4
  for (; p + 2 <= A.n; p += 2) {
    const float* B0 = A.bval + (size_t)A.idx[p] * K;
    const float* B1 = A.bval + (size_t)A.idx[p + 1] * K;
    const __m128 a0 = _mm_set1_ps(A.aval[p]);
    const __m128 a1 = _mm_set1_ps(A.aval[p + 1]);
    acc0 = _mm_fmadd_ps(a0, _mm_loadu_ps(B0), acc0);
    acc1 = _mm_fmadd_ps(a1, _mm_loadu_ps(B1), acc1);
  }
  const __m128 r = _mm_add_ps(acc0, acc1);
  alignas(16) float o[K]; _mm_storeu_ps(o, r);
  return o[0] + o[1] + o[2] + o[3];
}

// ---- V1: column index replaced by a sequential counter. Removes the INDEX load. 2 ops/nnz.
NOINLINE float v1(const Args& A) {
  __m128 acc0 = _mm_setzero_ps(), acc1 = _mm_setzero_ps();
  int p = 0;
  #pragma GCC unroll 4
  for (; p + 2 <= A.n; p += 2) {
    const float* B0 = A.bval + (size_t)p * K;
    const float* B1 = A.bval + (size_t)(p + 1) * K;
    const __m128 a0 = _mm_set1_ps(A.aval[p]);
    const __m128 a1 = _mm_set1_ps(A.aval[p + 1]);
    acc0 = _mm_fmadd_ps(a0, _mm_loadu_ps(B0), acc0);
    acc1 = _mm_fmadd_ps(a1, _mm_loadu_ps(B1), acc1);
  }
  const __m128 r = _mm_add_ps(acc0, acc1);
  alignas(16) float o[K]; _mm_storeu_ps(o, r);
  return o[0] + o[1] + o[2] + o[3];
}

// ---- V2: A value hoisted out of the loop. Removes the BROADCAST. 2 ops/nnz.
NOINLINE float v2(const Args& A) {
  __m128 acc0 = _mm_setzero_ps(), acc1 = _mm_setzero_ps();
  const __m128 a0 = _mm_set1_ps(A.aval[0]);
  const __m128 a1 = _mm_set1_ps(A.aval[1]);
  int p = 0;
  #pragma GCC unroll 4
  for (; p + 2 <= A.n; p += 2) {
    const float* B0 = A.bval + (size_t)A.idx[p] * K;
    const float* B1 = A.bval + (size_t)A.idx[p + 1] * K;
    acc0 = _mm_fmadd_ps(a0, _mm_loadu_ps(B0), acc0);
    acc1 = _mm_fmadd_ps(a1, _mm_loadu_ps(B1), acc1);
  }
  const __m128 r = _mm_add_ps(acc0, acc1);
  alignas(16) float o[K]; _mm_storeu_ps(o, r);
  return o[0] + o[1] + o[2] + o[3];
}

// ---- V3: B operand register-resident. Removes the B LOAD. 2 ops/nnz.
// The index load is KEPT and consumed into the address computation of nothing, so it must still
// be issued: it is summed into a sink to stay live.
NOINLINE float v3(const Args& A) {
  __m128 acc0 = _mm_setzero_ps(), acc1 = _mm_setzero_ps();
  const __m128 b0 = _mm_loadu_ps(A.bval);
  const __m128 b1 = _mm_loadu_ps(A.bval + K);
  int32_t keep = 0;
  int p = 0;
  #pragma GCC unroll 4
  for (; p + 2 <= A.n; p += 2) {
    keep += A.idx[p] + A.idx[p + 1];
    const __m128 a0 = _mm_set1_ps(A.aval[p]);
    const __m128 a1 = _mm_set1_ps(A.aval[p + 1]);
    acc0 = _mm_fmadd_ps(a0, b0, acc0);
    acc1 = _mm_fmadd_ps(a1, b1, acc1);
  }
  const __m128 r = _mm_add_ps(acc0, acc1);
  alignas(16) float o[K]; _mm_storeu_ps(o, r);
  return o[0] + o[1] + o[2] + o[3] + (float)keep * 1e-30f;
}

// ---- AMENDMENT 12, forced by the emitted code and made before any timing was read.
// A sink that is OVERWRITTEN each iteration is still dead: only the last iteration's value is
// stored, so -O3 deleted V4's loads (delta +0.12 mem/nnz) and ALL of V5's vpors (0.00/nnz).
// Consuming a value means ACCUMULATING it, and an accumulate costs an ALU op -- which would
// confound V4's intended load-only delta. So V4 carries the accumulate AND a new arm V4b carries
// the accumulate WITHOUT the load, and V4 - V4b isolates exactly one load per nonzero.
//
// `_mm_or_ps` is the accumulate in every case: non-idiom, distinct live sources (a sink and an
// operand), and on ports 0/1/5 rather than the FMA's 0/1, so it can retire on port 5 and does not
// contend with FMA throughput. Four independent sink chains, so the sink's own 1-cycle latency
// cannot become the binding constraint. All four are stored after the loop.

// ---- V4: V0 plus one EXTRA LOAD per nonzero, accumulated. POSITIVE CONTROL. 4 load ops/nnz.
NOINLINE float v4(const Args& A) {
  __m128 acc0 = _mm_setzero_ps(), acc1 = _mm_setzero_ps();
  __m128 s0 = _mm_setzero_ps(), s1 = _mm_setzero_ps(),
         s2 = _mm_setzero_ps(), s3 = _mm_setzero_ps();
  int p = 0;
  #pragma GCC unroll 2
  for (; p + 4 <= A.n; p += 4) {
    const float* B0 = A.bval + (size_t)A.idx[p] * K;
    const float* B1 = A.bval + (size_t)A.idx[p + 1] * K;
    const float* B2 = A.bval + (size_t)A.idx[p + 2] * K;
    const float* B3 = A.bval + (size_t)A.idx[p + 3] * K;
    const __m128 a0 = _mm_set1_ps(A.aval[p]);
    const __m128 a1 = _mm_set1_ps(A.aval[p + 1]);
    const __m128 a2 = _mm_set1_ps(A.aval[p + 2]);
    const __m128 a3 = _mm_set1_ps(A.aval[p + 3]);
    s0 = _mm_or_ps(s0, _mm_loadu_ps(A.dval + (size_t)p * K));
    s1 = _mm_or_ps(s1, _mm_loadu_ps(A.dval + (size_t)(p + 1) * K));
    s2 = _mm_or_ps(s2, _mm_loadu_ps(A.dval + (size_t)(p + 2) * K));
    s3 = _mm_or_ps(s3, _mm_loadu_ps(A.dval + (size_t)(p + 3) * K));
    acc0 = _mm_fmadd_ps(a0, _mm_loadu_ps(B0), acc0);
    acc1 = _mm_fmadd_ps(a1, _mm_loadu_ps(B1), acc1);
    acc0 = _mm_fmadd_ps(a2, _mm_loadu_ps(B2), acc0);
    acc1 = _mm_fmadd_ps(a3, _mm_loadu_ps(B3), acc1);
  }
  _mm_storeu_ps(g_sink[0], s0); _mm_storeu_ps(g_sink[1], s1);
  _mm_storeu_ps(g_sink[2], s2); _mm_storeu_ps(g_sink[3], s3);
  const __m128 r = _mm_add_ps(acc0, acc1);
  alignas(16) float o[K]; _mm_storeu_ps(o, r);
  return o[0] + o[1] + o[2] + o[3];
}

// ---- V4b: V0 plus the ACCUMULATE ONLY, no extra load. 3 load ops/nnz, issue +1.
// V4 - V4b is one load per nonzero with the ALU op held fixed. NEW IN AMENDMENT 12.
NOINLINE float v4b(const Args& A) {
  __m128 acc0 = _mm_setzero_ps(), acc1 = _mm_setzero_ps();
  __m128 s0 = _mm_setzero_ps(), s1 = _mm_setzero_ps(),
         s2 = _mm_setzero_ps(), s3 = _mm_setzero_ps();
  int p = 0;
  #pragma GCC unroll 2
  for (; p + 4 <= A.n; p += 4) {
    const float* B0 = A.bval + (size_t)A.idx[p] * K;
    const float* B1 = A.bval + (size_t)A.idx[p + 1] * K;
    const float* B2 = A.bval + (size_t)A.idx[p + 2] * K;
    const float* B3 = A.bval + (size_t)A.idx[p + 3] * K;
    const __m128 a0 = _mm_set1_ps(A.aval[p]);
    const __m128 a1 = _mm_set1_ps(A.aval[p + 1]);
    const __m128 a2 = _mm_set1_ps(A.aval[p + 2]);
    const __m128 a3 = _mm_set1_ps(A.aval[p + 3]);
    s0 = _mm_or_ps(s0, a0);
    s1 = _mm_or_ps(s1, a1);
    s2 = _mm_or_ps(s2, a2);
    s3 = _mm_or_ps(s3, a3);
    acc0 = _mm_fmadd_ps(a0, _mm_loadu_ps(B0), acc0);
    acc1 = _mm_fmadd_ps(a1, _mm_loadu_ps(B1), acc1);
    acc0 = _mm_fmadd_ps(a2, _mm_loadu_ps(B2), acc0);
    acc1 = _mm_fmadd_ps(a3, _mm_loadu_ps(B3), acc1);
  }
  _mm_storeu_ps(g_sink[0], s0); _mm_storeu_ps(g_sink[1], s1);
  _mm_storeu_ps(g_sink[2], s2); _mm_storeu_ps(g_sink[3], s3);
  const __m128 r = _mm_add_ps(acc0, acc1);
  alignas(16) float o[K]; _mm_storeu_ps(o, r);
  return o[0] + o[1] + o[2] + o[3];
}

// ---- V4c: V0 plus one extra load from the SAME CACHE LINE, accumulated.
// THE CLEAN POSITIVE CONTROL, added after V4 read +12% on both core types. V4's extra load comes
// from a separate 64 KB array, so it adds a cache stream as well as a load-port operation and
// cannot isolate port pressure.
//
// The first attempt reloaded B0 itself and the compiler CSE'd it away -- mem/nnz read 3.12 instead
// of 4.00, the third time in this probe that the optimiser deleted the thing being measured. So the
// extra load is from `B0 + K`: a DIFFERENT address, so no common-subexpression elimination, and
// with K=4 floats per B row it is 16 bytes along, inside the same 64-byte line three times in four.
// That adds exactly one load-port operation per nonzero and essentially no new footprint, so
// V4c - V4b is port pressure alone with the ALU op held fixed.
NOINLINE float v4c(const Args& A) {
  __m128 acc0 = _mm_setzero_ps(), acc1 = _mm_setzero_ps();
  __m128 s0 = _mm_setzero_ps(), s1 = _mm_setzero_ps(),
         s2 = _mm_setzero_ps(), s3 = _mm_setzero_ps();
  int p = 0;
  #pragma GCC unroll 2
  for (; p + 4 <= A.n; p += 4) {
    const float* B0 = A.bval + (size_t)A.idx[p] * K;
    const float* B1 = A.bval + (size_t)A.idx[p + 1] * K;
    const float* B2 = A.bval + (size_t)A.idx[p + 2] * K;
    const float* B3 = A.bval + (size_t)A.idx[p + 3] * K;
    const __m128 a0 = _mm_set1_ps(A.aval[p]);
    const __m128 a1 = _mm_set1_ps(A.aval[p + 1]);
    const __m128 a2 = _mm_set1_ps(A.aval[p + 2]);
    const __m128 a3 = _mm_set1_ps(A.aval[p + 3]);
    s0 = _mm_or_ps(s0, _mm_loadu_ps(B0 + K));
    s1 = _mm_or_ps(s1, _mm_loadu_ps(B1 + K));
    s2 = _mm_or_ps(s2, _mm_loadu_ps(B2 + K));
    s3 = _mm_or_ps(s3, _mm_loadu_ps(B3 + K));
    acc0 = _mm_fmadd_ps(a0, _mm_loadu_ps(B0), acc0);
    acc1 = _mm_fmadd_ps(a1, _mm_loadu_ps(B1), acc1);
    acc0 = _mm_fmadd_ps(a2, _mm_loadu_ps(B2), acc0);
    acc1 = _mm_fmadd_ps(a3, _mm_loadu_ps(B3), acc1);
  }
  _mm_storeu_ps(g_sink[0], s0); _mm_storeu_ps(g_sink[1], s1);
  _mm_storeu_ps(g_sink[2], s2); _mm_storeu_ps(g_sink[3], s3);
  const __m128 r = _mm_add_ps(acc0, acc1);
  alignas(16) float o[K]; _mm_storeu_ps(o, r);
  return o[0] + o[1] + o[2] + o[3];
}

// ---- V5: V1 plus the same ACCUMULATE. THE DISCRIMINATOR. 2 load ops/nnz, issue +1.
// V5 is exactly V4b with the index load removed. If V5 lands at V0 the binding constraint was the
// load port; if V5 lands at V1 it was issue width, and that closes the line-reuse lever.
NOINLINE float v5(const Args& A) {
  __m128 acc0 = _mm_setzero_ps(), acc1 = _mm_setzero_ps();
  __m128 s0 = _mm_setzero_ps(), s1 = _mm_setzero_ps(),
         s2 = _mm_setzero_ps(), s3 = _mm_setzero_ps();
  int p = 0;
  #pragma GCC unroll 2
  for (; p + 4 <= A.n; p += 4) {
    const float* B0 = A.bval + (size_t)p * K;
    const float* B1 = A.bval + (size_t)(p + 1) * K;
    const float* B2 = A.bval + (size_t)(p + 2) * K;
    const float* B3 = A.bval + (size_t)(p + 3) * K;
    const __m128 a0 = _mm_set1_ps(A.aval[p]);
    const __m128 a1 = _mm_set1_ps(A.aval[p + 1]);
    const __m128 a2 = _mm_set1_ps(A.aval[p + 2]);
    const __m128 a3 = _mm_set1_ps(A.aval[p + 3]);
    s0 = _mm_or_ps(s0, a0);
    s1 = _mm_or_ps(s1, a1);
    s2 = _mm_or_ps(s2, a2);
    s3 = _mm_or_ps(s3, a3);
    acc0 = _mm_fmadd_ps(a0, _mm_loadu_ps(B0), acc0);
    acc1 = _mm_fmadd_ps(a1, _mm_loadu_ps(B1), acc1);
    acc0 = _mm_fmadd_ps(a2, _mm_loadu_ps(B2), acc0);
    acc1 = _mm_fmadd_ps(a3, _mm_loadu_ps(B3), acc1);
  }
  _mm_storeu_ps(g_sink[0], s0); _mm_storeu_ps(g_sink[1], s1);
  _mm_storeu_ps(g_sink[2], s2); _mm_storeu_ps(g_sink[3], s3);
  const __m128 r = _mm_add_ps(acc0, acc1);
  alignas(16) float o[K]; _mm_storeu_ps(o, r);
  return o[0] + o[1] + o[2] + o[3];
}

// ---- V6 and V7 ARE CANDIDATE KERNELS, not throughput probes. Unlike V1/V2/V3/V5 they compute
// the CORRECT answer: more independent accumulator chains is a reassociation of the same sum.
//
// They exist because the registered probe's result identified the bound. V0 measures 1.994
// cycles/nnz on a P-core at 5.700 GHz and 2.991 on an E-core at 4.400 GHz, against a prediction of
// 4/2 = 2.00 and 6/2 = 3.00 from TWO accumulator chains, one FMA entering each per two nonzeros,
// at Golden Cove's 4-cycle and Gracemont's 6-cycle FMA latency. Both match to 0.3%. So the kernel
// is latency-bound on its accumulator chains and the remedy is more of them.
//
// Predicted, before measuring: 4 chains gives 4/4 = 1.00 cycles/nnz on P and 6/4 = 1.50 on E, both
// 2.0x. 8 chains gives 0.50 and 0.75 -- but the P-core cannot go below 3 load ops / 3 ports =
// 1.00 cycles/nnz, so V7 should show NO further gain on P while still gaining on E, whose load
// width is narrower but whose latency is longer. If V7 beats V6 on P, the load-port model is wrong.

// ---- V6: four accumulator chains, 4-way unrolled over nonzeros.
NOINLINE float v6(const Args& A) {
  const int* SCORCH_RESTRICT SCORCH_IDX = A.idx;
  const float* SCORCH_RESTRICT SCORCH_AV = A.aval;
  const float* SCORCH_RESTRICT SCORCH_BV = A.bval;
  const int SCORCH_N = A.n;
  __m128 acc0 = _mm_setzero_ps(), acc1 = _mm_setzero_ps(),
         acc2 = _mm_setzero_ps(), acc3 = _mm_setzero_ps();
  int p = 0;
  #pragma GCC unroll 2
  for (; p + 4 <= SCORCH_N; p += 4) {
    const float* B0 = SCORCH_BV + (size_t)SCORCH_IDX[p] * K;
    const float* B1 = SCORCH_BV + (size_t)SCORCH_IDX[p + 1] * K;
    const float* B2 = SCORCH_BV + (size_t)SCORCH_IDX[p + 2] * K;
    const float* B3 = SCORCH_BV + (size_t)SCORCH_IDX[p + 3] * K;
    acc0 = _mm_fmadd_ps(_mm_set1_ps(SCORCH_AV[p]),     _mm_loadu_ps(B0), acc0);
    acc1 = _mm_fmadd_ps(_mm_set1_ps(SCORCH_AV[p + 1]), _mm_loadu_ps(B1), acc1);
    acc2 = _mm_fmadd_ps(_mm_set1_ps(SCORCH_AV[p + 2]), _mm_loadu_ps(B2), acc2);
    acc3 = _mm_fmadd_ps(_mm_set1_ps(SCORCH_AV[p + 3]), _mm_loadu_ps(B3), acc3);
  }
  for (; p < SCORCH_N; ++p)
    acc0 = _mm_fmadd_ps(_mm_set1_ps(SCORCH_AV[p]),
                        _mm_loadu_ps(SCORCH_BV + (size_t)SCORCH_IDX[p] * K), acc0);
  const __m128 r = _mm_add_ps(_mm_add_ps(acc0, acc1), _mm_add_ps(acc2, acc3));
  alignas(16) float o[K]; _mm_storeu_ps(o, r);
  return o[0] + o[1] + o[2] + o[3];
}

// ---- V7: eight accumulator chains, 8-way unrolled over nonzeros.
NOINLINE float v7(const Args& A) {
  const int* SCORCH_RESTRICT SCORCH_IDX = A.idx;
  const float* SCORCH_RESTRICT SCORCH_AV = A.aval;
  const float* SCORCH_RESTRICT SCORCH_BV = A.bval;
  const int SCORCH_N = A.n;
  __m128 a0 = _mm_setzero_ps(), a1 = _mm_setzero_ps(), a2 = _mm_setzero_ps(), a3 = _mm_setzero_ps();
  __m128 a4 = _mm_setzero_ps(), a5 = _mm_setzero_ps(), a6 = _mm_setzero_ps(), a7 = _mm_setzero_ps();
  int p = 0;
  #pragma GCC unroll 1
  for (; p + 8 <= SCORCH_N; p += 8) {
    a0 = _mm_fmadd_ps(_mm_set1_ps(SCORCH_AV[p]),     _mm_loadu_ps(SCORCH_BV + (size_t)SCORCH_IDX[p] * K), a0);
    a1 = _mm_fmadd_ps(_mm_set1_ps(SCORCH_AV[p + 1]), _mm_loadu_ps(SCORCH_BV + (size_t)SCORCH_IDX[p + 1] * K), a1);
    a2 = _mm_fmadd_ps(_mm_set1_ps(SCORCH_AV[p + 2]), _mm_loadu_ps(SCORCH_BV + (size_t)SCORCH_IDX[p + 2] * K), a2);
    a3 = _mm_fmadd_ps(_mm_set1_ps(SCORCH_AV[p + 3]), _mm_loadu_ps(SCORCH_BV + (size_t)SCORCH_IDX[p + 3] * K), a3);
    a4 = _mm_fmadd_ps(_mm_set1_ps(SCORCH_AV[p + 4]), _mm_loadu_ps(SCORCH_BV + (size_t)SCORCH_IDX[p + 4] * K), a4);
    a5 = _mm_fmadd_ps(_mm_set1_ps(SCORCH_AV[p + 5]), _mm_loadu_ps(SCORCH_BV + (size_t)SCORCH_IDX[p + 5] * K), a5);
    a6 = _mm_fmadd_ps(_mm_set1_ps(SCORCH_AV[p + 6]), _mm_loadu_ps(SCORCH_BV + (size_t)SCORCH_IDX[p + 6] * K), a6);
    a7 = _mm_fmadd_ps(_mm_set1_ps(SCORCH_AV[p + 7]), _mm_loadu_ps(SCORCH_BV + (size_t)SCORCH_IDX[p + 7] * K), a7);
  }
  for (; p < SCORCH_N; ++p)
    a0 = _mm_fmadd_ps(_mm_set1_ps(SCORCH_AV[p]),
                      _mm_loadu_ps(SCORCH_BV + (size_t)SCORCH_IDX[p] * K), a0);
  const __m128 r = _mm_add_ps(_mm_add_ps(_mm_add_ps(a0, a1), _mm_add_ps(a2, a3)),
                              _mm_add_ps(_mm_add_ps(a4, a5), _mm_add_ps(a6, a7)));
  alignas(16) float o[K]; _mm_storeu_ps(o, r);
  return o[0] + o[1] + o[2] + o[3];
}

typedef float (*Fn)(const Args&);
struct Arm { const char* name; Fn fn; };
// ---- V8 and V9: THE LEVER THE PROBE LACKED. Both are probes, not candidates -- they compute the
// wrong answer -- and both exist to discriminate two readings of V6's measured 1.5705 cycles/nnz
// on the P-core, which is above BOTH named floors (4-chain latency 1.00 and 3-loads/3-ports 1.00)
// and so is unexplained by the model that predicted it.
//
// THE TWO READINGS. The peer's hypothesis: something restricts this loop to two effective load
// slots on Golden Cove, making the port floor 3 ops / 2 slots = 1.50 cycles/nnz, which V6 sits
// 4.7% above. The alternative: 1.50 arises from something else entirely and the port count is
// slack, in which case removing a load buys only what it bought at two chains.
//
// Recorded before measuring, and the two predictions differ by an ORDER OF MAGNITUDE rather than
// by a floor width, which is what makes this worth running:
//
//   V8 = V6 with the index load removed, so 2 load ops/nnz at four chains.
//     PORT READING:  2 ops / 2 slots = 1.00 cycles/nnz, so V8 beats V6 by roughly 30-36%.
//     SLACK READING: V8 beats V6 by about 1%, which is what V1 bought over V0 at two chains
//                    (-1.041% on P, -0.243% on E).
//     There is no reading that puts it between 5% and 25%. If it lands there, both are wrong.
//
//   V9 = V6 with the B load removed (register-resident B, index still loaded), so 2 load ops/nnz
//        at four chains and NO random memory access.
//     This asks whether V6's 21.259% depends on the random column-indexed load at all. V3 bought
//     -1.420% over V0 at two chains, so the whole cost of that load -- port slot and latency
//     together -- was 1.42% when the accumulator chain was the deep one. If V9 over V6 is again
//     about -1.4%, the random load is equally cheap once the chain is shallower and the gain is
//     dependency. If V9 over V6 is much larger, the load's latency became exposed when the
//     dependency was lifted, and the four-chain gain is partly memory-level parallelism.
//     Note this is at a 128 KB B (nrowsB=8192, K=4), which is L2-resident: the answer is for the
//     L2 band only and a footprint sweep is the companion measurement.

// ---- V8: four accumulator chains, index load removed (B walked sequentially). Not a kernel.
NOINLINE float v8(const Args& A) {
  const float* SCORCH_RESTRICT SCORCH_AV = A.aval;
  const float* SCORCH_RESTRICT SCORCH_BV = A.bval;
  const int SCORCH_N = A.n;
  __m128 acc0 = _mm_setzero_ps(), acc1 = _mm_setzero_ps(),
         acc2 = _mm_setzero_ps(), acc3 = _mm_setzero_ps();
  int p = 0;
  #pragma GCC unroll 2
  for (; p + 4 <= SCORCH_N; p += 4) {
    acc0 = _mm_fmadd_ps(_mm_set1_ps(SCORCH_AV[p]),
                        _mm_loadu_ps(SCORCH_BV + (size_t)p * K), acc0);
    acc1 = _mm_fmadd_ps(_mm_set1_ps(SCORCH_AV[p + 1]),
                        _mm_loadu_ps(SCORCH_BV + (size_t)(p + 1) * K), acc1);
    acc2 = _mm_fmadd_ps(_mm_set1_ps(SCORCH_AV[p + 2]),
                        _mm_loadu_ps(SCORCH_BV + (size_t)(p + 2) * K), acc2);
    acc3 = _mm_fmadd_ps(_mm_set1_ps(SCORCH_AV[p + 3]),
                        _mm_loadu_ps(SCORCH_BV + (size_t)(p + 3) * K), acc3);
  }
  for (; p < SCORCH_N; ++p)
    acc0 = _mm_fmadd_ps(_mm_set1_ps(SCORCH_AV[p]),
                        _mm_loadu_ps(SCORCH_BV + (size_t)p * K), acc0);
  const __m128 r = _mm_add_ps(_mm_add_ps(acc0, acc1), _mm_add_ps(acc2, acc3));
  alignas(16) float o[K]; _mm_storeu_ps(o, r);
  return o[0] + o[1] + o[2] + o[3];
}

// ---- V9: four accumulator chains, B register-resident, index still loaded. Not a kernel.
// `keep` consumes the index exactly as V3 does, so the index load is ISSUED and only the random
// B access is gone -- otherwise -O3 deletes the load and V9 becomes V8 under another symbol,
// which is the failure mode the disassembly gate exists for.
NOINLINE float v9(const Args& A) {
  const int* SCORCH_RESTRICT SCORCH_IDX = A.idx;
  const float* SCORCH_RESTRICT SCORCH_AV = A.aval;
  const int SCORCH_N = A.n;
  const __m128 b0 = _mm_loadu_ps(A.bval);
  const __m128 b1 = _mm_loadu_ps(A.bval + K);
  const __m128 b2 = _mm_loadu_ps(A.bval + 2 * K);
  const __m128 b3 = _mm_loadu_ps(A.bval + 3 * K);
  __m128 acc0 = _mm_setzero_ps(), acc1 = _mm_setzero_ps(),
         acc2 = _mm_setzero_ps(), acc3 = _mm_setzero_ps();
  int32_t keep = 0;
  int p = 0;
  #pragma GCC unroll 2
  for (; p + 4 <= SCORCH_N; p += 4) {
    keep += SCORCH_IDX[p] + SCORCH_IDX[p + 1] + SCORCH_IDX[p + 2] + SCORCH_IDX[p + 3];
    acc0 = _mm_fmadd_ps(_mm_set1_ps(SCORCH_AV[p]),     b0, acc0);
    acc1 = _mm_fmadd_ps(_mm_set1_ps(SCORCH_AV[p + 1]), b1, acc1);
    acc2 = _mm_fmadd_ps(_mm_set1_ps(SCORCH_AV[p + 2]), b2, acc2);
    acc3 = _mm_fmadd_ps(_mm_set1_ps(SCORCH_AV[p + 3]), b3, acc3);
  }
  for (; p < SCORCH_N; ++p) {
    keep += SCORCH_IDX[p];
    acc0 = _mm_fmadd_ps(_mm_set1_ps(SCORCH_AV[p]), b0, acc0);
  }
  const __m128 r = _mm_add_ps(_mm_add_ps(acc0, acc1), _mm_add_ps(acc2, acc3));
  alignas(16) float o[K]; _mm_storeu_ps(o, r);
  return o[0] + o[1] + o[2] + o[3] + (float)keep * 1e-30f;
}

static Arm ARMS[] = {{"V0", v0}, {"V0b", v0b}, {"V1", v1}, {"V2", v2},
                     {"V3", v3}, {"V4", v4},   {"V4b", v4b}, {"V4c", v4c}, {"V5", v5},
                     {"V6", v6},  {"V7", v7},
                     {"V8", v8}, {"V9", v9}};
// Derived, never restated. This read `= 11` while ARMS[] grew to 13, so V8 and V9 were
// compiled, disassembled and ATTESTED -- and never timed. A constant beside a table whose
// length it duplicates is wrong the moment the table changes, and silently: the run produced a
// self-consistent 11-arm CSV that passed every internal check. What caught it was the analyzer
// requiring the CSV's arm set to EQUAL the attestation's, which is the bidirectional form added
// one commit earlier for an unrelated reason.
static const int NARM = (int)(sizeof(ARMS) / sizeof(ARMS[0]));

static double now_ns() {
  struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
  return (double)t.tv_sec * 1e9 + (double)t.tv_nsec;
}

int main(int argc, char** argv) {
  // n: nonzeros per call. reps: timed blocks per arm. inner: calls per block, set so a block
  // exceeds 5 ms as the pre-registration requires. seed fixes the index pattern, which is
  // IDENTICAL across variants so no variant sees a different cache footprint.
  const int n      = argc > 1 ? atoi(argv[1]) : 4096;
  const int reps   = argc > 2 ? atoi(argv[2]) : 31;
  const int nrowsB = argc > 3 ? atoi(argv[3]) : 4096;
  const unsigned seed = argc > 4 ? (unsigned)atoi(argv[4]) : 12345u;

  std::vector<int> idx(n);
  // bval carries 2 rows of slack because V4c loads B0 + K, one row past the row it is
  // given, and idx can name the last row.
  std::vector<float> aval(n), bval((size_t)(nrowsB + 2) * K), dval((size_t)(n + 2) * K);
  unsigned st = seed;
  auto rnd = [&st]() { st = st * 1664525u + 1013904223u; return st; };
  // V1 and V5 index B sequentially, so B must have at least n rows for their addresses to be
  // in bounds; nrowsB >= n is asserted below rather than assumed.
  if (nrowsB < n + 2) { fprintf(stderr, "REFUSE: nrowsB %d < n+2 %d\n", nrowsB, n + 2); return 2; }
  for (int i = 0; i < n; i++) idx[i] = (int)(rnd() % (unsigned)nrowsB);
  for (int i = 0; i < n; i++) aval[i] = (float)((rnd() >> 8) & 0xffff) / 65536.0f;
  for (size_t i = 0; i < bval.size(); i++) bval[i] = (float)((rnd() >> 8) & 0xffff) / 65536.0f;
  for (size_t i = 0; i < dval.size(); i++) dval[i] = (float)((rnd() >> 8) & 0xffff) / 65536.0f;

  Args A{idx.data(), aval.data(), bval.data(), dval.data(), n};

  // Calibrate `inner` once, from V0, and hand the SAME value to every arm -- the batch is fixed by
  // the caller for exactly the reason mrgate.py fixes it: an arm that drew a slow calibration would
  // otherwise carry the penalty for its whole row.
  double t0 = now_ns(); float sink = 0;
  for (int i = 0; i < 64; i++) sink += v0(A);
  double per = (now_ns() - t0) / 64.0;
  int inner = (int)(5.0e6 / (per > 1.0 ? per : 1.0));
  if (inner < 1) inner = 1;

  // Interleaved, randomised order per repetition: arm order is permuted inside each rep so a
  // monotone drift in machine state cannot load onto one arm.
  // Per-repetition values are emitted, not just the minimum: the pre-registration defines the
  // noise floor as the p95 of the PER-REPETITION deviation of V0a/V0b, which a min cannot give.
  std::vector<double> best(NARM, 1e300);
  std::vector<std::vector<double> > all(NARM);
  std::vector<int> order(NARM);
  for (int i = 0; i < NARM; i++) order[i] = i;
  unsigned ost = seed ^ 0x5bf03635u;
  for (int r = 0; r < reps; r++) {
    for (int i = NARM - 1; i > 0; i--) {
      ost = ost * 1664525u + 1013904223u;
      std::swap(order[i], order[(int)(ost % (unsigned)(i + 1))]);
    }
    for (int oi = 0; oi < NARM; oi++) {
      const int a = order[oi];
      double s = now_ns();
      for (int i = 0; i < inner; i++) sink += ARMS[a].fn(A);
      double e = now_ns();
      double nspn = (e - s) / ((double)inner * (double)n);
      all[a].push_back(nspn);
      if (nspn < best[a]) best[a] = nspn;
    }
  }

  printf("# probe n=%d reps=%d inner=%d nrowsB=%d seed=%u sink=%g sinkguard=%g\n",
         n, reps, inner, nrowsB, seed, (double)sink, (double)g_sink[0][0]);
  printf("arm,rep,ns_per_nnz\n");
  for (int a = 0; a < NARM; a++)
    for (size_t r = 0; r < all[a].size(); r++)
      printf("%s,%zu,%.6f\n", ARMS[a].name, r, all[a][r]);
  return 0;
}
