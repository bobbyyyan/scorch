// spmm_micro.cpp — standalone CSR x dense SpMM kernel-variant bench.
//
// Purpose: iterate on the row kernel without rebuilding the torch extension, under
// the same flags the shipped .so uses (-O3 -march=native -ffast-math -funroll-loops
// -fopenmp). Every variant runs against the same matrix, the same B, and the same
// parallel skeleton as spmm_csr_float_v2 (atomic row work-stealing over `chunk`
// rows), so a difference here is a difference in the row kernel and nothing else.
//
// Timing follows the house method: each round runs every variant once in a fresh
// random permutation (rotating a fixed order lets a heavy arm poison whichever arm
// follows it), and the reported number is the median over rounds. `base` is entered
// twice under two names so |aa/base - 1| is the per-cell noise floor.
//
// build: g++ -O3 -march=native -ffast-math -funroll-loops -fopenmp -std=c++17 \
//            -o spmm_micro spmm_micro.cpp
// usage: ./spmm_micro <matrix.bin> <N> <reps> <threads> [variant,variant,...]
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <random>
#include <string>
#include <vector>

#include <immintrin.h>
#include <omp.h>
#include <time.h>

#define RESTRICT __restrict__

static double now(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

// --------------------------------------------------------------------------- //
// matrix
// --------------------------------------------------------------------------- //
struct CSR {
  int64_t M = 0, J = 0, nnz = 0;
  std::vector<int> pos, crd;
  std::vector<int64_t> crd64;   // same coordinates, int64 — for the wide-index arms
  std::vector<float> val;
};

static CSR load_csr(const char *path) {
  CSR a;
  FILE *f = fopen(path, "rb");
  if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
  int64_t hdr[3];
  if (fread(hdr, sizeof(int64_t), 3, f) != 3) { fprintf(stderr, "short read\n"); exit(1); }
  a.M = hdr[0]; a.J = hdr[1]; a.nnz = hdr[2];
  a.pos.resize(a.M + 1); a.crd.resize(a.nnz); a.val.resize(a.nnz);
  if (fread(a.pos.data(), sizeof(int), a.M + 1, f) != (size_t)(a.M + 1)) exit(1);
  if (fread(a.crd.data(), sizeof(int), a.nnz, f) != (size_t)a.nnz) exit(1);
  if (fread(a.val.data(), sizeof(float), a.nnz, f) != (size_t)a.nnz) exit(1);
  fclose(f);
  a.crd64.resize(a.nnz);
  for (int64_t i = 0; i < a.nnz; i++) a.crd64[i] = a.crd[i];
  return a;
}

// --------------------------------------------------------------------------- //
// prefetch hint plumbing: gcc's __builtin_prefetch takes a compile-time locality,
// 3 -> PREFETCHT0 (L1), 2 -> T1, 1 -> T2, 0 -> NTA. The shipped kernel uses 1,
// i.e. it never brings a B row nearer than L2/L3, so every demand load still pays
// an L1 miss even when the prefetch landed.
// --------------------------------------------------------------------------- //
template <int H>
static inline void pf(const void *p) {
  if (H == 0) __builtin_prefetch(p, 0, 0);
  else if (H == 1) __builtin_prefetch(p, 0, 1);
  else if (H == 2) __builtin_prefetch(p, 0, 2);
  else if (H == 3) __builtin_prefetch(p, 0, 3);
}

static inline __m256i lane_mask(int ml) {
  return _mm256_setr_epi32(ml > 0 ? -1 : 0, ml > 1 ? -1 : 0, ml > 2 ? -1 : 0,
                           ml > 3 ? -1 : 0, ml > 4 ? -1 : 0, ml > 5 ? -1 : 0,
                           ml > 6 ? -1 : 0, ml > 7 ? -1 : 0);
}

// --------------------------------------------------------------------------- //
// Narrow-N row kernel, parameterised on everything the shipped one hardcodes.
//   NVEC  output-row vectors held in registers (N <= 8*NVEC)
//   ILP   nonzeros consumed per iteration (independent accumulator sets)
//   PFD   prefetch distance in NONZEROS (0 = no prefetch)
//   PFH   prefetch hint (see pf<H>)
//   FULL  true when N % 8 == 0, so the last vector needs no mask at all
// --------------------------------------------------------------------------- //
template <int NVEC, int ILP, int PFD, int PFH, bool FULL>
static inline void row_narrow(const int *RESTRICT crd, const float *RESTRICT val,
                              const float *RESTRICT B, int N, float *RESTRICT C,
                              int p0, int p1, __m256i mask) {
  __m256 acc[ILP][NVEC];
#pragma unroll
  for (int c = 0; c < ILP; c++)
#pragma unroll
    for (int v = 0; v < NVEC; v++) acc[c][v] = _mm256_setzero_ps();

  int p = p0;
  for (; p + ILP <= p1; p += ILP) {
    if (PFD > 0 && p + PFD < p1)
      pf<PFH>(B + (size_t)crd[p + PFD] * (size_t)N);
#pragma unroll
    for (int c = 0; c < ILP; c++) {
      const float *RESTRICT Bp = B + (size_t)crd[p + c] * (size_t)N;
      const __m256 a = _mm256_set1_ps(val[p + c]);
#pragma unroll
      for (int v = 0; v < NVEC; v++) {
        const bool m = (v == NVEC - 1) && !FULL;
        const __m256 b = m ? _mm256_maskload_ps(Bp + 8 * v, mask)
                           : _mm256_loadu_ps(Bp + 8 * v);
        acc[c][v] = _mm256_fmadd_ps(a, b, acc[c][v]);
      }
    }
  }
  for (; p < p1; p++) {  // ragged tail nonzeros
    const float *RESTRICT Bp = B + (size_t)crd[p] * (size_t)N;
    const __m256 a = _mm256_set1_ps(val[p]);
#pragma unroll
    for (int v = 0; v < NVEC; v++) {
      const bool m = (v == NVEC - 1) && !FULL;
      const __m256 b = m ? _mm256_maskload_ps(Bp + 8 * v, mask)
                         : _mm256_loadu_ps(Bp + 8 * v);
      acc[0][v] = _mm256_fmadd_ps(a, b, acc[0][v]);
    }
  }
#pragma unroll
  for (int v = 0; v < NVEC; v++) {
    __m256 r = acc[0][v];
#pragma unroll
    for (int c = 1; c < ILP; c++) r = _mm256_add_ps(r, acc[c][v]);
    if ((v == NVEC - 1) && !FULL) _mm256_maskstore_ps(C + 8 * v, mask, r);
    else _mm256_storeu_ps(C + 8 * v, r);
  }
}

// Wide-N: 64-wide k-tiles, 8 accumulators per tile, A re-read per tile (L1-hot).
template <int ILP, int PFD, int PFH>
static inline void row_wide(const int *RESTRICT crd, const float *RESTRICT val,
                            const float *RESTRICT B, int N, float *RESTRICT C,
                            int p0, int p1) {
  int k0 = 0;
  for (; k0 + 64 <= N; k0 += 64) {
    __m256 acc[ILP][8];
#pragma unroll
    for (int c = 0; c < ILP; c++)
#pragma unroll
      for (int v = 0; v < 8; v++) acc[c][v] = _mm256_setzero_ps();
    int p = p0;
    for (; p + ILP <= p1; p += ILP) {
      if (PFD > 0 && p + PFD < p1)
        pf<PFH>(B + (size_t)crd[p + PFD] * (size_t)N + k0);
#pragma unroll
      for (int c = 0; c < ILP; c++) {
        const float *RESTRICT Bp = B + (size_t)crd[p + c] * (size_t)N + k0;
        const __m256 a = _mm256_set1_ps(val[p + c]);
#pragma unroll
        for (int v = 0; v < 8; v++)
          acc[c][v] = _mm256_fmadd_ps(a, _mm256_loadu_ps(Bp + 8 * v), acc[c][v]);
      }
    }
    for (; p < p1; p++) {
      const float *RESTRICT Bp = B + (size_t)crd[p] * (size_t)N + k0;
      const __m256 a = _mm256_set1_ps(val[p]);
#pragma unroll
      for (int v = 0; v < 8; v++)
        acc[0][v] = _mm256_fmadd_ps(a, _mm256_loadu_ps(Bp + 8 * v), acc[0][v]);
    }
#pragma unroll
    for (int v = 0; v < 8; v++) {
      __m256 r = acc[0][v];
#pragma unroll
      for (int c = 1; c < ILP; c++) r = _mm256_add_ps(r, acc[c][v]);
      _mm256_storeu_ps(C + k0 + 8 * v, r);
    }
  }
  const int kw = N - k0;
  if (kw > 0) {  // ragged last tile, scalar-safe and rarely hot
    const int nv = (kw + 7) / 8;
    const int ml = kw - 8 * (nv - 1);
    const __m256i mask = lane_mask(ml);
    __m256 acc[8];
    for (int v = 0; v < nv; v++) acc[v] = _mm256_setzero_ps();
    for (int p = p0; p < p1; p++) {
      const float *RESTRICT Bp = B + (size_t)crd[p] * (size_t)N + k0;
      const __m256 a = _mm256_set1_ps(val[p]);
      for (int v = 0; v < nv; v++) {
        const __m256 b = (v == nv - 1 && ml != 8) ? _mm256_maskload_ps(Bp + 8 * v, mask)
                                                  : _mm256_loadu_ps(Bp + 8 * v);
        acc[v] = _mm256_fmadd_ps(a, b, acc[v]);
      }
    }
    for (int v = 0; v < nv; v++) {
      if (v == nv - 1 && ml != 8) _mm256_maskstore_ps(C + k0 + 8 * v, mask, acc[v]);
      else _mm256_storeu_ps(C + k0 + 8 * v, acc[v]);
    }
  }
}

template <int NVEC, int ILP, int PFD, int PFH, bool FULL>
static inline void row_narrow64(const int64_t *RESTRICT crd, const float *RESTRICT val,
                                const float *RESTRICT B, int N, float *RESTRICT C,
                                int p0, int p1, __m256i mask) {
  __m256 acc[ILP][NVEC];
#pragma unroll
  for (int c = 0; c < ILP; c++)
#pragma unroll
    for (int v = 0; v < NVEC; v++) acc[c][v] = _mm256_setzero_ps();
  int p = p0;
  for (; p + ILP <= p1; p += ILP) {
    if (PFD > 0 && p + PFD < p1) pf<PFH>(B + (size_t)crd[p + PFD] * (size_t)N);
#pragma unroll
    for (int c = 0; c < ILP; c++) {
      const float *RESTRICT Bp = B + (size_t)crd[p + c] * (size_t)N;
      const __m256 a = _mm256_set1_ps(val[p + c]);
#pragma unroll
      for (int v = 0; v < NVEC; v++) {
        const bool m = (v == NVEC - 1) && !FULL;
        const __m256 b = m ? _mm256_maskload_ps(Bp + 8 * v, mask)
                           : _mm256_loadu_ps(Bp + 8 * v);
        acc[c][v] = _mm256_fmadd_ps(a, b, acc[c][v]);
      }
    }
  }
  for (; p < p1; p++) {
    const float *RESTRICT Bp = B + (size_t)crd[p] * (size_t)N;
    const __m256 a = _mm256_set1_ps(val[p]);
#pragma unroll
    for (int v = 0; v < NVEC; v++) {
      const bool m = (v == NVEC - 1) && !FULL;
      const __m256 b = m ? _mm256_maskload_ps(Bp + 8 * v, mask)
                         : _mm256_loadu_ps(Bp + 8 * v);
      acc[0][v] = _mm256_fmadd_ps(a, b, acc[0][v]);
    }
  }
#pragma unroll
  for (int v = 0; v < NVEC; v++) {
    __m256 r = acc[0][v];
#pragma unroll
    for (int c = 1; c < ILP; c++) r = _mm256_add_ps(r, acc[c][v]);
    if ((v == NVEC - 1) && !FULL) _mm256_maskstore_ps(C + 8 * v, mask, r);
    else _mm256_storeu_ps(C + 8 * v, r);
  }
}

template <int ILP, int PFD, int PFH>
static inline void row_wide64(const int64_t *RESTRICT crd, const float *RESTRICT val,
                              const float *RESTRICT B, int N, float *RESTRICT C,
                              int p0, int p1) {
  int k0 = 0;
  for (; k0 + 64 <= N; k0 += 64) {
    __m256 acc[8];
#pragma unroll
    for (int v = 0; v < 8; v++) acc[v] = _mm256_setzero_ps();
    for (int p = p0; p < p1; p++) {
      const float *RESTRICT Bp = B + (size_t)crd[p] * (size_t)N + k0;
      if (PFD > 0 && p + PFD < p1)
        pf<PFH>(B + (size_t)crd[p + PFD] * (size_t)N + k0);
      const __m256 a = _mm256_set1_ps(val[p]);
#pragma unroll
      for (int v = 0; v < 8; v++)
        acc[v] = _mm256_fmadd_ps(a, _mm256_loadu_ps(Bp + 8 * v), acc[v]);
    }
#pragma unroll
    for (int v = 0; v < 8; v++) _mm256_storeu_ps(C + k0 + 8 * v, acc[v]);
  }
  for (int k = k0; k < N; k++) {
    float s = 0.f;
    for (int p = p0; p < p1; p++) s += val[p] * B[(size_t)crd[p] * (size_t)N + k];
    C[k] = s;
  }
}

template <int ILP, int PFD, int PFH, bool NOMASK>
static void spmm64(const CSR &a, const float *RESTRICT B, int N, float *RESTRICT C,
                   int nthreads, int chunk) {
  const int nvec = (N + 7) / 8;
  const int ml = N - 8 * (nvec - 1);
  const __m256i mask = lane_mask(ml);
  const bool full = NOMASK && (N % 8 == 0);
  std::atomic<int> next_row{0};
  const int M = (int)a.M;
  const int *RESTRICT pos = a.pos.data();
  const int64_t *RESTRICT crd = a.crd64.data();
  const float *RESTRICT val = a.val.data();
#pragma omp parallel num_threads(nthreads)
  {
    while (true) {
      const int start = next_row.fetch_add(chunk, std::memory_order_relaxed);
      if (start >= M) break;
      const int end = std::min(start + chunk, M);
      for (int i = start; i < end; i++) {
        const int p0 = pos[i], p1 = pos[i + 1];
        float *RESTRICT Ci = C + (size_t)i * (size_t)N;
        if (p0 == p1) { memset(Ci, 0, sizeof(float) * N); continue; }
        if (N > 32) { row_wide64<ILP, PFD, PFH>(crd, val, B, N, Ci, p0, p1); continue; }
        switch (nvec) {
#define CASE64(NV)                                                                   \
  case NV:                                                                           \
    if (full) row_narrow64<NV, ILP, PFD, PFH, true>(crd, val, B, N, Ci, p0, p1, mask); \
    else row_narrow64<NV, ILP, PFD, PFH, false>(crd, val, B, N, Ci, p0, p1, mask);     \
    break;
          CASE64(1) CASE64(2) CASE64(3) CASE64(4)
#undef CASE64
          default: break;
        }
      }
    }
  }
}

// --------------------------------------------------------------------------- //
// TWO-ROW BLOCKED narrow-N kernel.
//
// Consecutive CSR rows are column-sorted, so two rows can be walked with a
// two-pointer merge; where they share a column, ONE B-row load feeds two FMAs.
// The measured adjacent-row column-set overlap is 0.81-0.89 on the structural
// matrices this study loses on (inline_1, audikw_1, crankseg_1) and 0.007-0.12 on
// the scattered graphs it wins on, so the shared-column branch is both the common
// case and the well-predicted one exactly where the reuse exists.
//
// This halves B *load* traffic on the shared columns, which is what the row-at-a-
// time kernel cannot do: v2 relies on the cache still holding row i's B lines when
// row i+1 asks for them, and pays an L1/L2 access per nonzero either way.
// --------------------------------------------------------------------------- //
template <int NVEC, int PFD, int PFH, bool FULL>
static inline void row_narrow_2r(const int *RESTRICT crd, const float *RESTRICT val,
                                 const float *RESTRICT B, int N,
                                 float *RESTRICT C0, float *RESTRICT C1,
                                 int p0, int p1, int q0, int q1, __m256i mask) {
  __m256 a0[NVEC], a1[NVEC];
#pragma unroll
  for (int v = 0; v < NVEC; v++) { a0[v] = _mm256_setzero_ps(); a1[v] = _mm256_setzero_ps(); }

  int p = p0, q = q0;
  while (p < p1 && q < q1) {
    const int cp = crd[p], cq = crd[q];
    if (PFD > 0) {
      if (p + PFD < p1) pf<PFH>(B + (size_t)crd[p + PFD] * (size_t)N);
      if (q + PFD < q1) pf<PFH>(B + (size_t)crd[q + PFD] * (size_t)N);
    }
    if (cp == cq) {                                   // shared column: one load, two FMAs
      const float *RESTRICT Bp = B + (size_t)cp * (size_t)N;
      const __m256 s0 = _mm256_set1_ps(val[p]);
      const __m256 s1 = _mm256_set1_ps(val[q]);
#pragma unroll
      for (int v = 0; v < NVEC; v++) {
        const bool m = (v == NVEC - 1) && !FULL;
        const __m256 b = m ? _mm256_maskload_ps(Bp + 8 * v, mask)
                           : _mm256_loadu_ps(Bp + 8 * v);
        a0[v] = _mm256_fmadd_ps(s0, b, a0[v]);
        a1[v] = _mm256_fmadd_ps(s1, b, a1[v]);
      }
      p++; q++;
    } else if (cp < cq) {
      const float *RESTRICT Bp = B + (size_t)cp * (size_t)N;
      const __m256 s0 = _mm256_set1_ps(val[p]);
#pragma unroll
      for (int v = 0; v < NVEC; v++) {
        const bool m = (v == NVEC - 1) && !FULL;
        const __m256 b = m ? _mm256_maskload_ps(Bp + 8 * v, mask)
                           : _mm256_loadu_ps(Bp + 8 * v);
        a0[v] = _mm256_fmadd_ps(s0, b, a0[v]);
      }
      p++;
    } else {
      const float *RESTRICT Bp = B + (size_t)cq * (size_t)N;
      const __m256 s1 = _mm256_set1_ps(val[q]);
#pragma unroll
      for (int v = 0; v < NVEC; v++) {
        const bool m = (v == NVEC - 1) && !FULL;
        const __m256 b = m ? _mm256_maskload_ps(Bp + 8 * v, mask)
                           : _mm256_loadu_ps(Bp + 8 * v);
        a1[v] = _mm256_fmadd_ps(s1, b, a1[v]);
      }
      q++;
    }
  }
  for (; p < p1; p++) {
    const float *RESTRICT Bp = B + (size_t)crd[p] * (size_t)N;
    const __m256 s0 = _mm256_set1_ps(val[p]);
#pragma unroll
    for (int v = 0; v < NVEC; v++) {
      const bool m = (v == NVEC - 1) && !FULL;
      const __m256 b = m ? _mm256_maskload_ps(Bp + 8 * v, mask) : _mm256_loadu_ps(Bp + 8 * v);
      a0[v] = _mm256_fmadd_ps(s0, b, a0[v]);
    }
  }
  for (; q < q1; q++) {
    const float *RESTRICT Bp = B + (size_t)crd[q] * (size_t)N;
    const __m256 s1 = _mm256_set1_ps(val[q]);
#pragma unroll
    for (int v = 0; v < NVEC; v++) {
      const bool m = (v == NVEC - 1) && !FULL;
      const __m256 b = m ? _mm256_maskload_ps(Bp + 8 * v, mask) : _mm256_loadu_ps(Bp + 8 * v);
      a1[v] = _mm256_fmadd_ps(s1, b, a1[v]);
    }
  }
#pragma unroll
  for (int v = 0; v < NVEC; v++) {
    if ((v == NVEC - 1) && !FULL) {
      _mm256_maskstore_ps(C0 + 8 * v, mask, a0[v]);
      _mm256_maskstore_ps(C1 + 8 * v, mask, a1[v]);
    } else {
      _mm256_storeu_ps(C0 + 8 * v, a0[v]);
      _mm256_storeu_ps(C1 + 8 * v, a1[v]);
    }
  }
}

// Wide-N two-row variant: 32-wide k-tiles so two rows' accumulators (2 x 4 YMM)
// leave registers free for the broadcast and the B vector. A 64-wide tile would
// need 16 YMM for accumulators alone and spill.
template <int PFD, int PFH>
static inline void row_wide_2r(const int *RESTRICT crd, const float *RESTRICT val,
                               const float *RESTRICT B, int N,
                               float *RESTRICT C0, float *RESTRICT C1,
                               int p0, int p1, int q0, int q1) {
  int k0 = 0;
  for (; k0 + 32 <= N; k0 += 32) {
    __m256 a0[4], a1[4];
#pragma unroll
    for (int v = 0; v < 4; v++) { a0[v] = _mm256_setzero_ps(); a1[v] = _mm256_setzero_ps(); }
    int p = p0, q = q0;
    while (p < p1 && q < q1) {
      const int cp = crd[p], cq = crd[q];
      if (PFD > 0) {
        if (p + PFD < p1) pf<PFH>(B + (size_t)crd[p + PFD] * (size_t)N + k0);
        if (q + PFD < q1) pf<PFH>(B + (size_t)crd[q + PFD] * (size_t)N + k0);
      }
      if (cp == cq) {
        const float *RESTRICT Bp = B + (size_t)cp * (size_t)N + k0;
        const __m256 s0 = _mm256_set1_ps(val[p]), s1 = _mm256_set1_ps(val[q]);
#pragma unroll
        for (int v = 0; v < 4; v++) {
          const __m256 b = _mm256_loadu_ps(Bp + 8 * v);
          a0[v] = _mm256_fmadd_ps(s0, b, a0[v]);
          a1[v] = _mm256_fmadd_ps(s1, b, a1[v]);
        }
        p++; q++;
      } else if (cp < cq) {
        const float *RESTRICT Bp = B + (size_t)cp * (size_t)N + k0;
        const __m256 s0 = _mm256_set1_ps(val[p]);
#pragma unroll
        for (int v = 0; v < 4; v++) a0[v] = _mm256_fmadd_ps(s0, _mm256_loadu_ps(Bp + 8 * v), a0[v]);
        p++;
      } else {
        const float *RESTRICT Bp = B + (size_t)cq * (size_t)N + k0;
        const __m256 s1 = _mm256_set1_ps(val[q]);
#pragma unroll
        for (int v = 0; v < 4; v++) a1[v] = _mm256_fmadd_ps(s1, _mm256_loadu_ps(Bp + 8 * v), a1[v]);
        q++;
      }
    }
    for (; p < p1; p++) {
      const float *RESTRICT Bp = B + (size_t)crd[p] * (size_t)N + k0;
      const __m256 s0 = _mm256_set1_ps(val[p]);
#pragma unroll
      for (int v = 0; v < 4; v++) a0[v] = _mm256_fmadd_ps(s0, _mm256_loadu_ps(Bp + 8 * v), a0[v]);
    }
    for (; q < q1; q++) {
      const float *RESTRICT Bp = B + (size_t)crd[q] * (size_t)N + k0;
      const __m256 s1 = _mm256_set1_ps(val[q]);
#pragma unroll
      for (int v = 0; v < 4; v++) a1[v] = _mm256_fmadd_ps(s1, _mm256_loadu_ps(Bp + 8 * v), a1[v]);
    }
#pragma unroll
    for (int v = 0; v < 4; v++) {
      _mm256_storeu_ps(C0 + k0 + 8 * v, a0[v]);
      _mm256_storeu_ps(C1 + k0 + 8 * v, a1[v]);
    }
  }
  // Ragged remainder (N % 32 != 0): a scalar strip. Correctness first — this is at
  // most 31 columns and the variants that matter are measured at N a multiple of 32.
  if (k0 < N) {
    for (int k = k0; k < N; k++) { C0[k] = 0.f; C1[k] = 0.f; }
    for (int p = p0; p < p1; p++) {
      const float *RESTRICT Bp = B + (size_t)crd[p] * (size_t)N;
      const float s = val[p];
      for (int k = k0; k < N; k++) C0[k] += s * Bp[k];
    }
    for (int q = q0; q < q1; q++) {
      const float *RESTRICT Bp = B + (size_t)crd[q] * (size_t)N;
      const float s = val[q];
      for (int k = k0; k < N; k++) C1[k] += s * Bp[k];
    }
  }
}

// --------------------------------------------------------------------------- //
// Whole-SpMM driver, parameterised the same way. Parallel skeleton copies
// spmm_csr_float_v2: one omp team, atomic fetch_add over `chunk` rows.
// --------------------------------------------------------------------------- //
struct Cfg {
  int ilp = 2, pfd = 2, pfh = 1;
  bool nomask = false;   // specialise N%8==0 to drop the mask entirely
};

template <int ILP, int PFD, int PFH, bool NOMASK>
static void spmm(const CSR &a, const float *RESTRICT B, int N, float *RESTRICT C,
                 int nthreads, int chunk) {
  const int nvec = (N + 7) / 8;
  const int ml = N - 8 * (nvec - 1);
  const __m256i mask = lane_mask(ml);
  const bool full = NOMASK && (N % 8 == 0);
  std::atomic<int> next_row{0};
  const int M = (int)a.M;
  const int *RESTRICT pos = a.pos.data();
  const int *RESTRICT crd = a.crd.data();
  const float *RESTRICT val = a.val.data();

#pragma omp parallel num_threads(nthreads)
  {
    while (true) {
      const int start = next_row.fetch_add(chunk, std::memory_order_relaxed);
      if (start >= M) break;
      const int end = std::min(start + chunk, M);
      for (int i = start; i < end; i++) {
        const int p0 = pos[i], p1 = pos[i + 1];
        float *RESTRICT Ci = C + (size_t)i * (size_t)N;
        if (p0 == p1) { memset(Ci, 0, sizeof(float) * N); continue; }
        if (N > 32) { row_wide<ILP, PFD, PFH>(crd, val, B, N, Ci, p0, p1); continue; }
        switch (nvec) {
#define CASE(NV)                                                                \
  case NV:                                                                      \
    if (full) row_narrow<NV, ILP, PFD, PFH, true>(crd, val, B, N, Ci, p0, p1, mask); \
    else row_narrow<NV, ILP, PFD, PFH, false>(crd, val, B, N, Ci, p0, p1, mask);     \
    break;
          CASE(1) CASE(2) CASE(3) CASE(4)
#undef CASE
          default: break;
        }
      }
    }
  }
}

// --------------------------------------------------------------------------- //
using Fn = void (*)(const CSR &, const float *, int, float *, int, int);

struct Variant { const char *name; Fn fn; };

#define MK(NAME, ILP, PFD, PFH, NOMASK)                                          \
  static void NAME(const CSR &a, const float *B, int N, float *C, int nt, int ch) { \
    spmm<ILP, PFD, PFH, NOMASK>(a, B, N, C, nt, ch);                             \
  }

// base reproduces the shipped kernel exactly: 2-nnz ILP, prefetch 2 nnz ahead with
// gcc locality 1 (PREFETCHT2), mask on the last vector even when N%8==0.
MK(v_base,   2,  2, 1, false)
MK(v_aa,     2,  2, 1, false)
MK(v_nopf,   2,  0, 1, false)
MK(v_pfT0,   2,  2, 3, false)
MK(v_pfT1,   2,  2, 2, false)
MK(v_d8T0,   2,  8, 3, false)
MK(v_d16T0,  2, 16, 3, false)
MK(v_d32T0,  2, 32, 3, false)
MK(v_d64T0,  2, 64, 3, false)
MK(v_d16T2,  2, 16, 1, false)
MK(v_d16NTA, 2, 16, 0, false)
MK(v_nomask, 2,  2, 1, true)
MK(v_nm_d16, 2, 16, 3, true)
MK(v_ilp1,   1,  2, 1, false)
MK(v_ilp4,   4,  2, 1, false)
MK(v_ilp4d16,4, 16, 3, true)
MK(v_ilp8d16,8, 16, 3, true)

#define MK64(NAME, ILP, PFD, PFH, NOMASK)                                          \
  static void NAME(const CSR &a, const float *B, int N, float *C, int nt, int ch) { \
    spmm64<ILP, PFD, PFH, NOMASK>(a, B, N, C, nt, ch);                             \
  }
// Identical arithmetic to v_base / v_nm_d16; the only difference is that each column
// index is loaded as 8 bytes instead of 4, so A's stream is 12 bytes per nonzero
// instead of 8. This measures what an int64-index kernel family would cost per call.
MK64(v_base64, 2, 2, 1, false)
MK64(v_nm_d16_64, 2, 16, 3, true)

// --------------------------------------------------------------------------- //
// TWO-ROW BLOCKED whole-SpMM driver. Same work-stealing skeleton; rows are consumed
// in pairs inside a stolen chunk, with a single-row tail when the chunk is odd.
//
// Two effects are in play and they need separating in the results. On matrices whose
// consecutive rows share columns (structural/FEM: adjacent-row overlap 0.81-0.89) a
// shared column costs one B-row load instead of two. On SHORT-row matrices (graphs at
// degree 6-16) the win is different and does not need any overlap at all: the per-row
// prologue and the output store are amortised over two rows.
// --------------------------------------------------------------------------- //
template <int PFD, int PFH, bool NOMASK>
static void spmm_2r(const CSR &a, const float *RESTRICT B, int N, float *RESTRICT C,
                    int nthreads, int chunk) {
  const int nvec = (N + 7) / 8;
  const int ml = N - 8 * (nvec - 1);
  const __m256i mask = lane_mask(ml);
  const bool full = NOMASK && (N % 8 == 0);
  std::atomic<int> next_row{0};
  const int M = (int)a.M;
  const int *RESTRICT pos = a.pos.data();
  const int *RESTRICT crd = a.crd.data();
  const float *RESTRICT val = a.val.data();

#pragma omp parallel num_threads(nthreads)
  {
    while (true) {
      const int start = next_row.fetch_add(chunk, std::memory_order_relaxed);
      if (start >= M) break;
      const int end = std::min(start + chunk, M);
      int i = start;
      for (; i + 1 < end; i += 2) {
        const int p0 = pos[i], p1 = pos[i + 1], q1 = pos[i + 2];
        float *RESTRICT C0 = C + (size_t)i * (size_t)N;
        float *RESTRICT C1 = C0 + (size_t)N;
        if (N > 32) {
          row_wide_2r<PFD, PFH>(crd, val, B, N, C0, C1, p0, p1, p1, q1);
          continue;
        }
        switch (nvec) {
#define CASE2R(NV)                                                                    \
  case NV:                                                                            \
    if (full)                                                                         \
      row_narrow_2r<NV, PFD, PFH, true>(crd, val, B, N, C0, C1, p0, p1, p1, q1, mask); \
    else                                                                              \
      row_narrow_2r<NV, PFD, PFH, false>(crd, val, B, N, C0, C1, p0, p1, p1, q1, mask);\
    break;
          CASE2R(1) CASE2R(2) CASE2R(3) CASE2R(4)
#undef CASE2R
          default: break;
        }
      }
      for (; i < end; i++) {  // odd tail row in this chunk
        const int p0 = pos[i], p1 = pos[i + 1];
        float *RESTRICT Ci = C + (size_t)i * (size_t)N;
        if (p0 == p1) { memset(Ci, 0, sizeof(float) * N); continue; }
        if (N > 32) { row_wide<2, PFD, PFH>(crd, val, B, N, Ci, p0, p1); continue; }
        switch (nvec) {
#define CASE1R(NV)                                                                   \
  case NV:                                                                           \
    if (full) row_narrow<NV, 2, PFD, PFH, true>(crd, val, B, N, Ci, p0, p1, mask);   \
    else row_narrow<NV, 2, PFD, PFH, false>(crd, val, B, N, Ci, p0, p1, mask);       \
    break;
          CASE1R(1) CASE1R(2) CASE1R(3) CASE1R(4)
#undef CASE1R
          default: break;
        }
      }
    }
  }
}

#define MK2R(NAME, PFD, PFH, NOMASK)                                              \
  static void NAME(const CSR &a, const float *B, int N, float *C, int nt, int ch) { \
    spmm_2r<PFD, PFH, NOMASK>(a, B, N, C, nt, ch);                                \
  }
MK2R(v_r2, 2, 1, false)
MK2R(v_r2_d16, 16, 3, true)

static Variant VARIANTS[] = {
    {"base", v_base},       {"aa", v_aa},           {"nopf", v_nopf},
    {"pfT0", v_pfT0},       {"pfT1", v_pfT1},       {"d8T0", v_d8T0},
    {"d16T0", v_d16T0},     {"d32T0", v_d32T0},     {"d64T0", v_d64T0},
    {"d16T2", v_d16T2},     {"d16NTA", v_d16NTA},   {"nomask", v_nomask},
    {"nm_d16", v_nm_d16},   {"ilp1", v_ilp1},       {"ilp4", v_ilp4},
    {"ilp4d16", v_ilp4d16}, {"ilp8d16", v_ilp8d16},
    {"base64", v_base64},   {"nm_d16_64", v_nm_d16_64},
    {"r2", v_r2},           {"r2_d16", v_r2_d16},
};

// --------------------------------------------------------------------------- //
int main(int argc, char **argv) {
  if (argc < 3) {
    fprintf(stderr, "usage: %s <matrix.bin> <N> [reps] [threads] [v1,v2,...]\n", argv[0]);
    return 1;
  }
  CSR a = load_csr(argv[1]);
  const int N = atoi(argv[2]);
  const int reps = argc > 3 ? atoi(argv[3]) : 11;
  const int nthreads = argc > 4 ? atoi(argv[4]) : omp_get_num_procs();
  std::string want = argc > 5 ? argv[5] : "";

  // chunk: same formula as scorch_chunk (7 chunks per worker, clamped 4..64)
  long ch = a.M / ((long)nthreads * 7);
  if (ch < 4) ch = 4;
  if (ch > 64) ch = 64;
  const int chunk = (int)ch;

  const size_t bsz = (size_t)a.J * (size_t)N, csz = (size_t)a.M * (size_t)N;
  float *B = (float *)aligned_alloc(64, bsz * sizeof(float));
  float *C = (float *)aligned_alloc(64, csz * sizeof(float));
  if (!B || !C) { fprintf(stderr, "alloc failed\n"); return 1; }
  std::mt19937 rng(12345);
  std::uniform_real_distribution<float> ud(-1.f, 1.f);
  for (size_t i = 0; i < bsz; i++) B[i] = ud(rng);
  memset(C, 0, csz * sizeof(float));

  // double-precision reference on a bounded row sample (whole-matrix double SpMM
  // would dominate the run on the big matrices)
  const int nchk = (int)std::min<int64_t>(a.M, 512);
  std::vector<int> chk(nchk);
  for (int i = 0; i < nchk; i++) chk[i] = (int)((int64_t)i * a.M / nchk);
  std::vector<double> ref((size_t)nchk * N, 0.0);
  for (int t = 0; t < nchk; t++) {
    const int i = chk[t];
    for (int p = a.pos[i]; p < a.pos[i + 1]; p++)
      for (int k = 0; k < N; k++)
        ref[(size_t)t * N + k] += (double)a.val[p] * (double)B[(size_t)a.crd[p] * N + k];
  }

  std::vector<Variant> vs;
  for (auto &v : VARIANTS) {
    if (want.empty() || want.find(v.name) != std::string::npos) vs.push_back(v);
  }

  // warmup + correctness
  std::vector<double> relerr(vs.size(), 0.0);
  for (size_t vi = 0; vi < vs.size(); vi++) {
    vs[vi].fn(a, B, N, C, nthreads, chunk);
    double num = 0, den = 0;
    for (int t = 0; t < nchk; t++)
      for (int k = 0; k < N; k++) {
        const double r = ref[(size_t)t * N + k];
        const double g = C[(size_t)chk[t] * N + k];
        num = std::max(num, std::fabs(g - r));
        den = std::max(den, std::fabs(r));
      }
    relerr[vi] = den > 0 ? num / den : 0.0;
  }

  // random-permutation interleaved rounds
  std::vector<std::vector<double>> samples(vs.size());
  std::mt19937 orng(999);
  std::vector<int> order(vs.size());
  for (size_t i = 0; i < vs.size(); i++) order[i] = (int)i;
  for (int r = 0; r < reps; r++) {
    std::shuffle(order.begin(), order.end(), orng);
    for (int vi : order) {
      const double t0 = now();
      vs[vi].fn(a, B, N, C, nthreads, chunk);
      samples[vi].push_back(now() - t0);
    }
  }

  printf("# matrix=%s M=%lld J=%lld nnz=%lld N=%d threads=%d chunk=%d reps=%d\n",
         argv[1], (long long)a.M, (long long)a.J, (long long)a.nnz, N, nthreads,
         chunk, reps);
  double basemed = 0;
  for (size_t vi = 0; vi < vs.size(); vi++) {
    auto s = samples[vi];
    std::sort(s.begin(), s.end());
    const double med = s[s.size() / 2];
    if (std::string(vs[vi].name) == "base") basemed = med;
    samples[vi] = s;
  }
  for (size_t vi = 0; vi < vs.size(); vi++) {
    const double med = samples[vi][samples[vi].size() / 2];
    printf("MICRO %-9s med_ms=%9.4f vs_base=%7.4f relerr=%.2e\n", vs[vi].name,
           med * 1e3, basemed > 0 ? basemed / med : 0.0, relerr[vi]);
  }
  free(B); free(C);
  return 0;
}
