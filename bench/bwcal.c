// Calibrate the `cache-misses` (LONGEST_LAT_CACHE.MISS) -> DRAM-bytes proxy, and
// measure this machine's achievable memory bandwidth. Both are prerequisites for a
// roofline attribution when no uncore IMC PMU is available.
//
// Each mode touches a known number of compulsory DRAM bytes, so the ratio
// (bytes / cache-misses) tells us how many bytes a counted L3 miss really moves.
//
//   read   : sum over N floats                      -> N*4 read bytes
//   write  : store a constant over N floats         -> N*4 RFO-read + N*4 writeback
//   copy   : b[i]=a[i]                              -> 2*N*4 read + N*4 writeback
//   triad  : a[i]=b[i]+s*c[i]                       -> 3*N*4 read + N*4 writeback
//   gather : sum a[idx[i]] with random idx, stride>line -> N*4 lines pulled (64B each)
//
// build: gcc -O3 -march=native -fopenmp -o bwcal bwcal.c
// usage: ./bwcal <mode> <MiB> <reps> <threads>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <omp.h>

static double now(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

int main(int argc, char **argv) {
  const char *mode = argc > 1 ? argv[1] : "read";
  size_t mib = argc > 2 ? (size_t)atol(argv[2]) : 1024;
  int reps = argc > 3 ? atoi(argv[3]) : 5;
  int nthr = argc > 4 ? atoi(argv[4]) : omp_get_max_threads();
  omp_set_num_threads(nthr);

  size_t n = mib * 1024 * 1024 / sizeof(float);
  float *a = aligned_alloc(64, n * sizeof(float));
  float *b = aligned_alloc(64, n * sizeof(float));
  float *c = aligned_alloc(64, n * sizeof(float));
  int *idx = NULL;
  if (!a || !b || !c) { fprintf(stderr, "alloc failed\n"); return 1; }
#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < n; i++) { a[i] = 1.0f; b[i] = 2.0f; c[i] = 3.0f; }

  size_t ngather = n / 16;  // one float per 64B line
  if (!strcmp(mode, "gather")) {
    idx = aligned_alloc(64, ngather * sizeof(int));
    unsigned long s = 88172645463325252UL;
    for (size_t i = 0; i < ngather; i++) {
      s ^= s << 13; s ^= s >> 7; s ^= s << 17;
      idx[i] = (int)((s % (n / 16)) * 16);
    }
  }

  double best = 1e30, vol = 0;
  volatile double sink = 0;
  for (int r = 0; r < reps; r++) {
    double t0 = now();
    if (!strcmp(mode, "read")) {
      double s = 0;
#pragma omp parallel for schedule(static) reduction(+ : s)
      for (size_t i = 0; i < n; i++) s += a[i];
      sink += s; vol = (double)n * 4;
    } else if (!strcmp(mode, "write")) {
#pragma omp parallel for schedule(static)
      for (size_t i = 0; i < n; i++) a[i] = 1.5f;
      vol = (double)n * 8;  // RFO fill + writeback
    } else if (!strcmp(mode, "copy")) {
#pragma omp parallel for schedule(static)
      for (size_t i = 0; i < n; i++) b[i] = a[i];
      vol = (double)n * 12;
    } else if (!strcmp(mode, "triad")) {
#pragma omp parallel for schedule(static)
      for (size_t i = 0; i < n; i++) a[i] = b[i] + 1.5f * c[i];
      vol = (double)n * 16;
    } else if (!strcmp(mode, "gather")) {
      double s = 0;
#pragma omp parallel for schedule(static) reduction(+ : s)
      for (size_t i = 0; i < ngather; i++) s += a[idx[i]];
      sink += s; vol = (double)ngather * 64;
    } else { fprintf(stderr, "bad mode\n"); return 1; }
    double dt = now() - t0;
    if (dt < best) best = dt;
  }
  // Whole-process expected DRAM traffic, so a `perf stat` over the whole run can be
  // divided by it: the three init arrays are written once (RFO fill + writeback).
  double init_bytes = 3.0 * (double)n * 8.0;
  printf("mode=%s MiB=%zu threads=%d reps=%d time_ms=%.3f compulsory_bytes=%.0f "
         "total_expected_bytes=%.0f GB/s=%.2f sink=%g\n",
         mode, mib, nthr, reps, best * 1e3, vol, init_bytes + reps * vol,
         vol / best / 1e9, (double)sink);
  return 0;
}
