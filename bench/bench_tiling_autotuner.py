#!/usr/bin/env python3
r"""bench_tiling_autotuner.py — an ADAPTIVE SpMM tiling selector for ANY workload.

Unifies the whole tiling space behind one cost model:
    none | tile-i | tile-j | tile-ik | tile-ijk(+B width-panel relayout)

For each (matrix, N):
  * ORACLE  : measure every candidate config, report the best (ground truth).
  * MODEL   : predict the best (schedule, tile sizes) from cheap structural
              features (M, J, nnz, degree, wavefront W*) + N + cache C, by
              estimating each schedule's DRAM traffic and taking the min.
  * validate: is the model's pick within a few % of the oracle's best?

Mechanism recap (derived + measured earlier):
  B working set is a rectangle W* live-rows TALL x N WIDE.  It overflows cache two
  ways, each fixed by shrinking a different side:
    - large N on a LOCAL matrix (small W*)      -> tile-ik  (shrink width Nc)
    - scattered access (large W*)               -> tile-j   (shrink live rows Jc)
    - scattered AND very wide N                 -> tile-ijk (shrink both; relaid B)
    - fits (W*.4N <= C)                         -> none / tile-i (scheduling only)
"""
from __future__ import annotations
import os, sys, time, math, statistics, csv as csvmod, platform, subprocess
# redwood-only HOME/MPLCONFIGDIR forcing (harmless if the dir is absent, e.g. M5)
if os.path.isdir("/scratch/bobbyy"):
    os.environ.setdefault("HOME", "/scratch/bobbyy")
    os.environ.setdefault("MPLCONFIGDIR", "/scratch/bobbyy/.mplcache")
import numpy as np, scipy.sparse, scipy.io, torch
os.environ["PATH"] = os.path.dirname(sys.executable) + os.pathsep + os.environ.get("PATH", "")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)


def _detect_llc():
    """Effective shared last-level cache in bytes. macOS = SLC (not exposed by
    sysctl -> env override / perflevel0 L2 fallback); Linux = L3."""
    env = os.environ.get("SCORCH_LLC_BYTES")
    if env:
        return int(env)
    if platform.system() == "Darwin":
        # Apple hides SLC; M5 Max SLC is large. Default 24MB (fit empirically in
        # Phase A); can override with SCORCH_LLC_BYTES.
        return 24 * 1024 * 1024
    # Linux: read L3 from sysfs, else 36MB (redwood i9-14900K)
    try:
        for idx in range(0, 8):
            base = f"/sys/devices/system/cpu/cpu0/cache/index{idx}"
            if os.path.isfile(base + "/level"):
                with open(base + "/level") as fh:
                    lvl = fh.read().strip()
                if lvl == "3":
                    with open(base + "/size") as fh:
                        s = fh.read().strip()
                    return int(s[:-1]) * 1024 if s.endswith("K") else int(s[:-1]) * 1024 * 1024
    except Exception:
        pass
    return 36 * 1024 * 1024


C_LLC = _detect_llc()
# threads: explicit + consistent so the schedule RANKING is fair. Default = every
# physical core (matches the E-core-recruit BW-bound production launch).
NT = int(os.environ.get("SCORCH_TILING_NT", "0")) or (
    os.cpu_count() if platform.system() == "Darwin" else 0)
# SuiteSparse roots to probe (redwood scratch, then M5 caches)
SS_ROOTS = ["/scratch/suitesparse",
            os.path.expanduser("~/.cache/scorch/ss_full_envelope"),
            os.path.expanduser("~/.cache/scorch_suitesparse")]

CPP = r"""
#include <torch/extension.h>
#include <vector>
#include <algorithm>
#include <omp.h>
static inline int et(int nt){ return nt>0?nt:omp_get_max_threads(); }
static inline void accum(int i,int k0,int kw,int N,const int*pos,const int*crd,
    const float*av,const float*bv,float*C){
  float* cr=C+(size_t)i*N+k0; for(int k=0;k<kw;++k) cr[k]=0.f;
  for(int p=pos[i];p<pos[i+1];++p){ float a=av[p]; const float*br=bv+(size_t)crd[p]*N+k0;
    if(p+1<pos[i+1]) __builtin_prefetch(bv+(size_t)crd[p+1]*N+k0,0,1);
    for(int k=0;k<kw;++k) cr[k]+=a*br[k]; } }
torch::Tensor spmm_none(std::vector<int> rs,torch::Tensor Ap,torch::Tensor Ac,
    torch::Tensor Av,torch::Tensor Bv,int N,int nt){
  const int M=rs[0]; const int*pos=Ap.data_ptr<int>();const int*crd=Ac.data_ptr<int>();
  const float*av=Av.data_ptr<float>();const float*bv=Bv.data_ptr<float>();
  torch::Tensor Ct=torch::empty({M,N},torch::kFloat32);float*C=Ct.data_ptr<float>();
  #pragma omp parallel for schedule(dynamic,64) num_threads(et(nt))
  for(int i=0;i<M;++i) accum(i,0,N,N,pos,crd,av,bv,C);
  return Ct;
}
torch::Tensor spmm_tile_i(std::vector<int> rs,torch::Tensor Ap,torch::Tensor Ac,
    torch::Tensor Av,torch::Tensor Bv,int N,int Ti,int nt){
  const int M=rs[0]; const int*pos=Ap.data_ptr<int>();const int*crd=Ac.data_ptr<int>();
  const float*av=Av.data_ptr<float>();const float*bv=Bv.data_ptr<float>();
  torch::Tensor Ct=torch::empty({M,N},torch::kFloat32);float*C=Ct.data_ptr<float>();
  const int nb=(M+Ti-1)/Ti;
  #pragma omp parallel for schedule(dynamic,1) num_threads(et(nt))
  for(int b=0;b<nb;++b){int i0=b*Ti,i1=std::min(i0+Ti,M);
    for(int i=i0;i<i1;++i) accum(i,0,N,N,pos,crd,av,bv,C);}
  return Ct;
}
torch::Tensor spmm_tile_ik(std::vector<int> rs,torch::Tensor Ap,torch::Tensor Ac,
    torch::Tensor Av,torch::Tensor Bv,int N,int Ti,int Nc,int nt){
  const int M=rs[0]; const int*pos=Ap.data_ptr<int>();const int*crd=Ac.data_ptr<int>();
  const float*av=Av.data_ptr<float>();const float*bv=Bv.data_ptr<float>();
  torch::Tensor Ct=torch::empty({M,N},torch::kFloat32);float*C=Ct.data_ptr<float>();
  const int nb=(M+Ti-1)/Ti;
  #pragma omp parallel for schedule(dynamic,1) num_threads(et(nt))
  for(int b=0;b<nb;++b){int i0=b*Ti,i1=std::min(i0+Ti,M);
    for(int k0=0;k0<N;k0+=Nc){int kw=std::min(Nc,N-k0);
      for(int i=i0;i<i1;++i) accum(i,k0,kw,N,pos,crd,av,bv,C);}}
  return Ct;
}
void spmm_accum(torch::Tensor Ct,std::vector<int> rs,torch::Tensor Ap,torch::Tensor Ac,
    torch::Tensor Av,torch::Tensor Bv,int N,int nt){
  const int M=rs[0]; const int*pos=Ap.data_ptr<int>();const int*crd=Ac.data_ptr<int>();
  const float*av=Av.data_ptr<float>();const float*bv=Bv.data_ptr<float>();float*C=Ct.data_ptr<float>();
  #pragma omp parallel for schedule(dynamic,64) num_threads(et(nt))
  for(int i=0;i<M;++i){int b=pos[i],e=pos[i+1];if(b==e)continue;float*cr=C+(size_t)i*N;
    for(int p=b;p<e;++p){float a=av[p];const float*br=bv+(size_t)crd[p]*N;
      if(p+1<e)__builtin_prefetch(bv+(size_t)crd[p+1]*N,0,1);
      for(int k=0;k<N;++k) cr[k]+=a*br[k];}}
}
void spmm_accum_relaid(torch::Tensor Cp,std::vector<int> rs,torch::Tensor Ap,torch::Tensor Ac,
    torch::Tensor Av,torch::Tensor Bp,int Nc,int nt){
  const int M=rs[0]; const int*pos=Ap.data_ptr<int>();const int*crd=Ac.data_ptr<int>();
  const float*av=Av.data_ptr<float>();const float*bp=Bp.data_ptr<float>();float*C=Cp.data_ptr<float>();
  #pragma omp parallel for schedule(dynamic,64) num_threads(et(nt))
  for(int i=0;i<M;++i){int b=pos[i],e=pos[i+1];if(b==e)continue;float*cr=C+(size_t)i*Nc;
    for(int p=b;p<e;++p){float a=av[p];const float*br=bp+(size_t)crd[p]*Nc;
      if(p+1<e)__builtin_prefetch(bp+(size_t)crd[p+1]*Nc,0,1);
      for(int k=0;k<Nc;++k) cr[k]+=a*br[k];}}
}
void write_strip(torch::Tensor Ct,torch::Tensor Cp,int N,int k0,int Nc,int nt){
  const int M=Ct.size(0); float*C=Ct.data_ptr<float>(); const float*cp=Cp.data_ptr<float>();
  #pragma omp parallel for schedule(static) num_threads(et(nt))
  for(int i=0;i<M;++i){float*dst=C+(size_t)i*N+k0;const float*src=cp+(size_t)i*Nc;
    for(int k=0;k<Nc;++k) dst[k]=src[k];}
}
"""


def build():
    from torch.utils.cpp_extension import load_inline
    # platform-aware OpenMP flags (macOS: -Xpreprocessor -fopenmp + torch libomp;
    # Linux: -fopenmp + bundled libgomp) — reuse scorch's own build helpers.
    from scorch.utils import get_extra_cflags, get_extra_ldflags
    cflags = get_extra_cflags(["-O3", "-march=native", "-ffast-math", "-funroll-loops"])
    print(f"compiling unified kernels... (cflags={cflags})", flush=True)
    return load_inline(name="tiling_autotuner", cpp_sources=[CPP],
        functions=["spmm_none", "spmm_tile_i", "spmm_tile_ik", "spmm_accum",
                   "spmm_accum_relaid", "write_strip"],
        extra_cflags=cflags, extra_ldflags=get_extra_ldflags(), verbose=False)


def timed(thunks, warmup=2, min_rounds=3, max_rounds=6, budget=2.0):
    import random
    K = len(thunks)
    for _ in range(warmup):
        for th in thunks: th()
    T = [[] for _ in range(K)]; order = list(range(K)); rng = random.Random(7); r = 0
    while r < max_rounds:
        rng.shuffle(order)
        for idx in order:
            t0 = time.perf_counter(); thunks[idx](); T[idx].append(time.perf_counter()-t0)
        r += 1
        if r >= min_rounds and sum(sum(t) for t in T) >= budget: break
    return [statistics.median(t) for t in T]


# ---- matrix zoo ----------------------------------------------------------
def m_band(M, bw, seed=0):
    rng = np.random.default_rng(seed); rows = []; cols = []
    for i in range(M):
        lo = max(0, i - bw); hi = min(M, i + bw + 1)
        c = np.arange(lo, hi); rows.append(np.full(c.size, i)); cols.append(c)
    r = np.concatenate(rows); c = np.concatenate(cols)
    d = rng.random(r.size, dtype=np.float32)
    x = scipy.sparse.csr_matrix((d, (r, c)), shape=(M, M)); x.sort_indices(); return x


def m_scatter(M, deg, seed=0):
    rng = np.random.default_rng(seed)
    indptr = np.arange(0, (M + 1) * deg, deg, dtype=np.int64)
    cols = rng.integers(0, M, size=M * deg, dtype=np.int64)
    data = rng.random(M * deg, dtype=np.float32)
    x = scipy.sparse.csr_matrix((data, cols, indptr), shape=(M, M))
    x.sum_duplicates(); x.sort_indices(); return x


def m_suitesparse(name):
    for root in SS_ROOTS:
        for p in (os.path.join(root, name, name + ".mtx"),
                  os.path.join(root, name + ".mtx")):
            if os.path.exists(p):
                x = scipy.sparse.csr_matrix(scipy.io.mmread(p), dtype=np.float32)
                x.sort_indices(); return x
    return None


def m_gcn(name):
    import bench_gcn as G
    ds = G.load_dataset(name); a = G.compute_normalized_adj(ds.edge_index, ds.num_nodes)[0]
    x = scipy.sparse.csr_matrix((a.values().numpy().astype(np.float32), a.col_indices().numpy(),
        a.crow_indices().numpy().astype(np.int64)), shape=(ds.num_nodes, ds.num_nodes))
    x.sort_indices(); return x


def wavefront(csr):
    M, J = csr.shape; csc = csr.tocsc(); ind, ptr = csc.indices, csc.indptr
    nz = np.diff(ptr) > 0
    lo = ind[ptr[:-1][nz]]; hi = ind[ptr[1:][nz] - 1]
    d = np.zeros(M + 1, dtype=np.int64); np.add.at(d, lo, 1); np.add.at(d, hi + 1, -1)
    return int(np.cumsum(d[:-1]).max())


def make_panels(csr, Jc):
    J = csr.shape[1]; csc = csr.tocsc(); out = []
    for j0 in range(0, J, Jc):
        s = csc[:, j0:j0 + Jc].tocsr(); s.sort_indices()
        out.append((torch.from_numpy(s.indptr.astype(np.int32)),
                    torch.from_numpy((s.indices + j0).astype(np.int32)),
                    torch.from_numpy(s.data.astype(np.float32))))
    return out


# ---- COST MODEL: predicted DRAM bytes per schedule -----------------------
def model_costs(M, J, nnz, Wstar, N, C, relayout_amortized=True):
    """Return dict schedule -> (bytes, params). Lower bytes = predicted faster."""
    BN = 4.0 * N
    deg = nnz / J
    f = max(0.0, 1.0 - C / (Wstar * BN))          # envelope overflow fraction
    A = 8.0 * nnz
    Cwr = M * BN
    out = {}
    # none / tile-i : no re-traffic; B floor J*4N if it fits, else thrash
    b_none = (J + f * (nnz - J)) * BN
    out["none"] = (b_none + Cwr + A, {})
    out["tile-i"] = (b_none + Cwr + A + M * 4, {"Ti": 256})   # ~none + tiny sched
    # tile-j : Jc = C/4N ; C re-traffic P times
    Jc = max(1.0, C / BN); P = math.ceil(J / Jc)
    out["tile-j"] = (J * BN + P * 2 * Cwr + A + P * M * 4, {"Jc": int(min(J, Jc))})
    # tile-ik : row-block holds band's Nc-panel; needs band (W*) to fit at some Nc
    #   Nc_ik = largest Nc with W*.4Nc <= C ; invalid if even Nc=1 doesn't fit
    Nc_ik = int(C / (4.0 * Wstar)) if Wstar > 0 else N
    if Nc_ik >= 1:
        Nc = min(N, max(1, Nc_ik)); nk = math.ceil(N / Nc)
        out["tile-ik"] = (J * BN + Cwr + A * nk, {"Ti": 64, "Nc": Nc})
    # tile-ijk (relaid) : Nc bounded by Cp (M) + panel; A re-scanned nk times
    Nc_ijk = max(1, int(C / (4.0 * (M + min(J, C / (4.0 * 64))))))
    Nc = min(N, Nc_ijk); nk = math.ceil(N / Nc)
    Jc2 = min(J, max(256, int(C / (4.0 * max(1, Nc)))))
    relay = 0.0 if relayout_amortized else 2 * J * BN
    out["tile-ijk"] = (J * BN + Cwr + A * nk + relay, {"Nc": Nc, "Jc": int(Jc2)})
    return out


def model_pick(M, J, nnz, Wstar, N, C):
    costs = model_costs(M, J, nnz, Wstar, N, C)
    sched = min(costs, key=lambda k: costs[k][0])
    return sched, costs[sched][1], costs


# ---- oracle measurement --------------------------------------------------
def eval_matrix(mod, csr, N, Wstar, nt=None):
    if nt is None: nt = NT
    M, J = csr.shape; nnz = csr.nnz; rs = [M, J]
    Ap = torch.from_numpy(csr.indptr.astype(np.int32)); Ac = torch.from_numpy(csr.indices.astype(np.int32))
    Av = torch.from_numpy(csr.data.astype(np.float32))
    torch.manual_seed(0); B = torch.rand(J, N, dtype=torch.float32)
    ref = torch.from_numpy(csr.astype(np.float64) @ B.double().numpy()); rn = ref.norm().item() + 1e-30
    flops = 2.0 * nnz * N
    cfgs = []  # (schedule, label, thunk)
    cfgs.append(("none", "none", lambda: mod.spmm_none(rs, Ap, Ac, Av, B, N, nt)))
    for Ti in (64, 256):
        cfgs.append(("tile-i", f"i{Ti}", lambda Ti=Ti: mod.spmm_tile_i(rs, Ap, Ac, Av, B, N, Ti, nt)))
    for Ti, Nc in [(16, 64), (64, 128), (256, 256)]:
        if Nc > N: continue
        cfgs.append(("tile-ik", f"ik{Ti},{Nc}", lambda Ti=Ti, Nc=Nc: mod.spmm_tile_ik(rs, Ap, Ac, Av, B, N, Ti, Nc, nt)))
    pcache = {}
    def panels(Jc):
        if Jc not in pcache: pcache[Jc] = make_panels(csr, Jc)
        return pcache[Jc]
    base_jc = max(256, int(C_LLC / (4 * N)))
    jc_set = sorted({max(256, base_jc // 4), max(256, base_jc // 2), base_jc,
                     base_jc * 2, base_jc * 4})
    for Jc in jc_set:
        if Jc >= J: continue
        ps = panels(Jc); Cb = torch.empty(M, N, dtype=torch.float32)
        def run(ps=ps, Cb=Cb):
            Cb.zero_()
            for (a, b, c) in ps: mod.spmm_accum(Cb, rs, a, b, c, B, N, nt)
            return Cb
        cfgs.append(("tile-j", f"j{Jc}", run))
    for Nc in [64, 128, 256]:
        if Nc > N: continue
        Jc = min(J, max(256, int(C_LLC / (4 * Nc))))
        ps = panels(Jc); nk = (N + Nc - 1) // Nc
        Bp = [B[:, p * Nc:min((p + 1) * Nc, N)].contiguous() for p in range(nk)]
        Cb = torch.empty(M, N, dtype=torch.float32); Cp = torch.empty(M, Nc, dtype=torch.float32)
        def run(ps=ps, Bp=Bp, Cb=Cb, Cp=Cp, Nc=Nc, nk=nk):
            for p in range(nk):
                kw = min(Nc, N - p * Nc); cp = Cp if kw == Nc else Cp[:, :kw].contiguous(); cp.zero_()
                for (a, b, c) in ps: mod.spmm_accum_relaid(cp, rs, a, b, c, Bp[p], kw, nt)
                mod.write_strip(Cb, cp, N, p * Nc, kw, nt)
            return Cb
        cfgs.append(("tile-ijk", f"ijk{Nc},{Jc}", run))

    worst = max((th().double() - ref).norm().item() / rn for _, _, th in cfgs)
    meds = timed([c[2] for c in cfgs]); gf = [flops / t / 1e9 for t in meds]
    # best per schedule + overall oracle
    per = {}
    for (s, lab, _), g in zip(cfgs, gf):
        if s not in per or g > per[s][0]: per[s] = (g, lab)
    oidx = int(np.argmax(gf)); oracle = (gf[oidx], cfgs[oidx][0], cfgs[oidx][1])
    return per, oracle, worst, (M, J, nnz)


def main():
    print(f"[cfg] platform={platform.system()} NT={NT or 'omp_max'} "
          f"C_LLC={C_LLC/1024/1024:.0f}MB", flush=True)
    mod = build()
    grid = []  # (name, csr_builder, [Ns])
    NARROW = [64, 256, 1024]
    WIDE = [64, 256, 1024, 4096, 16384]
    grid += [
        ("band_bw16 (well-ordered)", lambda: m_band(40000, 16), WIDE),
        ("scatter_deg16 (low-deg)", lambda: m_scatter(40000, 16), WIDE),
        ("scatter_deg200 (high-deg)", lambda: m_scatter(30000, 200), WIDE),
        ("cant (FEM)", lambda: m_suitesparse("cant"), WIDE),
        ("ogbn-arxiv (real graph)", lambda: m_gcn("ogbn-arxiv"), NARROW),
        ("reddit (real graph)", lambda: m_gcn("reddit"), [64, 256, 1024]),
    ]
    rows = []
    agree = 0; tot = 0; ratios = []
    for name, builder, Ns in grid:
        try:
            csr = builder()
        except Exception as ex:
            print(f"[skip] {name}: {type(ex).__name__}: {ex}", flush=True); continue
        if csr is None:
            print(f"[skip] {name}", flush=True); continue
        M, J = csr.shape; nnz = csr.nnz; Wstar = wavefront(csr)
        print(f"\n########## {name}  M={M} J={J} nnz={nnz} ({nnz/M:.0f}/row)  W*={Wstar} (W*/J={Wstar/J:.2f}) ##########", flush=True)
        for N in Ns:
            per, oracle, worst, _ = eval_matrix(mod, csr, N, Wstar)
            psched, pparams, costs = model_pick(M, J, nnz, Wstar, N, C_LLC)
            og, os_, olab = oracle
            pg = per.get(psched, (float("nan"), "-"))[0]  # measured gflops of model's pick
            ratio = pg / og if og else float("nan")
            ok = (psched == os_) or (ratio >= 0.97)
            agree += ok; tot += 1; ratios.append(ratio)
            allfam = "  ".join(f"{s}={per[s][0]:.0f}" for s in ["none", "tile-i", "tile-ik", "tile-j", "tile-ijk"] if s in per)
            print(f"  N={N:5d}  ORACLE={os_}({olab}) {og:6.1f}  |  MODEL={psched}{pparams} -> {pg:6.1f} ({ratio:.2f}x oracle) {'OK' if ok else 'MISS'}", flush=True)
            print(f"           measured: {allfam}   relerr={worst:.0e}", flush=True)
            rows.append(dict(matrix=name, M=M, J=J, nnz=nnz, Wstar=Wstar, N=N,
                             oracle_sched=os_, oracle_gflops=og,
                             model_sched=psched, model_gflops=pg, ratio=ratio, ok=int(ok)))
    gm = math.exp(sum(math.log(max(1e-9, r)) for r in ratios) / len(ratios)) if ratios else float("nan")
    print(f"\n===== SELECTOR VALIDATION =====", flush=True)
    print(f"  agreement (same family or >=0.97x oracle): {agree}/{tot}", flush=True)
    print(f"  geomean(model_pick / oracle_best throughput): {gm:.3f}", flush=True)
    tag = platform.system().lower()
    outdir = os.path.join(REPO, "bench", "bench_results")
    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, f"tiling_autotuner_{tag}.csv")
    with open(out, "w", newline="") as f:
        w = csvmod.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"  wrote {out}", flush=True)


if __name__ == "__main__":
    main()
