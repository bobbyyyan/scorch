#!/usr/bin/env python3
"""Per (corpus, key, k): how far the drop-in SpMM's output is from the exact product, in units of
the standard rounding bound. For an entry c_ij = sum_p a_ip b_pj computed in any order, the
rounding error is at most about deg * eps * (|A||B|)_ij, so the ratio |c - c_exact| / (eps *
(|A||B|)_ij) measures how much of that budget an ordering used. For float32 output the exact
product is computed in float64, whose own error is far below float32's eps, so the ratio is read
against the bound. For float64 output the float64 reference carries error of the same order as the
output, so that column is read only as a comparison BETWEEN builds, never against the bound.

Run under two builds and compare: a change that only reorders a sum leaves both builds inside the
bound with the same distribution.

SCREEN_DTYPE=float64 runs double. usage: output_error.py <cache_dir> <sample_dir> <out_csv> [ks]
"""
import csv, os, sys, zlib
import numpy as np, torch, scorch_ops as so
sys.path.insert(0, "/matx/u/bobbyy/pfsstage")
from csrresolve import resolve_all
DT = {"float32": (torch.float32, np.float32, so.spmm_csr_float_v2),
      "float64": (torch.float64, np.float64, so.spmm_csr_double_v2)}[os.environ.get("SCREEN_DTYPE", "float32")]
cache, sdir, out = sys.argv[1:4]; KS = [int(x) for x in (sys.argv[4] if len(sys.argv) > 4 else "1").split(",")]
P = torch.get_num_threads(); EPS = float(torch.finfo(DT[0]).eps)
w = csv.writer(open(out, "w")); w.writerow(["corpus", "key", "k", "deg", "max_ratio", "mean_ratio", "max_rel"])
for corpus in ("ss", "dlmc"):
    keys = [l.strip() for l in open(os.path.join(sdir, "screen_%s.txt" % corpus)) if l.strip()]
    paths, _, _ = resolve_all(os.path.join(cache, corpus), keys)
    for key in keys:
        z = np.load(paths[key]); M, J = int(z["shape"][0]), int(z["shape"][1])
        ipt = torch.from_numpy(z["indptr"].astype(np.int32)); ixt = torch.from_numpy(z["indices"].astype(np.int32))
        vt = torch.from_numpy(z["data"].astype(DT[1]))
        A64 = torch.sparse_csr_tensor(ipt.long(), ixt.long(), vt.double(), size=(M, J))
        Aabs = torch.sparse_csr_tensor(ipt.long(), ixt.long(), vt.double().abs(), size=(M, J))
        for k in KS:
            g = torch.Generator().manual_seed(zlib.crc32(("%s|%d" % (key, k)).encode()) % (2**31))
            B = torch.rand((J, k), generator=g, dtype=DT[0])
            o = DT[2]([M, k], [M, J], [[], [ipt, ixt]], vt, [J, k], [[], []], B.reshape(-1).contiguous(), 256, P, True)
            c = o.storage.value.view(M, k).double()
            ref = torch.sparse.mm(A64, B.double()); bound = torch.sparse.mm(Aabs, B.double().abs()) * EPS
            err = (c - ref).abs(); nz = bound > 0
            ratio = err[nz] / bound[nz]
            rel = (err / ref.abs().clamp_min(1e-300))[nz]
            w.writerow([corpus, key, k, int(z["indptr"][-1]) // max(M, 1), "%.4f" % float(ratio.max()) if ratio.numel() else 0,
                        "%.5f" % float(ratio.mean()) if ratio.numel() else 0, "%.3e" % float(rel.max()) if rel.numel() else 0])
