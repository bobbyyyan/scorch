#!/usr/bin/env python3
"""End-to-end forward passes through the PRODUCTION path (scorch.matmul) vs torch.sparse.mm (MKL).

GCN (Kipf & Welling, 2 layers) on the cached normalized adjacencies, and the sparse autoencoder
shapes of bench_sparse_autoencoder.py with uniformly pruned weights. Both interleave the SpMM
with dense GEMMs and elementwise ops -- the composition a single-op screen cannot see.

usage: workload_pipe.py <gcn_npz_dir> [repeats]
"""
import os, sys, time, numpy as np, torch
import scorch
from scorch import STensor
gdir = sys.argv[1]; REP = int(sys.argv[2]) if len(sys.argv) > 2 else 30
torch.manual_seed(0)
def bench(fn):
    for _ in range(5): fn()
    ts = []
    for _ in range(REP):
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
    ts.sort(); return ts[len(ts) // 2] * 1e6
rows = []
GCN = {"cora": (1433, 16, 7), "citeseer": (3703, 16, 6), "pubmed": (500, 16, 3)}
with torch.no_grad():
    for g, (fin, hid, ncls) in GCN.items():
        z = np.load(os.path.join(gdir, g + ".npz")); n = int(z["shape"][0])
        A = torch.sparse_csr_tensor(torch.from_numpy(z["indptr"].astype(np.int64)), torch.from_numpy(z["indices"].astype(np.int64)),
                                    torch.from_numpy(z["data"].astype(np.float32)), size=(n, n))
        Ast = STensor.from_csr(A, "A")
        X = torch.rand(n, fin); W1 = torch.rand(fin, hid); b1 = torch.rand(hid); W2 = torch.rand(hid, ncls); b2 = torch.rand(ncls)
        def mkl():
            h = torch.relu(torch.sparse.mm(A, X @ W1) + b1); return torch.sparse.mm(A, h @ W2) + b2
        def sc():
            h = torch.relu(scorch.matmul(Ast, X @ W1, format="dd") + b1); return scorch.matmul(Ast, h @ W2, format="dd") + b2
        rel = float((mkl() - sc()).norm() / mkl().norm()); assert rel < 1e-5, ("gcn rel err", g, rel)
        rows.append(("gcn", g, n, int(z["indptr"][-1]), bench(sc), bench(mkl)))
    for name, din, h1, h2 in [("mnist", 784, 1024, 512), ("mnist_big", 784, 4096, 2048), ("svhn", 3072, 2048, 1024)]:
        for sp in (0.9, 0.99):
            dims = [din, h1, h2, h1, din]; Ws, Sts, bs = [], [], []
            for i in range(4):
                Wd = torch.randn(dims[i + 1], dims[i]) / ((1 - sp) * dims[i]) ** 0.5; Wd[torch.rand_like(Wd) < sp] = 0
                Wc = Wd.to_sparse_csr(); Ws.append(Wc); Sts.append(STensor.from_csr(Wc, "W%d" % i)); bs.append(torch.rand(dims[i + 1]))
            x = torch.rand(256, din)
            acts = [torch.relu, torch.relu, torch.relu, torch.sigmoid]
            def mkl():
                h = x
                for W, b, act in zip(Ws, bs, acts): h = act(torch.sparse.mm(W, h.T).T + b)
                return h
            def sc():
                h = x
                for W, b, act in zip(Sts, bs, acts): h = act(scorch.matmul(W, h.T.contiguous(), format="dd").T + b)
                return h
            h = x                                     # per-layer relative error against MKL
            for W, St, b in zip(Ws, Sts, bs):
                ref = torch.sparse.mm(W, h.T.contiguous()); got = scorch.matmul(St, h.T.contiguous(), format="dd")
                rel = float((got - ref).norm() / ref.norm()); assert rel < 1e-5, ("layer rel err", name, sp, rel)
                h = torch.relu(ref.T + b)
            rows.append(("ae", "%s@%.2f" % (name, sp), dims[1], sum(int(W._nnz()) for W in Ws), bench(sc), bench(mkl)))
print("%-4s %-16s %8s %10s %12s %12s %8s" % ("kind", "case", "rows", "nnz", "scorch_us", "mkl_us", "mkl/sc"))
for r in rows:
    print("%-4s %-16s %8d %10d %12.1f %12.1f %8.3f" % (r[0], r[1], r[2], r[3], r[4], r[5], r[5] / r[4]))
