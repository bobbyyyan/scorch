#!/usr/bin/env python3
"""One line per (corpus, key, k): the SHA-256 of the drop-in SpMM's output bytes, through the
production launch path, with a fixed B. Run it under two builds and `diff` the files: a change
claimed to be bit-identical must produce no diff, and the line count shows how many cells the
claim covers.

SCREEN_DTYPE=float64 runs double.

usage: output_hashes.py <cache_dir> <sample_dir> <out_txt> [ks]
"""
import hashlib, os, sys, zlib
import numpy as np, torch, scorch_ops as so
sys.path.insert(0, "/matx/u/bobbyy/pfsstage")
from csrresolve import resolve_all
DT = {"float32": (torch.float32, np.float32, so.spmm_csr_float_v2),
      "float64": (torch.float64, np.float64, so.spmm_csr_double_v2)}[os.environ.get("SCREEN_DTYPE", "float32")]
cache, sdir, out = sys.argv[1:4]; KS = [int(x) for x in (sys.argv[4] if len(sys.argv) > 4 else "1").split(",")]
P = torch.get_num_threads()
with open(out, "w") as f:
    for corpus in ("ss", "dlmc"):
        keys = [l.strip() for l in open(os.path.join(sdir, "screen_%s.txt" % corpus)) if l.strip()]
        paths, _, _ = resolve_all(os.path.join(cache, corpus), keys)
        for key in keys:
            z = np.load(paths[key]); M, J = int(z["shape"][0]), int(z["shape"][1])
            ipt = torch.from_numpy(z["indptr"].astype(np.int32)); ixt = torch.from_numpy(z["indices"].astype(np.int32))
            vt = torch.from_numpy(z["data"].astype(DT[1]))
            for k in KS:
                g = torch.Generator().manual_seed(zlib.crc32(("%s|%d" % (key, k)).encode()) % (2**31))
                B = torch.rand((J, k), generator=g, dtype=DT[0]).reshape(-1).contiguous()
                o = DT[2]([M, k], [M, J], [[], [ipt, ixt]], vt, [J, k], [[], []], B, 256, P, True)
                f.write("%s %s %d %d %s\n" % (corpus, key, k, int(z["indptr"][-1]) // max(M, 1), hashlib.sha256(o.storage.value.numpy().tobytes()).hexdigest()[:20]))
