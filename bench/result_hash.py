#!/usr/bin/env python3
"""Print a hash of scorch.matmul's output plus its error against a float64 reference.

Run once per tree; identical hashes mean the change did not perturb a single output
bit. The float64 comparison is what says the shared answer is also the right one.
"""
import hashlib
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import phase0_attrib as P  # noqa: E402
import scorch  # noqa: E402

matrix, n, level = sys.argv[1], int(sys.argv[2]), sys.argv[3]
csr = P.load_matrix(matrix)
J = csr.shape[1]
rng = np.random.default_rng(0)
B = torch.from_numpy(rng.standard_normal((J, n), dtype=np.float32))
A = P.H.to_st(csr)
scorch.set_autotune(level)
out = scorch.matmul(A, B).contiguous()
arr = out.numpy()
h = hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()[:32]

# float64 reference on a bounded row sample
M = csr.shape[0]
rows = np.unique(np.linspace(0, M - 1, min(M, 256)).astype(np.int64))
sub = csr[rows].astype(np.float64)
ref = sub @ B.numpy().astype(np.float64)
got = arr[rows].astype(np.float64)
den = max(np.abs(ref).max(), 1e-30)
print(f"HASH {matrix} N={n} {level} sha={h} shape={arr.shape} "
      f"relerr={np.abs(got - ref).max() / den:.3e}")
