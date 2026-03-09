"""Run SDDMM benchmark on larger problems."""
import sys, time, statistics, signal
sys.path.insert(0, 'bench')
from _utils import download_matrix, scipy_to_torch, to_scorch_coo, to_scorch_dense, suppress_torch_warnings
import torch, scorch

suppress_torch_warnings()
torch.manual_seed(42)

WARMUP = 2
REPEATS = 5

benchmarks = [
    # (ssid, group, name, k)
    # Warmup: small problem to compile kernel first
    (6,   'HB',     'arc130',   128),  # 130x130, nnz=1.3K
    # Scaling up
    (39,  'HB',     'bcsstk17', 512),  # 10974x10974, nnz=429K
    (39,  'HB',     'bcsstk17', 1024), # 10974x10974, nnz=429K
    (39,  'HB',     'bcsstk17', 2048), # 10974x10974, nnz=429K
    (39,  'HB',     'bcsstk17', 4096), # 10974x10974, nnz=429K
    # (52,  'HB',     'bcsstk30', 512),  # 28924x28924, nnz=2M
    # (356, 'Boeing', 'ct20stif', 128),  # 52329x52329, nnz=2.7M
    # (537, 'Gupta',  'gupta2',   64),   # 62064x62064, nnz=4.2M
]

print("%-12s %5s %10s %12s %12s %10s %8s" % (
    "Matrix", "k", "NNZ", "PyTorch(ms)", "Scorch(ms)", "Speedup", "Correct"))
print("-" * 75)

for ssid, group, name, k in benchmarks:
    csr = download_matrix(ssid, group, name)
    n_rows, n_cols = csr.shape
    nnz = csr.nnz

    dense_A = torch.rand(n_rows, k, dtype=torch.float32)
    dense_B = torch.rand(k, n_cols, dtype=torch.float32)
    torch_coo = scipy_to_torch(csr, fmt='coo')

    # --- PyTorch ---
    for _ in range(WARMUP):
        torch.mul(torch_coo, torch.matmul(dense_A, dense_B))
    pt_times = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        ref = torch.mul(torch_coo, torch.matmul(dense_A, dense_B))
        pt_times.append(time.perf_counter() - t0)
    pt_med = statistics.median(pt_times) * 1e3

    # --- Scorch ---
    s_st = to_scorch_coo(torch_coo, 'S')
    a_st = to_scorch_dense(dense_A, 'A')
    b_st = to_scorch_dense(dense_B, 'B')
    for _ in range(WARMUP):
        scorch.einsum('ij,ik,kj->ij', s_st, a_st, b_st)
    sc_times = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        r = scorch.einsum('ij,ik,kj->ij', s_st, a_st, b_st)
        sc_times.append(time.perf_counter() - t0)
    sc_med = statistics.median(sc_times) * 1e3

    # Correctness
    ref_dense = ref.to_dense() if ref.is_sparse else ref
    sc_dense = r.to_torch()
    if sc_dense.is_sparse or sc_dense.is_sparse_csr:
        sc_dense = sc_dense.to_dense()
    correct = torch.allclose(sc_dense, ref_dense, atol=1e-2, rtol=1e-2)

    ratio = pt_med / sc_med if sc_med > 0 else float('inf')
    print("%-12s %5d %10s %12.1f %12.1f %9.1fx %8s" % (
        name, k, "%d" % nnz, pt_med, sc_med, ratio, correct))
