#!/usr/bin/env python3
"""Dump a benchmark matrix to a flat binary the standalone kernel harness can read.

layout: int64 M, int64 J, int64 nnz, int32 indptr[M+1], int32 indices[nnz],
        float32 values[nnz]
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import phase0_attrib as P  # noqa: E402


def main():
    out_dir = sys.argv[1]
    os.makedirs(out_dir, exist_ok=True)
    for spec in sys.argv[2:]:
        m = P.load_matrix(spec).tocsr()
        m.sort_indices()
        path = os.path.join(out_dir, spec.replace(":", "__") + ".bin")
        with open(path + ".tmp", "wb") as f:
            np.array([m.shape[0], m.shape[1], m.nnz], dtype=np.int64).tofile(f)
            m.indptr.astype(np.int32).tofile(f)
            m.indices.astype(np.int32).tofile(f)
            m.data.astype(np.float32).tofile(f)
        os.replace(path + ".tmp", path)
        print(f"{spec}: M={m.shape[0]} J={m.shape[1]} nnz={m.nnz} -> {path}")


if __name__ == "__main__":
    main()
