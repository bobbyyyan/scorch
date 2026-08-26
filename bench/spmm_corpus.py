"""Matrix corpus for the mass SpMM sweep: SuiteSparse ``.mtx``, DLMC ``.smtx``, and
the canonical CSR cache both convert into.

Parsing Matrix Market text is the dominant cost of a whole-collection sweep -- 41 GB
of it for the SuiteSparse matrices this sweep can afford -- and paying it once per k
value, per arm, or per host is pure waste. So every source converts once into an
uncompressed ``.npz`` holding int32 ``indptr``/``indices`` and float32 ``data``, and
the timing stage only ever opens the cache. int32 is safe because the sweep caps nnz
well below 2^31; ``convert`` refuses anything that would overflow rather than
silently truncating.

The two formats need different handling and neither is quite what its documentation
says:

*SuiteSparse ``.mtx``* stores a symmetric matrix as one triangle, so the nnz in the
header is not the nnz you compute with; ``scipy.io.mmread`` expands it, and the
manifest scan therefore doubles the header count to predict cost. ``pattern``
matrices carry no values and read back as all-ones, which is fine for timing and
fine for a correctness check.

The scan also flags each SuiteSparse file as ``primary`` or not. A tarball contains
``Group/Name/Name.mtx`` -- the matrix -- alongside files like ``Name_b.mtx``, which are
right-hand-side *vectors*: shapes like 1447360x1 holding 440 nonzeros. They are real
files in the collection and worth timing, but pooling them with the matrices would
report a library's behaviour on operators using measurements of its behaviour on
vectors, so they are labelled rather than dropped.

*DLMC ``.smtx``* is documented as "each line contains the column indices for a row",
which is wrong: it is three lines -- ``rows, cols, nnz``, then ``rows+1`` row
offsets, then ``nnz`` column indices. It carries no values at all (it is a pruning
*mask* collection), so this module fills them from a per-matrix seeded generator.
Values that are all 1.0 would make every arm's output identical under any summation
order and hide a real ordering bug in the correctness check.
"""

import csv
import os
import sys

import numpy as np
import scipy.io
import scipy.sparse

# int32 index arrays are what both MKL's fast path and scorch's kernels want; a
# matrix that cannot be expressed in them is skipped with a reason, not truncated.
INT32_MAX = 2**31 - 1


def canonical(csr, dtype=np.float32):
    """CSR with sorted indices, no duplicate or explicitly-stored-zero entries.

    Every arm is handed the same array, so canonicalizing here is what makes the
    comparison about kernels rather than about which arm tolerated a stray duplicate.
    """
    csr = scipy.sparse.csr_matrix(csr, dtype=dtype)
    csr.sum_duplicates()
    csr.eliminate_zeros()
    csr.sort_indices()
    return csr


def read_mtx(path):
    """Load a SuiteSparse ``.mtx``. Returns (csr, None) or (None, reason)."""
    with open(path, "rb") as f:
        banner = f.readline().decode("utf-8", "replace").lower().split()
    if len(banner) > 2 and banner[2] != "coordinate":
        return None, "array (dense) matrix market file, not a sparse matrix"
    if len(banner) > 3 and banner[3] == "complex":
        return None, "complex field"
    m = scipy.io.mmread(path)
    if not scipy.sparse.issparse(m):
        return None, "mmread returned a dense array"
    return canonical(m), None


def read_smtx(path, seed):
    """Load a DLMC ``.smtx``. Returns (csr, None) or (None, reason).

    Values are drawn from a generator seeded per matrix so the same file gives the
    same numbers on every host and in every pass -- a sweep that reseeded per run
    could not tell a real cross-host difference from a different draw.
    """
    with open(path, "rb") as f:
        hdr = f.readline().decode()
        rows, cols, nnz = (int(x) for x in hdr.replace(",", " ").split()[:3])
        indptr = np.fromstring(f.readline().decode(), dtype=np.int64, sep=" ")
        indices = np.fromstring(f.readline().decode(), dtype=np.int64, sep=" ")
    if indptr.size != rows + 1:
        return None, f"row-offset line has {indptr.size} ints, expected {rows + 1}"
    if indices.size != nnz:
        return None, f"index line has {indices.size} ints, expected {nnz}"
    rng = np.random.default_rng(seed)
    data = rng.random(nnz, dtype=np.float32) + 0.5
    m = scipy.sparse.csr_matrix((data, indices, indptr), shape=(rows, cols))
    return canonical(m), None


def save_cache(path, csr):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    np.savez(
        tmp,
        indptr=csr.indptr.astype(np.int32),
        indices=csr.indices.astype(np.int32),
        data=csr.data.astype(np.float32),
        shape=np.asarray(csr.shape, dtype=np.int64),
    )
    os.replace(tmp + ".npz", path)


def load_cache(path):
    z = np.load(path)
    return scipy.sparse.csr_matrix(
        (z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"])
    )


def convert(src, kind, dest, nnz_cap, seed=0):
    """Read `src`, canonicalize, write `dest`. Returns (csr, reason_or_None)."""
    if kind == "mtx":
        csr, why = read_mtx(src)
    elif kind == "smtx":
        csr, why = read_smtx(src, seed)
    else:
        return None, f"unknown kind {kind!r}"
    if csr is None:
        return None, why
    if csr.nnz == 0:
        return None, "empty after canonicalization"
    if csr.nnz > nnz_cap:
        return None, f"nnz {csr.nnz} over cap {nnz_cap:.0f}"
    if csr.nnz > INT32_MAX or max(csr.shape) > INT32_MAX:
        return None, "does not fit int32 indices"
    save_cache(dest, csr)
    return csr, None


def row_stats(csr):
    """(mean, std, max, empty) nonzeros per row -- the regime axes the sweep groups by.

    `empty` is the number of all-zero rows. It matters because the output of an SpMM
    is O(rows * k) whether or not a row has any nonzeros, so a matrix that is mostly
    empty rows is an output-writing problem wearing a sparse matrix's clothes.
    """
    rl = np.diff(csr.indptr).astype(np.int64)
    if rl.size == 0:
        return 0.0, 0.0, 0, 0
    return float(rl.mean()), float(rl.std()), int(rl.max()), int((rl == 0).sum())


# --------------------------------------------------------------------------- #
# manifests
# --------------------------------------------------------------------------- #
def scan_suitesparse(root):
    """Every ``.mtx`` under `root`, with its header's dims and a predicted nnz.

    Only the first few hundred bytes of each file are read. The predicted nnz
    doubles the header count for symmetric/hermitian storage because that is what
    ``mmread`` will expand it to, and the sweep's cost model has to see the expanded
    size to schedule honestly.
    """
    rows = []
    dropped = []
    for d, _, fs in os.walk(root):
        for f in sorted(fs):
            if not f.endswith(".mtx"):
                continue
            p = os.path.join(d, f)
            try:
                with open(p, "rb") as fh:
                    head = fh.read(4096).decode("utf-8", "replace")
            except OSError:
                continue
            lines = head.splitlines()
            if not lines:
                continue
            banner = lines[0].lower().split()
            field = banner[3] if len(banner) > 3 else "?"
            symm = banner[4] if len(banner) > 4 else "?"
            dims = None
            for ln in lines[1:]:
                s = ln.strip()
                if s and not s.startswith("%"):
                    dims = s.split()
                    break
            if not dims or len(dims) < 3:
                # MatrixMarket *array* storage: the header carries dims but no
                # nonzero count, so it is a dense vector or a coordinate list,
                # not a sparse matrix. Usually that is an auxiliary file next to
                # the matrix and dropping it is right. But redwood's shared cache
                # holds, for 37 matrices, an auxiliary array *under the matrix's
                # own name* -- lp_nug30.mtx there is the LP cost vector, and
                # onera_dual.mtx is a node-coordinate list -- so dropping it
                # silently removed the whole LP family from that host's corpus
                # while the other host measured all of it. Record the drops whose
                # stem matches their directory, since those are the ones that
                # cost a matrix rather than skipping a right-hand side.
                if os.path.basename(p)[:-4] == os.path.basename(os.path.dirname(p)):
                    dropped.append(p)
                continue
            M, J, ent = (int(x) for x in dims[:3])
            mult = 2 if symm in ("symmetric", "hermitian", "skew-symmetric") else 1
            rel = os.path.relpath(p, root)
            stem = os.path.basename(p)[:-4]
            rows.append(
                dict(
                    primary=int(stem == os.path.basename(os.path.dirname(p))),
                    key="ss:" + rel[:-4].replace(os.sep, "/"),
                    kind="mtx",
                    path=p,
                    rows=M,
                    cols=J,
                    nnz_pred=ent * mult,
                    field=field,
                    symmetry=symm,
                    family=rel.split(os.sep)[0],
                )
            )
    if dropped:
        print(
            "scan_suitesparse(%s): %d file(s) named as their own matrix are in "
            "MatrixMarket array storage and were dropped -- that host is missing "
            "those matrices, fetch them rather than assuming the path is the "
            "matrix:" % (root, len(dropped)),
            file=sys.stderr,
        )
        for p in sorted(dropped):
            print("  %s" % p, file=sys.stderr)
    return rows


def scan_dlmc(root):
    """Every ``.smtx`` under `root`, keyed by model/technique/sparsity/name."""
    rows = []
    for d, _, fs in os.walk(root):
        for f in sorted(fs):
            if not f.endswith(".smtx"):
                continue
            p = os.path.join(d, f)
            try:
                with open(p, "rb") as fh:
                    hdr = fh.readline().decode()
                M, J, nnz = (int(x) for x in hdr.replace(",", " ").split()[:3])
            except (OSError, ValueError):
                continue
            rel = os.path.relpath(p, root)
            parts = rel.split(os.sep)
            rows.append(
                dict(
                    primary=1,
                    key="dlmc:" + rel[:-5].replace(os.sep, "/"),
                    kind="smtx",
                    path=p,
                    rows=M,
                    cols=J,
                    nnz_pred=nnz,
                    field="pattern",
                    symmetry="general",
                    # model/technique/sparsity, e.g. rn50/magnitude_pruning/0.9
                    family="/".join(parts[:3]),
                )
            )
    return rows


FIELDS = [
    "primary",
    "key",
    "kind",
    "path",
    "rows",
    "cols",
    "nnz_pred",
    "field",
    "symmetry",
    "family",
]


def write_manifest(rows, path):
    rows = sorted(rows, key=lambda r: (r["nnz_pred"], r["key"]))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def read_manifest(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for c in ("rows", "cols", "nnz_pred", "primary"):
            r[c] = int(r[c])
    return rows


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="build a corpus manifest")
    ap.add_argument("--suitesparse", default=None)
    ap.add_argument("--dlmc", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = []
    if a.suitesparse:
        rows += scan_suitesparse(a.suitesparse)
    if a.dlmc:
        rows += scan_dlmc(a.dlmc)
    n = write_manifest(rows, a.out)
    print(f"{n} matrices -> {a.out}")
