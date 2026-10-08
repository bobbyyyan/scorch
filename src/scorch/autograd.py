"""Autograd for scorch's sparse x dense products.

``scorch.matmul`` computes a sparse x dense product with scorch's own kernels. When a
gradient is wanted -- grad mode is on and an input requires grad, or forward-mode AD has
given an input a tangent -- the product is recorded on torch's tape through the functions
here, so ``import scorch as torch`` code trains through a sparse product as it does under
torch. Before this module, every such product returned a tensor with no ``grad_fn``.

For ``out = A @ B`` with ``A`` sparse (stored values ``v``), ``B`` dense and
``G = dL/d(out)``::

    dL/dB = A^T @ G              an SpMM against A's transpose
    dL/dv = SDDMM(A; G, B)       G[i, :] . B[j, :] at each stored (i, j)

Three Functions -- ``A @ B``, ``A^T @ B`` and the SDDMM -- are closed under
differentiation: each one's backward is built from the three, so a second derivative
(``create_graph=True``) is recorded the same way as the first, and each has a ``jvp`` for
forward-mode AD.

Which inputs carry gradients:

- a torch sparse CSR or COO tensor that requires grad: its gradient arrives through
  ``values()``, so ``A.grad`` is a sparse tensor with A's structure, as
  ``torch.sparse.mm`` gives (``torch.matmul`` raises on such an input);
- an STensor: never. Its values are stored detached and its ``requires_grad`` is inert
  (see STensor), so only its dense partner gets a gradient;
- a dense torch tensor: as under torch.

The products run on scorch's ordinary path. ``A @ B`` runs on the caller's own operand,
so it is the product the same call computes without autograd, plans included. ``A^T @ B``
runs on an STensor this module builds once per sparsity structure over A's transpose and
keeps with the operand; the transpose's values are refilled from A's only when A's have
changed, so a training loop over a fixed matrix does no per-step work for them, and the
STensor gets plans of its own like any other operand.

A product this module does not cover (a sparse x sparse product, a batched one, keyword
arguments, an unsupported dtype or device) is computed without autograd and given a
``grad_fn`` that raises when a gradient is actually asked of it. That is torch's own
convention for an operation with no derivative, and it keeps a forward pass that never
calls ``backward`` -- inference with parameters that require grad, say -- working.
"""

from __future__ import annotations

import threading
import weakref
from typing import Any, Callable, Dict, Optional, Tuple, cast

import torch
import torch.autograd.forward_ad as _fwad

import scorch_ops as _native

from .exceptions import TensorIndexError
from .format import LevelType
from .plan import GRAD_ATTR, torch_operand_state
from .stensor import STensor
from .storage import tensor_version

_Stamp = Tuple[int, int, int]

# The value types scorch's SpMM kernels serve.
_DTYPES = (torch.float32, torch.float64)

# ops.matmul, the public binding (the C entry point where it is built), bound on first use:
# ops imports this module, so it cannot be imported here at load time.
_MATMUL: Optional[Callable[..., Any]] = None
_NTHREADS: Optional[Callable[[], int]] = None
_ATPARALLEL = False
_TRANSPOSE: Optional[Callable[[torch.Tensor], torch.Tensor]] = None


def _matmul(a: Any, b: torch.Tensor) -> torch.Tensor:
    global _MATMUL
    if _MATMUL is None:
        from . import ops

        _MATMUL = ops.matmul
    # A sparse operand times a dense matrix: a dense torch.Tensor.
    return cast(torch.Tensor, _MATMUL(a, b))


def _nthreads() -> int:
    """The thread count the drop-in SpMM is handed, so the SDDMM runs on the same team."""
    global _NTHREADS, _ATPARALLEL
    if _NTHREADS is None:
        from . import ops

        _ATPARALLEL = ops._PLAN_ATPARALLEL
        _NTHREADS = ops._PLAN_NTHREADS
    return _NTHREADS()


def _transpose(t: torch.Tensor) -> torch.Tensor:
    """``t.t()`` made contiguous, by scorch's cache-blocked transpose where it applies
    (float32): torch's own copy of a transposed view runs well below memory bandwidth.
    """
    global _TRANSPOSE
    if _TRANSPOSE is None:
        from . import ops

        _TRANSPOSE = ops.fast_transpose
    return _TRANSPOSE(t)


def _transposed_layout(t: torch.Tensor) -> bool:
    """Whether ``t`` is the transpose of a contiguous matrix: ``x.t()``, which is how
    ``x @ A`` hands ``x`` to ``A^T @ x^T``."""
    return t.dim() == 2 and not t.is_contiguous() and t.t().is_contiguous()


def _in_layout(result: torch.Tensor, transposed: bool) -> torch.Tensor:
    """A contiguous ``result`` in the layout of the dense operand it came from.

    A product's output, and the gradient of its dense operand, follow that operand's
    layout: ``x @ A`` is computed as ``(A^T @ x^T)^T``, and laying ``A^T @ x^T`` out as the
    transpose of a contiguous matrix makes the ``x @ A`` the caller receives contiguous, as
    torch's is, and makes ``x``'s gradient contiguous, which is the layout autograd
    accumulates into a contiguous ``x.grad`` without another copy.
    """
    return _transpose(result).t() if transposed else result


def _stamp(t: torch.Tensor) -> _Stamp:
    """Buffer, version counter and length: what identifies an array's contents without
    reading them (the same triple storage.py's validation stamp and plan.h record). An
    inference tensor has no version counter and reads as version 0, with the blind spot
    storage.tensor_version describes."""
    return (t.data_ptr(), tensor_version(t), t.numel())


# --------------------------------------------------------------------------- #
# A CSR structure
# --------------------------------------------------------------------------- #


class _Structure:
    """An ``rows x cols`` CSR structure, as the gradients of a product need it.

    It holds the index arrays (screened once, here), its transpose (built on first use,
    with the permutation that carries values across), and an STensor over each through
    which a product with given values runs on the ordinary path, plans included.

    ``sources`` are the operand's own index arrays this structure was derived from (for a
    CSR operand, the two arrays themselves; for COO or DCSR, the arrays the positions were
    computed from). :meth:`describes` compares their stamps, so a structure whose operand
    has changed is rebuilt rather than reused, and the transpose is shared by every
    structure over the same ones (``_TRANSPOSES``).
    """

    def __init__(
        self,
        positions: torch.Tensor,
        coordinates: torch.Tensor,
        rows: int,
        cols: int,
        sources: Tuple[torch.Tensor, ...] = (),
    ) -> None:
        # Detached: an operand's index accessor can return a view whose base is the operand
        # itself (a COO tensor's indices()), and this structure lives in the operand's own
        # entry in plan.TORCH_OPERANDS, so a view held here would keep the operand alive.
        positions = positions.detach().contiguous()
        coordinates = coordinates.detach().contiguous()
        if positions.numel() != rows + 1 or _native.abi_screen_compressed_level(
            positions, coordinates, cols, False
        ):
            raise TensorIndexError(
                "the sparse operand's index arrays do not describe a valid "
                f"{rows} x {cols} CSR structure"
            )
        self.positions = positions
        self.coordinates = coordinates
        self.rows = rows
        self.cols = cols
        self.nnz = coordinates.numel()
        # Detached for the reason above.
        self.sources = tuple(t.detach() for t in sources) or (positions, coordinates)
        self.source = tuple(_stamp(t) for t in self.sources)
        self._transpose: Optional[Tuple[_Structure, torch.Tensor]] = None
        # dtype -> (STensor over these arrays, its values buffer)
        self._carriers: Dict[torch.dtype, Tuple[STensor, torch.Tensor]] = {}
        self._lock = threading.Lock()

    def describes(self, *arrays: torch.Tensor) -> bool:
        return self.source == tuple(_stamp(a) for a in arrays)

    def transpose(self) -> Tuple["_Structure", torch.Tensor]:
        """The ``cols x rows`` transpose and ``perm``: its values are ``values[perm]``."""
        held = self._transpose
        if held is None:
            held = self._transpose = _shared_transpose(self)
        return held

    def sddmm(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """``out[q] = x[i, :] . y[j, :]`` for each stored ``q = (i, j)``."""
        nthreads = _nthreads()  # binds _ATPARALLEL on first use
        return _native.sddmm_csr(
            self.positions,
            self.coordinates,
            _plain(x),
            _plain(y),
            nthreads,
            _ATPARALLEL,
        )

    def mm(
        self,
        values: torch.Tensor,
        b: torch.Tensor,
        perm: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """This structure with ``values`` (or ``values[perm]``) times ``b``.

        Runs on an STensor this structure keeps for the dtype, so the call is planned
        like any repeated product. Its values buffer is refilled on every call: a write
        to the values that bypasses their version counter -- through ``.data``, which is
        how gradcheck perturbs an input and how older optimizers step, or a numpy view --
        would otherwise leave the product reading the values it was last filled from.
        torch reads the values on every call, and so does every plan; this copy is the
        one place they are held, so it is refreshed rather than compared.
        """
        b = _plain(b)
        with self._lock:
            held = self._carriers.get(values.dtype)
            if held is None:
                # An ordinary tensor even when the first call is under inference_mode:
                # an inference tensor cannot be refilled outside it.
                with torch.inference_mode(False):
                    buffer = torch.empty(self.nnz, dtype=values.dtype)
                    carrier = STensor.from_csr(
                        torch.sparse_csr_tensor(
                            self.positions,
                            self.coordinates,
                            buffer,
                            size=(self.rows, self.cols),
                        )
                    )
                held = self._carriers[values.dtype] = (carrier, buffer)
            carrier, buffer = held
            if perm is None:
                buffer.copy_(values)
                return _matmul(carrier, b)
            # A transpose's values are a gather through perm. Where the carrier holds a plan
            # for this product, the plan does the gather (SpmmCsrPlan::run_gathered: on the
            # product's team, where it has one, each worker right before it reads them);
            # until then, gather_values fills the buffer and the ordinary path plans the
            # product.
            values = values.contiguous()
            served = _native.planned_matmul_gathered(carrier, b, values, perm)
            if served is not None:
                return cast(torch.Tensor, served)
            _native.gather_values(values, perm, buffer, _nthreads())
            return _matmul(carrier, b)


# --------------------------------------------------------------------------- #
# Transposes, shared by every structure over the same index arrays
# --------------------------------------------------------------------------- #
#
# A structure is cached on its operand object, like a plan. A program that wraps the same
# index arrays in a new operand on every call -- torch.sparse_csr_tensor(crow, col, values),
# with values that change -- would then build the transpose x @ A runs on, an O(nnz)
# counting sort plus the STensor that carries it and that STensor's plans, on every call:
# 103 us against 32 us for a call that finds it, on a 2000 x 2000 matrix with 20k nonzeros
# at k = 1 on an M5, where the product ran 86 us before autograd.py. So the transpose is also
# kept here, keyed on the index arrays themselves, for exactly as long as they live.
#
# torch keeps one Python object per storage for as long as the storage lives, and a
# finalizer on it runs when the storage dies, so an entry is dropped with the arrays it
# describes and a storage's id cannot be reused while an entry names it. The key holds, per
# array, the storage's id, the array's address and length and its dtype; the entry holds
# the arrays' versions, so a structure changed in place replaces its entry rather than
# adding one. (The validation memo in csrc/native_abi.h keys index arrays the same way,
# with a weak pointer to the storage in place of the finalizer.)

_ArrayKey = Tuple[int, int, int, torch.dtype]
_TRANSPOSES: Dict[
    Tuple[Any, ...], Tuple[Tuple[int, ...], Tuple["_Structure", torch.Tensor]]
] = {}
_TRANSPOSES_LOCK = threading.Lock()


def _shared_transpose(s: "_Structure") -> Tuple["_Structure", torch.Tensor]:
    storages = [t.untyped_storage() for t in s.sources]
    key = (s.rows, s.cols) + tuple(
        (id(storage), t.data_ptr(), t.numel(), t.dtype)
        for storage, t in zip(storages, s.sources)
    )
    versions = tuple(tensor_version(t) for t in s.sources)
    with _TRANSPOSES_LOCK:
        entry = _TRANSPOSES.get(key)
    if entry is not None and entry[0] == versions:
        return entry[1]
    positions, coordinates, perm = _native.csr_transpose(
        s.positions, s.coordinates, s.cols
    )
    held = (_Structure(positions, coordinates, s.cols, s.rows), perm)
    with _TRANSPOSES_LOCK:
        new = key not in _TRANSPOSES
        _TRANSPOSES[key] = (versions, held)
    if new:
        for storage in {id(storage): storage for storage in storages}.values():
            weakref.finalize(storage, _TRANSPOSES.pop, key, None)
    return held


def _plain(t: torch.Tensor) -> torch.Tensor:
    """``t`` as the kernels read it: contiguous, with no lazy conjugate or negation."""
    if t.is_neg() or t.is_conj():
        t = t.resolve_neg().resolve_conj()
    if _transposed_layout(t):
        return _transpose(t.t())
    return t.contiguous()


# --------------------------------------------------------------------------- #
# A sparse operand
# --------------------------------------------------------------------------- #


class _Operand:
    """One sparse operand of one product: the structure its values are stored in, and
    whether that structure is the operand's own CSR (``transposed`` false) or its
    transpose's (a CSC layout stores ``A^T``'s CSR).

    ``values`` given to the methods are always in storage order. ``owner`` is the
    caller's object. When it is a CSR operand (``direct``), ``A @ B`` with the owner's
    own values runs on it directly: that is the product the same call computes without
    autograd, on the drop-in kernel, with the owner's plans. Any other layout's ordinary
    path is the generated kernel, so its products run on the CSR structure instead -- the
    same drop-in kernel. (That path also has defects of its own: float64 COO x dense
    computes in float32 precision at some widths and fails to compile at others.)
    """

    __slots__ = ("owner", "owner_values", "stored", "transposed", "direct", "shape")

    def __init__(
        self,
        owner: Any,
        owner_values: torch.Tensor,
        stored: _Structure,
        transposed: bool,
        direct: bool,
    ) -> None:
        self.owner = owner
        self.owner_values = owner_values
        self.stored = stored
        self.transposed = transposed
        self.direct = direct
        self.shape = (
            (stored.cols, stored.rows) if transposed else (stored.rows, stored.cols)
        )

    def _owns(self, values: torch.Tensor) -> bool:
        own = self.owner_values
        return (
            values.data_ptr() == own.data_ptr()
            and values.numel() == own.numel()
            and values.stride() == own.stride()
        )

    def mm(self, values: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """``A @ b``."""
        if self.direct and self._owns(values):
            return _matmul(self.owner, _plain(b))
        if self.transposed:
            structure, perm = self.stored.transpose()
            return structure.mm(values, b, perm)
        return self.stored.mm(values, b)

    def mm_t(self, values: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """``A^T @ b``."""
        if self.transposed:
            return self.stored.mm(values, b)
        structure, perm = self.stored.transpose()
        return structure.mm(values, b, perm)

    def sddmm(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """``x[i, :] . y[j, :]`` for each stored ``A[i, j]``, in storage order."""
        if self.transposed:
            return self.stored.sddmm(y, x)
        return self.stored.sddmm(x, y)


# --------------------------------------------------------------------------- #
# The three Functions
# --------------------------------------------------------------------------- #


def _record(ctx: Any, op: _Operand, first: torch.Tensor, second: torch.Tensor) -> None:
    """What a backward or forward-mode pass reads, saved on ``ctx``.

    The three Functions share a shape: ``(op, first, second)``, where the gradient of
    ``first`` reads ``second`` and the gradient of ``second`` reads ``first``. So only what a
    requested gradient reads is saved for backward (``second`` can be a large activation);
    a forward-mode pass reads both.

    Classic Functions (``forward(ctx, ...)``), not the ``setup_context`` form: torch binds
    a ``setup_context`` Function's arguments through ``inspect.signature`` on every call,
    which was most of the Python a training step through a small matrix executed.
    """
    ctx.op = op
    ctx.b_transposed = ctx.y_transposed = _transposed_layout(second)
    ctx.save_for_backward(
        first if ctx.needs_input_grad[2] else None,
        second if ctx.needs_input_grad[1] else None,
    )
    ctx.save_for_forward(first, second)


def _add(
    x: Optional[torch.Tensor], y: Optional[torch.Tensor]
) -> Optional[torch.Tensor]:
    if x is None:
        return y
    return x if y is None else x + y


class _SparseMM(torch.autograd.Function):
    """``A(values) @ b``."""

    @staticmethod
    def forward(
        ctx: Any, op: _Operand, values: torch.Tensor, b: torch.Tensor
    ) -> torch.Tensor:
        _record(ctx, op, values, b)
        return _in_layout(op.mm(values, b), ctx.b_transposed)

    @staticmethod
    def backward(ctx: Any, g: torch.Tensor):
        values, b = ctx.saved_tensors
        need = ctx.needs_input_grad
        return (None,) + _mm_backward(
            ctx.op, values, b, g, need[1], need[2], ctx.b_transposed
        )

    @staticmethod
    def jvp(ctx: Any, _op: Any, dv: Optional[torch.Tensor], db: Optional[torch.Tensor]):
        values, b = ctx.saved_tensors
        op = ctx.op
        total = _add(
            op.mm(dv, b) if dv is not None else None,
            op.mm(values, db) if db is not None else None,
        )
        assert total is not None  # forward AD asks for this only with some tangent
        return _in_layout(total, ctx.b_transposed)


def _mm_backward(
    op: _Operand,
    values: Optional[torch.Tensor],
    b: Optional[torch.Tensor],
    g: torch.Tensor,
    want_values: bool,
    want_b: bool,
    b_transposed: bool,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """The gradients of ``A(values) @ b`` given the output's gradient ``g``: of the values,
    ``SDDMM(g, b)``, and of ``b``, ``A^T @ g``. Under grad mode (a backward asked to build
    its own graph) each is recorded as the Function computing it, so the set is closed
    under differentiation."""
    graph = torch.is_grad_enabled()
    gv = gb = None
    if want_values:
        assert b is not None  # saved whenever the values' gradient is wanted (_record)
        gv = _SparseSDDMM.apply(op, g, b) if graph else op.sddmm(g, b)
    if want_b:
        assert values is not None  # and the values whenever b's is
        if graph:
            gb = _SparseMMT.apply(op, values, g)
        else:
            gb = _in_layout(op.mm_t(values, g), b_transposed)
    return gv, gb


def planned_mm_backward(
    a: torch.Tensor,
    values: Optional[torch.Tensor],
    b: Optional[torch.Tensor],
    g: torch.Tensor,
    want_values: bool,
    want_b: bool,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """The backward of ``a @ b`` for a torch CSR ``a`` whose product the C entry point
    served from its plan and recorded itself (csrc/planned_matmul.h, PlannedMatmulGrad).

    The same gradients as :class:`_SparseMM`'s, by the same code: only the forward moved
    to C, because that is where a step through a small matrix spent its cold time -- the
    Python layers in front of the kernel, not the kernel. ``values`` is ``a.values()`` as
    the forward saw it (version-checked by the node) and ``b`` the dense operand, each
    present only when the other's gradient is wanted, as :func:`_record` saves them.
    """
    held = _torch_operand(a)
    if held is None:  # unreachable: the C entry serves only 2-D torch CSR operands
        raise RuntimeError(
            "scorch: a planned product's operand is no longer sparse CSR"
        )
    return _mm_backward(held[0], values, b, g, want_values, want_b, False)


class _SparseMMT(torch.autograd.Function):
    """``A(values)^T @ b``."""

    @staticmethod
    def forward(
        ctx: Any, op: _Operand, values: torch.Tensor, b: torch.Tensor
    ) -> torch.Tensor:
        _record(ctx, op, values, b)
        return _in_layout(op.mm_t(values, b), ctx.b_transposed)

    @staticmethod
    def backward(ctx: Any, g: torch.Tensor):
        values, b = ctx.saved_tensors
        op = ctx.op
        graph = torch.is_grad_enabled()
        gv = gb = None
        if ctx.needs_input_grad[1]:
            # d/dA[i, j] of sum_j (A^T b)[j, :] . g[j, :] is b[i, :] . g[j, :].
            gv = _SparseSDDMM.apply(op, b, g) if graph else op.sddmm(b, g)
        if ctx.needs_input_grad[2]:
            if graph:
                gb = _SparseMM.apply(op, values, g)
            else:
                gb = _in_layout(op.mm(values, g), ctx.b_transposed)
        return None, gv, gb

    @staticmethod
    def jvp(ctx: Any, _op: Any, dv: Optional[torch.Tensor], db: Optional[torch.Tensor]):
        values, b = ctx.saved_tensors
        op = ctx.op
        total = _add(
            op.mm_t(dv, b) if dv is not None else None,
            op.mm_t(values, db) if db is not None else None,
        )
        assert total is not None  # forward AD asks for this only with some tangent
        return _in_layout(total, ctx.b_transposed)


class _SparseSDDMM(torch.autograd.Function):
    """``out[q] = x[i, :] . y[j, :]`` for each stored ``A[i, j]``."""

    @staticmethod
    def forward(
        ctx: Any, op: _Operand, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        ctx.x_transposed = _transposed_layout(x)
        _record(ctx, op, x, y)
        return op.sddmm(x, y)

    @staticmethod
    def backward(ctx: Any, g: torch.Tensor):
        x, y = ctx.saved_tensors
        op = ctx.op
        graph = torch.is_grad_enabled()
        gx = gy = None
        if ctx.needs_input_grad[1]:
            if graph:
                gx = _SparseMM.apply(op, g, y)
            else:
                gx = _in_layout(op.mm(g, y), ctx.x_transposed)
        if ctx.needs_input_grad[2]:
            if graph:
                gy = _SparseMMT.apply(op, g, x)
            else:
                gy = _in_layout(op.mm_t(g, x), ctx.y_transposed)
        return None, gx, gy

    @staticmethod
    def jvp(ctx: Any, _op: Any, dx: Optional[torch.Tensor], dy: Optional[torch.Tensor]):
        x, y = ctx.saved_tensors
        op = ctx.op
        return _add(
            op.sddmm(dx, y) if dx is not None else None,
            op.sddmm(x, dy) if dy is not None else None,
        )


# --------------------------------------------------------------------------- #
# Operands
# --------------------------------------------------------------------------- #


def _cached(
    state: Dict[str, Any],
    sources: Tuple[torch.Tensor, ...],
    build: Callable[[], _Structure],
) -> _Structure:
    held = state.get(GRAD_ATTR)
    if held is None or not held.describes(*sources):
        held = state[GRAD_ATTR] = build()
    return held


def _torch_operand(a: torch.Tensor) -> Optional[Tuple[_Operand, torch.Tensor]]:
    rows, cols = a.shape
    if a.layout is torch.sparse_csr:
        positions, coordinates = a.crow_indices(), a.col_indices()
        state = torch_operand_state(a, create=True)
        stored = _cached(
            state,
            (positions, coordinates),
            lambda: _Structure(positions, coordinates, rows, cols),
        )
        values = a.values()
        return _Operand(a, values, stored, False, True), values
    if a.layout is torch.sparse_coo:
        if not a.is_coalesced():
            # Differentiable, and the order the values then have is row-major: CSR's. An
            # inference tensor is coalesced under inference_mode: torch's coalesce of one
            # outside it fails an internal assert (variable_excluded_from_dispatch), where
            # torch.mm on the same operand runs.
            if a.is_inference():
                with torch.inference_mode():
                    a = a.coalesce()
            else:
                a = a.coalesce()
        indices = a.indices()
        row_of, coordinates = indices[0], indices[1]
        state = torch_operand_state(a, create=True)

        def build() -> _Structure:
            positions = torch._convert_indices_from_coo_to_csr(row_of, rows)
            return _Structure(positions, coordinates, rows, cols, sources=(indices,))

        stored = _cached(state, (indices,), build)
        values = a.values()
        return _Operand(a, values, stored, False, False), values
    return None


def _stensor_operand(s: STensor) -> Optional[Tuple[_Operand, torch.Tensor]]:
    """An STensor's stored structure as CSR, for the two-level layouts scorch stores
    matrices in: dense-compressed (CSR, or CSC under mode order [1, 0]),
    compressed-compressed (DCSR) and coordinate (COO, lexicographically sorted)."""
    if s.dim() != 2:
        return None
    order = list(s.storage.index.mode_order or [0, 1])
    if order not in ([0, 1], [1, 0]):
        return None
    # An STensor's shape is in storage order: a CSC matrix of 7 rows and 5 columns
    # reports (5, 7). So the stored structure is shape[0] x shape[1] either way.
    outer, inner = s.shape[0], s.shape[1]
    levels = s.format.level_types
    indices = s._native_mode_indices()
    state = s.__dict__
    sources: Tuple[torch.Tensor, ...]
    build: Callable[[], _Structure]
    if levels == (LevelType.DENSE, LevelType.COMPRESSED):
        positions, coordinates = indices[1]
        sources = (positions, coordinates)

        def build() -> _Structure:
            return _Structure(positions, coordinates, outer, inner)

    elif levels == (LevelType.COMPRESSED, LevelType.COMPRESSED):
        top_positions, top_coordinates = indices[0]
        positions_1, coordinates = indices[1]
        sources = (top_coordinates, positions_1, coordinates)

        def build() -> _Structure:
            counts = positions_1[1:].long() - positions_1[:-1].long()
            positions = torch.zeros(outer + 1, dtype=torch.int64)
            positions.index_add_(0, top_coordinates.long() + 1, counts)
            return _Structure(
                positions.cumsum_(0).to(coordinates.dtype),
                coordinates,
                outer,
                inner,
                sources=sources,
            )

    elif levels[0] is LevelType.COORDINATE and levels[1] in (
        LevelType.COORDINATE,
        LevelType.SINGLETON,
    ):
        (row_of,) = indices[0]
        (coordinates,) = indices[1]
        sources = (row_of, coordinates)
        if _native.abi_screen_lex_levels([row_of, coordinates], row_of.numel()):
            return None  # not row-major order, so not a CSR order of the values

        def build() -> _Structure:
            positions = torch._convert_indices_from_coo_to_csr(
                row_of, outer, out_int32=coordinates.dtype == torch.int32
            )
            return _Structure(
                positions,
                coordinates,
                outer,
                inner,
                sources=sources,
            )

    else:
        return None
    stored = _cached(state, sources, build)
    values = s._raw_values
    csr = order == [0, 1] and levels == (LevelType.DENSE, LevelType.COMPRESSED)
    return _Operand(s, values, stored, order == [1, 0], csr), values


def _sparse_operand(x: Any) -> Optional[Tuple[_Operand, torch.Tensor]]:
    if type(x) is STensor:
        return _stensor_operand(x)
    if isinstance(x, torch.Tensor) and x.dim() == 2:
        return _torch_operand(x)
    return None


def is_sparse(x: Any) -> bool:
    return type(x) is STensor or (
        isinstance(x, torch.Tensor) and x.layout is not torch.strided
    )


def _dtype_of(x: Any) -> torch.dtype:
    return x._raw_values.dtype if type(x) is STensor else x.dtype


def _device_of(x: Any) -> torch.device:
    return x._raw_values.device if type(x) is STensor else x.device


# --------------------------------------------------------------------------- #
# Entry points
# --------------------------------------------------------------------------- #


def _fw_tangent(x: Any) -> bool:
    if not isinstance(x, torch.Tensor):
        return False
    try:
        return _fwad.unpack_dual(x).tangent is not None
    except RuntimeError:
        # A layout unpack_dual cannot read (sparse CSR: "do not have strides") is one
        # torch cannot make dual either.
        return False


def wanted(a: Any, b: Any) -> bool:
    """Whether ``a @ b`` must be recorded for autograd: grad mode is on and a torch
    input requires grad, or forward-mode AD has given a torch input a tangent."""
    if torch.is_grad_enabled() and (
        (isinstance(a, torch.Tensor) and a.requires_grad)
        or (isinstance(b, torch.Tensor) and b.requires_grad)
    ):
        return True
    return _fwad._current_level >= 0 and (_fw_tangent(a) or _fw_tangent(b))


def serve(a: Any, b: Any, ordinary: Callable[..., Any], kwargs: Dict[str, Any]) -> Any:
    """What ``ops.matmul`` asks of this module, for a call with a sparse operand the C entry
    point did not probe: ``a @ b`` recorded for autograd when a gradient is wanted through
    it (``matmul``); ``x @ A`` for a dense ``x`` and a sparse ``A`` computed unrecorded when
    not (``right_product``), so that product is one function in every grad mode; None for
    every other call, which ``ops.matmul`` computes itself."""
    if wanted(a, b):
        if is_sparse(a) or is_sparse(b):
            return matmul(a, b, ordinary, kwargs)
        return None
    if kwargs or not is_sparse(b):
        return None
    return right_product(a, b)


def matmul(a: Any, b: Any, ordinary: Callable[..., Any], kwargs: Dict[str, Any]) -> Any:
    """``a @ b`` recorded for autograd.

    ``ordinary`` is scorch's matmul itself, which computes a product this module does not
    cover; its result is given a ``grad_fn`` that raises if a gradient is asked of it.
    """
    served = None if kwargs else _differentiable(a, b)
    if served is not None:
        return served
    return _not_differentiable(a, b, ordinary, kwargs)


def _covered(
    a: Any, b: Any
) -> Optional[Tuple[bool, _Operand, torch.Tensor, torch.Tensor]]:
    """``(left, op, values, dense)`` when this module computes ``a @ b``: one operand
    sparse and 2-D, the other a dense CPU matrix or vector of the same float dtype;
    ``left`` says whether the sparse one is ``a``. None otherwise."""
    left = is_sparse(a)
    sparse, dense = (a, b) if left else (b, a)
    if (
        not is_sparse(sparse)
        or type(dense) not in (torch.Tensor, torch.nn.Parameter)
        or dense.layout is not torch.strided
        or dense.dim() not in (1, 2)
        or dense.dtype not in _DTYPES
        or dense.dtype != _dtype_of(sparse)
        or dense.device.type != "cpu"
        or _device_of(sparse).type != "cpu"
    ):
        return None
    held = _sparse_operand(sparse)
    if held is None:
        return None
    op, values = held
    rows, cols = op.shape
    inner = dense.shape[0] if left else dense.shape[-1]
    if inner != (cols if left else rows):
        return None  # the ordinary path raises the shape error
    return left, op, values, dense


def _differentiable(a: Any, b: Any) -> Optional[torch.Tensor]:
    covered = _covered(a, b)
    if covered is None:
        return None
    left, op, values, dense = covered
    if left:
        if dense.dim() == 1:
            return _SparseMM.apply(op, values, dense.unsqueeze(1)).squeeze(1)
        return _SparseMM.apply(op, values, dense)
    # x @ A = (A^T @ x^T)^T
    if dense.dim() == 1:
        return _SparseMMT.apply(op, values, dense.unsqueeze(1)).squeeze(1)
    return _SparseMMT.apply(op, values, dense.t()).t()


def right_product(a: Any, b: Any) -> Optional[torch.Tensor]:
    """``a @ b`` for a dense ``a`` and a sparse ``b`` with nothing to record: the forward
    ``_differentiable`` records, run unrecorded, so a product returns the same type and the
    same values whether or not a gradient is wanted through it (model.train() and
    model.eval() see one function). None when this module does not cover the product."""
    covered = _covered(a, b)
    if covered is None or covered[0]:
        return None
    _, op, values, x = covered
    if x.dim() == 1:
        return op.mm_t(values, x.unsqueeze(1)).squeeze(1)
    xt = x.t()
    return _in_layout(op.mm_t(values, xt), _transposed_layout(xt)).t()


class _NotDifferentiable(torch.autograd.Function):
    """Carries a result computed without autograd; raises if a gradient is asked of it."""

    @staticmethod
    def forward(
        ctx: Any, result: torch.Tensor, what: str, *inputs: torch.Tensor
    ) -> torch.Tensor:
        ctx.what = what
        return result

    @staticmethod
    def backward(ctx: Any, *grads: torch.Tensor):
        raise NotImplementedError(
            f"scorch.matmul has no derivative for {ctx.what}; differentiable products "
            "are a 2-D sparse operand (torch CSR or COO, or a CSR, CSC, DCSR or sorted "
            "COO STensor) times a dense float32/float64 CPU matrix or vector, either way "
            "round, with no keyword arguments"
        )

    @staticmethod
    def jvp(ctx: Any, *tangents: Any):
        raise NotImplementedError(
            f"scorch.matmul has no forward derivative for {ctx.what}"
        )


def _describe(x: Any) -> str:
    if type(x) is STensor:
        return f"an STensor of format {x.format}"
    if isinstance(x, torch.Tensor):
        return f"a {x.dim()}-D {x.layout} {x.dtype} tensor"
    return type(x).__name__


def _not_differentiable(
    a: Any, b: Any, ordinary: Callable[..., Any], kwargs: Dict[str, Any]
) -> Any:
    inputs = [x for x in (a, b) if isinstance(x, torch.Tensor)]
    # Computed from the primals, outside autograd: the ordinary path records nothing, and
    # with forward-mode AD active it would otherwise be asked again.
    with torch.no_grad():
        result = ordinary(
            *(x.detach() if isinstance(x, torch.Tensor) else x for x in (a, b)),
            **kwargs,
        )
    if type(result) is not torch.Tensor:
        return result  # an STensor result has no autograd at all (see STensor)
    what = f"{_describe(a)} @ {_describe(b)}" + (
        f" with {', '.join(sorted(kwargs))}" if kwargs else ""
    )
    return _NotDifferentiable.apply(result, what, *inputs)
