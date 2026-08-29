from __future__ import annotations
import os
import weakref
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import replace
from typing import Any, Optional, Tuple, Union, List

import torch

from .compiler.cin import (
    TensorVar,
    ForAll,
    IndexVar,
    Workspace,
    Where,
    TensorAssign,
)
from .compiler.cin_lowerer import CINLowerer
from .compiler.codegen import LLIRLowerer
from .exceptions import (
    TensorIndexError,
    TensorLayoutError,
    TensorStorageError,
    TensorTypeError,
    TensorValidationError,
)
from .format import TensorFormat, LevelFormat, LevelType
from .layout import TensorLayout, TensorMetadata

# Names only -- plan.py imports torch and the native extension, nothing from this
# module, so there is no cycle. Imported as constants because _set_state runs on
# every STensor construction and must not pay a function call for this.
from .plan import (
    PLANS_ATTR as _PLANS_ATTR,
    SEEN_ATTR as _PLAN_SEEN_ATTR,
    DECLINES_ATTR as _PLAN_DECLINES_ATTR,
)
from .storage import SparseStorage, TensorIndex, tensor_version
from .utils import (
    parse_format,
    get_extra_cflags,
    get_extra_ldflags,
    jit_preamble_text,
    _kernel_name,
    _load_kernel,
)

# Whether `_wrap_generated_result` may take a kernel's index arrays instead of copying
# them; see that function for what makes it safe and for the one call site that opts
# out. `SCORCH_ADOPT_KERNEL_INDICES=0` turns it off, so the two states can be compared
# in one binary.
_ADOPT_KERNEL_INDICES = os.environ.get("SCORCH_ADOPT_KERNEL_INDICES", "1") not in (
    "0",
    "false",
    "False",
)

# The same value in a mutable cell. `_wrap_generated_result` reads the cell rather than
# taking the flag as a default argument, because a default argument is bound once when
# the function is defined -- which would confine any measurement of what adoption is
# worth to a comparison between two processes. Timing both states in one process is
# strictly better evidence, and this is what makes it possible.
_ADOPT_CELL = [_ADOPT_KERNEL_INDICES]


def _finalize_generated_mode_indices(
    tensor_format: TensorFormat,
    mode_indices: Sequence[Sequence[torch.Tensor]],
) -> List[List[torch.Tensor]]:
    """Finish zero-initialized trailing compressed positions from JIT output."""
    finalized = [list(arrays) for arrays in mode_indices]
    for level, level_format in enumerate(tensor_format.get_level_formats()):
        if level_format.get_level_type() != LevelType.COMPRESSED:
            continue
        positions = finalized[level][0]
        if positions.numel() == 0 or positions[-1].item() != 0:
            continue
        nonzero = torch.nonzero(positions, as_tuple=False).flatten()
        if nonzero.numel() == 0:
            continue
        last_written = int(nonzero[-1].item())
        repaired = positions.clone()
        repaired[last_written + 1 :] = repaired[last_written]
        finalized[level][0] = repaired
    return finalized


# --------------------------------------------------------------------------- #
# Generated dense results
#
# A generated kernel's *dense* result is described completely by (physical shape, format,
# mode order, name, dtype, device): the index -- every level dense, so every per-mode
# array tuple is empty -- the layout, and the metadata. Only the values tensor differs
# between calls, and rebuilding the rest was 7.3 us of the 19.4 us a warm 64x64
# `einsum("ik,kj->ij", ..., format="dd")` spent in Python.
#
# This is the same trade as `_DENSE_PARTS_CACHE` above, made sound the same way: each
# cached part is an immutable value object -- TensorIndex, TensorLayout and
# TensorMetadata, whose only `object.__setattr__` calls are inside their own
# constructors -- and a dense tensor's per-mode index arrays are empty tuples, so there
# is nothing shared that a holder could write through.
#
# A *sparse* result keeps the ordinary path. Its index arrays are different arrays on
# every call, so the index cannot be shared, and what is left to hold is the layout,
# which `TensorLayout.from_physical_shape` already caches.
#
# Set the bound to 0 for a same-binary control arm: installation stops and every call
# takes the ordinary path, which is what happened before this existed.
# --------------------------------------------------------------------------- #
_RESULT_PARTS_CACHE: dict = {}
_RESULT_PARTS_CACHE_MAX = 512


def _dense_result_parts(
    shape: Tuple[int, ...],
    tensor_format: TensorFormat,
    mode_indices: Sequence[Sequence[torch.Tensor]],
    mode_order: Optional[Sequence[int]],
    name: Optional[str],
    value: torch.Tensor,
    adopt: bool,
) -> Optional[Tuple[TensorIndex, TensorLayout, TensorMetadata]]:
    """The constant parts of a dense generated result, or None to take the long way.

    Declines unless the format is all-dense *and* the kernel handed back no index
    arrays. The second condition is not redundant: an all-dense format requires zero
    arrays per level, and it is `_normalize_mode_indices` inside the ordinary path that
    enforces it, so a shortcut that assumed it would turn a fail-closed check into a
    silent one.
    """
    if not tensor_format.is_dense():
        return None
    if any(arrays for arrays in mode_indices):
        return None
    # `adopt` is deliberately absent: it decides whether TensorIndex copies the arrays it
    # is given, and an all-dense index has none, so the two settings build the same index.
    key = (
        tuple(shape),
        tensor_format,
        None if mode_order is None else tuple(mode_order),
        name,
        value.dtype,
        value.device,
    )
    cached = _RESULT_PARTS_CACHE.get(key)
    if cached is not None:
        return cached
    # Build the parts with the ordinary constructors, so what is cached is what the
    # ordinary path produces rather than a second implementation of it.
    index = TensorIndex(
        tensor_format=tensor_format,
        mode_indices=_finalize_generated_mode_indices(tensor_format, mode_indices),
        mode_order=mode_order,
        _adopt=adopt,
    )
    layout = TensorLayout.from_physical_shape(
        shape, index.format, index.mode_order, index.index_dtype
    )
    metadata = TensorMetadata(
        "tensor" if name is None else name,
        value.dtype,
        value.device,
        layout,
        False,
    )
    parts = (index, layout, metadata)
    if len(_RESULT_PARTS_CACHE) < _RESULT_PARTS_CACHE_MAX:
        _RESULT_PARTS_CACHE[key] = parts
    return parts


def _wrap_generated_result(
    shape: Tuple[int, ...],
    tensor_format: Any,
    result_cpp: Any,
    mode_order: Optional[Sequence[int]] = None,
    name: Optional[str] = None,
    adopt: Optional[bool] = None,
) -> "STensor":
    """Turn a kernel's result into an STensor.

    Every dispatch path that runs a kernel -- prebuilt or generated -- ends with the
    same four steps: finish the zero-initialized trailing positions, describe the
    arrays, build the storage, name the tensor. This was written out at nine call sites --
    eight in ops.py and one in `STensor.to_sparse` -- which is nine chances to forget the
    first step; it is one function now.

    The result's index is declared *trusted*, so the O(nnz) structural walk is skipped
    unless `storage._VALIDATE_KERNEL_RESULTS` is on (`tests/conftest.py` turns it on, so
    every test still walks every generated result). The cheap per-array checks -- dtype,
    rank, contiguity, device, arity -- run either way. See that flag for why this follows
    `torch.sparse.check_sparse_tensor_invariants` rather than inventing a policy.

    An **all-dense** result takes a shortcut: its index, layout and metadata are
    constants of (shape, format, mode order, name, dtype, device), so they are built once
    per key and shared. Only the values tensor differs between calls. See
    `_dense_result_parts` for what it declines and why the "no index arrays" half of that
    test is not redundant. A sparse result takes the ordinary path below.

    ``adopt`` takes the kernel's index arrays as they are rather than copying them;
    ``None`` means "whatever the process is configured for" (``_ADOPT_CELL``), and an
    explicit ``False`` opts a single call site out for good.
    The copy exists so that a *caller* cannot invalidate a validated tensor by mutating
    what it passed in, which does not apply to a buffer a kernel allocated for its own
    output microseconds ago -- and Scorch already treats the result's *values* exactly
    that way, sharing them through ``detach`` rather than cloning. Adopting and skipping
    the walk are together worth 1.04-1.15x of a whole sparse-result ``einsum`` on both
    hosts (``bench/bench_index_validation.py --what adopt``, three arms in one process).

    Adopting is only sound where the kernel's output index arrays do not alias an
    *input's*. Sharing index buffers between a result and its operand would be safe only
    while nothing ever writes through an STensor's index arrays in place, which is not a
    property this code establishes anywhere, so the rule is per call site and not a
    global bet:

    * **Generated kernels allocate their output arrays.** Every sparse output level in
      the emitted C++ comes from a `torch::empty` sized from the counted extent
      (`compiler/cin_lowerer.py`, the output-buffer block), so no generated result can
      alias an operand. This is structural, not sampled.
    * **Prebuilt kernels with a sparse result allocate too** -- `kernels.h` builds fresh
      `C0_crd_torch` / `C1_pos_torch` tensors -- and the dense-output ones have no index
      arrays at all.
    * **SDDMM is the exception, and it is a real one.** `sddmm_coo_float_prebuilt`
      assigns `D.storage.index.mode_indices = S_mode_indices` (`csrc/kernels.h`): its
      result keeps the sparsity pattern of its operand and returns that operand's own
      arrays. The copy is what makes that safe today, so `ops.einsum` passes
      ``adopt=False`` at the SDDMM site. `test_sddmm_result_does_not_alias_its_operand`
      fails if that opt-out is ever removed.
    """
    if adopt is None:
        adopt = _ADOPT_CELL[0]
    # Accept either spelling of a format: callers pass a TensorFormat in most places
    # and a string in one, and `_finalize_generated_mode_indices` needs the parsed form.
    tensor_format = parse_format(tensor_format)
    # One `.storage` hop rather than two: these are pybind property reads, and the pair
    # of them was 0.6 us of a 7.3 us wrap.
    result_storage = result_cpp.storage
    kernel_mode_indices = result_storage.index.mode_indices
    value = result_storage.value
    parts = _dense_result_parts(
        shape, tensor_format, kernel_mode_indices, mode_order, name, value, adopt
    )
    if parts is not None:
        cached_index, cached_layout, cached_metadata = parts
        # `_from_validated` skips the shape-against-storage check that `STensor.__init__`
        # does; the layout was built from this same `shape` and the cache key contains it,
        # so that check could only ever pass. `_set_state` still compares the metadata's
        # layout, dtype and device against the storage's.
        return STensor._from_validated(
            cached_metadata,
            SparseStorage(
                cached_layout, value, index=cached_index, _trusted_index=True
            ),
        )
    index = TensorIndex(
        tensor_format=tensor_format,
        mode_indices=_finalize_generated_mode_indices(
            tensor_format, kernel_mode_indices
        ),
        mode_order=mode_order,
        _adopt=adopt,
    )
    # Build the storage here rather than letting STensor do it, because this is the one
    # construction site that may declare the index trusted: these arrays came out of our
    # own codegen microseconds ago. `SparseStorage` then skips the O(nnz) structural walk
    # unless `storage._VALIDATE_KERNEL_RESULTS` is on, which the test suite turns on. Going
    # through `storage=` keeps that decision local -- STensor's public signature does not
    # grow a way for a caller to declare their own arrays trusted.
    layout = TensorLayout.from_physical_shape(
        shape, index.format, index.mode_order, index.index_dtype
    )
    storage = SparseStorage(layout, value, index=index, _trusted_index=True)
    if name is None:
        return STensor(shape=shape, storage=storage)
    return STensor(name=name, shape=shape, storage=storage)


class Window(object):
    """A tensor window object that describes the slice into a physical storage (TensorStorage)
    or another logical tensor (Tensor)
    Contains:
        - an offset for the starting coordinate of the window
        - a shape tuple for the shape of the window
        - a step tuple for the step of the window
    """

    def __init__(self, offset: Tuple[int], shape: Tuple[int], step: Tuple[int]):
        self.offset = offset
        self.shape = shape
        self.step = step

    def __str__(self):
        return f"Window(offset={self.offset}, shape={self.shape}, step={self.step})"

    def __repr__(self):
        return f"Window(offset={self.offset}, shape={self.shape}, step={self.step})"

    def __copy__(self):
        return Window(deepcopy(self.offset), deepcopy(self.shape), deepcopy(self.step))


# --------------------------------------------------------------------------- #
# Dense from_torch
#
# Everything about a dense operand except its values is a function of
# (shape, dtype, device, name, mode_order) alone: the format, the per-mode index arrays
# (all empty), the layout, and the metadata. Rebuilding them on every call is the single
# largest cost in a small matmul -- wrapping the dense right-hand operand was ~51% of
# `scorch.matmul` on a 64x64 SpMM, all of it discarded when the call returned.
#
# So build them once per key with the ordinary constructor and reuse them. The cached
# parts are whatever the real path produced, not a reimplementation of it, and a test
# asserts the fast and ordinary paths agree field by field.
#
# Sharing is safe because each cached part is an immutable value object: TensorLayout,
# TensorMetadata and TensorFormat are frozen dataclasses whose only object.__setattr__
# calls are inside their own __post_init__, and a dense tensor's mode-index arrays are
# empty tuples.
# --------------------------------------------------------------------------- #
_DENSE_PARTS_CACHE: dict = {}
_DENSE_PARTS_CACHE_MAX = 512


# --------------------------------------------------------------------------- #
# The copy a non-contiguous dense operand needs
#
# `SparseStorage` holds a flat contiguous values array, so a dense operand that is a
# transposed view -- `W.T`, `x.permute(1, 0)`, a strided slice -- must be materialized.
# For a contiguous operand `.reshape(-1)` is a view and the STensor shares the caller's
# buffer, which is what it has always done; for a non-contiguous one it is a full copy,
# paid again on every call even when the operand has not changed since the last one.
#
# So remember it. The entry is keyed on the identity of the *base* tensor rather than the
# view, because the view is a fresh object per call (`W.T` inside a loop) while its base is
# the parameter that persists. Three things must hold for a hit, and the first is why a
# reused allocator address cannot fool it:
#
#   * the weak reference still resolves to the same base object,
#   * the base's version counter is unchanged -- torch shares one counter between a tensor
#     and every view of it, so any in-place torch write through any of them is a miss,
#   * the copy's own counter is unchanged, because the copy is handed out as the STensor's
#     values and a write through those would otherwise be served to the next caller.
#
# Two STensors built from the same unmodified operand therefore share one values buffer,
# where before they held separate copies. That is not a new kind of sharing: an STensor over
# a *contiguous* operand has always shared the caller's buffer, and the result of a kernel
# shares its values through `detach`. It does mean a write through one sibling's `.values`
# is visible in the other until the next lookup notices the version move.
#
# What none of that sees is a write through a raw pointer or a numpy view, which bumps no
# counter. Nothing in Scorch sees those -- index validation has the same blind spot, and so
# does autograd -- but here the consequence is a stale *value* rather than a skipped check,
# so it is said plainly: an operand mutated behind torch's back and then multiplied again
# reads as unchanged. `SCORCH_MEMO_OPERAND_COPY=0` turns this off in the same binary, which
# is also how it is measured.
#
# The retained copies are the other cost. Peak memory during a call does not change -- the
# copy was always being made -- but steady state grows by one copy per live memoized
# operand. Hence a small bound, and a sweep of entries whose base has died, so the bound
# counts copies that are still reachable rather than corpses.
#
# A memo that cannot hit is a tax on every call, and `plan.py` met this exact problem
# first: "a plan that declines costs the call it declined ... which is above the noise
# floor and therefore a regression on any call site that repeats a product a plan cannot
# serve." Measured here, an operand refilled in place every call -- a dataloader's buffer --
# paid 1.11x on the smallest cell for a lookup that could never pay. So the same answer:
# after enough consecutive misses with nothing served, stop consulting, which returns such a
# call site to exactly what it cost before this existed.
#
# What is counted is the *stale* miss -- the key was present and the version had moved --
# and not the cold miss, where the key was simply new. That distinction is the whole signal.
# A changing operand produces a stale miss on every call; a program with more distinct
# stable operands than the bound, or a deep model on its first forward, produces cold
# misses and then hits. Counting all misses alike would withdraw the memo from the second
# case, which is the case it exists for.
# --------------------------------------------------------------------------- #
_MEMO_OPERAND_COPY = [
    os.environ.get("SCORCH_MEMO_OPERAND_COPY", "1") not in ("0", "false", "False")
]
_OPERAND_COPY_CACHE: dict = {}
_OPERAND_COPY_CACHE_MAX = 16
# Eight is enough because the streak resets on any hit, so this only fires when eight
# stale misses arrive with nothing served in between. `plan.py` relies on the same
# property: "a plan that helps most calls and declines the odd one is never touched." A
# weight updated once per step and multiplied ten times within it never withdraws. The
# trade is that the withdrawal is permanent and process-wide, so a program whose first
# eight stale misses all precede its first hit loses the memo for good -- which costs it
# the gain, not correctness, and is the direction to err in.
_OPERAND_COPY_GIVE_UP = 8
# [consecutive stale misses, insert attempts left before the next sweep]
_OPERAND_COPY_STATE = [0, _OPERAND_COPY_CACHE_MAX]


# Resolved once. An editable install whose extension has not been rebuilt will not have
# the entry point and must keep building tensors, so its absence is a missing optimization
# rather than an error -- the same arrangement as storage.py's validation screens.
try:  # pragma: no cover - exercised by whichever branch this import takes
    import scorch_ops as _native_ops

    _NATIVE_TRANSPOSE = getattr(_native_ops, "scorch_transpose_2d_float", None)
except Exception:
    _NATIVE_TRANSPOSE = None


def _contiguous_copy(tensor: torch.Tensor) -> torch.Tensor:
    """``tensor.contiguous()``, using the cache-blocked kernel when that is what it is.

    A transposed 2-D float32 operand is column-major, and `.contiguous()` on one runs
    torch's element-scatter, which is several times below memory bandwidth. Scorch already
    ships a cache-blocked transpose for exactly that shape (AVX2 8x8 / NEON 4x4, also
    reachable as `scorch.fast_transpose`), and transposing the row-major view of a
    column-major matrix *is* its contiguous copy -- the same floats in the same order, bit
    for bit. Every other layout, dtype and rank still takes `.contiguous()`.

    **Serial deliberately**, by passing the kernel's "no thread override". Not because a
    threaded transpose is slow -- on a 2000x256 operand on a 32-core x86 it takes 19 us
    against 73 us serial and 188 us for `.contiguous()` -- but because of what it does to
    the kernel that runs next. Measured on that host, materializing this operand and then
    running scorch's SpMM on it costs 321 us with torch's copy, 182 us with the serial
    kernel, and **2681 us with the threaded one**, where the two pieces alone are 19 + 47.
    Two threaded transposes back to back cost 66 us, and a threaded transpose followed by
    `torch.sparse.mm` costs 423 against 361 for the pieces, so it is not the transpose and
    it is not threading as such: it is an ATen parallel region opening immediately before
    scorch's own team. This is the same neighbour effect the validation screens ran into
    (see `csrc/native_abi.h`), an order of magnitude larger.

    The serial path has no region to leave behind, and it still beats `.contiguous()` on
    39 of the 40 shapes in the measured grid -- 2.0-7.8x on the small ones, and never worse
    than 0.95x. What it gives up is the threaded win on very large operands (up to 2-3x of
    the copy itself at 15M elements), where the neighbour cost is a smaller share of a much
    bigger copy and threading might well pay. That crossover is not chased here: it would
    mean picking a second threshold against an interaction that is measured but not
    explained.
    """
    if (
        _NATIVE_TRANSPOSE is not None
        and tensor.dim() == 2
        and tensor.dtype == torch.float32
        and tensor.t().is_contiguous()
    ):
        return _NATIVE_TRANSPOSE(tensor.t(), -1)
    return tensor.contiguous()


def _remember_operand_copy(key, base: torch.Tensor, copy: torch.Tensor) -> None:
    """Install an entry, dropping dead ones so the bound counts live copies.

    The sweep runs on a countdown rather than on every full-cache insert: walking sixteen
    weak references per call is itself the tax this is trying to avoid, and a copy whose
    base died a few calls ago costs nothing but the memory it is about to release.
    """
    if len(_OPERAND_COPY_CACHE) >= _OPERAND_COPY_CACHE_MAX:
        _OPERAND_COPY_STATE[1] -= 1
        if _OPERAND_COPY_STATE[1] > 0:
            return
        _OPERAND_COPY_STATE[1] = _OPERAND_COPY_CACHE_MAX
        for dead in [k for k, e in _OPERAND_COPY_CACHE.items() if e[0]() is None]:
            del _OPERAND_COPY_CACHE[dead]
        if len(_OPERAND_COPY_CACHE) >= _OPERAND_COPY_CACHE_MAX:
            return
    try:
        held = weakref.ref(base)
    except TypeError:  # not weak-referenceable, so there is nothing to hold on to
        return
    _OPERAND_COPY_CACHE[key] = (held, tensor_version(base), copy, tensor_version(copy))


def _flat_contiguous_values(tensor: torch.Tensor) -> torch.Tensor:
    """The flat contiguous values array for a dense operand, copying only when needed."""
    if tensor.is_conj() or tensor.is_neg():
        # Resolving produces a tensor whose relation to the caller's base is no longer the
        # one the key describes, so this rare shape is left alone.
        return tensor.resolve_conj().resolve_neg().contiguous().reshape(-1)
    if tensor.is_contiguous():
        return tensor.reshape(-1)
    if not _MEMO_OPERAND_COPY[0] or _OPERAND_COPY_STATE[0] >= _OPERAND_COPY_GIVE_UP:
        return _contiguous_copy(tensor).reshape(-1)
    base = tensor._base if tensor._base is not None else tensor
    key = (
        id(base),
        tensor.shape,
        tensor.stride(),
        tensor.storage_offset(),
        tensor.dtype,
    )
    entry = _OPERAND_COPY_CACHE.get(key)
    if entry is not None:
        held, base_version, remembered, copy_version = entry
        if (
            held() is base
            and tensor_version(base) == base_version
            and tensor_version(remembered) == copy_version
        ):
            _OPERAND_COPY_STATE[0] = 0
            return remembered
        # The key was here and no longer answers: this operand is being changed under us.
        _OPERAND_COPY_STATE[0] += 1
        if _OPERAND_COPY_STATE[0] >= _OPERAND_COPY_GIVE_UP:
            # Withdrawing and still holding the copies is the worst of both: retained
            # blocks keep torch's caching allocator from handing the same warm one back
            # for the copy about to be made, which measured as a further 1.6-5% on the
            # small cells. Release them, and do not install this call's copy either --
            # every later call takes the early return above.
            _OPERAND_COPY_CACHE.clear()
            return _contiguous_copy(tensor).reshape(-1)
    copy = _contiguous_copy(tensor).reshape(-1)
    _remember_operand_copy(key, base, copy)
    return copy


def _dense_from_torch(
    tensor: torch.Tensor,
    name: Optional[str],
    mode_order: List[int],
    rank: int,
) -> "STensor":
    """Dense ``STensor`` over ``tensor``; the tail of :meth:`STensor.from_torch`.

    The caller has already checked that ``tensor`` is a strided CPU tensor and applied
    any permutation, so the value below is contiguous, flat, 1-D and free of lazy view
    bits by construction -- which is why the assembly skips re-checking those.
    """
    key = (tuple(tensor.shape), tensor.dtype, tensor.device, name, tuple(mode_order))
    parts = _DENSE_PARTS_CACHE.get(key)
    value = _flat_contiguous_values(tensor)
    if parts is None:
        built = STensor(
            name=name,
            shape=tuple(tensor.shape),
            index=TensorIndex(
                tensor_format="d" * rank,
                mode_indices=[[] for _ in range(rank)],
                mode_order=mode_order,
            ),
            value=value,
        )
        if len(_DENSE_PARTS_CACHE) < _DENSE_PARTS_CACHE_MAX:
            storage = built._storage
            _DENSE_PARTS_CACHE[key] = (
                storage.layout,
                storage._mode_indices,
                built._metadata,
            )
        return built

    # Assembled without re-validating, because validation is a function of the key that
    # was already validated when this entry was built. `_set_state` compares
    # metadata.layout against storage.layout (the same object here), metadata.dtype and
    # .device against the value's (both in the key), and `storage.validate()` checks the
    # value length against the layout -- and the length is product(shape), also in the
    # key. Nothing left to check that could differ between two tensors sharing a key.
    layout, mode_indices, metadata = parts
    storage = object.__new__(SparseStorage)
    object.__setattr__(storage, "_layout", layout)
    object.__setattr__(storage, "_mode_indices", mode_indices)
    object.__setattr__(storage, "_value", value.detach())
    built = object.__new__(STensor)
    built._metadata = metadata
    built._storage = storage
    return built


class STensor:
    """A sparse tensor stored in a custom, per-mode layout.

    ``STensor`` is Scorch's user-facing sparse tensor. It is a thin *logical*
    handle: one immutable :class:`~scorch.layout.TensorMetadata` owns its name,
    dtype, device, and :class:`~scorch.layout.TensorLayout`; numeric payload and
    structural indices live in a validated
    :class:`~scorch.storage.SparseStorage`. The layout's
    :class:`~scorch.format.TensorFormat` has one
    :class:`~scorch.format.LevelType` per physical mode (dense, compressed, or
    coordinate).

    Users almost never construct an ``STensor`` directly. Build one from a torch
    tensor with the factories :meth:`from_torch`, :meth:`from_csr`, or
    :meth:`from_coo` (also re-exported at module scope as ``scorch.from_torch``,
    ``scorch.from_csr``, ``scorch.from_coo``), or use
    ``scorch.from_components`` for explicit storage. Exit back to PyTorch with
    :meth:`to_torch`. Matmul is the top-level function ``scorch.matmul(a, b)`` /
    ``scorch.einsum(...)`` — ``STensor`` deliberately defines no ``__matmul__``,
    so ``a @ b`` will not work.

    Notes
    -----
    This is a plain Python class, not an ``nn.Module``. Because the payload lives
    in ``self._storage`` and never as a direct tensor attribute, ``nn.Module``
    registered nothing useful — it only added per-instance
    ``__init__``/``__setattr__``/``isinstance`` overhead that dominated matmul
    latency on small matrices. STensors are transient data, are never registered
    as submodules, and carry no autograd, so dropping ``nn.Module`` is
    behaviour-preserving. ``requires_grad`` is stored but inert (no autograd).

    Scorch is a CPU compiler library. ``repr(stensor)`` is uninformative (it
    always prints ``"Tensor"``); inspect a tensor via :attr:`shape`,
    ``str(x.format)`` (e.g. ``"d,s"``), :attr:`values`, and
    ``x.index.mode_indices`` instead.

    Examples
    --------
    >>> import torch
    >>> import scorch
    >>> t = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    >>> a = scorch.from_torch(t, "A")     # dense STensor, format "d,d"
    >>> a.shape
    (3, 4)
    >>> torch.equal(a.to_torch(), t)
    True

    See Also
    --------
    from_torch : Build an STensor from a dense/COO/CSR torch tensor.
    to_torch : Materialize back to a dense ``torch.Tensor``.
    """

    _metadata: TensorMetadata
    _storage: SparseStorage

    def __init__(
        self,
        name: Optional[str] = None,
        shape: Optional[Tuple[int, ...]] = None,
        storage: Optional[SparseStorage] = None,
        index: Optional[TensorIndex] = None,
        value: Optional[torch.Tensor] = None,
        requires_grad: Optional[bool] = False,
    ) -> None:
        tensor_name = "tensor" if name is None else name
        if not isinstance(requires_grad, bool):
            raise TensorTypeError("requires_grad must be a bool")
        if storage is not None:
            if index is not None or value is not None:
                raise TensorStorageError(
                    "storage cannot be combined with separate index or value arguments"
                )
            if not isinstance(storage, SparseStorage):
                raise TensorTypeError("storage must be a SparseStorage")
            if shape is not None:
                if isinstance(shape, (str, bytes)) or not isinstance(shape, Sequence):
                    raise TensorTypeError("shape must be a sequence of integers")
                declared_shape = tuple(shape)
                if declared_shape != storage.layout.physical_shape:
                    raise TensorLayoutError(
                        f"shape {declared_shape} does not match storage physical "
                        f"shape {storage.layout.physical_shape}"
                    )
            runtime_storage = storage
        else:
            missing = [
                field
                for field, item in (
                    ("shape", shape),
                    ("index", index),
                    ("value", value),
                )
                if item is None
            ]
            if missing:
                raise TensorValidationError(
                    "runtime STensor construction requires shape, index, and value; "
                    f"missing {', '.join(missing)}. Use TensorSpec for compile-only tensors."
                )
            if not isinstance(index, TensorIndex):
                raise TensorTypeError("index must be a TensorIndex")
            if not isinstance(value, torch.Tensor):
                raise TensorTypeError("value must be a torch.Tensor")
            if shape is None:
                raise TensorLayoutError("shape is required for runtime storage")
            layout = TensorLayout.from_physical_shape(
                shape, index.format, index.mode_order, index.index_dtype
            )
            runtime_storage = SparseStorage(layout, value, index=index)
        metadata = TensorMetadata(
            tensor_name,
            runtime_storage.value.dtype,
            runtime_storage.value.device,
            runtime_storage.layout,
            requires_grad,
        )
        self._set_state(metadata, runtime_storage)

    @classmethod
    def _from_validated(
        cls, metadata: TensorMetadata, storage: SparseStorage
    ) -> "STensor":
        """Internal constructor for already assembled value objects."""
        tensor = object.__new__(cls)
        tensor._set_state(metadata, storage)
        return tensor

    def _set_state(
        self,
        metadata: TensorMetadata,
        storage: SparseStorage,
        *,
        recheck_storage: bool = False,
    ) -> None:
        """Adopt ``metadata``/``storage`` after checking they agree and are sound.

        ``recheck_storage`` forces the O(nnz) structural walk even when the storage
        already carries a passing verdict for these exact arrays. Only the public
        :meth:`validate` sets it: every other caller is handing over a storage that
        ``SparseStorage.__init__`` validated moments earlier, and walking it again is
        pure duplicated work (see ``validate_unless_already_checked``).
        """
        if not isinstance(metadata, TensorMetadata):
            raise TensorTypeError("metadata must be TensorMetadata")
        if not isinstance(storage, SparseStorage):
            raise TensorTypeError("storage must be SparseStorage")
        if metadata.layout != storage.layout:
            raise TensorLayoutError(
                "metadata and storage must reference the same layout"
            )
        if metadata.dtype != storage.value.dtype:
            raise TensorStorageError(
                f"metadata dtype {metadata.dtype} does not match values dtype "
                f"{storage.value.dtype}"
            )
        if metadata.device != storage.value.device:
            raise TensorStorageError(
                f"metadata device {metadata.device} does not match values device "
                f"{storage.value.device}"
            )
        if recheck_storage:
            storage.validate()
        else:
            storage.validate_unless_already_checked()
        self._metadata = metadata
        self._storage = storage
        # Every in-place structural change funnels through here (insert,
        # change_mode_order, to_sparse, ...), so this is the one place a cached
        # call plan describing the old structure has to be dropped. See plan.py;
        # a plan would also decline on its own, having recorded the index arrays'
        # identity and version, but dropping it releases the narrowed copy too.
        state = self.__dict__
        if _PLANS_ATTR in state or _PLAN_SEEN_ATTR in state:
            state.pop(_PLANS_ATTR, None)
            state.pop(_PLAN_SEEN_ATTR, None)
            state.pop(_PLAN_DECLINES_ATTR, None)

    def __getstate__(self) -> dict:
        """The state to copy or serialize: everything except the plan cache.

        A cached call plan lives in the native extension and cannot be pickled, so
        without this an operand that had been multiplied twice could no longer be
        deep-copied or pickled -- both work on an operand that has not. Dropping the
        cache is also the right answer independent of that: a plan is memoized work,
        not state, and the copy will build its own on its second use.

        Python routes ``pickle``, ``copy.deepcopy`` and ``copy.copy`` through
        ``__reduce_ex__``, which consults this, so one method covers all three.
        """
        state = dict(self.__dict__)
        state.pop(_PLANS_ATTR, None)
        state.pop(_PLAN_SEEN_ATTR, None)
        state.pop(_PLAN_DECLINES_ATTR, None)
        return state

    def insert(self, indices, values):
        """Insert values into the tensor.

        Parameters
        ----------
        indices : array-like
            Coordinates at which to insert.
        values : array-like
            Values to insert at ``indices``.

        Raises
        ------
        NotImplementedError
            Always. Build a new tensor with a factory instead.
        """
        raise NotImplementedError("STensor insertion is not implemented")

    def _nnz(self):
        """Get the number of non-zero elements in the tensor."""
        return self.values.numel()

    @property
    def has_index(self) -> bool:
        """Whether the tensor's storage carries a sparsity index.

        Returns
        -------
        bool
            ``True`` if the underlying storage has a
            :class:`~scorch.storage.TensorIndex` (format + coordinates),
            ``False`` for a value-only storage. Delegates to
            ``self.storage.has_index``.
        """
        return self.storage.has_index

    @property
    def name(self) -> str:
        """The tensor's name.

        Returns
        -------
        str
            The name assigned at construction (the factories default it to
            ``"tensor"``).

        Runtime tensors always have a validated non-empty name; factories use
        ``"tensor"`` when none is supplied.
        """
        return self._metadata.name

    @name.setter
    def name(self, name: str) -> None:
        self._metadata = replace(self._metadata, name=name)

    @property
    def values(self) -> torch.Tensor:
        """The flat 1-D tensor of stored (nonzero) values.

        Returns
        -------
        torch.Tensor
            A 1-D tensor holding every stored value in physical order — the
            entire numeric payload. For a dense tensor this is the row-major
            flattening; for CSR/COO it is the ``nnz`` nonzeros. Equivalent to
            ``self.storage.value``.
        """
        return self.storage.value

    @property
    def _raw_values(self) -> torch.Tensor:
        """The stored value tensor, skipping the defensive copy :attr:`values` makes.

        ``values`` goes through ``storage.value``, which returns ``self._value.detach()``
        -- but ``_value`` was already detached when the storage was built, so that
        detach only allocates a second Python tensor object with identical contents and
        the same ``requires_grad=False``. Internal callers that read a dtype or hand the
        buffer to a native kernel therefore use this instead; a matmul was paying four
        of those allocations per call.
        """
        return self._storage._value

    @property
    def index(self) -> TensorIndex:
        """The sparsity index (format plus coordinate arrays).

        Returns
        -------
        TensorIndex
            The :class:`~scorch.storage.TensorIndex` describing structure: the
            :class:`~scorch.format.TensorFormat`, the ``mode_indices``
            (e.g. CSR ``[[], [crow, col]]`` or COO ``[[row], [col]]``), and the
            ``mode_order`` permutation. Equivalent to ``self.storage.index``.
        """
        return self.storage.index

    def _native_mode_indices(self) -> List[List[torch.Tensor]]:
        """Return trusted internal index handles for native kernel calls."""
        return self.storage._native_mode_indices()

    @property
    def format(self) -> TensorFormat:
        """The per-mode storage format.

        Returns
        -------
        TensorFormat
            The :class:`~scorch.format.TensorFormat`, one
            :class:`~scorch.format.LevelType` per mode. ``str(fmt)`` renders a
            comma-joined level string, e.g. ``"d,d"`` (dense), ``"d,s"`` (CSR),
            or ``"o,o"`` (COO).

        Runtime tensors always have a format whose rank matches their layout.
        """
        return self.layout.format

    @property
    def storage(self) -> SparseStorage:
        """The physical storage container.

        Returns
        -------
        SparseStorage
            The frozen :class:`~scorch.storage.SparseStorage` holding flat
            values, index arrays, and the canonical layout.
        """
        return self._storage

    @property
    def dtype(self):
        """The authoritative component dtype.

        Returns
        -------
        torch.dtype
            The metadata dtype, cross-validated against stored values.
        """
        return self._metadata.dtype

    @property
    def device(self) -> torch.device:
        """The authoritative runtime device."""
        return self._metadata.device

    @property
    def layout(self) -> TensorLayout:
        """The immutable logical-to-physical layout."""
        return self._metadata.layout

    @property
    def metadata(self) -> TensorMetadata:
        """The immutable runtime metadata value."""
        return self._metadata

    @property
    def logical_shape(self) -> Tuple[int, ...]:
        return self.layout.logical_shape

    @property
    def physical_shape(self) -> Tuple[int, ...]:
        return self.layout.physical_shape

    @property
    def index_dtype(self) -> torch.dtype:
        return self.layout.index_dtype

    @property
    def mode_order(self) -> Tuple[int, ...]:
        return self.layout.permutation

    @property
    def requires_grad(self) -> bool:
        return self._metadata.requires_grad

    @requires_grad.setter
    def requires_grad(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise TensorTypeError("requires_grad must be a bool")
        self._metadata = replace(self._metadata, requires_grad=value)

    @property
    def shape(self) -> Tuple[int, ...]:
        """The current physical shape (retained for API compatibility).

        Returns
        -------
        tuple of int
            Extents in physical storage-level order. Use
            :attr:`logical_shape` for the original logical axis order.
        """
        # Compatibility: historically ``shape`` is the current physical shape.
        return self.layout.physical_shape

    def __str__(self):
        """Get a string representation of the tensor."""
        # return f"TacoTensor_{self._name}({self._storage})"
        return "Tensor"

    def __repr__(self):
        """Get a string representation of the tensor."""
        return self.__str__()

    def validate(self) -> None:
        """Validate the tensor's internal consistency.

        Re-runs metadata/storage agreement and sparse storage invariant checks.
        Returns ``None`` when the tensor is valid and raises a Scorch domain
        exception otherwise. The structural checks are re-run in full, not taken
        from the verdict the storage already carries -- an explicit call asks for
        the work to be done.
        """
        self._set_state(self._metadata, self._storage, recheck_storage=True)

    def to(self, device):
        """Move the tensor to a device.

        Parameters
        ----------
        device
            Target device.

        Raises
        ------
        NotImplementedError
            Always. Scorch is a CPU-only compiler library; there is no device
            transfer.
        """
        raise NotImplementedError("STensor is CPU-only and does not support transfer")

    def cuda(self):
        """Move the tensor to the GPU.

        Raises
        ------
        NotImplementedError
            Always — this delegates to :meth:`to`, which is unimplemented.
            Scorch is CPU-only.
        """
        return self.to(torch.cuda.current_device())

    def clone(self):
        """Clone the tensor.

        Returns
        -------
        STensor
            An independent validated copy.

        See Also
        --------
        copy : The supported way to duplicate an ``STensor``.
        """
        return self.copy()

    def dim(self):
        """Number of tensor dimensions (order).

        Returns
        -------
        int
            ``len(self.shape)``.

        Notes
        -----
        ``STensor`` has no ``.ndim`` property; use this method instead.
        """
        return len(self.shape)

    def __add__(self, other) -> STensor:
        """Element-wise addition of two tensors (``self + other``).

        JIT-compiles and runs a codegen kernel that adds the two operands
        element-wise. ``other`` is first relaid out (via
        :meth:`change_mode_order`) to match ``self``'s ``mode_order``. The
        output takes ``self``'s format.

        Parameters
        ----------
        other : STensor
            The right-hand operand. Must have the same shape as ``self``.

        Returns
        -------
        STensor
            A new tensor holding the element-wise sum.

        Notes
        -----
        No broadcasting is performed and the output format is not inferred from
        the inputs (it is fixed to ``self.format``). The first call incurs a C++
        compile; compiled kernels are cached.

        Examples
        --------
        >>> import torch
        >>> import scorch
        >>> A = scorch.from_torch(torch.rand(4, 4), "A")
        >>> B = scorch.from_torch(torch.rand(4, 4), "B")
        >>> C = A + B
        >>> torch.allclose(C.to_torch(), A.to_torch() + B.to_torch(), atol=1e-3)
        True
        """
        if not isinstance(other, STensor):
            raise TensorTypeError("STensor addition requires another STensor")
        if self.logical_shape != other.logical_shape:
            raise TensorLayoutError(
                f"addition requires equal logical shapes, got "
                f"{self.logical_shape} and {other.logical_shape}"
            )
        # Scheduling must not mutate the caller's right-hand operand.
        if self.storage.index.mode_order != other.storage.index.mode_order:
            other = other.copy()
            other.change_mode_order(self.storage.index.mode_order)

        # Perform element-wise addition
        # TODO: support broadcasting
        a_index_vars = [IndexVar(f"i{i}") for i in self.storage.index.mode_order]
        index_vars = [IndexVar(f"i{i}") for i in range(len(self.shape))]
        # TODO: output format inferred from input formats
        output_format = self.format
        result_shape = self.shape

        A = TensorVar(
            name="A",
            fmt=output_format,
            shape=result_shape,
            dtype=self.dtype,
            mode_order=self.storage.index.mode_order,
        )
        B = TensorVar(
            name="B",
            fmt=self.format,
            shape=self.shape,
            dtype=self.dtype,
            mode_order=self.storage.index.mode_order,
        )
        C = TensorVar(
            name="C",
            fmt=other.format,
            shape=other.shape,
            dtype=other.dtype,
            mode_order=other.storage.index.mode_order,
        )

        # Generate the python code for the element-wise addition
        # e.g. A[i0, i1, ...] = B[i0, i1, ...] + C[i0, i1, ...]
        lhs = f'A[{", ".join(["index_vars[{i}]".format(i=i) for i in range(len(self.shape))])}]'
        rhs = f'B[{", ".join(["index_vars[{i}]".format(i=i) for i in range(len(self.shape))])}]'
        rhs += f' + C[{", ".join(["index_vars[{i}]".format(i=i) for i in range(len(self.shape))])}]'
        code = f"{lhs} = {rhs}"
        exec(code)

        # Generate the python code for constructing the ForAll's and execute it
        # e.g. cin_stmt = ForAll(i0, ForAll(i1, ForAll(i2, A._assignment)))
        rhs = "A._assignment"
        for i in range(len(self.shape))[::-1]:
            rhs = f"ForAll(a_index_vars[{i}], {rhs})"
        cin_stmt = eval(rhs)

        lowerer = CINLowerer()
        lowered_llir = lowerer.lower_IndexStmt(cin_stmt)
        llir_lowerer = LLIRLowerer()
        cpp_code = llir_lowerer.lower_llir(lowered_llir)

        # print("\n\ncpp_code:\n\n", cpp_code)

        header_cpp_code = jit_preamble_text()

        module = _load_kernel(
            name=_kernel_name(header_cpp_code, cpp_code),
            cpp_sources=[header_cpp_code, cpp_code],
            functions=["evaluate"],
            extra_cflags=get_extra_cflags(),
            extra_ldflags=get_extra_ldflags(),
        )

        result_cpp = module.evaluate(
            result_shape,
            self.shape,
            self._native_mode_indices(),
            self.storage.value,
            other.shape,
            other._native_mode_indices(),
            other.storage.value,
        )

        result = STensor(
            shape=result_shape,
            index=TensorIndex(
                mode_indices=_finalize_generated_mode_indices(
                    output_format, result_cpp.storage.index.mode_indices
                ),
                tensor_format=output_format,
                mode_order=self.storage.index.mode_order,
            ),
            value=result_cpp.storage.value,
        )

        return result

    def __mul__(self, other) -> STensor:
        """Element-wise multiplication of two tensors (``self * other``).

        Parameters
        ----------
        other : STensor
            The right-hand operand.

        Raises
        ------
        NotImplementedError
            Always. Element-wise multiply is not yet implemented; only
            :meth:`__add__` is available among the element-wise operators.

        See Also
        --------
        __add__ : Element-wise addition (implemented).
        """
        raise NotImplementedError()

    def copy(self) -> STensor:
        """Return a deep copy of the tensor.

        Duplicates the storage: the values tensor is cloned and every index
        array is cloned and detached, so the copy shares no state with the
        original. Name and shape are preserved.

        Returns
        -------
        STensor
            An independent copy.

        :meth:`clone` is an alias for this operation.
        """
        return STensor._from_validated(self.metadata, self.storage.copy())

    @classmethod
    def from_components(
        cls,
        shape: Sequence[int],
        tensor_format: Union[TensorFormat, str, List[str]],
        mode_indices: Sequence[Sequence[torch.Tensor]],
        values: torch.Tensor,
        *,
        name: Optional[str] = None,
        mode_order: Optional[Sequence[int]] = None,
        index_dtype: Optional[torch.dtype] = None,
        requires_grad: bool = False,
    ) -> STensor:
        """Build a fully validated runtime tensor from explicit components.

        ``shape`` and ``mode_order`` describe physical storage. The constructor
        derives logical shape, validates format rank and permutation, then checks
        all dense/COO/compressed storage invariants before returning.
        """
        if isinstance(shape, (str, bytes)) or not isinstance(shape, Sequence):
            raise TensorTypeError("shape must be a sequence of integers")
        index = TensorIndex(
            tensor_format=tensor_format,
            mode_indices=mode_indices,
            mode_order=mode_order,
            index_dtype=index_dtype,
        )
        return cls(
            name=name,
            shape=tuple(shape),
            index=index,
            value=values,
            requires_grad=requires_grad,
        )

    @staticmethod
    def from_csr(
        csr_matrix: torch.Tensor,
        name: Optional[str] = None,
    ) -> STensor:
        """Create an STensor from a PyTorch CSR matrix.

        Wraps a 2-D ``torch.sparse_csr`` tensor as a Scorch tensor with the
        canonical CSR layout: format ``"d,s"`` (dense rows, compressed cols),
        ``mode_indices = [[], [crow_indices, col_indices]]``, and the CSR values
        as the flat payload. Structural index arrays are copied into immutable
        storage; the values buffer remains an isolated tensor view of the input
        payload and is not silently cast.

        Parameters
        ----------
        csr_matrix : torch.Tensor
            A 2-D sparse tensor in CSR format (``is_sparse_csr`` must be True).
        name : str, optional
            Name for the tensor. Defaults to ``"tensor"``.

        Returns
        -------
        STensor
            A Scorch tensor in CSR (``"d,s"``) format.

        Raises a Scorch domain exception if the input is not a valid rank-2 CPU
        CSR tensor.

        Notes
        -----
        Unlike :meth:`from_torch`, this does not set an explicit ``mode_order``,
        so the index uses the identity permutation. For n-D sparse data use
        :meth:`from_coo`. Re-exported as ``scorch.from_csr``.

        Examples
        --------
        >>> import torch
        >>> import scorch
        >>> dense = torch.tensor([[0., 2., 0.], [1., 0., 3.]])
        >>> a = scorch.from_csr(dense.to_sparse_csr(), "W")
        >>> str(a.format)
        'd,s'
        >>> torch.equal(a.to_torch(), dense)
        True

        See Also
        --------
        from_torch : Auto-detects dense/COO/CSR inputs.
        from_coo : Build from COO (arbitrary rank).
        """
        if not isinstance(csr_matrix, torch.Tensor):
            raise TensorTypeError("from_csr expects a torch.Tensor")
        if csr_matrix.layout != torch.sparse_csr:
            raise TensorStorageError("from_csr expects a sparse CSR tensor")
        if csr_matrix.device.type != "cpu":
            raise TensorStorageError("Scorch only supports CPU CSR tensors")

        # Extract the crow_indices, col_indices, and values
        crow_indices = csr_matrix.crow_indices()
        col_indices = csr_matrix.col_indices()
        values = csr_matrix.values().resolve_conj().resolve_neg()
        shape = csr_matrix.size()

        if len(shape) != 2:
            raise TensorLayoutError("CSR format is only valid for rank-2 tensors")

        return STensor(
            name=name,
            shape=tuple(shape),
            index=TensorIndex(
                tensor_format="ds",
                mode_indices=[[], [crow_indices, col_indices]],
            ),
            value=values,
        )

    @staticmethod
    def from_coo(
        coo_matrix: Optional[torch.Tensor] = None,
        indices: Optional[torch.Tensor] = None,
        values: Optional[torch.Tensor] = None,
        shape: Optional[Tuple[int, ...]] = None,
        name: Optional[str] = None,
    ) -> STensor:
        """Create an STensor from COO data (arbitrary rank).

        Two calling conventions are supported:

        1. Pass ``coo_matrix`` — a ``torch.sparse_coo_tensor``. It is coalesced
           and its indices, values, and shape are read from it.
        2. Pass ``indices``, ``values``, and ``shape`` directly to build COO
           from raw arrays without a torch sparse tensor.

        Every mode is stored as ``LevelType.COORDINATE`` (format ``"o,o,..."``)
        with ``mode_indices[i] = [indices[i]]``. Works for any number of modes,
        unlike :meth:`from_csr` (2-D only).

        Parameters
        ----------
        coo_matrix : torch.Tensor, optional
            A sparse COO tensor. If given, ``indices``/``values``/``shape`` are
            derived from it (after coalescing) and need not be passed.
        indices : torch.Tensor, optional
            Coordinate array of shape ``[ndim, nnz]``. Used when ``coo_matrix``
            is not supplied.
        values : torch.Tensor, optional
            The ``nnz`` nonzero values, shape ``[nnz]``.
        shape : tuple of int, optional
            The logical shape. Required in the raw-arrays form.
        name : str, optional
            Name for the tensor. Defaults to ``"tensor"``.

        Returns
        -------
        STensor
            A Scorch tensor in COO (``"o,o,..."``) format.

        Notes
        -----
        The caller's indices are never mutated or narrowed. Raw COO input is
        coalesced into canonical int64 coordinates, and that dtype is recorded
        by :attr:`layout`. Re-exported as ``scorch.from_coo``.

        Examples
        --------
        >>> import torch
        >>> import scorch
        >>> i = torch.tensor([[0, 1, 1], [2, 0, 2]])
        >>> v = torch.tensor([3., 4., 5.])
        >>> coo = torch.sparse_coo_tensor(i, v, (2, 3)).coalesce()
        >>> a = scorch.from_coo(coo, name="S")
        >>> b = scorch.from_coo(indices=i, values=v, shape=(2, 3), name="S")
        >>> str(a.format)
        'o,o'
        >>> torch.equal(a.to_torch(), coo.to_dense())
        True
        """
        if coo_matrix is not None:
            if any(item is not None for item in (indices, values, shape)):
                raise TensorStorageError(
                    "coo_matrix cannot be combined with indices, values, or shape"
                )
            if not isinstance(coo_matrix, torch.Tensor):
                raise TensorTypeError("coo_matrix must be a torch.Tensor")
            if coo_matrix.layout != torch.sparse_coo:
                raise TensorStorageError("from_coo expects a sparse COO tensor")
            if coo_matrix.device.type != "cpu":
                raise TensorStorageError("Scorch only supports CPU COO tensors")
            if coo_matrix.dense_dim() != 0:
                raise TensorStorageError(
                    "hybrid COO tensors with dense value dimensions are unsupported"
                )
            coo_matrix = coo_matrix.coalesce()
            indices = coo_matrix.indices()
            values = coo_matrix.values().resolve_conj().resolve_neg()
            shape = tuple(coo_matrix.shape)
        else:
            missing = [
                field
                for field, item in (
                    ("indices", indices),
                    ("values", values),
                    ("shape", shape),
                )
                if item is None
            ]
            if missing:
                raise TensorStorageError(
                    "raw COO construction requires indices, values, and shape; "
                    f"missing {', '.join(missing)}"
                )
            if not isinstance(indices, torch.Tensor) or not isinstance(
                values, torch.Tensor
            ):
                raise TensorTypeError("COO indices and values must be torch tensors")
            if isinstance(shape, (str, bytes)) or not isinstance(shape, Sequence):
                raise TensorTypeError("COO shape must be a sequence of integers")
            shape = tuple(shape)
            # Normalize every extent before handing it to PyTorch so malformed
            # shapes are reported as Scorch domain exceptions.
            shape_layout = TensorLayout.from_physical_shape(
                shape,
                "o" * len(shape),
                index_dtype=torch.int64,
            )
            shape = shape_layout.physical_shape
            if indices.device.type != "cpu" or values.device.type != "cpu":
                raise TensorStorageError("Scorch only supports CPU COO tensors")
            if indices.dtype not in (torch.int32, torch.int64):
                raise TensorIndexError("COO indices must use int32 or int64")
            if indices.dim() != 2:
                raise TensorIndexError("COO indices must have shape [rank, nnz]")
            if values.dim() != 1:
                raise TensorStorageError("COO values must be one-dimensional")
            if indices.shape[0] != len(shape):
                raise TensorIndexError("COO index rank does not match shape rank")
            if indices.shape[1] != values.numel():
                raise TensorStorageError("COO coordinate and value counts must match")
            # Coalesce without ever modifying the caller's tensors. PyTorch's COO
            # builder canonicalizes coordinates to int64, which is recorded in
            # the resulting layout rather than silently narrowed to int32.
            try:
                canonical = torch.sparse_coo_tensor(
                    indices.to(torch.int64),
                    values.resolve_conj().resolve_neg(),
                    tuple(shape),
                    check_invariants=True,
                ).coalesce()
            except (RuntimeError, TypeError, ValueError, OverflowError) as error:
                raise TensorIndexError(f"invalid COO coordinates: {error}") from error
            indices = canonical.indices()
            values = canonical.values()
            shape = tuple(canonical.shape)

        if indices is None or values is None or shape is None:
            raise TensorStorageError("COO components were not initialized")
        mode_indices = [[indices[mode]] for mode in range(len(shape))]
        return STensor(
            name=name,
            shape=tuple(shape),
            index=TensorIndex(
                tensor_format="o" * len(shape),
                mode_indices=mode_indices,
            ),
            value=values,
        )

    @staticmethod
    def from_torch(
        tensor: torch.Tensor,
        name: Optional[str] = None,
        mode_order: Optional[List[int]] = None,
    ) -> STensor:
        """Create an STensor from a ``torch.Tensor``, auto-detecting layout.

        The primary constructor. Accepts a dense tensor, a ``torch.sparse_coo``
        tensor, or a 2-D ``torch.sparse_csr`` tensor and picks the Scorch format
        automatically:

        - **dense** input → every mode ``DENSE`` (format ``"d,d,..."``); values
          are the row-major flattening.
        - **sparse_coo** input → coalesced; every mode ``COORDINATE``
          (format ``"o,o,..."``).
        - **sparse_csr** input (2-D) → format ``"d,s"`` (canonical CSR).

        Parameters
        ----------
        tensor : torch.Tensor
            The source tensor (dense, COO, or CSR).
        name : str, optional
            Name for the tensor. Defaults to ``"tensor"``.
        mode_order : list of int, optional
            A permutation of the axes. If given, the input is first
            ``tensor.permute(*mode_order)`` and the permutation is recorded on
            the index (physical axis → logical axis), so :meth:`to_torch` can
            invert it. Defaults to the identity order.

        Returns
        -------
        STensor
            A Scorch tensor whose format matches the input layout.

        Notes
        -----
        ``mode_order`` is how Scorch represents a transposed/relaid-out operand
        without recomputing; most tensors use the identity order.

        Examples
        --------
        >>> import torch
        >>> import scorch
        >>> t = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        >>> a = scorch.from_torch(t, "A")
        >>> str(a.format)
        'd,d'
        >>> torch.equal(a.to_torch(), t)
        True

        See Also
        --------
        from_csr : Build specifically from a CSR matrix.
        from_coo : Build from COO indices/values.
        """
        if not isinstance(tensor, torch.Tensor):
            raise TensorTypeError("from_torch expects a torch.Tensor")
        if tensor.device.type != "cpu":
            raise TensorStorageError("Scorch only supports CPU tensors")
        if tensor.layout not in (
            torch.strided,
            torch.sparse_coo,
            torch.sparse_csr,
        ):
            raise TensorStorageError(f"unsupported torch tensor layout {tensor.layout}")
        rank = tensor.dim()
        identity_order = list(range(rank))
        if mode_order is None:
            mode_order = identity_order
        else:
            if isinstance(mode_order, (str, bytes)) or not isinstance(
                mode_order, Sequence
            ):
                raise TensorLayoutError("mode_order must be a sequence of integers")
            if any(
                isinstance(mode, bool) or not isinstance(mode, int)
                for mode in mode_order
            ):
                raise TensorLayoutError("mode_order entries must be integers")
            if len(mode_order) != rank or sorted(mode_order) != identity_order:
                raise TensorLayoutError(
                    f"mode_order must be a permutation of range({rank})"
                )
        mode_order = list(mode_order)
        if tensor.layout == torch.sparse_csr and mode_order != identity_order:
            raise TensorLayoutError(
                "CSR tensors only support the identity mode_order; convert to COO "
                "before applying a permutation"
            )
        if tensor.layout == torch.sparse_coo and tensor.dense_dim() != 0:
            raise TensorStorageError(
                "hybrid COO tensors with dense value dimensions are unsupported"
            )
        if mode_order != identity_order:
            tensor = tensor.permute(*mode_order)

        if tensor.is_sparse or tensor.is_sparse_csr:
            if tensor.layout == torch.sparse_coo:
                mode_indices = []
                tensor = tensor.coalesce()
                tensor_indices = tensor.indices()
                for i in range(tensor.dim()):
                    mode_indices.append([tensor_indices[i]])

                return STensor(
                    name=name,
                    shape=tuple(tensor.shape),
                    index=TensorIndex(
                        tensor_format="o" * rank,
                        mode_indices=mode_indices,
                        mode_order=mode_order,
                    ),
                    value=tensor.values().resolve_conj().resolve_neg(),
                )

            elif tensor.layout == torch.sparse_csr:
                crow_indices = tensor.crow_indices()
                col_indices = tensor.col_indices()
                values = tensor.values()
                shape = tensor.size()

                return STensor(
                    name=name,
                    shape=shape,
                    index=TensorIndex(
                        tensor_format="ds",
                        mode_indices=[[], [crow_indices, col_indices]],
                        mode_order=mode_order,
                    ),
                    value=values.resolve_conj().resolve_neg(),
                )
            raise TensorStorageError(f"unsupported sparse layout {tensor.layout}")

        return _dense_from_torch(tensor, name, mode_order, rank)

    def to_torch(self, in_place=True) -> torch.Tensor:
        """Materialize to a dense ``torch.Tensor`` (exit to PyTorch).

        Densifies the tensor (via :meth:`to_dense`), casts the values back to
        the logical :attr:`dtype`, reshapes to the tensor's shape, and — if the
        tensor has a non-identity ``mode_order`` — permutes by the inverse
        permutation so the result is returned in the original logical axis
        order. The result is always a dense ``torch.Tensor``.

        Parameters
        ----------
        in_place : bool, default True
            Forwarded to :meth:`to_dense`. When ``True`` the underlying storage
            may be replaced with the densified storage as a side effect.

        Returns
        -------
        torch.Tensor
            A dense tensor equal to the sparse tensor's contents.

        Examples
        --------
        >>> import torch
        >>> import scorch
        >>> t = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        >>> a = scorch.from_torch(t.to_sparse_csr(), "A")
        >>> torch.equal(a.to_torch(), t)
        True
        """
        # Get a dense Scorch tensor
        dense_tensor = self.to_dense(in_place=in_place)
        # Convert the dense Scorch tensor to a torch.Tensor
        # torch_tensor = dense_tensor.storage.value.clone().detach()
        torch_tensor = dense_tensor.storage.value
        if torch_tensor.dtype != self.dtype:
            torch_tensor = torch_tensor.type(self.dtype)
        # Reshape the torch.Tensor to the original shape
        torch_tensor = torch_tensor.reshape(dense_tensor.shape)

        # Permute back if tensor has non-default mode order
        default_mode_order = [i for i in range(self.dim())]
        if (
            self.storage.index.mode_order
            and self.storage.index.mode_order != default_mode_order
        ):
            # Compute inverse permutation
            inv_perm = [0] * len(self.storage.index.mode_order)
            for i, m in enumerate(self.storage.index.mode_order):
                inv_perm[m] = i
            torch_tensor = torch_tensor.permute(*inv_perm)

        return torch_tensor

    def to_dense(
        self,
        fmt: Optional[Union[TensorFormat, str, List[str]]] = None,
        in_place: bool = False,
    ) -> STensor:
        """Densify to an all-dense ``STensor`` (stays within Scorch).

        Returns an ``STensor`` (not a ``torch.Tensor``) whose format is
        all-``DENSE`` by default. If the tensor is already dense, returns
        ``self`` (when ``in_place``) or a copy. Otherwise Scorch JIT-compiles a
        C++ kernel to scatter the stored values into a dense buffer.

        Parameters
        ----------
        fmt : TensorFormat or str or list of str, optional
            Target format (e.g. ``"dd"`` or ``["dense", "dense"]``), parsed via
            ``parse_format``. If ``None`` the output is all-dense. Passing a
            non-dense ``fmt`` here is an under-specified path — this method is
            intended for densification.
        in_place : bool, default False
            When ``True`` the densified storage replaces ``self._storage`` and
            ``self`` is returned; otherwise a new ``STensor`` is returned.

        Returns
        -------
        STensor
            A dense Scorch tensor.

        Notes
        -----
        The first densification of a given shape/format incurs a C++ compile;
        compiled kernels are cached. To exit to PyTorch use :meth:`to_torch`.

        See Also
        --------
        to_sparse : The inverse (compress to a sparse format).
        to_torch : Materialize to a dense ``torch.Tensor``.
        """

        # If self is already dense at every level, return self
        if self.format.is_dense():
            if in_place:
                return self
            else:
                return self.copy()

        default_index_vars = [IndexVar(name) for name in ["i", "j", "k", "l", "m", "n"]]

        if len(self.shape) > len(default_index_vars):
            index_vars = [IndexVar(f"i{i}") for i in range(len(self.shape))]
        else:
            index_vars = default_index_vars[: len(self.shape)]

        # Permute index_vars by mode_order so ForAll nesting matches
        # the physical level order. Don't pass mode_order to TensorVars
        # because the permuted index_vars already reflect physical order;
        # get_sorted_index_vars() with identity mode_order will then
        # correctly map subscript position k to physical level k.
        if self.storage.index.mode_order:
            index_vars = [index_vars[i] for i in self.storage.index.mode_order]

        if self.has_index:
            B = TensorVar(
                name="B",
                fmt=self.format,
                shape=self.shape,
                dtype=self.dtype,
            )
        else:
            B = TensorVar(
                name="B",
                fmt=TensorFormat(
                    level_formats=[
                        LevelFormat(mode=LevelType.DENSE)
                        for _ in range(len(self.shape))
                    ]
                ),
                shape=self.shape,
                dtype=self.dtype,
            )

        if fmt is None:
            # TODO: infer output format from input format
            # For now, make every level COMPRESSED
            output_format = TensorFormat(
                level_formats=[
                    LevelFormat(mode=LevelType.DENSE) for _ in range(len(self.shape))
                ]
            )
        else:
            output_format = parse_format(fmt)

        A = TensorVar(
            name="A",
            fmt=output_format,
            shape=self.shape,
            dtype=self.dtype,
        )

        # Generate the python code for A[i0, i1, etc.] = B[i0, i1, etc.] and execute it
        lhs = f'A[{", ".join(["index_vars[{i}]".format(i=i) for i in range(len(self.shape))])}]'
        rhs = f'B[{", ".join(["index_vars[{i}]".format(i=i) for i in range(len(self.shape))])}]'
        code = f"{lhs} = {rhs}"
        exec(code)

        # Generate the python code for constructing the ForAll's and execute it
        # e.g. cin_stmt = ForAll(i0, ForAll(i1, ForAll(i2, A._assignment)))
        rhs = "A._assignment"
        for i in range(len(self.shape))[::-1]:
            rhs = f"ForAll(index_vars[{i}], {rhs})"
        cin_stmt = eval(rhs)

        lowerer = CINLowerer(filter_zeros=True)
        lowered_llir = lowerer.lower_IndexStmt(cin_stmt)
        llir_lowerer = LLIRLowerer()
        cpp_code = llir_lowerer.lower_llir(lowered_llir)

        # print("\n\ncpp_code:\n\n", cpp_code)

        header_cpp_code = jit_preamble_text()

        module = _load_kernel(
            name=_kernel_name(header_cpp_code, cpp_code),
            cpp_sources=[header_cpp_code, cpp_code],
            functions=["evaluate"],
            extra_cflags=get_extra_cflags(),
            extra_ldflags=get_extra_ldflags(),
        )

        result_cpp = module.evaluate(
            self.shape,
            self.shape,
            self._native_mode_indices(),
            self.storage.value,
        )

        new_tensor = STensor(
            name=self.name,
            shape=self.shape,
            index=TensorIndex(
                tensor_format=output_format,
                mode_indices=_finalize_generated_mode_indices(
                    output_format, result_cpp.storage.index.mode_indices
                ),
                mode_order=self.storage.index.mode_order,
            ),
            value=result_cpp.storage.value,
        )

        if in_place:
            self._set_state(new_tensor.metadata, new_tensor.storage)
            return self

        return new_tensor

    def to_sparse(
        self, fmt: Optional[Union[TensorFormat, str, List[str]]] = None
    ) -> STensor:
        """Compress to a sparse ``STensor``, mutating in place.

        Filters out zeros and stores the tensor in a sparse format. **This
        method always mutates ``self._storage`` in place and returns ``self``**
        (there is no ``in_place`` flag). By default every mode becomes
        ``COMPRESSED``.

        Parameters
        ----------
        fmt : TensorFormat or str or list of str, optional
            Target sparse format, parsed via ``parse_format``. If ``None`` the
            output is all-``COMPRESSED``.

        Returns
        -------
        STensor
            ``self``, with its storage replaced by the sparse form.

        Notes
        -----
        The 1-D case is special-cased and builds a single compressed level
        directly from ``torch.nonzero`` (no kernel compile). For rank ≥ 2 Scorch
        JIT-compiles a filter-zeros kernel (honoring the tensor's ``mode_order``);
        the first call of a given shape/format incurs a C++ compile.

        Examples
        --------
        >>> import torch
        >>> import scorch
        >>> a = scorch.from_torch(torch.tensor([[0., 5.], [0., 0.]]), "A")
        >>> _ = a.to_sparse("ss")   # both modes COMPRESSED; a mutated in place
        >>> str(a.format)
        's,s'

        See Also
        --------
        to_dense : The inverse (densify).
        """
        if len(self.shape) == 1:
            # Find indexes of non-zero elements in self.values, flatten them
            nonzero_indices = torch.nonzero(self.values).flatten()
            size = len(nonzero_indices)
            # Create a filtered value tensor that only contains non-zero elements
            filtered_values = self.values[nonzero_indices]
            new_tensor = STensor(
                name=self.name,
                shape=self.shape,
                index=TensorIndex(
                    tensor_format=TensorFormat(
                        level_formats=[LevelFormat(mode=LevelType.COMPRESSED)]
                    ),
                    mode_indices=[
                        [
                            torch.tensor(
                                [0, size],
                                dtype=nonzero_indices.dtype,
                                device=nonzero_indices.device,
                            ),
                            nonzero_indices,
                        ]
                    ],
                    mode_order=self.storage.index.mode_order,
                ),
                value=filtered_values,
            )
            self._set_state(new_tensor.metadata, new_tensor.storage)
        else:
            default_index_vars = [
                IndexVar(name) for name in ["i", "j", "k", "l", "m", "n"]
            ]
            if len(self.shape) > len(default_index_vars):
                index_vars = [IndexVar(f"i{i}") for i in range(len(self.shape))]
            else:
                index_vars = default_index_vars[: len(self.shape)]

            # Permute index_vars by mode_order for ForAll construction
            ordered_index_vars = [index_vars[i] for i in self.storage.index.mode_order]

            if self.has_index:
                B = TensorVar(
                    name="B",
                    fmt=self.format,
                    shape=self.shape,
                    dtype=self.dtype,
                    mode_order=self.storage.index.mode_order,
                )
            else:
                B = TensorVar(
                    name="B",
                    fmt=TensorFormat(
                        level_formats=[
                            LevelFormat(mode=LevelType.DENSE)
                            for _ in range(len(self.shape))
                        ]
                    ),
                    shape=self.shape,
                    dtype=self.dtype,
                    mode_order=self.storage.index.mode_order,
                )

            if fmt is None:
                # TODO: infer output format from input format
                # For now, make every level COMPRESSED
                output_format = TensorFormat(
                    level_formats=[
                        LevelFormat(mode=LevelType.COMPRESSED)
                        for _ in range(len(self.shape))
                    ]
                )
            else:
                output_format = parse_format(fmt)

            A = TensorVar(
                name="A",
                fmt=output_format,
                shape=self.shape,
                dtype=self.dtype,
                mode_order=self.storage.index.mode_order,
            )

            # Generate the python code for A[i0, i1, etc.] = B[i0, i1, etc.] and execute it
            lhs = f'A[{", ".join(["index_vars[{i}]".format(i=i) for i in range(len(self.shape))])}]'
            rhs = f'B[{", ".join(["index_vars[{i}]".format(i=i) for i in range(len(self.shape))])}]'
            code = f"{lhs} = {rhs}"
            exec(code)

            # Generate the python code for constructing the ForAll's and execute it
            # e.g. cin_stmt = ForAll(i0, ForAll(i1, ForAll(i2, A._assignment)))
            rhs = "A._assignment"
            for i in range(len(self.shape))[::-1]:
                rhs = f"ForAll(ordered_index_vars[{i}], {rhs})"
            cin_stmt = eval(rhs)

            # print("\n\ncin_stmt: ", cin_stmt)

            lowerer = CINLowerer(filter_zeros=True)
            lowered_llir = lowerer.lower_IndexStmt(cin_stmt)
            llir_lowerer = LLIRLowerer()
            cpp_code = llir_lowerer.lower_llir(lowered_llir)

            # print("to_sparse cpp_code:\n\n", cpp_code)

            header_cpp_code = jit_preamble_text()

            module = _load_kernel(
                name=_kernel_name(header_cpp_code, cpp_code),
                cpp_sources=[header_cpp_code, cpp_code],
                functions=["evaluate"],
                extra_cflags=get_extra_cflags(),
                extra_ldflags=get_extra_ldflags(),
            )

            result_cpp = module.evaluate(
                self.shape,
                self.shape,
                self._native_mode_indices(),
                self.storage.value,
            )

            # A generated result like any other: `module.evaluate` above just ran a
            # kernel whose output levels codegen allocated with `torch::empty`. It went
            # through the same four steps written out by hand until now, which also meant
            # it was the one such result still copying its index arrays and still being
            # walked in release.
            new_tensor = _wrap_generated_result(
                shape=self.shape,
                tensor_format=output_format,
                result_cpp=result_cpp,
                mode_order=self.storage.index.mode_order,
                name=self.name,
            )
            self._set_state(new_tensor.metadata, new_tensor.storage)

        return self

    def change_mode_order(self, mode_order: List[int]) -> STensor:
        """Relay out the tensor into a new logical mode order (transpose).

        Permutes the tensor's modes, updating storage and shape in place. A
        fast path handles the common 2-D core formats (``"d,d"``, ``"d,s"``,
        ``"o,o"``) without compiling a kernel; the general path compiles and
        executes a ``Where(producer, consumer)`` CIN, where the producer
        iterates in the old mode order and the consumer in the new one, with a
        multi-dimensional workspace as intermediate.

        Parameters
        ----------
        mode_order : list of int
            The new mode-order permutation. Must be a permutation of
            ``range(self.dim())``.

        Returns
        -------
        STensor
            ``self``, with updated storage and shape. If the requested order
            already matches, ``self`` is returned unchanged.

        Raises
        ------
        TensorLayoutError
            If ``mode_order`` is not a valid permutation matching the tensor
            order.

        Notes
        -----
        ``mode_order`` maps physical axis → logical axis. This is how Scorch
        represents a transposed operand without recomputing; the general
        (non-fast-path) route triggers a JIT C++ compile on first use.
        """
        dim = len(self.shape)
        if not isinstance(mode_order, (list, tuple)):
            raise TensorTypeError("mode_order must be a sequence of integers")
        if any(
            isinstance(mode, bool) or not isinstance(mode, int) for mode in mode_order
        ):
            raise TensorTypeError("mode_order entries must be integers")
        if len(mode_order) != dim or sorted(mode_order) != list(range(dim)):
            raise TensorLayoutError(f"mode_order must be a permutation of range({dim})")

        old_mode_order = (
            self.storage.index.mode_order[:]
            if self.storage.index.mode_order is not None
            else [i for i in range(dim)]
        )

        if old_mode_order == mode_order:
            return self

        # old_mode_order maps physical_axis -> logical_axis.
        # Compute inverse: logical_axis -> physical_axis.
        inv_old_mode_order = [0] * dim
        for physical_axis, logical_axis in enumerate(old_mode_order):
            inv_old_mode_order[logical_axis] = physical_axis

        # Convert shape from current physical layout to logical layout, then
        # remap to the target physical layout described by mode_order.
        logical_shape = tuple(self.shape[inv_old_mode_order[i]] for i in range(dim))
        result_shape = tuple(logical_shape[i] for i in mode_order)
        perm_old_to_new = [inv_old_mode_order[i] for i in mode_order]

        # Fast path for 2D tensors in core formats. This avoids lowering/compiling
        # a transpose kernel for the common matmul operands.
        fmt_str = str(self.format)
        if dim == 2 and fmt_str in {"d,d", "d,s", "o,o"}:

            def _coalesce_2d_coo(
                row: torch.Tensor, col: torch.Tensor, vals: torch.Tensor, num_cols: int
            ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                if row.numel() == 0:
                    return (
                        row.to(self.index_dtype),
                        col.to(self.index_dtype),
                        vals,
                    )

                row64 = row.to(torch.int64)
                col64 = col.to(torch.int64)
                key = row64 * int(num_cols) + col64
                perm = torch.argsort(key)
                row_sorted = row64[perm]
                col_sorted = col64[perm]
                vals_sorted = vals[perm]
                key_sorted = key[perm]

                unique_mask = torch.ones_like(key_sorted, dtype=torch.bool)
                if key_sorted.numel() > 1:
                    unique_mask[1:] = key_sorted[1:] != key_sorted[:-1]

                if torch.all(unique_mask).item():
                    return (
                        row_sorted.to(self.index_dtype),
                        col_sorted.to(self.index_dtype),
                        vals_sorted,
                    )

                segment_ids = torch.cumsum(unique_mask.to(torch.int64), dim=0) - 1
                unique_count = int(segment_ids[-1].item() + 1)
                reduced_vals = torch.zeros(
                    unique_count, dtype=vals_sorted.dtype, device=vals_sorted.device
                )
                reduced_vals.scatter_add_(0, segment_ids, vals_sorted)
                unique_positions = torch.nonzero(unique_mask, as_tuple=False).flatten()
                return (
                    row_sorted[unique_positions].to(self.index_dtype),
                    col_sorted[unique_positions].to(self.index_dtype),
                    reduced_vals,
                )

            mode_indices: Optional[List[List[torch.Tensor]]] = None
            values: Optional[torch.Tensor] = None

            if fmt_str == "d,d":
                dense = self.values.reshape(self.shape).permute(*perm_old_to_new)
                values = dense.contiguous().reshape(-1)
                mode_indices = [[], []]
            elif fmt_str == "o,o":
                old_coords = [
                    self.storage._mode_indices[0][0].to(torch.int64),
                    self.storage._mode_indices[1][0].to(torch.int64),
                ]
                new_row = old_coords[perm_old_to_new[0]]
                new_col = old_coords[perm_old_to_new[1]]
                coalesced_row, coalesced_col, coalesced_values = _coalesce_2d_coo(
                    new_row,
                    new_col,
                    self.values,
                    result_shape[1],
                )
                mode_indices = [
                    [coalesced_row],
                    [coalesced_col],
                ]
                values = coalesced_values
            else:
                crow_indices, col_indices = self.storage._mode_indices[1]
                row_counts = (crow_indices[1:] - crow_indices[:-1]).to(torch.int64)
                old_row = torch.repeat_interleave(
                    torch.arange(
                        self.shape[0], dtype=torch.int64, device=col_indices.device
                    ),
                    row_counts,
                )
                old_col = col_indices.to(torch.int64)
                old_coords = [old_row, old_col]
                new_row = old_coords[perm_old_to_new[0]]
                new_col = old_coords[perm_old_to_new[1]]
                coalesced_row, coalesced_col, coalesced_values = _coalesce_2d_coo(
                    new_row,
                    new_col,
                    self.values,
                    result_shape[1],
                )
                transposed_crow = torch.zeros(
                    result_shape[0] + 1,
                    dtype=self.index_dtype,
                    device=coalesced_row.device,
                )
                if coalesced_row.numel() > 0:
                    row_nnz = torch.bincount(
                        coalesced_row.to(torch.int64), minlength=result_shape[0]
                    )
                    transposed_crow[1:] = torch.cumsum(row_nnz, dim=0)
                mode_indices = [
                    [],
                    [
                        transposed_crow,
                        coalesced_col.to(self.index_dtype),
                    ],
                ]
                values = coalesced_values

            if mode_indices is None or values is None:
                raise TensorStorageError(
                    "mode-order conversion did not produce storage"
                )
            new_tensor = STensor(
                name=self.name,
                shape=result_shape,
                index=TensorIndex(
                    tensor_format=self.format,
                    mode_indices=mode_indices,
                    mode_order=mode_order[:],
                    index_dtype=self.index_dtype,
                ),
                value=values,
            )
            self._set_state(new_tensor.metadata, new_tensor.storage)
            return self

        default_index_vars = [IndexVar(name) for name in ["i", "j", "k", "l", "m", "n"]]
        if dim > len(default_index_vars):
            index_vars = [IndexVar(f"i{i}") for i in range(dim)]
        else:
            index_vars = default_index_vars[:dim]

        b_index_vars = [index_vars[i] for i in old_mode_order]
        a_index_vars = [index_vars[i] for i in mode_order]

        B = TensorVar(
            name="B",
            fmt=self.format,
            shape=self.shape,
            dtype=self.dtype,
            mode_order=old_mode_order[:],
        )

        A = TensorVar(
            name="A",
            fmt=self.format,
            shape=result_shape,
            dtype=self.dtype,
            mode_order=mode_order[:],
        )

        workspace = Workspace(
            name="wksp",
            dim=len(self.shape),
            mode_order=mode_order[:],
        )

        producer_stmt = TensorAssign(
            workspace[tuple(index_vars)],
            B[tuple(index_vars)],
        )

        for index_var in b_index_vars[::-1]:
            producer_stmt = ForAll(index_var, producer_stmt)

        consumer_stmt = TensorAssign(
            A[tuple(index_vars)],
            workspace[tuple(index_vars)],
        )

        for index_var in a_index_vars[::-1]:
            consumer_stmt = ForAll(index_var, consumer_stmt)

        cin_stmt = Where(
            producer=producer_stmt,
            consumer=consumer_stmt,
        )

        lowerer = CINLowerer(filter_zeros=True)
        lowered_llir = lowerer.lower_IndexStmt(cin_stmt)
        llir_lowerer = LLIRLowerer()
        cpp_code = llir_lowerer.lower_llir(lowered_llir)

        header_cpp_code = jit_preamble_text()

        module = _load_kernel(
            name=_kernel_name(header_cpp_code, cpp_code),
            cpp_sources=[header_cpp_code, cpp_code],
            functions=["evaluate"],
            extra_cflags=get_extra_cflags(),
            extra_ldflags=get_extra_ldflags(),
        )

        result_cpp = module.evaluate(
            result_shape,
            self.shape,
            self._native_mode_indices(),
            self.storage.value,
        )

        new_tensor = STensor(
            name=self.name,
            shape=result_shape,
            index=TensorIndex(
                tensor_format=self.format,
                mode_indices=_finalize_generated_mode_indices(
                    self.format, result_cpp.storage.index.mode_indices
                ),
                mode_order=mode_order[:],
            ),
            value=result_cpp.storage.value,
        )
        self._set_state(new_tensor.metadata, new_tensor.storage)

        return self
