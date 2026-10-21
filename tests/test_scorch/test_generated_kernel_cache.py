"""A warm conversion, addition or SpMV call runs its kept kernel and lowers nothing.

``STensor.to_dense``, ``to_sparse``, ``change_mode_order``, ``__add__`` and ``ops.spmv``
keep their generated kernels in ``scorch.stensor._GENERATED_KERNELS``, keyed by every
input their lowering reads plus the call's options key, under ``einsum``'s store and hit
rules.  Before, each warm call built its CIN, lowered it, emitted and hashed the C++ and
prepared a build request: about 12,000 Python calls and 12-14 ms for a 1024 x 1024
``to_torch`` on sapling, against about 1,900 calls on 864745d (job 82763).  These tests
pin, for each entry point and for the conversions ``einsum`` and ``matmul`` make inside a
call:

1. an identical second call hits: it builds nothing, makes no options snapshot and no
   environment parse, and is correct; an in-place call leaves the tensor, field by
   field, as the full path leaves it;
2. a second call that changes one input misses, its key differs from the first call's
   in exactly the positions that input enters, and it does what the full path does with
   the cache off: the same correct result, or the same exception;
3. a second call on another tensor of the same layout hits and reads its own arrays;
4. one negative control per row of 2: with those key positions left out, the second call
   hits the first call's kernel, and the hit check (``SCORCH_CHECK_KERNEL_CACHE_HITS``)
   then refuses it when the input changes the kernel, and passes it when it does not;
5. ``matmul`` makes at most one snapshot per call, every step that needs one gets that
   object, a warm call makes none, and the schedule a call resolves does not depend on
   which step resolves it first.

Results are read back with torch alone (``_logical``), not with a conversion, so a
reading never goes through the cache under test.
"""

from __future__ import annotations

import contextlib
import dataclasses
import os
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import pytest
import torch

import scorch
import scorch.ops as ops  # type: ignore[import-untyped]
import scorch.stensor as stensor_module  # type: ignore[import-untyped]
from scorch import STensor  # type: ignore[import-untyped]
from scorch.compiler import compile_options  # type: ignore[import-untyped]
from scorch.compiler.cin_analysis import (  # type: ignore[import-untyped]
    full_cin_verification,
)
from scorch.compiler.compile_options import (  # type: ignore[import-untyped]
    CompileOptions,
)
from scorch.compiler.scheduler import (  # type: ignore[import-untyped]
    Schedule,
    regblock_force,
    schedule_force,
)
from scorch.exceptions import (  # type: ignore[import-untyped]
    GeneratedKernelCacheError,
)
from scorch.format import LevelType, parse_format  # type: ignore[import-untyped]

F64, F32, I32, I64 = torch.float64, torch.float32, torch.int32, torch.int64


@dataclasses.dataclass(frozen=True)
class _Sum:
    """An expected value that is a sum, which a kernel may add in another order."""

    value: torch.Tensor


Case = Callable[[], Tuple[Any, Any]]


# -- what a test sees -------------------------------------------------------------------


class _Seen:
    """Builds, hit checks, snapshots and environment parses since the last ``reset``."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.builds: List[str] = []
        self.build_options: List[CompileOptions] = []
        self.einsum_options: List[CompileOptions] = []
        self.checks = 0
        self.snapshots = 0
        self.parses = 0


class _Recording(dict):
    """The kernel cache, recording every lookup as (key, hit)."""

    def __init__(self, keep: bool = True) -> None:
        super().__init__()
        self.keep = keep
        self.lookups: List[Tuple[tuple, bool]] = []

    def _stored(self, key: tuple) -> tuple:
        return key

    def get(self, key, default=None):  # type: ignore[override]
        found = super().get(self._stored(key), default)
        self.lookups.append((key, found is not None))
        return found

    def __setitem__(self, key, value) -> None:
        if self.keep:
            super().__setitem__(self._stored(key), value)

    def keys_of(self, operation: str) -> List[tuple]:
        return [key for key, _ in self.lookups if key[0] == operation]


class _KeyWithout(_Recording):
    """The cache with some key positions left out of every key (a negative control)."""

    def __init__(self, positions: Tuple[int, ...]) -> None:
        super().__init__()
        self.positions = positions

    def _stored(self, key: tuple) -> tuple:
        return tuple(
            "left out" if position in self.positions else part
            for position, part in enumerate(key)
        )


def _kept() -> _Recording:
    return stensor_module._GENERATED_KERNELS


@pytest.fixture(autouse=True)
def _nothing_kept_between_tests(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    compile_options.clear_build_snapshot_cache()
    monkeypatch.setattr(stensor_module, "_GENERATED_KERNELS", _Recording())
    monkeypatch.setattr(ops, "_einsum_dispatch_cache", {})
    # These tests count snapshots, which a hit check makes; the negative controls and
    # the check's own tests turn it on.
    monkeypatch.setattr(stensor_module, "_CHECK_GENERATED_KERNEL_HITS", [False])
    yield
    compile_options.clear_build_snapshot_cache()


@pytest.fixture
def seen(monkeypatch: pytest.MonkeyPatch) -> _Seen:
    record = _Seen()
    build = stensor_module._compile_generated_cin

    def counted_build(cin_stmt, options, context, filter_zeros):
        record.build_options.append(options)
        module = build(cin_stmt, options, context, filter_zeros)
        record.builds.append(module.__name__)
        return module

    monkeypatch.setattr(stensor_module, "_compile_generated_cin", counted_build)
    spmv_build = ops._compile_spmv_cin

    def counted_spmv_build(cin_stmt, options, context):
        record.build_options.append(options)
        module = spmv_build(cin_stmt, options, context)
        record.builds.append(module.__name__)
        return module

    monkeypatch.setattr(ops, "_compile_spmv_cin", counted_spmv_build)
    prepare = ops._prepare_generated_kernel_build

    def counted_prepare(header, cpp, options, context):
        record.einsum_options.append(options)
        return prepare(header, cpp, options, context)

    monkeypatch.setattr(ops, "_prepare_generated_kernel_build", counted_prepare)
    check = stensor_module._check_generated_kernel_hit

    def counted_check(*args):
        record.checks += 1
        return check(*args)

    monkeypatch.setattr(stensor_module, "_check_generated_kernel_hit", counted_check)
    from_environment = CompileOptions.from_environment.__func__
    parse = CompileOptions._parse_environment.__func__

    def counted_snapshot(cls, *args, **kwargs):
        record.snapshots += 1
        return from_environment(cls, *args, **kwargs)

    def counted_parse(cls, *args, **kwargs):
        record.parses += 1
        return parse(cls, *args, **kwargs)

    monkeypatch.setattr(
        CompileOptions, "from_environment", classmethod(counted_snapshot)
    )
    monkeypatch.setattr(
        CompileOptions, "_parse_environment", classmethod(counted_parse)
    )
    return record


# -- tensors built and read with torch alone -------------------------------------------


def _dense(shape: Sequence[int], seed: int, dtype: torch.dtype = F64) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    values = torch.rand(*shape, generator=generator, dtype=F64) + 0.5
    keep = torch.rand(*shape, generator=generator) < 0.4
    return (values * keep).to(dtype)


def _components(
    physical: torch.Tensor, fmt: str, index_dtype: torch.dtype
) -> Tuple[List[List[torch.Tensor]], torch.Tensor]:
    """``physical`` stored in ``fmt``: a compressed level keeps a child whose subtree
    holds a nonzero, a dense level keeps every child, a coordinate (COO) format keeps
    each nonzero.

    A coordinate array is a column of an int64 matrix; ``.to(torch.int64)`` returns that
    column itself, which is not contiguous and which ``from_components`` refuses, so
    each one is made contiguous.
    """

    if set(fmt) == {"o"}:
        stored = physical.nonzero()
        return (
            [
                [stored[:, level].to(index_dtype).contiguous()]
                for level in range(len(fmt))
            ],
            physical[tuple(stored.T)],
        )
    nonzero = physical != 0
    paths = torch.zeros((1, 0), dtype=I64)
    mode_indices: List[List[torch.Tensor]] = []
    for level, kind in enumerate(fmt):
        extent = physical.shape[level]
        parents = paths.shape[0]
        children = torch.cat(
            [
                paths.repeat_interleave(extent, dim=0),
                torch.arange(extent).repeat(parents).unsqueeze(1),
            ],
            dim=1,
        )
        if kind == "d":
            mode_indices.append([])
            paths = children
            continue
        below = nonzero
        for _ in range(physical.dim() - level - 1):
            below = below.any(dim=-1)
        keep = below[tuple(children.T)]
        positions = torch.zeros(parents + 1, dtype=I64)
        positions[1:] = torch.cumsum(keep.reshape(parents, extent).sum(dim=1), 0)
        paths = children[keep]
        mode_indices.append(
            [positions.to(index_dtype), paths[:, -1].to(index_dtype).contiguous()]
        )
    return mode_indices, physical[tuple(paths.T)]


def _tensor(
    shape: Sequence[int] = (6, 5),
    fmt: str = "ds",
    dtype: torch.dtype = F64,
    index_dtype: torch.dtype = I32,
    mode_order: Optional[Sequence[int]] = None,
    seed: int = 1,
    name: str = "A",
    from_torch: bool = False,
) -> Tuple[STensor, torch.Tensor]:
    """A tensor of logical ``shape`` stored in ``fmt``, and its logical values.

    ``from_torch`` makes the tensor ``STensor.from_torch`` makes instead: every level
    dense, and an index like every storage on this tree (``SparseStorage.has_index`` is
    always ``True``).
    """

    logical = _dense(shape, seed, dtype)
    if from_torch:
        return STensor.from_torch(logical, name), logical
    order = list(range(len(shape))) if mode_order is None else list(mode_order)
    physical = logical.permute(*order).contiguous()
    mode_indices, values = _components(physical, fmt, index_dtype)
    tensor = STensor.from_components(
        tuple(physical.shape),
        fmt,
        mode_indices,
        values,
        name=name,
        mode_order=order,
        index_dtype=index_dtype,
    )
    return tensor, logical


def _logical(tensor: Any) -> torch.Tensor:
    """A result's logical values, rebuilt from its stored arrays with torch alone."""

    if isinstance(tensor, torch.Tensor):
        return tensor
    kinds = tensor.format.get_level_types()
    shape = tuple(tensor.physical_shape)
    values = tensor.values
    if all(kind is LevelType.DENSE for kind in kinds):
        physical = values.reshape(shape)
    else:
        mode_indices = tensor._native_mode_indices()
        paths = torch.zeros((1, 0), dtype=I64)
        for level, kind in enumerate(kinds):
            if kind is LevelType.DENSE:
                extent = shape[level]
                paths = torch.cat(
                    [
                        paths.repeat_interleave(extent, dim=0),
                        torch.arange(extent).repeat(paths.shape[0]).unsqueeze(1),
                    ],
                    dim=1,
                )
            elif kind is LevelType.COMPRESSED:
                positions, coordinates = mode_indices[level]
                counts = (positions[1:] - positions[:-1]).to(I64)
                paths = torch.cat(
                    [
                        paths.repeat_interleave(counts, dim=0),
                        coordinates.to(I64).unsqueeze(1),
                    ],
                    dim=1,
                )
            else:
                coordinates = mode_indices[level][0].to(I64).unsqueeze(1)
                paths = (
                    coordinates
                    if paths.shape[1] == 0
                    else torch.cat([paths, coordinates], dim=1)
                )
        physical = torch.zeros(shape, dtype=values.dtype)
        physical[tuple(paths.T)] = values.reshape(-1)
    inverse = [0] * len(shape)
    for axis, mode in enumerate(tensor.mode_order):
        inverse[mode] = axis
    logical = physical.permute(*inverse)
    assert tuple(logical.shape) == tuple(tensor.logical_shape)
    return logical


def _fields(tensor: STensor) -> tuple:
    """Everything an in-place call leaves on the tensor."""

    indexed = tensor.has_index
    indices = tensor._native_mode_indices() if indexed else []
    return (
        str(tensor.format),
        tuple(tensor.mode_order),
        tuple(tensor.physical_shape),
        indexed,
        tensor.index_dtype if indexed else None,
        tensor.name,
        tensor.dtype,
        [[array.clone() for array in level] for level in indices],
        tensor.values.clone(),
    )


def _same_fields(left: tuple, right: tuple) -> bool:
    if left[:7] != right[:7] or len(left[7]) != len(right[7]):
        return False
    for mine, theirs in zip(left[7], right[7]):
        if len(mine) != len(theirs) or not all(
            a.dtype == b.dtype and torch.equal(a, b) for a, b in zip(mine, theirs)
        ):
            return False
    return left[8].dtype == right[8].dtype and torch.equal(left[8], right[8])


# -- a call and what it did -------------------------------------------------------------


def _outcome(case: Case) -> tuple:
    """("correct",), ("wrong", got, expected), or ("raises", type, code, message).

    Correct is equal, in dtype, shape and every value, except that a sum (``_Sum``) may
    differ by its rounding: 1e-12 in float64 and 1e-5 in float32.
    """

    try:
        result, expected = case()
    except Exception as error:  # the full path's own refusal, compared below
        return ("raises", type(error), getattr(error, "code", None), str(error))
    summed = isinstance(expected, _Sum)
    wanted = expected.value if summed else expected
    got = _logical(result)
    if got.dtype != wanted.dtype or got.shape != wanted.shape:
        return ("wrong", got, wanted)
    if torch.equal(got, wanted):
        return ("correct",)
    if summed:
        tolerance = 1e-12 if got.dtype == F64 else 1e-5
        if torch.allclose(got, wanted, rtol=tolerance, atol=tolerance):
            return ("correct",)
    return ("wrong", got, wanted)


def _same_refusal(got: tuple, reference: tuple) -> bool:
    return got[0] == "raises" and got[1:3] == reference[1:3]


def _hit_pair(case: Case, seen: _Seen) -> None:
    """The positive control: an identical second call hits every lookup, and builds,
    snapshots and parses nothing."""

    assert _outcome(case) == ("correct",)
    del _kept().lookups[:]
    seen.reset()
    assert _outcome(case) == ("correct",)
    lookups = _kept().lookups
    assert lookups and all(hit for _, hit in lookups), lookups
    assert seen.builds == [], "the second call lowered again"
    assert (seen.snapshots, seen.parses) == (0, 0), "the second call made options"


# What a row's input does, with that input's key positions left out of the key:
CHANGES_KERNEL = "changes the kernel"  # the hit check refuses the kept kernel
SAME_KERNEL = "same kernel"  # the full path builds the same kernel; the hit is correct
REFUSED = "refused"  # the full path refuses the call after its lookup
REFUSED_BEFORE_LOOKUP = "refused before lookup"  # the call never reaches the cache


def _changed_row(
    operation: str,
    first: Case,
    second: Case,
    positions: Tuple[int, ...],
    verdict: str,
    seen: _Seen,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Call 2 changes one input: its key differs in ``positions`` alone, it misses, and
    it does what the full path does."""

    monkeypatch.setattr(stensor_module, "_GENERATED_KERNELS", _Recording(keep=False))
    reference = _outcome(second)
    cache = _Recording()
    monkeypatch.setattr(stensor_module, "_GENERATED_KERNELS", cache)
    assert _outcome(first) == ("correct",)
    (first_key,) = cache.keys_of(operation)
    del cache.lookups[:]
    seen.reset()
    got = _outcome(second)
    second_keys = cache.keys_of(operation)
    if verdict == REFUSED_BEFORE_LOOKUP:
        assert second_keys == [] and reference[0] == "raises", (second_keys, reference)
        assert _same_refusal(got, reference), (got, reference)
        return
    (second_key,) = second_keys
    changed = tuple(
        position
        for position, (old, new) in enumerate(zip(first_key, second_key))
        if old != new
    )
    assert changed == positions, f"the key changed in {changed}, not {positions}"
    if verdict == REFUSED:
        assert reference[0] == "raises", reference
        assert _same_refusal(got, reference), (got, reference)
    else:
        assert reference == ("correct",), reference
        assert got == ("correct",), got
        assert seen.builds, "call 2 hit a kernel kept for call 1"


def _negative_control(
    first: Case,
    second: Case,
    positions: Tuple[int, ...],
    verdict: str,
    seen: _Seen,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With ``positions`` left out of every key, call 2 hits call 1's kernel, so its row's
    miss fails, and the hit check decides what the call then does."""

    monkeypatch.setattr(stensor_module, "_GENERATED_KERNELS", _Recording(keep=False))
    reference = _outcome(second)
    monkeypatch.setattr(stensor_module, "_GENERATED_KERNELS", _KeyWithout(positions))
    monkeypatch.setattr(stensor_module, "_CHECK_GENERATED_KERNEL_HITS", [True])
    assert _outcome(first) == ("correct",)
    seen.reset()
    got = _outcome(second)
    if verdict == REFUSED_BEFORE_LOOKUP:
        assert seen.checks == 0 and _same_refusal(got, reference), (got, reference)
        return
    assert seen.checks == 1, "call 2 did not hit: the left-out input is keyed elsewhere"
    if verdict == CHANGES_KERNEL:
        assert got[0] == "raises" and issubclass(got[1], GeneratedKernelCacheError), (
            "the hit check passed a kernel this input was declared to change; if the "
            f"kernel no longer reads it, the row's verdict is SAME_KERNEL: {got}"
        )
    elif verdict == SAME_KERNEL:
        assert got == ("correct",), (
            "the hit check refused a kernel this input was declared not to change; if "
            f"the kernel now reads it, the row's verdict is CHANGES_KERNEL: {got}"
        )
    else:
        assert _same_refusal(got, reference), (got, reference)


# -- the rows ---------------------------------------------------------------------------
#
# Each row is (name, what call 2 changes, the key positions it changes, verdict).  The
# positions are those of the keys built in stensor.py and ops.py; the layout (position 1,
# and 2 for the second operand of __add__ and spmv) holds format, mode order, rank,
# extents and index dtype.  The device is keyed too and has no row: a tensor off the CPU
# is refused when it is made (storage.py, TensorDeviceError).  ``to_dense`` and
# ``to_sparse`` key whether the source has an index (their build_cin branches on it) and
# have no row for it either: every storage on this tree has one
# (``SparseStorage.has_index`` is always ``True``).


@contextlib.contextmanager
def _environment(name: str, value: str) -> Iterator[None]:
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv(name, value)
        yield


_ROW_SCHEDULE = Schedule(loop_order=("i", "j"), tag="conversion-row")

# (name, context, verdict); the context is entered around call 2.  Each one changes the
# options key alone.
_OPTION_ROWS: Tuple[Tuple[str, Callable[[], Any], str], ...] = (
    # Instrumented kernels carry the -DSCORCH_TUNE_HOOKS flag and another name.
    (
        "tune hooks",
        lambda: _environment("SCORCH_JIT_TUNE_HOOKS", "1"),
        CHANGES_KERNEL,
    ),
    ("forced register blocking", lambda: regblock_force(True), SAME_KERNEL),
    ("full CIN verification", lambda: full_cin_verification(), SAME_KERNEL),
    # Plain CIN lowering refuses a requested schedule it would otherwise ignore.
    ("forced schedule", lambda: schedule_force(_ROW_SCHEDULE), REFUSED),
)


def _under(context: Callable[[], Any], case: Case) -> Case:
    def run() -> Tuple[Any, Any]:
        with context():
            return case()

    return run


def _to_dense(**spec: Any) -> Case:
    def run() -> Tuple[Any, Any]:
        tensor, logical = _tensor(**spec)
        return tensor.to_dense(in_place=False), logical

    return run


_TO_DENSE_ROWS = (
    ("source format", {"fmt": "ss"}, (1,), CHANGES_KERNEL),
    ("source mode order", {"mode_order": (1, 0)}, (1,), CHANGES_KERNEL),
    ("value dtype", {"dtype": F32}, (3,), CHANGES_KERNEL),
    ("index dtype", {"index_dtype": I64}, (1,), SAME_KERNEL),
    ("rank", {"shape": (4, 3, 5), "fmt": "dss"}, (1, 5), CHANGES_KERNEL),
    ("first extent", {"shape": (7, 5)}, (1,), CHANGES_KERNEL),
    ("second extent", {"shape": (6, 4)}, (1,), CHANGES_KERNEL),
)


def _to_sparse(target: str = "ds", **spec: Any) -> Case:
    spec = {"fmt": "dd", **spec}

    def run() -> Tuple[Any, Any]:
        tensor, logical = _tensor(**spec)
        return tensor.to_sparse(target), logical

    return run


# No row changes has_index (position 2; see above).  A from_torch source has the same
# layout and hits: test_a_from_torch_source_hits_the_kernel_of_a_built_source_of_its_layout.
_TO_SPARSE_ROWS = (
    ("source format", {"fmt": "ss"}, (1,), CHANGES_KERNEL),
    ("target format", {"target": "ss"}, (5,), CHANGES_KERNEL),
    ("source mode order", {"mode_order": (1, 0)}, (1,), CHANGES_KERNEL),
    ("value dtype", {"dtype": F32}, (3,), CHANGES_KERNEL),
    ("index dtype", {"index_dtype": I64}, (1,), SAME_KERNEL),
    (
        "rank",
        {"shape": (4, 3, 5), "fmt": "ddd", "target": "dss"},
        (1, 5),
        CHANGES_KERNEL,
    ),
    ("first extent", {"shape": (7, 5)}, (1,), CHANGES_KERNEL),
    ("second extent", {"shape": (6, 4)}, (1,), CHANGES_KERNEL),
)


def _change_mode_order(
    to: Sequence[int] = (1, 0, 2),
    target: Optional[str] = None,
    keep: bool = False,
    **spec: Any,
) -> Case:
    spec = {"shape": (4, 3, 5), "fmt": "sss", **spec}

    def run() -> Tuple[Any, Any]:
        tensor, logical = _tensor(**spec)
        extra: Dict[str, Any] = {}
        if target is not None:
            extra["_target_format"] = parse_format(target)
        if keep:
            extra["_keep_every_stored_entry"] = True
        return tensor.change_mode_order(list(to), **extra), logical

    return run


# With no target format the result keeps the source's (change_mode_order's
# ``target_format = self.format``), so a source format change, and a rank change with it,
# changes position 5 as well as the layout.
_CHANGE_MODE_ORDER_ROWS = (
    ("source format", {"fmt": "dss"}, (1, 5), CHANGES_KERNEL),
    ("target format", {"target": "dss"}, (5,), CHANGES_KERNEL),
    ("source mode order", {"mode_order": (0, 2, 1)}, (1,), CHANGES_KERNEL),
    ("target mode order", {"to": (2, 1, 0)}, (4,), CHANGES_KERNEL),
    ("value dtype", {"dtype": F32}, (2,), CHANGES_KERNEL),
    ("index dtype", {"index_dtype": I64}, (1,), SAME_KERNEL),
    ("rank", {"shape": (6, 5), "fmt": "ss", "to": (1, 0)}, (1, 4, 5), CHANGES_KERNEL),
    ("first extent", {"shape": (5, 3, 5)}, (1,), CHANGES_KERNEL),
    ("second extent", {"shape": (4, 4, 5)}, (1,), CHANGES_KERNEL),
    ("third extent", {"shape": (4, 3, 6)}, (1,), CHANGES_KERNEL),
    ("every stored entry kept", {"keep": True}, (6,), CHANGES_KERNEL),
)


def _add(
    left: Optional[Dict[str, Any]] = None, right: Optional[Dict[str, Any]] = None
) -> Case:
    left_spec = {"seed": 1, **(left or {})}
    right_spec = {"seed": 2, "name": "B", **(right or {})}

    def run() -> Tuple[Any, Any]:
        a, a_logical = _tensor(**left_spec)
        b, b_logical = _tensor(**right_spec)
        dtype = torch.promote_types(a.dtype, b.dtype)
        return a + b, a_logical.to(dtype) + b_logical.to(dtype)

    return run


def _both(**spec: Any) -> Dict[str, Any]:
    return {"left": dict(spec), "right": dict(spec)}


_ADD_ROWS = (
    ("left format", {"left": {"fmt": "ss"}}, (1,), CHANGES_KERNEL),
    ("right format", {"right": {"fmt": "ss"}}, (2,), CHANGES_KERNEL),
    ("left value dtype", {"left": {"dtype": F32}}, (3,), CHANGES_KERNEL),
    ("right value dtype", {"right": {"dtype": F32}}, (4,), CHANGES_KERNEL),
    ("index dtype", _both(index_dtype=I64), (1, 2), SAME_KERNEL),
    ("rank", _both(shape=(4, 3, 5), fmt="sss"), (1, 2), CHANGES_KERNEL),
    ("first extent", _both(shape=(7, 5)), (1, 2), CHANGES_KERNEL),
)


def _spmv(vector_dtype: Optional[torch.dtype] = None, **spec: Any) -> Case:
    def run() -> Tuple[Any, Any]:
        matrix, logical = _tensor(**spec)
        x = _dense((logical.shape[1],), 7, vector_dtype or logical.dtype) + 1.0
        result = ops.spmv(matrix, STensor.from_torch(x, "x"))
        return result, _Sum((logical.to(F64) @ x.to(F64)).to(logical.dtype))

    return run


_SPMV_ROWS = (
    ("matrix format", {"fmt": "ss"}, (1,), CHANGES_KERNEL),
    # A product has one value dtype: the lowering refuses a mixed pair.
    ("matrix value dtype", {"dtype": F32, "vector_dtype": F64}, (3,), REFUSED),
    ("vector value dtype", {"vector_dtype": F32}, (4,), REFUSED),
    ("both value dtypes", {"dtype": F32}, (3, 4), CHANGES_KERNEL),
    ("index dtype", {"index_dtype": I64}, (1,), SAME_KERNEL),
    ("rows", {"shape": (7, 5)}, (1,), CHANGES_KERNEL),
    ("columns", {"shape": (6, 4)}, (1, 2), CHANGES_KERNEL),
)

# name: (case factory, rows, the operation the key names, position of the options key,
# and the verdicts of the option rows that differ from _OPTION_ROWS')
_ENTRIES: Dict[str, Tuple[Callable[..., Case], tuple, str, int, Dict[str, str]]] = {
    "to_dense": (_to_dense, _TO_DENSE_ROWS, "to_dense", 6, {}),
    "to_sparse": (_to_sparse, _TO_SPARSE_ROWS, "to_sparse", 6, {}),
    "change_mode_order": (
        _change_mode_order,
        _CHANGE_MODE_ORDER_ROWS,
        "change_mode_order",
        7,
        {},
    ),
    "__add__": (_add, _ADD_ROWS, "add", 7, {}),
    # spmv refuses a requested schedule before it builds its key.  Full CIN verification
    # accepts spmv's CIN (test_where_scoped_workspace_reduction.py), so its row is the
    # one every entry has.
    "spmv": (
        _spmv,
        _SPMV_ROWS,
        "spmv",
        8,
        {"forced schedule": REFUSED_BEFORE_LOOKUP},
    ),
}


def _component_rows() -> List[Any]:
    rows = []
    for entry, (_, entry_rows, _, options_position, overrides) in _ENTRIES.items():
        for name, changed, positions, verdict in entry_rows:
            rows.append(
                pytest.param(
                    entry, changed, None, positions, verdict, id=f"{entry}: {name}"
                )
            )
        for name, context, verdict in _OPTION_ROWS:
            rows.append(
                pytest.param(
                    entry,
                    {},
                    context,
                    (options_position,),
                    overrides.get(name, verdict),
                    id=f"{entry}: {name}",
                )
            )
    return rows


def _row_cases(
    entry: str, changed: Dict[str, Any], context: Optional[Callable[[], Any]]
) -> Tuple[Case, Case]:
    factory = _ENTRIES[entry][0]
    second = factory(**changed)
    return factory(), second if context is None else _under(context, second)


# -- 1. positive controls ---------------------------------------------------------------


@pytest.mark.parametrize("entry", sorted(_ENTRIES))
def test_an_identical_second_call_hits(entry: str, seen: _Seen) -> None:
    _hit_pair(_ENTRIES[entry][0](), seen)


def test_to_torch_hits_through_to_dense(seen: _Seen) -> None:
    def case() -> Tuple[Any, Any]:
        tensor, logical = _tensor()
        return tensor.to_torch(in_place=False), logical

    _hit_pair(case, seen)


def test_rank_one_to_sparse_hits_through_to_dense(seen: _Seen) -> None:
    def case() -> Tuple[Any, Any]:
        tensor, logical = _tensor(shape=(9,), fmt="s")
        return tensor.to_sparse("s"), logical

    _hit_pair(case, seen)


def _in_place_to_dense(tensor: STensor) -> STensor:
    tensor.to_dense(in_place=True)
    return tensor


def _in_place_to_torch(tensor: STensor) -> STensor:
    tensor.to_torch()
    return tensor


def _in_place_to_sparse(tensor: STensor) -> STensor:
    tensor.to_sparse("ds")
    return tensor


def _in_place_change_mode_order(tensor: STensor) -> STensor:
    tensor.change_mode_order([1, 0])
    return tensor


@pytest.mark.parametrize(
    "call",
    (
        _in_place_to_dense,
        _in_place_to_torch,
        _in_place_to_sparse,
        _in_place_change_mode_order,
    ),
    ids=lambda call: call.__name__.removeprefix("_in_place_"),
)
def test_an_in_place_hit_leaves_the_tensor_as_the_full_path_does(
    call: Callable[[STensor], STensor],
    seen: _Seen,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def operand() -> STensor:
        return _tensor(fmt="ss")[0]

    monkeypatch.setattr(stensor_module, "_GENERATED_KERNELS", _Recording(keep=False))
    reference = _fields(call(operand()))
    monkeypatch.setattr(stensor_module, "_GENERATED_KERNELS", _Recording())
    call(operand())
    seen.reset()
    hit = call(operand())
    assert seen.builds == []
    assert _kept().lookups[-1][1], "the third call did not hit"
    assert _same_fields(_fields(hit), reference)
    # The tensor is used again afterwards, and reads as the source.
    assert torch.equal(_logical(hit), _dense((6, 5), 1))


@contextlib.contextmanager
def _environment_reads() -> Iterator[List[str]]:
    """Every variable read through ``os.environ`` or ``os.getenv`` inside the block.

    The options key of a warm call reads the encoded variables from ``os.environ``'s own
    dict (``compile_options._environment_values``), which is not a read through these, and
    is einsum's warm key too.
    """

    reads: List[str] = []
    environ_type = type(os.environ)
    get, getitem, contains = (
        environ_type.get,
        environ_type.__getitem__,
        environ_type.__contains__,
    )
    getenv = os.getenv

    def counted_get(self, key, default=None):
        reads.append(f"environ.get({key!r})")
        return get(self, key, default)

    def counted_getitem(self, key):
        reads.append(f"environ[{key!r}]")
        return getitem(self, key)

    def counted_contains(self, key):
        reads.append(f"{key!r} in environ")
        return contains(self, key)

    def counted_getenv(key, default=None):
        reads.append(f"getenv({key!r})")
        return getenv(key, default)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(environ_type, "get", counted_get)
        patch.setattr(environ_type, "__getitem__", counted_getitem)
        patch.setattr(environ_type, "__contains__", counted_contains)
        patch.setattr(os, "getenv", counted_getenv)
        yield reads


@pytest.mark.parametrize("entry", sorted(_ENTRIES))
def test_another_tensor_of_the_same_layout_hits_and_reads_its_own_arrays(
    entry: str, seen: _Seen
) -> None:
    factory = _ENTRIES[entry][0]
    other: Case
    if entry == "__add__":
        other = factory(left={"seed": 11}, right={"seed": 12})
    else:
        other = factory(seed=11)
    assert _outcome(factory()) == ("correct",)
    seen.reset()
    assert _outcome(other) == ("correct",)
    assert seen.builds == [], "another pattern of the same layout lowered again"


def test_a_from_torch_source_hits_the_kernel_of_a_built_source_of_its_layout(
    seen: _Seen,
) -> None:
    """``STensor.from_torch`` and ``from_components`` make the same layout for an all-dense
    tensor, and both have an index, so the second ``to_sparse`` hits and reads its own
    values."""

    assert _outcome(_to_sparse()) == ("correct",)
    seen.reset()
    assert _outcome(_to_sparse(from_torch=True, seed=11)) == ("correct",)
    assert seen.builds == [], "a from_torch source of the same layout lowered again"


# -- 2. one input changed ---------------------------------------------------------------


@pytest.mark.parametrize(
    "entry, changed, context, positions, verdict", _component_rows()
)
def test_a_changed_input_misses_and_does_what_the_full_path_does(
    entry: str,
    changed: Dict[str, Any],
    context: Optional[Callable[[], Any]],
    positions: Tuple[int, ...],
    verdict: str,
    seen: _Seen,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first, second = _row_cases(entry, changed, context)
    operation = _ENTRIES[entry][2]
    _changed_row(operation, first, second, positions, verdict, seen, monkeypatch)


# -- 4. negative controls ---------------------------------------------------------------


@pytest.mark.parametrize(
    "entry, changed, context, positions, verdict", _component_rows()
)
def test_a_key_without_the_input_fails_its_row(
    entry: str,
    changed: Dict[str, Any],
    context: Optional[Callable[[], Any]],
    positions: Tuple[int, ...],
    verdict: str,
    seen: _Seen,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first, second = _row_cases(entry, changed, context)
    _negative_control(first, second, positions, verdict, seen, monkeypatch)


def test_the_hit_check_passes_a_correct_hit(
    seen: _Seen, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(stensor_module, "_CHECK_GENERATED_KERNEL_HITS", [True])
    case = _to_dense()
    assert _outcome(case) == ("correct",)
    seen.reset()
    assert _outcome(case) == ("correct",)
    assert seen.checks == 1


def test_the_hit_check_refuses_a_kept_module_that_is_not_the_built_one(
    seen: _Seen, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _to_dense()
    assert _outcome(case) == ("correct",)
    cache = _kept()
    (key,) = cache.keys()
    # Another kernel's module kept under this key, as a key that left out an input
    # would keep it.
    assert _outcome(_to_dense(fmt="ss")) == ("correct",)
    other = cache[next(k for k in cache.keys() if k != key)]
    dict.__setitem__(cache, key, other)
    monkeypatch.setattr(stensor_module, "_CHECK_GENERATED_KERNEL_HITS", [True])
    with pytest.raises(GeneratedKernelCacheError) as refused:
        case()
    assert refused.value.code == "kept_kernel_differs_from_built"


# -- conversions inside a call ----------------------------------------------------------


def _warm_call_hits_its_conversions(case: Case, seen: _Seen) -> None:
    assert _outcome(case) == ("correct",)
    assert seen.builds, "the first call built no conversion kernel"
    del _kept().lookups[:]
    seen.reset()
    assert _outcome(case) == ("correct",)
    lookups = _kept().lookups
    assert lookups and all(hit for _, hit in lookups), lookups
    assert seen.builds == [] and seen.einsum_options == []
    assert (seen.snapshots, seen.parses) == (0, 0)


def test_einsum_moves_an_operand_on_its_cached_path_without_options(
    seen: _Seen,
) -> None:
    # ``ijk->i`` over a dsd operand is converted on every call, the move recorded in
    # einsum's cache entry (test_einsum_operand_relayout's sweep class 2).
    def case() -> Tuple[Any, Any]:
        tensor, logical = _tensor(shape=(4, 3, 5), fmt="dsd")
        result = scorch.einsum("ijk->i", tensor, format="s")
        return result, _Sum(logical.sum(dim=(1, 2)))

    _warm_call_hits_its_conversions(case, seen)


def test_einsum_reorders_its_result_on_its_cached_path_without_options(
    seen: _Seen,
) -> None:
    # The result is computed in another mode order and moved to [1, 0] after: ``ij``,
    # ``jk`` and the requested ``ki`` order the indices in a cycle, and
    # topo_sort_characters gives up the edge of the smallest tensor, here the 4 x 4
    # result (A is 4 x 5, B 5 x 4), so the loops run i, j, k and the kernel writes the
    # result in [0, 1].  With a larger result (A 6 x 5) B's edge goes instead, the
    # loops run k, i, j and nothing is moved.
    def case() -> Tuple[Any, Any]:
        a, a_logical = _tensor(shape=(4, 5), fmt="ss", seed=3)
        b, b_logical = _tensor(shape=(5, 4), fmt="ss", seed=4, name="B")
        result = scorch.einsum("ij,jk->ik", a, b, format="ss", output_mode_order=[1, 0])
        return result, _Sum(a_logical @ b_logical)

    _warm_call_hits_its_conversions(case, seen)
    (entry,) = ops._einsum_dispatch_cache.values()
    assert entry[2] != entry[3], "einsum no longer reorders this result"


def test_addition_moves_its_right_operand_under_its_own_options(seen: _Seen) -> None:
    case = _add(
        left={"shape": (4, 3, 5), "fmt": "sss"},
        right={"shape": (4, 3, 5), "fmt": "sss", "mode_order": (1, 0, 2)},
    )
    seen.reset()
    _warm_call_hits_its_conversions(case, seen)
    assert {key[0] for key, _ in _kept().lookups} == {"change_mode_order", "add"}


def _spmv_branch() -> Tuple[Any, Any]:
    a, logical = _tensor(fmt="ss")
    x = _dense((5,), 8) + 1.0
    return scorch.matmul(a, x), _Sum(logical @ x)


def _spmv_branch_moved() -> Tuple[Any, Any]:
    a, logical = _tensor(fmt="ss", mode_order=(1, 0))
    x = _dense((5,), 8) + 1.0
    return scorch.matmul(a, x), _Sum(logical @ x)


def _generated_moved() -> Tuple[Any, Any]:
    # DCSR has no prebuilt product, and its 2-D move is a kernel (change_mode_order's
    # torch paths take dd, ds and oo).
    a, logical = _tensor(fmt="ss", mode_order=(1, 0))
    b = _dense((5, 3), 9)
    return scorch.matmul(a, STensor.from_torch(b, "B")), _Sum(logical @ b)


def _prebuilt_moved_in_torch() -> Tuple[Any, Any]:
    a, logical = _tensor(fmt="ds", mode_order=(1, 0))
    b = _dense((5, 3), 9)
    return scorch.matmul(a, b), _Sum(logical @ b)


def _dense_product_stored() -> Tuple[Any, Any]:
    a = _dense((6, 5), 10)
    b = _dense((5, 4), 11)
    result = scorch.matmul(
        STensor.from_torch(a, "A"), STensor.from_torch(b, "B"), format="ds"
    )
    return result, _Sum(a @ b)


# ops.spmv moves an A stored column first to row order on a copy (0054), under the call's
# holder: DCSR by a change_mode_order kernel, CSR in torch.
_spmv_moved = _spmv(fmt="ss", mode_order=(1, 0))
_spmv_moved_in_torch = _spmv(fmt="ds", mode_order=(1, 0))


@pytest.mark.parametrize(
    "case",
    [pytest.param(_ENTRIES[entry][0](), id=entry) for entry in sorted(_ENTRIES)]
    + [
        pytest.param(_spmv_moved, id="spmv, operand moved"),
        pytest.param(_spmv_branch_moved, id="matmul spmv branch, operand moved"),
        pytest.param(_dense_product_stored, id="matmul dense product stored"),
    ],
)
def test_a_warm_hit_reads_no_environment_variable(case: Case) -> None:
    # The hit check's switch is read once, at import (_CHECK_GENERATED_KERNEL_HITS);
    # a per-call read would cost the warm call what the cache removes.
    assert _outcome(case) == ("correct",)
    del _kept().lookups[:]
    with _environment_reads() as reads:
        assert _outcome(case) == ("correct",)
    lookups = _kept().lookups
    assert lookups and all(hit for _, hit in lookups), lookups
    assert reads == []


@pytest.mark.parametrize(
    "case, operations",
    (
        pytest.param(
            _spmv_moved,
            ["change_mode_order", "spmv"],
            id="spmv, operand moved by a kernel",
        ),
        pytest.param(_spmv_moved_in_torch, ["spmv"], id="spmv, operand moved in torch"),
        pytest.param(_spmv_branch, ["spmv"], id="spmv branch"),
        pytest.param(
            _spmv_branch_moved,
            ["change_mode_order", "spmv"],
            id="spmv branch, operand moved by a kernel",
        ),
        pytest.param(
            _generated_moved,
            ["change_mode_order"],
            id="generated product, operand moved by a kernel",
        ),
        pytest.param(
            _prebuilt_moved_in_torch, [], id="prebuilt product, operand moved in torch"
        ),
        pytest.param(_dense_product_stored, ["to_sparse"], id="dense product stored"),
    ),
)
def test_a_warm_call_makes_no_snapshot_and_no_parse(
    case: Case, operations: List[str], seen: _Seen
) -> None:
    assert _outcome(case) == ("correct",)
    assert seen.snapshots <= 1, "a call made more than one snapshot"
    del _kept().lookups[:]
    seen.reset()
    assert _outcome(case) == ("correct",)
    lookups = _kept().lookups
    assert [key[0] for key, _ in lookups] == operations
    assert all(hit for _, hit in lookups), lookups
    assert seen.builds == [] and seen.einsum_options == []
    assert (seen.snapshots, seen.parses) == (0, 0)


def test_spmv_of_a_moved_operand_hits_the_kernel_of_one_stored_row_first() -> None:
    """spmv moves A before it builds its key, so the key holds the moved layout: the
    product with an A stored column first hits the kernel kept for an A stored row
    first, and reads its own values."""

    assert _outcome(_spmv(fmt="ss")) == ("correct",)
    del _kept().lookups[:]
    assert _outcome(_spmv(fmt="ss", mode_order=(1, 0), seed=11)) == ("correct",)
    lookups = _kept().lookups
    assert [hit for key, hit in lookups if key[0] == "spmv"] == [True], lookups


def test_one_snapshot_serves_a_relayout_miss_and_an_einsum_miss(
    seen: _Seen, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ops, "_kernel_cache", {})
    seen.reset()
    assert _outcome(_generated_moved) == ("correct",)
    assert seen.snapshots == 1
    assert len(seen.build_options) == 1 and len(seen.einsum_options) == 1
    assert seen.build_options[0] is seen.einsum_options[0]


_SCHEDULE = Schedule(loop_order=("i", "j", "k"), tag="matmul-one-holder")


def _keyword() -> Tuple[Dict[str, Any], Any]:
    return {"schedule": _SCHEDULE}, contextlib.nullcontext()


def _forced() -> Tuple[Dict[str, Any], Any]:
    return {}, schedule_force(_SCHEDULE)


def _handed() -> Tuple[Dict[str, Any], Any]:
    options = CompileOptions.from_environment(requested_schedule=_SCHEDULE)
    return {"_compile_options": options}, contextlib.nullcontext()


@pytest.mark.parametrize(
    "spelling", (_keyword, _forced, _handed), ids=("keyword", "forced", "handed")
)
@pytest.mark.parametrize(
    "moved", (False, True), ids=("einsum resolves", "relayout resolves")
)
def test_the_schedule_does_not_depend_on_which_step_resolves_it(
    spelling: Callable[[], Tuple[Dict[str, Any], Any]],
    moved: bool,
    seen: _Seen,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ops, "_kernel_cache", {})
    a, logical = _tensor(fmt="ss", mode_order=(1, 0) if moved else None)
    b = _dense((5, 3), 9)
    kwargs, context = spelling()
    seen.reset()
    with context:
        result = scorch.matmul(a, STensor.from_torch(b, "B"), **kwargs)
    assert torch.allclose(_logical(result), logical @ b, rtol=1e-12, atol=1e-12)
    (key,) = ops._einsum_dispatch_cache
    assert key[5] == ops._schedule_cache_key(_SCHEDULE)
    assert seen.snapshots == (0 if spelling is _handed else 1)
    (options,) = seen.einsum_options
    assert options.requested_schedule is not None
    assert options.requested_schedule.cache_key == _SCHEDULE.cache_key
    assert len(seen.build_options) == (1 if moved else 0)


def test_a_conversion_takes_a_holder_or_a_snapshot_not_both() -> None:
    tensor, _ = _tensor(fmt="ss")
    holder = ops._DeferredOptions({})
    with pytest.raises(TypeError, match="not both"):
        tensor.to_dense(_compile_options=holder.options(), _deferred_options=holder)


def test_a_holder_with_a_schedule_is_refused() -> None:
    tensor, _ = _tensor(fmt="ss")
    with schedule_force(_SCHEDULE):
        holder = ops._DeferredOptions({})
    with pytest.raises(TypeError, match="requested schedule"):
        tensor.to_dense(_deferred_options=holder)


def test_a_scheduled_conversion_gets_the_schedule_free_copy_of_the_snapshot(
    seen: _Seen, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ops, "_kernel_cache", {})
    a, logical = _tensor(fmt="ss", mode_order=(1, 0))
    b = _dense((5, 3), 9)
    seen.reset()
    result = scorch.matmul(a, STensor.from_torch(b, "B"), schedule=_SCHEDULE)
    assert torch.allclose(_logical(result), logical @ b, rtol=1e-12, atol=1e-12)
    assert seen.snapshots == 1
    (conversion,) = seen.build_options
    (product,) = seen.einsum_options
    assert conversion.requested_schedule is None
    assert conversion == dataclasses.replace(product, requested_schedule=None)
