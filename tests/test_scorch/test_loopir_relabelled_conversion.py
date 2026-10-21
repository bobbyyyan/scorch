"""An operand conversion is a relabelled copy, assembled by counting passes.

A conversion copies every entry an operand stores into another level order
(``operand_conversion``).  Its first form walks the operand in storage order and
appends each entry at its converted coordinates, out of the converted tensor's
storage order, under a recorded ``two_pass_serial`` assembly.  The verifier
admits such an append only as a relabelled copy of one input
(``verifier._check_append_order``), and the target builds the converted tensor
(``lower_llir._RelabelledCopyAssemblyLowering``).  Into dense levels over one
compressed level, by counting sort: count the entries of each segment of the
compressed level, prefix-sum the counts, store each entry at its slot.  Into
two or more compressed levels, by a stable radix sort of the entries' records
made of the same counting passes, then one sweep that builds every level.  A
layout the target refuses with ``unsupported_relabelled_assembly`` would take
the sparse-workspace copy instead; ``conversion_source(forms=...)`` selects
the forms for measurement.  The parallel form (``"relabelled_parallel"``,
``two_pass_parallel``) splits the walk into chunks of about equal entry counts:
per-thread counts into one compressed level, and per-thread digit histograms
and a two-pass sweep into more.  It is tested on forced thread counts through
the tuning hooks, against the same expected arrays.

Every executed conversion is compared level array by level array, and value by
value, with the arrays the converted layout must hold for the operand's stored
cells: every layout of dense and compressed levels at ranks 2 to 4 under every
mode order; every converted layout of dense levels over compressed ones at
ranks 2 and 3, and from an all-compressed operand at rank 4; on operands with
empty rows and columns, on an operand that stores nothing and on one whose
only stored entry is a zero; on hypersparse operands whose coordinates need
several radix digits, and on one with more entries than the widest digit
distinguishes; in floating point and integer types.
"""

import itertools
import os
from collections import defaultdict
from dataclasses import replace
from typing import Dict, List, Optional, Tuple

import pytest
import torch

from scorch import STensor
from scorch.compiler.cin import TensorVar
from scorch.compiler.compilation_context import CompilationContext
from scorch.compiler.compile_options import CompileOptions
from scorch.compiler.loop_plan import OperandConversion
from scorch.compiler.loopir.levels import LevelTensorStorage
from scorch.compiler.loopir.lower_llir import (
    UNSUPPORTED_RELABELLED_ASSEMBLY,
    LoopIRTargetError,
    lower_loopir_to_llir,
)
from scorch.compiler.loopir.nodes import (
    AppendEntry,
    AssemblyStrategy,
    Block,
    DenseFor,
    FloatConst,
    LevelKind,
    LoopIRNodeId,
    SparseFor,
)
from scorch.compiler.loopir.operand_conversion import (
    conversion_format,
    conversion_program,
    conversion_source,
    converted_shape,
    relabelled_copy_program,
)
from scorch.compiler.loopir.oracle import run_program
from scorch.compiler.loopir.verifier import LoopIRVerificationError, verify_program

SHAPES = {2: (6, 5), 3: (5, 4, 3), 4: (4, 3, 3, 2)}
COUNTING_SORT = "Count the entries of each segment of the compressed level"
RADIX_SORT = "one counting pass per digit"
PARALLEL = "Each thread counts its chunk's entries per segment"
PARALLEL_RADIX = "one counting pass per digit, on every thread"
PARALLEL_SWEEP = "Build each compressed level from the sorted records, on every thread"


def layouts(rank: int) -> List[str]:
    """Every layout with a compressed level; an all-dense operand never converts."""

    return [
        "".join(letters)
        for letters in itertools.product("ds", repeat=rank)
        if "s" in letters
    ]


def reorders(rank: int) -> List[Tuple[int, ...]]:
    identity = tuple(range(rank))
    return [order for order in itertools.permutations(range(rank)) if order != identity]


CASES = [
    (fmt, order)
    for rank in (2, 3, 4)
    for fmt in layouts(rank)
    for order in reorders(rank)
]


def targets(rank: int) -> List[str]:
    """Every converted layout: dense levels over one or more compressed levels."""

    return ["d" * dense + "s" * (rank - dense) for dense in range(rank)]


TARGET_CASES = [
    (fmt, order, target)
    for rank in (2, 3)
    for fmt in layouts(rank)
    for order in reorders(rank)
    for target in targets(rank)
    if target != conversion_format(fmt)
] + [
    (fmt, order, target)
    for fmt in ("ssss", "dsds")
    for order in reorders(4)
    for target in targets(4)
    if target != conversion_format(fmt)
]


def assert_in_order_copy(source: str) -> None:
    """The conversion is one walk that appends in order: no hash map, no sort,
    no counting passes."""

    assert "coo_workspace" not in source and "sort(" not in source
    assert COUNTING_SORT not in source and RADIX_SORT not in source


IN_ORDER_CASES = [
    (fmt, tuple(range(rank))) for rank in (2, 3, 4) for fmt in layouts(rank)
]


@pytest.mark.parametrize(
    "fmt, order", IN_ORDER_CASES, ids=[f"{f}-{o}" for f, o in IN_ORDER_CASES]
)
def test_a_conversion_that_keeps_the_level_order_copies_in_order(fmt, order):
    """Every layout at ranks 2 to 4, converted without reordering its levels (a
    layout with a dense level below a compressed one becomes all compressed):
    the in-order copy, executed and compared array by array."""

    target = conversion_format(fmt)
    source, run = run_conversion(fmt, order, torch.float64)
    assert_in_order_copy(source)
    for family, values, keep in operands(SHAPES[len(fmt)], torch.float64):
        converted = run(stored(values, keep, fmt))
        assert actual_arrays(converted) == expected_arrays(
            values, keep, fmt, order, target
        ), family


def assert_counting_form(source: str, target: str) -> None:
    """The conversion is the counting sort into one compressed level, the radix
    sort into more, and never the sparse-workspace copy."""

    if target.count("s") == 1:
        assert COUNTING_SORT in source and RADIX_SORT not in source
    else:
        assert RADIX_SORT in source and COUNTING_SORT not in source
    assert PARALLEL not in source and PARALLEL_RADIX not in source
    assert "coo_workspace" not in source


def assert_parallel_form(source: str, target: str) -> None:
    """The parallel counting sort into one compressed level, the parallel radix
    sort and sweep into more; each keeps the one-thread kernel for one thread."""

    if target.count("s") == 1:
        assert PARALLEL in source and COUNTING_SORT in source
        assert RADIX_SORT not in source
    else:
        assert PARALLEL_RADIX in source and PARALLEL_SWEEP in source
        assert COUNTING_SORT not in source and PARALLEL not in source
    assert "coo_workspace" not in source


def level_arrays(indicator: torch.Tensor, fmt: str):
    """The leaf cells and per-level ``(pos, crd)`` arrays of ``indicator`` in ``fmt``."""

    shape = tuple(indicator.shape)
    children: Dict[Tuple[int, ...], set] = defaultdict(set)
    for coord in indicator.nonzero().tolist():
        for level in range(len(shape)):
            children[tuple(coord[:level])].add(coord[level])
    parents: List[Tuple[int, ...]] = [()]
    levels: List[Optional[Tuple[List[int], List[int]]]] = []
    for level, kind in enumerate(fmt):
        if kind == "d":
            levels.append(None)
            parents = [p + (c,) for p in parents for c in range(shape[level])]
            continue
        pos, crd, below = [0], [], []
        for parent in parents:
            kids = sorted(children.get(parent, ()))
            crd.extend(kids)
            below.extend(parent + (kid,) for kid in kids)
            pos.append(len(crd))
        levels.append((pos, crd))
        parents = below
    return parents, levels


def stored_cells(keep: torch.Tensor, fmt: str) -> torch.Tensor:
    """The cells a tensor holding ``keep`` in ``fmt`` stores (a dense level below a
    compressed one stores every cell under each stored parent)."""

    leaves, _ = level_arrays(keep, fmt)
    cells = torch.zeros(tuple(keep.shape), dtype=torch.bool)
    for leaf in leaves:
        cells[leaf] = True
    return cells


def stored(values: torch.Tensor, keep: torch.Tensor, fmt: str) -> STensor:
    leaves, levels = level_arrays(keep, fmt)
    mode_indices = [
        (
            []
            if arrays is None
            else [
                torch.tensor(arrays[0], dtype=torch.int64),
                torch.tensor(arrays[1], dtype=torch.int64),
            ]
        )
        for arrays in levels
    ]
    data = torch.stack([values[leaf] for leaf in leaves]) if leaves else values[:0]
    return STensor.from_components(
        tuple(values.shape), fmt, mode_indices, data.reshape(-1), name="A"
    )


def scattered(
    shape: Tuple[int, ...], count: int, dtype: torch.dtype, seed: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``count`` distinct random cells of ``shape`` holding values in 1..49."""

    generator = torch.Generator().manual_seed(seed)
    cells = torch.randperm(int(torch.tensor(shape).prod()), generator=generator)
    keep = torch.zeros(shape, dtype=torch.bool)
    keep.view(-1)[cells[:count]] = True
    values = torch.randint(1, 50, shape, generator=generator).to(dtype)
    return values * keep, keep


def operands(shape: Tuple[int, ...], dtype: torch.dtype):
    """Operand families: random with an empty first slice and last column, empty,
    and one stored zero."""

    generator = torch.Generator().manual_seed(11)
    if dtype.is_floating_point:
        values = torch.rand(shape, generator=generator, dtype=torch.float64) + 0.5
    else:
        values = torch.randint(1, 50, shape, generator=generator, dtype=torch.int64)
    values = values.to(dtype)
    keep = torch.rand(shape, generator=generator) < 0.4
    keep[0] = False
    keep[..., -1] = False
    empty = torch.zeros(shape, dtype=torch.bool)
    single = torch.zeros(shape, dtype=torch.bool)
    single[tuple(extent - 1 for extent in shape)] = True
    zero_values = torch.zeros(shape, dtype=dtype)
    return [
        ("random", values * keep, keep),
        ("empty", values * empty, empty),
        ("one stored zero", zero_values, single),
    ]


def run_conversion(
    fmt: str,
    order: Tuple[int, ...],
    dtype: torch.dtype,
    target: Optional[str] = None,
    shape: Optional[Tuple[int, ...]] = None,
    compile_options: Optional[CompileOptions] = None,
    forms: Optional[Tuple[str, ...]] = None,
):
    """Emit, build and return (source, run) for one conversion (into
    ``conversion_format(fmt)`` unless ``target`` names another layout, of an
    operand of ``SHAPES[len(fmt)]`` unless ``shape`` names another, by the
    default forms unless ``forms`` names others)."""

    from scorch.ops import (
        _load_validated_prepared_kernel,
        _prepare_generated_kernel_build,
    )
    from scorch.stensor import _wrap_generated_result

    shape = SHAPES[len(fmt)] if shape is None else shape
    target = conversion_format(fmt) if target is None else target
    if compile_options is None:
        compile_options = CompileOptions.from_environment()
    conversion = OperandConversion(
        TensorVar("A", fmt=fmt, dtype=dtype).symbol_id, order, target
    )
    extra = {} if forms is None else {"forms": forms}
    source = conversion_source(
        fmt, conversion, dtype, shape, compile_options=compile_options, **extra
    )
    module = _load_validated_prepared_kernel(
        _prepare_generated_kernel_build(
            compile_options.build.preamble_source,
            source,
            compile_options,
            CompilationContext(compile_options),
        )
    )
    new_shape = converted_shape(shape, conversion)

    def run(operand: STensor) -> STensor:
        return _wrap_generated_result(
            new_shape,
            target,
            module.evaluate(
                list(new_shape),
                list(operand.shape),
                operand._native_mode_indices(),
                operand.values,
            ),
            mode_order=tuple(range(len(fmt))),
        )

    return source, run


def expected_arrays(values, keep, fmt, order, target):
    cells = stored_cells(keep, fmt).permute(order)
    leaves, levels = level_arrays(cells, target)
    arrays = [() if level is None else (level[0], level[1]) for level in levels]
    moved = values.permute(order)
    return arrays, [moved[leaf].item() for leaf in leaves]


def actual_arrays(converted: STensor):
    arrays = [
        tuple(t.reshape(-1).tolist() for t in level)
        for level in converted.storage.index.mode_indices
    ]
    return arrays, converted.storage.value.reshape(-1).tolist()


@pytest.mark.parametrize("fmt, order", CASES, ids=[f"{f}-{o}" for f, o in CASES])
def test_a_conversion_stores_exactly_the_operands_cells(fmt, order):
    """Every layout and mode order: the converted arrays, entry for entry."""

    target = conversion_format(fmt)
    source, run = run_conversion(fmt, order, torch.float64)
    assert_counting_form(source, target)
    for family, values, keep in operands(SHAPES[len(fmt)], torch.float64):
        converted = run(stored(values, keep, fmt))
        assert actual_arrays(converted) == expected_arrays(
            values, keep, fmt, order, target
        ), family


@pytest.mark.parametrize(
    "fmt, order, target",
    TARGET_CASES,
    ids=[f"{f}-{o}-{t}" for f, o, t in TARGET_CASES],
)
def test_a_conversion_into_any_converted_layout(fmt, order, target):
    """Every converted layout an explicit plan can ask for, not only the one the
    automatic plan picks: more or fewer dense levels than the operand has."""

    source, run = run_conversion(fmt, order, torch.float64, target)
    assert_counting_form(source, target)
    for family, values, keep in operands(SHAPES[len(fmt)], torch.float64):
        converted = run(stored(values, keep, fmt))
        assert actual_arrays(converted) == expected_arrays(
            values, keep, fmt, order, target
        ), family


SPREAD_CASES = [
    # Hypersparse: a few entries over wide extents.  The digit is
    # bit_width(entries) bits wide, so each wide level takes two or more
    # passes.  Extents stay small enough for the dense reference tensors.
    ("ss", (1, 0), "ss", (700, 900), 20),
    ("ss", (1, 0), "ds", (700, 900), 20),
    ("sss", (2, 0, 1), "sss", (3, 700, 300), 40),
    ("sss", (0, 2, 1), "dss", (3, 700, 300), 40),
    ("dss", (1, 0, 2), "dss", (6, 2500, 9), 30),
    # More entries than the widest digit distinguishes (2**11): full-width
    # digits, two passes over a compressed level, and over the dense level's
    # coordinates, of extent 5000.
    ("ss", (1, 0), "ss", (64, 5000), 3000),
    ("dss", (2, 0, 1), "dss", (3, 40, 5000), 2500),
]


@pytest.mark.parametrize(
    "fmt, order, target, shape, count",
    SPREAD_CASES,
    ids=[f"{f}-{o}-{t}-{n}" for f, o, t, _, n in SPREAD_CASES],
)
def test_a_conversion_whose_coordinates_need_several_digits(
    fmt, order, target, shape, count
):
    source, run = run_conversion(fmt, order, torch.float64, target, shape)
    assert_counting_form(source, target)
    values, keep = scattered(shape, count, torch.float64, seed=count)
    converted = run(stored(values, keep, fmt))
    assert actual_arrays(converted) == expected_arrays(values, keep, fmt, order, target)


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.int32, torch.int64, torch.int8, torch.uint8]
)
@pytest.mark.parametrize(
    "fmt, order",
    [("ds", (1, 0)), ("dds", (2, 0, 1)), ("ss", (1, 0)), ("dss", (0, 2, 1))],
)
def test_a_counting_conversion_keeps_the_value_type(fmt, order, dtype):
    source, run = run_conversion(fmt, order, dtype)
    assert_counting_form(source, conversion_format(fmt))
    for family, values, keep in operands(SHAPES[len(fmt)], dtype):
        converted = run(stored(values, keep, fmt))
        assert converted.storage.value.dtype == dtype, family
        assert actual_arrays(converted) == expected_arrays(
            values, keep, fmt, order, conversion_format(fmt)
        ), family


PARALLEL_SHAPES = {2: (8, 8), 3: (8, 8, 8), 4: (8, 8, 8, 8)}
PARALLEL_CASES = [
    (fmt, order, conversion_format(fmt))
    for fmt, order in CASES
    if conversion_format(fmt).count("s") == 1
] + [
    (fmt, order, target)
    for fmt, order, target in TARGET_CASES
    if target.count("s") == 1 and (len(fmt) < 4 or fmt == "ssss")
]


def crowded(shape: Tuple[int, ...], dtype: torch.dtype):
    """Operand families with enough entries per segment for several threads: three
    quarters of the cells with an empty first slice and last column, and the last
    two slices full with every other cell empty (the chunks are then uneven in
    outermost iterations)."""

    generator = torch.Generator().manual_seed(23)
    values = (torch.randint(1, 50, shape, generator=generator)).to(dtype)
    keep = torch.rand(shape, generator=generator) < 0.75
    keep[0] = False
    keep[..., -1] = False
    tail = torch.zeros(shape, dtype=torch.bool)
    tail[-2:] = True
    return [("crowded", values * keep, keep), ("last two slices", values * tail, tail)]


def segments_of(target: str, shape: Tuple[int, ...], order: Tuple[int, ...]) -> int:
    moved = [shape[mode] for mode in order]
    count = 1
    for level, kind in enumerate(target):
        if kind == "d":
            count *= moved[level]
    return count


@pytest.mark.parametrize(
    "fmt, order, target",
    PARALLEL_CASES,
    ids=[f"{f}-{o}-{t}" for f, o, t in PARALLEL_CASES],
)
def test_the_parallel_counting_sort_stores_what_one_thread_stores(
    fmt, order, target, monkeypatch
):
    """Three threads forced through the tuning hooks (``SCORCH_TUNE_THREADS``),
    on operands with at least two entries per segment, so the kernel's cap of
    entries / segments threads still leaves two or more: the arrays equal the
    expected ones entry for entry, as on one thread."""

    shape = PARALLEL_SHAPES[len(fmt)]
    source, run = run_parallel(fmt, order, torch.float64, target, shape)
    assert_parallel_form(source, target)
    monkeypatch.setenv("SCORCH_TUNE_THREADS", "3")
    for family, values, keep in crowded(shape, torch.float64):
        entries = int(stored_cells(keep, fmt).sum())
        assert entries >= 2 * segments_of(target, shape, order), family
        converted = run(stored(values, keep, fmt))
        assert actual_arrays(converted) == expected_arrays(
            values, keep, fmt, order, target
        ), family


def run_parallel(fmt, order, dtype, target=None, shape=None):
    """``run_conversion`` of the parallel form, with the tuning hooks compiled in
    so ``SCORCH_TUNE_THREADS`` sets the thread count when the kernel runs."""

    options = CompileOptions.from_environment(
        environ={**os.environ, "SCORCH_JIT_TUNE_HOOKS": "1"}
    )
    return run_conversion(
        fmt, order, dtype, target, shape, options, forms=("relabelled_parallel",)
    )


PARALLEL_RADIX_CASES = [
    (fmt, order, conversion_format(fmt))
    for fmt, order in CASES
    if conversion_format(fmt).count("s") >= 2
] + [
    (fmt, order, target)
    for fmt, order, target in TARGET_CASES
    if target.count("s") >= 2 and (len(fmt) < 4 or fmt == "ssss")
]


@pytest.mark.parametrize(
    "fmt, order, target",
    PARALLEL_RADIX_CASES,
    ids=[f"{f}-{o}-{t}" for f, o, t in PARALLEL_RADIX_CASES],
)
def test_the_parallel_radix_sort_stores_what_one_thread_stores(
    fmt, order, target, monkeypatch
):
    """Two or more compressed levels on three threads: chunks of uneven
    outermost iterations, empty chunks (the last two slices), an operand that
    stores nothing and one stored zero.  The arrays equal the expected ones
    entry for entry, as on one thread."""

    shape = PARALLEL_SHAPES[len(fmt)]
    source, run = run_parallel(fmt, order, torch.float64, target, shape)
    assert_parallel_form(source, target)
    monkeypatch.setenv("SCORCH_TUNE_THREADS", "3")
    families = crowded(shape, torch.float64) + operands(shape, torch.float64)
    for family, values, keep in families:
        converted = run(stored(values, keep, fmt))
        assert actual_arrays(converted) == expected_arrays(
            values, keep, fmt, order, target
        ), family


@pytest.mark.parametrize("threads", ["2", "3", "7"])
@pytest.mark.parametrize(
    "fmt, order, target, shape, count",
    SPREAD_CASES,
    ids=[f"{f}-{o}-{t}-{n}" for f, o, t, _, n in SPREAD_CASES],
)
def test_a_parallel_conversion_whose_coordinates_need_several_digits(
    fmt, order, target, shape, count, threads, monkeypatch
):
    """The hypersparse operands on several thread counts: every digit pass of
    the parallel radix sort, and segments that no chunk starts (dense prefixes
    of extent 2500 and 5000 over 30 and 2500 entries)."""

    source, run = run_parallel(fmt, order, torch.float64, target, shape)
    assert_parallel_form(source, target)
    monkeypatch.setenv("SCORCH_TUNE_THREADS", threads)
    values, keep = scattered(shape, count, torch.float64, seed=count)
    converted = run(stored(values, keep, fmt))
    assert actual_arrays(converted) == expected_arrays(values, keep, fmt, order, target)


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.int32, torch.int64, torch.int8, torch.uint8]
)
@pytest.mark.parametrize(
    "fmt, order",
    [("ds", (1, 0)), ("dds", (2, 0, 1)), ("ss", (1, 0)), ("dss", (0, 2, 1))],
)
def test_a_parallel_conversion_keeps_the_value_type(fmt, order, dtype, monkeypatch):
    shape = PARALLEL_SHAPES[len(fmt)]
    target = conversion_format(fmt)
    source, run = run_parallel(fmt, order, dtype, shape=shape)
    assert_parallel_form(source, target)
    monkeypatch.setenv("SCORCH_TUNE_THREADS", "3")
    for family, values, keep in crowded(shape, dtype):
        converted = run(stored(values, keep, fmt))
        assert converted.storage.value.dtype == dtype, family
        assert actual_arrays(converted) == expected_arrays(
            values, keep, fmt, order, target
        ), family


def relabelled(fmt: str, order: Tuple[int, ...], parallel: bool = False):
    conversion = OperandConversion(
        TensorVar("A", fmt=fmt, dtype=torch.float64).symbol_id,
        order,
        conversion_format(fmt),
    )
    return relabelled_copy_program(
        fmt,
        conversion,
        torch.float64,
        compile_options=CompileOptions.from_environment(),
        parallel=parallel,
    )


def with_leaf(program, rebuild):
    """``program`` with its append leaf replaced by ``rebuild(leaf)``."""

    def walk(node):
        if type(node) is AppendEntry:
            return rebuild(node)
        if type(node) is Block:
            return replace(node, statements=tuple(walk(s) for s in node.statements))
        if type(node) in (DenseFor, SparseFor):
            return replace(node, body=walk(node.body))
        raise AssertionError(type(node).__name__)

    return replace(program, body=walk(program.body))


@pytest.mark.parametrize("assembly", [None, AssemblyStrategy.SINGLE_PASS_SERIAL])
def test_an_unordered_append_needs_a_two_pass_record(assembly):
    program, _, _ = relabelled("ds", (1, 0))
    verify_program(program)
    with pytest.raises(LoopIRVerificationError) as refused:
        verify_program(replace(program, assembly=assembly))
    assert refused.value.defect.code == "unordered_append"
    assert "two-pass" in str(refused.value)


def test_an_unordered_append_must_copy_the_stored_value():
    program, _, _ = relabelled("ds", (1, 0))
    forged = with_leaf(
        program, lambda leaf: replace(leaf, value=FloatConst(LoopIRNodeId(10_000), 1.0))
    )
    with pytest.raises(LoopIRVerificationError) as refused:
        verify_program(forged)
    assert refused.value.defect.code == "unordered_append"


def test_an_unordered_append_must_not_repeat_a_coordinate():
    """A coordinate appended twice, out of order, is refused.

    Every level of a converted tensor has its own dimension, so a repeated
    coordinate sits at a level of another dimension and the coordinate's
    domain check refuses it before the order rule is reached; the order
    rule's permutation check is the second line.  At rank 2 the repeat is
    bound at one depth twice, which is in order, so rank 3 is used.
    """

    program, _, _ = relabelled("dds", (2, 0, 1))
    forged = with_leaf(
        program,
        lambda leaf: replace(
            leaf,
            coords=(
                leaf.coords[0],
                leaf.coords[1],
                # The same coordinate as a node of its own: a node may appear once.
                replace(leaf.coords[1], node_id=LoopIRNodeId(10_001)),
            ),
        ),
    )
    with pytest.raises(LoopIRVerificationError) as refused:
        verify_program(forged)
    assert refused.value.defect.code == "domain_mismatch"


IDENTITY_CASES = [
    (fmt, order)
    for rank in (2, 3)
    for fmt in layouts(rank)
    for order in itertools.permutations(range(rank))
]


@pytest.mark.parametrize(
    "fmt, order", IDENTITY_CASES, ids=[f"{f}-{o}" for f, o in IDENTITY_CASES]
)
def test_a_conversion_into_the_order_the_operand_holds_is_the_in_order_copy(fmt, order):
    """An operand stored in mode order ``order`` and converted to ``order``:
    the copy appends in order, records no strategy, verifies as before, and
    is lowered as it is, one walk with no hash map and no sort
    (``_run_conversions`` rebuilds a conversion against the operand's stored
    order, so this is reached whenever the operand already holds the planned
    order)."""

    conversion = OperandConversion(
        TensorVar("A", fmt=fmt, dtype=torch.float64).symbol_id,
        order,
        conversion_format(fmt),
    )
    options = CompileOptions.from_environment()
    program, _, _ = relabelled_copy_program(
        fmt, conversion, torch.float64, source_mode_order=order, compile_options=options
    )
    assert program.assembly is None
    verify_program(program)
    source = conversion_source(
        fmt,
        conversion,
        torch.float64,
        SHAPES[len(fmt)],
        source_mode_order=order,
        compile_options=options,
    )
    assert_in_order_copy(source)


OUTER_COMPRESSED_CASES = [
    ("sd", (1, 0), "ds"),
    ("ss", (1, 0), "ds"),
    ("sds", (1, 2, 0), "dds"),
    ("ssd", (2, 1, 0), "dds"),
]


@pytest.mark.parametrize(
    "fmt, order, target",
    OUTER_COMPRESSED_CASES,
    ids=[f"{f}-{o}-{t}" for f, o, t in OUTER_COMPRESSED_CASES],
)
def test_a_counting_sort_conversion_from_a_compressed_outer_level(fmt, order, target):
    """A source whose outermost level is compressed declares that level's
    iterator bounds once, for both walks (an explicit plan can ask for these
    targets; the automatic plan keeps the source's letters)."""

    source, run = run_conversion(fmt, order, torch.float64, target)
    assert COUNTING_SORT in source and "coo_workspace" not in source
    for family, values, keep in operands(SHAPES[len(fmt)], torch.float64):
        converted = run(stored(values, keep, fmt))
        assert actual_arrays(converted) == expected_arrays(
            values, keep, fmt, order, target
        ), family


def test_an_in_order_append_is_judged_as_before():
    """The identity copy appends in order: no record is needed, as before."""

    program, _, _ = relabelled("ds", (0, 1))
    verify_program(replace(program, assembly=None))


def test_a_relabelled_copy_records_two_pass_on_any_receiver_that_stores_its_entries():
    """``dds`` is not a partitionable receiver, which a shared two-phase pass
    needs; a relabelled copy numbers its segments by the dense prefix, and its
    parallel form splits the input's walk, so either record is legal."""

    program, _, _ = relabelled("dds", (1, 2, 0))
    assert program.assembly is AssemblyStrategy.TWO_PASS_SERIAL
    verify_program(program)
    parallel, _, _ = relabelled("dds", (1, 2, 0), parallel=True)
    assert parallel.assembly is AssemblyStrategy.TWO_PASS_PARALLEL
    verify_program(parallel)


def conversion_of(fmt: str, order: Tuple[int, ...]) -> OperandConversion:
    return OperandConversion(
        TensorVar("A", fmt=fmt, dtype=torch.float64).symbol_id,
        order,
        conversion_format(fmt),
    )


def test_a_layout_the_target_refuses_takes_the_workspace_copy(monkeypatch):
    """A refusal with ``unsupported_relabelled_assembly`` falls back to the
    sparse-workspace copy; any other target refusal propagates."""

    from scorch.compiler.loopir import operand_conversion
    from scorch.compiler.loopir.lower_llir import LoopIRTargetDefect

    lower = operand_conversion.lower_loopir_to_llir
    seen = []

    def refusing(code):
        def lower_or_refuse(program, **kwargs):
            seen.append(program.assembly)
            if program.assembly is AssemblyStrategy.TWO_PASS_SERIAL:
                raise LoopIRTargetError(LoopIRTargetDefect(code, "refused"))
            return lower(program, **kwargs)

        return lower_or_refuse

    options = CompileOptions.from_environment()
    monkeypatch.setattr(
        operand_conversion,
        "lower_loopir_to_llir",
        refusing(UNSUPPORTED_RELABELLED_ASSEMBLY),
    )
    source = conversion_source(
        "ss",
        conversion_of("ss", (1, 0)),
        torch.float64,
        SHAPES[2],
        compile_options=options,
    )
    assert "coo_workspace" in source and RADIX_SORT not in source
    assert seen == [AssemblyStrategy.TWO_PASS_SERIAL, None]
    monkeypatch.setattr(
        operand_conversion, "lower_loopir_to_llir", refusing("some_other_defect")
    )
    with pytest.raises(LoopIRTargetError) as refused:
        conversion_source(
            "ss",
            conversion_of("ss", (1, 0)),
            torch.float64,
            SHAPES[2],
            compile_options=options,
        )
    assert refused.value.defect.code == "some_other_defect"


@pytest.mark.parametrize("fmt, order", [("ds", (1, 0)), ("dss", (0, 2, 1))])
def test_forms_select_the_conversion_for_measurement(fmt, order):
    """``forms=("region",)`` builds the sparse-workspace copy of any layout,
    ``("relabelled",)`` the counting form; the default tries both in order."""

    options = CompileOptions.from_environment()
    conversion = conversion_of(fmt, order)
    shape = SHAPES[len(fmt)]
    region = conversion_source(
        fmt,
        conversion,
        torch.float64,
        shape,
        compile_options=options,
        forms=("region",),
    )
    assert "coo_workspace" in region
    assert COUNTING_SORT not in region and RADIX_SORT not in region
    relabelled_only = conversion_source(
        fmt,
        conversion,
        torch.float64,
        shape,
        compile_options=options,
        forms=("relabelled",),
    )
    default = conversion_source(
        fmt, conversion, torch.float64, shape, compile_options=options
    )
    assert relabelled_only == default
    assert_counting_form(default, conversion_format(fmt))
    parallel = conversion_source(
        fmt,
        conversion,
        torch.float64,
        shape,
        compile_options=options,
        forms=("relabelled_parallel",),
    )
    assert_parallel_form(parallel, conversion_format(fmt))


def test_forms_name_known_forms_and_keep_the_in_order_copy():
    """A relabelled form alone builds a copy that keeps the level order as the
    in-order copy, and unknown form names are rejected."""

    options = CompileOptions.from_environment()
    for fmt in ("ds", "sd"):
        source = conversion_source(
            fmt,
            conversion_of(fmt, (0, 1)),
            torch.float64,
            SHAPES[2],
            compile_options=options,
            forms=("relabelled",),
        )
        assert_in_order_copy(source)
    unknown = ((), ("hash",), ("relabelled", "workspace"), ("parallel",))
    for forms in (*unknown, ("relabelled_serial",)):
        with pytest.raises(ValueError):
            conversion_source(
                "ds",
                conversion_of("ds", (1, 0)),
                torch.float64,
                SHAPES[2],
                compile_options=options,
                forms=forms,
            )


@pytest.mark.parametrize("fmt", ["ds", "ss"])
def test_the_serial_record_is_the_default_and_both_records_are_emitted(fmt):
    """The copy records the serial strategy unless asked for the parallel one;
    the target emits both, into one compressed level and into more."""

    expected = {
        False: AssemblyStrategy.TWO_PASS_SERIAL,
        True: AssemblyStrategy.TWO_PASS_PARALLEL,
    }
    for parallel, strategy in expected.items():
        # Each program declares its own operand symbol.
        program, operand, _ = relabelled(fmt, (1, 0), parallel=parallel)
        assert program.assembly is strategy
        verify_program(program)
        lower_loopir_to_llir(
            program,
            input_shapes={operand: SHAPES[2]},
            result_shape=SHAPES[2][::-1],
            compile_options=CompileOptions.from_environment(),
        )


@pytest.mark.parametrize(
    "fmt, order", [("ds", (1, 0)), ("dds", (2, 0, 1)), ("ss", (1, 0))]
)
def test_the_oracle_assembles_a_relabelled_copy_like_the_workspace_copy(fmt, order):
    """The oracle collects a relabelled copy's appends and assembles them in
    coordinate order: the same tensor as the sparse-workspace copy."""

    conversion = OperandConversion(
        TensorVar("A", fmt=fmt, dtype=torch.float64).symbol_id,
        order,
        conversion_format(fmt),
    )
    options = CompileOptions.from_environment()
    shape = SHAPES[len(fmt)]
    values, keep = operands(shape, torch.float64)[0][1:]
    kinds = tuple(
        LevelKind.DENSE if kind == "d" else LevelKind.COMPRESSED for kind in fmt
    )
    operand_tensor = LevelTensorStorage.from_dense(
        (values * keep).tolist(), shape, tuple(range(len(fmt))), kinds
    )
    results = []
    for form in (relabelled_copy_program, conversion_program):
        program, operand, result = form(
            fmt, conversion, torch.float64, compile_options=options
        )
        out = run_program(
            program,
            {operand: operand_tensor},
            {result: converted_shape(shape, conversion)},
        )
        results.append(out[result])
    assert results[0] == results[1]
