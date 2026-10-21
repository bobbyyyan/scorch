"""The LoopIR route reads and writes an all-dense tensor in any loop order.

A dense level's position is its parent's position times the level's extent plus
its coordinate, so it needs no loop of its own: the target computes it in the
innermost loop that binds a coordinate of that level or of a level above it,
wherever the loop nest visits the tensor.  That holds for an all-dense operand,
for an all-dense result written by a store, and for one written by a strip-mined
dense workspace's copy-out loop.  A compressed level still needs its parent's
position before its loop starts, so every compressed tensor is still visited in
storage order.

Each program below runs into several result formats.  An all-dense result must
compile on both automatic arms; every other format either compiles or is refused
with a structured code.  Whatever compiles is executed on stored operand values
and compared with dense PyTorch: values, exact zeros where nothing is stored
below, and, for a compressed result, the exact level arrays its storage rule
gives.  The larger extents cross the 32-wide strip of the dense workspace, so a
partial strip is exercised.
"""

from collections import defaultdict
from dataclasses import replace
from typing import Dict, List, Optional, Sequence, Tuple

import pytest
import torch

from scorch import STensor
from scorch.compiler.cin import (
    BinaryOp as CINBinaryOp,
    ForAll,
    IndexVar,
    Operation,
    TensorAssign,
    TensorVar,
)
from scorch.compiler.compile_options import CompileOptions
from scorch.compiler.diagnostics import UnsupportedFeature
from scorch.compiler.loop_plan_legality import InvalidSchedule
from scorch.compiler.loopir.lower_cin import LoopIRLoweringError
from scorch.compiler.loopir.lower_llir import LoopIRTargetError
from scorch.compiler.loopir.pipeline import (
    compile_cin_via_loopir,
    execute_cin_via_loopir,
)
from scorch.compiler.loopir.schedule_passes import SchedulePassError
from scorch.compiler.scheduler import Schedule

REFUSALS = (
    LoopIRLoweringError,
    LoopIRTargetError,
    InvalidSchedule,
    UnsupportedFeature,
    SchedulePassError,
)

SMALL = {"i": 5, "j": 4, "k": 3, "l": 2}
# Larger than the dense workspace's 32-wide strip and not a multiple of it.
LARGE = {"i": 37, "j": 35, "k": 33, "l": 3}

# (name, operands as (format, indices), result indices, result formats).  The
# first result format is all dense.
PROGRAMS = [
    ("dense copy, transposed", (("dd", "ij"),), "ji", ("dd", "ds", "sd", "ss")),
    ("CSR copy, transposed", (("ds", "ij"),), "ji", ("dd", "ds", "sd", "ss")),
    ("DCSR copy, transposed", (("ss", "ij"),), "ji", ("dd", "ds", "sd", "ss")),
    ("dense 3-tensor, ikj", (("ddd", "ijk"),), "ikj", ("ddd", "dds", "dss", "sss")),
    ("compressed 3-tensor, kji", (("dss", "ijk"),), "kji", ("ddd", "dds", "dsd")),
    (
        "dense operand read transposed",
        (("dd", "ji"), ("ds", "ij")),
        "ij",
        ("dd", "ds", "ss"),
    ),
    (
        "CSR times dense into a transposed result",
        (("ds", "ij"), ("dd", "jk")),
        "ki",
        ("dd", "ds", "sd", "ss"),
    ),
    (
        "dense times dense into a transposed result",
        (("dd", "ij"), ("dd", "jk")),
        "ki",
        ("dd", "ds", "ss"),
    ),
    (
        "TTM into a reversed result",
        (("sss", "ijk"), ("dd", "kl")),
        "lji",
        ("ddd", "dds", "sss"),
    ),
    (
        "outer contraction into a transposed result",
        (("ds", "ij"), ("ds", "kl")),
        "lj",
        ("dd", "ds", "ss"),
    ),
    # In storage order everywhere: the change must leave these as they were.
    ("CSR times dense, in order", (("ds", "ij"), ("dd", "jk")), "ik", ("dd", "ds")),
]


def options(arm: int) -> CompileOptions:
    return replace(
        CompileOptions.from_environment().with_regblock_enabled(bool(arm)),
        requested_schedule=Schedule(),
    )


def program_cin(operands, result_indices, result_format, dtype):
    letters = "".join(dict.fromkeys("".join(indices for _, indices in operands)))
    variables = {letter: IndexVar(letter) for letter in letters}
    product = None
    for position, (fmt, indices) in enumerate(operands):
        access = TensorVar("AB"[position], fmt=fmt, dtype=dtype)[
            tuple(variables[letter] for letter in indices)
        ]
        product = (
            access if product is None else CINBinaryOp(Operation.MUL, product, access)
        )
    result = TensorVar("C", fmt=result_format, dtype=dtype)[
        tuple(variables[letter] for letter in result_indices)
    ]
    if any(letter not in result_indices for letter in letters):
        statement = TensorAssign(result, product, op=Operation.ADD)
    else:
        statement = TensorAssign(result, product)
    for letter in reversed(letters):
        statement = ForAll(variables[letter], statement)
    return statement


def level_arrays(
    indicator: torch.Tensor, fmt: str
) -> Tuple[List[Tuple[int, ...]], List[Optional[Tuple[List[int], List[int]]]]]:
    """Stored leaves and per-level (pos, crd) of ``indicator`` stored as ``fmt``.

    A dense level stores every coordinate below every stored parent; a
    compressed level stores exactly the coordinates something nonzero lies
    under, in increasing order.
    """

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


def operand(values: torch.Tensor, keep: torch.Tensor, fmt: str, name: str) -> STensor:
    """An STensor built from explicit level arrays: no conversion kernel runs."""

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
    stored = torch.tensor([float(values[leaf]) for leaf in leaves], dtype=values.dtype)
    return STensor.from_components(
        tuple(values.shape), fmt, mode_indices, stored, name=name
    )


def inputs(operands, extents, dtype, seed: int):
    """(operands, float64 reference values, stored-structure indicators)."""

    built, references, indicators = [], [], []
    for position, (fmt, indices) in enumerate(operands):
        shape = tuple(extents[letter] for letter in indices)
        generator = torch.Generator().manual_seed(1000 * seed + position)
        values = (torch.rand(shape, generator=generator, dtype=torch.float64) + 0.5).to(
            dtype
        )
        if "s" in fmt:
            keep = torch.rand(shape, generator=generator) < 0.6
        else:
            keep = torch.ones(shape, dtype=torch.bool)
        built.append(operand(values, keep, fmt, "AB"[position]))
        references.append(values.to(torch.float64) * keep)
        indicators.append(keep.to(torch.float64))
    return built, references, indicators


def result_order(result: STensor, rank: int) -> List[int]:
    """The result's mode order: storage level -> logical mode."""

    order = getattr(result, "mode_order", None)
    return list(range(rank)) if order is None else [int(mode) for mode in order]


def result_dense(result: STensor, fmt: str, shape: Sequence[int]) -> torch.Tensor:
    """The result's logical cells, from its level arrays and mode order.

    No conversion kernel runs.  Execution may return the result in another
    mode order than the program declares (it lays it out in the loop order),
    so the levels are read in the result's own order.
    """

    order = result_order(result, len(shape))
    physical = [shape[mode] for mode in order]
    mode_indices = result.storage.index.mode_indices
    values = result.storage.value.reshape(-1).to(torch.float64)
    coords: List[Tuple[int, ...]] = [()]
    for level, kind in enumerate(fmt):
        if kind == "d":
            coords = [c + (x,) for c in coords for x in range(physical[level])]
            continue
        pos, crd = (t.reshape(-1).tolist() for t in mode_indices[level])
        coords = [
            parent + (crd[p],)
            for index, parent in enumerate(coords)
            for p in range(pos[index], pos[index + 1])
        ]
    assert len(coords) == values.numel()
    dense = torch.zeros(tuple(shape), dtype=torch.float64)
    for coord, value in zip(coords, values.tolist()):
        logical = [0] * len(order)
        for level, mode in enumerate(order):
            logical[mode] = coord[level]
        dense[tuple(logical)] = value
    return dense


def judge(result, fmt, shape, reference, structure, dtype) -> None:
    got = result_dense(result, fmt, shape)
    tolerance = 1e-12 if dtype is torch.float64 else 1e-5
    scale = max(1.0, float(reference.abs().max())) if reference.numel() else 1.0
    error = float((got - reference).abs().max()) if reference.numel() else 0.0
    assert error <= tolerance * scale, f"max error {error:.3e} at scale {scale:.3e}"
    assert torch.all(got[~structure] == 0), "a value where nothing is stored below"
    order = result_order(result, len(shape))
    expected_leaves, expected = level_arrays(structure.permute(order), fmt)
    actual = [
        (
            None
            if kind == "d"
            else tuple(
                t.reshape(-1).tolist() for t in result.storage.index.mode_indices[level]
            )
        )
        for level, kind in enumerate(fmt)
    ]
    assert actual == [
        None if arrays is None else tuple(arrays) for arrays in expected
    ], "stored coordinates differ from the operands' structure"
    assert result.storage.value.numel() == len(expected_leaves)


def run_emitted(kernel, shape, tensors, arm):
    """Run the emitted kernel itself, as the execution entry does after layout.

    The execution entry first lays every operand and the result out in the
    loop order, so it never runs a kernel that reads or writes a dense tensor
    out of storage order; this runs the kernel ``compile_cin_via_loopir``
    emitted on the operands as declared.
    """

    from scorch.compiler.compilation_context import CompilationContext
    from scorch.compiler.loopir.nodes import LevelKind
    from scorch.ops import (
        _load_validated_prepared_kernel,
        _prepare_generated_kernel_build,
    )
    from scorch.stensor import _wrap_generated_result

    compile_options = options(arm)
    prepared = _prepare_generated_kernel_build(
        compile_options.build.preamble_source,
        kernel.cpp_source,
        compile_options,
        CompilationContext(compile_options),
    )
    module = _load_validated_prepared_kernel(prepared)
    arguments: List[object] = [list(shape)]
    for tensor in tensors:
        arguments.extend(
            [list(tensor.shape), tensor._native_mode_indices(), tensor.values]
        )
    result_cpp = module.evaluate(*arguments)
    decl = next(
        d
        for d in kernel.lowering.program.tensors
        if d.symbol == kernel.lowering.result_symbol
    )
    fmt = "".join(
        "d" if level.kind is LevelKind.DENSE else "s" for level in decl.levels
    )
    result = _wrap_generated_result(
        tuple(shape),
        fmt,
        result_cpp,
        mode_order=tuple(level.mode for level in decl.levels),
    )
    return result, fmt


def refusal_code(error: BaseException) -> Optional[str]:
    """The structured reason of a refusal: a defect or diagnostic code."""

    defect = getattr(error, "defect", None)
    if defect is not None:
        return str(defect.code)
    for diagnostic in getattr(error, "diagnostics", ()):
        code = getattr(diagnostic, "code", None)
        if code is not None:
            return str(code)
    return None


def run(name, operands, result_indices, result_format, arm, extents, dtype):
    """``"emitted"`` after a correct execution, or ``"refused:<code>"``."""

    shape = tuple(extents[letter] for letter in result_indices)
    bindings = tuple(
        (tuple(extents[letter] for letter in indices), dtype) for _, indices in operands
    )
    cin = program_cin(operands, result_indices, result_format, dtype)
    try:
        kernel = compile_cin_via_loopir(
            cin, shape, bindings, compile_options=options(arm)
        )
    except REFUSALS as error:
        code = refusal_code(error)
        assert code, f"{name} [{result_format}] arm{arm}: unstructured {error!r}"
        return f"refused:{code}"
    seed = [program[0] for program in PROGRAMS].index(name)
    built, references, indicators = inputs(operands, extents, dtype, seed)
    expression = ",".join(indices for _, indices in operands) + "->" + result_indices
    reference = torch.einsum(expression, *references)
    structure = torch.einsum(expression, *indicators) > 0
    if "s" not in result_format:
        structure = torch.ones(shape, dtype=torch.bool)
    out = execute_cin_via_loopir(
        program_cin(operands, result_indices, result_format, dtype),
        shape,
        *built,
        compile_options=options(arm),
    )
    result, executed = out
    # Execution runs the kernel the same plan emits at compile time: no dense
    # tensor is copied into the loop order first.
    assert executed.cpp_source == kernel.cpp_source, (name, result_format, arm)
    assert executed.conversion_sources == kernel.conversion_sources
    judge(result, result_format, shape, reference, structure, dtype)
    # The emitted kernel itself, on the operands as declared (converted first
    # where the plan converts one, in the identity-order form it reads).
    conversions = {c.operand_id: c for c in kernel.conversions}
    symbols = [t.symbol_id for t in cin.get_rhs_tensor_vars()]
    direct_inputs = []
    for position, (fmt, indices) in enumerate(operands):
        conversion = conversions.get(symbols[position])
        if conversion is None:
            direct_inputs.append(built[position])
            continue
        # No operand here has a dense level below a compressed one, so the
        # cells it stores are the cells kept, and it keeps its layout.
        assert "sd" not in fmt and conversion.format == fmt
        order = conversion.mode_order
        values = references[position].to(dtype).permute(order).contiguous()
        keep = (indicators[position] > 0).permute(order).contiguous()
        direct_inputs.append(operand(values, keep, conversion.format, "AB"[position]))
    direct, direct_format = run_emitted(kernel, shape, direct_inputs, arm)
    judge(direct, direct_format, shape, reference, structure, dtype)
    return "emitted"


@pytest.mark.parametrize(
    "name,operands,result_indices,result_formats",
    PROGRAMS,
    ids=[program[0] for program in PROGRAMS],
)
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32], ids=str)
def test_dense_tensors_in_any_loop_order(
    name, operands, result_indices, result_formats, dtype
):
    outcomes = {}
    for result_format in result_formats:
        for arm in (0, 1):
            outcomes[(result_format, arm)] = run(
                name, operands, result_indices, result_format, arm, SMALL, dtype
            )
    dense_format = result_formats[0]
    assert "s" not in dense_format
    assert (
        outcomes[(dense_format, 0)] == outcomes[(dense_format, 1)] == "emitted"
    ), outcomes


@pytest.mark.parametrize(
    "name,operands,result_indices,result_formats",
    PROGRAMS,
    ids=[program[0] for program in PROGRAMS],
)
def test_dense_results_at_extents_past_one_strip(
    name, operands, result_indices, result_formats
):
    for arm in (0, 1):
        assert (
            run(
                name,
                operands,
                result_indices,
                result_formats[0],
                arm,
                LARGE,
                torch.float64,
            )
            == "emitted"
        )
