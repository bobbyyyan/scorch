"""A LoopIR plan may convert operands so that one loop order follows every tensor.

A compressed level needs its parent's position before its loop starts, so a loop
order must visit every tensor with a compressed level in storage order.  When the
tensors' storage orders disagree (``A[i,j] -> C[j,i]`` with both compressed, or
``A[i,j] * B[j,i]``), no loop order does, and the new compiler used to refuse.  The
plan now records operand conversions (``LoopPlan.conversions``): the loop order
determines them, since every compressed operand the order does not follow is
converted to that order restricted to its own indices.  When no loop order follows
every compressed tensor, the automatic plan tries every set of convertible operands
and keeps the cheapest by the scheduler's own objective, whose transposition term
prices each conversion.  A requested loop order converts exactly the operands it
breaks.

A conversion is itself a kernel of this compiler: a copy of every entry the
operand stores into the new level order, emitted with the main kernel
(``LoopIRCompiledKernel.conversion_sources``) and run by the execution entry
before it.  The copy is structural, so a stored zero stays stored, and the
converted tensor is laid out to store exactly the operand's entries
(``OperandConversion.format``): a dense level below a compressed one would store
cells the operand does not hold, so an operand with one (``"sd"``, ``"dsd"``)
converts to every level compressed, and any other keeps its layout (CSR becomes
CSC).  The main kernel reads a converted operand in the identity-order form
``converted_cin`` writes: a CSC matrix is the CSR form of its transpose.

Every test that executes compares against dense PyTorch in float64, through the
execution entry and by running the emitted kernel directly on operands built in
the converted layout, over several result formats.  A result with a compressed
level must also store exactly the coordinates the operands' stored entries
imply, zeros included; the operands hold stored zeros for that reason.  Each
converting program also runs on operands that store nothing and on operands
whose only stored entry is a zero.
"""

import itertools
from collections import defaultdict
from dataclasses import replace
from typing import Dict, List, Optional, Tuple

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
from scorch.compiler.compilation_context import CompilationContext
from scorch.compiler.compile_options import CompileOptions
from scorch.compiler.diagnostics import (
    InvalidSchedule,
    UnsupportedFeature,
    VerificationError,
)
from scorch.compiler.loop_plan import (
    LoopPlan,
    OperandConversion,
    converted_cin,
    verify_loop_plan,
)
from scorch.compiler.loopir.lower_cin import LoopIRLoweringError
from scorch.compiler.loopir.lower_llir import LoopIRTargetError
from scorch.compiler.loopir.pipeline import (
    compile_cin_via_loopir,
    execute_cin_via_loopir,
)
from scorch.compiler.loopir.plan_identity import canonical_plan_dump
from scorch.compiler.loopir.schedule_passes import SchedulePassError
from scorch.compiler.scheduler import Schedule, Scheduler
from scorch.compiler.cin_analysis import normalize_cin

REFUSALS = (
    LoopIRLoweringError,
    LoopIRTargetError,
    InvalidSchedule,
    UnsupportedFeature,
    SchedulePassError,
)

EXTENT = {"i": 5, "j": 4, "k": 3, "l": 2}
DTYPE = torch.float64


def options(arm: int, loop_order: Optional[Tuple[str, ...]] = None) -> CompileOptions:
    return replace(
        CompileOptions.from_environment().with_regblock_enabled(bool(arm)),
        requested_schedule=Schedule(loop_order=loop_order),
    )


def program(operands, result_indices, result_format):
    """``C[result] (+)= product of operands``, operands as (name, format, indices)."""

    letters = "".join(dict.fromkeys("".join(indices for _, _, indices in operands)))
    variables = {letter: IndexVar(letter) for letter in letters}
    product = None
    for name, fmt, indices in operands:
        access = TensorVar(name, fmt=fmt, dtype=DTYPE)[
            tuple(variables[letter] for letter in indices)
        ]
        product = (
            access if product is None else CINBinaryOp(Operation.MUL, product, access)
        )
    result = TensorVar("C", fmt=result_format, dtype=DTYPE)[
        tuple(variables[letter] for letter in result_indices)
    ]
    if any(letter not in result_indices for letter in letters):
        statement = TensorAssign(result, product, op=Operation.ADD)
    else:
        statement = TensorAssign(result, product)
    for letter in reversed(letters):
        statement = ForAll(variables[letter], statement)
    return statement


def shapes(operands, result_indices):
    result_shape = tuple(EXTENT[letter] for letter in result_indices)
    bindings = tuple(
        (tuple(EXTENT[letter] for letter in indices), DTYPE)
        for _, _, indices in operands
    )
    return result_shape, bindings


def symbol_of(cin, name):
    return next(t.symbol_id for t in cin.get_rhs_tensor_vars() if t.name == name)


# -- the plan field ------------------------------------------------------------


TRANSPOSE = ((("A", "ds", "ij"),), "ji", "ss")


def transpose_cin():
    operands, result_indices, result_format = TRANSPOSE
    return normalize_cin(program(operands, result_indices, result_format))


def loop_ids(cin, names):
    by_name = {}
    stmt = cin
    while isinstance(stmt, ForAll):
        by_name[stmt.index_var.name] = stmt.index_var.index_id
        stmt = stmt.stmt
    return tuple(by_name[name] for name in names)


def test_a_conversion_makes_the_transposed_order_verify():
    cin = transpose_cin()
    order = loop_ids(cin, "ji")
    with pytest.raises(InvalidSchedule):
        verify_loop_plan(cin, LoopPlan(loop_order=order))
    plan = LoopPlan(
        loop_order=order,
        conversions=(OperandConversion(symbol_of(cin, "A"), (1, 0), "ds"),),
    )
    assert verify_loop_plan(cin, plan) is plan


@pytest.mark.parametrize(
    "conversions,message",
    [
        (lambda a, c: (OperandConversion(c, (1, 0), "ds"),), "right-hand-side"),
        (
            lambda a, c: (
                OperandConversion(a, (1, 0), "ds"),
                OperandConversion(a, (1, 0), "ds"),
            ),
            "twice",
        ),
        (lambda a, c: (OperandConversion(a, (0, 0), "ds"),), "permute"),
        (lambda a, c: (OperandConversion(a, (0, 1), "ds"),), "change"),
        (lambda a, c: (OperandConversion(a, (1, 0), "sd"),), "only the operand's"),
        (lambda a, c: (OperandConversion(a, (1, 0), "dd"),), "keep a compressed"),
        (lambda a, c: (OperandConversion(a, (1, 0), "dss"),), "one level per"),
        (lambda a, c: (OperandConversion(a, (1, 0), "do"),), "dense.*compressed"),
    ],
    ids=[
        "result",
        "twice",
        "not a permutation",
        "unchanged",
        "dense below compressed",
        "no compressed level",
        "wrong rank",
        "another level kind",
    ],
)
def test_a_malformed_conversion_is_rejected(conversions, message):
    cin = transpose_cin()
    result = next(
        stmt.lhs.tensor.symbol_id
        for stmt in [cin.stmt.stmt]
        if isinstance(stmt, TensorAssign)
    )
    plan = LoopPlan(
        loop_order=loop_ids(cin, "ji"),
        conversions=conversions(symbol_of(cin, "A"), result),
    )
    with pytest.raises(VerificationError, match=message):
        verify_loop_plan(cin, plan)


def test_converting_an_all_dense_operand_is_rejected():
    cin = normalize_cin(program((("A", "dd", "ij"),), "ji", "dd"))
    plan = LoopPlan(
        loop_order=loop_ids(cin, "ij"),
        conversions=(OperandConversion(symbol_of(cin, "A"), (1, 0), "ss"),),
    )
    with pytest.raises(VerificationError, match="compressed level"):
        verify_loop_plan(cin, plan)


def test_a_plan_without_conversions_keeps_its_identity_bytes():
    cin = transpose_cin()
    order = loop_ids(cin, "ji")
    converting = LoopPlan(
        loop_order=order,
        conversions=(OperandConversion(symbol_of(cin, "A"), (1, 0), "ds"),),
    )
    assert '"conversions"' in canonical_plan_dump(cin, converting)
    other_layout = replace(
        converting,
        conversions=(OperandConversion(symbol_of(cin, "A"), (1, 0), "ss"),),
    )
    assert canonical_plan_dump(cin, other_layout) != canonical_plan_dump(
        cin, converting
    )
    in_order = normalize_cin(program((("A", "ds", "ij"),), "ij", "ss"))
    dump = canonical_plan_dump(in_order, LoopPlan(loop_order=loop_ids(in_order, "ij")))
    assert '"conversions"' not in dump


def test_the_converted_program_stores_the_operand_in_identity_order():
    cin = transpose_cin()
    for tensor in cin.get_rhs_tensor_vars():
        tensor.shape = (5, 4)
    converted = converted_cin(
        cin, (OperandConversion(symbol_of(cin, "A"), (1, 0), "ss"),)
    )
    (access,) = converted.get_rhs_tensor_accesses()
    assert list(access.tensor.mode_order) == [0, 1]
    assert tuple(access.tensor.shape) == (4, 5)
    assert [index.name for index in access.indices] == ["j", "i"]
    assert str(access.tensor.format).replace(",", "") == "ss"
    # The caller's program is not changed.
    (original,) = cin.get_rhs_tensor_accesses()
    assert [index.name for index in original.indices] == ["i", "j"]
    assert tuple(original.tensor.shape) == (5, 4)
    assert str(original.tensor.format).replace(",", "") == "ds"


# -- the automatic plan ---------------------------------------------------------

# (operands, result indices, result formats).  With a compressed result each
# needs a conversion (no loop order follows every compressed tensor); a dense
# result constrains no order, so with it the operands' orders may already agree.
CONVERTING = [
    ((("A", "ds", "ij"),), "ji", ("ss", "ds", "dd")),
    ((("A", "ss", "ij"),), "ji", ("ss", "dd")),
    ((("A", "dss", "ijk"),), "kji", ("dss", "sss", "ddd")),
    ((("A", "sss", "ijk"),), "ikj", ("sss", "dss")),
    ((("A", "ds", "ij"), ("B", "ds", "jk")), "ki", ("ds", "dd")),
    ((("A", "ds", "ij"), ("B", "ds", "ji")), "ij", ("ds", "ss", "dd")),
    ((("A", "ds", "ij"), ("B", "ss", "ji")), "ij", ("dd", "ds")),
    # A dense level below a compressed one: the operand stores zeros.
    ((("A", "sd", "ij"),), "ji", ("ss", "ds", "dd")),
    ((("A", "dsd", "ijk"),), "kji", ("sss", "dss", "ssd")),
    ((("A", "sdsd", "ijkl"),), "lkji", ("sssd", "ssss")),
    ((("A", "ssd", "ijk"), ("B", "ds", "kj")), "ik", ("ss", "ds", "dd")),
]

# The layout each operand layout converts to.
CONVERTED_LAYOUT = {
    "ds": "ds",
    "ss": "ss",
    "dss": "dss",
    "sss": "sss",
    "sd": "ss",
    "dsd": "sss",
    "sdsd": "ssss",
    "ssd": "sss",
}


def needs_conversion(operands, result_indices, result_format) -> bool:
    """Whether no loop order follows every tensor with a compressed level.

    Written from the definition (every permutation of the loops is tried), not
    with the compiler's storage-order walk.
    """

    letters = "".join(dict.fromkeys("".join(indices for _, _, indices in operands)))
    chains = [indices for _, fmt, indices in operands if "s" in fmt]
    if "s" in result_format:
        chains.append(result_indices)
    for order in itertools.permutations(letters):
        position = {letter: n for n, letter in enumerate(order)}
        if all(
            [position[c] for c in chain] == sorted(position[c] for c in chain)
            for chain in chains
        ):
            return False
    return True


def plan_of(operands, result_indices, result_format, arm, loop_order=None):
    """(kernel or None, refusal or None, the CIN compiled)."""

    result_shape, bindings = shapes(operands, result_indices)
    cin = program(operands, result_indices, result_format)
    try:
        kernel = compile_cin_via_loopir(
            cin,
            result_shape,
            bindings,
            compile_options=options(arm, loop_order),
        )
    except REFUSALS as error:
        return None, error, cin
    return kernel, None, cin


@pytest.mark.parametrize("arm", [0, 1])
def test_programs_with_disagreeing_storage_orders_convert_operands(arm):
    converted = []
    for operands, result_indices, result_formats in CONVERTING:
        for result_format in result_formats:
            key = (operands, result_indices, result_format)
            kernel, error, cin = plan_of(operands, result_indices, result_format, arm)
            if kernel is None:
                # Refused for another reason, never for the storage orders.
                assert not any(
                    getattr(d, "code", None)
                    in ("sparse_parent_dominance", "result_storage_order")
                    for d in getattr(error, "diagnostics", ())
                ), (key, error)
                continue
            assert bool(kernel.conversions) == needs_conversion(*key), key
            assert len(kernel.conversion_sources) == len(kernel.conversions)
            layouts = {name: fmt for name, fmt, _ in operands}
            names = {t.symbol_id: t.name for t in cin.get_rhs_tensor_vars()}
            for conversion in kernel.conversions:
                operand_layout = layouts[names[conversion.operand_id]]
                assert conversion.format == CONVERTED_LAYOUT[operand_layout], key
            if kernel.conversions:
                converted.append(key)
    # Not a vacuous pass: programs convert, and some to another layout.
    assert len(converted) >= 10, converted
    assert any("sd" in fmt for ops, _, _ in converted for _, fmt, _ in ops)


def test_the_cheaper_operand_is_converted():
    """A (CSR) times B stored ``ji`` (DCSR): converting either one fixes the order.

    The cost model estimates DCSR's stored entries at ``rho`` squared times its
    cells and CSR's at ``rho`` times, so the transposition term prices B lower;
    the chosen plan must be the cheaper of the two by the scheduler's objective.
    """

    operands = (("A", "ds", "ij"), ("B", "ss", "ji"))
    kernel, error, cin = plan_of(operands, "ij", "dd", 0)
    assert kernel is not None, error
    (conversion,) = kernel.conversions
    cin = normalize_cin(cin)
    names = {t.symbol_id: t.name for t in cin.get_rhs_tensor_vars()}
    costs = {}
    for name, order in (("A", "ji"), ("B", "ij")):
        candidate = (
            OperandConversion(
                symbol_of(cin, name), (1, 0), {"A": "ds", "B": "ss"}[name]
            ),
        )
        converted = converted_cin(cin, candidate)
        costs[name] = Scheduler.loop_order_cost(
            converted, loop_ids(cin, order), transposed=cin
        )
    assert names[conversion.operand_id] == min(costs, key=costs.get), costs


def test_a_requested_order_converts_the_operands_it_breaks():
    operands = (("A", "ds", "ij"), ("B", "dd", "jk"))
    kernel, error, cin = plan_of(operands, "ik", "dd", 0, loop_order=("j", "i", "k"))
    assert kernel is not None, error
    assert kernel.conversions == (OperandConversion(symbol_of(cin, "A"), (1, 0), "ds"),)


def test_a_program_that_needs_no_conversion_records_none():
    kernel, error, _ = plan_of((("A", "ds", "ij"), ("B", "dd", "jk")), "ik", "dd", 0)
    assert kernel is not None, error
    assert kernel.conversions == ()


# -- execution ------------------------------------------------------------------


def level_arrays(indicator: torch.Tensor, fmt: str):
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
    """The cells a tensor holding ``keep`` in layout ``fmt`` stores.

    More than ``keep`` when a dense level lies below a compressed one: every
    cell of that level under each stored parent is stored.
    """

    leaves, _ = level_arrays(keep, fmt)
    cells = torch.zeros(tuple(keep.shape), dtype=torch.bool)
    for leaf in leaves:
        cells[leaf] = True
    return cells


def stored(values: torch.Tensor, keep: torch.Tensor, fmt: str, name: str) -> STensor:
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
    data = torch.tensor([float(values[leaf]) for leaf in leaves], dtype=values.dtype)
    return STensor.from_components(
        tuple(values.shape), fmt, mode_indices, data, name=name
    )


def result_cells(result: STensor, fmt: str, logical_shape) -> torch.Tensor:
    """The result's logical cells, read from its level arrays and mode order."""

    order = list(getattr(result, "mode_order", None) or range(len(logical_shape)))
    physical = [logical_shape[mode] for mode in order]
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
    dense = torch.zeros(tuple(logical_shape), dtype=torch.float64)
    for coord, value in zip(coords, values.tolist()):
        logical = [0] * len(order)
        for level, mode in enumerate(order):
            logical[mode] = coord[level]
        dense[tuple(logical)] = value
    return dense


def run_emitted(kernel, result_shape, tensors, arm):
    """Run the emitted kernel itself on ``tensors``, as the execution entry does."""

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
    arguments: List[object] = [list(result_shape)]
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
    return (
        _wrap_generated_result(
            tuple(result_shape),
            fmt,
            result_cpp,
            mode_order=kernel.result_mode_order,
        ),
        fmt,
    )


INPUTS = ("random", "empty", "one stored zero")


def operand_values(kind: str, shape, position: int):
    """``(values, keep)`` of one operand: its cells' values and the cells kept.

    ``random`` keeps 60% of the cells and sets 15% of all cells to zero, so
    some kept entries hold zero.  ``empty`` keeps nothing.  ``one stored
    zero`` keeps exactly the last cell, holding zero: an operand whose only
    entry no value-based copy would keep.
    """

    generator = torch.Generator().manual_seed(100 + position)
    values = torch.rand(shape, generator=generator, dtype=DTYPE) + 0.5
    keep = torch.rand(shape, generator=generator) < 0.6
    values = values * keep
    values[torch.rand(shape, generator=generator) < 0.15] = 0.0
    if kind == "empty":
        return torch.zeros(shape, dtype=DTYPE), torch.zeros(shape, dtype=torch.bool)
    if kind == "one stored zero":
        keep = torch.zeros(shape, dtype=torch.bool)
        keep[tuple(extent - 1 for extent in shape)] = True
        return torch.zeros(shape, dtype=DTYPE), keep
    assert kind == "random"
    return values, keep


@pytest.mark.parametrize(
    "operands,result_indices,result_formats",
    CONVERTING,
    ids=[f"{','.join(o[1] for o in c[0])}->{c[1]}" for c in CONVERTING],
)
@pytest.mark.parametrize("arm", [0, 1])
def test_converting_programs_match_pytorch(
    operands, result_indices, result_formats, arm
):
    expression = ",".join(indices for _, _, indices in operands) + "->" + result_indices
    executed = 0
    for result_format in result_formats:
        kernel, error, cin = plan_of(operands, result_indices, result_format, arm)
        if kernel is None:
            continue
        result_shape, _ = shapes(operands, result_indices)
        conversions = {c.operand_id: c for c in kernel.conversions}
        symbols = {t.name: t.symbol_id for t in cin.get_rhs_tensor_vars()}
        # One program is one set of kernels: every input below runs the kernels
        # the first one built.
        for kind in INPUTS:
            tensors, references, indicators, converted_inputs = [], [], [], []
            stored_zeros = 0
            for position, (name, fmt, indices) in enumerate(operands):
                shape = tuple(EXTENT[letter] for letter in indices)
                values, keep = operand_values(kind, shape, position)
                # Every cell a dense level below a compressed one stores
                # without keeping is a stored zero too.
                cells = stored_cells(keep, fmt)
                stored_zeros += int((cells & (values == 0)).sum())
                tensors.append(stored(values, keep, fmt, name))
                references.append(values)
                indicators.append(cells.to(DTYPE))
                conversion = conversions.get(symbols[name])
                if conversion is None:
                    converted_inputs.append(tensors[-1])
                else:
                    order = conversion.mode_order
                    assert "sd" not in conversion.format
                    converted_inputs.append(
                        stored(
                            values.permute(order).contiguous(),
                            cells.permute(order).contiguous(),
                            conversion.format,
                            name,
                        )
                    )
            # The inputs' premises: stored zeros where the input promises
            # them, and nothing stored in the empty one.
            if kind == "empty":
                # An all-dense operand stores every cell whatever it is given.
                assert not any(
                    bool(indicator.any())
                    for indicator, (_, fmt, *_) in zip(indicators, operands)
                    if "s" in fmt
                )
            else:
                assert stored_zeros, "the operands must hold stored zeros"
            reference = torch.einsum(expression, *references)
            structure = torch.einsum(expression, *indicators) > 0
            _, expected_levels = level_arrays(structure, result_format)
            out, executed_kernel = execute_cin_via_loopir(
                program(operands, result_indices, result_format),
                result_shape,
                *tensors,
                compile_options=options(arm),
            )
            # Execution runs the conversion kernels and the main kernel the
            # plan emits at compile time.
            assert executed_kernel.cpp_source == kernel.cpp_source, result_format
            assert executed_kernel.conversion_sources == kernel.conversion_sources
            # The two kernels come from two constructions of the program, so
            # their operands' symbols differ; the decisions are compared.
            assert [(c.mode_order, c.format) for c in executed_kernel.conversions] == [
                (c.mode_order, c.format) for c in kernel.conversions
            ]
            direct, fmt = run_emitted(kernel, result_shape, converted_inputs, arm)
            assert fmt == result_format
            for route, result in (("execution entry", out), ("emitted kernel", direct)):
                got = result_cells(result, result_format, result_shape)
                assert torch.allclose(got, reference, rtol=1e-12, atol=1e-12), (
                    kind,
                    route,
                    result_format,
                    float((got - reference).abs().max()),
                )
                # The stored coordinates are the ones the operands' stored
                # entries imply, whatever their values.
                actual_levels = [
                    (
                        None
                        if level_kind == "d"
                        else tuple(t.reshape(-1).tolist() for t in arrays)
                    )
                    for level_kind, arrays in zip(
                        result_format, result.storage.index.mode_indices
                    )
                ]
                assert actual_levels == [
                    None if level is None else (level[0], level[1])
                    for level in expected_levels
                ], (kind, route, result_format)
        executed += 1
    assert executed, "no result format compiled"


def other_orders(rank: int, target: Tuple[int, ...]) -> List[Tuple[int, ...]]:
    """Up to three stored mode orders that are not the declared (identity) one.

    The order the plan converts to (the caller already stores what the plan
    wants), the reversed order and a rotation.
    """

    identity = tuple(range(rank))
    candidates = [tuple(target), identity[::-1], identity[1:] + identity[:1]]
    return [order for order in dict.fromkeys(candidates) if order != identity]


def stored_as(values, keep, fmt: str, name: str, order: Tuple[int, ...]):
    """``(tensor, cells)``: the operand stored with mode order ``order``.

    Storage level ``l`` holds logical mode ``order[l]`` in the layout
    ``fmt[l]``.  ``cells`` are the logical cells that storage holds: a dense
    level below a compressed one stores every cell under each stored parent,
    in the stored order, so they depend on ``order``.
    """

    physical_values = values.permute(order).contiguous()
    physical_keep = keep.permute(order).contiguous()
    leaves, levels = level_arrays(physical_keep, fmt)
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
    data = torch.tensor(
        [float(physical_values[leaf]) for leaf in leaves], dtype=values.dtype
    )
    tensor = STensor.from_components(
        tuple(physical_values.shape),
        fmt,
        mode_indices,
        data,
        name=name,
        mode_order=order,
    )
    inverse = [order.index(mode) for mode in range(len(order))]
    cells = stored_cells(physical_keep, fmt).permute(inverse).contiguous()
    return tensor, cells


def storage_arrays(tensor: STensor):
    """Copies of everything a tensor stores, to compare after a call."""

    return (
        tuple(tensor.shape),
        tuple(tensor.mode_order),
        [[array.clone() for array in level] for level in tensor._native_mode_indices()],
        tensor.values.clone(),
    )


def same_storage(before, tensor: STensor) -> bool:
    shape, mode_order, levels, values = before
    now = tensor._native_mode_indices()
    return (
        shape == tuple(tensor.shape)
        and mode_order == tuple(tensor.mode_order)
        and len(levels) == len(now)
        and all(
            len(old) == len(new) and all(torch.equal(a, b) for a, b in zip(old, new))
            for old, new in zip(levels, now)
        )
        and torch.equal(values, tensor.values)
    )


@pytest.mark.parametrize(
    "operands,result_indices,result_formats",
    CONVERTING,
    ids=[f"{','.join(o[1] for o in c[0])}->{c[1]}" for c in CONVERTING],
)
@pytest.mark.parametrize("arm", [0, 1])
def test_a_converted_operand_stored_in_another_order_is_converted_as_stored(
    operands, result_indices, result_formats, arm
):
    """The caller stores a converted operand in a mode order the program does not declare.

    The shipped compiler relayouts such an operand; the new compiler refused
    it (``CompileSpecError``) until 2026-10-12.  The alignment leaves an
    operand the plan converts as stored, and its conversion kernel is now
    built for the stored order: the same structural copy, reading the levels
    the caller holds, so one pass moves it into the planned layout.  The main
    kernel is the one the plan emits at compile time.  Checked per stored
    order (``other_orders``), on every input kind: values against dense
    PyTorch to 1e-12, the result's stored coordinates against the ones the
    operands' stored cells imply, the conversion source that ran, and the
    caller's operand, which the call must leave as it was.
    """

    expression = ",".join(indices for _, _, indices in operands) + "->" + result_indices
    result_shape, _ = shapes(operands, result_indices)
    executed = 0
    for result_format in result_formats:
        kernel, error, cin = plan_of(operands, result_indices, result_format, arm)
        if kernel is None or not kernel.conversions:
            continue
        symbols = {t.name: t.symbol_id for t in cin.get_rhs_tensor_vars()}
        plan_position = {c.operand_id: n for n, c in enumerate(kernel.conversions)}
        for moved, (moved_name, moved_fmt, _) in enumerate(operands):
            if symbols[moved_name] not in plan_position:
                continue
            which = plan_position[symbols[moved_name]]
            conversion = kernel.conversions[which]
            for order in other_orders(len(moved_fmt), conversion.mode_order):
                for kind in INPUTS:
                    tensors, references, indicators = [], [], []
                    for position, (name, fmt, indices) in enumerate(operands):
                        shape = tuple(EXTENT[letter] for letter in indices)
                        values, keep = operand_values(kind, shape, position)
                        if position == moved:
                            tensor, cells = stored_as(values, keep, fmt, name, order)
                        else:
                            tensor = stored(values, keep, fmt, name)
                            cells = stored_cells(keep, fmt)
                        tensors.append(tensor)
                        references.append(values)
                        indicators.append(cells.to(DTYPE))
                    assert tuple(tensors[moved].mode_order) == order
                    before = storage_arrays(tensors[moved])
                    reference = torch.einsum(expression, *references)
                    structure = torch.einsum(expression, *indicators) > 0
                    _, expected_levels = level_arrays(structure, result_format)
                    out, ran = execute_cin_via_loopir(
                        program(operands, result_indices, result_format),
                        result_shape,
                        *tensors,
                        compile_options=options(arm),
                    )
                    label = (result_format, moved_name, order, kind)
                    # The plan's main kernel, and a conversion kernel built for
                    # the stored order in place of the declared-order one.
                    assert ran.cpp_source == kernel.cpp_source, label
                    assert len(ran.conversion_sources) == len(kernel.conversions)
                    for n, source in enumerate(ran.conversion_sources):
                        assert (source == kernel.conversion_sources[n]) == (
                            n != which
                        ), label
                    assert same_storage(before, tensors[moved]), label
                    got = result_cells(out, result_format, result_shape)
                    assert torch.allclose(got, reference, rtol=1e-12, atol=1e-12), (
                        label,
                        float((got - reference).abs().max()),
                    )
                    actual_levels = [
                        (
                            None
                            if level_kind == "d"
                            else tuple(t.reshape(-1).tolist() for t in arrays)
                        )
                        for level_kind, arrays in zip(
                            result_format, out.storage.index.mode_indices
                        )
                    ]
                    assert actual_levels == [
                        None if level is None else (level[0], level[1])
                        for level in expected_levels
                    ], label
                executed += 1
    assert executed, "no converting program ran with an operand in another order"


def test_an_operand_stored_in_an_undeclared_order_must_be_one_the_plan_converts():
    """The check between compiling the kernel and building it.

    The kernel is compiled for the declared order.  An operand still stored in
    another order is read correctly only by its conversion kernel, so one the
    plan does not convert is refused: with equal extents no extent check would
    tell, and the kernel would read its levels in the wrong order.
    """

    from scorch.compiler.loopir.pipeline import (
        _require_converted_or_stored_as_declared,
    )
    from scorch.ops import _snapshot_runtime_tensors

    operands = (("A", "ss", "ij"),)
    kernel, _, cin = plan_of(operands, "ji", "ss", 0)
    assert kernel is not None and len(kernel.conversions) == 1
    values, keep = operand_values("random", (EXTENT["i"], EXTENT["j"]), 0)

    as_declared = _snapshot_runtime_tensors(
        (stored(values, keep, "ss", "A"),), expected_count=1
    )
    _require_converted_or_stored_as_declared(cin, as_declared, ())
    _require_converted_or_stored_as_declared(cin, as_declared, kernel.conversions)

    tensor, _ = stored_as(values, keep, "ss", "A", (1, 0))
    moved = _snapshot_runtime_tensors((tensor,), expected_count=1)
    _require_converted_or_stored_as_declared(cin, moved, kernel.conversions)
    with pytest.raises(LoopIRLoweringError) as refused:
        _require_converted_or_stored_as_declared(cin, moved, ())
    assert refused.value.defect.code == "unconverted_operand_order"
    assert "'A' is stored in mode order (1, 0)" in str(refused.value)
    assert "declares (0, 1)" in str(refused.value)


def test_a_conversion_of_a_level_kind_the_copy_does_not_read_is_refused():
    """A plan that converts a coordinate-level operand is refused with a code.

    The candidate search never proposes one (``_conversion_candidates``); the
    helper that names an operand's levels refuses instead of asserting, so
    the refusal survives ``python -O``.
    """

    from scorch.compiler.loopir.pipeline import _conversion_letters

    assert _conversion_letters(TensorVar("A", fmt="ds", dtype=DTYPE)) == "ds"
    with pytest.raises(LoopIRLoweringError) as refused:
        _conversion_letters(TensorVar("A", fmt="oo", dtype=DTYPE))
    assert refused.value.defect.code == "unsupported_operand_conversion"


def test_a_conversion_kernel_keeps_every_stored_entry():
    """The conversion kernel alone: ``"sdsd"`` to every level compressed, reversed.

    The converted tensor's arrays are exactly those of the operand's stored
    cells (zeros included) in the new order.
    """

    from scorch.ops import (
        _load_validated_prepared_kernel,
        _prepare_generated_kernel_build,
    )
    from scorch.compiler.loopir.operand_conversion import (
        conversion_format,
        conversion_source,
        converted_shape,
    )

    fmt, order = "sdsd", (3, 2, 1, 0)
    assert conversion_format(fmt) == "ssss"
    assert conversion_format("ds") == "ds" and conversion_format("dss") == "dss"
    shape = (5, 4, 3, 2)
    generator = torch.Generator().manual_seed(7)
    values = torch.rand(shape, generator=generator, dtype=DTYPE) + 0.5
    keep = torch.rand(shape, generator=generator) < 0.3
    values = values * keep
    cells = stored_cells(keep, fmt)
    assert int(cells.sum()) > int(keep.sum()), "the operand must store zeros"
    operand = stored(values, keep, fmt, "A")
    compile_options = CompileOptions.from_environment()
    conversion = OperandConversion(
        TensorVar("A", fmt=fmt, dtype=DTYPE).symbol_id, order, "ssss"
    )
    source = conversion_source(
        fmt, conversion, DTYPE, shape, compile_options=compile_options
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
    assert new_shape == (2, 3, 4, 5)
    from scorch.stensor import _wrap_generated_result

    converted = _wrap_generated_result(
        new_shape,
        "ssss",
        module.evaluate(
            list(new_shape),
            list(operand.shape),
            operand._native_mode_indices(),
            operand.values,
        ),
        mode_order=(0, 1, 2, 3),
    )
    leaves, levels = level_arrays(cells.permute(order), "ssss")
    actual = [
        tuple(t.reshape(-1).tolist() for t in arrays)
        for arrays in converted.storage.index.mode_indices
    ]
    assert actual == [(level[0], level[1]) for level in levels]
    expected = torch.tensor(
        [float(values.permute(order)[leaf]) for leaf in leaves], dtype=DTYPE
    )
    assert torch.equal(converted.storage.value.reshape(-1).to(DTYPE), expected)
