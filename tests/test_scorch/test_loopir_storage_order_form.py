"""A tensor with a compressed level may be stored in any mode order.

A tensor's mode order says which logical mode each storage level holds.  The
public einsum declares one whenever it stores a result in the loop order
instead of the order the caller wrote (``ijk->ikj`` kept as ``i, j, k`` with
mode order ``(0, 2, 1)``), and a caller holds one whenever a tensor was built
that way (a CSC matrix is ``"ds"`` in mode order ``(1, 0)``).  The new compiler
used to refuse every such tensor that has a compressed level
(``unsupported_mode_order``).

It now lowers the program in storage-order form
(``loop_plan.storage_order_form``): the tensor stored in mode order ``m`` and
accessed ``A[x_0, ..., x_n]`` holds the same arrays as the identity-ordered
tensor of the same storage-order shape accessed ``A[x_{m_0}, ..., x_{m_n}]``.
Nothing below that rewrite sees a permuted compressed tensor, the kernel's
arguments are unchanged, and the result is wrapped in its declared order
(``LoopIRCompiledKernel.result_mode_order``).

Three kinds of test.  The rewrite itself, on small statements.  Each permuted
program against its twin, the same program written by hand in identity order:
the two must emit the same C++, conversion kernels included.  And execution
against dense PyTorch in float64 on operands built level by level in the
declared layout and mode order: values to 1e-12, and for a result with a
compressed level exactly the stored coordinates the operands' stored entries
imply, compared in the result's storage order.  The operands hold stored
zeros, and the mode orders include three-cycles, which are not their own
inverses, so reading a permutation backwards fails.
"""

from typing import List, Optional, Sequence, Tuple

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
from scorch.compiler.cin_analysis import normalize_cin
from scorch.compiler.loop_plan import out_of_order_modes, storage_order_form
from scorch.compiler.loopir.pipeline import (
    compile_cin_via_loopir,
    execute_cin_via_loopir,
)

from tests.test_scorch.test_loopir_operand_conversions import (
    DTYPE,
    EXTENT,
    REFUSALS,
    level_arrays,
    operand_values,
    options,
    result_cells,
    stored_cells,
)

Operand = Tuple[str, str, str, Optional[Tuple[int, ...]]]
Result = Tuple[str, str, Optional[Tuple[int, ...]]]


def order_of(indices: str, mode_order: Optional[Sequence[int]]) -> List[int]:
    return list(range(len(indices))) if mode_order is None else list(mode_order)


def program(operands: Sequence[Operand], result: Result, letters: str):
    """``C[result] (+)= product of operands`` over index variables ``letters``.

    An operand is ``(name, format, indices, mode order)`` and the result
    ``(indices, format, mode order)``; a mode order of ``None`` is identity.
    ``letters`` fixes the order the index variables are created and nested
    in, so a program and its twin differ in nothing but what is under test.
    """

    variables = {letter: IndexVar(letter) for letter in letters}

    def access(name, fmt, indices, mode_order):
        tensor = TensorVar(name, fmt=fmt, dtype=DTYPE)
        if mode_order is not None:
            tensor.mode_order = list(mode_order)
        return tensor[tuple(variables[letter] for letter in indices)]

    product = None
    for name, fmt, indices, mode_order in operands:
        factor = access(name, fmt, indices, mode_order)
        product = (
            factor if product is None else CINBinaryOp(Operation.MUL, product, factor)
        )
    result_indices, result_format, result_order = result
    target = access("C", result_format, result_indices, result_order)
    if any(letter not in result_indices for letter in letters):
        statement = TensorAssign(target, product, op=Operation.ADD)
    else:
        statement = TensorAssign(target, product)
    for letter in reversed(letters):
        statement = ForAll(variables[letter], statement)
    return statement


def twin(operands: Sequence[Operand], result: Result):
    """The same program with every tensor written in its storage order."""

    def stored_indices(indices, mode_order):
        return "".join(indices[mode] for mode in order_of(indices, mode_order))

    return (
        [
            (name, fmt, stored_indices(indices, mode_order), None)
            for name, fmt, indices, mode_order in operands
        ],
        (stored_indices(result[0], result[2]), result[1], None),
    )


def storage_shape(indices: str, mode_order) -> Tuple[int, ...]:
    return tuple(EXTENT[indices[mode]] for mode in order_of(indices, mode_order))


def letters_of(operands: Sequence[Operand]) -> str:
    return "".join(dict.fromkeys("".join(indices for _, _, indices, _ in operands)))


def compiled(operands, result, arm):
    """``(kernel, None)``, or ``(None, "<type>/<code>")`` for a refusal."""

    try:
        return (
            compile_cin_via_loopir(
                program(operands, result, letters_of(operands)),
                storage_shape(result[0], result[2]),
                tuple(
                    (storage_shape(indices, mode_order), DTYPE)
                    for _, _, indices, mode_order in operands
                ),
                compile_options=options(arm),
            ),
            None,
        )
    except REFUSALS as error:
        defect = getattr(error, "defect", None)
        code = getattr(defect, "code", None) or getattr(error, "code", None)
        return None, f"{type(error).__name__}/{code}"


# -- the rewrite ---------------------------------------------------------------


def test_a_permuted_compressed_tensor_is_written_in_its_storage_order():
    statement = program(
        [("A", "sss", "ijk", (1, 2, 0))], ("ijk", "sss", (2, 0, 1)), "ijk"
    )
    before = normalize_cin(statement)
    rewritten = storage_order_form(statement)
    assert rewritten is not statement
    (operand,) = rewritten.get_rhs_tensor_accesses()
    (target,) = rewritten.get_result_tensor_accesses()
    # Level l of a tensor stored in mode order m holds logical mode m[l].
    assert [index.name for index in operand.indices] == ["j", "k", "i"]
    assert [index.name for index in target.indices] == ["k", "i", "j"]
    assert operand.tensor.mode_order == [0, 1, 2]
    assert target.tensor.mode_order == [0, 1, 2]
    # The copy keeps every identity, so a plan made on the declared program
    # names the same loops and tensors in the rewritten one.
    (declared,) = before.get_rhs_tensor_accesses()
    assert operand.tensor.symbol_id == declared.tensor.symbol_id
    assert {index.index_id for index in operand.indices} == {
        index.index_id for index in declared.indices
    }
    # The caller's statement is not touched.
    (original,) = statement.get_rhs_tensor_accesses()
    assert [index.name for index in original.indices] == ["i", "j", "k"]
    assert original.tensor.mode_order == [1, 2, 0]


def test_every_access_to_one_tensor_is_rewritten_from_the_same_order():
    i, j = IndexVar("i"), IndexVar("j")
    tensor = TensorVar("A", fmt="ss", dtype=DTYPE)
    tensor.mode_order = [1, 0]
    target = TensorVar("C", fmt="ss", dtype=DTYPE)[i, j]
    statement = ForAll(
        i,
        ForAll(
            j,
            TensorAssign(
                target, CINBinaryOp(Operation.MUL, tensor[i, j], tensor[j, i])
            ),
        ),
    )
    rewritten = storage_order_form(statement)
    first, second = rewritten.get_rhs_tensor_accesses()
    assert first.tensor is second.tensor
    assert first.tensor.mode_order == [0, 1]
    assert [index.name for index in first.indices] == ["j", "i"]
    assert [index.name for index in second.indices] == ["i", "j"]


@pytest.mark.parametrize(
    "operands,result",
    [
        ([("A", "ss", "ij", None)], ("ij", "ss", None)),
        ([("A", "ds", "ij", None), ("B", "sd", "jk", None)], ("ik", "ds", None)),
        # An all-dense tensor keeps its mode order: it is addressed by coordinate.
        ([("A", "dd", "ij", (1, 0))], ("ij", "dd", (1, 0))),
        ([("A", "ddd", "ijk", (2, 0, 1)), ("B", "ss", "ij", None)], ("ij", "ss", None)),
    ],
)
def test_a_program_with_no_permuted_compressed_tensor_is_returned_itself(
    operands, result
):
    statement = program(operands, result, letters_of(operands))
    assert storage_order_form(statement) is statement
    for access in [
        *statement.get_rhs_tensor_accesses(),
        *statement.get_result_tensor_accesses(),
    ]:
        assert out_of_order_modes(access) is None


def test_a_mode_order_that_is_no_permutation_is_left_for_the_lowering_to_refuse():
    statement = program([("A", "ss", "ij", (0, 0))], ("ij", "ss", None), "ij")
    assert storage_order_form(statement) is statement


# -- a permuted program and its identity-order twin -----------------------------

# Results stored in the loop order, as the public einsum declares them; operands
# stored in an order the caller chose; both; and three-cycles on each side.
PERMUTED = [
    ([("A", "ddd", "ijk", None)], ("ikj", "dss", (0, 2, 1))),
    ([("A", "sss", "ijk", None)], ("kji", "dsd", (2, 1, 0))),
    ([("A", "ss", "ij", None)], ("ji", "ss", (1, 0))),
    ([("A", "ds", "ij", None)], ("ji", "ds", (1, 0))),
    ([("A", "sss", "ijk", None)], ("jki", "sss", (2, 0, 1))),
    ([("A", "dss", "ijk", None)], ("kij", "dss", (1, 2, 0))),
    ([("A", "dsss", "ijkl", None)], ("ji", "ss", (1, 0))),
    ([("A", "ds", "ij", (1, 0))], ("ij", "ss", None)),
    ([("A", "ds", "ij", (1, 0))], ("ji", "ss", None)),
    ([("A", "ds", "ij", (1, 0)), ("x", "d", "j", None)], ("i", "d", None)),
    ([("A", "ss", "ij", (1, 0)), ("B", "ss", "jk", None)], ("ik", "ss", None)),
    ([("A", "sss", "ijk", (1, 2, 0))], ("ijk", "sss", None)),
    ([("A", "sss", "ijk", (1, 2, 0))], ("jki", "sss", None)),
    ([("A", "dss", "ijk", (2, 0, 1))], ("kij", "dss", (1, 2, 0))),
    ([("A", "sds", "ijk", (2, 0, 1)), ("B", "ds", "kj", (1, 0))], ("ik", "ss", (1, 0))),
    ([("A", "ssd", "ijk", (1, 0, 2)), ("B", "ds", "kj", None)], ("ik", "ds", None)),
]


def label(case) -> str:
    operands, result = case

    def spell(fmt, indices, mode_order):
        suffix = "" if mode_order is None else "@" + "".join(map(str, mode_order))
        return f"{fmt}:{indices}{suffix}"

    return (
        ",".join(spell(fmt, indices, order) for _, fmt, indices, order in operands)
        + "->"
        + spell(result[1], result[0], result[2])
    )


@pytest.mark.parametrize("case", PERMUTED, ids=[label(case) for case in PERMUTED])
@pytest.mark.parametrize("arm", [0, 1])
def test_a_permuted_program_emits_what_its_identity_order_twin_emits(case, arm):
    operands, result = case
    letters = letters_of(operands)
    twin_operands, twin_result = twin(operands, result)
    kernels = []
    for spelled_operands, spelled_result in (
        (operands, result),
        (twin_operands, twin_result),
    ):
        try:
            kernels.append(
                compile_cin_via_loopir(
                    program(spelled_operands, spelled_result, letters),
                    storage_shape(spelled_result[0], spelled_result[2]),
                    tuple(
                        (storage_shape(indices, mode_order), DTYPE)
                        for _, _, indices, mode_order in spelled_operands
                    ),
                    compile_options=options(arm),
                )
            )
        except REFUSALS as error:
            defect = getattr(error, "defect", None)
            kernels.append(
                f"{type(error).__name__}/"
                f"{getattr(defect, 'code', None) or getattr(error, 'code', None)}"
            )
    permuted, identity = kernels
    if isinstance(identity, str):
        # Whatever refuses the twin refuses the permuted program, by that code.
        assert permuted == identity
        return
    assert not isinstance(permuted, str), permuted
    assert permuted.cpp_source == identity.cpp_source
    assert permuted.conversion_sources == identity.conversion_sources
    assert permuted.result_mode_order == tuple(order_of(result[0], result[2]))
    assert identity.result_mode_order == tuple(range(len(result[0])))


def test_each_listed_program_has_a_permuted_compressed_tensor():
    """So each one was refused for its mode order before the rewrite existed."""

    for operands, result in PERMUTED:
        statement = program(operands, result, letters_of(operands))
        assert storage_order_form(statement) is not statement, label((operands, result))


# -- execution -----------------------------------------------------------------


def stored_in_order(values, keep, fmt, name, mode_order) -> STensor:
    """The tensor holding ``keep`` of ``values``, stored in ``mode_order``."""

    order = order_of("x" * values.dim(), mode_order)
    physical_values = values.permute(order).contiguous()
    leaves, levels = level_arrays(keep.permute(order).contiguous(), fmt)
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
    return STensor.from_components(
        tuple(physical_values.shape),
        fmt,
        mode_indices,
        data,
        name=name,
        mode_order=order,
    )


def logical_stored_cells(keep, fmt, mode_order) -> torch.Tensor:
    """The logical cells a tensor holding ``keep`` stores in that layout and order."""

    order = order_of("x" * keep.dim(), mode_order)
    inverse = [order.index(mode) for mode in range(len(order))]
    return stored_cells(keep.permute(order).contiguous(), fmt).permute(inverse)


def test_the_operand_builder_stores_what_it_is_given():
    """The harness's own premise, on a three-cycle: the tensor reads back exactly."""

    shape = (EXTENT["i"], EXTENT["j"], EXTENT["k"])
    values, keep = operand_values("random", shape, 0)
    order = (1, 2, 0)
    tensor = stored_in_order(values, keep, "sss", "A", order)
    assert tuple(tensor.mode_order) == order
    assert tuple(tensor.shape) == tuple(shape[mode] for mode in order)
    assert torch.equal(result_cells(tensor, "sss", shape), values * keep)
    assert torch.equal(logical_stored_cells(keep, "sss", order), keep)
    # Dense levels at the bottom, below a compressed one, store every cell
    # under a stored parent; which cells those are depends on the storage
    # order.  One kept cell (i=1, j=2, k=0): stored in order (j, k, i) the
    # dense last level runs over i, and stored in order (i, j, k) over k.
    one = torch.zeros(shape, dtype=torch.bool)
    one[1, 2, 0] = True
    over_i = torch.zeros(shape, dtype=torch.bool)
    over_i[:, 2, 0] = True
    over_k = torch.zeros(shape, dtype=torch.bool)
    over_k[1, 2, :] = True
    assert torch.equal(logical_stored_cells(one, "ssd", order), over_i)
    assert torch.equal(logical_stored_cells(one, "ssd", None), over_k)


@pytest.mark.parametrize("case", PERMUTED, ids=[label(case) for case in PERMUTED])
@pytest.mark.parametrize("arm", [0, 1])
def test_permuted_programs_match_pytorch(case, arm):
    operands, result = case
    result_indices, result_format, result_order = result
    kernel, refusal = compiled(operands, result, arm)
    if kernel is None:
        # Refused exactly as its identity-order twin is (the test above); a
        # refusal for the mode order itself is the defect this file is about.
        assert "unsupported_mode_order" not in refusal
        pytest.skip(refusal)
    expression = (
        ",".join(indices for _, _, indices, _ in operands) + "->" + result_indices
    )
    logical_shape = tuple(EXTENT[letter] for letter in result_indices)
    declared = order_of(result_indices, result_order)
    for kind in ("random", "empty", "one stored zero"):
        tensors, references, indicators = [], [], []
        stored_zeros = 0
        for position, (name, fmt, indices, mode_order) in enumerate(operands):
            shape = tuple(EXTENT[letter] for letter in indices)
            values, keep = operand_values(kind, shape, position)
            cells = logical_stored_cells(keep, fmt, mode_order)
            stored_zeros += int((cells & (values == 0)).sum())
            tensors.append(stored_in_order(values, keep, fmt, name, mode_order))
            references.append(values)
            indicators.append(cells.to(DTYPE))
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
        # The result's levels are in its declared storage order.
        _, expected_levels = level_arrays(
            structure.permute(declared).contiguous(), result_format
        )
        out, executed = execute_cin_via_loopir(
            program(operands, result, letters_of(operands)),
            storage_shape(result_indices, result_order),
            *tensors,
            compile_options=options(arm),
        )
        assert executed.cpp_source == kernel.cpp_source
        assert executed.conversion_sources == kernel.conversion_sources
        assert tuple(out.mode_order) == tuple(declared)
        assert tuple(out.shape) == storage_shape(result_indices, result_order)
        got = result_cells(out, result_format, logical_shape)
        assert torch.allclose(got, reference, rtol=1e-12, atol=1e-12), (
            kind,
            float((got - reference).abs().max()),
        )
        actual_levels = [
            (
                None
                if level_kind == "d"
                else tuple(t.reshape(-1).tolist() for t in arrays)
            )
            for level_kind, arrays in zip(result_format, out.storage.index.mode_indices)
        ]
        assert actual_levels == [
            None if level is None else (level[0], level[1]) for level in expected_levels
        ], kind
