"""Execution against dense PyTorch for programs the new compiler used to refuse.

Several test files pinned a refusal of the new compiler (normalized CIN ->
LoopPlan -> LoopIR -> LLIR) that it no longer makes: a result stored in an
order no operand is stored in, a compressed tensor stored in a non-identity
mode order, an all-dense tensor read or written against its storage order.
Each of those files now states what the program compiles to and runs it here.

A program is spelled as in ``test_loopir_storage_order_form``: an operand is
``(name, format, indices, mode order)`` and the result ``(indices, format,
mode order)``, where a mode order of ``None`` is the identity.  The check is
that file's: operands built level by level in the declared layout and mode
order (no conversion kernel), holding stored zeros; float64 values against
``torch.einsum`` to 1e-12; and for a result with a compressed level exactly
the stored coordinates the operands' stored entries imply, compared in the
result's storage order.  Every program runs on a random input, on operands
that store nothing, and on operands whose only stored entry is a zero.

The program goes through the scheduled execution entry, which leaves an
all-dense tensor as the caller stored it and runs the conversion kernels the
plan records, so the kernels that run are the ones the same request emits at
compile time; the check asserts that too.
"""

from typing import Mapping, Optional, Sequence, Tuple

import torch

from scorch.compiler.loopir.pipeline import (
    compile_cin_via_loopir,
    execute_cin_via_loopir,
)

from tests.test_scorch.test_loopir_operand_conversions import (
    DTYPE,
    EXTENT,
    INPUTS,
    level_arrays,
    operand_values,
    options,
    result_cells,
)
from tests.test_scorch.test_loopir_storage_order_form import (
    letters_of,
    logical_stored_cells,
    order_of,
    program,
    stored_in_order,
)

Operand = Tuple[str, str, str, Optional[Tuple[int, ...]]]
Result = Tuple[str, str, Optional[Tuple[int, ...]]]


def storage_shape(indices: str, mode_order, extent: Mapping[str, int]):
    return tuple(extent[indices[mode]] for mode in order_of(indices, mode_order))


def assert_program_matches_pytorch(
    operands: Sequence[Operand],
    result: Result,
    arm: int,
    *,
    loop_order: Optional[Sequence[str]] = None,
    extent: Mapping[str, int] = EXTENT,
):
    """Compile and run one program; return ``(kernel, statement)`` of the compile.

    ``arm`` is the register-blocking setting and ``loop_order`` a requested
    loop order (the automatic plan when omitted).  ``statement`` is the CIN
    the returned kernel was compiled from, so a caller can ask the scheduler
    about the same index variables the kernel's plan names.  Every execution
    must run the text and the conversion kernels of that compile.
    """

    result_indices, result_format, result_order = result
    letters = letters_of(operands)
    compile_options = options(arm, None if loop_order is None else tuple(loop_order))
    result_shape = storage_shape(result_indices, result_order, extent)
    statement = program(operands, result, letters)
    kernel = compile_cin_via_loopir(
        statement,
        result_shape,
        tuple(
            (storage_shape(indices, mode_order, extent), DTYPE)
            for _, _, indices, mode_order in operands
        ),
        compile_options=compile_options,
    )
    expression = (
        ",".join(indices for _, _, indices, _ in operands) + "->" + result_indices
    )
    logical_shape = tuple(extent[letter] for letter in result_indices)
    declared = order_of(result_indices, result_order)
    for kind in INPUTS:
        tensors, references, indicators = [], [], []
        stored_zeros = 0
        for position, (name, fmt, indices, mode_order) in enumerate(operands):
            shape = tuple(extent[letter] for letter in indices)
            values, keep = operand_values(kind, shape, position)
            cells = logical_stored_cells(keep, fmt, mode_order)
            stored_zeros += int((cells & (values == 0)).sum())
            tensors.append(stored_in_order(values, keep, fmt, name, mode_order))
            references.append(values)
            indicators.append(cells.to(DTYPE))
        # The inputs' premises: nothing stored in the empty one (an all-dense
        # operand stores every cell whatever it is given), stored zeros in the
        # others.
        if kind == "empty":
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
            program(operands, result, letters),
            result_shape,
            *tensors,
            compile_options=compile_options,
        )
        assert executed.cpp_source == kernel.cpp_source, kind
        assert executed.conversion_sources == kernel.conversion_sources, kind
        stored_order = getattr(out, "mode_order", None)
        assert tuple(stored_order or range(len(declared))) == tuple(declared), kind
        assert tuple(out.shape) == result_shape, kind
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
    return kernel, statement
