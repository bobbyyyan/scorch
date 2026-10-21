"""The new compiler on results with a dense level below a compressed one.

``test_dense_below_compressed_result.py`` states the storage rules and runs a
table of programs through the public ``scorch.einsum``, which the shipped
compiler serves.  This file runs the same table through the new compiler
(normalized CIN -> LoopPlan -> LoopIR -> LLIR) on both automatic arms, with the
same inputs and the same judge: canonical level arrays, values against a
float64 ``torch.einsum``, and the stored coordinates the operands' stored
structure implies.

A program the new compiler does not lower is listed in ``REFUSED`` with the
code it exits with, so the table is also the list of what the new compiler
still owes this family.  The check is strict in both directions: a listed
program that starts compiling fails until it is removed, and an unlisted
program that is refused fails.  The table is empty: the last two entries, the
transposes ``ds ij->ji`` and ``ss ij->ji`` into ``sd``, compile since the plan
can convert an operand to the result's order.
"""

import pytest
import torch

from scorch.compiler.cin import (
    BinaryOp as CINBinaryOp,
    ForAll,
    IndexVar,
    Operation,
    TensorAssign,
    TensorVar,
)
from scorch.compiler.loop_plan_legality import InvalidSchedule
from scorch.compiler.loopir.lower_cin import LoopIRLoweringError
from scorch.compiler.loopir.lower_llir import LoopIRTargetError
from scorch.compiler.loopir.pipeline import (
    compile_cin_via_loopir,
    execute_cin_via_loopir,
    legacy_generated_cpp,
)
from tests.test_scorch.loopir_executed_programs import (
    assert_program_matches_pytorch,
)
from tests.test_scorch.test_dense_below_compressed_result import (
    CELLS,
    EXTENT,
    INPUTS,
    _cell_id,
    _inputs,
    _judge,
    _operand,
)
from tests.test_scorch.test_loopir_sparse_workspace_target import auto_options

# Every program of the table is lowered.  Since 2026-10-12: reductions that
# bind a result coordinate below a reduced loop (``ijk->ik``, ``ijk->jk``,
# ``ij,jk->ik``, ``ijk,kl->ijl``) drain a sorted coordinate workspace into the
# dense block below the committed coordinate, and intersections at a level the
# result stores dense (``ij,ij->ij``, ``ijk,ijk->ijk``) bind the dense level
# from the intersection's aligned coordinate.  The two transposes, whose result
# is stored in an order the operand is not stored in, were refused by the
# plan's legality check (``InvalidSchedule``) until the plan could convert the
# operand; ``TRANSPOSED`` records the conversion each one gets.
REFUSED: dict = {}

# program -> (mode order the operand is converted to, layout it is converted into)
TRANSPOSED = {
    "ds ij->ji as sd": ((1, 0), "ds"),
    "ss ij->ji as sd": ((1, 0), "ss"),
}


def _statement(expr, operand_fmts, result_fmt):
    """The einsum as CIN, loops in order of each index's first appearance."""

    terms, output = expr.split("->")[0].split(","), expr.split("->")[1]
    letters = "".join(dict.fromkeys("".join(terms)))
    variables = {letter: IndexVar(letter) for letter in letters}
    product = None
    for name, term, fmt in zip("ABD", terms, operand_fmts):
        access = TensorVar(name, fmt=fmt, dtype=torch.float32)[
            tuple(variables[letter] for letter in term)
        ]
        product = (
            access if product is None else CINBinaryOp(Operation.MUL, product, access)
        )
    result = TensorVar("C", fmt=result_fmt, dtype=torch.float32)[
        tuple(variables[letter] for letter in output)
    ]
    reduces = set(letters) - set(output)
    statement = TensorAssign(result, product, op=Operation.ADD if reduces else None)
    for letter in reversed(letters):
        statement = ForAll(variables[letter], statement)
    return statement


def _refusal(expr, operand_fmts, result_fmt, arm):
    """The code the new compiler exits with for this program, or ``None``."""

    terms, output = expr.split("->")[0].split(","), expr.split("->")[1]
    try:
        compile_cin_via_loopir(
            _statement(expr, operand_fmts, result_fmt),
            tuple(EXTENT[letter] for letter in output),
            tuple(
                (tuple(EXTENT[letter] for letter in term), torch.float32)
                for term in terms
            ),
            compile_options=auto_options(arm),
        )
    except (LoopIRLoweringError, LoopIRTargetError) as error:
        return error.defect.code
    except InvalidSchedule:
        return "InvalidSchedule"
    return None


def _run(expr, operand_fmts, result_fmt, kind, seed, arm):
    terms = expr.split("->")[0].split(",")
    operands, indicators, denses = [], [], []
    pairs = _inputs(kind, terms, seed)
    for (dense, pattern, ghosts), fmt, name in zip(pairs, operand_fmts, "ABD"):
        operand, indicator = _operand(dense, pattern, fmt, name, ghosts)
        operands.append(operand)
        indicators.append(indicator)
        denses.append(dense.to(torch.float64))
    reference = torch.einsum(expr, *denses)
    result = execute_cin_via_loopir(
        _statement(expr, operand_fmts, result_fmt),
        tuple(reference.shape),
        *operands,
        compile_options=auto_options(arm, jit=True),
    )
    result = result[0] if isinstance(result, tuple) else result
    structural = torch.einsum(expr, *indicators) > 0
    _judge(result, result_fmt, reference, structural, dtype=pairs[0][0].dtype)


def assert_statement_matches_pytorch(statement, arm, seed=0):
    """Run a product statement on the new compiler and judge it as this file does.

    For the tests of neighbouring families whose programs with a dense level
    below a compressed one used to be refused: the statement is read back into
    an einsum (index names in loop order, one term per operand), run on every
    input of ``INPUTS``, and held to the same three checks.
    """

    loops, body = [], statement
    while isinstance(body, ForAll):
        loops.append(body.get_index_var())
        body = body.stmt
    assert isinstance(body, TensorAssign)
    letter = {id(index_var): "ijkl"[n] for n, index_var in enumerate(loops)}

    def term(access):
        return "".join(letter[id(index_var)] for index_var in access.indices)

    def layout(access):
        return str(access.tensor.format).replace(",", "")

    def factors(expression):
        if isinstance(expression, CINBinaryOp):
            assert expression.op is Operation.MUL, "a product statement"
            return factors(expression.left) + factors(expression.right)
        return [expression]

    operands = factors(body.rhs)
    expr = ",".join(term(access) for access in operands) + "->" + term(body.lhs)
    operand_fmts = tuple(layout(access) for access in operands)
    failures = []
    for kind in INPUTS:
        try:
            _run(expr, operand_fmts, layout(body.lhs), kind, seed, arm)
        except Exception as error:  # every input is reported, not the first
            failures.append(f"{kind}: {type(error).__name__}: {str(error)[:400]}")
    assert not failures, f"{expr} {operand_fmts}\n" + "\n".join(failures)


def test_the_refusal_table_names_only_programs_of_this_table():
    assert set(REFUSED) <= {_cell_id(cell) for cell in CELLS}


@pytest.mark.parametrize("arm", [False, True], ids=["plain", "blocked"])
@pytest.mark.parametrize(
    "cell,seed",
    [pytest.param(cell, seed, id=_cell_id(cell)) for seed, cell in enumerate(CELLS)],
)
def test_result_is_canonical_exact_and_structural(cell, seed, arm):
    expr, operand_fmts, result_fmt = cell
    listed = REFUSED.get(_cell_id(cell))
    code = _refusal(expr, operand_fmts, result_fmt, arm)
    assert code == listed, (
        f"the new compiler {'compiles this' if code is None else 'exits ' + code}; "
        f"the table says {'it compiles' if listed is None else listed}"
    )
    if code is not None:
        return
    failures = []
    for kind in INPUTS:
        try:
            _run(expr, operand_fmts, result_fmt, kind, seed, arm)
        except Exception as error:  # every input is reported, not the first
            failures.append(f"{kind}: {type(error).__name__}: {str(error)[:400]}")
    assert not failures, "\n".join(failures)


@pytest.mark.parametrize("arm", [False, True], ids=["plain", "blocked"])
@pytest.mark.parametrize("name", sorted(TRANSPOSED))
def test_a_transpose_converts_the_operand_and_the_shipped_plan_refuses_it(name, arm):
    """What the two formerly refused programs compile to, and against what.

    The plan records one conversion of the operand to the result's order.  The
    shipped compiler's plan origin has none and still raises
    ``InvalidSchedule`` (``sparse_parent_dominance``), so there is no shipped
    text to compare with.  ``test_result_is_canonical_exact_and_structural``
    runs both programs on every input of the table, and the test below runs
    them in float64.
    """

    cell = next(cell for cell in CELLS if _cell_id(cell) == name)
    expr, operand_fmts, result_fmt = cell
    terms, output = expr.split("->")[0].split(","), expr.split("->")[1]
    result_shape = tuple(EXTENT[letter] for letter in output)
    operand_bindings = tuple(
        (tuple(EXTENT[letter] for letter in term), torch.float32) for term in terms
    )
    kernel = compile_cin_via_loopir(
        _statement(expr, operand_fmts, result_fmt),
        result_shape,
        operand_bindings,
        compile_options=auto_options(arm),
    )
    assert [(c.mode_order, c.format) for c in kernel.conversions] == [TRANSPOSED[name]]
    with pytest.raises(InvalidSchedule) as error:
        legacy_generated_cpp(
            _statement(expr, operand_fmts, result_fmt),
            result_shape,
            operand_bindings,
            compile_options=auto_options(arm),
        )
    assert [d.code for d in error.value.diagnostics] == ["sparse_parent_dominance"]


@pytest.mark.parametrize("arm", [0, 1], ids=["plain", "blocked"])
@pytest.mark.parametrize("name", sorted(TRANSPOSED))
def test_a_transpose_matches_pytorch_in_float64(name, arm):
    """Values to 1e-12 and exactly the stored coordinates the operand implies."""

    operand_fmt, expr = name.split(" as ")[0].split(" ")
    operand_indices, result_indices = expr.split("->")
    kernel, _ = assert_program_matches_pytorch(
        (("A", operand_fmt, operand_indices, None),),
        (result_indices, "sd", None),
        arm,
    )
    assert [(c.mode_order, c.format) for c in kernel.conversions] == [TRANSPOSED[name]]
