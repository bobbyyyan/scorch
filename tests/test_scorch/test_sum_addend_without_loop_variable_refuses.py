"""A sum whose addends do not share a loop variable is refused.

In ``C[i,j] = A[i,j] + x[j]`` the vector has no ``i``.  It is defined at every ``i``, so
the sum is defined at every ``i`` too.  The lowering built the loop over ``i`` from the
addends that have ``i`` (``A`` alone) and then removed ``x`` from the loop's body as an
operand that had run out of stored values: the kernel returned ``A``.  Every layout did
this -- ``A`` dense or compressed, ``x`` dense or compressed, the missing variable outer
or inner -- and nothing was refused.

No public operation builds such a program (``A + B`` requires equal shapes and
``einsum`` multiplies), so it is reached only through hand-written CIN on the developer
entry.  A loop over the whole extent of the missing variable is not written; the
lowering now refuses with ``unsupported_sum_addend_without_loop_variable`` and names the
variable.  A product is unaffected: a factor without the loop's variable is constant
across the loop.
"""

from dataclasses import replace

import pytest
import torch

from scorch.compiler.cin import (
    BinaryOp,
    ForAll,
    IndexVar,
    Operation,
    TensorAssign,
    TensorVar,
)
from scorch.compiler.compile_options import CompileOptions
from scorch.compiler.diagnostics import UnsupportedFeature
from scorch.compiler.loopir.pipeline import legacy_generated_cpp
from scorch.compiler.scheduler import Schedule
from scorch.ops import lower_and_exec_cin
from tests.test_scorch.level_array_support import stored

EXTENTS = {"i": 3, "j": 4, "k": 2}

# (operand indices, operand formats, result indices, result format, missing variable)
MISSING = [
    (("ij", "j"), ("ds", "d"), "ij", "dd", "i"),
    (("ij", "j"), ("ds", "d"), "ij", "ds", "i"),
    (("ij", "j"), ("dd", "d"), "ij", "dd", "i"),
    (("ij", "j"), ("ss", "s"), "ij", "ss", "i"),
    (("ij", "j"), ("ds", "s"), "ij", "ds", "i"),
    (("j", "ij"), ("d", "ds"), "ij", "dd", "i"),
    (("ij", "i"), ("ds", "d"), "ij", "dd", "j"),
    (("ij", "i"), ("ss", "s"), "ij", "ss", "j"),
    (("ijk", "jk"), ("dss", "ds"), "ijk", "ddd", "i"),
    (("ijk", "ij"), ("dss", "dd"), "ijk", "ddd", "k"),
    (("ij", "ij", "j"), ("ss", "ds", "d"), "ij", "ss", "i"),
]
# Every addend has every variable: lowered as before.
COMPLETE = [
    (("ij", "ij"), ("ds", "dd"), "ij", "dd"),
    (("ij", "ij"), ("ss", "ds"), "ij", "ss"),
    (("ij", "ij", "ij"), ("ss", "ds", "dd"), "ij", "ss"),
]
ROUTES = {
    "as given": CompileOptions.from_environment(environ={}),
    "scheduled": replace(
        CompileOptions.from_environment(environ={}), requested_schedule=Schedule()
    ),
}


def access(name, fmt, indices, ivars):
    key = tuple(ivars[index] for index in indices)
    return TensorVar(name, fmt=fmt, dtype=torch.float64)[
        key[0] if len(key) == 1 else key
    ]


def program(terms, formats, result_indices, result_format, operation=Operation.ADD):
    ivars = {index: IndexVar(index) for index in result_indices}
    rhs = access("A", formats[0], terms[0], ivars)
    for name, fmt, indices in zip("BD", formats[1:], terms[1:]):
        rhs = BinaryOp(operation, rhs, access(name, fmt, indices, ivars))
    stmt = TensorAssign(access("C", result_format, result_indices, ivars), rhs)
    for index in reversed(result_indices):
        stmt = ForAll(ivars[index], stmt)
    return stmt


def emitted(terms, formats, result_indices, result_format, options, **kwargs):
    return legacy_generated_cpp(
        program(terms, formats, result_indices, result_format, **kwargs),
        tuple(EXTENTS[index] for index in result_indices),
        tuple(
            (tuple(EXTENTS[index] for index in term), torch.float64) for term in terms
        ),
        compile_options=options,
    )


@pytest.mark.parametrize("route", sorted(ROUTES))
@pytest.mark.parametrize(
    ("terms", "formats", "result_indices", "result_format", "missing"), MISSING
)
def test_a_sum_with_an_addend_missing_a_loop_variable_refuses(
    terms, formats, result_indices, result_format, missing, route
):
    with pytest.raises(UnsupportedFeature) as refusal:
        emitted(terms, formats, result_indices, result_format, ROUTES[route])

    (diagnostic,) = refusal.value.diagnostics
    assert diagnostic.code == "unsupported_sum_addend_without_loop_variable"
    assert f"no tensor indexed by {missing!r}" in diagnostic.message


def test_the_developer_entry_refuses_before_it_builds_anything():
    a = 0.5 + torch.rand((3, 4), dtype=torch.float64)
    x = 0.5 + torch.rand((4,), dtype=torch.float64)

    with pytest.raises(UnsupportedFeature) as refusal:
        lower_and_exec_cin(
            program(("ij", "j"), ("ds", "d"), "ij", "dd"),
            (3, 4),
            stored(a, "ds", "A"),
            stored(x, "d", "B"),
        )

    assert [diagnostic.code for diagnostic in refusal.value.diagnostics] == [
        "unsupported_sum_addend_without_loop_variable"
    ]


@pytest.mark.parametrize("route", sorted(ROUTES))
@pytest.mark.parametrize(
    ("terms", "formats", "result_indices", "result_format"), COMPLETE
)
def test_a_sum_whose_addends_have_every_variable_still_lowers(
    terms, formats, result_indices, result_format, route
):
    source = emitted(terms, formats, result_indices, result_format, ROUTES[route])
    assert "evaluate(" in source


@pytest.mark.parametrize("route", sorted(ROUTES))
def test_a_product_with_a_factor_missing_a_loop_variable_still_lowers(route):
    source = emitted(
        ("ij", "j"), ("ds", "d"), "ij", "ds", ROUTES[route], operation=Operation.MUL
    )
    assert "A_val[pA1] * " in source
