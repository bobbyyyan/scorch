"""``C = A + B + D`` with two compressed operands and a dense one: the dense counter.

Along an index where two operands are compressed and a third is dense, the sum visits
every coordinate of the dense extent and merges the two compressed operands as it goes.
That lowers to a ``while`` loop that advances the dense variable together with the two
positions::

    int64_t j = 0;
    while (j < D1_size) {
      int j_A = pA1 < pA1_end ? A1_crd[pA1] : 0;
      int j_B = pB1 < pB1_end ? B1_crd[pB1] : 0;
      ...
      pA1 += (int)(j_A == j);
      pB1 += (int)(j_B == j);
      j++;
    }

The first line was missing.  With one compressed operand the loop is a ``for`` whose
header declares the variable, and with none it is a plain counted loop, so only the
merge under a dense extent named a variable nothing declared, and the kernel did not
build (``use of undeclared identifier 'j'``).  No public operation adds three operands;
the programs are hand-written CIN through ``lower_and_exec_cin``.

Operands are built and results read as level arrays in PyTorch; float64 to 1e-12 of
each cell.
"""

import itertools
import re

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
from scorch.compiler.loopir.pipeline import legacy_generated_cpp
from scorch.ops import lower_and_exec_cin
from tests.test_scorch.level_array_support import assert_matches, stored

SMALL = {1: (7,), 2: (5, 4), 3: (4, 3, 5)}
LARGE = {1: (500,), 2: (200, 9), 3: (60, 5, 4)}
_WHILE_OVER_INDEX = re.compile(r"^while \(([ijk]) < ")


def sum_cin(formats, result_format, dtype=torch.float64):
    ivars = tuple(IndexVar(name) for name in "ijk"[: len(result_format)])
    key = ivars[0] if len(ivars) == 1 else ivars
    total = None
    for position, fmt in enumerate(formats):
        access = TensorVar("ABD"[position], fmt=fmt, dtype=dtype)[key]
        total = access if total is None else BinaryOp(Operation.ADD, total, access)
    stmt = TensorAssign(TensorVar("C", fmt=result_format, dtype=dtype)[key], total)
    for index_var in reversed(ivars):
        stmt = ForAll(index_var, stmt)
    return stmt


def emitted(formats, result_format):
    """The kernel's statements, one per entry; blank lines and comments dropped."""

    shape = SMALL[len(result_format)]
    source = legacy_generated_cpp(
        sum_cin(formats, result_format),
        shape,
        tuple((shape, torch.float64) for _ in formats),
        compile_options=CompileOptions.from_environment(environ={}),
    )
    lines = [line.strip() for line in source.splitlines()]
    return [line for line in lines if line and not line.startswith("//")]


def while_loops_over_an_undeclared_index(source):
    """Index variables a ``while`` loop counts and its block does not declare before it.

    The declaration need not be the line just before the loop: a statement such as the
    dense-block check of a dense result level may sit between them.
    """

    return [
        match.group(1)
        for at, line in enumerate(source)
        for match in [_WHILE_OVER_INDEX.match(line)]
        if match and not _declared_before(source, at, match.group(1))
    ]


def _declared_before(source, at, name):
    """Whether ``int64_t name = 0;`` comes before line ``at`` in the block holding it.

    Walks back over the lines and their braces from the right, so a block that closes
    before the loop is skipped whole and the walk stops where the loop's block opens.
    """

    depth = 0
    for line in reversed(source[:at]):
        if depth == 0 and line == f"int64_t {name} = 0;":
            return True
        for char in reversed(line):
            if char == "}":
                depth += 1
            elif char == "{":
                depth -= 1
                if depth < 0:
                    return False
    return False


def layouts(rank):
    return ["".join(kinds) for kinds in itertools.product("ds", repeat=rank)]


def test_the_check_finds_a_while_loop_over_an_undeclared_index():
    source = ["int pB1_end = B1_pos[pB0 + 1];", "while (j < D1_size) {", "j++;", "}"]
    assert while_loops_over_an_undeclared_index(source) == ["j"]
    assert while_loops_over_an_undeclared_index(["int64_t j = 0;"] + source[1:]) == []


def test_the_check_accepts_a_statement_between_the_declaration_and_the_loop():
    source = [
        "int64_t j = 0;",
        "scorch_check_dense_block(pC0, C1_size, false, 1, 1);",
        "while (j < A1_size) {",
        "j++;",
        "}",
    ]
    assert while_loops_over_an_undeclared_index(source) == []


def test_the_check_finds_a_declaration_in_a_block_that_closed_before_the_loop():
    loop = ["while (j < D1_size) {", "j++;", "}"]
    sibling = ["if (pB1 < pB1_end) {", "int64_t j = 0;", "}"]
    assert while_loops_over_an_undeclared_index(sibling + loop) == ["j"]
    outer = "for (int64_t i = 0; i < A0_size; i++) {"
    earlier_loop = [outer, "int64_t j = 0;", "}"]
    assert while_loops_over_an_undeclared_index(earlier_loop + [outer] + loop) == ["j"]
    branch = ["if (a) {", "int64_t j = 0;", "} else {"]
    assert while_loops_over_an_undeclared_index(branch + loop + ["}"]) == ["j"]


def test_the_merge_under_a_dense_extent_declares_its_counter():
    source = emitted(("ss", "ds", "dd"), "ss")

    loop = source.index("while (j < D1_size) {")
    assert source[loop - 1] == "int64_t j = 0;"
    assert "j++;" in source[loop:]
    assert source.count("int64_t j = 0;") == 1


@pytest.mark.parametrize(
    ("formats", "result_format"),
    [
        (formats, result_format)
        for rank in (1, 2)
        for formats in itertools.product(layouts(rank), repeat=3)
        for result_format in layouts(rank)
    ],
)
def test_no_while_loop_counts_an_index_nothing_declares(formats, result_format):
    assert while_loops_over_an_undeclared_index(emitted(formats, result_format)) == []


PROGRAMS = [
    (("ss", "ds", "dd"), "ss"),
    (("ss", "ds", "dd"), "dd"),
    (("ss", "ds", "dd"), "ds"),
    (("ss", "ss", "dd"), "ss"),
    (("ss", "ss", "dd"), "dd"),
    (("ds", "ds", "dd"), "ds"),
    (("sd", "ds", "dd"), "dd"),
    (("dd", "ss", "ss"), "dd"),
    (("s", "s", "d"), "d"),
    (("s", "s", "d"), "s"),
    (("sss", "dss", "ddd"), "sss"),
    (("sss", "sss", "ddd"), "ddd"),
    # No dense operand: a merge of three compressed operands, built before.
    (("ss", "ss", "ss"), "ss"),
]


@pytest.mark.parametrize("extents", [SMALL, LARGE], ids=["small", "large"])
@pytest.mark.parametrize(("formats", "result_format"), PROGRAMS)
def test_sum_of_three_operands_matches(formats, result_format, extents):
    shape = extents[len(result_format)]
    generator = torch.Generator().manual_seed(61)
    dense = []
    for fmt in formats:
        values = 0.5 + torch.rand(shape, generator=generator, dtype=torch.float64)
        if set(fmt) != {"d"}:
            keep = torch.rand(shape, generator=generator, dtype=torch.float64) < 0.5
            rows = torch.rand(shape[0], generator=generator, dtype=torch.float64) < 0.5
            values = values * (keep & rows.reshape((-1,) + (1,) * (len(shape) - 1)))
        dense.append(values)

    result = lower_and_exec_cin(
        sum_cin(formats, result_format),
        shape,
        *[
            stored(values, fmt, "ABD"[position])
            for position, (values, fmt) in enumerate(zip(dense, formats))
        ],
    )

    assert_matches(result, dense[0] + dense[1] + dense[2], torch.float64)
