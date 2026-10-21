"""A lowered kernel that reads a variable outside its declaring block is refused.

``reads_out_of_scope`` walks the function the target lowering returns, block by
block.  The first tests build small functions by hand, one per rule of the
block structure; the last two show the target lowering refusing, by code, the
kernel it emitted until 2026-10-12 for a four-dimensional sum (the lowering's
fix is switched off for that test), and compiling it with the fix in place.
"""

from __future__ import annotations

from typing import List

import pytest
import torch

from scorch.compiler import llir
from scorch.compiler.loopir import lower_llir
from scorch.compiler.loopir.llir_scope import reads_out_of_scope
from scorch.compiler.loopir.lower_llir import LoopIRTargetError
from scorch.compiler.loopir.pipeline import compile_cin_via_loopir
from tests.test_scorch.test_loopir_dense_union import SPARSE_EXTENTS, _statement
from tests.test_scorch.test_loopir_sparse_workspace_target import auto_options

INT = llir.DataType.INT


def var(name: str) -> llir.Var:
    return llir.Var(name=name, type=INT)


def declare(name: str) -> llir.VarInit:
    return llir.VarInit(var=var(name), value=llir.Literal(0, INT))


def read(name: str) -> llir.Assign:
    """``sink = name;``: one read of ``name``."""

    return llir.Assign(var=var("sink"), value=var(name))


def function(body: List[llir.Stmt]) -> llir.Function:
    return llir.Function(
        llir.DataType.VOID, "evaluate", [var("n")], [declare("sink"), *body]
    )


def names(body: List[llir.Stmt]) -> List[str]:
    return [name for name, _ in reads_out_of_scope(function(body))]


def less(left: str, right: str) -> llir.BinOp:
    return llir.BinOp(op="<", left=var(left), right=var(right))


def test_a_read_in_the_declaring_block_and_in_a_nested_block_is_in_scope():
    loop = llir.WhileLoop(less("sink", "n"), [read("p"), read("n")])
    assert names([declare("p"), read("p"), loop]) == []


def test_a_name_declared_in_one_branch_is_not_visible_in_the_next():
    """The defect this check exists for: ``pB2`` declared where both operands
    have an entry, and read in the case that iterates one of them."""

    chain = llir.IfThenElse(
        cond_list=[less("sink", "n"), less("n", "sink")],
        then_body_list=[[declare("pB2"), read("pB2")], [read("pB2")]],
    )
    found = reads_out_of_scope(function([chain]))
    assert [name for name, _ in found] == ["pB2"]
    assert "then_body_list" in found[0][1] and "[1]" in found[0][1]


def test_a_single_branch_and_its_else_are_separate_blocks():
    branch = llir.IfThenElse(
        cond=less("sink", "n"), then_body=[declare("p")], else_body=[read("p")]
    )
    assert names([branch, read("p")]) == ["p", "p"]


def test_a_for_init_is_visible_in_the_header_and_body_and_not_after():
    loop = llir.ForLoop(
        init=declare("i"),
        cond=less("i", "n"),
        update=llir.Increment(var("i")),
        body=[read("i")],
    )
    assert names([loop]) == []
    assert names([loop, read("i")]) == ["i"]


def test_a_while_body_ends_its_declarations():
    loop = llir.WhileLoop(less("sink", "n"), [declare("p"), read("p")])
    assert names([loop, read("p")]) == ["p"]


def test_a_nested_statement_list_belongs_to_the_block_that_holds_it():
    assert names([[declare("p")], read("p")]) == []  # type: ignore[list-item]


def test_a_read_before_the_declaration_in_one_block_is_reported():
    assert names([read("p"), declare("p")]) == ["p"]


def test_a_name_no_node_declares_is_left_to_the_cpp_compiler():
    assert names([read("A2_pos")]) == []


def test_a_lambda_is_visible_after_it_and_its_parameters_only_inside():
    worker = llir.LambdaDef(var("worker"), [var("lo")], [read("lo"), read("n")])
    assert names([worker, read("worker")]) == []
    assert names([worker, read("lo")]) == ["lo"]


def test_statements_hoisted_before_a_loop_belong_to_the_enclosing_block():
    loop = llir.ForLoop(
        init=declare("i"),
        cond=less("i", "n"),
        update=llir.Increment(var("i")),
        body=[read("total")],
        before_parallel_body=[declare("total")],
    )
    assert names([loop, read("total")]) == []


CELL = ("ijkl,ijkl->ijkl", ("sdsd", "sdsd"), "ssss")


def _compile(arm: bool):
    expr, fmts, result_fmt = CELL
    extents = SPARSE_EXTENTS[0]
    terms, output = expr.split("->")[0].split(","), expr.split("->")[1]
    return compile_cin_via_loopir(
        _statement(expr, fmts, result_fmt, torch.float64),
        tuple(extents[c] for c in output),
        tuple((tuple(extents[c] for c in term), torch.float64) for term in terms),
        compile_options=auto_options(arm, jit=True),
    )


@pytest.mark.parametrize("arm", (False, True), ids=("plain", "blocked"))
def test_the_lowering_refuses_the_kernel_it_emitted_before_the_fix(arm, monkeypatch):
    """With the fix off, the one-operand cases resolve the absent operand's
    dense position again, from a position variable those cases do not declare."""

    monkeypatch.setattr(
        lower_llir._TargetLowering,
        "_operand_absent_here",
        lambda self, symbol, level: False,
    )
    with pytest.raises(LoopIRTargetError) as refused:
        _compile(arm)
    assert refused.value.defect.code == "variable_read_out_of_scope"
    assert "pA2" in refused.value.defect.message
    assert "pB2" in refused.value.defect.message


@pytest.mark.parametrize("arm", (False, True), ids=("plain", "blocked"))
def test_the_fixed_kernel_reads_every_variable_in_scope(arm):
    assert reads_out_of_scope(_compile(arm).llir_function) == ()
