"""Operands and results that disagree on an index's extent are refused (patch 0076).

A kernel bounds an index's loop by one argument's extent and reads or writes the others by
it.  Each argument used to be checked against its own shape only, so ``scorch.matmul(A, v)``
with ``len(v) != A.shape[1]`` read ``v`` past its end (``matmul`` and ``spmv`` now refuse that
at their front door, patch 0074), and through ``scorch.ops.lower_and_exec_cin`` a result one
row short was written past its end.  Now the kernel ABI records which argument dimensions
each index variable is bound to (``index_extent_bindings``); the ABI refuses known extents
that disagree before a kernel is built, and a kernel whose extents are not all compiled in, as
a kept kernel is, compares them at its entry (``scorch_native::validate_jit_index_extents``,
out of line).  A post-op's extra tensor must hold exactly the result's cells.  All raise
``IndexExtentMismatchError`` with the code ``einsum_index_extent_mismatch``.

The end-to-end cases run in a child interpreter: on a tree without the check they read or
write past an argument's end, which may crash, and must fail the test, not end the process.
Values are compared in float64 to 1e-12.
"""

import subprocess
import sys
import textwrap
from typing import List

import pytest
import torch

from scorch.compiler.torch_cpp_abi import (  # type: ignore[import-untyped]
    KernelTensorABI,
    TorchCppKernelABI,
    index_extent_bindings,
)
from scorch.exceptions import IndexExtentMismatchError, defect_code
from scorch.format import LevelType

CODE = "einsum_index_extent_mismatch"
REFUSED = f"IndexExtentMismatchError {CODE}"
D, S = LevelType.DENSE, LevelType.COMPRESSED

_PRELUDE = """
import torch
import scorch
from scorch import STensor
from scorch.exceptions import defect_code

torch.manual_seed(0)


def outcome(call):
    try:
        call()
    except Exception as error:
        return f"{type(error).__name__} {defect_code(error)}"
    return "RETURNED"


def sparse(rows, cols):
    return torch.rand(rows, cols, dtype=torch.float64) * (torch.rand(rows, cols) < 0.3)
"""


def _child(body: str) -> List[str]:
    """``body``'s printed lines, run in a fresh interpreter after ``_PRELUDE``."""

    proc = subprocess.run(
        [sys.executable, "-c", _PRELUDE + textwrap.dedent(body)],
        capture_output=True,
        text=True,
        timeout=3600,
    )
    assert (
        proc.returncode == 0
    ), f"child exited {proc.returncode}: {proc.stderr[-3000:]}"
    return proc.stdout.splitlines()


def _matvec_abi(x_shape, result_shape=(5,), a_shape=(5, 4)):
    return TorchCppKernelABI(
        result_shape=result_shape,
        result_level_types=(D,),
        input_tensors=(
            KernelTensorABI("A", (D, S), (0, 1), a_shape, torch.float64),
            KernelTensorABI("x", (D,), (0,), x_shape, torch.float64),
        ),
        index_names=("i", "j"),
        index_bindings=((0, 0, 0), (0, 1, 0), (1, 1, 1), (1, 2, 0)),
    )


# -- the bindings ------------------------------------------------------------------------


def test_bindings_of_a_matrix_vector_product() -> None:
    names, bindings = index_extent_bindings(
        [
            ((), [[("i", "i")]]),
            ((), [[("i", "i"), ("j", "j")]]),
            ((), [[("j", "j")]]),
        ]
    )
    assert names == ("i", "j")
    assert bindings == ((0, 0, 0), (0, 1, 0), (1, 1, 1), (1, 2, 0))


def test_bindings_name_the_physical_level() -> None:
    # A stored column-major: logical axis 0 (i) is at level 1.
    names, bindings = index_extent_bindings(
        [((), [[("i", "i")]]), ((1, 0), [[("i", "i"), ("j", "j")]])]
    )
    assert names == ("i",)
    assert bindings == ((0, 0, 0), (0, 1, 1))


def test_an_index_twice_in_one_access_is_compared_with_itself() -> None:
    # y(i) = A(i, i): both of A's levels and the result's carry i.
    names, bindings = index_extent_bindings(
        [((), [[("i", "i")]]), ((), [[("i", "i"), ("i", "i")]])]
    )
    assert names == ("i",)
    assert bindings == ((0, 0, 0), (0, 1, 0), (0, 1, 1))


def test_an_index_bound_once_is_left_out() -> None:
    # s = sum_i x(i): nothing to compare.
    assert index_extent_bindings([((), []), ((), [[("i", "i")]])]) == ((), ())


# -- the ABI -----------------------------------------------------------------------------


def test_known_extents_that_disagree_are_refused_before_a_kernel_is_built() -> None:
    with pytest.raises(IndexExtentMismatchError) as caught:
        _matvec_abi(x_shape=(3,))
    assert defect_code(caught.value) == CODE
    assert "index 'j' has extent 4 in A (level 1) and 3 in x (level 0)" in str(
        caught.value
    )
    with pytest.raises(IndexExtentMismatchError, match="index 'i' has extent 6"):
        _matvec_abi(x_shape=(4,), result_shape=(6,))


def test_the_error_is_both_a_spec_error_and_a_runtime_error() -> None:
    from scorch.exceptions import CompileSpecError

    assert issubclass(IndexExtentMismatchError, CompileSpecError)
    assert issubclass(IndexExtentMismatchError, RuntimeError)


def _calls(abi: TorchCppKernelABI) -> List[str]:
    return [statement.name for statement in abi.emit_validation()]


def test_known_extents_that_agree_leave_the_entry_unchanged() -> None:
    assert "scorch_native::validate_jit_index_extents" not in _calls(
        _matvec_abi(x_shape=(4,))
    )


def test_an_unknown_extent_is_compared_at_the_entry_last() -> None:
    abi = _matvec_abi(x_shape=())
    statements = abi.emit_validation()
    assert statements[-1].name == "scorch_native::validate_jit_index_extents"
    assert _calls(abi).count("scorch_native::validate_jit_index_extents") == 1
    # i is known on both sides and was compared at construction; only j is left, as
    # index 0: (0, A, level 1) and (0, x, level 0).  The shape arguments come last.
    _, arguments, indices, triples, *shapes = statements[-1].args
    assert [literal.value for literal in arguments.values] == ["result", "A", "x"]
    assert [literal.value for literal in indices.values] == ["j"]
    assert [literal.value for literal in triples.values] == [0, 1, 1, 0, 2, 0]
    assert [shape.name for shape in shapes] == ["result_shape", "A_shape", "x_shape"]


@pytest.mark.parametrize(
    "bindings",
    [
        ((0, 0, 0),),  # bound once
        ((0, 0, 0), (0, 3, 0)),  # no argument 3
        ((0, 0, 0), (0, 1, 2)),  # A has 2 levels
        ((1, 0, 0), (1, 1, 0), (0, 1, 1), (0, 2, 0)),  # indices out of order
    ],
)
def test_malformed_bindings_are_refused(bindings) -> None:
    from scorch.exceptions import ScorchValueError

    with pytest.raises(ScorchValueError) as caught:
        TorchCppKernelABI(
            result_shape=(),
            result_level_types=(D,),
            input_tensors=(
                KernelTensorABI("A", (D, S), (0, 1), (), torch.float64),
                KernelTensorABI("x", (D,), (0,), (), torch.float64),
            ),
            index_names=("i", "j"),
            index_bindings=bindings,
        )
    assert defect_code(caught.value) == "kernel_abi_index_binding_invalid"


def test_an_extra_tensor_on_a_sparse_result_is_refused() -> None:
    from scorch.exceptions import ScorchValueError

    with pytest.raises(ScorchValueError) as caught:
        TorchCppKernelABI(
            result_shape=(),
            result_level_types=(D, S),
            input_tensors=(KernelTensorABI("A", (D, S), (0, 1), (), torch.float64),),
            extra_tensor_names=("bias",),
            extra_tensor_dtype=torch.float64,
        )
    assert defect_code(caught.value) == "post_op_extra_tensor_sparse_result"


# -- end to end, each family in a child interpreter ---------------------------------------


@pytest.mark.parametrize("delta", [-1, 1])
def test_a_vector_of_the_wrong_length_is_refused_on_every_format(delta) -> None:
    # The public routes: the front door (0074) refuses each before a kernel; without it
    # every route but "ds" with the cache (a prebuilt kernel) reached a generated one.
    lines = _child(f"""
        a = sparse(50, 40)
        v = torch.rand(40 + ({delta}), dtype=torch.float64)
        for name, operand, kwargs in [
            ("ds nocache", STensor.from_torch(a, "A").to_sparse("ds"), {{"use_cache": False}}),
            ("ss", STensor.from_torch(a, "A").to_sparse("ss"), {{}}),
            ("sd", STensor.from_torch(a, "A").to_sparse("sd"), {{}}),
            ("oo", STensor.from_torch(a, "A").to_sparse("oo"), {{}}),
            ("torch coo", a.to_sparse(), {{}}),
        ]:
            print(name, "|", outcome(lambda: scorch.matmul(operand, v, **kwargs)))
        operand = STensor.from_torch(a, "A").to_sparse("ss")
        x = STensor.from_torch(v, "x")
        print("spmv |", outcome(lambda: scorch.ops.spmv(operand, x, "d")))
        """)
    assert len(lines) == 6
    for line in lines:
        assert line.split(" | ")[1] == REFUSED, line


def test_a_kept_kernel_refuses_a_mismatch_at_its_entry() -> None:
    # A kernel compiled for no particular shape (its tensor variables carry none, so no
    # extent is compiled in) is built and kept the way the production lowering emits it,
    # then called past every front door: once with agreeing arguments, once with the
    # vector one entry short.  Only the kernel's entry check stands in the way.  (The
    # public entries compile the operands' extents into the kernel, whose per-argument
    # check then refuses a wrong length before this one.)
    lines = _child("""
        import re

        from scorch import ops
        from scorch.compiler.cin import (
            BinaryOp, ForAll, IndexVar, Operation, TensorAssign, TensorVar)
        from scorch.compiler.cin_lowerer import CINLowerer
        from scorch.compiler.codegen import LLIRLowerer
        from scorch.compiler.compilation_context import CompilationContext
        from scorch.compiler.compile_options import CompileOptions
        from scorch.compiler.scheduler import Scheduler

        f64 = torch.float64
        i, j = IndexVar("i"), IndexVar("j")
        options = CompileOptions.from_environment()


        def kept(fmt):
            a, x, y = (TensorVar(n, fmt=f, dtype=f64) for n, f in zip("Axy", (fmt, "d", "d")))
            body = TensorAssign(y[i], BinaryOp(Operation.MUL, a[i, j], x[j]), op=Operation.ADD)
            scheduled = Scheduler.auto_schedule(ForAll(i, ForAll(j, body)), compile_options=options)
            lowered = CINLowerer(compile_options=options).lower_IndexStmt(scheduled)
            cpp = LLIRLowerer(compile_options=options).lower_llir(lowered)
            assert "validate_jit_index_extents(" in cpp
            prepared = ops._prepare_generated_kernel_build(
                options.build.preamble_source, cpp, options, CompilationContext(options))
            return ops._load_validated_prepared_kernel(prepared).evaluate


        def arguments(*tensors):
            args = [[50]]
            for snapshot in ops._snapshot_runtime_tensors(tensors, expected_count=len(tensors)):
                args += [snapshot.shape, snapshot.native_mode_indices, snapshot.values]
            return args


        a = sparse(50, 40)
        for fmt in ("ss", "sd", "oo"):
            kernel = kept(fmt)
            held = STensor.from_torch(a, "A").to_sparse(fmt)
            good = torch.rand(40, dtype=torch.float64)
            got = ops._evaluate_native(
                kernel, *arguments(held, STensor.from_torch(good, "x").to_dense()))
            correct = (got.storage.value - a @ good).abs().max().item() <= 1e-12
            short = STensor.from_torch(good[:39].clone(), "x").to_dense()
            try:
                ops._evaluate_native(kernel, *arguments(held, short))
                print(fmt, "|", correct, "|", "RETURNED")
            except Exception as error:
                text = str(error).splitlines()[0]
                named = re.search(r"has extent 40 in .* and 39 in ", text) is not None
                kind = f"{type(error).__name__} {defect_code(error)}"
                print(fmt, "|", correct, "|", kind, named)
        """)
    assert len(lines) == 3
    for line in lines:
        _, correct, refused = line.split(" | ")
        assert correct == "True" and refused == REFUSED + " True", line


def test_direct_entries_refuse_every_disagreeing_extent() -> None:
    lines = _child("""
        from scorch.compiler.cin import (
            BinaryOp, ForAll, IndexVar, Operation, TensorAssign, TensorVar)
        from scorch.ops import lower_and_exec_cin

        f64 = torch.float64
        i, j = IndexVar("i"), IndexVar("j")


        def matvec():
            a, x, y = (TensorVar(n, fmt=f, dtype=f64) for n, f in zip("Axy", ("ds", "d", "d")))
            body = TensorAssign(y[i], BinaryOp(Operation.MUL, a[i, j], x[j]), op=Operation.ADD)
            return ForAll(i, ForAll(j, body))


        def diagonal():
            a, y = TensorVar("A", fmt="dd", dtype=f64), TensorVar("y", fmt="d", dtype=f64)
            return ForAll(i, TensorAssign(y[i], a[i, i]))


        def transpose():
            a, c = TensorVar("A", fmt="ds", dtype=f64), TensorVar("C", fmt="dd", dtype=f64)
            return ForAll(j, ForAll(i, TensorAssign(c[j, i], a[i, j])))


        def held(t, fmt):
            s = STensor.from_torch(t, "T")
            return s.to_dense() if set(fmt) == {"d"} else s.to_sparse(fmt)


        a = sparse(5, 4)
        cases = [
            ("matvec x short", matvec, (5,), [held(a, "ds"), held(torch.rand(3, dtype=f64), "d")]),
            ("matvec x long", matvec, (5,), [held(a, "ds"), held(torch.rand(5, dtype=f64), "d")]),
            ("matvec result short", matvec, (4,), [held(a, "ds"), held(torch.rand(4, dtype=f64), "d")]),
            ("matvec result long", matvec, (6,), [held(a, "ds"), held(torch.rand(4, dtype=f64), "d")]),
            ("diagonal of 4 x 5", diagonal, (4,), [held(sparse(4, 5), "dd")]),
            ("diagonal of 4 x 4", diagonal, (4,), [held(sparse(4, 4), "dd")]),
            ("transpose result rows", transpose, (5, 5), [held(a, "ds")]),
            ("transpose result cols", transpose, (4, 4), [held(a, "ds")]),
        ]
        for name, build, shape, args in cases:
            print(name, "|", outcome(lambda: lower_and_exec_cin(build(), shape, *args)))
        """)
    # The verifier refuses an index named twice in one access, with or without agreeing
    # extents, before any extent is compared: the diagonal never reaches the check.
    repeated = "VerificationError unsupported_repeated_access_index"
    assert lines == [
        "matvec x short | " + REFUSED,
        "matvec x long | " + REFUSED,
        "matvec result short | " + REFUSED,
        "matvec result long | " + REFUSED,
        "diagonal of 4 x 5 | " + repeated,
        "diagonal of 4 x 4 | " + repeated,
        "transpose result rows | " + REFUSED,
        "transpose result cols | " + REFUSED,
    ]


def test_a_post_op_tensor_must_hold_the_result() -> None:
    lines = _child("""
        from scorch.compiler.cin import PostOp, PostOps

        a = sparse(6, 5)
        b = torch.rand(5, 3, dtype=torch.float64)
        for count in (17, 18, 19):
            extra = torch.rand(count, dtype=torch.float64)
            post = PostOps(ops=[PostOp("add", "postop_0")], extra_tensors=["postop_0"])

            def call():
                return scorch.einsum(
                    "ij,jk->ik",
                    STensor.from_torch(a, "A").to_sparse("ds"),
                    STensor.from_torch(b, "B"),
                    format="dd",
                    _post_ops=post,
                    _post_ops_tensors=[extra],
                )

            print(count, "|", outcome(call))
        """)
    # 18 values, the result's cells, pass the entry check.  The value the call returns is
    # the post-op's to get right, not this check's.
    assert lines == ["17 | " + REFUSED, "18 | RETURNED", "19 | " + REFUSED]


def test_agreeing_shapes_compute_what_torch_does() -> None:
    lines = _child("""
        a = sparse(50, 40)
        v = torch.rand(40, dtype=torch.float64)
        for fmt in ("ds", "ss", "sd", "oo"):
            for kwargs in ({}, {"use_cache": False}):
                got = scorch.matmul(STensor.from_torch(a, "A").to_sparse(fmt), v, **kwargs)
                print(fmt, kwargs, "|", (got - a @ v).abs().max().item() <= 1e-12)
        """)
    assert len(lines) == 8
    for line in lines:
        assert line.endswith("| True"), line


def test_the_known_extent_check_does_not_run_on_a_kept_kernel() -> None:
    # In a child: a kernel another test kept would make the first call a hit too.
    lines = _child("""
        from scorch.compiler.torch_cpp_abi import TorchCppKernelABI

        calls = []
        original = TorchCppKernelABI._refuse_disagreeing_extents


        def counted(self):
            calls.append(self.function_name)
            return original(self)


        TorchCppKernelABI._refuse_disagreeing_extents = counted
        operand = STensor.from_torch(sparse(30, 20), "A").to_sparse("ss")
        v = torch.rand(20, dtype=torch.float64)
        scorch.matmul(operand, v)
        built = len(calls)
        scorch.matmul(operand, v)
        print(built, len(calls))
        """)
    built, after = (int(part) for part in lines[-1].split())
    assert built > 0
    assert after == built
