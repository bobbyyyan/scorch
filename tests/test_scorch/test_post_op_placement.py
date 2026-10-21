"""Every post-op einsum is asked for is applied to every result cell, once.

``einsum(..., _post_ops=..., _post_ops_tensors=...)`` fuses a chain of post-ops (a
bias add, a scale, an activation) into the contraction's kernel.  The lowering applied
them in one place only: the element loop that copies a dense workspace into the
result.  A program without that loop, such as ``A @ B`` with ``A`` stored ``ds`` (an
``i, j, k`` nest that adds into the result directly), dropped them.  The kernel took
the post-op tensor and never read it, and the call returned ``A @ B``.  The copy-out
that did apply them also took the memcpy path for a whole dense row, which applies
nothing.  And it reached only the rows its loop visited: with ``A`` stored ``ss`` the
rows ``A`` does not store were left zero, without the bias.  No einsum program had its
post-ops applied: where the copy-out sat inside a split reduction loop (it adds partial
sums there) the lowering refused them, and every other program returned its result
without them.

Now the copy-out applies the post-ops only where it gives every cell its final value
(``CINLowerer._copy_out_applies_post_ops``).  Everywhere else the result's assembler
applies them to the finished value tensor, after the loop nest and before the values
become the result's storage (``ResultTensorAssembler.emit_value_post_ops``).  A sparse
result takes only post-ops that map zero to zero, with no tensor; the others are
refused with a code.  einsum hands a post-op tensor to the kernel in the order the
kernel stores the result.

Values are compared with a float64 reference to a relative 1e-10 (float32: 1e-5).  The
end-to-end cases run in a child interpreter so that kernels other tests built are not
reused.
"""

import subprocess
import sys
import textwrap
from typing import List

import pytest
import torch

from scorch import ops
from scorch.compiler import llir
from scorch.compiler.cin import (
    ForAll,
    IndexVar,
    PostOp,
    PostOps,
    TensorAssign,
    TensorVar,
)
from scorch.compiler.cin_lowerer import CINLowerer, _result_tensor_abi_assembler
from scorch.compiler.codegen import LLIRLowerer
from scorch.compiler.torch_cpp_abi import ResultTensorAssembler
from scorch.exceptions import ScorchError, defect_code
from scorch.format import LevelType
from tests.test_scorch.level_array_support import emitted_einsum_cpp

D, S = LevelType.DENSE, LevelType.COMPRESSED
VALUE_PASS = "Apply the post-ops to every stored value"
SPARSE_CHANGES_ZERO = "post_op_sparse_result_changes_zero"
SPARSE_EXTRA = "post_op_extra_tensor_sparse_result"

_PRELUDE = """
import torch
import scorch
from scorch import STensor, ops
from scorch.compiler.cin import PostOp, PostOps
from scorch.exceptions import defect_code

torch.manual_seed(0)
GELU = 0.7978845608
VALUE_PASS = "Apply the post-ops to every stored value"
BUILT = []
_build = ops._prepare_generated_kernel_build


def _recording_build(header, source, *args, **kwargs):
    BUILT.append(source)
    return _build(header, source, *args, **kwargs)


ops._prepare_generated_kernel_build = _recording_build


def operand(shape, fmt, dtype):
    # Signed values, about 40 percent stored, with the second index of every mode empty:
    # a row and a column no stored entry reaches.  The reference takes the values as
    # rounded to ``dtype``.
    values = (torch.rand(shape, dtype=torch.float64) - 0.5) * (torch.rand(shape) < 0.4)
    for mode in range(len(shape)):
        if shape[mode] > 2:
            values.select(mode, 1).zero_()
    values = values.to(dtype)
    tensor = STensor.from_torch(values, "T")
    if fmt is not None and set(fmt) != {"d"}:
        tensor = tensor.to_sparse(fmt)
    return tensor, values.double()


def assembly_path(result):
    # How the einsum kernel assembles its result, read from its source: the one built
    # with the value pass.
    sources = [source for source in BUILT if VALUE_PASS in source]
    if len(sources) != 1:
        return f"{len(sources)} value passes"
    if set(result) == {"d"}:
        return "dense"
    if "_values_data" in sources[0]:
        return "two-phase"
    if "_known_nnz" in sources[0]:
        return "known nnz"
    return "vectors"


def reference(expr, dense, chain, extras):
    out = torch.einsum(expr, *dense)
    extras = iter(extras)
    for kind in chain:
        if kind == "add":
            out = out + next(extras)
        elif kind == "mul":
            out = out * next(extras)
        elif kind == "relu":
            out = out.clamp_min(0)
        elif kind == "sigmoid":
            out = 1 / (1 + torch.exp(-out))
        elif kind == "tanh":
            out = torch.tanh(out)
        elif kind == "gelu":
            out = out * 0.5 * (1 + torch.tanh(GELU * (out + 0.044715 * out**3)))
    return out


def run(name, expr, formats, result, chain, extents, dtype=torch.float64, path=False):
    terms, out_term = expr.split("->")[0].split(","), expr.split("->")[1]
    built = [
        operand(tuple(extents[c] for c in term), fmt, dtype)
        for term, fmt in zip(terms, formats)
    ]
    shape = tuple(extents[c] for c in out_term)
    binary = [kind for kind in chain if kind in ("add", "mul")]
    names = [f"postop_{n}" for n in range(len(binary))]
    extras = [(torch.rand(shape, dtype=torch.float64) - 0.5).to(dtype) for _ in names]
    names_iter = iter(names)
    post = PostOps(
        ops=[
            PostOp(kind, next(names_iter) if kind in ("add", "mul") else None)
            for kind in chain
        ],
        extra_tensors=names,
    )
    want = reference(
        expr, [b[1] for b in built], chain, [extra.double() for extra in extras]
    )
    BUILT.clear()
    try:
        got = scorch.einsum(
            expr,
            *[b[0] for b in built],
            format=result,
            _post_ops=post,
            _post_ops_tensors=extras,
        )
    except Exception as error:
        print(name, "|", type(error).__name__, defect_code(error))
        return
    got = got.to_torch().double() if isinstance(got, STensor) else got.double()
    tol = 1e-10 if dtype == torch.float64 else 1e-5
    error = ((got - want).abs() / want.abs().clamp_min(1)).max().item()
    verdict = "OK" if error <= tol else f"WRONG {error:.3e}"
    print(name, "|", verdict, *(["|", assembly_path(result)] if path else []))
"""


def _child(body: str, threads: int = 0) -> List[str]:
    """``body``'s printed lines, run in a fresh interpreter after ``_PRELUDE``.

    ``threads``, when given, is the worker count the body's shapes were built for, and the child
    sets torch's count to it, capped at its processors, before the body runs. The child inherits
    this process's environment, and a thread setting there (MKL_NUM_THREADS=1) would otherwise hold
    every team scorch launches to torch's count (scorch_thread_ceiling in scorch_policy.h).
    """

    setup = ""
    if threads:
        setup = (
            "import os\n"
            "procs = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') "
            "else (os.cpu_count() or 1)\n"
            "torch.set_num_threads(max(1, min(%d, procs)))\n" % threads
        )
    proc = subprocess.run(
        [sys.executable, "-c", _PRELUDE + setup + textwrap.dedent(body)],
        capture_output=True,
        text=True,
        timeout=3600,
    )
    assert (
        proc.returncode == 0
    ), f"child exited {proc.returncode}: {proc.stderr[-3000:]}"
    return proc.stdout.splitlines()


def _all_ok(lines: List[str], count: int) -> None:
    assert len(lines) == count, lines
    wrong = [line for line in lines if not line.endswith("| OK")]
    assert not wrong, wrong


# -- values: dense results -------------------------------------------------------------


def test_the_reported_program_adds_its_bias() -> None:
    # A stored ds: an i, j, k nest with no workspace.  It returned A @ B without the
    # add.
    lines = _child("""
        extents = dict(i=7, j=6, k=5)
        for chain in (["add"], ["add", "relu"], ["mul", "sigmoid"], ["relu", "add"]):
            name = "ds dd " + "+".join(chain)
            run(name, "ij,jk->ik", ["ds", "dd"], "dd", chain, extents)
        """)
    _all_ok(lines, 4)


def test_rows_no_loop_visits_still_take_the_post_ops() -> None:
    # Row 1 of A stores nothing.  A loop over the rows A stores (ss, sd's compressed
    # rows, COO) never reaches it, and its cells must still hold bias, not zero.
    lines = _child("""
        extents = dict(i=7, j=6, k=5)
        for a in ("dd", "ds", "sd", "ss", "oo"):
            for b in ("dd", "ds", "ss"):
                run(f"{a} {b}", "ij,jk->ik", [a, b], "dd", ["add", "relu"], extents)
        """)
    _all_ok(lines, 15)


@pytest.mark.parametrize("extents", [dict(i=7, j=6, k=5), dict(i=37, j=29, k=23)])
def test_every_kind_on_every_dense_result_shape(extents) -> None:
    lines = _child(f"""
        extents = {extents!r}
        cases = [
            ("ij,j->i", ["ds", None], "d"),
            ("ij,j->i", ["ss", None], "d"),
            ("ij,ij->ij", ["ds", "dd"], "dd"),
            ("ij,ij->ij", ["ss", "ss"], "dd"),
            ("ij->ij", ["sd"], "dd"),
            ("ij->i", ["ss"], "d"),
            ("ij,kj->ik", ["ds", "ds"], "dd"),
        ]
        for chain in (["add", "relu"], ["mul", "gelu"], ["tanh", "add"], ["sigmoid"]):
            for expr, formats, result in cases:
                name = expr + " " + ",".join(str(f) for f in formats)
                name += " " + "+".join(chain)
                run(name, expr, formats, result, chain, extents)
        """)
    _all_ok(lines, 28)


def test_a_result_stored_in_another_order_pairs_each_cell_with_its_entry() -> None:
    # einsum stores ij,jk->ki in the loop order (i, k) and transposes it afterwards; the
    # post-op tensor, given in the logical (k, i) shape, is handed over in that order.
    lines = _child("""
        extents = dict(i=7, j=6, k=5)
        for a in ("dd", "ds", "ss"):
            run(f"{a} ki", "ij,jk->ki", [a, "dd"], "dd", ["add", "mul"], extents)
        run("abc->cab", "abc->cab", ["dss"], "ddd", ["add"], dict(a=4, b=5, c=3))
        extents = dict(a=3, b=6, c=4, d=5)
        run("abc,bd->adc", "abc,bd->adc", ["dsd", "dd"], "ddd", ["add"], extents)
        """)
    _all_ok(lines, 5)


def test_a_copy_out_of_partial_sums_takes_the_post_ops_after_the_last() -> None:
    # A compressed A puts the workspace over f inside the loop over c, a reduction: the
    # copy-out adds one c's share per iteration.  Post-ops refused this program before;
    # now the finished values take them.
    lines = _child("""
        extents = dict(a=3, b=4, c=5, e=4, f=6)
        for a in ("dsss", "ssss", "dsds"):
            run(a, "abce,cef->abf", [a, "ddd"], "ddd", ["add", "relu"], extents)
        """)
    _all_ok(lines, 3)


def test_float32() -> None:
    lines = _child("""
        extents = dict(i=37, j=29, k=23)
        for a in ("ds", "ss"):
            for b in ("dd", "ds"):
                run(f"{a} {b}", "ij,jk->ik", [a, b], "dd", ["add", "relu"], extents,
                    dtype=torch.float32)
        """)
    _all_ok(lines, 4)


_PATH_PROGRAMS = [
    ("ij,jk->ik", ["ds", "dd"], "dd"),
    ("ij,jk->ik", ["ds", "ds"], "ds"),
    ("ij,jk->ik", ["ss", "ss"], "ss"),
    ("ij,jk->ik", ["ds", "ds"], "sd"),
    ("ij,ij->ij", ["ds", "dd"], "ds"),
    ("ij->ij", ["ss"], "ss"),
    ("ij,ij->ij", ["oo", "oo"], "oo"),
    # Sampled dense-dense product with a COO sample: one scalar sum over k per stored
    # entry of the sample, so the result stores exactly its entries (the known count).
    ("ij,ik,jk->ij", ["oo", "dd", "dd"], "oo"),
]


@pytest.mark.parametrize("dtype", ["float64", "float32"])
@pytest.mark.parametrize(
    "extents", [dict(i=7, j=6, k=5), dict(i=64, j=40, k=48)], ids=["1 thread", "n"]
)
def test_the_value_pass_on_every_assembly_path(dtype, extents) -> None:
    """Dense, compressed two-phase, known count and appended vectors, every kind.

    The path is read from the kernel's source.  A dense result takes every kind; a
    sparse one the kinds that keep zero.  The pass splits its values over
    ``scorch_nthreads(n, n)`` threads, one per 500 values up to the processor count:
    the small results (35 cells) take one, the large (3,072 cells) up to six.
    """

    lines = _child(f"""
        extents = {extents!r}
        dtype = torch.{dtype}
        programs = {_PATH_PROGRAMS!r}
        for expr, formats, result in programs:
            if set(result) == {{"d"}}:
                chains = [["add", "relu"], ["mul", "sigmoid"], ["gelu", "tanh"]]
            else:
                chains = [["relu"], ["gelu", "tanh"]]
            for chain in chains:
                name = expr + " " + ",".join(formats) + "->" + result
                name += " " + "+".join(chain)
                run(name, expr, formats, result, chain, extents, dtype, path=True)
        """, threads=6)  # the large results' 3,072 values: 3072 // 500 = 6 workers
    assert len(lines) == 1 * 3 + 7 * 2, lines
    wrong = [line for line in lines if line.split(" | ")[1] != "OK"]
    assert not wrong, wrong
    paths = {line.split(" | ")[2] for line in lines}
    assert {"dense", "two-phase", "known nnz", "vectors"} <= paths, paths


def test_the_prebuilt_sampled_product_is_not_taken_with_post_ops() -> None:
    # The float32 sampled product with a COO sample (ij,ik,jk->ij, oo dd dd) has a
    # prebuilt kernel that computes the bare product.  einsum took it for a call with
    # post-ops too and returned the product without them.  Such a call now keeps the
    # compiler path; a call without post-ops still takes the prebuilt kernel.
    lines = _child("""
        import scorch_ops
        real = scorch_ops.sddmm_coo_float_prebuilt
        calls = []

        def counted(*args):
            calls.append(1)
            return real(*args)

        scorch_ops.sddmm_coo_float_prebuilt = counted
        extents = dict(i=37, j=29, k=23)
        run("relu", "ij,ik,jk->ij", ["oo", "dd", "dd"], "oo", ["relu"], extents,
            dtype=torch.float32)
        print("prebuilt calls with post-ops", len(calls))
        s, _ = operand((37, 29), "oo", torch.float32)
        a, _ = operand((37, 23), "dd", torch.float32)
        b, _ = operand((29, 23), "dd", torch.float32)
        scorch.einsum("ij,ik,jk->ij", s, a, b, format="oo")
        print("prebuilt calls without post-ops", len(calls))
        """)
    assert lines == [
        "relu | OK",
        "prebuilt calls with post-ops 0",
        "prebuilt calls without post-ops 1",
    ], lines


# -- values and refusals: sparse results -----------------------------------------------


def test_a_sparse_result_takes_the_post_ops_that_keep_zero() -> None:
    lines = _child("""
        extents = dict(i=7, j=6, k=5)
        for result in ("ds", "ss"):
            for a, b in (("ds", "ds"), ("ss", "ss"), ("ds", "dd")):
                for chain in (["relu"], ["gelu", "tanh"]):
                    name = f"{a} {b} -> {result} " + "+".join(chain)
                    run(name, "ij,jk->ik", [a, b], result, chain, extents)
        run("ij,ij->ij ss", "ij,ij->ij", ["ss", "ss"], "ss", ["tanh"], extents)
        """)
    _all_ok(lines, 13)


def test_a_sparse_result_refuses_the_post_ops_that_change_zero() -> None:
    # sigmoid(0) is 1/2, and a post-op tensor is read at a cell a sparse result does not
    # have: both would change the entries the result does not store.
    lines = _child("""
        extents = dict(i=7, j=6, k=5)
        for result in ("ds", "ss"):
            run(result + " sigmoid", "ij,jk->ik", ["ds", "ds"], result, ["sigmoid"],
                extents)
            run(result + " relu+sigmoid", "ij,jk->ik", ["ds", "ds"], result,
                ["relu", "sigmoid"], extents)
            run(result + " add", "ij,jk->ik", ["ds", "ds"], result, ["add"], extents)
        """)
    changes_zero = f"UnsupportedFeature {SPARSE_CHANGES_ZERO}"
    extra = f"ScorchValueError {SPARSE_EXTRA}"
    assert lines == [
        f"ds sigmoid | {changes_zero}",
        f"ds relu+sigmoid | {changes_zero}",
        f"ds add | {extra}",
        f"ss sigmoid | {changes_zero}",
        f"ss relu+sigmoid | {changes_zero}",
        f"ss add | {extra}",
    ]


# -- where the post-ops are emitted: once, and never nowhere ---------------------------

_EXPRESSIONS = [
    "ab,bc->ac",
    "ab,b->a",
    "ab,ab->ab",
    "ab->ab",
    "ab,bc->ca",
    "abc,cd->abd",
]


def _layouts(rank: int) -> List[str]:
    if rank == 1:
        return ["d", "s"]
    return ["d" * rank, "d" + "s" * (rank - 1), "s" * rank, "o" * rank]


def _post_ops_text_sample():
    for expr in _EXPRESSIONS:
        terms, result = expr.split("->")[0].split(","), expr.split("->")[1]
        formats = [[]]
        for term in terms:
            formats = [f + [layout] for f in formats for layout in _layouts(len(term))]
        for fmt in formats:
            yield expr, fmt, "d" * len(result)


def test_every_emitted_kernel_applies_each_post_op_once(monkeypatch) -> None:
    """Text only, nothing built: a bias add and a relu over a sample of programs.

    A kernel with the value pass reads the bias once, there.  A kernel without it
    applies the post-ops in its copy-out, which may be emitted once per case of a merged
    loop but must be present.  A program not emitted carries a code.  (This sample takes
    the value pass throughout, since none of its copy-outs reaches every cell; the
    copy-out placement has its own test above.)
    """

    post = PostOps(
        ops=[PostOp("add", "postop_0"), PostOp("relu")], extra_tensors=["postop_0"]
    )
    extents = dict(a=5, b=6, c=4, d=3)
    seen = {"value pass": 0, "copy-out": 0, "refused": 0}
    for expr, formats, result in _post_ops_text_sample():
        try:
            text = emitted_einsum_cpp(
                monkeypatch, expr, formats, extents, format=result, _post_ops=post
            )
        except ScorchError as error:
            assert defect_code(error), (expr, formats, error)
            seen["refused"] += 1
            continue
        finally:
            monkeypatch.undo()
        reads, relus = text.count("postop_0_val["), text.count("scorch_relu(")
        where = (expr, formats)
        if VALUE_PASS in text:
            assert text.count(VALUE_PASS) == 1, where
            assert (reads, relus) == (1, 1), where
            seen["value pass"] += 1
        else:
            assert reads >= 1 and reads == relus, where
            seen["copy-out"] += 1
    assert seen["value pass"] > 0, seen


def test_a_copy_out_that_reaches_every_cell_applies_them_as_it_writes() -> None:
    """The one-pass placement, on a program that has the copy-out.

    einsum's schedules copy a dense workspace into a dense result only inside a split
    reduction loop, where the copy-out adds partial sums and cannot apply the post-ops.
    So this program is lowered directly: C = A @ B through a dense workspace over j, the
    copy-out not enclosed by any reduction.  With A's rows dense the row loop visits
    every row, and the copy-out applies the bias and relu as it writes each row, with
    no memcpy and no value pass.  With A's rows compressed, the loop may skip rows A
    does not store, and those rows still need the bias.  In both cases each post-op is
    applied once, and the values match a float64 reference.
    """

    lines = _child("""
        from scorch.compiler.cin import (
            ForAll, IndexVar, Operation, TensorAssign, TensorVar, Where, Workspace)
        from scorch.compiler.cin_lowerer import CINLowerer
        from scorch.compiler.codegen import LLIRLowerer
        from scorch.compiler.compilation_context import CompilationContext
        from scorch.compiler.compile_options import CompileOptions

        options = CompileOptions.from_environment()
        f64 = torch.float64
        for extents in ((7, 6, 5), (37, 29, 23)):
            n, kk, m = extents
            for a_format in ("ds", "ss"):
                i, j, k = IndexVar("i"), IndexVar("j"), IndexVar("k")
                C = TensorVar("C", fmt="dd", shape=(n, m), dtype=f64)
                A = TensorVar("A", fmt=a_format, shape=(n, kk), dtype=f64)
                B = TensorVar("B", fmt="ds", shape=(kk, m), dtype=f64)
                w = Workspace(name="w", dim=1, dtype=f64, dense=True)
                program = ForAll(i, Where(
                    producer=ForAll(k, ForAll(j, TensorAssign(
                        w[j], A[i, k] * B[k, j], op=Operation.ADD))),
                    consumer=ForAll(j, TensorAssign(C[i, j], w[j]))))
                post = PostOps(ops=[PostOp("add", "bias"), PostOp("relu")],
                               extra_tensors=["bias"])
                lowered = CINLowerer(post_ops=post, compile_options=options).lower_IndexStmt(
                    program)
                cpp = LLIRLowerer(compile_options=options).lower_llir(lowered)
                placement = "value pass" if VALUE_PASS in cpp else "copy-out"
                once = cpp.count("bias_val[") == 1 and cpp.count("scorch_relu(") == 1
                prepared = ops._prepare_generated_kernel_build(
                    options.build.preamble_source, cpp, options, CompilationContext(options))
                evaluate = ops._load_validated_prepared_kernel(prepared).evaluate
                a, a_values = operand((n, kk), a_format, f64)
                b, b_values = operand((kk, m), "ds", f64)
                bias = torch.rand(n, m, dtype=f64) - 0.5
                args = [[n, m]]
                for snapshot in ops._snapshot_runtime_tensors((a, b), expected_count=2):
                    args += [snapshot.shape, snapshot.native_mode_indices, snapshot.values]
                got = ops._evaluate_native(evaluate, *args, bias).storage.value.reshape(n, m)
                want = (a_values @ b_values + bias).clamp_min(0)
                error = ((got - want).abs() / want.abs().clamp_min(1)).max().item()
                verdict = "OK" if error <= 1e-10 else f"WRONG {error:.3e}"
                print(f"{n} {a_format} | {placement} | once {once} | memcpy "
                      f"{'memcpy(' in cpp} | {verdict}")
        """)
    assert len(lines) == 4, lines
    for line in lines:
        name, placement, once, memcpy, verdict = line.split(" | ")
        assert (once, verdict) == ("once True", "OK"), line
        if name.endswith(" ds"):
            assert placement == "copy-out", line
        if placement == "copy-out":
            # It writes element by element, applying the post-ops; memcpy applies none.
            assert memcpy == "memcpy False", line


def test_without_post_ops_there_is_no_value_pass(monkeypatch) -> None:
    # The value pass and the copy-out's choice exist only with post-ops.  (An empty
    # chain still turns off einsum's register-block dual path, so its kernel is not
    # compared with the plain one here; the series census compares kernels without
    # post-ops across the change.)
    empty_chain = PostOps(ops=[], extra_tensors=[])
    assert CINLowerer(post_ops=empty_chain)._value_post_ops() == ()
    assert CINLowerer()._value_post_ops() == ()
    extents = dict(a=5, b=6, c=4)
    for formats in (["ds", "dd"], ["ds", "ds"], ["ss", "ds"]):
        plain = emitted_einsum_cpp(
            monkeypatch, "ab,bc->ac", formats, extents, format="dd"
        )
        monkeypatch.undo()
        empty = emitted_einsum_cpp(
            monkeypatch,
            "ab,bc->ac",
            formats,
            extents,
            format="dd",
            _post_ops=PostOps(ops=[], extra_tensors=[]),
        )
        monkeypatch.undo()
        assert VALUE_PASS not in plain
        assert VALUE_PASS not in empty


# -- the lowering and the assembler ----------------------------------------------------


def test_the_lowering_emits_the_value_pass_before_the_storage_assignment() -> None:
    index = IndexVar("i")
    result = TensorVar("Result", fmt="d")
    source = TensorVar("Input", fmt="d")
    statement = ForAll(index, TensorAssign(result[index], source[index]))
    post_ops = PostOps(
        ops=[PostOp(kind="add", tensor_name="bias"), PostOp(kind="gelu")],
        extra_tensors=["bias"],
    )
    function = CINLowerer(post_ops=post_ops).lower_IndexStmt(statement)
    assert type(function) is llir.Function
    cpp = LLIRLowerer().lower_llir(function)
    pass_at = cpp.index(VALUE_PASS)
    assert pass_at < cpp.index("return Result;")
    block = cpp[pass_at:]
    assert (
        "double* __restrict__ Result_post_op_values = "
        "Result_values_torch.data_ptr<double>();" in block
        or "float* __restrict__ Result_post_op_values = "
        "Result_values_torch.data_ptr<float>();" in block
    )
    assert "int64_t Result_post_op_count = Result_values_torch.numel();" in block
    assert (
        "#pragma omp parallel for num_threads(scorch_nthreads(Result_post_op_count, "
        "Result_post_op_count)) schedule(static)" in block
    )
    add_at = block.index(
        "Result_post_op_values[Result_post_op_position] += "
        "bias_val[Result_post_op_position];"
    )
    gelu_at = block.index(
        "Result_post_op_values[Result_post_op_position] = "
        "scorch_gelu(Result_post_op_values[Result_post_op_position]);"
    )
    assert add_at < gelu_at
    assert cpp.count("bias_val[") == 1


def _assembler(level_types, value_post_ops) -> ResultTensorAssembler:
    return ResultTensorAssembler(
        name="C",
        level_types=level_types,
        dtype=torch.float64,
        value_post_ops=value_post_ops,
    )


@pytest.mark.parametrize(
    "level_types, value_post_ops, code",
    [
        ((D, S), (("sigmoid", None),), SPARSE_CHANGES_ZERO),
        ((S,), (("relu", None), ("sigmoid", None)), SPARSE_CHANGES_ZERO),
        ((D, S), (("add", "bias"),), SPARSE_EXTRA),
        ((D,), (("mul", "bias"),), None),
        ((D, S), (("relu", None), ("gelu", None), ("tanh", None)), None),
        ((D,), (("clip", None),), "post_op_kind_unsupported"),
        ((D,), (("add", None),), "internal_argument_type_invalid"),
        ((D,), (("relu", "bias"),), "internal_argument_type_invalid"),
        ((D,), ["relu"], "internal_argument_type_invalid"),
    ],
)
def test_the_assembler_takes_only_post_ops_the_result_can_hold(
    level_types, value_post_ops, code
) -> None:
    if code is None:
        assert _assembler(level_types, value_post_ops).emit_value_post_ops()
        return
    with pytest.raises(ScorchError) as raised:
        _assembler(level_types, value_post_ops)
    assert defect_code(raised.value) == code


def test_an_assembler_without_post_ops_emits_what_it_emitted() -> None:
    plain = ResultTensorAssembler(name="C", level_types=(D, D), dtype=torch.float64)
    assert plain.emit_value_post_ops() == []
    lowered = [LLIRLowerer().lower_llir(stmt) for stmt in plain.emit_storage_epilogue()]
    assert not any("post_op" in line for line in lowered)


@pytest.mark.parametrize(
    "entry", ["emit_value_post_ops", "emit_storage_epilogue", "emit_final_assembly"]
)
@pytest.mark.parametrize("level_types", [(D, D), (D, S)], ids=["dense", "sparse"])
def test_an_assembler_with_undecided_post_ops_refuses_to_finish(
    level_types, entry
) -> None:
    """``value_post_ops=None``: built before the lowering decided the post-ops.

    The lowering builds such an assembler to allocate the result's storage before
    its loop nest is lowered, when it cannot know whether the copy-out will apply
    the post-ops.  It allocates (those emitters never read the post-ops), but the
    value pass and both entries that reach it refuse with a code, so a path that
    finished a result with it would fail closed instead of dropping the post-ops
    or applying them twice.  The lowering's helper leaves them undecided unless
    its caller passes them.
    """

    undecided = _assembler(level_types, None)
    assert undecided.emit_value_array_init()
    assert undecided.emit_level_indices_init() is not None
    with pytest.raises(ScorchError) as raised:
        getattr(undecided, entry)()
    assert defect_code(raised.value) == "result_post_ops_undecided"
    fmt = "".join("d" if t is D else "s" for t in level_types)
    result = TensorVar("C", fmt=fmt, dtype=torch.float64)
    assert _result_tensor_abi_assembler(result).value_post_ops is None
    decided = _result_tensor_abi_assembler(result, value_post_ops=())
    assert decided.emit_value_post_ops() == []


def test_the_post_op_tensor_is_handed_over_in_storage_order() -> None:
    extra = torch.arange(6.0).reshape(2, 3)
    same = ops._post_op_tensor_in_storage_order(extra, (2, 3), [0, 1])
    assert same is extra
    moved = ops._post_op_tensor_in_storage_order(extra, (2, 3), [1, 0])
    assert moved.is_contiguous()
    assert torch.equal(moved, extra.t())
    flat = ops._post_op_tensor_in_storage_order(extra.reshape(6), (2, 3), [1, 0])
    assert torch.equal(flat, extra.t())
    short = torch.zeros(5)
    assert ops._post_op_tensor_in_storage_order(short, (2, 3), [1, 0]) is short
