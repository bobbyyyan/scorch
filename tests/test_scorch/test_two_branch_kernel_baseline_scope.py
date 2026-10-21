"""The two-branch kernel's else branch declares what its own loop reads.

For a dense result with a tiled reduction, public ``einsum`` lowers the program twice --
the register-block nest and the baseline nest -- and emits one kernel that picks a nest
by the reduced index's extent (``ops._stitch_regblock_dual_path``):

    if (A1_size <= 8) { <register-block loop> } else { <baseline loop> }

The stitch kept the first function's statements and took the second function's loop
alone.  When the baseline nest starts with the tile loop over the reduced index, no
parallel loop owns its dense workspace, so the workspace is declared serially at function
scope ahead of the loop -- and that declaration was left behind.
``einsum('ab,bc->ac', A, B, format='dd')`` with a dense ``A`` and a CSR ``B`` used
``wksp`` in the else branch with no declaration and did not build, at any extent.

The else branch now carries the function-scope statements only the baseline function
has around its loop, and the two-branch kernel is not formed when a baseline statement
can be neither found in the first function nor carried.

One baseline statement is deliberately not carried: its tile width.  Both nests name
the width of a tiled index alike (``kTile_b``), the register-block nest declares 8 and
the baseline nest 32, and the stitched kernel has one declaration, the first
function's.  So the else branch of every two-branch kernel with a tiled baseline nest
runs 8 wide, not 32.  That holds for kernels that build today; changing it is a
schedule change that needs a runtime measurement, and this repair leaves their text
as it is.
"""

import re

import pytest
import torch

import scorch
from scorch import ops
from scorch.compiler import llir
from tests.test_scorch.level_array_support import (
    assert_matches,
    emitted_einsum_cpp,
    reference,
    stored,
)

_DECLARATION = re.compile(
    r"^(?:for \()?(?:(?:const|constexpr|static)\s+)*"
    r"[A-Za-z_][\w:]*(?:<[^;()]*>)?\s*(?:[*&]\s*)*(?:__restrict__\s+)?"
    r"([A-Za-z_]\w*)\s*(?:=|;|\[|\{|:|\()"
)
_USE = re.compile(r"(?<![\w.:])(?<!->)([A-Za-z_]\w*)")
_NOT_A_TYPE = {"return", "else", "delete", "case", "goto", "throw", "new"}


def uses_outside_scope(source):
    """Uses of a name the kernel declares, at a point where no declaration is in scope.

    Walks the kernel by brace depth.  A name counts if some statement of the kernel
    declares it; a use is in scope when a declaration of the name sits in a block that
    is still open.  Returns (line number, name, line) for every use that is not.
    """

    lines = [
        re.sub(r'"[^"]*"', '""', line.split("//")[0]).strip()
        for line in source[source.index("evaluate(") :].splitlines()
    ]
    lines = [line for line in lines if line and not line.startswith("#")]

    def declared_by(line):
        match = _DECLARATION.match(line.lstrip("} "))
        if match is None or line.lstrip("} ").split()[0] in _NOT_A_TYPE:
            return None
        return match.group(1)

    declared_somewhere = {name for name in map(declared_by, lines) if name}
    scopes, found = [set()], []
    for number, line in enumerate(lines):
        while line.startswith("}"):
            scopes.pop()
            line = line[1:].strip()
        if not line:
            continue
        opens = line.endswith("{")
        name = declared_by(line)
        if opens:
            scopes.append(set())
        if name:
            scopes[-1].add(name)
        for used in _USE.findall(line):
            if used in declared_somewhere and not any(used in s for s in scopes):
                found.append((number, used, line))
    return found


def test_the_scope_walk_finds_a_use_in_the_other_branch():
    source = """
    Tensor evaluate(int n) {
      if (n <= 8) {
        double* __restrict__ wksp = owner.get();
        wksp[0] = 1.0;
      } else {
        for (int64_t a = 0; a < n; a++) {
          wksp[a] = 2.0;
        }
      }
    }
    """

    assert [(name, line) for _, name, line in uses_outside_scope(source)] == [
        ("wksp", "wksp[a] = 2.0;")
    ]


# (expression, operand formats, result format, extents): a dense first operand and a CSR
# second one into a dense result reach the two-branch kernel with a serial baseline
# workspace, and so does the three-matrix chain that ends in a CSR matrix; the three
# before it are the same expression where they do not.
DENSE_TIMES_CSR = [
    ("ab,bc->ac", ["dd", "ds"], "dd", {"a": 3, "b": 9, "c": 4}),
    ("ij,jk->ik", ["dd", "ds"], "dd", {"i": 3, "j": 9, "k": 4}),
    ("abc,cd->abd", ["ddd", "ds"], "ddd", {"a": 2, "b": 3, "c": 9, "d": 4}),
    ("ab,bc->ac", ["ds", "dd"], "dd", {"a": 3, "b": 9, "c": 4}),
    ("ab,bc->ac", ["dd", "ss"], "dd", {"a": 3, "b": 9, "c": 4}),
    ("ab,bc->ac", ["dd", "ds"], "ds", {"a": 3, "b": 9, "c": 4}),
    ("ab,bc,cd->ad", ["dd", "dd", "ds"], "dd", {"a": 3, "b": 5, "c": 9, "d": 4}),
]

# Every row of the high-rank layout sweep whose result is dense in every level
# (expression, operand formats, extents), as test_high_rank_programs generates them.
SWEEP_DENSE_RESULT_ROWS = [
    ("ijkl->ijkl", "ddds", "i:3,j:2,k:2,l:2"),
    ("ijkl->lkji", "ddss", "i:2,j:2,k:2,l:4"),
    ("ijkl->jkli", "dsss", "i:2,j:4,k:4,l:2"),
    ("ijklm->ijklm", "ddsss", "i:4,j:2,k:3,l:4,m:4"),
    ("ijklm->mlkji", "sdssd", "i:2,j:3,k:4,l:2,m:2"),
    ("ijklm->jklmi", "dssdd", "i:3,j:3,k:4,l:2,m:3"),
    ("ijklmn->ijklmn", "dsdssd", "i:3,j:2,k:2,l:2,m:3,n:3"),
    ("ijklmn->nmlkji", "sdddsd", "i:4,j:3,k:3,l:2,m:4,n:2"),
    ("ijklmn->jklmni", "dddsds", "i:2,j:2,k:3,l:3,m:3,n:3"),
    ("ijkl->ijl", "ddsd", "i:4,j:2,k:2,l:4"),
    ("ijkl->ij", "sdds", "i:3,j:2,k:2,l:4"),
    ("ijkl->ik", "sddd", "i:3,j:3,k:3,l:2"),
    ("ijkl->il", "sdss", "i:2,j:4,k:2,l:3"),
    ("ijkl->kl", "ssds", "i:3,j:3,k:4,l:4"),
    ("ijkl->i", "ddsd", "i:2,j:4,k:2,l:3"),
    ("ijkl->jk", "dsdd", "i:4,j:2,k:4,l:3"),
    ("ijkl->jl", "ddds", "i:2,j:2,k:4,l:2"),
    ("ijkl->li", "sdds", "i:3,j:4,k:3,l:2"),
    ("ijklm->ik", "sdsds", "i:3,j:4,k:2,l:3,m:4"),
    ("ijklm->ijkm", "dddsd", "i:4,j:2,k:4,l:4,m:4"),
    ("ijklm->im", "sssds", "i:3,j:3,k:2,l:2,m:2"),
    ("ijklm->ijlm", "ddsds", "i:3,j:2,k:4,l:2,m:2"),
    ("ijklm->l", "ssdss", "i:4,j:4,k:3,l:4,m:4"),
    ("ijklm->kl", "sssss", "i:3,j:3,k:3,l:4,m:2"),
    ("ijklm->i", "ssssd", "i:2,j:2,k:4,l:3,m:4"),
    ("ijklm->ijkl", "ddsds", "i:4,j:4,k:2,l:3,m:2"),
    ("ijklm->mi", "ssdsd", "i:4,j:2,k:4,l:2,m:4"),
    ("ijklmn->i", "sddssd", "i:4,j:4,k:2,l:2,m:4,n:3"),
    ("ijklmn->jk", "ssddds", "i:2,j:3,k:2,l:2,m:4,n:4"),
    ("ijklmn->ijkmn", "ddsdsd", "i:4,j:4,k:2,l:2,m:2,n:4"),
    ("ijklmn->in", "dddsdd", "i:3,j:2,k:2,l:4,m:2,n:2"),
    ("ijklmn->l", "sddddd", "i:2,j:4,k:4,l:2,m:4,n:4"),
    ("ijklmn->kn", "ddssds", "i:3,j:3,k:4,l:2,m:4,n:2"),
    ("ijklmn->im", "dssddd", "i:2,j:4,k:4,l:3,m:4,n:4"),
    ("ijklmn->k", "dsdsds", "i:4,j:3,k:2,l:2,m:3,n:4"),
    ("ijklmn->ni", "dddsss", "i:2,j:2,k:3,l:2,m:4,n:2"),
    ("ijkl,ia->ajkl", "ddsd,ds", "i:3,j:4,k:4,l:3,a:4"),
    ("ijkl,ja->iakl", "ssdd,ds", "i:2,j:4,k:4,l:2,a:2"),
    ("ijkl,ka->ijal", "ddss,sd", "i:3,j:4,k:3,l:4,a:3"),
    ("ijkl,la->ijka", "ddds,sd", "i:3,j:4,k:2,l:3,a:2"),
    ("ijklm,ka->ijalm", "ssssd,dd", "i:3,j:4,k:2,l:2,m:4,a:3"),
    ("ijklm,ma->ijkla", "sssds,sd", "i:3,j:4,k:3,l:4,m:3,a:3"),
    ("ijkl,ia,jb->abkl", "dsds,ds,dd", "i:4,j:2,k:2,l:2,a:2,b:2"),
    ("ijkl,ka,lb->ijab", "sdsd,ds,ss", "i:3,j:4,k:4,l:4,a:4,b:4"),
    ("ijklm,ia,jb,kc->abclm", "sdssd,ds,sd,ds", "i:3,j:2,k:2,l:4,m:2,a:3,b:3,c:4"),
    ("ijkl,jr,kr,lr->ir", "dsdd,sd,ss,sd", "i:4,j:4,k:3,l:4,r:2"),
    ("ijkl,ir,kr,lr->jr", "dssd,sd,ss,sd", "i:2,j:3,k:2,l:3,r:3"),
    ("ijklm,jr,kr,lr,mr->ir", "ddsdd,dd,sd,dd,dd", "i:4,j:2,k:2,l:2,m:3,r:2"),
    ("ijkl,klmn->ijmn", "ddds,ddsd", "i:4,j:3,k:4,l:3,m:3,n:2"),
    ("ijkl,jklm->im", "sddd,ssss", "i:3,j:4,k:3,l:3,m:3"),
    ("ijkl,lmkn->ijmn", "ddsd,dddd", "i:3,j:2,k:3,l:3,m:2,n:4"),
    ("bijk,bkl->bijl", "sssd,ddd", "b:3,i:3,j:4,k:3,l:3"),
    ("abij,abjk->abik", "dsds,ddds", "a:4,b:2,i:3,j:2,k:2"),
    ("bijk,bjk->bi", "dsss,dss", "b:3,i:2,j:4,k:4"),
    ("ij,kl->ijkl", "sd,ss", "i:4,j:3,k:3,l:2"),
    ("ijk,lmn->ijklmn", "sdd,dss", "i:3,j:4,k:4,l:2,m:2,n:2"),
    ("ij,klm->ijklm", "ds,sds", "i:4,j:3,k:4,l:2,m:4"),
    ("ijklm,ijklm->ijklm", "ddsss,ssdds", "i:3,j:2,k:2,l:2,m:4"),
    ("ijkl,ijkl->ijkl", "sdds,dsss", "i:3,j:3,k:2,l:2"),
    ("ijkl,ijkl->il", "ssss,ssdd", "i:2,j:4,k:4,l:4"),
]


def _extents(text):
    return {pair.split(":")[0]: int(pair.split(":")[1]) for pair in text.split(",")}


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32, torch.int64])
@pytest.mark.parametrize(
    "expr, formats, result_format, extents",
    DENSE_TIMES_CSR,
    ids=[f"{e}|{','.join(f)}|{r}" for e, f, r, _ in DENSE_TIMES_CSR],
)
def test_dense_times_csr_reads_no_name_outside_its_scope(
    monkeypatch, expr, formats, result_format, extents, dtype
):
    source = emitted_einsum_cpp(
        monkeypatch, expr, formats, extents, dtype=dtype, format=result_format
    )

    assert uses_outside_scope(source) == []


@pytest.mark.parametrize(
    "expr, formats, extents",
    SWEEP_DENSE_RESULT_ROWS,
    ids=[f"{e}|{f}" for e, f, _ in SWEEP_DENSE_RESULT_ROWS],
)
def test_sweep_dense_result_row_reads_no_name_outside_its_scope(
    monkeypatch, expr, formats, extents
):
    result_format = "d" * len(expr.split("->")[1])

    source = emitted_einsum_cpp(
        monkeypatch, expr, formats.split(","), _extents(extents), format=result_format
    )

    assert uses_outside_scope(source) == []


def test_else_branch_declares_its_workspace(monkeypatch):
    source = emitted_einsum_cpp(
        monkeypatch, "ab,bc->ac", ["dd", "ds"], {"a": 3, "b": 9, "c": 4}, format="dd"
    )
    lines = [line.strip() for line in source.splitlines() if line.strip()]

    # The guard is on b's extent: at most 8 takes the first nest, more the second.
    guard = lines.index("if (A1_size <= 8) {")
    otherwise = lines.index("} else {", guard)
    loop = lines.index(
        "for (int64_t b_out = 0; b_out < A1_size; b_out += kTile_b) {", otherwise
    )
    carried = lines[otherwise + 1 : loop]
    assert carried == [
        "auto wksp_pool_owner = scorch_make_aligned_buffer<double>((size_t)C1_size);",
        "double* __restrict__ wksp = wksp_pool_owner.get();",
    ]
    # One tile width for both nests, the first function's: the baseline nest's own
    # (32) is not carried.
    assert [line for line in lines if line.startswith("constexpr int kTile_b")] == [
        "constexpr int kTile_b = 8;"
    ]
    assert lines.index("constexpr int kTile_b = 8;") < guard
    # More than one tile of b reaches the result: the copy-out adds.
    assert "C_values[pC1] += wksp[c];" in lines[loop:]


def _operands(expr, formats, extents, dtype, seed):
    generator = torch.Generator().manual_seed(seed)
    dense = []
    for term, fmt in zip(expr.split("->")[0].split(","), formats):
        shape = tuple(extents[index] for index in term)
        if dtype.is_floating_point:
            values = 0.5 + torch.rand(shape, generator=generator, dtype=torch.float64)
        else:
            values = torch.randint(1, 9, shape, generator=generator)
        if set(fmt) != {"d"}:
            values = values * (torch.rand(shape, generator=generator) < 0.6)
        dense.append(values.to(dtype))
    return dense


# The guard index's extent: 5 and 8 take the first nest (one tile of 8); 9, 40 and 70
# take the second, over two, five and nine tiles of 8.
GUARD_EXTENTS = [5, 8, 9, 40, 70]
EXECUTED = [
    ("ab,bc->ac", ["dd", "ds"], "dd", {"a": 6, "c": 7}, "b"),
    ("abc,cd->abd", ["ddd", "ds"], "ddd", {"a": 3, "b": 4, "d": 5}, "c"),
    ("ab,bc,cd->ad", ["dd", "dd", "ds"], "dd", {"a": 4, "b": 11, "d": 6}, "c"),
]


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32, torch.int64])
@pytest.mark.parametrize("guard_extent", GUARD_EXTENTS)
@pytest.mark.parametrize(
    "expr, formats, result_format, other_extents, guard_index",
    EXECUTED,
    ids=[e for e, *_ in EXECUTED],
)
def test_dense_times_csr_into_a_dense_result_matches(
    expr, formats, result_format, other_extents, guard_index, guard_extent, dtype
):
    extents = {**other_extents, guard_index: guard_extent}
    dense = _operands(expr, formats, extents, dtype, 100 + guard_extent)
    tensors = [
        stored(values, fmt, "ABDE"[position])
        for position, (values, fmt) in enumerate(zip(dense, formats))
    ]

    result = scorch.einsum(expr, *tensors, format=result_format)

    assert_matches(result, reference(expr, dense), dtype)


# The same contraction spelled with other index names (the reported form), and under
# the other result formats, on both sides of the guard.
@pytest.mark.parametrize("guard_extent", [8, 40])
@pytest.mark.parametrize(
    "expr, result_format",
    [
        ("ij,jk->ik", "dd"),
        ("ab,bc->ac", "ds"),
        ("ab,bc->ac", "ss"),
        ("ab,bc->ac", "sd"),
    ],
)
def test_dense_times_csr_under_other_names_and_result_formats(
    expr, result_format, guard_extent
):
    left, right = expr.split("->")[0].split(",")
    extents = {left[0]: 6, left[1]: guard_extent, right[1]: 7}
    dense = _operands(expr, ["dd", "ds"], extents, torch.float64, 200 + guard_extent)
    tensors = [stored(dense[0], "dd", "A"), stored(dense[1], "ds", "B")]

    result = scorch.einsum(expr, *tensors, format=result_format)

    assert_matches(result, reference(expr, dense), torch.float64)


def _int(name):
    return llir.Var(name=name, type=llir.DataType.INT64)


def _declare(name, value):
    return llir.VarInit(var=_int(name), value=llir.Literal(value))


def _call(function, *names):
    return llir.FunctionCallStmt(function, [_int(name) for name in names])


def _loop(variable, bound, body):
    return llir.ForLoop(
        init=_declare(variable, 0),
        cond=llir.BinOp(op="<", left=_int(variable), right=_int(bound)),
        update=llir.Increment(var=_int(variable)),
        body=body,
    )


def _function(before, after=()):
    """A lowered function: before, one row loop over a tile loop, after."""

    row_loop = _loop("i", "rows", [_loop("k_out", "free_size", [_call("work", "i")])])
    body = [_declare("rows", 4), *before, row_loop, *after]
    return llir.Function(llir.DataType.INT64, "evaluate", [], body), row_loop


def _stitched_else(base_before, base_after=(), rb_before=(), rb_after=()):
    rb, _ = _function(rb_before, rb_after)
    base, base_loop = _function(base_before, base_after)
    stitched = ops._stitch_regblock_dual_path(rb, base, 8)
    if stitched is None:
        return None, base_loop
    (branch,) = [s for s in stitched.body if type(s) is llir.IfThenElse]
    return branch.else_body, base_loop


def test_a_baseline_function_with_nothing_of_its_own_carries_nothing():
    else_body, base_loop = _stitched_else([], rb_before=[_declare("kTile", 8)])

    assert len(else_body) == 1 and else_body[0] is base_loop


def test_declarations_next_to_the_baseline_loop_are_carried_in_order():
    owner, workspace = _declare("owner", 1), _declare("wksp", 2)

    else_body, base_loop = _stitched_else([owner, workspace])

    assert [s is t for s, t in zip(else_body, [owner, workspace, base_loop])] == [
        True,
        True,
        True,
    ]
    assert len(else_body) == 3


def test_a_declaration_the_first_function_makes_differently_is_carried():
    extent = _declare("extent", 32)

    else_body, base_loop = _stitched_else([extent], rb_before=[_declare("extent", 8)])

    assert else_body[0] is extent and else_body[1] is base_loop


def _constant(name, value):
    return llir.VarInit(
        var=llir.Var(name=name, type=llir.DataType.CONSTEXPR_INT),
        value=llir.Literal(value),
    )


def test_a_tile_width_both_functions_name_is_not_carried():
    workspace = _declare("wksp", 2)

    else_body, base_loop = _stitched_else(
        [_constant("kTile_k", 32), workspace], rb_before=[_constant("kTile_k", 8)]
    )

    assert len(else_body) == 2
    assert else_body[0] is workspace and else_body[1] is base_loop


def test_a_tile_width_only_the_baseline_function_names_is_carried():
    width = _constant("kTile_j", 32)

    else_body, base_loop = _stitched_else([width], rb_before=[_constant("kTile_k", 8)])

    assert else_body[0] is width and else_body[1] is base_loop


def test_a_statement_behind_the_loop_that_closes_a_carried_name_is_carried():
    workspace, release = _declare("wksp", 2), _call("release", "wksp")

    else_body, base_loop = _stitched_else([workspace], [release])

    assert else_body[0] is workspace and else_body[1] is base_loop
    assert else_body[2] is release


@pytest.mark.parametrize(
    "base_before, base_after, rb_before, rb_after",
    [
        # Its own declaration, but a statement both functions make sits between it
        # and the loop: carrying it would move it past that statement.
        ([_declare("mine", 1), _declare("both", 2)], [], [_declare("both", 2)], []),
        # Next to the loop, but not a declaration.
        ([_call("touch", "rows")], [], [], []),
        # Carried, and a statement that stays reads it.
        ([_declare("wksp", 2)], [_call("use", "wksp")], [], [_call("use", "wksp")]),
        # Behind the loop, and closing nothing that was carried.
        ([], [_call("touch", "rows")], [], []),
        # Its own statement behind the loop, with a shared one between.
        (
            [_declare("wksp", 2)],
            [_call("shared", "rows"), _call("release", "wksp")],
            [],
            [_call("shared", "rows")],
        ),
        # A tile width both name, between a carried declaration and the loop.
        (
            [_declare("wksp", 2), _constant("kTile_k", 32)],
            [],
            [_constant("kTile_k", 8)],
            [],
        ),
    ],
    ids=[
        "not-next-to-the-loop",
        "not-a-declaration",
        "read-by-a-statement-that-stays",
        "closes-nothing",
        "behind-a-shared-statement",
        "shared-width-inside-the-carry",
    ],
)
def test_the_two_branch_kernel_is_not_formed(
    base_before, base_after, rb_before, rb_after
):
    else_body, _ = _stitched_else(base_before, base_after, rb_before, rb_after)

    assert else_body is None
