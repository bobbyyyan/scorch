"""The SpGEMM work estimate reads every position array inside its length, and no
team is larger than the workspace pool it indexes.

The two-phase compressed-``Where`` kernels size their team, their per-thread workspace
pool and their chunk from a work estimate, ``A_nnz * (B_nnz / B0_size + 1)``, which
reads each operand's count as ``X{L}_pos[X0_size]``, with ``L`` the operand's deepest
compressed level.  Level ``L``'s position array has one entry per stored position of
level ``L - 1``, plus one, so for ``L > 1`` that read lies past the array's end when
level ``L - 1`` stores fewer than ``X0_size`` positions (dennis, region scan 82815:
``ijkl,ijkl->il`` with ``A`` in ``dssd`` read ``A2_pos[4]`` of a three-entry array).

Two changes, and no other:

(b) for ``L > 1`` the index is clamped to the array's last entry,
    ``X{L}_pos[std::min<int64_t>(X0_size, X_mode_indices[L][0].numel() - 1)]``.  It reads
    the same entry wherever the unclamped read was inside the array, so the estimate,
    and with it the team, the chunk and the pool size, are unchanged there.
(a) the pool's worker count, one evaluation of the policy, is also the team of both
    regions that index the pool by ``omp_get_thread_num()``, or caps each region's own
    policy where the two regions' policies differ.  Before, each region evaluated the
    policy again, so a read past the end could give a region more threads than the pool
    has workspaces.  The same holds for the dense per-worker workspaces the parallel
    marking pass places (``<ws>_pool_owner``, one buffer of count x extent elements,
    each worker's slice at ``omp_get_thread_num() * extent``): the region that slices
    them takes their count, ``std::min`` over the counts when there are several.

The pool check reads the kernel the way the compiler would: each pool use is matched to
its innermost enclosing ``#pragma omp parallel`` block by bracket matching, and every use
other than an index in such a region, the declaration and the construction calls outside
every region fails.  Planted kernels check the check.
"""

import re
from typing import Dict, List, Optional, Sequence, Tuple

import pytest
import torch

import scorch
from scorch.compiler import llir
from scorch.compiler.compressed_where_openmp_pass import (
    _spgemm_flop_work_estimate,
    _stored_count_read,
)
from scorch.compiler.llir_traversal import LLIRTraversalContext
from tests.test_scorch import test_high_rank_programs as high_rank
from tests.test_scorch.level_array_support import (
    assert_matches,
    emitted_einsum_cpp,
    level_arrays,
    reference,
    stored,
)

TRAVERSAL = LLIRTraversalContext(stage="test", pass_name="spgemm_work_estimate")
BOUND = llir.Var(name="A0_size", type=llir.DataType.INT)


def _var(name: str, data_type: llir.DataType) -> llir.Var:
    return llir.Var(name=name, type=data_type)


def _end(operand: str, level: int) -> llir.VarInit:
    """``int pX{level}_end = X{level}_pos[pX{level - 1} + 1];``"""

    return llir.VarInit(
        _var(f"p{operand}{level}_end", llir.DataType.INT),
        llir.ArrayAccess(
            _var(f"{operand}{level}_pos", llir.DataType.PTR_INT),
            llir.Add(
                _var(f"p{operand}{level - 1}", llir.DataType.INT),
                llir.Literal(1, llir.DataType.INT),
            ),
        ),
    )


def _body(left: Sequence[int], right: Sequence[int]) -> List[llir.Stmt]:
    return [_end("A", level) for level in left] + [_end("B", level) for level in right]


def _bound(left: Sequence[int], right: Sequence[int]) -> frozenset:
    return frozenset(
        [f"A{level}_pos" for level in left] + [f"B{level}_pos" for level in right]
    )


def _read(tensor: str, deepest: int) -> str:
    """The estimate's read of ``tensor``'s count, as its text spells it."""

    name = f"{tensor}{deepest}_pos"
    if deepest == 1:
        return f"{name}[{tensor}0_size]"
    return (
        f"{name}[std::min<int64_t>({tensor}0_size, "
        f"{tensor}_mode_indices[{deepest}][0].numel() - 1)]"
    )


def test_two_level_csr_keeps_its_text():
    estimate = _spgemm_flop_work_estimate(
        _body([1], [1]), BOUND, TRAVERSAL, _bound([1], [1])
    )
    assert estimate is not None
    assert estimate.text == (
        "(long)A1_pos[A0_size] * (B0_size > 0 ? B1_pos[B0_size] / B0_size + 1 : 1)"
    )


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ([1, 2], [1]),
        ([1], [1, 2]),
        ([1, 2], [1, 2]),
        ([1, 2, 3], [1, 2, 3]),
        # A dense level between compressed ones reads the deepest array as well.
        ([2], [1]),
        ([1, 3], [1]),
        ([2, 3], [1, 3]),
    ],
)
def test_a_deeper_operand_reads_its_array_clamped(left, right):
    estimate = _spgemm_flop_work_estimate(
        _body(left, right), BOUND, TRAVERSAL, _bound(left, right)
    )
    assert estimate is not None
    assert estimate.text == (
        f"(long){_read('A', max(left))} * "
        f"(B0_size > 0 ? {_read('B', max(right))} / B0_size + 1 : 1)"
    )
    for tensor, levels in (("A", left), ("B", right)):
        assert f"{tensor}{max(levels)}_pos[{tensor}0_size]" not in estimate.text or (
            max(levels) == 1
        )


def test_the_clamped_read_is_typed_as_its_text_spells_it():
    chain = _stored_count_read(
        "A", {1, 2}, _var("A0_size", llir.DataType.INT64), frozenset({"A2_pos"})
    )
    assert chain is not None
    text, expr = chain
    assert text == _read("A", 2)
    assert type(expr) is llir.ArrayAccess
    assert expr.array.name == "A2_pos"
    index = expr.index
    assert type(index) is llir.FunctionCall
    assert index.name == "std::min"
    assert index.template_args == (llir.DataType.INT64,)
    outer, last = index.args
    assert outer.name == "A0_size"
    assert type(last) is llir.BinOp and last.op == "-"
    assert type(last.left) is llir.MemberCall and last.left.member == "numel"


def test_a_deeper_array_the_kernel_does_not_bind_takes_the_guarded_estimate():
    assert _spgemm_flop_work_estimate(_body([1, 2], [1]), BOUND, TRAVERSAL) is None
    assert (
        _spgemm_flop_work_estimate(
            _body([1], [1, 2]), BOUND, TRAVERSAL, frozenset({"A1_pos", "B1_pos"})
        )
        is None
    )
    # Level 1 is read inside the array without an entry count.
    assert _spgemm_flop_work_estimate(_body([1], [1]), BOUND, TRAVERSAL) is not None


# dennis's examples from region scan 82815 (job 82954), one per pattern, with the
# clamped reads their estimates make.  Three are places in test_high_rank_programs'
# population, which seeds a program's inputs by its place.  3530 and 3620 are the scan's
# "wide" family (merge_class/vmscan's expressions over every layout, extents a=3 b=9
# c=4 d=5), seeded by their number in the scan (dennis, job 83089).
WIDE_EXTENTS = (("a", 3), ("b", 9), ("c", 4), ("d", 5))
HIGH_RANK = [
    (228, "population", ("ijkl,ijkl->il", ("dssd", "dsdd"), "ds", None), ("A", 2)),
    (3620, "wide", ("abc,cd->abd", ("dss", "ds"), "dss", WIDE_EXTENTS), ("A", 2)),
    (3530, "wide", ("abc,cd->abd", ("dds", "ds"), "dss", WIDE_EXTENTS), ("A", 2)),
    (1004, "population", ("bijk,bjk->bi", ("dsss", "dss"), "ds", None), ("A", 3)),
    (187, "population", ("ijkl,jklm->im", ("ddss", "dsds"), "ds", None), ("A", 3)),
]
HIGH_RANK_IDS = [str(program[0]) for program in HIGH_RANK]
# A deeper level's position array indexed by the level-0 extent: the unclamped read.
OLD_READ = re.compile(r"\b(?P<tensor>[A-Z])([2-9])_pos\[(?P=tensor)0_size\]")
POOL_COUNT = re.compile(r"\bint (\w+)_thread_count = ([^;]*);")
POOL_TOKEN = re.compile(r"\b(\w+)_pool\b")
POOL_CONSTRUCTION = re.compile(
    r"\s*\.\s*(reserve|emplace_back|push_back|resize|size)\s*\("
)
# parallel_marking_pass's dense per-worker workspaces: one buffer of count x extent
# elements, each worker's slice taken at omp_get_thread_num() * extent.  Its serial
# placement (a cluster no parallel loop placed) declares an owner sized for one worker
# and takes slice zero; a kernel can hold both under one name, in different blocks.
DENSE_OWNER = re.compile(r"\bauto (\w+)_pool_owner = scorch_make_aligned_buffer<")
DENSE_COUNTED = re.compile(
    r"[\w:<> ]*>\(\s*scorch_checked_size_product\("
    r"\s*\(size_t\)\s*(\w+)_thread_count\s*,"
)
DENSE_ONE_WORKER = re.compile(r"[\w:<> ]*>\(\s*\(size_t\)\s*\w+\s*\)\s*;")
DENSE_TOKEN = re.compile(r"\b(\w+)_pool_owner\b")
DENSE_GET = re.compile(r"\s*\.\s*get\s*\(\s*\)")
DENSE_SLICE = re.compile(r"\s*\+\s*\(size_t\)\s*omp_get_thread_num\s*\(\s*\)\s*\*")
# The pointer a slice-zero .get() initializes: "<type>* [__restrict__] <name> = ".
SLICE_TARGET = re.compile(r"\s*[\w:]+\s*\*\s*(?:__restrict__\s+)?(\w+)\s*=\s*")
REGION = re.compile(r"#pragma omp parallel\b([^\n]*)")
FOR_HEADER = re.compile(r"for\s*\(")
SPACE = re.compile(r"\s*")


def _high_rank_program(monkeypatch, number, family, spec):
    expr, formats, result, extents = spec
    if family == "population":
        program = high_rank._POPULATION[number]
        assert (program.expr, program.operand_formats, program.result_format) == (
            expr,
            formats,
            result,
        ), program.name
        return program
    program = high_rank.Program(
        family=family,
        expr=expr,
        operand_formats=formats,
        result_format=result,
        extents=extents,
    )
    # Seeded by its number in the scan, as the harness seeds a population program by
    # its place.
    monkeypatch.setitem(high_rank._POPULATION_INDEX, program.name, number)
    return program


def _emitted(monkeypatch, program, blocking: bool) -> str:
    return emitted_einsum_cpp(
        monkeypatch,
        program.expr,
        program.operand_formats,
        dict(program.extents),
        format=program.result_format,
        _compile_options=high_rank._options(blocking),
    )


def _past_comment_or_literal(cpp: str, i: int) -> int:
    """The index after a comment or a string or character literal at ``i``, or ``i``."""

    if cpp.startswith("//", i):
        end = cpp.find("\n", i)
        return len(cpp) if end < 0 else end
    if cpp.startswith("/*", i):
        end = cpp.find("*/", i + 2)
        return len(cpp) if end < 0 else end + 2
    if cpp[i] in "\"'":
        j = i + 1
        while j < len(cpp) and cpp[j] != cpp[i]:
            j += 2 if cpp[j] == "\\" else 1
        return j + 1
    return i


def _past_bracket(cpp: str, i: int) -> int:
    """The index after the bracket that closes the one at ``cpp[i]``."""

    opening = cpp[i]
    closing = {"(": ")", "{": "}"}[opening]
    depth = 0
    while i < len(cpp):
        j = _past_comment_or_literal(cpp, i)
        if j != i:
            i = j
            continue
        depth += {opening: 1, closing: -1}.get(cpp[i], 0)
        i += 1
        if depth == 0:
            return i
    raise AssertionError(f"unbalanced {opening}")


def _regions(cpp: str) -> List[Tuple[int, int, Optional[str]]]:
    """``(start, end, team)`` of every ``#pragma omp parallel`` construct.  Its block is
    found after any directive or comment lines that follow the pragma: a braced block,
    the loop of ``parallel for``, or else one statement.  ``team`` is the
    ``num_threads`` operand, or None without one."""

    regions = []
    for match in REGION.finditer(cpp):
        clause, team = match.group(1), None
        if "num_threads(" in clause:
            begin = clause.index("num_threads(") + len("num_threads")
            team = clause[begin + 1 : _past_bracket(clause, begin) - 1].strip()
        i = SPACE.match(cpp, match.end()).end()
        while cpp.startswith(("#", "//", "/*"), i):
            i = (
                cpp.index("\n", i)
                if cpp[i] == "#"
                else _past_comment_or_literal(cpp, i)
            )
            i = SPACE.match(cpp, i).end()
        if FOR_HEADER.match(cpp, i):
            i = SPACE.match(cpp, _past_bracket(cpp, cpp.index("(", i))).end()
        end = _past_bracket(cpp, i) if cpp[i] == "{" else cpp.index(";", i) + 1
        regions.append((match.start(), end, team))
    return regions


def _blocks(cpp: str) -> List[Tuple[int, int]]:
    """``(open, close)`` of every braced block, comments and literals skipped."""

    blocks, stack, i = [], [], 0
    while i < len(cpp):
        j = _past_comment_or_literal(cpp, i)
        if j != i:
            i = j
            continue
        if cpp[i] == "{":
            stack.append(i)
        elif cpp[i] == "}":
            blocks.append((stack.pop(), i))
        i += 1
    return blocks


def _visible(
    starts: List[int], at: int, blocks: List[Tuple[int, int]]
) -> Optional[int]:
    """The declaration (of ``starts``, one name's) that a use of the name at ``at``
    refers to: the latest one before ``at`` whose innermost enclosing block holds
    ``at``, as one function's name lookup resolves it; None when none is visible."""

    for start in sorted((s for s in starts if s < at), reverse=True):
        scope = max((b for b in blocks if b[0] < start < b[1]), default=None)
        if scope is None or scope[0] < at < scope[1]:
            return start
    return None


def _dense_index_use(
    cpp: str,
    token: "re.Match[str]",
    declaration: int,
    regions: List[Tuple[int, int, Optional[str]]],
    blocks: List[Tuple[int, int]],
) -> str:
    """The use a ``.get()`` of a dense owner makes, the owner being the declaration at
    ``declaration`` (resolved by name lookup).  A worker's slice (``+ (size_t)
    omp_get_thread_num() *``) is "uncounted" on an owner not sized by a count, "outside
    a region" outside every region, and else its region's team against the owner's
    count (_team_use).  Slice zero is "slice zero" only outside every region, on an
    owner sized for one worker, into a pointer no region uses; else "slice zero in a
    region", "slice zero of a counted pool", "slice zero, unrecognized target" or "a
    one-worker slice used in a region"."""

    size_at = DENSE_OWNER.match(cpp, declaration)
    assert size_at is not None
    counted = DENSE_COUNTED.match(cpp, size_at.end())
    enclosing = [r for r in regions if r[0] <= token.start() < r[1]]
    get = DENSE_GET.match(cpp, token.end())
    assert get is not None
    if DENSE_SLICE.match(cpp, get.end()):
        if counted is None:
            return "uncounted"
        if not enclosing:
            return "outside a region"
        return _team_use(counted.group(1), max(enclosing)[2])
    if enclosing:
        return "slice zero in a region"
    if not DENSE_ONE_WORKER.match(cpp, size_at.end()):
        return "slice zero of a counted pool"
    line_start = cpp.rfind("\n", 0, token.start()) + 1
    target = SLICE_TARGET.fullmatch(cpp, line_start, token.start())
    if target is None:
        return "slice zero, unrecognized target"
    name = re.escape(target.group(1))
    declared = [
        m.start(1)
        for m in re.finditer(
            rf"(?:\*\s*(?:__restrict__\s+)?|\bauto\s*[*&]?\s*)({name})\s*=(?!=)", cpp
        )
    ]
    for use in re.finditer(rf"\b{name}\b", cpp):
        inside = any(r[0] <= use.start() < r[1] for r in regions)
        if inside and use.start() not in declared:
            if _visible(declared, use.start(), blocks) == target.start(1):
                return "a one-worker slice used in a region"
    return "slice zero"


def _min_arguments(team: str) -> Optional[List[str]]:
    """The arguments of a ``std::min`` call spanning all of ``team``, nested ``std::min``
    calls flattened; None when ``team`` is not such a call."""

    if not team.startswith("std::min(") or _past_bracket(team, 8) != len(team):
        return None
    arguments, depth, start, i = [], 0, 9, 9
    while i < len(team) - 1:
        j = _past_comment_or_literal(team, i)
        if j != i:
            i = j
            continue
        depth += {"(": 1, ")": -1}.get(team[i], 0)
        if team[i] == "," and depth == 0:
            arguments.append(team[start:i].strip())
            start = i + 1
        i += 1
    arguments.append(team[start:-1].strip())
    flat: List[str] = []
    for argument in arguments:
        nested = _min_arguments(argument)
        flat.extend([argument] if nested is None else nested)
    return flat


def _team_use(workspace: str, team: Optional[str]) -> str:
    count = f"{workspace}_thread_count"
    if team == count:
        return "count"
    if team is not None and count in (_min_arguments(team) or ()):
        return "min"
    return f"team {team}"


def _pool_uses(cpp: str) -> List[Tuple[str, str]]:
    """``(workspace, use)`` for every token of a workspace pool of either family: a
    linked-list ``<workspace>_pool`` or a dense ``<workspace>_pool_owner``.  An index of
    a linked-list pool (``[``) is "count" or "min" in a region whose team is the pool's
    count or a ``std::min`` with the count among its arguments, "team <operand>" (or
    "team None") in a region with another team, "serial" in a kernel with no region,
    "outside a region" in a kernel with one, and "undeclared" for a pool with no count.
    A ``.get()`` of a dense owner is classified against the declaration it names
    (_dense_index_use).  The declaration and a construction call outside every region
    are "setup"; any other use is "unrecognized"."""

    pools = {workspace for workspace, _ in POOL_COUNT.findall(cpp)}
    owner_at: Dict[str, List[int]] = {}
    for owner in DENSE_OWNER.finditer(cpp):
        owner_at.setdefault(owner.group(1), []).append(owner.start())
    regions = _regions(cpp)
    blocks = _blocks(cpp)
    uses = []
    for match in POOL_TOKEN.finditer(cpp):
        workspace = match.group(1)
        after = SPACE.match(cpp, match.end()).end()
        indexed = cpp.startswith("[", after)
        if workspace not in pools:
            if indexed:
                uses.append((workspace, "undeclared"))
            continue
        enclosing = [r for r in regions if r[0] <= match.start() < r[1]]
        if not indexed:
            before = cpp[max(0, match.start() - 200) : match.start()]
            declared = cpp.startswith(";", after) and before.rstrip().endswith(">")
            constructed = POOL_CONSTRUCTION.match(cpp, match.end()) is not None
            setup = (declared or constructed) and not enclosing
            uses.append((workspace, "setup" if setup else "unrecognized"))
            continue
        if not enclosing:
            uses.append((workspace, "outside a region" if regions else "serial"))
            continue
        uses.append((workspace, _team_use(workspace, max(enclosing)[2])))
    for match in DENSE_TOKEN.finditer(cpp):
        workspace = match.group(1)
        if workspace not in owner_at:
            continue
        enclosing = [r for r in regions if r[0] <= match.start() < r[1]]
        if not DENSE_GET.match(cpp, match.end()):
            line = cpp[cpp.rfind("\n", 0, match.start()) + 1 : match.start()]
            declared = line.strip() == "auto" and cpp.startswith(
                " = scorch_make_aligned_buffer<", match.end()
            )
            setup = declared and not enclosing
            uses.append((workspace, "setup" if setup else "unrecognized"))
            continue
        declaration = _visible(owner_at[workspace], match.start(), blocks)
        if declaration is None:
            uses.append((workspace, "undeclared"))
            continue
        uses.append(
            (workspace, _dense_index_use(cpp, match, declaration, regions, blocks))
        )
    return uses


def _assert_teams_within_pool(cpp: str) -> int:
    """Every index of a workspace pool of either family lies in a region that takes the
    pool's count as its team, or a ``std::min`` with it, or is the serial placement's
    slice zero outside every region; every other use is its declaration or a
    construction call outside every region; and some region indexes each counted pool.
    Returns the number of counted pools."""

    pools = POOL_COUNT.findall(cpp)
    uses = _pool_uses(cpp)
    assert all(use in ("count", "min", "setup", "slice zero") for _, use in uses), uses
    indexed = {workspace for workspace, use in uses if use in ("count", "min")}
    for workspace, _ in pools:
        assert workspace in indexed, f"no region indexes {workspace}_pool"
    return len(pools)


_PLANTED_POOL = (
    "int wksp_thread_count = scorch_nthreads(n, 1);\n"
    "std::vector<dense_workspace_1d<double>> wksp_pool;\n"
    "wksp_pool.reserve((size_t)wksp_thread_count);\n"
)
_PLANTED_INDEX = "  auto wksp = wksp_pool[(size_t)omp_get_thread_num()].make_view();\n"
# More than the old check's 400-character window, with braces in a comment and a string.
_FAR = '  x += 1; /* } { */ const char* s = "}";\n' * 20
_COUNT = "num_threads(wksp_thread_count)"
_MIN = "num_threads(std::min(scorch_nthreads(m, 1), wksp_thread_count))"
_OTHER = "num_threads(scorch_nthreads(m, 1))"
# (name, the region's clause, text before the index, the use that fails or None)
PLANTED_KERNELS = [
    ("count, far in", _COUNT, _FAR, None),
    ("min", _MIN, "", None),
    ("no num_threads", "", "", "team None"),
    ("another team, far in", _OTHER, _FAR, "team scorch_nthreads(m, 1)"),
    ("an alias", _COUNT, "  auto& p = wksp_pool;\n", "unrecognized"),
    ("a pointer", _COUNT, "  auto* q = wksp_pool.data();\n", "unrecognized"),
]


@pytest.mark.parametrize(
    ("clause", "filler", "failing_use"),
    [case[1:] for case in PLANTED_KERNELS],
    ids=[case[0] for case in PLANTED_KERNELS],
)
def test_the_pool_check_reads_planted_kernels(clause, filler, failing_use):
    cpp = (
        _PLANTED_POOL
        + f"#pragma omp parallel for {clause}\n#pragma GCC ivdep\n"
        + "for (int64_t a = 0; a < n; a++) {\n"
        + filler
        + _PLANTED_INDEX
        + "}\n"
    )
    if failing_use is None:
        assert _assert_teams_within_pool(cpp) == 1
    else:
        assert ("wksp", failing_use) in _pool_uses(cpp)
        with pytest.raises(AssertionError, match=re.escape(failing_use)):
            _assert_teams_within_pool(cpp)


def test_the_pool_check_fails_an_index_outside_every_region():
    cpp = (
        _PLANTED_POOL
        + _PLANTED_INDEX
        + "#pragma omp parallel num_threads(wksp_thread_count)\n{\n"
        + _PLANTED_INDEX
        + "}\n"
    )
    assert ("wksp", "outside a region") in _pool_uses(cpp)
    with pytest.raises(AssertionError):
        _assert_teams_within_pool(cpp)


def _dense_owner(workspace: str) -> str:
    return (
        f"int {workspace}_thread_count = scorch_nthreads(n, 1);\n"
        f"auto {workspace}_pool_owner = scorch_make_aligned_buffer<double>("
        f"scorch_checked_size_product((size_t){workspace}_thread_count, (size_t)m));\n"
    )


def _dense_slice(workspace: str) -> str:
    return (
        f"  double* __restrict__ {workspace} = {workspace}_pool_owner.get() + "
        "(size_t)omp_get_thread_num() * (size_t)m;\n"
    )


_SERIAL_OWNER = "auto w_pool_owner = scorch_make_aligned_buffer<double>((size_t)m);\n"
_TWO_COUNTS = "num_threads(std::min(w_thread_count, v_thread_count))"
# (name, setup, the region's clause, the region's body, the use that fails or None)
PLANTED_DENSE_KERNELS = [
    ("count, far in", _dense_owner("w"), _COUNT.replace("wksp", "w"), _FAR, None),
    (
        "two pools, the smaller count",
        _dense_owner("w") + _dense_owner("v"),
        _TWO_COUNTS,
        _dense_slice("v"),
        None,
    ),
    ("the policy again", _dense_owner("w"), _OTHER, "", "team scorch_nthreads(m, 1)"),
    ("no num_threads", _dense_owner("w"), "", "", "team None"),
    ("an owner for one worker", _SERIAL_OWNER, _OTHER, "", "uncounted"),
    (
        "an alias",
        _dense_owner("w"),
        _COUNT.replace("wksp", "w"),
        "  auto& o = w_pool_owner;\n",
        "unrecognized",
    ),
]


@pytest.mark.parametrize(
    ("setup", "clause", "body", "failing_use"),
    [case[1:] for case in PLANTED_DENSE_KERNELS],
    ids=[case[0] for case in PLANTED_DENSE_KERNELS],
)
def test_the_pool_check_reads_planted_dense_workspaces(
    setup, clause, body, failing_use
):
    cpp = (
        setup
        + f"#pragma omp parallel for {clause}\n#pragma GCC ivdep\n"
        + "for (int64_t a = 0; a < n; a++) {\n"
        + body
        + _dense_slice("w")
        + "}\n"
    )
    if failing_use is None:
        assert _assert_teams_within_pool(cpp) == setup.count("_thread_count = ")
    else:
        assert ("w", failing_use) in _pool_uses(cpp)
        with pytest.raises(AssertionError, match=re.escape(failing_use)):
            _assert_teams_within_pool(cpp)


_ONE_WORKER = "auto w_pool_owner = scorch_make_aligned_buffer<double>((size_t)m);\n"
_SLICE_ZERO = "double* __restrict__ w = w_pool_owner.get();\n"


def _counted_loop(body: str, clause: str = "num_threads(w_thread_count)") -> str:
    return (
        f"#pragma omp parallel for {clause}\n"
        "for (int64_t a = 0; a < n; a++) {\n" + body + "}\n"
    )


# The serial placement (slice zero of an owner sized for one worker) and the shapes it
# must not let through: (name, kernel text, the use that fails or None).
PLANTED_SERIAL_PLACEMENTS = [
    (
        # The reduce program's kernel (dev2): both placements under one name.
        "the serial placement beside a pool of the same name",
        "{\n" + _ONE_WORKER + _SLICE_ZERO + "  w[0] = 1.0;\n}\n{\n" + _dense_owner("w")
        + _counted_loop(_dense_slice("w") + "  w[a] = 2.0;\n") + "}\n",
        None,
    ),
    (
        "slice zero of a counted pool, offset in the region",
        _dense_owner("w") + "double* q = w_pool_owner.get();\n"
        + _counted_loop("  double* w = q + omp_get_thread_num() * m;\n"),
        "slice zero of a counted pool",
    ),
    (
        "slice zero in a region",
        _ONE_WORKER + "#pragma omp parallel\n{\n" + _SLICE_ZERO + "}\n",
        "slice zero in a region",
    ),
    (
        "a one-worker slice used in a region",
        _ONE_WORKER + _SLICE_ZERO
        + _counted_loop("  w[a] = 1.0;\n", "num_threads(scorch_nthreads(n, 1))"),
        "a one-worker slice used in a region",
    ),
    (
        "a worker's slice outside every region",
        _dense_owner("w") + _dense_slice("w") + _counted_loop("  x();\n"),
        "outside a region",
    ),
    (
        "slice zero into an unrecognized target",
        _ONE_WORKER + "w_buffers[0] = w_pool_owner.get();\n",
        "slice zero, unrecognized target",
    ),
]


@pytest.mark.parametrize(
    ("cpp", "failing_use"),
    [case[1:] for case in PLANTED_SERIAL_PLACEMENTS],
    ids=[case[0] for case in PLANTED_SERIAL_PLACEMENTS],
)
def test_the_pool_check_reads_planted_serial_placements(cpp, failing_use):
    if failing_use is None:
        assert ("w", "slice zero") in _pool_uses(cpp)
        assert _assert_teams_within_pool(cpp) == 1
    else:
        assert ("w", failing_use) in _pool_uses(cpp)
        with pytest.raises(AssertionError, match=re.escape(failing_use)):
            _assert_teams_within_pool(cpp)


# The two programs of test_high_rank_programs' population whose kernels hold a dense
# per-worker workspace (dev1's census): each region that slices it takes its count, and
# the reduce program's kernel also places a second cluster serially (slice zero).
DENSE_WORKSPACE_PROGRAMS = [
    "batched: abij,abjk->abik dsds,ddds -> [dddd] 4x2x3x2x2",
    "reduce: ijkl->jl ddds -> [dd] 2x2x4x2",
]


def _population_program(name: str):
    (program,) = [p for p in high_rank._POPULATION if p.name == name]
    return program


@pytest.mark.parametrize("blocking", (False, True), ids=("plain", "blocked"))
@pytest.mark.parametrize("name", DENSE_WORKSPACE_PROGRAMS)
def test_a_dense_workspace_region_takes_its_team_from_the_pool(
    monkeypatch, blocking, name
):
    cpp = _emitted(monkeypatch, _population_program(name), blocking)
    owners = set(DENSE_OWNER.findall(cpp))
    assert owners, "the kernel holds a dense per-worker workspace"
    assert owners <= {workspace for workspace, _ in POOL_COUNT.findall(cpp)}, owners
    assert _assert_teams_within_pool(cpp) >= len(owners)


@pytest.mark.parametrize("blocking", (False, True), ids=("plain", "blocked"))
@pytest.mark.parametrize("name", DENSE_WORKSPACE_PROGRAMS)
def test_a_dense_workspace_program_matches_pytorch(blocking, name):
    outcome = high_rank._outcome(
        high_rank._run_shipped,
        _population_program(name),
        blocking,
        high_rank._SHIPPED_REFUSALS,
    )
    assert outcome == "correct", outcome


@pytest.mark.parametrize("blocking", (False, True), ids=("plain", "blocked"))
@pytest.mark.parametrize(
    ("number", "family", "spec", "deeper"), HIGH_RANK, ids=HIGH_RANK_IDS
)
def test_the_programs_that_read_past_the_end_read_clamped(
    monkeypatch, blocking, number, family, spec, deeper
):
    program = _high_rank_program(monkeypatch, number, family, spec)
    cpp = _emitted(monkeypatch, program, blocking)
    assert OLD_READ.search(cpp) is None, OLD_READ.search(cpp)
    tensor, level = deeper
    clamped = re.escape(_read(tensor, level))
    # The chunk is spelled from the estimate's text, and the team (the pool's count, or
    # the region's own policy where there is no pool) from its typed value.
    assert re.search(rf"scorch_chunk\([^;]*{clamped}", cpp), "scorch_chunk"
    typed = (
        rf"{tensor}{level}_pos\[std::min<int64_t>\({tensor}0_size, \(?"
        rf"{tensor}_mode_indices\[{level}\]\[0\]\.numel\(\) - 1\)?\)\]"
    )
    assert re.search(rf"scorch_nthreads\([^;]*{typed}", cpp), "scorch_nthreads"


@pytest.mark.parametrize("blocking", (False, True), ids=("plain", "blocked"))
@pytest.mark.parametrize(
    ("number", "family", "spec", "deeper"), HIGH_RANK, ids=HIGH_RANK_IDS
)
def test_no_team_is_larger_than_the_workspace_pool(
    monkeypatch, blocking, number, family, spec, deeper
):
    program = _high_rank_program(monkeypatch, number, family, spec)
    _assert_teams_within_pool(_emitted(monkeypatch, program, blocking))


@pytest.mark.parametrize("blocking", (False, True), ids=("plain", "blocked"))
def test_a_two_level_product_takes_its_team_from_the_pool(monkeypatch, blocking):
    cpp = emitted_einsum_cpp(
        monkeypatch,
        "ij,jk->ik",
        ("ds", "ds"),
        {"i": 64, "j": 48, "k": 40},
        format="ds",
        _compile_options=high_rank._options(blocking),
    )
    assert "A1_pos[A0_size]" in cpp and "_mode_indices[1][0].numel()" not in cpp
    assert _assert_teams_within_pool(cpp) == 1


@pytest.mark.parametrize("blocking", (False, True), ids=("plain", "blocked"))
@pytest.mark.parametrize(
    ("number", "family", "spec", "deeper"), HIGH_RANK, ids=HIGH_RANK_IDS
)
def test_the_programs_that_read_past_the_end_still_match_pytorch(
    monkeypatch, blocking, number, family, spec, deeper
):
    program = _high_rank_program(monkeypatch, number, family, spec)
    outcome = high_rank._outcome(
        high_rank._run_shipped, program, blocking, high_rank._SHIPPED_REFUSALS
    )
    assert outcome == "correct", outcome


def _gap_operands(rows: int):
    """82972's gap input for program 228, ``rows`` rows high: ``A`` (dssd) stores two
    (i, j) pairs, rows 1 and 2 one each, so ``A2_pos`` has 3 entries and the unclamped
    read ``A2_pos[rows]`` lies ``rows - 2`` entries past its end."""

    generator = torch.Generator().manual_seed(228)
    a = torch.zeros((rows, 2, 3, 2), dtype=torch.float64)
    a[1, 1] = torch.rand((3, 2), generator=generator, dtype=torch.float64) + 0.5
    a[2, 0] = torch.rand((3, 2), generator=generator, dtype=torch.float64) + 0.5
    a[1, 1, 0] = 0.0
    b = torch.rand((rows, 2, 3, 2), generator=generator, dtype=torch.float64) + 0.5
    b[0] = 0.0
    b[rows - 1] = 0.0
    return a, b


@pytest.mark.parametrize("rows", (4, 4096, 65536))
def test_the_gap_input_matches_pytorch(rows):
    a, b = _gap_operands(rows)
    positions = level_arrays(a, "dssd")[0][2][0]
    # The unclamped read A2_pos[rows] would lie past the array's last entry.
    assert positions.numel() == 3 and rows > positions.numel() - 1
    result = scorch.einsum(
        "ijkl,ijkl->il", stored(a, "dssd", "A"), stored(b, "dsdd", "B"), format="ds"
    )
    assert_matches(result, reference("ijkl,ijkl->il", [a, b]), torch.float64)
