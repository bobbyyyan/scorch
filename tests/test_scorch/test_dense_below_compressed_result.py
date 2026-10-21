"""Sparse results whose levels are not "dense levels, then compressed levels".

A result's format gives each storage level a kind, and two rules fix what the
level arrays must hold:

* a dense level of extent ``E`` below a parent with ``P`` positions has
  ``P * E`` positions, and coordinate ``c`` under parent position ``q`` sits at
  position ``q * E + c``;
* a compressed level below ``P`` positions has ``P + 1`` position entries.

So a compressed coordinate ABOVE a dense level owns a block of ``E`` child
positions (``sd``, ``ssd``, ``dsd``, ``sdd``), a compressed level below such a
block owns one segment per cell of it (``sds``), and a compressed level below
two dense levels owns one segment per cell of the flattened pair (``dds``).

The shipped lowering was written for dense levels above compressed ones.  For
every other arrangement it returned malformed arrays or raised:

* values under a dense level were appended once per visited cell, so the cells
  a stored stream skipped got no slot;
* a compressed level recorded its coordinate only when a compressed child
  gained entries, so above a dense child it recorded nothing;
* a compressed level below a dense one was numbered by the dense coordinate
  instead of the dense position;
* a serial kernel's dense workspace was never declared, because only a
  parallel loop placed it;
* a position array built by appending stopped at the last visited parent
  cell, so it came back short when the cells after it stored nothing;
* the parallel two-pass assembly of compressed rows numbered the rows by the
  outermost loop's variable, which for a loop over an operand's stored rows is
  a position among them: every row after the first one the operand does not
  store was assembled one row early, and the position array had one entry per
  stored row.

Each cell below runs through the public ``scorch.einsum`` and is judged from
the result's own arrays, read here rather than through the library:

``canonical``  every compressed level has ``parents + 1`` positions that start
               at 0 and never decrease, coordinates strictly increasing and in
               range inside each segment, and one value per leaf position;
``values``     the arrays, made dense here, equal a float64 ``torch.einsum``;
``structure``  the stored coordinates are the ones the operands' STORED
               structure implies: a compressed result coordinate is stored
               exactly when a stored operand cell lies under it, whatever the
               value there.  An explicit stored zero stays stored.

The format's letters describe storage levels.  ``scorch.einsum`` stores a
result's levels in its loop order when the caller names none and records that
order as the tensor's ``mode_order``, so the arrays are read in that order and
the values compared after the same permutation.
"""

import pytest
import torch

import scorch
from scorch.compiler.cin import (
    BinaryOp as CINBinaryOp,
    ForAll,
    IndexVar,
    Operation,
    TensorAssign,
    TensorVar,
)
from scorch.compiler.cin_analysis import normalize_cin
from scorch.compiler.cin_lowerer import CINLowerer
from scorch.compiler.compilation_context import CompilationContext
from scorch.compiler.compile_options import CompileOptions
from scorch.compiler.loopir.pipeline import (
    _bind_runtime_metadata,
    legacy_generated_cpp,
)
from scorch.compiler.scheduler import Scheduler
from scorch.ops import _lower_generated_llir
from scorch.stensor import STensor

ATOL = RTOL = 1e-3
EXTENT = {"i": 5, "j": 4, "k": 3, "l": 6}

# One kernel serves every input of a cell, so each cell costs one build.
#   ragged       slice 1 along each operand's first index stores nothing
#   ends         the first and the last slice store nothing
#   stored_zero  ragged, with one stored cell whose value is exactly 0.0
#   empty        nothing is stored anywhere
#   zero_extent  the first operand's last index has extent 0
#   hollow       ragged, and each compressed level with a compressed level
#                below it also stores coordinates that have nothing under
#                them: their child segments are empty.  Such a coordinate
#                stores no entry, so it implies no result coordinate.
INPUTS = ("ragged", "ends", "stored_zero", "empty", "zero_extent", "hollow")


def _stored(pattern, fmt, ghosts=None):
    """Leaf cells and level arrays of a tensor that stores ``pattern`` in ``fmt``.

    ``ghosts`` is a second pattern whose cells contribute coordinates but no
    entries: every compressed level that has a compressed level below it also
    stores the coordinate each ghost cell has there, and the lowest compressed
    level stores none, so the segments under a ghost coordinate are empty.
    """

    cells = {tuple(int(t) for t in index) for index in pattern.nonzero()}
    hollow = (
        set()
        if ghosts is None
        else {tuple(int(t) for t in index) for index in ghosts.nonzero()}
    )
    lowest_compressed = max(
        (level for level, kind in enumerate(fmt) if kind != "d"), default=-1
    )
    parents, levels = [()], []
    for level, kind in enumerate(fmt):
        if kind == "d":
            levels.append(None)
            parents = [p + (c,) for p in parents for c in range(pattern.shape[level])]
            continue
        positions, coordinates, children = [0], [], []
        named = cells if level == lowest_compressed else cells | hollow
        for parent in parents:
            stored = sorted({c[level] for c in named if c[:level] == parent})
            coordinates += stored
            children += [parent + (c,) for c in stored]
            positions.append(len(coordinates))
        levels.append((positions, coordinates))
        parents = children
    return parents, levels


def _operand(dense, pattern, fmt, name, ghosts=None):
    """The operand in ``fmt`` storing ``pattern``, and its stored-cell indicator.

    Built level by level with ``from_components``, never through ``to_sparse``:
    a conversion into one of these layouts runs the kernels under test.
    """

    leaves, levels = _stored(pattern, fmt, ghosts)
    if (
        ghosts is not None
        and fmt.count("s") >= 2
        and dense.shape[0] > 1
        and dense.numel()
    ):
        # The input's premise, checked: the operand stores more coordinates
        # than its entries account for.
        minimal = _stored(pattern, fmt)[1]
        assert sum(len(level[1]) for level in levels if level) > sum(
            len(level[1]) for level in minimal if level
        ), f"{name} in {fmt}: the hollow input has no empty segment"
    mode_indices = [
        (
            []
            if level is None
            else [
                torch.tensor(level[0], dtype=torch.int64),
                torch.tensor(level[1], dtype=torch.int64),
            ]
        )
        for level in levels
    ]
    values = torch.tensor([float(dense[c]) for c in leaves], dtype=dense.dtype)
    indicator = torch.zeros(dense.shape, dtype=torch.float64)
    for cell in leaves:
        indicator[cell] = 1.0
    tensor = STensor.from_components(
        tuple(dense.shape), fmt, mode_indices, values, name=name
    )
    return tensor, indicator


def _inputs(kind, terms, seed, extent=None, dtype=torch.float32):
    """One ``(dense, stored pattern, ghost pattern or None)`` per operand.

    ``extent`` maps each index letter to its extent (``EXTENT`` by default);
    ``dtype`` is the operands' scalar type.
    """

    extent = dict(EXTENT if extent is None else extent)
    if kind == "zero_extent":
        extent[terms[0][-1]] = 0
    pairs = []
    for position, term in enumerate(terms):
        shape = tuple(extent[c] for c in term)
        generator = torch.Generator().manual_seed(1000 * seed + position)
        values = torch.rand(shape, generator=generator, dtype=torch.float64) + 0.5
        pattern = torch.rand(shape, generator=generator) < 0.55
        if kind == "empty":
            pattern = torch.zeros_like(pattern)
        elif pattern.numel():
            if kind == "ends":
                pattern[0] = False
                pattern[-1] = False
            elif shape[0] > 1:
                pattern[1] = False
        dense = (values * pattern).to(dtype)
        if kind == "stored_zero" and bool(pattern.any()):
            dense[tuple(int(t) for t in pattern.nonzero()[0])] = 0.0
        ghosts = None
        if kind == "hollow":
            ghosts = (torch.rand(shape, generator=generator) < 0.5) & ~pattern
            if shape[0] > 1 and pattern.numel():
                # The slice the pattern leaves empty is all ghost, so a
                # coordinate with nothing under it exists wherever the layout
                # can hold one.
                ghosts[1] = True
        pairs.append((dense, pattern, ghosts))
    return pairs


def _read(result, fmt, shape):
    """Leaf cells and level arrays from the result's own arrays, checked."""

    mode_indices = result.storage.index.mode_indices
    assert len(mode_indices) == len(fmt)
    parents, levels = [()], []
    for level, kind in enumerate(fmt):
        if kind == "d":
            assert len(mode_indices[level]) == 0, f"level {level} is dense"
            levels.append(None)
            parents = [p + (c,) for p in parents for c in range(shape[level])]
            continue
        assert len(mode_indices[level]) == 2, f"level {level} is compressed"
        positions, coordinates = (
            array.reshape(-1).tolist() for array in mode_indices[level]
        )
        assert len(positions) == len(parents) + 1, (
            f"level {level} holds {len(positions)} positions for "
            f"{len(parents)} parents"
        )
        assert positions[0] == 0 and positions == sorted(
            positions
        ), f"level {level} positions {positions}"
        assert positions[-1] == len(coordinates), (
            f"level {level} ends at {positions[-1]} with "
            f"{len(coordinates)} coordinates"
        )
        children = []
        for number, parent in enumerate(parents):
            segment = coordinates[positions[number] : positions[number + 1]]
            assert segment == sorted(
                set(segment)
            ), f"level {level} segment {number} is {segment}"
            assert all(
                0 <= c < shape[level] for c in segment
            ), f"level {level} segment {number} is {segment}, extent {shape[level]}"
            children += [parent + (c,) for c in segment]
        levels.append((positions, coordinates))
        parents = children
    assert result.storage.value.numel() == len(
        parents
    ), f"{result.storage.value.numel()} values for {len(parents)} leaf positions"
    return parents, levels


def _judge(result, result_fmt, reference, structural):
    """Canonical arrays, float64 values, and the stored-structure rule."""

    assert str(result.format) == ",".join(result_fmt)
    order = tuple(int(mode) for mode in result.mode_order)
    assert sorted(order) == list(range(reference.dim()))
    shape = tuple(reference.shape[mode] for mode in order)
    leaves, levels = _read(result, result_fmt, shape)
    got = torch.zeros(shape, dtype=torch.float64)
    for cell, value in zip(leaves, result.storage.value.reshape(-1).tolist()):
        got[cell] = value
    torch.testing.assert_close(got, reference.permute(order), atol=ATOL, rtol=RTOL)
    expected = _stored(structural.permute(order), result_fmt)[1]
    assert levels == expected, f"stored {levels}, the operands imply {expected}"


def _run(expr, operand_fmts, result_fmt, kind, seed):
    terms = expr.split("->")[0].split(",")
    operands, indicators, denses = [], [], []
    pairs = _inputs(kind, terms, seed)
    for (dense, pattern, ghosts), fmt, name in zip(pairs, operand_fmts, "ABD"):
        operand, indicator = _operand(dense, pattern, fmt, name, ghosts)
        operands.append(operand)
        indicators.append(indicator)
        denses.append(dense.to(torch.float64))
    result = scorch.einsum(expr, *operands, format=result_fmt)
    _judge(
        result,
        result_fmt,
        torch.einsum(expr, *denses),
        torch.einsum(expr, *indicators) > 0,
    )


def _cells():
    cells = []
    for operand in ("dd", "ds", "sd", "ss"):
        cells.append(("ij->ij", (operand,), "sd"))
    for operand in ("ds", "ss"):
        cells.append(("ij->ji", (operand,), "sd"))
    for result in ("ssd", "dsd", "sdd", "sds", "dds"):
        for operand in ("sss", "dss", "dds", "ddd"):
            cells.append(("ijk->ijk", (operand,), result))
    for expr in ("ijk->ik", "ijk->ij", "ijk->jk"):
        for operand in ("sss", "dss", "dds"):
            cells.append((expr, (operand,), "sd"))
    for result in ("ssd", "dsd", "sds", "dds"):
        for tensor in ("sss", "dds"):
            for matrix in ("dd", "ds"):
                cells.append(("ijk,kl->ijl", (tensor, matrix), result))
    for result in ("sd", "ds"):
        cells.append(("ij,ij->ij", ("ss", "ss"), result))
    for result in ("dsd", "sds", "dds", "dss"):
        cells.append(("ijk,ijk->ijk", ("sss", "sss"), result))
    for result in ("ssds", "sdss", "sdsd", "dsds", "ssdd", "ddss"):
        cells.append(("ijkl->ijkl", ("ssss",), result))
    # Compressed rows below a dense level zero, accumulated per row by a loop
    # over an operand's STORED rows.  Every row that loop skips still owns a
    # segment, wherever it is: the middle, the first row, or the last.
    for expr, operands in (
        ("ij,jk->ik", ("ss", "dd")),
        ("ij,jk->ik", ("ss", "ss")),
        ("ijk->ik", ("sss",)),
        ("ijk->ij", ("sss",)),
    ):
        cells.append((expr, operands, "ds"))
    for matrix in ("dd", "ss"):
        cells.append(("ijk,kl->ijl", ("sss", matrix), "dss"))
    # A matrix product into compressed rows of dense columns: the dense row
    # workspace is sized from the result's column extent when no all-dense
    # operand carries the column index (it raised IndexError before).
    for matrix in ("ss", "dd"):
        cells.append(("ij,jk->ik", ("ss", matrix), "sd"))
    # The operand stored in the result's own layout: a copy, and a product of
    # two such operands, whose dense levels both operands store in full and
    # whose compressed levels are intersected.
    for layout in ("sd", "ssd", "dsd", "sdd", "sds"):
        indices = "ijk"[: len(layout)]
        if layout != "sd":
            cells.append((f"{indices}->{indices}", (layout,), layout))
        cells.append((f"{indices},{indices}->{indices}", (layout, layout), layout))
    return cells


CELLS = _cells()


def _cell_id(cell):
    expr, operand_fmts, result_fmt = cell
    return f"{','.join(operand_fmts)} {expr} as {result_fmt}"


# Cells the shipped compiler still gets wrong, each with what it does instead.
# Strict: a cell that starts passing must leave this table.
KNOWN_DEFECTS = {}


def _parameters():
    parameters = []
    for seed, cell in enumerate(CELLS):
        reason = KNOWN_DEFECTS.get(_cell_id(cell))
        marks = [pytest.mark.xfail(strict=True, reason=reason)] if reason else []
        parameters.append(pytest.param(cell, seed, id=_cell_id(cell), marks=marks))
    return parameters


@pytest.mark.parametrize("cell,seed", _parameters())
def test_result_is_canonical_exact_and_structural(cell, seed):
    expr, operand_fmts, result_fmt = cell
    failures = []
    for kind in INPUTS:
        try:
            _run(expr, operand_fmts, result_fmt, kind, seed)
        except Exception as error:  # every input is reported, not the first
            failures.append(f"{kind}: {type(error).__name__}: {str(error)[:400]}")
    assert not failures, "\n".join(failures)


@pytest.mark.parametrize("fmt", ["sd", "ssd", "dsd", "sdd", "sds", "dds"])
def test_to_sparse_from_a_dense_tensor_reaches_these_layouts(fmt):
    """``to_sparse`` runs the same copy kernels; it stores the nonzero cells."""

    failures = []
    for kind in ("ragged", "ends", "empty"):
        dense, _, _ = _inputs(kind, ["ijk"[: len(fmt)]], seed=len(fmt))[0]
        try:
            tensor = STensor.from_torch(dense.clone(), "A")
            tensor.to_sparse(fmt)
            _judge(tensor, fmt, dense.to(torch.float64), dense != 0)
        except Exception as error:  # every input is reported, not the first
            failures.append(f"{kind}: {type(error).__name__}: {str(error)[:400]}")
    assert not failures, "\n".join(failures)


# -- the mechanisms, read from the emitted source (no kernel is built) --------


def _options():
    return CompileOptions.from_environment(environ={}).with_regblock_enabled(False)


def _product_source(result_fmt, operand_fmts):
    """Shipped C++ for ``C = A * B`` over one shared index nest."""

    rank = len(result_fmt)
    ivars = tuple(IndexVar(name) for name in "ijkl"[:rank])
    rhs = None
    for name, fmt in zip("AB", operand_fmts):
        access = TensorVar(name, fmt=fmt, dtype=torch.float32)[ivars]
        rhs = access if rhs is None else CINBinaryOp(Operation.MUL, rhs, access)
    statement = TensorAssign(
        TensorVar("C", fmt=result_fmt, dtype=torch.float32)[ivars], rhs
    )
    for index_var in reversed(ivars):
        statement = ForAll(index_var, statement)
    shape = tuple(EXTENT[name] for name in "ijkl"[:rank])
    return legacy_generated_cpp(
        statement,
        shape,
        tuple((shape, torch.float32) for _ in operand_fmts),
        compile_options=_options(),
    )


def _top_level_position_loops(source, level):
    """The ``for (; C<level>_pos_index < ...`` headers at function depth."""

    return [
        line
        for line in source.splitlines()
        if line.startswith(f"  for (; C{level}_pos_index < ")
    ]


@pytest.mark.parametrize(
    "result_fmt,operand_fmts,level,bound",
    [
        ("ds", ("ss", "ss"), 1, "C0_size"),
        ("dss", ("sss", "sss"), 1, "C0_size"),
        ("dds", ("sss", "sss"), 2, "C0_size * C1_size"),
    ],
)
def test_stored_driven_dense_levels_close_the_positions_after_the_loop(
    result_fmt, operand_fmts, level, bound
):
    """A loop over stored coordinates stops at the last one; the close does not.

    The compressed level below the dense levels owns one segment per dense
    cell.  A loop that visits only the cells the operands store never reaches
    the cells after the last stored one, so their position entries are written
    after the loop, up to the full cell count.
    """

    source = _product_source(result_fmt, operand_fmts)
    assert _top_level_position_loops(source, level) == [
        f"  for (; C{level}_pos_index < {bound}; C{level}_pos_index++) {{"
    ]


@pytest.mark.parametrize(
    "result_fmt,operand_fmts,level",
    [("ds", ("ds", "ds"), 1), ("dss", ("dss", "dss"), 1), ("dds", ("dds", "dds"), 2)],
)
def test_dense_driven_dense_levels_need_no_close_after_the_loop(
    result_fmt, operand_fmts, level
):
    """A loop over the whole dense range visits every cell; nothing is left."""

    source = _product_source(result_fmt, operand_fmts)
    assert _top_level_position_loops(source, level) == []


def test_canonical_csr_positions_stay_as_they_were():
    """``ds`` below one dense level keeps its emission byte for byte.

    The fix changes how a position array is built only where the old layout
    was wrong; the copy of a compressed-row matrix is the control.
    """

    source = _product_source("ds", ("ds",))
    assert "C1_pos[i + 1]" in source or "C1_pos_index + 1" in source
    assert _top_level_position_loops(source, 1) == []
    assert "Commit the compressed coordinate" not in source


@pytest.mark.parametrize(
    "result_fmt,operand_fmts",
    [("sd", ("ss",)), ("ssd", ("sss",)), ("dsd", ("sss",)), ("sds", ("sss",))],
)
def test_a_compressed_coordinate_above_a_dense_block_is_committed_once(
    result_fmt, operand_fmts
):
    """The coordinate is recorded after its block, exactly when it was written."""

    source = _product_source(result_fmt, operand_fmts)
    level = result_fmt.index("sd")
    assert source.count("// Commit the compressed coordinate above") == 1
    assert source.count(f"C{level}_crd.push_back(") == 1


def _scheduled_source(statement, result_shape, operand_shapes):
    """Shipped C++ after the automatic schedule, which places the workspace."""

    options = _options()
    context = CompilationContext(options)
    normalized = normalize_cin(statement, compile_options=options)
    _bind_runtime_metadata(
        normalized,
        tuple((shape, torch.float32) for shape in operand_shapes),
        result_shape,
    )
    scheduled = Scheduler.auto_schedule(
        normalized, compile_options=options, compilation_context=context
    )
    lowered = CINLowerer(
        compile_options=options, compilation_context=context
    )._lower_owned_IndexStmt(scheduled)
    return _lower_generated_llir(lowered, options, context)


def _matrix_product_source(a_fmt):
    """``C[i,k] += A[i,j] * B[j,k]`` into compressed rows, ``B`` dense."""

    i, j, k = (IndexVar(name) for name in "ijk")
    statement = TensorAssign(
        TensorVar("C", fmt="ds", dtype=torch.float32)[i, k],
        CINBinaryOp(
            Operation.MUL,
            TensorVar("A", fmt=a_fmt, dtype=torch.float32)[i, j],
            TensorVar("B", fmt="dd", dtype=torch.float32)[j, k],
        ),
        op=Operation.ADD,
    )
    for index_var in (k, j, i):
        statement = ForAll(index_var, statement)
    return _scheduled_source(statement, (5, 3), ((5, 4), (4, 3)))


def test_two_pass_rows_are_numbered_by_coordinate_under_a_stored_outer_loop():
    """A position among an operand's stored rows is not a row of the result.

    With ``A`` doubly compressed the outermost loop runs over the rows ``A``
    stores.  The two-pass assembly keeps one count and one offset per row of
    the RESULT, so both are indexed by the coordinate at the loop's position
    and sized by the result's extent, and the position array has one entry per
    result row plus one.
    """

    source = _matrix_product_source("ss")
    assert "for (int pA0 = A0_pos[0]; pA0 < pA0_end; pA0++) {" in source
    assert "std::vector<int> _count1((size_t)C0_size, 0);" in source
    assert "_count1[A0_crd[pA0]] = _cnt1;" in source
    assert "std::vector<int64_t> _offset1((size_t)C0_size + 1);" in source
    assert "int64_t _base1 = _offset1[A0_crd[pA0]];" in source
    assert (
        "torch::Tensor C1_pos_torch = torch::empty({(int64_t)(C0_size + 1)}, "
        "torch::kInt);" in source
    )
    # The loop's own variable and bound, which is what the defect used.
    assert "_count1[pA0]" not in source
    assert "_offset1[pA0]" not in source
    assert "_count1((size_t)pA0_end" not in source


def test_two_pass_rows_keep_the_loop_variable_under_a_dense_outer_loop():
    """The control: a loop over the dense range already numbers the rows.

    With ``A`` in compressed rows the outermost loop variable is the row and
    its bound is the row count, so this kernel is emitted as it always was.
    """

    source = _matrix_product_source("ds")
    assert "for (int64_t i = 0; i < A0_size; i++) {" in source
    assert "std::vector<int> _count1((size_t)A0_size, 0);" in source
    assert "_count1[i] = _cnt1;" in source
    assert "int64_t _base1 = _offset1[i];" in source
    assert "_crd[" not in "".join(
        line for line in source.splitlines() if "_count1[" in line
    )
