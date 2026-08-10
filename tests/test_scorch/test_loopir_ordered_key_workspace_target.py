"""The rank-general ordered-key sparse-workspace vertical.

This is the semantic half of Phase-7 cluster 2: multi-compressed sparse
reductions and tensor-times-matrix, lowered through one workspace whose key
domain has any rank.  ``K == 1`` is the instance the migrated B1 / row-scope /
dense-row CSR families already are, so those keep their exact generated C++;
everything here is what the ordered key domain newly makes reachable.

Three properties are locked, and they are the three the inherited review said
were missing.

**Placement is derived, not tabulated.**  The region anchors at the outermost
reduction and keys on the result coordinates at or below that anchor, in
result level order.  ``prefix == 0`` -- the region owning the program root --
is the ordinary shape of a whole-tensor reduction, not a special case.

**Coordinates are not positions.**  A rank-K drain assembles K genuinely
nested levels: each key level above the leaf appends a coordinate only when
the drained key opens a new segment there, and each compressed level's
position array is written at its own parent's coordinate count.  Zipping the
coordinate arrays would pass a numeric check and still be wrong, so the locks
compare exact ``(pos, crd)`` storage against the format-neutral oracle.

**The legacy comparand is not a gate here.**  Every cell in this file is one
the legacy assembler either rejects, corrupts, or terminates on, so
correctness is gated on the LoopIR oracle plus a dense PyTorch reference.
Byte parity is asserted only where it is meaningful: that the migrated K == 1
families' sources did not move.
"""

from dataclasses import replace

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
from scorch.compiler.compile_options import CompileOptions
from scorch.compiler.loopir.levels import LevelTensorStorage
from scorch.compiler.loopir.lower_cin import LoopIRLoweringError
from scorch.compiler.loopir.lower_llir import LoopIRTargetError
from scorch.compiler.loopir.nodes import LevelKind
from scorch.compiler.loopir.oracle import run_program
from scorch.compiler.loopir.pipeline import (
    compile_cin_via_loopir,
    execute_cin_via_loopir,
)
from scorch.compiler.loopir.printer import canonical_program_dump
from scorch.compiler.loopir.schedule_passes import erase_schedule
from scorch.compiler.loopir.verifier import verify_program
from scorch.compiler.scheduler import Schedule
from scorch.stensor import STensor

_F32 = torch.float32


def auto_options(regblock_enabled, *, jit=False):
    base = (
        CompileOptions.from_environment()
        if jit
        else CompileOptions.from_environment(environ={})
    )
    return replace(
        base.with_regblock_enabled(regblock_enabled),
        requested_schedule=Schedule(),
    )


# -- programs ---------------------------------------------------------------


def reduction_cin(operand_fmt, result_fmt, operand_indices, result_indices, dtype=_F32):
    """``C[result] += A[operand]`` over one shared index nest."""

    ivars = {name: IndexVar(name) for name in operand_indices}
    operand = TensorVar("A", fmt=operand_fmt, dtype=dtype)[
        tuple(ivars[name] for name in operand_indices)
    ]
    result = TensorVar("C", fmt=result_fmt, dtype=dtype)[
        tuple(ivars[name] for name in result_indices)
    ]
    statement = TensorAssign(result, operand, op=Operation.ADD)
    for name in reversed(operand_indices):
        statement = ForAll(ivars[name], statement)
    return statement


def ttm_cin(a_fmt, b_fmt, c_fmt, dtype=_F32, *, commuted=False):
    """``C[i,j,l] += A[i,j,k] * B[k,l]`` -- tensor times matrix."""

    ivars = {name: IndexVar(name) for name in "ijkl"}
    a = TensorVar("A", fmt=a_fmt, dtype=dtype)[ivars["i"], ivars["j"], ivars["k"]]
    b = TensorVar("B", fmt=b_fmt, dtype=dtype)[ivars["k"], ivars["l"]]
    c = TensorVar("C", fmt=c_fmt, dtype=dtype)[ivars["i"], ivars["j"], ivars["l"]]
    value = (
        CINBinaryOp(Operation.MUL, b, a)
        if commuted
        else CINBinaryOp(Operation.MUL, a, b)
    )
    statement = TensorAssign(c, value, op=Operation.ADD)
    for name in reversed("ijkl"):
        statement = ForAll(ivars[name], statement)
    return statement


def sparse(dense, fmt, name):
    return STensor.from_torch(dense.clone(), name).to_sparse(fmt)


def random_dense(shape, seed, dtype=_F32, density=0.4):
    if 0 in shape:
        return torch.zeros(shape, dtype=dtype)
    generator = torch.Generator().manual_seed(seed)
    values = torch.rand(shape, generator=generator, dtype=torch.float64)
    mask = torch.rand(shape, generator=generator, dtype=torch.float64) < density
    return (values * mask).to(dtype)


def oracle_bindings(program, densities):
    """Bind each declared input to storage matching its own level kinds."""

    decls = {decl.symbol: decl for decl in program.tensors}
    bindings = {}
    for symbol, dense in zip(program.inputs, densities):
        decl = decls[symbol]
        kinds = tuple(level.kind for level in decl.levels)
        payload = dense.to(torch.float64).tolist()
        if all(kind is LevelKind.DENSE for kind in kinds):
            bindings[symbol] = payload
            continue
        bindings[symbol] = LevelTensorStorage.from_dense(
            payload,
            tuple(dense.shape),
            tuple(level.mode for level in decl.levels),
            kinds,
        )
    return bindings


def compiled_storage(result):
    """The public result's exact ``(pos, crd)`` per level plus its values."""

    levels = tuple(
        tuple(tuple(int(x) for x in part.tolist()) for part in mode)
        for mode in result.storage.index.mode_indices
    )
    return levels, tuple(float(v) for v in result.storage.value.tolist())


def oracle_level_storage(oracle):
    levels = []
    for level in range(len(oracle.level_kinds)):
        if oracle.positions[level] is None:
            levels.append(())
            continue
        levels.append(
            (
                tuple(int(x) for x in oracle.positions[level]),
                tuple(int(x) for x in oracle.coordinates[level]),
            )
        )
    return tuple(levels), tuple(float(v) for v in oracle.values)


def assert_ordered_hierarchy(levels, shape):
    """Storage well-formedness, read only from the stored arrays.

    Positions are parent links, not coordinates: a level's coordinate array is
    segmented by its child's position array, so this walks the links rather
    than zipping.
    """

    def walk(level, parent_position, prefix):
        if level == len(shape):
            entries.append(tuple(prefix))
            return
        if not levels[level]:
            for coordinate in range(shape[level]):
                walk(
                    level + 1,
                    parent_position * shape[level] + coordinate,
                    prefix + [coordinate],
                )
            return
        positions, coordinates = levels[level]
        assert positions[0] == 0
        assert list(positions) == sorted(positions)
        assert positions[-1] == len(coordinates)
        assert parent_position + 1 < len(positions)
        segment = coordinates[
            positions[parent_position] : positions[parent_position + 1]
        ]
        assert list(segment) == sorted(
            set(segment)
        ), "coordinates must strictly increase inside one parent segment"
        assert all(0 <= coordinate < shape[level] for coordinate in segment)
        for offset, coordinate in enumerate(segment):
            walk(level + 1, positions[parent_position] + offset, prefix + [coordinate])

    entries = []
    walk(0, 0, [])
    assert len(entries) == len(set(entries))
    return entries


# -- the migrated cells ------------------------------------------------------

# ``(name, operand format, result format, operand indices, result indices,
#   shape)``.  Each is a cell the inherited census recorded as fail-closed.
REDUCTION_CELLS = [
    ("ss ij->j", "ss", "s", "ij", "j", (4, 5)),
    ("ds ij->j", "ds", "s", "ij", "j", (4, 5)),
    ("sd ij->j", "sd", "s", "ij", "j", (4, 5)),
    ("sss ijk->k", "sss", "s", "ijk", "k", (3, 4, 5)),
    ("dss ijk->k", "dss", "s", "ijk", "k", (3, 4, 5)),
    ("sss ijk->ik", "sss", "ss", "ijk", "ik", (3, 4, 5)),
    ("sss ijk->jk", "sss", "ss", "ijk", "jk", (3, 4, 5)),
    ("dss ijk->jk", "dss", "ss", "ijk", "jk", (3, 4, 5)),
    ("ssss ijkl->l", "ssss", "s", "ijkl", "l", (2, 3, 4, 5)),
    ("ssss ijkl->kl", "ssss", "ss", "ijkl", "kl", (2, 3, 4, 5)),
    ("ssss ijkl->jkl", "ssss", "sss", "ijkl", "jkl", (2, 3, 4, 5)),
    ("ssss ijkl->il", "ssss", "ss", "ijkl", "il", (2, 3, 4, 5)),
    # The only shape where a bound prefix and a rank>1 key meet: the region
    # runs once per prefix cell AND assembles two nested key levels, so a
    # repeated outer key coordinate across two regions must not merge.
    ("ssss ijkl->ikl", "ssss", "sss", "ijkl", "ikl", (2, 3, 4, 5)),
    ("ssss ijkl->ijl", "ssss", "sss", "ijkl", "ijl", (2, 3, 4, 5)),
]

TTM_CELLS = [
    ("sss x dd", "sss", "dd", "sss"),
    ("sss x ss", "sss", "ss", "sss"),
    ("sss x ds", "sss", "ds", "sss"),
    ("sss x sd", "sss", "sd", "sss"),
    ("dss x dd", "dss", "dd", "dss"),
    ("dss x ss", "dss", "ss", "dss"),
]


@pytest.mark.parametrize("arm", [False, True])
@pytest.mark.parametrize(
    "cell", REDUCTION_CELLS, ids=[cell[0] for cell in REDUCTION_CELLS]
)
def test_reduction_cells_compile_in_both_arms(cell, arm):
    """Every migrated reduction cell now reaches a complete kernel."""

    _, operand_fmt, result_fmt, operand_indices, result_indices, shape = cell
    result_shape = tuple(shape[operand_indices.index(c)] for c in result_indices)
    kernel = compile_cin_via_loopir(
        reduction_cin(operand_fmt, result_fmt, operand_indices, result_indices),
        result_shape,
        ((shape, _F32),),
        compile_options=auto_options(arm),
    )
    source = kernel.cpp_source
    assert "wksp.sort();" in source
    # One ordered drain, and the assembly writes the deepest level's
    # coordinates rather than reusing a CSR-shaped shortcut.
    assert source.count("wksp.sort();") == 1
    leaf_level = len(result_indices) - 1
    assert f"C{leaf_level}_crd.emplace_back(" in source


@pytest.mark.parametrize("cell", TTM_CELLS, ids=[cell[0] for cell in TTM_CELLS])
@pytest.mark.parametrize("arm", [False, True])
def test_ttm_cells_compile_in_both_arms(cell, arm):
    _, a_fmt, b_fmt, c_fmt = cell
    kernel = compile_cin_via_loopir(
        ttm_cin(a_fmt, b_fmt, c_fmt),
        (4, 5, 3),
        (((4, 5, 6), _F32), ((6, 3), _F32)),
        compile_options=auto_options(arm),
    )
    assert "wksp.sort();" in kernel.cpp_source


def test_arms_generate_identical_sources():
    """The two automatic policy arms agree for every migrated cell."""

    for (
        _,
        operand_fmt,
        result_fmt,
        operand_indices,
        result_indices,
        shape,
    ) in REDUCTION_CELLS:
        result_shape = tuple(shape[operand_indices.index(c)] for c in result_indices)
        sources = {
            arm: compile_cin_via_loopir(
                reduction_cin(operand_fmt, result_fmt, operand_indices, result_indices),
                result_shape,
                ((shape, _F32),),
                compile_options=auto_options(arm),
            ).cpp_source
            for arm in (False, True)
        }
        assert sources[False] == sources[True], operand_fmt


# -- runtime spelling and assembly discipline --------------------------------


def test_rank_one_key_keeps_the_retained_runtime_spelling():
    """``K == 1`` is the same instance the migrated families already emit."""

    kernel = compile_cin_via_loopir(
        reduction_cin("ss", "s", "ij", "j"),
        (5,),
        (((4, 5), _F32),),
        compile_options=auto_options(False),
    )
    assert "coo_workspace_1d<float, 1>(1024)" in kernel.cpp_source
    assert "coo_workspace<" not in kernel.cpp_source
    assert "int64_t j = it.first;" in kernel.cpp_source


def test_rank_two_key_passes_key_domain_extents_not_the_result_shape():
    """The rank-K container is constructed over the KEY domain.

    Passing the whole result shape is precisely what made every insertion of
    a rank-2 key throw ``workspace coordinate rank mismatch``; the extents
    here are the two drained levels' own extents.
    """

    kernel = compile_cin_via_loopir(
        reduction_cin("sss", "ss", "ijk", "jk"),
        (4, 5),
        (((3, 4, 5), _F32),),
        compile_options=auto_options(False),
    )
    source = kernel.cpp_source
    assert "coo_workspace<float, 2>(1024, {result_shape[0], result_shape[1]})" in source
    assert "int64_t j = it.first[0];" in source
    assert "int64_t k = it.first[1];" in source


def test_rank_one_key_below_a_prefix_uses_the_prefix_level_extent():
    """A bound prefix shifts which result levels the key domain names."""

    kernel = compile_cin_via_loopir(
        ttm_cin("sss", "dd", "sss"),
        (4, 5, 3),
        (((4, 5, 6), _F32), ((6, 3), _F32)),
        compile_options=auto_options(False),
    )
    # K == 1 keeps the 1-D container, and the drained level is the trailing
    # one, so no result-shape argument appears at all.
    assert "coo_workspace_1d<float, 1>(1024)" in kernel.cpp_source
    assert "int64_t l = it.first;" in kernel.cpp_source


def test_multi_level_assembly_parent_links_every_compressed_segment():
    """Each child position array is written at its own parent's crd count."""

    kernel = compile_cin_via_loopir(
        reduction_cin("ssss", "sss", "ijkl", "jkl"),
        (3, 4, 5),
        (((2, 3, 4, 5), _F32),),
        compile_options=auto_options(False),
    )
    source = kernel.cpp_source
    assert "coo_workspace<float, 3>(1024, " in source
    assert "scorch_vector_set(C1_pos, C0_crd.size(), C1_crd.size());" in source
    assert "scorch_vector_set(C2_pos, C1_crd.size(), C2_crd.size());" in source
    assert "scorch_vector_set(C0_pos, C0_pos_index + 1, C0_crd.size());" in source
    # The cascade: a new outer segment forces new inner segments even when
    # the inner coordinate repeats.
    assert "bool wksp_opened = false;" in source
    assert source.count("wksp_opened = true;") == 2


def test_target_uses_no_layout_string_or_rendered_name_discovery():
    """Admission reads structure; nothing sniffs a format spelling.

    Checked from the parsed source rather than by substring search: every
    string literal the class evaluates is collected and none may be a level
    alias or a layout spelling, and no pattern-matching module may be used.
    """

    import ast
    import inspect
    import textwrap

    from scorch.compiler.loopir import lower_llir

    source = textwrap.dedent(
        inspect.getsource(lower_llir._OrderedKeySparseWorkspaceLowering)
    )
    tree = ast.parse(source)
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            doc = ast.get_docstring(node, clean=False)
            if doc is not None:
                docstrings.add(doc)

    aliases = {"d", "s", "c", "o", "dense", "compressed", "sparse", "singleton"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value in docstrings:
                continue
            text = node.value
            assert text.strip().lower() not in aliases, text
            assert not (
                text and len(text) <= 5 and set(text.lower()) <= {"d", "s", "c", "o"}
            ), f"layout-looking literal {text!r}"
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            assert node.value.id not in {"re", "fnmatch"}, node.value.id


# -- compiled public differentials -------------------------------------------


def execute_reduction(cell, arm, dtype, seed, shape_override=None):
    _, operand_fmt, result_fmt, operand_indices, result_indices, shape = cell
    shape = shape_override or shape
    dense = random_dense(shape, seed, dtype)
    result_shape = tuple(shape[operand_indices.index(c)] for c in result_indices)
    result, kernel = execute_cin_via_loopir(
        reduction_cin(operand_fmt, result_fmt, operand_indices, result_indices, dtype),
        result_shape,
        sparse(dense, operand_fmt, "A"),
        compile_options=auto_options(arm, jit=True),
    )
    return dense, result, kernel, result_shape


def assert_matches_oracle_and_pytorch(
    dense_operands, result, kernel, result_shape, reference
):
    scheduled = kernel.lowering.program
    base = erase_schedule(scheduled)
    verify_program(base)
    bindings = oracle_bindings(scheduled, dense_operands)
    shapes = {scheduled.outputs[0]: result_shape}
    scheduled_storage = run_program(scheduled, bindings, shapes)[scheduled.outputs[0]]
    base_storage = run_program(base, bindings, shapes)[base.outputs[0]]
    assert scheduled_storage == base_storage, "erasure changed the semantics"

    got_levels, got_values = compiled_storage(result)
    want_levels, want_values = oracle_level_storage(scheduled_storage)
    assert got_levels == want_levels, "compiled storage diverges from the oracle"
    assert len(got_values) == len(want_values)
    for got, want in zip(got_values, want_values):
        assert got == pytest.approx(want, abs=1e-4, rel=1e-4)

    assert_ordered_hierarchy(got_levels, tuple(result_shape))
    dense_result = result.to_torch(in_place=False).to(torch.float64)
    assert torch.allclose(dense_result, reference, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize(
    "cell", REDUCTION_CELLS, ids=[cell[0] for cell in REDUCTION_CELLS]
)
def test_reduction_storage_matches_oracle_and_pytorch(cell):
    """float32 over every migrated reduction cell."""

    _, operand_fmt, _, operand_indices, result_indices, shape = cell
    dense, result, kernel, result_shape = execute_reduction(
        cell, False, torch.float32, 4242 + len(operand_indices)
    )
    axes = tuple(
        position
        for position, name in enumerate(operand_indices)
        if name not in result_indices
    )
    reference = dense.to(torch.float64).sum(dim=axes)
    assert_matches_oracle_and_pytorch((dense,), result, kernel, result_shape, reference)


@pytest.mark.parametrize(
    "cell",
    [
        cell
        for cell in REDUCTION_CELLS
        if cell[0]
        in {
            "sss ijk->jk",
            "ssss ijkl->jkl",
            "ssss ijkl->ikl",
            "sss ijk->ik",
            "ss ij->j",
        }
    ],
    ids=lambda cell: cell[0],
)
@pytest.mark.parametrize("arm", [False, True])
def test_reduction_storage_matches_in_float64_and_both_arms(cell, arm):
    """float64 and the second automatic arm over the structural shapes.

    Restricted to the cells whose ``(prefix, key rank)`` splits differ, which
    is what the dtype and arm axes can actually interact with; the float32
    sweep above covers every cell.
    """

    _, operand_fmt, _, operand_indices, result_indices, shape = cell
    dense, result, kernel, result_shape = execute_reduction(
        cell, arm, torch.float64, 4242 + len(operand_indices)
    )
    axes = tuple(
        position
        for position, name in enumerate(operand_indices)
        if name not in result_indices
    )
    reference = dense.to(torch.float64).sum(dim=axes)
    assert_matches_oracle_and_pytorch((dense,), result, kernel, result_shape, reference)


@pytest.mark.parametrize("cell", TTM_CELLS, ids=[cell[0] for cell in TTM_CELLS])
@pytest.mark.parametrize("commuted", [False, True])
def test_ttm_storage_matches_oracle_and_pytorch(cell, commuted):
    _, a_fmt, b_fmt, c_fmt = cell
    a_shape, b_shape = (4, 5, 6), (6, 3)
    a_dense = random_dense(a_shape, 8181)
    b_dense = random_dense(b_shape, 9191, density=0.7)
    result_shape = (4, 5, 3)
    operands = (
        (sparse(b_dense, b_fmt, "B"), sparse(a_dense, a_fmt, "A"))
        if commuted
        else (sparse(a_dense, a_fmt, "A"), sparse(b_dense, b_fmt, "B"))
    )
    dense_operands = (b_dense, a_dense) if commuted else (a_dense, b_dense)
    result, kernel = execute_cin_via_loopir(
        ttm_cin(a_fmt, b_fmt, c_fmt, commuted=commuted),
        result_shape,
        *operands,
        compile_options=auto_options(False, jit=True),
    )
    reference = torch.einsum(
        "ijk,kl->ijl", a_dense.to(torch.float64), b_dense.to(torch.float64)
    )
    assert_matches_oracle_and_pytorch(
        dense_operands, result, kernel, result_shape, reference
    )


@pytest.mark.parametrize(
    "shape",
    [(1, 1, 1), (3, 1, 5), (0, 4, 5), (3, 4, 0), (2, 3, 4)],
    ids=["singleton", "ragged", "empty-outer", "zero-trailing", "small"],
)
def test_rank_two_key_handles_degenerate_extents(shape):
    cell = ("sss ijk->jk", "sss", "ss", "ijk", "jk", shape)
    dense, result, kernel, result_shape = execute_reduction(cell, False, _F32, 1717)
    reference = dense.to(torch.float64).sum(dim=(0,))
    assert_matches_oracle_and_pytorch((dense,), result, kernel, result_shape, reference)


def test_repeated_outer_key_across_regions_stays_two_segments():
    """A bound prefix plus a rank-2 key: segments must not merge across cells.

    ``ssss ijkl->ikl`` runs one region per ``i`` and assembles two nested key
    levels inside it.  The operand here is built so that EVERY prefix cell
    drains the same single ``k``, which is the exact input that a
    ``crd.back() != k`` test without a per-region base would silently fold
    into one segment -- producing storage that still passes a numeric check
    while losing a whole level of structure.
    """

    shape = (3, 2, 4, 5)
    dense = torch.zeros(shape, dtype=_F32)
    for i in range(shape[0]):
        for j in range(shape[1]):
            # Same k for every i, so consecutive regions open with equal
            # outer key coordinates.
            dense[i, j, 2, (i + j) % shape[3]] = float(i + 1) * (j + 2)
    result_shape = (3, 4, 5)
    result, kernel = execute_cin_via_loopir(
        reduction_cin("ssss", "sss", "ijkl", "ikl"),
        result_shape,
        sparse(dense, "ssss", "A"),
        compile_options=auto_options(False, jit=True),
    )
    levels, _ = compiled_storage(result)
    # One i segment per prefix cell, each with its own single k coordinate.
    assert levels[0][1] == (0, 1, 2)
    assert levels[1][0] == (0, 1, 2, 3), "each region must open a new k segment"
    assert levels[1][1] == (2, 2, 2)
    reference = dense.to(torch.float64).sum(dim=(1,))
    assert_matches_oracle_and_pytorch((dense,), result, kernel, result_shape, reference)


def test_explicitly_stored_operand_zeros_reach_the_result():
    """An operand zero that is STORED, not absent, still contributes a key.

    Built through ``from_coo`` so the zero survives into the operand's stored
    values instead of being filtered out by densify/re-sparsify.  Its key must
    therefore appear in the result even though its contribution is 0.0 --
    which is the sparse-structure contract, not a numeric one.
    """

    indices = torch.tensor(
        [
            [0, 0, 1, 1],
            [0, 1, 0, 1],
            [2, 3, 2, 4],
        ]
    )
    values = torch.tensor([1.5, 0.0, 0.0, 2.5], dtype=_F32)
    shape = (2, 2, 5)
    operand = STensor.from_coo(
        indices=indices, values=values, shape=shape, name="A"
    ).to_sparse("sss")
    stored = operand.values.tolist()
    assert 0.0 in stored, "the fixture must retain an explicitly stored zero"

    result, kernel = execute_cin_via_loopir(
        reduction_cin("sss", "ss", "ijk", "jk"),
        (2, 5),
        operand,
        compile_options=auto_options(False, jit=True),
    )
    levels, drained = compiled_storage(result)
    # Four stored operand entries over four distinct (j, k) keys.
    assert levels[1][1] == (2, 3, 2, 4)
    assert 0.0 in drained, "a stored zero must keep its coordinate"

    dense = torch.zeros(shape, dtype=torch.float64)
    for position in range(indices.shape[1]):
        dense[tuple(int(x) for x in indices[:, position])] = float(values[position])
    assert_matches_oracle_and_pytorch(
        (dense.to(_F32),), result, kernel, (2, 5), dense.sum(dim=(0,))
    )


def test_cancellation_keeps_the_stored_entry():
    """Exactly cancelling contributions still occupy their key."""

    shape = (4, 3, 4)
    dense = random_dense(shape, 3131, density=1.0)
    dense[1] = -dense[0]
    dense[2:] = 0.0
    result, kernel = execute_cin_via_loopir(
        reduction_cin("sss", "ss", "ijk", "jk"),
        (3, 4),
        sparse(dense, "sss", "A"),
        compile_options=auto_options(False, jit=True),
    )
    _, values = compiled_storage(result)
    assert values, "a cancelled key must remain stored, not vanish"
    assert all(abs(value) < 1e-6 for value in values)
    reference = dense.to(torch.float64).sum(dim=(0,))
    assert_matches_oracle_and_pytorch((dense,), result, kernel, (3, 4), reference)


# -- erasure and representation ---------------------------------------------


@pytest.mark.parametrize(
    "cell", REDUCTION_CELLS, ids=[cell[0] for cell in REDUCTION_CELLS]
)
def test_scheduled_program_erases_to_the_semantic_source(cell):
    _, operand_fmt, result_fmt, operand_indices, result_indices, shape = cell
    result_shape = tuple(shape[operand_indices.index(c)] for c in result_indices)
    kernel = compile_cin_via_loopir(
        reduction_cin(operand_fmt, result_fmt, operand_indices, result_indices),
        result_shape,
        ((shape, _F32),),
        compile_options=auto_options(False),
    )
    scheduled = kernel.lowering.program
    erased = erase_schedule(scheduled)
    verify_program(erased)
    assert "sparse_workspace" not in canonical_program_dump(erased)
    assert erase_schedule(erased) is erased


def test_canonical_dump_is_arm_stable():
    dumps = set()
    for arm in (False, True):
        kernel = compile_cin_via_loopir(
            reduction_cin("sss", "ss", "ijk", "jk"),
            (4, 5),
            (((3, 4, 5), _F32),),
            compile_options=auto_options(arm),
        )
        dumps.add(canonical_program_dump(kernel.lowering.program))
    assert len(dumps) == 1


# -- fail-closed neighbours (recorded seam occupants) ------------------------


@pytest.mark.parametrize("arm", [False, True])
@pytest.mark.parametrize(
    ("operand_fmt", "result_fmt", "operand_indices", "result_indices", "code"),
    [
        # A dense result level bound by a STORED prefix loop would need the
        # row-scope catch-up against a dynamic parent count; that neighbour
        # is not migrated here.
        ("sss", "ds", "ijk", "ik", "unsupported_sparse_output_domain"),
        ("sss", "ds", "ijk", "jk", "unsupported_sparse_output_domain"),
        # A stored result prefix level driven by a DENSE domain likewise has
        # no coordinate stream to append.
        ("dds", "ss", "ijk", "ik", "unsupported_sparse_output_domain"),
        # A compressed-parent/dense-leaf result keeps its own reduction seam.
        ("sss", "sd", "ijk", "ik", "unsupported_sparse_output_reduction"),
    ],
    ids=["dense-row-ik", "dense-row-jk", "dense-driven-prefix", "dense-leaf"],
)
def test_unmigrated_neighbours_keep_precise_domain_codes(
    operand_fmt, result_fmt, operand_indices, result_indices, code, arm
):
    shape = (3, 4, 5)
    result_shape = tuple(shape[operand_indices.index(c)] for c in result_indices)
    with pytest.raises(LoopIRLoweringError) as error:
        compile_cin_via_loopir(
            reduction_cin(operand_fmt, result_fmt, operand_indices, result_indices),
            result_shape,
            ((shape, _F32),),
            compile_options=auto_options(arm),
        )
    assert error.value.defect.code == code


def test_row_scope_dense_prefix_is_rejected_by_the_target():
    """A dense result level under a stored loop stops at the target.

    Reached through an explicit legal order, so the rejection is a program
    boundary rather than the automatic origin's; the arm axis does not apply
    to an explicitly scheduled compile.
    """

    options = CompileOptions.from_environment(
        environ={}, requested_schedule=Schedule(loop_order=("i", "j", "k"))
    )
    with pytest.raises((LoopIRTargetError, LoopIRLoweringError)) as error:
        compile_cin_via_loopir(
            reduction_cin("sss", "ds", "ijk", "ik"),
            (3, 5),
            (((3, 4, 5), _F32),),
            compile_options=options,
        )
    assert error.value.defect.code in (
        "unsupported_program_shape",
        "unsupported_sparse_output_domain",
    )
