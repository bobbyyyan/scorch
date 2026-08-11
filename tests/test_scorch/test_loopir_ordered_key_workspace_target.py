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

**Legacy evidence is cell-specific, and every cell has some.**  The LoopIR
oracle plus a dense PyTorch reference gate all twenty migrated cells.  The
inherited claim that the legacy assembler "rejects, corrupts or terminates on"
every one of them is wrong: the legacy generator emits C++ for all twenty in
both automatic arms, and for **nine** of them that C++ is sound -- the
``ss``/``ds``/``sd`` rank-one reductions, ``sss ijk->ik``, ``sss ijk->jk``,
``dss ijk->jk``, ``ssss ijkl->jkl``, and the two ``dss`` tensor-times-matrix
cases.  Those nine additionally require exact runtime and sparse-storage
equality, in both arms, at f32 and f64, and under cancellation.  The remaining
**eleven** are unsound in exactly three measured ways -- duplicate drained
coordinates, C++ that does not compile, and a malformed child position array
-- and are characterized as such rather than waved away.  Every legacy source
differs from ours: the nine locks are semantic parity, never byte parity.
"""

from copy import deepcopy
from dataclasses import replace

import pytest
import torch

from scorch.compiler import llir
from scorch.compiler.cin import (
    BinaryOp as CINBinaryOp,
    ForAll,
    IndexVar,
    Operation,
    TensorAssign,
    TensorVar,
)
from scorch.compiler.compile_options import CompileOptions
from scorch.compiler.llir_traversal import (
    LLIRRewriter,
    LLIRStatementSequence,
    LLIRTraversalContext,
)
from scorch.compiler.loopir.levels import (
    CompressedLevel,
    LevelTensorStorage,
)
from scorch.compiler.loopir.lower_cin import LoopIRLoweringError
from scorch.compiler.loopir.lower_llir import LoopIRTargetError
from scorch.compiler.loopir.nodes import LevelKind
from scorch.compiler.loopir.oracle import run_program
from scorch.compiler.loopir.pipeline import (
    compile_cin_via_loopir,
    execute_cin_via_loopir,
    legacy_generated_cpp,
)
from scorch.compiler.loopir.printer import canonical_program_dump
from scorch.compiler.loopir.schedule_passes import erase_schedule
from scorch.compiler.loopir.verifier import verify_program
from scorch.compiler.scheduler import Schedule
from scorch.storage import TensorIndex
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


# -- completion integrity ---------------------------------------------------


def owned_objects(root):
    """Every distinct object the produced body owns, read without hooks.

    Reads go through ``type`` and ``object.__getattribute__`` so a forged
    ``__class__`` or container protocol on managed state cannot steer the
    walk the tamper helpers do.
    """

    found = []
    seen = set()
    stack = [root]
    while stack:
        value = stack.pop()
        if id(value) in seen:
            continue
        kind = type(value)
        if kind is list or kind is tuple:
            seen.add(id(value))
            found.append(value)
            stack.extend(value)
            continue
        if isinstance(value, llir.Node):
            seen.add(id(value))
            found.append(value)
            stack.extend(object.__getattribute__(value, "__dict__").values())
    return found


def equal_owned_pair(root, kind):
    """Two distinct owned objects of ``kind`` with identical stored state."""

    candidates = [value for value in owned_objects(root) if type(value) is kind]
    for index, first in enumerate(candidates):
        for second in candidates[index + 1 :]:
            if kind is tuple:
                if first and first == second:
                    return first, second
                continue
            if isinstance(first, llir.Node) and object.__getattribute__(
                first, "__dict__"
            ) == object.__getattribute__(second, "__dict__"):
                return first, second
    raise AssertionError(f"no equal owned {kind.__name__} pair to alias")


def replace_object(root, target, replacement):
    """Install ``replacement`` wherever the produced body stored ``target``."""

    replaced = 0
    for owner in owned_objects(root):
        if type(owner) is list:
            for index, value in enumerate(owner):
                if value is target:
                    owner[index] = replacement
                    replaced += 1
            continue
        if not isinstance(owner, llir.Node):
            continue
        state = object.__getattribute__(owner, "__dict__")
        for field_name in tuple(state):
            value = state[field_name]
            if value is target:
                state[field_name] = replacement
                replaced += 1
            elif type(value) is tuple and any(item is target for item in value):
                state[field_name] = tuple(
                    replacement if item is target else item for item in value
                )
                replaced += 1
    assert replaced, "the alias tamper found nowhere to install its replacement"
    return replaced


def final_metadata(root):
    """Every ``TensorAccessMetadata`` the produced body still carries."""

    carried = []
    for owner in owned_objects(root):
        if not isinstance(owner, llir.Node):
            continue
        state = object.__getattribute__(owner, "__dict__")
        metadata = state.get("tensor_access")
        if type(metadata) is llir.TensorAccessMetadata:
            carried.append((owner, metadata))
    return carried


def install_hostile_pass(monkeypatch, tamper):
    """Run the real dynamic-vector pass, then let ``tamper`` corrupt its result."""

    import scorch.compiler.llir_pass_manager as pass_manager

    original = pass_manager.rewrite_dynamic_vector_accesses
    state = {"changed": False}

    def hostile(value, context):
        result = original(value, context)
        state["changed"] = bool(tamper(result))
        return result

    monkeypatch.setattr(
        pass_manager,
        "rewrite_dynamic_vector_accesses",
        hostile,
    )
    return state


def compile_ordered_key_probe(arm=False):
    """One rank-2-key ordered workspace compile, the completion probe cell."""

    return compile_cin_via_loopir(
        reduction_cin("sss", "ss", "ijk", "jk"),
        (4, 5),
        (((3, 4, 5), _F32),),
        compile_options=auto_options(arm),
    )


class RewriteInsert(LLIRRewriter):
    """Rewrite, drop, duplicate or wrap the one workspace insertion."""

    def __init__(self, tamper):
        super().__init__(
            LLIRTraversalContext(
                stage="test",
                pass_name="tamper_ordered_key_workspace_insert",
            )
        )
        self.tamper = tamper
        self.changed = False
        self.removed = None

    def rewrite_statement_sequence_member(self, node, path):
        if (
            self.changed
            or type(node) is not llir.FunctionCallStmt
            or node.name != "wksp.insert"
        ):
            return (node,)
        self.changed = True
        if self.tamper in {"drop_insert", "relocate_insert"}:
            self.removed = node
            return ()
        if self.tamper == "duplicate_insert":
            return (node, deepcopy(node))
        if self.tamper == "mutate_insert_value":
            return (
                llir.FunctionCallStmt(
                    name=node.name,
                    args=(
                        *node.args[:-1],
                        llir.Literal(0.0, llir.DataType.FLOAT32),
                    ),
                    template_args=node.template_args,
                ),
            )
        if self.tamper == "mutate_insert_key":
            # The leading arguments are the inserted key coordinates; a
            # pass that rewrote one of them would file every entry under
            # the wrong key and still produce well-formed storage.
            assert len(node.args) >= 2
            return (
                llir.FunctionCallStmt(
                    name=node.name,
                    args=(
                        llir.Literal(0, llir.DataType.INT),
                        *node.args[1:],
                    ),
                    template_args=node.template_args,
                ),
            )
        if self.tamper == "mutate_insert_callee":
            return (
                llir.FunctionCallStmt(
                    name="wksp.insert_entry",
                    args=node.args,
                    template_args=node.template_args,
                ),
            )
        if self.tamper == "wrap_insert":
            return (
                llir.IfThenElse(
                    cond=llir.Literal(True),
                    then_body=[node],
                ),
            )
        if self.tamper == "wrap_insert_in_function":
            return (
                llir.Function(
                    return_type=llir.DataType.VOID,
                    name="nested_workspace_insert",
                    args=[],
                    body=[node],
                ),
            )
        if self.tamper == "member_clear":
            return (
                node,
                llir.MemberCallStmt(
                    base=llir.Var("wksp", llir.DataType.NO_TYPE),
                    member="clear",
                ),
            )
        return (node,)


class RelocateInsert(LLIRRewriter):
    def __init__(self, insertion):
        super().__init__(
            LLIRTraversalContext(
                stage="test",
                pass_name="relocate_ordered_key_workspace_insert",
            )
        )
        self.insertion = insertion
        self.changed = False

    def prepare_statement_sequence(
        self,
        statements: LLIRStatementSequence,
        path,
    ):
        prepared = list(statements)
        if self.changed:
            return prepared
        for index, statement in enumerate(prepared):
            if (
                type(statement) is llir.FunctionCallStmt
                and statement.name == "wksp.sort"
            ):
                prepared.insert(index, self.insertion)
                self.changed = True
                break
        return prepared


class RelocateInsertBeforeDeclaration(LLIRRewriter):
    def __init__(self):
        super().__init__(
            LLIRTraversalContext(
                stage="test",
                pass_name="move_ordered_key_insert_before_declaration",
            )
        )
        self.changed = False

    def prepare_statement_sequence(
        self,
        statements: LLIRStatementSequence,
        path,
    ):
        prepared = list(statements)
        if self.changed:
            return prepared
        for index, statement in enumerate(prepared):
            if (
                index > 0
                and type(statement) is llir.FunctionCallStmt
                and statement.name == "wksp.insert"
            ):
                prepared.insert(0, prepared.pop(index))
                self.changed = True
                break
        return prepared


class RelocateDrain(LLIRRewriter):
    """Move the ordered drain, either up to the allocation or past the return."""

    def __init__(self, tamper):
        super().__init__(
            LLIRTraversalContext(
                stage="test",
                pass_name="relocate_ordered_key_workspace_drain",
            )
        )
        self.tamper = tamper
        self.changed = False

    def prepare_statement_sequence(
        self,
        statements: LLIRStatementSequence,
        path,
    ):
        prepared = list(statements)
        if self.changed:
            return prepared
        allocation = next(
            (
                index
                for index, statement in enumerate(prepared)
                if type(statement) is llir.VarInit
                and type(statement.var) is llir.Var
                and statement.var.name == "wksp"
            ),
            None,
        )
        drain = next(
            (
                index
                for index, statement in enumerate(prepared)
                if type(statement) is llir.ForLoopAuto
                and type(statement.array) is llir.Var
                and statement.array.name == "wksp"
            ),
            None,
        )
        if allocation is not None and drain is not None:
            statement = prepared.pop(drain)
            if self.tamper == "drain_after_return":
                return_index = next(
                    (
                        index
                        for index, candidate in enumerate(prepared)
                        if type(candidate) is llir.Return
                    ),
                    None,
                )
                if return_index is None:
                    return prepared
                prepared.insert(return_index + 1, statement)
            else:
                prepared.insert(allocation + 1, statement)
            self.changed = True
        return prepared


class WrapRegion(LLIRRewriter):
    def __init__(self):
        super().__init__(
            LLIRTraversalContext(
                stage="test",
                pass_name="wrap_ordered_key_workspace_region",
            )
        )
        self.changed = False

    def prepare_statement_sequence(
        self,
        statements: LLIRStatementSequence,
        path,
    ):
        prepared = list(statements)
        if self.changed:
            return prepared
        allocation = next(
            (
                index
                for index, statement in enumerate(prepared)
                if type(statement) is llir.VarInit
                and type(statement.var) is llir.Var
                and statement.var.name == "wksp"
            ),
            None,
        )
        drain = next(
            (
                index
                for index, statement in enumerate(prepared)
                if type(statement) is llir.ForLoopAuto
                and type(statement.array) is llir.Var
                and statement.array.name == "wksp"
            ),
            None,
        )
        if allocation is not None and drain is not None and allocation < drain:
            region = prepared[allocation : drain + 1]
            prepared[allocation : drain + 1] = [
                llir.IfThenElse(
                    cond=llir.Literal(False),
                    then_body=region,
                )
            ]
            self.changed = True
        return prepared


class RelocateRegion(LLIRRewriter):
    """Move the whole allocation-to-drain region, keeping every effect."""

    def __init__(self):
        super().__init__(
            LLIRTraversalContext(
                stage="test",
                pass_name="relocate_ordered_key_workspace_region",
            )
        )
        self.changed = False

    def prepare_statement_sequence(
        self,
        statements: LLIRStatementSequence,
        path,
    ):
        prepared = list(statements)
        if self.changed:
            return prepared
        allocation = next(
            (
                index
                for index, statement in enumerate(prepared)
                if type(statement) is llir.VarInit
                and type(statement.var) is llir.Var
                and statement.var.name == "wksp"
            ),
            None,
        )
        drain = next(
            (
                index
                for index, statement in enumerate(prepared)
                if type(statement) is llir.ForLoopAuto
                and type(statement.array) is llir.Var
                and statement.array.name == "wksp"
            ),
            None,
        )
        if allocation is None or drain is None or allocation >= drain:
            return prepared
        region = prepared[allocation : drain + 1]
        del prepared[allocation : drain + 1]
        prepared[0:0] = region
        self.changed = prepared != list(statements)
        return prepared


class RewriteSurroundingBody(LLIRRewriter):
    def __init__(self):
        super().__init__(
            LLIRTraversalContext(
                stage="test",
                pass_name="tamper_ordered_key_surrounding_body",
            )
        )
        self.changed = False

    def prepare_statement_sequence(
        self,
        statements: LLIRStatementSequence,
        path,
    ):
        prepared = list(statements)
        if self.changed:
            return prepared
        values_index = next(
            (
                index
                for index, statement in enumerate(prepared)
                if type(statement) is llir.VarDecl
                and type(statement.var) is llir.Var
                and statement.var.name == "C_values"
            ),
            None,
        )
        position_index = next(
            (
                index
                for index, statement in enumerate(prepared)
                if type(statement) is llir.FunctionCallStmt
                and statement.name == "scorch_vector_set"
                and statement.args
                and type(statement.args[0]) is llir.Var
                and statement.args[0].name == "C0_pos"
            ),
            None,
        )
        if values_index is not None and position_index is not None:
            prepared[values_index], prepared[position_index] = (
                prepared[position_index],
                prepared[values_index],
            )
            self.changed = True
        return prepared


def apply_workspace_effect_tamper(tamper, rewritten):
    """Apply one named tamper to a produced body; report whether it landed."""

    if tamper in {"relocate_drain", "drain_after_return"}:
        rewriter = RelocateDrain(tamper)
        return rewriter.rewrite(rewritten), rewriter.changed
    if tamper == "wrap_region_false":
        rewriter = WrapRegion()
        return rewriter.rewrite(rewritten), rewriter.changed
    if tamper == "relocate_region":
        rewriter = RelocateRegion()
        return rewriter.rewrite(rewritten), rewriter.changed
    if tamper == "insert_before_declaration":
        rewriter = RelocateInsertBeforeDeclaration()
        return rewriter.rewrite(rewritten), rewriter.changed
    if tamper == "swap_surrounding_dependencies":
        rewriter = RewriteSurroundingBody()
        return rewriter.rewrite(rewritten), rewriter.changed
    if tamper in {
        "share_blank_line",
        "share_equal_variable",
        "share_equal_template_args",
    }:
        # Ownership-only tampers: nothing the emitter would print changes,
        # only how many distinct objects the produced body owns.
        return share_one_owned_object(tamper, rewritten), True
    rewriter = RewriteInsert(tamper)
    result = rewriter.rewrite(rewritten)
    assert rewriter.changed
    if tamper != "relocate_insert":
        return result, True
    assert rewriter.removed is not None
    relocator = RelocateInsert(rewriter.removed)
    return relocator.rewrite(result), relocator.changed


def share_one_owned_object(tamper, rewritten):
    """Collapse one equal-but-distinct pair in the body into a single object."""

    if tamper == "share_blank_line":
        blanks = [
            index
            for index, statement in enumerate(rewritten)
            if type(statement) is llir.BlankLine
        ]
        assert len(blanks) >= 2
        rewritten[blanks[1]] = rewritten[blanks[0]]
        return rewritten
    kind = llir.Var if tamper == "share_equal_variable" else tuple
    first, second = equal_owned_pair(rewritten, kind)
    replace_object(rewritten, second, first)
    return rewritten


@pytest.mark.parametrize(
    "tamper",
    [
        "drop_insert",
        "duplicate_insert",
        "mutate_insert_value",
        "mutate_insert_key",
        "mutate_insert_callee",
        "wrap_insert",
        "wrap_insert_in_function",
        "member_clear",
        "insert_before_declaration",
        "relocate_insert",
        "relocate_drain",
        "drain_after_return",
        "wrap_region_false",
        "relocate_region",
        "swap_surrounding_dependencies",
        "share_blank_line",
        "share_equal_variable",
        "share_equal_template_args",
    ],
)
def test_ordered_key_completion_rejects_workspace_effect_tampering(
    monkeypatch,
    tamper,
):
    """Managed passes cannot change or relocate any workspace-owned effect."""

    import scorch.compiler.llir_pass_manager as pass_manager

    original = pass_manager.rewrite_dynamic_vector_accesses
    state = {"changed": False}

    def hostile_rewrite(value, context):
        result, changed = apply_workspace_effect_tamper(
            tamper, original(value, context)
        )
        state["changed"] = changed
        return result

    monkeypatch.setattr(
        pass_manager,
        "rewrite_dynamic_vector_accesses",
        hostile_rewrite,
    )
    with pytest.raises(LoopIRTargetError) as error:
        compile_ordered_key_probe()
    assert state["changed"]
    assert error.value.defect.code == "sparse_workspace_completion_lost"


def test_ordered_key_completion_rejects_legacy_atomic_ancestor_mutation(
    monkeypatch,
):
    """Every codegen-active field of an enclosing loop is checkpointed."""

    import scorch.compiler.llir_pass_manager as pass_manager

    original = pass_manager.rewrite_dynamic_vector_accesses

    class MutateAncestor(LLIRRewriter):
        def __init__(self):
            super().__init__(
                LLIRTraversalContext(
                    stage="test",
                    pass_name="mutate_ordered_key_atomic_ancestor",
                )
            )
            self.changed = False

        def rewrite_for_loop(self, node, path):
            rewritten = super().rewrite_for_loop(node, path)
            if not self.changed:
                rewritten._use_atomic_scheduling = True
                rewritten._atomic_chunk_var = "_chunk"
                rewritten._atomic_counter_var = "_next_row"
                rewritten._loop_bound = "A0_size"
                self.changed = True
            return rewritten

    state = {"changed": False}

    def hostile_rewrite(value, context):
        rewriter = MutateAncestor()
        result = rewriter.rewrite(original(value, context))
        state["changed"] = rewriter.changed
        return result

    monkeypatch.setattr(
        pass_manager,
        "rewrite_dynamic_vector_accesses",
        hostile_rewrite,
    )
    with pytest.raises(LoopIRTargetError) as error:
        compile_cin_via_loopir(
            reduction_cin("ssss", "sss", "ijkl", "ikl"),
            (2, 4, 5),
            (((2, 3, 4, 5), _F32),),
            compile_options=auto_options(False),
        )
    assert state["changed"]
    assert error.value.defect.code == "sparse_workspace_completion_lost"


def test_ordered_key_completion_rejects_enclosing_openmp_mutation(monkeypatch):
    """An enclosing loop cannot acquire a parallel discipline after the fact."""

    import scorch.compiler.llir_pass_manager as pass_manager

    original = pass_manager.rewrite_dynamic_vector_accesses

    class ParallelizeAncestor(LLIRRewriter):
        def __init__(self):
            super().__init__(
                LLIRTraversalContext(
                    stage="test",
                    pass_name="parallelize_ordered_key_ancestor",
                )
            )
            self.changed = False

        def rewrite_for_loop(self, node, path):
            rewritten = super().rewrite_for_loop(node, path)
            if not self.changed:
                rewritten.omp_parallel_for = True
                rewritten.omp_schedule = "dynamic"
                rewritten.omp_num_threads = 4
                self.changed = True
            return rewritten

    state = {"changed": False}

    def hostile_rewrite(value, context):
        rewriter = ParallelizeAncestor()
        result = rewriter.rewrite(original(value, context))
        state["changed"] = rewriter.changed
        return result

    monkeypatch.setattr(
        pass_manager,
        "rewrite_dynamic_vector_accesses",
        hostile_rewrite,
    )
    with pytest.raises(LoopIRTargetError) as error:
        compile_ordered_key_probe()
    assert state["changed"]
    assert error.value.defect.code == "sparse_workspace_completion_lost"


@pytest.mark.parametrize(
    "field", ["role", "access_id", "tensor_id", "index_ids", "whole_object"]
)
def test_ordered_key_completion_rejects_final_metadata_mutation(monkeypatch, field):
    """Access provenance is checkpointed by value, not by shared reference.

    The shared LLIR rewriter deliberately carries ``TensorAccessMetadata``
    across by reference.  Before the checkpoint was built by a detaching
    rewriter of the target's own, the reference body and the pipeline body
    therefore pointed at one frozen object, and a ``__dict__`` write on it
    moved both sides at once -- the comparison accepted its own corruption.
    """

    from scorch.compiler.identity import AccessId, IndexId, SymbolId

    identities = {"access_id": AccessId, "tensor_id": SymbolId}

    def tamper(statements):
        carried = final_metadata(statements)
        assert carried, "the probe cell carries no access provenance to attack"
        owner, metadata = carried[0]
        stored = object.__getattribute__(metadata, "__dict__")
        if field == "whole_object":
            # Swap in a distinct object whose stored value genuinely differs;
            # an aliased *equal* value is legal and is locked separately.
            object.__getattribute__(owner, "__dict__")["tensor_access"] = (
                llir.TensorAccessMetadata(
                    access_id=AccessId(stored["access_id"].value + 1),
                    tensor_id=stored["tensor_id"],
                    index_ids=stored["index_ids"],
                    role=stored["role"],
                )
            )
            return True
        if field == "role":
            stored["role"] = (
                llir.TensorAccessRole.RESULT_WRITE
                if stored["role"] is llir.TensorAccessRole.INPUT_READ
                else llir.TensorAccessRole.INPUT_READ
            )
            return True
        if field == "index_ids":
            stored["index_ids"] = (*stored["index_ids"], IndexId(4242))
            return True
        stored[field] = identities[field](stored[field].value + 1)
        return True

    state = install_hostile_pass(monkeypatch, tamper)
    with pytest.raises(LoopIRTargetError) as error:
        compile_ordered_key_probe()
    assert state["changed"]
    assert error.value.defect.code == "sparse_workspace_completion_lost"


def test_ordered_key_completion_accepts_value_equal_metadata_substitution(monkeypatch):
    """Provenance is value state: an equal copy, or an alias, is not a defect.

    This is the positive control for the lock above.  The production two-phase
    rewrite legitimately duplicates a work body whose detached statements keep
    the same provenance *values*, so the boundary must reject a changed value
    without rejecting a re-owned or shared equal one.
    """

    def tamper(statements):
        carried = final_metadata(statements)
        assert carried
        owner, metadata = carried[0]
        stored = object.__getattribute__(metadata, "__dict__")
        replacement = llir.TensorAccessMetadata(
            access_id=stored["access_id"],
            tensor_id=stored["tensor_id"],
            index_ids=tuple(stored["index_ids"]),
            role=stored["role"],
        )
        assert replacement is not metadata
        for other_owner, _ in carried:
            object.__getattribute__(other_owner, "__dict__")[
                "tensor_access"
            ] = replacement
        return True

    state = install_hostile_pass(monkeypatch, tamper)
    kernel = compile_ordered_key_probe()
    assert state["changed"]
    assert "coo_workspace<float, 2>" in kernel.cpp_source


@pytest.mark.parametrize("tamper_name", ["cycle", "missing_field", "extra_field"])
def test_ordered_key_completion_rejects_forged_node_state(monkeypatch, tamper_name):
    """A cyclic or field-forged body is refused, and refused in finite time."""

    def tamper(statements):
        for owner in owned_objects(statements):
            if type(owner) is not llir.ForLoop:
                continue
            state = object.__getattribute__(owner, "__dict__")
            if tamper_name == "cycle":
                # The loop's body becomes the whole function body, so the
                # comparison meets one list twice.  It must reject, not spin.
                state["body"] = statements
            elif tamper_name == "missing_field":
                del state["simd"]
            else:
                state["_forged_field"] = True
            return True
        return False

    state = install_hostile_pass(monkeypatch, tamper)
    with pytest.raises(LoopIRTargetError) as error:
        compile_ordered_key_probe()
    assert state["changed"]
    assert error.value.defect.code == "sparse_workspace_completion_lost"


def test_ordered_key_completion_rejects_enum_singleton_mutation(monkeypatch):
    """The one class of state the checkpoint shares is independently pinned.

    Enum members are process-wide singletons, so the detached checkpoint must
    share them with the pipeline body -- the comparison requires identity for
    them.  Their *stored* state is therefore pinned against an import-time
    snapshot instead, which is what makes that sharing safe.
    """

    member = llir.TensorAccessRole.INPUT_READ
    stored = object.__getattribute__(member, "__dict__")
    original_value = stored["_value_"]

    def tamper(statements):
        assert final_metadata(statements)
        stored["_value_"] = "forged_role"
        return True

    state = install_hostile_pass(monkeypatch, tamper)
    try:
        with pytest.raises(LoopIRTargetError) as error:
            compile_ordered_key_probe()
    finally:
        stored["_value_"] = original_value
    assert stored["_value_"] == original_value
    assert state["changed"]
    assert error.value.defect.code == "sparse_workspace_completion_lost"


def test_exact_completion_comparator_rejects_shared_statement_lists():
    """List ownership is censused even where a real body cannot alias one.

    Every list in this family's emitted body is a statement body, and no two
    of them are equal, so a *structure-preserving* shared-list tamper is not
    constructible end to end -- an end-to-end share would be caught as a
    structural difference and would prove nothing about ownership.  The census
    is therefore locked directly, on a pair that differs in ownership alone.
    """

    from scorch.compiler.loopir.lower_llir import _exact_sparse_completion_matches

    def leaf():
        return [llir.Comment("shared"), llir.BlankLine()]

    shared = leaf()
    actual = [
        llir.IfThenElse(cond=llir.Literal(True), then_body=shared),
        llir.IfThenElse(cond=llir.Literal(True), then_body=shared),
    ]
    expected = [
        llir.IfThenElse(cond=llir.Literal(True), then_body=leaf()),
        llir.IfThenElse(cond=llir.Literal(True), then_body=leaf()),
    ]
    detached = [
        llir.IfThenElse(cond=llir.Literal(True), then_body=leaf()),
        llir.IfThenElse(cond=llir.Literal(True), then_body=leaf()),
    ]
    assert _exact_sparse_completion_matches(detached, expected)
    assert not _exact_sparse_completion_matches(actual, expected)


def test_ordered_key_checkpoint_never_runs_foreign_object_hooks(monkeypatch):
    """Malformed managed state is rejected without invoking its callbacks."""

    import scorch.compiler.llir_pass_manager as pass_manager

    original = pass_manager.rewrite_dynamic_vector_accesses
    calls = []

    class Hostile:
        @property
        def __class__(self):
            calls.append("__class__")
            raise RuntimeError("foreign class hook ran")

        def __eq__(self, other):
            calls.append("__eq__")
            raise RuntimeError("foreign equality hook ran")

        def __hash__(self):
            calls.append("__hash__")
            raise RuntimeError("foreign hash hook ran")

        def __reduce_ex__(self, protocol):
            calls.append(("__reduce_ex__", protocol))
            raise RuntimeError("foreign pickle hook ran")

    class ForgeComment(LLIRRewriter):
        def __init__(self):
            super().__init__(
                LLIRTraversalContext(
                    stage="test",
                    pass_name="forge_ordered_key_comment",
                )
            )
            self.changed = False

        def rewrite_comment(self, node, path):
            if self.changed:
                return super().rewrite_comment(node, path)
            self.changed = True
            return llir.Comment(Hostile())

    state = {"changed": False}

    def hostile_rewrite(value, context):
        rewriter = ForgeComment()
        result = rewriter.rewrite(original(value, context))
        state["changed"] = rewriter.changed
        return result

    monkeypatch.setattr(
        pass_manager,
        "rewrite_dynamic_vector_accesses",
        hostile_rewrite,
    )
    with pytest.raises(LoopIRTargetError) as error:
        compile_ordered_key_probe()
    assert state["changed"]
    assert calls == []
    assert error.value.defect.code == "sparse_workspace_completion_lost"


def test_ordered_key_checkpoint_shares_no_forgeable_state_with_the_pipeline():
    """The reference and the managed graphs overlap only in pinned state.

    Every object the checkpoint reaches is compared against everything the
    pipeline's *input* body and its *final* body reach.  The intersection may
    contain only values a managed pass cannot forge: ``None``, the immutable
    scalars, the interned empty tuple, and the LLIR enum singletons -- whose
    stored state ``_exact_sparse_completion_matches`` pins separately (locked
    above).  No node, no list, no non-empty tuple, no access provenance and no
    provenance identity may be shared.
    """

    from enum import Enum

    from scorch.compiler.identity import AccessId, IndexId, SymbolId
    from scorch.compiler.loopir import lower_llir

    captured = {}

    def reachable(root):
        """Every object reachable from ``root``, scalars included."""

        found = {}
        stack = [root]
        seen = set()
        while stack:
            value = stack.pop()
            if id(value) in seen:
                continue
            seen.add(id(value))
            found[id(value)] = value
            kind = type(value)
            if kind is list or kind is tuple:
                stack.extend(value)
                continue
            if kind is llir.TensorAccessMetadata:
                state = object.__getattribute__(value, "__dict__")
                stack.extend(state.values())
                continue
            if kind in (AccessId, SymbolId, IndexId):
                stack.extend(object.__getattribute__(value, "__dict__").values())
                continue
            if isinstance(value, llir.Node):
                stack.extend(object.__getattribute__(value, "__dict__").values())
        return found

    real_checkpoint = lower_llir._ordered_key_expected_checkpoint
    real_require = lower_llir._require_ordered_key_completion_checkpoint

    def capture_checkpoint(lowering, kernel_abi, body):
        expected = real_checkpoint(lowering, kernel_abi, body)
        if expected is not None:
            captured["input"] = reachable(body)
            # The reference now carries the required ABI signature beside the
            # body, so the residual-sharing proof binds on both.
            captured["expected"] = reachable(
                [expected.body, expected.return_type, expected.name, expected.args]
            )
        return expected

    def capture_require(actual, expected):
        if expected is not None:
            captured["final"] = reachable(actual)
        return real_require(actual, expected)

    lower_llir._ordered_key_expected_checkpoint = capture_checkpoint
    lower_llir._require_ordered_key_completion_checkpoint = capture_require
    try:
        compile_ordered_key_probe()
    finally:
        lower_llir._ordered_key_expected_checkpoint = real_checkpoint
        lower_llir._require_ordered_key_completion_checkpoint = real_require

    assert set(captured) == {"input", "expected", "final"}
    expected = captured["expected"]
    # The reference is a real body, not an empty shell.
    assert sum(1 for value in expected.values() if isinstance(value, llir.Node)) > 100
    assert any(
        type(value) is llir.TensorAccessMetadata for value in expected.values()
    ), "the probe cell must carry access provenance for this proof to bind"

    pinned = getattr(lower_llir, "_SPARSE_COMPLETION_ENUM_STATES")
    for side in ("input", "final"):
        overlap = set(expected) & set(captured[side])
        for identity in overlap:
            value = expected[identity]
            if value is None or type(value) in (bool, int, float, str):
                continue
            if type(value) is tuple and not value:
                continue
            if isinstance(value, Enum):
                assert (type(value), id(value)) in pinned, value
                continue
            raise AssertionError(
                f"the checkpoint shares forgeable {type(value).__name__} state "
                f"with the pipeline {side} body: {value!r}"
            )


# -- honest legacy comparands ------------------------------------------------


@pytest.mark.parametrize("operand_fmt", ["dd", "ds", "sd", "ss"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("arm", [False, True])
def test_rank_one_reduction_matches_the_sound_legacy_route(
    operand_fmt,
    dtype,
    arm,
):
    """The valid legacy subset is a semantic shadow, not an oracle-only cell."""

    import scorch

    dense = torch.tensor(
        [
            [1.0, 0.0, 2.0, 0.0, -3.0],
            [-1.0, 4.0, 0.0, 5.0, 0.0],
            [0.0, -2.0, 3.0, 0.0, 1.0],
            [2.0, 0.0, -5.0, -1.0, 2.0],
        ],
        dtype=dtype,
    )
    cin = reduction_cin(operand_fmt, "s", "ij", "j", dtype)
    result, kernel = execute_cin_via_loopir(
        cin,
        (5,),
        sparse(dense, operand_fmt, "A"),
        compile_options=auto_options(arm, jit=True),
    )
    legacy = scorch.einsum(
        "ij->j",
        sparse(dense, operand_fmt, "A"),
        format="s",
        _compile_options=auto_options(arm, jit=True),
    )

    assert str(result.format) == str(legacy.format) == "s"
    assert compiled_storage(result) == compiled_storage(legacy)
    assert torch.equal(
        result.to_torch(in_place=False),
        legacy.to_torch(in_place=False),
    )
    assert torch.equal(result.to_torch(in_place=False), dense.sum(dim=0))

    legacy_source = legacy_generated_cpp(
        reduction_cin(operand_fmt, "s", "ij", "j", dtype),
        (5,),
        (((4, 5), dtype),),
        compile_options=auto_options(arm),
    )
    # The two correct routes currently make different lowering choices.  Keep
    # that fact explicit: this is runtime/storage parity, not byte parity.
    assert kernel.cpp_source != legacy_source


_HONEST_HIGHER_REDUCTIONS = [
    cell
    for cell in REDUCTION_CELLS
    if cell[0]
    in {
        "sss ijk->ik",
        "sss ijk->jk",
        "dss ijk->jk",
        "ssss ijkl->jkl",
    }
]


def cancelling_dense(shape, seed, dtype, reduced_axis):
    """A operand whose leading two slabs along ``reduced_axis`` cancel exactly.

    Cancellation is the case where "sums to the right number" and "stores the
    right keys" come apart: the reduced entry is exactly zero but its key was
    genuinely produced, so a route that drops it is wrong in storage while
    still right in every dense comparison.
    """

    dense = random_dense(shape, seed, dtype, density=0.75)
    if shape[reduced_axis] >= 2:
        first = [slice(None)] * len(shape)
        first[reduced_axis] = 0
        second = [slice(None)] * len(shape)
        second[reduced_axis] = 1
        dense[tuple(second)] = -dense[tuple(first)]
    return dense


@pytest.mark.parametrize(
    "cell",
    _HONEST_HIGHER_REDUCTIONS,
    ids=[cell[0] for cell in _HONEST_HIGHER_REDUCTIONS],
)
@pytest.mark.parametrize("pattern", ["random", "cancelling"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("arm", [False, True])
def test_higher_rank_reduction_matches_the_sound_legacy_route(
    cell, pattern, dtype, arm
):
    """Every measured-correct higher-rank reduction keeps a legacy shadow."""

    import scorch

    _, operand_fmt, result_fmt, operand_indices, result_indices, shape = cell
    result_shape = tuple(shape[operand_indices.index(c)] for c in result_indices)
    if pattern == "random":
        dense = random_dense(shape, 5151 + len(operand_indices), dtype, density=0.55)
    else:
        reduced_axis = next(
            position
            for position, name in enumerate(operand_indices)
            if name not in result_indices
        )
        dense = cancelling_dense(
            shape, 5151 + len(operand_indices), dtype, reduced_axis
        )
    cin = reduction_cin(
        operand_fmt,
        result_fmt,
        operand_indices,
        result_indices,
        dtype,
    )
    result, kernel = execute_cin_via_loopir(
        cin,
        result_shape,
        sparse(dense, operand_fmt, "A"),
        compile_options=auto_options(arm, jit=True),
    )
    legacy = scorch.einsum(
        f"{operand_indices}->{result_indices}",
        sparse(dense, operand_fmt, "A"),
        format=result_fmt,
        _compile_options=auto_options(arm, jit=True),
    )

    assert str(result.format) == str(legacy.format) == ",".join(result_fmt)
    assert compiled_storage(result) == compiled_storage(legacy)
    assert torch.equal(
        result.to_torch(in_place=False),
        legacy.to_torch(in_place=False),
    )
    assert kernel.cpp_source != legacy_generated_cpp(
        cin,
        result_shape,
        ((shape, dtype),),
        compile_options=auto_options(arm),
    )


@pytest.mark.parametrize("b_fmt", ["dd", "ss"])
@pytest.mark.parametrize("pattern", ["random", "cancelling"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("arm", [False, True])
def test_dss_ttm_matches_the_sound_legacy_route(b_fmt, pattern, dtype, arm):
    """The two measured-correct TTM cells keep exact sparse-storage parity."""

    import scorch

    a_shape, b_shape = (4, 5, 6), (6, 3)
    result_shape = (4, 5, 3)
    a_dense = random_dense(a_shape, 6262, dtype, density=0.45)
    if pattern == "random":
        b_dense = random_dense(b_shape, 7373, dtype, density=0.65)
    else:
        # ``k`` is the contracted axis; mirroring its leading two slabs makes
        # whole reduced entries cancel to an exact zero that was still keyed.
        b_dense = cancelling_dense(b_shape, 7373, dtype, 0)
    cin = ttm_cin("dss", b_fmt, "dss", dtype)
    result, kernel = execute_cin_via_loopir(
        cin,
        result_shape,
        sparse(a_dense, "dss", "A"),
        sparse(b_dense, b_fmt, "B"),
        compile_options=auto_options(arm, jit=True),
    )
    legacy = scorch.einsum(
        "ijk,kl->ijl",
        sparse(a_dense, "dss", "A"),
        sparse(b_dense, b_fmt, "B"),
        format="dss",
        _compile_options=auto_options(arm, jit=True),
    )

    assert str(result.format) == str(legacy.format) == "d,s,s"
    assert compiled_storage(result) == compiled_storage(legacy)
    assert torch.equal(
        result.to_torch(in_place=False),
        legacy.to_torch(in_place=False),
    )
    assert kernel.cpp_source != legacy_generated_cpp(
        cin,
        result_shape,
        ((a_shape, dtype), (b_shape, dtype)),
        compile_options=auto_options(arm),
    )


@pytest.mark.parametrize("arm", [False, True])
def test_every_migrated_cell_has_a_legacy_source_route_that_is_not_identical(arm):
    """All twenty routes generate legacy C++, and none of it is our C++.

    The inherited claim that "the legacy comparand is not a gate here" was
    read as "there is no legacy comparand".  There is one for every cell: the
    legacy generator produces a source for all twenty in both arms.  What
    varies is whether that source is *sound* -- the nine honest cells are
    locked against it above, and the eleven defective ones are characterized
    below.  Recording that the sources always differ keeps the honest cells'
    parity claim unambiguous: it is semantic, never byte parity.
    """

    for (
        name,
        operand_fmt,
        result_fmt,
        operand_indices,
        result_indices,
        shape,
    ) in REDUCTION_CELLS:
        result_shape = tuple(shape[operand_indices.index(c)] for c in result_indices)
        cin = reduction_cin(operand_fmt, result_fmt, operand_indices, result_indices)
        kernel = compile_cin_via_loopir(
            cin,
            result_shape,
            ((shape, _F32),),
            compile_options=auto_options(arm),
        )
        legacy_source = legacy_generated_cpp(
            reduction_cin(operand_fmt, result_fmt, operand_indices, result_indices),
            result_shape,
            ((shape, _F32),),
            compile_options=auto_options(arm),
        )
        assert legacy_source.strip(), name
        assert kernel.cpp_source != legacy_source, name

    for name, a_fmt, b_fmt, c_fmt in TTM_CELLS:
        a_shape, b_shape = (4, 5, 6), (6, 3)
        kernel = compile_cin_via_loopir(
            ttm_cin(a_fmt, b_fmt, c_fmt),
            (4, 5, 3),
            ((a_shape, _F32), (b_shape, _F32)),
            compile_options=auto_options(arm),
        )
        legacy_source = legacy_generated_cpp(
            ttm_cin(a_fmt, b_fmt, c_fmt),
            (4, 5, 3),
            ((a_shape, _F32), (b_shape, _F32)),
            compile_options=auto_options(arm),
        )
        assert legacy_source.strip(), name
        assert kernel.cpp_source != legacy_source, name


# -- the defective legacy comparands, characterized ---------------------------

# The eleven cells whose legacy route generates C++ that is not sound.  Each
# entry names the measured disposition, not a guess: three distinct defect
# classes, arm-invariant, every one reproduced below in a disposable process.
DEFECTIVE_LEGACY_CELLS = [
    ("sss ijk->k", "duplicate_coordinates"),
    ("dss ijk->k", "duplicate_coordinates"),
    ("ssss ijkl->l", "duplicate_coordinates"),
    ("ssss ijkl->il", "duplicate_coordinates"),
    ("ssss ijkl->kl", "uncompilable_source"),
    ("ssss ijkl->ikl", "uncompilable_source"),
    ("ssss ijkl->ijl", "malformed_child_positions"),
    ("sss x dd", "malformed_child_positions"),
    ("sss x ss", "malformed_child_positions"),
    ("sss x ds", "malformed_child_positions"),
    ("sss x sd", "malformed_child_positions"),
]

_DEFECTIVE_LEGACY_WORKER = '''
"""Characterize each defective legacy route in one disposable process."""

import json
import resource
import sys
from dataclasses import replace

resource.setrlimit(resource.RLIMIT_CPU, (1800, 1800))
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

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
from scorch.compiler.loopir.pipeline import execute_cin_via_loopir
from scorch.compiler.scheduler import Schedule
from scorch.stensor import STensor
import scorch

F32 = torch.float32


def auto_options(arm, jit):
    base = (
        CompileOptions.from_environment()
        if jit
        else CompileOptions.from_environment(environ={})
    )
    return replace(
        base.with_regblock_enabled(arm), requested_schedule=Schedule()
    )


def reduction_cin(operand_fmt, result_fmt, operand_indices, result_indices):
    ivars = {name: IndexVar(name) for name in operand_indices}
    operand = TensorVar("A", fmt=operand_fmt, dtype=F32)[
        tuple(ivars[name] for name in operand_indices)
    ]
    result = TensorVar("C", fmt=result_fmt, dtype=F32)[
        tuple(ivars[name] for name in result_indices)
    ]
    statement = TensorAssign(result, operand, op=Operation.ADD)
    for name in reversed(operand_indices):
        statement = ForAll(ivars[name], statement)
    return statement


def ttm_cin(a_fmt, b_fmt, c_fmt):
    ivars = {name: IndexVar(name) for name in "ijkl"}
    a = TensorVar("A", fmt=a_fmt, dtype=F32)[ivars["i"], ivars["j"], ivars["k"]]
    b = TensorVar("B", fmt=b_fmt, dtype=F32)[ivars["k"], ivars["l"]]
    c = TensorVar("C", fmt=c_fmt, dtype=F32)[ivars["i"], ivars["j"], ivars["l"]]
    statement = TensorAssign(
        c, CINBinaryOp(Operation.MUL, a, b), op=Operation.ADD
    )
    for name in reversed("ijkl"):
        statement = ForAll(ivars[name], statement)
    return statement


def sparse(dense, fmt, name):
    return STensor.from_torch(dense.clone(), name).to_sparse(fmt)


def random_dense(shape, seed, density):
    generator = torch.Generator().manual_seed(seed)
    values = torch.rand(shape, generator=generator, dtype=torch.float64)
    mask = torch.rand(shape, generator=generator, dtype=torch.float64) < density
    return (values * mask).to(F32)


def classify(error):
    name = type(error).__name__
    message = str(error)
    if name == "TensorIndexError":
        if "coordinates must be strictly increasing" in message:
            return "duplicate_coordinates"
        if "position array must start at zero" in message:
            return "malformed_child_positions"
        return "other_malformed_storage:" + message[:80]
    if name == "CalledProcessError":
        return "uncompilable_source"
    return name + ":" + message[:80]


CELLS = json.loads(sys.argv[1])

for cell in CELLS:
    for arm in (False, True):
        record = {"name": cell["name"], "arm": arm}
        try:
            if cell["kind"] == "reduction":
                shape = tuple(cell["shape"])
                operand_indices = cell["operand"]
                result_indices = cell["result"]
                result_shape = tuple(
                    shape[operand_indices.index(c)] for c in result_indices
                )
                dense = random_dense(shape, 5151 + len(operand_indices), 0.55)
                make_operands = lambda: [sparse(dense, cell["operand_fmt"], "A")]
                subscripts = operand_indices + "->" + result_indices
                legacy_fmt = cell["result_fmt"]
                builder = lambda: reduction_cin(
                    cell["operand_fmt"],
                    cell["result_fmt"],
                    operand_indices,
                    result_indices,
                )
            else:
                a_shape, b_shape = (4, 5, 6), (6, 3)
                result_shape = (4, 5, 3)
                a_dense = random_dense(a_shape, 6262, 0.45)
                b_dense = random_dense(b_shape, 7373, 0.65)
                make_operands = lambda: [
                    sparse(a_dense, cell["a_fmt"], "A"),
                    sparse(b_dense, cell["b_fmt"], "B"),
                ]
                subscripts = "ijk,kl->ijl"
                legacy_fmt = cell["c_fmt"]
                builder = lambda: ttm_cin(
                    cell["a_fmt"], cell["b_fmt"], cell["c_fmt"]
                )

            loopir_result, _ = execute_cin_via_loopir(
                builder(),
                result_shape,
                *make_operands(),
                compile_options=auto_options(arm, True),
            )
            record["loopir"] = "ok"
            record["loopir_nnz"] = int(loopir_result.storage.value.numel())
            try:
                scorch.einsum(
                    subscripts,
                    *make_operands(),
                    format=legacy_fmt,
                    _compile_options=auto_options(arm, True),
                )
                record["legacy"] = "ok"
            except Exception as error:  # noqa: BLE001
                record["legacy"] = classify(error)
        except Exception as error:  # noqa: BLE001
            record["harness"] = type(error).__name__ + ": " + str(error)[:200]
        print("@@CELL@@" + json.dumps(record), flush=True)
'''


def test_defective_legacy_routes_keep_their_measured_disposition(tmp_path):
    """The eleven unsound legacy comparands, characterized rather than waved off.

    Every cell here generates legacy C++ (see the twenty-route lock above),
    so "there is no comparand" was the wrong claim; the right one is that
    these eleven comparands are unsound, in exactly three measured ways.  Each
    is reproduced in a disposable process with ``RLIMIT_CPU``/``RLIMIT_CORE``
    and a wall-clock timeout, and each cell's verdict streams out as it is
    produced -- so a route that terminated the interpreter would be *named*
    rather than swallowing the whole table.  The LoopIR route is required to
    succeed on every one of them in both arms.
    """

    import json
    import subprocess
    import sys

    reduction_by_name = {cell[0]: cell for cell in REDUCTION_CELLS}
    ttm_by_name = {cell[0]: cell for cell in TTM_CELLS}
    specs = []
    for name, _ in DEFECTIVE_LEGACY_CELLS:
        if name in reduction_by_name:
            _, operand_fmt, result_fmt, operand_indices, result_indices, shape = (
                reduction_by_name[name]
            )
            specs.append(
                {
                    "name": name,
                    "kind": "reduction",
                    "operand_fmt": operand_fmt,
                    "result_fmt": result_fmt,
                    "operand": operand_indices,
                    "result": result_indices,
                    "shape": list(shape),
                }
            )
        else:
            _, a_fmt, b_fmt, c_fmt = ttm_by_name[name]
            specs.append(
                {
                    "name": name,
                    "kind": "ttm",
                    "a_fmt": a_fmt,
                    "b_fmt": b_fmt,
                    "c_fmt": c_fmt,
                }
            )

    worker = tmp_path / "characterize_defective_legacy.py"
    worker.write_text(_DEFECTIVE_LEGACY_WORKER)
    completed = subprocess.run(
        [sys.executable, str(worker), json.dumps(specs)],
        capture_output=True,
        text=True,
        timeout=5400,
    )
    measured = [
        json.loads(line[len("@@CELL@@") :])
        for line in completed.stdout.splitlines()
        if line.startswith("@@CELL@@")
    ]
    expected_pairs = [
        (name, arm) for name, _ in DEFECTIVE_LEGACY_CELLS for arm in (False, True)
    ]
    got_pairs = [(record["name"], record["arm"]) for record in measured]
    assert got_pairs == expected_pairs, (
        f"characterization stopped early (rc={completed.returncode}); "
        f"last measured {got_pairs[-1:]}\n{completed.stderr[-2000:]}"
    )
    dispositions = dict(DEFECTIVE_LEGACY_CELLS)
    for record in measured:
        assert "harness" not in record, record
        assert record["loopir"] == "ok", record
        assert record["legacy"] == dispositions[record["name"]], record


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("arm", [False, True])
def test_stored_compressed_zero_survives_both_arms_and_dtypes(dtype, arm):
    """A compressed input's explicitly stored zero remains an output key."""

    row_pos = torch.tensor([0, 2], dtype=torch.int32)
    row_crd = torch.tensor([0, 2], dtype=torch.int32)
    leaf_pos = torch.tensor([0, 2, 3], dtype=torch.int32)
    leaf_crd = torch.tensor([1, 3, 2], dtype=torch.int32)
    values = torch.tensor([0.0, 2.0, -1.0], dtype=dtype)
    operand = STensor(
        name="A",
        shape=(3, 4),
        index=TensorIndex(
            "ss",
            [[row_pos, row_crd], [leaf_pos, leaf_crd]],
        ),
        value=values,
    )
    result, kernel = execute_cin_via_loopir(
        reduction_cin("ss", "s", "ij", "j", dtype),
        (4,),
        operand,
        compile_options=auto_options(arm, jit=True),
    )
    levels, drained = compiled_storage(result)
    assert levels == (((0, 3), (1, 2, 3)),)
    assert drained == (0.0, -1.0, 2.0)

    scheduled = kernel.lowering.program
    oracle_operand = LevelTensorStorage(
        shape=(3, 4),
        modes=(0, 1),
        levels=(
            CompressedLevel((0, 2), (0, 2)),
            CompressedLevel((0, 2, 3), (1, 3, 2)),
        ),
        values=(0.0, 2.0, -1.0),
    )
    oracle = run_program(
        scheduled,
        {scheduled.inputs[0]: oracle_operand},
        {scheduled.outputs[0]: (4,)},
    )[scheduled.outputs[0]]
    assert oracle_level_storage(oracle) == (levels, drained)


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


def test_stored_operand_zeros_keep_their_key():
    """A STORED operand zero still contributes a key to the result.

    Which entries a drained result holds is a property of the iteration
    domain, not of the values: a key that was inserted must appear even when
    everything inserted under it was 0.0.  The operand is all-dense so every
    cell really is stored -- ``to_sparse`` filters structural zeros out of a
    compressed operand, so a compressed fixture could not state this.
    """

    shape = (3, 5)
    dense = torch.zeros(shape, dtype=_F32)
    dense[0, 1] = 2.0
    dense[2, 4] = -1.0
    # Column 3 is stored everywhere and zero everywhere.
    result, kernel = execute_cin_via_loopir(
        reduction_cin("dd", "s", "ij", "j"),
        (5,),
        STensor.from_torch(dense.clone(), "A"),
        compile_options=auto_options(False, jit=True),
    )
    levels, drained = compiled_storage(result)
    assert levels[0][1] == (0, 1, 2, 3, 4), "every stored column keeps its key"
    assert drained[3] == 0.0, "an all-zero stored column keeps an explicit zero"
    assert_matches_oracle_and_pytorch(
        (dense,), result, kernel, (5,), dense.to(torch.float64).sum(dim=(0,))
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
    ("operand_fmt", "result_fmt", "operand_indices", "result_indices", "shape"),
    [
        ("ss", "ss", "ij", "ji", (4, 5)),
        ("ds", "ds", "ij", "ji", (4, 5)),
        ("sss", "sss", "ijk", "ikj", (3, 4, 5)),
        ("sss", "ds", "ijk", "ki", (3, 4, 5)),
    ],
    ids=["ss->ji", "ds->ji", "sss->ikj", "sss->ki"],
)
def test_reorder_blocked_neighbours_carry_a_plan_diagnostic_not_a_defect_code(
    operand_fmt, result_fmt, operand_indices, result_indices, shape, arm
):
    """These routes are stable, but not through the defect-code channel.

    The reorder-blocked cells stop inside loop-plan legality, which raises
    ``InvalidSchedule``.  That exception carries no ``defect``: its stable
    identifier is a structured ``LoopPlanDiagnostic`` in ``diagnostics``.
    Recording that distinction matters because the retained frontier receipt
    derived the same string by matching the *message text*, which is a
    classification of a diagnostic rather than a read of one.

    The occupants are now the PERMUTED-result cells, whose pre-forced order is
    refused by the result's own storage-order rule as well, so the automatic
    plan origin has no legal order to fall back to.  The cells this test used
    to name -- ``ss->i``, ``ds->i``, ``sss->i``, ``sss->j`` and ``sss->ij`` --
    moved when the bound-prefix family landed: three are admitted and the rest
    reach their own shape-specific codes.  ``test_loopir_bound_prefix_target``
    owns all five, and the two-way gating that separates them.
    """

    from scorch.compiler.diagnostics import InvalidSchedule

    result_shape = tuple(shape[operand_indices.index(c)] for c in result_indices)
    with pytest.raises(InvalidSchedule) as error:
        compile_cin_via_loopir(
            reduction_cin(operand_fmt, result_fmt, operand_indices, result_indices),
            result_shape,
            ((shape, _F32),),
            compile_options=auto_options(arm),
        )
    assert getattr(error.value, "defect", None) is None
    diagnostics = error.value.diagnostics
    assert len(diagnostics) == 1
    assert diagnostics[0].code == "sparse_parent_dominance"
    assert diagnostics[0].path == ("loop_order", "sparse_access")
    assert diagnostics[0].stage == "loop_plan"


@pytest.mark.parametrize("arm", [False, True])
@pytest.mark.parametrize(
    ("operand_fmt", "result_fmt", "operand_indices", "result_indices", "code"),
    [
        # A stored result prefix level driven by a DENSE domain has no
        # coordinate stream to append, and CIN says so.
        ("dds", "ss", "ijk", "ik", "unsupported_sparse_output_domain"),
        # A compressed-parent/dense-leaf result keeps its own reduction seam.
        ("sss", "sd", "ijk", "ik", "unsupported_sparse_output_reduction"),
    ],
    ids=["dense-driven-prefix", "dense-leaf"],
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


@pytest.mark.parametrize("arm", [False, True])
@pytest.mark.parametrize(
    ("result_indices", "message"),
    [
        ("jk", "every drained result level to be compressed"),
    ],
    ids=["dense-row-drained"],
)
def test_dense_result_level_under_a_stored_loop_stops_at_the_target(
    result_indices, message, arm
):
    """A DRAINED dense result level is still a target boundary.

    Recorded seam move.  This test carried two cells, differing in which half
    of the split the dense level lands in.  The PREFIX half (``ik``) is blocker
    3 and is migrated -- see ``test_loopir_row_scope_prefix_target.py``, and
    ``test_the_migrated_row_scope_prefix_no_longer_stops_here`` below, which
    states the migration where the lock used to be.  The DRAINED half survives:
    a workspace key level owns its own ordering and appends one coordinate per
    drained entry, so a dense level cannot be one, and no catch-up applies.
    """

    shape = (3, 4, 5)
    result_shape = tuple(shape["ijk".index(c)] for c in result_indices)
    with pytest.raises(LoopIRTargetError) as error:
        compile_cin_via_loopir(
            reduction_cin("sss", "ds", "ijk", result_indices),
            result_shape,
            ((shape, _F32),),
            compile_options=auto_options(arm),
        )
    assert error.value.defect.code == "unsupported_program_shape"
    assert message in str(error.value)


@pytest.mark.parametrize("arm", [False, True])
def test_the_migrated_row_scope_prefix_no_longer_stops_here(arm):
    """``sss ijk->ik [ds]`` compiles now, and this is where that is recorded.

    It was the ``dense-row-prefix`` cell above.  CIN routes a canonical-CSR
    receiver to ``CSR_SPARSE_ROW``, which already permits a stored row domain,
    so its refusal was never a CIN one -- the ordered-key target's prefix
    require was the whole boundary, and blocker 3 moves exactly that.  The
    migration is asserted at the old lock's site so the seam's history stays
    readable from the seam.
    """

    assert compile_cin_via_loopir(
        reduction_cin("sss", "ds", "ijk", "ik"),
        (3, 5),
        (((3, 4, 5), _F32),),
        compile_options=auto_options(arm),
    ).cpp_source


def test_the_explicitly_ordered_route_stops_at_the_operand_not_the_prefix():
    """Corrected: this cell no longer stops where its name used to say.

    It was ``test_row_scope_dense_prefix_is_rejected_by_the_target``, asserting
    that ``sss ijk->ik [ds]`` under an explicit legal order stops at the target
    because of its dense result prefix level.  Blocker 3 migrated that prefix,
    and the test kept PASSING -- for an unrelated reason: the explicitly ordered
    route builds no workspace region, so the program reaches the generic target,
    which refuses ``sss`` at its hierarchical-compressed-INPUT rule before any
    prefix rule is consulted.  A test that passes for the wrong reason is worse
    than one that fails, so the boundary it actually proves is named here and
    the message is asserted, not just the code.
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
    assert error.value.defect.code == "unsupported_program_shape"
    assert "hierarchical compressed" in str(error.value)


# -- the post-assembly window ------------------------------------------------
#
# The sealed checkpoint verifies the statement list the managed pipeline
# returns.  Four stages then run before the caller sees a function: ABI
# assembly, then the parallel, panel, result-tile and relayout completions.
# The two-node check the seal replaced DID cover that window, because it ran on
# the assembled function, so leaving it unchecked was a real narrowing -- a
# drain duplicated there is exactly the silent storage corruption the verified
# body is checked for.  These lock the window shut from both ends.


def test_post_assembly_drain_duplication_is_rejected(monkeypatch):
    """Duplicating the ordered drain during ABI assembly fails closed.

    ``assemble_function`` is shared by every target, so it is the one piece of
    the post-checkpoint window that a change elsewhere could plausibly reach.
    """

    import scorch.compiler.torch_cpp_abi as torch_cpp_abi

    original = torch_cpp_abi.TorchCppKernelABI.assemble_function
    state = {"duplicated": False}

    def hostile(self, body):
        function = original(self, body)
        statements = object.__getattribute__(function, "__dict__")["body"]
        for index, statement in enumerate(list(statements)):
            if type(statement) is llir.ForLoopAuto:
                statements.insert(index + 1, statement)
                state["duplicated"] = True
                break
        return function

    monkeypatch.setattr(torch_cpp_abi.TorchCppKernelABI, "assemble_function", hostile)
    with pytest.raises(LoopIRTargetError) as error:
        compile_ordered_key_probe()
    assert state["duplicated"], "the probe must actually duplicate the drain"
    assert error.value.defect.code == "sparse_workspace_completion_lost"


def test_post_assembly_statement_substitution_is_rejected(monkeypatch):
    """A verified statement replaced by a value-equal copy fails closed.

    Identity is the right test here precisely because these statements were
    deep-compared against the detached reference moments earlier: carrying that
    verification forward requires the same objects, not merely equal ones.
    """

    import scorch.compiler.torch_cpp_abi as torch_cpp_abi

    original = torch_cpp_abi.TorchCppKernelABI.assemble_function
    state = {"substituted": False}

    def hostile(self, body):
        function = original(self, body)
        statements = object.__getattribute__(function, "__dict__")["body"]
        for index, statement in enumerate(statements):
            if type(statement) is llir.Comment:
                statements[index] = llir.Comment(statement.value)
                state["substituted"] = True
                break
        return function

    monkeypatch.setattr(torch_cpp_abi.TorchCppKernelABI, "assemble_function", hostile)
    with pytest.raises(LoopIRTargetError) as error:
        compile_ordered_key_probe()
    assert state["substituted"]
    assert error.value.defect.code == "sparse_workspace_completion_lost"


def test_a_post_assembly_completion_stage_is_rejected(monkeypatch):
    """A completion stage that SUBSTITUTES a function has no replay contract.

    All four stages are inert for every ordered-key cell today, and each is
    guarded by its own ``is None`` test.  Requiring the returned function to BE
    the assembled function proves that per compile instead of assuming it.

    It proves exactly that and no more: because every stage returns the object
    it was handed, this requirement cannot detect a stage that *ran*.  A fused
    workspace-plus-tile plan would satisfy it -- the gate such a plan has to
    open deliberately is the structural comparison, which sees the nested
    in-place rewrites result-tile completion performs.  See
    ``test_no_completion_stage_constructs_a_new_function``.
    """

    from scorch.compiler.loopir import lower_llir

    original = lower_llir._TargetLowering.complete_relayout
    state = {"rewrapped": False}

    def hostile(self, function):
        returned = original(self, function)
        state["rewrapped"] = True
        return llir.Function(
            return_type=returned.return_type,
            name=returned.name,
            args=returned.args,
            body=list(object.__getattribute__(returned, "__dict__")["body"]),
        )

    monkeypatch.setattr(lower_llir._TargetLowering, "complete_relayout", hostile)
    with pytest.raises(LoopIRTargetError) as error:
        compile_ordered_key_probe()
    assert state["rewrapped"]
    assert error.value.defect.code == "sparse_workspace_completion_lost"


def test_ordered_key_cells_still_reach_a_kernel_through_the_window():
    """The post-assembly requirements are inert for a correct compile."""

    for cell in REDUCTION_CELLS:
        _, operand_fmt, result_fmt, operand_indices, result_indices, shape = cell
        result_shape = tuple(shape[operand_indices.index(c)] for c in result_indices)
        for arm in (False, True):
            kernel = compile_cin_via_loopir(
                reduction_cin(operand_fmt, result_fmt, operand_indices, result_indices),
                result_shape,
                ((shape, _F32),),
                compile_options=auto_options(arm),
            )
            assert "wksp.sort();" in kernel.cpp_source


# -- extended tamper classes -------------------------------------------------


def test_ordered_key_completion_rejects_a_same_typed_subclass(monkeypatch):
    """A node rewritten into a SUBCLASS of its own type fails closed.

    The comparator's leading type-identity test already refuses this, and the
    exact-type node dispatch keeps it refused; the point of the lock is that
    the refusal never consults the subclass's own hooks.
    """

    hooks = []

    class Recording(llir.ForLoop):
        def __eq__(self, other):
            hooks.append("__eq__")
            raise AssertionError("__eq__ was consulted")

        def __hash__(self):
            hooks.append("__hash__")
            raise AssertionError("__hash__ was consulted")

        def __reduce_ex__(self, protocol):
            hooks.append("__reduce_ex__")
            raise AssertionError("__reduce_ex__ was consulted")

    def tamper(rewritten):
        for value in owned_objects(rewritten):
            if type(value) is llir.ForLoop:
                clone = object.__new__(Recording)
                object.__getattribute__(clone, "__dict__").update(
                    object.__getattribute__(value, "__dict__")
                )
                return replace_object(rewritten, value, clone)
        return 0

    state = install_hostile_pass(monkeypatch, tamper)
    with pytest.raises(LoopIRTargetError) as error:
        compile_ordered_key_probe()
    assert state["changed"]
    assert error.value.defect.code == "sparse_workspace_completion_lost"
    assert hooks == [], f"the boundary consulted subclass hooks: {hooks}"


def test_ordered_key_completion_rejects_a_tuple_where_a_list_was(monkeypatch):
    """A pass returning an immutable sequence for a mutable one fails closed."""

    def tamper(rewritten):
        for value in owned_objects(rewritten):
            if not isinstance(value, llir.Node):
                continue
            state = object.__getattribute__(value, "__dict__")
            for field_name in tuple(state):
                member = state[field_name]
                if type(member) is list and member:
                    state[field_name] = tuple(member)
                    return 1
        return 0

    state = install_hostile_pass(monkeypatch, tamper)
    with pytest.raises(LoopIRTargetError) as error:
        compile_ordered_key_probe()
    assert state["changed"]
    assert error.value.defect.code == "sparse_workspace_completion_lost"


def test_ordered_key_completion_rejects_a_distinct_enum_twin(monkeypatch):
    """A DIFFERENT object of an accepted enum type with identical stored state.

    The pinned-state check compares stored state, so a twin would satisfy it;
    the identity requirement is what refuses this, and both are needed.
    """

    def tamper(rewritten):
        for owner in owned_objects(rewritten):
            if not isinstance(owner, llir.Node):
                continue
            state = object.__getattribute__(owner, "__dict__")
            for field_name in tuple(state):
                member = state[field_name]
                if type(member) in (
                    llir.AssignOp,
                    llir.DataType,
                    llir.TensorAccessRole,
                ):
                    twin = object.__new__(type(member))
                    object.__getattribute__(twin, "__dict__").update(
                        object.__getattribute__(member, "__dict__")
                    )
                    if twin is member:
                        continue
                    state[field_name] = twin
                    return 1
        return 0

    state = install_hostile_pass(monkeypatch, tamper)
    with pytest.raises(LoopIRTargetError) as error:
        compile_ordered_key_probe()
    assert state["changed"]
    assert error.value.defect.code == "sparse_workspace_completion_lost"


def test_ordered_key_completion_rejects_a_partially_applied_tamper(monkeypatch):
    """One of two structurally equal subtrees changed leaves the body valid.

    The result still compiles as C++ and is internally consistent; only the
    reference disagrees.  That is the direction a whole-body comparison must
    catch and a two-node check cannot.
    """

    def tamper(rewritten):
        groups = {}
        for value in owned_objects(rewritten):
            if type(value) is llir.Comment:
                text = object.__getattribute__(value, "__dict__").get("value")
                groups.setdefault(text, []).append(value)
        for text, members in groups.items():
            if len(members) >= 2 and type(text) is str:
                object.__getattribute__(members[0], "__dict__")["value"] = text + " "
                return 1
        return 0

    state = install_hostile_pass(monkeypatch, tamper)
    with pytest.raises(LoopIRTargetError) as error:
        compile_ordered_key_probe()
    assert state["changed"]
    assert error.value.defect.code == "sparse_workspace_completion_lost"


def test_ordered_key_completion_rejects_a_benign_consistent_rename(monkeypatch):
    """An internally consistent rename is still refused, and that is the point.

    This family's contract is that the managed pipeline hands its emission back
    unchanged, so a transformation that would be semantically harmless is
    refused rather than guessed about.  Any future pass that legitimately needs
    to touch this body has to extend the contract deliberately.
    """

    def tamper(rewritten):
        variables = [v for v in owned_objects(rewritten) if type(v) is llir.Var]
        targets = [v for v in variables if v.name.startswith("i")]
        if not targets:
            return 0
        old = targets[0].name
        changed = 0
        for variable in variables:
            if variable.name == old:
                object.__getattribute__(variable, "__dict__")["name"] = old + "_r"
                changed += 1
        return changed

    state = install_hostile_pass(monkeypatch, tamper)
    with pytest.raises(LoopIRTargetError) as error:
        compile_ordered_key_probe()
    assert state["changed"]
    assert error.value.defect.code == "sparse_workspace_completion_lost"


def test_ordered_key_completion_rejects_pre_pipeline_mutation(monkeypatch):
    """Mutating the PRE-pipeline body after the reference was built fails closed.

    ``assemble_body`` splices the prologue and final-assembly statements into
    the pipeline's body by reference, so a pass that reaches back and mutates
    one of those nodes in place moves the pipeline body without moving the
    already-detached reference.
    """

    from scorch.compiler.loopir import lower_llir

    original = lower_llir._ordered_key_expected_checkpoint
    state = {"mutated": False}

    def checkpoint(lowering, kernel_abi, body):
        expected = original(lowering, kernel_abi, body)
        if expected is not None:
            for statement in reversed(body):
                if type(statement) is llir.Comment:
                    stored = object.__getattribute__(statement, "__dict__")
                    if type(stored.get("value")) is str:
                        stored["value"] += " tampered"
                        state["mutated"] = True
                        break
        return expected

    monkeypatch.setattr(lower_llir, "_ordered_key_expected_checkpoint", checkpoint)
    with pytest.raises(LoopIRTargetError) as error:
        compile_ordered_key_probe()
    assert state["mutated"], "the probe must actually mutate the pre-pipeline body"
    assert error.value.defect.code == "sparse_workspace_completion_lost"


# -- why the detaching mirror's two refusals are exact -----------------------


def test_for_loop_update_is_the_only_non_sequence_assign_position():
    """The mirror's "sequences only" rule equals the pass's "not update" rule.

    ``_OrderedKeyExpectedBody`` converts a dynamic-vector store only when the
    ``Assign`` is a member of a sequence; the shared pass converts one unless
    its path ends in ``update``.  Those two rules agree exactly because
    ``ForLoop.update`` is the ONLY field in the whole LLIR schema that can hold
    a bare ``Assign`` outside a list -- which is a schema fact, so it is locked
    as one rather than left as an empirical observation about today's bodies.
    """

    import collections.abc
    import typing

    from scorch.compiler.llir_traversal import SUPPORTED_LLIR_NODE_TYPES

    sequence_origins = {
        list,
        tuple,
        set,
        frozenset,
        collections.abc.Sequence,
        collections.abc.MutableSequence,
        collections.abc.Iterable,
    }

    def admits_a_bare_assign(annotation):
        origin = typing.get_origin(annotation)
        if origin in sequence_origins:
            return False
        if origin is typing.Union:
            return any(
                admits_a_bare_assign(argument)
                for argument in typing.get_args(annotation)
            )
        if isinstance(annotation, type):
            try:
                return issubclass(llir.Assign, annotation)
            except TypeError:
                return False
        return False

    bare_assign_fields = []
    for node_type in SUPPORTED_LLIR_NODE_TYPES:
        # ``__init__`` hints cover the hand-written nodes as well as the
        # dataclasses, so the derivation is over the whole schema.
        hints = typing.get_type_hints(node_type.__init__, vars(llir))
        for name, annotation in hints.items():
            if name in ("self", "return"):
                continue
            if admits_a_bare_assign(annotation):
                bare_assign_fields.append((node_type.__name__, name))
    assert bare_assign_fields == [("ForLoop", "update")], bare_assign_fields


def test_the_expected_body_mirror_reproduces_the_shared_pass():
    """The detaching mirror computes exactly what the managed pass computes.

    This is the seal's central fidelity claim, and it is checkable directly
    rather than through the list of empirical facts that motivated it: run the
    same pre-pipeline body through both constructions and require structural
    equality, with enums compared by identity and provenance identities by
    stored value -- exactly what the comparator accepts.
    """

    from enum import Enum

    from scorch.compiler.dynamic_vector_access_pass import (
        DYNAMIC_VECTOR_ACCESS_CONTEXT,
        rewrite_dynamic_vector_accesses,
    )
    from scorch.compiler.llir_traversal import SUPPORTED_LLIR_NODE_TYPES
    from scorch.compiler.loopir import lower_llir

    node_types = frozenset(SUPPORTED_LLIR_NODE_TYPES)

    def differs(left, right, path=(), depth=0):
        if depth > 400:
            return [(path, "depth cap")]
        if type(left) is not type(right):
            return [(path, f"type {type(left).__name__}/{type(right).__name__}")]
        if left is None:
            return []
        if type(left) in (bool, int, float, str):
            return [] if left == right else [(path, f"{left!r}/{right!r}")]
        if isinstance(left, Enum):
            return [] if left is right else [(path, f"enum {left!r}/{right!r}")]
        if type(left) in (list, tuple):
            if len(left) != len(right):
                return [(path, f"len {len(left)}/{len(right)}")]
            found = []
            for index, (a, b) in enumerate(zip(left, right)):
                found += differs(a, b, path + (str(index),), depth + 1)
            return found
        if type(left) in (
            lower_llir.AccessId,
            lower_llir.SymbolId,
            lower_llir.IndexId,
        ):
            a = lower_llir._stored_identity_value(left, type(left))
            b = lower_llir._stored_identity_value(right, type(right))
            return [] if (a == b and a is not None) else [(path, f"id {a}/{b}")]
        if type(left) is llir.TensorAccessMetadata or type(left) in node_types:
            left_state = object.__getattribute__(left, "__dict__")
            right_state = object.__getattribute__(right, "__dict__")
            if set(left_state) != set(right_state):
                return [(path, "field names")]
            found = []
            for key in sorted(left_state):
                found += differs(
                    left_state[key], right_state[key], path + (key,), depth + 1
                )
            return found
        return [(path, f"unknown {type(left).__name__}")]

    compared = []
    original = lower_llir._ordered_key_expected_checkpoint

    def checkpoint(lowering, kernel_abi, body):
        if type(lowering) is lower_llir._OrderedKeySparseWorkspaceLowering:
            mirror = lower_llir._OrderedKeyExpectedBody(deepcopy(body)).build()
            shared = rewrite_dynamic_vector_accesses(
                deepcopy(body), DYNAMIC_VECTOR_ACCESS_CONTEXT
            )
            compared.append(differs(shared, mirror))
        return original(lowering, kernel_abi, body)

    try:
        lower_llir._ordered_key_expected_checkpoint = checkpoint
        for cell in REDUCTION_CELLS:
            _, operand_fmt, result_fmt, operand_indices, result_indices, shape = cell
            result_shape = tuple(
                shape[operand_indices.index(c)] for c in result_indices
            )
            for arm in (False, True):
                compile_cin_via_loopir(
                    reduction_cin(
                        operand_fmt, result_fmt, operand_indices, result_indices
                    ),
                    result_shape,
                    ((shape, _F32),),
                    compile_options=auto_options(arm),
                )
        for cell in TTM_CELLS:
            _, a_fmt, b_fmt, c_fmt = cell
            for arm in (False, True):
                compile_cin_via_loopir(
                    ttm_cin(a_fmt, b_fmt, c_fmt),
                    (4, 5, 3),
                    (((4, 5, 6), _F32), ((6, 3), _F32)),
                    compile_options=auto_options(arm),
                )
    finally:
        lower_llir._ordered_key_expected_checkpoint = original

    assert len(compared) == 2 * (len(REDUCTION_CELLS) + len(TTM_CELLS))
    divergent = [found for found in compared if found]
    assert not divergent, divergent[:3]


# -- the window closed by structure rather than by identity -------------------
#
# Verifying the pipeline's statement list and then carrying that verification
# across the completion window by object identity leaves one class open: an
# in-place rewrite INSIDE an already-verified statement preserves every
# identity, so a nested change one level down is invisible.  That was measured,
# not supposed -- before the comparison moved to the far side of the window, a
# comment rewritten inside a verified statement reached the emitted source and a
# statement duplicated inside the ordered drain compiled.  The single structural
# comparison now runs on the assembled body, so the whole window is covered by
# structure; these three lock the classes that closure adds, and the fourth
# locks the cost claim that made it affordable.


def _last_completion_stage_tamper(monkeypatch, tamper):
    """Run ``tamper`` on the function the last completion stage returns."""

    from scorch.compiler.loopir import lower_llir

    original = lower_llir._TargetLowering.complete_relayout
    state = {"landed": False}

    def hostile(self, function):
        returned = original(self, function)
        state["landed"] = tamper(returned)
        return returned

    monkeypatch.setattr(lower_llir._TargetLowering, "complete_relayout", hostile)
    return state


def test_post_assembly_nested_rewrite_is_rejected(monkeypatch):
    """A rewrite INSIDE a verified statement fails closed.

    Every top-level statement is still the same object, so the identity tests
    are all satisfied; only the structural comparison on the assembled body
    sees this.
    """

    def tamper(function):
        body = object.__getattribute__(function, "__dict__")["body"]
        for top in body:
            if type(top) is llir.Comment:
                continue
            for node in _reachable_nodes(top):
                if type(node) is not llir.Comment:
                    continue
                state = object.__getattribute__(node, "__dict__")
                text = state.get("value")
                if isinstance(text, str):
                    state["value"] = text + " tampered"
                    return True
        return False

    state = _last_completion_stage_tamper(monkeypatch, tamper)
    with pytest.raises(LoopIRTargetError) as error:
        compile_ordered_key_probe()
    assert state["landed"], "the probe must actually rewrite a nested comment"
    assert error.value.defect.code == "sparse_workspace_completion_lost"


def test_post_assembly_nested_duplication_is_rejected(monkeypatch):
    """A statement duplicated INSIDE the ordered drain fails closed.

    This is the semantic-damage member of the class: a duplicated append inside
    the drain files every entry twice, and no top-level identity moves.
    """

    def tamper(function):
        body = object.__getattribute__(function, "__dict__")["body"]
        for top in body:
            if type(top) is not llir.ForLoopAuto:
                continue
            nested = object.__getattribute__(top, "__dict__").get("body")
            if type(nested) is list and nested:
                nested.append(nested[-1])
                return True
        return False

    state = _last_completion_stage_tamper(monkeypatch, tamper)
    with pytest.raises(LoopIRTargetError) as error:
        compile_ordered_key_probe()
    assert state["landed"], "the probe must actually duplicate a drained statement"
    assert error.value.defect.code == "sparse_workspace_completion_lost"


def test_post_assembly_function_subclass_is_rejected(monkeypatch):
    """A ``Function`` SUBCLASS carrying identical state fails closed here.

    It satisfies both identity requirements, and codegen would refuse it later
    on exact-type dispatch, but this family owns the diagnosis, so the root type
    is pinned at the boundary with the family's own defect code.
    """

    import scorch.compiler.torch_cpp_abi as torch_cpp_abi

    subclass = type("SubclassedFunction", (llir.Function,), {})
    original = torch_cpp_abi.TorchCppKernelABI.assemble_function
    state = {"landed": False}

    def hostile(self, body):
        function = original(self, body)
        clone = object.__new__(subclass)
        object.__getattribute__(clone, "__dict__").update(
            object.__getattribute__(function, "__dict__")
        )
        state["landed"] = True
        return clone

    monkeypatch.setattr(torch_cpp_abi.TorchCppKernelABI, "assemble_function", hostile)
    with pytest.raises(LoopIRTargetError) as error:
        compile_ordered_key_probe()
    assert state["landed"]
    assert error.value.defect.code == "sparse_workspace_completion_lost"
    assert type(subclass) is type


def test_the_completion_comparison_runs_exactly_once_per_compile(monkeypatch):
    """Moving the comparison across the window added no second traversal.

    The comparison covers assembly and all four completion stages instead of
    the pipeline's list alone, and it is still the same single call -- which is
    what keeps the boundary inside its compile-latency ceiling.
    """

    from scorch.compiler.loopir import lower_llir

    original = lower_llir._exact_sparse_completion_matches
    calls = []

    def counted(actual, expected):
        calls.append(type(actual).__name__)
        return original(actual, expected)

    monkeypatch.setattr(lower_llir, "_exact_sparse_completion_matches", counted)
    for arm in (False, True):
        del calls[:]
        assert compile_ordered_key_probe(arm) is not None
        assert calls == ["list"], calls


# -- the ABI signature across the completion window ---------------------------


def _install_hostile_completion_stage(monkeypatch, tamper):
    """Let ``tamper`` corrupt the function the LAST completion stage returns."""

    from scorch.compiler.loopir import lower_llir

    original = lower_llir._TargetLowering.complete_relayout
    state = {"landed": False}

    def hostile(self, function):
        returned = original(self, function)
        state["landed"] = bool(tamper(returned))
        return returned

    monkeypatch.setattr(lower_llir._TargetLowering, "complete_relayout", hostile)
    return state


def _tamper_rename(function):
    state = object.__getattribute__(function, "__dict__")
    state["name"] = str(state["name"]) + "_TAMPERED"
    return 1


def _tamper_return_type(function):
    state = object.__getattribute__(function, "__dict__")
    before = state["return_type"]
    replacement = next(
        candidate
        for candidate in (llir.DataType.FLOAT32, llir.DataType.INT32)
        if candidate is not before
    )
    state["return_type"] = replacement
    return 1


def _tamper_drop_one_argument(function):
    state = object.__getattribute__(function, "__dict__")
    args = state["args"]
    assert type(args) is list and len(args) > 1
    del args[-1]
    return 1


def _tamper_duplicate_last_argument(function):
    state = object.__getattribute__(function, "__dict__")
    args = state["args"]
    assert type(args) is list and args
    args.append(args[-1])
    return 1


def _tamper_rename_one_argument(function):
    state = object.__getattribute__(function, "__dict__")
    args = state["args"]
    assert type(args) is list and args
    victim = args[-1]
    inner = object.__getattribute__(victim, "__dict__")
    inner["name"] = str(inner["name"]) + "_TAMPERED"
    return 1


@pytest.mark.parametrize("arm", [False, True])
@pytest.mark.parametrize(
    "tamper",
    [
        _tamper_rename,
        _tamper_return_type,
        _tamper_drop_one_argument,
        _tamper_duplicate_last_argument,
        _tamper_rename_one_argument,
    ],
    ids=["name", "return_type", "drop_arg", "duplicate_arg", "rename_arg"],
)
def test_the_abi_signature_is_covered_across_the_completion_window(
    monkeypatch,
    tamper,
    arm,
):
    """A body-only reference left three ``llir.Function`` fields unverified.

    That was measured, not supposed.  Against a body-only reference, each of
    these five tampers **compiled**, three of them putting a corrupted public
    signature straight into the emitted C++ -- a renamed entry point, a
    ``float`` return type where a tensor was declared, and a signature one
    argument short of the body's own references.  Codegen type-checks the
    argument *elements* but never which arguments they are, so it is no
    backstop for content, and no identity requirement in the window observes
    any of the three fields because every completion stage returns the object
    it was handed.
    """

    state = _install_hostile_completion_stage(monkeypatch, tamper)
    with pytest.raises(LoopIRTargetError) as error:
        compile_ordered_key_probe(arm)
    assert state["landed"]
    assert error.value.defect.code == "sparse_workspace_completion_lost"


@pytest.mark.parametrize("arm", [False, True])
def test_a_wholesale_function_dict_swap_preserving_every_field_is_accepted(
    monkeypatch,
    arm,
):
    """The positive control: container identity alone is not a corruption.

    Replacing the function's ``__dict__`` with an equal mapping, and its body
    with a fresh list holding the same statements in the same order, changes
    nothing the emitted C++ can observe.  The boundary checks per-element
    identity and structure, never the identity of the containers, so this is
    accepted -- and it has to be, or the check would be pinning an
    implementation detail of assembly rather than the program.
    """

    def tamper(function):
        state = object.__getattribute__(function, "__dict__")
        replacement = dict(state)
        replacement["body"] = list(replacement["body"])
        state.clear()
        state.update(replacement)
        return 1

    state = _install_hostile_completion_stage(monkeypatch, tamper)
    assert compile_ordered_key_probe(arm) is not None
    assert state["landed"]


@pytest.mark.parametrize("arm", [False, True])
def test_the_assembled_root_must_carry_exactly_the_declared_function_fields(
    monkeypatch,
    arm,
):
    """An extra stored field on the root would otherwise ride along unread."""

    def tamper(function):
        object.__getattribute__(function, "__dict__")["smuggled"] = 1
        return 1

    state = _install_hostile_completion_stage(monkeypatch, tamper)
    with pytest.raises(LoopIRTargetError) as error:
        compile_ordered_key_probe(arm)
    assert state["landed"]
    assert error.value.defect.code == "sparse_workspace_completion_lost"


def test_no_completion_stage_constructs_a_new_function():
    """Why the returned-function identity is not the fused-plan tripwire.

    §51.4 and the boundary's own first draft claimed that requiring
    ``returned is assembled`` "fails closed the moment a future plan composes a
    workspace with a tile, panel or relayout".  It does not: all four stages
    return the object they were given -- result-tile and relayout completion
    each have exactly one return expression, ``function``, and both mutate
    nested loop bodies in place -- so a fused plan would satisfy this
    requirement.  What fails closed for such a plan is the structural
    comparison, because those in-place rewrites are not in the reference.  This
    test pins the measurement so the claim cannot drift back.
    """

    import ast
    import inspect
    import textwrap

    from scorch.compiler.loopir import lower_llir

    for name in ("_complete_result_tile_impl", "_complete_relayout_impl"):
        source = textwrap.dedent(inspect.getsource(getattr(lower_llir, name)))
        spellings = sorted(
            {
                ast.unparse(node.value)
                for node in ast.walk(ast.parse(source))
                if isinstance(node, ast.Return) and node.value is not None
            }
        )
        assert spellings == ["function"], (name, spellings)

    for name in (
        "complete_parallel",
        "complete_panel",
        "complete_result_tile",
        "complete_relayout",
    ):
        source = textwrap.dedent(
            inspect.getsource(getattr(lower_llir._TargetLowering, name))
        )
        spellings = sorted(
            {
                ast.unparse(node.value)
                for node in ast.walk(ast.parse(source))
                if isinstance(node, ast.Return) and node.value is not None
            }
        )
        assert not any("llir.Function(" in spelling for spelling in spellings), (
            name,
            spellings,
        )


def test_the_signature_reference_is_the_abis_own_authority():
    """The required signature is asked for, not restated.

    ``assemble_function`` and the completion reference read one method, so the
    reference cannot drift from the spelling assembly actually emits; and each
    call hands back freshly built arguments, so asking twice cannot alias the
    assembled function's own state.
    """

    from scorch.compiler.torch_cpp_abi import KernelTensorABI, TorchCppKernelABI
    from scorch.format import LevelType

    abi = TorchCppKernelABI(
        result_shape=(4, 5),
        result_rank=2,
        input_tensors=(
            KernelTensorABI(
                name="A",
                level_types=(LevelType.COMPRESSED, LevelType.COMPRESSED),
                mode_order=(0, 1),
                shape=(4, 5),
                dtype=_F32,
            ),
        ),
    )
    return_type, name, args = abi.signature()
    assembled = abi.assemble_function([])
    assert assembled.return_type is return_type
    assert assembled.name == name
    assert [(var.name, var.type) for var in assembled.args] == [
        (var.name, var.type) for var in args
    ]
    # Fresh, unshared arguments on every call.
    again = abi.signature()[2]
    assert all(first is not second for first, second in zip(args, again))
    assert all(first is not second for first, second in zip(args, assembled.args))


def _reachable_nodes(root):
    """Every LLIR node reachable from ``root``, without consulting ``__eq__``."""

    found = []
    seen = set()
    stack = [root]
    while stack:
        value = stack.pop()
        if id(value) in seen:
            continue
        kind = type(value)
        if kind is list or kind is tuple:
            seen.add(id(value))
            stack.extend(value)
            continue
        if isinstance(value, llir.Node):
            seen.add(id(value))
            found.append(value)
            stack.extend(object.__getattribute__(value, "__dict__").values())
    return found
