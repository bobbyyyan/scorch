"""Every mechanism that rebuilds or compares an LLIR statement field by field
has to carry the result-storage marker across, and the set of such mechanisms is
locked against the tree.

Why this file exists, in one paragraph.  ``result_write_pass`` decides whether a
statement writes the result tensor's own storage by reading a marker
(``llir.ResultStorageMetadata``) that the emitting lowering attaches.  The
marker is declared ``compare=False, repr=False`` on five statement types, which
hides it from ``==`` and ``hash`` but not from ``__dict__``, from
``dataclasses.fields`` or from ``get_type_hints``.  So any pass that rebuilds a
statement from an explicit field list drops the marker silently, and any
comparator that dispatches on an exhaustive type table refuses a marked
statement it has no branch for.  Review section 64.4 measured what that costs:
reinstating one untaught rewriter delivers **0** markers to the guard instead of
764, while the emitted C++ stays byte-identical, the suite stays green and the
static checks stay clean.  Nothing on the route that ships can tell a working
feature from a dead one.  That is what this file is for.

Three things it does that a list of six hand-written assertions would not:

* It **discovers** the mechanisms from the tree rather than restating them, so a
  seventh one written next month fails a test instead of going unnoticed.  Three
  detectors run: generic field readers, statement rebuilds, and
  ``LLIRRewriter`` subclasses.  Each is locked against
  :data:`_MECHANISMS` by set equality in both directions.
* It **classifies** every discovered mechanism with a verdict and a reason, so a
  mechanism that is deliberately blind to the marker says so at a place a reader
  will find, rather than being absent from a list for no recorded reason.
* It **exercises** every rebuild it can reach, so "threads the field" is a
  measured property of the code rather than a property of the source text.

What it does NOT do, stated because the limits matter:

* The two static detectors are syntactic.  A rebuild that reaches its source
  node through something the scanner cannot follow -- a dict lookup, a list
  element, a computed ``getattr`` -- is invisible to them.  That class of miss is
  real rather than hypothetical: ``parallel_chunk_assembly._body_lambda`` was
  missed until the scanner learned to see through ``cast(T, x).field``.
* Driving a rebuild helper directly proves the helper threads the field.  It does
  not prove the enclosing pass reaches that helper on any real program.  That is
  what ``harness/marker_reach.py`` measures, over the 1,139-program matrix, and
  it stays a harness you run rather than a gate that runs itself.
* Mechanisms outside ``src/scorch`` are out of scope.
"""

from __future__ import annotations

import ast
import pathlib
from dataclasses import dataclass
from importlib import import_module
from typing import Dict, FrozenSet, List, Optional, Set, Tuple, get_type_hints

import pytest

import scorch
from scorch.compiler import llir
from scorch.compiler.cin import SymbolId
from scorch.compiler.llir_traversal import (  # type: ignore[import-untyped]
    SUPPORTED_LLIR_NODE_TYPES,
    LLIRRewriter,
)

# --------------------------------------------------------------------------
# The schema, derived rather than restated
# --------------------------------------------------------------------------


def _marker_carrying_type_names() -> FrozenSet[str]:
    """The statement types that declare ``result_storage``, from the schema.

    Derived from ``get_type_hints`` over every supported LLIR node type rather
    than hand-written, so a sixth type gaining the field widens both static
    detectors automatically instead of quietly falling outside them.
    """

    names = set()
    for node_type in SUPPORTED_LLIR_NODE_TYPES:
        initializer = getattr(node_type, "__init__", None)
        if initializer is None:
            continue
        try:
            hints = get_type_hints(initializer, vars(llir))
        except Exception:  # pragma: no cover - a node with unresolvable hints
            continue
        if "result_storage" in hints:
            names.add(node_type.__name__)
    return frozenset(names)


MARKER_CARRYING_TYPES = _marker_carrying_type_names()

#: The declared field names of each marker-carrying type, used by the rebuild
#: detector to tell "copies a field off a statement" from "happens to read an
#: attribute with the same name off something else".
DECLARED_FIELDS: Dict[str, FrozenSet[str]] = {}
for _name in sorted(MARKER_CARRYING_TYPES):
    _hints = get_type_hints(getattr(llir, _name).__init__, vars(llir))
    _hints.pop("return", None)
    DECLARED_FIELDS[_name] = frozenset(_hints)


# --------------------------------------------------------------------------
# The registry
# --------------------------------------------------------------------------

#: Reads a statement's fields generically and decides something from them.
KIND_READER = "reader"
#: Reconstructs a statement from an explicit field list.
KIND_REBUILD = "rebuild"
#: Subclasses ``LLIRRewriter``, so it inherits or overrides the field-by-field
#: statement reconstruction the base class performs.
KIND_REWRITER = "rewriter"

#: Has a branch for the marker and carries it, or refuses truthfully.
HANDLES = "handles"
#: Cannot see the marker, for a reason recorded on the entry.
INERT = "inert"
#: Sees the marker and deliberately does nothing with it.
BLIND_BY_DESIGN = "blind-by-design"
#: Rebuilds a statement without the field, deliberately, reason on the entry.
UNTHREADED_BY_DESIGN = "unthreaded-by-design"


@dataclass(frozen=True)
class Mechanism:
    """One classified mechanism, with the reason its verdict is what it is."""

    key: str
    kind: str
    verdict: str
    why: str
    #: ``module::test_name`` of a test that exercises this mechanism's handling
    #: of the marker, where one exists elsewhere.  Locked to exist, so deleting
    #: that test fails here rather than silently removing the coverage.
    covered_by: Optional[str] = None
    #: For a rebuild site, how many unthreaded constructions the detector should
    #: find in this function.  A drift in either direction is a finding.
    unthreaded_constructions: int = 0


_MECHANISMS: Tuple[Mechanism, ...] = (
    # ------------------------------------------------------------------
    # Readers and comparators that HANDLE the marker
    # ------------------------------------------------------------------
    Mechanism(
        "compiler/llir_traversal.py::_validate_result_storage_metadata",
        KIND_READER,
        HANDLES,
        "the marker's own validator; every managed pass runs it on every "
        "marker-carrying statement, which is what makes a forged marker fail "
        "closed rather than reach codegen",
        covered_by=(
            "test_scorch.test_llir_traversal::"
            "test_forged_result_storage_marker_fails_closed"
        ),
    ),
    Mechanism(
        "compiler/loopir/lower_llir.py::_result_storage_state_matches",
        KIND_READER,
        HANDLES,
        "IS the marker branch of the sparse-completion comparator: compares "
        "tensor_id and the references tuple by stored state",
    ),
    Mechanism(
        "compiler/loopir/lower_llir.py::_exact_sparse_completion_matches",
        KIND_READER,
        HANDLES,
        "exhaustive type table with a fall-through 'return False'; without its "
        "marker branch every marked statement would compare unequal and four "
        "call sites would raise sparse_workspace_completion_lost",
        covered_by=(
            "test_scorch.test_loopir_ordered_key_workspace_target::"
            "test_completion_comparator_handles_the_result_storage_marker"
        ),
    ),
    Mechanism(
        "compiler/loopir/lower_llir.py::_TargetLowering._exact_panel_state_matches",
        KIND_READER,
        HANDLES,
        "the same shape as the sparse-completion comparator and the sixth "
        "mechanism review section 64.3 found untaught; measured returning False "
        "for two structurally identical MARKED statements and True for the same "
        "pair unmarked, which turned a program that compiled before the marker "
        "existed into a structured refusal",
        covered_by=(
            "test_scorch.test_loopir_ordered_key_workspace_target::"
            "test_panel_comparator_handles_the_result_storage_marker"
        ),
    ),
    Mechanism(
        "compiler/loopir/lower_llir.py::_capture_sparse_completion_enum_states",
        KIND_READER,
        HANDLES,
        "the import-time enum-singleton snapshot the two comparators pin "
        "against; it covers ResultStorageArray and ResultStorageDirection, so a "
        "later mutation of either enum is observed rather than trusted",
    ),
    Mechanism(
        "compiler/loopir/lower_llir.py::_OrderedKeyExpectedBody._node",
        KIND_READER,
        HANDLES,
        "clones each stored field through _value, whose exhaustive type table "
        "RAISES on an unrecognized value; its ResultStorageMetadata branch "
        "detaches the marker so the expected body carries the same marker the "
        "actual side does",
    ),
    # ------------------------------------------------------------------
    # Readers that cannot see the marker, by type
    # ------------------------------------------------------------------
    Mechanism(
        "compiler/cin_lowerer.py::CINLowerer._coo_preallocation_receiver",
        KIND_READER,
        INERT,
        "vars() on an llir.Var; Var does not declare the field",
    ),
    Mechanism(
        "compiler/cin_lowerer.py::CINLowerer._transform_coo_loop_for_openmp",
        KIND_READER,
        INERT,
        "vars() on the loop's end-bound llir.Var; Var does not declare the field",
    ),
    Mechanism(
        "compiler/cin_lowerer.py::"
        "CINLowerer._transform_coo_loop_for_openmp.preallocation_statements",
        KIND_READER,
        INERT,
        "the same read, seen again in the nested scope that encloses it",
    ),
    Mechanism(
        "compiler/llir_traversal.py::_validate_tensor_access_metadata",
        KIND_READER,
        INERT,
        "validates TensorAccessMetadata, which describes an element ACCESS and "
        "is deliberately not this marker widened",
    ),
    Mechanism(
        "compiler/loopir/lower_llir.py::_TargetLowering._panel_loop_header_matches",
        KIND_READER,
        INERT,
        "reads an llir.ForLoop's stored state and delegates every field value to "
        "the panel comparator, which handles the marker",
    ),
    Mechanism(
        "compiler/loopir/lower_llir.py::_TargetLowering._panel_var_state_is_valid",
        KIND_READER,
        INERT,
        "an llir.Var's five stored fields; Var does not declare the field",
    ),
    Mechanism(
        "compiler/loopir/lower_llir.py::_metadata_state_matches",
        KIND_READER,
        INERT,
        "the access-provenance leaf of the sparse-completion comparator, a "
        "different type from the marker",
    ),
    Mechanism(
        "compiler/loopir/lower_llir.py::_require_ordered_key_completed_body",
        KIND_READER,
        INERT,
        "an exact four-field check on an llir.Function; Function does not "
        "declare the field, and the statements inside are compared by identity",
    ),
    Mechanism(
        "compiler/parallel_marking_pass.py::_validate_pool_spec_fields",
        KIND_READER,
        INERT,
        "a pass-local frozen dataclass, not an LLIR node",
    ),
    Mechanism(
        "compiler/parallel_marking_pass.py::atomic_work_stealing_prelude",
        KIND_READER,
        INERT,
        "the loop bound's llir.Var state; Var does not declare the field",
    ),
    Mechanism(
        "compiler/schedule_lowerer.py::_is_plain_prefetch_reference",
        KIND_READER,
        INERT,
        "vars() on an llir.Var; Var does not declare the field",
    ),
    Mechanism(
        "compiler/schedule_lowerer.py::_is_prefetch_int_literal",
        KIND_READER,
        INERT,
        "vars() on an llir.Literal; Literal does not declare the field",
    ),
    Mechanism(
        "compiler/torch_cpp_abi.py::"
        "ResultTensorAssembler.emit_first_compressed_position_allocation",
        KIND_READER,
        INERT,
        "an EXACT field-name set on an llir.Var -- the one reader shape a new "
        "declared field does break -- but on Var, which does not declare it",
    ),
    Mechanism(
        "compiler/torch_cpp_abi.py::"
        "ResultTensorAssembler.emit_deeper_compressed_position_allocations",
        KIND_READER,
        INERT,
        "the same exact field-name set on an llir.Var",
    ),
    Mechanism(
        "compiler/torch_cpp_abi.py::"
        "ResultTensorAssembler.emit_compressed_value_allocation",
        KIND_READER,
        INERT,
        "the same exact field-name set on an llir.Var",
    ),
    # ------------------------------------------------------------------
    # Readers that DO see a marker-carrying statement and tolerate the field
    # ------------------------------------------------------------------
    Mechanism(
        "compiler/loopir/lower_llir.py::_TargetLowering._nested_statement_lists",
        KIND_READER,
        INERT,
        "does read an llir.IfThenElse, which carries the field, but asks whether "
        "three named fields are PRESENT rather than whether the field set is "
        "exactly those three, so a new declared field passes; it extracts nested "
        "bodies and rebuilds nothing",
    ),
    Mechanism(
        "compiler/loopir/lower_llir.py::_TargetLowering.complete_panel",
        KIND_READER,
        INERT,
        "the same presence-not-exhaustiveness shape on an llir.VarInit's four "
        "named fields; its own reconstructions are classified separately below",
    ),
    Mechanism(
        "compiler/parallel_marking_pass.py::_validate_cluster_fields",
        KIND_READER,
        INERT,
        "validates VarInit and MemberCallStmt pool templates through .get(), so "
        "a new declared field passes; nothing marks a workspace pool template",
    ),
    # ------------------------------------------------------------------
    # The one reader that sees the marker and must keep ignoring it
    # ------------------------------------------------------------------
    Mechanism(
        "compiler/codegen.py::LLIRLowerer._validate_exact_codegen_tree",
        KIND_READER,
        BLIND_BY_DESIGN,
        "recurses vars(value), so it DOES visit result_storage -- and must go on "
        "doing nothing with it.  ResultStorageMetadata declares no base class, so "
        "it is not an llir.Node, not a str and not a sequence, and the function "
        "falls off the end.  Teaching this one would be teaching codegen to see "
        "compile-time-only metadata, which is exactly what byte-identical "
        "emission forbids.  THE COST: this validator gives the marker no "
        "coverage at all, so a forged marker arriving here is not caught here.  "
        "It is caught upstream, by _validate_result_storage_metadata, which every "
        "managed pass runs on every marker-carrying statement it walks",
        covered_by=(
            "test_scorch.test_llir_traversal::"
            "test_result_storage_marker_is_invisible_to_codegen"
        ),
    ),
    # ------------------------------------------------------------------
    # LLIRRewriter subclasses
    # ------------------------------------------------------------------
    Mechanism(
        "compiler/llir_traversal.py::LLIRRewriter",
        KIND_REWRITER,
        HANDLES,
        "the base class, and the one that is load-bearing everywhere: "
        "compressed_where_openmp_pass rewrites the whole work body through it at "
        "position 1 of the frozen LLIR order, immediately before RESULT_WRITE, so "
        "a marker it drops is a marker the guard never sees on every program",
    ),
    Mechanism(
        "compiler/result_write_pass.py::_ResultWriteRewriter",
        KIND_REWRITER,
        HANDLES,
        "the guard itself; it reads the marker rather than carrying it, and "
        "nothing it constructs carries one, which is what makes a surviving "
        "marker mean the rewrite left the statement alone",
    ),
    Mechanism(
        "compiler/dynamic_vector_access_pass.py::_DynamicVectorAccessRewriter",
        KIND_REWRITER,
        HANDLES,
        "overrides rewrite_assign, so the base class's fix does not cover it; it "
        "converts an indexed store into an append or a checked set and carries "
        "the marker onto whichever shape it produces",
    ),
    Mechanism(
        "compiler/compressed_where_openmp_pass.py::_WorkspaceInsertRewriter",
        KIND_REWRITER,
        HANDLES,
        "overrides rewrite_statement_sequence and renames one workspace insert "
        "call; the rename threads the field",
    ),
    Mechanism(
        "compiler/dense_pointer_hoist_pass.py::_DensePointerDetacher",
        KIND_REWRITER,
        HANDLES,
        "overrides _rewrite_stmt only to carry one legacy dynamic attribute, and "
        "delegates the statement itself to the base class",
    ),
    Mechanism(
        "compiler/schedule_lowerer.py::_TensorAccessRewriter",
        KIND_REWRITER,
        HANDLES,
        "overrides _rewrite_expr only; statements go through the base class",
    ),
    # ------------------------------------------------------------------
    # Statement rebuilds that thread the field
    # ------------------------------------------------------------------
    Mechanism(
        "compiler/cin_lowerer.py::CINLowerer._rewrite_val_refs",
        KIND_REBUILD,
        HANDLES,
        "rebuilds a MemberCallStmt and a FunctionCallStmt with rewritten "
        "operands; both thread the field.  On the legacy lowering chain, so a "
        "drop here reaches result_write_pass",
    ),
    Mechanism(
        "compiler/compressed_where_openmp_pass.py::"
        "_WorkspaceInsertRewriter._rewrite_legacy_statement",
        KIND_REBUILD,
        HANDLES,
        "the workspace-insert rename, at position 1 of the frozen LLIR order -- "
        "the one rebuild whose drop the guard itself would see",
    ),
    Mechanism(
        "compiler/dense_pointer_hoist_pass.py::_rewrite_statement_references",
        KIND_REBUILD,
        HANDLES,
        "rebuilds FunctionCallStmt, GuardedCallStmt.call and MemberCallStmt with "
        "rewritten references; Assign and VarInit are mutated in place and keep "
        "the field without help",
    ),
    Mechanism(
        "compiler/single_iteration_loop_pass.py::_rewrite_statement_references",
        KIND_REBUILD,
        HANDLES,
        "the same three call shapes as the dense-pointer pass",
    ),
    Mechanism(
        "compiler/schedule_lowerer.py::_rewrite_stmt_access_sequence",
        KIND_REBUILD,
        HANDLES,
        "rebuilds the three call shapes; Assign, VarInit and IfThenElse are "
        "mutated in place here",
    ),
    Mechanism(
        "compiler/loop_invariant_factor_pass.py::_replace_body_assignment",
        KIND_REBUILD,
        HANDLES,
        "rebuilds the same Assign with a hoisted invariant factor: same target, "
        "same direction, same arrays named by the target",
    ),
    Mechanism(
        "compiler/dynamic_vector_access_pass.py::"
        "_DynamicVectorAccessRewriter.rewrite_assign",
        KIND_REBUILD,
        HANDLES,
        "the three shapes an indexed store can become; review section 64.3 "
        "measured 4,710 markers changing statement type here with none lost",
    ),
    # ------------------------------------------------------------------
    # Statement rebuilds that deliberately do NOT thread the field
    # ------------------------------------------------------------------
    Mechanism(
        "compiler/loopir/lower_llir.py::_TargetLowering.complete_panel",
        KIND_REBUILD,
        UNTHREADED_BY_DESIGN,
        "the panel window's bound reconstruction: ONE VarInit becomes THREE, and "
        "its two carried values go to different ones -- the original initializer "
        "to the new row-end declaration, the original variable to the clamped "
        "upper bound.  A marker describes a statement, and there is no honest "
        "answer to which of the three inherits it, so a preserved marker would be "
        "a guess and _require_truthful_marker is the check that would catch the "
        "guess.  Nothing marks a loop bound today, so the population is empty",
        unthreaded_constructions=3,
    ),
    Mechanism(
        "compiler/schedule_lowerer.py::_window_sparse_loop",
        KIND_REBUILD,
        UNTHREADED_BY_DESIGN,
        "the legacy twin of the panel window's bound reconstruction, and "
        "unthreaded for the same reason",
        unthreaded_constructions=3,
    ),
    Mechanism(
        "compiler/loopir/parallel_chunk_assembly.py::_body_lambda",
        KIND_REBUILD,
        UNTHREADED_BY_DESIGN,
        "rebuilds the chunked loop's own init VarInit with the per-chunk lower "
        "bound; a loop iterator declaration, which the marker's vocabulary "
        "excludes and no lowering marks",
        unthreaded_constructions=1,
    ),
    # ------------------------------------------------------------------
    # Detector false positives: the copied attribute is not a statement field
    # ------------------------------------------------------------------
    Mechanism(
        "compiler/cin_lowerer.py::CINLowerer.lower_TensorAssign",
        KIND_REBUILD,
        INERT,
        "not a rebuild: the detector sees 'ivar.name', an IndexVar's name, which "
        "collides with FunctionCallStmt.name",
        unthreaded_constructions=1,
    ),
    Mechanism(
        "compiler/cin_lowerer.py::CINLowerer.lower_Where",
        KIND_REBUILD,
        INERT,
        "not a rebuild: 'wksp_ctype.value' is an enum member's value, which "
        "collides with VarInit.value",
        unthreaded_constructions=2,
    ),
    Mechanism(
        "compiler/loopir/lower_llir.py::"
        "_SparseWorkspaceLowering._workspace_init_statement",
        KIND_REBUILD,
        INERT,
        "not a rebuild: 'element_type.value' is a DataType member's value",
        unthreaded_constructions=1,
    ),
    Mechanism(
        "compiler/loopir/lower_llir.py::"
        "_OrderedKeySparseWorkspaceLowering._workspace_init_statement",
        KIND_REBUILD,
        INERT,
        "not a rebuild: two 'element_type.value' reads",
        unthreaded_constructions=2,
    ),
    Mechanism(
        "compiler/loopir/lower_llir.py::"
        "_ParallelSparseWorkspaceLowering._workspace_init_statement",
        KIND_REBUILD,
        INERT,
        "not a rebuild: 'element_type.value' again",
        unthreaded_constructions=1,
    ),
    Mechanism(
        "compiler/torch_cpp_abi.py::TorchCppKernelABI.emit_validation",
        KIND_REBUILD,
        INERT,
        "not a rebuild: 'tensor.name' is the ABI's own tensor name",
        unthreaded_constructions=1,
    ),
)


def _registry_by_kind(kind: str) -> Dict[str, Mechanism]:
    return {
        mechanism.key: mechanism for mechanism in _MECHANISMS if mechanism.kind == kind
    }


def test_the_registry_has_no_duplicate_keys() -> None:
    """One key can appear under two kinds and must not appear twice under one."""

    for kind in (KIND_READER, KIND_REBUILD, KIND_REWRITER):
        keys = [m.key for m in _MECHANISMS if m.kind == kind]
        assert len(keys) == len(set(keys)), f"duplicate {kind} keys: {keys}"
    for mechanism in _MECHANISMS:
        assert mechanism.why.strip(), f"{mechanism.key} carries no reason"
        assert mechanism.verdict in (
            HANDLES,
            INERT,
            BLIND_BY_DESIGN,
            UNTHREADED_BY_DESIGN,
        )


# --------------------------------------------------------------------------
# The static detectors
# --------------------------------------------------------------------------

_PACKAGE_ROOT = pathlib.Path(scorch.__file__).resolve().parent


def _package_sources() -> List[Tuple[str, ast.Module]]:
    """Every module in the installed ``scorch`` package, parsed once."""

    parsed: List[Tuple[str, ast.Module]] = []
    for path in sorted(_PACKAGE_ROOT.rglob("*.py")):
        relative = path.relative_to(_PACKAGE_ROOT).as_posix()
        parsed.append((relative, ast.parse(path.read_text())))
    return parsed


_SOURCES = _package_sources()


def _base_name(node: ast.AST) -> Optional[str]:
    """The variable a field read is rooted at, seeing through ``cast``.

    ``cast(llir.VarInit, ranged.init).var`` roots at ``ranged``.  Learning this
    case is what found ``parallel_chunk_assembly._body_lambda``, which the first
    version of this detector missed -- the standing evidence that a syntactic
    detector's recall is a claim about the scanner rather than about the tree.
    """

    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return _base_name(node.value)
    if isinstance(node, ast.Call):
        function = node.func
        if isinstance(function, ast.Name) and function.id == "cast" and node.args:
            return _base_name(node.args[-1])
    return None


class _MechanismScan(ast.NodeVisitor):
    """One module's generic field reads and marker-carrying statement builds."""

    #: The ways a mechanism can read a statement's fields without naming them:
    #: ``compare=False`` hides ``result_storage`` from ``==`` and ``hash`` but
    #: from none of these.
    GENERIC_READS = ("__dict__", "vars", "get_type_hints", "fields")

    def __init__(self, module: str) -> None:
        self.module = module
        self._scope: List[str] = []
        #: qualname -> the generic reads found in it, and the llir names it uses
        self.readers: Dict[str, Tuple[Set[str], Set[str]]] = {}
        #: (qualname, statement class, threaded, copied fields, base variable)
        self.builds: List[Tuple[str, str, bool, FrozenSet[str], str]] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._scope.append(node.name)
        self.generic_visit(node)
        self._scope.pop()

    def _visit_function(self, node: ast.AST) -> None:
        name = getattr(node, "name", "<lambda>")
        self._scope.append(name)
        reads: Set[str] = set()
        llir_names: Set[str] = set()
        for child in ast.walk(node):
            if isinstance(child, ast.Attribute) and child.attr == "__dict__":
                reads.add("__dict__")
            elif isinstance(child, ast.Constant) and child.value == "__dict__":
                reads.add("__dict__")
            elif isinstance(child, ast.Call):
                function_name = ast.unparse(child.func)
                leaf = function_name.rsplit(".", 1)[-1]
                if leaf == "vars":
                    reads.add("vars")
                elif leaf == "get_type_hints":
                    reads.add("get_type_hints")
                elif leaf == "fields" and "dataclass" in function_name:
                    reads.add("fields")
            if (
                isinstance(child, ast.Attribute)
                and isinstance(child.value, ast.Name)
                and child.value.id == "llir"
            ):
                llir_names.add(child.attr)
        if reads:
            self.readers[".".join(self._scope)] = (reads, llir_names)
        self.generic_visit(node)
        self._scope.pop()

    visit_FunctionDef = _visit_function  # type: ignore[assignment]
    visit_AsyncFunctionDef = _visit_function  # type: ignore[assignment]

    def visit_Call(self, node: ast.Call) -> None:
        spelling = ast.unparse(node.func)
        statement_class = spelling.rsplit(".", 1)[-1]
        if statement_class in MARKER_CARRYING_TYPES and spelling in (
            statement_class,
            f"llir.{statement_class}",
        ):
            declared = DECLARED_FIELDS[statement_class]
            per_base: Dict[str, Set[str]] = {}
            for child in ast.walk(node):
                base: Optional[str] = None
                field: Optional[str] = None
                if isinstance(child, ast.Attribute):
                    base, field = _base_name(child.value), child.attr
                elif (
                    isinstance(child, ast.Call)
                    and ast.unparse(child.func) == "getattr"
                    and len(child.args) >= 2
                    and isinstance(child.args[1], ast.Constant)
                ):
                    base = _base_name(child.args[0])
                    constant = child.args[1].value
                    field = constant if isinstance(constant, str) else None
                if (
                    base is None
                    or field is None
                    or base == "self"
                    or field not in declared
                ):
                    continue
                per_base.setdefault(base, set()).add(field)
            if per_base:
                base, fields = max(per_base.items(), key=lambda item: len(item[1]))
                threaded = any(
                    keyword.arg == "result_storage" for keyword in node.keywords
                )
                self.builds.append(
                    (
                        ".".join(self._scope),
                        statement_class,
                        threaded,
                        frozenset(fields),
                        base,
                    )
                )
        self.generic_visit(node)


def _scan_all() -> List[_MechanismScan]:
    scans = []
    for module, tree in _SOURCES:
        scan = _MechanismScan(module)
        scan.visit(tree)
        scans.append(scan)
    return scans


_SCANS = _scan_all()


def _discovered_readers() -> Dict[str, Tuple[Set[str], Set[str]]]:
    """Generic field readers whose function also names an ``llir`` type.

    The ``llir`` qualifier is the scope rule: a function that reads stored
    fields generically and never mentions an LLIR type is reading something
    else -- a CIN node, a format, a schedule, a storage object -- and those live
    in type universes the marker does not reach.
    """

    found: Dict[str, Tuple[Set[str], Set[str]]] = {}
    for scan in _SCANS:
        for qualname, (reads, llir_names) in scan.readers.items():
            if llir_names:
                found[f"{scan.module}::{qualname}"] = (reads, llir_names)
    return found


def _discovered_builds() -> List[Tuple[str, str, bool, FrozenSet[str], str]]:
    rows = []
    for scan in _SCANS:
        for qualname, cls, threaded, fields, base in scan.builds:
            rows.append((f"{scan.module}::{qualname}", cls, threaded, fields, base))
    return rows


# --------------------------------------------------------------------------
# Closure: a seventh mechanism has to be classified
# --------------------------------------------------------------------------


def test_every_generic_field_reader_over_llir_is_classified() -> None:
    """The set of field-by-field readers is locked in both directions.

    This is the test that fires for a mechanism nobody has written yet.  Adding
    a function that reads an LLIR node's stored fields generically -- through
    ``__dict__``, ``vars``, ``get_type_hints`` or ``dataclasses.fields`` -- fails
    here until it is classified in :data:`_MECHANISMS` with a verdict and a
    reason.  Deleting one fails here too, so the registry cannot rot into a list
    of functions that no longer exist.
    """

    discovered = set(_discovered_readers())
    registered = set(_registry_by_kind(KIND_READER))

    unclassified = sorted(discovered - registered)
    assert not unclassified, (
        "a mechanism reads an LLIR node's fields generically and is not "
        "classified.  compare=False hides result_storage from == and hash, not "
        "from __dict__, so decide what this one does with the marker and add it "
        "to _MECHANISMS with a verdict and a reason:\n  " + "\n  ".join(unclassified)
    )
    stale = sorted(registered - discovered)
    assert not stale, (
        "_MECHANISMS classifies readers that no longer exist; delete them or fix "
        "the key:\n  " + "\n  ".join(stale)
    )


def test_every_statement_rebuild_threads_the_marker_or_is_classified() -> None:
    """A rebuild either passes ``result_storage=`` or is registered.

    The default is threading the field.  A rebuild that does not is a landmine
    of exactly the shape review section 64.3's sixth mechanism turned out to be,
    so each one has to say at this registry why it is not, and how many
    constructions in that function the detector should be finding.
    """

    exceptions = {
        mechanism.key: mechanism
        for mechanism in _MECHANISMS
        if mechanism.kind == KIND_REBUILD
        and mechanism.verdict in (UNTHREADED_BY_DESIGN, INERT)
    }
    counted: Dict[str, int] = {}
    unclassified: List[str] = []
    for key, cls, threaded, fields, base in _discovered_builds():
        if threaded:
            continue
        if key in exceptions:
            counted[key] = counted.get(key, 0) + 1
            continue
        unclassified.append(f"{key} builds {cls} copying {sorted(fields)} off {base!r}")

    assert not unclassified, (
        "a construction of a marker-carrying statement type copies fields off "
        "an existing node and does not pass result_storage=.  If it rebuilds the "
        "same statement, thread the field.  If it does not, register it in "
        "_MECHANISMS with the reason:\n  " + "\n  ".join(sorted(unclassified))
    )
    for key, mechanism in sorted(exceptions.items()):
        assert counted.get(key, 0) == mechanism.unthreaded_constructions, (
            f"{key} was registered with "
            f"{mechanism.unthreaded_constructions} unthreaded constructions and "
            f"the tree now has {counted.get(key, 0)}; re-read the site rather "
            "than adjusting the count"
        )


def test_every_llir_rewriter_subclass_is_classified() -> None:
    """Subclassing ``LLIRRewriter`` is the house way to rebuild statements.

    Discovered at runtime from the class hierarchy rather than from the source
    text, after importing every module in the package, so a subclass declared
    anywhere is found.  ``LLIRRewriter`` itself is registered too, since it is
    the mechanism the whole marker rests on.
    """

    for module, _ in _SOURCES:
        if module.endswith("__init__.py"):
            continue
        dotted = "scorch." + module[: -len(".py")].replace("/", ".")
        try:
            import_module(dotted)
        except Exception:  # pragma: no cover - an optional or broken module
            continue

    discovered: Set[str] = set()
    frontier = [LLIRRewriter]
    while frontier:
        current = frontier.pop()
        module_path = current.__module__.replace(".", "/")
        assert module_path.startswith("scorch/"), module_path
        discovered.add(f"{module_path[len('scorch/'):]}.py::{current.__qualname__}")
        frontier.extend(current.__subclasses__())

    registered = set(_registry_by_kind(KIND_REWRITER))
    unclassified = sorted(discovered - registered)
    assert not unclassified, (
        "an LLIRRewriter subclass is not classified.  Every rewrite_* method "
        "reconstructs its statement from an explicit field list, so decide "
        "whether this one threads result_storage and register it:\n  "
        + "\n  ".join(unclassified)
    )
    stale = sorted(registered - discovered)
    assert not stale, (
        "_MECHANISMS classifies rewriter subclasses that no longer exist:\n  "
        + "\n  ".join(stale)
    )


def test_every_registered_mechanism_with_a_named_test_still_has_it() -> None:
    """A ``covered_by`` reference has to resolve, or the coverage is gone."""

    for mechanism in _MECHANISMS:
        if mechanism.covered_by is None:
            continue
        module_name, _, test_name = mechanism.covered_by.partition("::")
        module = import_module(module_name)
        covering = getattr(module, test_name, None)
        assert callable(covering), (
            f"{mechanism.key} names {mechanism.covered_by} as its coverage and "
            "that test no longer exists"
        )


# --------------------------------------------------------------------------
# Behaviour: drive each threading rebuild and watch the marker come out
# --------------------------------------------------------------------------

#: One marked statement handed in, and the statement the mechanism produced.
Probe = Tuple[llir.Stmt, llir.Stmt]


def _marker(tensor: int = 11) -> llir.ResultStorageMetadata:
    return llir.ResultStorageMetadata(
        tensor_id=SymbolId(tensor),
        references=(
            llir.ResultStorageReference(
                llir.ResultStorageArray.CRD,
                1,
                llir.ResultStorageDirection.WRITE,
            ),
        ),
    )


def _var(name: str, data_type: llir.DataType = llir.DataType.NO_TYPE) -> llir.Var:
    return llir.Var(name=name, type=data_type)


def _statement(value: object) -> llir.Stmt:
    assert isinstance(value, llir.Stmt), value
    return value


Marker = Optional[llir.ResultStorageMetadata]


def _call(marker: Marker, name: str = "C1_crd.push_back") -> llir.FunctionCallStmt:
    """A call whose one argument is the string both reference passes replace."""

    return llir.FunctionCallStmt(name, [_var("_old")], result_storage=marker)


def _member_call(marker: Marker) -> llir.MemberCallStmt:
    return llir.MemberCallStmt(
        _var("C1_crd"), "push_back", args=[_var("_old")], result_storage=marker
    )


def _indexed_store(marker: Marker, vector: str) -> llir.Assign:
    return llir.Assign(
        var=llir.ArrayAccess(array=_var(vector), index=_var("pC1")),
        value=llir.Literal(7),
        result_storage=marker,
    )


# ---- the base rewriter, once per marker-carrying statement type ----------


def _identity_rewrite(statement: llir.Stmt) -> llir.Stmt:
    from scorch.compiler.llir_traversal import LLIRTraversalContext

    context = LLIRTraversalContext(
        stage="LLIR transformation", pass_name="result_storage_marker_probe"
    )
    return _statement(LLIRRewriter(context).rewrite([statement])[0])


def _probe_rewriter_call(marker: Marker) -> Probe:
    handed_in = llir.FunctionCallStmt(
        "C1_crd.push_back", [llir.Literal(1)], result_storage=marker
    )
    return handed_in, _identity_rewrite(handed_in)


def _probe_rewriter_assign(marker: Marker) -> Probe:
    handed_in = _indexed_store(marker, "C1_crd")
    return handed_in, _identity_rewrite(handed_in)


def _probe_rewriter_var_init(marker: Marker) -> Probe:
    handed_in = llir.VarInit(
        var=_var("count", llir.DataType.INT),
        value=llir.Literal(0),
        result_storage=marker,
    )
    return handed_in, _identity_rewrite(handed_in)


def _probe_rewriter_member_call(marker: Marker) -> Probe:
    handed_in = llir.MemberCallStmt(
        _var("C1_crd"), "push_back", args=[llir.Literal(1)], result_storage=marker
    )
    return handed_in, _identity_rewrite(handed_in)


def _probe_rewriter_if(marker: Marker) -> Probe:
    handed_in = llir.IfThenElse(
        cond=llir.Literal(1), then_body=[], result_storage=marker
    )
    return handed_in, _identity_rewrite(handed_in)


# ---- the one subclass that overrides a per-statement rewrite method ------


def _dynamic_vector_rewrite(handed_in: llir.Assign, vector: str) -> llir.Stmt:
    from scorch.compiler.dynamic_vector_access_pass import (
        DYNAMIC_VECTOR_ACCESS_CONTEXT,
        rewrite_dynamic_vector_accesses,
    )

    body: List[llir.Stmt] = [
        llir.VarDecl(_var(vector, llir.DataType.STD_VECTOR_INT)),
        handed_in,
    ]
    rewritten = rewrite_dynamic_vector_accesses(body, DYNAMIC_VECTOR_ACCESS_CONTEXT)
    produced = _statement(rewritten[1])
    assert type(produced) is llir.FunctionCallStmt, produced
    return produced


def _probe_dynamic_vector_append(marker: Marker) -> Probe:
    """``_crd`` is an append suffix, so the indexed store becomes an append."""

    handed_in = _indexed_store(marker, "C1_crd")
    produced = _dynamic_vector_rewrite(handed_in, "C1_crd")
    assert produced.name == "C1_crd.emplace_back"  # type: ignore[attr-defined]
    return handed_in, produced


def _probe_dynamic_vector_checked_set(marker: Marker) -> Probe:
    """Any other suffix takes the other branch, a checked set."""

    handed_in = _indexed_store(marker, "C1_scratch")
    produced = _dynamic_vector_rewrite(handed_in, "C1_scratch")
    assert produced.name == "scorch_vector_set"  # type: ignore[attr-defined]
    return handed_in, produced


# ---- the legacy CIN lowerer's value-reference rewrite --------------------


def _val_ref_rewrite(handed_in: llir.Stmt) -> llir.Stmt:
    from scorch.compiler.cin_lowerer import CINLowerer

    body: List[llir.Stmt] = [handed_in]
    CINLowerer._rewrite_val_refs(body, {"_old": "_new"})
    return _statement(body[0])


def _probe_val_refs_call(marker: Marker) -> Probe:
    handed_in = _call(marker)
    return handed_in, _val_ref_rewrite(handed_in)


def _probe_val_refs_member_call(marker: Marker) -> Probe:
    handed_in = _member_call(marker)
    return handed_in, _val_ref_rewrite(handed_in)


# ---- the workspace-insert rename, at position 1 of the frozen order ------


def _probe_workspace_insert_rename(marker: Marker) -> Probe:
    import torch

    from scorch.compiler.compressed_where_openmp_pass import (
        CompressedWhereOpenMPContext,
        _WorkspaceInsertRewriter,
    )
    from scorch.compiler.torch_cpp_abi import ResultTensorAssembler
    from scorch.format import LevelType

    context = CompressedWhereOpenMPContext(
        result_name="Result",
        result_id=SymbolId(1),
        compressed_levels=(1,),
        result_assembler=ResultTensorAssembler(
            name="Result",
            level_types=(LevelType.DENSE, LevelType.COMPRESSED),
            dtype=torch.float32,
        ),
        workspace_name="wksp",
        workspace_ctype="float",
    )
    handed_in = llir.FunctionCallStmt(
        "wksp.insert", [llir.Literal(1)], result_storage=marker
    )
    rewritten = _WorkspaceInsertRewriter(context).rewrite_statement_sequence(
        [handed_in], ("body",)
    )
    produced = _statement(list(rewritten)[0])
    assert type(produced) is llir.FunctionCallStmt, produced
    assert produced.name == "wksp.insert_unchecked"
    return handed_in, produced


# ---- the two reference-rewrite passes, which share three shapes ----------


def _reference_rewrite(module_name: str, handed_in: llir.Stmt) -> llir.Stmt:
    """Drive either reference-rewrite pass's statement rebuild.

    The two passes have the same private helper name and the same three call
    shapes; their replacement records differ by one field name, which is read off
    the dataclass rather than restated.
    """

    module = import_module(module_name)
    replacements_type = getattr(module, "_ReferenceReplacements")
    keywords: Dict[str, object] = {"generated_strings": (("_old", "_new"),)}
    for field in get_type_hints(replacements_type):
        keywords.setdefault(field, ())
    if module_name.endswith("dense_pointer_hoist_pass"):
        context: object = module.DensePointerHoistContext(value_array_ctypes=())
    else:
        context = module.SINGLE_ITERATION_LOOP_ELIMINATION_CONTEXT
    rewritten = module._rewrite_statement_references(
        [handed_in], replacements_type(**keywords), context, ("body",)
    )
    return _statement(list(rewritten)[0])


def _guarded(marker: Marker) -> Tuple[llir.FunctionCallStmt, llir.GuardedCallStmt]:
    inner = _call(marker)
    guard = llir.BinOp("<", _var("pC1", llir.DataType.INT64), llir.Literal(4))
    return inner, llir.GuardedCallStmt(cond=guard, call=inner)


def _probe_dense_pointer_call(marker: Marker) -> Probe:
    handed_in = _call(marker)
    module = "scorch.compiler.dense_pointer_hoist_pass"
    return handed_in, _reference_rewrite(module, handed_in)


def _probe_dense_pointer_member_call(marker: Marker) -> Probe:
    handed_in = _member_call(marker)
    module = "scorch.compiler.dense_pointer_hoist_pass"
    return handed_in, _reference_rewrite(module, handed_in)


def _probe_dense_pointer_guarded_call(marker: Marker) -> Probe:
    inner, guarded = _guarded(marker)
    module = "scorch.compiler.dense_pointer_hoist_pass"
    produced = _reference_rewrite(module, guarded)
    assert type(produced) is llir.GuardedCallStmt, produced
    return inner, produced.call


def _probe_single_iteration_call(marker: Marker) -> Probe:
    handed_in = _call(marker)
    module = "scorch.compiler.single_iteration_loop_pass"
    return handed_in, _reference_rewrite(module, handed_in)


def _probe_single_iteration_member_call(marker: Marker) -> Probe:
    handed_in = _member_call(marker)
    module = "scorch.compiler.single_iteration_loop_pass"
    return handed_in, _reference_rewrite(module, handed_in)


def _probe_single_iteration_guarded_call(marker: Marker) -> Probe:
    inner, guarded = _guarded(marker)
    module = "scorch.compiler.single_iteration_loop_pass"
    produced = _reference_rewrite(module, guarded)
    assert type(produced) is llir.GuardedCallStmt, produced
    return inner, produced.call


# ---- the schedule lowerer's tensor-access rewrite ------------------------


def _access_rewrite(handed_in: llir.Stmt) -> llir.Stmt:
    from scorch.compiler.cin import IndexId
    from scorch.compiler.schedule_lowerer import _rewrite_stmt_access_sequence

    rewritten, _ = _rewrite_stmt_access_sequence(
        [handed_in],
        SymbolId(4_000_900),
        (IndexId(1),),
        llir.TensorAccessRole.INPUT_READ,
        llir.Literal(0),
    )
    return _statement(list(rewritten)[0])


def _probe_schedule_lowerer_call(marker: Marker) -> Probe:
    handed_in = _call(marker)
    return handed_in, _access_rewrite(handed_in)


def _probe_schedule_lowerer_member_call(marker: Marker) -> Probe:
    handed_in = _member_call(marker)
    return handed_in, _access_rewrite(handed_in)


def _probe_schedule_lowerer_guarded_call(marker: Marker) -> Probe:
    inner, guarded = _guarded(marker)
    produced = _access_rewrite(guarded)
    assert type(produced) is llir.GuardedCallStmt, produced
    return inner, produced.call


# ---- the loop-invariant hoist's assignment replacement -------------------


def _probe_loop_invariant_assign(marker: Marker) -> Probe:
    from scorch.compiler.loop_invariant_factor_pass import _replace_body_assignment

    handed_in = llir.Assign(
        var=llir.ArrayAccess(array=_var("C1_crd"), index=_var("p")),
        value=llir.Mul(_var("a"), _var("b")),
        result_storage=marker,
    )
    loop = llir.ForLoop(
        init=llir.VarInit(_var("i", llir.DataType.INT), llir.Literal(0)),
        cond=llir.BinOp("<", _var("i"), llir.Literal(4)),
        update=llir.Increment(_var("i")),
        body=[handed_in],
    )
    _replace_body_assignment(loop, 0, handed_in, _var("hoisted"))
    return handed_in, _statement(loop.body[0])


_REBUILD_PROBES = {
    "LLIRRewriter/Assign": _probe_rewriter_assign,
    "LLIRRewriter/FunctionCallStmt": _probe_rewriter_call,
    "LLIRRewriter/IfThenElse": _probe_rewriter_if,
    "LLIRRewriter/MemberCallStmt": _probe_rewriter_member_call,
    "LLIRRewriter/VarInit": _probe_rewriter_var_init,
    "cin_lowerer/_rewrite_val_refs/FunctionCallStmt": _probe_val_refs_call,
    "cin_lowerer/_rewrite_val_refs/MemberCallStmt": _probe_val_refs_member_call,
    "compressed_where_openmp/insert_rename": _probe_workspace_insert_rename,
    "dense_pointer_hoist/FunctionCallStmt": _probe_dense_pointer_call,
    "dense_pointer_hoist/GuardedCallStmt.call": _probe_dense_pointer_guarded_call,
    "dense_pointer_hoist/MemberCallStmt": _probe_dense_pointer_member_call,
    "dynamic_vector_access/append": _probe_dynamic_vector_append,
    "dynamic_vector_access/checked_set": _probe_dynamic_vector_checked_set,
    "loop_invariant_factor/Assign": _probe_loop_invariant_assign,
    "schedule_lowerer/FunctionCallStmt": _probe_schedule_lowerer_call,
    "schedule_lowerer/GuardedCallStmt.call": _probe_schedule_lowerer_guarded_call,
    "schedule_lowerer/MemberCallStmt": _probe_schedule_lowerer_member_call,
    "single_iteration_loop/FunctionCallStmt": _probe_single_iteration_call,
    "single_iteration_loop/GuardedCallStmt.call": _probe_single_iteration_guarded_call,
    "single_iteration_loop/MemberCallStmt": _probe_single_iteration_member_call,
}


@pytest.mark.parametrize("probe", sorted(_REBUILD_PROBES))
def test_a_rebuild_carries_the_marker_across(probe: str) -> None:
    """Drive one rebuild and assert the marker arrives on what it produced.

    Two assertions, and the first is what makes the second mean something.
    ``produced is not handed_in`` proves a rebuild actually happened: a probe
    that landed on a mutate-in-place branch instead would otherwise pass while
    testing nothing, since the marker would still be on the object it was
    attached to.
    """

    marker = _marker()
    handed_in, produced = _REBUILD_PROBES[probe](marker)

    assert produced is not handed_in, (
        f"{probe} handed back the statement it was given, so this probe does "
        "not reach a rebuild and proves nothing about threading the field"
    )
    assert produced.result_storage == marker, (
        f"{probe} dropped the result-storage marker.  A rebuild that omits the "
        "field silently un-marks a result write, which is the one failure mode "
        "the marker exists to prevent: review section 64.4 measured it as 764 "
        "markers arriving at the guard becoming 0, with the emitted C++ "
        "byte-identical, the suite green and the static checks clean"
    )


@pytest.mark.parametrize("probe", sorted(_REBUILD_PROBES))
def test_a_rebuild_invents_no_marker(probe: str) -> None:
    """The same rebuilds hand back ``None`` when nothing was marked.

    Not redundant with the test above: a rebuild that hardcoded a marker rather
    than threading the field would pass that one and fail this one.
    """

    handed_in, produced = _REBUILD_PROBES[probe](None)

    assert handed_in.result_storage is None
    assert produced.result_storage is None, (
        f"{probe} produced a marker from an unmarked statement, so it is not "
        "threading the field but inventing one"
    )


# --------------------------------------------------------------------------
# The ABI's per-result identity, and the two sides that must move together
# --------------------------------------------------------------------------


def _assembler(result_id: object = None) -> object:
    """One two-level CSR-shaped result assembler, optionally given an identity."""

    import torch

    from scorch.compiler.torch_cpp_abi import (  # type: ignore[import-untyped]
        ResultTensorAssembler,
    )
    from scorch.format import LevelType  # type: ignore[import-untyped]

    return ResultTensorAssembler(
        name="C",
        level_types=(LevelType.DENSE, LevelType.COMPRESSED),
        dtype=torch.float32,
        result_id=result_id,
    )


def _position_sentinel_markers(
    assembler: object,
) -> List[Optional[llir.ResultStorageMetadata]]:
    """The markers on the ``C{L}_pos[0] = 0`` sentinels the assembler emits."""

    return [
        statement.result_storage
        for statement in assembler.emit_level_indices_init()  # type: ignore[attr-defined]
        if type(statement) is llir.Assign
        and type(statement.var) is llir.ArrayAccess
        and type(statement.var.array) is llir.Var
        and statement.var.array.name.endswith("_pos")
    ]


def test_the_abi_marks_nothing_without_an_identity() -> None:
    """No identity, no marker -- which is what keeps the legacy route unchanged.

    ``ResultTensorAssembler`` builds every storage name from ``self.name`` and a
    level number, and a name says which vector a statement touches, not whose.
    The legacy route builds this assembler from a ``TensorVar`` whose identity IS
    a name, so it passes no ``SymbolId`` and nothing it emits gains a marker.
    """

    markers = _position_sentinel_markers(_assembler())

    assert markers, "the fixture must reach at least one position sentinel"
    assert all(marker is None for marker in markers)
    assert (
        _assembler().result_storage_marker(  # type: ignore[attr-defined]
            (llir.ResultStorageArray.POS, 1, True)
        )
        is None
    )


def test_the_abi_marks_its_position_sentinel_from_the_identity() -> None:
    """With an identity, the sentinel says whose position array it writes."""

    markers = _position_sentinel_markers(_assembler(SymbolId(7)))

    assert markers
    for marker in markers:
        assert marker is not None
        assert type(marker) is llir.ResultStorageMetadata
        assert marker.tensor_id == SymbolId(7)
        assert marker.writes() is True
        assert [
            (reference.array, reference.level, reference.direction)
            for reference in marker.references
        ] == [
            (
                llir.ResultStorageArray.POS,
                1,
                llir.ResultStorageDirection.WRITE,
            )
        ]


def test_the_abi_rebuilds_the_identity_rather_than_sharing_it() -> None:
    """Nothing reachable from an emitted node aliases the caller's identity.

    The discipline every other marker builder in the compiler follows, and the
    reason is that a managed pass must own no program state.
    """

    identity = SymbolId(7)
    marker = _assembler(identity).result_storage_marker(  # type: ignore[attr-defined]
        (llir.ResultStorageArray.POS, 1, True)
    )

    assert marker.tensor_id == identity
    assert marker.tensor_id is not identity


def test_the_abi_refuses_an_identity_that_is_not_a_symbol_id() -> None:
    """Fail closed on the field, the way every other ABI field does."""

    with pytest.raises(TypeError, match="result identity"):
        _assembler(7)


def test_both_sides_of_the_position_completion_carry_the_same_marker() -> None:
    """THE TRAP, locked.

    The serial sparse-workspace families compare an assembled function against a
    completion reference they build themselves, FOR EQUALITY.  The assembler emits
    ``C1_pos[0] = 0``; the dynamic-vector pass rewrites it into
    ``scorch_vector_set(C1_pos, 0, 0)`` carrying its marker; and
    ``_completed_position_init_statement`` reproduces that rewrite as the
    reference.  Mark one side and not the other and the two differ, and a program
    that compiles today becomes a ``sparse_workspace_completion_lost`` refusal.

    So the two markers are asserted EQUAL here.  Changing either side's reference
    triple, or its identity, or removing either marker, fails this test -- which
    is the whole point of asserting it rather than reading the two sites and
    believing they agree.

    The reference side is driven unbound against a stub carrying only the two
    attributes it reads, because building a real lowering costs a LoopIR plan and
    several seconds and would prove nothing more.
    """

    from types import SimpleNamespace

    from scorch.compiler.loopir.lower_llir import (  # type: ignore[import-untyped]
        _SparseWorkspaceLowering,
        _TargetLowering,
    )

    class _ReferenceSide:
        result_symbol = SymbolId(7)
        result_decl = SimpleNamespace(name="C")
        _result_storage = _TargetLowering._result_storage

    reference = _SparseWorkspaceLowering._completed_position_init_statement(
        _ReferenceSide(), 1
    )
    emitted_markers = _position_sentinel_markers(_assembler(SymbolId(7)))

    assert type(reference) is llir.FunctionCallStmt
    assert reference.name == "scorch_vector_set"
    assert reference.result_storage is not None
    assert len(emitted_markers) == 1
    assert reference.result_storage == emitted_markers[0], (
        "the completion reference and the emitted statement carry different "
        "result-storage markers, so the completion comparison will refuse a "
        "program that compiles today.  Both sides move together or neither does"
    )


#: Where a ``ResultTensorAssembler`` is built without handing over an identity,
#: and why that is right there.  Keyed by ``module::function``.
_IDENTITY_FREE_ASSEMBLERS = {
    "compiler/cin_lowerer.py::_result_tensor_abi_assembler": (
        "the legacy route, whose result identity IS a name: it snapshots a "
        "TensorVar, which has no SymbolId to hand over.  Nothing legacy emits "
        "gains a marker, and that is what keeps the descriptor off the shipping "
        "path -- measured, not assumed: 764 markers arrive at the guard on the "
        "legacy route and the emission digest is unchanged"
    ),
}


def test_every_result_assembler_hands_over_an_identity_or_is_registered() -> None:
    """A construction that forgets ``result_id`` breaks the completion comparison.

    The mutation this closes: dropping ``result_id=`` from one
    ``result_assembler()`` override leaves the assembler emitting an UNMARKED
    ``C{L}_pos[0] = 0`` while ``_completed_position_init_statement`` still marks
    its reference, so the two sides differ and a program that compiles today
    becomes a refusal.  Every behavioural test above builds its own assembler with
    an explicit identity, so none of them can see that.  This can, and it also
    covers an override written later.
    """

    unclassified = []
    for module, tree in _SOURCES:
        scope: List[str] = []

        class _Visit(ast.NodeVisitor):
            def visit_ClassDef(self, node: ast.ClassDef) -> None:
                scope.append(node.name)
                self.generic_visit(node)
                scope.pop()

            def _function(self, node: ast.AST) -> None:
                scope.append(getattr(node, "name", "<lambda>"))
                self.generic_visit(node)
                scope.pop()

            visit_FunctionDef = _function  # type: ignore[assignment]
            visit_AsyncFunctionDef = _function  # type: ignore[assignment]

            def visit_Call(self, node: ast.Call) -> None:
                if ast.unparse(node.func).rsplit(".", 1)[-1] == (
                    "ResultTensorAssembler"
                ):
                    hands_over = any(
                        keyword.arg == "result_id" for keyword in node.keywords
                    )
                    if not hands_over:
                        key = f"{module}::{'.'.join(scope)}"
                        if key not in _IDENTITY_FREE_ASSEMBLERS:
                            unclassified.append(f"{key} (line {node.lineno})")
                self.generic_visit(node)

        _Visit().visit(tree)

    assert not unclassified, (
        "a ResultTensorAssembler is built without handing over result_id.  Pass "
        "the result's SymbolId so the statements it emits can be marked, or "
        "register the site in _IDENTITY_FREE_ASSEMBLERS with the reason it has "
        "no identity to hand over:\n  " + "\n  ".join(sorted(unclassified))
    )


#: Every statement the ABI emits that a lowering ALSO hand-builds as a completion
#: reference, and the marker both sides must carry.  The comparison between them is
#: for equality, so a marker on one side alone turns a program that compiles today
#: into a refusal.  ``reference_site`` is ``module::function::local`` naming the
#: hand-built side, asserted below to pass ``result_storage=``.
_PAIRED_COMPLETION_REFERENCES = (
    (
        "the level-index position sentinel: C{L}_pos[0] = 0, which the "
        "dynamic-vector pass rewrites into scorch_vector_set",
        (llir.ResultStorageArray.POS, 1, True),
        "compiler/loopir/lower_llir.py" "::_completed_position_init_statement",
    ),
    (
        "the dense-result zero fill: scorch_zero_dense(C_values, C_capacity), a "
        "free function writing the value vector through an ARGUMENT",
        (llir.ResultStorageArray.VALUES, None, True),
        "compiler/loopir/lower_llir.py::_complete_result_tile_impl::expected_zero",
    ),
)


#: How the reference side spells the array and the direction, for the AST check.
_ARRAY_SPELLINGS = {
    "RESULT_VALUES": llir.ResultStorageArray.VALUES,
    "RESULT_POS": llir.ResultStorageArray.POS,
    "RESULT_CRD": llir.ResultStorageArray.CRD,
}
_DIRECTION_SPELLINGS = {
    "STORAGE_WRITE": True,
    "STORAGE_READ": False,
}


def _reference_triples_at(site: str) -> List[Tuple[object, object, object]]:
    """The ``(array, level, writes)`` triples a hand-built reference passes.

    Read off the source, because both reference builders sit inside functions that
    need a whole LoopIR plan to call.  The level is returned as ``None`` when it is
    a variable rather than a literal, which is the case for the position sentinel,
    whose level is its argument.
    """

    module, _, rest = site.partition("::")
    function, _, local = rest.partition("::")
    tree = dict(_SOURCES)[module]
    triples: List[Tuple[object, object, object]] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name != function:
            continue
        for call in ast.walk(node):
            if not isinstance(call, ast.Call):
                continue
            if ast.unparse(call.func).rsplit(".", 1)[-1] not in MARKER_CARRYING_TYPES:
                continue
            if local and not any(
                isinstance(parent, ast.Assign)
                and parent.value is call
                and any(
                    isinstance(target, ast.Name) and target.id == local
                    for target in parent.targets
                )
                for parent in ast.walk(node)
            ):
                continue
            marker = [k.value for k in call.keywords if k.arg == "result_storage"]
            assert marker, (
                f"{site} builds a completion reference without a result-storage "
                "marker.  Its emitted counterpart carries one and the two are "
                "compared for EQUALITY, so this refuses a program that compiles "
                "today"
            )
            for element in ast.walk(marker[0]):
                if not isinstance(element, ast.Tuple) or len(element.elts) != 3:
                    continue
                array, level, direction = element.elts
                if (
                    not isinstance(array, ast.Name)
                    or array.id not in _ARRAY_SPELLINGS
                    or not isinstance(direction, ast.Name)
                    or direction.id not in _DIRECTION_SPELLINGS
                ):
                    continue
                triples.append(
                    (
                        _ARRAY_SPELLINGS[array.id],
                        level.value if isinstance(level, ast.Constant) else None,
                        _DIRECTION_SPELLINGS[direction.id],
                    )
                )
    return triples


def _emitted_marker(
    assembler: object, callee: str
) -> Optional[llir.ResultStorageMetadata]:
    """The marker on the one statement the assembler emits with this callee."""

    emitted = [
        statement
        for statement in assembler.emit_value_array_init()  # type: ignore[attr-defined]
        if type(statement) is llir.FunctionCallStmt and statement.name == callee
    ]
    assert len(emitted) == 1, f"expected one {callee}, found {len(emitted)}"
    marker = emitted[0].result_storage
    assert marker is None or type(marker) is llir.ResultStorageMetadata
    return marker


def _dense_assembler(result_id: object = None) -> object:
    """An all-dense receiver, which is the branch that emits the zero fill."""

    import torch

    from scorch.compiler.torch_cpp_abi import (  # type: ignore[import-untyped]
        ResultTensorAssembler,
    )
    from scorch.format import LevelType  # type: ignore[import-untyped]

    return ResultTensorAssembler(
        name="C",
        level_types=(LevelType.DENSE, LevelType.DENSE),
        dtype=torch.float32,
        result_id=result_id,
    )


def test_the_dense_zero_fill_is_marked_on_both_sides() -> None:
    """The SECOND instance of gap A's shape, and section 64.6 named only the first.

    ``scorch_zero_dense(C_values, C_capacity)`` is a free function that WRITES the
    result's value vector through an argument: the callee-name match cannot see it,
    and the syntax cannot say the argument is written.
    ``_complete_result_tile_impl`` hand-builds the completion reference for it and
    compares the two for equality, so both sides carry the same marker or a program
    that compiles today is refused.

    Both sides are checked against the same declared triple: the emitted side by
    driving the assembler, the reference side by reading the triple it passes.
    """

    triple = (llir.ResultStorageArray.VALUES, None, True)

    assert _emitted_marker(_dense_assembler(), "scorch_zero_dense") is None
    emitted = _emitted_marker(_dense_assembler(SymbolId(7)), "scorch_zero_dense")
    assert emitted == _dense_assembler(SymbolId(7)).result_storage_marker(  # type: ignore[attr-defined]
        triple
    )
    assert emitted.tensor_id == SymbolId(7)
    assert [
        (reference.array, reference.level, reference.direction)
        for reference in emitted.references
    ] == [(llir.ResultStorageArray.VALUES, None, llir.ResultStorageDirection.WRITE)]

    assert _reference_triples_at(
        "compiler/loopir/lower_llir.py::_complete_result_tile_impl::expected_zero"
    ) == [triple]


def test_the_position_sentinel_reference_declares_a_position_write() -> None:
    """The sentinel's reference triple, read off the source.

    ``test_both_sides_of_the_position_completion_carry_the_same_marker`` compares
    the two markers directly; this adds what that cannot see, which is that the
    reference is spelled from the position array and a write rather than reaching
    the right value some other way.  Its level is the builder's argument, so the
    level reads as ``None`` here.
    """

    assert _reference_triples_at(
        "compiler/loopir/lower_llir.py::_completed_position_init_statement"
    ) == [(llir.ResultStorageArray.POS, None, True)]


#: Statements the result ABI emits that reference result storage by name and must
#: stay UNMARKED, with the reason.  Locked in this direction too, because the
#: completion comparison is symmetric: marking one of these while its hand-built
#: reference stays unmarked refuses a program that compiles today, exactly as the
#: reverse does.
_UNMARKED_ABI_STATEMENTS = {
    "C_values pointer declaration": (
        "DECLARES C_values as a pointer into the Torch tensor.  It neither reads "
        "nor writes the storage's contents, and its value expression names "
        "C_values_torch, a different thing.  ResultStorageArray's docstring "
        "excludes the two-phase pass's _data pointers and the p{R}{L} cursor on "
        "the same ground.  _complete_result_tile_impl hand-builds the completion "
        "reference for this statement and leaves it unmarked to match"
    ),
}


def test_the_values_pointer_declaration_stays_unmarked() -> None:
    """A declaration is not an access, and both sides agree by staying silent.

    The mutation this closes: marking this statement while
    ``_complete_result_tile_impl``'s ``expected_result_pointer_init`` stays
    unmarked makes the two differ and refuses a program that compiles today.  The
    completion comparison is symmetric, so the unmarked direction needs a lock as
    much as the marked one does.
    """

    assert "C_values pointer declaration" in _UNMARKED_ABI_STATEMENTS
    for assembler in (_dense_assembler(), _dense_assembler(SymbolId(7))):
        declarations = [
            statement
            for statement in assembler.emit_value_array_init()  # type: ignore[attr-defined]
            if type(statement) is llir.VarInit
            and type(statement.var) is llir.Var
            and statement.var.name == "C_values"
        ]
        assert len(declarations) == 1, len(declarations)
        assert declarations[0].result_storage is None, (
            "the result value pointer's DECLARATION gained a marker.  It is not an "
            "access -- see _UNMARKED_ABI_STATEMENTS -- and "
            "_complete_result_tile_impl's hand-built reference for it is unmarked, "
            "so marking this side alone refuses a program that compiles today"
        )


#: The result ABI's statements that name result storage with no dotted member and
#: stay unmarked, with the reason.  Empty, and that is the point: a bare storage
#: name is what the guard's callee-name match structurally cannot see -- review
#: section 61.4's gap A -- so an unmarked one passes through silently.
_UNMARKED_BARE_ABI_WRITES: Dict[str, str] = {}


def test_the_result_abi_leaves_no_bare_storage_name_unmarked() -> None:
    """Gap A, closed in the assembler and kept closed.

    Why this is a separate check from everything above: the statement-marker census
    lists ``torch_cpp_abi.py`` among its files and reports ZERO result references in
    it, because that file builds every storage name from ``self.name`` and a level
    number and the census cannot attribute a polymorphic name.  So the census cannot
    see an unmarked result write inside the assembler, which is the file gap A lives
    in.  This reads the file directly.

    Only the BARE shape is required to be marked.  A DOTTED member call is one the
    existing callee-name match already sees, so an unmarked one is refused rather
    than accepted; and a ``VarDecl`` cannot carry the field, while a ``VarInit``
    declaring the vector or a pointer to it neither reads nor writes the storage's
    contents.
    """

    tree = dict(_SOURCES)["compiler/torch_cpp_abi.py"]
    assembler = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ResultTensorAssembler"
    )
    unmarked: List[str] = []
    bare_marked = 0
    for function in assembler.body:
        if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for call in ast.walk(function):
            if not isinstance(call, ast.Call):
                continue
            statement_class = ast.unparse(call.func).rsplit(".", 1)[-1]
            if statement_class not in MARKER_CARRYING_TYPES:
                continue
            spellings: Set[str] = set()
            for child in ast.walk(call):
                if not isinstance(child, ast.JoinedStr):
                    continue
                pieces: List[str] = []
                for part in child.values:
                    if isinstance(part, ast.Constant) and isinstance(part.value, str):
                        pieces.append(part.value)
                    else:
                        pieces.append("{}")
                text = "".join(pieces)
                if any(
                    text.endswith(tail) or f"{tail}." in text
                    for tail in ("_values", "_pos", "_crd")
                ):
                    spellings.add(text)
            if not spellings:
                continue
            if statement_class == "VarInit" or any("." in s for s in spellings):
                continue  # DECLARATION or DOTTED; see the docstring
            key = f"{function.name}:{call.lineno}"
            if any(keyword.arg == "result_storage" for keyword in call.keywords):
                bare_marked += 1
            elif key not in _UNMARKED_BARE_ABI_WRITES:
                unmarked.append(f"{key} {statement_class} {sorted(spellings)}")

    assert bare_marked == 2, (
        "expected the two bare-name result writes the assembler emits -- the "
        f"position sentinel and the dense zero fill -- and found {bare_marked}"
    )
    assert not unmarked, (
        "the result ABI emits a statement naming result storage with no dotted "
        "member and no marker.  The guard's callee-name match cannot see a bare "
        "name, so this is review section 61.4's gap A: mark it from the "
        "assembler's result_id, and mark whatever completion reference is built "
        "for it, or register it in _UNMARKED_BARE_ABI_WRITES with the reason:\n  "
        + "\n  ".join(sorted(unmarked))
    )
