"""Parallel single-pass sparse output assembly, by per-chunk buffers.

The generated single-pass sparse builders append into ``std::vector`` in
outer-loop order.  That is sequential by construction, so the migrated families
that use it emit a SERIAL kernel where the legacy pipeline emits a parallel
two-phase one -- and the legacy kernel wins by up to 2.8x on the shapes where
its own ``scorch_nthreads`` gate opens.

The measured decomposition (``~/.cache/scorch-codex/ttm-density-mechanism/``)
says that gap is 100% parallelism and 0% allocation strategy: with legacy's two
``#pragma omp parallel for`` lines deleted and nothing else changed, the typed
single pass is 1.18-1.47x FASTER than legacy's serial two-pass, because legacy's
counting pass costs it more than vector growth costs the single pass.  Adopting
the two-phase strategy would therefore surrender a real advantage to buy
parallelism.  This module buys the parallelism and keeps the advantage.

THE STRATEGY.  Partition the outer dense loop into chunks.  Each chunk appends
into its own private output buffers, exactly as the serial builder does, and the
buffers are concatenated afterwards in chunk order -- which is outer-loop order,
which is the required lexicographic order, because the outer loop binds result
level 0 and every compressed level sits under it.

WHY CHUNKS AND NOT THREADS OWN THE BUFFERS.  Under a dynamic schedule one thread
may take a low row range and then a high one, so concatenating per-THREAD
buffers would not reproduce lexicographic order.  Per-CHUNK buffers do, whatever
the schedule did and whichever thread ran what, so dynamic load balancing costs
nothing in ordering.

WHY BOTH ARMS SHARE ONE BODY.  The nest is emitted ONCE, as a lambda both arms
call: whole-range with the shared result vectors for the serial arm, once per
chunk with that chunk's private buffers for the parallel one.  This is not
tidiness.  The first build of this transformation gave each arm its own copy,
and that was measured to slow the arm that RUNS by up to 34% -- with the other
copy never executed -- because the optimizer's per-function budgets are then
spent on both.  A source-level ablation isolated it: stubbing the unexecuted
arm's body out, with the gate still in place, recovered the base's time exactly
at all twelve configurations, so the cost is the second copy and not the branch
(``ttm-parallel-singlepass/DUPLICATION.md``).  Sharing the body is what makes
the serial arm cost nothing, and it also makes "both arms compute the same
thing" true by construction rather than by test.

The buffer parameters are ``auto&`` and both call sites pass the same types, so
the generic lambda has exactly one instantiation.

WHY THE SERIAL ARM IS A REAL BRANCH.  At ``scorch_nthreads == 1`` an OpenMP
region is not merely inert, it costs 4-10% -- measured -- and the low-density
wins this family already holds are exactly the saving from paying neither a
counting pass nor a region.  So the gate is a runtime ``if`` whose ``else`` arm
is one call over the whole range, with no pragma anywhere on it, no chunk buffer
constructed and nothing to merge afterwards.  An OpenMP ``if()`` clause was
rejected for this: a false clause still enters the runtime and still outlines
the body, which would make inertness an empirical claim instead of a structural
one.

The design this implements is ``FIX_DESIGN.md`` in the ttm-density-mechanism
ledger; ``ttm-parallel-singlepass/`` holds this one's measurements.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple, TypeVar, cast

from .. import llir
from ..llir_traversal import LLIRRewriter, LLIRTraversalContext
from ..parallel_marking_pass import (
    extract_loop_bound,
    find_sparse_pos_array,
    sparse_pos_work_expr,
)

#: Every runtime spelling this transformation puts into a generated kernel.
#: Adopting families reserve these against user identifiers.
PARALLEL_CHUNK_RUNTIME_SPELLINGS: Tuple[str, ...] = (
    "scorch_chunk_rows",
    "scorch_chunk_end",
    "scorch_chunk_buffers",
    "scorch_presize_positions",
    "scorch_concat_chunks",
    "scorch_concat_chunk_positions",
    "scorch_shift_chunk_positions",
)

#: Every generated identifier this transformation declares.
GENERATED_NAMES: Tuple[str, ...] = (
    "_assembly_threads",
    "_assembly_width",
    "_assembly_chunks",
    "_assembly_chunk",
    "_assembly_lo",
    "_assembly_hi",
    "_assembly_body",
)

#: The prefix of every per-chunk buffer name.
CHUNK_BUFFER_PREFIX = "_chunk_"

#: Anything the transformation copies structurally.
_DetachedT = TypeVar("_DetachedT")

_TRAVERSAL = LLIRTraversalContext(
    stage="LoopIR target lowering",
    pass_name="parallel_chunk_assembly",
)


@dataclass(frozen=True)
class ParallelChunkAssemblyContext:
    """Everything the transformation needs that is not the statement list.

    ``shared_position_level`` is the first compressed level.  Its position array
    is indexed by the outer dense loop over a statically known range, so it is
    shared and pre-sized rather than chunked: every chunk writes a disjoint
    slice, and only its VALUES (running coordinate counts) need the merge's
    offset.  That removes one array from the concatenation entirely and removes
    the only place a chunk-local copy would have needed its index expression
    rewritten.
    """

    result_name: str
    #: The first compressed result level; its ``_pos`` array is shared.
    shared_position_level: int
    #: Every compressed result level, ascending, including the shared one.
    compressed_levels: Tuple[int, ...]
    #: C++ element spelling of the result value array (e.g. ``"float"``).
    value_ctype: str
    #: C++ element spelling of every position/coordinate array (e.g. ``"int"``).
    index_ctype: str


def buffer_name(vector_name: str) -> str:
    """The per-chunk buffer holding one result vector's chunk-local entries."""

    return f"{CHUNK_BUFFER_PREFIX}{vector_name}"


def chunked_vector_names(context: ParallelChunkAssemblyContext) -> Tuple[str, ...]:
    """Every result vector whose entries are chunk-local, in emission order."""

    result = context.result_name
    names: List[str] = []
    for level in context.compressed_levels:
        if level != context.shared_position_level:
            names.append(f"{result}{level}_pos")
        names.append(f"{result}{level}_crd")
    names.append(f"{result}_values")
    return tuple(names)


def parallel_chunk_generated_names(
    context: ParallelChunkAssemblyContext,
) -> Tuple[str, ...]:
    """Every identifier the transformation declares for one result."""

    return (
        *GENERATED_NAMES,
        *(buffer_name(name) for name in chunked_vector_names(context)),
    )


def _var(name: str, data_type: llir.DataType = llir.DataType.NO_TYPE) -> llir.Var:
    return llir.Var(name=name, type=data_type)


def _int64(name: str) -> llir.Var:
    return _var(name, llir.DataType.INT64)


def _literal(value: int) -> llir.Literal:
    return llir.Literal(value=value, data_type=llir.DataType.INT)


def _detach(value: _DetachedT) -> _DetachedT:
    """Return a fully owned structural copy, sharing no mutable node.

    Every node the parallel arm places is freshly owned, so the two arms of the
    gate cannot alias one mutable statement and a later in-place rewrite of one
    cannot silently move the other.
    """

    return cast(_DetachedT, LLIRRewriter(_TRAVERSAL).rewrite(cast(Any, value)))


@dataclass(frozen=True)
class _PolicyExpressions:
    """The work/rows pair legacy's own pragma is derived from, reused exactly."""

    work_text: str
    rows_text: str
    work: llir.Expr
    rows: llir.Expr


def _policy_expressions(loop: llir.ForLoop) -> Optional[_PolicyExpressions]:
    """Derive ``scorch_nthreads``'s arguments from the outer loop.

    This is deliberately the SAME derivation the legacy pragma uses -- the
    shared ``sparse_pos_work_expr`` over the first sparse position array in the
    body -- so the gate this transformation opens on is the same gate whose
    behaviour the mechanism study characterized, evaluated on the same operands.
    A different work estimate would silently move the parallel/serial crossover
    away from the one that was measured.
    """

    if type(loop.update) is not llir.Increment:
        # A strided outer loop would make `rows` a trip count rather than the
        # bound, and the chunk arithmetic below indexes rows directly.
        return None
    bound = extract_loop_bound(loop)
    if bound is None:
        return None
    if type(loop.cond) is not llir.BinOp or type(loop.cond.right) is not llir.Var:
        return None
    bound_type = cast(llir.Var, loop.cond.right).type
    position = find_sparse_pos_array(loop.body)
    work_text = sparse_pos_work_expr(position, bound)
    if work_text is None or position is None:
        return None
    return _PolicyExpressions(
        work_text=work_text,
        rows_text=bound,
        work=llir.ArrayAccess(
            array=_var(position),
            index=_var(bound, bound_type),
        ),
        rows=_var(bound, bound_type),
    )


def is_transformable(loop: llir.ForLoop) -> bool:
    """Whether the outer loop supports the chunk partition and the policy."""

    return _policy_expressions(loop) is not None


def _body_lambda(
    outer: llir.ForLoop,
    context: ParallelChunkAssemblyContext,
) -> llir.LambdaDef:
    """The ONE copy of the assembly nest, parameterized by row range and buffers.

    Emitting the nest twice -- once per arm of the gate -- was measured to slow
    the arm that runs by up to 34%, with the other copy never executed, because
    the optimizer's per-function budgets are then spent on both.  A source-level
    ablation isolated it: stubbing the unexecuted arm's body out, with the gate
    left in place, recovered the base's time exactly, so the cost is the second
    copy and not the branch (``DUPLICATION.md`` in the ledger).  Both arms
    therefore call one lambda.

    The buffer parameters are ``auto&``, so this is a generic lambda; both call
    sites pass the same types, so it is one instantiation and the "one body"
    property is real rather than intended.  Four scalars are re-initialized per
    call, and only one takes a value that differs from its serial initializer:
    the shared level's ``_pos_index`` starts at the call's first row, which is
    exactly what makes the existing catch-up loop correct over a sub-range.
    """

    result = context.result_name
    prologue: List[llir.Stmt] = []
    for level in context.compressed_levels:
        if level == context.shared_position_level:
            continue
        # A chunk-local position array needs its own leading zero.  For the
        # whole-range call the shared array already carries one, and rewriting
        # slot zero with zero is idempotent, so one spelling serves both.
        prologue.append(
            llir.FunctionCallStmt(
                name="scorch_vector_set",
                args=[_var(f"{result}{level}_pos"), _literal(0), _literal(0)],
            )
        )
    for level in context.compressed_levels:
        prologue.append(
            llir.VarInit(var=_int64(f"p{result}{level}"), value=_literal(0))
        )
        prologue.append(
            llir.VarInit(
                var=_int64(f"{result}{level}_pos_index"),
                value=(
                    _int64("_assembly_lo")
                    if level == context.shared_position_level
                    else _literal(0)
                ),
            )
        )

    ranged = _detach(outer)
    ranged.init = llir.VarInit(
        var=cast(llir.VarInit, ranged.init).var,
        value=_int64("_assembly_lo"),
    )
    ranged.cond = llir.BinOp(
        "<", cast(llir.BinOp, ranged.cond).left, _int64("_assembly_hi")
    )

    params = [_int64("_assembly_lo"), _int64("_assembly_hi")]
    params.extend(
        _var(name, llir.DataType.AUTO_REF) for name in chunked_vector_names(context)
    )
    return llir.LambdaDef(
        var=_var("_assembly_body", llir.DataType.AUTO),
        params=params,
        body=[*prologue, ranged],
    )


def _body_call(
    context: ParallelChunkAssemblyContext,
    low: llir.Expr,
    high: llir.Expr,
    buffers: List[llir.Expr],
) -> llir.FunctionCallStmt:
    return llir.FunctionCallStmt(name="_assembly_body", args=[low, high, *buffers])


def _merge_statements(
    context: ParallelChunkAssemblyContext,
    policy: _PolicyExpressions,
) -> List[llir.Stmt]:
    """Concatenate the per-chunk buffers into the shared result vectors.

    Every helper takes the assembly thread count.  This copy is the whole price
    of keeping the single-pass strategy -- it stands in for legacy's counting
    pass -- and once the compute term has been cut by the thread count a SERIAL
    merge is a large fraction of what is left, so the merge is parallel over
    chunks too.  Each helper falls back to a straight-line copy below
    ``SCORCH_MEMSET_GRAIN_BYTES``, where the region would cost more than the
    copy.
    """

    result = context.result_name
    shared = context.shared_position_level
    statements: List[llir.Stmt] = [
        # The shared position array already holds each chunk's LOCAL running
        # coordinate count in that chunk's own slice; the shift adds the count
        # every earlier chunk contributed.
        llir.FunctionCallStmt(
            name="scorch_shift_chunk_positions",
            args=[
                _var(f"{result}{shared}_pos"),
                _var(buffer_name(f"{result}{shared}_crd")),
                _int64("_assembly_width"),
                _detach(policy.rows),
                _var("_assembly_threads", llir.DataType.INT),
            ],
        )
    ]
    for level in context.compressed_levels:
        if level != shared:
            statements.append(
                llir.FunctionCallStmt(
                    name="scorch_concat_chunk_positions",
                    args=[
                        _var(f"{result}{level}_pos"),
                        _var(buffer_name(f"{result}{level}_pos")),
                        _var(buffer_name(f"{result}{level}_crd")),
                        _var("_assembly_threads", llir.DataType.INT),
                    ],
                )
            )
        statements.append(
            llir.FunctionCallStmt(
                name="scorch_concat_chunks",
                args=[
                    _var(f"{result}{level}_crd"),
                    _var(buffer_name(f"{result}{level}_crd")),
                    _var("_assembly_threads", llir.DataType.INT),
                ],
            )
        )
    statements.append(
        llir.FunctionCallStmt(
            name="scorch_concat_chunks",
            args=[
                _var(f"{result}_values"),
                _var(buffer_name(f"{result}_values")),
                _var("_assembly_threads", llir.DataType.INT),
            ],
        )
    )
    return statements


def _parallel_branch(
    context: ParallelChunkAssemblyContext,
    policy: _PolicyExpressions,
) -> List[llir.Stmt]:
    """Buffers, the chunk region, then the merge.  Carries no copy of the nest."""

    result = context.result_name
    width = _int64("_assembly_width")
    chunks = _int64("_assembly_chunks")
    chunk = _int64("_assembly_chunk")

    statements: List[llir.Stmt] = [
        llir.VarInit(
            var=width,
            value=llir.FunctionCall(
                name="scorch_chunk_rows",
                args=[_detach(policy.rows), _detach(policy.work)],
            ),
        ),
        llir.VarInit(
            var=chunks,
            value=llir.BinOp(
                "/",
                llir.BinOp(
                    "-", llir.Add(_detach(policy.rows), _detach(width)), _literal(1)
                ),
                _detach(width),
            ),
        ),
    ]
    for name in chunked_vector_names(context):
        element = (
            context.value_ctype if name == f"{result}_values" else context.index_ctype
        )
        statements.append(
            llir.VarInit(
                var=_var(buffer_name(name), llir.DataType.AUTO),
                value=llir.FunctionCall(
                    name=f"scorch_chunk_buffers<{element}>",
                    args=[_detach(chunks)],
                ),
            )
        )
    statements.append(
        llir.FunctionCallStmt(
            name="scorch_presize_positions",
            args=[
                _var(f"{result}{context.shared_position_level}_pos"),
                llir.Add(_detach(policy.rows), _literal(1)),
            ],
        )
    )

    low = _int64("_assembly_lo")
    chunk_body: List[llir.Stmt] = [
        llir.VarInit(var=low, value=llir.BinOp("*", _detach(chunk), _detach(width))),
        _body_call(
            context,
            _detach(low),
            llir.FunctionCall(
                name="scorch_chunk_end",
                args=[_detach(low), _detach(width), _detach(policy.rows)],
            ),
            [
                llir.ArrayAccess(array=_var(buffer_name(name)), index=_detach(chunk))
                for name in chunked_vector_names(context)
            ],
        ),
    ]

    # "dynamic" is OpenMP's chunk-size-one dynamic schedule.  One iteration is
    # one output chunk, and scorch_chunk_rows already sized the chunks at about
    # SCORCH_CHUNKS_PER_THREAD per worker, so the schedule needs no chunking of
    # its own.  The thread count is spelled exactly as the legacy pragma spells
    # it, from the same derivation, so the two kernels open on the same gate.
    statements.append(
        llir.ForLoop(
            init=llir.VarInit(var=chunk, value=_literal(0)),
            cond=llir.BinOp("<", _detach(chunk), _detach(chunks)),
            update=llir.Increment(var=_detach(chunk)),
            body=chunk_body,
            omp_parallel_for=True,
            omp_schedule="dynamic",
            omp_num_threads=(
                f"scorch_nthreads({policy.work_text}, {policy.rows_text})"
            ),
        )
    )
    statements.extend(_merge_statements(context, policy))
    return statements


def build_parallel_chunk_assembly(
    statements: List[llir.Stmt],
    context: ParallelChunkAssemblyContext,
) -> Optional[List[llir.Stmt]]:
    """Return the gated statement list, or ``None`` to keep the serial one.

    ``None`` is not a failure: a declined optimization is still a correct
    kernel, and every caller keeps emitting exactly what it emits today.
    """

    if type(context) is not ParallelChunkAssemblyContext:
        raise TypeError("an exact ParallelChunkAssemblyContext is required")
    if type(statements) is not list:
        raise TypeError("the serial statement list must be a list")
    outer_index = _sole_outer_loop(statements)
    if outer_index is None:
        return None
    outer = cast(llir.ForLoop, statements[outer_index])
    policy = _policy_expressions(outer)
    if policy is None:
        return None
    return [
        _body_lambda(outer, context),
        llir.VarInit(
            var=_var("_assembly_threads", llir.DataType.INT),
            value=llir.FunctionCall(
                name="scorch_nthreads",
                args=[_detach(policy.work), _detach(policy.rows)],
            ),
        ),
        llir.IfThenElse(
            cond=llir.BinOp(
                ">", _var("_assembly_threads", llir.DataType.INT), _literal(1)
            ),
            then_body=_parallel_branch(context, policy),
            # The serial arm is ONE CALL: the whole row range, straight into the
            # shared result vectors, so there is no chunk buffer to construct
            # and nothing to merge afterwards.  No pragma appears anywhere on
            # this path -- inertness at one thread stays structural -- and
            # because the body it calls is the same object the parallel arm
            # calls, the function holds one copy of the nest rather than two.
            else_body=[
                _body_call(
                    context,
                    _literal(0),
                    _detach(policy.rows),
                    [_var(name) for name in chunked_vector_names(context)],
                )
            ],
        ),
    ]


def _sole_outer_loop(statements: Sequence[llir.Stmt]) -> Optional[int]:
    """Index of the one top-level loop, or ``None`` when there is not exactly one."""

    found: Optional[int] = None
    for index, statement in enumerate(statements):
        if type(statement) is llir.ForLoop:
            if found is not None:
                return None
            found = index
        elif type(statement) in (llir.ForLoopAuto, llir.WhileLoop):
            return None
    return found
