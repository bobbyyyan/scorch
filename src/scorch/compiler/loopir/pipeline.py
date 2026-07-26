"""Test/debug LoopIR strangler pipeline and curated shadow comparison.

This module is the executable end of the Phase-4/5/6 vertical slices: it
carries one normalized supported-family CIN program through

``normalized CIN -> LoopPlan -> LoopIR -> scheduled LoopIR ->
structured LLIR -> C++ -> compiled kernel -> execution``

using the same JIT build, cache, and execution helpers as the legacy public
entry (:func:`scorch.ops.lower_and_exec_cin`).  When the compile options
carry a requested :class:`~scorch.compiler.scheduler.Schedule`, the shared
``Scheduler.apply_schedule`` boundary validates it and produces the
verified ``LoopPlan``; the migrated explicit families (loop reorder,
affine ``accum='direct'`` tiles, and ``accum='stack'`` workspace
accumulation) are then applied as typed LoopIR passes,
and every other schedule family fails closed with a stable code —
nothing silently ignores a requested schedule.  Kernel cache identity
remains source-derived, exactly as on the legacy path: the schedule
affects the generated source, identical source is the identical kernel
artifact, and no LoopIR- or plan-level artifact is ever cached.

It exists for dedicated LoopIR tests and curated shadow comparison only:

- production never imports it, so the legacy default path, its stage
  sequences, and its cache keys are untouched;
- kernel cache identity remains derived from the generated source exactly as
  before.  For the migrated families the LoopIR path generates byte-identical
  source, so it shares the legacy kernel artifact honestly — identical source
  is identical kernel — while LoopIR-level artifacts (programs, dumps) are
  never cached at all;
- shadow comparison (:func:`compare_generated_sources`,
  :func:`execute_shadow`) is invoked explicitly by curated differential
  tests; nothing turns it on globally, so ordinary pytest and release JIT do
  not double compile.

Stage timing: the LoopIR stages are recorded through the same
:class:`~scorch.compiler.compilation_context.CompilationContext` mechanism
under the new ``CIN_TO_LOOPIR_LOWERING`` and ``LOOPIR_TO_LLIR_LOWERING``
identities; downstream C++ generation and build-request assembly reuse the
existing canonical stage identities.
"""

from __future__ import annotations

import copy
import hashlib
import time
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch

from ...exceptions import CompileSpecError
from ...utils import parse_format
from .. import llir
from ..cin import IndexStmt
from ..cin_analysis import normalize_cin
from ..cin_lowerer import CINLowerer
from ..compilation_context import CompilationContext, CompilerStageId
from ..compile_options import CompileOptions
from ..identity import SymbolId
from ..loop_plan import LoopPlan, ScheduledCIN
from .lower_cin import LoopIRLoweringResult, lower_normalized_cin_to_loopir
from .lower_llir import lower_loopir_to_llir
from .nodes import LevelKind as LoopIRLevelKind, LoopProgram
from .printer import canonical_program_dump, print_program
from .schedule_passes import ScheduledLoopIR, apply_schedule_plan


@dataclass(frozen=True)
class LoopIRCompiledKernel:
    """One compiled supported-family kernel and its LoopIR provenance.

    For scheduled compilations ``schedule`` retains the full scheduling
    artifact (base program, exact plan, scheduled program, provenance) and
    ``program_text``/``program_dump`` describe the scheduled program — the
    artifact the target lowering consumed.  ``request_dump`` and
    ``request_identity`` are the canonical strangler request serialization
    and its SHA-256 content key, computed at the compile/shadow request
    boundary from the canonical normalized CIN, the canonical verified plan
    (or the explicit unscheduled marker), the result shape, and the runtime
    input bindings; no cache consumes them yet, and the release
    source-derived cache is untouched.
    """

    lowering: LoopIRLoweringResult
    program_text: str
    program_dump: str
    llir_function: llir.Function
    cpp_source: str
    schedule: Optional[ScheduledLoopIR] = None
    request_dump: str = ""
    request_identity: str = ""


@dataclass(frozen=True)
class ShadowSourceComparison:
    """Curated legacy-versus-LoopIR generated-source comparison."""

    loopir_cpp: str
    legacy_cpp: str

    @property
    def identical(self) -> bool:
        return self.loopir_cpp == self.legacy_cpp


def _resolve_options(
    compile_options: Optional[CompileOptions],
    compilation_context: Optional[CompilationContext] = None,
) -> CompileOptions:
    if (
        compilation_context is not None
        and type(compilation_context) is not CompilationContext
    ):
        raise TypeError("compilation_context must be a CompilationContext")
    if compile_options is None:
        options = (
            compilation_context.compile_options
            if compilation_context is not None
            else CompileOptions.from_environment()
        )
    elif type(compile_options) is not CompileOptions:
        raise TypeError("compile_options must be a CompileOptions snapshot")
    else:
        options = compile_options
    if compilation_context is not None:
        compilation_context.require_compile_options(options)
    return options


def _plan_mode_orders_to_planned_order(
    normalized: IndexStmt,
    args: Sequence[object],
    plan: LoopPlan,
):
    """The scheduled-path twin of ``ops._plan_mode_orders_to_loop_order``.

    The legacy helper reads the iteration order from the CIN's own ForAll
    chain; a scheduled compilation iterates in the plan's logical order
    instead, so runtime mode-order alignment must target that order — the
    same storage-vs-iteration consistency the reorder pass and the target
    boundary enforce.  Only the order source differs; the relayout executor
    and CIN alignment application are the shared production helpers.
    """

    from ..loop_plan import entity_display_names

    index_names, _ = entity_display_names(normalized)
    loop_order_names = [index_names[index_id] for index_id in plan.loop_order]
    if not loop_order_names:
        return tuple(None for _ in args), None

    rhs_accesses = normalized.get_rhs_tensor_accesses()
    if len(rhs_accesses) != len(args):
        return tuple(None for _ in args), None

    rhs_mode_orders: List[Optional[Tuple[int, ...]]] = []
    for access in rhs_accesses:
        tensor_var = access.get_tensor()
        access_names = [index_var.name for index_var in access.get_index_vars()]
        desired_names = [name for name in loop_order_names if name in access_names]
        desired = tuple(access_names.index(name) for name in desired_names)
        mode_order = tensor_var.mode_order
        if mode_order is None or len(desired) != len(mode_order):
            rhs_mode_orders.append(None)
            continue
        rhs_mode_orders.append(desired)

    lhs_mode_order: Optional[Tuple[int, ...]] = None
    leaf: IndexStmt = normalized
    while hasattr(leaf, "stmt"):
        leaf = leaf.stmt
    lhs_access = getattr(leaf, "lhs", None)
    if lhs_access is not None:
        lhs_tensor = lhs_access.get_tensor()
        lhs_names = [index_var.name for index_var in lhs_access.get_index_vars()]
        desired_names = [name for name in loop_order_names if name in lhs_names]
        desired = tuple(lhs_names.index(name) for name in desired_names)
        lhs_modes = lhs_tensor.mode_order
        if lhs_modes is not None and len(desired) == len(lhs_modes):
            lhs_mode_order = desired
    return tuple(rhs_mode_orders), lhs_mode_order


def _execute_mode_order_alignment(
    normalized: IndexStmt,
    args: tuple,
    alignment_plan,
    options: CompileOptions,
    context: Optional[CompilationContext],
) -> tuple:
    """Apply one alignment without leaking a parent schedule to prerequisites.

    A storage relayout may compile an auxiliary kernel.  That prerequisite is
    not the scheduled operation, so it receives a schedule-free copy of the
    already-resolved options and a matching independent instrumentation owner.
    Every other build and compiler option remains identical.
    """

    from ...ops import _apply_mode_order_alignment, _relayout_mode_order_args

    relayout_options = options
    relayout_context = context
    if options.requested_schedule is not None:
        relayout_options = replace(options, requested_schedule=None)
        relayout_context = CompilationContext(relayout_options)
    aligned = _relayout_mode_order_args(
        args,
        alignment_plan,
        relayout_options,
        relayout_context,
    )
    _apply_mode_order_alignment(normalized, alignment_plan)
    return aligned


def _apply_requested_schedule(
    cin_stmt: IndexStmt,
    options: CompileOptions,
    context: Optional[CompilationContext],
) -> Tuple[IndexStmt, LoopPlan]:
    """Adapt the public Schedule through the shared scheduler boundary.

    ``Scheduler.apply_schedule`` is the single owner of Schedule validation
    and Schedule-to-LoopPlan translation for both pipelines; the paths only
    diverge downstream (legacy replays CIN tree surgery, the LoopIR path
    applies typed passes to the verified base program).  It normalizes the
    CIN and records the scheduling stage itself.
    """

    from ..scheduler import Scheduler

    schedule = options.requested_schedule
    assert schedule is not None
    scheduled = Scheduler.apply_schedule(
        cin_stmt,
        schedule,
        compile_options=options,
        compilation_context=context,
    )
    return scheduled.normalized_cin, scheduled.verified_loop_plan


def _bind_runtime_metadata(
    cin_stmt: IndexStmt,
    input_bindings: Sequence[Tuple[Tuple[int, ...], torch.dtype]],
    result_shape: Tuple[int, ...],
    input_formats: Optional[Sequence[object]] = None,
) -> None:
    """Bind shapes/dtypes exactly like the legacy public entry does."""

    rhs_tensor_vars = cin_stmt.get_rhs_tensor_vars()
    if len(rhs_tensor_vars) != len(input_bindings):
        raise CompileSpecError(
            f"CIN expects {len(rhs_tensor_vars)} runtime tensors, got "
            f"{len(input_bindings)}"
        )
    if input_formats is not None and len(input_formats) != len(input_bindings):
        raise CompileSpecError(
            "runtime tensor formats must match the input binding count"
        )
    for position, (tensor_var, (shape, dtype)) in enumerate(
        zip(rhs_tensor_vars, input_bindings)
    ):
        if input_formats is not None and tensor_var.format != input_formats[position]:
            raise CompileSpecError(
                f"CIN tensor {tensor_var.name!r} expects format "
                f"{tensor_var.format}, got {input_formats[position]}"
            )
        tensor_var.shape = tuple(shape)
        tensor_var.dtype = dtype
        if tensor_var.mode_order is None:
            tensor_var.mode_order = list(range(len(shape)))
    output_dtype = input_bindings[0][1] if input_bindings else torch.float32
    for tensor_var in cin_stmt.get_result_tensor_vars():
        tensor_var.shape = tuple(result_shape)
        tensor_var.dtype = output_dtype


def _validate_runtime_formats(cin_stmt: IndexStmt, args: Sequence[object]) -> None:
    """Reject storage-format mismatches before any runtime relayout work."""

    rhs_tensor_vars = cin_stmt.get_rhs_tensor_vars()
    if len(rhs_tensor_vars) != len(args):
        raise CompileSpecError(
            f"CIN expects {len(rhs_tensor_vars)} runtime tensors, got {len(args)}"
        )
    for tensor_var, arg in zip(rhs_tensor_vars, args):
        actual_format = getattr(arg, "format", None)
        if tensor_var.format != actual_format:
            raise CompileSpecError(
                f"CIN tensor {tensor_var.name!r} expects format "
                f"{tensor_var.format}, got {actual_format}"
            )


def _compile_normalized_cin_via_loopir(
    normalized: IndexStmt,
    result_shape: Sequence[int],
    input_bindings: Sequence[Tuple[Tuple[int, ...], torch.dtype]],
    *,
    options: CompileOptions,
    context: CompilationContext,
    input_formats: Optional[Sequence[object]] = None,
    plan: Optional[LoopPlan] = None,
) -> LoopIRCompiledKernel:
    """Lower one owned normalized CIN program through the LoopIR path."""

    binding_token = context.begin_stage(
        CompilerStageId.FRONTEND_VALIDATED_OPERATION_CONSTRUCTION,
        compile_options=options,
    )
    try:
        _bind_runtime_metadata(
            normalized,
            input_bindings,
            tuple(result_shape),
            input_formats,
        )
    except Exception:
        context.fail_stage(binding_token)
        raise
    context.complete_stage(binding_token)

    lowering_token = context.begin_stage(
        CompilerStageId.CIN_TO_LOOPIR_LOWERING,
        compile_options=options,
    )
    try:
        lowering = lower_normalized_cin_to_loopir(
            normalized,
            planned_loop_order=None if plan is None else plan.loop_order,
        )
    except Exception:
        context.fail_stage(lowering_token)
        raise
    context.complete_stage(lowering_token)

    schedule: Optional[ScheduledLoopIR] = None
    program: LoopProgram = lowering.program
    if plan is not None:
        schedule_token = context.begin_stage(
            CompilerStageId.LOOPIR_SCHEDULE_APPLICATION,
            compile_options=options,
        )
        try:
            schedule = apply_schedule_plan(lowering.program, plan)
        except Exception:
            context.fail_stage(schedule_token)
            raise
        context.complete_stage(schedule_token)
        program = schedule.program

    # The canonical strangler request identity, owned at this compile/shadow
    # request boundary.  Derived only from canonical semantic content — the
    # normalized CIN, the verified plan (or the unscheduled marker), the
    # result shape, and the runtime bindings — inside the stages already
    # recorded above; no new stage and no cache participate.
    from .plan_identity import loopir_request_dump

    request_dump = loopir_request_dump(
        normalized,
        plan,
        tuple(result_shape),
        tuple(input_bindings),
    )
    request_identity = hashlib.sha256(request_dump.encode("utf-8")).hexdigest()

    program_text = print_program(program)
    program_dump = canonical_program_dump(program)

    input_shapes: Dict[SymbolId, Tuple[int, ...]] = {}
    rhs_tensor_vars = normalized.get_rhs_tensor_vars()
    for tensor_var in rhs_tensor_vars:
        assert tensor_var.shape is not None
        input_shapes[tensor_var.symbol_id] = tuple(tensor_var.shape)

    llir_function = lower_loopir_to_llir(
        program,
        input_shapes=input_shapes,
        result_shape=tuple(result_shape),
        compile_options=options,
        compilation_context=context,
    )

    from ...ops import _lower_generated_llir

    cpp_source = _lower_generated_llir(llir_function, options, context)
    return LoopIRCompiledKernel(
        lowering=lowering,
        program_text=program_text,
        program_dump=program_dump,
        llir_function=llir_function,
        cpp_source=cpp_source,
        schedule=schedule,
        request_dump=request_dump,
        request_identity=request_identity,
    )


def compile_cin_via_loopir(
    cin_stmt: IndexStmt,
    result_shape: Sequence[int],
    input_bindings: Sequence[Tuple[Tuple[int, ...], torch.dtype]],
    *,
    compile_options: Optional[CompileOptions] = None,
    compilation_context: Optional[CompilationContext] = None,
) -> LoopIRCompiledKernel:
    """Lower one supported-family CIN program to C++ through LoopIR."""

    options = _resolve_options(compile_options, compilation_context)
    context = compilation_context
    if context is None:
        context = CompilationContext(options)
    plan: Optional[LoopPlan] = None
    if options.requested_schedule is not None:
        normalized, plan = _apply_requested_schedule(cin_stmt, options, context)
    else:
        normalized = normalize_cin(
            cin_stmt,
            compile_options=options,
            compilation_context=context,
        )
    return _compile_normalized_cin_via_loopir(
        normalized,
        result_shape,
        input_bindings,
        options=options,
        context=context,
        plan=plan,
    )


def legacy_generated_cpp(
    cin_stmt: IndexStmt,
    result_shape: Sequence[int],
    input_bindings: Sequence[Tuple[Tuple[int, ...], torch.dtype]],
    *,
    compile_options: Optional[CompileOptions] = None,
) -> str:
    """Generate the legacy pipeline's C++ for the same program, untouched.

    When the options carry a requested schedule, the legacy side is the
    scheduled route production uses: ``Scheduler.apply_schedule`` followed
    by the legacy lowering of the verified ``ScheduledCIN`` (which replays
    the legacy tree surgery inside the lowering adapter).
    """

    options = _resolve_options(compile_options)
    context = CompilationContext(options)
    lowering_stmt: Union[IndexStmt, ScheduledCIN]
    if options.requested_schedule is not None:
        from ..scheduler import Scheduler

        scheduled = Scheduler.apply_schedule(
            cin_stmt,
            options.requested_schedule,
            compile_options=options,
            compilation_context=context,
        )
        _bind_runtime_metadata(
            scheduled.normalized_cin, input_bindings, tuple(result_shape)
        )
        lowering_stmt = scheduled
    else:
        normalized = normalize_cin(cin_stmt, compile_options=options)
        _bind_runtime_metadata(normalized, input_bindings, tuple(result_shape))
        lowering_stmt = normalized
    lowerer = CINLowerer(compile_options=options, compilation_context=context)
    lowered = lowerer._lower_owned_IndexStmt(lowering_stmt)

    from ...ops import _lower_generated_llir

    return _lower_generated_llir(lowered, options, context)


def compare_generated_sources(
    cin_stmt: IndexStmt,
    result_shape: Sequence[int],
    input_bindings: Sequence[Tuple[Tuple[int, ...], torch.dtype]],
    *,
    compile_options: Optional[CompileOptions] = None,
) -> ShadowSourceComparison:
    """Curated shadow comparison of generated sources; never on by default.

    Both pipelines consume independent detached normalizations of the same
    semantic CIN, so neither can observe the other's working state.
    """

    options = _resolve_options(compile_options)
    loopir_kernel = compile_cin_via_loopir(
        copy.deepcopy(cin_stmt),
        result_shape,
        input_bindings,
        compile_options=options,
    )
    legacy_cpp = legacy_generated_cpp(
        copy.deepcopy(cin_stmt),
        result_shape,
        input_bindings,
        compile_options=options,
    )
    return ShadowSourceComparison(
        loopir_cpp=loopir_kernel.cpp_source,
        legacy_cpp=legacy_cpp,
    )


def execute_cin_via_loopir(
    cin_stmt: IndexStmt,
    result_shape: Sequence[int],
    *args,
    compile_options: Optional[CompileOptions] = None,
    time_dict: Optional[dict] = None,
    _compilation_context: Optional[CompilationContext] = None,
):
    """Compile through LoopIR and execute on ``args`` (STensor inputs).

    Mirrors ``scorch.ops.lower_and_exec_cin`` for the migrated families:
    the same public argument marshalling and JIT build/caching helpers,
    with result wrapping derived from the verified LoopIR declaration.
    """

    from ...stensor import STensor
    from ...storage import TensorIndex
    from ...ops import (
        _finalize_generated_mode_indices,
        _load_validated_prepared_kernel,
        _plan_mode_orders_to_loop_order,
        _prepare_generated_kernel_build,
    )

    options = _resolve_options(compile_options, _compilation_context)
    context = _compilation_context
    if context is None:
        context = CompilationContext(options)
    plan: Optional[LoopPlan] = None
    if options.requested_schedule is not None:
        normalized, plan = _apply_requested_schedule(cin_stmt, options, context)
    else:
        normalized = normalize_cin(
            cin_stmt,
            compile_options=options,
            compilation_context=context,
        )
    planning_token = context.begin_stage(
        CompilerStageId.FRONTEND_VALIDATED_OPERATION_CONSTRUCTION,
        compile_options=options,
    )
    try:
        _validate_runtime_formats(normalized, args)
        if plan is not None:
            alignment_plan = _plan_mode_orders_to_planned_order(normalized, args, plan)
            # A scheduled relayout executes with an independent schedule-free
            # context, but its success or failure still belongs to this
            # compilation's frontend binding stage.  Keep the parent token
            # active so elapsed time includes the prerequisite and a child
            # failure makes the caller-owned context terminal.
            args = _execute_mode_order_alignment(
                normalized,
                args,
                alignment_plan,
                options,
                context,
            )
        else:
            alignment_plan = _plan_mode_orders_to_loop_order(normalized, args)
    except Exception:
        context.fail_stage(planning_token)
        raise
    context.complete_stage(planning_token)
    if plan is None:
        # The unscheduled route shares its context with nested relayout
        # compilation, so its root frontend stage must finish first.
        args = _execute_mode_order_alignment(
            normalized,
            args,
            alignment_plan,
            options,
            context,
        )
    input_bindings = tuple((tuple(arg.shape), arg.dtype) for arg in args)
    kernel = _compile_normalized_cin_via_loopir(
        normalized,
        result_shape,
        input_bindings,
        options=options,
        context=context,
        input_formats=tuple(arg.format for arg in args),
        plan=plan,
    )
    header_cpp_code = options.build.preamble_source
    prepared_build = _prepare_generated_kernel_build(
        header_cpp_code,
        kernel.cpp_source,
        options,
        context,
    )
    module = _load_validated_prepared_kernel(prepared_build)

    module_args: List[object] = [result_shape]
    for arg in args:
        module_args.append(arg.shape)
        module_args.append(arg._native_mode_indices())
        module_args.append(arg.values)

    start_time = time.time()
    result_cpp = module.evaluate(*module_args)
    end_time = time.time()
    if time_dict is not None:
        time_dict["eval_time"] = end_time - start_time

    result_decl = next(
        decl
        for decl in kernel.lowering.program.tensors
        if decl.symbol == kernel.lowering.result_symbol
    )
    result_format = "".join(
        "d" if level.kind is LoopIRLevelKind.DENSE else "s"
        for level in result_decl.levels
    )
    result = STensor(
        shape=tuple(result_shape),
        index=TensorIndex(
            mode_indices=_finalize_generated_mode_indices(
                parse_format(result_format),
                result_cpp.storage.index.mode_indices,
            ),
            tensor_format=result_format,
        ),
        value=result_cpp.storage.value,
    )
    return result, kernel


def _execute_legacy_scheduled(
    cin_stmt: IndexStmt,
    result_shape: Sequence[int],
    *args,
    compile_options: CompileOptions,
):
    """Execute the legacy scheduled route on ``args`` (dense outputs only).

    The untouched low-level entry rejects requested schedules, so scheduled
    shadow execution drives the same production components directly:
    ``Scheduler.apply_schedule``, the legacy lowering of the verified
    ``ScheduledCIN``, and the shared JIT build/cache/execution helpers.
    Runtime mode-order prerequisites are aligned to the verified plan, and
    the dense result wrapper is derived from the normalized result rank.
    """

    from ...stensor import STensor
    from ...storage import TensorIndex
    from ...ops import (
        _finalize_generated_mode_indices,
        _load_validated_prepared_kernel,
        _lower_generated_llir,
        _prepare_generated_kernel_build,
    )
    from ..scheduler import Scheduler

    options = compile_options
    assert options.requested_schedule is not None
    context = CompilationContext(options)
    scheduled = Scheduler.apply_schedule(
        cin_stmt,
        options.requested_schedule,
        compile_options=options,
        compilation_context=context,
    )
    _validate_runtime_formats(scheduled.normalized_cin, args)
    alignment_plan = _plan_mode_orders_to_planned_order(
        scheduled.normalized_cin,
        args,
        scheduled.verified_loop_plan,
    )
    args = _execute_mode_order_alignment(
        scheduled.normalized_cin,
        args,
        alignment_plan,
        options,
        context,
    )
    _bind_runtime_metadata(
        scheduled.normalized_cin,
        tuple((tuple(arg.shape), arg.dtype) for arg in args),
        tuple(result_shape),
        tuple(arg.format for arg in args),
    )
    lowerer = CINLowerer(compile_options=options, compilation_context=context)
    lowered = lowerer._lower_owned_IndexStmt(scheduled)
    cpp_source = _lower_generated_llir(lowered, options, context)
    prepared_build = _prepare_generated_kernel_build(
        options.build.preamble_source,
        cpp_source,
        options,
        context,
    )
    module = _load_validated_prepared_kernel(prepared_build)

    module_args: List[object] = [result_shape]
    for arg in args:
        module_args.append(arg.shape)
        module_args.append(arg._native_mode_indices())
        module_args.append(arg.values)
    result_cpp = module.evaluate(*module_args)
    result_vars = scheduled.normalized_cin.get_result_tensor_vars()
    if len(result_vars) != 1 or result_vars[0].format is None:
        raise CompileSpecError(
            "legacy scheduled shadow execution requires one formatted result"
        )
    result_format = result_vars[0].format
    if not result_format.is_dense():
        raise CompileSpecError(
            "legacy scheduled shadow execution requires a dense result"
        )
    return STensor(
        shape=tuple(result_shape),
        index=TensorIndex(
            mode_indices=_finalize_generated_mode_indices(
                result_format, result_cpp.storage.index.mode_indices
            ),
            tensor_format=result_format,
        ),
        value=result_cpp.storage.value,
    )


def execute_shadow(
    cin_stmt: IndexStmt,
    result_shape: Sequence[int],
    *args,
    compile_options: Optional[CompileOptions] = None,
):
    """Execute both pipelines on ``args`` for curated differential tests.

    Returns ``(loopir_result, legacy_result, source_comparison)``.  The
    legacy execution uses the untouched low-level entry (or, when the
    options request a schedule, the legacy scheduled route through
    ``Scheduler.apply_schedule``); both wrappers support dense outputs.
    Sparse-output comparisons must use source parity plus direct
    LoopIR/PyTorch/oracle execution instead.
    """

    from ...ops import (
        _plan_mode_orders_to_loop_order,
        lower_and_exec_cin,
    )

    options = _resolve_options(compile_options)
    shadow_plan: Optional[LoopPlan] = None
    downstream_options = options
    if options.requested_schedule is not None:
        normalized, shadow_plan = _apply_requested_schedule(
            cin_stmt,
            options,
            CompilationContext(options),
        )
        # Freeze any policy-selected shorthand (notably
        # ``Schedule.loop_order=None``) to the verified plan before the
        # comparison invokes either pipeline again.  Runtime alignment below
        # may update compiler-owned mode-order metadata; it must not cause a
        # second policy selection to reinterpret the same shadow request.
        from ..scheduler import materialize_legacy_schedule

        canonical_schedule, _, _, _ = materialize_legacy_schedule(
            normalized,
            shadow_plan,
        )
        if canonical_schedule != options.requested_schedule:
            downstream_options = replace(
                options,
                requested_schedule=canonical_schedule,
            )
    else:
        normalized = normalize_cin(cin_stmt, compile_options=options)
    if any(
        tensor_var.format is None or not tensor_var.format.is_dense()
        for tensor_var in normalized.get_result_tensor_vars()
    ):
        raise CompileSpecError(
            "execute_shadow requires dense outputs because the legacy "
            "low-level result wrapper does not preserve sparse indices"
        )
    _validate_runtime_formats(normalized, args)
    if shadow_plan is None:
        alignment_plan = _plan_mode_orders_to_loop_order(normalized, args)
    else:
        alignment_plan = _plan_mode_orders_to_planned_order(
            normalized,
            args,
            shadow_plan,
        )
    # Align once before source comparison so compile-only bindings describe
    # logical storage, then both execution routes independently observe the
    # same already-aligned inputs.
    args = _execute_mode_order_alignment(
        normalized,
        args,
        alignment_plan,
        downstream_options,
        None,
    )

    comparison = compare_generated_sources(
        normalized,
        result_shape,
        tuple((tuple(arg.shape), arg.dtype) for arg in args),
        compile_options=downstream_options,
    )
    loopir_result, _ = execute_cin_via_loopir(
        copy.deepcopy(normalized),
        result_shape,
        *args,
        compile_options=downstream_options,
    )
    if downstream_options.requested_schedule is not None:
        legacy_result = _execute_legacy_scheduled(
            copy.deepcopy(normalized),
            result_shape,
            *args,
            compile_options=downstream_options,
        )
    else:
        legacy_result = lower_and_exec_cin(
            copy.deepcopy(normalized),
            result_shape,
            *args,
            _compile_options=downstream_options,
        )
    return loopir_result, legacy_result, comparison
