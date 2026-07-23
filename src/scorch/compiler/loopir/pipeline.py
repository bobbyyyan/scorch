"""Test/debug LoopIR strangler pipeline and curated shadow comparison.

This module is the executable end of the Phase-4 dense vertical slice: it
carries one normalized dense-family CIN program through

``normalized CIN -> LoopPlan -> LoopIR -> structured LLIR -> C++ ->
compiled kernel -> execution``

using the same JIT build, cache, and execution helpers as the legacy public
entry (:func:`scorch.ops.lower_and_exec_cin`).  It exists for dedicated
LoopIR tests and curated shadow comparison only:

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
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

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
from .lower_cin import LoopIRLoweringResult, lower_normalized_cin_to_loopir
from .lower_llir import lower_loopir_to_llir
from .printer import canonical_program_dump, print_program


@dataclass(frozen=True)
class LoopIRCompiledKernel:
    """One compiled dense-family kernel and its LoopIR provenance."""

    lowering: LoopIRLoweringResult
    program_text: str
    program_dump: str
    llir_function: llir.Function
    cpp_source: str


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
) -> CompileOptions:
    if compile_options is None:
        return CompileOptions.from_environment()
    if type(compile_options) is not CompileOptions:
        raise TypeError("compile_options must be a CompileOptions snapshot")
    if compile_options.requested_schedule is not None:
        raise CompileSpecError(
            "the LoopIR dense pipeline does not consume requested schedules"
        )
    return compile_options


def _bind_runtime_metadata(
    cin_stmt: IndexStmt,
    input_bindings: Sequence[Tuple[Tuple[int, ...], torch.dtype]],
    result_shape: Tuple[int, ...],
) -> None:
    """Bind shapes/dtypes exactly like the legacy public entry does."""

    rhs_tensor_vars = cin_stmt.get_rhs_tensor_vars()
    if len(rhs_tensor_vars) != len(input_bindings):
        raise CompileSpecError(
            f"CIN expects {len(rhs_tensor_vars)} runtime tensors, got "
            f"{len(input_bindings)}"
        )
    for tensor_var, (shape, dtype) in zip(rhs_tensor_vars, input_bindings):
        tensor_var.shape = tuple(shape)
        tensor_var.dtype = dtype
        tensor_var.mode_order = list(range(len(shape)))
    output_dtype = input_bindings[0][1] if input_bindings else torch.float32
    for tensor_var in cin_stmt.get_result_tensor_vars():
        tensor_var.shape = tuple(result_shape)
        tensor_var.dtype = output_dtype


def compile_cin_via_loopir(
    cin_stmt: IndexStmt,
    result_shape: Sequence[int],
    input_bindings: Sequence[Tuple[Tuple[int, ...], torch.dtype]],
    *,
    compile_options: Optional[CompileOptions] = None,
    compilation_context: Optional[CompilationContext] = None,
) -> LoopIRCompiledKernel:
    """Lower one dense-family CIN program to C++ through the LoopIR path."""

    options = _resolve_options(compile_options)
    context = compilation_context
    if context is None:
        context = CompilationContext(options)
    normalized = normalize_cin(cin_stmt, compile_options=options)
    _bind_runtime_metadata(normalized, input_bindings, tuple(result_shape))

    lowering_token = context.begin_stage(
        CompilerStageId.CIN_TO_LOOPIR_LOWERING,
        compile_options=options,
    )
    try:
        lowering = lower_normalized_cin_to_loopir(normalized)
    except Exception:
        context.fail_stage(lowering_token)
        raise
    context.complete_stage(lowering_token)

    input_shapes: Dict[SymbolId, Tuple[int, ...]] = {}
    rhs_tensor_vars = normalized.get_rhs_tensor_vars()
    for tensor_var in rhs_tensor_vars:
        assert tensor_var.shape is not None
        input_shapes[tensor_var.symbol_id] = tuple(tensor_var.shape)

    target_token = context.begin_stage(
        CompilerStageId.LOOPIR_TO_LLIR_LOWERING,
        compile_options=options,
    )
    try:
        llir_function = lower_loopir_to_llir(
            lowering.program,
            input_shapes=input_shapes,
            result_shape=tuple(result_shape),
            compile_options=options,
        )
    except Exception:
        context.fail_stage(target_token)
        raise
    context.complete_stage(target_token)

    from ...ops import _lower_generated_llir

    cpp_source = _lower_generated_llir(llir_function, options, context)
    return LoopIRCompiledKernel(
        lowering=lowering,
        program_text=print_program(lowering.program),
        program_dump=canonical_program_dump(lowering.program),
        llir_function=llir_function,
        cpp_source=cpp_source,
    )


def legacy_generated_cpp(
    cin_stmt: IndexStmt,
    result_shape: Sequence[int],
    input_bindings: Sequence[Tuple[Tuple[int, ...], torch.dtype]],
    *,
    compile_options: Optional[CompileOptions] = None,
) -> str:
    """Generate the legacy pipeline's C++ for the same program, untouched."""

    options = _resolve_options(compile_options)
    context = CompilationContext(options)
    normalized = normalize_cin(cin_stmt, compile_options=options)
    _bind_runtime_metadata(normalized, input_bindings, tuple(result_shape))
    lowerer = CINLowerer(compile_options=options, compilation_context=context)
    lowered = lowerer._lower_owned_IndexStmt(normalized)

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

    loopir_kernel = compile_cin_via_loopir(
        copy.deepcopy(cin_stmt),
        result_shape,
        input_bindings,
        compile_options=compile_options,
    )
    legacy_cpp = legacy_generated_cpp(
        copy.deepcopy(cin_stmt),
        result_shape,
        input_bindings,
        compile_options=compile_options,
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

    Mirrors ``scorch.ops.lower_and_exec_cin`` for the dense families: the
    same public argument marshalling, the same JIT build/caching helpers,
    and the same dense result wrapping.
    """

    from ...stensor import STensor
    from ...storage import TensorIndex
    from ...ops import (
        _finalize_generated_mode_indices,
        _load_validated_prepared_kernel,
        _prepare_generated_kernel_build,
    )

    options = _resolve_options(compile_options)
    context = _compilation_context
    if context is None:
        context = CompilationContext(options)
    input_bindings = tuple((tuple(arg.shape), arg.dtype) for arg in args)
    kernel = compile_cin_via_loopir(
        cin_stmt,
        result_shape,
        input_bindings,
        compile_options=options,
        compilation_context=context,
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

    result_rank = len(tuple(result_shape))
    result_format = "d" * result_rank
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


def execute_shadow(
    cin_stmt: IndexStmt,
    result_shape: Sequence[int],
    *args,
    compile_options: Optional[CompileOptions] = None,
):
    """Execute both pipelines on ``args`` for curated differential tests.

    Returns ``(loopir_result, legacy_result, source_comparison)``.  The
    legacy execution uses the untouched public entry.
    """

    from ...ops import lower_and_exec_cin

    comparison = compare_generated_sources(
        cin_stmt,
        result_shape,
        tuple((tuple(arg.shape), arg.dtype) for arg in args),
        compile_options=compile_options,
    )
    loopir_result, _ = execute_cin_via_loopir(
        copy.deepcopy(cin_stmt),
        result_shape,
        *args,
        compile_options=compile_options,
    )
    legacy_result = lower_and_exec_cin(
        copy.deepcopy(cin_stmt),
        result_shape,
        *args,
    )
    return loopir_result, legacy_result, comparison
