import itertools
import math
import os
import re
import string
import sys
import time
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
)

import scorch_ops as _scorch_ops
import torch
import torch.autograd.forward_ad as _fwad
from torch.fx import Proxy

from .compiler import compile_options as _compile_options_module
from .compiler import llir
from .compiler.cin import (
    IndexExpr,
    IndexVar,
    TensorVar,
    ForAll,
    Workspace,
    Where,
    TensorAssign,
    Operation,
    IndexStmt,
)
from .compiler.cin_lowerer import CINLowerer
from .compiler.cin_analysis import (
    _trusted_normalized_cin,
    normalize_cin,
    verify_cin_structure,
)
from .compiler.compile_options import (
    CompileOptions,
    environment_reuse_key,
    note_environment_parsed,
)
from .compiler.codegen import LLIRLowerer
from .compiler.diagnostics import (
    CompileOptionsDiagnostic,
    CompileOptionsError,
    ScheduleNotThreadedThroughSpMVError,
)
from .compiler.loop_plan import ScheduledCIN
from .compiler.scheduler import (
    FORCE_SEEN_CELL as _SCHEDULE_FORCE_SEEN,
    _ON_FIRST_SCHEDULE_FORCE,
    _SCHEDULE_FORCE,
    Schedule,
    Scheduler,
    get_forced_schedule,
)
from .compiler.compilation_context import (
    CompilerStageId,
    CompilerStageToken,
    CompilationContext,
    CompilationContextDiagnostic,
    CompilationContextError,
)
from .exceptions import (
    CompileSpecError,
    EinsumOperandCountError,
    EinsumRepeatedIndexError,
    EinsumResultLabelError,
    EinsumScalarResultError,
    IndexExtentMismatchError,
    KeywordArgumentError,
    MatmulOperandRankError,
    OperandRankMismatchError,
    ScheduleWithoutSparseOperandError,
    ScorchAssertionError,
    ScorchNotImplementedError,
    ScorchRuntimeError,
    ScorchTypeError,
    ScorchValueError,
    ShapeCapacitySpecError,
    TensorFormatError,
    TensorTypeError,
    defect_code,
    raise_if_native_coded,
    shape_capacity_error,
)
from .format import FormatInput, TensorFormat, LevelFormat, LevelType
from .format import (
    conversion_format,
    dense_prefix_unaddressable,
    stores_only_its_entries,
)
from .matmul_rules import matmul_result_rank, operand_shape, operands_expression
from .layout import (
    TensorLayout,
    TensorSpec,
    common_value_dtype,
    require_same_value_dtype,
)
from .plan import ENABLED_CELL as _PLAN_ENABLED
from .plan import GENERATION as _PLAN_GENERATION
from .plan import PLANS_ATTR as _PLANS_ATTR
from .plan import MAX_PLANS_PER_OPERAND as _PLAN_MAX_PER_OPERAND
from .plan import SEEN_ATTR as _PLAN_SEEN_ATTR
from .plan import TRANSPOSED_PLANS_ATTR as _PLAN_TRANSPOSED_ATTR
from .plan import install as _plan_install
from .plan import torch_operand_state as _plan_torch_operand_state
from .prebuilt_kernels import (
    csr_csr_symbol,
    csr_dense_symbol,
    execute_prebuilt_binary_kernel,
    resolve_prebuilt_matmul,
)
from .tiling import maybe_dispatch as _tiling_maybe_dispatch
from .tiling import is_candidate as _tiling_is_candidate
from .tiling import shape_is_candidate as _tiling_shape_is_candidate
from .tiling import decided as _tiling_decided
from .tiling import _current_level as _tiling_current_level
from .storage import _WRAP_GRAIN as _STORAGE_WRAP_GRAIN
from .stensor import (
    STensor,
    _generated_kernel,
    _wrap_generated_result,
    _compilation_context_at_boundary,
)

# Re-exported, not used here: the LoopIR pipeline module imports this name from `ops`
# at four sites, though it is defined in stensor.py. Dropping it because `ops` itself
# has no caller is what broke 351 tests in the merge -- "unused in this file" and
# "unused" are different claims for a module other modules import through. Kept on its
# own line so the suppression covers this name only.
from .stensor import _finalize_generated_mode_indices  # noqa: F401 - re-export
from . import autograd as _autograd
from . import einsum_grad as _einsum_grad
from .result_bound import EntryBound, entry_bound, level_totals_in_source
from .utils import (
    _PreparedJITBuild,
    parse_format,
    topo_sort_characters,
    _kernel_name,
    _load_validated_prepared_kernel,
    _prepare_jit_build,
)

_kernel_cache = {}
#: The result levels each kept kernel checks the total of before allocating them (the
#: two-pass assembly), read from its source when it is built (patch 0056): beside
#: ``_kernel_cache``, under the same key.  A key missing here counts no level as checked.
_kernel_level_totals: Dict[Any, FrozenSet[int]] = {}
_einsum_dispatch_cache = {}


def _requested_schedule(kwargs: dict) -> Optional[Schedule]:
    """Resolve only the explicit public/internal schedule spelling."""

    public = kwargs.get("schedule")
    internal = kwargs.get("_schedule")
    if public is not None and internal is not None and public != internal:
        raise ScorchValueError(
            "schedule and _schedule specify different schedules",
            code="conflicting_schedule",
        )
    schedule = public if public is not None else internal
    if schedule is not None and not isinstance(schedule, Schedule):
        raise ScorchTypeError(
            "schedule must be a scorch.compiler.scheduler.Schedule",
            code="schedule_type_invalid",
        )
    return schedule


def _effective_schedule(kwargs: dict) -> Optional[Schedule]:
    """Resolve an explicit or context-forced compiler schedule."""

    schedule = _requested_schedule(kwargs)
    return schedule if schedule is not None else get_forced_schedule()


def _compile_options_from_kwargs(kwargs: dict) -> CompileOptions:
    """Create or validate the one snapshot owned by a public compile call."""

    provided = kwargs.get("_compile_options")
    requested = _requested_schedule(kwargs)
    if provided is not None:
        if type(provided) is not CompileOptions:
            raise ScorchTypeError(
                "_compile_options must be a CompileOptions instance",
                code="compile_options_type_invalid",
            )
        if requested is not None and provided.requested_schedule != requested:
            raise CompileOptionsError(
                (
                    CompileOptionsDiagnostic(
                        code="conflicting_schedule",
                        field="requested_schedule",
                        message=(
                            "_compile_options and the requested schedule disagree"
                        ),
                    ),
                )
            )
        return provided

    return CompileOptions.from_environment(requested_schedule=requested)


def _compilation_context_from_kwargs(
    kwargs: dict,
    compile_options: CompileOptions,
) -> CompilationContext:
    """Create or validate the one timing owner paired with the one snapshot."""

    return _compilation_context_at_boundary(
        kwargs.get("_compilation_context"), compile_options
    )


class _DeferredOptions:
    """The options snapshot of one public call, made when the call first needs it.

    A call whose kernel is in the dispatch cache compiles nothing.  It needs only the
    snapshot's cache key, which decides whether an earlier call's kernel is the one it
    would select, plus the knowledge that its options would not be refused.  Making the
    snapshot anyway cost 85 us of a 329 us warm ``einsum`` on sapling (job 82682), with
    the parse itself kept.  So the key comes from the last parse under the same
    environment and schedule (``parsed_options_key``), and the snapshot is made
    (``options()``) only where something reads it: a compilation, or a relayout or
    conversion on the cached path.

    It is made at once, as before, when:
    - the call was handed one (``_compile_options``), which it validates.  Nothing else
      is read then: the snapshot already holds what the environment and the context
      held;
    - the call was handed a context (``_compilation_context``), which it validates
      against the snapshot;
    - no parse under the call's environment and schedule has yet succeeded in this
      process.  So the parse still refuses every environment it refuses, at the same
      point of the call.

    ``reuse_key`` is the options' part of the dispatch key: the snapshot's cache
    fingerprint (``CompileOptions.cache_fingerprint``, the SHA-256 of its cache key's
    canonical encoding), however the call came by it, so a call handed a snapshot and a
    call that made one from the same environment select the same kernel.  The
    fingerprint and not the cache key itself, because a dict lookup hashes its key on
    every call and CPython keeps no hash for a tuple: the cache key is a nested tuple of
    about 100 objects (the build flags, the include paths, every ``PATH`` entry), and
    hashing it visited each of them on every warm call, a cache miss apiece after the
    caches were evicted.  A string keeps its hash, and the fingerprint the parse memo
    hands back is the same string object on every call.  Two cache keys give the same
    fingerprint only when they are equal (the encoding tags each value's type), so
    the key selects the kernels it selected before.

    ``schedule_key`` is ``requested_schedule``'s cache key, read once here.  The parse
    memo and the dispatch key (``_einsum_cache_key``) both take this one read, so the
    two describe one state of the schedule even if it is changed during the call, and
    the call pays for one read rather than two.
    """

    __slots__ = (
        "_kwargs",
        "_options",
        "_environment_key",
        "requested_schedule",
        "reuse_key",
        "schedule_key",
    )

    def __init__(self, kwargs: dict) -> None:
        self._kwargs = kwargs
        self._options: Optional[CompileOptions] = None
        self._environment_key: Optional[tuple] = None
        if not kwargs:
            # A call with no keyword arguments, which every warm product of a model
            # makes: no snapshot or context was handed in and no schedule named, so the
            # schedule is the context's forced one.  The same steps as below, with the
            # context variable and the parse memo read in place of their accessors
            # (``get_forced_schedule``, ``_schedule_cache_key`` and
            # ``parsed_options_key``), so the holder makes one Python call,
            # ``environment_reuse_key``.
            key = environment_reuse_key()
            self._environment_key = key
            if key is not None:
                forced = _SCHEDULE_FORCE.get()
                forced_key = forced.cache_key if forced is not None else None
                known = _compile_options_module._parsed_options_keys.get(
                    (key, forced_key)
                )
                if known is not None:
                    self.requested_schedule: Optional[Schedule] = forced
                    self.schedule_key: Optional[str] = forced_key
                    self.reuse_key: str = known
                    return
            self._take_options()
            return
        if kwargs.get("_compile_options") is None:
            key = environment_reuse_key()
            self._environment_key = key
            if key is not None and kwargs.get("_compilation_context") is None:
                # The steps above with a schedule named first: ``_requested_schedule``
                # checks one named by either keyword, and answers None when neither is
                # given.  Then, as above, the context's forced schedule and the parse
                # memo are read in place of their accessors.
                requested = (
                    _requested_schedule(kwargs)
                    if "schedule" in kwargs or "_schedule" in kwargs
                    else None
                )
                schedule = requested if requested is not None else _SCHEDULE_FORCE.get()
                schedule_key = schedule.cache_key if schedule is not None else None
                known = _compile_options_module._parsed_options_keys.get(
                    (key, schedule_key)
                )
                if known is not None:
                    self.requested_schedule = schedule
                    self.schedule_key = schedule_key
                    self.reuse_key = known
                    return
        self._take_options()

    def _take_options(self) -> None:
        """Take the schedule and the reuse key from the call's snapshot."""

        options = self.options()
        self.requested_schedule = options.requested_schedule
        self.schedule_key = _schedule_cache_key(self.requested_schedule)
        self.reuse_key = options.cache_fingerprint

    def options(self) -> CompileOptions:
        """The call's snapshot, made now if need be."""

        if self._options is None:
            options = _compile_options_from_kwargs(self._kwargs)
            if self._environment_key is not None:
                note_environment_parsed(self._environment_key, options)
            self._options = options
        return self._options

    def still_reusable(self) -> bool:
        """Whether a kernel built during the call may be kept under ``reuse_key``.

        Only when the snapshot it was built from has that fingerprint.  It may not when
        the host changed since the parse ``reuse_key`` came from (a header file, the
        working directory): the kernel then belongs to the new snapshot, and the next
        call, keyed by the new snapshot through ``parsed_options_key``, builds and keeps
        its own.  And only when the schedule still has the state ``schedule_key`` was
        read from: one changed during the call (with ``object.__setattr__``) may have
        been compiled in its new state, and that kernel does not belong under the key
        of the old one.  A schedule whose new state is invalid is refused here, as at
        every read.
        """

        options_kept = (
            self._options is None or self._options.cache_fingerprint == self.reuse_key
        )
        return options_kept and (
            _schedule_cache_key(self.requested_schedule) == self.schedule_key
        )


class _DeferredCompilation:
    """The compilation context of one public call, made when the call first needs one.

    A context times the stages of a compilation.  A call whose kernel is already in the
    dispatch cache compiles nothing: it begins the frontend stage, cancels it at the
    cache hit, and the context is dropped unread.  Making it, beginning the stage and
    cancelling it cost 24 us of a 390 us warm ``einsum`` call on sapling (job 82239),
    against a kernel of about 146 us.  So a call that was not handed a context
    makes one only when it reaches a stage that records something: the frontend stage
    of a compilation, which then starts at the reading taken when the call began it
    here, or a relayout or conversion on the cached path.  A call handed a context
    (``_compilation_context``) begins its frontend stage at once, as before, since the
    caller reads that context afterwards.
    """

    __slots__ = ("_options", "_context", "_frontend", "_started_ns", "_cancelled")

    def __init__(self, kwargs: dict, options: _DeferredOptions) -> None:
        self._options = options
        provided = kwargs.get("_compilation_context")
        self._context: Optional[CompilationContext] = None
        self._frontend: Optional[CompilerStageToken] = None
        self._cancelled = False
        if provided is not None:
            compile_options = options.options()
            self._context = _compilation_context_at_boundary(provided, compile_options)
            self._frontend = self._context.begin_stage(
                CompilerStageId.FRONTEND_VALIDATED_OPERATION_CONSTRUCTION,
                compile_options=compile_options,
            )
        self._started_ns = time.perf_counter_ns()

    def frontend(self) -> Tuple[CompilationContext, CompilerStageToken]:
        """The context and its active frontend stage, made now if need be."""

        if self._cancelled:
            raise CompilationContextError(
                CompilationContextDiagnostic(
                    code="cancelled_frontend_stage",
                    message="the frontend stage of this call was already cancelled",
                    stage_id=CompilerStageId.FRONTEND_VALIDATED_OPERATION_CONSTRUCTION,
                )
            )
        if self._context is None:
            self._context = CompilationContext(compile_options=self._options.options())
        if self._frontend is None:
            self._frontend = self._context.begin_stage(
                CompilerStageId.FRONTEND_VALIDATED_OPERATION_CONSTRUCTION,
                compile_options=self._options.options(),
                started_ns=self._started_ns,
            )
        return self._context, self._frontend

    def cancel_frontend(self) -> None:
        """Retire the frontend stage at a cache hit: it produced no artifact."""

        if self._frontend is not None and self._context is not None:
            self._context.cancel_stage(self._frontend)
        self._cancelled = True

    def context(self) -> CompilationContext:
        """The context after the frontend stage was cancelled, made now if need be."""

        if self._context is None:
            self._context = CompilationContext(compile_options=self._options.options())
        return self._context

    def nested_context(self) -> Optional[CompilationContext]:
        """The context a conversion inside the call records into, or None.

        The caller's context when it handed one, which reads the conversion's stages
        afterwards; else one this call already made.  None otherwise: a conversion that
        hits its kernel cache then needs none, and one that misses makes its own.
        """

        return self._context

    def fail_frontend_if_active(self) -> None:
        context, token = self._context, self._frontend
        if context is not None and token is not None and context.is_stage_active(token):
            context.fail_stage(token)


def _lower_generated_llir(
    lowered_llir: Union[llir.Stmt, List[llir.Stmt]],
    compile_options: CompileOptions,
    compilation_context: CompilationContext,
) -> str:
    """Render final generated LLIR inside the canonical C++ stage."""

    llir_lowerer = LLIRLowerer(compile_options=compile_options)
    cpp_token = compilation_context.begin_stage(
        CompilerStageId.LLIR_TO_CPP_GENERATION,
        compile_options=compile_options,
    )
    try:
        cpp_code = llir_lowerer.lower_llir(lowered_llir)
    except Exception:
        compilation_context.fail_stage(cpp_token)
        raise
    compilation_context.complete_stage(cpp_token)
    return cpp_code


def _prepare_generated_kernel_build(
    header_cpp_code: str,
    cpp_code: str,
    compile_options: CompileOptions,
    compilation_context: CompilationContext,
) -> _PreparedJITBuild:
    """Assemble and validate the generated kernel request before native work."""

    kernel_name_token = compilation_context.begin_stage(
        CompilerStageId.KERNEL_NAME_AND_BUILD_REQUEST_ASSEMBLY,
        compile_options=compile_options,
    )
    try:
        kernel_name = _kernel_name(
            header_cpp_code,
            cpp_code,
            compile_options=compile_options,
        )
        prepared_build = _prepare_jit_build(
            name=kernel_name,
            cpp_sources=[header_cpp_code, cpp_code],
            functions=["evaluate"],
            extra_cflags=list(compile_options.build.extra_cflags),
            extra_ldflags=list(compile_options.build.extra_ldflags),
            compile_options=compile_options,
        )
    except Exception:
        compilation_context.fail_stage(kernel_name_token)
        raise
    compilation_context.complete_stage(kernel_name_token)
    return prepared_build


def _matmul_compile_policy(
    kwargs: dict,
) -> Tuple[Optional[CompileOptions], Optional[Schedule]]:
    """Resolve an existing snapshot or defer creation until compilation starts."""

    if kwargs.get("_compile_options") is None:
        return None, _effective_schedule(kwargs)
    options = _compile_options_from_kwargs(kwargs)
    return options, options.requested_schedule


def _schedule_cache_key(schedule: Optional[Schedule]) -> Optional[str]:
    return schedule.cache_key if schedule is not None else None


#: ``_einsum_cache_key``'s default for ``schedule_key``: the caller did not read it.
_SCHEDULE_KEY_UNREAD = object()


def _reject_unsupported_requested_schedule(
    compile_options: CompileOptions,
    boundary: str,
) -> None:
    """Fail closed when a compile entry cannot apply a requested schedule."""

    if compile_options.requested_schedule is not None:
        raise CompileOptionsError(
            (
                CompileOptionsDiagnostic(
                    code="unsupported_requested_schedule",
                    field="requested_schedule",
                    message=f"{boundary} does not support explicit scheduling",
                ),
            )
        )


def _layout_contract(tensor: Any) -> Any:
    """An operand's layout as ``_einsum_cache_key`` keys it."""

    layout = getattr(tensor, "layout", None)
    if isinstance(layout, TensorLayout):
        # The layout *is* the contract. It is a frozen dataclass over logical shape,
        # physical shape, format, permutation and index dtype, so the __eq__ and
        # __hash__ it already has distinguish exactly what a dispatch key has to
        # distinguish -- test_einsum_dispatch_key_discriminates_every_layout_field
        # pins that against serialize() field by field. This cache is an in-process
        # dict, so there is nothing the string bought: serializing to JSON to get a
        # dict key cost 2.32 us per sparse operand on an M5 for no information at
        # all, and 3.20 us for the whole key -- 7% of the smallest einsum there is.
        return layout
    serialize = getattr(layout, "serialize", None)
    if callable(serialize):
        # Not one of ours, but it offers its own canonical string, so keep keying on
        # that rather than dropping to the weaker tensor-like contract below. Every
        # `.layout` in this tree returns a TensorLayout, so this branch costs one
        # failed isinstance on the hot path and is otherwise unreachable.
        return serialize()
    # Scheduling tools may use lightweight tensor-like objects. Preserve a
    # deterministic fallback without weakening real tensors' canonical key.
    shape = getattr(tensor, "shape", ()) or ()
    mode_order = getattr(tensor, "mode_order", ()) or ()
    return (
        "tensor_like",
        str(getattr(tensor, "format", None)),
        tuple(shape),
        tuple(mode_order),
        str(getattr(tensor, "index_dtype", None)),
    )


def _runtime_key(value: Any) -> Any:
    """An operand's dtype or device as ``_einsum_cache_key`` keys it: torch's own
    ``torch.dtype`` and ``torch.device`` as they are, else the value's string.

    Two of torch's dtypes or devices are equal exactly when their strings are, and
    they hash, so keying the objects keys what the strings keyed without making two
    strings per operand and call.  A value of another type (a tensor-like's) keeps
    its string, as the key held before."""

    kind = type(value)
    if kind is torch.dtype or kind is torch.device:
        return value
    return str(value)


def _einsum_cache_key(
    expression: str,
    tensors: Sequence[Any],
    output_format: Any,
    output_mode_order: Any,
    schedule: Optional[Schedule],
    compile_options: Optional[CompileOptions] = None,
    *,
    options_key: object = None,
    schedule_key: object = _SCHEDULE_KEY_UNREAD,
) -> tuple:
    """Build the early dispatch key, including every scheduling decision.

    The options enter as ``compile_options.cache_fingerprint`` when a snapshot is
    given, else as ``options_key`` (``_DeferredOptions.reuse_key``, the same
    fingerprint), which names them without making them.  The schedule enters as
    ``schedule_key`` when the caller has read it (``_DeferredOptions.schedule_key``,
    ``schedule``'s key read once per call), else as ``schedule.cache_key`` read here.

    An exact STensor's layout, dtype and device are read from its metadata, where its
    ``layout``, ``dtype`` and ``device`` properties read them.  Every einsum that
    reaches a kernel builds this key, so it is one loop and no nested function.
    """

    contracts = []
    runtimes = []
    for t in tensors:
        if type(t) is STensor:
            metadata = t._metadata
            layout = metadata.layout
            dtype = metadata.dtype
            device = metadata.device
            if (
                type(layout) is TensorLayout
                and type(dtype) is torch.dtype
                and type(device) is torch.device
            ):
                # What the three calls below give such an operand, read in place.
                contracts.append(layout)
                runtimes.append((dtype, device))
                continue
        contracts.append(_layout_contract(t))
        runtimes.append(
            (
                _runtime_key(getattr(t, "dtype", None)),
                _runtime_key(getattr(t, "device", "cpu")),
            )
        )
    return (
        expression,
        tuple(contracts),
        tuple(runtimes),
        str(output_format) if output_format is not None else None,
        tuple(output_mode_order) if output_mode_order else None,
        (
            _schedule_cache_key(schedule)
            if schedule_key is _SCHEDULE_KEY_UNREAD
            else schedule_key
        ),
        (
            compile_options.cache_fingerprint
            if compile_options is not None
            else options_key
        ),
    )


def _codegen_kernel_cache_key(
    cin: Union[IndexStmt, ScheduledCIN],
    post_ops: Any,
    schedule: Optional[Schedule],
    compile_options: Optional[CompileOptions] = None,
) -> str:
    """Cache a lowered CIN without allowing dtype/schedule fields to alias."""
    normalized_cin = cin.normalized_cin if isinstance(cin, ScheduledCIN) else cin
    tensor_contracts = tuple(
        sorted(
            {
                (
                    access.tensor.name,
                    str(access.tensor.dtype),
                    access.tensor.shape,
                    str(access.tensor.format),
                    tuple(access.tensor.mode_order or ()),
                )
                for access in normalized_cin.tensor_accesses
            }
        )
    )
    key = str(normalized_cin) + f"|contracts:{tensor_contracts!r}"
    if post_ops:
        key += f"|post_ops:{post_ops}"
    if schedule is not None:
        key += f"|schedule:{schedule.cache_key}"
    if compile_options is not None:
        key += f"|compile_options:{compile_options.cache_fingerprint}"
    return key


# An einsum expression's labels are checked character by character on every call:
# `isascii()`/`isalpha()` per label, a set per operand group, a set for the result, and a
# membership test per result label. That is a pure function of the expression string --
# the same ~2 us of it on every iteration of a training loop -- so memoize it. The memo is
# consulted in exactly the position the checks ran in, after the operand count/None/type
# checks, so a malformed call still raises the same exception it raised before.
#
# The bound exists for the reason the layout cache has one: a program that generates
# expressions must not grow this without limit. Past the bound the checks simply run,
# which is what happened before the memo, so a bench can set it to 0 for a same-binary
# control arm.
_EXPR_LABELS_CACHE: Dict[str, Tuple[List[List[str]], List[str]]] = {}
_EXPR_LABELS_CACHE_MAX = 512


def _refuse_repeated_index(input_groups: Sequence[str], result_expression: str) -> None:
    """Raise ``EinsumRepeatedIndexError`` if one term of an expression repeats a letter.

    An operand term that repeats a letter (``"ii->i"``, ``"iij->ij"``) asks for that
    operand's diagonal.  No kernel computes one: the loop-order sort and the operand
    relayout both take a term's letters to be distinct, so the expression is refused
    here, before either sees it.  A result term that repeats a letter has no meaning.
    """
    for operand, group in enumerate(input_groups):
        if len(set(group)) != len(group):
            index = next(label for label in group if group.count(label) > 1)
            raise EinsumRepeatedIndexError(
                f"einsum operand {operand} repeats index {index!r} in its term "
                f"{group!r}; the diagonal of an operand is not computed",
                operand,
                index,
            )
    if len(set(result_expression)) != len(result_expression):
        index = next(
            label for label in result_expression if result_expression.count(label) > 1
        )
        raise EinsumRepeatedIndexError(
            f"einsum result labels must be unique; {result_expression!r} repeats "
            f"index {index!r}",
            None,
            index,
        )


def _validated_labels(
    expression: str,
    input_groups: Sequence[str],
    result_expression: str,
) -> Tuple[List[List[str]], List[str]]:
    """Check an einsum expression's labels; return them split per operand and result.

    ``input_groups`` and ``result_expression`` must be the two halves of ``expression``,
    which is the memo key: the caller has already split it, and re-splitting here to
    prove it would cost what the memo saves.
    """
    cached = _EXPR_LABELS_CACHE.get(expression)
    if cached is not None:
        # Fresh lists: callers downstream sort and rewrite what they get back.
        return [list(group) for group in cached[0]], list(cached[1])
    # An empty term is a rank-0 tensor: a result that sums over every index (``"ij->"``),
    # or an operand that holds one value (``",i->i"``).
    if any(
        not all(label.isascii() and label.isalpha() for label in group)
        for group in input_groups
    ):
        raise CompileSpecError(
            "einsum input labels must be ASCII letters",
            code="einsum_label_invalid",
        )
    if not all(label.isascii() and label.isalpha() for label in result_expression):
        raise CompileSpecError(
            "einsum result labels must be ASCII letters",
            code="einsum_label_invalid",
        )
    if not result_expression and not any(input_groups):
        raise EinsumScalarResultError(
            f"einsum {expression!r} has no index: its scalar result is a product "
            "without a loop, which no kernel computes"
        )
    _refuse_repeated_index(input_groups, result_expression)
    input_labels = set("".join(input_groups))
    unknown_result_labels = [
        label for label in result_expression if label not in input_labels
    ]
    if unknown_result_labels:
        raise EinsumResultLabelError(
            "einsum result labels must appear in an input; unknown labels: "
            + ", ".join(unknown_result_labels)
        )
    input_index_strs = [list(group) for group in input_groups]
    result_index_strs = list(result_expression)
    if len(_EXPR_LABELS_CACHE) < _EXPR_LABELS_CACHE_MAX:
        _EXPR_LABELS_CACHE[expression] = (
            [list(group) for group in input_index_strs],
            list(result_index_strs),
        )
    return input_index_strs, result_index_strs


def _einsum_tensor_names() -> Iterator[str]:
    """The names einsum gives its operands, in order, and then its result: ``A`` to
    ``Z``, then ``AA``, ``AB``, ..., ``ZZ``, ``AAA`` and so on, without end.

    The first 26 are the single letters einsum has always used, in the same order, so a
    call with 25 operands or fewer names its tensors as before and emits the same
    kernel. A 26th operand used to leave no name for the result (``IndexError: pop from
    empty list``).

    Capital letters only, and no digit: every identifier a kernel builds from a tensor
    name puts a digit, an underscore or nothing after the name, and ``p``, an index name
    with an underscore, or nothing before it (``A_values``, ``A1_crd``, ``pA1_end``,
    ``i_A``, ``M_A_coord``). A run of capitals that is not an index name is therefore
    one whole tensor name, and two names never share an identifier; a name that ends in
    a digit cannot promise that (``T26`` at level 1 and ``T261`` would both give
    ``pT261``).

    No name is left out. Some of them would complete a macro of the C library behind an
    upper-case index letter and an underscore (``OK``: ``F_OK``; ``PI``: ``M_PI``;
    ``IFMT``: ``S_IFMT``); ``scorch.compiler.merged_coordinate`` spells the one
    identifier of that form so that it cannot, and its tests read the preprocessor's
    macro list against every identifier a kernel builds from a tensor name of any
    length.
    """

    length = 1
    while True:
        for letters in itertools.product(string.ascii_uppercase, repeat=length):
            yield "".join(letters)
        length += 1


def _logical_index_sizes(
    input_index_strs: Sequence[Sequence[str]],
    tensors: Sequence[Any],
) -> dict:
    """Map logical einsum indices to sizes after physical mode reordering.

    Each operand's mode order and shape are read once (every einsum that reaches a
    kernel asks this), and an index's size is stored the first time it is seen.
    """
    index_to_size: Dict[str, int] = {}
    for operand, (index_strs, tensor) in enumerate(zip(input_index_strs, tensors)):
        mode_order = tuple(tensor.mode_order)
        if len(index_strs) != len(mode_order):
            raise OperandRankMismatchError(
                f"einsum operand {operand} has {len(index_strs)} indices but "
                f"tensor rank {len(mode_order)}"
            )
        shape = tensor.shape
        for logical_axis, index_str in enumerate(index_strs):
            size = shape[mode_order.index(logical_axis)]
            previous = index_to_size.get(index_str)
            if previous is None:
                index_to_size[index_str] = size
            elif previous != size:
                raise IndexExtentMismatchError(
                    f"einsum index {index_str!r} has incompatible dimensions "
                    f"{previous} and {size}"
                )
    return index_to_size


def _post_op_tensor_in_storage_order(
    tensor: Any,
    logical_shape: Sequence[int],
    storage_order: Sequence[int],
) -> Any:
    """A post-op tensor laid out the way the kernel stores the result's cells.

    The kernel reads a post-op tensor at the result's storage position
    (``CINLowerer._emit_post_ops``, ``ResultTensorAssembler.emit_value_post_ops``).
    The caller's tensor holds the result's entries in row-major order of its
    logical shape.  When the kernel stores the result in another mode order
    (``storage_order[physical] = logical``, the order einsum picks from the
    loop order), an entry sits at a different position there, so the tensor is
    permuted to that order and made contiguous.  A tensor whose element count
    is not the result's is passed unchanged; the kernel's argument check then
    refuses it with its code.
    """

    if list(storage_order) == list(range(len(storage_order))):
        return tensor
    if not isinstance(tensor, torch.Tensor) or tensor.numel() != math.prod(
        logical_shape
    ):
        return tensor
    return (
        tensor.reshape(tuple(logical_shape)).permute(tuple(storage_order)).contiguous()
    )


def _storage_letters(tensor_format: TensorFormat) -> str:
    """One letter per storage level: ``"d"`` dense, ``"s"`` any level that stores
    only the coordinates put in it (compressed, coordinate or singleton)."""

    return "".join(
        "d" if level_type is LevelType.DENSE else "s"
        for level_type in tensor_format.get_level_types()
    )


@dataclass(frozen=True)
class _OperandRelayout:
    """How einsum relays one operand; see :func:`_einsum_operand_relayout`.

    ``mode_orders`` are the storage orders the operand is moved to, in turn
    (empty: it stays as it is stored).  ``format_letters`` are its letters after
    the last move, or ``None`` to keep its own format.  ``keep_every_stored_entry``
    copies every stored entry whatever its value; otherwise a move keeps only the
    entries whose value is nonzero, as the relayout always has.
    """

    mode_orders: Tuple[Tuple[int, ...], ...]
    format_letters: Optional[str]
    keep_every_stored_entry: bool

    def move_kwargs(self) -> Dict[str, Any]:
        """The keyword arguments :func:`_with_compiler_mode_order` takes for a move."""

        if self.format_letters is None and not self.keep_every_stored_entry:
            return {}
        return {
            "format_letters": self.format_letters,
            "keep_every_stored_entry": self.keep_every_stored_entry,
        }


def _einsum_operand_relayout(
    operand: int,
    operand_indices: Sequence[Sequence[str]],
    operand_letters: Sequence[str],
    input_orders: Sequence[Sequence[int]],
    aligned_orders: Sequence[Sequence[int]],
    result_indices: Sequence[str],
    result_letters: str,
    result_order: Sequence[int],
    extents: Mapping[str, int],
) -> _OperandRelayout:
    """Decide how einsum moves operand ``operand`` to the storage order its kernel reads.

    Metadata only.  For every operand: its index letters (``operand_indices``,
    logical order), its storage letters (``operand_letters``, one per storage
    level, from :func:`_storage_letters`) and its storage order on entry
    (``input_orders``, storage level -> logical mode).  For this operand, the
    orders einsum's alignment moves it to, in turn (``aligned_orders``: the
    topological order, then the selected loop order's).  For the result: its index
    letters, storage letters and storage order.  ``extents`` maps each index
    letter to its extent.

    What a move stores.  For a tensor stored with letters ``L`` under an order,
    let ``A`` be the modes at or above its deepest compressed level and ``D`` the
    modes below it (all on dense levels).  It stores exactly the projection of
    its entries onto ``A`` times every cell of ``D``: a dense level stores every
    cell under each stored parent.  A move keeps ``L`` by storage position.  So
    after moves to orders ``o1, o2, ...`` the operand stores cells that differ
    from a cell it stored only in the modes
    ``F = (D(o1) | D(o2) | ...) - D(input order)``.  A dense level of extent 1
    has one cell, so a mode of extent 1 adds no cell.

    When an added cell changes the result.  A mode ``m`` is exposed when its value
    can decide a stored result coordinate: ``m`` is a result mode at or above the
    result's deepest compressed level (result storage order), or another operand
    holds ``m`` at or above its own deepest compressed level (its input order; it
    may store only some values of ``m``).  If no mode of ``F`` with extent >= 2 is
    exposed, any combination of entries that reaches a result coordinate through
    an added cell reaches the same coordinate through the stored cell it differs
    from, so the result's stored coordinates are unchanged; ``F`` of two operands
    are disjoint whenever neither is converted (a mode in both would be exposed to
    each by the other), so the substitution is consistent.  If one is exposed, an
    input exists whose result stores a coordinate the entries do not imply.  When
    some index has extent 0 no operand holding it stores an entry and the
    iteration space is empty, so nothing is added either.

    The decision.
      - The operand does not move: nothing to decide.
      - The result has no compressed level (it stores every cell), or the operand
        has none (it stores every cell, so a zero value is a cell like any
        other): the moves einsum always made, letters kept, nonzero values
        copied.  Kernel text is unchanged.
      - Otherwise every stored entry is copied, whatever its value: copying only
        nonzero values drops entries the operand stores explicitly as zero, and
        the result then misses the coordinates they imply.
      - Converted, when an exposed mode of ``F`` has extent >= 2 and no index has
        extent 0: one move from the input order to the last aligned order, with
        letters :func:`scorch.format.conversion_format` (every level compressed),
        which stores exactly the operand's entries; the main kernel is compiled
        for those letters.  The operand moves once because a move that keeps
        ``L`` adds cells a later conversion would keep.  If the last aligned order
        is the input order, it does not move at all.
      - Otherwise the moves einsum always made, letters kept.

    One case comes before all of these: a move that keeps ``L`` can put a dense level
    on a mode whose extent the operand's own layout never multiplied (``"ds"`` of
    shape (1, 2**31 - 1) moved to its column order has a dense level of 2**31 - 1
    positions for two stored entries).  When the dense levels a moved layout starts
    with hold more positions than a kernel addresses
    (:func:`scorch.format.dense_prefix_unaddressable`), no kernel could read the moved
    operand, and building it would allocate that many position entries first.  The
    operand is then converted as above, with every level compressed, which stores its
    entries and nothing else; every stored entry is copied.
    """

    letters = operand_letters[operand]
    input_order = tuple(input_orders[operand])
    moves: List[Tuple[int, ...]] = []
    current = input_order
    for aligned_order in aligned_orders:
        order = tuple(aligned_order)
        if order != current:
            moves.append(order)
            current = order
    if not moves:
        return _OperandRelayout((), None, False)
    indices = operand_indices[operand]
    if any(
        dense_prefix_unaddressable(letters, [extents[indices[mode]] for mode in order])
        for order in moves
    ):
        # Unaddressable: converted, one move, whatever the result stores.
        final_order = moves[-1]
        if final_order == input_order:
            return _OperandRelayout((), None, False)
        return _OperandRelayout((final_order,), "s" * len(letters), True)
    if "s" not in result_letters or "s" not in letters:
        return _OperandRelayout(tuple(moves), None, False)
    if stores_only_its_entries(letters) or any(
        extent == 0 for extent in extents.values()
    ):
        return _OperandRelayout(tuple(moves), None, True)

    def above_and_below(
        indices: Sequence[str], level_letters: str, order: Sequence[int]
    ) -> Tuple[set, set]:
        deepest = max(
            (level for level, letter in enumerate(level_letters) if letter != "d"),
            default=-1,
        )
        modes = [indices[mode] for mode in order]
        return set(modes[: deepest + 1]), set(modes[deepest + 1 :])

    input_below = above_and_below(indices, letters, input_order)[1]
    added = (
        set().union(*(above_and_below(indices, letters, order)[1] for order in moves))
        - input_below
    )
    exposed = above_and_below(result_indices, result_letters, result_order)[0]
    for other, other_indices in enumerate(operand_indices):
        if other != operand:
            exposed |= above_and_below(
                other_indices, operand_letters[other], input_orders[other]
            )[0]
    if any(extents[mode] >= 2 for mode in added & exposed):
        final_order = moves[-1]
        if final_order == input_order:
            return _OperandRelayout((), None, False)
        return _OperandRelayout((final_order,), conversion_format(letters), True)
    return _OperandRelayout(tuple(moves), None, True)


def _bind_frontend_operand_mode_orders(
    input_index_strs: Sequence[Sequence[str]],
    selected_loop_order_names: Sequence[str],
    tensors: Sequence[Union[STensor, TensorSpec]],
    tensor_vars: Sequence[TensorVar],
) -> List[List[int]]:
    """Bind final frontend layout metadata without executing runtime relayouts."""

    desired_mode_orders: List[List[int]] = []
    for tensor_index, input_index_str in enumerate(input_index_strs):
        desired_index_strs = [
            index_str
            for index_str in selected_loop_order_names
            if index_str in input_index_str
        ]
        desired_mode_order = [
            input_index_str.index(index_str) for index_str in desired_index_strs
        ]
        desired_mode_orders.append(desired_mode_order)
        tensor_vars[tensor_index].shape = tuple(
            tensors[tensor_index].logical_shape[mode] for mode in desired_mode_order
        )
        if tensor_vars[tensor_index].mode_order != desired_mode_order:
            tensor_vars[tensor_index].mode_order = desired_mode_order
    return desired_mode_orders


def _select_and_bind_einsum_operand_mode_orders(
    cin_stmt: IndexStmt,
    input_index_strs: Sequence[Sequence[str]],
    tensors: Sequence[Union[STensor, TensorSpec]],
    tensor_vars: Sequence[TensorVar],
    effective_schedule: Optional[Schedule],
    compile_options: CompileOptions,
    compilation_context: CompilationContext,
) -> List[List[int]]:
    """Record the schedule-derived operand-alignment decision before relayout."""

    prealignment_token = compilation_context.begin_stage(
        CompilerStageId.SCHEDULING_AND_LOOP_PLAN_CONSTRUCTION,
        compile_options=compile_options,
    )
    try:
        if effective_schedule is not None and effective_schedule.loop_order is not None:
            selected_loop_order = Scheduler.resolve_loop_order(
                cin_stmt, effective_schedule.loop_order
            )
        else:
            selected_loop_order = Scheduler.select_loop_order(
                cin_stmt,
                costs=compile_options.scheduler.cost_model,
            )
        selected_loop_order_names = [
            index_var.name for index_var in selected_loop_order
        ]
        desired_mode_orders = _bind_frontend_operand_mode_orders(
            input_index_strs,
            selected_loop_order_names,
            tensors,
            tensor_vars,
        )
    except Exception:
        compilation_context.fail_stage(prealignment_token)
        raise
    compilation_context.complete_stage(prealignment_token)
    return desired_mode_orders


def _conversion_compile_inputs(
    compile_options: CompileOptions,
    compilation_context: Optional[CompilationContext],
) -> Tuple[CompileOptions, Optional[CompilationContext]]:
    """The options and timing owner for a conversion kernel inside a compile call.

    A requested schedule belongs to the caller's own kernel.  A mode-order or
    format conversion the call performs on an operand or result is a separate
    kernel with no schedule of its own, and plain CIN lowering refuses -- by
    design -- to ignore a requested schedule.  So the conversion lowers with the
    schedule removed, under its own timing owner, exactly as the typed
    pipeline's mode-order relayout prerequisite does.  Without a requested
    schedule the caller's pair is returned unchanged.
    """

    if compile_options.requested_schedule is None:
        return compile_options, compilation_context
    conversion_options = replace(compile_options, requested_schedule=None)
    return conversion_options, CompilationContext(compile_options=conversion_options)


def _conversion_kwargs(
    options: Union[CompileOptions, _DeferredOptions],
    compilation_context: Optional[CompilationContext],
) -> Dict[str, Any]:
    """The keyword arguments that hand a conversion inside a compile call its options.

    A call's options holder (``_DeferredOptions``) is handed on as it is, so the
    conversion and the call make at most one snapshot between them, and a conversion
    that hits its kernel cache makes none.  Under a requested schedule the conversion
    needs the schedule-free copy of the snapshot (``_conversion_compile_inputs``), so
    the snapshot is made, as it always was for these conversions.
    """

    if type(options) is _DeferredOptions:
        if options.requested_schedule is None:
            return {
                "_deferred_options": options,
                "_compilation_context": compilation_context,
            }
        options = options.options()
    conversion_options, conversion_context = _conversion_compile_inputs(
        cast(CompileOptions, options), compilation_context
    )
    return {
        "_compile_options": conversion_options,
        "_compilation_context": conversion_context,
    }


def _with_compiler_mode_order(
    tensor: Union[STensor, TensorSpec],
    mode_order: Sequence[int],
    compile_options: Union[CompileOptions, _DeferredOptions],
    compilation_context: Optional[CompilationContext] = None,
    *,
    format_letters: Optional[str] = None,
    keep_every_stored_entry: bool = False,
) -> Union[STensor, TensorSpec]:
    """Relayout a runtime tensor or functionally update a compile-only spec.

    ``format_letters`` and ``keep_every_stored_entry`` come from
    :func:`_einsum_operand_relayout`; the defaults are the relayout as it always
    was (letters kept by storage position, nonzero values copied).
    ``compile_options`` is the call's snapshot, or its holder on einsum's cached path
    (see :func:`_conversion_kwargs`).
    """
    target_format = None if format_letters is None else parse_format(format_letters)
    if isinstance(tensor, TensorSpec):
        return tensor.with_mode_order(mode_order, tensor_format=target_format)
    result = tensor.copy()
    conversion_kwargs = _conversion_kwargs(compile_options, compilation_context)
    if target_format is None and not keep_every_stored_entry:
        result.change_mode_order(list(mode_order), **conversion_kwargs)
    else:
        result.change_mode_order(
            list(mode_order),
            _target_format=target_format,
            _keep_every_stored_entry=keep_every_stored_entry,
            **conversion_kwargs,
        )
    return result


def _first_einsum_operand_moves(
    tensors: Sequence[Union[STensor, TensorSpec]],
    operand_letters: Sequence[str],
    compiler_mode_orders: Sequence[Sequence[int]],
    compile_options: CompileOptions,
    compilation_context: Optional[CompilationContext],
) -> Tuple[Union[STensor, TensorSpec], ...]:
    """The first of einsum's two alignment moves, for the operands that make it now.

    An operand with no compressed level moves to the topological order before the
    loop order is selected, as it always has; that is what
    :func:`_einsum_operand_relayout` answers for it whatever the orders.  An operand
    with a compressed level stays as it is: how it moves depends on the selected
    order, and selection reads only metadata (:func:`_finish_einsum_operand_moves`).
    """

    return tuple(
        (
            tensor
            if "s" in operand_letters[tensor_index]
            else _with_compiler_mode_order(
                tensor,
                compiler_mode_orders[tensor_index],
                compile_options,
                compilation_context,
            )
        )
        for tensor_index, tensor in enumerate(tensors)
    )


def _finish_einsum_operand_moves(
    tensors: Sequence[Union[STensor, TensorSpec]],
    tensor_vars: Sequence[TensorVar],
    input_index_strs: Sequence[Sequence[str]],
    operand_letters: Sequence[str],
    input_orders: Sequence[Sequence[int]],
    compiler_mode_orders: Sequence[Sequence[int]],
    desired_mode_orders: Sequence[Sequence[int]],
    result_index_strs: Sequence[str],
    result_letters: str,
    result_mode_order: Sequence[int],
    extents: Mapping[str, int],
    compile_options: CompileOptions,
    compilation_context: Optional[CompilationContext],
) -> Tuple[Tuple[Union[STensor, TensorSpec], ...], List[_OperandRelayout]]:
    """Decide every operand's relayout and make the moves that remain.

    An operand with a compressed level makes the moves
    :func:`_einsum_operand_relayout` decides; when it is converted, its TensorVar
    takes the converted format, so the kernel is compiled for the letters the operand
    is stored with.  An operand with no compressed level makes the second legacy move,
    to the selected order.  Returns the operands and the decisions, which the
    dispatch cache keeps for the calls it serves.
    """

    operand_relayouts = [
        _einsum_operand_relayout(
            tensor_index,
            input_index_strs,
            operand_letters,
            input_orders,
            (compiler_mode_orders[tensor_index], desired_mode_order),
            result_index_strs,
            result_letters,
            result_mode_order,
            extents,
        )
        for tensor_index, desired_mode_order in enumerate(desired_mode_orders)
    ]
    moved = list(tensors)
    for tensor_index, desired_mode_order in enumerate(desired_mode_orders):
        relayout = operand_relayouts[tensor_index]
        if "s" in operand_letters[tensor_index]:
            for mode_order in relayout.mode_orders:
                moved[tensor_index] = _with_compiler_mode_order(
                    moved[tensor_index],
                    mode_order,
                    compile_options,
                    compilation_context,
                    **relayout.move_kwargs(),
                )
            if relayout.format_letters is not None:
                tensor_vars[tensor_index].format = moved[tensor_index].format
        elif list(moved[tensor_index].mode_order) != list(desired_mode_order):
            moved[tensor_index] = _with_compiler_mode_order(
                moved[tensor_index],
                desired_mode_order,
                compile_options,
                compilation_context,
            )
    return tuple(moved), operand_relayouts


# The drop-in CSR@dense SpMM symbols: one kernel (spmm_csr_v2_core) instantiated at
# float32 and float64. Both take the composition hints below.
_DROP_IN_SPMM_SYMBOLS = frozenset({"spmm_csr_float_v2", "spmm_csr_double_v2"})

# Composition thread-count matching for the drop-in CSR-SpMM.
# When a scorch SpMM is used as a torch op inside a host pipeline (e.g. a GCN
# layer: F.linear -> scorch.matmul -> ...), the SpMM's throttled policy thread
# count differs from the host's (torch) thread count, forcing a libgomp team
# reshape at every op boundary (~15% of a sub-ms GCN forward). Passing
# torch.get_num_threads() lets the kernel keep one warm team spanning the
# pipeline; the kernel only adopts it above the fork/join floor and caps it by
# row-parallelism, so tiny/standalone products (the SuiteSparse panel) are
# unaffected. On by default (non-fused GCN forward: pubmed 0.78x -> 1.15x vs
# PyTorch, cora/citeseer +5-6%, big graphs neutral, panel safe). Set
# SCORCH_MATCH_HOST_THREADS=0 to fall back to the pure standalone policy.
_MATCH_HOST_THREADS = os.environ.get("SCORCH_MATCH_HOST_THREADS", "1") == "1"

# Composition, part two: for the same drop-in pipeline SpMM, launch the kernel's
# workers on torch's own intra-op pool (at::parallel_for) instead of a private
# libgomp team, so the SpMM shares one warm pool with the surrounding torch
# epilogue (bias/activation) — a private omp team's threads sleep between ops
# under the default passive OMP policy and re-wake per call, a pool-transition tax
# that dominated the sparse-AE @0.99 mid-size gap (cur/mkl 1.2-1.3 -> ~0.9, 6/7
# apples-to-apples wins vs MKL). Fires together with _MATCH_HOST_THREADS; the
# kernel gates it to products that can feed the full host pool, so row-starved
# tiny products (the SuiteSparse panel's 130-row cell) keep omp. On by default;
# set SCORCH_ATPARALLEL_PIPELINE=0 to force the private-team launch.
_ATPARALLEL_PIPELINE = os.environ.get("SCORCH_ATPARALLEL_PIPELINE", "1") == "1"

# The same two composition hints, resolved once, for the planned dispatch: a plan
# call passes them positionally, so binding them here keeps that path branchless.
# `-1` is the kernel ABI's "no override", which is what the ordinary path passes
# when host-thread matching is off.
_PLAN_NTHREADS = torch.get_num_threads if _MATCH_HOST_THREADS else (lambda: -1)
_PLAN_ATPARALLEL = _ATPARALLEL_PIPELINE and _MATCH_HOST_THREADS

# `torch.Tensor`, bound once. The plan probe tests it on every matmul, and reading
# it through the module is a global lookup plus an attribute lookup each time.
_TENSOR = torch.Tensor
_SPARSE_CSR = torch.sparse_csr
_SPARSE_CSC = torch.sparse_csc

# start_time = time.time()
# # Register custom classes
# load(
#     name="pybind",
#     sources=[...],
# )
# end_time = time.time()
# compile_time = end_time - start_time
# print(f"Pybind load time: {compile_time:.5f} seconds")
#
# load_to_kernel_cache("spmm_csr", _kernel_cache, "spmm-csr.cpp")
# load_to_kernel_cache("spmm_csr_ones", _kernel_cache, "spmm-csr-ones.cpp")


def spmv(
    a: STensor,
    b: STensor,
    output_format: Optional[Union[TensorFormat, str, List[str]]] = None,
    **kwargs,
) -> STensor:
    """Sparse matrix-vector product (internal; reached through ``matmul``).

    Computes ``y[i] = sum_j a[i, j] * b[j]`` by lowering a per-row ``Where`` with a
    scalar accumulating ``Workspace`` through the CIN compiler (CIN -> LLIR -> C++),
    JIT-compiling, and executing. This is the generic-path SpMV used when the
    prebuilt ``prebuilt_spmv_csr_*`` kernel does not match.

    This function is **not exported**: ``scorch.spmv`` is not defined (it would fall
    through ``__getattr__`` to ``torch``, which has no ``spmv``). The user-facing
    way to reach it is ``matmul(a, x)`` with a 2-D sparse ``a`` and a 1-D ``x`` —
    ``matmul`` tries the prebuilt CSR-SpMV kernel first and falls back to this
    function on a miss. Reachable directly as ``scorch.ops.spmv`` for advanced use.

    Parameters
    ----------
    a : STensor
        2-D sparse matrix operand of shape ``(m, n)``.
    b : STensor
        1-D vector operand of shape ``(n,)``.
    output_format : TensorFormat or str or list of str, optional
        Requested output format. Defaults to ``"d"`` (a dense 1-D vector). Accepts
        anything :func:`scorch.utils.parse_format` understands.
    **kwargs
        ``format``: ``output_format`` by another name; the two given together must
        name one format (``format_keywords_conflict``).  ``time_dict`` : dict,
        optional -- if given, ``time_dict["eval_time"]`` is set to the kernel
        wall-clock time in seconds.  Also ``use_cache``, ``schedule`` (refused: the
        SpMV path takes none yet), ``_compile_options`` and ``_compilation_context``.
        Any other name raises ``KeywordArgumentError`` (``unknown_keyword_argument``).

    Returns
    -------
    STensor
        Result of shape ``(a.shape[0],)`` in ``output_format``.

    Notes
    -----
    The kernel is kept in the generated-kernel cache that the conversions use
    (``scorch.stensor._GENERATED_KERNELS``), keyed by both operands' layouts and
    value dtypes, the result format and the options key, so a warm call lowers
    nothing and makes no options snapshot.

    Examples
    --------
    >>> import torch
    >>> import scorch
    >>> A = scorch.from_torch((torch.rand(128, 128) < 0.1).float(), "A").to_sparse("ds")
    >>> x = torch.rand(128)
    >>> y = scorch.matmul(A, x)   # 2-D x 1-D -> dense vector via prebuilt/spmv
    >>> torch.allclose(y, A.to_torch() @ x, atol=1e-3, rtol=1e-3)
    True
    """
    # The product entries' keyword rule (patch 0072); format= is output_format= by
    # another name here.
    if kwargs:
        _entry_keywords(kwargs, _SPMV_KEYWORDS, "spmv")
        if "format" in kwargs:
            output_format = _format_alias("spmv", kwargs.pop("format"), output_format)
    return _spmv_owned(a, b, output_format, kwargs, _DeferredOptions(kwargs))


def _compile_spmv_cin(
    cin_stmt: IndexStmt,
    compile_options: CompileOptions,
    compilation_context: CompilationContext,
) -> Any:
    """Lower, emit, name and load an SpMV kernel, as ``spmv`` always has."""

    cin_stmt = normalize_cin(
        cin_stmt,
        compile_options=compile_options,
        compilation_context=compilation_context,
    )
    lowerer = CINLowerer(
        compile_options=compile_options,
        compilation_context=compilation_context,
    )
    lowered_llir = lowerer._lower_owned_IndexStmt(cin_stmt)
    cpp_code = _lower_generated_llir(
        lowered_llir,
        compile_options,
        compilation_context,
    )

    header_cpp_code = compile_options.build.preamble_source

    prepared_build = _prepare_generated_kernel_build(
        header_cpp_code,
        cpp_code,
        compile_options,
        compilation_context,
    )
    return _load_validated_prepared_kernel(prepared_build)


def _spmv_owned(
    a: STensor,
    b: STensor,
    output_format: Optional[Union[TensorFormat, str, List[str]]],
    kwargs: dict,
    options: _DeferredOptions,
) -> STensor:
    """``spmv`` under the options holder of the call it belongs to.

    ``matmul`` calls this with its own holder, so a matmul and its SpMV make at most
    one snapshot, and a warm one makes none.
    """

    # Every spelling of a schedule lands in requested_schedule, a schedule_force context
    # included.  The SpMV path takes none yet: the refusal matmul's SpMV branch makes.
    # It is a refusal of the call's options, which no operand changes, so it comes
    # first and a scheduled call gets this code whatever its operands are; the operand
    # checks follow, still before anything reads an operand.
    if options.requested_schedule is not None:
        raise ScheduleNotThreadedThroughSpMVError()
    _require_matrix_vector_operands(a, b)
    compilation = _DeferredCompilation(kwargs, options)
    # The kernel below reads A by rows, in its storage order, so an A stored in another
    # mode order is moved first, as matmul's matrix-vector branch moves it.  The move
    # takes the call's holder, so a warm call hits its kernel with no snapshot, and the
    # key below holds the moved operand's layout.
    if a.storage.index.mode_order not in (None, [0, 1]):
        a = a.copy()
        a.change_mode_order(
            [0, 1], **_conversion_kwargs(options, compilation.nested_context())
        )

    def requested_format() -> TensorFormat:
        if output_format is None:
            return parse_format("d")
        if isinstance(output_format, TensorFormat):
            return output_format
        return parse_format(output_format)

    parsed_format: Optional[TensorFormat]
    try:
        parsed_format = requested_format()
    except Exception:
        # Parsed again in the frontend stage, which raises it as it always did.
        parsed_format = None
    if parsed_format is not None and parsed_format.get_order() != 1:
        # The result is a vector; refused before any CIN is built (see
        # _require_output_format_rank), for a caller of scorch.ops.spmv itself.
        _require_output_format_rank(parsed_format, 1, "spmv")
    result_shape = (a.shape[0],)
    # What the lowering reads: both layouts and value dtypes and the result format.
    key = (
        None
        if parsed_format is None
        else (
            "spmv",
            a.layout,
            b.layout,
            a.dtype,
            b.dtype,
            a.device,
            b.device,
            parsed_format,
            options.reuse_key,
        )
    )

    def build_cin() -> IndexStmt:
        nonlocal parsed_format
        if parsed_format is None:
            parsed_format = requested_format()

        # One value dtype, as torch requires of a product (it raises RuntimeError
        # and does not promote); a mixed pair raises before anything is built.
        y = TensorVar(
            "y",
            shape=result_shape,
            fmt=parsed_format,
            dtype=require_same_value_dtype(
                a.dtype, b.dtype, operation="matrix-vector product"
            ),
        )
        A = TensorVar("A", shape=a.shape, fmt=a.format, dtype=a.dtype)
        x = TensorVar("x", shape=b.shape, fmt=b.format, dtype=b.dtype)

        i = IndexVar("i")
        j = IndexVar("j")

        workspace = Workspace(
            name="wksp",
            dim=0,
            dtype=y.dtype,
        )

        cin_stmt: IndexStmt = ForAll(
            i,
            Where(
                producer=ForAll(
                    j,
                    TensorAssign(
                        workspace.get_default_access(),
                        A[i, j] * x[j],
                        op=Operation.ADD,
                    ),
                ),
                consumer=TensorAssign(
                    y[i],
                    workspace.get_default_access(),
                ),
            ),
        )
        return cin_stmt

    module = _generated_kernel(key, options, compilation, build_cin, _compile_spmv_cin)
    # set above or by build_cin
    if parsed_format is None:
        raise ScorchAssertionError(
            "parsed_format is not None", code="generated_kernel_format_missing"
        )

    args = [result_shape]

    for tensor in [a, b]:
        args.append(tensor.shape)  # type: ignore
        args.append(tensor._native_mode_indices())  # type: ignore
        # The detached values: native code never reads autograd's tensor (stensor_grad).
        args.append(tensor._raw_values)  # type: ignore

    start_time = time.time()
    try:
        result_cpp = module.evaluate(*args)
    except RuntimeError as error:
        raise_if_native_coded(error)
        raise
    end_time = time.time()
    eval_time = end_time - start_time
    if "time_dict" in kwargs:
        time_dict = kwargs["time_dict"]
        time_dict["eval_time"] = eval_time
    # m

    result = _wrap_generated_result(
        shape=result_shape,
        tensor_format=parsed_format,
        result_cpp=result_cpp,
    )

    return result


def _composition_hints(resolved):
    """Thread-count and pipelining hints for the drop-in CSR@dense SpMM.

    Returns ``(nthreads, atparallel)``, both inert unless the resolved kernel is one
    of the two drop-in SpMMs, ``spmm_csr_float_v2`` or ``spmm_csr_double_v2``. Those
    are one kernel instantiated at two value types, so they take the same hints and
    the thread-count match is worth the same thing to both.

    Derived here and nowhere else because two callers dispatch this kernel family --
    ``matmul`` and ``scorch.compile``'s fused SpMM+bias+act path -- and the tiling
    selector memoizes a verdict measured against the baseline these hints configure,
    then applies the same ``nthreads`` to the tiled kernels it picks. A second caller
    deriving them differently would write verdicts the first cannot reproduce, and
    would forfeit the host-thread match (pubmed 0.78 -> 1.15x, commit e795127).
    """
    if _MATCH_HOST_THREADS and resolved.symbol_name in _DROP_IN_SPMM_SYMBOLS:
        return torch.get_num_threads(), _ATPARALLEL_PIPELINE
    return None, False


def tiling_gate(a, b, resolved):
    """Whether the tiling selector may serve this SpMM at all, and at which level.

    Returns ``(level, may_serve)``, with ``level`` ``None`` when the resolved kernel is
    not the drop-in SpMM.

    Split out of ``_dispatch_tiled`` so a caller that has to build closures in order
    to consult the selector can ask first and skip building them on the shapes the
    gate declines — which is every GCN-small layer, every autoencoder layer, and
    anything whose dense operand fits in cache. ``scorch.compile``'s fused path needs
    that; ``matmul`` does not, and lets ``_dispatch_tiled`` call this itself.

    The level is resolved exactly once, here, and handed back, so the gate and the
    dispatch that follows it cannot be asked at two different levels — a
    ``set_autotune`` context manager could otherwise exit between the two reads.
    """
    # float32 only, deliberately: the tiled kernels the selector can pick
    # (spmm_csr_float_tilej / _tileijk) have no float64 instantiation, so a float64
    # product must not enter the selector at all. Widening this without widening
    # those would hand the probe a candidate list with nothing in it.
    if resolved.symbol_name != "spmm_csr_float_v2":
        return None, False
    level = _tiling_current_level()
    return level, _tiling_is_candidate(a, b, level=level)


def _dispatch_tiled(
    a,
    b,
    resolved,
    nthreads,
    atparallel,
    time_dict,
    baseline_fn=None,
    baseline_tag="v2",
    epilogue=None,
    gate=None,
):
    """Run the adaptive tiling selector's choice, or report that v2 should run.

    Returns ``(result_cpp, result_shape, plan_kind, plan_param)``, with the first two
    ``None`` when the caller must run its ordinary v2 dispatch, and the last two
    naming the kernel that *did* serve the call so a plan can reproduce it.

    On the drop-in CSR@dense SpMM this routes the high-degree operand-over-LLC thrash
    regime (reddit/products-class) to the column-panel kernel
    ``spmm_csr_float_tilej``; v2 serves everything else byte-unchanged. Provably
    no-regression: the baseline is always a probe candidate, so the memoized choice
    is never slower than it (see ``tiling.maybe_dispatch``). It only engages when the
    cheap O(1) pre-filter says a shape can even benefit — no overhead on
    GCN-small/AE/panel.

    ``baseline_fn`` / ``baseline_tag`` / ``epilogue`` default to the drop-in SpMM
    comparison this function was written for. A caller with a different alternative
    passes its own — see ``dispatch_tiled_fused``, which is how
    ``scorch.compile``'s fused path enters. ``resolved`` always names the *untiled
    SpMM* kernel either way, so the symbol gate and the composition hints are the
    same question for both callers.
    """
    # `gate` carries the symbol check, the autotune level resolved ONCE, and the O(1)
    # pre-filter. A caller that already asked passes the answer in rather than paying
    # for it twice; everyone else lets it be asked here. Nothing outside the v2 symbol
    # is touched either way, so other prebuilt kernels (bias_act/fused) stay
    # byte-identical.
    if gate is None:
        gate = tiling_gate(a, b, resolved)
    level, may_serve = gate
    if not may_serve:
        return None, None, "v2", None
    result_shape = [a.shape[0], b.shape[1]]

    if baseline_fn is None:

        def baseline_fn(nt):
            rc, _ = execute_prebuilt_binary_kernel(
                resolved.fn, a, b, nthreads=nt, atparallel=atparallel
            )
            return rc

    disp = _tiling_maybe_dispatch(
        a,
        b,
        result_shape,
        baseline_fn,
        nthreads,
        time_dict=time_dict,
        level=level,
        epilogue=epilogue,
        baseline_tag=baseline_tag,
    )
    if disp is None:
        return None, None, "v2", None
    result_cpp, _ = disp
    # A tiled kernel served this call. Read which one HERE and nowhere else: the
    # memo is only authoritative where the selector actually ran. Reading it
    # wherever an entry merely exists let a plan carry tile-j for a shape the gate
    # had rejected, so the plan ran a different kernel than the ordinary path --
    # caught by a test comparing the two bitwise.
    verdict = _tiling_decided(a, b.shape[1], level=level, baseline_tag=baseline_tag)
    if verdict is None:
        return result_cpp, tuple(result_shape), "v2", None
    return result_cpp, tuple(result_shape), verdict[0], verdict[1]


def dispatch_tiled_fused(
    a, b, resolved_spmm, baseline_fn, baseline_tag, epilogue, gate=None
):
    """Consult the adaptive tiling selector on behalf of a fused SpMM + tail.

    ``scorch.compile``'s only entry into the selector, and the reason
    ``_dispatch_tiled`` takes a baseline at all. Sharing that function rather than
    mirroring it is deliberate: the symbol gate, the autotune level resolved once,
    the O(1) candidate pre-filter and — critically — the composition hints that
    configure the tiled kernels all come from the same code, so the two callers
    cannot drift into running the same kernel two different ways.

    What legitimately differs is the alternative being measured against.
    ``matmul``'s is the drop-in SpMM; a fused caller's is a kernel that folds
    ``bias``/``relu`` into the SpMM's row epilogue and is therefore faster than the
    drop-in SpMM plus a separate pass over the output. So the fused caller supplies
    both its own ``baseline_fn`` and the ``epilogue`` the tiled kernels must run
    out-of-line, and the verdict is memoized under ``baseline_tag``.

    Returns ``_dispatch_tiled``'s 4-tuple; the first element is ``None`` when the
    caller should run its own fused kernel.
    """
    nthreads, atparallel = _composition_hints(resolved_spmm)
    return _dispatch_tiled(
        a,
        b,
        resolved_spmm,
        nthreads,
        atparallel,
        None,
        baseline_fn=baseline_fn,
        baseline_tag=baseline_tag,
        epilogue=epilogue,
        gate=gate,
    )


# The keyword the C entry point passes the plan key under when it hands a torch operand's
# call to `matmul` (see there). Private: no caller is meant to pass it.
_C_PROBED_KW = "_scorch_plan_probed"


# What _c_probed returns: no probe made anywhere yet, or the C entry point's probe found
# nothing for the installer to do. Equal values, distinct objects: `matmul` tells them
# apart by identity.
_NOT_PROBED: Tuple[None, None, None] = (None, None, None)
_PROBED_NOTHING = cast(Tuple[None, None, None], tuple([None, None, None]))


def _c_probed(a: Any, b: Any, kwargs: Dict[str, Any]) -> Tuple[Any, Any, Any]:
    """(plan_a, plan_b, key) for a call the C entry point (planned_matmul.h) handed over.

    That is a torch sparse CSR operand its plan did not serve: it has done the probe, and
    it records a first sighting itself once this call returns, so the key comes only when
    the installer has something to do (install on a repeat, account for a decline). The
    keyword is popped so nothing below sees it. A call with only ordinary keywords gets
    `_NOT_PROBED`.
    """
    if _C_PROBED_KW not in kwargs:
        return _NOT_PROBED
    key = kwargs.pop(_C_PROBED_KW)
    return _PROBED_NOTHING if key is None else (a, b, key)


def _run_torch_operand_plan(a: torch.Tensor, b: torch.Tensor, key: Any) -> Any:
    """The planned call for a torch sparse CSR operand, or None to take the ordinary path.

    Its plans live in its index arrays' entry (plan.torch_operand_state; see
    plan.TORCH_OPERANDS). Without them every call converted the operand afresh -- two
    copies of its index arrays and two O(nnz) walks -- which made the whole call 2-4x slower than
    torch.sparse.mm's on mkt1 although the kernel was faster. `run_csr` is given the
    operand's arrays as they are now and declines unless they are the plan's own.

    Not inline in `matmul` like the STensor probe: the C entry point (planned_matmul.h)
    serves these calls, and passes the ones it does not serve back with `_C_PROBED_KW`,
    which skips this. So it runs only in a build without that entry point, or when
    `_matmul_python` is called directly.
    """
    state = _plan_torch_operand_state(a)
    plans = state.get(_PLANS_ATTR) if state is not None else None
    plan = plans.get(key) if plans is not None else None
    if plan is None:
        return None
    positions, coordinates = _scorch_ops.torch_sparse_index_members(a)
    return plan.run_csr(
        positions,
        coordinates,
        a.values(),
        b,
        _PLAN_NTHREADS(),
        _PLAN_ATPARALLEL,
    )


# The value dtypes at which two torch CSR operands' product is the prebuilt CSR x CSR kernel's
# in this build (prebuilt_kernels.csr_csr_symbol), and so is made by _torch_csr_spgemm; empty in
# a build without spgemm_csr_torch_result.
_SPGEMM_SERVES: Dict[torch.dtype, bool] = (
    {
        dtype: csr_csr_symbol(dtype) == f"prebuilt_spmspm_csr_{suffix}"
        for dtype, suffix in (
            (torch.float32, "f32"),
            (torch.float64, "f64"),
            (torch.int32, "i32"),
            (torch.int64, "i64"),
        )
    }
    if hasattr(_scorch_ops, "spgemm_csr_torch_result")
    else {}
)
# The index dtype of that product's result: int64, torch's own for a CSR result
# (to_sparse_csr, torch.sparse.mm), so a caller mixing scorch's results with torch's sees one.
# PROVISIONAL: patch A's grid measures an int32 result (half the index bytes, written by the
# same kernel) against it, and this changes only on that measurement.
_SPGEMM_INT64_INDICES = True


def _torch_compressed_product(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Whether ``matmul`` returns ``a @ b`` in torch's type: two 2-D torch compressed matrices
    on the CPU, CSR or CSC, at a value dtype the prebuilt CSR x CSR kernel takes in this build,
    whose inner extents match (else the ordinary path raises the shape error). The caller has
    checked the dtypes match. Without keywords such a product is the torch sparse CSR tensor
    torch returns for it; ``matmul`` reads its keywords against ``_TORCH_RESULT_KEYWORDS`` for
    the type with them. Torch's product of CSC with CSR, either way round, is that same CSR
    tensor (torch 2.14.1 with MKL), so a CSC operand, which from_torch adopts, gives the type
    CSR in its place gives. Two CSC operands never come here: they are torch's own product
    (``matmul``'s tier 1).
    """
    return (
        (a.layout is _SPARSE_CSR or a.layout is _SPARSE_CSC)
        and (b.layout is _SPARSE_CSR or b.layout is _SPARSE_CSC)
        and a.dim() == 2
        and b.dim() == 2
        and a.device.type == "cpu"
        and b.device.type == "cpu"
        and _SPGEMM_SERVES.get(a.dtype, False)
        and a.shape[1] == b.shape[0]
    )


def _torch_csr_spgemm_serves(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Whether ``_torch_csr_spgemm`` makes ``a @ b``: a product ``_torch_compressed_product``
    accepts whose operands are both CSR, the pair the prebuilt CSR x CSR kernel reads as they
    are. A CSC operand's product is computed on the ordinary path, which reads the CSC as the
    CSR of its transpose (stensor._csc_from_torch), and returned in torch's type all the same
    (``_as_torch_compressed``).
    """
    return (
        a.layout is _SPARSE_CSR
        and b.layout is _SPARSE_CSR
        and _torch_compressed_product(a, b)
    )


def _torch_csr_spgemm(
    a: torch.Tensor, b: torch.Tensor, time_dict: Optional[dict] = None
) -> Optional[torch.Tensor]:
    """``a @ b`` for two operands ``_torch_csr_spgemm_serves`` accepts, as the torch sparse CSR
    tensor torch returns for the same product; ``time_dict["eval_time"]``, when given, the
    native call's wall-clock time, as the prebuilt kernels report theirs.

    The native function is the one the C entry point calls for the same pair (planned_matmul.h,
    planned_matmul_torch_spgemm), so both routes return the same tensor. Before it, the pair was
    wrapped into two STensors and the product came back as an STensor, a type no caller of
    ``import scorch as torch`` holding two torch tensors asked for. A row's coordinates need not
    be in order: torch.sparse.mm's own CSR x CSR result, whose rows are not sorted, is read as
    it is (the kernel is order-free; see spgemm_csr_torch_result). Both operands are recorded as
    write witnesses of their index arrays, as every reading of a torch operand's arrays is
    (torch_sparse_index_members).

    The native function checks the operands' arrays (spans and bounds) with uncoded text, but
    for a coded refusal (an index past the 32-bit bound), which leaves as its structured error
    (``raise_if_native_coded``). For any other refusal it returns None, and ``matmul`` computes
    the product on the ordinary path, as the C entry point hands it over: that path refuses a
    malformed operand by code, as it did before this function, and a product it computes is
    returned in torch's type all the same (``_as_torch_compressed``).
    """
    a_positions, a_coordinates = _scorch_ops.torch_sparse_index_members(a)
    b_positions, b_coordinates = _scorch_ops.torch_sparse_index_members(b)
    start_time = time.time() if time_dict is not None else 0.0
    try:
        result = _scorch_ops.spgemm_csr_torch_result(
            a_positions,
            a_coordinates,
            a.values(),
            b_positions,
            b_coordinates,
            b.values(),
            a.shape[0],
            a.shape[1],
            b.shape[1],
            _SPGEMM_INT64_INDICES,
        )
    except RuntimeError as error:
        raise_if_native_coded(error)
        return None
    if time_dict is not None:
        time_dict["eval_time"] = time.time() - start_time
    return result


def _require_one_product_dtype(a: object, b: object) -> None:
    """Raise ``OperandDtypeMismatchError`` when ``matmul``'s operands differ in dtype.

    ``matmul`` calls it for every pair its inline test did not admit.  That test admits
    only exact types, so every other pair comes here: an ``nn.Parameter`` or another
    tensor subclass, a dense left operand of an STensor, a ``TensorSpec``, and every
    mistake.  An operand that is not an STensor, a ``torch.Tensor`` or a ``TensorSpec``
    is refused first, by code (``matmul_operand_type_invalid``, from
    ``_product_operand_extents``), as ``einsum`` refuses one.

    Each operand's type is tested once, by the test that reads its dtype: an STensor's
    from its metadata, as the inline test reads it, and a tensor's as its attribute.  So
    a correct call reads no extents and calls no property.  On the M5 that costs 20 ns
    more than reading both dtypes with ``getattr``, which refused nothing; refusing
    through ``_product_operand_extents`` on every call cost 140 ns, 1.8-2.3% of a 9 us
    call with an ``nn.Parameter`` operand (fd3).  The two operands' tests are written
    out because a helper's call would cost more than the test.

    A separate function so ``matmul`` stays within the project's complexity bound.
    """

    a_dtype: Optional[torch.dtype]
    b_dtype: Optional[torch.dtype]
    if isinstance(a, STensor):
        a_dtype = a._metadata.dtype
    elif isinstance(a, torch.Tensor):
        a_dtype = a.dtype
    else:
        _product_operand_extents(a, "first")
        a_dtype = getattr(a, "dtype", None)
    if isinstance(b, STensor):
        b_dtype = b._metadata.dtype
    elif isinstance(b, torch.Tensor):
        b_dtype = b.dtype
    else:
        _product_operand_extents(b, "second")
        b_dtype = getattr(b, "dtype", None)
    if a_dtype != b_dtype and a_dtype is not None and b_dtype is not None:
        require_same_value_dtype(a_dtype, b_dtype, operation="matmul")


def _product_operand_extents(
    operand: object, position: str
) -> Optional[Tuple[int, ...]]:
    """``operand``'s logical extents, or ``None`` for a ``TensorSpec`` (``einsum``
    checks a spec's extents itself).  Anything else is refused with a code, as
    ``einsum`` refuses an operand of a type it does not take.
    """

    if isinstance(operand, STensor):
        return operand._metadata.layout.logical_shape
    if isinstance(operand, torch.Tensor):
        return operand.shape
    if isinstance(operand, TensorSpec):
        return None
    raise TensorTypeError(
        f"matmul's {position} operand must be a torch.Tensor or an STensor, "
        f"got {type(operand).__name__}",
        code="matmul_operand_type_invalid",
    )


def _require_product_operands(a: object, b: object) -> None:
    """Refuse a ``matmul`` whose operands cannot be multiplied, by ``torch.matmul``'s
    rule: the last extent of ``a`` meets the first extent of a vector ``b`` and the
    second-to-last extent of any other ``b``.

    Batched ranks and rank 0 are left to the rank rules further on, which refuse them
    by code.  A separate function so ``matmul`` stays within the project's complexity
    bound.
    """

    a_shape = _product_operand_extents(a, "first")
    b_shape = _product_operand_extents(b, "second")
    if not a_shape or not b_shape:
        return
    contracted = b_shape[0] if len(b_shape) == 1 else b_shape[-2]
    if a_shape[-1] != contracted:
        raise IndexExtentMismatchError(
            f"matmul operands of shapes {tuple(a_shape)} and {tuple(b_shape)} cannot "
            f"be multiplied: contracted extents {a_shape[-1]} and {contracted}"
        )


def _dense_layouts_product(a: STensor, b: STensor) -> torch.Tensor:
    """``torch.matmul`` of two dense STensors, its refusal of operands that cannot be
    multiplied raised under the class every other route raises.  A separate function so
    ``matmul`` stays within the project's complexity bound.
    """

    try:
        return torch.matmul(a.to_torch(in_place=False), b.to_torch(in_place=False))
    except RuntimeError:
        _require_product_operands(a, b)
        raise


# The keyword arguments each product entry takes (patch 0072).  Any other name is refused
# with KeywordArgumentError (unknown_keyword_argument): every entry used to ignore it, so a
# misspelt format=, an out= or a torch keyword returned a result computed without it.
_PRODUCT_KEYWORDS = frozenset(
    {
        "format",
        "output_format",
        "use_cache",
        "schedule",
        "_schedule",
        "time_dict",
        "output_mode_order",
        "_compile_options",
        "_compilation_context",
    }
)
_MATMUL_KEYWORDS = _PRODUCT_KEYWORDS | {"out"}
# einsum takes no out=: torch.einsum has none, and nothing here implements one.
_EINSUM_KEYWORDS = _PRODUCT_KEYWORDS | {
    "_post_ops",
    "_post_ops_tensors",
    "_deferred_options",
}
# spmv names its result format output_format, a parameter; format= is its alias.  A vector
# result has no other mode order, so output_mode_order is not one of them.
_SPMV_KEYWORDS = (_PRODUCT_KEYWORDS - {"output_format", "output_mode_order"}) | {
    "format"
}
# matmul's keywords by what each does to the type of two torch compressed operands' product,
# which without keywords is torch's (_torch_compressed_product, _torch_csr_spgemm). A result format (format=, and
# output_format=, read as it) or a schedule (schedule=, _schedule=) asks for the product in
# scorch's form, an STensor, as it does on every other route; so does a forced schedule given
# by _compile_options= or by the context. Every other keyword leaves torch's type: time_dict=
# is served by the same function, timed; use_cache=, output_mode_order=, _compile_options= and
# _compilation_context= by the route they choose, whose product is returned as the torch sparse
# tensor holding its arrays (_as_torch_compressed): CSR, or CSC for output_mode_order=[1, 0].
# out= is neither: out= is the result (_matmul_into), which must be dense. The two sets and out=
# are every keyword matmul takes (a test checks it), so a keyword added to matmul is placed in
# one of them before it can change a torch operand's result type.
_STENSOR_RESULT_KEYWORDS = frozenset(
    {"format", "output_format", "schedule", "_schedule"}
)
_TORCH_RESULT_KEYWORDS = frozenset(
    {
        "time_dict",
        "use_cache",
        "output_mode_order",
        "_compile_options",
        "_compilation_context",
    }
)
_SPGEMM_SERVED_KEYWORDS = frozenset({"time_dict"})

# A keyword given this value is dropped at the entry, compared by identity: every reader
# takes the same value when the keyword is absent.  So a call whose keywords all hold
# their defaults is the plain call (autograd.py, plans, a dense einsum's route).
_KEYWORD_DEFAULTS: Dict[str, Any] = {
    "format": None,
    "output_format": None,
    "use_cache": True,
    "schedule": None,
    "_schedule": None,
    "time_dict": None,
    "output_mode_order": None,
    "_compile_options": None,
    "_compilation_context": None,
    "out": None,
    "_post_ops": None,
    "_post_ops_tensors": None,
    "_deferred_options": None,
}
_NO_DEFAULT = object()
# What _matmul_keyword_call returns when matmul goes on with the call itself.
_PLAIN_CALL = object()


def _same_format(first: FormatInput, second: FormatInput) -> bool:
    """Whether two format spellings name one format; a spelling that is no format raises
    the error it raises wherever a format is read."""

    return first is second or parse_format(first) == parse_format(second)


def _format_alias(entry: str, value: Any, alias: Any) -> Any:
    """The result format of a call given it both as ``format=`` (``value``) and by its
    alias (``alias``): either one when the other is absent (``None``), refused with
    ``format_keywords_conflict`` when they name different formats."""

    if value is None:
        return alias
    if alias is not None and not _same_format(value, alias):
        raise KeywordArgumentError(
            f"{entry}() got format={value!r} and output_format={alias!r}, which name "
            "different formats",
            code="format_keywords_conflict",
        )
    return value


def _entry_keywords(
    kwargs: Dict[str, Any], accepted: FrozenSet[str], entry: str
) -> None:
    """Apply the product entries' keyword rule to ``kwargs``, in place (patch 0072).

    A name ``entry`` does not take is refused, with Python's own message for it.  A
    keyword holding its default is dropped (``_KEYWORD_DEFAULTS``).  ``output_format=``
    is ``format=`` by another name: it is read as ``format``, and the two given together
    must name the same format.  Called only with keywords: a call with none pays nothing.
    One pass reads every name, and the first one refused raises before anything is
    dropped.
    """

    defaults = None
    for name, value in kwargs.items():
        if name not in accepted:
            raise KeywordArgumentError(
                f"{entry}() got an unexpected keyword argument '{name}'",
                code="unknown_keyword_argument",
            )
        if value is _KEYWORD_DEFAULTS.get(name, _NO_DEFAULT):
            if defaults is None:
                defaults = [name]
            else:
                defaults.append(name)
    if defaults is not None:
        for name in defaults:
            del kwargs[name]
    if "output_format" in kwargs:
        alias = kwargs.pop("output_format")
        kwargs["format"] = _format_alias(entry, kwargs.get("format"), alias)


def _checked_output_mode_order(order: Any, rank: int) -> List[int]:
    """``order`` as a list, refused with einsum's codes unless it is a permutation of a
    result of ``rank`` modes."""

    if isinstance(order, (str, bytes)) or not isinstance(order, Sequence):
        raise CompileSpecError(
            "output_mode_order must be a sequence of integers",
            code="output_mode_order_type_invalid",
        )
    if any(isinstance(mode, bool) or not isinstance(mode, int) for mode in order):
        raise CompileSpecError(
            "output_mode_order entries must be integers",
            code="output_mode_order_type_invalid",
        )
    if len(order) != rank or sorted(order) != list(range(rank)):
        raise CompileSpecError(
            "output_mode_order must be a permutation of the result modes",
            code="output_mode_order_not_permutation",
        )
    return list(order)


def _matmul_result_rank(a: Any, b: Any) -> Optional[int]:
    """The rank of ``a @ b`` by torch.matmul's rules, or ``None`` for a pair of ranks
    matmul refuses further on, or an operand that is not a tensor (an fx Proxy among
    them: asking its rank would record a node)."""

    if not (
        isinstance(a, (torch.Tensor, STensor))
        and isinstance(b, (torch.Tensor, STensor))
    ):
        return None
    return matmul_result_rank(operand_shape(a), operand_shape(b))


def _is_sparse_matrix(x: Any) -> bool:
    if isinstance(x, STensor):
        return x.dim() == 2 and not x.format.is_dense()
    return (
        isinstance(x, torch.Tensor)
        and x.dim() == 2
        and x.layout in (torch.sparse_csr, torch.sparse_coo)
    )


def _is_dense_operand(x: Any) -> bool:
    if type(x) is STensor:
        # ``x.format``, read from the metadata where that property reads it: every
        # einsum asks this of its first operand.
        return x._metadata.layout.format.is_dense()
    if isinstance(x, STensor):
        return x.format.is_dense()
    return isinstance(x, torch.Tensor) and x.layout is torch.strided


def _plain_dense_result_rank(a: Any, b: Any) -> Optional[int]:
    """The rank of ``a @ b`` when the product's plain result is a dense tensor on every
    route, else ``None``.  That holds for a sparse matrix times a dense matrix or vector
    (einsum infers the dense last index from the dense operand, spmv's default is dense,
    and the prebuilt kernels and autograd.py return dense), and for a dense operand times
    a sparse matrix where autograd.py computes the product (``autograd.computes``).  Any
    other dense x sparse product can have a sparse plain result (an integer x @ A is an
    STensor: einsum infers "ds" from A's compressed level)."""

    if _is_sparse_matrix(a) and _is_dense_operand(b) and b.dim() in (1, 2):
        return int(b.dim())
    if (
        _is_dense_operand(a)
        and a.dim() in (1, 2)
        and _is_sparse_matrix(b)
        and _autograd.computes(a, b)
    ):
        return int(a.dim())
    return None


def _forced_result_format(a: Any, b: Any, kwargs: Dict[str, Any]) -> Optional[str]:
    """The result format a forced schedule's generated kernel is given for ``a @ b``: the
    one the product has without the schedule, so a schedule changes the kernel, never the
    result's type, layout or format.  A product whose plain result is dense on every route
    (``_plain_dense_result_rank``) gets the dense layout of its rank, which matmul returns
    as a torch.Tensor (``_dense_result_as_tensor``), as the unforced routes do; a schedule
    the dense result cannot take is refused by the scheduler with its code.  Otherwise
    None: einsum's default, which is the unforced result (the prebuilt kernels' formats are
    einsum's for every pair they serve), or the ``format=`` the call names.  Read on the
    caller's operands, before matmul converts them, by every forced call: the C entry
    point hands one to matmul, and autograd.py's forced forward calls matmul."""

    if "format" in kwargs:
        return None
    rank = _plain_dense_result_rank(a, b)
    return None if rank is None else "d" * rank


def _names_plain_dense_result(a: Any, b: Any, requested: FormatInput) -> bool:
    """Whether ``format=requested`` names what ``a @ b`` returns without it: a dense
    layout of the result's rank, on a product whose plain result is a dense tensor on
    every route (``_plain_dense_result_rank``)."""

    rank = _plain_dense_result_rank(a, b)
    if rank is None:
        return False
    tensor_format = parse_format(requested)
    return tensor_format.is_dense() and tensor_format.get_order() == rank


def _matmul_keyword_call(a: Any, b: Any, kwargs: Dict[str, Any]) -> Any:
    """Apply ``matmul``'s keyword rule to ``kwargs``, in place (patch 0072).

    The rule every product entry applies (``_entry_keywords``), and three more that make
    a keyword which changes nothing about the computation leave it the plain call's:
    an identity ``output_mode_order`` is taken out of the computation (any other order
    is checked, by einsum's codes, against the result's rank); so is a ``format=``
    naming the plain result's own dense layout (``_names_plain_dense_result``).  Any
    other order changes the result's storage order and nothing else, so with no
    ``format=`` a product whose plain result is dense (``_plain_dense_result_rank``)
    asks for a dense result: the generated kernel the order goes to infers "ds" for
    x @ A from A's compressed level, where the plain call returns a dense tensor.
    ``out=`` is filled here (``_matmul_into``), except for an fx Proxy operand, whose
    call is recorded as it was made.

    A keyword taken out of the computation is still one the caller passed, which is
    not the same as one left at its default (``_entry_keywords`` drops those, by
    identity with the default): the drop-in rule reads a passed result keyword as a
    request for the result in scorch's form, whatever its value.  So a call that passed
    one is computed here as the plain call -- the same bits, its derivative, its plan,
    the plain call's cache keys -- and its result goes to ``_result_as_requested`` with
    the keywords taken out, their values normalised (a ``TensorFormat``, an order as a
    list).  Returns the result of a call completed here, else ``_PLAIN_CALL``.  A
    separate function so ``matmul`` stays within the project's complexity bound.
    """

    _entry_keywords(kwargs, _MATMUL_KEYWORDS, "matmul")
    taken_out: Dict[str, Any] = {}
    if "output_mode_order" in kwargs:
        rank = _matmul_result_rank(a, b)
        if rank is not None:
            order = _checked_output_mode_order(kwargs["output_mode_order"], rank)
            if order == list(range(rank)):
                del kwargs["output_mode_order"]
                taken_out["output_mode_order"] = order
            else:
                kwargs["output_mode_order"] = order
    requested = kwargs.get("format")
    if "output_mode_order" in kwargs:
        if requested is None:
            dense_rank = _plain_dense_result_rank(a, b)
            if dense_rank is not None:
                kwargs["format"] = "d" * dense_rank
    elif requested is not None and _names_plain_dense_result(a, b, requested):
        del kwargs["format"]
        taken_out["format"] = parse_format(requested)
    if "out" in kwargs and not (isinstance(a, Proxy) or isinstance(b, Proxy)):
        # out= is the result (torch.matmul's contract), so nothing is left to present.
        return _matmul_into(a, b, kwargs.pop("out"), kwargs)
    if taken_out:
        return _result_as_requested(matmul(a, b, **kwargs), taken_out)
    return _PLAIN_CALL


def _result_as_requested(result: Any, taken_out: Dict[str, Any]) -> Any:
    """The result of a call whose caller passed result keywords that did not change the
    computation (``_matmul_keyword_call``): ``format=`` naming the plain result's dense
    layout, an identity ``output_mode_order``.  ``taken_out`` holds them, normalised.

    The drop-in rule reads such a keyword as a request for the result in scorch's form
    (rule (a), scorch-merge 2026-10-14 18:48); building that form is the result-type
    patch's (0073, one helper with the dense einsum's), not this one's.  Until it lands
    the result is the plain call's, as it was before the keyword was taken out; this is
    the one place that patch changes, and the keywords reach it.
    """

    return result


def _operand_requires_grad(x: Any) -> bool:
    return isinstance(x, (torch.Tensor, STensor)) and bool(x.requires_grad)


def _matmul_into(a: Any, b: Any, out: Any, kwargs: Dict[str, Any]) -> torch.Tensor:
    """``matmul(a, b, out=out, **kwargs)`` by ``torch.matmul``'s contract for ``out=``.

    Refused before anything is computed, each by code: an ``out`` that is not a tensor;
    a call in grad mode with an operand or ``out`` that requires grad (torch refuses
    automatic differentiation through ``out=``); a sparse result format, or a result
    mode order (``out``'s own strides are the result's layout); an ``out`` of another
    dtype than the operands' values, or on another device.  Two dense torch tensors with
    nothing else asked are ``torch.matmul(a, b, out=out)`` itself, with torch's own
    checks.  Every other product is computed by ``matmul(a, b, **kwargs)``, refused by
    code when it is sparse (an STensor, or a torch sparse tensor: two torch compressed
    operands' product, ``_torch_compressed_product``), and ``out`` resized by torch's rule (``_resize_out``, which warns when a non-empty ``out`` has
    another shape) and filled.  A product computed into a temporary cannot
    alias an operand, so these routes accept an ``out`` that overlaps one.
    """

    if not isinstance(out, torch.Tensor) or out.layout is not torch.strided:
        raise ScorchTypeError(
            "matmul(): argument 'out' must be a dense torch.Tensor, not "
            f"{type(out).__name__}",
            code="out_argument_type_invalid",
        )
    if torch.is_grad_enabled() and (
        out.requires_grad or _operand_requires_grad(a) or _operand_requires_grad(b)
    ):
        raise ScorchRuntimeError(
            "matmul(): functions with out=... arguments don't support automatic "
            "differentiation, but one of the arguments requires grad.",
            code="out_argument_requires_grad",
        )
    requested = kwargs.get("format")
    if "output_mode_order" in kwargs or (
        requested is not None and not parse_format(requested).is_dense()
    ):
        raise KeywordArgumentError(
            "matmul(): out= takes a dense result in out's own layout; a sparse format "
            "or an output_mode_order cannot be given with it",
            code="out_requires_dense_result",
        )
    dtype = getattr(a, "dtype", None)
    if dtype is not None and out.dtype != dtype:
        raise ScorchRuntimeError(
            f"Expected out tensor to have dtype {dtype}, but got {out.dtype} instead",
            code="out_dtype_mismatch",
        )
    if out.device.type != "cpu":
        raise ScorchRuntimeError(
            f"Expected out tensor to be on the cpu, but got {out.device} instead",
            code="out_device_mismatch",
        )
    dense_tensors = (
        type(a) is torch.Tensor
        and type(b) is torch.Tensor
        and a.layout is torch.strided
        and b.layout is torch.strided
    )
    if dense_tensors and set(kwargs) <= {"time_dict"}:
        start_time = time.time()
        torch.matmul(a, b, out=out)
        if "time_dict" in kwargs:
            kwargs["time_dict"]["eval_time"] = time.time() - start_time
        return out
    result = matmul(a, b, **kwargs)
    if not isinstance(result, torch.Tensor) or result.layout is not torch.strided:
        raise KeywordArgumentError(
            "matmul(): out= takes a dense result, and this product's is "
            + (
                f"a torch tensor of layout {result.layout}"
                if isinstance(result, torch.Tensor)
                else f"{type(result).__name__} of format {getattr(result, 'format', None)}"
            ),
            code="out_requires_dense_result",
        )
    _resize_out(out, result.shape)
    out.copy_(result)
    return out


def _resize_out(out: torch.Tensor, shape: torch.Size) -> None:
    """Resize ``out`` to ``shape`` by torch's rule for an ``out=`` argument
    (``at::native::resize_output``): unchanged when the shape matches, else resized, with
    torch's own deprecation warning when ``out`` had elements.  torch's operator for it,
    ``torch._resize_output_``, has no CPU kernel."""

    if out.shape == shape:
        return
    if out.numel() != 0:
        warnings.warn(
            f"An output with one or more elements was resized since it had shape "
            f"{list(out.shape)}, which does not match the required output shape "
            f"{list(shape)}. This behavior is deprecated, and in a future PyTorch "
            "release outputs will not be resized unless they have zero elements. You "
            "can explicitly reuse an out tensor t by resizing it, inplace, to zero "
            "elements with t.resize_(0).",
            UserWarning,
            stacklevel=4,
        )
    out.resize_(shape)


def _as_stensor(x: Union[torch.Tensor, STensor]) -> STensor:
    return x if isinstance(x, STensor) else STensor.from_torch(x)


def _in_storage_order(tensor: torch.Tensor, order: List[int]) -> torch.Tensor:
    """``tensor`` with its elements stored in mode order ``order``, read back in logical
    order: the strides of the dense result a generated kernel stores in that order
    (``STensor.to_torch`` returns it permuted back the same way)."""

    inverse = [0] * len(order)
    for position, mode in enumerate(order):
        inverse[mode] = position
    return tensor.permute(*order).contiguous().permute(*inverse)


def _dense_product(
    a: Union[torch.Tensor, STensor],
    b: Union[torch.Tensor, STensor],
    kwargs: Dict[str, Any],
    compilation_context: Optional[CompilationContext],
) -> Union[torch.Tensor, STensor]:
    """``matmul`` of two dense operands, torch tensors or dense STensors, called with
    keywords: torch's product, with the keywords honoured as on every other route.

    ``time_dict`` is filled; a sparse ``format`` converts the result; an
    ``output_mode_order`` stores a dense result in that order (``_in_storage_order``).
    A sparse result in another mode order is the generated kernel's, which stores it so.
    A separate function so ``matmul`` stays within the project's complexity bound.
    """

    requested = kwargs.get("format")
    output_format = None if requested is None else parse_format(requested)
    sparse_result = output_format is not None and not output_format.is_dense()
    order = kwargs.get("output_mode_order")
    if sparse_result and order is not None:
        return _dense_result_as_tensor(
            _generated_matmul(_as_stensor(a), _as_stensor(b), dict(kwargs), None)
        )
    start_time = time.time()
    if isinstance(a, STensor):
        result_torch = _dense_layouts_product(a, cast(STensor, b))
    else:
        result_torch = torch.matmul(a, cast(torch.Tensor, b))
    time_dict = kwargs.get("time_dict")
    if time_dict is not None:
        time_dict["eval_time"] = time.time() - start_time
    if sparse_result:
        options = _matmul_options(None, kwargs, compilation_context)
        return STensor.from_torch(result_torch).to_sparse(
            output_format, **_conversion_kwargs(options, compilation_context)
        )
    return result_torch if order is None else _in_storage_order(result_torch, order)


def _dense_tensors_product(
    a: torch.Tensor, b: torch.Tensor, kwargs: Dict[str, Any]
) -> Union[torch.Tensor, STensor]:
    """``matmul`` of two dense torch tensors called with keywords (``_dense_product``).

    A schedule is refused, as for every all-dense product: the policy's own question, so
    a schedule carried only inside ``_compile_options`` is refused too."""

    if _matmul_compile_policy(kwargs)[1] is not None:
        raise ScheduleWithoutSparseOperandError()
    return _dense_product(a, b, kwargs, _matmul_compilation_context(kwargs))


def _matmul_compilation_context(kwargs: Dict[str, Any]) -> Optional[CompilationContext]:
    """The ``_compilation_context`` a ``matmul`` call was handed, refused by code unless
    it is a ``CompilationContext``.  A separate function so ``matmul`` stays within the
    project's complexity bound."""

    compilation_context = kwargs.get("_compilation_context")
    if (
        compilation_context is not None
        and type(compilation_context) is not CompilationContext
    ):
        raise ScorchTypeError(
            "_compilation_context must be a CompilationContext instance",
            code="compilation_context_type_invalid",
        )
    return compilation_context


def _require_matrix_vector_operands(a: object, b: object) -> None:
    """Refuse an ``spmv`` whose operands are not a matrix and a vector of its width,
    with the codes ``einsum("ij,j->i", a, b)`` gives the same mistakes, before any CIN
    exists.  The CIN compiler checked only each operand's rank against its access
    (at a stage boundary, with its own code) and no extent: its kernel bounds ``j`` by
    ``a`` and read a shorter vector past its end.

    Every correct call passes one condition; the refusal is formed out of line
    (``_refuse_matrix_vector_operands``), so a correct call builds no tuple and runs no
    loop: 50 ns on the M5, where checking each part in turn cost 120 ns.
    """

    if isinstance(a, STensor) and isinstance(b, STensor):
        a_shape = a._metadata.layout.logical_shape
        b_shape = b._metadata.layout.logical_shape
        if len(a_shape) == 2 and len(b_shape) == 1 and a_shape[1] == b_shape[0]:
            return
    _refuse_matrix_vector_operands(a, b)
    # Not reached: the condition above fails only on a mistake the refusal names.
    raise ScorchAssertionError(
        "spmv operands were refused but no mistake was found",
        code="spmv_operand_refusal_unfounded",
    )


def _refuse_matrix_vector_operands(a: object, b: object) -> None:
    """The refusal ``_require_matrix_vector_operands`` makes for operands that failed its
    condition: the first mistake, in the order type, rank, extent.  It returns only for
    operands with no mistake, which its caller then refuses as an internal error.
    """

    if not isinstance(a, STensor) or not isinstance(b, STensor):
        position, operand = (
            ("first", a) if not isinstance(a, STensor) else ("second", b)
        )
        raise TensorTypeError(
            f"spmv's {position} operand must be an STensor, "
            f"got {type(operand).__name__}",
            code="spmv_operand_type_invalid",
        )
    a_shape = a._metadata.layout.logical_shape
    b_shape = b._metadata.layout.logical_shape
    for operand, (shape, rank) in enumerate(((a_shape, 2), (b_shape, 1))):
        if len(shape) != rank:
            raise OperandRankMismatchError(
                f"spmv operand {operand} has rank {len(shape)}, not {rank}"
            )
    if a_shape[1] != b_shape[0]:
        raise IndexExtentMismatchError(
            f"spmv operands of shapes {tuple(a_shape)} and {tuple(b_shape)} cannot be "
            f"multiplied: contracted extents {a_shape[1]} and {b_shape[0]}"
        )


def _require_output_format_rank(
    requested: Optional[FormatInput], result_rank: int, operation: str
) -> Optional[TensorFormat]:
    """``requested`` parsed, after checking it has one level per mode of the result.

    A requested output format of another rank is the caller's argument mistake, so it
    is refused here, at the front door, with the code ``einsum`` gives it, before any
    CIN exists for the compiler's verifier to reject at a stage boundary.  ``None``
    (nothing requested) comes back as ``None``.
    """

    if requested is None:
        return None
    parsed = parse_format(requested)
    if parsed.get_order() != result_rank:
        raise CompileSpecError(
            f"{operation} output format rank {parsed.get_order()} "
            f"does not match result rank {result_rank}",
            code="einsum_output_format_rank_mismatch",
        )
    return parsed


def _dense_result_as_tensor(
    result: Union[torch.Tensor, STensor],
) -> Union[torch.Tensor, STensor]:
    """What ``matmul`` returns for ``result``: a ``torch.Tensor`` when it is an
    ``STensor`` in a dense format, the object itself otherwise.

    The return type follows the result's format, not the kernel that computed the
    product: a prebuilt kernel and a generated one that both produce a dense result
    hand back the same type.  A separate function so ``matmul`` has one statement
    of the rule and stays within the project's complexity bound.

    The tensor is not a view (``_not_a_view``) unless it records a gradient, as
    torch.matmul's product is not: ``to_torch`` lays the stored values out as a view of
    their flat buffer.
    """

    if isinstance(result, STensor) and result.format.is_dense():
        return _not_a_view(result.to_torch())
    return result


def _matmul_result(
    result: Union[torch.Tensor, STensor], torch_result: bool
) -> Union[torch.Tensor, STensor]:
    """What ``matmul`` returns for the product ``result``: for two torch sparse CSR
    operands whose result is torch's type (``_TORCH_RESULT_KEYWORDS``), the torch sparse
    tensor holding its arrays (``_as_torch_compressed``); else ``_dense_result_as_tensor``'s.
    A batched product computed by its rank-two products (``_batched_by_matrix_products``)
    is a torch tensor already, which ``_dense_result_as_tensor`` returns as it is.
    """

    if torch_result:
        return _as_torch_compressed(cast(STensor, result))
    return _dense_result_as_tensor(result)


def _as_torch_compressed(result: STensor) -> torch.Tensor:
    """``result``, the product of two torch compressed operands (``_torch_compressed_product``)
    made by the route a keyword or a CSC operand chose, as the torch sparse tensor over its arrays: a CSR result (format ``ds``) stored in
    mode order [0, 1] as a torch sparse CSR tensor, one stored in [1, 0] -- the compressed
    columns output_mode_order=[1, 0] asks for -- as a torch sparse CSC tensor, torch's type for
    that order. Its index type is ``_torch_csr_spgemm``'s (``_SPGEMM_INT64_INDICES``), so the
    result's type and index type do not depend on the keywords. The arrays are the result's own,
    widened where the index type differs, as nothing else holds the result.

    Every route such a call takes makes a ``ds`` result in one of the two orders: the prebuilt
    CSR x CSR kernel's, einsum's inferred format for ``ij,jk->ik`` over two compressed operands
    (a CSC one read as the CSR of its transpose), and
    its relayout to the order asked for, which keeps the format's letters by storage position
    (``_in_default_order``). A result in any other form is refused by code, not converted.
    """

    order = tuple(result.mode_order)
    if result.format != _CSR_RESULT_FORMAT or order not in _TORCH_COMPRESSED_LAYOUTS:
        raise ScorchNotImplementedError(
            f"matmul(): the product of two torch compressed operands came back in format "
            f"{result.format} stored in mode order {list(order)}, which has no torch sparse "
            "layout here",
            code="torch_result_layout_unsupported",
        )
    positions, coordinates = result.storage._native_mode_indices()[1]
    stored = int(positions[-1])
    index = torch.int64 if _SPGEMM_INT64_INDICES else torch.int32
    return _TORCH_COMPRESSED_LAYOUTS[order](
        positions.to(index),
        coordinates[:stored].to(index),
        result.storage.value[:stored].to(result.dtype),
        size=tuple(result.logical_shape),
    )


_CSR_RESULT_FORMAT = parse_format("ds")
_TORCH_COMPRESSED_LAYOUTS: Dict[Tuple[int, ...], Callable[..., torch.Tensor]] = {
    (0, 1): torch.sparse_csr_tensor,
    (1, 0): torch.sparse_csc_tensor,
}


def _not_a_view(t: torch.Tensor) -> torch.Tensor:
    """``t``, or a new tensor over its memory when it is a view that records no gradient.

    A product a custom autograd Function returns must not be a view: torch refuses an
    in-place write in grad mode to a view made inside the Function's forward, which a
    product of torch's never is.  autograd.py's Functions call ``matmul`` only with a CSR
    operand of float32 or float64 and a dense matrix (``autograd.serve``; a vector is
    unsqueezed), which the prebuilt CSR kernel covers, so the product they get is the
    kernel's own ``[rows, k]`` tensor (spmm.h allocates it in that shape, for a plan too),
    which is not a view.  This is for a product made elsewhere: every dense result
    ``matmul`` builds as an STensor passes through it (``_dense_result_as_tensor``).
    A view that records a gradient is returned as it is: detaching it would drop the
    gradient, and an ordinary autograd view takes in-place writes (only a view made inside
    a custom Function's forward, or under ``no_grad``, refuses them).
    """
    return t if t._base is None or t.requires_grad else t.detach()


#: The keyword arguments a batched product may carry and still take one of the matrix
#: routes below (``_batched_by_matrix_products``): any other names a result or a kernel
#: the batched einsum honours, so it goes there.
_BATCHED_ROUTE_KEYWORDS = frozenset({"time_dict"})


def _stacked_matrix(x: Any) -> Optional[STensor]:
    """``x``, a sparse STensor of rank three or more stored in logical order, as the CSR
    matrix of all its rows: batch modes and row mode stacked into one row mode of their
    product's extent, the column mode kept.  None for any other ``x``.

    - Batch and row levels dense, column level compressed (format ``d...ds``): the
      compressed level's position array already runs over the stacked rows, so the
      matrix adopts ``x``'s arrays and values as they are, with no copy.
    - Every level a coordinate level (format ``o...o``, a torch COO tensor's): when the
      entries are in row-major order (each one's stacked row no smaller than the one
      before, which a coalesced COO tensor's are), the position array is the running
      count of entries per stacked row, one pass over the entries; the column array and
      the values are adopted.  Out of order entries give None.
    """

    if type(x) is not STensor or x.dim() < 3:
        return None
    if list(x.mode_order) != list(range(x.dim())):
        return None
    levels = x.format.get_level_types()
    shape = tuple(int(extent) for extent in x.logical_shape)
    rows = 1
    for extent in shape[:-1]:
        rows *= extent
    arrays = x._native_mode_indices()
    if levels[-1] is LevelType.COMPRESSED and all(
        level is LevelType.DENSE for level in levels[:-1]
    ):
        mode_indices = [[], list(arrays[-1])]
    elif all(level is LevelType.COORDINATE for level in levels):
        stacked = arrays[0][-1].to(torch.int64)
        for level, extent in zip(arrays[1:-1], shape[1:-1]):
            stacked = stacked * extent + level[-1]
        if stacked.numel() > 1 and not bool((stacked[1:] >= stacked[:-1]).all()):
            return None
        counts = torch.bincount(stacked, minlength=rows)
        columns = arrays[-1][-1]
        pos = torch.zeros(rows + 1, dtype=columns.dtype)
        pos[1:] = torch.cumsum(counts, 0)
        mode_indices = [[], [pos, columns]]
    else:
        return None
    return STensor._from_parts(
        None,
        (rows, shape[-1]),
        "ds",
        mode_indices,
        x._raw_values,
        adopt=True,
    )


def _dense_torch(x: Any) -> Optional[torch.Tensor]:
    """``x`` as a dense torch tensor in logical order when it is one already (a strided
    torch tensor, or a dense STensor, whose values are viewed, not copied); else None.
    """

    if isinstance(x, torch.Tensor) and x.layout is torch.strided:
        return x
    if type(x) is STensor and x.format.is_dense():
        return x.to_torch(in_place=False)
    return None


def _batched_by_matrix_products(
    a: Any, b: Any, time_dict: Optional[dict]
) -> Optional[torch.Tensor]:
    """A batched product computed by the rank-two products it is made of, when the
    operands' storage lets it be, else None (the batched einsum computes it).  Each route
    returns what the batched einsum returns for the call (``matmul_rules``): a dense
    result, as a torch tensor, since each takes a sparse operand times a dense one.

    - A sparse operand of rank three or more whose storage is its stacked rows' CSR matrix
      (``_stacked_matrix``) times a dense matrix or vector, or a dense operand whose batch
      extents are all 1 (the matrix it wraps): that matrix times the dense matrix, one
      rank-two product over every batch's rows at once, reshaped.
    - A sparse matrix, or a sparse operand whose batch extents are all 1 with that storage,
      times a dense operand of rank three or more: the matrix times the dense operand's
      batches laid side by side as one wide matrix (its contracted mode first, then batch
      and column modes), one rank-two product, reshaped back.
    - A sparse operand with that storage times a dense operand of rank three or more with
      the same batch shape: the block-diagonal CSR matrix of its batches (its column
      indices offset by the batch's position times the contracted extent, one pass over the
      nonzeros into a new index array) times the dense operand's batches stacked, one
      rank-two product.

    ``time_dict`` is the rank-two product's.  Gradients are recorded by autograd.py as the
    batched einsum's (``einsum_grad``), whose forward calls ``matmul`` on the operands
    without history and so may come here.

    A route's result is its rank-two product reshaped, which can be a view of that
    product; ``_generated_matmul`` returns it through ``_not_a_view``, as the batched
    einsum's dense result is returned (``_dense_result_as_tensor``), so the product the
    caller receives is not a view, as torch.matmul's is not.
    """

    dense_b = _dense_torch(b)
    if dense_b is None or type(a) is not STensor or a.format.is_dense():
        return None
    a_shape = tuple(a.logical_shape)
    b_shape = tuple(dense_b.shape)
    p, q = len(a_shape), len(b_shape)
    if a_shape[-1] != (b_shape[0] if q == 1 else b_shape[-2]):
        return None  # the batched einsum refuses it by code
    kwargs = {} if time_dict is None else {"time_dict": time_dict}
    stacked = _stacked_matrix(a) if p >= 3 else None
    if stacked is not None and q <= 2:
        product = _sparse_dense_product(stacked, dense_b, kwargs)
        return product.reshape(a_shape[:-1] + b_shape[1:])
    if stacked is not None and all(e == 1 for e in b_shape[:-2]):
        # Every batch extent of the dense operand is 1: it is the matrix it wraps, and
        # the result's batch shape is ``a``'s, after any further leading 1s it has.
        product = _sparse_dense_product(stacked, dense_b.reshape(b_shape[-2:]), kwargs)
        lead = (1,) * max(q - p, 0)
        return product.reshape(lead + a_shape[:-1] + (b_shape[-1],))
    if q >= 3 and (
        p == 2 or (stacked is not None and all(e == 1 for e in a_shape[:-2]))
    ):
        matrix = a if stacked is None else stacked
        rows, inner = a_shape[-2], a_shape[-1]
        batch = b_shape[:-2]
        width = dense_b.movedim(-2, 0).reshape(inner, -1)
        product = _sparse_dense_product(matrix, width, kwargs)
        out = product.reshape((rows,) + batch + (b_shape[-1],)).movedim(0, -2)
        lead = (1,) * max(p - q, 0)
        return out.reshape(lead + out.shape).contiguous()
    if stacked is not None and q == p and a_shape[:-2] == b_shape[:-2]:
        return _block_diagonal_product(stacked, a_shape, dense_b, kwargs)
    return None


#: The largest column index an int32 index array holds.
_INT32_MAX = 2**31 - 1


def _sparse_dense_product(
    a: STensor, b: torch.Tensor, kwargs: Dict[str, Any]
) -> torch.Tensor:
    """``matmul(a, b)`` for a sparse matrix ``a`` and a dense matrix ``b``, whose result
    is a dense torch tensor on every route (``_plain_dense_result_rank``)."""

    return cast(torch.Tensor, matmul(a, b, **kwargs))


def _block_diagonal_product(
    stacked: STensor,
    a_shape: Tuple[int, ...],
    b: torch.Tensor,
    kwargs: Dict[str, Any],
) -> torch.Tensor:
    """``a @ b`` for ``a`` (of shape ``a_shape``) given as its stacked rows' CSR matrix
    ``stacked`` and a dense ``b`` of the same batch shape: the block-diagonal matrix of
    ``a``'s batches, whose column indices are ``a``'s offset by each batch's position
    times the contracted extent, times ``b``'s batches stacked as rows (a reshape of a
    contiguous ``b``).  The offset indices are a new array, made on every call, so no
    copy of ``a``'s structure outlives the call."""

    pos, crd = stacked._native_mode_indices()[-1]
    rows, inner = a_shape[-2], a_shape[-1]
    batches = 1
    for extent in a_shape[:-2]:
        batches *= int(extent)
    if crd.numel() == 0:
        offset_crd = crd
    else:
        # Batch t's entries are pos[t * rows] to pos[(t + 1) * rows]; each one's column
        # moves by t * inner.  The offsets are computed in int64 and kept there only
        # when the block matrix's columns do not fit in int32.
        starts = pos[::rows].to(torch.int64)
        offsets = torch.repeat_interleave(
            torch.arange(batches, dtype=torch.int64) * inner, starts[1:] - starts[:-1]
        )
        offset_crd = crd.to(torch.int64) + offsets
        if batches * inner > _INT32_MAX:
            pos = pos.to(torch.int64)
        else:
            offset_crd = offset_crd.to(crd.dtype)
    block = STensor._from_parts(
        None,
        (batches * rows, batches * inner),
        "ds",
        [[], [pos, offset_crd]],
        stacked._raw_values,
        adopt=True,
    )
    stacked_b = b.reshape(batches * inner, b.shape[-1])
    product = _sparse_dense_product(block, stacked_b, kwargs)
    return product.reshape(a_shape[:-1] + (b.shape[-1],))


def _generated_matmul(
    a: Union[torch.Tensor, STensor],
    b: Union[torch.Tensor, STensor],
    einsum_kwargs: dict[str, Any],
    options: Optional[_DeferredOptions],
) -> Union[torch.Tensor, STensor]:
    """The product ``matmul`` sends to the generated path, as an ``einsum``.

    The expression is ``torch.matmul``'s rank rules written as einsum labels
    (``matmul_rules.matmul_expression``), batched products included: their batch modes
    broadcast as torch broadcasts them, an operand's batch mode of extent 1 against the
    other's ``n`` taking a label of its own that the product sums over.  A pair with no
    expression (a rank-0 operand, or more batch modes than there are labels) is refused
    by code (``MatmulOperandRankError``) before anything is built; vector x vector is
    refused by ``einsum``, by code, until a scalar result is computed, and batch extents
    that do not broadcast by ``einsum``'s extent check (``einsum_index_extent_mismatch``).

    ``options`` is the call's options holder when an earlier step of the call made it
    (a relayout), and ``einsum`` then runs under it, so the call makes at most one
    snapshot.  Otherwise ``einsum`` makes its own holder from the same keyword
    arguments, which resolves the same schedule.  ``einsum_kwargs`` already carries any
    context the caller handed in.  An ``output_mode_order`` the generated kernel refuses
    by code is honoured by a relayout of the product in the default order
    (``_in_default_order``).  The result is an ``STensor``: ``einsum`` returns a
    ``TensorSpec`` only for ``compile_only=True``, which ``matmul`` does not take
    (``_MATMUL_KEYWORDS``).  A separate function so ``matmul`` stays within the
    project's complexity bound.
    """

    expression = operands_expression(a, b)
    if expression is None:
        raise MatmulOperandRankError((a.dim(), b.dim()))
    if (
        (a.dim() > 2 or b.dim() > 2)
        and options is None
        and einsum_kwargs.keys() <= _BATCHED_ROUTE_KEYWORDS
        and get_forced_schedule() is None
    ):
        served = _batched_by_matrix_products(a, b, einsum_kwargs.get("time_dict"))
        if served is not None:
            return _not_a_view(served)
    if options is not None:
        einsum_kwargs["_deferred_options"] = options
    if "output_mode_order" not in einsum_kwargs:
        return cast(STensor, einsum(expression, a, b, **einsum_kwargs))
    try:
        return cast(STensor, einsum(expression, a, b, **einsum_kwargs))
    except Exception as refusal:
        code = defect_code(refusal)
        if code is None:
            raise
        return _in_default_order(expression, a, b, einsum_kwargs, code)


#: The ``matmul`` calls whose generated kernel refused the ``output_mode_order`` asked
#: for, by the refusal's code: each was computed in the default order and relaid out
#: (``_in_default_order``).  The tests read it to record the route a call took.  The
#: scheduler choosing between the two routes itself is a compiler frontier item.
_ORDER_RELAYOUTS: Dict[str, int] = {}


def _in_default_order(
    expression: str,
    a: Union[torch.Tensor, STensor],
    b: Union[torch.Tensor, STensor],
    einsum_kwargs: dict[str, Any],
    code: str,
) -> STensor:
    """The product ``einsum(expression, a, b, **einsum_kwargs)`` refused by code (``code``)
    for its ``output_mode_order``: computed in the default order and relaid out to the
    order asked for (``STensor.change_mode_order``), so the request is honoured and
    nothing raises.  Before patch 0072 every route but the generated kernel's dropped the
    order and returned the product, so such a call must not raise now.  The relayout
    keeps the format's letters by storage position, as einsum's own relayout from the
    order it computes in does.  An error of the product in the default order
    propagates, its context the refusal.
    """

    order = einsum_kwargs["output_mode_order"]
    kwargs = {k: v for k, v in einsum_kwargs.items() if k != "output_mode_order"}
    result = cast(STensor, einsum(expression, a, b, **kwargs))
    compilation_context = kwargs.get("_compilation_context")
    options = _matmul_options(
        kwargs.get("_deferred_options"), kwargs, compilation_context
    )
    result.change_mode_order(
        list(order), **_conversion_kwargs(options, compilation_context)
    )
    _ORDER_RELAYOUTS[code] = _ORDER_RELAYOUTS.get(code, 0) + 1
    return result


def _matmul_options(
    options: Optional[_DeferredOptions],
    kwargs: dict,
    compilation_context: Optional[CompilationContext],
) -> _DeferredOptions:
    """The options holder of one ``matmul`` call, made by the first step that needs it.

    Every later step of the call (a relayout, the dense product's conversion, ``spmv``,
    ``einsum``) takes the same holder, so a call makes at most one snapshot, and a call
    whose kernels are all kept makes none.  Made from the call's keyword arguments, so
    it resolves the schedule ``_matmul_compile_policy`` resolved.  A context the caller
    handed in is checked against the call's options here, where the call first needs
    them, as before; the holder made its snapshot already, being handed a context.
    """

    if options is not None:
        return options
    options = _DeferredOptions(kwargs)
    if compilation_context is not None:
        _compilation_context_at_boundary(compilation_context, options.options())
    return options


def matmul(
    a: Union[torch.Tensor, STensor],
    b: Union[torch.Tensor, STensor],
    **kwargs: Any,
) -> Union[torch.Tensor, STensor]:
    """Matrix multiplication for any mix of dense and sparse operands.

    The tuned public entry point for two-operand products (SpMV, SpMM, SpGEMM, and
    the dense-dense passthrough). Dispatch proceeds through three tiers, tried in
    order:

    1. **Both-dense torch delegate.** If both operands are dense (dense
       ``torch.Tensor`` x dense ``torch.Tensor``, or dense ``STensor`` x dense
       ``STensor``), the product is handed to ``torch.matmul`` directly.
    2. **Prebuilt hand-written C++ kernel.** ``resolve_prebuilt_matmul`` matches the
       operand ranks/formats/dtype against the native ``scorch_ops`` extension
       (CSR x dense -> dense, CSR x CSR -> CSR, COO x COO -> COO, COO x dense ->
       dense, CSR x vector -> dense vector). On the drop-in ``spmm_csr_float_v2``
       symbol it additionally applies the adaptive tiling selector
       (``tiling.maybe_dispatch``, provably no-regression because v2 is always the
       probe baseline) and the host-thread composition hints.
    3. **Generic JIT compiler pipeline.** On a prebuilt miss the product is emitted
       as the ``einsum`` of ``torch.matmul``'s rank rules, ``einsum("ij,jk->ik", a,
       b, ...)`` for two matrices (2-D x 1-D goes to :func:`spmv`), and for an operand
       of rank three or more the batched product, its batch modes broadcast as torch
       broadcasts them (``matmul_rules``), compiling a C++ kernel through the CIN ->
       LLIR -> codegen pipeline.

    Sparse 2-D operands are normalized to canonical mode order ``[0, 1]`` before
    dispatch so transposed / non-default layouts stay correct on the fast paths.

    Parameters
    ----------
    a, b : torch.Tensor or STensor
        Operands. ``torch.Tensor`` inputs are auto-converted: dense-dense returns
        immediately via ``torch.matmul``; two ``sparse_coo`` tensors are promoted
        to CSR; any other sparse input triggers ``STensor.from_torch``.
    **kwargs
        The keywords below, on every route; any other name raises
        ``KeywordArgumentError`` (``unknown_keyword_argument``).  A keyword given its
        default value is the call without it.
        format : str or list or TensorFormat, optional
            Requested output format. When omitted for a compiled product the output
            format is inferred (see :func:`einsum`). A format naming the dense layout
            the product returns anyway (a sparse matrix times a dense operand) is the
            call without it, so it keeps its gradient and its plan.
        output_format : str or list or TensorFormat, optional
            Alias for ``format``. Given together they must name one format
            (``format_keywords_conflict``).
        output_mode_order : sequence of int, optional
            The result's storage order, a permutation of its modes. A dense result is
            returned with its elements stored in that order; a sparse one is the
            generated kernel's. The identity order is the call without it.
        out : torch.Tensor, optional
            Filled with a dense result by ``torch.matmul``'s rules for ``out=``
            (resized as torch resizes it; refused in grad mode when an argument
            requires grad, and for another dtype or a sparse result).
        use_cache : bool, default True
            When ``True``, allow the prebuilt-kernel fast path and the adaptive
            tiling selector. ``False`` forces the generic ``einsum`` path (used by
            the compiler gap tests).
        schedule : Schedule, optional
            Explicit JIT loop order and affine tiling decisions. Supplying one
            forces the generic compiler path even when a prebuilt kernel matches.
        time_dict : dict, optional
            If given, ``time_dict["eval_time"]`` is set to the kernel wall-clock
            time in seconds.

    Returns
    -------
    torch.Tensor or STensor
        A dense ``torch.Tensor`` when the result format is dense (the common SpMM
        case, e.g. ``ds @ dd -> dd``, a matrix times a vector in any layout, and
        every dense-dense product), otherwise an ``STensor`` (e.g. the SpGEMM
        ``ds @ ds -> ds``).

    Other Parameters
    ----------------
    SCORCH_MATCH_HOST_THREADS : env, default "1"
        Set to ``"0"`` to stop passing ``torch.get_num_threads()`` to the drop-in
        ``spmm_csr_float_v2`` kernel (thread-team matching for GCN pipelines).
    SCORCH_ATPARALLEL_PIPELINE : env, default "1"
        Set to ``"0"`` to launch the SpMM on a private libgomp team instead of
        torch's intra-op pool.

    See Also
    --------
    einsum : The generic compiler front door matmul falls through to.

    Notes
    -----
    Compiled kernels are memoized aggressively; a stale extensions build directory
    can mask a codegen edit, so clear ``TORCH_EXTENSIONS_DIR`` to force a recompile.

    Examples
    --------
    >>> import torch
    >>> import scorch
    >>> A_dense = (torch.rand(256, 256) * (torch.rand(256, 256) < 0.1)).float()
    >>> x = torch.rand(256, 128)
    >>> A = scorch.from_torch(A_dense, "A").to_sparse("ds")   # CSR STensor
    >>> Y = scorch.matmul(A, x)                                # -> dense torch.Tensor
    >>> torch.allclose(Y, A_dense @ x, atol=1e-3, rtol=1e-3)
    True

    Sparse x sparse (SpGEMM) producing a sparse ``STensor``:

    >>> B = scorch.from_torch(A_dense, "B").to_sparse("ds")
    >>> C = scorch.matmul(A, B, format="ds")                   # -> STensor (CSR)
    >>> torch.allclose(C.to_torch(), A_dense @ A_dense, atol=1e-3, rtol=1e-3)
    True
    """

    # A repeated CSR x dense product is served by a plan: the resolution, the
    # selector's verdict, the validated index structure and the kernel arguments
    # were all settled on an earlier call and live in the native extension (see
    # plan.py). What is left is a dict lookup and one Python->C++ hop.
    #
    # Deliberately the first thing this function does, and written as inline
    # dict/attribute reads rather than a helper call, because everything below is
    # what a plan exists to skip. The exact-type tests keep it off every other
    # shape of call at the cost of two pointer comparisons: a torch.fx Proxy, an
    # STensor right-hand operand, a dense left-hand operand and a sparse-sparse
    # product all fail them and fall through unchanged. The keyword arguments left
    # once _matmul_keyword_call has dropped those that change nothing are excluded
    # wholesale -- a sparse format, use_cache=False, a schedule, output_mode_order and
    # time_dict all change what the call means or what it must report.
    # The same test decides whether a plan may serve this call and whether one may
    # be installed for it at the end of the prebuilt branch, so it is made once.
    # plan_a/plan_b are the caller's own objects: a plan is keyed on what the
    # caller will pass again, never on an operand this function wrapped or relaid
    # out, which is why the installer below also checks `a is plan_a`.
    #
    # `_PLAN_ENABLED[0]` comes first so that switching plans off leaves this whole
    # paragraph -- and the installation below, which is what `plan_b is not None`
    # tests for -- costing one list index. That is what makes the off state a
    # control arm for measuring the on state rather than a different code path with
    # its own overhead. Measured: with the probe ungated, the off state ran 2-4%
    # slower than the tree before plans on the small cells of both hosts.
    # The caller's left operand: an STensor, or a torch sparse CSR tensor.
    plan_a: Optional[Union[STensor, torch.Tensor]]
    probed = _NOT_PROBED
    if kwargs:
        # The C entry point's keyword first: a call it handed over has no other.  Then
        # matmul's keyword rule (_matmul_keyword_call), which refuses a name matmul does
        # not take and drops what does not change the call, so a call left with no
        # keywords is the plain call from here on: autograd.py, plans, the same value.
        probed = _c_probed(a, b, kwargs)
        if kwargs and (done := _matmul_keyword_call(a, b, kwargs)) is not _PLAIN_CALL:
            return done
    plan_a, plan_b, plan_key = probed
    # Tier 1 first: two dense torch tensors are torch's product. Testing for them before
    # anything else keeps every test below off the path of a dense model's products, all
    # of which `import scorch as torch` sends here.  torch sets `is_sparse` only for COO
    # and `is_sparse_csr` only for CSR, so this also sends torch every pair of torch
    # tensors whose sparse operands are all CSC, BSR or BSC, keywords included, as it has
    # since dcdcdd2: what torch computes of those is computed as before, and what it
    # refuses is refused by torch as before.  CSC with a COO or CSR operand comes past
    # here, and from_torch adopts the CSC (stensor.TORCH_ADOPTED_LAYOUTS); BSR and BSC
    # there are refused by from_torch's code.  Sending CSC pairs to scorch's own kernels
    # waits for a measurement against torch's (patch 0073b).
    if (
        isinstance(a, torch.Tensor)
        and isinstance(b, torch.Tensor)
        and not (a.is_sparse or a.is_sparse_csr or b.is_sparse or b.is_sparse_csr)
    ):
        # Keywords are honoured as on every other route (_dense_product).  Without them
        # the policy is the context's forced schedule alone, read here directly: every
        # dense product a dense model makes comes this way.
        if kwargs:
            return _dense_tensors_product(a, b, kwargs)
        if get_forced_schedule() is not None:
            raise ScheduleWithoutSparseOperandError()
        return torch.matmul(a, b)
    # ``scorch.compile`` traces user functions with torch.fx.  Keep the leaf
    # behavior on the function itself so tracing never has to replace the
    # public ``scorch.matmul`` binding (or aliases in user globals).  A Proxy
    # carries its tracer, which can record this exact call without descending
    # into the eager dispatch below, autograd.py's included.
    proxy = a if isinstance(a, Proxy) else b if isinstance(b, Proxy) else None
    if proxy is not None:
        return proxy.tracer.create_proxy(
            "call_function", matmul, (a, b), kwargs
        )  # type: ignore[return-value]

    # A product's operands hold one value dtype: torch raises RuntimeError for a mixed
    # pair and does not promote, so this raises the same, before anything below looks
    # up or builds a kernel (autograd.py, plans, prebuilt kernels, spmv, einsum).  Two
    # dense torch tensors were handed to torch above, which raises it itself.
    # The three pairs a warm call makes are compared inline, because torch dtypes are
    # singletons, so ``is`` is equality.  The function's frame, its two getattr calls
    # and STensor.dtype cost 0.7 us of a 41 us warm STensor call on mkt1 (beat-mkl,
    # sc8r).  The third pair, two STensors, is every sparse x sparse product.  Every
    # other pair, and every mismatch, goes through the function.
    if not (
        (
            type(b) is _TENSOR
            and (
                (type(a) is STensor and a._metadata.dtype is b.dtype)
                or (type(a) is _TENSOR and a.dtype is b.dtype)
            )
        )
        or (
            type(b) is STensor
            and type(a) is STensor
            and a._metadata.dtype is b._metadata.dtype
        )
    ):
        _require_one_product_dtype(a, b)
    # Operands whose contracted extents differ are refused where the product is computed,
    # with IndexExtentMismatchError (einsum_index_extent_mismatch) on every route, and
    # nothing reads either operand first: a prebuilt kernel's native entry check carries
    # the code (raise_if_native_coded), einsum checks its index extents before it builds
    # a kernel, spmv checks its operands before its CIN exists, and two dense layouts get
    # torch's refusal under the same class.  A check here instead cost every call
    # (0.2-0.4 us of a warm call on the M5, fd1) to refuse what those checks refuse.  The
    # C entry (planned_matmul.h) never serves a mismatch: a plan is installed only for a
    # product computed here, under the dense operand's shape.

    plan_via = None
    # autograd.py next (autograd.serve): a product a gradient is wanted through is recorded
    # there, which has to come before the plan probe, since a plan would serve the product
    # and drop the gradient; and a dense x times a sparse A is computed there whether or
    # not a gradient is wanted, so that product has one type and one value in every grad
    # mode. The C entry point hands calls that want a gradient straight here. A call it
    # probed is one with a sparse left operand it found no gradient wanted through
    # (planned_matmul.h, planned_matmul_wants_grad), so autograd.py is skipped for it:
    # that is every call with a torch CSR operand made afresh each time.
    if (
        probed is _NOT_PROBED
        and (served := _autograd.serve(a, b, matmul, kwargs)) is not None
    ):
        return served
    # `probed is _NOT_PROBED` as well as `not kwargs`: _c_probed pops the C entry point's
    # keyword, so a call it handed over has no keywords left by now. A plan runs a prebuilt
    # kernel, which takes no schedule, so none serves a call made under a forced schedule:
    # the generated kernel below honours the schedule or refuses it with a code. Read last,
    # where only a call a plan could serve pays for it, as the C entry point reads it, and
    # only once schedule_force has run in this process (scheduler.FORCE_SEEN_CELL).
    if (
        _PLAN_ENABLED[0]
        and probed is _NOT_PROBED
        and not kwargs
        and type(b) is _TENSOR
        and (not _SCHEDULE_FORCE_SEEN[0] or _SCHEDULE_FORCE.get() is None)
    ):
        if type(a) is STensor:
            plan_a, plan_b = a, b
            # Built once and handed to the installer below, because every call that
            # gets here needs it and it used to be built twice: hashing it means
            # hashing a torch.Size and a dtype, which is most of what the probe costs
            # on a call that has no plan to find.
            plan_key = (b.shape, b.dtype, _PLAN_GENERATION[0])
            plans = a.__dict__.get(_PLANS_ATTR)
            if plans is not None:
                plan = plans.get(plan_key)
                # A miss means no plan for this free dimension; None back from `run`
                # means a plan that declined (see plan.py). Both fall through.
                if plan is not None:
                    served = plan.run(
                        a._storage._value, b, _PLAN_NTHREADS(), _PLAN_ATPARALLEL
                    )
                    if served is not None:
                        return served
        elif type(a) is _TENSOR and a.layout is _SPARSE_CSR:
            # The same for a torch sparse CSR operand (see _run_torch_operand_plan).
            plan_a, plan_b = a, b
            plan_key = (b.shape, b.dtype, _PLAN_GENERATION[0])
            served = _run_torch_operand_plan(a, b, plan_key)
            if served is not None:
                return served

    # With no keyword arguments the policy is the context's forced schedule.  Read
    # directly, it skips the policy's two frames on every call that gets this far.  The
    # policy also checks a ``_compile_options`` handed in against the call's schedule.
    effective_schedule = (
        _matmul_compile_policy(kwargs)[1] if kwargs else get_forced_schedule()
    )
    compilation_context = _matmul_compilation_context(kwargs) if kwargs else None
    forced_format = (
        None if effective_schedule is None else _forced_result_format(a, b, kwargs)
    )
    # The call's options holder, made by the first step that needs options
    # (``_matmul_options``) and handed to every later one.
    options: Optional[_DeferredOptions] = None

    # Whether the product is two torch compressed operands' whose result is torch's type
    # (_torch_compressed_product, _TORCH_RESULT_KEYWORDS): for two CSR operands with at most
    # time_dict= asked of the call, the torch sparse CSR tensor from the function the C entry
    # point calls for it; for any other such keyword or a CSC operand, the product of the
    # route it chooses, returned in torch's type (_matmul_result).
    torch_result = False
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        torch_result = (
            effective_schedule is None
            and kwargs.keys() <= _TORCH_RESULT_KEYWORDS
            and _torch_compressed_product(a, b)
        )
        if (
            torch_result
            and kwargs.keys() <= _SPGEMM_SERVED_KEYWORDS
            and a.layout is _SPARSE_CSR
            and b.layout is _SPARSE_CSR
            and (served := _torch_csr_spgemm(a, b, kwargs.get("time_dict"))) is not None
        ):
            return served
        # Two COO matrices go over as CSR, the format chosen for them (a schedule choice
        # kept as the candidate until the coordinate route is measured against it).  Only
        # matrices whose two modes are both sparse: torch's to_sparse_csr refuses any other
        # COO tensor (a vector, a rank-3 tensor, a hybrid one with dense value dimensions)
        # with an uncoded RuntimeError.  Those are handed over as they are, and from_torch
        # makes them what it makes of every COO tensor ("o" per mode, or its own code for a
        # hybrid one), to be computed or refused by code like any operand of their rank.
        if (
            a.layout is torch.sparse_coo
            and b.layout is torch.sparse_coo
            and a.dim() == 2
            and b.dim() == 2
            and a.dense_dim() == 0
            and b.dense_dim() == 0
        ):
            a = a.to_sparse_csr()
            b = b.to_sparse_csr()
        # One of the two is sparse: two dense ones were returned at the top.
        a = STensor.from_torch(a)
        b = STensor.from_torch(b)
        # The STensor a plan for the caller's torch operand may be installed from:
        # this one, provided nothing below relays it out into a copy. (Installation
        # also needs `plan_b`, which is set only when the probe above saw a torch
        # sparse CSR operand.)
        plan_via = a

    if isinstance(a, torch.Tensor):
        a = STensor.from_torch(a)
    if isinstance(b, torch.Tensor):
        b = STensor.from_torch(b)

    # Dense STensor x dense STensor: use torch dense matmul directly.
    # This is both faster and more reliable than lowering through sparse
    # scheduling paths for fully dense operands.
    if a.format.is_dense() and b.format.is_dense():
        if effective_schedule is not None:
            raise ScheduleWithoutSparseOperandError()
        if kwargs:
            return _dense_product(a, b, kwargs, compilation_context)
        return _dense_layouts_product(a, b)

    # Never silently swallow a codegen schedule in a prebuilt dispatch.  A result
    # stored in another mode order (output_mode_order: only a non-identity order is
    # left by now) is the generated kernel's: the prebuilt kernels store their own.
    use_cache = (
        kwargs.get("use_cache", True)
        and effective_schedule is None
        and "output_mode_order" not in kwargs
    )
    time_dict = kwargs.get("time_dict", None)
    # output_format= arrived as format= (_entry_keywords).
    requested_output_format = kwargs.get("format")
    # A forced schedule's kernel makes the unforced result's format (_forced_result_format).
    einsum_kwargs = (
        dict(kwargs) if forced_format is None else dict(kwargs, format=forced_format)
    )

    if a.dim() == 2 and b.dim() == 1:
        if effective_schedule is not None:
            raise ScheduleNotThreadedThroughSpMVError()
        # A matrix times a vector is a vector: a format of any other rank is refused
        # here, as einsum refuses it on every other branch, before the prebuilt
        # resolution or spmv sees it (the CIN verifier caught it at a stage boundary,
        # with its own code; pp's prebuilt path crashed on it).
        _require_output_format_rank(requested_output_format, 1, "matmul")
        default_mode_order = [0, 1]
        if (
            not a.format.is_dense()
        ) and a.storage.index.mode_order != default_mode_order:
            a = a.copy()
            options = _matmul_options(options, kwargs, compilation_context)
            a.change_mode_order(
                default_mode_order, **_conversion_kwargs(options, compilation_context)
            )

        if use_cache:
            resolved = resolve_prebuilt_matmul(
                a, b, output_format=requested_output_format
            )
            if resolved is not None:
                result_cpp, result_shape = execute_prebuilt_binary_kernel(
                    resolved.fn, a, b, time_dict=time_dict
                )
                result: Union[torch.Tensor, STensor] = _wrap_generated_result(
                    shape=result_shape,
                    tensor_format=resolved.output_format,
                    result_cpp=result_cpp,
                )
                return _dense_result_as_tensor(result)

        spmv_kwargs = dict(kwargs)
        spmv_output_format = spmv_kwargs.pop("format", None)
        # Under the call's holder, so the SpMV makes a snapshot only when its kernel is
        # not kept.  ``spmv_kwargs`` carries any context the caller handed in.
        # ``spmv`` itself still returns the STensor it built; ``matmul`` hands a dense
        # result back as a torch.Tensor, as its prebuilt branch above does.
        return _dense_result_as_tensor(
            _spmv_owned(
                a,
                b,
                spmv_output_format,
                spmv_kwargs,
                _matmul_options(options, kwargs, compilation_context),
            )
        )

    # Normalize sparse 2D operands to canonical mode order before dispatch.
    # This keeps fast kernels correct and avoids known non-default mode-order
    # issues in the generic path.
    if a.dim() == 2 and b.dim() == 2:
        default_mode_order = [0, 1]
        has_non_default_mode_order = (
            a.storage.index.mode_order != default_mode_order
            or b.storage.index.mode_order != default_mode_order
        )
        has_sparse_input = (not a.format.is_dense()) or (not b.format.is_dense())
        if has_non_default_mode_order and has_sparse_input:
            if a.storage.index.mode_order != default_mode_order:
                a = a.copy()
                options = _matmul_options(options, kwargs, compilation_context)
                a.change_mode_order(
                    default_mode_order,
                    **_conversion_kwargs(options, compilation_context),
                )
            if b.storage.index.mode_order != default_mode_order:
                b = b.copy()
                options = _matmul_options(options, kwargs, compilation_context)
                b.change_mode_order(
                    default_mode_order,
                    **_conversion_kwargs(options, compilation_context),
                )

    if use_cache:
        resolved = resolve_prebuilt_matmul(a, b, output_format=requested_output_format)
        if resolved is not None:
            nthreads, atparallel = _composition_hints(resolved)
            result_cpp, result_shape, _plan_kind, _plan_param = _dispatch_tiled(
                a, b, resolved, nthreads, atparallel, time_dict
            )
            if result_cpp is None:
                result_cpp, result_shape = execute_prebuilt_binary_kernel(
                    resolved.fn,
                    a,
                    b,
                    time_dict=time_dict,
                    nthreads=nthreads,
                    atparallel=atparallel,
                )
            # Fast path: a dense output kernel already produced the contiguous
            # row-major [rows, k] product, allocated in that shape (spmm.h), so it
            # is returned as it is: no reshape, and not a view. This skips the
            # STensor/TensorIndex construction and the to_dense()/to_torch()
            # round-trip — all pure per-call Python overhead here, since the
            # result is default mode order with matching dtype (exactly what
            # to_torch() would have reconstructed).
            if resolved.output_format.is_dense():
                out = result_cpp.storage.value
                # Everything above is a function of the operands' structure and
                # the free dimension, so a second call with the same pair need not
                # repeat it: hand the resolution and the kernel that actually ran to
                # a plan the next call can invoke directly (plan.py). `plan_b` is
                # None unless plans are on and the caller passed a sparse STensor or
                # torch sparse CSR tensor and a dense torch.Tensor with no keywords;
                # `a is plan_a` (or, for a torch operand, `a is plan_via`) fails when
                # the operand was relaid out into a copy above, and a plan must
                # describe the operand the caller holds, not one this function made.
                # The installer builds a torch operand's plan from its own arrays.
                if plan_b is not None and (a is plan_a or a is plan_via):
                    _plan_install(
                        plan_a,
                        plan_b,
                        plan_key,
                        resolved.symbol_name,
                        _plan_kind,
                        _plan_param,
                    )
                return out
            result = _wrap_generated_result(
                shape=result_shape,
                tensor_format=resolved.output_format,
                result_cpp=result_cpp,
            )
        else:
            # A snapshot made here would be made on every call, kernel cached or not.
            # ``einsum`` makes one only where it needs one, under the holder a relayout
            # above made, or under its own from the same keyword arguments and context,
            # which resolves the same schedule.  So the call has at most one snapshot.
            result = _generated_matmul(a, b, einsum_kwargs, options)
    else:
        # As above: ``einsum`` makes the snapshot where it needs one.
        result = _generated_matmul(a, b, einsum_kwargs, options)

    return _matmul_result(result, torch_result)


# The planned fast path in C (csrc/planned_matmul.h). It serves exactly the calls the probe at
# the top of `matmul` serves -- two positional operands, no keywords, `type(a) is STensor` or a
# 2-D torch sparse CSR tensor, `type(b) is torch.Tensor`, plans enabled, a plan under the same
# key -- and passes every other call, arguments untouched, to the Python function above, which
# still carries the probe for a build without this entry point. The one exception is a torch
# operand it does not serve, which arrives with `_C_PROBED_KW` (see the top of `matmul`). A cold planned call through the Python probe cost 1.2-1.5 us
# more than the legacy entry point on mkt1, which is what this removes.
#
# Rebinding the module global is what keeps tracing whole: a Proxy is never served here, so it
# reaches the Python function, which records its global `matmul` -- this object -- as the
# node's target, and `node.target is ops.matmul` (trace.py) holds. `_matmul_python` stays
# importable for tests that compare the two paths.
_matmul_python = matmul
_install_planned_matmul = getattr(_scorch_ops, "install_planned_matmul", None)
if _install_planned_matmul is not None:
    matmul = _install_planned_matmul(  # type: ignore[assignment]
        _matmul_python,
        STensor,
        _TENSOR,
        _PLAN_GENERATION,
        _PLAN_ENABLED,
        _PLANS_ATTR,
        _PLAN_SEEN_ATTR,
        _PLAN_MAX_PER_OPERAND,
        _C_PROBED_KW,
        _MATCH_HOST_THREADS,
        _PLAN_ATPARALLEL,
        sys.modules[__name__],
        _matmul_python.__doc__ or "",
    )
    # A torch CSR operand's product that autograd must record is then recorded by the C
    # entry itself when the operand has a plan (planned_matmul.h, PlannedMatmulGrad), with
    # this as its backward; a build without the setter sends such products here instead.
    _set_mm_backward = getattr(_scorch_ops, "set_planned_matmul_backward", None)
    if _set_mm_backward is not None:
        _set_mm_backward(_autograd.planned_mm_backward)
    # And x @ A for a torch CSR A, once its transposed product has a plan (planned_matmul.h,
    # PlannedMatmulGradT), with this as its backward.
    _set_right = getattr(_scorch_ops, "set_planned_matmul_right", None)
    if _set_right is not None:
        _set_right(_autograd.planned_mm_t_backward, _PLAN_TRANSPOSED_ATTR)
    # And a torch CSR operand's product no plan serves, where it is the call this function would
    # make: the drop-in SpMM, where the product resolves to it, and at float32 only for a shape the
    # tiling selector's gate declines (planned_matmul.h, planned_matmul_torch_ordinary).
    _set_ordinary = getattr(_scorch_ops, "set_planned_matmul_ordinary", None)
    if _set_ordinary is not None:
        _set_ordinary(
            _tiling_shape_is_candidate,
            csr_dense_symbol(torch.float32) == "spmm_csr_float_v2",
            csr_dense_symbol(torch.float64) == "spmm_csr_double_v2",
            _STORAGE_WRAP_GRAIN,
        )
    # And two torch CSR operands' product, at the dtypes where _torch_csr_spgemm makes it, the
    # same way (planned_matmul.h, planned_matmul_torch_spgemm).
    _set_spgemm = getattr(_scorch_ops, "set_planned_matmul_spgemm", None)
    if _set_spgemm is not None:
        _set_spgemm(
            _SPGEMM_SERVES.get(torch.float32, False),
            _SPGEMM_SERVES.get(torch.float64, False),
            _SPGEMM_SERVES.get(torch.int32, False),
            _SPGEMM_SERVES.get(torch.int64, False),
            _SPGEMM_INT64_INDICES,
        )
    # And every call made under a forced schedule goes to this function, whose generated
    # kernel honours it (planned_matmul.h, planned_matmul_schedule_forced).
    _set_schedule_force = getattr(
        _scorch_ops, "set_planned_matmul_schedule_force", None
    )
    if _set_schedule_force is not None:
        _set_schedule_force(_SCHEDULE_FORCE)
        # It reads the variable only once its own flag is set, by the first schedule_force
        # in the process, before the variable is (scheduler.FORCE_SEEN_CELL).
        _set_force_seen = _scorch_ops.note_planned_matmul_schedule_force
        _ON_FIRST_SCHEDULE_FORCE.append(_set_force_seen)
        if _SCHEDULE_FORCE_SEEN[0]:
            _set_force_seen()


# Activation codes for the fused Linear kernel (must match spmm.h
# scorch_act_scalar / scorch_apply_row_bias_act).
_ACT_CODES = {None: 0, "none": 0, "identity": 0, "relu": 1, "sigmoid": 2}
# Empty mode-indices for a dense operand (reused so the chain allocates no lists).
_EMPTY_MODE_INDICES = [[], []]


def fast_transpose(x: torch.Tensor) -> torch.Tensor:
    """Materialize the transpose of a 2D float32 tensor, fast.

    Returns a contiguous ``[C, R]`` tensor equal (bit-identical) to
    ``x.T.contiguous()`` for a ``[R, C]`` input, using the prebuilt cache-blocked
    ``scorch_transpose_2d_float`` (AVX2 8x8 / NEON 4x4 / scalar micro-tiles). This
    exists because ``torch.Tensor.T.contiguous()`` is a naive element-scatter that
    runs ~5x below memory bandwidth; once the fused ``sparse_linear`` epilogue is
    folded away, that input transpose is 40-66% of the whole forward.

    Falls back to ``x.T.contiguous()`` if the kernel is unavailable or the input
    is not a 2D float32 tensor (exactness is preserved either way).

    Parameters
    ----------
    x : torch.Tensor
        2-D ``float32`` tensor ``[R, C]`` (fast path); any other rank/dtype takes
        the exact ``x.T.contiguous()`` fallback.

    Returns
    -------
    torch.Tensor
        Contiguous ``[C, R]`` tensor, bit-identical to ``x.T.contiguous()``.

    Notes
    -----
    Uses ``torch.get_num_threads()`` when ``SCORCH_MATCH_HOST_THREADS`` is on, else
    ``-1``. Backs the input transpose inside :func:`sparse_linear`.

    A call through which a gradient is wanted (``x`` requires grad in grad mode, or has
    a forward-mode tangent) is recorded as ``_FastTranspose``, whose derivative is the
    same transpose, so ``sparse_linear``'s input gets its gradient.

    Examples
    --------
    >>> import torch
    >>> import scorch
    >>> x = torch.rand(1024, 768)
    >>> xt = scorch.fast_transpose(x)              # [768, 1024]
    >>> torch.equal(xt, x.T.contiguous())          # bit-identical
    True
    """
    import scorch_ops as _native

    fn = getattr(_native, "scorch_transpose_2d_float", None)
    if fn is None or x.dim() != 2 or x.dtype != torch.float32:
        return x.T.contiguous()
    # autograd.wanted_for written out, the flag first and forward-mode AD after: a call
    # through which no gradient is wanted makes no call into autograd.py.
    if (x.requires_grad and torch.is_grad_enabled()) or (
        _fwad._current_level >= 0 and _autograd.wanted_for(x)
    ):
        return _FastTranspose.apply(x)
    nthreads = torch.get_num_threads() if _MATCH_HOST_THREADS else -1
    return fn(x.contiguous(), nthreads)


class _FastTranspose(torch.autograd.Function):
    """``fast_transpose(x)`` recorded for autograd. The derivative of a transpose is the
    transpose of the incoming gradient, computed by ``fast_transpose`` again, which
    records itself under grad mode, so a second derivative is recorded the same way."""

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # Never a view (the fallback's contiguous() returns one when x.T is contiguous
        # already): see the note above autograd._in_layout.
        return fast_transpose(x.detach()).detach()

    @staticmethod
    def backward(ctx: Any, g: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return fast_transpose(g)

    @staticmethod
    def jvp(ctx: Any, dx: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return fast_transpose(dx)


def sparse_softmax_csr(
    crow_indices: torch.Tensor,
    values: torch.Tensor,
    scale: float = 1.0,
) -> torch.Tensor:
    """Row-wise softmax over the nonzeros of a CSR value array, ``scale`` folded in.

    Returns ``softmax(scale * values)`` computed per CSR row span
    ``[crow[i], crow[i+1])``, matching a torch scatter softmax up to float
    rounding. Backs the sparse-attention chain (SDDMM -> softmax -> SpMM): the
    prebuilt ``scorch_sparse_softmax_csr_float`` walks each row's contiguous span
    sequentially (max, exp+sum, normalize) with no scatter and no nnz-sized
    intermediates, parallel over rows on torch's warm intra-op pool.

    Falls back to a torch segment-softmax if the kernel is unavailable or the
    values are not float32 (exactness preserved either way).

    Parameters
    ----------
    crow_indices : torch.Tensor
        CSR row-pointer tensor of length ``nrows + 1``.
    values : torch.Tensor
        CSR nonzero values (``float32`` for the fast path); softmax is computed
        within each row's span.
    scale : float, default 1.0
        Multiplier folded into the logits before softmax (e.g. the attention
        scaling ``1 / sqrt(d)``).

    Returns
    -------
    torch.Tensor
        Dense tensor the same length as ``values``, normalized per CSR row span.

    Examples
    --------
    >>> import torch
    >>> import scorch
    >>> M = (torch.rand(8, 8) < 0.4).float()
    >>> csr = M.to_sparse_csr()
    >>> w = scorch.sparse_softmax_csr(csr.crow_indices(), csr.values().float())
    >>> w.shape[0] == csr.values().numel()       # sums to 1 within each row span
    True
    """
    import scorch_ops as _native

    fn = getattr(_native, "scorch_sparse_softmax_csr_float", None)
    if fn is None or values.dtype != torch.float32:
        crow = crow_indices.long()
        nrows = crow.numel() - 1
        row_ids = torch.repeat_interleave(
            torch.arange(nrows, device=values.device), crow[1:] - crow[:-1]
        )
        logits = values * scale
        row_max = torch.full((nrows,), float("-inf"), device=values.device)
        row_max.scatter_reduce_(0, row_ids, logits, reduce="amax", include_self=False)
        exp_vals = torch.exp(logits - row_max[row_ids])
        row_sum = torch.zeros(nrows, device=values.device)
        row_sum.scatter_add_(0, row_ids, exp_vals)
        return exp_vals / row_sum[row_ids]
    nthreads = torch.get_num_threads() if _MATCH_HOST_THREADS else -1
    return fn(crow_indices, values.contiguous(), float(scale), nthreads)


def _sparse_attention_fallback(
    crow_indices: torch.Tensor,
    col_indices: torch.Tensor,
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Pure-torch reference for ``sparse_attention`` (kernel unavailable/dtype).

    Reproduces the fused kernel's math head-by-head with a gather SDDMM, the
    CSR-native row softmax (``sparse_softmax_csr``, itself torch-fallback safe),
    and a torch sparse SpMM — so the API always returns the right ``[S,H,D]``.
    """
    S, H, D = int(Q.shape[0]), int(Q.shape[1]), int(Q.shape[2])
    crow = crow_indices.long()
    col = col_indices.long()
    counts = crow[1:] - crow[:-1]
    rows = torch.repeat_interleave(torch.arange(S, device=Q.device), counts)
    out = torch.empty_like(Q)
    for h in range(H):
        Qh, Kh, Vh = Q[:, h, :], K[:, h, :], V[:, h, :]
        score = (Qh[rows] * Kh[col]).sum(dim=-1)  # (nnz,)
        attn = sparse_softmax_csr(crow_indices, score.contiguous(), scale)
        attn_csr = torch.sparse_csr_tensor(crow, col, attn, (S, S))
        out[:, h, :] = torch.sparse.mm(attn_csr, Vh.contiguous())
    return out


def sparse_attention(
    crow_indices: torch.Tensor,
    col_indices: torch.Tensor,
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    scale: float = 1.0,
) -> torch.Tensor:
    """Fused sparse (masked) multi-head attention over a shared CSR mask.

    Computes, for each query row ``i`` and head ``h``,
    ``out[i,h] = sum_j softmax_j(scale * Q[i,h]·K[j,h]) * V[j,h]`` over the row's
    attended columns ``j`` (the nonzeros of the CSR mask ``crow``/``col``) — i.e.
    the same masked-attention math the dense path does with ``-inf`` fills, over
    exactly the mask's structural nonzeros.

    ``Q``/``K``/``V`` are dense ``[S, H, D]`` float32 tensors (as produced by
    ``q_proj(x).view(S, H, D)``); the CSR mask structure is passed once. Returns a
    dense ``[S, H, D]`` tensor. The prebuilt ``scorch_sparse_attention_csr_float``
    does the whole thing in ONE parallel pass over rows (inline SDDMM + two-pass
    row softmax in registers + weighted-V accumulation), batched over heads — no
    per-head kernel dispatch, no nnz-sized intermediates, no CSR round-trip. This
    is "lever 2" for the sparse-attention bench: it removes the residual per-head
    Python-dispatch/materialization overhead left after the CSR-native softmax
    (``sparse_softmax_csr``, "lever 1").

    Falls back to a pure-torch per-head reference if the kernel is unavailable or
    the tensors are not float32 (result identical up to float rounding).

    Parameters
    ----------
    crow_indices, col_indices : torch.Tensor
        Shared CSR attention **mask** structure over the ``[S, S]`` query x key
        grid (passed once, shared across heads).
    Q, K, V : torch.Tensor
        Dense ``[S, H, D]`` ``float32`` tensors (``S`` = sequence length,
        ``H`` = heads, ``D`` = head dim).
    scale : float, default 1.0
        Attention scale, e.g. ``1 / sqrt(D)``, folded into the logits.

    Returns
    -------
    torch.Tensor
        Dense ``[S, H, D]`` attention output.

    Examples
    --------
    >>> import torch
    >>> import scorch
    >>> S, H, D = 128, 4, 32
    >>> csr = (torch.rand(S, S) < 0.1).float().to_sparse_csr()
    >>> Q, K, V = torch.rand(S, H, D), torch.rand(S, H, D), torch.rand(S, H, D)
    >>> out = scorch.sparse_attention(
    ...     csr.crow_indices(), csr.col_indices(), Q, K, V, scale=1.0 / (D ** 0.5),
    ... )
    >>> tuple(out.shape)
    (128, 4, 32)
    """
    import scorch_ops as _native

    fn = getattr(_native, "scorch_sparse_attention_csr_float", None)
    if (
        fn is None
        or Q.dtype != torch.float32
        or K.dtype != torch.float32
        or V.dtype != torch.float32
    ):
        return _sparse_attention_fallback(crow_indices, col_indices, Q, K, V, scale)
    nthreads = torch.get_num_threads() if _MATCH_HOST_THREADS else -1
    return fn(
        crow_indices,
        col_indices,
        Q.contiguous(),
        K.contiguous(),
        V.contiguous(),
        float(scale),
        nthreads,
    )


def sparse_linear_fm(
    x_fm: Union[torch.Tensor, STensor],
    weight: Union[STensor, torch.Tensor],
    bias: Optional[torch.Tensor] = None,
    activation: Optional[str] = None,
) -> torch.Tensor:
    """Fused feature-major sparse Linear layer.

    Computes ``Y = act(weight @ x_fm + bias[:, None])`` in ONE prebuilt parallel
    region (SpMM + per-output-channel bias + activation), returning a dense,
    FEATURE-MAJOR result — so an autoencoder forward stays feature-major
    throughout (transpose the input once in, transpose the output once out) with
    NO per-layer transpose and NO separate torch bias/act epilogue.

    Parameters
    ----------
    x_fm : torch.Tensor or STensor
        Dense activation ``[in, batch]`` in FEATURE-MAJOR (row = input feature)
        layout. The output of a previous ``sparse_linear_fm`` (already
        ``[out, batch]``) feeds straight in.
    weight : STensor or torch.Tensor
        Sparse CSR weight ``[out, in]`` (as in ``F.linear``: ``y = x @ weight.T``).
        A torch tensor -- dense or CSR -- is converted to a CSR STensor first,
        exactly as :func:`sparse_linear` converts it; build the STensor once to
        keep the conversion out of a per-layer call.
    bias : torch.Tensor, optional
        Dense ``[out]`` bias, added per output channel.
    activation : str, optional
        ``None`` / ``"relu"`` / ``"sigmoid"``.

    Returns
    -------
    torch.Tensor
        Dense ``[out, batch]`` result (feature-major).

    Raises
    ------
    ValueError
        If ``activation`` is not one of ``None`` / ``"none"`` / ``"identity"`` /
        ``"relu"`` / ``"sigmoid"``.

    Notes
    -----
    Routes to the prebuilt ``spmm_csr_linear_fused_float`` kernel (v2's fast
    regtile/regblock inner loops + the fused epilogue), falling back to a dense
    torch reference when the kernel is unavailable or the dtype is not float32.
    Carries the same pipeline composition hints as the drop-in ``matmul`` SpMM
    (``SCORCH_MATCH_HOST_THREADS`` / ``SCORCH_ATPARALLEL_PIPELINE``). This is a
    DISTINCT entry point from :func:`matmul`, so the FEM/GCN SpMM path
    (``scorch.matmul`` -> ``spmm_csr_float_v2`` / ``spmm_csr_bias_act``) never
    reaches this kernel — the guardrail is the API boundary.

    See Also
    --------
    sparse_linear : Natural-layout ``F.linear`` drop-in built on this kernel.
    """
    import scorch_ops as _native

    if isinstance(weight, torch.Tensor) and not isinstance(weight, STensor):
        weight = STensor.from_csr(weight.to_sparse_csr(), "weight")
    act = _ACT_CODES.get(activation, None)
    if act is None:
        raise ValueError(f"unsupported activation {activation!r}")

    # Extract the dense feature-major operand [in, batch] WITHOUT wrapping it in an
    # STensor — a dense operand needs only its shape, empty mode-indices, and flat
    # values, so the full STensor.from_torch construction (TensorFormat/TensorIndex
    # objects) is pure per-layer overhead in a chain. A dense STensor input is
    # unwrapped the same cheap way.
    if isinstance(x_fm, torch.Tensor):
        x_vals = x_fm.reshape(-1)
        in_dim = int(x_fm.shape[0])
        batch = int(x_fm.shape[1])
    else:
        x_vals = x_fm.values
        in_dim = int(x_fm.shape[0])
        batch = int(x_fm.shape[1])
    out_dim = int(weight.shape[0])
    result_shape = [out_dim, batch]

    fn = getattr(_native, "spmm_csr_linear_fused_float", None)
    if (
        fn is None
        or weight._raw_values.dtype != torch.float32
        or x_vals.dtype != torch.float32
    ):
        # Fallback (kernel unavailable / unsupported dtype): dense reference in
        # feature-major layout so the op still produces the right result.  Torch
        # operations throughout, so torch records its gradient: ``to_torch`` and
        # ``values`` read the tensor autograd tracks for an STensor's values.
        w_dense = weight.to_torch()
        x_dense = x_vals.reshape(in_dim, batch)
        y = torch.matmul(w_dense, x_dense)
        if bias is not None:
            # In the product's dtype, so the result keeps the operands' dtype (float64;
            # float16 and bfloat16 weights are refused earlier), as F.linear's does.
            # .to() only on a dtype change: a same-dtype .to() returns the bias itself
            # but still goes through the dispatcher on every call.
            if bias.dtype != y.dtype:
                bias = bias.to(y.dtype)
            y = y + bias.view(-1, 1)
        if act == 1:
            return torch.relu(y)
        if act == 2:
            return torch.sigmoid(y)
        return y

    # The kernel's float32 bias, made only on the kernel's path: before step 13 the
    # fallback above added this copy, so every float64 call carried a float32-rounded
    # bias (L16: 3e6-5e6 times the float64 error bound on the M5).
    if bias is None:
        bias_t = torch.zeros(out_dim, dtype=torch.float32)
    else:
        bias_t = bias if bias.dtype == torch.float32 else bias.to(torch.float32)
        bias_t = bias_t.contiguous()

    nthreads = -1
    atparallel = False
    if _MATCH_HOST_THREADS:
        nthreads = torch.get_num_threads()
        atparallel = _ATPARALLEL_PIPELINE

    # autograd.wanted_by_any written out, as _einsum_compiler_path has it: a call through
    # which no gradient is wanted makes no call into autograd.py.
    grad_wanted = False
    if torch.is_grad_enabled():
        for operand in (x_fm, weight, bias):
            if (isinstance(operand, torch.Tensor) and operand.requires_grad) or (
                type(operand) is STensor and operand._metadata.requires_grad
            ):
                grad_wanted = True
                break
    if not grad_wanted and _fwad._current_level >= 0:
        grad_wanted = _autograd.wanted_by_any((x_fm, weight, bias))
    if grad_wanted:

        def kernel(
            values: torch.Tensor, x: torch.Tensor, bias_values: torch.Tensor
        ) -> torch.Tensor:
            # The call below, on the values, activation and bias autograd hands it.
            result_cpp = fn(
                result_shape,
                list(weight.shape),
                weight._native_mode_indices(),
                values,
                [in_dim, batch],
                _EMPTY_MODE_INDICES,
                x,
                bias_values,
                act,
                256,
                nthreads,
                atparallel,
            )
            # The [out, batch] product as the kernel allocated it, as below.
            return result_cpp.storage.value

        # Recorded by autograd.linear_fm on the same kernel, so the layer returns the
        # same values whether or not a gradient is wanted through it.
        return _autograd.linear_fm(
            weight, x_vals.reshape(in_dim, batch), bias_t, act, kernel
        )

    result_cpp = fn(
        result_shape,
        list(weight.shape),
        weight._native_mode_indices(),
        # The detached values: native code never reads autograd's tensor.
        weight._raw_values,
        [in_dim, batch],
        _EMPTY_MODE_INDICES,
        x_vals if isinstance(x_fm, torch.Tensor) else x_fm._raw_values,
        bias_t,
        act,
        256,
        nthreads,
        atparallel,
    )
    # The kernel allocates the [out, batch] product in that shape (spmm.h): it is
    # returned as it is, not as a view of a flat buffer.
    return result_cpp.storage.value


def sparse_linear(
    x: torch.Tensor,
    weight: Union[STensor, torch.Tensor],
    bias: Optional[torch.Tensor] = None,
    activation: Optional[str] = None,
) -> torch.Tensor:
    """Drop-in ``F.linear`` with a sparse weight, fused with bias + activation.

    Computes ``act(x @ weight.T + bias)`` for a NATURAL-layout activation
    ``x`` ``[batch, in]``, returning ``[batch, out]`` — the exact signature of
    ``torch.nn.functional.linear`` (plus an optional fused activation). This is
    the transparent per-call entry point: replace ``act(F.linear(x, W, b))`` with
    ``scorch.sparse_linear(x, W, b, "relu")`` and the SpMM + bias + activation run
    in ONE prebuilt parallel region.

    ``weight`` may be a sparse CSR ``STensor`` ``[out, in]`` (preferred — build it
    once and reuse) or a dense ``torch.Tensor`` ``[out, in]`` (converted to CSR per
    call; pass a pre-built STensor in a hot loop to avoid the conversion).
    ``activation`` -- ``None`` / ``"relu"`` / ``"sigmoid"``.

    Internally transposes ``x`` into feature-major ``[in, batch]`` (via the fast
    cache-blocked ``fast_transpose`` — NOT torch's cache-hostile
    ``x.T.contiguous()``, which would dominate the fused forward), runs the fused
    feature-major kernel (``sparse_linear_fm``), and returns the result's
    transpose as a lazy view — the SAME transpose pattern
    ``torch.sparse.mm(W, x.T).T`` MKL already pays, minus MKL's separate bias/act
    epilogue. For a multi-layer chain, ``sparse_linear_fm`` (staying feature-major
    end to end) avoids the intermediate transposes.

    Parameters
    ----------
    x : torch.Tensor
        Dense ``[batch, in]`` ``float32`` activation in natural layout.
    weight : STensor or torch.Tensor
        Sparse CSR ``STensor`` ``[out, in]`` (**preferred** — build it once and
        reuse), or a dense ``torch.Tensor`` ``[out, in]`` which is converted to CSR
        *per call*; pass a pre-built STensor in a hot loop to avoid the conversion.
    bias : torch.Tensor, optional
        Dense ``[out]`` bias.
    activation : str, optional
        ``None`` / ``"relu"`` / ``"sigmoid"``.

    Returns
    -------
    torch.Tensor
        Dense ``[batch, out]`` result.

    See Also
    --------
    sparse_linear_fm : Feature-major variant that avoids per-layer transposes.
    fast_transpose : The cache-blocked transpose used on the input.

    Examples
    --------
    >>> import torch
    >>> import scorch
    >>> W_dense = (torch.rand(256, 512) < 0.1).float()
    >>> W = scorch.from_csr(W_dense.to_sparse_csr(), "W")   # build CSR STensor once
    >>> b = torch.rand(256)
    >>> x = torch.rand(64, 512)
    >>> y = scorch.sparse_linear(x, W, b, activation="relu")   # [64, 256]
    >>> torch.allclose(y, torch.relu(x @ W_dense.T + b), atol=1e-3, rtol=1e-3)
    True
    """
    x_fm = fast_transpose(x)
    y_fm = sparse_linear_fm(x_fm, weight, bias, activation)
    return y_fm.T


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2b (codegen-parity): register-block dual-path kernel emission.
#
# The free-dim register-block transform (holding the output tile in a stack-local
# accumulator across a row's nonzeros) WINS for narrow N (<=~3) but REGRESSES for
# wide N (re-traversal of the sparse row per free-dim tile). Because ONE
# format-keyed kernel serves all N, the fix is a RUNTIME branch inside the kernel on
# the free-dim size: register-block nest for small N (single tile, no re-traversal),
# the byte-identical baseline memory-destination nest for large N. This preserves
# wide-k parity by construction (the regblock arm cannot fire for large N) — the same
# discipline as the spmm.h nfloor / E-core-recruit gates.
#
# We build it by lowering the SAME (unscheduled) CIN twice — regblock forced OFF and
# ON — and splicing the two top-level compute loops under one llir.IfThenElse. See
# codegen-parity/04-phase2-results.md for the route evaluation (LLIR if-stitch chosen
# over an N-bucketed cache key).
#
# ROLLOUT: dual-path emission now defaults ON via `_regblock_dual_active()` after
# validation on M5 (2026-07-09) and x86/redwood (2026-07-10) showed it is
# neutral-or-better across the whole grid — narrow-k register-block WIN (x86 rb/base
# geomean 0.89), wide-k byte-identical else-arm PARITY. This is DECOUPLED from the
# scheduler's `SCORCH_REGBLOCK` env (which stays default OFF, so a direct
# `Scheduler.auto_schedule` caller — and the two schedule-shape tests — is unchanged):
# `_build_regblock_dual_path` derives two LOCAL immutable option snapshots, so it
# needs nothing from the global env. Escape hatch: `SCORCH_REGBLOCK_DUAL=0` restores
# the pre-flip single baseline path (also byte-identical to before the whole feature).
# ─────────────────────────────────────────────────────────────────────────────


def _regblock_dual_active(compile_options: CompileOptions) -> bool:
    """Whether `einsum` emits the register-block dual-path kernel for the qualifying
    pattern (dense output, sparse contraction, a tileable free dim).

    Default TRUE post-validation (see the module comment above). Set
    ``SCORCH_REGBLOCK_DUAL=0`` to restore the pre-flip single baseline path. This is
    intentionally SEPARATE from the scheduler's ``SCORCH_REGBLOCK`` policy
    (default OFF): the dual builder configures its two lowerings explicitly, so flipping the
    per-op default here leaves every direct ``auto_schedule`` caller untouched. When the
    dual-path does not apply (non-qualifying pattern → `_build_regblock_dual_path`
    returns None), einsum falls back to the byte-identical baseline single path.
    """
    return compile_options.regblock_dual


def _find_top_compute_loop(
    body: List[llir.Stmt],
) -> Tuple[Optional[int], Optional[llir.ForLoop]]:
    """Locate the single top-level compute loop in a lowered function body.

    Prefer the omp-parallel-for row loop; fall back to a lone top-level ForLoop.
    Returns (None, None) when the shape is ambiguous (0 or >1 candidates) so the
    caller can safely decline the dual-path stitch.
    """
    for_loops = [(i, s) for i, s in enumerate(body) if isinstance(s, llir.ForLoop)]
    if not for_loops:
        return None, None
    parallel = [(i, s) for i, s in for_loops if getattr(s, "omp_parallel_for", False)]
    if len(parallel) == 1:
        return parallel[0]
    if len(for_loops) == 1:
        return for_loops[0]
    return None, None


def _regblock_free_size_expr(row_loop: llir.ForLoop) -> Optional[llir.Expr]:
    """Extract the free-dim size expression from the register-block row loop.

    The tiled nest contains a `for (k_out = 0; k_out < <free_size>; k_out += kTile)`
    loop; its bound (`cond.right`) IS the free-dim extent as the lowerer computed it
    (e.g. `B1_size`). Using it avoids hard-coding a variable name. Returns None if no
    `*_out` tile loop is found.
    """

    def _walk(stmts: List[llir.Stmt]) -> Optional[llir.Expr]:
        for s in stmts:
            if isinstance(s, llir.ForLoop):
                init = getattr(s, "init", None)
                var = getattr(init, "var", None) if init is not None else None
                name = getattr(var, "name", "") or ""
                if name.endswith("_out"):
                    cond = getattr(s, "cond", None)
                    right = getattr(cond, "right", None) if cond is not None else None
                    if right is not None:
                        return right
                found = _walk(getattr(s, "body", None) or [])
                if found is not None:
                    return found
        return None

    return _walk(getattr(row_loop, "body", None) or [])


# The statements that introduce a name: the only kind `_stitch_regblock_dual_path`
# moves from the baseline function's scope into the else branch ahead of the loop.
_CARRIED_DECLARATION_TYPES = (
    llir.VarInit,
    llir.VarDecl,
    llir.DirectInit,
    llir.FixedStackArrayDecl,
)
_LAYOUT_ONLY_TYPES = (llir.BlankLine, llir.Comment)
_CPP_IDENTIFIER = re.compile(r"[A-Za-z_]\w*")


def _is_compile_time_constant(stmt: llir.Stmt) -> bool:
    """Whether `stmt` declares a `constexpr int`, the form of a tile width."""

    return type(stmt) is llir.VarInit and stmt.var.type is llir.DataType.CONSTEXPR_INT


def _carried_declaration_name(stmt: llir.Stmt) -> Optional[str]:
    """The name a carried statement declares, or None when it is not a declaration."""

    if type(stmt) not in _CARRIED_DECLARATION_TYPES:
        return None
    if type(stmt) is llir.FixedStackArrayDecl:
        name = stmt.name
    else:
        name = cast(llir.VarInit, stmt).var.name
    return name if type(name) is str and name.isidentifier() else None


def _baseline_statements_for_else_branch(
    rb_body: List[llir.Stmt],
    idx_rb: int,
    base_body: List[llir.Stmt],
    idx_base: int,
    compile_options: Optional[CompileOptions] = None,
) -> Optional[Tuple[List[llir.Stmt], List[llir.Stmt]]]:
    """The baseline function's own function-scope statements, to go around its loop.

    The stitched kernel keeps the register-block function's statements and takes
    the loop from the baseline function, so anything else the baseline function
    declares at function scope for that loop has to be carried with it.  A
    baseline nest whose outermost loop is the tile loop over a reduced index has
    no loop a parallel region can own, so its dense workspace is declared
    serially ahead of the loop (`CINLowerer._place_unplaced_workspace_clusters`).
    `einsum('ab,bc->ac', A dd, B ds, format='dd')` is such a nest: the stitch
    took the loop alone, the else branch used `wksp` with no declaration, and the
    kernel did not build at any extent.

    Returns `(before, after)`, the statements to emit inside the else branch
    ahead of and behind the baseline loop, where only that loop sees them.
    Returns None when the two-branch kernel cannot be formed with every name the
    baseline loop reads declared as the baseline function declares it:

    - each baseline statement is either emitted identically by the
      register-block function on the same side of its loop and in the same order
      (the stitched kernel then already has it at function scope), or carried;
    - a compile-time constant both functions declare under one name is the
      exception: it is not carried, and the else branch reads the register-block
      function's value, as it always has.  The two nests name their tile widths
      alike (`kTile_k`, 8 for the register-block nest and 32 for the baseline
      one), so every stitched kernel whose baseline nest is tiled runs its else
      branch 8 wide.  Carrying the baseline's own width would change that for
      kernels that build today -- a schedule change, to be judged by a runtime
      measurement and not folded into this repair;
    - the carried statements are the ones next to the loop: carrying a statement
      past one that stays would reorder the two;
    - a statement carried ahead of the loop is a declaration, and no statement
      that stays mentions the name it declares;
    - a statement carried behind the loop mentions a carried name: it closes
      what a carried declaration opened.

    "Identical" is by emitted C++.  The same code generator renders the stitched
    kernel, so two statements with one rendering are one statement in it: the
    comparison renders under the compilation's own `compile_options`, like every
    other stage of it.  Comments and blank lines are not compared.
    """

    lowerer = LLIRLowerer(compile_options=compile_options)
    rb_constants = {
        cast(llir.VarInit, statement).var.name
        for statement in rb_body[:idx_rb]
        if _is_compile_time_constant(statement)
    }

    def stays_out(statement: llir.Stmt) -> bool:
        """Not compared: layout, or a constant the first function names too."""

        return type(statement) in _LAYOUT_ONLY_TYPES or (
            _is_compile_time_constant(statement)
            and cast(llir.VarInit, statement).var.name in rb_constants
        )

    def rendered(statements: List[llir.Stmt]) -> List[Tuple[int, str]]:
        return [
            (position, lowerer.lower_llir(statement).strip())
            for position, statement in enumerate(statements)
            if not stays_out(statement)
        ]

    def without_counterpart(
        base_side: List[llir.Stmt], rb_side: List[llir.Stmt]
    ) -> List[int]:
        rb_texts = [text for _, text in rendered(rb_side)]
        cursor, missing = 0, []
        for position, text in rendered(base_side):
            try:
                cursor = rb_texts.index(text, cursor) + 1
            except ValueError:
                missing.append(position)
        return missing

    base_before = list(base_body[:idx_base])
    base_after = list(base_body[idx_base + 1 :])
    missing_before = without_counterpart(base_before, list(rb_body[:idx_rb]))
    missing_after = without_counterpart(base_after, list(rb_body[idx_rb + 1 :]))

    before: List[llir.Stmt] = []
    if missing_before:
        start = missing_before[0]
        next_to_loop = [p for p, _ in rendered(base_before) if p >= start]
        if next_to_loop != missing_before:
            return None
        before = base_before[start:]
    after: List[llir.Stmt] = []
    if missing_after:
        stop = missing_after[-1] + 1
        next_to_loop = [p for p, _ in rendered(base_after) if p < stop]
        if next_to_loop != missing_after:
            return None
        after = base_after[:stop]

    carried_names = set()
    for statement in before:
        if stays_out(statement):
            # A shared constant between the carried declarations and the loop
            # would be redeclared by the carry; the stitch is not formed.
            if type(statement) not in _LAYOUT_ONLY_TYPES:
                return None
            continue
        name = _carried_declaration_name(statement)
        if name is None:
            return None
        carried_names.add(name)

    def mentions_carried_name(statement: llir.Stmt) -> bool:
        text = lowerer.lower_llir(statement)
        return not carried_names.isdisjoint(_CPP_IDENTIFIER.findall(text))

    staying = base_before[: len(base_before) - len(before)] + base_after[len(after) :]
    for statement in staying:
        if type(statement) in _LAYOUT_ONLY_TYPES:
            continue
        if mentions_carried_name(statement):
            return None
    for statement in after:
        if type(statement) in _LAYOUT_ONLY_TYPES:
            continue
        if not mentions_carried_name(statement):
            return None
    return before, after


def _stitch_regblock_dual_path(
    fn_rb: llir.Function,
    fn_base: llir.Function,
    cutoff_n: int,
    compile_options: Optional[CompileOptions] = None,
) -> Optional[llir.Function]:
    """Splice the register-block and baseline compute loops under one runtime branch.

    Keeps `fn_rb`'s prologue and epilogue, replacing its single top-level compute
    loop with `if (free_size <= cutoff_n) { <regblock loop> } else { <baseline
    loop> }`.  The else branch also holds the function-scope statements only the
    baseline function has around its loop -- a serially placed dense workspace --
    so that loop reads every name as its own function declared it
    (`_baseline_statements_for_else_branch`).

    Returns None (decline) if either function's compute region is ambiguous, the
    free-dim size expression can't be found, or the baseline function has a
    function-scope statement the stitched kernel would neither already have nor
    be able to carry.  The caller then emits the baseline nest alone.
    """
    idx_rb, loop_rb = _find_top_compute_loop(fn_rb.body)
    idx_base, loop_base = _find_top_compute_loop(fn_base.body)
    if loop_rb is None or loop_base is None or idx_rb is None or idx_base is None:
        return None
    free_expr = _regblock_free_size_expr(loop_rb)
    if free_expr is None:
        return None
    carried = _baseline_statements_for_else_branch(
        fn_rb.body, idx_rb, fn_base.body, idx_base, compile_options
    )
    if carried is None:
        return None
    before, after = carried
    cond = llir.BinOp(
        op="<=",
        left=free_expr,
        right=llir.Literal(value=cutoff_n, data_type=llir.DataType.INT64),
    )
    branch = llir.IfThenElse(
        cond=cond, then_body=[loop_rb], else_body=[*before, loop_base, *after]
    )
    new_body = list(fn_rb.body)
    new_body[idx_rb] = branch
    fn_rb.body = new_body
    return fn_rb


def _build_regblock_dual_path(
    cin_unscheduled: IndexStmt,
    post_ops: Any,
    compile_options: Optional[CompileOptions] = None,
    compilation_context: Optional[CompilationContext] = None,
) -> Optional[Tuple[llir.Function, str]]:
    """Build the Phase 2b dual-path kernel from an unscheduled CIN.

    Schedules + lowers the CIN twice (regblock forced OFF -> baseline nest, ON ->
    tiled nest) and stitches a runtime free-dim branch. Returns (Function, cache_key)
    or None when the register-block path doesn't apply (schedules identical) or the
    stitch can't be formed safely — the caller then falls back to the baseline path.
    """
    if compile_options is None:
        compile_options = CompileOptions.from_environment()
    elif type(compile_options) is not CompileOptions:
        raise ScorchTypeError(
            "compile_options must be a CompileOptions instance",
            code="internal_argument_type_invalid",
        )

    if not Scheduler._has_dense_output(cin_unscheduled):
        return None

    cin_base = Scheduler._auto_schedule_regblock_arm(
        cin_unscheduled,
        enabled=False,
        compile_options=compile_options,
        compilation_context=compilation_context,
    )
    cin_rb = Scheduler._auto_schedule_regblock_arm(
        cin_unscheduled,
        enabled=True,
        compile_options=compile_options,
        compilation_context=compilation_context,
    )
    if str(cin_base) == str(cin_rb):
        # Register-block didn't change the schedule (pattern doesn't qualify) ->
        # nothing to branch on; let the caller use the plain baseline path.
        return None

    fn_base = CINLowerer(
        post_ops=post_ops,
        compile_options=compile_options,
        compilation_context=compilation_context,
    )._lower_owned_IndexStmt(cin_base)
    fn_rb = CINLowerer(
        post_ops=post_ops,
        compile_options=compile_options,
        compilation_context=compilation_context,
    )._lower_owned_IndexStmt(cin_rb)
    if not (isinstance(fn_base, llir.Function) and isinstance(fn_rb, llir.Function)):
        return None

    stitch_token = (
        compilation_context.begin_stage(
            CompilerStageId.CIN_LOWERING,
            compile_options=compile_options,
        )
        if compilation_context is not None
        else None
    )
    try:
        stitched = _stitch_regblock_dual_path(
            fn_rb,
            fn_base,
            compile_options.scheduler.regblock_max_n,
            compile_options,
        )
    except Exception:
        if stitch_token is not None and compilation_context is not None:
            compilation_context.fail_stage(stitch_token)
        raise
    if stitch_token is not None and compilation_context is not None:
        if stitched is None:
            compilation_context.cancel_stage(stitch_token)
        else:
            compilation_context.complete_stage(stitch_token)
    if stitched is None:
        return None
    return (
        stitched,
        _codegen_kernel_cache_key(
            cin_rb,
            post_ops,
            None,
            compile_options=compile_options,
        )
        + "|rbdual",
    )


@contextmanager
def _trusted_compiler_built_cin(
    cin_stmt: IndexStmt,
    compile_options: CompileOptions,
    compilation_context: CompilationContext,
) -> Iterator[None]:
    """Prove one compiler-built CIN root and scope its structural receipt."""

    receipt_token = compilation_context.begin_stage(
        CompilerStageId.CIN_NORMALIZATION_AND_VERIFICATION,
        compile_options=compile_options,
    )
    try:
        verify_cin_structure(cin_stmt)
    except Exception:
        compilation_context.fail_stage(receipt_token)
        raise
    compilation_context.cancel_stage(receipt_token)
    with _trusted_normalized_cin(cin_stmt):
        yield


def _without_mixed_coordinate_levels(
    output_level_formats: List[LevelFormat],
) -> List[LevelFormat]:
    """Replace coordinate levels by compressed ones when the output would mix them.

    Per-level inference copies each output index's level type from the inputs, so a
    product of a compressed operand and a coordinate one infers a mixed output -- CSR x
    COO gives dense rows over coordinate columns. Code generation cannot lower that: a
    coordinate level under a dense or compressed parent needs per-parent grouping that
    the iterator model does not represent, and `CINLowerer` refuses such a tensor
    (`unsupported_mixed_coordinate_hierarchy`). The performance line had no refusal and
    returned the product in that format with the row grouping missing, so the result
    could not be interpreted -- its own `to_dense` failed to compile.

    A compressed level stores the same coordinates and composes with dense and
    compressed parents, so substituting it keeps every level sparse that inference made
    sparse and yields CSR or doubly-compressed CSR. An output that is entirely
    coordinate (COO) is coherent and kept. This only changes outputs that would
    otherwise be refused, and an explicitly requested format is never passed here.
    """
    level_types = [lf.get_level_type() for lf in output_level_formats]
    if LevelType.COORDINATE not in level_types or all(
        level_type is LevelType.COORDINATE for level_type in level_types
    ):
        return output_level_formats
    return [
        (
            LevelFormat(LevelType.COMPRESSED)
            if level_format.get_level_type() is LevelType.COORDINATE
            else level_format
        )
        for level_format in output_level_formats
    ]


def _evaluate_native(kernel: Callable[..., Any], *args: Any) -> Any:
    """Call a native kernel; a refusal its checks mark with a code (a repeated COO
    coordinate, an index past the 32-bit position bound) is raised as its structured
    error (``raise_if_native_coded``), anything else unchanged.

    A separate function so that ``_einsum_owned``, whose three native calls need the
    mapping, does not grow past its complexity (the project bound is 43, .flake8).  Its
    dispatch-cache hit, the path of every warm call, maps inline instead, so that path
    makes no Python call for the mapping.
    """

    try:
        return kernel(*args)
    except RuntimeError as error:
        raise_if_native_coded(error)
        raise


def _refuse_einsum_result_past_bound(
    bound: Optional[EntryBound],
    checked: FrozenSet[int],
    tensors: Sequence[Any],
    compile_only: Optional[bool],
) -> Optional[EntryBound]:
    """An einsum result's bound (patch 0056) with the levels its kernel checks itself,
    after refusing the call when a level assembled one entry at a time may pass the
    position bound -- read from the shapes and stored counts, and where those pass it,
    from the operands' coordinates (result_bound, J).  ``None`` (a shape that cannot
    pass) is returned as it is.  A ``compile_only`` call has no stored counts and is not
    refused.

    A separate function so that ``_einsum_owned`` keeps within its complexity.
    """

    if bound is None:
        return None
    bound = bound.with_checked(checked)
    if not compile_only:
        bound.refuse_past_limit(_stored_counts(tensors), tensors)
    return bound


def _stored_counts(tensors: Sequence[Any]) -> List[int]:
    """Each operand's stored entries, the counts a result bound reads (patch 0056)."""

    return [tensor._raw_values.numel() for tensor in tensors]


def _validated_einsum_request(
    expression: Any, tensors: Sequence[Any], kwargs: Mapping[str, Any]
) -> Tuple[
    List[str],
    str,
    Any,
    Any,
    Optional[TensorFormat],
    Optional[List[int]],
    dict,
]:
    """The checks every einsum request passes before anything is computed.

    ``tensors`` are STensors or TensorSpecs (torch operands already converted).  Returns
    ``(input_groups, result_expression, input_index_strs, result_index_strs,
    requested_output_format, output_mode_order, logical_index_sizes)``.  Shared by the
    compiler path (``_einsum_owned``) and the all-dense path (``_dense_einsum``), so a
    mistake is refused with the same code whichever path the operands take.
    """

    if not isinstance(expression, str) or expression.count("->") != 1:
        raise CompileSpecError(
            "einsum expression must contain an explicit '->'",
            code="einsum_implicit_output_unsupported",
        )
    input_expression, result_expression = expression.split("->")
    input_groups = input_expression.split(",")
    if len(input_groups) != len(tensors):
        raise EinsumOperandCountError(
            f"einsum expression expects {len(input_groups)} operands, got {len(tensors)}"
        )
    for tensor in tensors:
        if not isinstance(tensor, (STensor, TensorSpec)):
            # Built once an operand fails: None first, then every operand's type.
            if any(operand is None for operand in tensors):
                raise TensorTypeError(
                    "einsum operands cannot be None",
                    code="einsum_operand_type_invalid",
                )
            invalid = [
                type(operand).__name__
                for operand in tensors
                if not isinstance(operand, (STensor, TensorSpec))
            ]
            raise TensorTypeError(
                "einsum operands must be torch.Tensor, STensor, or TensorSpec; got "
                + ", ".join(invalid),
                code="einsum_operand_type_invalid",
            )
    input_index_strs, result_index_strs = _validated_labels(
        expression, input_groups, result_expression
    )
    requested_output_format = _require_output_format_rank(
        kwargs.get("format"), len(result_index_strs), "einsum"
    )
    if requested_output_format is not None:
        # ``level_types`` is the format's own tuple; ``in`` compares as ``==`` does.
        if LevelType.SINGLETON in requested_output_format.level_types:
            raise CompileSpecError(
                "singleton output levels are not supported by the compiler",
                code="singleton_level_unsupported",
            )
    for operand, (labels, tensor) in enumerate(zip(input_index_strs, tensors)):
        if len(labels) != tensor.dim():
            raise OperandRankMismatchError(
                f"einsum operand {operand} has {len(labels)} labels but tensor "
                f"rank {tensor.dim()}"
            )
    output_mode_order = kwargs.get("output_mode_order")
    if output_mode_order is not None:
        output_mode_order = _checked_output_mode_order(
            output_mode_order, len(result_index_strs)
        )
    logical_index_sizes = _logical_index_sizes(input_index_strs, tensors)
    return (
        input_groups,
        result_expression,
        input_index_strs,
        result_index_strs,
        requested_output_format,
        output_mode_order,
        logical_index_sizes,
    )


def _einsum_owned(
    expression: str,
    *tensors: Optional[Union[torch.Tensor, STensor, TensorSpec]],
    compile_only: Optional[bool] = False,
    options: _DeferredOptions,
    compilation: _DeferredCompilation,
    **kwargs: Any,
) -> Union[STensor, TensorSpec]:
    """Compile and evaluate a numpy-style einsum over sparse/dense operands.

    The generic front door to the sparse-tensor compiler. Given an index-notation
    expression it builds Compiler Index Notation (CIN), schedules and lowers it
    through LLIR to a C++ kernel, JIT-compiles (memoized), and executes — covering
    the binary contraction / elementwise / SDDMM patterns the library supports.
    :func:`matmul` falls through to this for anything the prebuilt kernels miss.

    Parameters
    ----------
    expression : str
        A numpy-style einsum string that **must contain an explicit** ``"->"``
        output, e.g. ``"ik,kj->ij"`` (matmul / SpMM / SpGEMM), ``"ij,ij->ij"``
        (elementwise multiply), or ``"ij,ik,jk->ij"`` (SDDMM). One character per
        index. An implicit-output form (no ``"->"``) is **not** supported and
        raises ``CompileSpecError``. A term may be empty: an empty result term
        sums over every index into a rank-0 result (``"ij->"``, ``"i,i->"``), and
        an empty operand term names a rank-0 operand (``",i->i"``). An expression
        with no index in any term (``"->"``) raises ``EinsumScalarResultError``.
    *tensors : torch.Tensor, STensor, or TensorSpec
        The operands, one per comma-separated input group. ``torch.Tensor`` inputs
        are auto-converted with ``STensor.from_torch``. ``TensorSpec`` operands
        are accepted only for compile-only calls.
    compile_only : bool, default False
        When ``True``, compile and cache the kernel without executing and return
        an immutable output ``TensorSpec`` (used by :func:`precompile_kernels`).
    **kwargs
        format : str or list or TensorFormat, optional
            Requested output format. If omitted, the output format is inferred
            per-level (see Notes).
        output_mode_order : list of int, optional
            Permutation applied to the output modes.
        schedule : Schedule, optional
            Exact loop order and affine tiling plan for the JIT scheduler. The
            complete plan participates in both compiler cache keys.
        time_dict : dict, optional
            If given, ``time_dict["eval_time"]`` is set to the kernel wall-clock
            time in seconds.
        _post_ops, _post_ops_tensors : advanced/internal
            Fused post-op chain (bias/scale/activation) and its extra operand
            tensors, used by ``scorch.compile``. When ``_post_ops`` is set the fast
            dispatch cache and the register-block dual-path are bypassed.

    Returns
    -------
    STensor
        The result (dense or sparse per the output format). Callers such as
        :func:`matmul` convert a dense-format result back to a ``torch.Tensor``.
        A rank-0 result is an ``STensor`` of shape ``()`` and format ``""``
        holding one value.

    Raises
    ------
    CompileSpecError
        If the expression or requested output contract is malformed.

    Notes
    -----
    **Output-format inference** (when ``format=`` is omitted) is decided per level:
    *sparse * anything = sparse; dense + anything = dense; otherwise compressed*,
    with ties preferring compressed over coordinate. A sparse level may not precede
    a dense level (a preceding sparse level is forced dense). As a special case,
    an SDDMM pattern (sparse output with reduction vars plus an input mirroring the
    output sparsity) yields an all-COO output to enable the scalar-accum codegen
    path. An output that would still mix coordinate levels with dense or compressed
    ones takes compressed levels in their place, since such a tensor cannot be lowered
    (see ``_without_mixed_coordinate_levels``) — and there is a dedicated prebuilt fast path inside ``einsum`` for exactly
    ``"ij,ik,jk->ij"`` with ``S`` (COO), ``A`` (dense), ``B`` (dense) float32.

    For the qualifying pattern (dense output, sparse contraction, tileable free dim,
    ``_post_ops is None``) ``einsum`` emits ONE kernel that branches at runtime on
    the free-dim size: a register-block nest for small ``N`` and the byte-identical
    baseline nest for large ``N``. Set ``SCORCH_REGBLOCK_DUAL=0`` to restore the
    single baseline path (distinct from the scheduler's ``SCORCH_REGBLOCK``, default
    off).

    Historically fragile format / mode-order / loop-order combinations (transposed
    operands, dense-STensor products, broadcast vectors) are covered by the green
    regression tests in ``tests/test_scorch/test_known_compiler_gaps*.py`` — despite
    the name, those are *supported* cases. Unsupported patterns surface as compiler
    exceptions rather than documented errors.

    Examples
    --------
    >>> import torch
    >>> import scorch
    >>> A = scorch.from_torch((torch.rand(64, 32) < 0.2).float(), "A").to_sparse("ds")
    >>> B = torch.rand(32, 48)
    >>> C = scorch.einsum("ik,kj->ij", A, B, format="dd")   # STensor, dense format
    >>> torch.allclose(C.to_torch(), A.to_torch() @ B, atol=1e-3, rtol=1e-3)
    True

    SDDMM (sampled ``A @ B.T`` on the nonzeros of a COO mask ``S``):

    >>> S = scorch.from_coo(torch.rand(50, 50).to_sparse_coo(), "S")
    >>> Ad = torch.rand(50, 16)
    >>> Bd = torch.rand(50, 16)
    >>> Sd = scorch.einsum("ij,ik,jk->ij", S, Ad, Bd)       # STensor (COO)
    """
    # e.g. expression might be e.g. "i,i->i" and "ij,ij->ij" for
    # elementwise multiplication or "ik,kj->ij" for matrix multiplication

    # # If any of the tensors have the same name, rename them
    # tensor_names = [tensor.name for tensor in tensors]
    # tensor_name_counts = {name: tensor_names.count(name) for name in tensor_names}
    # for i, tensor in enumerate(tensors):
    #     if tensor_name_counts[tensor.name] > 1:
    #         tensor.name = tensor.name + str(i)

    # Convert all torch.Tensor inputs to STensor early
    tensors = tuple(
        STensor.from_torch(t) if isinstance(t, torch.Tensor) else t for t in tensors
    )
    (
        input_groups,
        result_expression,
        input_index_strs,
        result_index_strs,
        requested_output_format,
        output_mode_order,
        logical_index_sizes,
    ) = _validated_einsum_request(expression, tensors, kwargs)
    if not compile_only and any(isinstance(tensor, TensorSpec) for tensor in tensors):
        raise CompileSpecError(
            "TensorSpec has no runtime payload; pass compile_only=True or use STensor",
            code="tensor_spec_without_payload",
        )
    effective_schedule = options.requested_schedule

    # ── Prebuilt SDDMM dispatch ────────────────────────────────────────
    # Pattern: 'ij,ik,jk->ij' with S(COO), A(dense), B(dense).  The prebuilt kernel
    # computes the bare product and has no post-operations, so a call that carries
    # them (``_post_ops``, ``_post_ops_tensors``) keeps the compiler path, which
    # applies them; taking this route returned the product without them.
    if (
        effective_schedule is None
        and not compile_only
        and "_post_ops" not in kwargs
        and "_post_ops_tensors" not in kwargs
        and expression == "ij,ik,jk->ij"
        and len(tensors) == 3
        and cast(STensor, tensors[0])._raw_values.dtype == torch.float32
        and str(tensors[0].format) == "o,o"
        and str(tensors[1].format) == "d,d"
        and str(tensors[2].format) == "d,d"
        and all(
            tuple(tensor.mode_order) == tuple(range(tensor.dim())) for tensor in tensors
        )
    ):
        import scorch_ops as _ops

        _sddmm_fn = getattr(_ops, "sddmm_coo_float_prebuilt", None)
        if _sddmm_fn is not None:
            compilation.cancel_frontend()
            S, A, B = tensors
            result_shape = S.shape
            result_cpp = _evaluate_native(
                _sddmm_fn,
                result_shape,
                S.shape,
                S._native_mode_indices(),
                cast(STensor, S)._raw_values,
                A.shape,
                A._native_mode_indices(),
                cast(STensor, A)._raw_values,
                B.shape,
                B._native_mode_indices(),
                cast(STensor, B)._raw_values,
            )
            return _wrap_generated_result(
                shape=result_shape,
                tensor_format=S.format,
                result_cpp=result_cpp,
                mode_order=S.mode_order,
                # The one wrap that must copy. SDDMM's result has S's sparsity pattern
                # and the kernel returns S's own index arrays rather than a copy of them
                # (`D.storage.index.mode_indices = S_mode_indices` in csrc/kernels.h),
                # so adopting here would make the result share index buffers with its
                # operand. Flip this only together with that kernel.
                adopt=False,
            )

    # ── Fast dispatch cache ─────────────────────────────────────────────
    # On a cache hit, skip the entire scheduling pipeline (select_loop_order
    # + auto_schedule) which dominates wall-clock time for cached kernels.
    _dispatch_key = None
    if not compile_only and "_post_ops" not in kwargs:
        _dispatch_key = _einsum_cache_key(
            expression,
            tensors,
            kwargs.get("format", None),
            output_mode_order,
            effective_schedule,
            options_key=options.reuse_key,
            schedule_key=options.schedule_key,
        )
        _cached = _einsum_dispatch_cache.get(_dispatch_key)
        if _cached is not None:
            # ``cancel_frontend`` without its call: only a call handed a context began
            # a frontend stage to retire, and a warm call is handed none.
            if compilation._frontend is None:
                compilation._cancelled = True
            else:
                compilation.cancel_frontend()
            # The result bound (patch 0056): None unless the result's shape can pass
            # the 32-bit position bound at a level assembled one entry at a time, so a
            # call whose shape cannot pays this test and no call.  Before any move.  The
            # operands' coordinates are read only where the counts pass the bound, and
            # once per index structure (result_bound._cached_join).
            if _cached[8] is not None:
                _cached[8].refuse_past_limit(_stored_counts(tensors), tensors)
            _module = _cached[0]
            _output_fmt = _cached[1]
            _temp_mo = _cached[2]
            _final_mo = _cached[3]
            _input_mos = _cached[4]
            _input_idx_strs = _cached[5]
            _result_idx_strs = _cached[6]
            _move_kwargs = _cached[7]

            # Set correct mode orders on input tensors.  One move each, to the order
            # the kernel reads, as _einsum_operand_relayout decided for this entry:
            # its letters, and whether every stored entry is copied.  The key holds
            # each operand's layout, so the decision is this call's.  The move is
            # handed the call's options holder, not a snapshot, so a move whose kernel
            # is kept makes none either (``_conversion_kwargs``).
            cached_tensors = list(tensors)
            _relayouted = False
            for _index, (_t, _mo) in enumerate(zip(cached_tensors, _input_mos)):
                if list(_t.mode_order) != _mo:
                    cached_tensors[_index] = _with_compiler_mode_order(
                        _t,
                        _mo,
                        options,
                        compilation.nested_context(),
                        **_move_kwargs[_index],
                    )
                    _relayouted = True
            tensors = tuple(cached_tensors)

            # Compute result shape from expression + current tensor shapes. Unless an
            # operand was just relayouted this is the map the validating call above
            # already built -- the same expression over the same tensors -- so reuse it
            # rather than walking the operands again. A relayout permutes an operand's
            # physical shape and its mode order together, so it cannot change a logical
            # index's size either (test_logical_index_sizes_survive_relayout), but there
            # is no reason to lean on that: recompute when one happened.
            _idx_to_size = (
                logical_index_sizes
                if not _relayouted and _input_idx_strs == input_index_strs
                else _logical_index_sizes(_input_idx_strs, tensors)
            )
            _logical_result_shape = tuple(_idx_to_size[_s] for _s in _result_idx_strs)
            _result_mode_order = _temp_mo or list(range(len(_logical_result_shape)))
            _physical_result_shape = tuple(
                _logical_result_shape[logical_mode]
                for logical_mode in _result_mode_order
            )

            # Build args and evaluate
            _args: List[Any] = [_physical_result_shape]
            for _t in tensors:
                _args.append(_t.shape)
                _args.append(_t._native_mode_indices())
                _args.append(cast(STensor, _t)._raw_values)

            # ``_evaluate_native`` without its call.  A try block costs nothing until
            # it raises (CPython 3.11 and later).
            _t0 = time.time()
            try:
                _result_cpp = _module.evaluate(*_args)
            except RuntimeError as error:
                raise_if_native_coded(error)
                raise
            _eval_time = time.time() - _t0

            _result = _wrap_generated_result(
                shape=_physical_result_shape,
                tensor_format=_output_fmt,
                result_cpp=_result_cpp,
                mode_order=_temp_mo if _temp_mo else _final_mo,
            )

            if "time_dict" in kwargs:
                kwargs["time_dict"]["eval_time"] = _eval_time

            # Both orders are constants of this cache entry, so compare them here rather
            # than calling a method that has to re-derive the same answer: it validates
            # the permutation, takes a defensive copy of the result's own mode order and
            # sorts it, all to discover there is nothing to do. 1.0 us of a 15 us call.
            if _temp_mo and _temp_mo != _final_mo:
                _result.change_mode_order(
                    _final_mo,
                    **_conversion_kwargs(options, compilation.nested_context()),
                )

            return _result
    # ── End fast dispatch ────────────────────────────────────────────────
    compile_options = options.options()
    # This call compiles: its frontend stage, timed from where `einsum` began the call.
    compilation_context, frontend_token = compilation.frontend()

    # unique_index_strs should be a list of unique index strings
    # e.g. ["i", "j", "k"]
    unique_index_strs = list("".join(input_groups) + result_expression)
    # Make sure the index strings are unique, keeping the order
    unique_index_strs = list(dict.fromkeys(unique_index_strs))
    # Reorder input index strings by each tensor's mode_order
    input_index_strs_sorted = [
        [input_index_strs[i][idx] for idx in tensors[i].mode_order]
        for i in range(len(tensors))
    ]

    # Reorder result index strings by output_mode_order if provided
    result_index_strs_sorted = (
        [result_index_strs[i] for i in output_mode_order]
        if output_mode_order
        else result_index_strs
    )

    # Build concatenated substrings for topo_sort_characters
    index_strs_concat = ["".join(s) for s in input_index_strs_sorted] + [
        "".join(result_index_strs_sorted)
    ]
    index_strs_by_schedule = topo_sort_characters(index_strs_concat, tensors)

    # Create a list of IndexVar objects, and a dict mapping index strings
    # to IndexVar objects
    index_vars = [IndexVar(index_str) for index_str in unique_index_strs]
    index_var_dict = {index_var.name: index_var for index_var in index_vars}

    # Compute temp_mode_order: the scheduler-recommended mode order for result
    index_str_to_mode_index = {s: i for i, s in enumerate(result_index_strs)}
    temp_mode_order = [
        index_str_to_mode_index[s]
        for s in index_strs_by_schedule
        if s in result_index_strs
    ]
    final_mode_order = output_mode_order if output_mode_order else temp_mode_order

    # Predict compiler-visible mode orders without executing a relayout while
    # the enclosing frontend stage is open. Runtime relayout compilation is a
    # distinct nested production path and happens only after frontend work has
    # completed below.
    compiler_mode_orders: List[List[int]] = []
    for tensor_index, input_index_str in enumerate(input_index_strs):
        new_mode_order = []
        str_to_mode = {s: i for i, s in enumerate(input_index_str)}
        for s in index_strs_by_schedule:
            if s in str_to_mode:
                new_mode_order.append(str_to_mode[s])
        compiler_mode_orders.append(new_mode_order)

    # Create a mapping from each index string to the list of LevelFormats
    # of the levels it indexes into each input tensor
    index_str_to_level_formats = {}
    tensors_new = []
    for sorted_index_strs, tensor in zip(input_index_strs_sorted, tensors):
        if not isinstance(tensor, (STensor, TensorSpec)):
            raise TensorTypeError(
                "input tensor is not a Scorch tensor or spec",
                code="einsum_operand_type_invalid",
            )
        tensors_new.append(tensor)

        for i, index_str in enumerate(sorted_index_strs):
            if index_str not in index_str_to_level_formats:
                index_str_to_level_formats[index_str] = []
            index_str_to_level_formats[index_str].append(
                tensor.format.get_level_formats()[i]
            )

    tensors = tensors_new

    # Create TensorVar's for each tensor
    tensor_vars = []
    tensor_names = _einsum_tensor_names()
    output_tensor_dtype = None
    for i, tensor in enumerate(tensors):
        if isinstance(tensor, (STensor, TensorSpec)):
            tensor_name = next(tensor_names)
            tensor_vars.append(
                TensorVar(
                    name=tensor_name,
                    fmt=tensor.format,
                    shape=tuple(
                        tensor.logical_shape[mode] for mode in compiler_mode_orders[i]
                    ),
                    dtype=tensor.dtype,
                    mode_order=compiler_mode_orders[i],
                )
            )
            if output_tensor_dtype is None:
                output_tensor_dtype = tensor.dtype
            else:
                if output_tensor_dtype != tensor.dtype:
                    require_same_value_dtype(
                        output_tensor_dtype, tensor.dtype, operation="einsum"
                    )

    # Get output format from kwargs
    output_format = requested_output_format

    # If output format is not specified, do sparse for all levels first
    if output_format is None:
        # Use format inference rules to infer the optimal format of the output
        # tensor
        # The format inference rules are decided on a per-level basis:
        # 1. Let the index variable indexing into the level be called i
        # 2. If the index variable is used to index into any input tensor's sparse
        #    dimension and multiplied with any other tensor, then the level is
        #    sparse
        # 3. If the index variable is used to index into any input tensor's dense
        #    dimension and added with any other tensor, then the level is dense
        # 4. Otherwise, the level is compressed
        # i.e sparse * anything = sparse
        #     dense + anything = dense
        # Note that LevelType.COMPRESSED and LevelType.COORDINATE are both "sparse"
        # levels
        # To break ties, compressed wins over coordinate (it is checked first below)

        # Create a list of LevelFormat objects
        output_level_formats = []
        for index_str in result_index_strs:
            level_format = LevelFormat(LevelType.DENSE)
            # Use the index_str_to_level_formats to get the list of LevelFormats
            # of the levels it indexes into each input tensor
            level_formats: List[LevelFormat] = index_str_to_level_formats[index_str]
            # If any of them is sparse, then the output level is sparse
            if any(
                level_format.get_level_type() == LevelType.COMPRESSED
                for level_format in level_formats
            ):
                level_format = LevelFormat(LevelType.COMPRESSED)
            # If any of them is coordinate, then the output level is coordinate format
            elif any(
                level_format.get_level_type() == LevelType.COORDINATE
                for level_format in level_formats
            ):
                level_format = LevelFormat(LevelType.COORDINATE)

            output_level_formats.append(level_format)

        # Make sure that the output format doesn't have a sparse level preceding
        # a dense level
        # If it does, then we need to make the preceding level dense as well
        # e.g. if the output level formats are [sparse, dense, dense], then we
        # need to make it [dense, dense, dense]
        # TODO: unless we are dealing with block tensors
        for i in range(len(output_level_formats) - 1, 0, -1):
            if (
                output_level_formats[i].get_level_type() == LevelType.DENSE
                and output_level_formats[i - 1].get_level_type() != LevelType.DENSE
            ):
                output_level_formats[i - 1] = LevelFormat(LevelType.DENSE)

        # For SDDMM-like patterns (sparse output with reduction variables
        # where an input tensor mirrors the output sparsity), use all-COO
        # output to enable the scalar-accum codegen path.  This gives
        # optimal loop order (reduction innermost), no workspace, and SIMD
        # vectorization of the dense reduction.
        _has_sparse_output = any(
            lf.get_level_type() in (LevelType.COMPRESSED, LevelType.COORDINATE)
            for lf in output_level_formats
        )
        if _has_sparse_output:
            _result_set = set(result_index_strs)
            _reduction_strs = [s for s in unique_index_strs if s not in _result_set]
            if _reduction_strs:
                # Check if any input tensor has the same index variables as
                # the output and contains sparse levels (SDDMM pattern).
                for inp_strs, tensor in zip(input_index_strs, tensors):
                    if set(inp_strs) == _result_set:
                        _inp_level_types = tensor.format.get_level_types()
                        if any(
                            lt in (LevelType.COMPRESSED, LevelType.COORDINATE)
                            for lt in _inp_level_types
                        ):
                            output_level_formats = [
                                LevelFormat(LevelType.COORDINATE)
                                for _ in output_level_formats
                            ]
                            break

        output_level_formats = _without_mixed_coordinate_levels(output_level_formats)
        output_format = TensorFormat(output_level_formats)
        # print(f"\nUnspecified output format, using inferred {output_format}")
    result_index_sizes = _logical_index_sizes(input_index_strs, tensors)
    logical_result_shape = tuple(
        result_index_sizes[index_str] for index_str in result_index_strs
    )
    result_mode_order = temp_mode_order or list(range(len(logical_result_shape)))
    physical_result_shape = tuple(
        logical_result_shape[logical_mode] for logical_mode in result_mode_order
    )
    # Patch 0056: the result's levels bounded from the shapes and stored counts, for a
    # result whose shape can pass the position bound at a sparse level (else None).  The
    # expression is one product of every operand, each carrying its own indices; their
    # order per operand lets the bound read the coordinates where the counts pass it.
    result_bound = entry_bound(
        "einsum",
        [result_index_strs[logical_mode] for logical_mode in result_mode_order],
        physical_result_shape,
        output_format.level_types,
        [[(k, letters) for k, letters in enumerate(input_index_strs)]],
        operand_names=input_index_strs,
    )

    # Create the result TensorVar
    if output_tensor_dtype is None:
        raise CompileSpecError(
            "einsum requires at least one typed operand", code="einsum_operand_missing"
        )
    result_tensor_var = TensorVar(
        name=next(tensor_names),
        fmt=output_format,
        shape=physical_result_shape,
        dtype=output_tensor_dtype,
        mode_order=temp_mode_order,
    )

    # Build RHS expression: product of all tensor accesses
    if not index_var_dict:
        raise CompileSpecError(
            "einsum expression does not define any indices",
            code="einsum_expression_empty",
        )
    rhs_expr = None
    for i, tensor_var in enumerate(tensor_vars):
        indices = [index_var_dict[s] for s in input_index_strs[i]]
        access = (
            tensor_var[indices[0]] if len(indices) == 1 else tensor_var[tuple(indices)]
        )
        rhs_expr = access if rhs_expr is None else rhs_expr * access

    # Build LHS access and create assignment
    lhs_indices = [index_var_dict[s] for s in result_index_strs]
    lhs_key = lhs_indices[0] if len(lhs_indices) == 1 else tuple(lhs_indices)
    lhs_access = result_tensor_var[lhs_key]
    cin_stmt: IndexStmt = TensorAssign(lhs_access, cast(IndexExpr, rhs_expr))

    # Wrap in nested ForAll loops (outermost first in schedule, built inside-out)
    for index_str in reversed(index_strs_by_schedule):
        cin_stmt = ForAll(index_var_dict[index_str], cin_stmt)

    # print("CIN:\n", cin_stmt)

    compilation_context.complete_stage(frontend_token)

    # Preserve the legacy two-step operand alignment.  The frontend above uses
    # the predicted compiler-visible metadata so that it can finish before a
    # runtime relayout starts; the physical relayout still happens here, once
    # for the topological order and again below if the selected loop order
    # differs.  An operand with a compressed level waits for both until the
    # selected order is known (_einsum_operand_relayout decides how it moves).
    operands = cast(Sequence[Union[STensor, TensorSpec]], tensors)
    input_orders = [tuple(operand.mode_order) for operand in operands]
    operand_letters = [_storage_letters(operand.format) for operand in operands]
    tensors = _first_einsum_operand_moves(
        operands,
        operand_letters,
        compiler_mode_orders,
        compile_options,
        compilation_context,
    )

    # Align input tensor mode orders with the selected loop order to keep
    # parent-child level traversal valid during lowering for non-canonical
    # schedules.
    desired_mode_orders = _select_and_bind_einsum_operand_mode_orders(
        cin_stmt,
        input_index_strs,
        cast(Sequence[Union[STensor, TensorSpec]], tensors),
        tensor_vars,
        effective_schedule,
        compile_options,
        compilation_context,
    )
    tensors, operand_relayouts = _finish_einsum_operand_moves(
        tensors,
        tensor_vars,
        input_index_strs,
        operand_letters,
        input_orders,
        compiler_mode_orders,
        desired_mode_orders,
        result_index_strs,
        _storage_letters(output_format),
        result_mode_order,
        result_index_sizes,
        compile_options,
        compilation_context,
    )

    # Extract PostOps for fused kernel compilation
    _post_ops = kwargs.get("_post_ops", None)
    _post_ops_tensors = kwargs.get("_post_ops_tensors", None)
    _cache_key_suffix = str(_post_ops) if _post_ops else ""
    lowering_stmt: Union[IndexStmt, ScheduledCIN] = cin_stmt

    # Phase 2b (codegen-parity): emit ONE kernel that branches at runtime on the
    # free-dim size — register-block for small N (single tile, no re-traversal), the
    # byte-identical baseline nest for large N. Default ON via `_regblock_dual_active()`
    # (post M5+x86 validation); `SCORCH_REGBLOCK_DUAL=0` restores the single path.
    # When the pattern doesn't qualify, `_build_regblock_dual_path` returns None and
    # the else-branch below emits the byte-identical baseline (unchanged from before).
    # The two internal dual arms and the fallback all detach the same freshly
    # constructed CIN root.  Prove that root once, then carry the receipt only
    # across this synchronous compiler-owned scheduling block.  No caller code
    # can mutate the graph between the verification and these consumers.
    with _trusted_compiler_built_cin(
        cin_stmt,
        compile_options,
        compilation_context,
    ):
        _dual_llir: Optional[llir.Function] = None
        if (
            effective_schedule is None
            and _regblock_dual_active(compile_options)
            and _post_ops is None
        ):
            _dual = _build_regblock_dual_path(
                cin_stmt,
                _post_ops,
                compile_options=compile_options,
                compilation_context=compilation_context,
            )
            if _dual is not None:
                _dual_llir, _kernel_cache_key = _dual
                _kernel_cache_key = _kernel_cache_key + _cache_key_suffix

        if _dual_llir is None:
            if effective_schedule is not None:
                lowering_stmt = Scheduler.apply_schedule(
                    cin_stmt,
                    effective_schedule,
                    compile_options=compile_options,
                    compilation_context=compilation_context,
                )
            # Default single path. When regblock is on but the dual-path wasn't
            # applicable, force it OFF here so we never ship the wide-k-regressing
            # single-path tiled kernel as a silent fallback.
            elif compile_options.scheduler.regblock_enabled:
                lowering_stmt = cast(
                    IndexStmt,
                    Scheduler._auto_schedule_regblock_arm(
                        cin_stmt,
                        enabled=False,
                        compile_options=compile_options,
                        compilation_context=compilation_context,
                    ),
                )
            else:
                lowering_stmt = cast(
                    IndexStmt,
                    Scheduler.auto_schedule(
                        cin_stmt,
                        compile_options=compile_options,
                        compilation_context=compilation_context,
                    ),
                )
            _kernel_cache_key = _codegen_kernel_cache_key(
                lowering_stmt,
                _post_ops,
                effective_schedule,
                compile_options=compile_options,
            )

    # print("Auto-scheduled CIN:\n", lowering_stmt)

    if _kernel_cache_key in _kernel_cache:
        # print(f"Using cached kernel for {cin_stmt}")
        module = _kernel_cache[_kernel_cache_key]
        result_bound = _refuse_einsum_result_past_bound(
            result_bound,
            _kernel_level_totals.get(_kernel_cache_key, frozenset()),
            tensors,
            compile_only,
        )
    else:
        if _dual_llir is not None:
            lowered_llir: Union[llir.Stmt, List[llir.Stmt]] = _dual_llir
        else:
            lowerer = CINLowerer(
                post_ops=_post_ops,
                compile_options=compile_options,
                compilation_context=compilation_context,
            )
            lowered_llir = lowerer._lower_owned_IndexStmt(lowering_stmt)

        cpp_code = _lower_generated_llir(
            lowered_llir,
            compile_options,
            compilation_context,
        )
        level_totals = level_totals_in_source(cpp_code)
        # After lowering, before the build: the levels the kernel checks itself.
        result_bound = _refuse_einsum_result_past_bound(
            result_bound, level_totals, tensors, compile_only
        )

        # print("\n\n", cpp_code)

        header_cpp_code = compile_options.build.preamble_source

        prepared_build = _prepare_generated_kernel_build(
            header_cpp_code,
            cpp_code,
            compile_options,
            compilation_context,
        )
        module = _load_validated_prepared_kernel(prepared_build)

        _kernel_cache[_kernel_cache_key] = module
        _kernel_level_totals[_kernel_cache_key] = level_totals

    # Populate the dispatch cache so future calls skip scheduling entirely.
    if _dispatch_key is not None and options.still_reusable():
        _einsum_dispatch_cache[_dispatch_key] = (
            module,
            output_format,
            temp_mode_order,
            final_mode_order,
            [list(t.mode_order) for t in tensors],
            input_index_strs,
            result_index_strs,
            [relayout.move_kwargs() for relayout in operand_relayouts],
            result_bound,
        )

    if compile_only:
        result_mode_order = final_mode_order or list(range(len(result_index_strs)))
        return TensorSpec(
            output_format,
            logical_result_shape,
            dtype=output_tensor_dtype,
            mode_order=result_mode_order,
            index_dtype=torch.int32,
            name="einsum_result",
        )

    # Call module.evaluate with the output shape,and the mode indices and values of each tensor
    args: Sequence[Any] = [physical_result_shape]
    for tensor in tensors:
        args.append(tensor.shape)  # type: ignore
        args.append(tensor._native_mode_indices())  # type: ignore
        args.append(tensor._raw_values)  # type: ignore

    # Append extra tensors for PostOps (bias, scale, etc.)
    if _post_ops_tensors:
        for extra_t in _post_ops_tensors:
            # The kernel reads it at the result's storage position.
            ordered = _post_op_tensor_in_storage_order(
                extra_t, logical_result_shape, result_mode_order
            )
            args.append(ordered)  # type: ignore

    start_time = time.time()
    result_cpp = _evaluate_native(module.evaluate, *args)
    end_time = time.time()
    eval_time = end_time - start_time
    # print("Time taken for evaluate:", eval_time)

    result = _wrap_generated_result(
        shape=physical_result_shape,
        tensor_format=output_format,
        result_cpp=result_cpp,
        mode_order=temp_mode_order if temp_mode_order else final_mode_order,
    )

    if "time_dict" in kwargs:
        time_dict = kwargs["time_dict"]
        time_dict["eval_time"] = eval_time

    # Convert to final mode order if it differs from temporary mode order
    if temp_mode_order:
        conversion_options, conversion_context = _conversion_compile_inputs(
            compile_options, compilation_context
        )
        result.change_mode_order(
            final_mode_order,
            _compile_options=conversion_options,
            _compilation_context=conversion_context,
        )

    return result


def _einsum_operands(expression: Any, tensors: Tuple[Any, ...]) -> Tuple[Any, ...]:
    """The operands of an einsum call, in each form ``torch.einsum`` accepts.

    ``einsum("ij,jk->ik", a, b)``; the operands as one list,
    ``einsum("ij,jk->ik", [a, b])``; and the sublist form
    ``einsum(a, [0, 1], b, [1, 2], [0, 2])``, whose operands are the arguments in even
    positions before the optional output sublist.
    """

    if isinstance(expression, str):
        if len(tensors) == 1 and isinstance(tensors[0], (list, tuple)):
            return tuple(tensors[0])
        return tensors
    arguments = (expression,) + tensors
    return arguments[0 : 2 * (len(arguments) // 2) : 2]


def _torch_argument(argument: Any) -> Any:
    """An einsum argument as ``torch.einsum`` takes it: a dense STensor as its tensor,
    a list of operands element by element, anything else (a sublist, a torch tensor)
    as it is."""

    if type(argument) is STensor:
        return argument.to_torch(in_place=False)
    if isinstance(argument, list):
        return [_torch_argument(item) for item in argument]
    return argument


def _requests_schedule(kwargs: Dict[str, Any]) -> bool:
    """Whether a call requests a compiler schedule, in any of its four forms.

    ``schedule=``, ``_schedule=``, a ``_compile_options`` snapshot whose
    ``requested_schedule`` is set, and the context's ``schedule_force(...)``.  It is the
    question the compiler path asks (``_matmul_compile_policy``, ``_DeferredOptions``):
    a snapshot that was handed in decides by itself, otherwise the keywords decide and
    then the forced schedule.  With no keyword arguments only the context variable is
    read.  A request is explicit and torch cannot honor it, so an all-dense call that
    makes one keeps the compiler path (scorch-merge, 2026-10-14).
    """

    return (
        _matmul_compile_policy(kwargs)[1] if kwargs else get_forced_schedule()
    ) is not None


def _requests_sparse_result(kwargs: Dict[str, Any]) -> bool:
    """Whether a call names a result ``format`` with a level that is not dense.

    ``output_format=`` has been read as ``format=`` by then (``_entry_keywords``).  A
    sparse result is scorch's computation even when every operand is dense (Bobby,
    2026-10-14: "for sparse stuff it should always be scorch").  Such a call keeps the
    compiler path, the route einsum ships for it: the shipped route is kept until the
    measurement in drop-in note 5.1 decides between it and torch's product followed by
    a conversion.  The two give the same values; their stored entries can differ,
    which is allowed when the values are the same.  A ``format`` that does not parse is
    the compiler path's to refuse, so it counts as not dense.
    """

    requested = kwargs.get("format")
    if requested is None:
        return False
    try:
        return not parse_format(requested).is_dense()
    except (TensorFormatError, TensorTypeError):
        return True


def _dense_einsum(
    expression: Any,
    tensors: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    operands: Tuple[Any, ...],
) -> Union[torch.Tensor, STensor]:
    """An einsum whose operands and result are all dense, computed by ``torch.einsum``.
    ``operands`` are the call's operands as ``einsum`` read them (``_einsum_operands``).

    Under ``import scorch as torch`` a dense model's every ``torch.einsum`` arrives here,
    so this is torch's own call: torch records its gradient and it compiles nothing.
    With no keyword the arguments reach ``torch.einsum`` exactly as given, in every form
    it accepts, and torch raises its own error for a mistake.  A keyword einsum does not
    take (``out``) was refused at the entry, with ``unknown_keyword_argument``
    (``_entry_keywords``).  A dense STensor operand reaches it through ``to_torch``,
    which reads the tensor autograd tracks for its values when it has one
    (``stensor_grad``), so its gradient is recorded by torch too.

    The result's type follows the call (scorch-merge, 2026-10-14 18:4x): torch tensors
    in and no result keyword give torch's tensor, as ``torch.einsum`` returns it; an
    STensor operand, or a ``format`` (``output_format``) or ``output_mode_order``, give a
    dense STensor over torch's tensor (``_dense_as_stensor``), as the compiler path
    returns for the same call.

    A call naming a scorch keyword is checked by ``_validated_einsum_request`` first, so a
    mistake gets the code the compiler path gives it; ``time_dict`` gets ``eval_time``.
    Two kinds of call never arrive here: one that requests a schedule
    (``_requests_schedule``) and one whose ``format`` has a sparse level
    (``_requests_sparse_result``).  The compiler builds both.
    """

    arguments = tuple(_torch_argument(argument) for argument in (expression,) + tensors)
    stensor_result = (
        "format" in kwargs
        or "output_mode_order" in kwargs
        or any(type(operand) is STensor for operand in operands)
    )
    if not kwargs:
        result = torch.einsum(*arguments)
        return _dense_as_stensor(result) if stensor_result else result
    # Read for their descriptors only, so under no_grad: a converted operand that
    # requires grad would otherwise record a copy of its values nothing reads.
    with torch.no_grad():
        stensors = tuple(
            STensor.from_torch(item) if isinstance(item, torch.Tensor) else item
            for item in operands
        )
    (
        _input_groups,
        _result_expression,
        _input_index_strs,
        _result_index_strs,
        _requested_output_format,
        output_mode_order,
        _logical_index_sizes,
    ) = _validated_einsum_request(expression, stensors, kwargs)
    start_time = time.time()
    result = torch.einsum(*arguments)
    if "time_dict" in kwargs:
        kwargs["time_dict"]["eval_time"] = time.time() - start_time
    if not stensor_result:
        return result
    return _dense_as_stensor(result, output_mode_order)


def _dense_as_stensor(
    result: torch.Tensor, output_mode_order: Optional[List[int]] = None
) -> STensor:
    """A dense torch result as a dense STensor over the same memory.

    ``STensor.from_torch`` stores a dense tensor's values as its row-major flattening in
    the STensor's mode order, a view when the tensor is laid out in that order, and gives
    them the tensor's history when it requires grad (``stensor_grad.link_dense_copy``),
    so the STensor differentiates as torch's tensor does.  A requested
    ``output_mode_order`` is the mode order, and a tensor laid out otherwise is copied
    into it.  With none requested the level order is scorch's choice (the format fixes
    each mode's level kind, not their order), so the mode order is the one torch's tensor
    is laid out in: a permutation of its axes that makes it contiguous, when there is one.
    ``torch.einsum`` can return such a permuted view (``"ij,jk->ki"``), and then nothing
    is copied either.  A tensor that is no permutation of a contiguous one (none comes
    from ``torch.einsum`` today) is copied into the identity order.
    """

    identity = list(range(result.dim()))
    order = identity if output_mode_order is None else output_mode_order
    if output_mode_order is None and result.dim() > 1 and not result.is_contiguous():
        # The axes by decreasing stride; ties (a size-1 axis can have any stride) keep
        # the logical order.
        strides = result.stride()
        by_stride = sorted(identity, key=lambda axis: (-strides[axis], axis))
        if result.permute(*by_stride).is_contiguous():
            order = by_stride
    return STensor.from_torch(result, mode_order=None if order == identity else order)


def einsum(
    expression: Any,
    *tensors: Optional[Union[torch.Tensor, STensor, TensorSpec]],
    compile_only: Optional[bool] = False,
    **kwargs: Any,
) -> Union[torch.Tensor, STensor, TensorSpec]:
    """Compile and evaluate a validated numpy-style sparse/dense einsum."""

    # The product entries' keyword rule first (patch 0072): a name einsum does not take
    # is refused, and output_format= is read as format=.  A call with no keywords pays
    # the test alone.
    if kwargs:
        _entry_keywords(kwargs, _EINSUM_KEYWORDS, "einsum")
    # Operands and a result that are all dense are torch's computation (drop-in
    # note, section 0.1, step 1): ``_dense_einsum``.  Tested before anything else
    # is built, so a dense model pays one pass over its operands.  A sparse result
    # format (``_requests_sparse_result``), a compile-only request, a schedule
    # request (``_requests_schedule``), and the internal calls that carry another
    # call's options (``_deferred_options``, from ``matmul``) or fused
    # post-operations (``_post_ops``, ``_post_ops_tensors``), keep the compiler path,
    # ``_einsum_compiler_path``.  This is the one decision between the two routes:
    # neither hands a call to the other.  The operands read here are handed on, so a
    # call reads them once.
    operands: Optional[Tuple[Any, ...]] = None
    if (
        not compile_only
        and "_deferred_options" not in kwargs
        and "_post_ops" not in kwargs
        and "_post_ops_tensors" not in kwargs
    ):
        operands = _einsum_operands(expression, tensors)
        dense = bool(operands)
        for operand in operands:
            if not _is_dense_operand(operand):
                dense = False
                break
        if (
            dense
            and not _requests_sparse_result(kwargs)
            and not _requests_schedule(kwargs)
        ):
            return _dense_einsum(expression, tensors, kwargs, operands)
    # ``kwargs`` is this call's own dict, so the compiler path takes it as it is.
    return _einsum_compiled(expression, tensors, compile_only, kwargs, operands)


def _einsum_via_compiler(
    expression: Any,
    *tensors: Optional[Union[torch.Tensor, STensor, TensorSpec]],
    compile_only: Optional[bool] = False,
    **kwargs: Any,
) -> Union[torch.Tensor, STensor, TensorSpec]:
    """``einsum`` computed by scorch's compiler, whatever the operands are.

    ``einsum`` sends an all-dense call with no schedule request to ``torch.einsum``
    and everything else to the compiler.  A caller that needs scorch's own kernel for
    an all-dense program, such as a harness comparing compilers or a test of dense
    code generation under the automatic schedule, calls this instead.  It checks a
    call's arguments as ``einsum`` does, then runs the same compiler path
    (``_einsum_compiler_path``), so the two entries build the same kernel for the same
    call.
    """

    return _einsum_compiler_path(
        expression, *tensors, compile_only=compile_only, **kwargs
    )


def _einsum_compiler_path(
    expression: Any,
    *tensors: Optional[Union[torch.Tensor, STensor, TensorSpec]],
    compile_only: Optional[bool] = False,
    **kwargs: Any,
) -> Union[torch.Tensor, STensor, TensorSpec]:
    """The compiler's einsum, after the entry's argument checks (``einsum``,
    ``_einsum_via_compiler``): the options holder, the compilation owner and
    ``_einsum_owned``.

    A call through which a gradient is wanted (``autograd.wanted_by_any``: grad mode and
    an operand that asks for one, or a forward-mode tangent) is computed without autograd
    and its result recorded with a node that raises ``einsum_derivative_unsupported``
    when a gradient is asked of it (``_einsum_without_derivative``), so a forward pass
    that never calls ``backward`` runs and a gradient is never dropped.  ``matmul``'s own
    calls (``_deferred_options``) are exempt: ``autograd.serve`` has decided their
    gradient, and computes what it does not cover under ``torch.no_grad`` on detached
    operands.
    """

    return _einsum_compiled(expression, tensors, compile_only, kwargs, None)


def _einsum_compiled(
    expression: Any,
    tensors: Tuple[Any, ...],
    compile_only: Optional[bool],
    kwargs: Dict[str, Any],
    operands: Optional[Tuple[Any, ...]],
) -> Union[torch.Tensor, STensor, TensorSpec]:
    """``_einsum_compiler_path`` with its arguments as they are held: ``kwargs`` a dict
    the call owns, which this changes, and ``operands`` the call's operands
    (``_einsum_operands``) when the caller has read them, else ``None``.  ``einsum``
    calls it directly, with the operands it chose the route by."""

    if not compile_only and "_deferred_options" not in kwargs:
        # autograd.wanted_by_any written out, grad mode and the operands' flags first and
        # forward-mode AD after, so a call through which no gradient is wanted makes no
        # call into autograd.py. The operands are the ones einsum handed on, else
        # _einsum_operands', read in place.
        grad_wanted = False
        if torch.is_grad_enabled():
            checked: Optional[Sequence[Any]] = operands
            if checked is None:
                if isinstance(expression, str):
                    checked = tensors
                    if len(tensors) == 1 and isinstance(tensors[0], (list, tuple)):
                        checked = tensors[0]
                else:
                    arguments = (expression,) + tensors
                    checked = arguments[0 : 2 * (len(arguments) // 2) : 2]
            for operand in checked:
                if (isinstance(operand, torch.Tensor) and operand.requires_grad) or (
                    type(operand) is STensor and operand._metadata.requires_grad
                ):
                    grad_wanted = True
                    break
        if not grad_wanted and _fwad._current_level >= 0:
            grad_wanted = _autograd.wanted_by_any(
                _einsum_operands(expression, tensors) if operands is None else operands
            )
        if grad_wanted:
            return _einsum_recorded(expression, tensors, kwargs)
    # ``_deferred_options``: the options holder of a call this one runs inside
    # (``matmul``), made from the same keyword arguments, so the two make at most one
    # snapshot between them.
    options = kwargs.pop("_deferred_options", None)
    if options is None:
        options = _DeferredOptions(kwargs)
    elif type(options) is not _DeferredOptions:
        raise ScorchTypeError(
            "_deferred_options must be a _DeferredOptions instance",
            code="deferred_options_type_invalid",
        )
    compilation = _DeferredCompilation(kwargs, options)
    try:
        return _einsum_owned(
            expression,
            *tensors,
            compile_only=compile_only,
            options=options,
            compilation=compilation,
            **kwargs,
        )
    except Exception:
        compilation.fail_frontend_if_active()
        raise


def _einsum_recorded(
    expression: Any, tensors: Tuple[Any, ...], kwargs: Dict[str, Any]
) -> Union[torch.Tensor, STensor]:
    """The compiler's einsum of a call through which a gradient is wanted, computed on
    detached operands under ``torch.no_grad`` and recorded by ``einsum_grad``: its
    result's values as a function of every operand's.  A call that module does not
    differentiate gets a node that raises ``einsum_derivative_unsupported`` when a
    gradient is asked of it instead (``autograd.not_differentiable``), whose inputs are
    every operand that requires grad, so the result requires grad exactly when torch's
    would and a forward pass that never calls ``backward`` runs."""

    operands = _einsum_operands(expression, tensors)

    def refuse(result: Any, why: str) -> Any:
        shown = expression if isinstance(expression, str) else "<sublist form>"
        described = ", ".join(_autograd._describe(operand) for operand in operands)
        return _autograd.not_differentiable(
            result,
            f"scorch.einsum has no derivative for {shown!r} with {described}: {why}. "
            "An einsum whose operands and result are all dense is differentiated by "
            "torch.einsum",
            "einsum_derivative_unsupported",
            operands,
        )

    return _einsum_grad.recorded(
        expression, tensors, kwargs, _einsum_compiler_path, refuse
    )


einsum.__doc__ = _einsum_owned.__doc__


_ModeOrderAlignmentPlan = Tuple[
    Tuple[Optional[Tuple[int, ...]], ...],
    Optional[Tuple[int, ...]],
]


@dataclass(frozen=True)
class _RuntimeTensorSnapshot:
    """One validated runtime tensor captured before compiler-side mutation."""

    tensor: STensor
    shape: Tuple[int, ...]
    dtype: torch.dtype
    tensor_format: TensorFormat
    mode_order: Tuple[int, ...]
    has_index: bool
    native_mode_indices: List[List[torch.Tensor]]
    values: torch.Tensor


def _checked_runtime_shape(shape: object, owner: str) -> Tuple[int, ...]:
    """Snapshot one safe public shape with signed-int64 ABI bounds."""

    if (
        type(shape) is not tuple
        and type(shape) is not list
        and type(shape) is not torch.Size
        and type(shape) is not range
    ):
        raise CompileSpecError(
            f"{owner} must be a tuple, list, torch.Size, or range of exact ints"
        )
    try:
        rank = len(shape)
    except OverflowError as error:
        raise CompileSpecError(f"{owner} rank exceeds the supported limit") from error
    if rank > 64:
        raise CompileSpecError(f"{owner} rank must not exceed 64")
    max_int64 = (1 << 63) - 1
    checked: List[int] = []
    for mode, extent in enumerate(shape):
        if type(extent) is not int or extent < 0:
            raise CompileSpecError(
                f"{owner} must contain nonnegative signed int64 extents"
            )
        if extent > max_int64:
            raise shape_capacity_error(
                ShapeCapacitySpecError,
                f"{owner} must contain nonnegative signed int64 extents",
                owner,
                checked,
                mode,
                extent,
            )
        checked.append(extent)
    # A zero extent holds no element whatever the extents before it multiply to.
    if 0 not in checked:
        count = math.prod(checked)
        if count > max_int64:
            raise shape_capacity_error(
                ShapeCapacitySpecError,
                f"{owner} element count exceeds signed int64",
                owner,
                checked,
                None,
                count,
            )
    return tuple(checked)


def _snapshot_runtime_tensors(
    args: tuple,
    *,
    expected_count: Optional[int] = None,
) -> Tuple[_RuntimeTensorSnapshot, ...]:
    """Validate every runtime operand before reading any operand descriptor."""

    if type(args) is not tuple:
        raise CompileSpecError("runtime tensor arguments must be an exact tuple")
    if expected_count is not None:
        if type(expected_count) is not int or expected_count < 0:
            raise TypeError("expected runtime tensor count must be a nonnegative int")
        if len(args) != expected_count:
            raise CompileSpecError(
                f"CIN expects {expected_count} runtime tensors, got {len(args)}"
            )
    for position, arg in enumerate(args):
        if type(arg) is not STensor:
            raise CompileSpecError(
                f"runtime tensor argument {position} must be an exact STensor"
            )

    snapshots: List[_RuntimeTensorSnapshot] = []
    for position, arg in enumerate(args):
        # The all-argument exact-type pass above is deliberately separate:
        # a hostile later object cannot make us execute a property, validation
        # hook, or relayout on an earlier argument before the request fails.
        try:
            shape = _checked_runtime_shape(
                STensor.shape.__get__(arg, STensor),
                f"runtime tensor argument {position} shape",
            )
            dtype = STensor.dtype.__get__(arg, STensor)
            tensor_format = STensor.format.__get__(arg, STensor)
            mode_order = STensor.mode_order.__get__(arg, STensor)
            has_index = STensor.has_index.__get__(arg, STensor)
            # The detached values: the snapshot reaches native code, which never
            # reads autograd's tensor (stensor_grad).
            values = STensor._raw_values.__get__(arg, STensor)
            native_mode_indices = STensor._native_mode_indices(arg)
        except CompileSpecError:
            raise
        except Exception as error:
            raise CompileSpecError(
                f"runtime tensor argument {position} has invalid internal state"
            ) from error
        if type(dtype) is not torch.dtype:
            raise CompileSpecError(
                f"runtime tensor argument {position} dtype must be an exact "
                "torch.dtype"
            )
        if type(tensor_format) is not TensorFormat:
            raise CompileSpecError(
                f"runtime tensor argument {position} format must be an exact "
                "TensorFormat"
            )
        if type(mode_order) is not tuple or any(
            type(mode) is not int for mode in mode_order
        ):
            raise CompileSpecError(
                f"runtime tensor argument {position} mode order must be an "
                "exact tuple of ints"
            )
        rank = len(shape)
        try:
            format_rank = TensorFormat.get_order(tensor_format)
            level_types = TensorFormat.get_level_types(tensor_format)
        except Exception as error:
            raise CompileSpecError(
                f"runtime tensor argument {position} has invalid format state"
            ) from error
        if (
            format_rank != rank
            or len(mode_order) != rank
            or sorted(mode_order) != list(range(rank))
        ):
            raise CompileSpecError(
                f"runtime tensor argument {position} format, shape, and mode "
                "order ranks must agree"
            )
        if type(has_index) is not bool:
            raise CompileSpecError(
                f"runtime tensor argument {position} has_index must be an exact bool"
            )
        if not isinstance(values, torch.Tensor):
            raise CompileSpecError(
                f"runtime tensor argument {position} values must be a torch.Tensor"
            )
        try:
            invalid_values = (
                values.dtype is not dtype
                or values.layout is not torch.strided
                or values.device.type != "cpu"
                or values.dim() != 1
                or not values.is_contiguous()
                or values.is_neg()
                or values.is_conj()
            )
        except Exception as error:
            raise CompileSpecError(
                f"runtime tensor argument {position} has invalid values state"
            ) from error
        if invalid_values:
            raise CompileSpecError(
                f"runtime tensor argument {position} values violate the native "
                "CPU storage contract"
            )
        if type(native_mode_indices) is not list or len(native_mode_indices) != rank:
            raise CompileSpecError(
                f"runtime tensor argument {position} native indices must be "
                "rank-sized lists of tensors"
            )
        if (
            any(type(arrays) is not list for arrays in native_mode_indices)
            or type(level_types) is not list
            or len(level_types) != rank
            or any(type(level_type) is not LevelType for level_type in level_types)
        ):
            raise CompileSpecError(
                f"runtime tensor argument {position} native index levels must "
                "match its exact TensorFormat"
            )
        for level, (arrays, level_type) in enumerate(
            zip(native_mode_indices, level_types)
        ):
            expected_arrays = (
                0
                if level_type is LevelType.DENSE
                else (2 if level_type is LevelType.COMPRESSED else 1)
            )
            if len(arrays) != expected_arrays:
                raise CompileSpecError(
                    f"runtime tensor argument {position} level {level} requires "
                    f"{expected_arrays} native index arrays"
                )
        if any(
            not isinstance(index, torch.Tensor)
            for arrays in native_mode_indices
            for index in arrays
        ):
            raise CompileSpecError(
                f"runtime tensor argument {position} native indices must be "
                "rank-sized lists of tensors"
            )
        try:
            invalid_index = any(
                index.layout is not torch.strided
                or index.device.type != "cpu"
                or index.dtype not in (torch.int32, torch.int64)
                or index.dim() != 1
                or not index.is_contiguous()
                for arrays in native_mode_indices
                for index in arrays
            )
            index_dtypes = {
                index.dtype for arrays in native_mode_indices for index in arrays
            }
        except Exception as error:
            raise CompileSpecError(
                f"runtime tensor argument {position} has invalid native index state"
            ) from error
        if invalid_index or len(index_dtypes) > 1:
            raise CompileSpecError(
                f"runtime tensor argument {position} native indices violate the "
                "CPU index contract"
            )
        try:
            stored_values = values.numel()
            parent_positions = 1
            for level, (extent, arrays, level_type) in enumerate(
                zip(shape, native_mode_indices, level_types)
            ):
                if level_type is LevelType.DENSE:
                    parent_positions *= extent
                    continue
                coordinates = arrays[-1]
                coordinate_count = coordinates.numel()
                if level_type in (LevelType.COORDINATE, LevelType.SINGLETON):
                    if coordinate_count != stored_values:
                        raise CompileSpecError(
                            f"runtime tensor argument {position} coordinate level "
                            f"{level} count must equal its stored value count"
                        )
                    parent_positions = coordinate_count
                    continue
                positions = arrays[0]
                if (
                    positions.numel() != parent_positions + 1
                    or positions[0].item() != 0
                    or positions[-1].item() != coordinate_count
                ):
                    raise CompileSpecError(
                        f"runtime tensor argument {position} compressed level "
                        f"{level} has an invalid positions/cardinality chain"
                    )
                parent_positions = coordinate_count
            if parent_positions != stored_values:
                raise CompileSpecError(
                    f"runtime tensor argument {position} storage cardinality "
                    "does not match its value count"
                )
        except CompileSpecError:
            raise
        except Exception as error:
            raise CompileSpecError(
                f"runtime tensor argument {position} has invalid storage "
                "cardinality state"
            ) from error
        snapshots.append(
            _RuntimeTensorSnapshot(
                tensor=arg,
                shape=shape,
                dtype=dtype,
                tensor_format=tensor_format,
                mode_order=mode_order,
                has_index=has_index,
                native_mode_indices=native_mode_indices,
                values=values,
            )
        )
    return tuple(snapshots)


def _validate_direct_runtime_formats(
    cin_stmt: IndexStmt,
    snapshots: Tuple[_RuntimeTensorSnapshot, ...],
) -> None:
    """Check every direct-CIN operand before relayout or metadata mutation."""

    rhs_tensor_vars = cin_stmt.get_rhs_tensor_vars()
    if len(rhs_tensor_vars) != len(snapshots):
        raise CompileSpecError(
            f"CIN expects {len(rhs_tensor_vars)} runtime tensors, got "
            f"{len(snapshots)}"
        )
    mismatches = [
        (tensor_var, snapshot.tensor_format)
        for tensor_var, snapshot in zip(rhs_tensor_vars, snapshots)
        if tensor_var.format != snapshot.tensor_format
    ]
    if mismatches:
        tensor_var, actual_format = mismatches[0]
        raise CompileSpecError(
            f"CIN tensor {tensor_var.name!r} expects format "
            f"{tensor_var.format}, got {actual_format}"
        )


def _plan_mode_orders_to_loop_order(
    cin_stmt: IndexStmt,
    args: tuple,
) -> _ModeOrderAlignmentPlan:
    """Plan detached CIN/runtime mode alignment without executing relayouts."""

    loop_order_names: List[str] = []
    curr: IndexStmt = cin_stmt
    while isinstance(curr, ForAll):
        loop_order_names.append(curr.index_var.name)
        curr = curr.stmt

    if not loop_order_names:
        return tuple(None for _ in args), None

    rhs_accesses = cin_stmt.get_rhs_tensor_accesses()
    if len(rhs_accesses) != len(args):
        return tuple(None for _ in args), None

    rhs_mode_orders: List[Optional[Tuple[int, ...]]] = []
    for ta in rhs_accesses:
        tv = ta.get_tensor()
        index_var_names = [iv.name for iv in ta.get_index_vars()]
        desired_names = [n for n in loop_order_names if n in index_var_names]
        desired_mode_order = tuple(index_var_names.index(n) for n in desired_names)
        if len(desired_mode_order) != len(tv.mode_order):
            rhs_mode_orders.append(None)
            continue
        rhs_mode_orders.append(desired_mode_order)

    lhs_mode_order: Optional[Tuple[int, ...]] = None
    if isinstance(curr, TensorAssign):
        lhs_tv = curr.lhs.get_tensor()
        lhs_names = [iv.name for iv in curr.lhs.get_index_vars()]
        desired_names = [n for n in loop_order_names if n in lhs_names]
        desired_mode_order = tuple(lhs_names.index(n) for n in desired_names)
        if len(desired_mode_order) == len(lhs_tv.mode_order):
            lhs_mode_order = desired_mode_order
    return tuple(rhs_mode_orders), lhs_mode_order


def _plan_direct_cin_runtime_binding(
    cin_stmt: IndexStmt,
    result_shape: Sequence[int],
    args: tuple,
    compile_options: CompileOptions,
    compilation_context: CompilationContext,
) -> Tuple[
    Tuple[int, ...],
    Tuple[_RuntimeTensorSnapshot, ...],
    _ModeOrderAlignmentPlan,
]:
    """Validate and plan direct-CIN binding without relayout or mutation."""

    planning_token = compilation_context.begin_stage(
        CompilerStageId.FRONTEND_VALIDATED_OPERATION_CONSTRUCTION,
        compile_options=compile_options,
    )
    try:
        checked_result_shape = _checked_runtime_shape(result_shape, "result shape")
        rhs_tensor_vars = cin_stmt.get_rhs_tensor_vars()
        snapshots = _snapshot_runtime_tensors(
            args,
            expected_count=len(rhs_tensor_vars),
        )
        _validate_direct_runtime_formats(cin_stmt, snapshots)
        result_tensor_vars: List[TensorVar] = []
        for tensor_var in cin_stmt.get_result_tensor_vars():
            if isinstance(tensor_var, Workspace) or any(
                tensor_var is retained for retained in result_tensor_vars
            ):
                continue
            result_tensor_vars.append(tensor_var)
        if len(result_tensor_vars) != 1:
            raise CompileSpecError(
                "direct CIN execution requires exactly one external result tensor"
            )
        result_tensor_var = result_tensor_vars[0]
        result_format = result_tensor_var.format
        if type(result_format) is not TensorFormat or result_format.get_order() != len(
            checked_result_shape
        ):
            raise CompileSpecError(
                "direct CIN result format rank must match the result shape rank"
            )
        plan = _plan_mode_orders_to_loop_order(
            cin_stmt,
            tuple(snapshot.tensor for snapshot in snapshots),
        )
    except Exception:
        compilation_context.fail_stage(planning_token)
        raise
    compilation_context.complete_stage(planning_token)
    return checked_result_shape, snapshots, plan


def _relayout_mode_order_args(
    args: tuple,
    plan: _ModeOrderAlignmentPlan,
    compile_options: CompileOptions,
    compilation_context: Optional[CompilationContext],
    *,
    snapshots: Optional[Tuple[_RuntimeTensorSnapshot, ...]] = None,
) -> tuple:
    """Execute only runtime relayout prerequisites from a frozen alignment plan."""

    if snapshots is None:
        snapshots = _snapshot_runtime_tensors(args)
    if len(args) != len(snapshots) or any(
        arg is not snapshot.tensor for arg, snapshot in zip(args, snapshots)
    ):
        raise CompileSpecError("runtime tensor snapshots do not match relayout inputs")
    aligned_args = list(args)
    for position, (snapshot, desired_mode_order) in enumerate(zip(snapshots, plan[0])):
        if (
            desired_mode_order is not None
            and snapshot.mode_order != desired_mode_order
            and snapshot.has_index
        ):
            aligned = STensor.copy(snapshot.tensor)
            conversion_options, conversion_context = _conversion_compile_inputs(
                compile_options, compilation_context
            )
            STensor.change_mode_order(
                aligned,
                list(desired_mode_order),
                _compile_options=conversion_options,
                _compilation_context=conversion_context,
            )
            aligned_args[position] = aligned
    return tuple(aligned_args)


def _apply_mode_order_alignment(
    cin_stmt: IndexStmt,
    plan: _ModeOrderAlignmentPlan,
) -> None:
    """Apply a validated alignment plan only to compiler-owned CIN metadata."""

    rhs_accesses = cin_stmt.get_rhs_tensor_accesses()
    for tensor_access, desired_mode_order in zip(rhs_accesses, plan[0]):
        if desired_mode_order is not None:
            tensor_access.get_tensor().mode_order = list(desired_mode_order)

    curr: IndexStmt = cin_stmt
    while isinstance(curr, ForAll):
        curr = curr.stmt
    if isinstance(curr, TensorAssign) and plan[1] is not None:
        curr.lhs.get_tensor().mode_order = list(plan[1])


def _align_mode_orders_to_loop_order(
    cin_stmt: IndexStmt,
    args: tuple,
    compile_options: CompileOptions,
    compilation_context: Optional[CompilationContext] = None,
) -> tuple:
    """Compatibility helper that aligns detached CIN and replacement inputs."""

    plan = _plan_mode_orders_to_loop_order(cin_stmt, args)
    aligned_args = _relayout_mode_order_args(
        args,
        plan,
        compile_options,
        compilation_context,
    )
    _apply_mode_order_alignment(cin_stmt, plan)
    return aligned_args


def lower_and_exec_cin(
    cin_stmt: IndexStmt, result_shape: Sequence[int], *args: STensor, **kwargs
) -> STensor:
    """Lower a CIN statement to LLIR then codegen and call on the input tensors.

    Low-level entry to the generic compiler path: aligns the operand mode orders to
    the CIN loop order (``_align_mode_orders_to_loop_order``), lowers CIN -> LLIR ->
    C++, JIT-loads the kernel, executes it on ``args``, and wraps the result as an
    ``STensor``. This is the primitive the compiler gap tests use to drive the
    pipeline directly; it is **not exported** — advanced users reach it as
    ``scorch.ops.lower_and_exec_cin``. Most code should use :func:`matmul` or
    :func:`einsum` instead.

    Parameters
    ----------
    cin_stmt : IndexStmt
        CIN statement (index-notation AST) to lower.
    result_shape : tuple[int, ...], list[int], torch.Size, or range
        Physical shape of the result tensor. Extents must be exact,
        nonnegative signed-64-bit integers and their product must fit in a
        signed 64-bit integer.
    *args : STensor
        Input tensors, matched left-to-right against the RHS tensor accesses.
    **kwargs
        ``time_dict`` : dict, optional -- if given, ``time_dict["eval_time"]`` is
        set to the kernel wall-clock time in seconds.

    Returns
    -------
    STensor
        Output tensor, in the format and mode order of the compiler-owned
        result declaration, sparse or dense.  Its value dtype is the operands'
        promoted dtype (``layout.common_value_dtype``, PyTorch's rule): the
        dtype of the statement's result ``TensorVar`` is ignored and replaced.
    """
    compile_options = _compile_options_from_kwargs(kwargs)
    _reject_unsupported_requested_schedule(compile_options, "lower_and_exec_cin")
    compilation_context = _compilation_context_from_kwargs(kwargs, compile_options)
    cin_stmt = normalize_cin(
        cin_stmt,
        compile_options=compile_options,
        compilation_context=compilation_context,
    )
    result_shape, runtime_args, alignment_plan = _plan_direct_cin_runtime_binding(
        cin_stmt,
        result_shape,
        args,
        compile_options,
        compilation_context,
    )
    try:
        args = _relayout_mode_order_args(
            tuple(snapshot.tensor for snapshot in runtime_args),
            alignment_plan,
            compile_options,
            compilation_context,
            snapshots=runtime_args,
        )
    except Exception:
        if compilation_context._failed_stage_id is None:
            relayout_failure_token = compilation_context.begin_stage(
                CompilerStageId.FRONTEND_VALIDATED_OPERATION_CONSTRUCTION,
                compile_options=compile_options,
            )
            compilation_context.fail_stage(relayout_failure_token)
        raise

    frontend_binding_token = compilation_context.begin_stage(
        CompilerStageId.FRONTEND_VALIDATED_OPERATION_CONSTRUCTION,
        compile_options=compile_options,
    )
    try:
        runtime_args = _snapshot_runtime_tensors(
            args,
            expected_count=len(runtime_args),
        )
        _apply_mode_order_alignment(cin_stmt, alignment_plan)
        rhs_tensor_vars = cin_stmt.get_rhs_tensor_vars()
        for tensor_var, snapshot in zip(rhs_tensor_vars, runtime_args):
            tensor_var.shape = snapshot.shape
            tensor_var.dtype = snapshot.dtype
            tensor_var.mode_order = list(snapshot.mode_order)

        # The operands' promoted type (PyTorch's rule), as ``A + B`` declares
        # it: the first operand's type stored an int64 plus a float64 as int64.
        output_dtype = (
            common_value_dtype(
                *(snapshot.dtype for snapshot in runtime_args),
                operation="lower_and_exec_cin",
            )
            if runtime_args
            else torch.float32
        )
        for tensor_var in cin_stmt.get_result_tensor_vars():
            if not isinstance(tensor_var, Workspace):
                tensor_var.shape = tuple(result_shape)
                tensor_var.dtype = output_dtype
    except Exception:
        compilation_context.fail_stage(frontend_binding_token)
        raise
    compilation_context.complete_stage(frontend_binding_token)

    # Lower to LLIR
    lowerer = CINLowerer(
        compile_options=compile_options,
        compilation_context=compilation_context,
    )
    lowered_llir = lowerer._lower_owned_IndexStmt(cin_stmt)
    cpp_code = _lower_generated_llir(
        lowered_llir,
        compile_options,
        compilation_context,
    )
    # print(cpp_code)
    header_cpp_code = compile_options.build.preamble_source

    prepared_build = _prepare_generated_kernel_build(
        header_cpp_code,
        cpp_code,
        compile_options,
        compilation_context,
    )
    module = _load_validated_prepared_kernel(prepared_build)

    module_args: List[Any] = [result_shape]

    for snapshot in runtime_args:
        module_args.append(snapshot.shape)
        module_args.append(snapshot.native_mode_indices)
        module_args.append(snapshot.values)

    start_time = time.time()
    try:
        result_cpp = module.evaluate(*module_args)
    except RuntimeError as error:
        raise_if_native_coded(error)
        raise
    end_time = time.time()
    eval_time = end_time - start_time
    if "time_dict" in kwargs:
        kwargs["time_dict"]["eval_time"] = eval_time

    # The result is wrapped in the format and mode order its declaration gives,
    # sparse or dense, as ``einsum`` and ``STensor.__add__`` wrap theirs.  A sparse
    # result used to be wrapped as a two-level dense tensor whatever it was, which
    # no sparse kernel's level arrays satisfy (``TensorIndexError``).
    declared_result = lowerer.final_result_tensor_var
    if declared_result is not None and declared_result.format is None:
        declared_result = None
    result_format = (
        declared_result.get_format()
        if declared_result is not None
        else parse_format("dd")
    )
    result_mode_order = (
        tuple(declared_result.mode_order)
        if declared_result is not None and declared_result.mode_order is not None
        else None
    )
    result = _wrap_generated_result(
        shape=tuple(result_shape),
        tensor_format=result_format,
        result_cpp=result_cpp,
        mode_order=result_mode_order,
    )

    return result


def precompile_kernels():
    """Warm the JIT cache by compiling the common SpMM / SpGEMM kernels.

    Calls :func:`einsum` with ``compile_only=True`` for the frequently used
    contraction/format combinations so the first real product does not pay the
    schedule + codegen + C++ compile cost:

    - ``ds @ dd -> dd``  (CSR x dense -> dense)
    - ``oo @ dd -> dd``  (COO x dense -> dense)
    - ``oo @ ds -> dd``  (COO x CSR -> dense)
    - ``ds @ ds -> dd``  (CSR x CSR -> dense)
    - ``ds @ ds -> ds``  (CSR x CSR -> CSR / SpGEMM)

    Prints ``"Precompiled kernels."`` when done. The call at module import in
    ``__init__.py`` is **commented out**, so this does not run automatically —
    invoke ``scorch.precompile_kernels()`` explicitly to warm up.

    Returns
    -------
    None

    Examples
    --------
    >>> import scorch
    >>> scorch.precompile_kernels()   # doctest: +SKIP
    Precompiled kernels.
    """
    # Extents are irrelevant to generated code; a real, immutable rank-2 spec
    # replaces the former format-only (and invalid) STensor placeholders.
    DS = TensorSpec("ds", (1, 1), name="csr_spec")
    DD = TensorSpec("dd", (1, 1), name="dense_spec")
    OO = TensorSpec("oo", (1, 1), name="coo_spec")

    einsum("ik,kj->ij", DS, DD, compile_only=True, format="dd")
    einsum("ik,kj->ij", OO, DD, compile_only=True, format="dd")
    einsum("ik,kj->ij", OO, DS, compile_only=True, format="dd")
    einsum("ik,kj->ij", DS, DS, compile_only=True, format="dd")
    einsum("ik,kj->ij", DS, DS, compile_only=True, format="ds")

    print("Precompiled kernels.")
