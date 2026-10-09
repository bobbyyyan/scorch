"""Cached call plans for the repeated CSR x dense product.

``ops.matmul`` re-derives the same facts on every call: which prebuilt symbol
serves this format and dtype, whether the adaptive tiling selector wants a tiled
kernel and with which panel widths, that the index arrays are structurally sound,
the argument list the legacy kernel ABI expects, and the shape of the result. For
an operand used more than once -- a graph adjacency in a training loop, an
attention mask, a weight matrix -- every one of those is a constant of the
operand and the free dimension.

A plan is that constant, held in the native extension (``csrc/plan.h``). With one
in hand a warm ``matmul`` is a dict lookup and a single Python->C++ hop; without
one, nothing changes. Concretely, on a 64x64 SpMM whose kernel runs in 2.4 us,
the ordinary path spent ~9.6 us in Python getting to it.

Three rules keep this honest:

* **A plan is installed only for an operand that has been seen before.** The
  first call with a given ``(operand, B shape, B dtype)`` records the key and
  takes the ordinary path; the second installs a plan. A program that wraps a
  fresh ``STensor`` per call therefore never pays for a plan it cannot reuse.
* **A plan is built from what the ordinary path already decided**, including the
  selector's memoized verdict, so the first call's semantics -- probes,
  measurements, structured errors -- are untouched.
* **A plan may always decline.** ``SpmmCsrPlan.run`` returns ``None`` when
  anything about the call is outside what it was built for, and the caller falls
  through to the ordinary path. Correctness never depends on the plan being
  right about what it can serve, only on it being conservative.

``SCORCH_DISPATCH_PLAN=0`` disables installation, which leaves the ordinary path
in sole charge -- the same binary, so it is the control arm for measuring this.
"""

from __future__ import annotations

import os
import weakref
from typing import Any, Dict, Optional, Tuple

import torch

import scorch_ops as _native

# Attributes carried on the sparse operand itself. The operand owns its plans:
# they are only valid for its structure, and they die with it -- no global cache
# to bound, no keys that could collide across tensors, and no reference from a
# module-level dict that would keep a graph-scale index array alive.
PLANS_ATTR = "_scorch_spmm_plans"
SEEN_ATTR = "_scorch_spmm_seen"
# Created only if a plan ever declines, so the ordinary path never touches it.
DECLINES_ATTR = "_scorch_spmm_declines"
# The CSR structure autograd.py builds for an operand a gradient flows through (its
# transpose, and the STensors its products run on). Kept and dropped with the plans, for
# the same reasons: it is only valid for this structure and it must die with the operand.
GRAD_ATTR = "_scorch_grad_structure"
# A torch CSR operand's TransposedProductPlans: for each key its transposed product A^T @ M has a
# plan under, the record the C entry serves x @ A from (csrc/planned_matmul.h). Made in C when that
# product is served for the operand; kept and dropped with its plans.
TRANSPOSED_PLANS_ATTR = "_scorch_transposed_plans"

# The same three entries for an operand that is a torch sparse CSR tensor, which cannot
# carry them itself. torch.save and pickle write a tensor's __dict__ along with it, so a
# native plan stored there makes the caller's own tensor unsaveable (both fail on the
# plan object), and anything importable stored instead would make the saved file
# unloadable wherever scorch is not installed. So these live here, keyed on the
# operand's id() and checked against a weak reference to it -- an id is only unique
# among live objects -- and the weak reference's callback drops the entry when the
# operand dies, which keeps the property the STensor attributes have: the plans die with
# the operand, and nothing here keeps a graph-scale index array alive.
#
# A plain dict probed with id() rather than a weak-keyed mapping, because the probe is
# on the path a planned call takes and must cost what the STensor probe costs: one dict
# lookup and one dereference. planned_matmul.h reads this dict directly.
TORCH_OPERANDS: Dict[int, Tuple["_OperandRef", Dict[str, Any]]] = {}


class _OperandRef(weakref.ref):
    """A weak reference to a torch operand that carries its TORCH_OPERANDS key, so one
    module-level callback serves every entry instead of a closure made per entry."""

    __slots__ = ("key",)
    key: int


def _drop_torch_operand(ref: "_OperandRef") -> None:
    held = TORCH_OPERANDS.get(ref.key)
    if held is not None and held[0] is ref:
        del TORCH_OPERANDS[ref.key]


def torch_operand_state(tensor: torch.Tensor, create: bool = False):
    """The plan state of a torch sparse CSR operand: its dict in TORCH_OPERANDS.

    None when there is none and ``create`` is false. An entry whose weak reference no
    longer names ``tensor`` belongs to an earlier object with the same id that has not
    been collected yet, and is replaced.
    """
    key = id(tensor)
    held = TORCH_OPERANDS.get(key)
    if held is not None and held[0]() is tensor:
        return held[1]
    if not create:
        return None
    state: Dict[str, Any] = {}
    ref = _OperandRef(tensor, _drop_torch_operand)
    ref.key = key
    TORCH_OPERANDS[key] = (ref, state)
    return state


def _is_torch_operand(a: Any) -> bool:
    return type(a) is torch.Tensor and a.layout is torch.sparse_csr


def _state_of(a: Any):
    """Where an operand's plans live, if it has any: an STensor's own __dict__, or for a
    torch sparse CSR operand, its entry in TORCH_OPERANDS."""
    if _is_torch_operand(a):
        return torch_operand_state(a)
    return getattr(a, "__dict__", None)


# One operand is realistically multiplied by a handful of distinct free
# dimensions (a GCN layer stack: 128, 64, 16). The bound exists so that sweeping
# N over hundreds of values cannot accumulate plans; past it, installation stops
# and the ordinary path serves, which is what happened before plans existed.
MAX_PLANS_PER_OPERAND = 8

# After this many declines with nothing ever served, the plan is withdrawn and the
# product is refused a plan for the rest of the operand's life.
#
# A plan that declines costs the call it declined: the lookup hits, `run` crosses
# into C++, screens, and returns nothing, and then the ordinary path runs anyway.
# Measured at 0.2-0.8 us on the small cells of both hosts -- 2-8% of those calls,
# which is above the noise floor and therefore a regression on any call site that
# repeats a product a plan cannot serve (a right-hand operand that is always a
# transposed view, say). Withdrawing the plan returns such a site to exactly what it
# cost before plans existed, because a withdrawn key is a plain dict miss.
#
# Only a plan that has served *nothing* is withdrawn, so a plan that helps most calls
# and declines the odd one is never touched, and no reset bookkeeping is needed. A
# call site that mixes servable and unservable right-hand operands at one shape is
# therefore safe: the first servable call makes `served` non-zero and the plan stays.
#
# The trade is that a withdrawal is permanent for that operand and that key, and the
# key does not record contiguity -- so a site whose first eight calls at a given shape
# and dtype are unservable, and whose ninth would have been servable, gets no plan at
# all thereafter. It runs at exactly the speed it did before plans existed, which is
# the right way round: refusing costs a call site the *gain*, keeping a fruitless plan
# would cost every call site a *loss*. Distinguishing the two would mean testing
# `b.is_contiguous()` on every matmul, which spends the common case to buy the rare one.
MAX_FRUITLESS_DECLINES = 8

# Prebuilt symbols a plan can serve, mapped to the kernel kind plan.h knows.
# Anything absent here -- SpGEMM, SpMV, COO, the int32/int64 reference SpMMs --
# simply never gets a plan.
#
# Every CSR x dense symbol the prebuilt registry can resolve must appear here, and
# `test_every_resolvable_csr_dense_symbol_has_a_plan_kind` fails if one does not.
# That test exists because adding `spmm_csr_double_v2` ahead of the two float64
# entries below silently took float64 off the plan path entirely: resolution moved to
# a symbol this table did not know, `.get` returned None, and float64 went back to
# paying the per-call dispatch cost the plan exists to remove. Nothing failed.
_SYMBOL_KINDS: Dict[str, str] = {
    "spmm_csr_float_v2": "v2",
    "spmm_csr_double_v2": "v2_double",
    "prebuilt_spmm_csr_f64": "reference",
    "spmm_csr_double": "reference",
}

_ENABLED = os.environ.get("SCORCH_DISPATCH_PLAN", "1") not in ("0", "false", "False")
_HAS_NATIVE_PLAN = hasattr(_native, "make_spmm_csr_plan")

# A plan carries the tiling selector's verdict, so it is only valid under the
# autotune policy that produced it. Rather than read the level on every call --
# which is a function call, a thread-local probe and a global read, on a path whose
# whole remaining cost is a few hundred nanoseconds -- the generation is part of
# every plan's key, and any policy change moves it on. A plan from an earlier
# generation is simply never found again.
#
# A mutable cell rather than a module-level int so that ``ops.matmul`` can bind it
# once at import and still see the current value: one list index per call.
#
# The cost of this is that a program that changes the level around every call --
# ``with scorch.autotune("max"): scorch.matmul(...)`` in a loop -- never gets a
# plan, because entering and leaving the context manager both move the generation
# on. That is the right trade: an explicit per-call policy override is asking for
# the selector to be consulted, and the ordinary path does exactly that.
GENERATION = [0]

# The same value as ``enabled()``, in a mutable cell, so that ops.matmul can decide
# whether to call the installer at all without a function call or a cross-module
# attribute lookup. With plans switched off this makes the ordinary path pay a single
# list index -- which is what lets the off state serve as an honest control arm.
ENABLED_CELL = [_ENABLED and _HAS_NATIVE_PLAN]


def invalidate_all() -> None:
    """Retire every plan in the process (any autotune policy change calls this)."""
    GENERATION[0] += 1


def enabled() -> bool:
    """Whether plan installation is on (env ``SCORCH_DISPATCH_PLAN``, default on)."""
    return _ENABLED and _HAS_NATIVE_PLAN


def set_enabled(value: bool) -> None:
    """Turn installation on or off for the rest of the process (tests, A/B arms).

    Existing plans are not dropped; call :func:`forget` on an operand for that.
    """
    global _ENABLED
    _ENABLED = bool(value)
    ENABLED_CELL[0] = _ENABLED and _HAS_NATIVE_PLAN


def forget(tensor: Any) -> None:
    """Drop every plan attached to ``tensor``.

    Called from ``STensor._set_state``, which every in-place structural change
    funnels through, so a relayout or an insert cannot leave a plan describing a
    structure that no longer exists. (A plan would also decline on its own: it
    records the index arrays' identity and version counters. This is the belt to
    that suspenders -- it also releases the narrowed index copy.)
    """
    state = _state_of(tensor)
    if state is None:
        return
    state.pop(PLANS_ATTR, None)
    state.pop(SEEN_ATTR, None)
    state.pop(DECLINES_ATTR, None)
    state.pop(GRAD_ATTR, None)
    state.pop(TRANSPOSED_PLANS_ATTR, None)


def _panels(kind: str, param: Any) -> Tuple[int, int]:
    """The selector's tile parameters in plan.h's (free, contraction) order."""
    if kind == "tilej":
        return 0, int(param)
    if kind == "tileijk":
        free, contraction = param
        return int(free), int(contraction)
    return 0, 0


def _account_for_a_decline(state: Dict[str, Any], plans: Dict[Any, Any], key: Any):
    """Called when a plan exists for ``key`` and did not serve the call.

    That inference is free and exact: ``ops.matmul`` looks the plan up before doing
    anything else and returns immediately when it serves, so reaching the installer
    with the key still in ``plans`` means the plan declined this very call. Nothing
    has to be recorded on the path that succeeds.

    A withdrawn plan is stored as ``None`` rather than deleted, which is what makes
    the withdrawal free for the caller: the probe in ``ops.matmul`` already tests the
    looked-up value for ``None``, so a refused key costs exactly what an absent one
    costs, and this function can tell "refused earlier" from "never planned".
    """
    if plans[key] is None:
        return None  # withdrawn on an earlier call
    counts = state.get(DECLINES_ATTR)
    if counts is None:
        counts = state[DECLINES_ATTR] = {}
    seen = counts[key] = counts.get(key, 0) + 1
    # `served` is counted in C++ for nothing, so this reads it only on a path that
    # is already paying for a declined call.
    if seen >= MAX_FRUITLESS_DECLINES and plans[key].served == 0:
        plans[key] = None
    return None


def install(
    a: Any,
    b: torch.Tensor,
    key: Any,
    symbol_name: str,
    kind: str = "v2",
    param: Any = None,
) -> Optional[Any]:
    """Attach a plan for ``a @ b`` if this call is one that should have one.

    Called from the end of every prebuilt dense-output ``matmul``, so the cheap
    tests come first: a call that is not going to install anything -- a product this
    operand has already been refused, an already-planned product, a first sighting, a
    full cache -- gets out for the price of one or two dict probes, before anything
    touches the index arrays. ``key`` is built by the probe in ``ops.matmul``, which
    needs it too, so no call builds it twice.

    ``kind``/``param`` name the kernel that actually served this call, read at the
    dispatch site inside the branch where a tiled kernel ran. They are deliberately
    not re-derived here: the selector's memo holds an entry for any shape it has ever
    probed, including ones whose gate later rejected tiling, so consulting it from
    here would let a plan run tile-j against an ordinary path running v2.
    """
    # An STensor keeps its plans in its own __dict__; a torch sparse CSR operand keeps
    # them in TORCH_OPERANDS (see there). ops.matmul passes a torch.Tensor here only when
    # it is a 2-D sparse CSR operand, so the type is the whole test. `create` because the
    # first sighting is recorded below, and a torch operand has no entry until then.
    if type(a) is torch.Tensor:
        state = torch_operand_state(a, create=True)
    else:
        state = a.__dict__
    # Products this operand already holds something for come first, because they are
    # the ones that reach here repeatedly: an installed plan that declined this call,
    # or a withdrawn one. Both are settled without touching the seen set.
    plans = state.get(PLANS_ATTR)
    if plans is not None:
        if key in plans:
            return _account_for_a_decline(state, plans, key)
        if len(plans) >= MAX_PLANS_PER_OPERAND:
            return None

    # First sighting of this exact product: record it and wait. A *set* of seen
    # keys, not the last one, because a layer stack alternates free dimensions --
    # 64, 16, 4, 64, 16, 4 -- and a single slot would never see a repeat. Bounded
    # for the same reason the plans themselves are.
    #
    # For a torch operand that came through the C entry point this branch is never
    # reached: planned_matmul.h records first sightings itself, by this same rule, and
    # calls the installer only for a product it has seen before or holds a plan for. A
    # call site that builds a new CSR tensor for every call only ever has first
    # sightings, and this function cost it 2 us a call on mkt1.
    seen = state.get(SEEN_ATTR)
    if seen is None:
        state[SEEN_ATTR] = {key}
        return None
    if key not in seen:
        if len(seen) < 2 * MAX_PLANS_PER_OPERAND:
            seen.add(key)
        return None

    plan_kind = _SYMBOL_KINDS.get(symbol_name)
    if plan_kind is None:
        # No plan kind serves this kernel (an SpMV, say), now or on any later call: refuse
        # the product, as a withdrawal does, so later calls settle it with one dict probe
        # -- and for a torch operand, in C without reaching this function at all.
        _refuse(state, plans, key)
        return None
    if kind in ("tilej", "tileijk"):
        # A tiled kernel is only ever chosen in place of v2 at float32 -- the tiled
        # kernels have no float64 instantiation and `tiling_gate` will not offer a
        # float64 product to the selector. Guard anyway rather than trust the caller.
        if plan_kind != "v2":
            return None
        plan_kind = kind
    panel_free, panel_contraction = _panels(plan_kind, param)

    if _is_torch_operand(a):
        # The operand's own arrays: the plan pins what it was built from, and `run_csr`
        # compares it against the operand's members on every call. The factory
        # validates them itself (plan.h). The members, not the accessors' aliases, whose
        # counter is the operand's and misses writes through the caller's arrays; the
        # operand becomes a witness of their storages (_native.torch_sparse_index_members).
        mode_indices = [[], list(_native.torch_sparse_index_members(a))]
        values = a.values()
    else:
        mode_indices = a._native_mode_indices()
        values = a._raw_values
    try:
        plan = _native.make_spmm_csr_plan(
            plan_kind,
            tuple(a.shape),
            mode_indices,
            values,
            int(b.shape[1]),
            panel_free,
            panel_contraction,
        )
    except Exception:
        # Nothing a plan does is load-bearing, so a refusal from the factory --
        # an extent the legacy int32 ABI cannot express, an index structure it
        # will not vouch for -- must not turn a working matmul into an error. Both are
        # properties of the operand, so the refusal is recorded rather than retried on
        # every call (each retry rebuilt the plan's inputs and called the factory).
        _refuse(state, plans, key)
        return None
    if plan is None:
        _refuse(state, plans, key)
        return None
    if plans is None:
        plans = state[PLANS_ATTR] = {}
    plans[key] = plan
    return plan


def _refuse(state: Dict[str, Any], plans: Optional[Dict[Any, Any]], key: Any) -> None:
    """Record that ``key`` will never get a plan on this operand: stored as None, the
    form a withdrawal takes (see `_account_for_a_decline`), which every probe already
    reads as "no plan" and the installer as "settled"."""
    if plans is None:
        plans = state[PLANS_ATTR] = {}
    plans[key] = None


def plans_of(tensor: Any) -> Dict[Any, Any]:
    """The live plans attached to ``tensor`` (empty dict if none). For tests.

    Withdrawn products are held as ``None`` (see :func:`_account_for_a_decline`) and
    are not plans, so they are not reported here; :func:`refused_of` has them.
    """
    held = (_state_of(tensor) or {}).get(PLANS_ATTR) or {}
    return {key: plan for key, plan in held.items() if plan is not None}


def transposed_plans_of(tensor: Any) -> Dict[Any, Any]:
    """The TransposedProductPlans a torch CSR operand holds, by key (empty dict if none). For
    tests."""
    return dict((_state_of(tensor) or {}).get(TRANSPOSED_PLANS_ATTR) or {})


def refused_of(tensor: Any) -> set:
    """The products this operand has been refused a plan for. For tests."""
    held = (_state_of(tensor) or {}).get(PLANS_ATTR) or {}
    return {key for key, plan in held.items() if plan is None}
