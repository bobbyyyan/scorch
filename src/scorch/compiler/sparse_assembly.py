"""Sparse-output assembly strategy: the vocabulary, and the one legality rule.

A sparse result can be assembled in more than one way, and which way is fastest
depends on the operands' formats and density rather than on which compiler class
happens to host the program.  Four strategies exist, each measured best in a
region of its own
(``~/.cache/scorch-codex/assembly-strategy/DESIGN.md`` collects the receipts):

``single_pass_serial``
    One traversal appending into ``std::vector``.  Fastest at one thread and at
    low density, where a counting pass buys nothing and an OpenMP region costs
    4-10% for a single worker.  Every family can emit it; nothing refuses it.

``single_pass_chunk_parallel``
    One traversal, appending into per-chunk buffers concatenated afterwards in
    chunk order -- which is outer-loop order, which is the required
    lexicographic order.  Carries the single-pass advantage into the parallel
    regime instead of discarding it.
    For a COO result that stores one input's entries
    (:func:`entry_positioned_receiver`) every entry's position is known before
    the loops run, so each worker writes its entries in place and there is
    nothing to concatenate: the shipped compiler's parallel loop over a COO
    input's entries.

``two_pass_serial``
    Count, allocate exactly, fill.  Wins where the output is large relative to
    the merge that produces it, because exact allocation then costs less than
    ``std::vector`` growth -- measured 0.857/0.953 against the single pass on
    ``TTM dss x dd -> dss`` at mid density, before any parallelism.

``two_pass_parallel``
    The same, with both phases parallel.  Exact per-outer-cell offsets make the
    fill pass independent, so it scales better than a chunked single pass on
    hosts where per-chunk buffer growth and concatenation are expensive.

**This module holds the vocabulary and the receiver contracts, once.**  Two
copies of one scheduling rule drifting apart is the defect that cost twelve
cells when the automatic tile heuristic was duplicated between the scheduler and
the legality boundary, so the token set and
:func:`partitionable_receiver_levels` have exactly one definition and every
layer imports them.

Legality and cost are deliberately different questions in different places:

* **legality** -- can this receiver be assembled this way at all?  Structural,
  provable, extent-free, compile-time, and it fails closed with a code.  It
  lives here and in the two layers that own its remaining inputs.
* **cost** -- should we?  A judgement about whether a legal transformation pays:
  the runtime thread gate, the chunk width, and the compile-time extent decline
  that keeps a gate out of a kernel too small to use it.  Those belong to the
  target's default choice and, next, to a selector.  Nothing in this module
  consults an extent, a density, a thread count or a measurement.

The tokens name assembly *structure*.  They do not name OpenMP spelling, a
thread count, a chunk width or a schedule clause: those are target-lowering
decisions and appear in neither the public schedule, the plan, nor LoopIR.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Optional, Sequence, Tuple

from ..format import LevelType

#: Every assembly strategy the compiler can be asked for, in the order the
#: design document names them.  A plan or public schedule recording ``None``
#: records *no* strategy decision, which is not the same as recording
#: ``single_pass_serial``: ``None`` defers to the target's own choice, and that
#: choice is per-receiver.
SPARSE_ASSEMBLY_STRATEGIES: Tuple[str, ...] = (
    "single_pass_serial",
    "single_pass_chunk_parallel",
    "two_pass_serial",
    "two_pass_parallel",
)

#: The strategies that assemble in one traversal.
SINGLE_PASS_STRATEGIES: Tuple[str, ...] = (
    "single_pass_serial",
    "single_pass_chunk_parallel",
)

#: The strategies that count, allocate exactly, then fill.
TWO_PASS_STRATEGIES: Tuple[str, ...] = (
    "two_pass_serial",
    "two_pass_parallel",
)

#: The strategies that may distribute the assembly across workers.  Each emits
#: its own gate; a closed gate is not the same kernel as the serial strategy,
#: which is why these are four strategies and not two orthogonal flags.
PARALLEL_ASSEMBLY_STRATEGIES: Tuple[str, ...] = (
    "single_pass_chunk_parallel",
    "two_pass_parallel",
)

#: The strategy every receiver can be assembled with.
DEFAULT_SERIAL_STRATEGY = "single_pass_serial"

# -- structured refusal codes -------------------------------------------------

#: Schedule-time: the recorded strategy is not a strategy, or is recorded on a
#: plan whose provenance may not carry one.
INVALID_SCHEDULE_ASSEMBLY = "invalid_schedule_assembly"

#: Schedule-time: a well-formed strategy the receiver's *shape* cannot support,
#: or one composed with a schedule decision that owns part of the same assembly.
UNSUPPORTED_SCHEDULE_ASSEMBLY = "unsupported_schedule_assembly"

#: Target-time: the receiver admits the strategy but this program does not.
UNSUPPORTED_ASSEMBLY_STRATEGY = "unsupported_assembly_strategy"

#: Target-time: the strategy is legal here and the family hosting the program
#: has no emission for it.  Fail-closed insurance for a family added later; a
#: gate requires it to be unreachable over the legal domain.
UNSUPPORTED_ASSEMBLY_HOST = "unsupported_assembly_host"


def is_assembly_strategy(strategy: object) -> bool:
    """Whether ``strategy`` is one of the exact recorded strategy tokens."""

    return type(strategy) is str and strategy in SPARSE_ASSEMBLY_STRATEGIES


def partitionable_receiver_levels(levels: Sequence[LevelType]) -> bool:
    """The one receiver contract every non-serial strategy requires.

    Level 0 dense and every level below it compressed -- equivalently,
    ``compressed_levels == (1, 2, ..., rank - 1)``.  Both existing
    implementations arrived at this independently: the shared two-phase pass
    requires exactly that tuple, and the per-chunk transformation requires a
    dense prefix of one with everything below compressed.  They are the same
    set, and the reason is the same in both cases -- the first compressed
    level's position array is indexed by the outer loop over a statically known
    range, so the assembly can be split by outer cell.

    A dense prefix DEEPER than one is excluded: the first compressed level's
    position array is then indexed by a flattened dense cell, so the pre-size,
    the per-worker starting index and the shift range all become products of the
    dense extents.  The mechanism generalizes and the derivation needs its own
    statement and its own measurement, so it is declined rather than guessed at.

    A STORED prefix is excluded permanently rather than pending work: there is
    no dense extent to split, and the outer loop's stored coordinates do not
    partition one.

    Extents, densities and thread counts are deliberately absent.  This
    predicate answers only whether the assembly *can* be split, never whether
    splitting pays.
    """

    if type(levels) not in (tuple, list):
        return False
    if len(levels) < 2:
        return False
    if levels[0] is not LevelType.DENSE:
        return False
    return all(level is LevelType.COMPRESSED for level in levels[1:])


@dataclass(frozen=True)
class AccessLayout:
    """One tensor access, as the receiver rules below read it.

    ``tensor`` is the accessed tensor's identity; ``logical`` the access's index
    identities in the tensor's logical mode order; ``storage`` the same
    identities in its storage-level order; ``levels`` the level types in storage
    order.  Built from a CIN access by the front end and from a CIN analysis
    record by plan legality, so both apply the same rule to the same facts.
    """

    tensor: Hashable
    logical: Tuple[Hashable, ...]
    storage: Tuple[Hashable, ...]
    levels: Tuple[LevelType, ...]


def entry_source(
    result: AccessLayout,
    factors: Optional[Sequence[AccessLayout]],
    loop_order: Sequence[Hashable],
) -> Optional[Hashable]:
    """The input whose stored entries are a COO result's entries, if one is.

    ``factors`` are the accesses of a right-hand side that is a product of tensor
    accesses, ``None`` for any other right-hand side; ``loop_order`` the loops,
    outermost first.  One input qualifies when:

    * the result is all COORDINATE levels;
    * the input has exactly the result's indices and every other factor is
      dense along each of them -- the shipped compiler's rule
      (:func:`scorch.compiler.cin.coordinate_result_driver`, the first such
      factor) -- so the loops over the result's indices walk that input's
      stored entries, one result entry each;
    * it has them in the result's logical order and in the result's storage
      order, so its entry ``p`` is the result's entry ``p``;
    * the loops over the result's indices are the outermost ones, in storage
      order, with a reduction loop below them.

    A schedule that names that input (``Schedule.entries_of``, recorded as
    ``LoopPlan.entries_of`` and ``LoopProgram.entries_of``) makes the result
    store every entry of it, one whose sum is empty included, and each entry's
    position is then known before the loops run: the input's own.  Nothing
    reads this rule unless a schedule names the input.
    """

    if not result.levels or any(
        level is not LevelType.COORDINATE for level in result.levels
    ):
        return None
    if factors is None:
        return None
    result_indices = set(result.logical)
    candidate: Optional[AccessLayout] = None
    for factor in factors:
        if set(factor.logical) != result_indices:
            continue
        if all(
            level is LevelType.DENSE
            for other in factors
            if other is not factor
            for index, level in zip(other.storage, other.levels)
            if index in result_indices
        ):
            candidate = factor
            break
    if candidate is None:
        return None
    if candidate.logical != result.logical or candidate.storage != result.storage:
        return None
    rank = len(result.storage)
    if len(loop_order) <= rank or tuple(loop_order[:rank]) != result.storage:
        return None
    return candidate.tensor


def entry_positioned_receiver(
    levels: Sequence[LevelType], stores_input_entries: bool
) -> bool:
    """The receiver contract ``single_pass_chunk_parallel`` also admits.

    A COO result that stores one input's entries (:func:`entry_source`, which a
    schedule requests as ``entries_of``): every entry's position is that
    input's, known before the loops run, so one traversal splits across
    workers with each writing its entries in place and nothing to concatenate.
    Plan legality also requires that input to be a COO tensor of two levels or
    more.  The two-pass strategies number segments of a compressed level and
    still require :func:`partitionable_receiver_levels`.
    """

    return (
        stores_input_entries
        and bool(levels)
        and all(level is LevelType.COORDINATE for level in levels)
    )


def receiver_requirement(strategy: str) -> str:
    """The receiver ``strategy`` needs, in words, for every refusal that names it.

    One wording for the four boundaries that refuse a strategy for its receiver
    (the explicit schedule surface, plan legality, schedule application and the
    LoopIR verifier), so a user is told the same requirement whichever refuses.
    The parallel single pass admits two receivers
    (:func:`partitionable_receiver_levels` and :func:`entry_positioned_receiver`
    with :func:`entry_source`); every other strategy admits the first.
    """

    partitionable = (
        "a result whose level zero is dense with every level below it compressed"
    )
    if strategy != "single_pass_chunk_parallel":
        return partitionable
    return (
        f"{partitionable}, or a COO result that stores the entries of a COO input "
        "of two levels or more, named by entries_of (the input has exactly the "
        "result's indices, in the result's logical and storage order, every other "
        "factor is dense along them, and the loops over them are the outermost "
        "loops, unsplit and in storage order, with a loop below them)"
    )


def compressed_levels_of(levels: Sequence[LevelType]) -> Tuple[int, ...]:
    """``(1, ..., rank - 1)`` for a partitionable receiver, else ``()``.

    The shared two-phase pass takes exactly this tuple and validates that it has
    exactly this shape, so deriving it here keeps one definition of the
    correspondence between a receiver's levels and that pass's contract.
    """

    if not partitionable_receiver_levels(levels):
        return ()
    return tuple(range(1, len(levels)))
