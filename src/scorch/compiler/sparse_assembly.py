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

``two_pass_serial``
    Count, allocate exactly, fill.  Wins where the output is large relative to
    the merge that produces it, because exact allocation then costs less than
    ``std::vector`` growth -- measured 0.857/0.953 against the single pass on
    ``TTM dss x dd -> dss`` at mid density, before any parallelism.

``two_pass_parallel``
    The same, with both phases parallel.  Exact per-outer-cell offsets make the
    fill pass independent, so it scales better than a chunked single pass on
    hosts where per-chunk buffer growth and concatenation are expensive.

**This module holds the vocabulary and the receiver contract, once.**  Two
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

from typing import Sequence, Tuple

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


def compressed_levels_of(levels: Sequence[LevelType]) -> Tuple[int, ...]:
    """``(1, ..., rank - 1)`` for a partitionable receiver, else ``()``.

    The shared two-phase pass takes exactly this tuple and validates that it has
    exactly this shape, so deriving it here keeps one definition of the
    correspondence between a receiver's levels and that pass's contract.
    """

    if not partitionable_receiver_levels(levels):
        return ()
    return tuple(range(1, len(levels)))
