"""The automatic rule for a sparse result's assembly strategy.

One definition, read by the two places that choose a strategy nobody
requested: the assembly stage, which records the rule's answer in an automatic
plan (:mod:`.assembly_selection`), and the target entry, which applies the same
answer to any program that reaches it without a recorded strategy -- an
unscheduled program, or an explicit plan that names none
(:func:`.lower_llir.lower_loopir_to_llir`).  A target family holds no default of
its own, so a program is assembled the same way whichever route compiled it.

The rule reads two facts and no others: the strategies the target can emit the
program with, and the extent of the result's first level.  It reads no density
and no thread count.
"""

from __future__ import annotations

from typing import Optional, Sequence

from ..sparse_assembly import SPARSE_ASSEMBLY_STRATEGIES
from .parallel_chunk_assembly import ROWS_PER_THREAD

#: The first-level extent from which the rule prefers a parallel strategy.  A
#: result with fewer coordinates cannot reach two threads for any operand:
#: ``scorch_nthreads``'s row term alone caps the count at one below
#: ``2 * ROWS_PER_THREAD`` rows, so a parallel strategy could only cost that
#: kernel a gate, a counting pass or an OpenMP region and never pay for it.
PARALLEL_FROM_EXTENT = 2 * ROWS_PER_THREAD

#: The strategies that distribute an assembly across workers, in the order the
#: rule prefers them.  The two-pass parallel assembly comes first: it is the
#: strategy the shipped compiler emits for these results, so wherever the
#: target can emit it the new compiler's kernel has the shipped kernel's
#: schedule (and, for most programs, its text).  The chunked single pass is a
#: different kernel whose only two-host runtime measurement (review section
#: 59.9, TTM) has it faster than the two-pass on some shapes and slower than
#: the serial kernel on others, with no condition yet that separates them; the
#: rule records it only where the two-pass cannot be emitted, and a plan can
#: request it anywhere it is legal.
AUTOMATIC_PARALLEL_PREFERENCE = ("two_pass_parallel", "single_pass_chunk_parallel")

#: The strategies that run on one thread, in the order the rule prefers them
#: below :data:`PARALLEL_FROM_EXTENT` and wherever no parallel strategy can be
#: emitted.  The single pass is the kernel an automatic plan emitted for these
#: results before plans recorded a strategy.
AUTOMATIC_SERIAL_PREFERENCE = ("single_pass_serial", "two_pass_serial")


def automatic_assembly(emittable: Sequence[str], split_extent: Optional[int]) -> str:
    """The strategy a sparse result is assembled with when none was requested.

    ``emittable`` is every strategy the target can emit the program with and
    must name at least one; ``split_extent`` is the extent of the result's
    first level (``None`` for a result with no level).  From
    :data:`PARALLEL_FROM_EXTENT` coordinates the first emittable strategy of
    :data:`AUTOMATIC_PARALLEL_PREFERENCE`; otherwise, and where neither can be
    emitted, the first emittable one of :data:`AUTOMATIC_SERIAL_PREFERENCE`;
    otherwise the first emittable strategy in the vocabulary's order, which is
    the only one for a family that emits a single parallel strategy.
    """

    if not emittable:
        raise ValueError("the rule needs at least one emittable strategy")
    if split_extent is not None and split_extent >= PARALLEL_FROM_EXTENT:
        for strategy in AUTOMATIC_PARALLEL_PREFERENCE:
            if strategy in emittable:
                return strategy
    for strategy in AUTOMATIC_SERIAL_PREFERENCE:
        if strategy in emittable:
            return strategy
    return next(s for s in SPARSE_ASSEMBLY_STRATEGIES if s in emittable)
