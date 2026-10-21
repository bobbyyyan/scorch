"""The assembly stage: which strategy a sparse result nobody requested one for gets.

The stage runs for a program that is about to be lowered, records no assembly
strategy and assembles a sparse result.  The target says which strategies the
program can be emitted with (:func:`~.lower_llir.assembly_choices`), the
automatic rule picks one (:func:`~.assembly_rule.automatic_assembly`), and the
stage returns what it saw and chose as an :class:`AssemblyDecision`, which the
compiled kernel keeps.

Where the strategy is recorded depends on who wrote the plan.  The automatic
scheduler records no assembly strategy; its plans are shared with the shipped
compiler, which has no such decision to replay.  For such a plan the stage
returns the plan with the strategy recorded (:func:`complete_automatic_plan`),
and the pipeline verifies and applies that plan.  A plan a caller wrote without
naming a strategy is left as written, and an unscheduled program has no plan:
for both the stage returns a copy of the program with the strategy recorded
(:func:`choose_unrequested_assembly`), which is the copy target lowering
receives.  The rule is the same function in every case, so a program is
assembled the same way whichever route compiled it.

The rule is total over programs that assemble a sparse result.  A program whose
result has only dense levels assembles nothing sparse; nothing records a
strategy for it, because the strategy names describe a sparse assembly and none
of them describes a dense result written by a parallel loop.  The pipeline does
not begin the stage for such a program (:func:`~.lower_llir.assembles_sparse_result`).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from typing import Mapping, Optional, Tuple

from ..identity import SymbolId
from ..loop_plan import LoopPlan
from ..sparse_assembly import UNSUPPORTED_ASSEMBLY_HOST
from .assembly_rule import (
    AUTOMATIC_PARALLEL_PREFERENCE,
    AUTOMATIC_SERIAL_PREFERENCE,
    PARALLEL_FROM_EXTENT,
    automatic_assembly,
)
from .lower_llir import (
    LoopIRTargetDefect,
    LoopIRTargetError,
    assembly_choices,
    with_recorded_assembly,
)
from .nodes import LoopProgram
from .schedule_passes import ScheduledLoopIR

__all__ = [
    "AUTOMATIC_PARALLEL_PREFERENCE",
    "AUTOMATIC_SERIAL_PREFERENCE",
    "PARALLEL_FROM_EXTENT",
    "AssemblyDecision",
    "automatic_assembly",
    "choose_unrequested_assembly",
    "complete_automatic_plan",
]


@dataclass(frozen=True)
class AssemblyDecision:
    """What the assembly stage saw and what it chose, for one compiled kernel.

    ``host`` is the target family that lowers the program; ``sparse_result``
    whether its result has a level that is not dense; ``emittable`` the
    strategies the target can emit it with; ``split_extent`` the extent of the
    result's first level; ``parallel_from`` the extent from which the rule
    prefers a parallel strategy; ``strategy`` the strategy the kernel is
    assembled with, ``None`` exactly when the result is all dense.  An
    automatic plan records the same strategy in its ``assembly`` field; a plan
    a caller wrote, and an unscheduled program, record it nowhere else.
    """

    host: str
    sparse_result: bool
    emittable: Tuple[str, ...]
    split_extent: Optional[int]
    parallel_from: int
    strategy: Optional[str]

    def dump(self) -> str:
        """The decision as canonical JSON: sorted keys, no spacing."""

        return json.dumps(
            {
                "emittable": list(self.emittable),
                "host": self.host,
                "parallel_from": self.parallel_from,
                "sparse_result": self.sparse_result,
                "split_extent": self.split_extent,
                "strategy": self.strategy,
            },
            sort_keys=True,
            separators=(",", ":"),
        )


def complete_automatic_plan(
    schedule: ScheduledLoopIR,
    plan: LoopPlan,
    *,
    input_shapes: Mapping[SymbolId, Tuple[int, ...]],
    result_shape: Tuple[int, ...],
) -> Tuple[LoopPlan, AssemblyDecision]:
    """``plan`` with its assembly strategy recorded, and the decision's record.

    ``schedule`` is ``plan`` applied to the base program: the strategies a
    program can be emitted with depend on the loops every other decision of
    the plan produced, so the question is asked of the scheduled program.
    ``plan`` must record no strategy (the target refuses the question of a
    program that already carries one, ``invalid_schedule_assembly``).  The
    returned plan is ``plan`` itself when the result is all dense, and a new
    plan otherwise; the caller verifies a new plan and applies it before
    anything is lowered from it.
    """

    choices = assembly_choices(
        schedule.program, input_shapes=input_shapes, result_shape=result_shape
    )
    strategy: Optional[str] = None
    if choices.sparse_result:
        if not choices.emittable:
            raise LoopIRTargetError(
                LoopIRTargetDefect(
                    UNSUPPORTED_ASSEMBLY_HOST,
                    f"{choices.host} reports no assembly strategy for a result "
                    "with a level that is not dense",
                )
            )
        strategy = automatic_assembly(choices.emittable, choices.split_extent)
    decision = AssemblyDecision(
        host=choices.host,
        sparse_result=choices.sparse_result,
        emittable=tuple(choices.emittable),
        split_extent=choices.split_extent,
        parallel_from=PARALLEL_FROM_EXTENT,
        strategy=strategy,
    )
    if strategy is None:
        return plan, decision
    return replace(plan, assembly=strategy), decision


def choose_unrequested_assembly(
    program: LoopProgram,
    *,
    input_shapes: Mapping[SymbolId, Tuple[int, ...]],
    result_shape: Tuple[int, ...],
) -> Tuple[LoopProgram, AssemblyDecision]:
    """``program`` with the rule's strategy recorded, and the decision's record.

    For a program that has no automatic plan to complete: it was not scheduled,
    or its plan was written by a caller who named no strategy.  ``program`` must
    record no strategy and assemble a sparse result.  The returned program is
    the copy target lowering receives; ``program`` itself, and the plan it came
    from, are not changed.  The target entry would record the same strategy on a
    program handed to it without one (:func:`~.lower_llir.with_recorded_assembly`,
    which this calls); going through the stage is what gives the kernel its
    record and the compile its stage time.
    """

    recorded, choices = with_recorded_assembly(
        program, input_shapes=input_shapes, result_shape=result_shape
    )
    if choices is None or recorded.assembly is None:
        raise ValueError(
            "the program records a strategy already or assembles no sparse result"
        )
    return recorded, AssemblyDecision(
        host=choices.host,
        sparse_result=choices.sparse_result,
        emittable=tuple(choices.emittable),
        split_extent=choices.split_extent,
        parallel_from=PARALLEL_FROM_EXTENT,
        strategy=recorded.assembly.value,
    )
