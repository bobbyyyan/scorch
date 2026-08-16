"""Sparse accumulation structure: the vocabulary, and the one legality rule.

A program that reduces into a sparse result accumulates through a workspace
keyed by coordinate, and more than one structure can hold that accumulation.
Two exist here, and which one is faster depends on the density and on the
receiver's compressed extent rather than on which compiler layer happens to
declare it -- measured over 64 configurations in
``~/.cache/scorch-codex/workspace-confound/`` and summarized in review section
67.3: the swap alone moves the emitted kernel's runtime between 0.872x and
1.574x, it changes SIGN with density, and it is outside its own same-binary A/A
floor on 56 of those 64.

``coordinate_list``
    The keys inserted are recorded as a list, and the drain reads them in sorted
    key order.  Realized as ``coo_workspace_1d<T,1>`` for a single-component key
    -- a value array indexed by the coordinate, a byte flag per coordinate, and
    an append-only list of the coordinates touched -- and as
    ``coo_workspace<T,K>`` for a K-component key, which flattens the key, dedups
    through a hash map and keeps an entry vector.  Both start at capacity 1024
    and grow on demand, so construction costs nothing that scales with the
    receiver.  This is what every family declares for itself.

``linked_list``
    Gustavson's accumulator: a value array and a ``next`` array, both indexed by
    the coordinate and both sized to its full extent, with the coordinates
    touched threaded through ``next`` as a chain the drain walks and sorts.
    Insertion is branchless and needs no bounds check, because the arrays
    already span the key's extent.  Realized as ``linked_list_workspace_1d<T>``.
    This is what the two-phase parallel assembly transform substitutes.

**This module holds the vocabulary once.**  Two copies of one scheduling rule
drifting apart is the defect that cost twelve cells when the automatic tile
heuristic was duplicated between the scheduler and the legality boundary, and it
is why :mod:`scorch.compiler.sparse_assembly` was written the same way: the token
set and :func:`single_coordinate_key` have exactly one definition and every layer
imports them.

Legality and cost are deliberately different questions in different places:

* **legality** -- can this program use this structure at all?  Structural,
  provable, extent-free, compile-time, and it fails closed with a code.  It lives
  here and in the layers that own its remaining inputs.
* **cost** -- should it?  Which structure pays depends on the density, on the
  receiver's compressed extent and partly on the worker count (section 67.4
  separates the two components), and those inputs belong to a cost model.  The
  seam is the target's ``default_accumulator()`` and, next, a selector.  Nothing
  in this module consults an extent, a density, a thread count or a measurement.

The tokens name accumulation *structure*.  They do not name a container, a
capacity, a pool, a thread count or an insert spelling.  In particular **pooling
is not a second axis** and the reason is worth stating, because the four-way
product looks plausible.  A workspace declared inside a parallel loop body is
private to the iteration by construction, so a pool is never needed for
correctness; it is needed when a structure is too expensive to construct per
iteration and must therefore be hoisted out of the loop.  ``linked_list``'s
constructor allocates and fills arrays of the receiver's compressed extent, so it
must be hoisted, so under a region it must be one per worker: pooling is entailed
by that structure rather than chosen beside it.  ``coordinate_list``'s
constructor is independent of the receiver, so hoisting buys nothing and nothing
pools it.  Of the four cells the product would admit, two have producers, a
pooled coordinate list is a candidate fix nobody has built, and a per-iteration
linked list has no producer and no reason to want one -- so this is an
enumeration of tokens, not a product of two booleans, exactly as the four
assembly strategies are.  A pooled coordinate list becomes a THIRD token when
somebody builds it.

The dense accumulator is deliberately absent.  In LoopIR the dense and sparse
accumulators are different node kinds (``WorkspaceDecl`` against
``SparseWorkspaceDecl``), and on the legacy side the CIN's ``dense`` flag selects
a different lowering entirely -- a per-worker slab with a ``memset`` per
iteration, pooled by the lowerer itself.  Choosing between them is choosing a
different program, not scheduling this one.
"""

from __future__ import annotations

from typing import Tuple

#: Every sparse accumulation structure the compiler can be asked for.  A plan or
#: public schedule recording ``None`` records *no* structure decision, which is
#: not the same as recording ``coordinate_list``: ``None`` defers to whatever the
#: existing layers choose, and where the two-phase transform fires that is
#: ``linked_list``.
SPARSE_ACCUMULATOR_STRUCTURES: Tuple[str, ...] = (
    "coordinate_list",
    "linked_list",
)

#: The structure every family that has a sparse workspace declares for itself.
DECLARED_ACCUMULATOR_STRUCTURE = "coordinate_list"

#: The structure the two-phase parallel assembly transform substitutes when the
#: workspace declaration is a direct child of the loop it replaces.
CHAINED_ACCUMULATOR_STRUCTURE = "linked_list"

# -- structured refusal codes -------------------------------------------------

#: Schedule-time: the recorded structure is not a structure, or is recorded on a
#: plan whose provenance may not carry one.
INVALID_SCHEDULE_ACCUMULATOR = "invalid_schedule_accumulator"

#: Schedule-time: a well-formed structure this program's shape cannot support.
UNSUPPORTED_SCHEDULE_ACCUMULATOR = "unsupported_schedule_accumulator"

#: Target- or pass-time: the family admits the structure and this program does
#: not -- the key has more than one component, or the emitted shape gives the
#: transform nothing to substitute.
UNSUPPORTED_ACCUMULATOR_STRUCTURE = "unsupported_accumulator_structure"

#: Target-time: the structure is legal here and the family hosting the program
#: has no emission for it.  A refusal that names the host beats an internal
#: completion failure three layers down.
UNSUPPORTED_ACCUMULATOR_HOST = "unsupported_accumulator_host"


def is_accumulator_structure(structure: object) -> bool:
    """Whether ``structure`` is one of the exact recorded structure tokens."""

    return type(structure) is str and structure in SPARSE_ACCUMULATOR_STRUCTURES


def single_coordinate_key(key_rank: object) -> bool:
    """The one contract the chained accumulator requires: one key component.

    The chain lives in a ``next`` array indexed by the key itself, so a key of
    more than one component has no single array to index -- the flattening a
    multi-component container does would have to materialize the product of the
    key extents, which is a different structure with a different cost and would
    need its own derivation and its own measurement.

    Extents are deliberately absent.  This predicate requires the key to HAVE a
    bounded extent; it never reads one, and it never asks whether an extent is
    small enough for the chain to pay.  That is cost.
    """

    return type(key_rank) is int and key_rank == 1
