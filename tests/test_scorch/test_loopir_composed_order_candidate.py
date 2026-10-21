"""The composed loop order as a second automatic candidate on the LoopIR route.

``Scheduler.select_loop_order`` first composes an order (initial order, cost-model
optimisation, mode-order constraints) and then, for a sparse result in a program
with a reduction, moves the last result index innermost when no result index
follows the last reduced one.  That forced reorder exists because legacy lowering
cannot lower a reduction innermost into a sparse result.  Its output is checked by
the plan's legality rules, but those check the parent-before-child order only of
tensors with a compressed level.  A forced order that reads an all-dense operand
against its storage order therefore verifies although the composed order reads
every tensor in storage order: SDDMM into CSR or DCSR, and reductions of a dense
operand into a compressed result, are such programs.  When this candidate was
added the LoopIR lowering refused such a forced order (``unsupported_loop_order``
in ``loopir/lower_cin.py``); it now lowers it, reading the dense operand by its
coordinates, so the candidate no longer decides whether these programs compile.
It decides which order they compile in.

``Scheduler.apply_schedule(..., composed_order_candidate=True)``, which only the
LoopIR pipeline passes, makes an automatic plan take the composed order exactly
when the forced reorder changed the order, the forced order visits some tensor's
levels out of storage order, the composed order visits every tensor's levels in
storage order, and the plan re-originated from the composed order verifies.
Every other caller leaves it off: legacy lowering refuses most of those composed
orders (``unscheduled_sparse_result_reduction``), so the shipped compiler's plans
must not move.  Both halves are locked here, as is execution against dense
PyTorch in float32 and float64.  Without the candidate these programs now compile
in the forced order, and that kernel is locked and executed too (float64, stored
coordinates exact).
"""

from dataclasses import replace

import pytest
import torch

from scorch.compiler.cin import (
    BinaryOp as CINBinaryOp,
    ForAll,
    IndexVar,
    Operation,
    TensorAssign,
    TensorVar,
)
from scorch.compiler.cin_analysis import normalize_cin
from scorch.compiler.compile_options import CompileOptions
from scorch.compiler.loopir.pipeline import (
    compile_cin_via_loopir,
    execute_cin_via_loopir,
    legacy_generated_cpp,
)
from scorch.compiler.scheduler import Schedule, Scheduler
from scorch.stensor import STensor
from tests.test_scorch.loopir_executed_programs import (
    assert_program_matches_pytorch,
)

EXTENT = {"i": 5, "j": 4, "k": 3, "l": 2, "m": 2, "n": 3}

# (name, operands as (symbol, format, indices), result format, result indices).
# Every operand's indices are listed in the nest's order, so the first order the
# program is written in is the storage order of every tensor.
REDUCTIONS = [
    ("dd ij->i [s]", (("A", "dd", "ij"),), "s", "i"),
    ("ddd ijk->i [s]", (("A", "ddd", "ijk"),), "s", "i"),
    ("ddd ijk->ij [ss]", (("A", "ddd", "ijk"),), "ss", "ij"),
    ("ddd ijk->j [s]", (("A", "ddd", "ijk"),), "s", "j"),
]
# ``ddd ijk->ij [ds]`` meets all four conditions too, but its composed order then
# reaches the canonical-CSR receiver, which this tree's sparse-output routing
# still refuses (``unsupported_sparse_output_reduction``).  It is not listed.
SDDMM = [
    (
        f"SDDMM ss x dd x dd -> {c_fmt}",
        (("A", "ss", "ij"), ("B", "dd", "ik"), ("D", "dd", "jk")),
        c_fmt,
        "ij",
    )
    for c_fmt in ("ss", "ds")
]
MIGRATED = REDUCTIONS + SDDMM

# Programs whose forced AND composed orders both read a dense factor against its
# storage order, while some order reads every tensor in storage order: the third
# candidate, the composed order re-sorted by ``Scheduler._storage_order_sort``.
SORTED = [
    (
        f"TTMc sss x dd x dd -> {c_fmt}",
        (("A", "sss", "ijk"), ("B", "dd", "jm"), ("D", "dd", "kn")),
        c_fmt,
        "imn",
    )
    for c_fmt in ("dss", "sss")
]

# Programs whose forced reorder changes the order but whose forced order already
# keeps every tensor in storage order: the composed order must not be taken.
FORCED_ORDER_KEPT = [
    ("MM ss x dd -> ss", (("A", "ss", "ik"), ("B", "dd", "kj")), "ss", "ij"),
    ("MM ds x dd -> ss", (("A", "ds", "ik"), ("B", "dd", "kj")), "ss", "ij"),
    (
        "MTTKRP sss x dd x dd -> ss",
        (("A", "sss", "ijk"), ("B", "dd", "jl"), ("D", "dd", "kl")),
        "ss",
        "il",
    ),
]

# Programs whose forced and composed orders BOTH read some tensor against its
# storage order: there is nothing better to take, and the forced order stays.
NEITHER_ORDER_KEEPS_STORAGE = [
    ("ddd ijk->ji [ss]", (("A", "ddd", "ijk"),), "ss", "ji"),
    ("ddd ijk->ji [ds]", (("A", "ddd", "ijk"),), "ds", "ji"),
]


def cell_id(cell):
    return cell[0]


def auto_options(regblock_enabled, *, jit=False):
    base = (
        CompileOptions.from_environment()
        if jit
        else CompileOptions.from_environment(environ={})
    )
    return replace(
        base.with_regblock_enabled(regblock_enabled),
        requested_schedule=Schedule(),
    )


def nest_order(operands, result_indices):
    order = []
    for _, _, indices in operands:
        for name in indices:
            if name not in order:
                order.append(name)
    for name in result_indices:
        if name not in order:
            order.append(name)
    return order


def build_cin(cell, dtype):
    _, operands, result_fmt, result_indices = cell
    order = nest_order(operands, result_indices)
    ivars = {name: IndexVar(name) for name in order}
    rhs = None
    for symbol, fmt, indices in operands:
        access = TensorVar(symbol, fmt=fmt, dtype=dtype)[
            tuple(ivars[name] for name in indices)
        ]
        rhs = access if rhs is None else CINBinaryOp(Operation.MUL, rhs, access)
    result = TensorVar("C", fmt=result_fmt, dtype=dtype)[
        tuple(ivars[name] for name in result_indices)
    ]
    stmt = TensorAssign(result, rhs, op=Operation.ADD)
    for name in reversed(order):
        stmt = ForAll(ivars[name], stmt)
    return stmt


def bindings(cell, dtype):
    _, operands, _, result_indices = cell
    return (
        tuple(
            (tuple(EXTENT[name] for name in indices), dtype)
            for _, _, indices in operands
        ),
        tuple(EXTENT[name] for name in result_indices),
    )


def candidate_orders(cin, options):
    """``select_loop_order``'s forced and composed orders, as index IDs."""

    composed = []
    forced = Scheduler.select_loop_order(
        normalize_cin(cin, compile_options=options),
        costs=options.scheduler.cost_model,
        pre_forced_order=composed,
    )
    return [v.index_id for v in forced], [v.index_id for v in composed]


def storage_orders(cin):
    """Each tensor's index IDs in storage order (indices permuted by mode order)."""

    return [
        [v.index_id for v in access.get_sorted_index_vars()]
        for access in normalize_cin(
            cin, compile_options=auto_options(False)
        ).tensor_accesses
    ]


def keeps_every_storage_order(order, cin):
    position = {index_id: n for n, index_id in enumerate(order)}
    for levels in storage_orders(cin):
        visited = [position[i] for i in levels if i in position]
        if visited != sorted(visited):
            return False
    return True


# -- which order each route records ------------------------------------------


@pytest.mark.parametrize("cell", MIGRATED, ids=cell_id)
@pytest.mark.parametrize("arm", [False, True], ids=["direct", "regblock"])
def test_the_preconditions_hold_for_every_migrated_program(cell, arm):
    """Measured, not assumed: all three order conditions hold for each program."""

    cin = build_cin(cell, torch.float32)
    forced, composed = candidate_orders(cin, auto_options(arm))
    assert forced != composed
    assert not keeps_every_storage_order(forced, cin)
    assert keeps_every_storage_order(composed, cin)


@pytest.mark.parametrize("cell", MIGRATED, ids=cell_id)
@pytest.mark.parametrize("arm", [False, True], ids=["direct", "regblock"])
def test_the_loopir_route_records_the_composed_order(cell, arm):
    cin = build_cin(cell, torch.float32)
    options = auto_options(arm)
    forced, composed = candidate_orders(cin, options)
    operand_bindings, result_shape = bindings(cell, torch.float32)
    kernel = compile_cin_via_loopir(
        cin, result_shape, operand_bindings, compile_options=options
    )
    assert kernel.schedule.plan.provenance == "auto"
    assert list(kernel.schedule.plan.loop_order) == composed


@pytest.mark.parametrize("cell", MIGRATED, ids=cell_id)
@pytest.mark.parametrize("arm", [False, True], ids=["direct", "regblock"])
def test_the_shared_boundary_still_records_the_forced_order(cell, arm):
    """``apply_schedule`` without the flag is the shipped route's plan origin.

    ``ops.py`` reaches it for every scheduled public call, so its plan for these
    programs must stay the forced order: legacy lowering cannot lower most of the
    composed orders.
    """

    cin = build_cin(cell, torch.float32)
    options = auto_options(arm)
    forced, composed = candidate_orders(cin, options)
    scheduled = Scheduler.apply_schedule(cin, Schedule(), compile_options=options)
    assert list(scheduled.verified_loop_plan.loop_order) == forced
    assert forced != composed


@pytest.fixture
def without_the_candidate(monkeypatch):
    """The LoopIR route with the composed-order candidate switched off."""

    real = Scheduler.apply_schedule

    def without_candidate(*args, **kwargs):
        kwargs["composed_order_candidate"] = False
        return real(*args, **kwargs)

    monkeypatch.setattr(Scheduler, "apply_schedule", staticmethod(without_candidate))


@pytest.mark.parametrize("cell", MIGRATED + SORTED, ids=cell_id)
@pytest.mark.parametrize("arm", [False, True], ids=["direct", "regblock"])
def test_without_the_candidate_the_loopir_route_lowers_the_forced_order(
    cell, arm, without_the_candidate
):
    """The control: what the candidate changes is the order, not admission.

    Without it the plan keeps the forced order, which reads a dense operand
    against its storage order.  The lowering refused that order
    (``unsupported_loop_order``) when the candidate was added; it now reads an
    all-dense operand by its coordinates in any loop order, so the forced order
    compiles, with no operand converted, to a kernel other than the one the
    candidate's order compiles to.
    """

    cin = build_cin(cell, torch.float32)
    options = auto_options(arm)
    forced, composed = candidate_orders(cin, options)
    operand_bindings, result_shape = bindings(cell, torch.float32)
    kernel = compile_cin_via_loopir(
        cin, result_shape, operand_bindings, compile_options=options
    )
    assert kernel.schedule.plan.provenance == "auto"
    assert list(kernel.schedule.plan.loop_order) == forced
    assert forced != composed
    assert not kernel.conversions
    # The shipped compiler lowers the same forced order to other text (measured
    # on every program here), so this kernel is the new compiler's own.
    shipped = legacy_generated_cpp(
        cin, result_shape, operand_bindings, compile_options=options
    )
    assert kernel.cpp_source != shipped


@pytest.mark.parametrize("cell", MIGRATED + SORTED, ids=cell_id)
@pytest.mark.parametrize("arm", [False, True], ids=["direct", "regblock"])
def test_the_candidate_changes_the_kernel(cell, arm, request):
    """With the candidate the same request compiles to different text."""

    cin = build_cin(cell, torch.float32)
    options = auto_options(arm)
    operand_bindings, result_shape = bindings(cell, torch.float32)
    with_candidate = compile_cin_via_loopir(
        cin, result_shape, operand_bindings, compile_options=options
    )
    request.getfixturevalue("without_the_candidate")
    forced_order = compile_cin_via_loopir(
        cin, result_shape, operand_bindings, compile_options=options
    )
    assert list(with_candidate.schedule.plan.loop_order) != list(
        forced_order.schedule.plan.loop_order
    )
    assert with_candidate.cpp_source != forced_order.cpp_source


@pytest.mark.parametrize("cell", FORCED_ORDER_KEPT, ids=cell_id)
@pytest.mark.parametrize("arm", [False, True], ids=["direct", "regblock"])
def test_a_forced_order_that_keeps_storage_order_is_kept(cell, arm):
    """The composed order of these programs reads the dense factor against its
    storage order; the forced one is the good order and stays."""

    cin = build_cin(cell, torch.float32)
    options = auto_options(arm)
    forced, composed = candidate_orders(cin, options)
    assert forced != composed
    assert keeps_every_storage_order(forced, cin)
    assert not keeps_every_storage_order(composed, cin)
    operand_bindings, result_shape = bindings(cell, torch.float32)
    kernel = compile_cin_via_loopir(
        cin, result_shape, operand_bindings, compile_options=options
    )
    assert list(kernel.schedule.plan.loop_order) == forced


@pytest.mark.parametrize("cell", NEITHER_ORDER_KEEPS_STORAGE, ids=cell_id)
def test_a_composed_order_that_breaks_storage_order_is_not_taken(cell):
    """Condition three: the forced order breaks storage order, and so does the
    composed one, so the plan keeps the forced order."""

    cin = build_cin(cell, torch.float32)
    options = auto_options(False)
    forced, composed = candidate_orders(cin, options)
    assert forced != composed
    assert not keeps_every_storage_order(forced, cin)
    assert not keeps_every_storage_order(composed, cin)
    scheduled = Scheduler.apply_schedule(
        cin, Schedule(), compile_options=options, composed_order_candidate=True
    )
    assert list(scheduled.verified_loop_plan.loop_order) == forced


def test_a_composed_plan_that_does_not_verify_leaves_the_forced_one(monkeypatch):
    """Condition four: if the composed order's plan is refused, nothing changes."""

    monkeypatch.setattr(
        Scheduler, "_from_pre_forced_order", staticmethod(lambda *a, **k: None)
    )
    cell = MIGRATED[0]
    cin = build_cin(cell, torch.float32)
    options = auto_options(False)
    forced, _ = candidate_orders(cin, options)
    scheduled = Scheduler.apply_schedule(
        cin, Schedule(), compile_options=options, composed_order_candidate=True
    )
    assert list(scheduled.verified_loop_plan.loop_order) == forced


@pytest.mark.parametrize("flag", [1, None, "yes"], ids=repr)
def test_the_flag_must_be_an_exact_bool(flag):
    cin = build_cin(MIGRATED[0], torch.float32)
    with pytest.raises(TypeError, match="must be an exact bool"):
        Scheduler.apply_schedule(
            cin,
            Schedule(),
            compile_options=auto_options(False),
            composed_order_candidate=flag,
        )


def test_storage_order_follows_the_mode_order():
    """A column-major dense operand's storage order is (j, i), not (i, j)."""

    i, j = IndexVar("i"), IndexVar("j")
    column_major = TensorVar("A", fmt="dd", dtype=torch.float32, mode_order=[1, 0])
    stmt = ForAll(
        i,
        ForAll(
            j,
            TensorAssign(
                TensorVar("C", fmt="s", dtype=torch.float32)[i],
                column_major[i, j],
                op=Operation.ADD,
            ),
        ),
    )
    assert Scheduler._visits_every_tensor_in_storage_order(stmt, [j, i])
    assert not Scheduler._visits_every_tensor_in_storage_order(stmt, [i, j])
    # An index the order does not bind is ignored.
    assert Scheduler._visits_every_tensor_in_storage_order(stmt, [j])


# -- execution against dense PyTorch -----------------------------------------


def operand_values(shape, dtype, fill, seed):
    generator = torch.Generator().manual_seed(seed)
    values = 0.5 + torch.rand(shape, generator=generator, dtype=torch.float64)
    keep = torch.rand(shape, generator=generator, dtype=torch.float64) < fill
    return (values * keep).to(dtype)


def reference(cell, dense):
    _, operands, _, result_indices = cell
    spec = ",".join(indices for _, _, indices in operands) + "->" + result_indices
    return torch.einsum(spec, *(d.to(torch.float64) for d in dense))


@pytest.mark.parametrize("cell", MIGRATED + SORTED, ids=cell_id)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=str)
@pytest.mark.parametrize("fill", [0.55, 1.0], ids=["fill0.55", "fill1"])
def test_execution_matches_dense_pytorch(cell, dtype, fill):
    _, operands, _, result_indices = cell
    dense = [
        operand_values(
            tuple(EXTENT[name] for name in indices),
            dtype,
            # Dense operands keep every entry; only stored-sparse ones are thinned.
            fill if "s" in fmt else 1.0,
            seed,
        )
        for seed, (_, fmt, indices) in enumerate(operands)
    ]
    tensors = [
        STensor.from_torch(d.clone(), symbol).to_sparse(fmt)
        for d, (symbol, fmt, _) in zip(dense, operands)
    ]
    result, kernel = execute_cin_via_loopir(
        build_cin(cell, dtype),
        tuple(EXTENT[name] for name in result_indices),
        *tensors,
        compile_options=auto_options(False, jit=True),
    )
    want = reference(cell, dense)
    assert bool((want != 0).any())
    got = result.to_torch(in_place=False).to(torch.float64)
    tolerance = 1e-12 if dtype == torch.float64 else 1e-5
    assert torch.equal(got == 0, want == 0)
    assert torch.allclose(got, want, rtol=tolerance, atol=0)
    assert "float " not in kernel.cpp_source or dtype == torch.float32


# -- the storage-order candidate ---------------------------------------------


def two_operand_nest(a_indices, b_indices, a_mode_order=None):
    """C[i] = sum A[a_indices] * B[b_indices], every tensor dense or compressed."""

    ivars = {name: IndexVar(name) for name in "ijk"}
    a = TensorVar("A", fmt="dd", dtype=torch.float32, mode_order=a_mode_order)[
        tuple(ivars[c] for c in a_indices)
    ]
    b = TensorVar("B", fmt="dd", dtype=torch.float32)[
        tuple(ivars[c] for c in b_indices)
    ]
    stmt = TensorAssign(
        TensorVar("C", fmt="s", dtype=torch.float32)[ivars["i"]],
        CINBinaryOp(Operation.MUL, a, b),
    )
    for name in reversed("ijk"):
        stmt = ForAll(ivars[name], stmt)
    return stmt, ivars


def test_an_order_already_in_storage_order_comes_back_unchanged():
    stmt, v = two_operand_nest("ij", "jk")
    order = [v["i"], v["j"], v["k"]]
    assert Scheduler._storage_order_sort(stmt, order) == order


def test_each_variable_moves_only_as_far_as_a_storage_order_forces_it():
    """A[i,j] needs i before j and B[j,k] needs j before k.  From (k, i, j)
    the earliest ready variable goes next: i, then j, then k."""

    stmt, v = two_operand_nest("ij", "jk")
    resorted = Scheduler._storage_order_sort(stmt, [v["k"], v["i"], v["j"]])
    assert resorted == [v["i"], v["j"], v["k"]]
    # Unconstrained relative order is kept: k only follows j, so (j, k) stays put
    # after i, and nothing pulls i behind anything.
    resorted = Scheduler._storage_order_sort(stmt, [v["i"], v["k"], v["j"]])
    assert resorted == [v["i"], v["j"], v["k"]]


def test_contradicting_storage_orders_have_no_sort():
    """A[i,j] and B[j,i], both stored row-major, cannot both be read in order."""

    stmt, v = two_operand_nest("ij", "ji")
    assert Scheduler._storage_order_sort(stmt, [v["i"], v["j"], v["k"]]) is None


def test_the_sort_reads_storage_order_through_the_mode_order():
    """The same A[i,j] stored column-major needs j before i."""

    stmt, v = two_operand_nest("ij", "jk", a_mode_order=[1, 0])
    resorted = Scheduler._storage_order_sort(stmt, [v["i"], v["j"], v["k"]])
    assert resorted == [v["j"], v["i"], v["k"]]
    assert Scheduler._visits_every_tensor_in_storage_order(stmt, resorted)


@pytest.mark.parametrize("cell", SORTED, ids=cell_id)
@pytest.mark.parametrize("arm", [False, True], ids=["direct", "regblock"])
def test_neither_scheduler_order_keeps_storage_order_but_a_sorted_one_does(cell, arm):
    """Measured, not assumed: the third candidate's preconditions hold."""

    cin = build_cin(cell, torch.float32)
    options = auto_options(arm)
    forced, composed = candidate_orders(cin, options)
    assert not keeps_every_storage_order(forced, cin)
    assert not keeps_every_storage_order(composed, cin)
    normalized = normalize_cin(cin, compile_options=options)
    by_id = {v.index_id: v for v in Scheduler.get_index_variables(normalized)}
    resorted = Scheduler._storage_order_sort(normalized, [by_id[i] for i in composed])
    assert resorted is not None
    assert keeps_every_storage_order([v.index_id for v in resorted], cin)


@pytest.mark.parametrize("cell", SORTED, ids=cell_id)
@pytest.mark.parametrize("arm", [False, True], ids=["direct", "regblock"])
def test_the_loopir_route_records_the_sorted_order(cell, arm):
    cin = build_cin(cell, torch.float32)
    options = auto_options(arm)
    forced, composed = candidate_orders(cin, options)
    operand_bindings, result_shape = bindings(cell, torch.float32)
    kernel = compile_cin_via_loopir(
        cin, result_shape, operand_bindings, compile_options=options
    )
    recorded = list(kernel.schedule.plan.loop_order)
    assert recorded not in (forced, composed)
    assert keeps_every_storage_order(recorded, cin)
    # The sort is stable: the recorded order is the composed order with each
    # index moved only as far as a storage order forces it.
    assert sorted(recorded) == sorted(composed)


@pytest.mark.parametrize("cell", SORTED, ids=cell_id)
def test_the_shared_boundary_still_records_the_forced_order_for_sorted_cells(cell):
    cin = build_cin(cell, torch.float32)
    options = auto_options(False)
    forced, _ = candidate_orders(cin, options)
    scheduled = Scheduler.apply_schedule(cin, Schedule(), compile_options=options)
    assert list(scheduled.verified_loop_plan.loop_order) == forced


@pytest.mark.parametrize("cell", MIGRATED + SORTED, ids=cell_id)
@pytest.mark.parametrize("arm", [0, 1], ids=["direct", "regblock"])
def test_the_forced_order_matches_dense_pytorch(cell, arm, without_the_candidate):
    """The forced order, executed: float64 to 1e-12 and exact stored coordinates.

    The kernel that runs is checked to be the forced order's: the plan of the
    compile whose text every execution must reproduce records that order.
    """

    name, operands, result_fmt, result_indices = cell
    kernel, statement = assert_program_matches_pytorch(
        tuple((symbol, fmt, indices, None) for symbol, fmt, indices in operands),
        (result_indices, result_fmt, None),
        arm,
        extent=EXTENT,
    )
    forced, composed = candidate_orders(statement, auto_options(bool(arm), jit=True))
    assert list(kernel.schedule.plan.loop_order) == forced, name
    assert forced != composed, name
