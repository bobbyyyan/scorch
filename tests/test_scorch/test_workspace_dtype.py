"""A workspace carries the element type of the tensors assigned from it.

``Workspace`` used to default to float32, and the matrix-vector product's scalar
workspace and ``STensor.change_mode_order``'s reordering workspace took that
default, so their float64 kernels accumulated or copied through ``float``: about
1e-7 relative error, invisible at the suite's usual 1e-3. These tests run those
two routes in float64 against a float64 PyTorch reference at 1e-12, and pin the
two guards that keep the defect from returning: the type is a required argument,
and the structural CIN preflight rejects a workspace whose type differs from the
tensor it is assigned into.
"""

import pytest
import torch

import scorch
import scorch.ops as ops
import scorch.stensor as stensor_module
from scorch import STensor
from scorch.compiler.cin import (
    ForAll,
    IndexVar,
    Operation,
    TensorAssign,
    TensorVar,
    UnaryOp,
    Where,
    Workspace,
)
from scorch.compiler.cin_analysis import normalize_cin
from scorch.compiler.cin_lowerer import CINLowerer
from scorch.compiler.compile_options import CompileOptions
from scorch.compiler.diagnostics import VerificationError
from scorch.compiler.scheduler import Scheduler
from scorch.format import LevelType


def _filled(shape, density, seed):
    generator = torch.Generator().manual_seed(seed)
    values = torch.rand(shape, generator=generator, dtype=torch.float64) + 0.25
    return values * (torch.rand(shape, generator=generator) < density)


def _stored(dense: torch.Tensor, fmt: str, name: str) -> STensor:
    """Build ``dense`` in ``fmt`` from arrays, with no conversion kernel."""

    if set(fmt) == {"o"}:
        return STensor.from_coo(dense.to_sparse_coo(), name=name)
    assert set(fmt) <= {"d", "s"}, fmt
    coordinates = dense.nonzero()
    position = torch.zeros(coordinates.shape[0], dtype=torch.int64)
    parents = 1
    mode_indices = []
    for level, letter in enumerate(fmt):
        extent = dense.shape[level]
        key = position * extent + coordinates[:, level]
        if letter == "d":
            mode_indices.append([])
            position, parents = key, parents * extent
            continue
        kept, position = torch.unique(key, sorted=True, return_inverse=True)
        parent = torch.div(kept, extent, rounding_mode="floor")
        pos = torch.zeros(parents + 1, dtype=torch.int64)
        pos[1:] = torch.cumsum(torch.bincount(parent, minlength=parents), 0)
        mode_indices.append([pos, kept - parent * extent])
        parents = kept.numel()
    values = torch.zeros(parents, dtype=dense.dtype)
    values[position] = dense[tuple(coordinates.t())]
    return STensor.from_components(
        tuple(dense.shape), fmt, mode_indices, values, name=name
    )


def _recording_workspaces(monkeypatch, module):
    """Record the element type of every workspace ``module`` constructs."""

    recorded = []

    def record(*args, **kwargs):
        workspace = Workspace(*args, **kwargs)
        recorded.append(workspace.dtype)
        return workspace

    monkeypatch.setattr(module, "Workspace", record)
    return recorded


def _logical_dense(tensor: STensor) -> torch.Tensor:
    """Densify from the tensor's own level arrays, with no conversion kernel."""

    shape = tuple(tensor.shape)
    level_types = tensor.format.get_level_types()
    levels = tensor.storage._mode_indices
    values = tensor.values
    if all(level_type is LevelType.COORDINATE for level_type in level_types):
        coordinates = tuple(level[0].to(torch.int64) for level in levels)
    else:
        assert set(level_types) <= {LevelType.DENSE, LevelType.COMPRESSED}
        stored = torch.zeros((1, 0), dtype=torch.int64)
        for level, level_type in enumerate(level_types):
            parents, extent = stored.shape[0], shape[level]
            if level_type is LevelType.DENSE:
                parent = torch.arange(parents).repeat_interleave(extent)
                crd = torch.arange(extent).repeat(parents)
            else:
                pos, crd = (array.to(torch.int64) for array in levels[level])
                parent = torch.arange(parents).repeat_interleave(pos[1:] - pos[:-1])
            stored = torch.cat([stored[parent], crd.reshape(-1, 1)], dim=1)
        coordinates = tuple(stored.t())
    physical = torch.zeros(shape, dtype=values.dtype)
    physical[coordinates] = values.reshape(-1)
    stored_order = tensor.storage.index.mode_order
    mode_order = list(range(len(shape))) if stored_order is None else stored_order
    return physical.permute([mode_order.index(axis) for axis in range(len(shape))])


@pytest.mark.parametrize("fmt", ["ss", "oo"])
def test_float64_matrix_vector_product_accumulates_in_float64(monkeypatch, fmt):
    matrix = _filled((48, 40), 0.3, 1746)
    vector = _filled((40,), 1.0, 1747)
    workspaces = _recording_workspaces(monkeypatch, ops)
    reached_generic_route = []
    generic_spmv = ops.spmv

    def spmv(*args, **kwargs):
        reached_generic_route.append(True)
        return generic_spmv(*args, **kwargs)

    monkeypatch.setattr(ops, "spmv", spmv)

    result = scorch.matmul(_stored(matrix, fmt, "A"), vector)

    assert reached_generic_route, "the call resolved to a prebuilt kernel"
    result = result.to_torch() if isinstance(result, STensor) else result
    assert result.dtype is torch.float64
    torch.testing.assert_close(result, matrix @ vector, rtol=1e-12, atol=0.0)
    assert workspaces == [torch.float64]


@pytest.mark.parametrize(
    ("fmt", "mode_order"),
    [("ddd", [2, 0, 1]), ("dss", [1, 2, 0]), ("sss", [1, 2, 0]), ("ooo", [2, 0, 1])],
)
def test_float64_reordering_copies_values_exactly(monkeypatch, fmt, mode_order):
    dense = _filled((3, 4, 5), 0.55, 1807)
    tensor = _stored(dense, fmt, "A")
    assert torch.equal(_logical_dense(tensor), dense)
    workspaces = _recording_workspaces(monkeypatch, stensor_module)

    tensor.change_mode_order(mode_order)

    assert list(tensor.storage.index.mode_order) == mode_order
    assert tensor.values.dtype is torch.float64
    torch.testing.assert_close(_logical_dense(tensor), dense, rtol=0.0, atol=0.0)
    assert workspaces == [torch.float64]


def test_workspace_requires_an_element_type():
    with pytest.raises(TypeError, match="dtype"):
        Workspace("wksp", dim=1)  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="torch.dtype"):
        Workspace("wksp", dim=1, dtype="float64")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        Workspace("wksp", 1, torch.float64)  # type: ignore[misc]
    assert Workspace("wksp", dim=1, dtype=torch.float64).dtype is torch.float64


def _scalar_workspace_program(result_dtype, workspace_dtype, consumer_rhs=None):
    i, j = IndexVar("i"), IndexVar("j")
    y = TensorVar("y", fmt="d", dtype=result_dtype)
    a = TensorVar("A", fmt="ds", dtype=result_dtype)
    x = TensorVar("x", fmt="d", dtype=result_dtype)
    workspace = Workspace("wksp", dim=0, dtype=workspace_dtype)
    read = workspace.get_default_access()
    return ForAll(
        i,
        Where(
            producer=ForAll(
                j,
                TensorAssign(
                    workspace.get_default_access(),
                    a[i, j] * x[j],
                    op=Operation.ADD,
                ),
            ),
            consumer=TensorAssign(
                y[i],
                read if consumer_rhs is None else consumer_rhs(read, x[i]),
            ),
        ),
    )


def _mismatch_codes(error):
    return [diagnostic.code for diagnostic in error.value.diagnostics]


@pytest.mark.parametrize(
    "consumer_rhs",
    [
        pytest.param(None, id="direct"),
        pytest.param(lambda read, other: other * read, id="inside-a-product"),
        pytest.param(
            lambda read, other: UnaryOp(Operation.ADD, read), id="inside-a-unary"
        ),
    ],
)
def test_preflight_rejects_a_workspace_of_another_type(consumer_rhs):
    release = CompileOptions.from_environment(environ={})
    mismatched = _scalar_workspace_program(torch.float64, torch.float32, consumer_rhs)

    with pytest.raises(VerificationError, match="workspace_type_mismatch") as error:
        normalize_cin(mismatched, compile_options=release)
    assert _mismatch_codes(error) == ["workspace_type_mismatch"]
    diagnostic = error.value.diagnostics[0]
    assert diagnostic.path[-1] != "lhs"
    assert "torch.float64" in diagnostic.message
    assert "torch.float32" in diagnostic.message

    # The raw legacy-lowering boundary runs the same preflight.
    with pytest.raises(VerificationError) as error:
        CINLowerer(compile_options=release).lower_IndexStmt(
            _scalar_workspace_program(torch.float64, torch.float32, consumer_rhs)
        )
    assert _mismatch_codes(error) == ["workspace_type_mismatch"]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_preflight_admits_a_workspace_of_the_assigned_type(dtype):
    release = CompileOptions.from_environment(environ={})
    normalized = normalize_cin(
        _scalar_workspace_program(dtype, dtype),
        compile_options=release,
    )
    assert normalized is not None


def test_scheduler_inserted_workspace_takes_the_result_type():
    # The scheduler's own workspace (scheduler.py, ``_insert_workspace``) is
    # created after normalization, where release lowering does not preflight
    # again, so its type must be correct by construction.
    i, j, k = IndexVar("i"), IndexVar("j"), IndexVar("k")
    c = TensorVar("C", shape=(6, 7), fmt="ds", dtype=torch.float64)
    a = TensorVar("A", shape=(6, 5), fmt="ds", dtype=torch.float64)
    b = TensorVar("B", shape=(5, 7), fmt="ds", dtype=torch.float64)
    program = ForAll(
        i,
        ForAll(
            j,
            ForAll(
                k,
                TensorAssign(c[i, k], a[i, j] * b[j, k], op=Operation.ADD),
            ),
        ),
    )

    scheduled = Scheduler.auto_schedule(program)

    workspaces = []
    pending = [scheduled]
    while pending:
        node = pending.pop()
        if isinstance(node, (ForAll,)):
            pending.append(node.stmt)
        elif isinstance(node, Where):
            pending.extend((node.producer, node.consumer))
        elif isinstance(node, TensorAssign):
            for access in (node.lhs, node.rhs):
                tensor = getattr(access, "tensor", None)
                if isinstance(tensor, Workspace):
                    workspaces.append(tensor)
    assert workspaces, "the scheduler inserted no workspace for this program"
    assert {workspace.dtype for workspace in workspaces} == {torch.float64}
