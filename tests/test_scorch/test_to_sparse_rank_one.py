"""Public rank-1 ``to_sparse`` conversion boundaries.

The rank-1 branch of :meth:`STensor.to_sparse` builds one compressed level
directly from ``torch.nonzero``.  It previously filtered ``self.values``
unconditionally, ignored the requested format entirely, and skipped the
compiler-options/timing-context ownership boundary that the rank>=2 route
owns.  Three consequences, all covered here:

- a sparse receiver was re-filtered through its own stored value array, so
  ``to_sparse('s')`` on an already-compressed rank-1 tensor reinterpreted
  positions as coordinates and silently corrupted it;
- a wrong-rank or dense request silently produced compressed storage;
- foreign ``_compile_options``/``_compilation_context`` objects were never
  rejected, and a sparse source recorded no densification work.

The branch now mirrors the rank>=2 discipline: validate the boundary, require
format rank equality, densify a sparse source out of place under the caller's
exact options and context, and commit receiver state only after the complete
output is built.
"""

import pytest
import torch

from scorch.compiler.compilation_context import CompilationContext
from scorch.compiler.compile_options import CompileOptions
from scorch.exceptions import TensorFormatError, TensorStorageError, TensorTypeError
from scorch.format import LevelFormat, LevelType, TensorFormat
from scorch.stensor import STensor

VECTOR = [0.0, 1.0, 0.0, 0.0, 2.0, 0.0]


def coordinates(stensor):
    return stensor.storage.index.mode_indices[0][1].tolist()


def positions(stensor):
    return stensor.storage.index.mode_indices[0][0].tolist()


def snapshot(stensor):
    return (
        stensor.name,
        stensor.shape,
        str(stensor.format),
        positions(stensor),
        coordinates(stensor),
        stensor.storage.value.tolist(),
    )


def test_dense_source_compresses_to_stored_coordinates():
    tensor = STensor.from_torch(torch.tensor(VECTOR), "A").to_sparse("s")
    assert str(tensor.format) == "s"
    assert coordinates(tensor) == [1, 4]
    assert positions(tensor) == [0, 2]
    assert tensor.storage.value.tolist() == [1.0, 2.0]
    assert tensor.to_torch().tolist() == VECTOR


def test_sparse_receiver_reconverts_without_corruption():
    """The defect: stored positions were re-filtered as if they were dense."""

    tensor = STensor.from_torch(torch.tensor(VECTOR), "A").to_sparse("s")
    before = snapshot(tensor)
    tensor.to_sparse("s")
    assert snapshot(tensor) == before
    assert tensor.to_torch().tolist() == VECTOR


def test_repeated_reconversion_is_idempotent():
    tensor = STensor.from_torch(torch.tensor(VECTOR), "A").to_sparse("s")
    for _ in range(3):
        tensor.to_sparse("s")
    assert coordinates(tensor) == [1, 4]
    assert tensor.to_torch().tolist() == VECTOR


def test_default_format_still_compresses():
    tensor = STensor.from_torch(torch.tensor(VECTOR), "A").to_sparse()
    assert str(tensor.format) == "s"
    assert coordinates(tensor) == [1, 4]


def test_wrong_rank_request_fails_closed():
    tensor = STensor.from_torch(torch.tensor(VECTOR), "A")
    with pytest.raises(TensorStorageError, match="does not match tensor rank 1"):
        tensor.to_sparse("ss")
    assert str(tensor.format) == "d"


def test_dense_rank_one_request_fails_closed():
    """A rank-1 ``d`` request no longer returns sparse storage."""

    tensor = STensor.from_torch(torch.tensor(VECTOR), "A")
    with pytest.raises(TensorStorageError, match="requests no supported sparse mode"):
        tensor.to_sparse("d")
    assert str(tensor.format) == "d"


@pytest.mark.parametrize("kind", ["s", "o"])
def test_exact_requested_sparse_format_metadata_is_preserved(kind):
    requested = TensorFormat([LevelFormat(kind, bit_width=64)])
    tensor = STensor.from_torch(torch.tensor(VECTOR), "A").to_sparse(requested)
    assert tensor.format == requested
    assert tensor.format.serialize() == requested.serialize()


def test_requested_format_and_nested_level_are_deeply_detached():
    level = LevelFormat("s", bit_width=64)
    requested = TensorFormat([level])
    tensor = STensor.from_torch(torch.tensor(VECTOR), "A").to_sparse(requested)

    object.__setattr__(level, "_mode", LevelType.DENSE)
    object.__setattr__(requested, "_level_formats", (LevelFormat("o"),))

    assert str(tensor.format) == "s"
    assert tensor.format.get_level_formats()[0].bit_width == 64
    assert tensor.to_torch().tolist() == VECTOR
    tensor.storage.validate()


def test_level_sequence_input_is_deeply_detached():
    level = LevelFormat("o", bit_width=32)
    tensor = STensor.from_torch(torch.tensor(VECTOR), "A").to_sparse([level])
    object.__setattr__(level, "_mode", LevelType.DENSE)
    assert str(tensor.format) == "o"
    assert tensor.format.get_level_formats()[0].bit_width == 32
    assert tensor.to_torch().tolist() == VECTOR


def test_tensor_format_subclass_fails_closed_without_mutating_receiver():
    class DerivedFormat(TensorFormat):
        pass

    tensor = STensor.from_torch(torch.tensor(VECTOR), "A")
    with pytest.raises(TensorTypeError, match="exact TensorFormat"):
        tensor.to_sparse(DerivedFormat("s"))
    assert str(tensor.format) == "d"


@pytest.mark.parametrize(
    "malformation",
    ["missing_levels", "list_levels", "foreign_level", "missing_mode", "bad_mode"],
)
def test_forged_format_structure_fails_closed(malformation):
    requested = TensorFormat([LevelFormat("s")])
    level = requested.get_level_formats()[0]
    if malformation == "missing_levels":
        object.__delattr__(requested, "_level_formats")
    elif malformation == "list_levels":
        object.__setattr__(requested, "_level_formats", [level])
    elif malformation == "foreign_level":
        object.__setattr__(requested, "_level_formats", (object(),))
    elif malformation == "missing_mode":
        object.__delattr__(level, "_mode")
    else:
        object.__setattr__(level, "_mode", "s")

    tensor = STensor.from_torch(torch.tensor(VECTOR), "A")
    with pytest.raises(TensorFormatError, match="to_sparse format"):
        tensor.to_sparse(requested)
    assert str(tensor.format) == "d"


@pytest.mark.parametrize("bit_width", [True, 0, -1, "64", 1 << 63])
def test_forged_format_bit_width_fails_closed(bit_width):
    requested = TensorFormat([LevelFormat("s")])
    object.__setattr__(requested.get_level_formats()[0], "_bit_width", bit_width)
    tensor = STensor.from_torch(torch.tensor(VECTOR), "A")
    with pytest.raises(TensorFormatError, match="positive signed-int64 exact int"):
        tensor.to_sparse(requested)
    assert str(tensor.format) == "d"


@pytest.mark.parametrize("owner", ["format", "level"])
def test_hostile_stored_field_keys_fail_closed(owner):
    class HostileKey(str):
        __hash__ = str.__hash__

        def __eq__(self, other):
            raise RuntimeError("stored-key equality must not execute")

    requested = TensorFormat([LevelFormat("s")])
    target = requested if owner == "format" else requested.get_level_formats()[0]
    state = object.__getattribute__(target, "__dict__")
    key = next(iter(state))
    value = state.pop(key)
    state[HostileKey(key)] = value

    tensor = STensor.from_torch(torch.tensor(VECTOR), "A")
    with pytest.raises(TensorFormatError, match="malformed stored state"):
        tensor.to_sparse(requested)
    assert str(tensor.format) == "d"


@pytest.mark.parametrize(
    "dtype",
    [torch.float32, torch.float64, torch.int32, torch.int64, torch.int8, torch.uint8],
)
def test_coordinate_rank_one_request_is_honored(dtype):
    dense = torch.tensor([0, 2, 0, 3], dtype=dtype)
    tensor = STensor.from_torch(dense.clone(), "A").to_sparse("o")
    assert str(tensor.format) == "o"
    assert len(tensor.storage.index.mode_indices[0]) == 1
    assert tensor.storage.index.mode_indices[0][0].tolist() == [1, 3]
    assert tensor.storage.value.tolist() == [2, 3]
    assert tensor.storage.value.dtype is dtype


def test_coordinate_rank_one_request_round_trips_sparse_source():
    dense = torch.tensor(VECTOR)
    tensor = STensor.from_torch(dense.clone(), "A").to_sparse("s").to_sparse("o")
    assert str(tensor.format) == "o"
    assert tensor.storage.index.mode_indices[0][0].tolist() == [1, 4]
    assert torch.equal(tensor.to_torch(), dense)


def test_dense_rank_one_source_does_not_take_the_densification_path(monkeypatch):
    tensor = STensor.from_torch(torch.tensor(VECTOR), "A")

    def unexpected_densification(*args, **kwargs):
        raise AssertionError("dense rank-1 conversion must not clone through to_dense")

    monkeypatch.setattr(STensor, "to_dense", unexpected_densification)
    tensor.to_sparse("s")
    assert coordinates(tensor) == [1, 4]


def test_unparseable_format_keeps_the_historical_path():
    tensor = STensor.from_torch(torch.tensor(VECTOR), "A").to_sparse("not-a-format")
    assert str(tensor.format) == "s"
    assert coordinates(tensor) == [1, 4]


@pytest.mark.parametrize("bad", [object(), "options", 3])
def test_foreign_compile_options_are_rejected(bad):
    tensor = STensor.from_torch(torch.tensor(VECTOR), "A")
    with pytest.raises(TypeError, match="_compile_options"):
        tensor.to_sparse("s", _compile_options=bad)
    assert str(tensor.format) == "d"


@pytest.mark.parametrize("bad", [object(), 3])
def test_foreign_compilation_context_is_rejected(bad):
    tensor = STensor.from_torch(torch.tensor(VECTOR), "A")
    with pytest.raises(TypeError, match="_compilation_context"):
        tensor.to_sparse("s", _compilation_context=bad)
    assert str(tensor.format) == "d"


def test_context_owned_by_foreign_options_is_rejected():
    tensor = STensor.from_torch(torch.tensor(VECTOR), "A")
    options = CompileOptions.from_environment(environ={})
    foreign = CompilationContext(CompileOptions.from_environment(environ={}))
    with pytest.raises(TypeError):
        tensor.to_sparse("s", _compile_options=options, _compilation_context=foreign)
    assert str(tensor.format) == "d"


def test_sparse_source_densification_uses_the_supplied_context():
    """A sparse receiver's densification is recorded by the caller's owner."""

    options = CompileOptions.from_environment()
    context = CompilationContext(options)
    tensor = STensor.from_torch(torch.tensor(VECTOR), "A").to_sparse("s")
    tensor.to_sparse("s", _compile_options=options, _compilation_context=context)
    assert coordinates(tensor) == [1, 4]
    assert context.stage_run_records


def test_conversion_is_exception_atomic(monkeypatch):
    tensor = STensor.from_torch(torch.tensor(VECTOR), "A").to_sparse("s")
    before = snapshot(tensor)

    def fail_nonzero(*args, **kwargs):
        raise RuntimeError("injected rank-1 failure")

    monkeypatch.setattr(torch, "nonzero", fail_nonzero)
    with pytest.raises(RuntimeError, match="injected rank-1 failure"):
        tensor.to_sparse("s")

    assert snapshot(tensor) == before
    assert tensor.to_torch().tolist() == VECTOR


def test_all_zero_and_empty_vectors_convert_canonically():
    zeros = STensor.from_torch(torch.zeros(4), "A").to_sparse("s")
    assert positions(zeros) == [0, 0]
    assert coordinates(zeros) == []
    assert zeros.storage.value.numel() == 0
    empty = STensor.from_torch(torch.zeros(0), "B").to_sparse("s")
    assert positions(empty) == [0, 0]
    assert empty.storage.value.numel() == 0


def test_float64_and_explicit_zero_values_are_preserved():
    dense = torch.tensor([0.0, 3.5, 0.0, -2.25], dtype=torch.float64)
    tensor = STensor.from_torch(dense.clone(), "A").to_sparse("s")
    assert tensor.storage.value.dtype is torch.float64
    assert tensor.storage.value.tolist() == [3.5, -2.25]
    assert torch.equal(tensor.to_torch(), dense)


def test_produced_values_do_not_alias_the_source():
    dense = torch.tensor([0.0, 1.0, 2.0])
    tensor = STensor.from_torch(dense, "A").to_sparse("s")
    stored = tensor.storage.value
    dense.mul_(0.0)
    assert stored.tolist() == [1.0, 2.0]
