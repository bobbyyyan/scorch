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
- an unparseable request was swallowed and silently treated as the default
  compressed format;
- foreign ``_compile_options``/``_compilation_context`` objects were never
  rejected, and a sparse source recorded no densification work.

The branch now mirrors the rank>=2 discipline: validate the boundary, require
format rank equality, densify a sparse source out of place under the caller's
exact options and context, and commit receiver state only after the complete
output is built.
"""

import pytest
import torch

import scorch.stensor as stensor_module  # type: ignore[import-untyped]
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


def test_requested_format_and_nested_level_cannot_change_the_result():
    """The result may hold the requested format itself: it cannot change."""
    level = LevelFormat("s", bit_width=64)
    requested = TensorFormat([level])
    tensor = STensor.from_torch(torch.tensor(VECTOR), "A").to_sparse(requested)

    with pytest.raises(AttributeError):
        object.__setattr__(level, "_mode", LevelType.DENSE)
    with pytest.raises(AttributeError):
        object.__setattr__(requested, "_level_formats", (LevelFormat("o"),))

    assert str(tensor.format) == "s"
    assert tensor.format.get_level_formats()[0].bit_width == 64
    assert tensor.to_torch().tolist() == VECTOR
    tensor.storage.validate()


def test_level_sequence_input_cannot_change_the_result():
    level = LevelFormat("o", bit_width=32)
    tensor = STensor.from_torch(torch.tensor(VECTOR), "A").to_sparse([level])
    with pytest.raises(AttributeError):
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
def test_format_structure_cannot_be_forged(malformation):
    requested = TensorFormat([LevelFormat("s")])
    level = requested.get_level_formats()[0]
    with pytest.raises(AttributeError):
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

    tensor = STensor.from_torch(torch.tensor(VECTOR), "A").to_sparse(requested)
    assert str(tensor.format) == "s"
    assert tensor.to_torch().tolist() == VECTOR


def test_a_format_that_was_never_built_fails_closed():
    tensor = STensor.from_torch(torch.tensor(VECTOR), "A")
    with pytest.raises(TensorFormatError, match="to_sparse format"):
        tensor.to_sparse(TensorFormat.__new__(TensorFormat))
    assert str(tensor.format) == "d"


@pytest.mark.parametrize("bit_width", [True, 0, -1, "64"])
def test_an_invalid_bit_width_can_be_neither_built_nor_forged(bit_width):
    with pytest.raises((TensorTypeError, TensorFormatError)):
        LevelFormat("s", bit_width=bit_width)
    requested = TensorFormat([LevelFormat("s")])
    with pytest.raises(AttributeError):
        object.__setattr__(requested.get_level_formats()[0], "_bit_width", bit_width)
    assert requested.get_level_formats()[0].bit_width is None


def test_a_bit_width_past_int64_fails_closed():
    """The constructor accepts any positive width; to_sparse allows a signed int64."""
    requested = TensorFormat([LevelFormat("s", bit_width=1 << 63)])
    tensor = STensor.from_torch(torch.tensor(VECTOR), "A")
    with pytest.raises(TensorFormatError, match="positive signed-int64 exact int"):
        tensor.to_sparse(requested)
    assert str(tensor.format) == "d"


@pytest.mark.parametrize("owner", ["format", "level"])
def test_no_stored_field_keys_exist_to_poison(owner):
    """A format and a level have no instance dict whose keys could be replaced with a
    hostile ``str`` subclass."""
    requested = TensorFormat([LevelFormat("s")])
    target = requested if owner == "format" else requested.get_level_formats()[0]
    with pytest.raises(AttributeError):
        object.__getattribute__(target, "__dict__")

    tensor = STensor.from_torch(torch.tensor(VECTOR), "A").to_sparse(requested)
    assert str(tensor.format) == "s"


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


@pytest.mark.parametrize("shape", [(6,), (2, 3)], ids=["rank1", "rank2"])
@pytest.mark.parametrize(
    "bad_format", ["not-a-format", ["s", "not-a-format"], object()]
)
def test_unparseable_format_fails_atomically_at_every_rank(shape, bad_format):
    tensor = STensor.from_torch(
        torch.arange(6, dtype=torch.float32).reshape(shape), "A"
    )
    metadata = tensor.metadata
    storage = tensor.storage
    values = tensor.storage.value.clone()

    with pytest.raises((TensorFormatError, TensorTypeError)):
        tensor.to_sparse(bad_format)

    assert tensor.metadata is metadata
    assert tensor.storage is storage
    assert tensor.format.is_dense()
    assert torch.equal(tensor.storage.value, values)


@pytest.mark.parametrize("mixed_format", ["do", "so", "od", "os"])
def test_rank2_mixed_coordinate_request_fails_atomically(mixed_format):
    dense = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    tensor = STensor.from_torch(dense.clone(), "A")
    metadata = tensor.metadata
    storage = tensor.storage
    values = storage.value.clone()

    with pytest.raises(
        TensorStorageError, match="does not support mixed coordinate hierarchies"
    ):
        tensor.to_sparse(mixed_format)

    assert tensor.metadata is metadata
    assert tensor.storage is storage
    assert torch.equal(tensor.storage.value, values)
    assert torch.equal(tensor.to_torch(), dense)


def test_rank2_all_coordinate_request_still_round_trips():
    dense = torch.tensor([[0.0, 2.0, 0.0], [3.0, 0.0, 4.0]])
    tensor = STensor.from_torch(dense.clone(), "A").to_sparse("oo")
    assert str(tensor.format) == "o,o"
    assert torch.equal(tensor.to_torch(), dense)


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


def test_sparse_source_densification_uses_the_supplied_context(monkeypatch):
    """A sparse receiver's densification is recorded by the caller's owner when it
    compiles, and records nothing when the kept kernel serves it."""

    # A densification the generated-kernel cache serves lowers nothing and records no
    # stage, in the caller's context or any other.  A kernel an earlier test in this
    # process kept for the same call (stensor._GENERATED_KERNELS) would serve this one.
    monkeypatch.setattr(stensor_module, "_GENERATED_KERNELS", {})
    options = CompileOptions.from_environment()
    context = CompilationContext(options)
    tensor = STensor.from_torch(torch.tensor(VECTOR), "A").to_sparse("s")
    tensor.to_sparse("s", _compile_options=options, _compilation_context=context)
    assert coordinates(tensor) == [1, 4]
    assert context.stage_run_records

    # Warm: the same densification under another supplied context runs the kept kernel,
    # keeps nothing new and records nothing (the cache's design, stensor.py).
    warm = STensor.from_torch(torch.tensor(VECTOR), "A").to_sparse("s")
    kept = dict(stensor_module._GENERATED_KERNELS)
    assert kept
    warm_context = CompilationContext(options)
    warm.to_sparse("s", _compile_options=options, _compilation_context=warm_context)
    assert coordinates(warm) == [1, 4]
    assert stensor_module._GENERATED_KERNELS.keys() == kept.keys()
    assert all(stensor_module._GENERATED_KERNELS[key] is kept[key] for key in kept)
    assert warm_context.stage_run_records == ()


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
