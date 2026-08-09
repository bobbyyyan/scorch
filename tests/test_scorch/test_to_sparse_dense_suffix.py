"""Public ``to_sparse`` dense-suffix block materialization.

Formats whose value-bearing suffix is DENSE below ``d``/``s`` prefix
structure (``sd``, ``sdd``, ``dsd``, ``ssd``, ...) previously ran the
per-entry legacy filter kernel, which mis-assembles blocked values
(values without parent coordinates), so every runtime test needed
hand-built ``TensorIndex`` storage.  The public conversion now
materializes these layouts directly: one complete dense block per
stored prefix path, a path stored exactly when its block contains any
nonzero, and stored blocks keeping their interior zeros — the same
conditional-parent discipline the compiled assembly families use.
"""

import pytest
import torch

from scorch.compiler.compilation_context import CompilationContext
from scorch.compiler.compile_options import CompileOptions
from scorch.exceptions import TensorStorageError
from scorch.format import LevelFormat, LevelType, TensorFormat
from scorch.stensor import STensor
from tests.test_scorch.test_loopir_mixed_operand_target import (
    sparse_dsd,
    sparse_sd,
    sparse_sdd,
)


def storage_snapshot(stensor):
    return (
        [
            [tensor.tolist() for tensor in level]
            for level in stensor.storage.index.mode_indices
        ],
        stensor.storage.value.tolist(),
    )


def tensor_snapshot(stensor):
    return (
        stensor.name,
        stensor.shape,
        str(stensor.format),
        tuple(stensor.storage.index.mode_order),
        storage_snapshot(stensor),
    )


@pytest.mark.parametrize("fmt", ["sd", "sdd", "dsd", "ssd", "dssd"])
def test_dense_suffix_conversion_round_trips(fmt):
    torch.manual_seed(20260731)
    rank = len(fmt)
    shape = {2: (4, 5), 3: (3, 4, 5), 4: (2, 3, 4, 3)}[rank]
    dense = (torch.rand(shape) < 0.25) * torch.randn(shape)
    converted = STensor.from_torch(dense.clone(), "A").to_sparse(fmt)
    assert str(converted.format) == ",".join(fmt)
    assert torch.equal(converted.to_torch(), dense)


@pytest.mark.parametrize(
    "fmt,builder",
    [("sd", sparse_sd), ("sdd", sparse_sdd), ("dsd", sparse_dsd)],
)
def test_conversion_matches_the_hand_built_storage_exactly(fmt, builder):
    torch.manual_seed(7)
    shape = (4, 5) if fmt == "sd" else (3, 4, 5)
    dense = (torch.rand(shape) < 0.3) * torch.randn(shape)
    converted = STensor.from_torch(dense.clone(), "A").to_sparse(fmt)
    hand_built = builder(dense, "A")
    assert storage_snapshot(converted) == storage_snapshot(hand_built)


def test_stored_blocks_keep_interior_zeros():
    dense = torch.tensor([[0.0, 5.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 2.0]])
    converted = STensor.from_torch(dense.clone(), "A").to_sparse("sd")
    modes, values = storage_snapshot(converted)
    assert modes[0] == [[0, 2], [0, 2]]
    assert modes[1] == []
    assert values == [0.0, 5.0, 0.0, 1.0, 0.0, 2.0]


def test_all_zero_tensors_convert_to_canonical_empty_storage():
    converted = STensor.from_torch(torch.zeros(3, 4), "A").to_sparse("sd")
    modes, values = storage_snapshot(converted)
    assert modes[0] == [[0, 0], []]
    assert values == []

    converted = STensor.from_torch(torch.zeros(3, 4, 5), "A").to_sparse("dsd")
    modes, values = storage_snapshot(converted)
    assert modes[0] == []
    assert modes[1] == [[0, 0, 0, 0], []]
    assert values == []


def test_dense_prefix_empty_slots_keep_full_position_arrays():
    dense = torch.zeros(3, 4, 5)
    dense[2, 1, 3] = 4.0
    converted = STensor.from_torch(dense.clone(), "A").to_sparse("dsd")
    modes, values = storage_snapshot(converted)
    assert modes[1] == [[0, 0, 0, 1], [1]]
    assert values == dense[2, 1].tolist()


def test_float64_conversion_preserves_dtype_and_values():
    torch.manual_seed(3)
    dense = ((torch.rand(4, 5) < 0.3) * torch.randn(4, 5)).double()
    converted = STensor.from_torch(dense.clone(), "A").to_sparse("sd")
    assert converted.storage.value.dtype == torch.float64
    assert torch.equal(converted.to_torch(), dense)


def test_sparse_sources_reconvert_through_densification():
    dense = torch.tensor([[0.0, 5.0], [3.0, 0.0], [0.0, 0.0]])
    converted = STensor.from_torch(dense.clone(), "A").to_sparse("ss").to_sparse("sd")
    assert str(converted.format) == "s,d"
    assert torch.equal(converted.to_torch(), dense)


@pytest.mark.parametrize("field", ["options", "context"])
def test_dense_suffix_conversion_validates_compiler_boundary(field):
    dense = STensor.from_torch(torch.eye(2), "A")
    options = CompileOptions.from_environment(environ={})
    kwargs = (
        {"_compile_options": object()}
        if field == "options"
        else {"_compile_options": options, "_compilation_context": object()}
    )
    with pytest.raises(TypeError):
        dense.to_sparse("sd", **kwargs)


def test_dense_suffix_conversion_rejects_a_context_for_foreign_options():
    dense = STensor.from_torch(torch.eye(2), "A")
    options = CompileOptions.from_environment(environ={})
    foreign_options = CompileOptions.from_environment(environ={})
    context = CompilationContext(foreign_options)
    with pytest.raises(TypeError, match="exact CompileOptions snapshot"):
        dense.to_sparse(
            "sd",
            _compile_options=options,
            _compilation_context=context,
        )


def test_rank_mismatch_fails_atomically_before_opening_a_stage():
    dense = STensor.from_torch(torch.eye(2), "A")
    options = CompileOptions.from_environment(environ={})
    context = CompilationContext(options)

    with pytest.raises(TensorStorageError, match="format rank"):
        dense.to_sparse(
            "sdd",
            _compile_options=options,
            _compilation_context=context,
        )

    assert context.stage_run_records == ()


def test_sparse_source_densification_uses_the_supplied_context():
    dense = torch.tensor([[0.0, 5.0], [3.0, 0.0], [0.0, 0.0]])
    source = STensor.from_torch(dense.to_sparse_csr(), "A")
    options = CompileOptions.from_environment()
    context = CompilationContext(options)

    converted = source.to_sparse(
        "sd",
        _compile_options=options,
        _compilation_context=context,
    )

    assert torch.equal(converted.to_torch(), dense)
    assert context.stage_run_records
    assert context.llir_pass_run_records


def test_dense_suffix_conversion_is_exception_atomic(monkeypatch):
    dense = torch.tensor([[0.0, 5.0], [3.0, 0.0], [0.0, 0.0]])
    source = STensor.from_torch(dense.to_sparse_csr(), "A")
    before = tensor_snapshot(source)

    def fail_nonzero(*args, **kwargs):
        raise RuntimeError("injected materialization failure")

    monkeypatch.setattr(torch, "nonzero", fail_nonzero)
    with pytest.raises(RuntimeError, match="injected materialization failure"):
        source.to_sparse("sd")

    assert tensor_snapshot(source) == before
    assert torch.equal(source.to_torch(), dense)


def test_compressed_leaf_formats_keep_the_kernel_path():
    """Formats with compressed value-bearing leaves are untouched."""

    dense = torch.tensor([[0.0, 5.0], [3.0, 0.0]])
    ss = STensor.from_torch(dense.clone(), "A").to_sparse("ss")
    assert str(ss.format) == "s,s"
    assert ss.storage.value.tolist() == [5.0, 3.0]
    ds = STensor.from_torch(dense.clone(), "B").to_sparse("ds")
    assert str(ds.format) == "d,s"
    assert ds.storage.value.tolist() == [5.0, 3.0]


@pytest.mark.parametrize(
    "shape,mode_order,fmt",
    [
        ((2, 3), [1, 0], "sd"),
        ((2, 3, 4), [2, 0, 1], "dds"),
        ((2, 3, 4), [1, 2, 0], "sds"),
    ],
)
def test_nonidentity_mode_order_materializes_in_physical_order(shape, mode_order, fmt):
    dense = torch.arange(
        int(torch.tensor(shape).prod().item()), dtype=torch.float32
    ).reshape(shape)
    dense[dense.remainder(3) != 0] = 0
    tensor = STensor.from_torch(dense.clone(), "A").change_mode_order(mode_order)

    tensor.to_sparse(fmt)

    assert str(tensor.format) == ",".join(fmt)
    assert tuple(tensor.storage.index.mode_order) == tuple(mode_order)
    assert tensor.logical_shape == shape
    assert tensor.physical_shape == tuple(shape[mode] for mode in mode_order)
    assert torch.equal(tensor.to_torch(), dense)


def test_nonidentity_sparse_source_materializes_without_losing_logical_axes():
    dense = torch.tensor(
        [
            [[0.0, 1.0, 0.0, 2.0], [0.0, 0.0, 3.0, 0.0], [4.0, 0.0, 0.0, 0.0]],
            [[0.0, 5.0, 0.0, 0.0], [6.0, 0.0, 0.0, 7.0], [0.0, 0.0, 8.0, 0.0]],
        ]
    )
    tensor = STensor.from_torch(dense.clone(), "A").change_mode_order([2, 0, 1])
    tensor.to_sparse("sss")

    tensor.to_sparse("dds")

    assert str(tensor.format) == "d,d,s"
    assert tuple(tensor.storage.index.mode_order) == (2, 0, 1)
    assert torch.equal(tensor.to_torch(), dense)


def test_converted_inputs_execute_through_the_compiled_mixed_route():
    """to_sparse('sd') inputs drive the mixed-operand family end-to-end."""

    from tests.test_scorch.test_loopir_mixed_operand_target import (
        build_copy_cin,
    )
    from tests.test_scorch.test_loopir_sparse_workspace_target import (
        auto_options,
    )
    from scorch.compiler.loopir.pipeline import execute_cin_via_loopir

    torch.manual_seed(11)
    dense = (torch.rand(4, 5) < 0.4) * torch.randn(4, 5)
    converted = STensor.from_torch(dense.clone(), "A").to_sparse("sd")
    out = execute_cin_via_loopir(
        build_copy_cin(),
        (4, 5),
        converted,
        compile_options=auto_options(False, jit=True),
    )
    result = out[0] if isinstance(out, tuple) else out
    assert torch.allclose(result.to_torch(), dense, atol=1e-3, rtol=1e-3)


# --- The widened directly-materialized family ---------------------------
#
# The per-entry filter kernel sizes a compressed level's position array from
# its immediately enclosing level alone, so it only assembles ``d?s+``
# layouts.  Every other ``d``/``s`` layout — a dense value-bearing suffix, or
# a dense level above a compressed level other than one leading prefix — is
# materialized directly by the same block walk.  Before that widening, the
# public conversion raised ``TensorIndexError`` for ``dds``/``sds``/``ddss``
# and friends: the position array was sized by one dense extent instead of
# the product of every dense parent.

_LEGACY_ASSEMBLED = ["ds", "ss", "dss", "sss", "dsss", "ssss"]
_DIRECT_COMPRESSED_LEAF = ["dds", "sds", "ddss", "dsds", "sdds", "sdss", "ssds"]


def _dense_reference_from_storage(stensor, fmt, shape):
    """Decode stored blocks back to a dense tensor, from the layout alone."""

    from scorch.format import LevelType, parse_format

    kinds = parse_format(fmt).get_level_types()
    rank = len(kinds)
    suffix = 0
    while suffix < rank and kinds[rank - 1 - suffix] is LevelType.DENSE:
        suffix += 1
    split = rank - suffix
    block_numel = 1
    for extent in shape[split:]:
        block_numel *= extent
    mode_indices = stensor.storage.index.mode_indices
    values = stensor.storage.value.reshape(-1)
    paths = [()]
    for level in range(split):
        nxt = []
        if kinds[level] is LevelType.DENSE:
            for path in paths:
                for coordinate in range(shape[level]):
                    nxt.append(path + (coordinate,))
        else:
            pos, crd = mode_indices[level]
            for parent, path in enumerate(paths):
                for slot in range(int(pos[parent]), int(pos[parent + 1])):
                    nxt.append(path + (int(crd[slot]),))
        paths = nxt
    out = torch.zeros(shape, dtype=values.dtype)
    if block_numel == 0 or any(extent == 0 for extent in shape):
        return out
    for block_index, path in enumerate(paths):
        block = values[block_index * block_numel : (block_index + 1) * block_numel]
        assert block.numel() == block_numel, (path, block.numel(), block_numel)
        view = out
        for coordinate in path:
            view = view[coordinate]
        view.reshape(-1)[:] = block
    return out


@pytest.mark.parametrize("fmt", _LEGACY_ASSEMBLED)
def test_filter_kernel_layouts_are_not_rerouted(fmt):
    """``d?s+`` layouts keep the per-entry kernel path byte-for-byte."""

    from scorch.format import parse_format
    from scorch.stensor import _is_directly_materialized_format

    assert not _is_directly_materialized_format(parse_format(fmt))


@pytest.mark.parametrize("fmt", ["dd", "ddd", "dddd"])
def test_all_dense_layouts_are_not_rerouted(fmt):
    """An all-dense request carries no compressed structure to assemble."""

    from scorch.format import parse_format
    from scorch.stensor import _is_directly_materialized_format

    assert not _is_directly_materialized_format(parse_format(fmt))


@pytest.mark.parametrize(
    "fmt", ["sd", "sdd", "dsd", "ssd", "dssd", *_DIRECT_COMPRESSED_LEAF]
)
def test_directly_materialized_family_is_routed(fmt):
    from scorch.format import parse_format
    from scorch.stensor import _is_directly_materialized_format

    assert _is_directly_materialized_format(parse_format(fmt))


@pytest.mark.parametrize("fmt", _DIRECT_COMPRESSED_LEAF)
def test_multi_dense_parent_layouts_convert_and_decode(fmt):
    """Compressed levels below several dense parents now assemble exactly."""

    torch.manual_seed(20260807)
    rank = len(fmt)
    shape = {3: (3, 4, 5), 4: (2, 3, 4, 3)}[rank]
    dense = (torch.rand(shape) < 0.3) * torch.randn(shape)
    converted = STensor.from_torch(dense.clone(), "A").to_sparse(fmt)
    assert str(converted.format) == ",".join(fmt)
    assert torch.equal(
        _dense_reference_from_storage(converted, fmt, shape), dense.float()
    )


def test_materialized_format_is_deeply_detached_from_caller():
    requested_level = LevelFormat("s", bit_width=64)
    requested = TensorFormat([requested_level, LevelFormat("d")])
    dense = torch.tensor([[0.0, 2.0, 0.0], [3.0, 0.0, 4.0]])
    converted = STensor.from_torch(dense.clone(), "A").to_sparse(requested)

    object.__setattr__(requested_level, "_mode", LevelType.DENSE)
    object.__setattr__(requested, "_level_formats", ())

    assert str(converted.format) == "s,d"
    assert converted.format.get_level_formats()[0].bit_width == 64
    assert torch.equal(converted.to_torch(), dense)
    converted.storage.validate()


@pytest.mark.parametrize("fmt", _DIRECT_COMPRESSED_LEAF)
def test_multi_dense_parent_position_arrays_span_every_dense_parent(fmt):
    """The defect: a position array sized by one dense extent, not the
    product of every dense parent above its compressed level."""

    torch.manual_seed(5)
    rank = len(fmt)
    shape = {3: (3, 4, 5), 4: (2, 3, 4, 3)}[rank]
    dense = (torch.rand(shape) < 0.5) * torch.randn(shape)
    converted = STensor.from_torch(dense.clone(), "A").to_sparse(fmt)
    mode_indices = converted.storage.index.mode_indices
    parents = 1
    for level, kind in enumerate(fmt):
        if kind == "d":
            assert list(mode_indices[level]) == []
            parents *= shape[level]
            continue
        pos, crd = mode_indices[level]
        assert len(pos) == parents + 1, (level, len(pos), parents + 1)
        assert int(pos[0]) == 0 and int(pos[-1]) == len(crd)
        parents = len(crd)
    assert converted.storage.value.numel() == parents


@pytest.mark.parametrize("fmt", ["dds", "sds", "ddss"])
def test_multi_dense_parent_zero_extents_and_all_zero(fmt):
    rank = len(fmt)
    for axis in range(rank):
        shape = [2] * rank
        shape[axis] = 0
        converted = STensor.from_torch(torch.zeros(tuple(shape)), "A").to_sparse(fmt)
        assert converted.storage.value.numel() == 0
    converted = STensor.from_torch(torch.zeros((2,) * rank), "A").to_sparse(fmt)
    assert converted.storage.value.numel() == 0
    assert torch.equal(
        _dense_reference_from_storage(converted, fmt, (2,) * rank),
        torch.zeros((2,) * rank),
    )


@pytest.mark.parametrize("fmt", ["dds", "ddss"])
def test_multi_dense_parent_float64_and_sparse_sources(fmt):
    torch.manual_seed(9)
    rank = len(fmt)
    shape = {3: (3, 4, 5), 4: (2, 3, 4, 3)}[rank]
    dense = ((torch.rand(shape) < 0.3) * torch.randn(shape)).double()
    converted = STensor.from_torch(dense.clone(), "A").to_sparse(fmt)
    assert converted.storage.value.dtype is torch.float64
    assert torch.equal(_dense_reference_from_storage(converted, fmt, shape), dense)

    source = STensor.from_torch(dense.clone().float(), "B").to_sparse("s" * rank)
    source.to_sparse(fmt)
    assert torch.equal(_dense_reference_from_storage(source, fmt, shape), dense.float())


def test_multi_dense_parent_conversion_is_exception_atomic(monkeypatch):
    torch.manual_seed(3)
    dense = (torch.rand((3, 4, 5)) < 0.3) * torch.randn((3, 4, 5))
    source = STensor.from_torch(dense.clone(), "A").to_sparse("sss")
    before = tensor_snapshot(source)

    def fail_nonzero(*args, **kwargs):
        raise RuntimeError("injected materialization failure")

    monkeypatch.setattr(torch, "nonzero", fail_nonzero)
    with pytest.raises(RuntimeError, match="injected materialization failure"):
        source.to_sparse("dds")

    assert tensor_snapshot(source) == before


def test_multi_dense_parent_conversion_validates_the_compiler_boundary():
    dense = torch.zeros((3, 4, 5))
    with pytest.raises(TypeError):
        STensor.from_torch(dense.clone(), "A").to_sparse(
            "dds", _compile_options=object()
        )
    with pytest.raises(TypeError):
        STensor.from_torch(dense.clone(), "A").to_sparse(
            "dds", _compilation_context=object()
        )


@pytest.mark.parametrize(
    "fmt,shape",
    [
        (["dense", "singleton"], (3, 4)),
        (["singleton", "compressed"], (3, 4)),
        (["singleton", "dense", "compressed"], (2, 3, 4)),
        (["dense", "dense", "singleton"], (2, 3, 4)),
    ],
)
def test_requested_singleton_levels_are_rejected_atomically(fmt, shape, monkeypatch):
    """A singleton request must fail before materialization, like rank 1.

    ``layout.validate_runtime_contract`` already declares singleton levels
    unrunnable and the rank-1 arm rejects them up front.  The rank>=2 arm
    used to run the whole JIT pipeline and then leak a bare ``ValueError``
    out of code generation, so an invalid requested format was not being
    rejected atomically at all.
    """

    torch.manual_seed(11)
    dense = (torch.rand(shape) < 0.4) * torch.randn(shape)
    source = STensor.from_torch(dense.clone(), "A")
    metadata = source.metadata
    storage = source.storage
    before = tensor_snapshot(source)

    def fail_if_compiler_setup_starts(*args, **kwargs):
        raise AssertionError("singleton rejection must precede compiler setup")

    monkeypatch.setattr(
        CompileOptions,
        "from_environment",
        classmethod(fail_if_compiler_setup_starts),
    )
    with pytest.raises(TensorStorageError, match="singleton"):
        source.to_sparse(fmt)
    assert source.metadata is metadata
    assert source.storage is storage
    assert tensor_snapshot(source) == before
