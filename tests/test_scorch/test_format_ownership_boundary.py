"""The construction-side format-ownership boundary, and what it deliberately leaves open.

``TensorFormat`` and ``LevelFormat`` are frozen value objects, but
``object.__setattr__`` still rewrites their stored state.  Until now every
object that *retained* a caller's format kept the caller's exact instance, so a
caller who held on to one could rewrite a tensor's declared layout after
construction -- and, worse, could rewrite a format shared with a process-global
kernel-resolution memo, silently changing the behaviour of later, unrelated
calls.

The boundary is installed on the way **in**, at the two construction sites that
retain a format (``TensorLayout.__post_init__`` and ``TensorIndex.__init__``),
not on every read.  Read-side rebuilding was measured as a double-digit
regression on the live dispatch path and would still not close the write side,
so it is deliberately not done.

What remains open is recorded here as an explicit characterization lock, not
hidden: a caller can still forge the object *it* holds, and forging a
returned tensor's own retained value objects still corrupts that tensor's
metadata.  Closing that needs structurally unforgeable value types, which is a
separate change to four core types (equality, hashing, pickling, and the
dataclass surface) and is not attempted here.
"""

import torch

import scorch
from scorch.format import (
    LevelFormat,
    LevelType,
    TensorFormat,
    audit_format_state,
    owned_format,
    parse_format,
)
from scorch.layout import TensorLayout
from scorch.stensor import STensor
from scorch.storage import TensorIndex


def csr(name="A", size=3):
    tensor = STensor.from_torch(torch.eye(size), name)
    tensor.to_sparse("ds")
    return tensor


# -- the boundary holds ------------------------------------------------------


def test_layout_does_not_retain_the_caller_format():
    caller = TensorFormat("ds")
    layout = TensorLayout.from_logical_shape((4, 5), caller)
    assert layout.format is not caller
    assert layout.format == caller


def test_layout_does_not_retain_the_caller_level_formats():
    caller = TensorFormat([LevelFormat("d"), LevelFormat("s", bit_width=32)])
    layout = TensorLayout.from_logical_shape((4, 5), caller)
    for retained, given in zip(
        layout.format.get_level_formats(), caller.get_level_formats()
    ):
        assert retained is not given
        assert retained == given


def test_index_does_not_retain_the_caller_format():
    caller = TensorFormat("ds")
    index = TensorIndex(
        tensor_format=caller,
        mode_indices=[[], [torch.tensor([0, 1]), torch.tensor([0])]],
    )
    assert index.format is not caller


def test_forging_the_caller_format_cannot_change_the_layout():
    caller = TensorFormat("ds")
    layout = TensorLayout.from_logical_shape((4, 5), caller)
    object.__setattr__(caller, "_level_formats", (LevelFormat("o"), LevelFormat("o")))
    assert list(layout.format.get_level_types()) == [
        LevelType.DENSE,
        LevelType.COMPRESSED,
    ]


def test_forging_a_caller_level_format_cannot_change_the_layout():
    level = LevelFormat("s")
    caller = TensorFormat([LevelFormat("d"), level])
    layout = TensorLayout.from_logical_shape((4, 5), caller)
    object.__setattr__(level, "_mode", LevelType.COORDINATE)
    assert layout.format.get_level_types()[1] is LevelType.COMPRESSED


def test_two_tensors_built_from_one_caller_format_do_not_share_it():
    caller = TensorFormat("ds")
    first = TensorLayout.from_logical_shape((4, 5), caller)
    second = TensorLayout.from_logical_shape((6, 7), caller)
    assert first.format is not second.format


def test_forging_a_result_format_does_not_poison_later_unrelated_calls():
    """The worst verified consequence of the seam is closed.

    The prebuilt matmul resolution memo is process-global; before the boundary
    it handed its own cached ``TensorFormat`` object straight through to the
    result tensor, so one ``object.__setattr__`` on a returned result rewrote
    the memo and changed every later ``matmul(..., format='ds')`` in the
    process.
    """

    first = scorch.matmul(csr("A"), csr("B"), format="ds")
    object.__setattr__(
        first.format, "_level_formats", (LevelFormat(LevelType.COORDINATE),)
    )
    later = scorch.matmul(csr("C"), csr("D"), format="ds")
    assert list(later.format.get_level_types()) == [
        LevelType.DENSE,
        LevelType.COMPRESSED,
    ]
    assert torch.allclose(later.to_torch(), torch.eye(3), atol=1e-6)


# -- fail-open behaviour is preserved ----------------------------------------


class _FormatSubclass(TensorFormat):
    pass


def test_a_format_subclass_is_still_accepted_unchanged():
    """The boundary must not tighten what construction accepted before."""

    caller = _FormatSubclass("ds")
    layout = TensorLayout.from_logical_shape((4, 5), caller)
    assert layout.format is caller


def test_owned_format_returns_non_exact_inputs_unchanged():
    caller = _FormatSubclass("ds")
    assert owned_format(caller) is caller
    assert audit_format_state(caller) is None


def test_owned_format_rebuilds_an_exact_format():
    caller = TensorFormat("ds")
    owned = owned_format(caller)
    assert owned is not caller
    assert owned == caller


def test_audit_rejects_forged_stored_state():
    caller = TensorFormat("ds")
    object.__setattr__(caller, "_level_formats", [LevelFormat("d")])
    assert audit_format_state(caller) is None


def test_audit_rejects_a_forged_bit_width():
    level = LevelFormat("s")
    object.__setattr__(level, "_bit_width", -1)
    assert audit_format_state(TensorFormat([LevelFormat("d"), level])) is None


def test_audit_preserves_bit_widths():
    caller = TensorFormat([LevelFormat("d", bit_width=16), LevelFormat("s")])
    levels = audit_format_state(caller)
    assert levels is not None
    assert levels[0].bit_width == 16
    assert levels[1].bit_width is None


def test_parse_format_still_passes_through_an_exact_format():
    """The boundary is at construction, not in the parser."""

    caller = TensorFormat("ds")
    assert parse_format(caller) is caller


# -- the deliberately deferred half, locked explicitly -----------------------


def test_a_returned_tensor_still_exposes_its_own_retained_format():
    """CHARACTERIZATION LOCK -- this is known-open, not a passing guarantee.

    Reads are not defended: ``tensor.format`` is the tensor's own retained
    value object, and forging it still desynchronizes that tensor's declared
    layout from its index arrays.  The damage is now confined to the tensor
    whose format was forged -- it no longer escapes into a process-global memo
    or into an unrelated tensor built from the same caller value.

    Closing this needs structurally unforgeable value types (a change to
    ``LevelFormat``, ``TensorFormat``, ``TensorLayout`` and ``TensorMetadata``
    covering equality, hashing, pickling and the dataclass surface), which is
    not attempted here.  Update this test when that lands.
    """

    tensor = csr("A")
    retained = tensor.format
    assert retained is tensor.layout.format
    object.__setattr__(retained, "_level_formats", (LevelFormat("o"), LevelFormat("o")))
    assert list(tensor.format.get_level_types()) == [
        LevelType.COORDINATE,
        LevelType.COORDINATE,
    ]


def test_a_copy_still_shares_its_source_metadata():
    """CHARACTERIZATION LOCK -- ``copy()`` passes metadata through unchanged."""

    tensor = csr("A")
    duplicate = tensor.copy()
    assert duplicate.metadata is tensor.metadata
    assert duplicate.format is tensor.format


def test_index_arrays_are_already_defensive_on_public_reads():
    """The index arrays were already copied on every public accessor."""

    tensor = csr("A")
    first = tensor.index.mode_indices
    second = tensor.index.mode_indices
    assert first[1][0] is not second[1][0]
    assert torch.equal(first[1][0], second[1][0])
