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

from abc import ABCMeta
from array import array
from collections import OrderedDict, deque
from collections.abc import Sequence
import enum
import json
from types import MappingProxyType
from typing import ClassVar, List

import pytest
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
from scorch.exceptions import (
    CompileSpecError,
    TensorDeviceError,
    TensorFormatError,
    TensorIndexError,
    TensorLayoutError,
    TensorTypeError,
)
from scorch.layout import TensorLayout, TensorMetadata, TensorSpec
from scorch.stensor import STensor
from scorch.storage import SparseStorage, TensorIndex, TensorStorage


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


def test_storages_do_not_retain_or_cross_share_a_caller_layout():
    caller = TensorLayout.from_logical_shape((2, 2), "dd")
    first = SparseStorage(caller, torch.arange(4.0), mode_indices=((), ()))
    second = SparseStorage(caller, torch.arange(4.0, 8.0), mode_indices=((), ()))

    assert first.layout is not caller
    assert second.layout is not caller
    assert first.layout is not second.layout
    object.__setattr__(caller, "permutation", (1, 0))
    first.validate()
    second.validate()
    assert first.layout.permutation == (0, 1)
    assert second.layout.permutation == (0, 1)


def test_tensors_do_not_cross_share_an_explicit_caller_storage_graph():
    layout = TensorLayout.from_logical_shape((2, 2), "dd")
    caller = SparseStorage(layout, torch.arange(4.0), mode_indices=((), ()))
    first = STensor("A", storage=caller)
    second = STensor("B", storage=caller)

    assert first.storage is not caller
    assert second.storage is not caller
    assert first.storage is not second.storage
    assert first.storage.value.data_ptr() == caller.value.data_ptr()
    assert second.storage.value.data_ptr() == caller.value.data_ptr()
    assert first.layout is not second.layout
    assert first.format is not second.format

    object.__setattr__(first.format, "_level_formats", (LevelFormat("o"),))
    assert second.format == TensorFormat("dd")
    second.storage.validate()


def test_foreign_tensor_class_spoof_fails_closed_at_storage_boundary():
    class TensorSpoof:
        @property
        def __class__(self):
            return torch.Tensor

    layout = TensorLayout.from_logical_shape((1,), "d")
    with pytest.raises(TensorTypeError, match="torch.Tensor"):
        SparseStorage(layout, TensorSpoof(), mode_indices=[[]])  # type: ignore[arg-type]


def test_value_tensor_subclass_properties_cannot_interpose_on_storage():
    class ValueBomb(torch.Tensor):
        @staticmethod
        def __new__(cls, value):
            return torch.Tensor._make_subclass(cls, value, False)

        @property
        def layout(self):
            raise RuntimeError("subclass layout must not run")

    caller = ValueBomb(torch.tensor([2.0]))
    storage = SparseStorage(
        TensorLayout.from_logical_shape((1,), "d"), caller, mode_indices=[[]]
    )
    assert type(storage._value) is torch.Tensor
    assert storage.value.data_ptr() == caller.data_ptr()
    storage.validate()


def test_metadata_does_not_retain_a_caller_layout_graph():
    caller = TensorLayout.from_logical_shape((2, 3), "dd")
    metadata = TensorMetadata("A", torch.float32, torch.device("cpu"), caller)

    assert metadata.layout is not caller
    assert metadata.layout.format is not caller.format
    object.__setattr__(caller, "permutation", (1, 0))
    object.__setattr__(
        caller.format,
        "_level_formats",
        (LevelFormat("o"), LevelFormat("o")),
    )
    assert metadata.layout.permutation == (0, 1)
    assert metadata.layout.format == TensorFormat("dd")
    metadata.serialize()


def test_storages_do_not_retain_or_cross_share_tensor_index_arrays():
    caller_index = TensorIndex(
        "s",
        [[torch.tensor([0, 1]), torch.tensor([0])]],
    )
    layout = TensorLayout.from_logical_shape((3,), "s", index_dtype=torch.int64)
    first = SparseStorage(layout, torch.tensor([5.0]), index=caller_index)
    second = SparseStorage(layout, torch.tensor([7.0]), index=caller_index)

    caller_coordinate = caller_index._mode_indices[0][1]
    first_coordinate = first._mode_indices[0][1]
    second_coordinate = second._mode_indices[0][1]
    assert first_coordinate is not caller_coordinate
    assert second_coordinate is not caller_coordinate
    assert first_coordinate is not second_coordinate

    caller_coordinate[0] = 2
    first.validate()
    second.validate()
    assert first_coordinate.tolist() == [0]
    assert second_coordinate.tolist() == [0]


class _ToggleInt(int):
    armed = False

    def __int__(self):
        raise RuntimeError("subclass conversion must not run")

    def __hash__(self):
        if self.armed:
            raise RuntimeError("hash bomb")
        return int.__hash__(self)

    def __eq__(self, other):
        if self.armed:
            raise RuntimeError("equality bomb")
        return int.__eq__(self, other)


def test_retaining_boundaries_canonicalize_integer_subclasses():
    extent = _ToggleInt(3)
    mode = _ToggleInt(0)
    layout = TensorLayout.from_logical_shape(
        (extent,), "s", (mode,), index_dtype=torch.int64
    )
    index = TensorIndex(
        "s",
        [[torch.tensor([0, 1]), torch.tensor([1])]],
        mode_order=(mode,),
    )

    assert type(layout.logical_shape[0]) is int
    assert type(layout.physical_shape[0]) is int
    assert type(layout.permutation[0]) is int
    assert type(index._mode_order[0]) is int

    extent.armed = True
    mode.armed = True
    assert layout.element_count == 3
    SparseStorage(layout, torch.tensor([5.0]), index=index).validate()


def test_redundant_storage_shapes_canonicalize_integer_subclasses():
    class EqBombInt(int):
        def __eq__(self, other):
            raise RuntimeError("subclass equality must not run")

    layout = TensorLayout.from_logical_shape((2,), "d")
    storage = SparseStorage(layout, torch.ones(2), mode_indices=[[]])
    tensor = STensor("A", shape=(EqBombInt(2),), storage=storage)
    assert tensor.shape == (2,)
    compatibility = TensorStorage(
        layout=layout,
        shape=(EqBombInt(2),),
        value=torch.ones(2),
        mode_indices=[[]],
    )
    compatibility.validate()


def test_tensor_storage_owns_layout_before_redundant_shape_comparison():
    class LayoutBomb(TensorLayout):
        armed = False

        def __getattribute__(self, name):
            if name == "physical_shape" and type(self).armed:
                raise RuntimeError("subclass layout property must not run")
            return super().__getattribute__(name)

    caller = LayoutBomb.from_logical_shape((2,), "d")
    LayoutBomb.armed = True
    storage = TensorStorage(
        layout=caller,
        shape=(2,),
        value=torch.ones(2),
        mode_indices=[[]],
    )
    assert type(storage.layout) is TensorLayout
    storage.validate()

    class LayoutSpoof:
        @property
        def __class__(self):
            return TensorLayout

    with pytest.raises(TensorTypeError, match="TensorLayout"):
        TensorStorage(
            layout=LayoutSpoof(),  # type: ignore[arg-type]
            shape=(2,),
            value=torch.ones(2),
            mode_indices=[[]],
        )


@pytest.mark.parametrize("field", ["shape", "permutation", "mode_order"])
def test_foreign_integer_class_spoofs_fail_closed(field):
    class IntSpoof:
        @property
        def __class__(self):
            return int

    spoof = IntSpoof()
    with pytest.raises(TensorTypeError, match="integer"):
        if field == "shape":
            TensorLayout.from_logical_shape((spoof,), "d")  # type: ignore[arg-type]
        elif field == "permutation":
            TensorLayout.from_logical_shape((3,), "d", (spoof,))  # type: ignore[arg-type]
        else:
            TensorIndex("d", [[]], mode_order=(spoof,))  # type: ignore[arg-type]


def test_metadata_canonicalizes_a_string_subclass():
    class ToggleStr(str):
        armed = False

        def strip(self, *args):
            return self

        def __hash__(self):
            if self.armed:
                raise RuntimeError("hash bomb")
            return str.__hash__(self)

    caller = ToggleStr(" A ")
    metadata = TensorMetadata(
        caller,
        torch.float32,
        torch.device("cpu"),
        TensorLayout.from_logical_shape((1,), "d"),
    )
    assert type(metadata.name) is str
    assert metadata.name == "A"
    caller.armed = True
    hash(metadata)


def test_foreign_scalar_class_spoofs_fail_closed():
    class StrSpoof:
        @property
        def __class__(self):
            return str

    class BoolSpoof:
        @property
        def __class__(self):
            return bool

    class DtypeSpoof:
        @property
        def __class__(self):
            return torch.dtype

    layout = TensorLayout.from_logical_shape((1,), "d")
    with pytest.raises(TensorLayoutError, match="name"):
        TensorMetadata(  # type: ignore[arg-type]
            StrSpoof(), torch.float32, torch.device("cpu"), layout
        )
    with pytest.raises(TensorTypeError, match="requires_grad"):
        TensorMetadata(  # type: ignore[arg-type]
            "A", torch.float32, torch.device("cpu"), layout, BoolSpoof()
        )
    with pytest.raises(TensorTypeError, match="requires_grad"):
        STensor(storage=SparseStorage(layout, torch.ones(1), mode_indices=[[]]), requires_grad=BoolSpoof())  # type: ignore[arg-type]
    with pytest.raises(TensorTypeError, match="torch.dtype"):
        TensorLayout.from_logical_shape(  # type: ignore[arg-type]
            (1,), "d", index_dtype=DtypeSpoof()
        )
    with pytest.raises(TensorTypeError, match="torch.dtype"):
        TensorMetadata(  # type: ignore[arg-type]
            "A", DtypeSpoof(), torch.device("cpu"), layout
        )
    with pytest.raises(TensorIndexError, match="index_dtype"):
        TensorIndex("d", [[]], index_dtype=DtypeSpoof())  # type: ignore[arg-type]


def test_metadata_device_failure_does_not_render_a_hostile_value():
    class DeviceBomb:
        def __repr__(self):
            raise OverflowError("repr bomb")

    with pytest.raises(TensorDeviceError, match="DeviceBomb"):
        TensorMetadata(  # type: ignore[arg-type]
            "A",
            torch.float32,
            DeviceBomb(),
            TensorLayout.from_logical_shape((1,), "d"),
        )


def test_tensor_spec_dtype_failure_does_not_render_a_hostile_value():
    class DtypeBomb:
        def __str__(self):
            raise ValueError("str bomb")

    with pytest.raises(CompileSpecError, match="torch.dtype"):
        TensorSpec("d", (1,), dtype=DtypeBomb())  # type: ignore[arg-type]


def test_metadata_setters_preserve_the_storage_layout_owner():
    tensor = csr("A")
    tensor.name = "renamed"
    assert tensor.name == "renamed"
    assert tensor.metadata.layout is tensor.storage.layout
    tensor.storage.validate()

    tensor.requires_grad = True
    assert tensor.requires_grad is True
    assert tensor.metadata.layout is tensor.storage.layout
    tensor.storage.validate()


def test_index_tensor_subclasses_cannot_cross_retaining_boundaries():
    class StickyTensor(torch.Tensor):
        @staticmethod
        def __new__(cls, value):
            return torch.Tensor._make_subclass(cls, value, False)

        def detach(self):
            return self

        def clone(self, *args, **kwargs):
            return self

    caller_positions = StickyTensor(torch.tensor([0, 1], dtype=torch.int64))
    caller_coordinates = StickyTensor(torch.tensor([1], dtype=torch.int64))
    index = TensorIndex("s", [[caller_positions, caller_coordinates]])
    layout = TensorLayout.from_logical_shape((3,), "s", index_dtype=torch.int64)
    storage = SparseStorage(layout, torch.tensor([5.0]), index=index)

    for retained, caller in zip(
        index._mode_indices[0], (caller_positions, caller_coordinates)
    ):
        assert type(retained) is torch.Tensor
        assert retained.data_ptr() != caller.data_ptr()
    for retained, caller in zip(
        storage._mode_indices[0], (caller_positions, caller_coordinates)
    ):
        assert type(retained) is torch.Tensor
        assert retained.data_ptr() != caller.data_ptr()

    caller_coordinates[0] = 2
    storage.validate()
    assert storage._mode_indices[0][1].tolist() == [1]


def test_sparse_storage_audits_tensor_index_stored_state_without_properties():
    class PropertyBombIndex(TensorIndex):
        @property
        def format(self):
            raise RuntimeError("format property must not run")

        @property
        def mode_order(self):
            raise RuntimeError("mode-order property must not run")

        @property
        def index_dtype(self):
            raise RuntimeError("dtype property must not run")

    index = PropertyBombIndex("s", [[torch.tensor([0, 1]), torch.tensor([1])]])
    layout = TensorLayout.from_logical_shape((3,), "s", index_dtype=torch.int64)
    storage = SparseStorage(layout, torch.tensor([5.0]), index=index)
    storage.validate()
    wrapped = STensor("A", shape=(3,), index=index, value=torch.tensor([5.0]))
    wrapped.storage.validate()
    compatibility = TensorStorage(index=index, value=torch.tensor([5.0]), shape=(3,))
    compatibility.validate()


def test_tensor_index_subclass_dict_descriptor_cannot_interpose_on_audit():
    class DictBombIndex(TensorIndex):
        @property
        def __dict__(self):
            raise RuntimeError("subclass __dict__ must not run")

    index = DictBombIndex("s", [[torch.tensor([0, 1]), torch.tensor([1])]])
    layout = TensorLayout.from_logical_shape((3,), "s", index_dtype=torch.int64)
    SparseStorage(layout, torch.tensor([5.0]), index=index).validate()


@pytest.mark.parametrize(
    "malformation",
    ["missing_format", "missing_order", "missing_dtype", "missing_indices", "extra"],
)
def test_sparse_storage_rejects_malformed_tensor_index_state(malformation):
    index = TensorIndex("s", [[torch.tensor([0, 1]), torch.tensor([1])]])
    field = {
        "missing_format": "_format",
        "missing_order": "_mode_order",
        "missing_dtype": "_index_dtype",
        "missing_indices": "_mode_indices",
    }.get(malformation)
    if field is None:
        object.__setattr__(index, "_extra", object())
    else:
        del index.__dict__[field]
    layout = TensorLayout.from_logical_shape((3,), "s", index_dtype=torch.int64)
    with pytest.raises(TensorIndexError, match="malformed stored state"):
        SparseStorage(layout, torch.tensor([5.0]), index=index)


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


# -- canonical subclasses and malformed state -------------------------------


class _FormatSubclass(TensorFormat):
    pass


class _LevelFormatSubclass(LevelFormat):
    pass


def test_a_canonical_format_subclass_is_accepted_but_detached():
    """Subclass values keep their semantics without crossing ownership."""

    caller = _FormatSubclass("ds")
    layout = TensorLayout.from_logical_shape((4, 5), caller)
    assert layout.format is not caller
    assert type(layout.format) is TensorFormat
    assert layout.format.serialize() == caller.serialize()


def test_owned_format_canonicalizes_a_structurally_plain_subclass():
    caller = _FormatSubclass("ds")
    owned = owned_format(caller)
    assert owned is not caller
    assert type(owned) is TensorFormat
    assert owned.serialize() == caller.serialize()
    assert audit_format_state(caller) is not None


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


class _Width(enum.IntEnum):
    W32 = 32


class _HostileWidth(int):
    def __int__(self):
        raise AssertionError("subclass __int__ must not run")

    def __index__(self):
        raise AssertionError("subclass __index__ must not run")

    def __le__(self, other):
        raise AssertionError("subclass comparison must not run")


class _LyingNegativeWidth(int):
    def __le__(self, other):
        return False


class _ForeignIntSpoof:
    @property
    def __class__(self):
        return int

    def __le__(self, other):
        return False


class _ForeignLevelTypeSpoof:
    @property
    def __class__(self):
        return LevelType

    def __str__(self):
        raise AssertionError("foreign mode __str__ must not run")

    def __repr__(self):
        raise AssertionError("foreign mode __repr__ must not run")

    def __format__(self, spec):
        raise AssertionError("foreign mode __format__ must not run")


class _ForeignLevelFormatSpoof:
    @property
    def __class__(self):
        return LevelFormat

    def __str__(self):
        raise AssertionError("foreign level __str__ must not run")

    def __repr__(self):
        raise AssertionError("foreign level __repr__ must not run")


class _HostileLevelAlias(str):
    def strip(self, *args):
        raise AssertionError("alias strip override must not run")

    def lower(self):
        raise AssertionError("alias lower override must not run")

    def __str__(self):
        raise AssertionError("alias __str__ must not run")

    def __repr__(self):
        raise AssertionError("alias __repr__ must not run")

    def __format__(self, spec):
        raise AssertionError("alias __format__ must not run")


class _PlainLevelFormatSubclass(LevelFormat):
    pass


class _NameDescriptorBombMeta(type):
    reads = 0

    @property
    def __name__(cls):
        _NameDescriptorBombMeta.reads += 1
        raise AssertionError("metaclass __name__ descriptor must not run")


class _ForeignActualType(metaclass=_NameDescriptorBombMeta):
    pass


class _HostileClassName(str):
    callbacks = 0

    def __str__(self):
        type(self).callbacks += 1
        raise AssertionError("class-name __str__ must not run")

    def __repr__(self):
        type(self).callbacks += 1
        raise AssertionError("class-name __repr__ must not run")

    def __format__(self, spec):
        type(self).callbacks += 1
        raise AssertionError("class-name __format__ must not run")


class _ClassGuardMeta(ABCMeta):
    reads = 0

    def __getattribute__(cls, name):
        if name == "__class__":
            _ClassGuardMeta.reads += 1
            raise AssertionError("sequence metaclass __class__ must not run")
        return ABCMeta.__getattribute__(cls, name)


class _GuardedSequence(Sequence, metaclass=_ClassGuardMeta):
    def __init__(self, values):
        self._values = tuple(values)

    def __len__(self):
        return len(self._values)

    def __getitem__(self, index):
        return self._values[index]


def test_audit_canonicalizes_a_constructor_valid_int_subclass_bit_width():
    """An ``IntEnum`` width is a legal format, not malformed stored state.

    ``LevelFormat.__init__`` accepts any non-bool ``int`` subclass, so
    rejecting one at the retaining boundary would turn a construction the
    public constructor accepts into an error.
    """

    level = LevelFormat("s", bit_width=_Width.W32)
    assert type(level.bit_width) is not int
    levels = audit_format_state(TensorFormat([LevelFormat("d"), level]))
    assert levels is not None
    assert levels[1].bit_width == 32
    assert type(levels[1].bit_width) is int


def test_level_format_validates_integer_subclasses_without_callbacks():
    width = _HostileWidth(32)
    level = LevelFormat("s", bit_width=width)
    assert level.bit_width is width

    levels = audit_format_state(TensorFormat([level]))
    assert levels is not None
    assert levels[0].bit_width == 32
    assert type(levels[0].bit_width) is int


def test_level_format_rejects_a_lying_negative_integer_subclass():
    with pytest.raises(TensorFormatError, match="positive"):
        LevelFormat("s", bit_width=_LyingNegativeWidth(-8))


def test_level_format_rejects_a_foreign_int_class_spoof():
    spoof = _ForeignIntSpoof()
    assert isinstance(spoof, int)
    with pytest.raises(TensorTypeError, match="integer"):
        LevelFormat("s", bit_width=spoof)  # type: ignore[arg-type]


def test_level_format_rejects_a_foreign_level_type_class_spoof_without_callbacks():
    spoof = _ForeignLevelTypeSpoof()
    assert isinstance(spoof, LevelType)
    with pytest.raises(TensorTypeError, match="string alias or LevelType"):
        LevelFormat(spoof)  # type: ignore[arg-type]


@pytest.mark.parametrize("nested", [False, True])
def test_tensor_format_rejects_foreign_level_format_class_spoofs(nested):
    spoof = _ForeignLevelFormatSpoof()
    assert isinstance(spoof, LevelFormat)
    value = [spoof] if nested else spoof
    with pytest.raises(TensorTypeError, match="tensor format"):
        TensorFormat(value)  # type: ignore[arg-type]


@pytest.mark.parametrize("nested", [False, True])
def test_tensor_format_preserves_real_level_format_subclasses(nested):
    level = _PlainLevelFormatSubclass("s", bit_width=32)
    value = [level] if nested else level
    tensor_format = TensorFormat(value)
    assert tensor_format.get_level_formats()[0] is level

    retained = owned_format(tensor_format).get_level_formats()[0]
    assert type(retained) is LevelFormat
    assert retained.get_level_type() is LevelType.COMPRESSED
    assert retained.bit_width == 32


def test_real_string_subclass_aliases_bypass_every_override():
    alias = _HostileLevelAlias(" Dense ")
    formats = (
        TensorFormat([LevelFormat(alias)]),
        TensorFormat(alias),
        TensorFormat([alias]),
    )
    assert all(
        tensor_format.get_level_types() == [LevelType.DENSE]
        for tensor_format in formats
    )
    assert all(
        audit_format_state(tensor_format) is not None for tensor_format in formats
    )


def test_invalid_string_subclass_alias_reports_without_rendering_it():
    with pytest.raises(TensorFormatError, match="invalid level format 'not-a-mode'"):
        LevelFormat(_HostileLevelAlias("not-a-mode"))


def test_rejected_actual_types_bypass_metaclass_name_descriptors():
    _NameDescriptorBombMeta.reads = 0
    foreign = _ForeignActualType()

    with pytest.raises(TensorTypeError, match="got _ForeignActualType"):
        LevelFormat(foreign)  # type: ignore[arg-type]
    with pytest.raises(TensorTypeError, match="got _ForeignActualType"):
        TensorFormat([foreign])  # type: ignore[list-item]

    assert _NameDescriptorBombMeta.reads == 0


def test_rejected_actual_type_names_are_canonicalized_before_rendering():
    class Foreign:
        pass

    Foreign.__name__ = _HostileClassName("Foreign")
    _HostileClassName.callbacks = 0
    foreign = Foreign()

    with pytest.raises(TensorTypeError, match="got Foreign"):
        LevelFormat(foreign)  # type: ignore[arg-type]
    with pytest.raises(TensorTypeError, match="got Foreign"):
        TensorFormat([foreign])  # type: ignore[list-item]

    assert _HostileClassName.callbacks == 0


def test_real_sequence_mro_check_bypasses_the_candidate_metaclass():
    _ClassGuardMeta.reads = 0
    value = _GuardedSequence(("d", "s"))

    direct = TensorFormat(value)
    parsed = parse_format(value)  # type: ignore[arg-type]

    assert direct.get_level_types() == [LevelType.DENSE, LevelType.COMPRESSED]
    assert parsed.get_level_types() == [LevelType.DENSE, LevelType.COMPRESSED]
    assert _ClassGuardMeta.reads == 0


@pytest.mark.parametrize("owner", ["layout", "index"])
def test_retaining_boundaries_accept_an_int_subclass_bit_width(owner):
    caller = TensorFormat([LevelFormat("d"), LevelFormat("s", bit_width=_Width.W32)])
    if owner == "layout":
        retained = TensorLayout.from_logical_shape((3, 4), caller).format
    else:
        retained = TensorIndex(
            caller,
            [[], [torch.tensor([0, 0, 0, 0]), torch.tensor([], dtype=torch.long)]],
        ).format
    assert retained is not caller
    width = retained.get_level_formats()[1].bit_width
    assert width == 32
    assert type(width) is int


@pytest.mark.parametrize("forged", [True, 0, -8])
def test_audit_still_rejects_bool_and_non_positive_bit_widths(forged):
    """Canonicalizing int subclasses must not widen the accepted set."""

    level = LevelFormat("s")
    object.__setattr__(level, "_bit_width", forged)
    assert audit_format_state(TensorFormat([LevelFormat("d"), level])) is None


@pytest.mark.parametrize("owner", ["layout", "index"])
@pytest.mark.parametrize(
    "malformation", ["list_levels", "tuple_subclass", "extra_key", "invalid_level"]
)
def test_retaining_boundaries_reject_malformed_format_state(owner, malformation):
    class TupleSubclass(tuple):
        pass

    caller = TensorFormat("ds")
    levels = caller.get_level_formats()
    if malformation == "list_levels":
        object.__setattr__(caller, "_level_formats", list(levels))
    elif malformation == "tuple_subclass":
        object.__setattr__(caller, "_level_formats", TupleSubclass(levels))
    elif malformation == "extra_key":
        object.__setattr__(caller, "_extra", object())
    else:
        object.__setattr__(levels[1], "_mode", "s")

    with pytest.raises(TensorFormatError, match="malformed stored state"):
        if owner == "layout":
            TensorLayout.from_logical_shape((2, 3), caller)
        else:
            TensorIndex(
                tensor_format=caller,
                mode_indices=[[], [torch.tensor([0, 1, 1]), torch.tensor([0])]],
            )


def test_two_layouts_built_from_one_subclass_do_not_share_it():
    caller = _FormatSubclass("ds")
    first = TensorLayout.from_logical_shape((2, 3), caller)
    second = TensorLayout.from_logical_shape((4, 5), caller)
    assert first.format is not caller
    assert second.format is not caller
    assert first.format is not second.format


def test_canonical_level_subclass_is_accepted_and_detached():
    level = _LevelFormatSubclass("s", bit_width=64)
    caller = TensorFormat([LevelFormat("d"), level])
    layout = TensorLayout.from_logical_shape((2, 3), caller)
    retained = layout.format.get_level_formats()[1]
    assert type(retained) is LevelFormat
    assert retained is not level
    assert retained.bit_width == 64

    object.__setattr__(level, "_mode", LevelType.COORDINATE)
    assert retained.get_level_type() is LevelType.COMPRESSED


def test_outer_subclass_dict_descriptor_cannot_interpose_on_the_audit():
    class FormatWithDictBomb(TensorFormat):
        @property
        def __dict__(self):
            raise RuntimeError("subclass __dict__ must not run")

    caller = FormatWithDictBomb("ds")
    layout = TensorLayout.from_logical_shape((2, 3), caller)
    assert type(layout.format) is TensorFormat
    assert layout.format.serialize() == TensorFormat("ds").serialize()


def test_nested_subclass_dict_descriptor_cannot_interpose_on_the_audit():
    class LevelWithDictBomb(LevelFormat):
        @property
        def __dict__(self):
            raise RuntimeError("subclass __dict__ must not run")

    caller = TensorFormat([LevelFormat("d"), LevelWithDictBomb("s")])
    layout = TensorLayout.from_logical_shape((2, 3), caller)
    assert all(
        type(level) is LevelFormat for level in layout.format.get_level_formats()
    )
    assert layout.format.serialize() == TensorFormat("ds").serialize()


def test_a_foreign_class_spoof_cannot_cross_the_ownership_boundary():
    class Spoof:
        @property
        def __class__(self):
            return TensorFormat

        def get_order(self):
            return 2

        def get_level_types(self):
            return [LevelType.DENSE, LevelType.COMPRESSED]

    spoof = Spoof()
    assert isinstance(spoof, TensorFormat)
    with pytest.raises(TensorTypeError, match="tensor format must be"):
        TensorLayout.from_logical_shape((2, 3), spoof)  # type: ignore[arg-type]


def test_a_raising_class_spoof_is_rejected_without_reading_class():
    class ClassBomb:
        class_reads = 0

        @property
        def __class__(self):
            type(self).class_reads += 1
            raise RuntimeError("class bomb")

    with pytest.raises(TensorTypeError, match="tensor format must be"):
        parse_format(ClassBomb())  # type: ignore[arg-type]
    assert ClassBomb.class_reads == 0


def test_audit_preserves_bit_widths():
    caller = TensorFormat([LevelFormat("d", bit_width=16), LevelFormat("s")])
    levels = audit_format_state(caller)
    assert levels is not None
    assert levels[0].bit_width == 16
    assert levels[1].bit_width is None


def test_constructor_valid_large_bit_width_is_owned_without_aliasing():
    level = LevelFormat("d", bit_width=1 << 80)
    caller = TensorFormat([level])
    layout = TensorLayout.from_logical_shape((3,), caller)
    assert layout.format is not caller
    assert layout.format.get_level_formats()[0] is not level
    assert layout.format.get_level_formats()[0].bit_width == 1 << 80

    object.__setattr__(level, "_mode", LevelType.COORDINATE)
    assert layout.format.get_level_types() == [LevelType.DENSE]


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


def test_a_copy_detaches_the_complete_metadata_graph():
    tensor = csr("A")
    tensor.requires_grad = True
    duplicate = tensor.copy()
    assert duplicate.metadata is not tensor.metadata
    assert duplicate.layout is not tensor.layout
    assert duplicate.format is not tensor.format
    assert duplicate.metadata.layout is duplicate.storage.layout
    assert duplicate.metadata == tensor.metadata
    assert duplicate.requires_grad is True
    assert duplicate.name == tensor.name
    assert duplicate.shape == tensor.shape

    object.__setattr__(
        duplicate.format,
        "_level_formats",
        (LevelFormat("o"), LevelFormat("o")),
    )
    assert list(tensor.format.get_level_types()) == [
        LevelType.DENSE,
        LevelType.COMPRESSED,
    ]


def test_dense_to_dense_copy_detaches_metadata():
    tensor = STensor.from_torch(torch.eye(3), "A")
    duplicate = tensor.to_dense(in_place=False)
    assert duplicate.metadata is not tensor.metadata
    assert duplicate.layout is not tensor.layout
    assert duplicate.format is not tensor.format

    object.__setattr__(
        duplicate.format,
        "_level_formats",
        (LevelFormat("o"), LevelFormat("o")),
    )
    assert tensor.format.is_dense()


def test_index_arrays_are_already_defensive_on_public_reads():
    """The index arrays were already copied on every public accessor."""

    tensor = csr("A")
    first = tensor.index.mode_indices
    second = tensor.index.mode_indices
    assert first[1][0] is not second[1][0]
    assert torch.equal(first[1][0], second[1][0])


# -- runtime sequence/mapping recognition runs no caller-owned code ---------
#
# The recognizers used to be ``issubclass``/``isinstance`` against the
# ``collections.abc`` ABCs.  ``ABCMeta``'s subclass check inserts the
# candidate into its positive and negative ``WeakSet`` caches, and those
# inserts call the candidate metaclass's ``__hash__`` and ``__eq__`` -- so a
# caller-owned metaclass observed public format validation, and a raising one
# escaped the public constructor as a bare ``builtins.RuntimeError``.


class _HashCountingMeta(type):
    """A metaclass that records every hash/equality the runtime asks of it."""

    calls: ClassVar[List[str]] = []

    def __hash__(cls):
        _HashCountingMeta.calls.append("__hash__")
        return type.__hash__(cls)

    def __eq__(cls, other):
        _HashCountingMeta.calls.append("__eq__")
        return cls is other


class _HashCountingCandidate(metaclass=_HashCountingMeta):
    pass


class _RaisingHashMeta(type):
    def __hash__(cls):
        raise RuntimeError("caller metaclass code ran inside scorch validation")


class _RaisingHashCandidate(metaclass=_RaisingHashMeta):
    pass


class _CountingSequenceMeta(ABCMeta):
    calls: ClassVar[List[str]] = []

    def __hash__(cls):
        _CountingSequenceMeta.calls.append("__hash__")
        return ABCMeta.__hash__(cls)


class _CountingSequence(Sequence, metaclass=_CountingSequenceMeta):
    def __init__(self, values):
        self._values = tuple(values)

    def __len__(self):
        return len(self._values)

    def __getitem__(self, index):
        return self._values[index]


class _ExplodingSequence(Sequence):
    def __len__(self):
        return 1

    def __getitem__(self, index):
        raise RuntimeError("sequence consumption escaped")


class _ExplodingDict(dict):
    def __contains__(self, key):
        raise RuntimeError("mapping consumption escaped")


class _ExplodingFill:
    def __ne__(self, other):
        raise RuntimeError("fill comparison escaped")


class _ExplodingKey(str):
    def __eq__(self, other):
        raise RuntimeError("mapping key comparison escaped")

    __hash__ = str.__hash__


def test_foreign_sequence_rejection_invokes_no_metaclass_hook():
    """Rejecting a non-sequence must not consult the candidate's metaclass."""

    _HashCountingMeta.calls = []
    with pytest.raises(TensorTypeError, match="sequence of levels"):
        TensorFormat(_HashCountingCandidate())
    # The ABC-based check measured eight ``__hash__`` calls here.
    assert _HashCountingMeta.calls == []

    _HashCountingMeta.calls = []
    with pytest.raises(TensorTypeError, match="sequence of levels"):
        TensorFormat(_HashCountingCandidate())
    # ...and two ``__eq__`` calls on the second, from the negative cache.
    assert _HashCountingMeta.calls == []


def test_real_sequence_success_path_invokes_no_metaclass_hook():
    """A genuine ``Sequence`` subclass is accepted without running its hooks.

    This is the success-path half: the ABC check invoked the caller
    metaclass twice even when the value was perfectly valid, so a raising
    ``__hash__`` could kill an otherwise-legal construction.
    """

    _CountingSequenceMeta.calls = []
    assert str(TensorFormat(_CountingSequence(("d", "s")))) == "d,s"
    assert _CountingSequenceMeta.calls == []


def test_raising_metaclass_cannot_escape_public_format_construction():
    """A caller exception must never leave a public format entry point."""

    with pytest.raises(TensorTypeError, match="sequence of levels"):
        TensorFormat(_RaisingHashCandidate())

    with pytest.raises(TensorFormatError, match="levels"):
        TensorFormat.from_dict({"levels": _RaisingHashCandidate()})

    with pytest.raises(TensorFormatError, match="must contain 'levels'"):
        TensorFormat.from_dict(_RaisingHashCandidate())

    with pytest.raises(TensorFormatError, match="'type'"):
        TensorFormat.from_dict({"levels": [_RaisingHashCandidate()]})


def test_protocol_consumption_failures_are_translated_at_public_boundaries():
    """Recognized containers run their protocol, but never leak its errors."""

    with pytest.raises(TensorFormatError, match="sequence is malformed") as error:
        TensorFormat(_ExplodingSequence())
    assert isinstance(error.value.__cause__, RuntimeError)

    with pytest.raises(TensorFormatError, match="serialized.*malformed") as error:
        TensorFormat.from_dict(_ExplodingDict(levels=[]))
    assert isinstance(error.value.__cause__, RuntimeError)

    with pytest.raises(TensorFormatError, match="serialized.*malformed") as error:
        TensorFormat.from_dict({"levels": [], "fill_value": _ExplodingFill()})
    assert isinstance(error.value.__cause__, RuntimeError)

    with pytest.raises(TensorFormatError, match="serialized.*malformed") as error:
        TensorFormat.from_dict({_ExplodingKey("levels"): []})
    assert isinstance(error.value.__cause__, RuntimeError)


def test_from_dict_still_accepts_every_real_mapping_and_sequence():
    """The MRO recognizer must not narrow the accepted builtin shapes.

    ``collections.abc`` registers the concrete builtins virtually rather
    than by inheritance, so an MRO-identity recognizer has to name them; this
    pins that it does.
    """

    fmt = TensorFormat(["d", "s"])
    payload = fmt.to_dict()

    assert TensorFormat.from_dict(payload) == fmt
    assert TensorFormat.from_dict(MappingProxyType(payload)) == fmt
    assert TensorFormat.from_dict(OrderedDict(payload)) == fmt
    assert TensorFormat.from_dict({"levels": tuple(payload["levels"])}) == fmt
    assert TensorFormat.from_dict({"levels": deque(payload["levels"])}) == fmt
    assert TensorFormat.from_dict(json.loads(fmt.serialize())) == fmt

    class _ListSubclass(list):
        pass

    assert str(TensorFormat(_ListSubclass(["d", "s"]))) == "d,s"
    assert str(TensorFormat(deque(["d", "s"]))) == "d,s"
    assert str(TensorFormat(array("u", "ds"))) == "d,s"
    assert TensorFormat(memoryview(b"")) == TensorFormat()

    with pytest.raises(TensorFormatError, match="levels must be a list"):
        TensorFormat.from_dict({"levels": "ds"})
    with pytest.raises(TensorFormatError, match="levels must be a list"):
        TensorFormat.from_dict({"levels": b"ds"})
