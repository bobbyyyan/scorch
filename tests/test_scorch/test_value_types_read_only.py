"""Scorch's value types cannot change once built.

``LevelFormat``, ``TensorFormat``, ``TensorLayout`` and ``TensorMetadata`` keep their
fields in a base made by ``scorch_ops._sealed_type`` (csrc/sealed_values.cpp), whose
fields are read-only and are set once, by the constructor, after it validates. That is
what lets any number of tensors and caches share one value. These tests hold the
guarantee from every side a caller reaches without Scorch's private names: writing or
deleting a field through ``object.__setattr__``, running ``__init__`` or ``__setstate__``
again on an existing value, assigning ``__class__``, subclassing the base, and building a
value with ``__new__`` alone. They also hold what must not change: equality, hashing,
repr and str (all of which feed cache keys), the dataclass surface, pickling -- including
pickles written before the change -- and copying.
"""

import base64
import copy
import dataclasses
import itertools
import pickle

import pytest
import torch

from scorch.exceptions import TensorFormatError, TensorLayoutError
import scorch.format as format_module
from scorch.format import (
    LevelFormat,
    LevelType,
    TensorFormat,
    _seal,
    owned_format,
    parse_format,
)
from scorch.layout import TensorLayout, TensorMetadata, owned_layout


def _values():
    level = LevelFormat("s", bit_width=32)
    tensor_format = TensorFormat("ds")
    layout = TensorLayout.from_logical_shape((3, 4), "ds", (1, 0), torch.int64)
    metadata = TensorMetadata("A", torch.float64, torch.device("cpu"), layout, True)
    return {
        "level": level,
        "format": tensor_format,
        "layout": layout,
        "metadata": metadata,
    }


KINDS = ("level", "format", "layout", "metadata")

# Every field each class's base holds, the derived ones included.
FIELDS = {
    "level": ("_mode", "_bit_width"),
    "format": ("_level_formats", "_level_types", "_is_dense", "_str", "_canonical"),
    "layout": (
        "logical_shape",
        "physical_shape",
        "format",
        "permutation",
        "index_dtype",
    ),
    "metadata": ("name", "dtype", "device", "layout", "requires_grad"),
}

ERRORS = {
    "level": TensorFormatError,
    "format": TensorFormatError,
    "layout": TensorLayoutError,
    "metadata": TensorLayoutError,
}


def _snapshot(value):
    return repr(value), tuple(getattr(value, name) for name in FIELDS[_kind(value)])


def _kind(value):
    return {
        LevelFormat: "level",
        TensorFormat: "format",
        TensorLayout: "layout",
        TensorMetadata: "metadata",
    }[type(value)]


# What the tree before this change printed (recorded from it), so that a changed repr --
# which would cold-start every cache keyed on one -- cannot pass.
LEGACY_REPR = {
    "level": "s",
    "format": "TensorFormat('d,s')",
    "layout": (
        "TensorLayout(logical_shape=(3, 4), physical_shape=(4, 3), "
        "format=TensorFormat('d,s'), permutation=(1, 0), index_dtype=torch.int64)"
    ),
    "metadata": (
        "TensorMetadata(name='A', dtype=torch.float64, device=device(type='cpu'), "
        "layout=TensorLayout(logical_shape=(3, 4), physical_shape=(4, 3), "
        "format=TensorFormat('d,s'), permutation=(1, 0), index_dtype=torch.int64), "
        "requires_grad=True)"
    ),
}

LEGACY_FIELDS = {
    "level": ["_mode", "_bit_width"],
    "format": ["_level_formats"],
    "layout": [
        "logical_shape",
        "physical_shape",
        "format",
        "permutation",
        "index_dtype",
    ],
    "metadata": ["name", "dtype", "device", "layout", "requires_grad"],
}


@pytest.mark.parametrize("kind", KINDS)
def test_no_field_can_be_written_or_deleted(kind):
    value = _values()[kind]
    before = _snapshot(value)
    for name in FIELDS[kind]:
        with pytest.raises(AttributeError):
            object.__setattr__(value, name, None)
        with pytest.raises(AttributeError):
            object.__delattr__(value, name)
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(value, name, None)
    assert _snapshot(value) == before


@pytest.mark.parametrize("kind", KINDS)
def test_no_other_attribute_can_be_added(kind):
    value = _values()[kind]
    with pytest.raises(AttributeError):
        object.__setattr__(value, "extra", 1)
    with pytest.raises(TypeError):
        vars(value)
    assert not hasattr(value, "__dict__")


def _other_arguments(kind):
    return {
        "level": ("d",),
        "format": ("oo",),
        "layout": ((2,), (2,), "d", (0,)),
        "metadata": (
            "B",
            torch.float32,
            "cpu",
            TensorLayout.from_logical_shape((2,), "d"),
        ),
    }[kind]


@pytest.mark.parametrize("kind", KINDS)
def test_constructing_again_does_not_change_a_value(kind):
    value = _values()[kind]
    before = _snapshot(value)
    with pytest.raises(dataclasses.FrozenInstanceError):
        type(value).__init__(value, *_other_arguments(kind))
    assert _snapshot(value) == before


@pytest.mark.parametrize("kind", KINDS)
def test_setstate_does_not_change_a_value(kind):
    value = _values()[kind]
    other = type(value)(*_other_arguments(kind))
    before = _snapshot(value)
    state = {name: getattr(other, name) for name in LEGACY_FIELDS[kind]}
    with pytest.raises(ERRORS[kind]):
        value.__setstate__(state)
    with pytest.raises(ERRORS[kind]):
        value.__setstate__({"extra": 1})
    assert _snapshot(value) == before


def _invalid_state(kind):
    """Instance state an old pickle could hold that the constructor rejects, in the
    constructor's argument order."""
    layout = TensorLayout.from_logical_shape((3, 4), "ds")
    return {
        "level": {"_mode": "x", "_bit_width": None},
        "format": {"_level_formats": "dx"},
        "layout": {
            "logical_shape": (3, 4),
            "physical_shape": (4, 4),
            "format": TensorFormat("ds"),
            "permutation": (0, 1),
            "index_dtype": torch.int32,
        },
        "metadata": {
            "name": "",
            "dtype": torch.float32,
            "device": torch.device("cpu"),
            "layout": layout,
            "requires_grad": False,
        },
    }[kind]


@pytest.mark.parametrize("kind", KINDS)
def test_setstate_on_an_unbuilt_value_validates_as_the_constructor_does(kind):
    """An object made by ``__new__`` alone (as unpickling makes one) takes old instance
    state only through the constructor, so it refuses what the constructor refuses, with
    the same exception and message, and is left unbuilt."""
    cls = type(_values()[kind])
    state = _invalid_state(kind)
    with pytest.raises(Exception) as from_constructor:
        cls(*state.values())
    bare = cls.__new__(cls)
    with pytest.raises(type(from_constructor.value)) as from_setstate:
        bare.__setstate__(dict(state))
    assert str(from_setstate.value) == str(from_constructor.value)
    with pytest.raises(AttributeError):
        getattr(bare, FIELDS[kind][0])


class _FormatWithState(TensorFormat):
    pass


@pytest.mark.parametrize("kind", KINDS)
def test_class_cannot_be_reassigned(kind):
    values = _values()
    value = values[kind]
    with_state = type("WithState", (type(value),), {})
    other_kind = "format" if kind != "format" else "layout"
    for target in (with_state, type(values[other_kind]), object):
        with pytest.raises(TypeError):
            object.__setattr__(value, "__class__", target)
        with pytest.raises(TypeError):
            object.__dict__["__class__"].__set__(value, target)
    assert (
        type(value)
        is {
            "level": LevelFormat,
            "format": TensorFormat,
            "layout": TensorLayout,
            "metadata": TensorMetadata,
        }[kind]
    )


@pytest.mark.parametrize(
    "cls", [LevelFormat, TensorFormat, TensorLayout, TensorMetadata]
)
def test_a_subclass_adding_no_state_is_refused(cls):
    """Such a subclass would be layout-compatible with the value class, so assigning
    ``__class__`` could turn an existing (possibly shared) value into one."""
    with pytest.raises(TypeError, match="must add state"):
        type("NoState", (cls,), {"__slots__": ()})
    # Adding state is ordinary subclassing.
    type("WithState", (cls,), {})
    type("WithSlot", (cls,), {"__slots__": ("tag",)})


@pytest.mark.parametrize(
    "cls", [LevelFormat, TensorFormat, TensorLayout, TensorMetadata]
)
def test_the_field_base_admits_no_other_subclass_and_cannot_be_changed(cls):
    base = cls.__mro__[1]
    assert base.__module__ == "scorch_ops"
    for namespace in ({"__slots__": ()}, {}):
        with pytest.raises(TypeError, match="cannot be subclassed again"):
            type("Twin", (base,), dict(namespace))
    with pytest.raises(TypeError, match="cannot create"):
        base()
    # An instance of the base made without its __init__ never takes fields.
    bare = base.__new__(base)
    field_count = len(FIELDS[_kind(cls.__new__(cls))])
    with pytest.raises(TypeError, match="not a sealed value"):
        _seal(bare, *((None,) * field_count))
    with pytest.raises(TypeError):
        setattr(base, "extra", 1)
    with pytest.raises(TypeError):
        setattr(base, "__init_subclass__", classmethod(lambda c: None))


def test_a_value_built_by_new_alone_holds_nothing_and_fails_closed():
    # The value classes allocate with object's __new__: a protocol 0 or 1 pickle rebuilds
    # an object through object.__new__(cls), which CPython refuses for any other.
    for cls in (LevelFormat, TensorFormat, TensorLayout, TensorMetadata):
        assert cls.__new__ is object.__new__
    new = object.__new__
    empty_format = new(TensorFormat)
    with pytest.raises(AttributeError):
        empty_format._level_formats
    with pytest.raises(TensorFormatError):
        owned_format(empty_format)
    with pytest.raises(TensorFormatError):
        TensorLayout((3,), (3,), empty_format, (0,))
    empty_layout = new(TensorLayout)
    with pytest.raises(TensorLayoutError):
        owned_layout(empty_layout)
    with pytest.raises(TensorLayoutError):
        TensorMetadata("A", torch.float32, "cpu", empty_layout)


@pytest.mark.parametrize("kind", KINDS)
def test_equality_hash_repr_and_fields_are_unchanged(kind):
    value = _values()[kind]
    again = _values()[kind]
    assert value is not again
    assert value == again
    assert hash(value) == hash(again)
    # The formula a frozen dataclass generates, over the declared fields only.
    assert hash(value) == hash(
        tuple(getattr(value, name) for name in LEGACY_FIELDS[kind])
    )
    assert repr(value) == LEGACY_REPR[kind]
    assert [field.name for field in dataclasses.fields(value)] == LEGACY_FIELDS[kind]
    assert value != _other(kind)


def _other(kind):
    return type(_values()[kind])(*_other_arguments(kind))


def test_derived_format_fields_match_the_levels():
    tensor_format = TensorFormat("ds")
    assert str(tensor_format) == "d,s"
    assert tensor_format.level_types == (LevelType.DENSE, LevelType.COMPRESSED)
    assert tensor_format.get_level_types() == [LevelType.DENSE, LevelType.COMPRESSED]
    assert tensor_format.is_dense() is False
    assert TensorFormat("dd").is_dense() is True
    assert TensorFormat().is_dense() is True
    assert str(TensorFormat()) == ""


def test_replace_builds_a_new_validated_value():
    values = _values()
    layout = dataclasses.replace(values["layout"], index_dtype=torch.int32)
    assert layout == TensorLayout.from_logical_shape((3, 4), "ds", (1, 0), torch.int32)
    metadata = dataclasses.replace(values["metadata"], requires_grad=False)
    assert metadata.requires_grad is False
    assert metadata.layout == values["layout"]
    with pytest.raises(TensorLayoutError):
        dataclasses.replace(values["layout"], physical_shape=(3, 4))


@pytest.mark.parametrize("kind", KINDS)
def test_pickle_and_copy_round_trip(kind):
    value = _values()[kind]
    copies = [
        pickle.loads(pickle.dumps(value, protocol))
        for protocol in range(2, pickle.HIGHEST_PROTOCOL + 1)
    ]
    copies += [copy.copy(value), copy.deepcopy(value)]
    for duplicate in copies:
        assert type(duplicate) is type(value)
        assert duplicate == value
        assert repr(duplicate) == repr(value)


@pytest.mark.parametrize("rank", [0, 1, 2])
def test_a_format_without_bit_widths_is_pickled_as_its_text(rank):
    for levels in itertools.product(list(LevelType), repeat=rank):
        value = TensorFormat(list(levels))
        assert value.__reduce__() == (TensorFormat, (str(value),))
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            loaded = pickle.loads(pickle.dumps(value, protocol))
            assert type(loaded) is TensorFormat
            assert loaded == value
            assert str(loaded) == str(value)


def test_a_format_with_bit_widths_is_pickled_as_its_levels():
    value = TensorFormat([LevelFormat("d"), LevelFormat("s", bit_width=32)])
    assert value.__reduce__() == (TensorFormat, (value.get_level_formats(),))
    for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
        loaded = pickle.loads(pickle.dumps(value, protocol))
        assert loaded == value
        assert loaded.get_level_formats()[1].bit_width == 32


def test_a_layout_loads_as_the_shared_layout():
    layout = TensorLayout.from_logical_shape((3, 4), "ds", (1, 0), torch.int64)
    shared = TensorLayout.from_physical_shape(
        layout.physical_shape, layout.format, layout.permutation, torch.int64
    )
    for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
        assert pickle.loads(pickle.dumps(layout, protocol)) is shared


class _ForgedPickle:
    """Pickles as the given callable and arguments, as a corrupted pickle would hold."""

    def __init__(self, *reduced):
        self.reduced = reduced

    def __reduce__(self):
        return self.reduced


def test_a_malformed_pickle_fails_in_the_constructor():
    layout = _ForgedPickle(
        TensorLayout.from_physical_shape, ((3,), TensorFormat("ds"), (0,), torch.int32)
    )
    with pytest.raises(TensorLayoutError, match="format rank 2"):
        pickle.loads(pickle.dumps(layout))
    with pytest.raises(TensorFormatError):
        pickle.loads(pickle.dumps(_ForgedPickle(TensorFormat, ("dx",))))


def test_a_subclass_with_state_keeps_it_and_is_normalized_where_retained():
    tagged = _FormatWithState("ds")
    tagged.tag = "caller"
    restored = pickle.loads(pickle.dumps(tagged))
    assert type(restored) is _FormatWithState
    assert restored.tag == "caller"
    assert restored == tagged
    owned = owned_format(tagged)
    assert type(owned) is TensorFormat
    assert owned == TensorFormat("ds")
    layout = TensorLayout.from_logical_shape((3, 4), tagged)
    assert type(layout.format) is TensorFormat


def test_an_exact_value_is_retained_as_it_is_and_shared():
    """Sharing is now the intended design: nothing a holder does can change the value."""
    tensor_format = TensorFormat("ds")
    layout = TensorLayout.from_logical_shape((3, 4), tensor_format)
    assert layout.format is tensor_format
    assert owned_format(tensor_format) is tensor_format
    assert owned_layout(layout) is layout
    first = TensorLayout.from_physical_shape((5, 6), "ds")
    second = TensorLayout.from_physical_shape((5, 6), "ds")
    assert first is second
    with pytest.raises(AttributeError):
        object.__setattr__(first.format, "_level_formats", (LevelFormat("o"),) * 2)
    assert str(second.format) == "d,s"


def test_an_int_subclass_bit_width_is_normalized_where_retained():
    class Width(int):
        pass

    caller = TensorFormat([LevelFormat("d", bit_width=Width(16))])
    assert type(caller.get_level_formats()[0].bit_width) is Width
    owned = owned_format(caller)
    assert owned is not caller
    assert type(owned.get_level_formats()[0].bit_width) is int
    assert owned == caller


def test_a_format_string_is_parsed_once_and_shared():
    """parse_format hands every caller naming a string the same format, and formats built
    from one string share its levels; none of them can be changed through any holder."""
    first = parse_format("ds")
    assert parse_format("ds") is first
    assert first == TensorFormat("ds")
    built, again = TensorFormat("ds"), TensorFormat("ds")
    assert built is not again
    assert built.get_level_formats() is again.get_level_formats()
    with pytest.raises(AttributeError):
        object.__setattr__(first, "_level_formats", (LevelFormat("o"),) * 2)
    with pytest.raises(AttributeError):
        object.__setattr__(built.get_level_formats()[0], "_mode", LevelType.COORDINATE)
    assert str(parse_format("ds")) == "d,s"
    assert str(TensorFormat("ds")) == "d,s"


def test_only_an_exact_string_is_looked_up():
    """A str subclass could answer a lookup with its own __hash__ and __eq__; it is parsed
    from its characters every time instead."""

    class Text(str):
        def __hash__(self):
            return hash("ds")

        def __eq__(self, other):
            return True

    parse_format("ds")
    parsed = parse_format(Text("oo"))
    assert str(parsed) == "o,o"
    assert str(TensorFormat(Text("oo"))) == "o,o"
    assert parse_format(Text("oo")) is not parsed


def test_a_string_that_fails_to_parse_is_not_remembered():
    for _ in range(2):
        with pytest.raises(TensorFormatError):
            parse_format("dx")
    assert "dx" not in format_module._TEXT_FORMATS
    assert "dx" not in format_module._TEXT_FIELDS


def test_the_text_caches_are_bounded(monkeypatch):
    text = "dsdsdsdsdsdsdsdsdsds"
    assert text not in format_module._TEXT_FORMATS
    monkeypatch.setattr(format_module, "_TEXT_CACHE_MAX", 0)
    first = parse_format(text)
    second = parse_format(text)
    assert first is not second
    assert first == second == TensorFormat(text)
    assert text not in format_module._TEXT_FORMATS
    assert text not in format_module._TEXT_FIELDS


_LEGACY_PICKLES = {
    "level_p2": (
        "gAJjc2NvcmNoLmZvcm1hdApMZXZlbEZvcm1hdApxACmBcQF9cQIoWAUAAABfbW9kZXEDY3Njb3JjaC5m"
        "b3JtYXQKTGV2ZWxUeXBlCnEEWAEAAABzcQWFcQZScQdYCgAAAF9iaXRfd2lkdGhxCEsgdWIu"
    ),
    "format_p2": (
        "gAJjc2NvcmNoLmZvcm1hdApUZW5zb3JGb3JtYXQKcQApgXEBfXECWA4AAABfbGV2ZWxfZm9ybWF0c3ED"
        "Y3Njb3JjaC5mb3JtYXQKTGV2ZWxGb3JtYXQKcQQpgXEFfXEGKFgFAAAAX21vZGVxB2NzY29yY2guZm9y"
        "bWF0CkxldmVsVHlwZQpxCFgBAAAAZHEJhXEKUnELWAoAAABfYml0X3dpZHRocQxOdWJoBCmBcQ19cQ4o"
        "aAdoCFgBAAAAc3EPhXEQUnERaAxOdWKGcRJzYi4="
    ),
    "layout_p2": (
        "gAJjc2NvcmNoLmxheW91dApUZW5zb3JMYXlvdXQKcQApgXEBfXECKFgNAAAAbG9naWNhbF9zaGFwZXED"
        "SwNLBIZxBFgOAAAAcGh5c2ljYWxfc2hhcGVxBUsESwOGcQZYBgAAAGZvcm1hdHEHY3Njb3JjaC5mb3Jt"
        "YXQKVGVuc29yRm9ybWF0CnEIKYFxCX1xClgOAAAAX2xldmVsX2Zvcm1hdHNxC2NzY29yY2guZm9ybWF0"
        "CkxldmVsRm9ybWF0CnEMKYFxDX1xDihYBQAAAF9tb2RlcQ9jc2NvcmNoLmZvcm1hdApMZXZlbFR5cGUK"
        "cRBYAQAAAGRxEYVxElJxE1gKAAAAX2JpdF93aWR0aHEUTnViaAwpgXEVfXEWKGgPaBBYAQAAAHNxF4Vx"
        "GFJxGWgUTnVihnEac2JYCwAAAHBlcm11dGF0aW9ucRtLAUsAhnEcWAsAAABpbmRleF9kdHlwZXEdY3Rv"
        "cmNoCmludDY0CnEedWIu"
    ),
    "metadata_p2": (
        "gAJjc2NvcmNoLmxheW91dApUZW5zb3JNZXRhZGF0YQpxACmBcQF9cQIoWAQAAABuYW1lcQNYAQAAAEFx"
        "BFgFAAAAZHR5cGVxBWN0b3JjaApmbG9hdDY0CnEGWAYAAABkZXZpY2VxB2N0b3JjaApkZXZpY2UKcQhY"
        "AwAAAGNwdXEJhXEKUnELWAYAAABsYXlvdXRxDGNzY29yY2gubGF5b3V0ClRlbnNvckxheW91dApxDSmB"
        "cQ59cQ8oWA0AAABsb2dpY2FsX3NoYXBlcRBLA0sEhnERWA4AAABwaHlzaWNhbF9zaGFwZXESSwRLA4Zx"
        "E1gGAAAAZm9ybWF0cRRjc2NvcmNoLmZvcm1hdApUZW5zb3JGb3JtYXQKcRUpgXEWfXEXWA4AAABfbGV2"
        "ZWxfZm9ybWF0c3EYY3Njb3JjaC5mb3JtYXQKTGV2ZWxGb3JtYXQKcRkpgXEafXEbKFgFAAAAX21vZGVx"
        "HGNzY29yY2guZm9ybWF0CkxldmVsVHlwZQpxHVgBAAAAZHEehXEfUnEgWAoAAABfYml0X3dpZHRocSFO"
        "dWJoGSmBcSJ9cSMoaBxoHVgBAAAAc3EkhXElUnEmaCFOdWKGcSdzYlgLAAAAcGVybXV0YXRpb25xKEsB"
        "SwCGcSlYCwAAAGluZGV4X2R0eXBlcSpjdG9yY2gKaW50NjQKcSt1YlgNAAAAcmVxdWlyZXNfZ3JhZHEs"
        "iHViLg=="
    ),
    "dense_p2": (
        "gAJjc2NvcmNoLnN0ZW5zb3IKU1RlbnNvcgpxACmBcQF9cQIoWAkAAABfbWV0YWRhdGFxA2NzY29yY2gu"
        "bGF5b3V0ClRlbnNvck1ldGFkYXRhCnEEKYFxBX1xBihYBAAAAG5hbWVxB1gBAAAARHEIWAUAAABkdHlw"
        "ZXEJY3RvcmNoCmZsb2F0MzIKcQpYBgAAAGRldmljZXELY3RvcmNoCmRldmljZQpxDFgDAAAAY3B1cQ2F"
        "cQ5ScQ9YBgAAAGxheW91dHEQY3Njb3JjaC5sYXlvdXQKVGVuc29yTGF5b3V0CnERKYFxEn1xEyhYDQAA"
        "AGxvZ2ljYWxfc2hhcGVxFEsCSwOGcRVYDgAAAHBoeXNpY2FsX3NoYXBlcRZLAksDhnEXWAYAAABmb3Jt"
        "YXRxGGNzY29yY2guZm9ybWF0ClRlbnNvckZvcm1hdApxGSmBcRp9cRtYDgAAAF9sZXZlbF9mb3JtYXRz"
        "cRxjc2NvcmNoLmZvcm1hdApMZXZlbEZvcm1hdApxHSmBcR59cR8oWAUAAABfbW9kZXEgY3Njb3JjaC5m"
        "b3JtYXQKTGV2ZWxUeXBlCnEhWAEAAABkcSKFcSNScSRYCgAAAF9iaXRfd2lkdGhxJU51YmgdKYFxJn1x"
        "JyhoIGgkaCVOdWKGcShzYlgLAAAAcGVybXV0YXRpb25xKUsASwGGcSpYCwAAAGluZGV4X2R0eXBlcStj"
        "dG9yY2gKaW50MzIKcSx1YlgNAAAAcmVxdWlyZXNfZ3JhZHEtiXViWAgAAABfc3RvcmFnZXEuY3Njb3Jj"
        "aC5zdG9yYWdlClNwYXJzZVN0b3JhZ2UKcS8pgXEwfXExKFgHAAAAX2xheW91dHEyaBJYDQAAAF9tb2Rl"
        "X2luZGljZXNxMykphnE0WAYAAABfdmFsdWVxNWN0b3JjaC5fdXRpbHMKX3JlYnVpbGRfdGVuc29yX3Yy"
        "CnE2KGN0b3JjaC5zdG9yYWdlCl9sb2FkX2Zyb21fYnl0ZXMKcTdjX2NvZGVjcwplbmNvZGUKcThYJQEA"
        "AMKAAsKKCmzDvMKcRsO5IGrCqFAZLsKAAk3DqQMuwoACfXEAKFgQAAAAcHJvdG9jb2xfdmVyc2lvbnEB"
        "TcOpA1gNAAAAbGl0dGxlX2VuZGlhbnECwohYCgAAAHR5cGVfc2l6ZXNxA31xBChYBQAAAHNob3J0cQVL"
        "AlgDAAAAaW50cQZLBFgEAAAAbG9uZ3EHSwR1dS7CgAIoWAcAAABzdG9yYWdlcQBjdG9yY2gKRmxvYXRT"
        "dG9yYWdlCnEBWA8AAAAxMDU1NTMxNjQzNTE4NDBxAlgDAAAAY3B1cQNLBk50cQRRLsKAAl1xAFgPAAAA"
        "MTA1NTUzMTY0MzUxODQwcQFhLgYAAAAAAAAAAAAAAAAAwoA/AAAAQAAAQEAAAMKAQAAAwqBAcTlYBgAA"
        "AGxhdGluMXE6hnE7UnE8hXE9UnE+SwBLBoVxP0sBhXFAiWNjb2xsZWN0aW9ucwpPcmRlcmVkRGljdApx"
        "QSlScUJ0cUNScUR1YnViLg=="
    ),
    "csr_p2": (
        "gAJjc2NvcmNoLnN0ZW5zb3IKU1RlbnNvcgpxACmBcQF9cQIoWAkAAABfbWV0YWRhdGFxA2NzY29yY2gu"
        "bGF5b3V0ClRlbnNvck1ldGFkYXRhCnEEKYFxBX1xBihYBAAAAG5hbWVxB1gBAAAAQ3EIWAUAAABkdHlw"
        "ZXEJY3RvcmNoCmZsb2F0MzIKcQpYBgAAAGRldmljZXELY3RvcmNoCmRldmljZQpxDFgDAAAAY3B1cQ2F"
        "cQ5ScQ9YBgAAAGxheW91dHEQY3Njb3JjaC5sYXlvdXQKVGVuc29yTGF5b3V0CnERKYFxEn1xEyhYDQAA"
        "AGxvZ2ljYWxfc2hhcGVxFEsCSwOGcRVYDgAAAHBoeXNpY2FsX3NoYXBlcRZLAksDhnEXWAYAAABmb3Jt"
        "YXRxGGNzY29yY2guZm9ybWF0ClRlbnNvckZvcm1hdApxGSmBcRp9cRtYDgAAAF9sZXZlbF9mb3JtYXRz"
        "cRxjc2NvcmNoLmZvcm1hdApMZXZlbEZvcm1hdApxHSmBcR59cR8oWAUAAABfbW9kZXEgY3Njb3JjaC5m"
        "b3JtYXQKTGV2ZWxUeXBlCnEhWAEAAABkcSKFcSNScSRYCgAAAF9iaXRfd2lkdGhxJU51YmgdKYFxJn1x"
        "JyhoIGghWAEAAABzcSiFcSlScSpoJU51YoZxK3NiWAsAAABwZXJtdXRhdGlvbnEsSwBLAYZxLVgLAAAA"
        "aW5kZXhfZHR5cGVxLmN0b3JjaAppbnQ2NApxL3ViWA0AAAByZXF1aXJlc19ncmFkcTCJdWJYCAAAAF9z"
        "dG9yYWdlcTFjc2NvcmNoLnN0b3JhZ2UKU3BhcnNlU3RvcmFnZQpxMimBcTN9cTQoWAcAAABfbGF5b3V0"
        "cTVoElgNAAAAX21vZGVfaW5kaWNlc3E2KWN0b3JjaC5fdXRpbHMKX3JlYnVpbGRfdGVuc29yX3YyCnE3"
        "KGN0b3JjaC5zdG9yYWdlCl9sb2FkX2Zyb21fYnl0ZXMKcThjX2NvZGVjcwplbmNvZGUKcTlYIQEAAMKA"
        "AsKKCmzDvMKcRsO5IGrCqFAZLsKAAk3DqQMuwoACfXEAKFgQAAAAcHJvdG9jb2xfdmVyc2lvbnEBTcOp"
        "A1gNAAAAbGl0dGxlX2VuZGlhbnECwohYCgAAAHR5cGVfc2l6ZXNxA31xBChYBQAAAHNob3J0cQVLAlgD"
        "AAAAaW50cQZLBFgEAAAAbG9uZ3EHSwR1dS7CgAIoWAcAAABzdG9yYWdlcQBjdG9yY2gKTG9uZ1N0b3Jh"
        "Z2UKcQFYDwAAADEwNTU1MzE2NDM1MjEyOHECWAMAAABjcHVxA0sDTnRxBFEuwoACXXEAWA8AAAAxMDU1"
        "NTMxNjQzNTIxMjhxAWEuAwAAAAAAAAAAAAAAAAAAAAEAAAAAAAAAAwAAAAAAAABxOlgGAAAAbGF0aW4x"
        "cTuGcTxScT2FcT5ScT9LAEsDhXFASwGFcUGJY2NvbGxlY3Rpb25zCk9yZGVyZWREaWN0CnFCKVJxQ3Rx"
        "RFJxRWg3KGg4aDlYOQEAAMKAAsKKCmzDvMKcRsO5IGrCqFAZLsKAAk3DqQMuwoACfXEAKFgQAAAAcHJv"
        "dG9jb2xfdmVyc2lvbnEBTcOpA1gNAAAAbGl0dGxlX2VuZGlhbnECwohYCgAAAHR5cGVfc2l6ZXNxA31x"
        "BChYBQAAAHNob3J0cQVLAlgDAAAAaW50cQZLBFgEAAAAbG9uZ3EHSwR1dS7CgAIoWAcAAABzdG9yYWdl"
        "cQBjdG9yY2gKTG9uZ1N0b3JhZ2UKcQFYDwAAADEwNTU1MzE2NDM1MjIyNHECWAMAAABjcHVxA0sGTnRx"
        "BFEuwoACXXEAWA8AAAAxMDU1NTMxNjQzNTIyMjRxAWEuBgAAAAAAAAAAAAAAAAAAAAEAAAAAAAAAAQAA"
        "AAAAAAABAAAAAAAAAAAAAAAAAAAAAgAAAAAAAABxRmg7hnFHUnFIhXFJUnFKSwNLA4VxS0sBhXFMiWhC"
        "KVJxTXRxTlJxT4ZxUIZxUVgGAAAAX3ZhbHVlcVJoNyhoOGg5WBcBAADCgALCigpsw7zCnEbDuSBqwqhQ"
        "GS7CgAJNw6kDLsKAAn1xAChYEAAAAHByb3RvY29sX3ZlcnNpb25xAU3DqQNYDQAAAGxpdHRsZV9lbmRp"
        "YW5xAsKIWAoAAAB0eXBlX3NpemVzcQN9cQQoWAUAAABzaG9ydHEFSwJYAwAAAGludHEGSwRYBAAAAGxv"
        "bmdxB0sEdXUuwoACKFgHAAAAc3RvcmFnZXEAY3RvcmNoCkZsb2F0U3RvcmFnZQpxAVgPAAAAMTA1NTUz"
        "MTY0MzUyNTEycQJYAwAAAGNwdXEDSwNOdHEEUS7CgAJdcQBYDwAAADEwNTU1MzE2NDM1MjUxMnEBYS4D"
        "AAAAAAAAAAAAAEAAAMKAPwAAQEBxU2g7hnFUUnFVhXFWUnFXSwBLA4VxWEsBhXFZiWhCKVJxWnRxW1Jx"
        "XHVidWIu"
    ),
    "csr_p4": (
        "gASVVAYAAAAAAACMDnNjb3JjaC5zdGVuc29ylIwHU1RlbnNvcpSTlCmBlH2UKIwJX21ldGFkYXRhlIwN"
        "c2NvcmNoLmxheW91dJSMDlRlbnNvck1ldGFkYXRhlJOUKYGUfZQojARuYW1llIwBQ5SMBWR0eXBllIwF"
        "dG9yY2iUjAdmbG9hdDMylJOUjAZkZXZpY2WUjAV0b3JjaJSMBmRldmljZZSTlIwDY3B1lIWUUpSMBmxh"
        "eW91dJRoBowMVGVuc29yTGF5b3V0lJOUKYGUfZQojA1sb2dpY2FsX3NoYXBllEsCSwOGlIwOcGh5c2lj"
        "YWxfc2hhcGWUSwJLA4aUjAZmb3JtYXSUjA1zY29yY2guZm9ybWF0lIwMVGVuc29yRm9ybWF0lJOUKYGU"
        "fZSMDl9sZXZlbF9mb3JtYXRzlGgijAtMZXZlbEZvcm1hdJSTlCmBlH2UKIwFX21vZGWUaCKMCUxldmVs"
        "VHlwZZSTlIwBZJSFlFKUjApfYml0X3dpZHRolE51YmgpKYGUfZQoaCxoLowBc5SFlFKUaDJOdWKGlHNi"
        "jAtwZXJtdXRhdGlvbpRLAEsBhpSMC2luZGV4X2R0eXBllGgOjAVpbnQ2NJSTlHVijA1yZXF1aXJlc19n"
        "cmFklIl1YowIX3N0b3JhZ2WUjA5zY29yY2guc3RvcmFnZZSMDVNwYXJzZVN0b3JhZ2WUk5QpgZR9lCiM"
        "B19sYXlvdXSUaBuMDV9tb2RlX2luZGljZXOUKYwMdG9yY2guX3V0aWxzlIwSX3JlYnVpbGRfdGVuc29y"
        "X3YylJOUKIwNdG9yY2guc3RvcmFnZZSMEF9sb2FkX2Zyb21fYnl0ZXOUk5RCFAEAAIACigps/JxG+SBq"
        "qFAZLoACTekDLoACfXEAKFgQAAAAcHJvdG9jb2xfdmVyc2lvbnEBTekDWA0AAABsaXR0bGVfZW5kaWFu"
        "cQKIWAoAAAB0eXBlX3NpemVzcQN9cQQoWAUAAABzaG9ydHEFSwJYAwAAAGludHEGSwRYBAAAAGxvbmdx"
        "B0sEdXUugAIoWAcAAABzdG9yYWdlcQBjdG9yY2gKTG9uZ1N0b3JhZ2UKcQFYDwAAADEwNTU1MzE2NDM1"
        "MjEyOHECWAMAAABjcHVxA0sDTnRxBFEugAJdcQBYDwAAADEwNTU1MzE2NDM1MjEyOHEBYS4DAAAAAAAA"
        "AAAAAAAAAAAAAQAAAAAAAAADAAAAAAAAAJSFlFKUSwBLA4WUSwGFlImMC2NvbGxlY3Rpb25zlIwLT3Jk"
        "ZXJlZERpY3SUk5QpUpR0lFKUaEkoaExCLAEAAIACigps/JxG+SBqqFAZLoACTekDLoACfXEAKFgQAAAA"
        "cHJvdG9jb2xfdmVyc2lvbnEBTekDWA0AAABsaXR0bGVfZW5kaWFucQKIWAoAAAB0eXBlX3NpemVzcQN9"
        "cQQoWAUAAABzaG9ydHEFSwJYAwAAAGludHEGSwRYBAAAAGxvbmdxB0sEdXUugAIoWAcAAABzdG9yYWdl"
        "cQBjdG9yY2gKTG9uZ1N0b3JhZ2UKcQFYDwAAADEwNTU1MzE2NDM1MjIyNHECWAMAAABjcHVxA0sGTnRx"
        "BFEugAJdcQBYDwAAADEwNTU1MzE2NDM1MjIyNHEBYS4GAAAAAAAAAAAAAAAAAAAAAQAAAAAAAAABAAAA"
        "AAAAAAEAAAAAAAAAAAAAAAAAAAACAAAAAAAAAJSFlFKUSwNLA4WUSwGFlIloVClSlHSUUpSGlIaUjAZf"
        "dmFsdWWUaEkoaExCCQEAAIACigps/JxG+SBqqFAZLoACTekDLoACfXEAKFgQAAAAcHJvdG9jb2xfdmVy"
        "c2lvbnEBTekDWA0AAABsaXR0bGVfZW5kaWFucQKIWAoAAAB0eXBlX3NpemVzcQN9cQQoWAUAAABzaG9y"
        "dHEFSwJYAwAAAGludHEGSwRYBAAAAGxvbmdxB0sEdXUugAIoWAcAAABzdG9yYWdlcQBjdG9yY2gKRmxv"
        "YXRTdG9yYWdlCnEBWA8AAAAxMDU1NTMxNjQzNTI1MTJxAlgDAAAAY3B1cQNLA050cQRRLoACXXEAWA8A"
        "AAAxMDU1NTMxNjQzNTI1MTJxAWEuAwAAAAAAAAAAAABAAACAPwAAQECUhZRSlEsASwOFlEsBhZSJaFQp"
        "UpR0lFKUdWJ1Yi4="
    ),
}


def _legacy(name):
    return pickle.loads(base64.b64decode(_LEGACY_PICKLES[name]))


def test_pickles_written_before_the_change_load():
    """Recorded with protocol 2 (``torch.save``'s) from the tree before this change, where
    the fields were instance state; loading one goes through the constructor."""
    values = _values()
    for kind in KINDS:
        loaded = _legacy(f"{kind}_p2")
        assert type(loaded) is type(values[kind])
        assert loaded == values[kind]
        assert repr(loaded) == LEGACY_REPR[kind]
        with pytest.raises(AttributeError):
            object.__setattr__(loaded, FIELDS[kind][0], None)
    dense = _legacy("dense_p2")
    assert torch.equal(dense.to_torch(), torch.arange(6.0).reshape(2, 3))
    expected = torch.tensor([[0.0, 2.0, 0.0], [1.0, 0.0, 3.0]])
    for name in ("csr_p2", "csr_p4"):
        csr = _legacy(name)
        assert str(csr.format) == "d,s"
        assert torch.equal(csr.to_torch().to_dense(), expected)


_PICKLES_864745D = {
    "format_warm_p0": (
        "Y2NvcHlfcmVnCl9yZWNvbnN0cnVjdG9yCnAwCihjc2NvcmNoLmZvcm1hdApUZW5zb3JGb3JtYXQKcDEK"
        "Y19fYnVpbHRpbl9fCm9iamVjdApwMgpOdHAzClJwNAooZHA1ClZfbGV2ZWxfZm9ybWF0cwpwNgooZzAK"
        "KGNzY29yY2guZm9ybWF0CkxldmVsRm9ybWF0CnA3CmcyCk50cDgKUnA5CihkcDEwClZfbW9kZQpwMTEK"
        "Y3Njb3JjaC5mb3JtYXQKTGV2ZWxUeXBlCnAxMgooVmQKcDEzCnRwMTQKUnAxNQpzVl9iaXRfd2lkdGgK"
        "cDE2Ck5zYmcwCihnNwpnMgpOdHAxNwpScDE4CihkcDE5CmcxMQpnMTIKKFZzCnAyMAp0cDIxClJwMjIK"
        "c2cxNgpOc2J0cDIzCnNWX3N0cgpwMjQKVmQscwpwMjUKc1ZfbGV2ZWxfdHlwZXMKcDI2CihnMTUKZzIy"
        "CnRwMjcKc1ZfaXNfZGVuc2UKcDI4CkkwMApzYi4="
    ),
    "format_warm_p2": (
        "gAJjc2NvcmNoLmZvcm1hdApUZW5zb3JGb3JtYXQKcQApgXEBfXECKFgOAAAAX2xldmVsX2Zvcm1hdHNx"
        "A2NzY29yY2guZm9ybWF0CkxldmVsRm9ybWF0CnEEKYFxBX1xBihYBQAAAF9tb2RlcQdjc2NvcmNoLmZv"
        "cm1hdApMZXZlbFR5cGUKcQhYAQAAAGRxCYVxClJxC1gKAAAAX2JpdF93aWR0aHEMTnViaAQpgXENfXEO"
        "KGgHaAhYAQAAAHNxD4VxEFJxEWgMTnVihnESWAQAAABfc3RycRNYAwAAAGQsc3EUWAwAAABfbGV2ZWxf"
        "dHlwZXNxFWgLaBGGcRZYCQAAAF9pc19kZW5zZXEXiXViLg=="
    ),
    "layout_p1": (
        "Y2NvcHlfcmVnCl9yZWNvbnN0cnVjdG9yCnEAKGNzY29yY2gubGF5b3V0ClRlbnNvckxheW91dApxAWNf"
        "X2J1aWx0aW5fXwpvYmplY3QKcQJOdHEDUnEEfXEFKFgNAAAAbG9naWNhbF9zaGFwZXEGKEsDSwR0cQdY"
        "DgAAAHBoeXNpY2FsX3NoYXBlcQgoSwRLA3RxCVgGAAAAZm9ybWF0cQpoAChjc2NvcmNoLmZvcm1hdApU"
        "ZW5zb3JGb3JtYXQKcQtoAk50cQxScQ19cQ4oWA4AAABfbGV2ZWxfZm9ybWF0c3EPKGgAKGNzY29yY2gu"
        "Zm9ybWF0CkxldmVsRm9ybWF0CnEQaAJOdHERUnESfXETKFgFAAAAX21vZGVxFGNzY29yY2guZm9ybWF0"
        "CkxldmVsVHlwZQpxFShYAQAAAGRxFnRxF1JxGFgKAAAAX2JpdF93aWR0aHEZTnViaAAoaBBoAk50cRpS"
        "cRt9cRwoaBRoFShYAQAAAHNxHXRxHlJxH2gZTnVidHEgWAQAAABfc3RycSFYAwAAAGQsc3EiWAwAAABf"
        "bGV2ZWxfdHlwZXNxIyhoGGgfdHEkWAkAAABfaXNfZGVuc2VxJUkwMAp1YlgLAAAAcGVybXV0YXRpb25x"
        "JihLAUsAdHEnWAsAAABpbmRleF9kdHlwZXEoY3RvcmNoCmludDY0CnEpdWIu"
    ),
    "metadata_p2": (
        "gAJjc2NvcmNoLmxheW91dApUZW5zb3JNZXRhZGF0YQpxACmBcQF9cQIoWAQAAABuYW1lcQNYAQAAAEFx"
        "BFgFAAAAZHR5cGVxBWN0b3JjaApmbG9hdDY0CnEGWAYAAABkZXZpY2VxB2N0b3JjaApkZXZpY2UKcQhY"
        "AwAAAGNwdXEJhXEKUnELWAYAAABsYXlvdXRxDGNzY29yY2gubGF5b3V0ClRlbnNvckxheW91dApxDSmB"
        "cQ59cQ8oWA0AAABsb2dpY2FsX3NoYXBlcRBLA0sEhnERWA4AAABwaHlzaWNhbF9zaGFwZXESSwRLA4Zx"
        "E1gGAAAAZm9ybWF0cRRjc2NvcmNoLmZvcm1hdApUZW5zb3JGb3JtYXQKcRUpgXEWfXEXKFgOAAAAX2xl"
        "dmVsX2Zvcm1hdHNxGGNzY29yY2guZm9ybWF0CkxldmVsRm9ybWF0CnEZKYFxGn1xGyhYBQAAAF9tb2Rl"
        "cRxjc2NvcmNoLmZvcm1hdApMZXZlbFR5cGUKcR1YAQAAAGRxHoVxH1JxIFgKAAAAX2JpdF93aWR0aHEh"
        "TnViaBkpgXEifXEjKGgcaB1YAQAAAHNxJIVxJVJxJmghTnVihnEnWAQAAABfc3RycShYAwAAAGQsc3Ep"
        "WAwAAABfbGV2ZWxfdHlwZXNxKmggaCaGcStYCQAAAF9pc19kZW5zZXEsiXViWAsAAABwZXJtdXRhdGlv"
        "bnEtSwFLAIZxLlgLAAAAaW5kZXhfZHR5cGVxL2N0b3JjaAppbnQ2NApxMHViWA0AAAByZXF1aXJlc19n"
        "cmFkcTGIdWIu"
    ),
    "csr_p2": (
        "gAJjc2NvcmNoLnN0ZW5zb3IKU1RlbnNvcgpxACmBcQF9cQIoWAkAAABfbWV0YWRhdGFxA2NzY29yY2gu"
        "bGF5b3V0ClRlbnNvck1ldGFkYXRhCnEEKYFxBX1xBihYBAAAAG5hbWVxB1gBAAAAQ3EIWAUAAABkdHlw"
        "ZXEJY3RvcmNoCmZsb2F0MzIKcQpYBgAAAGRldmljZXELY3RvcmNoCmRldmljZQpxDFgDAAAAY3B1cQ2F"
        "cQ5ScQ9YBgAAAGxheW91dHEQY3Njb3JjaC5sYXlvdXQKVGVuc29yTGF5b3V0CnERKYFxEn1xEyhYDQAA"
        "AGxvZ2ljYWxfc2hhcGVxFEsCSwOGcRVYDgAAAHBoeXNpY2FsX3NoYXBlcRZLAksDhnEXWAYAAABmb3Jt"
        "YXRxGGNzY29yY2guZm9ybWF0ClRlbnNvckZvcm1hdApxGSmBcRp9cRsoWA4AAABfbGV2ZWxfZm9ybWF0"
        "c3EcY3Njb3JjaC5mb3JtYXQKTGV2ZWxGb3JtYXQKcR0pgXEefXEfKFgFAAAAX21vZGVxIGNzY29yY2gu"
        "Zm9ybWF0CkxldmVsVHlwZQpxIVgBAAAAZHEihXEjUnEkWAoAAABfYml0X3dpZHRocSVOdWJoHSmBcSZ9"
        "cScoaCBoIVgBAAAAc3EohXEpUnEqaCVOdWKGcStYBAAAAF9zdHJxLFgDAAAAZCxzcS1YDAAAAF9sZXZl"
        "bF90eXBlc3EuaCRoKoZxL1gJAAAAX2lzX2RlbnNlcTCJdWJYCwAAAHBlcm11dGF0aW9ucTFLAEsBhnEy"
        "WAsAAABpbmRleF9kdHlwZXEzY3RvcmNoCmludDY0CnE0dWJYDQAAAHJlcXVpcmVzX2dyYWRxNYl1YlgI"
        "AAAAX3N0b3JhZ2VxNmNzY29yY2guc3RvcmFnZQpTcGFyc2VTdG9yYWdlCnE3KYFxOH1xOShYBwAAAF9s"
        "YXlvdXRxOmgSWA0AAABfbW9kZV9pbmRpY2VzcTspY3RvcmNoLl91dGlscwpfcmVidWlsZF90ZW5zb3Jf"
        "djIKcTwoY3RvcmNoLnN0b3JhZ2UKX2xvYWRfZnJvbV9ieXRlcwpxPWNfY29kZWNzCmVuY29kZQpxPlgh"
        "AQAAwoACwooKbMO8wpxGw7kgasKoUBkuwoACTcOpAy7CgAJ9cQAoWBAAAABwcm90b2NvbF92ZXJzaW9u"
        "cQFNw6kDWA0AAABsaXR0bGVfZW5kaWFucQLCiFgKAAAAdHlwZV9zaXplc3EDfXEEKFgFAAAAc2hvcnRx"
        "BUsCWAMAAABpbnRxBksEWAQAAABsb25ncQdLBHV1LsKAAihYBwAAAHN0b3JhZ2VxAGN0b3JjaApMb25n"
        "U3RvcmFnZQpxAVgPAAAAMTA1NTUzMjQyNjU4NzIwcQJYAwAAAGNwdXEDSwNOdHEEUS7CgAJdcQBYDwAA"
        "ADEwNTU1MzI0MjY1ODcyMHEBYS4DAAAAAAAAAAAAAAAAAAAAAQAAAAAAAAADAAAAAAAAAHE/WAYAAABs"
        "YXRpbjFxQIZxQVJxQoVxQ1JxREsASwOFcUVLAYVxRoljY29sbGVjdGlvbnMKT3JkZXJlZERpY3QKcUcp"
        "UnFIdHFJUnFKaDwoaD1oPlg5AQAAwoACwooKbMO8wpxGw7kgasKoUBkuwoACTcOpAy7CgAJ9cQAoWBAA"
        "AABwcm90b2NvbF92ZXJzaW9ucQFNw6kDWA0AAABsaXR0bGVfZW5kaWFucQLCiFgKAAAAdHlwZV9zaXpl"
        "c3EDfXEEKFgFAAAAc2hvcnRxBUsCWAMAAABpbnRxBksEWAQAAABsb25ncQdLBHV1LsKAAihYBwAAAHN0"
        "b3JhZ2VxAGN0b3JjaApMb25nU3RvcmFnZQpxAVgPAAAAMTA1NTUzMjQyNjU4ODE2cQJYAwAAAGNwdXED"
        "SwZOdHEEUS7CgAJdcQBYDwAAADEwNTU1MzI0MjY1ODgxNnEBYS4GAAAAAAAAAAAAAAAAAAAAAQAAAAAA"
        "AAABAAAAAAAAAAEAAAAAAAAAAAAAAAAAAAACAAAAAAAAAHFLaECGcUxScU2FcU5ScU9LA0sDhXFQSwGF"
        "cVGJaEcpUnFSdHFTUnFUhnFVhnFWWAYAAABfdmFsdWVxV2g8KGg9aD5YFwEAAMKAAsKKCmzDvMKcRsO5"
        "IGrCqFAZLsKAAk3DqQMuwoACfXEAKFgQAAAAcHJvdG9jb2xfdmVyc2lvbnEBTcOpA1gNAAAAbGl0dGxl"
        "X2VuZGlhbnECwohYCgAAAHR5cGVfc2l6ZXNxA31xBChYBQAAAHNob3J0cQVLAlgDAAAAaW50cQZLBFgE"
        "AAAAbG9uZ3EHSwR1dS7CgAIoWAcAAABzdG9yYWdlcQBjdG9yY2gKRmxvYXRTdG9yYWdlCnEBWA8AAAAx"
        "MDU1NTMyNDI2NTkxMDRxAlgDAAAAY3B1cQNLA050cQRRLsKAAl1xAFgPAAAAMTA1NTUzMjQyNjU5MTA0"
        "cQFhLgMAAAAAAAAAAAAAQAAAwoA/AABAQHFYaECGcVlScVqFcVtScVxLAEsDhXFdSwGFcV6JaEcpUnFf"
        "dHFgUnFhdWJ1Yi4="
    ),
    "format_stale_p2": (
        "gAJjc2NvcmNoLmZvcm1hdApUZW5zb3JGb3JtYXQKcQApgXEBfXECKFgOAAAAX2xldmVsX2Zvcm1hdHNx"
        "A2NzY29yY2guZm9ybWF0CkxldmVsRm9ybWF0CnEEKYFxBX1xBihYBQAAAF9tb2RlcQdjc2NvcmNoLmZv"
        "cm1hdApMZXZlbFR5cGUKcQhYAQAAAHNxCYVxClJxC1gKAAAAX2JpdF93aWR0aHEMTnViaAQpgXENfXEO"
        "KGgHaAtoDE51YoZxD1gEAAAAX3N0cnEQWAMAAABkLHNxEVgMAAAAX2xldmVsX3R5cGVzcRJoCFgBAAAA"
        "ZHEThXEUUnEVaAuGcRZYCQAAAF9pc19kZW5zZXEXiXViLg=="
    ),
}


def _legacy_864745d(name):
    return pickle.loads(base64.b64decode(_PICKLES_864745D[name]))


def test_pickles_with_format_memos_load():
    """Recorded on 864745d, whose formats memoized their level types, density and string
    form in instance state, so its pickles carry them; protocols 0 and 1 rebuild an object
    through ``object.__new__``. Each memo equals what the constructor derives. (154320f
    loads the CSR tensor but fails on first use: its audit rejects the extra state.)"""
    expected_format = TensorFormat("ds")
    for name in ("format_warm_p0", "format_warm_p2"):
        loaded = _legacy_864745d(name)
        assert type(loaded) is TensorFormat
        assert loaded == expected_format
        assert str(loaded) == "d,s"
        assert loaded.level_types == expected_format.level_types
        assert not loaded.is_dense()
        with pytest.raises(AttributeError):
            object.__setattr__(loaded, "_level_formats", (LevelFormat("s"),) * 2)
    layout = _legacy_864745d("layout_p1")
    assert layout == TensorLayout.from_logical_shape((3, 4), "ds", (1, 0), torch.int64)
    metadata = _legacy_864745d("metadata_p2")
    assert metadata == TensorMetadata(
        "A", torch.float64, torch.device("cpu"), layout, True
    )
    csr = _legacy_864745d("csr_p2")
    assert str(csr.format) == "d,s"
    assert torch.equal(
        csr.to_torch().to_dense(), torch.tensor([[0.0, 2.0, 0.0], [1.0, 0.0, 3.0]])
    )


def test_a_pickled_format_with_a_stale_memo_fails_closed():
    """On 864745d a format's levels could be rewritten after its memos were taken
    (``object.__setattr__`` wrote the frozen field), and its pickle then carries a string
    naming other levels. Loading it raises instead of choosing between the two."""
    with pytest.raises(TensorFormatError, match="disagrees with its other fields"):
        _legacy_864745d("format_stale_p2")
