"""A construction inside Scorch audits its index arrays once.

``STensor.from_csr`` and the other factories start from arrays, not from an index a
caller kept. They used to make a ``TensorIndex`` only to hand it to the constructor, and
a storage handed an index audits it again: the stored state of the object, then every
array, and it copies a second time the arrays of an index that did not adopt them. On an
index that never left the constructing expression that second audit found nothing, at 75
of the 249 Python calls in one ``from_csr``.

``STensor._from_parts`` is the replacement, and this file pins it: one audit and at most
one copy, with the same tensor, the same sharing and the same errors, in the same order,
as the public constructor gives. The public constructor keeps its full audit of an index
object a caller brings, which the last tests here check has not moved.
"""

import ast
import inspect
from types import SimpleNamespace

import pytest
import torch

import scorch.stensor as stensor_module
import scorch.storage as storage_module
from scorch import STensor
from scorch.exceptions import TensorIndexError, TensorStorageError, TensorTypeError
from scorch.layout import TensorLayout
from scorch.stensor import _wrap_generated_result
from scorch.storage import SparseStorage, TensorIndex, _index_components
from scorch.utils import parse_format


def ints(values, dtype=torch.int64):
    return torch.tensor(values, dtype=dtype)


def floats(count, dtype=torch.float32):
    return torch.arange(1, count + 1, dtype=dtype)


def csr_parts(dtype=torch.int64):
    """A 3 x 4 CSR matrix with an empty middle row."""
    return dict(
        name="A",
        shape=(3, 4),
        fmt="ds",
        mode_indices=[[], [ints([0, 2, 2, 3], dtype), ints([0, 3, 1], dtype)]],
        value=floats(3),
    )


# Valid components, one per storage shape the factories build: each level kind, rank 3,
# a permuted mode order, both index dtypes, and two CSR inputs that the storage has to
# put in canonical form (a row out of order, a column stored twice).
VALID = {
    "csr int64": lambda: csr_parts(),
    "csr int32": lambda: csr_parts(torch.int32),
    "csr permuted": lambda: dict(csr_parts(), mode_order=[1, 0]),
    "csr float64 values": lambda: dict(csr_parts(), value=floats(3, torch.float64)),
    "csr row out of order": lambda: dict(
        csr_parts(), mode_indices=[[], [ints([0, 2, 2, 3]), ints([3, 0, 1])]]
    ),
    "csr column stored twice": lambda: dict(
        csr_parts(), mode_indices=[[], [ints([0, 2, 2, 3]), ints([1, 1, 1])]]
    ),
    "coo": lambda: dict(
        name="A",
        shape=(3, 4),
        fmt="oo",
        mode_indices=[[ints([0, 0, 2])], [ints([0, 3, 1])]],
        value=floats(3),
    ),
    "dcsr": lambda: dict(
        name="A",
        shape=(3, 4),
        fmt="ss",
        mode_indices=[
            [ints([0, 2]), ints([0, 2])],
            [ints([0, 2, 3]), ints([0, 3, 1])],
        ],
        value=floats(3),
    ),
    "rank 3": lambda: dict(
        name="A",
        shape=(2, 3, 4),
        fmt="dss",
        mode_indices=[
            [],
            [ints([0, 1, 2]), ints([0, 2])],
            [ints([0, 2, 3]), ints([1, 3, 0])],
        ],
        value=floats(3),
    ),
    "rank 3 permuted": lambda: dict(
        name="A",
        shape=(2, 3, 4),
        fmt="dss",
        mode_indices=[
            [],
            [ints([0, 1, 2]), ints([0, 2])],
            [ints([0, 2, 3]), ints([1, 3, 0])],
        ],
        value=floats(3),
        mode_order=[2, 0, 1],
    ),
    "dense": lambda: dict(
        name=None, shape=(2, 3), fmt="dd", mode_indices=[[], []], value=floats(6)
    ),
    "declared index dtype": lambda: dict(csr_parts(), index_dtype=torch.int64),
}


def by_index(name, shape, fmt, mode_indices, value, **options):
    """The public constructor, through an index object made for it."""
    return STensor(
        name=name,
        shape=shape,
        index=TensorIndex(
            fmt,
            mode_indices,
            options.get("mode_order"),
            options.get("index_dtype"),
            _adopt=options.get("adopt", False),
        ),
        value=value,
        requires_grad=options.get("requires_grad", False),
    )


def by_parts(name, shape, fmt, mode_indices, value, **options):
    return STensor._from_parts(name, shape, fmt, mode_indices, value, **options)


def observable(tensor):
    """Everything a holder of this STensor can see about it."""
    storage = tensor._storage
    return dict(
        shape=tuple(tensor.shape),
        fmt=str(tensor.format),
        name=tensor.name,
        dtype=storage._value.dtype,
        mode_order=tuple(storage.layout.permutation),
        index_dtype=storage.layout.index_dtype,
        logical=tuple(storage.layout.logical_shape),
        physical=tuple(storage.layout.physical_shape),
        mode_indices=[
            [(array.dtype, array.tolist()) for array in level]
            for level in storage._mode_indices
        ],
        values=storage._value.tolist(),
        requires_grad=tensor.requires_grad,
    )


def shared_with(tensor, mode_indices):
    """Per index array: whether the tensor holds the very buffer it was given."""
    return [
        [held.data_ptr() == given.data_ptr() for held, given in zip(level, arrays)]
        for level, arrays in zip(tensor._storage._mode_indices, mode_indices)
    ]


@pytest.fixture
def audits(monkeypatch):
    """Count the two audits: of index arrays (with whether each copied) and of the
    stored state of an index object."""
    seen = {"arrays": [], "objects": 0}
    audit_arrays = storage_module._normalize_mode_indices
    audit_object = storage_module._owned_tensor_index_components

    def arrays(mode_indices, tensor_format, copy=True):
        seen["arrays"].append(copy)
        return audit_arrays(mode_indices, tensor_format, copy)

    def stored_state(index, *, own_arrays=True):
        seen["objects"] += 1
        return audit_object(index, own_arrays=own_arrays)

    monkeypatch.setattr(storage_module, "_normalize_mode_indices", arrays)
    monkeypatch.setattr(storage_module, "_owned_tensor_index_components", stored_state)
    monkeypatch.setattr(stensor_module, "_owned_tensor_index_components", stored_state)
    return seen


# --------------------------------------------------------------------------- #
# The same tensor as the public constructor builds
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("adopt", [False, True])
@pytest.mark.parametrize("case", sorted(VALID))
def test_parts_build_what_the_public_constructor_builds(case, adopt):
    reference = by_index(**VALID[case](), adopt=adopt)
    built = by_parts(**VALID[case](), adopt=adopt)
    assert observable(built) == observable(reference)
    assert type(built) is STensor
    built.validate()


@pytest.mark.parametrize("adopt", [False, True])
@pytest.mark.parametrize("case", sorted(VALID))
def test_parts_share_exactly_the_arrays_the_public_constructor_shares(case, adopt):
    """Adopted arrays are the caller's buffers and any others are copies, except where
    the storage had to build a canonical private copy, which shares nothing either way.
    """
    reference_parts, parts = VALID[case](), VALID[case]()
    reference = by_index(**reference_parts, adopt=adopt)
    built = by_parts(**parts, adopt=adopt)
    sharing = shared_with(built, parts["mode_indices"])
    assert sharing == shared_with(reference, reference_parts["mode_indices"])
    if "out of order" not in case and "stored twice" not in case:
        assert all(shared is adopt for level in sharing for shared in level)


def test_the_storage_keeps_its_own_tensor_objects():
    """Sharing a buffer is not keeping the caller's object: what is retained is a
    detached tensor over the same memory, with the same version counter."""
    parts = csr_parts()
    positions, coordinates = parts["mode_indices"][1]
    built = by_parts(**parts, adopt=True)
    held = built._storage._mode_indices[1]
    assert held[0] is not positions and held[1] is not coordinates
    assert type(held[0]) is torch.Tensor and type(held[1]) is torch.Tensor
    assert built._storage._value is not parts["value"]
    before = held[1]._version
    coordinates[0] = 1
    assert held[1]._version == before + 1
    assert held[1].tolist() == [1, 3, 1]


def test_copied_arrays_do_not_follow_a_later_write():
    parts = csr_parts()
    built = by_parts(**parts)
    parts["mode_indices"][1][1][0] = 2
    assert built._storage._mode_indices[1][1].tolist() == [0, 3, 1]


def test_from_components_copies_its_index_arrays_once(audits):
    parts = csr_parts()
    STensor.from_components(
        parts["shape"], parts["fmt"], parts["mode_indices"], parts["value"]
    )
    assert audits == {"arrays": [True], "objects": 0}


# --------------------------------------------------------------------------- #
# One audit per construction
# --------------------------------------------------------------------------- #


def torch_csr():
    parts = csr_parts()
    positions, coordinates = parts["mode_indices"][1]
    return torch.sparse_csr_tensor(positions, coordinates, parts["value"], (3, 4))


def torch_coo():
    return torch.sparse_coo_tensor(
        torch.tensor([[0, 0, 2], [0, 3, 1]]), floats(3), (3, 4)
    ).coalesce()


FACTORIES = {
    "from_csr": lambda: STensor.from_csr(torch_csr(), "A"),
    "from_coo": lambda: STensor.from_coo(torch_coo(), name="A"),
    "from_torch of a CSR tensor": lambda: STensor.from_torch(torch_csr(), "A"),
    "from_torch of a COO tensor": lambda: STensor.from_torch(torch_coo(), "A"),
}


@pytest.mark.parametrize("factory", sorted(FACTORIES))
def test_a_factory_audits_its_index_arrays_once_and_shares_them(audits, factory):
    built = FACTORIES[factory]()
    assert audits == {"arrays": [False], "objects": 0}
    built.validate()


def test_a_dense_operand_audits_at_most_once(audits, monkeypatch):
    monkeypatch.setattr(stensor_module, "_DENSE_PARTS_CACHE", {})
    dense = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    first = STensor.from_torch(dense, "D")
    assert audits == {"arrays": [True], "objects": 0}
    second = STensor.from_torch(dense, "D")
    assert audits == {"arrays": [True], "objects": 0}
    assert observable(first) == observable(second)


def kernel_result(mode_indices, value):
    """The shape of a compiled module's return value, holding arrays we choose."""
    return SimpleNamespace(
        storage=SimpleNamespace(
            index=SimpleNamespace(mode_indices=mode_indices), value=value
        )
    )


@pytest.mark.parametrize("adopt", [False, True])
def test_a_sparse_kernel_result_is_audited_once(audits, adopt):
    parts = csr_parts(torch.int32)
    wrapped = _wrap_generated_result(
        parts["shape"],
        parts["fmt"],
        kernel_result(parts["mode_indices"], parts["value"]),
        adopt=adopt,
    )
    assert audits == {"arrays": [not adopt], "objects": 0}
    sharing = shared_with(wrapped, parts["mode_indices"])
    assert all(shared is adopt for level in sharing for shared in level)
    reference = by_index(**dict(parts, name=None))
    assert observable(wrapped) == observable(reference)


def test_a_dense_kernel_result_is_audited_once_per_key(audits, monkeypatch):
    monkeypatch.setattr(stensor_module, "_RESULT_PARTS_CACHE", {})
    results = [
        _wrap_generated_result(
            (2, 3), "dd", kernel_result([[], []], floats(6) * scale), name="C"
        )
        for scale in (1.0, 2.0, 3.0)
    ]
    assert audits == {"arrays": [not stensor_module._ADOPT_CELL[0]], "objects": 0}
    assert [result._storage._value.tolist()[0] for result in results] == [1.0, 2.0, 3.0]
    assert len({id(result._storage._mode_indices) for result in results}) == 1
    reference = by_index("C", (2, 3), "dd", [[], []], floats(6))
    assert observable(results[0]) == observable(reference)


def index_object_calls(source):
    """Line numbers of the calls that build a ``TensorIndex``, however it is spelled:
    ``TensorIndex(...)`` or ``storage.TensorIndex(...)``. A call of one of its methods,
    such as ``TensorIndex._from_layout(...)``, is not one."""
    names = ("id", "attr")
    return [
        node.lineno
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and any(getattr(node.func, name, "") == "TensorIndex" for name in names)
    ]


def test_no_construction_in_stensor_goes_through_an_index_object():
    """Every construction in stensor.py starts from arrays. One that built a
    ``TensorIndex`` for the constructor would be audited twice again, and nothing but a
    profile would show it; build with ``STensor._from_parts`` instead."""
    spelled = "x = TensorIndex(a)\ny = storage.TensorIndex(a)"
    assert index_object_calls(spelled) == [1, 2]
    assert index_object_calls("x = TensorIndex._from_layout(a, b)") == []
    assert index_object_calls(inspect.getsource(stensor_module)) == []


# --------------------------------------------------------------------------- #
# The same errors, in the same order
# --------------------------------------------------------------------------- #


def outcome(build, parts):
    try:
        build(**parts)
    except Exception as error:  # the comparison is of whatever is raised
        return type(error), str(error)
    return None


def strided(values):
    spaced = torch.tensor([entry for value in values for entry in (value, 99)])
    return spaced[::2]


MALFORMED = {
    "no format": lambda: dict(csr_parts(), fmt=None),
    "unknown format letter": lambda: dict(csr_parts(), fmt="dq"),
    "index arrays not a sequence": lambda: dict(csr_parts(), mode_indices=7),
    "one level too few": lambda: dict(csr_parts(), mode_indices=[[]]),
    "one array too few": lambda: dict(
        csr_parts(), mode_indices=[[], [ints([0, 2, 2, 3])]]
    ),
    "an array that is a list": lambda: dict(
        csr_parts(), mode_indices=[[], [ints([0, 2, 2, 3]), [0, 3, 1]]]
    ),
    "floating-point coordinates": lambda: dict(
        csr_parts(),
        mode_indices=[[], [ints([0, 2, 2, 3]), torch.tensor([0.0, 3.0, 1.0])]],
    ),
    "two-dimensional coordinates": lambda: dict(
        csr_parts(), mode_indices=[[], [ints([0, 2, 2, 3]), ints([[0, 3, 1]])]]
    ),
    "non-contiguous coordinates": lambda: dict(
        csr_parts(), mode_indices=[[], [ints([0, 2, 2, 3]), strided([0, 3, 1])]]
    ),
    "two index dtypes": lambda: dict(
        csr_parts(),
        mode_indices=[[], [ints([0, 2, 2, 3], torch.int32), ints([0, 3, 1])]],
    ),
    "index dtype that is not an integer type": lambda: dict(
        csr_parts(), index_dtype=torch.float32
    ),
    "index dtype other than the arrays'": lambda: dict(
        csr_parts(), index_dtype=torch.int32
    ),
    "mode order that is a string": lambda: dict(csr_parts(), mode_order="10"),
    "mode order that repeats": lambda: dict(csr_parts(), mode_order=[0, 0]),
    "mode order of booleans": lambda: dict(csr_parts(), mode_order=[True, False]),
    "shape of the wrong rank": lambda: dict(csr_parts(), shape=(3,)),
    "negative extent": lambda: dict(csr_parts(), shape=(3, -4)),
    "no values": lambda: dict(csr_parts(), value=None),
    "values that are a list": lambda: dict(csr_parts(), value=[1.0, 2.0, 3.0]),
    "two-dimensional values": lambda: dict(csr_parts(), value=floats(3).reshape(3, 1)),
    "non-contiguous values": lambda: dict(csr_parts(), value=strided([1.0, 2.0, 3.0])),
    "one value too many": lambda: dict(csr_parts(), value=floats(4)),
    "positions that decrease": lambda: dict(
        csr_parts(), mode_indices=[[], [ints([0, 2, 1, 3]), ints([0, 3, 1])]]
    ),
    "positions that do not start at zero": lambda: dict(
        csr_parts(), mode_indices=[[], [ints([1, 2, 2, 3]), ints([0, 3, 1])]]
    ),
    "coordinate outside the extent": lambda: dict(
        csr_parts(), mode_indices=[[], [ints([0, 2, 2, 3]), ints([0, 4, 1])]]
    ),
    "requires_grad that is not a bool": lambda: dict(csr_parts(), requires_grad="yes"),
    "values on another device": lambda: dict(
        csr_parts(), value=torch.empty(3, device="meta")
    ),
    "coordinates on another device": lambda: dict(
        csr_parts(),
        mode_indices=[
            [],
            [ints([0, 2, 2, 3]), torch.empty(3, dtype=torch.int64, device="meta")],
        ],
    ),
    "lazily negated values": lambda: dict(
        csr_parts(), value=torch._neg_view(floats(3))
    ),
    "coordinates in a sparse tensor": lambda: dict(
        csr_parts(),
        mode_indices=[
            [],
            [
                ints([0, 2, 2, 3]),
                torch.sparse_coo_tensor(ints([[0, 1, 2]]), ints([0, 3, 1]), (3,)),
            ],
        ],
    ),
    "value type the runtime does not store": lambda: dict(
        csr_parts(), value=floats(3, torch.float16)
    ),
    # Two defects at once: the one reported shows the order the checks run in.
    "bad index and bad requires_grad": lambda: dict(
        csr_parts(), mode_indices=[[]], requires_grad="yes"
    ),
    "bad index and no values": lambda: dict(csr_parts(), mode_indices=[[]], value=None),
    "bad requires_grad and no values": lambda: dict(
        csr_parts(), requires_grad="yes", value=None
    ),
    "values that are a list and a bad shape": lambda: dict(
        csr_parts(), value=[1.0], shape=(3,)
    ),
    "bad shape and two-dimensional values": lambda: dict(
        csr_parts(), shape=(3,), value=floats(3).reshape(3, 1)
    ),
    "two-dimensional values and bad positions": lambda: dict(
        csr_parts(),
        value=floats(3).reshape(3, 1),
        mode_indices=[[], [ints([0, 2, 1, 3]), ints([0, 3, 1])]],
    ),
}


@pytest.mark.parametrize("adopt", [False, True])
@pytest.mark.parametrize("case", sorted(MALFORMED))
def test_parts_raise_what_the_public_constructor_raises(case, adopt):
    expected = outcome(by_index, dict(MALFORMED[case](), adopt=adopt))
    assert expected is not None, "this input was meant to be refused"
    assert outcome(by_parts, dict(MALFORMED[case](), adopt=adopt)) == expected


# --------------------------------------------------------------------------- #
# An exact torch.Tensor is audited without a base-class view of it
# --------------------------------------------------------------------------- #


def test_an_attribute_set_on_an_index_array_does_not_answer_for_it():
    """The audit's method reads go through the type. `dim` and `is_contiguous` are
    ordinary methods, so an attribute of that name on the instance shadows them for a
    plain attribute read -- the two asserts on the premise show that it does."""
    positions = ints([0, 2, 2, 3])
    flat = parse_format("ds")

    two_dimensional = ints([[0, 3, 1]])
    two_dimensional.dim = lambda: 1
    assert two_dimensional.dim() == 1
    with pytest.raises(TensorIndexError, match="one-dimensional"):
        storage_module._normalize_mode_indices([[], [positions, two_dimensional]], flat)

    spaced = strided([0, 3, 1])
    spaced.is_contiguous = lambda: True
    assert spaced.is_contiguous() is True
    with pytest.raises(TensorIndexError, match="contiguous"):
        storage_module._normalize_mode_indices([[], [positions, spaced]], flat)


def test_an_attribute_set_on_the_values_does_not_answer_for_them():
    layout = TensorLayout.from_physical_shape((3, 4), "ds", None, torch.int64)
    arrays = csr_parts()["mode_indices"]

    two_dimensional = floats(3).reshape(3, 1)
    two_dimensional.dim = lambda: 1
    assert two_dimensional.dim() == 1
    with pytest.raises(TensorStorageError, match="flat one-dimensional"):
        SparseStorage(layout, two_dimensional, mode_indices=arrays)

    spaced = strided([1.0, 2.0, 3.0])
    spaced.is_contiguous = lambda: True
    assert spaced.is_contiguous() is True
    with pytest.raises(TensorStorageError, match="contiguous"):
        SparseStorage(layout, spaced, mode_indices=arrays)

    negated = torch._neg_view(floats(3))
    negated.is_neg = lambda: False
    assert negated.is_neg() is False
    with pytest.raises(TensorStorageError, match="lazy negative"):
        SparseStorage(layout, negated, mode_indices=arrays)


@pytest.mark.parametrize("adopt", [False, True])
@pytest.mark.parametrize("method", ["detach", "clone"])
def test_a_method_set_on_an_index_array_does_not_decide_what_is_kept(method, adopt):
    """The arrays a storage keeps are made by ``torch.Tensor.detach`` and
    ``torch.Tensor.clone``, read through the type. An attribute of either name on the
    instance shadows the method for a plain attribute read -- the assert on the premise
    shows that it does -- and a storage that asked the instance would keep whatever
    that attribute returned, here a coordinate outside the matrix."""
    parts = csr_parts()
    coordinates = parts["mode_indices"][1][1]
    setattr(coordinates, method, lambda: ints([0, 99, 1]))
    assert getattr(coordinates, method)().tolist() == [0, 99, 1]
    built = by_parts(**parts, adopt=adopt)
    held = built._storage._mode_indices[1][1]
    assert type(held) is torch.Tensor
    assert held.tolist() == [0, 3, 1]
    assert (held.data_ptr() == coordinates.data_ptr()) is adopt
    built.validate()


@pytest.mark.parametrize("build", ["storage", "parts"])
def test_a_method_set_on_the_values_does_not_decide_what_is_kept(build):
    parts = csr_parts()
    value = parts["value"]
    value.detach = lambda: floats(3) * 100
    assert value.detach().tolist() == [100.0, 200.0, 300.0]
    if build == "storage":
        layout = TensorLayout.from_physical_shape((3, 4), "ds", None, torch.int64)
        storage = SparseStorage(layout, value, mode_indices=parts["mode_indices"])
    else:
        storage = by_parts(**parts)._storage
    assert type(storage._value) is torch.Tensor
    assert storage._value.tolist() == [1.0, 2.0, 3.0]
    assert storage._value.data_ptr() == value.data_ptr()


@pytest.mark.parametrize("name", ["layout", "device", "dtype"])
def test_the_properties_the_audit_reads_cannot_be_shadowed(name):
    """They are data descriptors of the tensor type, which win over the instance's own
    attributes; that is why the audit may read them as plain attributes."""
    array = ints([0, 1, 2])
    expected = getattr(array, name)
    array.__dict__[name] = "shadow"
    assert getattr(array, name) is expected or getattr(array, name) == expected
    with pytest.raises(AttributeError):
        setattr(array, name, "shadow")


class OneDimensional(torch.Tensor):
    """A subclass that answers for its own rank."""

    def dim(self):
        return 1


def test_a_subclass_is_still_read_through_its_base_class():
    positions = ints([0, 2, 2, 3])
    claimed = ints([[0, 3, 1]]).as_subclass(OneDimensional)
    assert claimed.dim() == 1
    with pytest.raises(TensorIndexError, match="one-dimensional"):
        storage_module._normalize_mode_indices(
            [[], [positions, claimed]], parse_format("ds")
        )
    honest = ints([0, 3, 1]).as_subclass(OneDimensional)
    held = storage_module._normalize_mode_indices(
        [[], [positions, honest]], parse_format("ds"), False
    )
    assert type(held[1][1]) is torch.Tensor
    assert held[1][1].data_ptr() == honest.data_ptr()


class Refuses(torch.Tensor):
    """Answers a request for a base-class view of itself with an error."""

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if getattr(func, "__name__", "") == "as_subclass":
            raise RuntimeError("no view")
        return super().__torch_function__(func, types, args, kwargs or {})


class AnswersSeven(torch.Tensor):
    """Answers the same request with something that is not a tensor."""

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if getattr(func, "__name__", "") == "as_subclass":
            return 7
        return super().__torch_function__(func, types, args, kwargs or {})


def view_is_dispatched(subclass):
    """Whether this torch hands ``Tensor.as_subclass`` on a subclass instance to the
    subclass's ``__torch_function__``. Where it does not, the audit's "is malformed"
    branch around that call cannot be reached from Python and the test below skips."""
    probe = ints([0, 3, 1]).as_subclass(subclass)
    try:
        return type(torch.Tensor.as_subclass(probe, torch.Tensor)) is not torch.Tensor
    except RuntimeError:
        return True


@pytest.mark.parametrize("adopt", [False, True])
@pytest.mark.parametrize("subclass", [Refuses, AnswersSeven])
def test_an_array_with_no_base_class_view_is_malformed_on_both_paths(subclass, adopt):
    """The audit reads a subclass instance through a base-class view of it. A subclass
    can refuse that view or answer with something else; either way the array is
    reported as malformed, by the same error on both construction paths."""
    if not view_is_dispatched(subclass):
        pytest.skip("Tensor.as_subclass is not dispatched to __torch_function__ here")

    def index_case():
        parts = csr_parts()
        positions = parts["mode_indices"][1][0]
        coordinates = ints([0, 3, 1]).as_subclass(subclass)
        return dict(parts, mode_indices=[[], [positions, coordinates]], adopt=adopt)

    def value_case():
        return dict(csr_parts(), value=floats(3).as_subclass(subclass), adopt=adopt)

    expected = outcome(by_index, index_case())
    assert expected == (TensorIndexError, "mode_indices[1][1] is malformed")
    assert outcome(by_parts, index_case()) == expected
    expected = outcome(by_index, value_case())
    assert expected == (TensorStorageError, "values tensor is malformed")
    assert outcome(by_parts, value_case()) == expected


# --------------------------------------------------------------------------- #
# What has not moved
# --------------------------------------------------------------------------- #


def test_an_index_holds_what_the_shared_checks_return():
    parts = csr_parts()
    index = TensorIndex("ds", parts["mode_indices"], [1, 0], torch.int64)
    tensor_format, arrays, order, index_dtype = _index_components(
        "ds", parts["mode_indices"], [1, 0], torch.int64, True
    )
    assert index._format == tensor_format
    assert index._mode_order == order == (1, 0)
    assert index._index_dtype == index_dtype == torch.int64
    assert [[a.tolist() for a in level] for level in index._mode_indices] == [
        [a.tolist() for a in level] for level in arrays
    ]


def test_audited_values_exclude_the_other_two_spellings():
    parts = csr_parts()
    layout = TensorLayout.from_physical_shape((3, 4), "ds", None, torch.int64)
    audited = _index_components("ds", parts["mode_indices"], None, None, False)
    storage = SparseStorage(layout, parts["value"], _audited_index=audited)
    assert storage._mode_indices is audited[1]
    for other in (
        dict(mode_indices=parts["mode_indices"]),
        dict(index=TensorIndex("ds", parts["mode_indices"])),
    ):
        with pytest.raises(TensorStorageError, match="exactly one"):
            SparseStorage(layout, parts["value"], _audited_index=audited, **other)
    with pytest.raises(TensorStorageError, match="exactly one"):
        SparseStorage(layout, parts["value"])


# A layout that disagrees with what the arrays were audited against (format "ds", the
# identity mode order, int64 index arrays): (format, mode order, index dtype), message.
LAYOUT_MISMATCHES = {
    "format": (("dd", None, torch.int64), "index format does not match"),
    "mode order": (("ds", [1, 0], torch.int64), "index mode_order does not match"),
    "index dtype": (("ds", None, torch.int32), "index dtype does not match"),
}


@pytest.mark.parametrize("trusted", [False, True])
@pytest.mark.parametrize("case", sorted(LAYOUT_MISMATCHES))
def test_audited_values_are_held_to_the_layout(case, trusted):
    """A storage does not audit these arrays again, so it checks the one thing it can
    without touching them: that the layout it is handed is for the format, mode order
    and index dtype they were audited against. A kernel result's storage is trusted
    and skips the walk over the arrays; it does not skip this."""
    (fmt, order, index_dtype), message = LAYOUT_MISMATCHES[case]
    parts = csr_parts()
    audited = _index_components("ds", parts["mode_indices"], None, None, False)
    layout = TensorLayout.from_physical_shape((3, 4), fmt, order, index_dtype)
    with pytest.raises(TensorIndexError, match=message):
        SparseStorage(
            layout, parts["value"], _audited_index=audited, _trusted_index=trusted
        )


class RenamedLayout(TensorLayout):
    """A layout subclass; a caller's is rebuilt as an exact layout, see owned_layout."""


@pytest.mark.parametrize("trusted", [False, True])
def test_audited_values_need_an_exact_layout(trusted):
    """With audited values the storage keeps the layout object it is handed, because
    the caller's metadata names that object, so it cannot rebuild a subclass as it
    does for a caller: it refuses one."""
    parts = csr_parts()
    audited = _index_components("ds", parts["mode_indices"], None, None, False)
    subclass = RenamedLayout.from_physical_shape((3, 4), "ds", None, torch.int64)
    assert type(subclass) is RenamedLayout
    for layout in (subclass, "ds", None):
        with pytest.raises(TensorTypeError, match="layout must be a TensorLayout"):
            SparseStorage(
                layout, parts["value"], _audited_index=audited, _trusted_index=trusted
            )
    arrays = parts["mode_indices"]
    rebuilt = SparseStorage(subclass, parts["value"], mode_indices=arrays)
    assert type(rebuilt.layout) is TensorLayout


@pytest.mark.parametrize("adopt", [False, True])
def test_an_index_a_caller_kept_and_changed_is_still_refused(adopt, audits):
    """The audit this file removes from Scorch's own constructions is the one that
    protects this case, so it has to be still there for an index object a caller holds.
    """
    parts = csr_parts()
    index = TensorIndex("ds", parts["mode_indices"], _adopt=adopt)
    audits["arrays"].clear()
    positions = index._mode_indices[1][0]
    object.__setattr__(
        index, "_mode_indices", ((), (positions, torch.tensor([0.0, 3.0, 1.0])))
    )
    with pytest.raises(TensorIndexError, match="int32 or int64"):
        STensor(name="A", shape=(3, 4), index=index, value=parts["value"])
    assert audits["objects"] >= 1 and len(audits["arrays"]) == 1
