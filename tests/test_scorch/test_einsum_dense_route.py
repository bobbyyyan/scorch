"""An einsum whose operands and result are all dense is ``torch.einsum``'s call.

Drop-in note, section 0.1 step 1: under ``import scorch as torch`` a dense model's every
``torch.einsum`` reaches ``scorch.einsum``, and only entirely dense calls go to torch
(Bobby, desk t-shim-matmul-ops, 2026-10-14 09:09:08).  These tests pin:

* an all-dense call is computed by torch, bit-identical to ``torch.einsum``, in every
  call form torch accepts (explicit and implicit output, ellipsis, an operand list, the
  sublist form), with torch tensors, ``nn.Parameter`` operands and dense STensors, and
  keeps autograd;
* the result's type follows the call (scorch-merge, 2026-10-14 18:4x): torch tensors in
  and no result keyword give torch's tensor; a dense STensor operand, or a ``format``
  (``output_format``) or ``output_mode_order``, give a dense STensor over torch's
  tensor, as the compiler path does for the same call.  It shares torch's memory when
  torch's tensor is laid out in some order of its axes, and keeps the history;
* torch's kernel computes it (the profiler sees ``aten::einsum``) and nothing is built;
* a scorch keyword gets the compiler path's checks and codes, a non-identity
  ``output_mode_order`` gives a dense STensor in that order, holding every position, and
  ``time_dict`` is filled;
* a ``format`` with a sparse level, spelt ``format=`` or ``output_format=``, keeps the
  compiler path even when every operand is dense (scorch-merge, 2026-10-14: a sparse
  result is sparse work).  It is the route einsum ships, kept until the measurement in
  drop-in note 5.1 decides between it and torch's product followed by a conversion; the
  values are torch's, the stored entries are the compiler entry's, and the same call
  stores the same entries every time.  A format the compiler refuses (a mixed coordinate
  hierarchy, a malformed format) gets the compiler's refusal;
* a keyword einsum does not take (``out``) is refused at the entry with
  ``unknown_keyword_argument``, a TypeError as torch's is, before anything is computed
  (patch 0072's keyword rule, ``_entry_keywords``);
* a schedule request, in each of its four forms (``schedule=``, ``_schedule=``,
  ``_compile_options`` carrying ``requested_schedule``, ``schedule_force``), keeps the
  compiler path with all-dense operands and matches float64 (scorch-merge,
  2026-10-14: a request is explicit and torch cannot honor it);
* a compile-only request and any non-dense operand keep the compiler path;
* ``ops._einsum_via_compiler``, the compiler's entry for callers that need scorch's
  kernel for an all-dense program, builds the kernel ``einsum`` builds for the same
  sparse call: same source, same build cache key, same dispatch-cache key.
"""

from __future__ import annotations

from typing import Any, Callable

import pytest
import torch

import scorch
import scorch.stensor as stensor_module
import scorch.utils
from scorch import ops
from scorch.compiler.compile_options import CompileOptions
from scorch.compiler.scheduler import Schedule, schedule_force
from scorch.exceptions import defect_code
from scorch.format import parse_format
from scorch.stensor import STensor


def _operands(dtype: torch.dtype = torch.float64) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(7)
    a = torch.randn(3, 4, generator=generator, dtype=dtype)
    b = torch.randn(4, 5, generator=generator, dtype=dtype)
    return a, b


@pytest.fixture
def no_build(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail the test if anything reaches the kernel build boundary."""

    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("an all-dense einsum built a kernel")

    monkeypatch.setattr(scorch.utils, "_run_jit_build_child_privately", refuse)


@pytest.fixture
def compiler_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail the test if the all-dense path is taken."""

    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("the call took the all-dense path")

    monkeypatch.setattr(ops, "_dense_einsum", refuse)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    "call",
    [
        lambda e, a, b: e("ij,jk->ik", a, b),
        lambda e, a, b: e("ij,jk", a, b),
        lambda e, a, b: e("...j,jk->...k", a, b),
        lambda e, a, b: e("ij,jk->ik", [a, b]),
        lambda e, a, b: e(a, [0, 1], b, [1, 2], [0, 2]),
        lambda e, a, b: e(a, [0, 1], b, [1, 2]),
    ],
    ids=["explicit", "implicit", "ellipsis", "list", "sublist", "sublist_implicit"],
)
def test_all_dense_call_is_torch_einsum(
    call: Callable[..., Any], dtype: torch.dtype, no_build: None
) -> None:
    a, b = _operands(dtype)
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU]
    ) as profile:
        got = call(scorch.einsum, a, b)
    want = call(torch.einsum, a, b)
    assert type(got) is torch.Tensor
    assert got.dtype == want.dtype and torch.equal(got, want)
    assert "aten::einsum" in {event.key for event in profile.key_averages()}


def test_dense_stensor_operands_are_computed_by_torch(no_build: None) -> None:
    """STensor operands give an STensor (the compiler path's type for the call), whose
    values are torch's, bit for bit."""
    a, b = _operands()
    got = scorch.einsum("ij,jk->ik", STensor.from_torch(a), STensor.from_torch(b))
    assert type(got) is STensor and got.format.is_dense()
    assert tuple(got.mode_order) == (0, 1)
    assert torch.equal(got.to_torch(), torch.einsum("ij,jk->ik", a, b))


def test_parameter_operand_keeps_autograd(no_build: None) -> None:
    a, b = _operands()
    weight = torch.nn.Parameter(b.clone())
    reference = torch.nn.Parameter(b.clone())
    got = scorch.einsum("ij,jk->ik", a, weight)
    want = torch.einsum("ij,jk->ik", a, reference)
    assert type(got) is torch.Tensor and got.requires_grad
    got.sum().backward()
    want.sum().backward()
    assert torch.equal(weight.grad, reference.grad)


def test_dense_format_returns_a_dense_stensor(no_build: None) -> None:
    """A ``format`` is a scorch keyword, so the result is an STensor even with torch
    operands; torch computes it."""
    a, b = _operands()
    time_dict: dict = {}
    got = scorch.einsum("ij,jk->ik", a, b, format="dd", time_dict=time_dict)
    assert type(got) is STensor and got.format.is_dense()
    assert torch.equal(got.to_torch(), torch.einsum("ij,jk->ik", a, b))
    assert "eval_time" in time_dict


@pytest.mark.parametrize(
    "expression", ["ij,jk->ki", "ij,jk->ik"], ids=["permuted", "row_major"]
)
def test_a_dense_stensor_result_shares_torchs_memory(
    expression: str, no_build: None
) -> None:
    """No copy: the STensor's mode order is the order of the axes torch's tensor is laid
    out in, so its values are that tensor's memory.  ``torch.einsum`` returns
    ``"ij,jk->ki"`` as a transposed view, and the STensor stores it in mode order (1, 0)
    rather than copying it; with no ``output_mode_order`` the level order is scorch's
    choice."""
    a, b = _operands()
    want = torch.einsum(expression, a, b)
    by_stride = sorted(range(want.dim()), key=lambda axis: (-want.stride(axis), axis))
    assert want.permute(*by_stride).is_contiguous()
    got = scorch.einsum(expression, a, b, format="dd")
    assert type(got) is STensor and tuple(got.mode_order) == tuple(by_stride)
    assert got.values.numel() == want.numel()
    assert torch.equal(got.values, want.permute(*by_stride).reshape(-1))
    assert torch.equal(got.to_torch(), want)


def test_a_dense_stensor_result_keeps_autograd(no_build: None) -> None:
    """The STensor's values carry torch's history: its dense tensor differentiates to the
    operands as torch's result does, float64 to the last bit."""
    a, b = _operands()
    weight = torch.nn.Parameter(b.clone())
    reference = torch.nn.Parameter(b.clone())
    got = scorch.einsum("ij,jk->ik", a, weight, format="dd")
    want = torch.einsum("ij,jk->ik", a, reference)
    assert type(got) is STensor
    dense = got.to_torch()
    assert dense.requires_grad
    upstream = torch.rand(want.shape, generator=torch.Generator().manual_seed(3))
    (dense * upstream.double()).sum().backward()
    (want * upstream.double()).sum().backward()
    assert torch.equal(weight.grad, reference.grad)


@pytest.mark.parametrize("keyword", ["format", "output_format"])
@pytest.mark.parametrize("requested", ["ds", "sd", "ss", "oo"])
@pytest.mark.parametrize(
    "make",
    [lambda a: a, lambda a: STensor.from_torch(a)],
    ids=["torch", "dense_stensor"],
)
def test_sparse_result_format_keeps_the_compiler_path(
    keyword: str,
    requested: str,
    make: Callable[..., Any],
    monkeypatch: pytest.MonkeyPatch,
    compiler_only: None,
) -> None:
    a, _ = _operands()
    reached = []
    monkeypatch.setattr(
        ops, "_einsum_owned", lambda *args, **kwargs: reached.append(args) or "owned"
    )
    assert scorch.einsum("ij->ij", make(a), **{keyword: requested}) == "owned"
    parsed = parse_format(requested)
    assert scorch.einsum("ij->ij", make(a), **{keyword: parsed}) == "owned"
    assert len(reached) == 2


@pytest.mark.parametrize("keyword", ["format", "output_format"])
@pytest.mark.parametrize("requested", ["ds", "sd", "ss", "oo"])
def test_sparse_result_is_the_compiler_entrys_result(
    keyword: str, requested: str, compiler_only: None
) -> None:
    """The shipped route, kept until note 5.1's measurement: the values are torch's,
    the result's kinds per mode are the requested ones, and the stored entries are
    those of the compiler's own entry.  Which entries a dense result stores is not
    pinned beyond that: when the values are the same, the faster route decides."""

    a, b = _operands()
    a = a.masked_fill(a < 0, 0.0)
    got = scorch.einsum("ij,jk->ik", a, b, **{keyword: requested})
    via_compiler = ops._einsum_via_compiler("ij,jk->ik", a, b, format=requested)
    assert isinstance(got, STensor)
    assert got.format.get_level_types() == parse_format(requested).get_level_types()
    assert tuple(got.mode_order) == (0, 1)
    torch.testing.assert_close(
        got.to_torch().double(),
        torch.einsum("ij,jk->ik", a.double(), b.double()),
        rtol=1e-12,
        atol=1e-12,
    )
    assert torch.equal(got.values, via_compiler.values)
    for mine, theirs in zip(
        got._native_mode_indices(), via_compiler._native_mode_indices()
    ):
        assert all(torch.equal(x, y) for x, y in zip(mine, theirs))


def _stored(result: STensor) -> tuple:
    arrays = tuple(
        tuple(array.tolist())
        for level in result._native_mode_indices()
        for array in level
    )
    return arrays, tuple(result.values.tolist())


@pytest.mark.parametrize("requested", ["ds", "ss", "oo"])
def test_a_product_that_cancels_to_exact_zeros(requested: str) -> None:
    """Row 1 of a @ b is exactly zero: its two products cancel.  The values are right,
    and the same call stores the same entries every time it is made; whether the zeros
    are stored is the route's choice."""

    a = torch.tensor([[1.0, 2.0], [3.0, -3.0], [0.5, 0.25]], dtype=torch.float64)
    b = torch.tensor([[2.0, 1.0], [2.0, 1.0]], dtype=torch.float64)
    want = a @ b
    assert bool((want[1] == 0).all())
    runs = [scorch.einsum("ij,jk->ik", a, b, output_format=requested) for _ in range(3)]
    for got in runs:
        assert isinstance(got, STensor)
        assert got.format.get_level_types() == parse_format(requested).get_level_types()
        assert torch.equal(got.to_torch(), want)
    assert len({_stored(got) for got in runs}) == 1


def test_format_and_output_format_together(no_build: None) -> None:
    a, b = _operands()
    want = torch.einsum("ij,jk->ik", a, b)
    got = scorch.einsum("ij,jk->ik", a, b, format="dd", output_format=["d", "d"])
    assert type(got) is STensor and torch.equal(got.to_torch(), want)
    with pytest.raises(TypeError) as caught:
        scorch.einsum("ij,jk->ik", a, b, format="dd", output_format="ds")
    assert defect_code(caught.value) == "format_keywords_conflict"


@pytest.mark.parametrize("requested", ["do", "od", "so", "os", "q"])
def test_a_result_format_the_compiler_refuses_gets_its_refusal(
    requested: str, compiler_only: None
) -> None:
    """A mixed coordinate hierarchy, and a format that does not parse, get exactly the
    refusal the compiler's own entry gives them."""

    a, _ = _operands()
    with pytest.raises(Exception) as public:
        scorch.einsum("ij->ij", a, format=requested)
    with pytest.raises(Exception) as compiled:
        ops._einsum_via_compiler("ij->ij", a, format=requested)
    assert type(public.value) is type(compiled.value)
    assert str(public.value) == str(compiled.value)
    assert defect_code(public.value) == defect_code(compiled.value)
    assert defect_code(public.value) is not None


@pytest.mark.parametrize("requested", [None, "dd"])
def test_output_mode_order_returns_that_stensor(requested: Any, no_build: None) -> None:
    a, b = _operands()
    keywords = {} if requested is None else {"format": requested}
    got = scorch.einsum("ij,jk->ik", a, b, output_mode_order=[1, 0], **keywords)
    assert isinstance(got, STensor) and tuple(got.mode_order) == (1, 0)
    assert got.format.is_dense() and got.values.numel() == 15
    assert torch.equal(got.to_torch(), torch.einsum("ij,jk->ik", a, b))


@pytest.mark.parametrize(
    "expression, keywords, code",
    [
        (
            "ij,jk->ik",
            {"output_mode_order": [0, 0]},
            "output_mode_order_not_permutation",
        ),
        ("ij,jk->ik", {"output_mode_order": "10"}, "output_mode_order_type_invalid"),
        ("ij,jk->ik", {"format": "d"}, None),
        ("ij,jk->ik", {"format": "ddd"}, None),
        ("ij,jk->ik", {"format": "sss"}, None),
    ],
    ids=[
        "not_permutation",
        "mode_order_type",
        "format_rank_low",
        "format_rank_high",
        "sparse_format_rank_high",
    ],
)
def test_scorch_keyword_mistake_gets_the_compiler_path_code(
    expression: str, keywords: dict, code: Any, no_build: None
) -> None:
    a, b = _operands()
    with pytest.raises(Exception) as dense:
        scorch.einsum(expression, a, b, **keywords)
    with pytest.raises(Exception) as sparse:
        scorch.einsum(expression, a.to_sparse_csr(), b, **keywords)
    assert type(dense.value) is type(sparse.value)
    assert defect_code(dense.value) == defect_code(sparse.value)
    if code is not None:
        assert defect_code(dense.value) == code


@pytest.mark.parametrize("operands", ["dense", "sparse"])
def test_a_keyword_einsum_does_not_take_is_refused_with_a_code(
    operands: str, no_build: None
) -> None:
    a, b = _operands()
    first = a if operands == "dense" else a.to_sparse_csr()
    with pytest.raises(TypeError) as caught:
        scorch.einsum("ij,jk->ik", first, b, out=torch.empty(3, 5, dtype=a.dtype))
    assert defect_code(caught.value) == "unknown_keyword_argument"


_REQUESTS = {
    "schedule": lambda: ({"schedule": Schedule()}, None),
    "_schedule": lambda: ({"_schedule": Schedule()}, None),
    "_compile_options": lambda: (
        {
            "_compile_options": CompileOptions.from_environment(
                requested_schedule=Schedule()
            )
        },
        None,
    ),
    "schedule_force": lambda: ({}, Schedule()),
    "schedule_force_with_format": lambda: ({"format": "dd"}, Schedule()),
}


@pytest.mark.parametrize("request_form", sorted(_REQUESTS))
def test_schedule_request_keeps_the_compiler_path(
    request_form: str, compiler_only: None
) -> None:
    a, b = _operands()
    keywords, forced = _REQUESTS[request_form]()
    keywords.setdefault("format", "dd")
    if forced is None:
        got = scorch.einsum("ij,jk->ik", a, b, **keywords)
    else:
        with schedule_force(forced):
            if request_form == "schedule_force":
                got = scorch.einsum("ij,jk->ik", a, b)  # no keyword at all
            else:
                got = scorch.einsum("ij,jk->ik", a, b, **keywords)
    assert isinstance(got, STensor)
    torch.testing.assert_close(
        got.to_torch(), torch.einsum("ij,jk->ik", a, b), rtol=1e-12, atol=1e-12
    )


def test_same_call_without_a_request_goes_to_torch(no_build: None) -> None:
    a, b = _operands()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU]
    ) as profile:
        got = scorch.einsum("ij,jk->ik", a, b, format="dd")
    assert type(got) is STensor and got.format.is_dense()
    assert "aten::einsum" in {event.key for event in profile.key_averages()}


@pytest.mark.parametrize(
    "make",
    [
        lambda a, b: (a.to_sparse_csr(), b),
        lambda a, b: (a.to_sparse(), b),
        lambda a, b: (STensor.from_torch(a).to_sparse("ds"), b),
    ],
    ids=["torch_csr", "torch_coo", "stensor_ds"],
)
def test_any_sparse_operand_keeps_the_compiler_path(
    make: Callable[..., Any], monkeypatch: pytest.MonkeyPatch, compiler_only: None
) -> None:
    a, b = _operands()
    operands = make(a, b)
    reached = []
    monkeypatch.setattr(
        ops, "_einsum_owned", lambda *args, **kwargs: reached.append(args) or "owned"
    )
    assert scorch.einsum("ij,jk->ik", *operands) == "owned"
    assert len(reached) == 1


def test_compiler_entry_builds_what_einsum_builds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    a, _ = _operands()
    sparse = a.masked_fill(a < 0, 0.0).to_sparse_csr()
    other = torch.randn(3, 4, generator=torch.Generator().manual_seed(8), dtype=a.dtype)
    original = ops._load_validated_prepared_kernel
    builds: list = []

    def record(prepared: Any) -> Any:
        builds.append((prepared.cache_key, prepared.request.cpp_sources))
        return original(prepared)

    monkeypatch.setattr(ops, "_load_validated_prepared_kernel", record)
    runs = []
    for entry in (scorch.einsum, ops._einsum_via_compiler):
        monkeypatch.setattr(ops, "_kernel_cache", {})
        monkeypatch.setattr(ops, "_einsum_dispatch_cache", {})
        monkeypatch.setattr(stensor_module, "_GENERATED_KERNELS", {})
        del builds[:]
        result = entry("ij,ij->ij", sparse, other, format="ds")
        runs.append((list(builds), set(ops._einsum_dispatch_cache), result.to_torch()))
    (first_builds, first_keys, first), (second_builds, second_keys, second) = runs
    assert first_builds and first_builds == second_builds
    assert first_keys and first_keys == second_keys
    assert torch.equal(first, second)


def test_compiler_entry_compiles_an_all_dense_program(compiler_only: None) -> None:
    a, b = _operands()
    got = ops._einsum_via_compiler("ij,jk->ik", a, b, format="dd")
    assert isinstance(got, STensor)
    torch.testing.assert_close(
        got.to_torch(), torch.einsum("ij,jk->ik", a, b), rtol=1e-12, atol=1e-12
    )


def test_compile_only_keeps_the_compiler_path(
    monkeypatch: pytest.MonkeyPatch, compiler_only: None
) -> None:
    a, b = _operands()
    monkeypatch.setattr(ops, "_einsum_owned", lambda *args, **kwargs: "owned")
    assert scorch.einsum("ij,jk->ik", a, b, compile_only=True) == "owned"
