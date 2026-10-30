"""The checks every ``einsum`` call passes, made cheaper (series step 31b), refuse what
they refused, with the same codes, in the same order.

Every call that reaches a kernel runs these checks, so step 31b rewrote them to do less
work per call: ``_entry_keywords`` reads the keywords in one pass, ``einsum`` reads its
operands once and hands them on, the compiler path's gradient check reads the operands
it is handed, ``_validated_einsum_request`` builds its error lists only once an operand
fails, ``_logical_index_sizes`` reads each operand's shape once, and the options holder
reads the forced schedule and the parse memo in place.  These tests pin:

* a refused keyword raises before any keyword is dropped, and the first one refused is
  the one named; defaults are dropped and ``output_format`` is read as ``format``;
* a None operand is refused before an operand of another type, and every operand of a
  wrong type is named;
* a singleton output level is refused, a dense or compressed one is not;
* an index whose extents disagree, and an operand of the wrong rank, are refused;
* ``einsum`` reads its operands once on either route;
* the compiler path sends a call to be recorded when an operand it is handed, or one it
  reads itself, asks for a gradient, and asks ``autograd.wanted_by_any`` about the
  handed operands only under forward-mode AD;
* ``_is_dense_operand`` answers as before;
* the options holder checks a named schedule as before, and a call with keywords takes
  the context's forced schedule, or the named one over it.
"""

from __future__ import annotations

from typing import Any, List

import pytest
import torch
import torch.autograd.forward_ad as fwad

import scorch  # type: ignore[import-untyped]
import scorch.ops as ops  # type: ignore[import-untyped]
from scorch import STensor  # type: ignore[import-untyped]
from scorch.compiler.scheduler import (  # type: ignore[import-untyped]
    Schedule,
    schedule_force,
)
from scorch.format import LevelFormat, TensorFormat  # type: ignore[import-untyped]


@pytest.fixture(autouse=True)
def _nothing_kept_between_tests(fresh_kernel_caches: None):
    """The kernel caches and the build snapshots start empty (fresh_kernel_caches)."""


def _sparse(rows: int = 6, cols: int = 5, seed: int = 0) -> STensor:
    generator = torch.Generator().manual_seed(seed)
    dense = torch.rand(rows, cols, generator=generator, dtype=torch.float64)
    dense[dense < 0.5] = 0.0
    return STensor.from_torch(dense).to_sparse("ds")


def _dense(*shape: int, seed: int = 1) -> STensor:
    generator = torch.Generator().manual_seed(seed)
    values = torch.rand(*shape, generator=generator, dtype=torch.float64)
    return STensor.from_torch(values)


# The module einsum asks (``ops._autograd``), as einsum reaches it.
scorch_autograd = ops._autograd


def _code(error: BaseException) -> Any:
    return getattr(error, "code", None)


# ``_entry_keywords``


def test_a_refused_keyword_raises_before_anything_is_dropped() -> None:
    kwargs = {"format": None, "first_unknown": 1, "second_unknown": 2}
    with pytest.raises(Exception) as caught:
        ops._entry_keywords(kwargs, ops._EINSUM_KEYWORDS, "einsum")
    assert _code(caught.value) == "unknown_keyword_argument"
    assert "'first_unknown'" in str(caught.value)
    assert kwargs == {"format": None, "first_unknown": 1, "second_unknown": 2}


def test_defaults_are_dropped_and_output_format_is_read_as_format() -> None:
    kwargs = {"format": None, "use_cache": True, "schedule": None, "output_format": "d"}
    ops._entry_keywords(kwargs, ops._EINSUM_KEYWORDS, "einsum")
    assert kwargs == {"format": "d"}
    kept = {"use_cache": False, "format": "d"}
    ops._entry_keywords(kept, ops._EINSUM_KEYWORDS, "einsum")
    assert kept == {"use_cache": False, "format": "d"}


def test_format_and_output_format_naming_different_formats_are_refused() -> None:
    with pytest.raises(Exception) as caught:
        ops._entry_keywords(
            {"format": "d", "output_format": "s"}, ops._EINSUM_KEYWORDS, "einsum"
        )
    assert _code(caught.value) == "format_keywords_conflict"


# ``_validated_einsum_request``


def test_a_none_operand_is_refused_before_another_type() -> None:
    a = _sparse()
    for tensors in ((None, 3), (3, None), (a, None)):
        with pytest.raises(Exception) as caught:
            ops._validated_einsum_request("ij,jk->ik", tensors, {})
        assert _code(caught.value) == "einsum_operand_type_invalid"
        assert "cannot be None" in str(caught.value)


def test_every_operand_of_a_wrong_type_is_named() -> None:
    a = _sparse()
    with pytest.raises(Exception) as caught:
        ops._validated_einsum_request("ij,jk,k->i", (a, 3, "x"), {})
    assert _code(caught.value) == "einsum_operand_type_invalid"
    assert "got int, str" in str(caught.value)


def test_a_singleton_output_level_is_refused() -> None:
    a, b = _sparse(), _dense(5, 4)
    singleton = TensorFormat([LevelFormat("s"), LevelFormat("singleton")])
    with pytest.raises(Exception) as caught:
        ops._validated_einsum_request("ij,jk->ik", (a, b), {"format": singleton})
    assert _code(caught.value) == "singleton_level_unsupported"
    for allowed in ("dd", "ds"):
        ops._validated_einsum_request("ij,jk->ik", (a, b), {"format": allowed})


# ``_logical_index_sizes``


def test_index_sizes_are_read_through_the_mode_order() -> None:
    a, b = _sparse(6, 5), _dense(5, 4)
    sizes = ops._logical_index_sizes([["i", "j"], ["j", "k"]], (a, b))
    assert sizes == {"i": 6, "j": 5, "k": 4}


def test_an_index_whose_extents_disagree_is_refused() -> None:
    a, b = _sparse(6, 5), _dense(3, 4)
    with pytest.raises(Exception) as caught:
        ops._logical_index_sizes([["i", "j"], ["j", "k"]], (a, b))
    assert "incompatible dimensions 5 and 3" in str(caught.value)


def test_an_operand_of_the_wrong_rank_is_refused() -> None:
    a = _sparse(6, 5)
    with pytest.raises(Exception) as caught:
        ops._logical_index_sizes([["i", "j", "k"]], (a,))
    assert _code(caught.value) == "einsum_operand_rank_mismatch"


# ``einsum`` reads its operands once


@pytest.mark.parametrize("route", ["compiler", "dense"])
def test_einsum_reads_its_operands_once(
    route: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    reads: List[int] = []
    original = ops._einsum_operands

    def counted(expression: Any, tensors: Any) -> Any:
        reads.append(1)
        return original(expression, tensors)

    monkeypatch.setattr(ops, "_einsum_operands", counted)
    b = _dense(5, 4)
    a = _sparse(6, 5) if route == "compiler" else _dense(6, 5, seed=2)
    got = scorch.einsum("ij,jk->ik", a, b, format="dd")
    want = a.to_torch() @ b.to_torch()
    assert torch.allclose(got.to_torch(), want, rtol=1e-12, atol=0.0)
    assert len(reads) == 1


# The compiler path's gradient check


class _SentToBeRecorded(Exception):
    """Raised in place of ``ops._einsum_recorded``: the call was sent to be recorded."""


def _record_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    def recorded(expression: Any, tensors: Any, kwargs: Any) -> Any:
        raise _SentToBeRecorded

    monkeypatch.setattr(ops, "_einsum_recorded", recorded)


def test_the_compiler_path_reads_the_operands_it_is_not_handed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # ``_einsum_via_compiler`` hands no operands on, so the check reads them itself, in
    # the positional and the list form; an operand asking for a gradient, a torch tensor
    # or an STensor, sends the call to be recorded.
    _record_raises(monkeypatch)
    a = _sparse(6, 5)
    leaf = torch.rand(5, 4, dtype=torch.float64, requires_grad=True)
    asking = _dense(5, 4)
    asking.requires_grad = True
    for b in (leaf, asking):
        with pytest.raises(_SentToBeRecorded):
            ops._einsum_via_compiler("ij,jk->ik", a, b, format="dd")
        with pytest.raises(_SentToBeRecorded):
            ops._einsum_via_compiler("ij,jk->ik", [a, b], format="dd")
    b = _dense(5, 4)
    got = ops._einsum_via_compiler("ij,jk->ik", a, b, format="dd")
    want = a.to_torch() @ b.to_torch()
    assert torch.allclose(got.to_torch(), want, rtol=1e-12, atol=0.0)


def test_forward_mode_ad_asks_wanted_by_any_about_the_handed_operands(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _record_raises(monkeypatch)
    asked: List[Any] = []
    original = scorch_autograd.wanted_by_any

    def counted(operands: Any) -> bool:
        asked.append(operands)
        return original(operands)

    monkeypatch.setattr(scorch_autograd, "wanted_by_any", counted)
    a, b = _sparse(6, 5), _dense(5, 4)
    scorch.einsum("ij,jk->ik", a, b, format="dd")
    assert asked == []  # no forward-mode level: autograd.py is not asked
    primal = torch.rand(5, 4, dtype=torch.float64)
    with fwad.dual_level():
        dual = fwad.make_dual(primal, torch.ones(5, 4, dtype=torch.float64))
        with torch.no_grad():
            with pytest.raises(_SentToBeRecorded):
                scorch.einsum("ij,jk->ik", a, dual, format="dd")
            got = scorch.einsum("ij,jk->ik", a, b, format="dd")
    assert torch.allclose(
        got.to_torch(), a.to_torch() @ b.to_torch(), rtol=1e-12, atol=0.0
    )
    # Each call asked once, about the operands ``einsum`` read and handed on.
    assert [len(operands) for operands in asked] == [2, 2]
    assert asked[0][0] is a and asked[0][1] is dual
    assert asked[1][0] is a and asked[1][1] is b


# ``_is_dense_operand``


def test_is_dense_operand_answers_as_before() -> None:
    cases = [
        (_dense(3, 4), True),
        (_sparse(), False),
        (torch.rand(3, 4), True),
        (torch.rand(3, 4).to_sparse(), False),
        (torch.rand(3, 4).to_sparse_csr(), False),
        (None, False),
        (3, False),
    ]
    for x, expected in cases:
        assert ops._is_dense_operand(x) is expected


# The options holder's keyword path


def test_a_named_schedule_is_checked_as_before() -> None:
    with pytest.raises(Exception) as caught:
        ops._DeferredOptions({"format": "dd", "schedule": Schedule(), "_schedule": 3})
    assert _code(caught.value) == "conflicting_schedule"
    with pytest.raises(Exception) as caught:
        ops._DeferredOptions({"format": "dd", "schedule": "not a schedule"})
    assert _code(caught.value) == "schedule_type_invalid"


def test_a_call_with_keywords_takes_the_forced_or_the_named_schedule() -> None:
    forced = Schedule()
    with schedule_force(forced):
        first = ops._DeferredOptions({"format": "dd"})  # the parse, then its memo
        second = ops._DeferredOptions({"format": "dd"})
        assert first.requested_schedule == forced
        assert second.requested_schedule is forced
        assert second.schedule_key == forced.cache_key
        assert second.reuse_key == first.reuse_key
        named = Schedule()
        third = ops._DeferredOptions({"format": "dd", "schedule": named})
        assert third.requested_schedule is named
        assert third.schedule_key == named.cache_key
    plain = ops._DeferredOptions({"format": "dd"})
    assert plain.requested_schedule is None
    assert plain.schedule_key is None
