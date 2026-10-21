"""A loop that calls a member of an object declared outside it is not run in parallel.

``STensor.change_mode_order`` on an all-dense tensor of rank three or more (which
public ``einsum`` runs on an operand whose stored order is not the loop order) declares
one coordinate workspace, loops over the operand inserting every nonzero into it, then
sorts and drains it into the new layout.  The loop over the operand's outer extent was
marked ``omp parallel for``:

    coo_workspace<double, 3> wksp = coo_workspace<double, 3>(1024, result_shape);
    #pragma omp parallel for ...
    for (int64_t i = 0; i < B0_size; i++) {
      ...
            wksp.insert({i, k, j}, B_val[pB2]);

so as soon as the outer extent gave the loop two workers (32 rows), two threads grew
one ``std::vector`` and one ``std::unordered_map`` at once.  ``einsum('abc,cd->abd')``
on a dense ``abc`` of 40x5x4 ended in a segmentation fault inside the relayout.  A
smaller extent runs on one worker, which is why small tests passed.

Parallel marking now leaves such a loop serial
(``parallel_marking_pass.objects_shared_between_workers``): 28 of the 878 conversion
kernels of ranks one to four, all of them relayouts of an all-dense tensor.  A sparse
operand's relayout was never marked.

Operands are built and results read as level arrays in PyTorch.
"""

import itertools
import re

import pytest
import torch

import scorch
from scorch import stensor
from tests.test_scorch.level_array_support import (
    assert_matches,
    logical_dense,
    reference,
    stored,
)

MUTATING_CALL = re.compile(
    r"\b([A-Za-z_]\w*)\.(insert|insert_unchecked|push_back|emplace_back|sort|clear|resize)\("
)
SHAPES = {1: (6,), 2: (6, 5), 3: (6, 5, 4), 4: (6, 5, 4, 3)}
# 16 rows per worker: two workers on the loop over the outer stored extent.
LARGE = {3: (40, 5, 4), 4: (40, 3, 4, 3)}


class _Emitted(Exception):
    """Raised in place of building the kernel."""


def layouts(rank):
    return ["".join(levels) for levels in itertools.product("ds", repeat=rank)] + [
        "o" * rank
    ]


def operand(shape, seed):
    generator = torch.Generator().manual_seed(seed)
    keep = torch.rand(shape, generator=generator) < 0.7
    return (0.5 + torch.rand(shape, generator=generator, dtype=torch.float64)) * keep


def emitted(monkeypatch, call):
    """The kernel ``call`` emits, or ``None`` when it runs without one."""

    captured = []

    def stop_before_native(prepared):
        captured.append(prepared.request.cpp_sources[-1])
        raise _Emitted

    # A kernel an earlier test kept for this call would serve it without reaching the stop.
    monkeypatch.setattr(stensor, "_GENERATED_KERNELS", {})
    monkeypatch.setattr(stensor, "_load_validated_prepared_kernel", stop_before_native)
    try:
        call()
    except _Emitted:
        return captured[0][captured[0].index("evaluate(") :]
    return None


def objects_shared_in_parallel_regions(source):
    """``object.member`` for each mutating member call inside an OpenMP parallel region
    on an object the region does not declare."""

    lines = source.splitlines()
    found = []
    for at, line in enumerate(lines):
        if not line.strip().startswith("#pragma omp parallel"):
            continue
        depth, started, block = 0, False, []
        for inner in lines[at + 1 :]:
            block.append(inner)
            depth += inner.count("{") - inner.count("}")
            started = started or "{" in inner
            if started and depth <= 0:
                break
        region = "\n".join(block)
        for call in MUTATING_CALL.finditer(region):
            name = call.group(1)
            declared = re.search(
                rf"(?m)^\s*[A-Za-z_][\w:]*(?:<[^;\n]*>)?[\s*&]+{name}\s*[=;({{]", region
            )
            if not declared and f"{name}.{call.group(2)}" not in found:
                found.append(f"{name}.{call.group(2)}")
    return found


def test_the_scan_finds_an_outer_object_and_not_one_the_region_declares():
    shared = (
        "coo_workspace<double, 3> wksp = f();\n"
        "#pragma omp parallel for\n"
        "for (int64_t i = 0; i < n; i++) {\n"
        "  wksp.insert({i}, 1.0);\n"
        "}\n"
        "wksp.sort();\n"
    )
    assert objects_shared_in_parallel_regions(shared) == ["wksp.insert"]
    private = shared.replace(
        "  wksp.insert", "  coo_workspace<double, 3> wksp = f();\n  wksp.insert"
    )
    assert objects_shared_in_parallel_regions(private) == []
    assert objects_shared_in_parallel_regions(shared.replace("#pragma", "//")) == []


@pytest.mark.parametrize("rank", [3, 4])
def test_dense_relayout_inserts_into_its_workspace_from_a_serial_loop(
    monkeypatch, rank
):
    dense = operand(SHAPES[rank], 3)
    for order in itertools.permutations(range(rank)):
        if list(order) == sorted(order):
            continue
        source = emitted(
            monkeypatch,
            lambda: stored(dense, "d" * rank, "A").change_mode_order(list(order)),
        )
        assert "wksp.insert(" in source
        assert objects_shared_in_parallel_regions(source) == []
        assert "#pragma omp" not in source[: source.index("wksp.insert(")]


@pytest.mark.parametrize("rank", [1, 2, 3])
def test_no_conversion_kernel_shares_an_object_between_workers(monkeypatch, rank):
    dense = operand(SHAPES[rank], 5)
    kernels = 0
    for fmt in layouts(rank):
        calls = [lambda: stored(dense, fmt, "A").to_dense()]
        calls += [
            lambda target=target: stored(dense, fmt, "A").to_sparse(target)
            for target in layouts(rank)
            if set(target) != {"d"}
        ]
        calls += [
            lambda order=order: stored(dense, fmt, "A").change_mode_order(list(order))
            for order in itertools.permutations(range(rank))
            if list(order) != sorted(order)
        ]
        for call in calls:
            source = emitted(monkeypatch, call)
            if source is not None:
                kernels += 1
                assert objects_shared_in_parallel_regions(source) == []
    assert kernels


@pytest.mark.parametrize("extent", ["small", "large"])
@pytest.mark.parametrize(
    "order", [(0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]
)
def test_dense_rank_three_relayout_matches(order, extent):
    shape = SHAPES[3] if extent == "small" else LARGE[3]
    dense = operand(shape, 7)

    tensor = stored(dense, "ddd", "A")
    tensor.change_mode_order(list(order))

    assert [int(mode) for mode in tensor.mode_order] == list(order)
    assert torch.equal(logical_dense(tensor), dense)
    assert torch.equal(
        tensor.values.reshape(-1).to(torch.float64), dense.permute(order).reshape(-1)
    )


@pytest.mark.parametrize("order", [(0, 1, 3, 2), (3, 2, 1, 0), (1, 0, 2, 3)])
def test_dense_rank_four_relayout_matches(order):
    dense = operand(LARGE[4], 9)

    tensor = stored(dense, "dddd", "A")
    tensor.change_mode_order(list(order))

    assert [int(mode) for mode in tensor.mode_order] == list(order)
    assert torch.equal(logical_dense(tensor), dense)


@pytest.mark.parametrize("result_format", ["sss", "ddd", "dss"])
def test_einsum_that_relays_out_a_dense_operand_matches(result_format):
    """``abc,cd->abd`` with a compressed ``c`` in the matrix: the loops run a, c, b, d
    and the dense ``abc`` is relaid out to (a, c, b) first."""

    extents = {"a": 40, "b": 5, "c": 4, "d": 6}
    dense = [operand((40, 5, 4), 11), operand((4, 6), 12)]

    result = scorch.einsum(
        "abc,cd->abd",
        stored(dense[0], "ddd", "A"),
        stored(dense[1], "sd", "B"),
        format=result_format,
    )

    assert tuple(extents[name] for name in "abd") == tuple(result.shape)
    assert_matches(result, reference("abc,cd->abd", dense), torch.float64)
