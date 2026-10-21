"""A LoopIR execution's result goes through the kernel-result wrapper, never the public one.

The public ``STensor`` constructor accepts a compressed level whose coordinates are
out of order or repeated inside a parent and stores a sorted, repeat-summed copy, as
torch does for a caller's CSR arrays.  A kernel's result must never be repaired that
way: an unordered or repeated coordinate there means the compiler emitted something
wrong.  ``stensor._wrap_generated_result`` declares the index trusted, so it is walked
and a malformed one raises when kernel-result validation is on (the test suite turns
it on), and it is stored exactly as the kernel returned it when validation is off.

These tests replace the native module with one that returns a chosen result, so no
compiler runs.
"""

import types

import pytest
import torch

import scorch.ops as ops
from scorch import storage
from scorch.compiler.cin import ForAll, IndexVar, TensorAssign, TensorVar
from scorch.compiler.loopir.pipeline import execute_cin_via_loopir
from scorch.exceptions import TensorIndexError
from scorch.stensor import STensor

INDEX = torch.int32


def copy_program():
    """``C[i,j] = A[i,j]``, both DCSR."""

    i, j = IndexVar("i"), IndexVar("j")
    a = TensorVar("A", fmt="ss", dtype=torch.float32)
    c = TensorVar("C", fmt="ss", dtype=torch.float32)
    return ForAll(i, ForAll(j, TensorAssign(c[i, j], a[i, j])))


def operand():
    """A 1 x 2 DCSR matrix holding 2 at (0, 0) and 3 at (0, 1)."""

    return STensor.from_components(
        (1, 2),
        "ss",
        [
            [torch.tensor([0, 1], dtype=INDEX), torch.tensor([0], dtype=INDEX)],
            [torch.tensor([0, 2], dtype=INDEX), torch.tensor([0, 1], dtype=INDEX)],
        ],
        torch.tensor([2.0, 3.0]),
        name="A",
    )


def native_result(child_pos, child_crd, values):
    root = [torch.tensor([0, 1], dtype=INDEX), torch.tensor([0], dtype=INDEX)]
    child = [
        torch.tensor(child_pos, dtype=INDEX),
        torch.tensor(child_crd, dtype=INDEX),
    ]
    return types.SimpleNamespace(
        storage=types.SimpleNamespace(
            index=types.SimpleNamespace(mode_indices=[root, child]),
            value=torch.tensor(values),
        )
    )


@pytest.fixture
def validation_flag():
    """The process-wide kernel-result validation flag, restored afterwards."""

    saved = storage._VALIDATE_KERNEL_RESULTS[0]
    yield storage._VALIDATE_KERNEL_RESULTS
    storage._VALIDATE_KERNEL_RESULTS[0] = saved


@pytest.fixture
def run_with_native_result(monkeypatch, validation_flag):
    """Execute the copy program with the native module's result replaced."""

    def run(result, *, validate):
        calls = []

        def evaluate(*args):
            calls.append(args)
            return result

        monkeypatch.setattr(
            ops, "_prepare_generated_kernel_build", lambda *args, **kwargs: None
        )
        monkeypatch.setattr(
            ops,
            "_load_validated_prepared_kernel",
            lambda prepared: types.SimpleNamespace(evaluate=evaluate),
        )
        validation_flag[0] = validate
        out, _ = execute_cin_via_loopir(copy_program(), (1, 2), operand())
        assert len(calls) == 1
        return out

    return run


# Child coordinates [1, 0, 1] under one parent: out of order and repeated.
MALFORMED = ([0, 3], [1, 0, 1], [1.0, 2.0, 2.0])


def test_a_well_formed_result_is_returned_as_the_kernel_wrote_it(
    run_with_native_result,
):
    out = run_with_native_result(
        native_result([0, 2], [0, 1], [2.0, 3.0]), validate=True
    )
    child_pos, child_crd = out._native_mode_indices()[1]
    assert child_pos.tolist() == [0, 2]
    assert child_crd.tolist() == [0, 1]
    assert out.values.tolist() == [2.0, 3.0]


def test_a_malformed_result_is_an_error_when_results_are_validated(
    run_with_native_result,
):
    with pytest.raises(TensorIndexError, match="strictly increasing"):
        run_with_native_result(native_result(*MALFORMED), validate=True)


def test_a_malformed_result_is_not_repaired_when_validation_is_off(
    run_with_native_result,
):
    out = run_with_native_result(native_result(*MALFORMED), validate=False)
    child_pos, child_crd = out._native_mode_indices()[1]
    assert child_pos.tolist() == [0, 3]
    assert child_crd.tolist() == [1, 0, 1]
    assert out.values.tolist() == [1.0, 2.0, 2.0]
