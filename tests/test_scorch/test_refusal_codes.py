"""Every refusal Scorch raises names its defect with a code, read by ``defect_code``.

dennis's inventory (sapling job 82999) found 1,002 raise sites that reach a public entry
point with no code: built-in ``TypeError``, ``ValueError``, ``NotImplementedError`` and
``assert`` failures, and Scorch's own classes raised with only a message.  Each now
passes ``code="..."``: one code per cause for a refusal a caller can meet (invalid input,
an unsupported request, a failed build) and one per family of checks for an internal
invariant (classification in /scratch/bobby/emb0038/classify on sapling).  A built-in
is raised as its Scorch variant, still that built-in, so an ``except`` clause written
for it catches it as before.  ``test_every_refusal_has_a_code.py`` holds the line
statically; this file checks the classes, the reader and a sample of public refusals.
"""

import dataclasses
import pickle
from types import SimpleNamespace

import pytest
import torch

import scorch
from scorch import STensor
from scorch.compiler.diagnostics import (
    CodegenError,
    CompileOptionsDiagnostic,
    CompilerInvariantError,
    InvalidSchedule,
    UnsupportedFeature,
    VerificationError,
)
from scorch.exceptions import (
    CompileSpecError,
    EinsumOperandCountError,
    EinsumResultLabelError,
    OperandRankMismatchError,
    ScorchAssertionError,
    ScorchError,
    ScorchFrozenInstanceError,
    ScorchImportError,
    ScorchKeyError,
    ScorchLookupError,
    ScorchNotImplementedError,
    ScorchOSError,
    ScorchRuntimeError,
    ScorchTypeError,
    ScorchValueError,
    TensorStorageError,
    defect_code,
)
from tests.test_scorch.level_array_support import stored

VARIANTS = [
    (ScorchTypeError, TypeError),
    (ScorchValueError, ValueError),
    (ScorchAssertionError, AssertionError),
    (ScorchNotImplementedError, NotImplementedError),
    (ScorchRuntimeError, RuntimeError),
    (ScorchLookupError, LookupError),
    (ScorchKeyError, KeyError),
    (ScorchImportError, ImportError),
    (ScorchOSError, OSError),
    (ScorchFrozenInstanceError, dataclasses.FrozenInstanceError),
]
COMPILER_ERRORS = [
    CompilerInvariantError,
    CodegenError,
    UnsupportedFeature,
    InvalidSchedule,
    VerificationError,
]


@pytest.mark.parametrize(
    ("variant", "builtin"), VARIANTS, ids=[v.__name__ for v, _ in VARIANTS]
)
def test_each_variant_is_its_builtin_and_carries_the_code_of_its_raise(
    variant, builtin
):
    with pytest.raises(builtin) as raised:
        raise variant("m", code="a_code")
    assert isinstance(raised.value, ScorchError)
    assert raised.value.code == "a_code"
    assert defect_code(raised.value) == "a_code"
    # Printed as the built-in prints it: a KeyError shows its key quoted.
    assert str(raised.value) == str(builtin("m"))
    copy = pickle.loads(pickle.dumps(raised.value))
    assert type(copy) is variant
    assert copy.code == "a_code"


def test_an_error_with_no_code_reads_as_none():
    assert ScorchError("m").code is None
    assert defect_code(ScorchError("m")) is None
    assert defect_code(ValueError("m")) is None
    assert defect_code(None) is None


class _OneCause(TensorStorageError):
    code = "one_cause"


def test_a_class_with_one_cause_codes_every_raise():
    assert _OneCause("m").code == "one_cause"
    assert defect_code(_OneCause("m")) == "one_cause"
    assert _OneCause("m", code="other").code == "other"
    assert TensorStorageError("m").code is None
    assert TensorStorageError("m", code="c").code == "c"


@pytest.mark.parametrize("error_type", COMPILER_ERRORS, ids=lambda t: t.__name__)
def test_a_compiler_error_carries_the_code_of_its_raise(error_type):
    error = error_type("m", code="c")
    assert error.code == "c"
    assert defect_code(error) == "c"
    assert error_type("m").code is None
    assert defect_code(error_type("m")) is None


@pytest.mark.parametrize(
    "error_type", [UnsupportedFeature, InvalidSchedule, VerificationError]
)
def test_diagnostics_give_the_code_when_the_raise_names_none(error_type):
    diagnostic = CompileOptionsDiagnostic(
        code="from_diagnostic", field="f", message="m"
    )
    assert defect_code(error_type("m", diagnostics=(diagnostic,))) == (
        "from_diagnostic"
    )
    assert defect_code(error_type("m", diagnostics=(diagnostic,), code="own")) == (
        "own"
    )


def test_defect_code_reads_the_error_a_refusal_was_raised_from():
    with pytest.raises(InvalidSchedule) as raised:
        try:
            raise ScorchValueError("inner", code="inner_code")
        except ValueError as exc:
            raise InvalidSchedule(str(exc)) from exc
    assert defect_code(raised.value) == "inner_code"
    with pytest.raises(InvalidSchedule) as raised:
        try:
            raise ScorchValueError("inner", code="inner_code")
        except ValueError as exc:
            raise InvalidSchedule(str(exc), code="outer_code") from exc
    assert defect_code(raised.value) == "outer_code"


def test_defect_code_reads_a_new_compiler_defect():
    error = RuntimeError("m")
    error.defect = SimpleNamespace(code="from_defect")  # type: ignore[attr-defined]
    assert defect_code(error) == "from_defect"


def test_a_cycle_of_causes_ends():
    first, second = ValueError("a"), ValueError("b")
    first.__cause__, second.__cause__ = second, first
    assert defect_code(first) is None


# A sample of the public refusals the inventory found without a code, one per kind a
# caller can meet; none builds a kernel.
def _matrix(shape=(2, 3)):
    dense = torch.arange(1.0, 1.0 + shape[0] * shape[1], dtype=torch.float64)
    return stored(dense.reshape(shape), "ds", "A")


@pytest.mark.parametrize(
    ("call", "error_type", "code"),
    [
        (
            lambda: scorch.einsum("ij", _matrix()),
            CompileSpecError,
            "einsum_implicit_output_unsupported",
        ),
        (
            lambda: scorch.einsum("ij,jk->ik", _matrix()),
            EinsumOperandCountError,
            "einsum_operand_count_mismatch",
        ),
        (
            lambda: scorch.einsum("ij->k", _matrix()),
            EinsumResultLabelError,
            "einsum_result_label_unknown",
        ),
        (
            lambda: scorch.einsum("i,jk->ik", _matrix(), _matrix((3, 4))),
            OperandRankMismatchError,
            "einsum_operand_rank_mismatch",
        ),
        (
            lambda: scorch.einsum("ij,jk->ik", _matrix(), _matrix((4, 5))),
            CompileSpecError,
            "einsum_index_extent_mismatch",
        ),
        (
            lambda: STensor.from_torch(torch.ones(3, 3)).to_sparse("so"),
            TensorStorageError,
            "unsupported_mixed_coordinate_hierarchy",
        ),
    ],
    ids=["implicit-output", "operand-count", "unknown-label", "rank", "extent", "so-layout"],
)
def test_public_refusals_name_their_cause(call, error_type, code):
    with pytest.raises(error_type) as raised:
        call()
    assert defect_code(raised.value) == code
