from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = REPO_ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import tune_scheduler


def test_normalize_templates_all_alias():
    templates = tune_scheduler._normalize_templates(["all"])
    assert templates == tune_scheduler.SUPPORTED_TEMPLATES


def test_normalize_templates_accepts_commas_and_deduplicates():
    templates = tune_scheduler._normalize_templates(["spmm,spgemm", "spmm"])
    assert templates == ("spmm", "spgemm")


def test_normalize_templates_rejects_unknown_template():
    with pytest.raises(ValueError, match="Unknown templates"):
        tune_scheduler._normalize_templates(["spmm", "unknown_template"])


def test_generate_workload_specs_spmm_spgemm():
    specs = tune_scheduler._generate_workload_specs(
        count=16,
        min_n=16,
        max_n=24,
        seed=7,
        templates=("spmm", "spgemm"),
    )

    templates_seen = {spec.template for spec in specs}
    assert templates_seen == {"spmm", "spgemm"}

    for spec in specs:
        inputs = {inp.name: inp for inp in spec.inputs}
        assert set(inputs) == {"A", "B"}
        if spec.template == "spmm":
            assert inputs["B"].fmt == "dd"
        else:
            assert inputs["B"].fmt in {"ds", "oo"}


def test_required_workload_specs_include_canonical_spmm_spgemm():
    specs = tune_scheduler._required_workload_specs(
        min_n=64,
        max_n=192,
        seed=0,
        templates=("spmm", "spgemm"),
    )

    names = {spec.name for spec in specs}
    assert "required_spmm_csr_dense_n128_0" in names
    assert "required_spmm_csr_dense_n128_1" in names
    assert "required_spgemm_csr_csr_n128_0" in names
    assert "required_spgemm_coord_csr_n128_1" in names


def test_normalize_spec_for_scoring_normalizes_non_default_mode_orders():
    spec = tune_scheduler.WorkloadSpec(
        name="test_spmm",
        template="spmm",
        n=128,
        fmt_out="dd",
        output_mode_order=(1, 0),
        inputs=(
            tune_scheduler.InputTensorSpec("A", 2, "ds", 0.9, (1, 0)),
            tune_scheduler.InputTensorSpec("B", 2, "dd", 0.0, (1, 0)),
        ),
    )
    normalized = tune_scheduler._normalize_spec_for_scoring(spec)
    assert normalized.output_mode_order == (0, 1)
    for inp in normalized.inputs:
        assert inp.mode_order == (0, 1)


def test_normalize_spec_for_scoring_preserves_default_mode_orders():
    spec = tune_scheduler.WorkloadSpec(
        name="test_spmm",
        template="spmm",
        n=128,
        fmt_out="dd",
        output_mode_order=(0, 1),
        inputs=(
            tune_scheduler.InputTensorSpec("A", 2, "ds", 0.9, (0, 1)),
            tune_scheduler.InputTensorSpec("B", 2, "dd", 0.0, (0, 1)),
        ),
    )
    normalized = tune_scheduler._normalize_spec_for_scoring(spec)
    assert normalized is spec


def test_normalize_spec_for_scoring_skips_all_dense():
    spec = tune_scheduler.WorkloadSpec(
        name="test_dense",
        template="spmm",
        n=128,
        fmt_out="dd",
        output_mode_order=(1, 0),
        inputs=(
            tune_scheduler.InputTensorSpec("A", 2, "dd", 0.0, (1, 0)),
            tune_scheduler.InputTensorSpec("B", 2, "dd", 0.0, (1, 0)),
        ),
    )
    normalized = tune_scheduler._normalize_spec_for_scoring(spec)
    assert normalized is spec
