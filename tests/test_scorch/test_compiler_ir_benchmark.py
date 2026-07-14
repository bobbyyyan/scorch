import argparse
import json
from pathlib import Path
from typing import Dict, List, Union

import pytest

from tools import benchmark_compiler_ir

JsonValue = Union[str, float, List[Dict[str, object]], Dict[str, object]]


def _latency_result(p50_ms: float, p95_ms: float) -> Dict[str, JsonValue]:
    return {
        "schema_version": benchmark_compiler_ir.SCHEMA_VERSION,
        "kind": "compiler-ir-phase0-latency",
        "corpus_version": benchmark_compiler_ir.LATENCY_CORPUS_VERSION,
        "metadata": {"hostname": "audit-host", "machine": "audit-machine"},
        "configuration": {"warmup": 1, "samples": 2},
        "cases": [
            {
                "name": "small_dense",
                "p50_ms": p50_ms,
                "p95_ms": p95_ms,
            }
        ],
    }


def _write_result(path: Path, result: Dict[str, JsonValue]) -> None:
    path.write_text(json.dumps(result))


@pytest.mark.parametrize(
    ("candidate_p50", "candidate_p95", "expected_status", "expected_exit"),
    (
        (1.10, 1.10, "TARGET", 0),
        (1.11, 1.10, "INVESTIGATE", 1),
        (1.10, 1.11, "INVESTIGATE", 1),
    ),
)
def test_latency_comparison_treats_target_crossing_as_investigation(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    candidate_p50: float,
    candidate_p95: float,
    expected_status: str,
    expected_exit: int,
) -> None:
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    _write_result(baseline, _latency_result(1.0, 1.0))
    _write_result(candidate, _latency_result(candidate_p50, candidate_p95))

    exit_code = benchmark_compiler_ir.run_compare_latency(
        argparse.Namespace(baseline=baseline, candidate=candidate)
    )

    output = capsys.readouterr().out
    assert exit_code == expected_exit
    assert expected_status in output
    assert "not an automatic rejection" in output
    assert "FAIL" not in output
