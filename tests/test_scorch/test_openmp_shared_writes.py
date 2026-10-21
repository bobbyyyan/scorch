"""Positive and negative controls for ``openmp_shared_writes.shared_scalar_writes``.

The checker backs the assertion that a generated kernel writes no scalar shared by an
OpenMP parallel region (test_loop_invariant_factor_hoist_validity.py).  An assertion of
an empty list is only as good as the checker's ability to find a write, so each case
here is a small C++ text in the code generator's layout with the shared writes it must
report: the pre-0047 SDDMM shape and four other racing forms, which it must find, and
ten forms that are synchronised or private, which it must not report.
"""

from __future__ import annotations

from typing import List

import pytest

from tests.test_scorch.openmp_shared_writes import SharedWrite, shared_scalar_writes

RACING = [
    (
        "pre_0047_sddmm_counter",
        ["pSampled1"],
        """
  int64_t pSampled1 = 0;
  #pragma omp parallel for schedule(static)
  for (int64_t pMask0 = 0; pMask0 < pMask0_end; pMask0++) {
    float _accum = 0.0f;
    Sampled_values[pSampled1] = _accum;
    pSampled1++;
  }
""",
    ),
    (
        "post_increment_inside_an_index",
        ["pC1"],
        """
  int pC1 = 0;
  #pragma omp parallel for
  for (int i = 0; i < n; i++) {
    C1_crd[pC1++] = i;
  }
""",
    ),
    (
        "assignment_to_a_shared_scalar",
        ["last"],
        """
  int64_t last = 0;
  #pragma omp parallel for
  for (int64_t i = 0; i < n; i++) {
    last = i;
  }
""",
    ),
    (
        "inner_for_header_starts_a_shared_position",
        ["pA1"],
        """
  int pA1 = 0;
  #pragma omp parallel for
  for (int64_t i = 0; i < n; i++) {
    for (pA1 = A1_pos[i]; pA1 < A1_pos[i + 1]; pA1++) {
      x[pA1] = 0;
    }
  }
""",
    ),
    (
        "omp_for_inside_a_parallel_block",
        ["count"],
        """
  int64_t count = 0;
  #pragma omp parallel
  {
    #pragma omp for
    for (int64_t i = 0; i < n; i++) {
      count += 1;
    }
  }
""",
    ),
]

RACE_FREE = [
    (
        "counters_declared_inside_the_loop",
        """
  #pragma omp parallel for
  for (int64_t batch = 0; batch < n; batch++) {
    int _cnt1 = 0;
    int64_t _cnt2 = 0;
    _cnt1++;
    _cnt2 += 2;
  }
""",
    ),
    (
        "iteration_variable_declared_outside",
        """
  int64_t i;
  #pragma omp parallel for
  for (i = 0; i < n; i++) {
    y[i] = 0;
  }
""",
    ),
    (
        "collapse_makes_both_iteration_variables_private",
        """
  int i, j;
  #pragma omp parallel for collapse(2)
  for (i = 0; i < n; i++) {
    for (j = 0; j < m; j++) {
      y[i * m + j] = 0;
    }
  }
""",
    ),
    (
        "reduction_clause",
        """
  double total = 0;
  #pragma omp parallel for reduction(+:total)
  for (int64_t i = 0; i < n; i++) {
    total += x[i];
  }
""",
    ),
    (
        "private_and_firstprivate_clauses",
        """
  int64_t a = 0, b = 0;
  #pragma omp parallel for private(a) firstprivate(b)
  for (int64_t i = 0; i < n; i++) {
    a = i;
    b++;
  }
""",
    ),
    (
        "atomic_statement_and_atomic_capture_block",
        """
  int64_t next = 0;
  #pragma omp parallel
  {
    int64_t start;
    #pragma omp atomic capture
    { start = next; next += 64; }
    #pragma omp atomic
    next++;
  }
""",
    ),
    (
        "critical_block",
        """
  int64_t total = 0;
  #pragma omp parallel for
  for (int64_t i = 0; i < n; i++) {
    #pragma omp critical
    {
      total += i;
    }
  }
""",
    ),
    (
        "element_and_member_writes_are_not_scalars",
        """
  #pragma omp parallel for
  for (int64_t i = 0; i < n; i++) {
    count[row[i]]++;
    s.x++;
    p->y += 1;
    out[i] = 2;
  }
""",
    ),
    (
        "restrict_pointer_and_simd_loop_inside",
        """
  #pragma omp parallel for
  for (int64_t pKey0 = 0; pKey0 < n; pKey0++) {
    const float* __restrict__ _Key_val_ptr = &Key_val[pKey0 * 4];
    float _accum = 0.0f;
    #pragma omp simd
    for (int64_t q = 0; q < 4; q++) {
      // q++ in a comment, and _accum
      _accum += _Key_val_ptr[q];
    }
    _accum *= 2;
  }
""",
    ),
    (
        "writes_outside_every_region",
        """
  int pC0 = 0;
  for (int i = 0; i < n; i++) {
    pC0++;
  }
  #pragma omp parallel for
  for (int i = 0; i < n; i++) {
    y[i] = 1;
  }
  pC0 = 0;
""",
    ),
]


def _names(writes: List[SharedWrite]) -> List[str]:
    return sorted({write.name for write in writes})


@pytest.mark.parametrize(
    ("expected", "cpp"),
    [case[1:] for case in RACING],
    ids=[case[0] for case in RACING],
)
def test_a_racing_write_is_reported(expected: List[str], cpp: str) -> None:
    assert _names(shared_scalar_writes(cpp)) == expected


@pytest.mark.parametrize(
    "cpp", [case[1] for case in RACE_FREE], ids=[case[0] for case in RACE_FREE]
)
def test_a_private_or_synchronised_write_is_not_reported(cpp: str) -> None:
    assert shared_scalar_writes(cpp) == []


def test_the_report_names_the_line_and_its_text() -> None:
    cpp = RACING[0][2]
    (write,) = shared_scalar_writes(cpp)
    assert write == SharedWrite(
        cpp.split("\n").index("    pSampled1++;") + 1, "pSampled1", "pSampled1++;"
    )
