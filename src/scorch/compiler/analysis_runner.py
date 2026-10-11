"""Common stateless runner for current immutable compiler side-table analyses.

Only normalized-CIN ownership/use/access analysis satisfies the canonical common
analysis contract today.  LLIR transform scans remain pass-local derived facts,
and mutable scheduler/iteration/lowering state remains outside this service.
Every request recomputes its result; this runner owns no cache, registry,
dependency graph, or preservation/invalidation state.
"""

from __future__ import annotations

from dataclasses import dataclass

from .cin import IndexStmt
from .cin_analysis import CINAnalysis, _compute_cin_analysis


@dataclass(frozen=True)
class AnalysisRunner:
    """Run explicit typed analyses without retaining inputs or results."""

    def analyze_cin(self, cin: IndexStmt) -> CINAnalysis:
        """Recompute the immutable typed CIN side tables for ``cin``."""

        return _compute_cin_analysis(cin)


COMMON_ANALYSIS_RUNNER = AnalysisRunner()
