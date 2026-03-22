---
phase: 02-project-identity
plan: 01
subsystem: docs
tags: [readme, badges, shields.io, license, mit, markdown]

# Dependency graph
requires:
  - phase: 01-quick-start-and-api-verification
    provides: README.md with Quick Start section (preserved above)
provides:
  - README identity header with title, tagline, 6 shields.io badges
  - "What is Scorch?" prose section with compiler pipeline, format notation, PyTorch compatibility
  - MIT LICENSE file at repo root
affects: [03-installation, 04-format-system, 07-architecture, 09-community]

# Tech tracking
tech-stack:
  added: [shields.io badges]
  patterns: [flat-style badges, left-aligned markdown, prose-only descriptions]

key-files:
  created: [LICENSE]
  modified: [README.md]

key-decisions:
  - "Created MIT LICENSE file in Phase 2 to prevent license badge 404 (rather than deferring to Phase 9)"
  - "Used fredrikbk.com fallback URL for CGO 2026 paper badge since DOI is not yet assigned"
  - "Used en-dash (--) for speedup range in prose for typographic consistency"

patterns-established:
  - "Left-aligned standard markdown only, no center HTML or horizontal rules between sections"
  - "Prose-only descriptions in identity sections, no bullet lists"
  - "All technical claims in README traced to ARCHITECTURE.md or verified source code"

requirements-completed: [IDENT-01, IDENT-02, IDENT-03]

# Metrics
duration: 2min
completed: 2026-03-22
---

# Phase 2 Plan 1: Project Identity Summary

**README identity header with flame-emoji title, 6 flat shields.io badges (CI, MIT, Python, PyTorch, paper, platform), 3-paragraph "What is Scorch?" prose, and MIT LICENSE file**

## Performance

- **Duration:** 2 min
- **Started:** 2026-03-22T16:55:42Z
- **Completed:** 2026-03-22T16:57:52Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments
- README title with flame emoji and one-line tagline ("A compiler-based sparse tensor library for PyTorch")
- 6 shields.io badges with flat style: CI status, MIT license, Python 3.11, PyTorch 2.0+, CGO 2026 paper, Linux/macOS
- Three-paragraph "What is Scorch?" section covering compiler pipeline (CIN/LLIR/C++), format notation (d/cs/o/s with CSR/COO examples), and PyTorch compatibility with JIT caching and CGO 2026 performance results
- MIT LICENSE file at repo root so license badge link resolves

## Task Commits

Each task was committed atomically:

1. **Task 1: Write title, tagline, badges, and create LICENSE file** - `a062c1f` (feat)
2. **Task 2: Write "What is Scorch?" prose section** - `ff2846f` (feat)

## Files Created/Modified
- `LICENSE` - MIT license file with Bobby Yan copyright
- `README.md` - Added identity header (title, tagline, badges, What is Scorch) above existing Quick Start

## Decisions Made
- Created MIT LICENSE file in Phase 2 rather than deferring to Phase 9 (COMM-02) to prevent a broken license badge link
- Used `https://fredrikbk.com/cgo26scorch.html` as paper badge URL since DOI is not yet assigned for the CGO 2026 paper
- Used en-dash (--) for the 1.05--5.80x speedup range for typographic consistency

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- README now has complete identity section (title, tagline, badges, What is Scorch?, Quick Start)
- Ready for Phase 3 (Installation) to add installation instructions below Quick Start
- Paper badge URL should be updated to DOI when assigned

## Self-Check: PASSED

- README.md: FOUND
- LICENSE: FOUND
- SUMMARY.md: FOUND
- Commit a062c1f (Task 1): FOUND
- Commit ff2846f (Task 2): FOUND

---
*Phase: 02-project-identity*
*Completed: 2026-03-22*
