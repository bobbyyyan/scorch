# Roadmap: Scorch Open-Source README

## Overview

This roadmap delivers a complete README rewrite for Scorch's public open-source release. The build order prioritizes API verification first (Quick Start), then wraps identity and installation around it, then layers conceptual depth (format system, drop-in compatibility, operations), architectural context, practical references (examples, benchmarks), and community sections. Each phase produces a verifiable section of the final README.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [x] **Phase 1: Quick Start and API Verification** - Write runnable STensor code examples and verify they work against the current API
- [x] **Phase 2: Project Identity** - Title, badges, and "What is Scorch" description
- [ ] **Phase 3: Installation Guide** - Prerequisites listing and installation instructions with platform-specific notes
- [ ] **Phase 4: Format System** - Explanation of d/cs/o/s notation with mapping table to familiar formats
- [ ] **Phase 5: PyTorch Drop-in Compatibility** - Demonstrate `import scorch as torch` usage with before/after comparison
- [ ] **Phase 6: Supported Operations** - List of supported operations (matmul, einsum, spmv, spmm) with descriptions
- [ ] **Phase 7: Architecture Overview** - Compiler pipeline diagram (CIN to LLIR to C++ to JIT execution) with Mermaid
- [ ] **Phase 8: Examples and Benchmarks** - Link to bench/*.py working examples and document how to run benchmarks
- [ ] **Phase 9: Community, Citation, and License** - Contributing guidelines, CGO 2026 citation BibTeX, and MIT license section

## Phase Details

### Phase 1: Quick Start and API Verification
**Goal**: Users can see a working code example that proves the STensor API is approachable and correct
**Depends on**: Nothing (first phase)
**Requirements**: INST-03
**Success Criteria** (what must be TRUE):
  1. A runnable Quick Start section exists showing STensor creation, format notation, and a matmul operation
  2. Every code snippet in the Quick Start has been executed in the scorch conda environment and produces correct output
  3. The Quick Start includes a note about JIT compilation ("first call compiles, subsequent calls use cached kernels")
**Plans:** 1 plan

Plans:
- [ ] 01-01-PLAN.md -- Write and verify Quick Start section with STensor API examples and captured output

### Phase 2: Project Identity
**Goal**: A reader landing on the repo instantly understands what Scorch is and trusts it is maintained
**Depends on**: Phase 1 (identity language must reflect verified API capabilities)
**Requirements**: IDENT-01, IDENT-02, IDENT-03
**Success Criteria** (what must be TRUE):
  1. The repo title has a one-line description that communicates "compiler-based sparse tensor library for PyTorch"
  2. A badges row displays CI status, MIT license, and Python 3.11 version (3-5 shields.io badges, flat style)
  3. A "What is Scorch" section (2-3 paragraphs) explains the compiler pipeline, format-driven sparsity, and PyTorch compatibility
**Plans:** 1 plan

Plans:
- [ ] 02-01-PLAN.md -- Write title, tagline, 6 badges, "What is Scorch?" prose, and MIT LICENSE file

### Phase 3: Installation Guide
**Goal**: A new user can go from zero to a working Scorch installation on either macOS or Linux
**Depends on**: Phase 1 (installation must be tested by running Quick Start examples)
**Requirements**: INST-01, INST-02
**Success Criteria** (what must be TRUE):
  1. A prerequisites section lists Python 3.11, PyTorch 2.0+, C++ compiler, OpenMP, and CMake
  2. Installation instructions cover conda environment creation and pip install
  3. Platform-specific OpenMP notes exist for macOS (Homebrew libomp, Apple Silicon vs Intel paths) and Linux (libgomp)
**Plans**: TBD

Plans:
- [ ] 03-01: TBD

### Phase 4: Format System
**Goal**: A user understands Scorch's d/cs/o/s format notation and can map it to familiar sparse formats
**Depends on**: Phase 1 (format examples must use verified API patterns)
**Requirements**: CONC-01
**Success Criteria** (what must be TRUE):
  1. A format system section includes a table mapping notation letters to familiar formats (d=Dense, cs=CSR/CSC, o=COO, s=Singleton)
  2. The section shows how format strings like "ds" or "dd" translate to storage layouts
  3. At least one code example demonstrates creating an STensor with an explicit format string
**Plans**: TBD

Plans:
- [ ] 04-01: TBD

### Phase 5: PyTorch Drop-in Compatibility
**Goal**: A PyTorch user sees exactly how to use Scorch as a drop-in replacement with minimal code changes
**Depends on**: Phase 1 (drop-in examples must use verified API)
**Requirements**: CONC-02
**Success Criteria** (what must be TRUE):
  1. A section demonstrates the `import scorch as torch` pattern
  2. A before/after code comparison shows standard PyTorch code alongside the Scorch equivalent
  3. The section clarifies which PyTorch operations are supported through the drop-in interface
**Plans**: TBD

Plans:
- [ ] 05-01: TBD

### Phase 6: Supported Operations
**Goal**: A user can quickly see which sparse operations Scorch supports
**Depends on**: Phase 1 (operation names must match verified API)
**Requirements**: CONC-03
**Success Criteria** (what must be TRUE):
  1. A supported operations section lists matmul, einsum, spmv, and spmm with brief descriptions
  2. Each operation includes its function signature or usage pattern
**Plans**: TBD

Plans:
- [ ] 06-01: TBD

### Phase 7: Architecture Overview
**Goal**: A technically curious user understands how Scorch's compiler pipeline transforms tensor expressions into efficient code
**Depends on**: Phase 2 (architecture section references concepts introduced in "What is Scorch")
**Requirements**: DPTH-01
**Success Criteria** (what must be TRUE):
  1. A Mermaid diagram renders on GitHub showing the compiler pipeline: CIN -> LLIR -> C++ -> JIT execution
  2. The diagram is accompanied by 3-5 sentences explaining each stage
  3. The section is concise enough to scan in under 60 seconds
**Plans**: TBD

Plans:
- [ ] 07-01: TBD

### Phase 8: Examples and Benchmarks
**Goal**: A user can find real-world usage examples and knows how to run benchmarks
**Depends on**: Phase 3 (examples require working installation)
**Requirements**: DPTH-02, DPTH-03
**Success Criteria** (what must be TRUE):
  1. An examples section links to bench/*.py scripts (GCN, sparse transformer, sparse autoencoder) as the current working examples
  2. A benchmarks section documents how to run the benchmark scripts with specific commands
  3. The section does NOT include benchmark result data (documents how to run only)
**Plans**: TBD

Plans:
- [ ] 08-01: TBD

### Phase 9: Community, Citation, and License
**Goal**: An interested contributor or researcher can find contribution guidelines, cite the project, and verify the license
**Depends on**: Nothing (standalone sections)
**Requirements**: COMM-01, COMM-02, COMM-03
**Success Criteria** (what must be TRUE):
  1. A contributing section provides brief guidelines or a pointer to CONTRIBUTING.md
  2. A citation section includes a BibTeX block for the CGO 2026 paper "Fast Autoscheduling for Sparse ML Frameworks" by Yan et al.
  3. A license section references the MIT license with a badge linking to the LICENSE file
**Plans**: TBD

Plans:
- [ ] 09-01: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7 -> 8 -> 9

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Quick Start and API Verification | 1/1 | Complete | 2026-03-19 |
| 2. Project Identity | 1/1 | Complete | 2026-03-22 |
| 3. Installation Guide | 0/? | Not started | - |
| 4. Format System | 0/? | Not started | - |
| 5. PyTorch Drop-in Compatibility | 0/? | Not started | - |
| 6. Supported Operations | 0/? | Not started | - |
| 7. Architecture Overview | 0/? | Not started | - |
| 8. Examples and Benchmarks | 0/? | Not started | - |
| 9. Community, Citation, and License | 0/? | Not started | - |
