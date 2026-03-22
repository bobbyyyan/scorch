# Requirements: Scorch Open-Source README

**Defined:** 2026-03-18
**Core Value:** The README must accurately represent Scorch's current capabilities and make it easy for new users to install, understand, and start using the library within minutes.

## v1 Requirements

Requirements for the open-source README rewrite. Each maps to roadmap phases.

### Identity

- [x] **IDENT-01**: Project title with one-line description conveying compiler-based sparse tensor library for PyTorch
- [x] **IDENT-02**: Badges row with 3-5 shields.io badges (CI status, MIT license, Python 3.11)
- [x] **IDENT-03**: "What is Scorch" description (2-3 paragraphs covering compiler pipeline, format-driven sparsity, PyTorch compatibility)

### Installation

- [ ] **INST-01**: Prerequisites section listing Python 3.11, PyTorch 2.0+, C++ compiler, OpenMP, CMake
- [ ] **INST-02**: Installation instructions with conda env creation, pip install, platform-specific OpenMP notes (macOS Homebrew libomp vs Linux libgomp)
- [x] **INST-03**: Quick start with real STensor API examples (create tensor, run matmul, show format notation)

### Concepts

- [ ] **CONC-01**: Format system explanation with table mapping d/cs/o/s notation to familiar formats (Dense, CSR/CSC, COO, Singleton)
- [ ] **CONC-02**: PyTorch drop-in compatibility section demonstrating `import scorch as torch` usage with before/after comparison
- [ ] **CONC-03**: Supported operations list (matmul, einsum, spmv, spmm) with brief descriptions

### Depth

- [ ] **DPTH-01**: Architecture overview with Mermaid diagram of compiler pipeline (CIN -> LLIR -> C++ -> JIT execution)
- [ ] **DPTH-02**: Examples section linking to bench/*.py scripts (GCN, sparse transformer, sparse autoencoder) as current working examples
- [ ] **DPTH-03**: Benchmarks section documenting how to run benchmark scripts

### Community

- [ ] **COMM-01**: Contributing section with brief guidelines or pointer to CONTRIBUTING.md
- [ ] **COMM-02**: MIT License section with badge reference to LICENSE file
- [ ] **COMM-03**: Academic citation with CGO 2026 paper BibTeX (Yan et al., "Fast Autoscheduling for Sparse ML Frameworks")

## v2 Requirements

Deferred to future updates. Tracked but not in current roadmap.

### Positioning

- **DPTH-04**: "Why Scorch" value proposition section comparing approach to alternatives
- **DPTH-05**: Gotchas / known limitations (JIT warmup, platform-specific OpenMP quirks)

### Navigation

- **NAV-01**: Table of contents with anchor links (add when README exceeds ~200 lines)

## Out of Scope

| Feature | Reason |
|---------|--------|
| Full API reference in README | Belongs in docstrings or separate docs site; goes stale |
| Benchmark result tables/charts | Hardware-dependent, go stale; documenting how to run is sufficient |
| Animated GIFs / demo videos | Computation library, not UI; no meaningful visual to animate |
| Competitor comparison tables | Creates adversarial tone; let users draw own comparisons |
| Full tutorial content inline | Duplicates example READMEs; makes README 2000+ lines |
| FAQ section | Too early; FAQ grows organically from real user questions |
| Logo/branding | Not needed for v1 launch |
| Documentation site | README is sufficient for initial release |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| INST-03 | Phase 1 | Complete |
| IDENT-01 | Phase 2 | Complete |
| IDENT-02 | Phase 2 | Complete |
| IDENT-03 | Phase 2 | Complete |
| INST-01 | Phase 3 | Pending |
| INST-02 | Phase 3 | Pending |
| CONC-01 | Phase 4 | Pending |
| CONC-02 | Phase 5 | Pending |
| CONC-03 | Phase 6 | Pending |
| DPTH-01 | Phase 7 | Pending |
| DPTH-02 | Phase 8 | Pending |
| DPTH-03 | Phase 8 | Pending |
| COMM-01 | Phase 9 | Pending |
| COMM-02 | Phase 9 | Pending |
| COMM-03 | Phase 9 | Pending |

**Coverage:**
- v1 requirements: 15 total
- Mapped to phases: 15
- Unmapped: 0

---
*Requirements defined: 2026-03-18*
*Last updated: 2026-03-18 after roadmap creation*
