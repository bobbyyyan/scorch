# Scorch Open-Source README

## What This Is

A full rewrite of Scorch's README.md to prepare the project for public open-source release. Scorch is a compiler-based sparse tensor computation library for PyTorch, and the current README is outdated — it doesn't reflect the actual API (STensor, format notation, compiler pipeline) and uses stale examples. The new README targets both ML researchers and engineers.

## Core Value

The README must accurately represent Scorch's current capabilities and make it easy for new users to install, understand, and start using the library within minutes.

## Requirements

### Validated

- ✓ Scorch library exists with working compiler pipeline (CIN → LLIR → C++) — existing
- ✓ STensor abstraction with format notation (d, cs, o, s) — existing
- ✓ PyTorch drop-in compatibility via `import scorch as torch` — existing
- ✓ matmul, einsum, spmv operations — existing
- ✓ JIT C++ kernel compilation with caching — existing
- ✓ OpenMP parallelization — existing
- ✓ Example applications (GCN, sparse transformer, sparse autoencoder, kernels) — existing
- ✓ Benchmark infrastructure in bench/ — existing
- ✓ Conda-based development environment — existing
- ✓ CI via GitHub Actions — existing

### Active

- [ ] Full README rewrite with accurate, current API examples
- [ ] Installation section (conda + pip, requirements, platform notes)
- [ ] Quick start with real STensor/format notation examples
- [ ] Format system explanation (d, cs, o, s notation)
- [ ] PyTorch drop-in usage section
- [ ] Architecture overview (compiler pipeline diagram)
- [ ] Benchmarks section (how to run, not results)
- [ ] Examples section linking to existing example apps
- [ ] Contributing guidelines section
- [ ] MIT license badge and reference
- [ ] Academic references (TACO and related work)

### Out of Scope

- Writing new example applications — existing examples are sufficient
- Generating benchmark result data — document how to run only
- API reference docs — README is an entry point, not full docs
- Website or hosted documentation — just the README for now
- CI/CD changes — infrastructure is already in place

## Context

- Current README uses `torch.sparse_coo_tensor(...)` which doesn't reflect the actual STensor-based API
- The project has 4 example apps with their own READMEs (GCN, sparse transformer, sparse autoencoder, kernels)
- Benchmark scripts exist in `bench/` directory
- Project uses conda environment "scorch" with Python 3.11
- C++ extensions require OpenMP (platform-specific: Homebrew libomp on macOS, libgomp on Linux)
- Codebase map already exists at `.planning/codebase/`

## Constraints

- **Tone**: Developer-friendly with academic references where relevant (mixed tone)
- **Accuracy**: All code examples must reflect the actual current API — no outdated patterns
- **License**: MIT license
- **Audience**: Both ML researchers and ML engineers

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Full rewrite vs patch | Current README is too outdated to patch — stale examples, missing key features | — Pending |
| Benchmarks: script-only | User wants to document how to run, not include result data | — Pending |
| MIT license | User's choice for open-source release | — Pending |
| Mixed tone | Practical developer docs with academic citations where relevant | — Pending |

---
*Last updated: 2026-03-18 after initialization*
