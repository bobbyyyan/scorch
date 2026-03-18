# External Integrations

**Analysis Date:** 2026-03-18

## APIs & External Services

**None detected** - Scorch is a self-contained sparse ML library with no external API dependencies.

**Graph Data Sources (Optional):**
- Open Graph Benchmark (OGB) - Used in `examples/gcn/requirements.txt` for dataset loading
  - SDK/Client: `ogb` package
  - Purpose: Loads benchmark graph datasets for testing sparse operations on real-world graphs

## Data Storage

**Databases:**
- None - No persistent database integrations

**File Storage:**
- Local filesystem only - Tensors and sparse structures stored as PyTorch tensor files or custom binary formats
- Benchmark results: CSV files in `/scratch/bobbyy/scorch/bench_results/`

**Caching:**
- None - No distributed caching layer detected

## Authentication & Identity

**Auth Provider:**
- Not applicable - No user authentication required; library-only package

## Monitoring & Observability

**Error Tracking:**
- None detected

**Logs:**
- Console logging only (Python `logging` module not explicitly configured; uses print statements and pytest capture)
- Pytest verbose mode available for detailed test output

## CI/CD & Deployment

**Hosting:**
- GitHub (source control and CI/CD via GitHub Actions)

**CI Pipeline:**
- GitHub Actions (`/.github/workflows/pytest.yml`)
  - Runs on: Ubuntu Latest
  - Triggers: Pull requests to main branch, manual dispatch
  - Build Steps:
    1. Checkout code
    2. Setup Python 3.11.7
    3. Install system dependencies (ninja-build)
    4. Install Pipenv and dependencies (CPU PyTorch 2.0.1)
    5. Run pytest via Pipenv script
    6. Upload artifacts (test results XML)

**Deployment:**
- PyPI distribution (implicit; `setup.py` configured for pip installable package)
- No automatic deployment pipeline; releases are manual

## Environment Configuration

**Required env vars:**
- None detected - Package operates without environment variable configuration

**Secrets location:**
- Not applicable - No API keys, credentials, or secrets required

**Build-time Config:**
- `TORCH_INCLUDE_DIR` - Resolved dynamically from torch installation in CMakeLists.txt
- `CC`, `CXX` - Compiler selection (enforced on macOS in `setup.sh`)

## Webhooks & Callbacks

**Incoming:**
- None detected

**Outgoing:**
- None detected

## Dataset Loading

**Graph Benchmark Data:**
- Open Graph Benchmark (OGB): Used via `ogb` package in example workloads
- PyTorch Geometric Data API: Integrated via `torch_geometric` for graph loading
- DGL DataLoader: Available via `dgl` for alternative graph representations

**Example Datasets:**
- Node classification, graph classification, and link prediction benchmarks
- Loaded on-demand during example execution in `examples/gcn/`

## No External Service Dependencies

The codebase contains **zero runtime dependencies on external APIs, databases, or cloud services**. It is a self-contained library that:

1. Takes PyTorch tensors as input
2. Performs sparse computation locally
3. Returns PyTorch tensors as output

This makes Scorch suitable for:
- Offline sparse ML workloads
- Embedded sparse tensor computation
- Privacy-sensitive applications (no data leaves the user's machine)

---

*Integration audit: 2026-03-18*
