# Codebase Concerns

**Analysis Date:** 2026-03-18

## Tech Debt

**Incomplete Implementation API Methods:**
- Issue: Multiple core `STensor` methods are declared but not implemented, returning `NotImplementedError`
- Files: `src/scorch/stensor.py`
- Methods affected: `insert()`, `validate()`, `to()`, `clone()` (all raise NotImplementedError at lines 78, 145, 150, 159)
- Impact: Users cannot migrate tensors between devices, clone tensors, or validate tensor state. These are fundamental operations expected from a tensor API
- Fix approach: Implement these methods or clearly document as "not yet supported" with timeline

**Unimplemented Format Operations:**
- Issue: Format conversion methods are incomplete
- Files: `src/scorch/format.py` lines 81, 91
- Methods: `to_dense()`, fill value extension
- Impact: Limited format conversion capabilities restrict workflow flexibility
- Fix approach: Complete format conversion logic or identify blocking constraints

**Multiple TODO Comments in Compiler Pipeline:**
- Issue: 29 TODO/FIXME comments scattered across compiler codebase indicating incomplete features and optimization opportunities
- Files: `src/scorch/compiler/iterator.py:230`, `iter_lattice.py:796,829,876,1461`, `llir.py:9`, `scheduler.py:1316,1419`, `cin_lowerer.py:1545,1997,2132`, `format.py:81,91`, `ops.py:630`, `stensor.py:49,78,144,149,158,172,175,525,663`
- Examples:
  - `iter_lattice.py:1461`: "TODO: generate workspace for the index vars below"
  - `cin_lowerer.py:1545`: "TODO: need to handle assembly of workspace with {level_type} level"
  - `ops.py:630`: "TODO: unless we are dealing with block tensors"
  - `stensor.py:172`: "TODO: support broadcasting"
- Impact: These indicate incomplete code paths that may fail silently or produce incorrect results in edge cases
- Fix approach: Create a tracking issue for each category; prioritize by severity (crash vs. incorrect result vs. missing feature)

## Known Bugs

**3D Sparse Output Crash (Recently Fixed):**
- Issue: Compilation crash when generating code for 3D sparse output tensors
- Files: `src/scorch/compiler/cin_lowerer.py`
- Root cause: Function `_should_parallelize_compressed_where` hardcodes level-1 array names and doesn't handle inner consumer/assembly code for additional compressed levels beyond 2-level tensors
- Status: Fixed in commit 0390a04 by guarding transform to only apply to 2-level (ds) output tensors
- Risk: Similar hardcoding patterns may exist elsewhere in cin_lowerer for other tensor dimensionalities
- Fix approach: Audit cin_lowerer for other hardcoded level references (search for `{name}1_` patterns)

**Integer Overflow in Dense Output (Recently Fixed):**
- Issue: int32 overflow when generating code for large dense outputs (e.g., 50k × 50k matrices where rows*cols > 2^31)
- Files: `src/scorch/compiler/cin.py`, `src/scorch/compiler/cin_lowerer.py`
- Root cause: DataType.INT (32-bit) used for computational variables in index calculations. Fixed in commit 0390a04
- Status: Resolved by replacing all DataType.INT with DataType.INT64 for computational integer variables
- Recommendation: Ensure all generated C++ integer variables use int64_t for safety

**Workspace Assembly Gaps (Partially Fixed):**
- Issue: Comment at cin_lowerer.py:1545 indicates incomplete support for workspace assembly with certain level types
- Files: `src/scorch/compiler/cin_lowerer.py:1545`
- Message: "TODO: need to handle assembly of workspace with {level_type} level"
- Impact: May fail to generate correct code for operations with unsupported workspace/level combinations
- Status: Partially addressed; needs complete audit of which level_type combinations are unsupported

## Security Considerations

**Memory Management - malloc/free in Generated Code:**
- Risk: Generated C++ code uses malloc/free for tensor allocation (cin_lowerer.py lines 72-88, 114-128)
- Files: `src/scorch/compiler/cin_lowerer.py`
- Current approach: Direct malloc/free without RAII wrappers in some code paths
- Status: Recent refactor (commit c282fda) added custom deleter handling with lambda functions for known-nnz case
- Recommendation: Ensure all malloc'd memory has associated deletion (check all code paths in emit_value_array_init and similar)

**Global Cache State (Thread-Safety):**
- Risk: Global mutable caches without synchronization
- Files: `src/scorch/ops.py` lines 30-31, `src/scorch/utils.py:29`
- Caches: `_kernel_cache`, `_einsum_dispatch_cache`, `_so_cache`
- Issue: Multiple Python threads or async code could race on cache updates
- Current mitigation: Python GIL provides some protection, but not guaranteed for C++ extensions
- Recommendation: Add lock-based synchronization if multi-threaded usage is expected

**No Input Validation on Tensor Dimensions:**
- Risk: Operators don't validate tensor shape compatibility before compilation
- Files: `src/scorch/ops.py` (einsum, matmul operations)
- Examples: No checks that matrix dimensions match before matmul or that index strings match tensor orders
- Impact: Incorrect dimensions could produce cryptic C++ compile errors or wrong results
- Fix approach: Add pre-compilation shape validation with clear error messages

## Performance Bottlenecks

**O(capacity) Workspace Clear (Partially Fixed):**
- Problem: Original workspace clearing required iterating over full capacity, not actual nnz
- Files: `src/scorch/compiler/cin_lowerer.py` (workspace assembly)
- Status: Commit 5fd33c5 "Add linked-list workspace and fix O(capacity) clear bottleneck" addressed this
- Remaining risk: Unknown if all workspace paths use optimized clear (some may still use old approach)
- Recommendation: Audit workspace initialization code to verify all use nnz-based clearing

**Module Caching Effectiveness:**
- Problem: Cache keys based only on format strings (ops.py:156), not on all parameters affecting code generation
- Files: `src/scorch/ops.py` lines 155-217
- Issue: Multiple different operations might share same format signature but need different compiled kernels
- Current approach: Two-level cache (module cache + dispatch cache) helps but may still have misses
- Impact: Some operations recompile unnecessarily, affecting first-call latency
- Recommendation: Analyze cache hit rates and expand key to include mode_order and other generation parameters

**Compiler Complexity - Large Files:**
- Problem: Core compiler modules are very large, making optimization and debugging difficult
- Files:
  - `src/scorch/compiler/cin_lowerer.py` - 3765 lines (159KB)
  - `src/scorch/compiler/iter_lattice.py` - 1527 lines (63KB)
  - `src/scorch/compiler/scheduler.py` - 1483 lines (52KB)
- Impact: Difficult to understand dataflow, identify optimization opportunities, or fix bugs
- Recommendation: Consider refactoring into smaller modules (e.g., separate workspace handling, loop generation, etc.)

**cvector vs malloc Trade-off (Partially Optimized):**
- Problem: Dynamic array initialization strategy affects performance
- Files: `src/scorch/compiler/cin_lowerer.py` (emit_value_array_init, emit_level_indices_init)
- Status: Commit c282fda replaced cvector with raw malloc for known-nnz sparse outputs
- Trade-off: cvector has bounds checking overhead but automatic cleanup; malloc is faster but manual cleanup required
- Recommendation: Profile both paths; consider malloc-only approach if cleanup can be guaranteed

## Fragile Areas

**Iteration Lattice - Complex Graph Construction:**
- Files: `src/scorch/compiler/iter_lattice.py`
- Why fragile:
  - Complex lattice point dependency graph construction (1527 lines)
  - Multiple TODO comments about correctness (lines 796, 829, 876, 1461)
  - No formal verification of lattice properties
  - Index variable assignment depends on topological correctness
- Safe modification: Changes to lattice point merging or sparse level handling require full test suite execution
- Test coverage: See TESTING.md for current test patterns
- Recommendation: Add assertions to verify lattice invariants (e.g., no cycles, proper parent-child relationships)

**Scheduler - Loop Order Selection:**
- Files: `src/scorch/compiler/scheduler.py`
- Why fragile:
  - Cost model-based decisions without complete documentation
  - Previous bugs in cost model (commit 0285d7b: "Fix scheduler cost model bugs")
  - TODO at line 1316 and 1419 indicate uncertain conditions
  - Loop order affects generated code structure significantly
- Safe modification: Changes require extensive benchmarking across test matrices
- Risk: Suboptimal loop orders silently produce correct but slow code
- Recommendation: Add loop order reasoning to generated code as comments; log cost model decisions in verbose mode

**Index Variable Calculations - Mode Order Dependent:**
- Files: `src/scorch/compiler/cin.py` lines 357-370, 804-821
- Why fragile:
  - Dense-mode-to-linear-offset calculations depend on correct mode_order
  - Recently fixed int32→int64 overflow (0390a04)
  - No validation that mode_order matches tensor format before code generation
- Safe modification: Any changes to offset calculation require testing all mode_order permutations
- Test coverage: Recommend adding comprehensive tests for all mode_order combinations

## Scaling Limits

**Workspace Allocation for Large Tensors:**
- Current capacity: Workspace uses int64 size calculations (after fix 0390a04), supporting tensors with up to 2^63 elements
- Limit: Memory availability; each element requires storage
- Scaling path: Already using int64; further optimization would require specialized memory management (GPU buffers, out-of-core)
- Risk area: Output tensor assembly code (cin_lowerer.py) still has opportunity for optimization via known-nnz path

**Compiled Module Caching Unbounded Growth:**
- Current capacity: No limit on `_einsum_dispatch_cache` or `_so_cache` size
- Limit: Memory growth as more unique operations are compiled
- Symptom: Long-running inference servers may accumulate compiled modules
- Scaling path: Implement LRU eviction or module cleanup on application shutdown
- Impact: High if application processes many different tensor shapes/formats
- Recommendation: Add cache size limits and eviction policy (prioritize removing least-used modules)

**Code Generation Latency:**
- Current capacity: JIT compilation latency currently acceptable for typical workloads
- Limit: Complex operations with many index variables can take 100ms+ to compile
- Concern: First-call latency in latency-sensitive applications (e.g., serving)
- Scaling path: Pre-compilation or trace-based specialization could reduce latency
- Recommendation: Document compilation time and provide warm-up guidance for deployments

## Scaling Limits

**3D+ Tensor Support Incomplete:**
- Current capacity: Code generation works for 2D tensors and higher, but 3D+ has known issues (bug fixed in 0390a04)
- Limit: Higher-dimensional tensors more prone to index calculation errors and hardcoded assumptions
- Scaling path: Audit all level-related code for hardcoded indices; generalize to arbitrary ndim
- Risk: Each new tensor rank discovered in production may reveal new bugs
- Recommendation: Add systematic testing for 3D, 4D, 5D cases during CI

## Dependencies at Risk

**PyTorch Version Dependency:**
- Risk: Code uses torch.jit.load, torch.cuda, torch C++ extensions (pybind11)
- Files: `src/scorch/utils.py`, `src/scorch/ops.py`, build system
- Impact: PyTorch API changes could break compilation or execution
- Current status: Code targets PyTorch with C++ bindings
- Recommendation: Pin PyTorch version in requirements.txt or document minimum/maximum compatible versions

**C++ Compiler Version Dependency:**
- Risk: Code uses modern C++ features (11/14 standards) and inline assembly-like constructs
- Files: All csrc/ files, compiler-generated code
- Impact: Older compilers (pre-2015) may fail to compile
- Current mitigation: CMakeLists.txt likely specifies C++ standard
- Recommendation: Document minimum compiler versions and test on CI

**LibTorch/PyTorch C++ API Stability:**
- Risk: Generated code calls PyTorch C++ APIs that may change between versions
- Files: Generated C++ code by cin_lowerer.py, codegen.py
- Impact: Breaking PyTorch releases could require code regeneration
- Recommendation: Stabilize on specific PyTorch version range; add compatibility layer if needed

## Test Coverage Gaps

**Untested Areas:**

**Broadcasting Support:**
- What's not tested: Broadcasting with different tensor ranks and dimension sizes
- Files: `src/scorch/stensor.py:172` (TODO: support broadcasting)
- Risk: Broadcasting operations may fail silently or produce wrong shapes
- Priority: High (broadcasting is fundamental operation)

**Format Inference Edge Cases:**
- What's not tested: Format inference for all possible index patterns and tensor formats
- Files: `src/scorch/ops.py` lines 620-663
- Examples: Unusual mode orders, all-dense outputs, mixed sparse levels
- Risk: Unexpected formats generated, affecting performance or correctness
- Priority: High

**3D+ Tensor Operations:**
- What's not tested: Only recently made 3D work (commit 0390a04 fixed crash)
- Files: All compiler modules
- Risk: Higher dimensions may have untested code paths
- Priority: Medium (less common but important for some applications)

**Device Operations (GPU):**
- What's not tested: CUDA/GPU tensor operations
- Files: `src/scorch/stensor.py:150` (cuda() method not implemented)
- Risk: Entire GPU execution path untested
- Priority: Medium (depends on use case)

**Error Paths:**
- What's not tested: What happens with malformed inputs, dimension mismatches, unsupported formats
- Files: Core operator functions (ops.py)
- Risk: Users receive cryptic compilation errors or crashes instead of helpful messages
- Priority: Medium (affects usability)

---

*Concerns audit: 2026-03-18*
