# Performance

Scorch is a performance library first. This section covers what it achieves,
*how it is measured*, and how to get the best numbers on your own workloads —
plus the design discipline (the "no-regression" convention) that governs every
optimization in the codebase.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} {octicon}`graph;1.5em;sd-text-primary` Benchmarks
:link: benchmarks
:link-type: doc
The CGO 2026 results, the workload families, and the reproducible harnesses in
`bench/`.
:::

:::{grid-item-card} {octicon}`tools;1.5em;sd-text-primary` Tuning guide
:link: tuning_guide
:link-type: doc
Warm the cache, pick an autotune level, and understand when tiling helps.
:::

::::

```{toctree}
:hidden:

benchmarks
tuning_guide
```
