# Compiler internals

Scorch's defining idea is that a sparse kernel is *generated*, not hand-written.
This section is a tour of the compiler for contributors and curious users: how an
index-notation expression becomes a JIT-compiled C++ kernel, one intermediate
representation at a time.

```{mermaid}
flowchart LR
    D[Dispatch<br/>ops.py] --> C[CIN<br/>index notation]
    C --> L[CINLowerer<br/>iterators · scheduler]
    L --> I[LLIR<br/>typed loops]
    I --> G[Codegen<br/>C++ source]
    G --> J[JIT compile<br/>cached .so]
    J --> E[Execute<br/>→ STensor]
```

Read {doc}`pipeline` first for the end-to-end story, then drill into each stage.

```{toctree}
:hidden:

pipeline
index_notation
lowering
codegen
workspaces
```
