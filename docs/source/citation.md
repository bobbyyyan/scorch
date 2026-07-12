# Citing Scorch

Scorch accompanies the paper presented at **CGO 2026** (the International
Symposium on Code Generation and Optimization). If you use Scorch in academic
work, please cite it.

The authoritative citation — with the final author list, page numbers, and DOI —
is on the [paper page](https://fredrikbk.com/cgo26scorch.html). A template BibTeX
entry:

```bibtex
@inproceedings{scorch-cgo26,
  title     = {Scorch: A Compiler for Sparse Tensor Computation in PyTorch},
  booktitle = {Proceedings of the 2026 IEEE/ACM International Symposium on
               Code Generation and Optimization (CGO)},
  year      = {2026},
  note      = {See https://fredrikbk.com/cgo26scorch.html for the canonical entry},
}
```

To cite the software implementation specifically:

```bibtex
@software{scorch-software,
  title  = {Scorch: A compiler-based sparse tensor library for PyTorch},
  author = {Yan, Bobby},
  url    = {https://github.com/bobbyyyan/scorch},
  note   = {Version 0.0.1},
}
```

```{admonition} Reported results
:class: tip
The paper reports **1.05–5.80× speedups over PyTorch Sparse** across sparse-matrix
and graph-neural-network workloads. When quoting Scorch's performance, cite that
range and the paper; per-machine benchmark numbers vary with hardware.
```
