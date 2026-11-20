# Citing Scorch

Scorch accompanies the paper presented at **CGO 2026** (the International
Symposium on Code Generation and Optimization). If you use Scorch in academic
work, please cite it.

> Bobby Yan, Alexander J. Root, Trevor Gale, David Broman, and Fredrik Kjolstad.
> Fast Autoscheduling for Sparse ML Frameworks. In *2026 IEEE/ACM International
> Symposium on Code Generation and Optimization (CGO)*, pages 28–43, 2026.
> [DOI](https://doi.org/10.1109/CGO68049.2026.11394842) ·
> [PDF](https://fredrikbk.com/publications/scorch.pdf)

```bibtex
@inproceedings{yan2026fast,
  author    = {Yan, Bobby and Root, Alexander J. and Gale, Trevor and Broman, David and Kjolstad, Fredrik},
  title     = {Fast Autoscheduling for Sparse {ML} Frameworks},
  booktitle = {2026 IEEE/ACM International Symposium on Code Generation and Optimization (CGO)},
  pages     = {28--43},
  year      = {2026},
  doi       = {10.1109/CGO68049.2026.11394842}
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
The paper reports 1.05–5.80× speedups over PyTorch Sparse on end-to-end graph
neural network, sparse autoencoder and sparse transformer workloads. When quoting
Scorch's performance, cite that range and the paper; per-machine benchmark numbers
vary with hardware.
```
