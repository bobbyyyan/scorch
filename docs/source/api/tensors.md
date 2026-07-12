# Tensors

The `STensor` is Scorch's user-facing sparse tensor. It wraps a PyTorch tensor
together with a {class}`~scorch.TensorFormat` describing its layout. See the
{doc}`Sparse tensors user guide </user_guide/sparse_tensors>` for a narrative
introduction.

## STensor

```{eval-rst}
.. autoclass:: scorch.STensor
   :members: to_torch, to_dense, to_sparse, change_mode_order, copy, insert,
             validate, dim, shape, dtype, format, values, index, storage, name,
             has_index
   :show-inheritance:
```

## Constructors

The factory methods are bound as top-level functions and are the recommended way
to build an `STensor`:

```{eval-rst}
.. autofunction:: scorch.from_torch

.. autofunction:: scorch.from_coo

.. autofunction:: scorch.from_csr
```

```{note}
`from_torch` and `from_coo` are listed in `scorch.__all__`; `from_csr` is bound
as `scorch.from_csr` but omitted from `__all__`. All three are usable, and each
is equivalent to the corresponding `STensor.from_*` static method.
```
