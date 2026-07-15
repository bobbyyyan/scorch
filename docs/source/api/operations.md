# Operations

Sparse tensor operations. `matmul` and `einsum` are the general entry points;
the `sparse_*` functions are fused, prebuilt kernels for neural-network layers.

See the {doc}`Operations user guide </user_guide/operations>` and
{doc}`Neural-network ops </user_guide/neural_network_ops>` for narrative
explanations and worked examples.

## Matrix multiplication

```{eval-rst}
.. autofunction:: scorch.matmul
```

## Einstein summation

```{eval-rst}
.. autofunction:: scorch.einsum
```

## Sparse matrix–vector product

```{eval-rst}
.. autofunction:: scorch.ops.spmv
```

```{note}
`spmv` is an internal operation and is not re-exported at the top level. Reach it
through {func}`scorch.matmul` with a 2-D sparse left operand and a 1-D right
operand — `scorch.matmul(A, x)`.
```

## Fused neural-network kernels

```{eval-rst}
.. autofunction:: scorch.sparse_linear

.. autofunction:: scorch.sparse_linear_fm

.. autofunction:: scorch.sparse_attention

.. autofunction:: scorch.sparse_softmax_csr

.. autofunction:: scorch.fast_transpose
```

## Cache warm-up

```{eval-rst}
.. autofunction:: scorch.precompile_kernels
```
