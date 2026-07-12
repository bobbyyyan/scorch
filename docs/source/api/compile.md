# Compile

`@scorch.compile` traces a function with `torch.fx`, finds fusible contraction +
elementwise chains, and dispatches them to prebuilt or JIT-fused kernels. See the
{doc}`user guide </user_guide/scorch_compile>` for details.

```{eval-rst}
.. autofunction:: scorch.compile
```
