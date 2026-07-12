# Autotuning

Control how Scorch dispatches the CSR-sparse × dense SpMM path. See the
{doc}`Autotuning user guide </user_guide/autotuning>` for the level ladder and
the no-regression design.

```{eval-rst}
.. autoclass:: scorch.autotune
   :members:
   :special-members: __enter__, __exit__, __call__

.. autofunction:: scorch.set_autotune

.. autofunction:: scorch.get_autotune

.. autofunction:: scorch.clear_autotune_cache

.. autofunction:: scorch.compiler_schedule_search_space

.. autofunction:: scorch.schedule_from_tuner_choice
```

The compiler schedule helpers are opt-in and separate from the production native
SpMM selector. They neither read nor update its memoized decisions, persistent
cache, or learned cost model.
