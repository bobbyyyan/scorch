# Formats

The format notation names a tensor's physical layout as a per-mode sequence of
level types. See the {doc}`Format system user guide </user_guide/format_system>`
for the full explanation and the familiar-format reference table.

## TensorFormat

```{eval-rst}
.. autoclass:: scorch.TensorFormat
   :members: get_level_formats, get_level_types, get_order, is_dense
   :show-inheritance:
```

## Level types

```{eval-rst}
.. autoclass:: scorch.format.LevelType
   :members:
   :undoc-members:

.. autoclass:: scorch.format.LevelFormat
   :members: get_level_type
```

```{note}
`LevelType` and `LevelFormat` live in the `scorch.format` module.
`TensorFormat` is also re-exported at the top level as `scorch.TensorFormat`.
```
