"""Domain exceptions raised at Scorch's public tensor boundaries."""


class ScorchError(Exception):
    """Base class for errors reported by Scorch."""


class TensorValidationError(ScorchError, ValueError):
    """A tensor or one of its value objects is internally inconsistent."""


class TensorFormatError(TensorValidationError):
    """A tensor format cannot be parsed or is structurally invalid."""


class TensorLayoutError(TensorValidationError):
    """Logical and physical layout metadata is inconsistent."""


class TensorIndexError(TensorValidationError):
    """Sparse index arrays violate their declared layout."""


class TensorStorageError(TensorValidationError):
    """Tensor values and sparse index storage are incompatible."""


class TensorDeviceError(TensorValidationError):
    """A tensor uses a device unsupported by Scorch."""


class TensorTypeError(ScorchError, TypeError):
    """A public tensor API received an object of the wrong type."""


class CompileSpecError(TensorValidationError):
    """A compile-only tensor specification is invalid."""
