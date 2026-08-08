from __future__ import annotations
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import replace
from typing import Dict, Optional, Tuple, Union, List

import torch

from .compiler.cin import (
    IndexStmt,
    TensorVar,
    ForAll,
    IndexVar,
    Workspace,
    Where,
    TensorAssign,
)
from .compiler.cin_lowerer import CINLowerer
from .compiler.cin_analysis import normalize_cin
from .compiler.compile_options import CompileOptions
from .compiler.codegen import LLIRLowerer
from .compiler.compilation_context import CompilerStageId, CompilationContext
from .exceptions import (
    TensorFormatError,
    TensorIndexError,
    TensorLayoutError,
    TensorStorageError,
    TensorTypeError,
    TensorValidationError,
)
from .format import TensorFormat, LevelFormat, LevelType, audit_format_state
from .layout import TensorLayout, TensorMetadata
from .storage import SparseStorage, TensorIndex
from .utils import (
    parse_format,
    _kernel_name,
    _load_validated_prepared_kernel,
    _prepare_jit_build,
)


def _compilation_context_at_boundary(
    compilation_context: Optional[CompilationContext],
    compile_options: CompileOptions,
) -> CompilationContext:
    """Create or validate the one timing owner paired with the one snapshot."""

    if compilation_context is not None:
        if type(compilation_context) is not CompilationContext:
            raise TypeError(
                "_compilation_context must be a CompilationContext instance"
            )
        if compilation_context.compile_options is not compile_options:
            raise TypeError(
                "_compilation_context must retain this compilation's exact "
                "CompileOptions snapshot"
            )
        return compilation_context
    return CompilationContext(compile_options=compile_options)


def _finalize_generated_mode_indices(
    tensor_format: TensorFormat,
    mode_indices: Sequence[Sequence[torch.Tensor]],
) -> List[List[torch.Tensor]]:
    """Finish zero-initialized trailing compressed positions from JIT output."""
    finalized = [list(arrays) for arrays in mode_indices]
    for level, level_format in enumerate(tensor_format.get_level_formats()):
        if level_format.get_level_type() != LevelType.COMPRESSED:
            continue
        positions = finalized[level][0]
        if positions.numel() == 0 or positions[-1].item() != 0:
            continue
        nonzero = torch.nonzero(positions, as_tuple=False).flatten()
        if nonzero.numel() == 0:
            continue
        last_written = int(nonzero[-1].item())
        repaired = positions.clone()
        repaired[last_written + 1 :] = repaired[last_written]
        finalized[level][0] = repaired
    return finalized


_MAX_FORMAT_BIT_WIDTH = (1 << 63) - 1


def _owned_sparse_format(tensor_format: TensorFormat) -> TensorFormat:
    """Validate and detach one parsed public ``to_sparse`` format.

    ``TensorFormat`` and ``LevelFormat`` are frozen value objects, but Python's
    ``object.__setattr__`` can still forge or mutate their stored state.  A
    runtime tensor must not retain caller-owned format objects: changing one
    after conversion would otherwise change the tensor's declared layout while
    leaving its index arrays and values untouched.  Inspect exact stored fields
    without invoking overridable accessors, then rebuild both container layers.
    """

    if type(tensor_format) is not TensorFormat:
        raise TensorTypeError("to_sparse format must be an exact TensorFormat")
    audited = audit_format_state(tensor_format)
    if audited is not None:
        for audited_level in audited:
            if (
                audited_level.bit_width is not None
                and audited_level.bit_width > _MAX_FORMAT_BIT_WIDTH
            ):
                raise TensorFormatError(
                    "to_sparse format level bit_width must be a positive "
                    "signed-int64 exact int or None"
                )
        # The shared construction-side audit already proved every stored
        # field exact and rebuilt both container layers.
        return TensorFormat(audited)
    # Otherwise re-walk the same state to name the exact defect: this public
    # entry point owes a more precise error than the shared construction
    # boundary.
    state = object.__getattribute__(tensor_format, "__dict__")
    state_keys = tuple(state) if type(state) is dict else ()
    if (
        type(state) is not dict
        or len(state_keys) != 1
        or type(state_keys[0]) is not str
        or state_keys[0] != "_level_formats"
    ):
        raise TensorFormatError("to_sparse format has malformed stored state")
    levels = state["_level_formats"]
    if type(levels) is not tuple:
        raise TensorFormatError("to_sparse format levels must be an exact tuple")

    owned_levels = []
    for level, level_format in enumerate(levels):
        if type(level_format) is not LevelFormat:
            raise TensorFormatError(
                f"to_sparse format level {level} must be an exact LevelFormat"
            )
        level_state = object.__getattribute__(level_format, "__dict__")
        level_keys = tuple(level_state) if type(level_state) is dict else ()
        if (
            type(level_state) is not dict
            or len(level_keys) != 2
            or any(type(key) is not str for key in level_keys)
            or set(level_keys) != {"_mode", "_bit_width"}
        ):
            raise TensorFormatError(
                f"to_sparse format level {level} has malformed stored state"
            )
        mode = level_state["_mode"]
        bit_width = level_state["_bit_width"]
        if type(mode) is not LevelType:
            raise TensorFormatError(
                f"to_sparse format level {level} mode must be an exact LevelType"
            )
        if bit_width is not None and (
            type(bit_width) is not int
            or bit_width <= 0
            or bit_width > _MAX_FORMAT_BIT_WIDTH
        ):
            raise TensorFormatError(
                f"to_sparse format level {level} bit_width must be a positive "
                "signed-int64 exact int or None"
            )
        owned_levels.append(LevelFormat(mode, bit_width=bit_width))
    return TensorFormat(owned_levels)


class Window(object):
    """A tensor window object that describes the slice into a physical storage (TensorStorage)
    or another logical tensor (Tensor)
    Contains:
        - an offset for the starting coordinate of the window
        - a shape tuple for the shape of the window
        - a step tuple for the step of the window
    """

    def __init__(self, offset: Tuple[int], shape: Tuple[int], step: Tuple[int]):
        self.offset = offset
        self.shape = shape
        self.step = step

    def __str__(self):
        return f"Window(offset={self.offset}, shape={self.shape}, step={self.step})"

    def __repr__(self):
        return f"Window(offset={self.offset}, shape={self.shape}, step={self.step})"

    def __copy__(self):
        return Window(deepcopy(self.offset), deepcopy(self.shape), deepcopy(self.step))


def _is_directly_materialized_format(output_format: TensorFormat) -> bool:
    """A ``d``/``s`` layout the per-entry filter kernel cannot assemble.

    The legacy filter kernel sizes each compressed level's position array
    from its immediately enclosing level alone, so it assembles exactly the
    layouts whose compressed levels form one contiguous run under at most
    one leading dense level -- ``ds``, ``ss``, ``dss``, ``sss``, ``dsss``,
    ``ssss``, and so on.  Every other ``d``/``s`` layout is materialized
    directly instead:

    - a dense value-bearing suffix (``sd``, ``ssd``, ``dssd``, ...), whose
      values would otherwise be stored without parent coordinates;
    - a dense level above a compressed level that is not that single
      leading prefix (``dds``, ``sds``, ``ddss``, ``sdss``, ...), whose
      position array would otherwise be sized by one dense extent instead
      of the product of every dense parent.

    Both are materialized by the same rule: one complete dense block per
    stored prefix path, a prefix path stored exactly when its block holds
    a nonzero, and a stored block keeping its interior zeros.
    """

    kinds = output_format.get_level_types()
    if len(kinds) < 2 or not all(
        kind in (LevelType.DENSE, LevelType.COMPRESSED) for kind in kinds
    ):
        return False
    if not any(kind is LevelType.COMPRESSED for kind in kinds):
        return False
    # ``d?s+`` -- the exact family the per-entry filter kernel assembles.
    body = kinds[1:] if kinds[0] is LevelType.DENSE else kinds
    return not all(kind is LevelType.COMPRESSED for kind in body)


class STensor:
    """A sparse tensor stored in a custom, per-mode layout.

    ``STensor`` is Scorch's user-facing sparse tensor. It is a thin *logical*
    handle: one immutable :class:`~scorch.layout.TensorMetadata` owns its name,
    dtype, device, and :class:`~scorch.layout.TensorLayout`; numeric payload and
    structural indices live in a validated
    :class:`~scorch.storage.SparseStorage`. The layout's
    :class:`~scorch.format.TensorFormat` has one
    :class:`~scorch.format.LevelType` per physical mode (dense, compressed, or
    coordinate).

    Users almost never construct an ``STensor`` directly. Build one from a torch
    tensor with the factories :meth:`from_torch`, :meth:`from_csr`, or
    :meth:`from_coo` (also re-exported at module scope as ``scorch.from_torch``,
    ``scorch.from_csr``, ``scorch.from_coo``), or use
    ``scorch.from_components`` for explicit storage. Exit back to PyTorch with
    :meth:`to_torch`. Matmul is the top-level function ``scorch.matmul(a, b)`` /
    ``scorch.einsum(...)`` — ``STensor`` deliberately defines no ``__matmul__``,
    so ``a @ b`` will not work.

    Notes
    -----
    This is a plain Python class, not an ``nn.Module``. Because the payload lives
    in ``self._storage`` and never as a direct tensor attribute, ``nn.Module``
    registered nothing useful — it only added per-instance
    ``__init__``/``__setattr__``/``isinstance`` overhead that dominated matmul
    latency on small matrices. STensors are transient data, are never registered
    as submodules, and carry no autograd, so dropping ``nn.Module`` is
    behaviour-preserving. ``requires_grad`` is stored but inert (no autograd).

    Scorch is a CPU compiler library. ``repr(stensor)`` is uninformative (it
    always prints ``"Tensor"``); inspect a tensor via :attr:`shape`,
    ``str(x.format)`` (e.g. ``"d,s"``), :attr:`values`, and
    ``x.index.mode_indices`` instead.

    Examples
    --------
    >>> import torch
    >>> import scorch
    >>> t = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    >>> a = scorch.from_torch(t, "A")     # dense STensor, format "d,d"
    >>> a.shape
    (3, 4)
    >>> torch.equal(a.to_torch(), t)
    True

    See Also
    --------
    from_torch : Build an STensor from a dense/COO/CSR torch tensor.
    to_torch : Materialize back to a dense ``torch.Tensor``.
    """

    _metadata: TensorMetadata
    _storage: SparseStorage

    def __init__(
        self,
        name: Optional[str] = None,
        shape: Optional[Tuple[int, ...]] = None,
        storage: Optional[SparseStorage] = None,
        index: Optional[TensorIndex] = None,
        value: Optional[torch.Tensor] = None,
        requires_grad: Optional[bool] = False,
    ) -> None:
        tensor_name = "tensor" if name is None else name
        if not isinstance(requires_grad, bool):
            raise TensorTypeError("requires_grad must be a bool")
        if storage is not None:
            if index is not None or value is not None:
                raise TensorStorageError(
                    "storage cannot be combined with separate index or value arguments"
                )
            if not isinstance(storage, SparseStorage):
                raise TensorTypeError("storage must be a SparseStorage")
            if shape is not None:
                if isinstance(shape, (str, bytes)) or not isinstance(shape, Sequence):
                    raise TensorTypeError("shape must be a sequence of integers")
                declared_shape = tuple(shape)
                if declared_shape != storage.layout.physical_shape:
                    raise TensorLayoutError(
                        f"shape {declared_shape} does not match storage physical "
                        f"shape {storage.layout.physical_shape}"
                    )
            runtime_storage = storage
        else:
            missing = [
                field
                for field, item in (
                    ("shape", shape),
                    ("index", index),
                    ("value", value),
                )
                if item is None
            ]
            if missing:
                raise TensorValidationError(
                    "runtime STensor construction requires shape, index, and value; "
                    f"missing {', '.join(missing)}. Use TensorSpec for compile-only tensors."
                )
            if not isinstance(index, TensorIndex):
                raise TensorTypeError("index must be a TensorIndex")
            if not isinstance(value, torch.Tensor):
                raise TensorTypeError("value must be a torch.Tensor")
            if shape is None:
                raise TensorLayoutError("shape is required for runtime storage")
            layout = TensorLayout.from_physical_shape(
                shape, index.format, index.mode_order, index.index_dtype
            )
            runtime_storage = SparseStorage(layout, value, index=index)
        metadata = TensorMetadata(
            tensor_name,
            runtime_storage.value.dtype,
            runtime_storage.value.device,
            runtime_storage.layout,
            requires_grad,
        )
        self._set_state(metadata, runtime_storage)

    @classmethod
    def _from_validated(
        cls, metadata: TensorMetadata, storage: SparseStorage
    ) -> "STensor":
        """Internal constructor for already assembled value objects."""
        tensor = object.__new__(cls)
        tensor._set_state(metadata, storage)
        return tensor

    def _set_state(self, metadata: TensorMetadata, storage: SparseStorage) -> None:
        if not isinstance(metadata, TensorMetadata):
            raise TensorTypeError("metadata must be TensorMetadata")
        if not isinstance(storage, SparseStorage):
            raise TensorTypeError("storage must be SparseStorage")
        if metadata.layout != storage.layout:
            raise TensorLayoutError(
                "metadata and storage must reference the same layout"
            )
        if metadata.dtype != storage.value.dtype:
            raise TensorStorageError(
                f"metadata dtype {metadata.dtype} does not match values dtype "
                f"{storage.value.dtype}"
            )
        if metadata.device != storage.value.device:
            raise TensorStorageError(
                f"metadata device {metadata.device} does not match values device "
                f"{storage.value.device}"
            )
        storage.validate()
        self._metadata = metadata
        self._storage = storage

    def insert(self, indices, values):
        """Insert values into the tensor.

        Parameters
        ----------
        indices : array-like
            Coordinates at which to insert.
        values : array-like
            Values to insert at ``indices``.

        Raises
        ------
        NotImplementedError
            Always. Build a new tensor with a factory instead.
        """
        raise NotImplementedError("STensor insertion is not implemented")

    def _nnz(self):
        """Get the number of non-zero elements in the tensor."""
        return self.values.numel()

    @property
    def has_index(self) -> bool:
        """Whether the tensor's storage carries a sparsity index.

        Returns
        -------
        bool
            ``True`` if the underlying storage has a
            :class:`~scorch.storage.TensorIndex` (format + coordinates),
            ``False`` for a value-only storage. Delegates to
            ``self.storage.has_index``.
        """
        return self.storage.has_index

    @property
    def name(self) -> str:
        """The tensor's name.

        Returns
        -------
        str
            The name assigned at construction (the factories default it to
            ``"tensor"``).

        Runtime tensors always have a validated non-empty name; factories use
        ``"tensor"`` when none is supplied.
        """
        return self._metadata.name

    @name.setter
    def name(self, name: str) -> None:
        self._metadata = replace(self._metadata, name=name)

    @property
    def values(self) -> torch.Tensor:
        """The flat 1-D tensor of stored (nonzero) values.

        Returns
        -------
        torch.Tensor
            A 1-D tensor holding every stored value in physical order — the
            entire numeric payload. For a dense tensor this is the row-major
            flattening; for CSR/COO it is the ``nnz`` nonzeros. Equivalent to
            ``self.storage.value``.
        """
        return self.storage.value

    @property
    def index(self) -> TensorIndex:
        """The sparsity index (format plus coordinate arrays).

        Returns
        -------
        TensorIndex
            The :class:`~scorch.storage.TensorIndex` describing structure: the
            :class:`~scorch.format.TensorFormat`, the ``mode_indices``
            (e.g. CSR ``[[], [crow, col]]`` or COO ``[[row], [col]]``), and the
            ``mode_order`` permutation. Equivalent to ``self.storage.index``.
        """
        return self.storage.index

    def _native_mode_indices(self) -> List[List[torch.Tensor]]:
        """Return trusted internal index handles for native kernel calls."""
        return self.storage._native_mode_indices()

    @property
    def format(self) -> TensorFormat:
        """The per-mode storage format.

        Returns
        -------
        TensorFormat
            The :class:`~scorch.format.TensorFormat`, one
            :class:`~scorch.format.LevelType` per mode. ``str(fmt)`` renders a
            comma-joined level string, e.g. ``"d,d"`` (dense), ``"d,s"`` (CSR),
            or ``"o,o"`` (COO).

        Runtime tensors always have a format whose rank matches their layout.
        """
        return self.layout.format

    @property
    def storage(self) -> SparseStorage:
        """The physical storage container.

        Returns
        -------
        SparseStorage
            The frozen :class:`~scorch.storage.SparseStorage` holding flat
            values, index arrays, and the canonical layout.
        """
        return self._storage

    @property
    def dtype(self):
        """The authoritative component dtype.

        Returns
        -------
        torch.dtype
            The metadata dtype, cross-validated against stored values.
        """
        return self._metadata.dtype

    @property
    def device(self) -> torch.device:
        """The authoritative runtime device."""
        return self._metadata.device

    @property
    def layout(self) -> TensorLayout:
        """The immutable logical-to-physical layout."""
        return self._metadata.layout

    @property
    def metadata(self) -> TensorMetadata:
        """The immutable runtime metadata value."""
        return self._metadata

    @property
    def logical_shape(self) -> Tuple[int, ...]:
        return self.layout.logical_shape

    @property
    def physical_shape(self) -> Tuple[int, ...]:
        return self.layout.physical_shape

    @property
    def index_dtype(self) -> torch.dtype:
        return self.layout.index_dtype

    @property
    def mode_order(self) -> Tuple[int, ...]:
        return self.layout.permutation

    @property
    def requires_grad(self) -> bool:
        return self._metadata.requires_grad

    @requires_grad.setter
    def requires_grad(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise TensorTypeError("requires_grad must be a bool")
        self._metadata = replace(self._metadata, requires_grad=value)

    @property
    def shape(self) -> Tuple[int, ...]:
        """The current physical shape (retained for API compatibility).

        Returns
        -------
        tuple of int
            Extents in physical storage-level order. Use
            :attr:`logical_shape` for the original logical axis order.
        """
        # Compatibility: historically ``shape`` is the current physical shape.
        return self.layout.physical_shape

    def __str__(self):
        """Get a string representation of the tensor."""
        # return f"TacoTensor_{self._name}({self._storage})"
        return "Tensor"

    def __repr__(self):
        """Get a string representation of the tensor."""
        return self.__str__()

    def validate(self) -> None:
        """Validate the tensor's internal consistency.

        Re-runs metadata/storage agreement and sparse storage invariant checks.
        Returns ``None`` when the tensor is valid and raises a Scorch domain
        exception otherwise.
        """
        self._set_state(self._metadata, self._storage)

    def to(self, device):
        """Move the tensor to a device.

        Parameters
        ----------
        device
            Target device.

        Raises
        ------
        NotImplementedError
            Always. Scorch is a CPU-only compiler library; there is no device
            transfer.
        """
        raise NotImplementedError("STensor is CPU-only and does not support transfer")

    def cuda(self):
        """Move the tensor to the GPU.

        Raises
        ------
        NotImplementedError
            Always — this delegates to :meth:`to`, which is unimplemented.
            Scorch is CPU-only.
        """
        return self.to(torch.cuda.current_device())

    def clone(self):
        """Clone the tensor.

        Returns
        -------
        STensor
            An independent validated copy.

        See Also
        --------
        copy : The supported way to duplicate an ``STensor``.
        """
        return self.copy()

    def dim(self):
        """Number of tensor dimensions (order).

        Returns
        -------
        int
            ``len(self.shape)``.

        Notes
        -----
        ``STensor`` has no ``.ndim`` property; use this method instead.
        """
        return len(self.shape)

    def __add__(self, other) -> STensor:
        """Element-wise addition of two tensors (``self + other``).

        JIT-compiles and runs a codegen kernel that adds the two operands
        element-wise. ``other`` is first relaid out (via
        :meth:`change_mode_order`) to match ``self``'s ``mode_order``. The
        output takes ``self``'s format.

        Parameters
        ----------
        other : STensor
            The right-hand operand. Must have the same shape as ``self``.

        Returns
        -------
        STensor
            A new tensor holding the element-wise sum.

        Notes
        -----
        No broadcasting is performed and the output format is not inferred from
        the inputs (it is fixed to ``self.format``). The first call incurs a C++
        compile; compiled kernels are cached.

        Examples
        --------
        >>> import torch
        >>> import scorch
        >>> A = scorch.from_torch(torch.rand(4, 4), "A")
        >>> B = scorch.from_torch(torch.rand(4, 4), "B")
        >>> C = A + B
        >>> torch.allclose(C.to_torch(), A.to_torch() + B.to_torch(), atol=1e-3)
        True
        """
        if not isinstance(other, STensor):
            raise TensorTypeError("STensor addition requires another STensor")
        if self.logical_shape != other.logical_shape:
            raise TensorLayoutError(
                f"addition requires equal logical shapes, got "
                f"{self.logical_shape} and {other.logical_shape}"
            )
        compile_options = CompileOptions.from_environment()
        compilation_context = CompilationContext(compile_options=compile_options)

        # Scheduling must not mutate the caller's right-hand operand.
        if self.storage.index.mode_order != other.storage.index.mode_order:
            other = other.copy()
            other.change_mode_order(
                self.storage.index.mode_order,
                _compile_options=compile_options,
                _compilation_context=compilation_context,
            )

        frontend_token = compilation_context.begin_stage(
            CompilerStageId.FRONTEND_VALIDATED_OPERATION_CONSTRUCTION,
            compile_options=compile_options,
        )

        try:
            # Perform element-wise addition
            # TODO: support broadcasting
            index_vars = [IndexVar(f"i{i}") for i in range(len(self.shape))]
            ordered_index_vars = [index_vars[i] for i in self.storage.index.mode_order]
            # TODO: output format inferred from input formats
            output_format = self.format
            result_shape = self.shape

            A = TensorVar(
                name="A",
                fmt=output_format,
                shape=result_shape,
                dtype=self.dtype,
                mode_order=self.storage.index.mode_order,
            )
            B = TensorVar(
                name="B",
                fmt=self.format,
                shape=self.shape,
                dtype=self.dtype,
                mode_order=self.storage.index.mode_order,
            )
            C = TensorVar(
                name="C",
                fmt=other.format,
                shape=other.shape,
                dtype=other.dtype,
                mode_order=other.storage.index.mode_order,
            )

            access_key = index_vars[0] if len(index_vars) == 1 else tuple(index_vars)
            rhs_expr = B[access_key] + C[access_key]
            lhs_access = A[access_key]
            cin_stmt: IndexStmt = TensorAssign(lhs_access, rhs_expr)
            for index_var in reversed(ordered_index_vars):
                cin_stmt = ForAll(index_var, cin_stmt)
        except Exception:
            compilation_context.fail_stage(frontend_token)
            raise

        compilation_context.complete_stage(frontend_token)
        cin_stmt = normalize_cin(
            cin_stmt,
            compile_options=compile_options,
            compilation_context=compilation_context,
        )
        lowerer = CINLowerer(
            compile_options=compile_options,
            compilation_context=compilation_context,
        )
        lowered_llir = lowerer._lower_owned_IndexStmt(cin_stmt)
        llir_lowerer = LLIRLowerer(compile_options=compile_options)
        cpp_token = compilation_context.begin_stage(
            CompilerStageId.LLIR_TO_CPP_GENERATION,
            compile_options=compile_options,
        )
        try:
            cpp_code = llir_lowerer.lower_llir(lowered_llir)
        except Exception:
            compilation_context.fail_stage(cpp_token)
            raise
        compilation_context.complete_stage(cpp_token)

        # print("\n\ncpp_code:\n\n", cpp_code)

        header_cpp_code = compile_options.build.preamble_source

        kernel_name_token = compilation_context.begin_stage(
            CompilerStageId.KERNEL_NAME_AND_BUILD_REQUEST_ASSEMBLY,
            compile_options=compile_options,
        )
        try:
            kernel_name = _kernel_name(
                header_cpp_code,
                cpp_code,
                compile_options=compile_options,
            )
            prepared_build = _prepare_jit_build(
                name=kernel_name,
                cpp_sources=[header_cpp_code, cpp_code],
                functions=["evaluate"],
                extra_cflags=list(compile_options.build.extra_cflags),
                extra_ldflags=list(compile_options.build.extra_ldflags),
                compile_options=compile_options,
            )
        except Exception:
            compilation_context.fail_stage(kernel_name_token)
            raise
        compilation_context.complete_stage(kernel_name_token)
        module = _load_validated_prepared_kernel(prepared_build)

        result_cpp = module.evaluate(
            result_shape,
            self.shape,
            self._native_mode_indices(),
            self.storage.value,
            other.shape,
            other._native_mode_indices(),
            other.storage.value,
        )

        result = STensor(
            shape=result_shape,
            index=TensorIndex(
                mode_indices=_finalize_generated_mode_indices(
                    output_format, result_cpp.storage.index.mode_indices
                ),
                tensor_format=output_format,
                mode_order=self.storage.index.mode_order,
            ),
            value=result_cpp.storage.value,
        )

        return result

    def __mul__(self, other) -> STensor:
        """Element-wise multiplication of two tensors (``self * other``).

        Parameters
        ----------
        other : STensor
            The right-hand operand.

        Raises
        ------
        NotImplementedError
            Always. Element-wise multiply is not yet implemented; only
            :meth:`__add__` is available among the element-wise operators.

        See Also
        --------
        __add__ : Element-wise addition (implemented).
        """
        raise NotImplementedError()

    def copy(self) -> STensor:
        """Return a deep copy of the tensor.

        Duplicates the storage: the values tensor is cloned and every index
        array is cloned and detached, so the copy shares no state with the
        original. Name and shape are preserved.

        Returns
        -------
        STensor
            An independent copy.

        :meth:`clone` is an alias for this operation.
        """
        storage = self.storage.copy()
        metadata = TensorMetadata(
            name=self.metadata.name,
            dtype=self.metadata.dtype,
            device=self.metadata.device,
            layout=storage.layout,
            requires_grad=self.metadata.requires_grad,
        )
        return STensor._from_validated(metadata, storage)

    @classmethod
    def from_components(
        cls,
        shape: Sequence[int],
        tensor_format: Union[TensorFormat, str, List[str]],
        mode_indices: Sequence[Sequence[torch.Tensor]],
        values: torch.Tensor,
        *,
        name: Optional[str] = None,
        mode_order: Optional[Sequence[int]] = None,
        index_dtype: Optional[torch.dtype] = None,
        requires_grad: bool = False,
    ) -> STensor:
        """Build a fully validated runtime tensor from explicit components.

        ``shape`` and ``mode_order`` describe physical storage. The constructor
        derives logical shape, validates format rank and permutation, then checks
        all dense/COO/compressed storage invariants before returning.
        """
        if isinstance(shape, (str, bytes)) or not isinstance(shape, Sequence):
            raise TensorTypeError("shape must be a sequence of integers")
        index = TensorIndex(
            tensor_format=tensor_format,
            mode_indices=mode_indices,
            mode_order=mode_order,
            index_dtype=index_dtype,
        )
        return cls(
            name=name,
            shape=tuple(shape),
            index=index,
            value=values,
            requires_grad=requires_grad,
        )

    @staticmethod
    def from_csr(
        csr_matrix: torch.Tensor,
        name: Optional[str] = None,
    ) -> STensor:
        """Create an STensor from a PyTorch CSR matrix.

        Wraps a 2-D ``torch.sparse_csr`` tensor as a Scorch tensor with the
        canonical CSR layout: format ``"d,s"`` (dense rows, compressed cols),
        ``mode_indices = [[], [crow_indices, col_indices]]``, and the CSR values
        as the flat payload. Structural index arrays are copied into immutable
        storage; the values buffer remains an isolated tensor view of the input
        payload and is not silently cast.

        Parameters
        ----------
        csr_matrix : torch.Tensor
            A 2-D sparse tensor in CSR format (``is_sparse_csr`` must be True).
        name : str, optional
            Name for the tensor. Defaults to ``"tensor"``.

        Returns
        -------
        STensor
            A Scorch tensor in CSR (``"d,s"``) format.

        Raises a Scorch domain exception if the input is not a valid rank-2 CPU
        CSR tensor.

        Notes
        -----
        Unlike :meth:`from_torch`, this does not set an explicit ``mode_order``,
        so the index uses the identity permutation. For n-D sparse data use
        :meth:`from_coo`. Re-exported as ``scorch.from_csr``.

        Examples
        --------
        >>> import torch
        >>> import scorch
        >>> dense = torch.tensor([[0., 2., 0.], [1., 0., 3.]])
        >>> a = scorch.from_csr(dense.to_sparse_csr(), "W")
        >>> str(a.format)
        'd,s'
        >>> torch.equal(a.to_torch(), dense)
        True

        See Also
        --------
        from_torch : Auto-detects dense/COO/CSR inputs.
        from_coo : Build from COO (arbitrary rank).
        """
        if not isinstance(csr_matrix, torch.Tensor):
            raise TensorTypeError("from_csr expects a torch.Tensor")
        if csr_matrix.layout != torch.sparse_csr:
            raise TensorStorageError("from_csr expects a sparse CSR tensor")
        if csr_matrix.device.type != "cpu":
            raise TensorStorageError("Scorch only supports CPU CSR tensors")

        # Extract the crow_indices, col_indices, and values
        crow_indices = csr_matrix.crow_indices()
        col_indices = csr_matrix.col_indices()
        values = csr_matrix.values().resolve_conj().resolve_neg()
        shape = csr_matrix.size()

        if len(shape) != 2:
            raise TensorLayoutError("CSR format is only valid for rank-2 tensors")

        return STensor(
            name=name,
            shape=tuple(shape),
            index=TensorIndex(
                tensor_format="ds",
                mode_indices=[[], [crow_indices, col_indices]],
            ),
            value=values,
        )

    @staticmethod
    def from_coo(
        coo_matrix: Optional[torch.Tensor] = None,
        indices: Optional[torch.Tensor] = None,
        values: Optional[torch.Tensor] = None,
        shape: Optional[Tuple[int, ...]] = None,
        name: Optional[str] = None,
    ) -> STensor:
        """Create an STensor from COO data (arbitrary rank).

        Two calling conventions are supported:

        1. Pass ``coo_matrix`` — a ``torch.sparse_coo_tensor``. It is coalesced
           and its indices, values, and shape are read from it.
        2. Pass ``indices``, ``values``, and ``shape`` directly to build COO
           from raw arrays without a torch sparse tensor.

        Every mode is stored as ``LevelType.COORDINATE`` (format ``"o,o,..."``)
        with ``mode_indices[i] = [indices[i]]``. Works for any number of modes,
        unlike :meth:`from_csr` (2-D only).

        Parameters
        ----------
        coo_matrix : torch.Tensor, optional
            A sparse COO tensor. If given, ``indices``/``values``/``shape`` are
            derived from it (after coalescing) and need not be passed.
        indices : torch.Tensor, optional
            Coordinate array of shape ``[ndim, nnz]``. Used when ``coo_matrix``
            is not supplied.
        values : torch.Tensor, optional
            The ``nnz`` nonzero values, shape ``[nnz]``.
        shape : tuple of int, optional
            The logical shape. Required in the raw-arrays form.
        name : str, optional
            Name for the tensor. Defaults to ``"tensor"``.

        Returns
        -------
        STensor
            A Scorch tensor in COO (``"o,o,..."``) format.

        Notes
        -----
        The caller's indices are never mutated or narrowed. Raw COO input is
        coalesced into canonical int64 coordinates, and that dtype is recorded
        by :attr:`layout`. Re-exported as ``scorch.from_coo``.

        Examples
        --------
        >>> import torch
        >>> import scorch
        >>> i = torch.tensor([[0, 1, 1], [2, 0, 2]])
        >>> v = torch.tensor([3., 4., 5.])
        >>> coo = torch.sparse_coo_tensor(i, v, (2, 3)).coalesce()
        >>> a = scorch.from_coo(coo, name="S")
        >>> b = scorch.from_coo(indices=i, values=v, shape=(2, 3), name="S")
        >>> str(a.format)
        'o,o'
        >>> torch.equal(a.to_torch(), coo.to_dense())
        True
        """
        if coo_matrix is not None:
            if any(item is not None for item in (indices, values, shape)):
                raise TensorStorageError(
                    "coo_matrix cannot be combined with indices, values, or shape"
                )
            if not isinstance(coo_matrix, torch.Tensor):
                raise TensorTypeError("coo_matrix must be a torch.Tensor")
            if coo_matrix.layout != torch.sparse_coo:
                raise TensorStorageError("from_coo expects a sparse COO tensor")
            if coo_matrix.device.type != "cpu":
                raise TensorStorageError("Scorch only supports CPU COO tensors")
            if coo_matrix.dense_dim() != 0:
                raise TensorStorageError(
                    "hybrid COO tensors with dense value dimensions are unsupported"
                )
            coo_matrix = coo_matrix.coalesce()
            indices = coo_matrix.indices()
            values = coo_matrix.values().resolve_conj().resolve_neg()
            shape = tuple(coo_matrix.shape)
        else:
            missing = [
                field
                for field, item in (
                    ("indices", indices),
                    ("values", values),
                    ("shape", shape),
                )
                if item is None
            ]
            if missing:
                raise TensorStorageError(
                    "raw COO construction requires indices, values, and shape; "
                    f"missing {', '.join(missing)}"
                )
            if not isinstance(indices, torch.Tensor) or not isinstance(
                values, torch.Tensor
            ):
                raise TensorTypeError("COO indices and values must be torch tensors")
            if isinstance(shape, (str, bytes)) or not isinstance(shape, Sequence):
                raise TensorTypeError("COO shape must be a sequence of integers")
            shape = tuple(shape)
            # Normalize every extent before handing it to PyTorch so malformed
            # shapes are reported as Scorch domain exceptions.
            shape_layout = TensorLayout.from_physical_shape(
                shape,
                "o" * len(shape),
                index_dtype=torch.int64,
            )
            shape = shape_layout.physical_shape
            if indices.device.type != "cpu" or values.device.type != "cpu":
                raise TensorStorageError("Scorch only supports CPU COO tensors")
            if indices.dtype not in (torch.int32, torch.int64):
                raise TensorIndexError("COO indices must use int32 or int64")
            if indices.dim() != 2:
                raise TensorIndexError("COO indices must have shape [rank, nnz]")
            if values.dim() != 1:
                raise TensorStorageError("COO values must be one-dimensional")
            if indices.shape[0] != len(shape):
                raise TensorIndexError("COO index rank does not match shape rank")
            if indices.shape[1] != values.numel():
                raise TensorStorageError("COO coordinate and value counts must match")
            # Coalesce without ever modifying the caller's tensors. PyTorch's COO
            # builder canonicalizes coordinates to int64, which is recorded in
            # the resulting layout rather than silently narrowed to int32.
            try:
                canonical = torch.sparse_coo_tensor(
                    indices.to(torch.int64),
                    values.resolve_conj().resolve_neg(),
                    tuple(shape),
                    check_invariants=True,
                ).coalesce()
            except (RuntimeError, TypeError, ValueError, OverflowError) as error:
                raise TensorIndexError(f"invalid COO coordinates: {error}") from error
            indices = canonical.indices()
            values = canonical.values()
            shape = tuple(canonical.shape)

        if indices is None or values is None or shape is None:
            raise TensorStorageError("COO components were not initialized")
        mode_indices = [[indices[mode]] for mode in range(len(shape))]
        return STensor(
            name=name,
            shape=tuple(shape),
            index=TensorIndex(
                tensor_format="o" * len(shape),
                mode_indices=mode_indices,
            ),
            value=values,
        )

    @staticmethod
    def from_torch(
        tensor: torch.Tensor,
        name: Optional[str] = None,
        mode_order: Optional[List[int]] = None,
    ) -> STensor:
        """Create an STensor from a ``torch.Tensor``, auto-detecting layout.

        The primary constructor. Accepts a dense tensor, a ``torch.sparse_coo``
        tensor, or a 2-D ``torch.sparse_csr`` tensor and picks the Scorch format
        automatically:

        - **dense** input → every mode ``DENSE`` (format ``"d,d,..."``); values
          are the row-major flattening.
        - **sparse_coo** input → coalesced; every mode ``COORDINATE``
          (format ``"o,o,..."``).
        - **sparse_csr** input (2-D) → format ``"d,s"`` (canonical CSR).

        Parameters
        ----------
        tensor : torch.Tensor
            The source tensor (dense, COO, or CSR).
        name : str, optional
            Name for the tensor. Defaults to ``"tensor"``.
        mode_order : list of int, optional
            A permutation of the axes. If given, the input is first
            ``tensor.permute(*mode_order)`` and the permutation is recorded on
            the index (physical axis → logical axis), so :meth:`to_torch` can
            invert it. Defaults to the identity order.

        Returns
        -------
        STensor
            A Scorch tensor whose format matches the input layout.

        Notes
        -----
        ``mode_order`` is how Scorch represents a transposed/relaid-out operand
        without recomputing; most tensors use the identity order.

        Examples
        --------
        >>> import torch
        >>> import scorch
        >>> t = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        >>> a = scorch.from_torch(t, "A")
        >>> str(a.format)
        'd,d'
        >>> torch.equal(a.to_torch(), t)
        True

        See Also
        --------
        from_csr : Build specifically from a CSR matrix.
        from_coo : Build from COO indices/values.
        """
        if not isinstance(tensor, torch.Tensor):
            raise TensorTypeError("from_torch expects a torch.Tensor")
        if tensor.device.type != "cpu":
            raise TensorStorageError("Scorch only supports CPU tensors")
        if tensor.layout not in (
            torch.strided,
            torch.sparse_coo,
            torch.sparse_csr,
        ):
            raise TensorStorageError(f"unsupported torch tensor layout {tensor.layout}")
        rank = tensor.dim()
        identity_order = list(range(rank))
        if mode_order is None:
            mode_order = identity_order
        else:
            if isinstance(mode_order, (str, bytes)) or not isinstance(
                mode_order, Sequence
            ):
                raise TensorLayoutError("mode_order must be a sequence of integers")
            if any(
                isinstance(mode, bool) or not isinstance(mode, int)
                for mode in mode_order
            ):
                raise TensorLayoutError("mode_order entries must be integers")
            if len(mode_order) != rank or sorted(mode_order) != identity_order:
                raise TensorLayoutError(
                    f"mode_order must be a permutation of range({rank})"
                )
        mode_order = list(mode_order)
        if tensor.layout == torch.sparse_csr and mode_order != identity_order:
            raise TensorLayoutError(
                "CSR tensors only support the identity mode_order; convert to COO "
                "before applying a permutation"
            )
        if tensor.layout == torch.sparse_coo and tensor.dense_dim() != 0:
            raise TensorStorageError(
                "hybrid COO tensors with dense value dimensions are unsupported"
            )
        if mode_order != identity_order:
            tensor = tensor.permute(*mode_order)

        if tensor.is_sparse or tensor.is_sparse_csr:
            if tensor.layout == torch.sparse_coo:
                mode_indices = []
                tensor = tensor.coalesce()
                tensor_indices = tensor.indices()
                for i in range(tensor.dim()):
                    mode_indices.append([tensor_indices[i]])

                return STensor(
                    name=name,
                    shape=tuple(tensor.shape),
                    index=TensorIndex(
                        tensor_format="o" * rank,
                        mode_indices=mode_indices,
                        mode_order=mode_order,
                    ),
                    value=tensor.values().resolve_conj().resolve_neg(),
                )

            elif tensor.layout == torch.sparse_csr:
                crow_indices = tensor.crow_indices()
                col_indices = tensor.col_indices()
                values = tensor.values()
                shape = tensor.size()

                return STensor(
                    name=name,
                    shape=shape,
                    index=TensorIndex(
                        tensor_format="ds",
                        mode_indices=[[], [crow_indices, col_indices]],
                        mode_order=mode_order,
                    ),
                    value=values.resolve_conj().resolve_neg(),
                )
            raise TensorStorageError(f"unsupported sparse layout {tensor.layout}")

        return STensor(
            name=name,
            shape=tuple(tensor.shape),
            index=TensorIndex(
                tensor_format="d" * rank,
                mode_indices=[[] for _ in range(rank)],
                mode_order=mode_order,
            ),
            value=tensor.resolve_conj().resolve_neg().contiguous().reshape(-1),
        )

    def to_torch(self, in_place=True) -> torch.Tensor:
        """Materialize to a dense ``torch.Tensor`` (exit to PyTorch).

        Densifies the tensor (via :meth:`to_dense`), casts the values back to
        the logical :attr:`dtype`, reshapes to the tensor's shape, and — if the
        tensor has a non-identity ``mode_order`` — permutes by the inverse
        permutation so the result is returned in the original logical axis
        order. The result is always a dense ``torch.Tensor``.

        Parameters
        ----------
        in_place : bool, default True
            Forwarded to :meth:`to_dense`. When ``True`` the underlying storage
            may be replaced with the densified storage as a side effect.

        Returns
        -------
        torch.Tensor
            A dense tensor equal to the sparse tensor's contents.

        Examples
        --------
        >>> import torch
        >>> import scorch
        >>> t = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        >>> a = scorch.from_torch(t.to_sparse_csr(), "A")
        >>> torch.equal(a.to_torch(), t)
        True
        """
        # Get a dense Scorch tensor
        dense_tensor = self.to_dense(in_place=in_place)
        # Convert the dense Scorch tensor to a torch.Tensor
        # torch_tensor = dense_tensor.storage.value.clone().detach()
        torch_tensor = dense_tensor.storage.value
        if torch_tensor.dtype != self.dtype:
            torch_tensor = torch_tensor.type(self.dtype)
        # Reshape the torch.Tensor to the original shape
        torch_tensor = torch_tensor.reshape(dense_tensor.shape)

        # Permute back if tensor has non-default mode order
        default_mode_order = [i for i in range(self.dim())]
        if (
            self.storage.index.mode_order
            and self.storage.index.mode_order != default_mode_order
        ):
            # Compute inverse permutation
            inv_perm = [0] * len(self.storage.index.mode_order)
            for i, m in enumerate(self.storage.index.mode_order):
                inv_perm[m] = i
            torch_tensor = torch_tensor.permute(*inv_perm)

        return torch_tensor

    def to_dense(
        self,
        fmt: Optional[Union[TensorFormat, str, List[str]]] = None,
        in_place: bool = False,
        *,
        _compile_options: Optional[CompileOptions] = None,
        _compilation_context: Optional[CompilationContext] = None,
    ) -> STensor:
        """Densify to an all-dense ``STensor`` (stays within Scorch).

        Returns an ``STensor`` (not a ``torch.Tensor``) whose format is
        all-``DENSE`` by default. If the tensor is already dense, returns
        ``self`` (when ``in_place``) or a copy. Otherwise Scorch JIT-compiles a
        C++ kernel to scatter the stored values into a dense buffer.

        Parameters
        ----------
        fmt : TensorFormat or str or list of str, optional
            Target format (e.g. ``"dd"`` or ``["dense", "dense"]``), parsed via
            ``parse_format``. If ``None`` the output is all-dense. Passing a
            non-dense ``fmt`` here is an under-specified path — this method is
            intended for densification.
        in_place : bool, default False
            When ``True`` the densified storage replaces ``self._storage`` and
            ``self`` is returned; otherwise a new ``STensor`` is returned.

        Returns
        -------
        STensor
            A dense Scorch tensor.

        Notes
        -----
        The first densification of a given shape/format incurs a C++ compile;
        compiled kernels are cached. To exit to PyTorch use :meth:`to_torch`.

        See Also
        --------
        to_sparse : The inverse (compress to a sparse format).
        to_torch : Materialize to a dense ``torch.Tensor``.
        """

        # If self is already dense at every level, return self
        if self.format.is_dense():
            if in_place:
                return self
            else:
                return self.copy()

        compile_options = (
            CompileOptions.from_environment()
            if _compile_options is None
            else _compile_options
        )
        if type(compile_options) is not CompileOptions:
            raise TypeError("_compile_options must be a CompileOptions instance")
        compilation_context = _compilation_context_at_boundary(
            _compilation_context, compile_options
        )
        frontend_token = compilation_context.begin_stage(
            CompilerStageId.FRONTEND_VALIDATED_OPERATION_CONSTRUCTION,
            compile_options=compile_options,
        )

        try:
            default_index_vars = [
                IndexVar(name) for name in ["i", "j", "k", "l", "m", "n"]
            ]

            if len(self.shape) > len(default_index_vars):
                index_vars = [IndexVar(f"i{i}") for i in range(len(self.shape))]
            else:
                index_vars = default_index_vars[: len(self.shape)]

            # Permute index_vars by mode_order so ForAll nesting matches
            # the physical level order. Don't pass mode_order to TensorVars
            # because the permuted index_vars already reflect physical order;
            # get_sorted_index_vars() with identity mode_order will then
            # correctly map subscript position k to physical level k.
            if self.storage.index.mode_order:
                index_vars = [index_vars[i] for i in self.storage.index.mode_order]

            if self.has_index:
                B = TensorVar(
                    name="B",
                    fmt=self.format,
                    shape=self.shape,
                    dtype=self.dtype,
                )
            else:
                B = TensorVar(
                    name="B",
                    fmt=TensorFormat(
                        level_formats=[
                            LevelFormat(mode=LevelType.DENSE)
                            for _ in range(len(self.shape))
                        ]
                    ),
                    shape=self.shape,
                    dtype=self.dtype,
                )

            if fmt is None:
                # TODO: infer output format from input format
                # For now, make every level COMPRESSED
                output_format = TensorFormat(
                    level_formats=[
                        LevelFormat(mode=LevelType.DENSE)
                        for _ in range(len(self.shape))
                    ]
                )
            else:
                output_format = parse_format(fmt)

            A = TensorVar(
                name="A",
                fmt=output_format,
                shape=self.shape,
                dtype=self.dtype,
            )

            access_key = index_vars[0] if len(index_vars) == 1 else tuple(index_vars)
            rhs_access = B[access_key]
            lhs_access = A[access_key]
            cin_stmt: IndexStmt = TensorAssign(lhs_access, rhs_access)
            for index_var in reversed(index_vars):
                cin_stmt = ForAll(index_var, cin_stmt)
        except Exception:
            compilation_context.fail_stage(frontend_token)
            raise

        compilation_context.complete_stage(frontend_token)
        cin_stmt = normalize_cin(
            cin_stmt,
            compile_options=compile_options,
            compilation_context=compilation_context,
        )
        lowerer = CINLowerer(
            filter_zeros=True,
            compile_options=compile_options,
            compilation_context=compilation_context,
        )
        lowered_llir = lowerer._lower_owned_IndexStmt(cin_stmt)
        llir_lowerer = LLIRLowerer(compile_options=compile_options)
        cpp_token = compilation_context.begin_stage(
            CompilerStageId.LLIR_TO_CPP_GENERATION,
            compile_options=compile_options,
        )
        try:
            cpp_code = llir_lowerer.lower_llir(lowered_llir)
        except Exception:
            compilation_context.fail_stage(cpp_token)
            raise
        compilation_context.complete_stage(cpp_token)

        # print("\n\ncpp_code:\n\n", cpp_code)

        header_cpp_code = compile_options.build.preamble_source

        kernel_name_token = compilation_context.begin_stage(
            CompilerStageId.KERNEL_NAME_AND_BUILD_REQUEST_ASSEMBLY,
            compile_options=compile_options,
        )
        try:
            kernel_name = _kernel_name(
                header_cpp_code,
                cpp_code,
                compile_options=compile_options,
            )
            prepared_build = _prepare_jit_build(
                name=kernel_name,
                cpp_sources=[header_cpp_code, cpp_code],
                functions=["evaluate"],
                extra_cflags=list(compile_options.build.extra_cflags),
                extra_ldflags=list(compile_options.build.extra_ldflags),
                compile_options=compile_options,
            )
        except Exception:
            compilation_context.fail_stage(kernel_name_token)
            raise
        compilation_context.complete_stage(kernel_name_token)
        module = _load_validated_prepared_kernel(prepared_build)

        result_cpp = module.evaluate(
            self.shape,
            self.shape,
            self._native_mode_indices(),
            self.storage.value,
        )

        new_tensor = STensor(
            name=self.name,
            shape=self.shape,
            index=TensorIndex(
                tensor_format=output_format,
                mode_indices=_finalize_generated_mode_indices(
                    output_format, result_cpp.storage.index.mode_indices
                ),
                mode_order=self.storage.index.mode_order,
            ),
            value=result_cpp.storage.value,
        )

        if in_place:
            self._set_state(new_tensor.metadata, new_tensor.storage)
            return self

        return new_tensor

    def to_sparse(
        self,
        fmt: Optional[Union[TensorFormat, str, List[str]]] = None,
        *,
        _compile_options: Optional[CompileOptions] = None,
        _compilation_context: Optional[CompilationContext] = None,
    ) -> STensor:
        """Compress to a sparse ``STensor``, mutating in place.

        Filters out zeros and stores the tensor in a sparse format. **This
        method always mutates ``self._storage`` in place and returns ``self``**
        (there is no ``in_place`` flag). By default every mode becomes
        ``COMPRESSED``.

        Parameters
        ----------
        fmt : TensorFormat or str or list of str, optional
            Target sparse format, parsed via ``parse_format``. If ``None`` the
            output is all-``COMPRESSED``.

        Returns
        -------
        STensor
            ``self``, with its storage replaced by the sparse form.

        Notes
        -----
        The 1-D case is special-cased and builds a single compressed or
        coordinate level directly from ``torch.nonzero`` (no kernel compile).
        Rank-2-and-higher formats that the per-entry filter kernel cannot
        assemble -- a dense value-bearing suffix, or a dense level above a
        compressed level other than one leading prefix -- materialize their
        blocked output directly from a dense snapshot; sparse inputs may
        first use the ordinary densification kernel.  Other rank-2-and-higher
        formats JIT-compile a filter-zeros kernel honoring ``mode_order``.

        Examples
        --------
        >>> import torch
        >>> import scorch
        >>> a = scorch.from_torch(torch.tensor([[0., 5.], [0., 0.]]), "A")
        >>> _ = a.to_sparse("ss")   # both modes COMPRESSED; a mutated in place
        >>> str(a.format)
        's,s'

        See Also
        --------
        to_dense : The inverse (densify).
        """
        if len(self.shape) == 1:
            parsed_rank_one_format = None if fmt is None else parse_format(fmt)
            rank_one_format: Optional[TensorFormat]
            rank_one_format = (
                None
                if parsed_rank_one_format is None
                else _owned_sparse_format(parsed_rank_one_format)
            )
            rank_one_options = (
                CompileOptions.from_environment()
                if _compile_options is None
                else _compile_options
            )
            if type(rank_one_options) is not CompileOptions:
                raise TypeError("_compile_options must be a CompileOptions instance")
            rank_one_context = _compilation_context_at_boundary(
                _compilation_context, rank_one_options
            )
            if rank_one_format is not None:
                if rank_one_format.get_order() != 1:
                    raise TensorStorageError(
                        f"format rank {rank_one_format.get_order()} does not "
                        "match tensor rank 1"
                    )
                rank_one_kind = rank_one_format.get_level_types()[0]
                if rank_one_kind not in (
                    LevelType.COMPRESSED,
                    LevelType.COORDINATE,
                ):
                    # This branch assembles one supported sparse level; a
                    # dense request is ``to_dense``'s job and SINGLETON is
                    # not executable by the runtime.
                    raise TensorStorageError(
                        "to_sparse builds a compressed or coordinate rank-1 "
                        f"level; {str(rank_one_format)!r} requests no supported "
                        "sparse mode"
                    )
                output_format = rank_one_format
            else:
                rank_one_kind = LevelType.COMPRESSED
                output_format = TensorFormat(
                    level_formats=[LevelFormat(mode=rank_one_kind)]
                )
            # A sparse receiver must be read as coordinates, not as stored
            # positions: filtering ``self.values`` directly would reinterpret
            # a compressed value array as dense coordinates and silently
            # corrupt the tensor.  Densify out of place under the caller's
            # exact options and context, exactly like the rank>=2 route.
            source_values = (
                self.values
                if self.format.is_dense()
                else self.to_dense(
                    in_place=False,
                    _compile_options=rank_one_options,
                    _compilation_context=rank_one_context,
                ).storage.value.reshape(-1)
            )
            # Find indexes of non-zero elements, flatten them
            nonzero_indices = torch.nonzero(source_values).flatten()
            size = len(nonzero_indices)
            # Create a filtered value tensor that only contains non-zero elements
            filtered_values = source_values[nonzero_indices].clone()
            mode_indices = (
                [
                    [
                        torch.tensor(
                            [0, size],
                            dtype=nonzero_indices.dtype,
                            device=nonzero_indices.device,
                        ),
                        nonzero_indices,
                    ]
                ]
                if rank_one_kind is LevelType.COMPRESSED
                else [[nonzero_indices]]
            )
            new_tensor = STensor(
                name=self.name,
                shape=self.shape,
                index=TensorIndex(
                    tensor_format=output_format,
                    mode_indices=mode_indices,
                    mode_order=self.storage.index.mode_order,
                ),
                value=filtered_values,
            )
            self._set_state(new_tensor.metadata, new_tensor.storage)
        else:
            parsed_requested_format = None if fmt is None else parse_format(fmt)
            requested_format: Optional[TensorFormat]
            requested_format = (
                None
                if parsed_requested_format is None
                else _owned_sparse_format(parsed_requested_format)
            )
            if requested_format is not None:
                if requested_format.get_order() != len(self.shape):
                    raise TensorStorageError(
                        f"format rank {requested_format.get_order()} does not "
                        f"match tensor rank {len(self.shape)}"
                    )
                requested_level_types = tuple(requested_format.get_level_types())
                if any(
                    level_type is LevelType.COORDINATE
                    for level_type in requested_level_types
                ) and any(
                    level_type is not LevelType.COORDINATE
                    for level_type in requested_level_types
                ):
                    raise TensorStorageError(
                        "to_sparse does not support mixed coordinate hierarchies"
                    )
            compile_options = (
                CompileOptions.from_environment()
                if _compile_options is None
                else _compile_options
            )
            if type(compile_options) is not CompileOptions:
                raise TypeError("_compile_options must be a CompileOptions instance")
            compilation_context = _compilation_context_at_boundary(
                _compilation_context, compile_options
            )
            if (
                requested_format is not None
                and _is_directly_materialized_format(requested_format)
                and requested_format.get_order() == len(self.shape)
            ):
                return self._to_sparse_materialized_blocks(
                    requested_format,
                    compile_options=compile_options,
                    compilation_context=compilation_context,
                )
            frontend_token = compilation_context.begin_stage(
                CompilerStageId.FRONTEND_VALIDATED_OPERATION_CONSTRUCTION,
                compile_options=compile_options,
            )
            try:
                default_index_vars = [
                    IndexVar(name) for name in ["i", "j", "k", "l", "m", "n"]
                ]
                if len(self.shape) > len(default_index_vars):
                    index_vars = [IndexVar(f"i{i}") for i in range(len(self.shape))]
                else:
                    index_vars = default_index_vars[: len(self.shape)]

                # Permute index_vars by mode_order for ForAll construction
                ordered_index_vars = [
                    index_vars[i] for i in self.storage.index.mode_order
                ]

                if self.has_index:
                    B = TensorVar(
                        name="B",
                        fmt=self.format,
                        shape=self.shape,
                        dtype=self.dtype,
                        mode_order=self.storage.index.mode_order,
                    )
                else:
                    B = TensorVar(
                        name="B",
                        fmt=TensorFormat(
                            level_formats=[
                                LevelFormat(mode=LevelType.DENSE)
                                for _ in range(len(self.shape))
                            ]
                        ),
                        shape=self.shape,
                        dtype=self.dtype,
                        mode_order=self.storage.index.mode_order,
                    )

                if fmt is None:
                    # TODO: infer output format from input format
                    # For now, make every level COMPRESSED
                    output_format = TensorFormat(
                        level_formats=[
                            LevelFormat(mode=LevelType.COMPRESSED)
                            for _ in range(len(self.shape))
                        ]
                    )
                else:
                    assert requested_format is not None
                    output_format = requested_format

                A = TensorVar(
                    name="A",
                    fmt=output_format,
                    shape=self.shape,
                    dtype=self.dtype,
                    mode_order=self.storage.index.mode_order,
                )

                access_key = (
                    index_vars[0] if len(index_vars) == 1 else tuple(index_vars)
                )
                rhs_access = B[access_key]
                lhs_access = A[access_key]
                cin_stmt: IndexStmt = TensorAssign(lhs_access, rhs_access)
                for index_var in reversed(ordered_index_vars):
                    cin_stmt = ForAll(index_var, cin_stmt)

                # print("\n\ncin_stmt: ", cin_stmt)
            except Exception:
                compilation_context.fail_stage(frontend_token)
                raise

            compilation_context.complete_stage(frontend_token)
            cin_stmt = normalize_cin(
                cin_stmt,
                compile_options=compile_options,
                compilation_context=compilation_context,
            )
            lowerer = CINLowerer(
                filter_zeros=True,
                compile_options=compile_options,
                compilation_context=compilation_context,
            )
            lowered_llir = lowerer._lower_owned_IndexStmt(cin_stmt)
            llir_lowerer = LLIRLowerer(compile_options=compile_options)
            cpp_token = compilation_context.begin_stage(
                CompilerStageId.LLIR_TO_CPP_GENERATION,
                compile_options=compile_options,
            )
            try:
                cpp_code = llir_lowerer.lower_llir(lowered_llir)
            except Exception:
                compilation_context.fail_stage(cpp_token)
                raise
            compilation_context.complete_stage(cpp_token)

            # print("to_sparse cpp_code:\n\n", cpp_code)

            header_cpp_code = compile_options.build.preamble_source

            kernel_name_token = compilation_context.begin_stage(
                CompilerStageId.KERNEL_NAME_AND_BUILD_REQUEST_ASSEMBLY,
                compile_options=compile_options,
            )
            try:
                kernel_name = _kernel_name(
                    header_cpp_code,
                    cpp_code,
                    compile_options=compile_options,
                )
                prepared_build = _prepare_jit_build(
                    name=kernel_name,
                    cpp_sources=[header_cpp_code, cpp_code],
                    functions=["evaluate"],
                    extra_cflags=list(compile_options.build.extra_cflags),
                    extra_ldflags=list(compile_options.build.extra_ldflags),
                    compile_options=compile_options,
                )
            except Exception:
                compilation_context.fail_stage(kernel_name_token)
                raise
            compilation_context.complete_stage(kernel_name_token)
            module = _load_validated_prepared_kernel(prepared_build)

            result_cpp = module.evaluate(
                self.shape,
                self.shape,
                self._native_mode_indices(),
                self.storage.value,
            )

            new_tensor = STensor(
                name=self.name,
                shape=self.shape,
                index=TensorIndex(
                    tensor_format=output_format,
                    mode_indices=_finalize_generated_mode_indices(
                        output_format, result_cpp.storage.index.mode_indices
                    ),
                    mode_order=self.storage.index.mode_order,
                ),
                value=result_cpp.storage.value,
            )
            self._set_state(new_tensor.metadata, new_tensor.storage)

        return self

    def _to_sparse_materialized_blocks(
        self,
        output_format: TensorFormat,
        *,
        compile_options: CompileOptions,
        compilation_context: CompilationContext,
    ) -> STensor:
        """Materialize a ``d``/``s`` block layout directly, in place.

        A ``d``/``s`` prefix over zero-or-more trailing DENSE levels stores
        one complete dense value block per stored prefix path: a prefix
        path is stored exactly when its block contains any nonzero, and a
        stored block keeps its interior zeros.  With no trailing DENSE level
        the block is one scalar and the rule degenerates to ordinary sparse
        storage, so the same walk serves both directly materialized families
        (see :func:`_is_directly_materialized_format`).  This is the same
        conditional-parent discipline the compiled assembly families use.
        The defective filter kernel is never used: an already-dense source is
        copied directly, while a sparse source obtains its snapshot through
        ordinary non-mutating densification under the caller's exact compiler
        options and timing context.
        """

        kinds = output_format.get_level_types()
        rank = len(kinds)
        shape = tuple(self.shape)
        if rank != len(shape):
            raise TensorStorageError(
                f"format rank {rank} does not match tensor rank {len(shape)}"
            )
        suffix = 0
        while suffix < rank and kinds[rank - 1 - suffix] is LevelType.DENSE:
            suffix += 1
        split = rank - suffix
        dense_tensor = self.to_dense(
            in_place=False,
            _compile_options=compile_options,
            _compilation_context=compilation_context,
        )
        dense = dense_tensor.storage.value.reshape(shape)
        block_numel = 1
        for extent in shape[split:]:
            block_numel *= extent
        collapsed = dense.reshape(*shape[:split], block_numel)
        mask = (collapsed != 0).any(dim=-1)
        stored = torch.nonzero(mask)  # lexicographic prefix coordinates
        paths = [tuple(int(x) for x in row) for row in stored]

        mode_indices: List[List[torch.Tensor]] = []
        # ``paths`` is lexicographically sorted (``torch.nonzero`` is
        # row-major), so every prefix's stored children are contiguous and
        # non-decreasing.  Each level therefore groups its children in one
        # pass and carries the enumerated parent prefixes forward, instead of
        # rescanning the whole path list once per parent at every level.
        parents: List[Tuple[int, ...]] = [()]
        for level in range(split):
            if kinds[level] is LevelType.DENSE:
                mode_indices.append([])
                parents = [
                    parent + (coordinate,)
                    for parent in parents
                    for coordinate in range(shape[level])
                ]
                continue

            grouped: Dict[Tuple[int, ...], List[int]] = {}
            for path in paths:
                children = grouped.get(path[:level])
                if children is None:
                    children = []
                    grouped[path[:level]] = children
                if not children or children[-1] != path[level]:
                    children.append(path[level])

            pos = [0]
            crd: List[int] = []
            next_parents: List[Tuple[int, ...]] = []
            for parent in parents:
                for coordinate in grouped.get(parent, ()):
                    crd.append(coordinate)
                    next_parents.append(parent + (coordinate,))
                pos.append(len(crd))
            parents = next_parents
            mode_indices.append(
                [
                    torch.tensor(pos, dtype=torch.int32),
                    torch.tensor(crd, dtype=torch.int32),
                ]
            )
        for _ in range(suffix):
            mode_indices.append([])
        if paths:
            values = collapsed[mask].reshape(-1).clone()
        else:
            values = dense.reshape(-1)[:0].clone()
        new_tensor = STensor(
            name=self.name,
            shape=shape,
            index=TensorIndex(
                tensor_format=output_format,
                mode_indices=mode_indices,
                mode_order=self.storage.index.mode_order,
            ),
            value=values,
        )
        self._set_state(new_tensor.metadata, new_tensor.storage)
        return self

    def change_mode_order(
        self,
        mode_order: List[int],
        *,
        _compile_options: Optional[CompileOptions] = None,
        _compilation_context: Optional[CompilationContext] = None,
    ) -> STensor:
        """Relay out the tensor into a new logical mode order (transpose).

        Permutes the tensor's modes, updating storage and shape in place. A
        fast path handles the common 2-D core formats (``"d,d"``, ``"d,s"``,
        ``"o,o"``) without compiling a kernel; the general path compiles and
        executes a ``Where(producer, consumer)`` CIN, where the producer
        iterates in the old mode order and the consumer in the new one, with a
        multi-dimensional workspace as intermediate.

        Parameters
        ----------
        mode_order : list of int
            The new mode-order permutation. Must be a permutation of
            ``range(self.dim())``.

        Returns
        -------
        STensor
            ``self``, with updated storage and shape. If the requested order
            already matches, ``self`` is returned unchanged.

        Raises
        ------
        TensorLayoutError
            If ``mode_order`` is not a valid permutation matching the tensor
            order.

        Notes
        -----
        ``mode_order`` maps physical axis → logical axis. This is how Scorch
        represents a transposed operand without recomputing; the general
        (non-fast-path) route triggers a JIT C++ compile on first use.
        """
        dim = len(self.shape)
        if not isinstance(mode_order, (list, tuple)):
            raise TensorTypeError("mode_order must be a sequence of integers")
        if any(
            isinstance(mode, bool) or not isinstance(mode, int) for mode in mode_order
        ):
            raise TensorTypeError("mode_order entries must be integers")
        if len(mode_order) != dim or sorted(mode_order) != list(range(dim)):
            raise TensorLayoutError(f"mode_order must be a permutation of range({dim})")

        old_mode_order = (
            self.storage.index.mode_order[:]
            if self.storage.index.mode_order is not None
            else [i for i in range(dim)]
        )

        if old_mode_order == mode_order:
            return self

        # old_mode_order maps physical_axis -> logical_axis.
        # Compute inverse: logical_axis -> physical_axis.
        inv_old_mode_order = [0] * dim
        for physical_axis, logical_axis in enumerate(old_mode_order):
            inv_old_mode_order[logical_axis] = physical_axis

        # Convert shape from current physical layout to logical layout, then
        # remap to the target physical layout described by mode_order.
        logical_shape = tuple(self.shape[inv_old_mode_order[i]] for i in range(dim))
        result_shape = tuple(logical_shape[i] for i in mode_order)
        perm_old_to_new = [inv_old_mode_order[i] for i in mode_order]

        # Fast path for 2D tensors in core formats. This avoids lowering/compiling
        # a transpose kernel for the common matmul operands.
        fmt_str = str(self.format)
        if dim == 2 and fmt_str in {"d,d", "d,s", "o,o"}:

            def _coalesce_2d_coo(
                row: torch.Tensor, col: torch.Tensor, vals: torch.Tensor, num_cols: int
            ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                if row.numel() == 0:
                    return (
                        row.to(self.index_dtype),
                        col.to(self.index_dtype),
                        vals,
                    )

                row64 = row.to(torch.int64)
                col64 = col.to(torch.int64)
                key = row64 * int(num_cols) + col64
                perm = torch.argsort(key)
                row_sorted = row64[perm]
                col_sorted = col64[perm]
                vals_sorted = vals[perm]
                key_sorted = key[perm]

                unique_mask = torch.ones_like(key_sorted, dtype=torch.bool)
                if key_sorted.numel() > 1:
                    unique_mask[1:] = key_sorted[1:] != key_sorted[:-1]

                if torch.all(unique_mask).item():
                    return (
                        row_sorted.to(self.index_dtype),
                        col_sorted.to(self.index_dtype),
                        vals_sorted,
                    )

                segment_ids = torch.cumsum(unique_mask.to(torch.int64), dim=0) - 1
                unique_count = int(segment_ids[-1].item() + 1)
                reduced_vals = torch.zeros(
                    unique_count, dtype=vals_sorted.dtype, device=vals_sorted.device
                )
                reduced_vals.scatter_add_(0, segment_ids, vals_sorted)
                unique_positions = torch.nonzero(unique_mask, as_tuple=False).flatten()
                return (
                    row_sorted[unique_positions].to(self.index_dtype),
                    col_sorted[unique_positions].to(self.index_dtype),
                    reduced_vals,
                )

            mode_indices: Optional[List[List[torch.Tensor]]] = None
            values: Optional[torch.Tensor] = None

            if fmt_str == "d,d":
                dense = self.values.reshape(self.shape).permute(*perm_old_to_new)
                values = dense.contiguous().reshape(-1)
                mode_indices = [[], []]
            elif fmt_str == "o,o":
                old_coords = [
                    self.storage._mode_indices[0][0].to(torch.int64),
                    self.storage._mode_indices[1][0].to(torch.int64),
                ]
                new_row = old_coords[perm_old_to_new[0]]
                new_col = old_coords[perm_old_to_new[1]]
                coalesced_row, coalesced_col, coalesced_values = _coalesce_2d_coo(
                    new_row,
                    new_col,
                    self.values,
                    result_shape[1],
                )
                mode_indices = [
                    [coalesced_row],
                    [coalesced_col],
                ]
                values = coalesced_values
            else:
                crow_indices, col_indices = self.storage._mode_indices[1]
                row_counts = (crow_indices[1:] - crow_indices[:-1]).to(torch.int64)
                old_row = torch.repeat_interleave(
                    torch.arange(
                        self.shape[0], dtype=torch.int64, device=col_indices.device
                    ),
                    row_counts,
                )
                old_col = col_indices.to(torch.int64)
                old_coords = [old_row, old_col]
                new_row = old_coords[perm_old_to_new[0]]
                new_col = old_coords[perm_old_to_new[1]]
                coalesced_row, coalesced_col, coalesced_values = _coalesce_2d_coo(
                    new_row,
                    new_col,
                    self.values,
                    result_shape[1],
                )
                transposed_crow = torch.zeros(
                    result_shape[0] + 1,
                    dtype=self.index_dtype,
                    device=coalesced_row.device,
                )
                if coalesced_row.numel() > 0:
                    row_nnz = torch.bincount(
                        coalesced_row.to(torch.int64), minlength=result_shape[0]
                    )
                    transposed_crow[1:] = torch.cumsum(row_nnz, dim=0)
                mode_indices = [
                    [],
                    [
                        transposed_crow,
                        coalesced_col.to(self.index_dtype),
                    ],
                ]
                values = coalesced_values

            if mode_indices is None or values is None:
                raise TensorStorageError(
                    "mode-order conversion did not produce storage"
                )
            new_tensor = STensor(
                name=self.name,
                shape=result_shape,
                index=TensorIndex(
                    tensor_format=self.format,
                    mode_indices=mode_indices,
                    mode_order=mode_order[:],
                    index_dtype=self.index_dtype,
                ),
                value=values,
            )
            self._set_state(new_tensor.metadata, new_tensor.storage)
            return self

        compile_options = (
            CompileOptions.from_environment()
            if _compile_options is None
            else _compile_options
        )
        if type(compile_options) is not CompileOptions:
            raise TypeError("_compile_options must be a CompileOptions instance")
        compilation_context = _compilation_context_at_boundary(
            _compilation_context, compile_options
        )
        frontend_token = compilation_context.begin_stage(
            CompilerStageId.FRONTEND_VALIDATED_OPERATION_CONSTRUCTION,
            compile_options=compile_options,
        )
        try:
            default_index_vars = [
                IndexVar(name) for name in ["i", "j", "k", "l", "m", "n"]
            ]
            if dim > len(default_index_vars):
                index_vars = [IndexVar(f"i{i}") for i in range(dim)]
            else:
                index_vars = default_index_vars[:dim]

            b_index_vars = [index_vars[i] for i in old_mode_order]
            a_index_vars = [index_vars[i] for i in mode_order]

            B = TensorVar(
                name="B",
                fmt=self.format,
                shape=self.shape,
                dtype=self.dtype,
                mode_order=old_mode_order[:],
            )

            A = TensorVar(
                name="A",
                fmt=self.format,
                shape=result_shape,
                dtype=self.dtype,
                mode_order=mode_order[:],
            )

            workspace = Workspace(
                name="wksp",
                dim=len(self.shape),
                mode_order=mode_order[:],
            )

            producer_stmt = TensorAssign(
                workspace[tuple(index_vars)],
                B[tuple(index_vars)],
            )

            for index_var in b_index_vars[::-1]:
                producer_stmt = ForAll(index_var, producer_stmt)

            consumer_stmt = TensorAssign(
                A[tuple(index_vars)],
                workspace[tuple(index_vars)],
            )

            for index_var in a_index_vars[::-1]:
                consumer_stmt = ForAll(index_var, consumer_stmt)

            cin_stmt: IndexStmt = Where(
                producer=producer_stmt,
                consumer=consumer_stmt,
            )
        except Exception:
            compilation_context.fail_stage(frontend_token)
            raise

        compilation_context.complete_stage(frontend_token)
        cin_stmt = normalize_cin(
            cin_stmt,
            compile_options=compile_options,
            compilation_context=compilation_context,
        )
        lowerer = CINLowerer(
            filter_zeros=True,
            compile_options=compile_options,
            compilation_context=compilation_context,
        )
        lowered_llir = lowerer._lower_owned_IndexStmt(cin_stmt)
        llir_lowerer = LLIRLowerer(compile_options=compile_options)
        cpp_token = compilation_context.begin_stage(
            CompilerStageId.LLIR_TO_CPP_GENERATION,
            compile_options=compile_options,
        )
        try:
            cpp_code = llir_lowerer.lower_llir(lowered_llir)
        except Exception:
            compilation_context.fail_stage(cpp_token)
            raise
        compilation_context.complete_stage(cpp_token)

        header_cpp_code = compile_options.build.preamble_source

        kernel_name_token = compilation_context.begin_stage(
            CompilerStageId.KERNEL_NAME_AND_BUILD_REQUEST_ASSEMBLY,
            compile_options=compile_options,
        )
        try:
            kernel_name = _kernel_name(
                header_cpp_code,
                cpp_code,
                compile_options=compile_options,
            )
            prepared_build = _prepare_jit_build(
                name=kernel_name,
                cpp_sources=[header_cpp_code, cpp_code],
                functions=["evaluate"],
                extra_cflags=list(compile_options.build.extra_cflags),
                extra_ldflags=list(compile_options.build.extra_ldflags),
                compile_options=compile_options,
            )
        except Exception:
            compilation_context.fail_stage(kernel_name_token)
            raise
        compilation_context.complete_stage(kernel_name_token)
        module = _load_validated_prepared_kernel(prepared_build)

        result_cpp = module.evaluate(
            result_shape,
            self.shape,
            self._native_mode_indices(),
            self.storage.value,
        )

        new_tensor = STensor(
            name=self.name,
            shape=result_shape,
            index=TensorIndex(
                tensor_format=self.format,
                mode_indices=_finalize_generated_mode_indices(
                    self.format, result_cpp.storage.index.mode_indices
                ),
                mode_order=mode_order[:],
            ),
            value=result_cpp.storage.value,
        )
        self._set_state(new_tensor.metadata, new_tensor.storage)

        return self
