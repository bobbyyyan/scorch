from __future__ import annotations
from copy import deepcopy
from typing import Optional, Tuple, Union, List

import torch

from .compiler.cin import TensorVar, ForAll, IndexVar, Workspace, Where, TensorAssign, Operation
from .compiler.cin_lowerer import CINLowerer
from .compiler.codegen import LLIRLowerer
from .format import TensorFormat, LevelFormat, LevelType
from .storage import TensorStorage, TensorIndex, TensorStorageView
from .utils import (
    parse_format,
    get_extra_cflags,
    get_extra_ldflags,
    jit_preamble_text,
    _kernel_name,
    _load_kernel,
)


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


class STensor:
    """A sparse tensor stored in a custom, per-mode layout.

    ``STensor`` is Scorch's user-facing sparse tensor. It is a thin *logical*
    handle: it records the tensor's name, logical shape, and logical dtype and
    delegates all numeric payload to ``self._storage`` (a
    :class:`~scorch.storage.TensorStorage`). The storage in turn pairs a flat 1-D
    values tensor with a :class:`~scorch.storage.TensorIndex`, which holds the
    :class:`~scorch.format.TensorFormat` (one :class:`~scorch.format.LevelType`
    per mode — dense, compressed, or coordinate) and the coordinate arrays that
    describe where the nonzeros live.

    Users almost never construct an ``STensor`` directly. Build one from a torch
    tensor with the factories :meth:`from_torch`, :meth:`from_csr`, or
    :meth:`from_coo` (also re-exported at module scope as ``scorch.from_torch``,
    ``scorch.from_csr``, ``scorch.from_coo``), and exit back to PyTorch with
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

    _name: Optional[str]

    _shape: Optional[Tuple[int, ...]]

    # (Logical) component type, which might be different from the physical component type in TensorStorage
    _dtype: torch.dtype = torch.float32

    # TODO: storage can also be a secondary index (TensorStorageView)
    _storage: Optional[Union[TensorStorage, TensorStorageView]]

    def __init__(
        self,
        name: Optional[str] = None,
        shape: Optional[Tuple[int, ...]] = None,
        storage: Optional[Union[TensorStorage, TensorStorageView]] = None,
        index: Optional[TensorIndex] = None,
        value: Optional[torch.Tensor] = None,
        requires_grad: Optional[bool] = False,
    ) -> None:
        if storage is not None:
            self._storage = storage
        else:
            self._storage = TensorStorage(index=index, value=value, shape=shape)
        self._name = name
        self._shape = shape

        if value is not None:
            self._dtype = value.dtype
        elif storage and storage.value is not None:
            self._dtype = storage.value.dtype

        self.requires_grad = requires_grad

    def insert(self, indices, values):
        """Insert values into the tensor.

        Parameters
        ----------
        indices : array-like
            Coordinates at which to insert.
        values : array-like
            Values to insert at ``indices``.

        Notes
        -----
        Not yet implemented — this method is a placeholder (no-op) and performs
        no insertion. There is no supported in-place mutation API on ``STensor``
        today; build a new tensor with a factory instead.
        """
        # TODO: Implement this.
        pass

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

        Raises
        ------
        AssertionError
            If no name was ever set. A hand-built ``STensor(shape=...)`` with no
            ``name`` raises on access; factory-made tensors are always safe.
        """
        assert self._name is not None, "Tensor name is not set."
        return self._name

    @name.setter
    def name(self, name: str) -> None:
        self._name = name

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

        Raises
        ------
        AssertionError
            If the index has no format set.
        """
        tensor_format = self.index.format
        assert tensor_format is not None, "Tensor format is not set."
        return tensor_format

    @property
    def storage(self) -> TensorStorage:
        """The physical storage container.

        Returns
        -------
        TensorStorage
            The :class:`~scorch.storage.TensorStorage` holding the flat values
            tensor and the :class:`~scorch.storage.TensorIndex`.

        Raises
        ------
        AssertionError
            If no storage was set on this tensor.
        """
        assert self._storage is not None, "Tensor storage is not set."
        return self._storage

    @property
    def dtype(self):
        """The logical component dtype (default ``torch.float32``).

        Returns
        -------
        torch.dtype
            The logical dtype. This may differ from the physical dtype of the
            stored values (``storage._dtype``); :meth:`to_torch` casts back to
            this logical dtype on the way out.
        """
        return self._dtype

    @property
    def shape(self) -> Tuple[int, ...]:
        """The logical shape.

        Returns
        -------
        tuple of int
            The tensor's shape in logical (original) axis order, or the empty
            tuple ``()`` if the shape is unset.
        """
        return self._shape if self._shape is not None else tuple()

    def __str__(self):
        """Get a string representation of the tensor."""
        # return f"TacoTensor_{self._name}({self._storage})"
        return "Tensor"

    def __repr__(self):
        """Get a string representation of the tensor."""
        return self.__str__()

    def validate(self):
        """Validate the tensor's internal consistency.

        Raises
        ------
        NotImplementedError
            Always. This method is a placeholder and performs no validation.
        """
        # TODO: Implement this.
        raise NotImplementedError()

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
        # TODO: Implement this.
        raise NotImplementedError()

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

        Raises
        ------
        NotImplementedError
            Always. This method is a placeholder; use :meth:`copy` for a working
            deep copy.

        See Also
        --------
        copy : The supported way to duplicate an ``STensor``.
        """
        # TODO: Implement this.
        raise NotImplementedError()

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
        # Change mode order of other to match self if different
        if self.storage.index.mode_order != other.storage.index.mode_order:
            other.change_mode_order(self.storage.index.mode_order)

        # Perform element-wise addition
        # TODO: support broadcasting
        a_index_vars = [IndexVar(f"i{i}") for i in self.storage.index.mode_order]
        index_vars = [IndexVar(f"i{i}") for i in range(len(self.shape))]
        # TODO: output format inferred from input formats
        output_format = self.format
        result_shape = self.shape

        A = TensorVar(
            name="A",
            fmt=output_format,
            mode_order=self.storage.index.mode_order,
        )
        B = TensorVar(
            name="B",
            fmt=self.format,
            mode_order=self.storage.index.mode_order,
        )
        C = TensorVar(
            name="C",
            fmt=other.format,
            mode_order=other.storage.index.mode_order,
        )

        # Assert A, B, C, and index_vars are defined
        assert A is not None, "Tensor A is not defined."
        assert B is not None, "Tensor B is not defined."
        assert C is not None, "Tensor C is not defined."
        assert index_vars is not None, "Index variables are not defined."

        # Generate the python code for the element-wise addition
        # e.g. A[i0, i1, ...] = B[i0, i1, ...] + C[i0, i1, ...]
        lhs = f'A[{", ".join(["index_vars[{i}]".format(i=i) for i in range(len(self.shape))])}]'
        rhs = f'B[{", ".join(["index_vars[{i}]".format(i=i) for i in range(len(self.shape))])}]'
        rhs += f' + C[{", ".join(["index_vars[{i}]".format(i=i) for i in range(len(self.shape))])}]'
        code = f"{lhs} = {rhs}"
        exec(code)

        # Generate the python code for constructing the ForAll's and execute it
        # e.g. cin_stmt = ForAll(i0, ForAll(i1, ForAll(i2, A._assignment)))
        rhs = "A._assignment"
        assert ForAll is not None, "ForAll is not imported"
        for i in range(len(self.shape))[::-1]:
            rhs = f"ForAll(a_index_vars[{i}], {rhs})"
        cin_stmt = eval(rhs)

        lowerer = CINLowerer()
        lowered_llir = lowerer.lower_IndexStmt(cin_stmt)
        llir_lowerer = LLIRLowerer()
        cpp_code = llir_lowerer.lower_llir(lowered_llir)

        # print("\n\ncpp_code:\n\n", cpp_code)

        header_cpp_code = jit_preamble_text()

        module = _load_kernel(
            name=_kernel_name(header_cpp_code, cpp_code),
            cpp_sources=[header_cpp_code, cpp_code],
            functions=["evaluate"],
            extra_cflags=get_extra_cflags(),
            extra_ldflags=get_extra_ldflags(),
        )

        result_cpp = module.evaluate(
            result_shape,
            self.shape,
            self.index.mode_indices,
            self.storage.value,
            other.shape,
            other.index.mode_indices,
            other.storage.value,
        )

        result = STensor(
            shape=result_shape,
            index=TensorIndex(
                mode_indices=result_cpp.storage.index.mode_indices,
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

        Notes
        -----
        This is the supported way to duplicate an ``STensor`` — :meth:`clone`
        is an unimplemented placeholder.
        """
        return STensor(
            name=self._name,
            shape=self.shape,
            storage=self.storage.copy(),
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
        as the flat payload. No data is copied beyond referencing the input's
        index/value arrays.

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

        Raises
        ------
        AssertionError
            If ``csr_matrix`` is not a sparse CSR tensor, or if it is not 2-D
            (CSR is only valid for 2-D matrices).

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
        # If name is not provided, use the default name
        if name is None:
            name = "tensor"

        # Check if input is a PyTorch CSR tensor
        assert csr_matrix.is_sparse_csr, "Input tensor must be a sparse CSR tensor."

        # Extract the crow_indices, col_indices, and values
        crow_indices = csr_matrix.crow_indices()
        col_indices = csr_matrix.col_indices()
        values = csr_matrix.values()
        shape = csr_matrix.size()

        # We only handle 2D CSR tensors here
        assert len(shape) == 2, "CSR format is only valid for 2D matrices."

        tt_tensor = STensor(
            name=name,
            shape=shape,
            storage=TensorStorage(
                index=TensorIndex(
                    tensor_format=TensorFormat(
                        level_formats=[
                            LevelFormat(mode=LevelType.DENSE),
                            LevelFormat(mode=LevelType.COMPRESSED),
                        ]
                    ),
                    mode_indices=[[], [crow_indices, col_indices]],
                ),
                value=values,
            ),
        )

        return tt_tensor

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
        Index arrays are coerced to ``torch.int`` internally. Re-exported as
        ``scorch.from_coo``.

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
        # If name is not provided, use the default name
        if name is None:
            name = "tensor"

        if coo_matrix is not None:
            coo_matrix = coo_matrix.coalesce()
            indices = coo_matrix.indices()
            values = coo_matrix.values()
            shape = coo_matrix.shape

        mode_indices = []
        for i in range(len(shape)):
            mode_indices.append([indices[i]])

        tt_tensor = STensor(
            name=name,
            shape=tuple(shape),
            storage=TensorStorage(
                index=TensorIndex(
                    tensor_format=TensorFormat(
                        level_formats=[
                            LevelFormat(mode=LevelType.COORDINATE)
                            for _ in range(len(shape))
                        ]
                    ),
                    mode_indices=mode_indices,
                ),
                value=values,
            ),
        )

        return tt_tensor

    @staticmethod
    def from_torch(tensor: torch.Tensor, name: Optional[str] = None, mode_order: Optional[List[int]] = None) -> STensor:
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
        # torch.Tensor is dense, so shape is the same as torch tensor,
        # and format is dense at every level

        # If name is not provided, use the default name
        if name is None:
            name = "tensor"

        if mode_order:
            tensor = tensor.permute(*mode_order)
        else:
            mode_order = [i for i in range(len(tensor.shape))]

        if tensor.is_sparse or tensor.is_sparse_csr:
            if tensor.layout == torch.sparse_coo:
                mode_indices = []
                tensor = tensor.coalesce()
                tensor_indices = tensor.indices()
                for i in range(tensor.dim()):
                    mode_indices.append([tensor_indices[i]])

                tt_tensor = STensor(
                    name=name,
                    shape=tuple(tensor.shape),
                    storage=TensorStorage(
                        index=TensorIndex(
                            tensor_format=TensorFormat(
                                level_formats=[
                                    LevelFormat(mode=LevelType.COORDINATE)
                                    for _ in range(len(tensor.shape))
                                ]
                            ),
                            mode_indices=mode_indices,
                            mode_order=mode_order,
                        ),
                        value=tensor.values(),
                    ),
                )

            elif tensor.layout == torch.sparse_csr:
                crow_indices = tensor.crow_indices()
                col_indices = tensor.col_indices()
                values = tensor.values()
                shape = tensor.size()

                tt_tensor = STensor(
                    name=name,
                    shape=shape,
                    storage=TensorStorage(
                        index=TensorIndex(
                            tensor_format=TensorFormat(
                                level_formats=[
                                    LevelFormat(mode=LevelType.DENSE),
                                    LevelFormat(mode=LevelType.COMPRESSED),
                                ]
                            ),
                            mode_indices=[[], [crow_indices, col_indices]],
                            mode_order=mode_order,
                        ),
                        value=values,
                    ),
                )

            return tt_tensor

        tt_tensor = STensor(
            name=name,
            shape=tuple(tensor.shape),
            storage=TensorStorage(
                index=TensorIndex(
                    tensor_format=TensorFormat(
                        level_formats=[
                            LevelFormat(mode=LevelType.DENSE)
                            for _ in range(len(tensor.shape))
                        ]
                    ),
                    mode_indices=[[] for _ in range(len(tensor.shape))],
                    mode_order=mode_order,
                ),
                value=tensor.flatten(),
            ),
        )

        return tt_tensor

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
        if self.storage.index.mode_order and self.storage.index.mode_order != default_mode_order:
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

        default_index_vars = [IndexVar(name) for name in ["i", "j", "k", "l", "m", "n"]]

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
                dtype=self.dtype,
            )

        if fmt is None:
            # TODO: infer output format from input format
            # For now, make every level COMPRESSED
            output_format = TensorFormat(
                level_formats=[
                    LevelFormat(mode=LevelType.DENSE) for _ in range(len(self.shape))
                ]
            )
        else:
            output_format = parse_format(fmt)

        A = TensorVar(
            name="A",
            fmt=output_format,
            dtype=self.dtype,
        )

        # Assert A, B, and index_vars are defined
        assert A is not None, "A is not defined"
        assert B is not None, "B is not defined"
        assert index_vars is not None, "index_vars is not defined"

        # Generate the python code for A[i0, i1, etc.] = B[i0, i1, etc.] and execute it
        lhs = f'A[{", ".join(["index_vars[{i}]".format(i=i) for i in range(len(self.shape))])}]'
        rhs = f'B[{", ".join(["index_vars[{i}]".format(i=i) for i in range(len(self.shape))])}]'
        code = f"{lhs} = {rhs}"
        exec(code)

        # Generate the python code for constructing the ForAll's and execute it
        # e.g. cin_stmt = ForAll(i0, ForAll(i1, ForAll(i2, A._assignment)))
        rhs = "A._assignment"
        assert ForAll is not None, "ForAll is not imported"
        for i in range(len(self.shape))[::-1]:
            rhs = f"ForAll(index_vars[{i}], {rhs})"
        cin_stmt = eval(rhs)

        lowerer = CINLowerer(filter_zeros=True)
        lowered_llir = lowerer.lower_IndexStmt(cin_stmt)
        llir_lowerer = LLIRLowerer()
        cpp_code = llir_lowerer.lower_llir(lowered_llir)

        # print("\n\ncpp_code:\n\n", cpp_code)

        header_cpp_code = jit_preamble_text()

        module = _load_kernel(
            name=_kernel_name(header_cpp_code, cpp_code),
            cpp_sources=[header_cpp_code, cpp_code],
            functions=["evaluate"],
            extra_cflags=get_extra_cflags(),
            extra_ldflags=get_extra_ldflags(),
        )

        result_cpp = module.evaluate(
            self.shape,
            self.shape,
            self.index.mode_indices,
            self.storage.value,
        )

        new_storage = TensorStorage(
            index=TensorIndex(
                tensor_format=output_format,
                mode_indices=result_cpp.storage.index.mode_indices,
                mode_order=self.storage.index.mode_order,
            ),
            value=result_cpp.storage.value,
        )

        if in_place:
            self._storage = new_storage
            return self

        new_tensor = STensor(
            name=self._name,
            shape=self.shape,
            storage=new_storage,
        )

        return new_tensor

    def to_sparse(
        self, fmt: Optional[Union[TensorFormat, str, List[str]]] = None
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
        The 1-D case is special-cased and builds a single compressed level
        directly from ``torch.nonzero`` (no kernel compile). For rank ≥ 2 Scorch
        JIT-compiles a filter-zeros kernel (honoring the tensor's ``mode_order``);
        the first call of a given shape/format incurs a C++ compile.

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
            # Find indexes of non-zero elements in self.values, flatten them
            nonzero_indices = torch.nonzero(self.values).flatten()
            size = len(nonzero_indices)
            # Create a filtered value tensor that only contains non-zero elements
            filtered_values = self.values[nonzero_indices]
            self._storage = TensorStorage(
                index=TensorIndex(
                    tensor_format=TensorFormat(
                        level_formats=[LevelFormat(mode=LevelType.COMPRESSED)]
                    ),
                    mode_indices=[
                        [
                            torch.Tensor([0, size]),
                            nonzero_indices,
                        ]
                    ],
                ),
                value=filtered_values,
            )
        else:
            default_index_vars = [
                IndexVar(name) for name in ["i", "j", "k", "l", "m", "n"]
            ]
            if len(self.shape) > len(default_index_vars):
                index_vars = [IndexVar(f"i{i}") for i in range(len(self.shape))]
            else:
                index_vars = default_index_vars[: len(self.shape)]

            # Permute index_vars by mode_order for ForAll construction
            ordered_index_vars = [index_vars[i] for i in self.storage.index.mode_order]

            if self.has_index:
                B = TensorVar(
                    name="B",
                    fmt=self.format,
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
                output_format = parse_format(fmt)

            A = TensorVar(
                name="A",
                fmt=output_format,
                shape=self.shape,
                dtype=self.dtype,
                mode_order=self.storage.index.mode_order,
            )

            # Assert A, B, and index_vars are defined
            assert A is not None, "A is not defined"
            assert B is not None, "B is not defined"
            assert index_vars is not None, "index_vars is not defined"

            # Generate the python code for A[i0, i1, etc.] = B[i0, i1, etc.] and execute it
            lhs = f'A[{", ".join(["index_vars[{i}]".format(i=i) for i in range(len(self.shape))])}]'
            rhs = f'B[{", ".join(["index_vars[{i}]".format(i=i) for i in range(len(self.shape))])}]'
            code = f"{lhs} = {rhs}"
            exec(code)

            # Generate the python code for constructing the ForAll's and execute it
            # e.g. cin_stmt = ForAll(i0, ForAll(i1, ForAll(i2, A._assignment)))
            rhs = "A._assignment"
            assert ForAll is not None, "ForAll is not imported"
            for i in range(len(self.shape))[::-1]:
                rhs = f"ForAll(ordered_index_vars[{i}], {rhs})"
            cin_stmt = eval(rhs)

            # print("\n\ncin_stmt: ", cin_stmt)

            lowerer = CINLowerer(filter_zeros=True)
            lowered_llir = lowerer.lower_IndexStmt(cin_stmt)
            llir_lowerer = LLIRLowerer()
            cpp_code = llir_lowerer.lower_llir(lowered_llir)

            # print("to_sparse cpp_code:\n\n", cpp_code)

            header_cpp_code = jit_preamble_text()

            module = _load_kernel(
                name=_kernel_name(header_cpp_code, cpp_code),
                cpp_sources=[header_cpp_code, cpp_code],
                functions=["evaluate"],
                extra_cflags=get_extra_cflags(),
                extra_ldflags=get_extra_ldflags(),
            )

            result_cpp = module.evaluate(
                self.shape,
                self.shape,
                self.index.mode_indices,
                self.storage.value,
            )

            self._storage = TensorStorage(
                index=TensorIndex(
                    tensor_format=output_format,
                    mode_indices=result_cpp.storage.index.mode_indices,
                    mode_order=self.storage.index.mode_order,
                ),
                value=result_cpp.storage.value,
            )

        return self

    def change_mode_order(self, mode_order: List[int]) -> STensor:
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
        AssertionError
            If the tensor has no index/dtype/shape/format, or if ``mode_order``
            is not a valid permutation matching the tensor order.

        Notes
        -----
        ``mode_order`` maps physical axis → logical axis. This is how Scorch
        represents a transposed operand without recomputing; the general
        (non-fast-path) route triggers a JIT C++ compile on first use.
        """
        assert self.has_index, "self.storage.index is None"
        assert self.dtype is not None, "self.dtype is None"
        assert self.shape is not None, "self.shape is None"
        assert self.format is not None, "self.format is None"

        dim = len(self.shape)
        assert len(mode_order) == dim, "mode_order must match tensor order"
        assert sorted(mode_order) == list(
            range(dim)
        ), "mode_order must be a permutation"

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
                    return row.int(), col.int(), vals

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
                    return row_sorted.int(), col_sorted.int(), vals_sorted

                segment_ids = torch.cumsum(unique_mask.to(torch.int64), dim=0) - 1
                unique_count = int(segment_ids[-1].item() + 1)
                reduced_vals = torch.zeros(
                    unique_count, dtype=vals_sorted.dtype, device=vals_sorted.device
                )
                reduced_vals.scatter_add_(0, segment_ids, vals_sorted)
                unique_positions = torch.nonzero(unique_mask, as_tuple=False).flatten()
                return (
                    row_sorted[unique_positions].int(),
                    col_sorted[unique_positions].int(),
                    reduced_vals,
                )

            mode_indices = None
            values = None

            if fmt_str == "d,d":
                dense = self.values.reshape(self.shape).permute(*perm_old_to_new)
                values = dense.contiguous().reshape(-1)
                mode_indices = [[], []]
            elif fmt_str == "o,o":
                old_coords = [
                    self.index.mode_indices[0][0].to(torch.int64),
                    self.index.mode_indices[1][0].to(torch.int64),
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
                crow_indices, col_indices = self.index.mode_indices[1]
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
                    dtype=torch.int64,
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
                        transposed_crow.int(),
                        coalesced_col.int(),
                    ],
                ]
                values = coalesced_values

            self._storage = TensorStorage(
                index=TensorIndex(
                    tensor_format=self.format,
                    mode_indices=mode_indices,
                    mode_order=mode_order[:],
                ),
                value=values,
            )
            self._shape = result_shape
            return self

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

        cin_stmt = Where(
            producer=producer_stmt,
            consumer=consumer_stmt,
        )

        lowerer = CINLowerer(filter_zeros=True)
        lowered_llir = lowerer.lower_IndexStmt(cin_stmt)
        llir_lowerer = LLIRLowerer()
        cpp_code = llir_lowerer.lower_llir(lowered_llir)

        header_cpp_code = jit_preamble_text()

        module = _load_kernel(
            name=_kernel_name(header_cpp_code, cpp_code),
            cpp_sources=[header_cpp_code, cpp_code],
            functions=["evaluate"],
            extra_cflags=get_extra_cflags(),
            extra_ldflags=get_extra_ldflags(),
        )

        result_cpp = module.evaluate(
            result_shape,
            self.shape,
            self.index.mode_indices,
            self.storage.value,
        )

        self._storage = TensorStorage(
            index=TensorIndex(
                tensor_format=self.format,
                mode_indices=result_cpp.storage.index.mode_indices,
                mode_order=mode_order[:],
            ),
            value=result_cpp.storage.value,
        )

        self._shape = result_shape

        return self
