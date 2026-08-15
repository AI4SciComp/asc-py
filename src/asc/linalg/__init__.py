# Copyright 2026 AI4SciComp contributors.
# SPDX-License-Identifier: Apache-2.0
"""Portable linear algebra and stable multi-result records."""

from __future__ import annotations

import collections.abc
import functools
import math
import typing

from asc import _array_api_compat, errors
from asc import typing as asc_typing
from asc.backends import _namespace as backend_namespace
from asc.core import _dtype
from asc.core import namespace as namespace_module
from asc.core._scalar import normalize_real_scalar
from asc.extensions import _dispatch


class EighResult(typing.NamedTuple):
    """Eigenvalues and eigenvectors from a Hermitian decomposition."""

    eigenvalues: object
    eigenvectors: object


class EigResult(typing.NamedTuple):
    """Eigenvalues and right eigenvectors of a square array."""

    eigenvalues: object
    eigenvectors: object


class QRResult(typing.NamedTuple):
    """Orthogonal and upper-triangular QR factors."""

    Q: object
    R: object


class SlogdetResult(typing.NamedTuple):
    """Sign and logarithm of an absolute determinant."""

    sign: object
    logabsdet: object


class SVDResult(typing.NamedTuple):
    """Left vectors, singular values, and right vectors."""

    U: object
    S: object
    Vh: object


class LstsqResult(typing.NamedTuple):
    """Stable least-squares result across supported backends."""

    solution: object
    residuals: object
    rank: object
    singular_values: object


_STANDARD_LINALG_FUNCTIONS: typing.Final = frozenset(
    {
        "cholesky",
        "cross",
        "det",
        "diagonal",
        "eig",
        "eigh",
        "eigvals",
        "eigvalsh",
        "inv",
        "lstsq",
        "matmul",
        "matrix_norm",
        "matrix_power",
        "matrix_rank",
        "matrix_transpose",
        "outer",
        "pinv",
        "qr",
        "slogdet",
        "solve",
        "svd",
        "svdvals",
        "tensordot",
        "trace",
        "vecdot",
        "vector_norm",
    }
)
_FLOATING_LINALG_FUNCTIONS: typing.Final = frozenset(
    {
        "cholesky",
        "det",
        "eig",
        "eigh",
        "eigvals",
        "eigvalsh",
        "inv",
        "lstsq",
        "matrix_norm",
        "matrix_power",
        "pinv",
        "qr",
        "slogdet",
        "solve",
        "svd",
        "svdvals",
        "vector_norm",
    }
)
_NUMERIC_LINALG_FUNCTIONS: typing.Final = frozenset(
    {"cross", "matmul", "outer", "tensordot", "trace", "vecdot"}
)
_LOW_PRECISION_DTYPES: typing.Final = frozenset({"bfloat16", "float16"})
_LOW_PRECISION_SOLVER_FUNCTIONS: typing.Final = frozenset(
    {
        "cholesky",
        "eig",
        "eigh",
        "eigvals",
        "eigvalsh",
        "inv",
        "lstsq",
        "matrix_rank",
        "pinv",
        "qr",
        "solve",
        "svd",
        "svdvals",
    }
)
_LOW_PRECISION_DETERMINANT_BACKENDS: typing.Final = frozenset(
    {"numpy", "torch"}
)


def _low_precision_capability_error(name: str, dtype_name: str) -> None:
    """Reject a low-precision operation without leaking native failures."""
    raise errors.CapabilityNotSupportedError(
        f"linalg.{name}: low-precision CPU kernel is unavailable for "
        f"{dtype_name}"
    )


def _validate_linalg_dtypes(name: str, *operands: object) -> str | None:
    """Enforce the operation-specific linalg dtype domains."""
    category = (
        "floating"
        if name in _FLOATING_LINALG_FUNCTIONS
        else "numeric"
        if name in _NUMERIC_LINALG_FUNCTIONS or name == "matrix_rank"
        else None
    )
    if category is None:
        return None
    operation = f"linalg.{name}"
    dtype_names = [
        backend_namespace.require_dtype_category(operation, operand, category)
        for operand in operands
    ]
    dtype_name = (
        dtype_names[0]
        if len(dtype_names) == 1
        else backend_namespace.require_portable_promotion(operation, *operands)
    )
    if dtype_name not in _LOW_PRECISION_DTYPES:
        return dtype_name
    xp = namespace_module.array_namespace(*operands)
    backend = namespace_module.identify_backend(xp)
    if name in _LOW_PRECISION_SOLVER_FUNCTIONS or (
        name in {"det", "slogdet"}
        and backend in _LOW_PRECISION_DETERMINANT_BACKENDS
    ):
        _low_precision_capability_error(name, dtype_name)
    return dtype_name


def _validate_low_precision_linalg_controls(
    name: str,
    dtype_name: str | None,
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> None:
    """Reject controls that select unavailable low-precision solvers."""
    if dtype_name not in _LOW_PRECISION_DTYPES:
        return
    if name == "matrix_power":
        exponent = typing.cast(int, _linalg_control(args, kwargs, "n", 1))
        if exponent < 0:
            _low_precision_capability_error(name, dtype_name)
    if name == "matrix_norm":
        order = _linalg_control(args, kwargs, "ord")
        if order in {2, -2, "nuc"}:
            _low_precision_capability_error(name, dtype_name)


class LinalgNamespace:
    """Delegate standard linalg operations and normalize result records."""

    def __init__(
        self,
        native: object,
        *,
        backend: asc_typing.BackendName,
        dtype_validator: typing.Callable[[object], None] | None = None,
    ) -> None:
        self._native = native
        self._backend = backend
        self._dtype_validator = dtype_validator

    def __getattr__(self, name: str) -> object:
        """Delegate standard functions not requiring result normalization."""
        if name not in _STANDARD_LINALG_FUNCTIONS:
            raise errors.CapabilityNotSupportedError(
                f"linalg.{name}: selected symbol is outside the frozen "
                "linalg surface"
            )
        try:
            function = getattr(self._native, name)
        except AttributeError as exception:
            raise errors.CapabilityNotSupportedError(
                f"linalg.{name}: selected backend does not provide this "
                "capability"
            ) from exception
        if not callable(function):
            return function

        @functools.wraps(function)
        def checked(*args: object, **kwargs: object) -> object:
            from asc.tree import tree_leaves

            operation = f"linalg.{name}"
            array_count = 2 if name in _BINARY_OPERATIONS else 1
            if len(args) < array_count:
                raise errors.NamespaceError(
                    f"{operation}: expected {array_count} native array "
                    "operand(s)"
                )
            positional_arity = 2 if name == "matrix_power" else array_count
            if len(args) > positional_arity:
                raise TypeError(
                    f"{operation}: keyword-only controls cannot be passed "
                    "positionally"
                )
            xp = _validate_bound_backend(
                self._backend, operation, *args[:array_count]
            )
            array_arguments = tuple(
                value
                for value in tree_leaves((args, kwargs))
                if not isinstance(value, type)
                and _array_api_compat.compat.is_array_api_obj(value)
            )
            _validate_bound_backend(self._backend, operation, *array_arguments)
            _validate_linalg_controls(
                name,
                args,
                kwargs,
                dtype_validator=self._dtype_validator,
            )
            dtype_name = _validate_linalg_dtypes(name, *args[:array_count])
            _validate_low_precision_linalg_controls(
                name, dtype_name, args, kwargs
            )
            _validate_linalg_shapes(name, args, kwargs)
            dispatch_args = args
            if name in _BINARY_OPERATIONS:
                try:
                    promoted_name = (
                        backend_namespace.require_portable_promotion(
                            operation, *args[:2]
                        )
                    )
                except errors.DTypeError:
                    # Mixed-kind linalg promotion is implementation-defined.
                    pass
                else:
                    promoted_dtype = getattr(xp, promoted_name)
                    if self._dtype_validator is not None:
                        self._dtype_validator(promoted_dtype)
                    dispatch_args = (
                        (
                            args[0]
                            if args[0].dtype == promoted_dtype
                            else xp.astype(args[0], promoted_dtype, copy=True)
                        ),
                        (
                            args[1]
                            if args[1].dtype == promoted_dtype
                            else xp.astype(args[1], promoted_dtype, copy=True)
                        ),
                        *args[2:],
                    )
            if (
                name == "matrix_rank"
                and self._backend == "torch"
                and not (
                    xp.isdtype(
                        args[0].dtype, ("real floating", "complex floating")
                    )
                )
            ):
                dispatch_args = (
                    xp.astype(args[0], xp.float64, copy=True),
                    *args[1:],
                )
            dispatch_kwargs = kwargs
            if (
                self._backend == "torch"
                and name in {"matrix_norm", "vector_norm"}
                and kwargs.get("ord", _LINALG_MISSING) is None
            ):
                dispatch_kwargs = {
                    **kwargs,
                    "ord": "fro" if name == "matrix_norm" else 2,
                }
            return function(*dispatch_args, **dispatch_kwargs)

        return checked

    def eigh(
        self,
        array: object,
        /,
        *,
        UPLO: str = "L",  # noqa: N803 - Array API keyword spelling.
    ) -> EighResult:
        """Return a Hermitian eigen decomposition with stable field names."""
        _validate_bound_backend(self._backend, "linalg.eigh", array)
        _validate_linalg_dtypes("eigh", array)
        _validate_linalg_shapes("eigh", (array,), {})
        if type(UPLO) is not str or UPLO not in {"L", "U"}:
            raise TypeError("linalg.eigh: UPLO must be 'L' or 'U'")
        # JAX otherwise averages both triangles before decomposition, which
        # conflicts with the Array API's explicit triangle policy.
        keywords: dict[str, object] = (
            {"UPLO": UPLO, "symmetrize_input": False}
            if self._backend == "jax"
            else {"UPLO": UPLO}
        )
        result = self._native.eigh(array, **keywords)
        return EighResult(result[0], result[1])

    def eig(self, array: object, /) -> EigResult:
        """Return a general eigen decomposition with stable field names."""
        _validate_bound_backend(self._backend, "linalg.eig", array)
        _validate_linalg_dtypes("eig", array)
        _validate_linalg_shapes("eig", (array,), {})
        function = getattr(self._native, "eig", None)
        if function is None:
            raise errors.CapabilityNotSupportedError(
                "linalg.eig: selected backend does not provide complex eigen "
                "outputs"
            )
        result = function(array)
        return EigResult(result[0], result[1])

    def qr(self, array: object, /, *, mode: str = "reduced") -> QRResult:
        """Return a QR decomposition with stable field names."""
        _validate_bound_backend(self._backend, "linalg.qr", array)
        _validate_linalg_dtypes("qr", array)
        _validate_linalg_shapes("qr", (array,), {})
        if type(mode) is not str or mode not in {"reduced", "complete"}:
            raise TypeError("linalg.qr: mode must be 'reduced' or 'complete'")
        result = self._native.qr(array, mode=mode)
        return QRResult(result[0], result[1])

    def slogdet(self, array: object, /) -> SlogdetResult:
        """Return signed log-determinant values with stable field names."""
        _validate_bound_backend(self._backend, "linalg.slogdet", array)
        _validate_linalg_dtypes("slogdet", array)
        _validate_linalg_shapes("slogdet", (array,), {})
        result = self._native.slogdet(array)
        return SlogdetResult(result[0], result[1])

    def svd(self, array: object, /, *, full_matrices: bool = True) -> SVDResult:
        """Return a singular-value decomposition with stable field names."""
        _validate_bound_backend(self._backend, "linalg.svd", array)
        _validate_linalg_dtypes("svd", array)
        _validate_linalg_shapes("svd", (array,), {})
        if type(full_matrices) is not bool:
            raise TypeError(
                "linalg.svd: full_matrices must be a Python Boolean"
            )
        result = self._native.svd(array, full_matrices=full_matrices)
        return SVDResult(result[0], result[1], result[2])

    def lstsq(
        self, first: object, second: object, /, *, rcond: object = None
    ) -> LstsqResult:
        """Solve least squares and normalize all four result fields."""
        function = getattr(self._native, "lstsq", None)
        if function is None:
            raise errors.CapabilityNotSupportedError(
                "linalg.lstsq: selected backend does not provide least squares"
            )
        xp = _validate_bound_backend(
            self._backend, "linalg.lstsq", first, second
        )
        _validate_linalg_dtypes("lstsq", first, second)
        _validate_linalg_shapes("lstsq", (first, second), {})
        if rcond is not None:
            if isinstance(rcond, bool) or not isinstance(rcond, (int, float)):
                raise errors.DTypeError(
                    "linalg.lstsq: rcond must be a finite Python scalar of "
                    "real type "
                    "or None"
                )
            try:
                rcond = float(rcond)
            except OverflowError as exception:
                raise errors.DTypeError(
                    "linalg.lstsq: rcond must be a finite Python scalar of "
                    "real type "
                    "or None"
                ) from exception
            if not math.isfinite(rcond):
                raise errors.DTypeError(
                    "linalg.lstsq: rcond must be a finite Python scalar of "
                    "real type "
                    "or None"
                )
        first, second = _promote_operands("linalg.lstsq", xp, first, second)
        if not xp.isdtype(first.dtype, ("real floating", "complex floating")):
            raise errors.DTypeError(
                "linalg.lstsq: operands must promote to a floating dtype"
            )
        if rcond is not None:
            cutoff_dtype = (
                xp.real(first).dtype
                if xp.isdtype(first.dtype, "complex floating")
                else first.dtype
            )
            rcond = normalize_real_scalar(
                xp,
                cutoff_dtype,
                rcond,
                "linalg.lstsq",
                "rcond",
                device=_array_api_compat.compat.device(first),
            )
        keywords = (
            {"rcond": rcond, "driver": "gelsd"}
            if self._backend == "torch"
            else {"rcond": rcond}
        )
        result = function(first, second, **keywords)
        return LstsqResult(result[0], result[1], result[2], result[3])


def linalg_namespace(selected_backend: object) -> LinalgNamespace:
    """Return a normalized linalg namespace for a :class:`Backend`."""
    xp = selected_backend.xp
    candidate = xp.linalg
    if isinstance(candidate, LinalgNamespace):
        return candidate
    return LinalgNamespace(candidate, backend=selected_backend.name)


_BINARY_OPERATIONS: typing.Final = frozenset(
    {"cross", "matmul", "outer", "solve", "tensordot", "vecdot"}
)
_LINALG_KEYWORDS: typing.Final = {
    "cholesky": frozenset({"upper"}),
    "cross": frozenset({"axis"}),
    "diagonal": frozenset({"offset"}),
    "matrix_norm": frozenset({"keepdims", "ord"}),
    "matrix_rank": frozenset({"rtol"}),
    "pinv": frozenset({"rtol"}),
    "tensordot": frozenset({"axes"}),
    "trace": frozenset({"offset", "dtype"}),
    "vecdot": frozenset({"axis"}),
    "vector_norm": frozenset({"axis", "keepdims", "ord"}),
}
_LINALG_MISSING: typing.Final = object()


def _linalg_control(
    args: tuple[object, ...],
    kwargs: dict[str, object],
    parameter: str,
    position: int | None = None,
) -> object:
    """Return one linalg control from its standard calling position."""
    if parameter in kwargs:
        return kwargs[parameter]
    if position is not None and len(args) > position:
        return args[position]
    return _LINALG_MISSING


def _integer_or_tuple(value: object, *, allow_none: bool) -> bool:
    """Return whether a value is an exact integer axis control."""
    return (
        (allow_none and value is None)
        or type(value) is int
        or (
            isinstance(value, tuple)
            and all(type(item) is int for item in value)
        )
    )


def _validate_norm_order(name: str, order: object) -> None:
    """Reject norm orders outside the operation's standard domain."""
    operation = f"linalg.{name}"
    if order is _LINALG_MISSING or order is None:
        return
    if name == "matrix_norm":
        if type(order) is str and order in {"fro", "nuc"}:
            return
        if type(order) in {int, float}:
            if order in {-2, -1, 1, 2, -math.inf, math.inf}:
                return
            raise ValueError(
                "linalg.matrix_norm: numeric ord must be -2, -1, 1, 2, "
                "or an infinity"
            )
    elif type(order) in {int, float} and not (
        type(order) is float and math.isnan(order)
    ):
        return
    raise TypeError(f"{operation}: ord has an invalid Python type")


def _validate_linalg_tolerance(
    operation: str, args: tuple[object, ...], tolerance: object
) -> None:
    """Validate an optional real-floating tolerance and its batch shape."""
    if tolerance is _LINALG_MISSING or tolerance is None:
        return
    if not _array_api_compat.compat.is_array_api_obj(tolerance):
        if type(tolerance) not in {int, float}:
            raise TypeError(
                f"{operation}: rtol must be a Python real scalar, native "
                "array, or None"
            )
        return
    backend_namespace.require_dtype_category(
        f"{operation} rtol", tolerance, "real floating"
    )
    if len(args[0].shape) < 2:
        return
    tolerance_shape = tuple(tolerance.shape)
    batch_shape = tuple(args[0].shape[:-2])
    compatible = len(tolerance_shape) <= len(batch_shape) and all(
        source in {1, destination}
        for source, destination in zip(
            reversed(tolerance_shape), reversed(batch_shape), strict=False
        )
    )
    if not compatible:
        raise ValueError(
            f"{operation}: rtol shape must broadcast to the input batch shape"
        )


def _validate_linalg_controls(  # pylint: disable=too-many-branches
    name: str,
    args: tuple[object, ...],
    kwargs: dict[str, object],
    *,
    dtype_validator: typing.Callable[[object], None] | None,
) -> None:
    """Validate standard linalg controls before native scalar extraction."""
    allowed = _LINALG_KEYWORDS.get(name, frozenset())
    unexpected = set(kwargs).difference(allowed)
    if unexpected:
        parameter = sorted(unexpected)[0]
        raise TypeError(
            f"linalg.{name}: {parameter} is outside the frozen linalg surface"
        )
    operation = f"linalg.{name}"
    if name == "cholesky":
        upper = _linalg_control(args, kwargs, "upper")
        if upper is not _LINALG_MISSING and type(upper) is not bool:
            raise TypeError(f"{operation}: upper must be a Python Boolean")
    if name in {"cross", "vecdot"}:
        axis = _linalg_control(args, kwargs, "axis")
        if axis is not _LINALG_MISSING and type(axis) is not int:
            raise TypeError(f"{operation}: axis must be a Python integer")
    if name in {"diagonal", "trace"}:
        offset = _linalg_control(args, kwargs, "offset")
        if offset is not _LINALG_MISSING and type(offset) is not int:
            raise TypeError(f"{operation}: offset must be a Python integer")
    if name in {"matrix_norm", "vector_norm"}:
        keepdims = _linalg_control(args, kwargs, "keepdims")
        if keepdims is not _LINALG_MISSING and type(keepdims) is not bool:
            raise TypeError(f"{operation}: keepdims must be a Python Boolean")
        order = _linalg_control(args, kwargs, "ord")
        _validate_norm_order(name, order)
    if name == "vector_norm":
        axis = _linalg_control(args, kwargs, "axis")
        if axis is not _LINALG_MISSING and not _integer_or_tuple(
            axis, allow_none=True
        ):
            raise TypeError(
                "linalg.vector_norm: axis must be an integer, tuple of "
                "integers, or None"
            )
    if name == "tensordot":
        axes = _linalg_control(args, kwargs, "axes")
        valid_pair = (
            isinstance(axes, tuple)
            and len(axes) == 2
            and all(
                isinstance(side, collections.abc.Sequence)
                and not isinstance(side, (str, bytes, bytearray))
                and all(type(item) is int for item in side)
                for side in axes
            )
        )
        if (
            axes is not _LINALG_MISSING
            and type(axes) is not int
            and not valid_pair
        ):
            raise TypeError(f"{operation}: axes has an invalid Python type")
    if name == "matrix_power":
        exponent = _linalg_control(args, kwargs, "n", 1)
        if type(exponent) is not int:
            raise TypeError("linalg.matrix_power: n must be a Python integer")
    if name in {"matrix_rank", "pinv"}:
        tolerance = _linalg_control(args, kwargs, "rtol")
        _validate_linalg_tolerance(operation, args, tolerance)
    if name == "trace" and "dtype" in kwargs:
        if dtype_validator is None:
            raise errors.DTypeError(
                "linalg.trace: dtype validation is unavailable"
            )
        dtype_validator(kwargs["dtype"])


def _validate_linalg_shapes(
    name: str, args: tuple[object, ...], kwargs: dict[str, object]
) -> None:
    """Enforce frozen linalg rank requirements before native dispatch."""
    operation = f"linalg.{name}"
    minimum_two = {
        "cholesky",
        "det",
        "diagonal",
        "eig",
        "eigh",
        "eigvals",
        "eigvalsh",
        "inv",
        "matrix_norm",
        "matrix_power",
        "matrix_rank",
        "matrix_transpose",
        "pinv",
        "qr",
        "slogdet",
        "svd",
        "svdvals",
        "trace",
    }
    if name in minimum_two and len(args[0].shape) < 2:
        raise ValueError(f"{operation}: input must be at least two-dimensional")
    if name == "outer" and any(len(value.shape) != 1 for value in args[:2]):
        raise ValueError(f"{operation}: both inputs must be one-dimensional")
    if name in {"matmul", "vecdot"} and any(
        len(value.shape) < 1 for value in args[:2]
    ):
        raise ValueError(
            f"{operation}: inputs must be at least one-dimensional"
        )
    if name == "solve" and (len(args[0].shape) < 2 or len(args[1].shape) < 1):
        raise ValueError(
            "linalg.solve: coefficient and right-hand arrays have invalid ranks"
        )
    if name == "lstsq" and (len(args[0].shape) < 2 or len(args[1].shape) < 1):
        raise ValueError(
            "linalg.lstsq: coefficient and right-hand arrays have invalid ranks"
        )
    if name == "cross":
        axis = _linalg_control(args, kwargs, "axis")
        normalized_axis = -1 if axis is _LINALG_MISSING else axis
        for value in args[:2]:
            if len(value.shape) < 1:
                raise ValueError(
                    "linalg.cross: inputs must be at least one-dimensional"
                )
            selected_axis = typing.cast(int, normalized_axis)
            if not -len(value.shape) <= selected_axis < len(value.shape):
                raise ValueError(f"{operation}: axis is out of bounds")
            selected_axis %= len(value.shape)
            if value.shape[selected_axis] != 3:
                raise ValueError(
                    "linalg.cross: selected axes must have size three"
                )


def _namespace(operation: str, *arrays: object) -> asc_typing.ArrayNamespace:
    """Select a namespace after enforcing the public array-only boundary."""
    if not arrays or any(
        not _array_api_compat.compat.is_array_api_obj(array) for array in arrays
    ):
        raise errors.NamespaceError(
            f"{operation}: every operand must be a native array"
        )
    return namespace_module.array_namespace(*arrays)


def _validate_bound_backend(
    expected: asc_typing.BackendName,
    operation: str,
    *values: object,
) -> asc_typing.ArrayNamespace:
    """Require every operand to be a native array from the bound backend."""
    if not values or any(
        not _array_api_compat.compat.is_array_api_obj(value) for value in values
    ):
        raise errors.NamespaceError(
            f"{operation}: every operand must be a native array"
        )
    xp = namespace_module.array_namespace(*values)
    observed = namespace_module.identify_backend(xp)
    if observed != expected:
        raise errors.MixedBackendError(
            f"{operation}: the {expected!r} facade cannot consume "
            f"{observed!r} arrays; convert explicitly"
        )
    return xp


def lstsq(
    first: object, second: object, *, rcond: object = None
) -> LstsqResult:
    """Solve least squares using the native backend of both inputs."""
    xp = _namespace("linalg.lstsq", first, second)
    backend = namespace_module.identify_backend(xp)
    namespace = xp.linalg
    if not isinstance(namespace, LinalgNamespace):
        namespace = LinalgNamespace(namespace, backend=backend)
    return namespace.lstsq(first, second, rcond=rcond)


def einsum(subscripts: str, *operands: object) -> object:
    """Evaluate Einstein summation without crossing backend boundaries."""
    if not operands:
        raise ValueError("einsum: at least one native operand is required")
    xp = _namespace("einsum", *operands)
    backend = namespace_module.identify_backend(xp)
    function = (
        None
        if backend == "array_api_strict"
        else getattr(_dispatch.load_backend(backend), "einsum", None)
    )
    if function is None:
        raise errors.CapabilityNotSupportedError(
            "einsum: selected backend has no native einsum implementation"
        )
    promoted = _promote_operands("einsum", xp, *operands)
    if backend == "torch" and xp.isdtype(promoted[0].dtype, "bool"):
        integer_operands = tuple(
            xp.astype(operand, xp.int64, copy=True) for operand in promoted
        )
        return function(subscripts, *integer_operands) != 0
    return function(subscripts, *promoted)


def _promote_operands(
    operation: str,
    xp: asc_typing.ArrayNamespace,
    *operands: object,
) -> tuple[object, ...]:
    """Cast native operands to their backend-neutral result dtype."""
    try:
        dtype = _dtype.extension_result_type(xp, *operands, operation=operation)
        return tuple(
            operand
            if getattr(operand, "dtype", None) == dtype
            else xp.astype(operand, dtype, copy=True)
            for operand in operands
        )
    except errors.DTypeError:
        raise
    except (RuntimeError, TypeError, ValueError) as exception:
        raise errors.DTypeError(
            f"{operation}: operand dtypes do not have a portable promotion"
        ) from exception


def kron(first: object, second: object) -> object:
    """Return the native Kronecker product of two arrays."""
    xp = _namespace("kron", first, second)
    backend = namespace_module.identify_backend(xp)
    function = (
        None
        if backend == "array_api_strict"
        else getattr(_dispatch.load_backend(backend), "kron", None)
    )
    if function is None:
        raise errors.CapabilityNotSupportedError(
            "kron: selected backend has no native kron implementation"
        )
    first, second = _promote_operands("kron", xp, first, second)
    if not xp.isdtype(first.dtype, "bool"):
        return function(first, second)
    rank = max(len(first.shape), len(second.shape))
    first_shape = (1,) * (rank - len(first.shape)) + tuple(first.shape)
    second_shape = (1,) * (rank - len(second.shape)) + tuple(second.shape)
    first = xp.reshape(first, first_shape)
    second = xp.reshape(second, second_shape)
    first_expanded = xp.reshape(
        first,
        tuple(value for extent in first_shape for value in (extent, 1)),
    )
    second_expanded = xp.reshape(
        second,
        tuple(value for extent in second_shape for value in (1, extent)),
    )
    combined = xp.logical_and(first_expanded, second_expanded)
    return xp.reshape(
        combined,
        tuple(
            first_extent * second_extent
            for first_extent, second_extent in zip(
                first_shape, second_shape, strict=True
            )
        ),
    )


def gkron(arrays: typing.Sequence[object]) -> tuple[object, ...]:
    """Build DeepMLT-style generalized Kronecker expansion matrices."""
    if not arrays:
        raise ValueError("gkron: arrays must be a non-empty sequence")
    xp = _namespace("gkron", *arrays)
    if any(len(array.shape) != 2 for array in arrays):
        raise ValueError("gkron: every input must be a two-dimensional array")
    result: list[object] = []
    for index, array in enumerate(arrays):
        expanded = array
        for other_index, other in enumerate(arrays):
            if other_index == index:
                continue
            ones = xp.ones(
                (other.shape[0], 1),
                dtype=array.dtype,
                device=_array_api_compat.compat.device(array),
            )
            expanded = (
                kron(expanded, ones)
                if index < other_index
                else kron(ones, expanded)
            )
        result.append(expanded)
    return tuple(result)


__all__ = [
    "EigResult",
    "EighResult",
    "LinalgNamespace",
    "LstsqResult",
    "QRResult",
    "SVDResult",
    "SlogdetResult",
    "einsum",
    "gkron",
    "kron",
    "linalg_namespace",
    "lstsq",
]
