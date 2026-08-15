# Copyright 2026 AI4SciComp contributors.
# SPDX-License-Identifier: Apache-2.0
"""Shared validation wrapper for backend-bound namespace functions."""

from __future__ import annotations

import collections.abc
import contextlib
import contextvars
import dataclasses
import functools
import inspect
import math
import sys
import typing

from asc import _array_api_compat, errors
from asc import typing as asc_typing
from asc.core import _dtype
from asc.core._array_api import MANDATORY_SYMBOLS
from asc.core._scalar import require_representable_scalar

_DEVICE_CREATION_FUNCTIONS: typing.Final = frozenset(
    {
        "arange",
        "asarray",
        "astype",
        "empty",
        "empty_like",
        "eye",
        "full",
        "full_like",
        "from_dlpack",
        "linspace",
        "ones",
        "ones_like",
        "zeros",
        "zeros_like",
    }
)
_DEVICE_PRESERVING_CREATION_FUNCTIONS: typing.Final = frozenset(
    {
        "astype",
        "empty_like",
        "from_dlpack",
        "full_like",
        "ones_like",
        "zeros_like",
    }
)
_TRUSTED_DLPACK_CONVERSION: contextvars.ContextVar[bool] = (
    contextvars.ContextVar("asc_trusted_dlpack_conversion", default=False)
)
_ARRAY_FORBIDDEN_CONTROL_PARAMETERS: typing.Final = frozenset(
    {
        "M",
        "N",
        "axes",
        "axis",
        "correction",
        "descending",
        "endpoint",
        "fill_value",
        "include_initial",
        "indexing",
        "k",
        "kind",
        "keepdims",
        "n",
        "n_cols",
        "n_rows",
        "num",
        "repetitions",
        "reps",
        "shape",
        "shift",
        "source",
        "stable",
        "start",
        "step",
        "stop",
        "destination",
    }
)
_BOOLEAN_CONTROL_PARAMETERS: typing.Final = frozenset(
    {
        "descending",
        "endpoint",
        "include_initial",
        "keepdims",
        "stable",
    }
)
_FORBIDDEN_NATIVE_EXTENSION_KEYWORDS: typing.Final = frozenset(
    {
        "casting",
        "decimals",
        "initial",
        "mode",
        "order",
        "out",
        "signature",
        "sparse",
        "subok",
        "where",
    }
)


class _NativeNamespaceInspection(typing.Protocol):
    """Structural type for standard namespace inspection objects."""

    def capabilities(self) -> collections.abc.Mapping[str, object]:
        """Return namespace capability declarations."""
        ...  # pylint: disable=unnecessary-ellipsis

    def default_device(self) -> object:
        """Return the native namespace default device."""
        ...  # pylint: disable=unnecessary-ellipsis

    def default_dtypes(self, *, device: object | None = None) -> object:
        """Return native default dtype metadata."""
        ...  # pylint: disable=unnecessary-ellipsis

    def dtypes(
        self,
        *,
        device: object | None = None,
        kind: str | tuple[str, ...] | None = None,
    ) -> object:
        """Return native supported dtype metadata."""
        ...  # pylint: disable=unnecessary-ellipsis


@dataclasses.dataclass(frozen=True, slots=True)
class _NamespaceInspection:
    """Expose metadata constrained to one ASC backend device surface."""

    inspection: _NativeNamespaceInspection
    namespace: asc_typing.ArrayNamespace
    backend: asc_typing.BackendName
    dtype_validator: typing.Callable[[object], None]
    device_resolver: typing.Callable[[object | None], object | None]

    def capabilities(self) -> dict[str, object]:
        """Return the native standard capability declarations."""
        return dict(self.inspection.capabilities())

    def default_device(self) -> object:
        """Return the backend's sole usable CPU device identifier."""
        resolved = self.device_resolver(None)
        if resolved is not None:
            return resolved
        # Pylint misclassifies the typed Protocol stub as a procedure.
        # pylint: disable-next=assignment-from-no-return
        device = self.inspection.default_device()
        self.device_resolver(device)
        return device

    def _validated_device(self, device: object | None) -> object:
        """Normalize an inspection query to the advertised CPU device."""
        if device is None:
            return self.default_device()
        resolved = self.device_resolver(device)
        return device if resolved is None else resolved

    def default_dtypes(self, *, device: object | None = None) -> object:
        """Return default dtypes for the validated CPU device."""
        return self.inspection.default_dtypes(
            device=self._validated_device(device)
        )

    def devices(self) -> list[object]:
        """Return only devices accepted by the backend-bound namespace."""
        return [self.default_device()]

    def dtypes(
        self,
        *,
        device: object | None = None,
        kind: str | tuple[str, ...] | None = None,
    ) -> dict[str, object]:
        """Return native and ASC-emulated dtypes on the active surface."""
        reported = typing.cast(
            collections.abc.Mapping[str, object],
            self.inspection.dtypes(
                device=self._validated_device(device), kind=kind
            ),
        )
        result = dict(reported)
        for name in _dtype.supported_dtype_names(
            self.backend, jax_x64_enabled=True
        ):
            dtype = getattr(self.namespace, name, None)
            if dtype is None:
                continue
            try:
                self.dtype_validator(dtype)
                matches = kind is None or self.namespace.isdtype(dtype, kind)
            except (errors.DTypeError, AttributeError, TypeError, ValueError):
                continue
            if matches:
                result[name] = dtype
        return result


_AXIS_INT_OR_NONE_FUNCTIONS: typing.Final = frozenset(
    {
        "argmax",
        "argmin",
        "concat",
        "cumulative_prod",
        "cumulative_sum",
        "repeat",
        "take",
    }
)
_AXIS_INT_FUNCTIONS: typing.Final = frozenset(
    {
        "argsort",
        "diff",
        "expand_dims",
        "sort",
        "stack",
        "take_along_axis",
        "unstack",
        "vecdot",
    }
)
_AXIS_SEQUENCE_OR_NONE_FUNCTIONS: typing.Final = frozenset(
    {
        "all",
        "any",
        "count_nonzero",
        "flip",
        "max",
        "mean",
        "min",
        "prod",
        "roll",
        "std",
        "sum",
        "var",
    }
)
_CREATION_SHAPE_FUNCTIONS: typing.Final = frozenset(
    {"empty", "full", "ones", "zeros"}
)
_ARRAY_ONLY_FIRST_FUNCTIONS: typing.Final = frozenset(
    {
        "abs",
        "acos",
        "acosh",
        "all",
        "any",
        "argmax",
        "argmin",
        "argsort",
        "asin",
        "asinh",
        "astype",
        "atan",
        "atanh",
        "bitwise_invert",
        "broadcast_to",
        "ceil",
        "clip",
        "conj",
        "cos",
        "cosh",
        "count_nonzero",
        "cumulative_prod",
        "cumulative_sum",
        "diff",
        "empty_like",
        "exp",
        "expand_dims",
        "expm1",
        "flip",
        "floor",
        "full_like",
        "imag",
        "isfinite",
        "isinf",
        "isnan",
        "log",
        "log10",
        "log1p",
        "log2",
        "logical_not",
        "matrix_transpose",
        "max",
        "mean",
        "min",
        "negative",
        "nonzero",
        "ones_like",
        "permute_dims",
        "positive",
        "prod",
        "real",
        "reciprocal",
        "repeat",
        "reshape",
        "roll",
        "round",
        "searchsorted",
        "sign",
        "signbit",
        "sin",
        "sinh",
        "sort",
        "sqrt",
        "square",
        "squeeze",
        "std",
        "sum",
        "take",
        "take_along_axis",
        "tan",
        "tanh",
        "tile",
        "tril",
        "triu",
        "trunc",
        "unique_all",
        "unique_counts",
        "unique_inverse",
        "unique_values",
        "unstack",
        "var",
        "where",
        "zeros_like",
    }
)
_ARRAY_ONLY_BINARY_FUNCTIONS: typing.Final = frozenset(
    {
        "matmul",
        "searchsorted",
        "take",
        "take_along_axis",
        "tensordot",
        "vecdot",
    }
)
_SCALAR_ENABLED_BINARY_FUNCTIONS: typing.Final = frozenset(
    {
        "add",
        "atan2",
        "bitwise_and",
        "bitwise_left_shift",
        "bitwise_or",
        "bitwise_right_shift",
        "bitwise_xor",
        "copysign",
        "divide",
        "equal",
        "floor_divide",
        "greater",
        "greater_equal",
        "hypot",
        "less",
        "less_equal",
        "logaddexp",
        "logical_and",
        "logical_or",
        "logical_xor",
        "maximum",
        "minimum",
        "multiply",
        "nextafter",
        "not_equal",
        "pow",
        "remainder",
        "subtract",
    }
)
_STANDARD_VARIADIC_POSITIONAL_FUNCTIONS: typing.Final = frozenset(
    {"broadcast_arrays", "meshgrid", "result_type"}
)
_STANDARD_TWO_POSITIONAL_FUNCTIONS: typing.Final = (
    _ARRAY_ONLY_BINARY_FUNCTIONS
    | _SCALAR_ENABLED_BINARY_FUNCTIONS
    | frozenset(
        {
            "astype",
            "broadcast_to",
            "can_cast",
            "expand_dims",
            "eye",
            "full",
            "full_like",
            "isdtype",
            "permute_dims",
            "repeat",
            "reshape",
            "roll",
            "squeeze",
            "tile",
        }
    )
)
_STANDARD_THREE_POSITIONAL_FUNCTIONS: typing.Final = frozenset(
    {"arange", "clip", "linspace", "moveaxis", "where"}
)
_STANDARD_KEYWORD_PARAMETERS: typing.Final = {
    "all": frozenset({"axis", "keepdims"}),
    "any": frozenset({"axis", "keepdims"}),
    "arange": frozenset({"stop", "step", "dtype", "device"}),
    "argmax": frozenset({"axis", "keepdims"}),
    "argmin": frozenset({"axis", "keepdims"}),
    "argsort": frozenset({"axis", "descending", "stable"}),
    "asarray": frozenset({"dtype", "device", "copy"}),
    "astype": frozenset({"copy", "device"}),
    "broadcast_to": frozenset({"shape"}),
    "clip": frozenset({"min", "max"}),
    "concat": frozenset({"axis"}),
    "count_nonzero": frozenset({"axis", "keepdims"}),
    "cumulative_prod": frozenset({"axis", "dtype", "include_initial"}),
    "cumulative_sum": frozenset({"axis", "dtype", "include_initial"}),
    "diff": frozenset({"axis", "n", "prepend", "append"}),
    "empty": frozenset({"shape", "dtype", "device"}),
    "empty_like": frozenset({"dtype", "device"}),
    "expand_dims": frozenset({"axis"}),
    "eye": frozenset({"k", "dtype", "device"}),
    "flip": frozenset({"axis"}),
    "from_dlpack": frozenset({"device", "copy"}),
    "full": frozenset({"shape", "fill_value", "dtype", "device"}),
    "full_like": frozenset({"fill_value", "dtype", "device"}),
    "isdtype": frozenset({"dtype", "kind"}),
    "linspace": frozenset({"num", "dtype", "device", "endpoint"}),
    "max": frozenset({"axis", "keepdims"}),
    "mean": frozenset({"axis", "keepdims"}),
    "meshgrid": frozenset({"indexing"}),
    "min": frozenset({"axis", "keepdims"}),
    "ones": frozenset({"shape", "dtype", "device"}),
    "ones_like": frozenset({"dtype", "device"}),
    "permute_dims": frozenset({"axes"}),
    "prod": frozenset({"axis", "dtype", "keepdims"}),
    "repeat": frozenset({"axis"}),
    "reshape": frozenset({"shape", "copy"}),
    "roll": frozenset({"shift", "axis"}),
    "searchsorted": frozenset({"side", "sorter"}),
    "sort": frozenset({"axis", "descending", "stable"}),
    "squeeze": frozenset({"axis"}),
    "stack": frozenset({"axis"}),
    "std": frozenset({"axis", "correction", "keepdims"}),
    "sum": frozenset({"axis", "dtype", "keepdims"}),
    "take": frozenset({"axis"}),
    "take_along_axis": frozenset({"axis"}),
    "tensordot": frozenset({"axes"}),
    "tril": frozenset({"k"}),
    "triu": frozenset({"k"}),
    "unstack": frozenset({"axis"}),
    "var": frozenset({"axis", "correction", "keepdims"}),
    "vecdot": frozenset({"axis"}),
    "zeros": frozenset({"shape", "dtype", "device"}),
    "zeros_like": frozenset({"dtype", "device"}),
}
_DTYPE_QUERY_KINDS: typing.Final = frozenset(
    {
        "bool",
        "complex floating",
        "integral",
        "numeric",
        "real floating",
        "signed integer",
        "unsigned integer",
    }
)
_BOOLEAN_DTYPE_NAMES: typing.Final = frozenset({"bool"})
_SIGNED_INTEGER_BITS: typing.Final = {
    "int8": 8,
    "int16": 16,
    "int32": 32,
    "int64": 64,
}
_UNSIGNED_INTEGER_BITS: typing.Final = {
    "uint8": 8,
    "uint16": 16,
    "uint32": 32,
    "uint64": 64,
}
_INTEGER_DTYPE_NAMES: typing.Final = frozenset(
    {*_SIGNED_INTEGER_BITS, *_UNSIGNED_INTEGER_BITS}
)
_REAL_FLOAT_DTYPE_NAMES: typing.Final = frozenset(
    {"bfloat16", "float16", "float32", "float64"}
)
_COMPLEX_DTYPE_NAMES: typing.Final = frozenset({"complex64", "complex128"})
_FLOAT_DTYPE_NAMES: typing.Final = (
    _REAL_FLOAT_DTYPE_NAMES | _COMPLEX_DTYPE_NAMES
)
_NUMERIC_DTYPE_NAMES: typing.Final = _INTEGER_DTYPE_NAMES | _FLOAT_DTYPE_NAMES
_REAL_NUMERIC_DTYPE_NAMES: typing.Final = (
    _INTEGER_DTYPE_NAMES | _REAL_FLOAT_DTYPE_NAMES
)
_PROMOTION_TABLE: typing.Final[dict[frozenset[str], str]] = {
    frozenset(("int8", "int16")): "int16",
    frozenset(("int8", "int32")): "int32",
    frozenset(("int8", "int64")): "int64",
    frozenset(("int16", "int32")): "int32",
    frozenset(("int16", "int64")): "int64",
    frozenset(("int32", "int64")): "int64",
    frozenset(("uint8", "uint16")): "uint16",
    frozenset(("uint8", "uint32")): "uint32",
    frozenset(("uint8", "uint64")): "uint64",
    frozenset(("uint16", "uint32")): "uint32",
    frozenset(("uint16", "uint64")): "uint64",
    frozenset(("uint32", "uint64")): "uint64",
    frozenset(("int8", "uint8")): "int16",
    frozenset(("int8", "uint16")): "int32",
    frozenset(("int8", "uint32")): "int64",
    frozenset(("int16", "uint8")): "int16",
    frozenset(("int16", "uint16")): "int32",
    frozenset(("int16", "uint32")): "int64",
    frozenset(("int32", "uint8")): "int32",
    frozenset(("int32", "uint16")): "int32",
    frozenset(("int32", "uint32")): "int64",
    frozenset(("int64", "uint8")): "int64",
    frozenset(("int64", "uint16")): "int64",
    frozenset(("int64", "uint32")): "int64",
    frozenset(("float16", "bfloat16")): "float32",
    frozenset(("float16", "float32")): "float32",
    frozenset(("bfloat16", "float32")): "float32",
    frozenset(("float16", "float64")): "float64",
    frozenset(("bfloat16", "float64")): "float64",
    frozenset(("float32", "float64")): "float64",
    frozenset(("float16", "complex64")): "complex64",
    frozenset(("bfloat16", "complex64")): "complex64",
    frozenset(("float32", "complex64")): "complex64",
    frozenset(("float64", "complex64")): "complex128",
    frozenset(("float16", "complex128")): "complex128",
    frozenset(("bfloat16", "complex128")): "complex128",
    frozenset(("float32", "complex128")): "complex128",
    frozenset(("float64", "complex128")): "complex128",
    frozenset(("complex64", "complex128")): "complex128",
}
_DTYPE_CATEGORIES: typing.Final = {
    "all": _BOOLEAN_DTYPE_NAMES | _NUMERIC_DTYPE_NAMES,
    "boolean": _BOOLEAN_DTYPE_NAMES,
    "complex floating": _COMPLEX_DTYPE_NAMES,
    "floating": _FLOAT_DTYPE_NAMES,
    "integer": _INTEGER_DTYPE_NAMES,
    "integer or boolean": _INTEGER_DTYPE_NAMES | _BOOLEAN_DTYPE_NAMES,
    "numeric": _NUMERIC_DTYPE_NAMES,
    "real floating": _REAL_FLOAT_DTYPE_NAMES,
    "real numeric": _REAL_NUMERIC_DTYPE_NAMES,
}
_UNARY_DTYPE_CATEGORIES: typing.Final = {
    **dict.fromkeys(
        {
            "acos",
            "acosh",
            "asin",
            "asinh",
            "atan",
            "atanh",
            "cos",
            "cosh",
            "exp",
            "expm1",
            "log",
            "log10",
            "log1p",
            "log2",
            "mean",
            "reciprocal",
            "sin",
            "sinh",
            "sqrt",
            "tan",
            "tanh",
        },
        "floating",
    ),
    **dict.fromkeys(
        {
            "abs",
            "conj",
            "cumulative_prod",
            "cumulative_sum",
            "diff",
            "isfinite",
            "isinf",
            "isnan",
            "negative",
            "positive",
            "prod",
            "real",
            "round",
            "sign",
            "square",
            "sum",
        },
        "numeric",
    ),
    **dict.fromkeys(
        {
            "argmax",
            "argmin",
            "argsort",
            "ceil",
            "clip",
            "floor",
            "max",
            "min",
            "sort",
            "trunc",
        },
        "real numeric",
    ),
    "bitwise_invert": "integer or boolean",
    "imag": "complex floating",
    "logical_not": "boolean",
    "signbit": "real floating",
    "std": "real floating",
    "var": "real floating",
}
_BINARY_DTYPE_CATEGORIES: typing.Final = {
    "add": "numeric",
    "atan2": "real floating",
    "bitwise_and": "integer or boolean",
    "bitwise_left_shift": "integer",
    "bitwise_or": "integer or boolean",
    "bitwise_right_shift": "integer",
    "bitwise_xor": "integer or boolean",
    "copysign": "real floating",
    "divide": "floating",
    "equal": "all",
    "floor_divide": "real numeric",
    "greater": "real numeric",
    "greater_equal": "real numeric",
    "hypot": "real floating",
    "less": "real numeric",
    "less_equal": "real numeric",
    "logaddexp": "real floating",
    "logical_and": "boolean",
    "logical_or": "boolean",
    "logical_xor": "boolean",
    "maximum": "real numeric",
    "minimum": "real numeric",
    "multiply": "numeric",
    "nextafter": "real floating",
    "not_equal": "all",
    "pow": "numeric",
    "remainder": "real numeric",
    "subtract": "numeric",
}
_NONPROMOTING_BINARY_DTYPE_CATEGORIES: typing.Final = {
    "matmul": "numeric",
    "tensordot": "numeric",
    "vecdot": "numeric",
}
_BOOLEAN_RESULT_FUNCTIONS: typing.Final = frozenset(
    {
        "equal",
        "greater",
        "greater_equal",
        "less",
        "less_equal",
        "logical_and",
        "logical_or",
        "logical_xor",
        "not_equal",
    }
)
_MISSING: typing.Final = object()


def require_dtype_category(operation: str, value: object, category: str) -> str:
    """Require one native operand to belong to an Array API dtype family."""
    name = _dtype.dtype_name(getattr(value, "dtype", None))
    if name not in _DTYPE_CATEGORIES[category]:
        raise errors.DTypeError(
            f"{operation}: operand dtype must be {category}"
        )
    return typing.cast(str, name)


def _promote_dtype_names(first: str, second: str, operation: str) -> str:
    """Apply the frozen Array API promotion lattice to two dtype names."""
    if first == second:
        return first
    result = _PROMOTION_TABLE.get(frozenset((first, second)))
    if result is None:
        raise errors.DTypeError(
            f"{operation}: operand dtypes {first!r} and {second!r} do not "
            "have a portable promotion"
        )
    return result


def _integer_bounds(name: str) -> tuple[int, int]:
    bits = (
        _SIGNED_INTEGER_BITS[name]
        if name in _SIGNED_INTEGER_BITS
        else _UNSIGNED_INTEGER_BITS[name]
    )
    if name in _SIGNED_INTEGER_BITS:
        return -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
    return 0, 2**bits - 1


def _promote_python_scalar(
    dtype_name: str, scalar: object, operation: str
) -> str:
    """Apply the 2024.12 weak Python-scalar promotion rules."""
    if type(scalar) is bool:
        if dtype_name != "bool":
            raise errors.DTypeError(
                f"{operation}: a Boolean scalar requires a Boolean array"
            )
        return dtype_name
    if type(scalar) is int:
        if dtype_name == "bool":
            raise errors.DTypeError(
                f"{operation}: an integer scalar cannot promote with Boolean"
            )
        if dtype_name in _INTEGER_DTYPE_NAMES:
            lower, upper = _integer_bounds(dtype_name)
            if not lower <= scalar <= upper:
                raise errors.DTypeError(
                    f"{operation}: integer scalar is outside the array dtype"
                )
        return dtype_name
    if type(scalar) is float:
        if dtype_name not in _FLOAT_DTYPE_NAMES:
            raise errors.DTypeError(
                f"{operation}: a real scalar requires a floating array"
            )
        return dtype_name
    if type(scalar) is complex:
        if dtype_name not in _FLOAT_DTYPE_NAMES:
            raise errors.DTypeError(
                f"{operation}: a complex scalar requires a floating array"
            )
        if dtype_name in _REAL_FLOAT_DTYPE_NAMES:
            return "complex128" if dtype_name == "float64" else "complex64"
        return dtype_name
    raise errors.DTypeError(
        f"{operation}: operand must be a native array or Python scalar"
    )


def require_portable_promotion(operation: str, *operands: object) -> str:
    """Return the portable result dtype for arrays, dtypes, and scalars."""
    dtype_names: list[str] = []
    scalars: list[object] = []
    for operand in operands:
        if _is_direct_native_array(operand):
            name = _dtype.dtype_name(getattr(operand, "dtype", None))
        elif type(operand) in {bool, int, float, complex}:
            scalars.append(operand)
            continue
        else:
            name = _dtype.dtype_name(operand)
        if name is None:
            raise errors.DTypeError(
                f"{operation}: operand has no supported native dtype"
            )
        dtype_names.append(name)
    if not dtype_names:
        raise errors.DTypeError(
            f"{operation}: at least one native array or dtype is required"
        )
    result = dtype_names[0]
    for name in dtype_names[1:]:
        result = _promote_dtype_names(result, name, operation)
    for scalar in scalars:
        result = _promote_python_scalar(result, scalar, operation)
    return result


def _native_array_leaves(
    value: object, seen: set[int] | None = None
) -> tuple[object, ...]:
    """Find array operands without mistaking native dtype classes for arrays."""
    if isinstance(value, type):
        return ()
    if _array_api_compat.compat.is_array_api_obj(value):
        return (value,)
    if seen is None:
        seen = set()
    marker = id(value)
    if marker in seen:
        return ()
    if isinstance(value, collections.abc.Mapping):
        seen.add(marker)
        return tuple(
            leaf
            for item in value.values()
            for leaf in _native_array_leaves(item, seen)
        )
    if isinstance(value, collections.abc.Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        seen.add(marker)
        return tuple(
            leaf for item in value for leaf in _native_array_leaves(item, seen)
        )
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        seen.add(marker)
        return tuple(
            leaf
            for field in dataclasses.fields(value)
            for leaf in _native_array_leaves(getattr(value, field.name), seen)
        )
    return ()


def _is_direct_native_array(value: object) -> bool:
    """Return whether one operand is a native-array candidate."""
    if isinstance(value, type):
        return False
    return _array_api_compat.compat.is_array_api_obj(value)


def _validate_required_array_operands(
    backend: asc_typing.BackendName,
    name: str,
    args: tuple[object, ...],
) -> None:
    """Enforce the standard's direct native-array operand positions."""
    operation = f"{backend} namespace.{name}"
    if name in _ARRAY_ONLY_FIRST_FUNCTIONS and (
        not args or not _is_direct_native_array(args[0])
    ):
        raise errors.NamespaceError(
            f"{operation}: the first operand must be a native array"
        )
    if name in _ARRAY_ONLY_BINARY_FUNCTIONS and (
        len(args) < 2
        or not all(_is_direct_native_array(value) for value in args[:2])
    ):
        raise errors.NamespaceError(
            f"{operation}: both operands must be native arrays"
        )
    if name in _SCALAR_ENABLED_BINARY_FUNCTIONS:
        if len(args) < 2:
            raise errors.NamespaceError(
                f"{operation}: two array or Python scalar operands are required"
            )
        valid_operands = all(
            _is_direct_native_array(value)
            or type(value) in {bool, int, float, complex}
            for value in args[:2]
        )
        if not valid_operands or not any(
            _is_direct_native_array(value) for value in args[:2]
        ):
            raise errors.NamespaceError(
                f"{operation}: at least one operand must be a native array"
            )
    if name == "where":
        if len(args) < 3 or not _is_direct_native_array(args[0]):
            raise errors.NamespaceError(
                f"{operation}: condition must be a native array"
            )
        choices = args[1:3]
        if not all(
            _is_direct_native_array(value)
            or type(value) in {bool, int, float, complex}
            for value in choices
        ) or not any(_is_direct_native_array(value) for value in choices):
            raise errors.NamespaceError(
                f"{operation}: choices must be arrays or Python scalars and "
                "at least one choice must be a native array"
            )
    if name in {"concat", "stack"}:
        if (
            not args
            or not isinstance(args[0], (tuple, list))
            or not args[0]
            or not all(_is_direct_native_array(value) for value in args[0])
        ):
            raise errors.NamespaceError(
                f"{operation}: arrays must be a non-empty tuple or list of "
                "native arrays"
            )
    elif (
        name == "broadcast_arrays"
        and (
            not args
            or not all(_is_direct_native_array(value) for value in args)
        )
    ) or (
        name == "meshgrid"
        and not all(_is_direct_native_array(value) for value in args)
    ):
        raise errors.NamespaceError(
            f"{operation}: every operand must be a native array"
        )


def _require_promoted_category(
    operation: str, category: str, *operands: object
) -> str:
    """Validate operand families and their portable promoted result."""
    for operand in operands:
        if _is_direct_native_array(operand):
            require_dtype_category(operation, operand, category)
    result = require_portable_promotion(operation, *operands)
    if result not in _DTYPE_CATEGORIES[category]:
        raise errors.DTypeError(
            f"{operation}: promoted dtype must be {category}"
        )
    return result


def _argument_value(
    args: tuple[object, ...],
    kwargs: dict[str, object],
    name: str,
    position: int,
) -> object:
    if name in kwargs:
        return kwargs[name]
    return args[position] if len(args) > position else _MISSING


def _validate_clip_dtypes(
    args: tuple[object, ...], kwargs: dict[str, object]
) -> str:
    """Validate clip bounds without allowing backend-native coercion."""
    operation = "clip"
    array = args[0]
    source_name = require_dtype_category(operation, array, "real numeric")
    for parameter, position in (("min", 1), ("max", 2)):
        bound = _argument_value(args, kwargs, parameter, position)
        if bound is _MISSING or bound is None:
            continue
        if _is_direct_native_array(bound):
            require_dtype_category(operation, bound, "real numeric")
            promoted = require_portable_promotion(operation, array, bound)
            if promoted not in _REAL_NUMERIC_DTYPE_NAMES:
                raise errors.DTypeError(
                    "clip: array bounds must have a compatible real dtype"
                )
        elif type(bound) not in {int, float}:
            raise errors.DTypeError(
                f"clip: {parameter} must be a native array, Python real "
                "scalar, or None"
            )
        elif source_name in _INTEGER_DTYPE_NAMES and type(bound) is not int:
            raise errors.DTypeError(
                f"clip: {parameter} must be integral when the input is integral"
            )
    return source_name


def _validate_take_shape(
    args: tuple[object, ...], kwargs: dict[str, object]
) -> None:
    """Enforce the standard one-dimensional ``take`` index contract."""
    axis = _argument_value(args, kwargs, "axis", 2)
    if len(args[1].shape) != 1:
        raise ValueError("take: indices must be one-dimensional")
    if (axis is _MISSING or axis is None) and len(args[0].shape) != 1:
        raise ValueError(
            "take: axis must be specified unless the input is one-dimensional"
        )


def _validate_repeat_shape(
    args: tuple[object, ...], kwargs: dict[str, object], repeats: object
) -> None:
    """Require array repeat counts to broadcast to one selected axis."""
    if len(repeats.shape) > 1:
        raise ValueError("repeat: repeats must be zero- or one-dimensional")
    if len(repeats.shape) == 0:
        return
    axis = _argument_value(args, kwargs, "axis", 2)
    expected = math.prod(args[0].shape)
    if axis is not _MISSING and axis is not None:
        rank = len(args[0].shape)
        if (
            isinstance(axis, int)
            and not isinstance(axis, bool)
            and -rank <= axis < rank
        ):
            expected = args[0].shape[axis]
        else:
            return
    if repeats.shape[0] not in {1, expected}:
        raise ValueError("repeat: repeats must broadcast to the selected axis")


def take_axis_extent(
    name: str,
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> int | None:
    """Return the selected axis extent when its control is valid."""
    axis = _argument_value(args, kwargs, "axis", 2)
    if name == "take" and (axis is _MISSING or axis is None):
        return typing.cast(int, args[0].shape[0])
    if not isinstance(axis, int) or isinstance(axis, bool):
        return None
    rank = len(args[0].shape)
    if not -rank <= axis < rank:
        return None
    return typing.cast(int, args[0].shape[axis])


def namespace_argument_value(
    args: tuple[object, ...],
    kwargs: dict[str, object],
    parameter: str,
    position: int,
) -> object:
    """Return one positional-or-keyword namespace argument."""
    return _argument_value(args, kwargs, parameter, position)


def is_integer_dtype_name(name: str | None) -> bool:
    """Return whether a canonical dtype name is integral."""
    return name in _INTEGER_DTYPE_NAMES


def is_signed_integer_array(value: object) -> bool:
    """Return whether a value has a supported signed integer dtype."""
    return (
        _is_direct_native_array(value)
        and _dtype.dtype_name(getattr(value, "dtype", None))
        in _SIGNED_INTEGER_BITS
    )


def _validate_scalar_namespace_values(
    name: str,
    args: tuple[object, ...],
    kwargs: dict[str, object],
    expected_result_dtype: str | None,
) -> None:
    """Reject invalid Python scalar values before backend dispatch."""
    if name == "pow" and expected_result_dtype in _INTEGER_DTYPE_NAMES:
        exponent = args[1]
        if not _is_direct_native_array(exponent) and exponent < 0:
            raise ValueError("pow: integer exponents must be non-negative")
    if name in {"bitwise_left_shift", "bitwise_right_shift"}:
        shifts = args[1]
        if not _is_direct_native_array(shifts) and shifts < 0:
            raise ValueError(f"{name}: shift counts must be non-negative")
    if name == "repeat":
        repeats = _argument_value(args, kwargs, "repeats", 1)
        if not _is_direct_native_array(repeats) and repeats < 0:
            raise ValueError("repeat: repeat counts must be non-negative")


def _raise_concrete_invalid(
    invalid: object,
    operation: str,
    message: str,
    *,
    index_error: bool = False,
) -> None:
    """Raise a stable eager error for a scalar invalidity predicate."""
    if not bool(invalid):
        return
    if index_error:
        raise IndexError(f"{operation}: {message}")
    raise ValueError(f"{operation}: {message}")


def _validate_concrete_namespace_values(
    namespace: object,
    name: str,
    args: tuple[object, ...],
    kwargs: dict[str, object],
    expected_result_dtype: str | None,
) -> None:
    """Validate dynamic values for eager namespaces such as NumPy."""
    any_value = typing.cast(typing.Callable[[object], object], namespace.any)
    less = typing.cast(
        typing.Callable[[object, object], object], namespace.less
    )
    greater_equal = typing.cast(
        typing.Callable[[object, object], object], namespace.greater_equal
    )
    logical_or = typing.cast(
        typing.Callable[[object, object], object], namespace.logical_or
    )
    if name in {"take", "take_along_axis"}:
        extent = take_axis_extent(name, args, kwargs)
        if extent is not None:
            indices = args[1]
            invalid = greater_equal(indices, extent)
            if is_signed_integer_array(indices):
                invalid = logical_or(invalid, less(indices, -extent))
            _raise_concrete_invalid(
                any_value(invalid),
                name,
                "index is out of bounds",
                index_error=True,
            )
    value: object | None = None
    message = ""
    if name == "pow" and expected_result_dtype in _INTEGER_DTYPE_NAMES:
        value = args[1]
        message = "integer exponents must be non-negative"
    elif name in {"bitwise_left_shift", "bitwise_right_shift"}:
        value = args[1]
        message = "shift counts must be non-negative"
    elif name == "repeat":
        value = _argument_value(args, kwargs, "repeats", 1)
        message = "repeat counts must be non-negative"
    if is_signed_integer_array(value):
        _raise_concrete_invalid(any_value(less(value, 0)), name, message)


def _validate_namespace_dtypes(
    name: str,
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> str | None:
    """Validate 2024.12 dtype domains and promotions before dispatch."""
    if name in _UNARY_DTYPE_CATEGORIES:
        category = _UNARY_DTYPE_CATEGORIES[name]
        require_dtype_category(name, args[0], category)
    if name in _BINARY_DTYPE_CATEGORIES:
        category = _BINARY_DTYPE_CATEGORIES[name]
        promoted = _require_promoted_category(name, category, *args[:2])
        return "bool" if name in _BOOLEAN_RESULT_FUNCTIONS else promoted
    if name in _NONPROMOTING_BINARY_DTYPE_CATEGORIES:
        category = _NONPROMOTING_BINARY_DTYPE_CATEGORIES[name]
        for operand in args[:2]:
            require_dtype_category(name, operand, category)
        try:
            return _require_promoted_category(name, category, *args[:2])
        except errors.DTypeError:
            # The standard permits implementation-defined mixed-kind linalg
            # promotion. Enforce the frozen lattice whenever it has a result.
            return None
    if name == "clip":
        return _validate_clip_dtypes(args, kwargs)
    if name == "where":
        require_dtype_category(name, args[0], "boolean")
        return _require_promoted_category(name, "all", *args[1:3])
    if name == "searchsorted":
        for operand in args[:2]:
            require_dtype_category(name, operand, "real numeric")
        if len(args[0].shape) != 1:
            raise ValueError(
                "searchsorted: sorted boundaries must be one-dimensional"
            )
        sorter = _argument_value(args, kwargs, "sorter", len(args) + 1)
        if sorter is not _MISSING and sorter is not None:
            require_dtype_category(name, sorter, "integer")
    if name == "take":
        require_dtype_category(name, args[1], "integer")
        _validate_take_shape(args, kwargs)
    if name == "take_along_axis":
        require_dtype_category(name, args[1], "integer")
    if name == "repeat":
        repeats = _argument_value(args, kwargs, "repeats", 1)
        if _is_direct_native_array(repeats):
            require_dtype_category(name, repeats, "integer")
            _validate_repeat_shape(args, kwargs, repeats)
    if name in {"concat", "stack"}:
        return require_portable_promotion(
            name, *typing.cast(tuple[object, ...], args[0])
        )
    if (
        name == "meshgrid"
        and len(
            {_dtype.dtype_name(getattr(array, "dtype", None)) for array in args}
        )
        > 1
    ):
        raise errors.DTypeError(
            "meshgrid: every input must have the same dtype"
        )
    if name == "astype":
        source_name = _dtype.dtype_name(getattr(args[0], "dtype", None))
        destination = _argument_value(args, kwargs, "dtype", 1)
        destination_name = _dtype.dtype_name(destination)
        if (
            source_name in _COMPLEX_DTYPE_NAMES
            and destination_name not in _COMPLEX_DTYPE_NAMES
        ):
            raise errors.DTypeError(
                "astype: complex arrays cannot be cast to non-complex dtypes"
            )
    return None


def _is_numpy_scalar(value: object) -> bool:
    """Return whether ``value`` is scalar NumPy data, not an ndarray."""
    return any(
        base.__module__.startswith("numpy") and base.__name__ == "generic"
        for base in type(value).__mro__
    )


def _validate_inferred_numpy_scalar_dtypes(
    backend: asc_typing.BackendName,
    values: tuple[object, ...],
) -> None:
    """Reject typed construction data that the backend would silently narrow."""
    supported = _dtype.supported_dtype_names(
        backend,
        jax_x64_enabled=(backend != "jax" or _dtype.active_jax_x64_enabled()),
    )
    for value in values:
        name = _dtype.dtype_name(value.dtype)
        if name not in supported:
            raise errors.DTypeError(
                f"{backend} namespace.asarray: inferred NumPy scalar dtype "
                f"{name!r} is outside the active release surface; request "
                "an explicit supported dtype"
            )


@contextlib.contextmanager
def trusted_dlpack_conversion() -> typing.Iterator[None]:
    """Allow an explicit conversion boundary to perform its validated import."""
    token = _TRUSTED_DLPACK_CONVERSION.set(True)
    try:
        yield
    finally:
        _TRUSTED_DLPACK_CONVERSION.reset(token)


def _validate_dlpack_producer(
    value: object, backend: asc_typing.BackendName
) -> str | None:
    """Prove DLPack placement, dtype, and graph safety before dispatch."""
    from asc.core import namespace as namespace_module

    if _TRUSTED_DLPACK_CONVERSION.get():
        return None
    try:
        native_producer = _array_api_compat.compat.is_array_api_obj(value)
    except Exception as exception:  # pylint: disable=broad-exception-caught
        raise errors.UnsupportedCapabilityError(
            "namespace.from_dlpack: producer array protocol metadata is invalid"
        ) from exception
    if native_producer:
        _producer_namespace, producer_backend = namespace_module.array_metadata(
            value, 0
        )
        if namespace_module.has_active_graph(value, producer_backend):
            raise errors.ConversionError(
                "namespace.from_dlpack: active autodiff graphs require "
                "explicit detach before import"
            )
    else:
        try:
            device_function = getattr(value, "__dlpack_device__", None)
        except Exception as exception:  # pylint: disable=broad-exception-caught
            raise errors.UnsupportedCapabilityError(
                "namespace.from_dlpack: producer device provenance is invalid"
            ) from exception
        if not callable(device_function):
            raise errors.UnsupportedCapabilityError(
                "namespace.from_dlpack: producer lacks CPU device provenance"
            )
        try:
            device = device_function()
        except Exception as exception:  # pylint: disable=broad-exception-caught
            raise errors.UnsupportedCapabilityError(
                "namespace.from_dlpack: producer device provenance is invalid"
            ) from exception
        if (
            not isinstance(device, tuple)
            or len(device) != 2
            or any(
                isinstance(entry, bool) or not isinstance(entry, int)
                for entry in device
            )
            or device != (1, 0)
        ):
            raise errors.UnsupportedCapabilityError(
                "namespace.from_dlpack: only dense CPU producers are supported"
            )
    try:
        producer_dtype = getattr(value, "dtype", None)
    except Exception as exception:  # pylint: disable=broad-exception-caught
        raise errors.DTypeError(
            "namespace.from_dlpack: producer dtype metadata is invalid"
        ) from exception
    producer_name = _dtype.dtype_name(producer_dtype)
    if producer_name is None:
        if backend == "jax" and not _dtype.active_jax_x64_enabled():
            raise errors.DTypeError(
                "namespace.from_dlpack: producer dtype metadata is required "
                "when JAX x64 is disabled"
            )
        return None
    supported = _dtype.supported_dtype_names(
        backend,
        jax_x64_enabled=(backend != "jax" or _dtype.active_jax_x64_enabled()),
    )
    if producer_name not in supported:
        raise errors.DTypeError(
            "namespace.from_dlpack: producer dtype is unavailable to the "
            f"active {backend} release surface"
        )
    return producer_name


def _requires_torch_dlpack_copy(value: object) -> bool:
    """Return whether Torch needs a safety copy before DLPack import."""
    if _is_preexported_dlpack(value):
        return False
    try:
        native_producer = _array_api_compat.compat.is_array_api_obj(value)
    except Exception as exception:  # pylint: disable=broad-exception-caught
        raise errors.UnsupportedCapabilityError(
            "namespace.from_dlpack: producer array protocol metadata is invalid"
        ) from exception
    if not native_producer:
        return True
    flags = getattr(value, "flags", None)
    strides = getattr(value, "strides", None)
    return flags is not None and (
        getattr(flags, "c_contiguous", False) is not True
        or getattr(flags, "writeable", True) is False
        or (strides is not None and any(stride < 0 for stride in strides))
    )


def _is_preexported_dlpack(value: object) -> bool:
    """Authenticate the package's exact one-shot capsule wrapper type."""
    module = sys.modules.get("asc.conversion")
    capsule_type = (
        None if module is None else getattr(module, "_DLPackCapsule", None)
    )
    return isinstance(capsule_type, type) and type(value) is capsule_type


def _accepts_copy_keyword(value: object) -> bool:
    """Return whether an opaque producer can honor a safety-copy request."""
    try:
        exporter = value.__dlpack__
        parameters = inspect.signature(exporter).parameters.values()
    except Exception:  # pylint: disable=broad-exception-caught
        return False
    return any(
        parameter.name == "copy"
        or parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters
    )


def _dlpack_validation_call(
    function: typing.Callable[..., object],
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> tuple[object | None, tuple[object, ...], dict[str, object]]:
    """Bind a DLPack producer regardless of its native parameter spelling."""
    try:
        signature = inspect.signature(function)
        bound = signature.bind_partial(*args, **kwargs)
        producer_parameter = next(iter(signature.parameters))
        producer = bound.arguments.get(producer_parameter)
    except (StopIteration, TypeError, ValueError):
        return None, args, kwargs
    if producer is None:
        return None, args, kwargs
    options = dict(kwargs)
    options.pop(producer_parameter, None)
    remaining = args[1:] if args else ()
    return producer, (producer, *remaining), options


def validate_array_arguments(
    backend: asc_typing.BackendName,
    name: str,
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> str | None:
    """Reject unsupported or foreign arrays before native dispatch."""
    from asc.core import namespace as namespace_module

    producer_dtype_name: str | None = None
    numpy_scalars: tuple[object, ...] = ()
    construction_leaves: tuple[object, ...] = ()
    if name == "from_dlpack" and args:
        producer_dtype_name = _validate_dlpack_producer(args[0], backend)
        arrays = _native_array_leaves((args[1:], kwargs))
    elif name == "asarray" and args:
        construction_leaves = _native_array_leaves(args[0])
        numpy_scalars = tuple(
            value for value in construction_leaves if _is_numpy_scalar(value)
        )
        arrays = (
            *construction_leaves,
            *_native_array_leaves((args[1:], kwargs)),
        )
    else:
        _validate_required_array_operands(backend, name, args)
        arrays = _native_array_leaves((args, kwargs))
    if arrays:
        observed = tuple(
            dict.fromkeys(
                namespace_module.array_metadata(value, position)[1]
                for position, value in enumerate(arrays)
            )
        )
        if observed != (backend,):
            raise errors.MixedBackendError(
                f"{backend} namespace.{name}: the backend-bound namespace "
                f"cannot consume arrays from {observed!r}; convert explicitly "
                "first"
            )
    if (
        name == "asarray"
        and args
        and construction_leaves
        and not any(value is args[0] for value in construction_leaves)
    ):
        if kwargs.get("copy") is False:
            raise ValueError(
                "asarray: copy=False cannot construct an array from nested "
                "native arrays"
            )
        raise TypeError(
            f"{backend} namespace.asarray: nested native arrays must be "
            "stacked explicitly before construction"
        )
    if numpy_scalars and kwargs.get("dtype") is None:
        _validate_inferred_numpy_scalar_dtypes(backend, numpy_scalars)
    return producer_dtype_name


def _bound_first_argument(
    function: typing.Callable[..., object],
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> object | None:
    """Return the first declared argument when native binding succeeds."""
    try:
        signature = inspect.signature(function)
        parameter = next(iter(signature.parameters))
        return signature.bind_partial(*args, **kwargs).arguments.get(parameter)
    except (StopIteration, TypeError, ValueError):
        return None


def _bound_arguments(
    function: typing.Callable[..., object],
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> dict[str, object]:
    """Bind native arguments when its inspectable signature is well behaved."""
    try:
        return dict(
            inspect.signature(function).bind_partial(*args, **kwargs).arguments
        )
    except Exception:  # pylint: disable=broad-exception-caught
        return {}


def _control_value(
    bound: dict[str, object],
    kwargs: dict[str, object],
    args: tuple[object, ...],
    names: tuple[str, ...],
    position: int | None = None,
) -> object:
    """Read one standard control across backend-specific parameter aliases."""
    for parameter in names:
        if parameter in bound:
            return bound[parameter]
        if parameter in kwargs:
            return kwargs[parameter]
    if position is not None and len(args) > position:
        return args[position]
    return _MISSING


def _require_integer(value: object, operation: str, parameter: str) -> None:
    """Require an exact Python integer rather than a Boolean or scalar array."""
    if type(value) is not int:
        raise TypeError(f"{operation}: {parameter} must be a Python integer")


def _require_integer_tuple(
    value: object,
    operation: str,
    parameter: str,
    *,
    allow_integer: bool,
    allow_none: bool,
) -> None:
    """Validate an integer or tuple-of-integers standard control."""
    if value is None and allow_none:
        return
    if allow_integer and type(value) is int:
        return
    if isinstance(value, tuple) and all(type(item) is int for item in value):
        return
    expected = "a Python integer or tuple of Python integers"
    if allow_none:
        expected += ", or None"
    raise TypeError(f"{operation}: {parameter} must be {expected}")


def _validate_namespace_controls(  # pylint: disable=too-many-branches
    function: typing.Callable[..., object],
    name: str,
    args: tuple[object, ...],
    kwargs: dict[str, object],
    validate_dtype: typing.Callable[[object], None],
) -> None:
    """Validate every standard Python-only control before native dispatch."""
    if name in MANDATORY_SYMBOLS and (
        name not in _STANDARD_VARIADIC_POSITIONAL_FUNCTIONS
    ):
        positional_arity = 1
        if name == "__array_namespace_info__":
            positional_arity = 0
        elif name in _STANDARD_TWO_POSITIONAL_FUNCTIONS:
            positional_arity = 2
        elif name in _STANDARD_THREE_POSITIONAL_FUNCTIONS:
            positional_arity = 3
        if len(args) > positional_arity:
            raise TypeError(
                f"{name}: accepts at most {positional_arity} positional "
                "arguments in the frozen Array API signature"
            )
    allowed_keywords = _STANDARD_KEYWORD_PARAMETERS.get(name)
    if name == "from_dlpack" and allowed_keywords is not None:
        try:
            producer_parameter = next(
                iter(inspect.signature(function).parameters)
            )
        except (StopIteration, TypeError, ValueError):
            producer_parameter = "x"
        allowed_keywords = frozenset((*allowed_keywords, producer_parameter))
    unexpected = (
        set(kwargs).difference(allowed_keywords)
        if name in MANDATORY_SYMBOLS and allowed_keywords is not None
        else (
            set(kwargs)
            if name in MANDATORY_SYMBOLS
            else _FORBIDDEN_NATIVE_EXTENSION_KEYWORDS.intersection(kwargs)
        )
    )
    if unexpected:
        parameter = sorted(unexpected)[0]
        if parameter == "dtype":
            raise errors.DTypeError(
                f"{name}: dtype is outside the frozen Array API signature"
            )
        raise TypeError(
            f"{name}: {parameter} is outside the frozen Array API surface"
        )

    bound = _bound_arguments(function, args, kwargs)
    for parameter in _ARRAY_FORBIDDEN_CONTROL_PARAMETERS:
        value = _control_value(bound, kwargs, args, (parameter,))
        if _native_array_leaves(value):
            raise TypeError(
                f"{name}: {parameter} must contain Python scalar controls, "
                "not native arrays"
            )
    for parameter in _BOOLEAN_CONTROL_PARAMETERS:
        value = _control_value(bound, kwargs, args, (parameter,))
        if value is not _MISSING and type(value) is not bool:
            raise TypeError(f"{name}: {parameter} must be a Python Boolean")

    if name in _AXIS_INT_OR_NONE_FUNCTIONS:
        axis = _control_value(bound, kwargs, args, ("axis",), 2)
        if axis is not _MISSING and axis is not None:
            _require_integer(axis, name, "axis")
    elif name in _AXIS_INT_FUNCTIONS:
        axis_position = 2 if name == "diff" else 1
        if name in {"take_along_axis", "vecdot"}:
            axis_position = 2
        axis = _control_value(bound, kwargs, args, ("axis",), axis_position)
        if axis is not _MISSING:
            _require_integer(axis, name, "axis")
    elif name in _AXIS_SEQUENCE_OR_NONE_FUNCTIONS:
        axis = _control_value(bound, kwargs, args, ("axis",), 1)
        if axis is not _MISSING:
            _require_integer_tuple(
                axis,
                name,
                "axis",
                allow_integer=True,
                allow_none=True,
            )
    elif name == "squeeze":
        axis = _control_value(bound, kwargs, args, ("axis",), 1)
        if axis is not _MISSING:
            _require_integer_tuple(
                axis,
                name,
                "axis",
                allow_integer=True,
                allow_none=False,
            )

    if name in _CREATION_SHAPE_FUNCTIONS:
        shape = _control_value(bound, kwargs, args, ("shape",), 0)
        if shape is not _MISSING:
            _require_integer_tuple(
                shape,
                name,
                "shape",
                allow_integer=True,
                allow_none=False,
            )
    elif name in {"broadcast_to", "reshape"}:
        shape = _control_value(bound, kwargs, args, ("shape",), 1)
        if shape is not _MISSING:
            _require_integer_tuple(
                shape,
                name,
                "shape",
                allow_integer=False,
                allow_none=False,
            )

    if name == "eye":
        rows = _control_value(bound, kwargs, args, ("n_rows", "N"), 0)
        columns = _control_value(bound, kwargs, args, ("n_cols", "M"), 1)
        diagonal = _control_value(bound, kwargs, args, ("k",), 2)
        if rows is not _MISSING:
            _require_integer(rows, name, "n_rows")
        if columns is not _MISSING and columns is not None:
            _require_integer(columns, name, "n_cols")
        if diagonal is not _MISSING:
            _require_integer(diagonal, name, "k")
    elif name == "linspace":
        count = _control_value(bound, kwargs, args, ("num",), 2)
        if count is not _MISSING:
            _require_integer(count, name, "num")
    elif name == "diff":
        order = _control_value(bound, kwargs, args, ("n",), 1)
        if order is not _MISSING:
            _require_integer(order, name, "n")
    elif name in {"tril", "triu"}:
        if len(args[0].shape) < 2:
            raise ValueError(f"{name}: input must be at least two-dimensional")
        diagonal = _control_value(bound, kwargs, args, ("k",), 1)
        if diagonal is not _MISSING:
            _require_integer(diagonal, name, "k")

    if name == "cumulative_prod" and len(args[0].shape) == 0:
        raise ValueError(
            "cumulative_prod: input must be at least one-dimensional"
        )

    if name == "moveaxis":
        for parameter, position in (("source", 1), ("destination", 2)):
            value = _control_value(bound, kwargs, args, (parameter,), position)
            if value is not _MISSING:
                _require_integer_tuple(
                    value,
                    name,
                    parameter,
                    allow_integer=True,
                    allow_none=False,
                )
    elif name == "permute_dims":
        axes = _control_value(bound, kwargs, args, ("axes",), 1)
        if axes is not _MISSING:
            _require_integer_tuple(
                axes,
                name,
                "axes",
                allow_integer=False,
                allow_none=False,
            )
    elif name == "roll":
        shift = _control_value(bound, kwargs, args, ("shift",), 1)
        if shift is not _MISSING:
            _require_integer_tuple(
                shift,
                name,
                "shift",
                allow_integer=True,
                allow_none=False,
            )
    elif name == "tensordot":
        axes = _control_value(bound, kwargs, args, ("axes",), 2)
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
        if axes is not _MISSING and type(axes) is not int and not valid_pair:
            raise TypeError(
                "tensordot: axes must be a Python integer or a pair of "
                "integer sequences"
            )
    elif name == "tile":
        repetitions = _control_value(
            bound, kwargs, args, ("repetitions", "reps"), 1
        )
        if repetitions is not _MISSING:
            _require_integer_tuple(
                repetitions,
                name,
                "repetitions",
                allow_integer=False,
                allow_none=False,
            )

    if name == "repeat":
        repeats = _control_value(bound, kwargs, args, ("repeats",), 1)
        if (
            repeats is not _MISSING
            and type(repeats) is not int
            and not _array_api_compat.compat.is_array_api_obj(repeats)
        ):
            raise TypeError(
                "repeat: repeats must be a Python integer or native array"
            )
    if name in {"std", "var"}:
        correction = _control_value(bound, kwargs, args, ("correction",))
        if correction is not _MISSING and type(correction) not in {int, float}:
            raise TypeError(f"{name}: correction must be a Python real scalar")
    if name == "arange":
        for parameter, position, allow_none in (
            ("start", 0, False),
            ("stop", 1, True),
            ("step", 2, False),
        ):
            value = _control_value(bound, kwargs, args, (parameter,), position)
            if value is _MISSING or (value is None and allow_none):
                continue
            if type(value) not in {int, float}:
                raise TypeError(
                    f"arange: {parameter} must be a Python real scalar"
                )
    elif name == "linspace":
        for parameter, position in (("start", 0), ("stop", 1)):
            value = _control_value(bound, kwargs, args, (parameter,), position)
            if value is not _MISSING and type(value) not in {
                int,
                float,
                complex,
            }:
                raise TypeError(
                    f"linspace: {parameter} must be a Python numeric scalar"
                )
    elif name in {"full", "full_like"}:
        position = 1
        fill_value = _control_value(
            bound, kwargs, args, ("fill_value",), position
        )
        if fill_value is not _MISSING and type(fill_value) not in {
            bool,
            int,
            float,
            complex,
        }:
            raise TypeError(
                f"{name}: fill_value must be a Python numeric scalar"
            )
    if name == "meshgrid":
        indexing = _control_value(bound, kwargs, args, ("indexing",))
        if indexing is not _MISSING and (
            type(indexing) is not str or indexing not in {"xy", "ij"}
        ):
            raise TypeError("meshgrid: indexing must be 'xy' or 'ij'")
    if name == "searchsorted":
        side = _control_value(bound, kwargs, args, ("side",), 2)
        if side is not _MISSING and (
            type(side) is not str or side not in {"left", "right"}
        ):
            raise TypeError("searchsorted: side must be 'left' or 'right'")
        sorter = _control_value(bound, kwargs, args, ("sorter",))
        if (
            sorter is not _MISSING
            and sorter is not None
            and not _is_direct_native_array(sorter)
        ):
            raise TypeError(
                "searchsorted: sorter must be a native array or None"
            )
    if name == "diff":
        for parameter in ("prepend", "append"):
            value = _control_value(bound, kwargs, args, (parameter,))
            if (
                value is not _MISSING
                and value is not None
                and not _is_direct_native_array(value)
            ):
                raise TypeError(
                    f"diff: {parameter} must be a native array or None"
                )
    if name == "isdtype":
        dtype = _control_value(bound, kwargs, args, ("dtype",), 0)
        if dtype is not _MISSING:
            validate_dtype(dtype)
        kind = _control_value(bound, kwargs, args, ("kind",), 1)
        kinds = kind if isinstance(kind, tuple) else (kind,)
        if kind is _MISSING or not kinds:
            raise TypeError(
                "isdtype: kind must be a dtype or recognized string"
            )
        for entry in kinds:
            if type(entry) is str:
                if entry not in _DTYPE_QUERY_KINDS:
                    raise TypeError(
                        "isdtype: kind contains an unrecognized dtype category"
                    )
            else:
                validate_dtype(entry)
    elif name in {"finfo", "iinfo"}:
        dtype_or_array = _control_value(bound, kwargs, args, ("type",), 0)
        if dtype_or_array is not _MISSING and not _is_direct_native_array(
            dtype_or_array
        ):
            validate_dtype(dtype_or_array)
    elif name == "can_cast":
        source = _control_value(bound, kwargs, args, ("from_",), 0)
        destination = _control_value(bound, kwargs, args, ("to",), 1)
        if source is not _MISSING and not _is_direct_native_array(source):
            validate_dtype(source)
        if destination is not _MISSING:
            validate_dtype(destination)
    elif name == "result_type":
        for value in args:
            if not _is_direct_native_array(value) and type(value) not in {
                bool,
                int,
                float,
                complex,
            }:
                validate_dtype(value)


def _requested_dtype(
    function: typing.Callable[..., object],
    name: str,
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> object | None:
    """Return a dtype supplied through either native calling convention."""
    if "dtype" in kwargs:
        return kwargs["dtype"]
    try:
        bound = inspect.signature(function).bind_partial(*args, **kwargs)
    except (TypeError, ValueError):
        if name == "astype" and len(args) >= 2:
            return args[1]
        return None
    return bound.arguments.get("dtype")


def _portable_dtype_query(
    namespace: object,
    name: str,
    args: tuple[object, ...],
    validate_dtype: typing.Callable[[object], None],
) -> object:
    """Evaluate promotion queries using the frozen portable lattice."""
    if name == "result_type":
        result_name = require_portable_promotion(name, *args)
        result = getattr(namespace, result_name)
        validate_dtype(result)
        return result
    if name == "can_cast" and len(args) >= 2:
        source = (
            getattr(args[0], "dtype", None)
            if _is_direct_native_array(args[0])
            else args[0]
        )
        source_name = _dtype.dtype_name(source)
        destination_name = _dtype.dtype_name(args[1])
        if source_name is None or destination_name is None:
            raise errors.DTypeError(
                "can_cast: operands must be a native array and dtype, or two "
                "native dtypes"
            )
        try:
            return (
                _promote_dtype_names(source_name, destination_name, name)
                == destination_name
            )
        except errors.DTypeError:
            return False
    return _MISSING


def _cast_standard_operand(
    namespace: object,
    value: object,
    dtype: object,
    device: object | None,
) -> object:
    """Normalize an admitted array/scalar operand before native dispatch."""
    if _is_direct_native_array(value):
        if getattr(value, "dtype", None) == dtype:
            return value
        return typing.cast(typing.Callable[..., object], namespace.astype)(
            value, dtype, copy=True
        )
    return typing.cast(typing.Callable[..., object], namespace.asarray)(
        value, dtype=dtype, device=device
    )


def _clip_comparison_dtype_name(
    namespace: object,
    args: tuple[object, ...],
    kwargs: dict[str, object],
    validate_dtype: typing.Callable[[object], None],
) -> str:
    """Return a common dtype that preserves clip comparisons before casting."""
    target_name = typing.cast(
        str, _dtype.dtype_name(getattr(args[0], "dtype", None))
    )
    for parameter, position in (("min", 1), ("max", 2)):
        bound = _argument_value(args, kwargs, parameter, position)
        if bound is _MISSING or bound is None:
            continue
        if _is_direct_native_array(bound):
            bound_name = typing.cast(
                str, _dtype.dtype_name(getattr(bound, "dtype", None))
            )
            target_name = _promote_dtype_names(target_name, bound_name, "clip")
            continue
        if (
            target_name not in _INTEGER_DTYPE_NAMES
            or type(bound) is not int
            or _integer_bounds(target_name)[0]
            <= bound
            <= _integer_bounds(target_name)[1]
        ):
            continue
        widened_name: str | None = None
        for candidate_name in (
            "int8",
            "uint8",
            "int16",
            "uint16",
            "int32",
            "uint32",
            "int64",
            "uint64",
        ):
            lower, upper = _integer_bounds(candidate_name)
            if not lower <= bound <= upper:
                continue
            try:
                candidate_dtype = getattr(namespace, candidate_name)
                validate_dtype(candidate_dtype)
                promoted_name = _promote_dtype_names(
                    target_name, candidate_name, "clip"
                )
                validate_dtype(getattr(namespace, promoted_name))
            except (AttributeError, errors.DTypeError):
                continue
            widened_name = promoted_name
            break
        if widened_name is None:
            raise errors.DTypeError(
                f"clip: {parameter} cannot be compared without narrowing"
            )
        target_name = widened_name
    return target_name


def _normalize_standard_dispatch(
    namespace: object,
    name: str,
    args: tuple[object, ...],
    kwargs: dict[str, object],
    expected_result_dtype: str | None,
    validate_dtype: typing.Callable[[object], None],
) -> tuple[tuple[object, ...], dict[str, object]]:
    """Cast standard operands to the portable pre-dispatch dtype."""
    normalized_args = list(args)
    normalized_kwargs = dict(kwargs)

    if name in {"repeat", "searchsorted"}:
        parameter, position = (
            ("repeats", 1) if name == "repeat" else ("sorter", 3)
        )
        control = _argument_value(args, kwargs, parameter, position)
        if _is_direct_native_array(control):
            carrier = None
            for dtype_name in ("int64", "int32"):
                try:
                    candidate = getattr(namespace, dtype_name)
                    validate_dtype(candidate)
                except (AttributeError, errors.DTypeError):
                    continue
                carrier = candidate
                break
            if carrier is None:
                raise errors.DTypeError(
                    f"{name}: backend has no supported signed index carrier"
                )
            normalized = _cast_standard_operand(
                namespace,
                control,
                carrier,
                _array_api_compat.compat.device(control),
            )
            if parameter in normalized_kwargs:
                normalized_kwargs[parameter] = normalized
            elif len(normalized_args) > position:
                normalized_args[position] = normalized
            else:
                normalized_kwargs[parameter] = normalized

    target_name: str | None = None
    positions: tuple[int, ...] = ()
    if name in _BINARY_DTYPE_CATEGORIES:
        target_name = require_portable_promotion(name, *args[:2])
        positions = (0, 1)
    elif name in _NONPROMOTING_BINARY_DTYPE_CATEGORIES:
        target_name = expected_result_dtype
        positions = (0, 1)
    elif name == "where":
        target_name = require_portable_promotion(name, *args[1:3])
        positions = (1, 2)
    elif name in {"concat", "stack"}:
        target_name = expected_result_dtype
    elif name == "clip":
        target_name = _clip_comparison_dtype_name(
            namespace, args, kwargs, validate_dtype
        )

    if target_name is None:
        return tuple(normalized_args), normalized_kwargs
    target_dtype = getattr(namespace, target_name)
    validate_dtype(target_dtype)
    arrays = _native_array_leaves((args, kwargs))
    device = None if not arrays else _array_api_compat.compat.device(arrays[0])
    if positions:
        for position in positions:
            normalized_args[position] = _cast_standard_operand(
                namespace,
                normalized_args[position],
                target_dtype,
                device,
            )
    elif name in {"concat", "stack"}:
        values = typing.cast(collections.abc.Sequence[object], args[0])
        normalized_args[0] = type(values)(
            _cast_standard_operand(namespace, value, target_dtype, device)
            for value in values
        )
    else:
        normalized_args[0] = _cast_standard_operand(
            namespace, normalized_args[0], target_dtype, device
        )
        for parameter, position in (("min", 1), ("max", 2)):
            value = _argument_value(args, kwargs, parameter, position)
            if value is _MISSING or value is None:
                continue
            normalized = _cast_standard_operand(
                namespace, value, target_dtype, device
            )
            if parameter in normalized_kwargs:
                normalized_kwargs[parameter] = normalized
            else:
                normalized_args[position] = normalized
    return tuple(normalized_args), normalized_kwargs


def _validate_fill_value(
    namespace: object,
    name: str,
    args: tuple[object, ...],
    kwargs: dict[str, object],
    requested_dtype: object | None,
) -> None:
    """Reject explicit fills that the destination dtype cannot represent."""
    if name not in {"full", "full_like"}:
        return
    dtype = requested_dtype
    if dtype is None and name == "full_like":
        dtype = getattr(args[0], "dtype", None)
    fill_value = _argument_value(args, kwargs, "fill_value", 1)
    if fill_value is not _MISSING:
        require_representable_scalar(namespace, dtype, fill_value, name)


def _validate_jax_no_copy(
    function: typing.Callable[..., object],
    name: str,
    args: tuple[object, ...],
    kwargs: dict[str, object],
    requested_dtype: object | None,
    resolve_device: typing.Callable[[object | None], object | None],
) -> object | None:
    """Preflight every JAX operation exposing an explicit no-copy policy."""
    if kwargs.get("copy") is not False or name not in {
        "asarray",
        "astype",
        "reshape",
    }:
        return None
    source = _bound_first_argument(function, args, kwargs)
    if not _is_direct_native_array(source):
        raise ValueError(
            f"{name}: copy=False cannot create JAX storage from Python-owned "
            "input"
        )
    source_dtype = getattr(source, "dtype", None)
    is_weak = getattr(source, "weak_type", False) is True
    if requested_dtype is not None and (
        source_dtype != requested_dtype or is_weak
    ):
        raise ValueError(
            f"{name}: copy=False cannot change a JAX array dtype or weak type"
        )
    if name == "reshape":
        bound = _bound_arguments(function, args, kwargs)
        shape = _control_value(bound, kwargs, args, ("shape",), 1)
        source_shape = tuple(getattr(source, "shape", ()))
        if is_weak or shape is _MISSING or source_shape != shape:
            raise ValueError(
                "reshape: copy=False cannot prove a storage-preserving JAX "
                "reshape"
            )
    requested_device = kwargs.get("device")
    if requested_device is not None:
        resolved_requested_device = resolve_device(requested_device)
        source_device = _array_api_compat.compat.device(source)
        if source_device != resolved_requested_device:
            raise ValueError(
                f"{name}: copy=False cannot change a JAX array device"
            )
    return source


def checked_attribute(
    namespace: object,
    name: str,
    backend: asc_typing.BackendName,
    validate_dtype: typing.Callable[[object], None],
    resolve_device: typing.Callable[[object | None], object | None],
    dispatch_override: (
        typing.Callable[
            [str, tuple[object, ...], dict[str, object]],
            tuple[bool, object],
        ]
        | None
    ) = None,
    validate_values: (
        typing.Callable[
            [str, tuple[object, ...], dict[str, object], str | None], None
        ]
        | None
    ) = None,
) -> object:
    """Return one namespace attribute with backend-bound validation."""
    if (
        name not in MANDATORY_SYMBOLS
        and name not in {"fft", "linalg"}
        and (
            name
            not in _dtype.supported_dtype_names(backend, jax_x64_enabled=True)
        )
    ):
        raise AttributeError(
            f"{backend} namespace: {name!r} is outside the frozen Array API "
            "surface"
        )
    attribute = getattr(namespace, name)
    if name == "linalg":
        from asc.linalg import LinalgNamespace

        return LinalgNamespace(
            attribute,
            backend=backend,
            dtype_validator=validate_dtype,
        )
    if name == "fft":
        from asc.fft import FFTNamespace

        default_dtype = typing.cast(
            asc_typing.NativeArray,
            typing.cast(typing.Callable[..., object], namespace.asarray)(
                0.0, device=resolve_device(None)
            ),
        )
        default_dtype = default_dtype.dtype
        validate_dtype(default_dtype)
        return FFTNamespace(
            attribute,
            namespace=typing.cast(asc_typing.ArrayNamespace, namespace),
            backend=backend,
            device=resolve_device(None),
            dtype=None,
            default_dtype=default_dtype,
            dtype_validator=validate_dtype,
            device_resolver=resolve_device,
        )
    if name == "__array_namespace_info__":
        inspection_factory = typing.cast(typing.Callable[[], object], attribute)

        @functools.wraps(inspection_factory)
        def bounded_inspection_factory() -> _NamespaceInspection:
            inspection = typing.cast(
                _NativeNamespaceInspection, inspection_factory()
            )
            return _NamespaceInspection(
                inspection=inspection,
                namespace=typing.cast(asc_typing.ArrayNamespace, namespace),
                backend=backend,
                dtype_validator=validate_dtype,
                device_resolver=resolve_device,
            )

        return bounded_inspection_factory
    if not callable(attribute) or name in _dtype.supported_dtype_names(
        backend, jax_x64_enabled=True
    ):
        return attribute

    function = typing.cast(typing.Callable[..., object], attribute)

    @functools.wraps(function)
    def checked(*args: object, **kwargs: object) -> object:
        copy_arrays = _native_array_leaves(kwargs.get("copy"))
        if copy_arrays:
            from asc.core import namespace as namespace_module

            observed_copy_backends = tuple(
                dict.fromkeys(
                    namespace_module.array_metadata(value, position)[1]
                    for position, value in enumerate(copy_arrays)
                )
            )
            if observed_copy_backends != (backend,):
                raise errors.MixedBackendError(
                    f"{backend} namespace.{name}: the backend-bound namespace "
                    "cannot consume copy controls from "
                    f"{observed_copy_backends!r}"
                )
        if (
            "copy" in kwargs
            and kwargs["copy"] is not None
            and type(kwargs["copy"]) is not bool
        ):
            raise TypeError(f"{name}: copy must be a Boolean or None")
        deferred_dlpack_copy = False
        validation_args = args
        validation_kwargs = kwargs
        producer: object | None = None
        if name == "from_dlpack":
            producer, validation_args, validation_kwargs = (
                _dlpack_validation_call(function, args, kwargs)
            )
        if producer is not None:
            requested_copy = kwargs.get("copy")
            if backend == "torch" and _requires_torch_dlpack_copy(producer):
                if requested_copy is False:
                    raise errors.UnsupportedCapabilityError(
                        "torch namespace.from_dlpack: producer layout cannot "
                        "safely guarantee a no-copy import"
                    )
                if (
                    not _array_api_compat.compat.is_array_api_obj(producer)
                    and not _is_preexported_dlpack(producer)
                    and not _accepts_copy_keyword(producer)
                ):
                    raise errors.UnsupportedCapabilityError(
                        "torch namespace.from_dlpack: opaque producer must "
                        "support the copy keyword for safe import"
                    )
                if requested_copy is None:
                    kwargs["copy"] = True
            if _is_preexported_dlpack(producer) and requested_copy is True:
                kwargs["copy"] = None
            if (
                kwargs.get("copy") is True
                or (_is_preexported_dlpack(producer) and requested_copy is True)
            ) and not _TRUSTED_DLPACK_CONVERSION.get():
                deferred_dlpack_copy = True
        requested_dtype = (
            _requested_dtype(function, name, args, kwargs)
            if name in _dtype.DTYPE_ARGUMENT_FUNCTIONS
            else None
        )
        if (
            name == "asarray"
            and "dtype" not in validation_kwargs
            and requested_dtype is not None
        ):
            validation_kwargs = {
                **validation_kwargs,
                "dtype": requested_dtype,
            }
        producer_dtype_name = validate_array_arguments(
            backend,
            name,
            validation_args,
            validation_kwargs,
        )
        _validate_namespace_controls(
            function,
            name,
            args,
            kwargs,
            validate_dtype,
        )
        expected_result_dtype = _validate_namespace_dtypes(name, args, kwargs)
        _validate_scalar_namespace_values(
            name, args, kwargs, expected_result_dtype
        )
        if validate_values is None:
            _validate_concrete_namespace_values(
                namespace, name, args, kwargs, expected_result_dtype
            )
        else:
            validate_values(name, args, kwargs, expected_result_dtype)
        portable_query = _portable_dtype_query(
            namespace, name, args, validate_dtype
        )
        if portable_query is not _MISSING:
            return portable_query
        if name == "meshgrid" and not args:
            return []
        jax_no_copy_source = (
            _validate_jax_no_copy(
                function,
                name,
                args,
                kwargs,
                requested_dtype,
                resolve_device,
            )
            if backend == "jax"
            else None
        )
        if requested_dtype is not None:
            validate_dtype(requested_dtype)
        _validate_fill_value(
            namespace,
            name,
            args,
            kwargs,
            requested_dtype,
        )
        if jax_no_copy_source is not None:
            return jax_no_copy_source
        if name in _DEVICE_CREATION_FUNCTIONS:
            requested_device = kwargs.get("device")
            preserve_device = name in _DEVICE_PRESERVING_CREATION_FUNCTIONS or (
                name == "asarray"
                and bool(args)
                and _is_direct_native_array(args[0])
            )
            if requested_device is not None or not preserve_device:
                kwargs["device"] = resolve_device(requested_device)
        args, kwargs = _normalize_standard_dispatch(
            namespace,
            name,
            args,
            kwargs,
            expected_result_dtype,
            validate_dtype,
        )
        handled, result = (
            dispatch_override(name, args, kwargs)
            if dispatch_override is not None
            else (False, None)
        )
        if not handled:
            result = function(*args, **kwargs)
        if name == "clip" and expected_result_dtype is not None:
            result = typing.cast(
                typing.Callable[..., object], namespace.astype
            )(
                result,
                getattr(namespace, expected_result_dtype),
                copy=True,
            )
        if deferred_dlpack_copy:
            result = typing.cast(
                typing.Callable[..., object], namespace.asarray
            )(
                result,
                device=resolve_device(kwargs.get("device")),
                copy=True,
            )
        result_dtype = getattr(result, "dtype", None)
        if (
            expected_result_dtype is not None
            and _dtype.dtype_name(result_dtype) != expected_result_dtype
        ):
            raise errors.DTypeError(
                f"{name}: backend result dtype does not match portable "
                f"promotion {expected_result_dtype!r}"
            )
        if (
            name in _dtype.DTYPE_ARGUMENT_FUNCTIONS or name == "from_dlpack"
        ) and result_dtype is not None:
            validate_dtype(result_dtype)
        if (
            producer_dtype_name is not None
            and _dtype.dtype_name(result_dtype) != producer_dtype_name
        ):
            raise errors.DTypeError(
                f"{backend} namespace.from_dlpack: producer dtype was not "
                "preserved"
            )
        return result

    return checked


__all__ = [
    "checked_attribute",
    "require_dtype_category",
    "trusted_dlpack_conversion",
    "validate_array_arguments",
]
