# Copyright 2026 AI4SciComp contributors.
# SPDX-License-Identifier: Apache-2.0
"""Lazy PyTorch extension adapter."""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import math
import typing
import warnings

import array_api_compat.torch as array_api_namespace
import torch

from asc import _array_api_compat, errors
from asc import typing as asc_typing
from asc.backends import _namespace, _state, _torch_random
from asc.core import _dtype


class _TorchModule(typing.Protocol):  # pylint: disable=too-few-public-methods
    """Typed dynamic compilation surface used by this adapter."""

    def compile(
        self,
        function: typing.Callable[..., object],
        *,
        backend: str,
        fullgraph: bool,
    ) -> typing.Callable[..., object]:
        """Compile a native PyTorch callable."""
        ...  # pylint: disable=unnecessary-ellipsis


class _UnpackedDual(typing.Protocol):  # pylint: disable=too-few-public-methods
    """Typed result from the PyTorch forward-mode inspection API."""

    tangent: object | None


class _ForwardAD(typing.Protocol):  # pylint: disable=too-few-public-methods
    """Typed PyTorch forward-mode inspection surface."""

    def unpack_dual(self, value: torch.Tensor) -> _UnpackedDual:
        """Separate primal and tangent values."""
        ...  # pylint: disable=unnecessary-ellipsis


_torch = typing.cast(_TorchModule, torch)
_forward_ad = typing.cast(_ForwardAD, torch.autograd.forward_ad)
_check_tensor_all = typing.cast(
    typing.Callable[[torch.Tensor, typing.Callable[[], str]], None],
    vars(torch)["_check_tensor_all"],
)

_WIDE_UNSIGNED_DTYPES: typing.Final = frozenset(
    {torch.uint16, torch.uint32, torch.uint64}
)
_SIGNED_VIEW_DTYPE: typing.Final = {
    torch.uint16: torch.int16,
    torch.uint32: torch.int32,
    torch.uint64: torch.int64,
}
_WIDER_SIGNED_DTYPE: typing.Final = {
    torch.uint16: torch.int32,
    torch.uint32: torch.int64,
}


def _ordered_unsigned(array: torch.Tensor) -> torch.Tensor:
    """Map unsigned bit patterns monotonically into a signed dtype."""
    signed_dtype = _SIGNED_VIEW_DTYPE[array.dtype]
    signed = array.view(signed_dtype)
    return torch.bitwise_xor(signed, torch.iinfo(signed_dtype).min)


def _restore_ordered_unsigned(
    array: torch.Tensor, dtype: torch.dtype
) -> torch.Tensor:
    """Invert :func:`_ordered_unsigned` without numeric conversion."""
    signed_dtype = _SIGNED_VIEW_DTYPE[dtype]
    restored = torch.bitwise_xor(array, torch.iinfo(signed_dtype).min)
    return restored.view(dtype)


def _uint64_divmod(
    dividend: torch.Tensor, divisor: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute elementwise unsigned 64-bit division using signed bit ops."""
    dividend, divisor = torch.broadcast_tensors(dividend, divisor)
    try:
        _check_tensor_all(
            divisor != 0, lambda: "integer division or modulo by zero"
        )
    except RuntimeError as exception:
        raise ZeroDivisionError(
            "integer division or modulo by zero"
        ) from exception
    dividend_bits = dividend.view(torch.int64)
    divisor_bits = divisor.view(torch.int64)
    divisor_ordered = _ordered_unsigned(divisor)
    quotient = torch.zeros_like(dividend_bits)
    remainder = torch.zeros_like(dividend_bits)
    zero = torch.zeros_like(dividend_bits)
    for bit in range(63, -1, -1):
        carry = remainder < 0
        incoming = torch.bitwise_and(
            torch.bitwise_right_shift(dividend_bits, bit), 1
        )
        shifted = torch.bitwise_or(
            torch.bitwise_left_shift(remainder, 1), incoming
        )
        shifted_ordered = torch.bitwise_xor(
            shifted, torch.iinfo(torch.int64).min
        )
        subtract = carry | (shifted_ordered >= divisor_ordered)
        remainder = torch.where(subtract, shifted - divisor_bits, shifted)
        mask = -(2**63) if bit == 63 else 1 << bit
        quotient = torch.bitwise_or(
            quotient,
            torch.where(subtract, torch.full_like(quotient, mask), zero),
        )
    return quotient.view(torch.uint64), remainder.view(torch.uint64)


def _uint64_power(base: torch.Tensor, exponent: torch.Tensor) -> torch.Tensor:
    """Compute modular unsigned powers without unsupported Torch kernels."""
    base, exponent = torch.broadcast_tensors(base, exponent)
    factor = base.view(torch.int64)
    exponent_bits = exponent.view(torch.int64)
    result = torch.ones_like(factor)
    for bit in range(64):
        mask = -(2**63) if bit == 63 else 1 << bit
        selected = torch.bitwise_and(exponent_bits, mask) != 0
        result = torch.where(selected, result * factor, result)
        if bit != 63:
            factor = factor * factor
    return result.view(torch.uint64)


def _uint64_shift(
    array: torch.Tensor, shifts: torch.Tensor, *, left: bool
) -> torch.Tensor:
    """Shift unsigned 64-bit bit patterns without signed right-shift leaks."""
    array, shifts = torch.broadcast_tensors(array, shifts)
    shift_counts = shifts.view(torch.int64)
    oversized = (shift_counts < 0) | (shift_counts >= 64)
    safe_counts = torch.where(
        oversized, torch.zeros_like(shift_counts), shift_counts
    )
    bits = array.view(torch.int64)
    if left:
        shifted = torch.bitwise_left_shift(bits, safe_counts)
        return torch.where(oversized, torch.zeros_like(bits), shifted).view(
            torch.uint64
        )

    result = torch.zeros_like(bits)
    one = torch.ones_like(bits)
    for output_bit in range(64):
        source_bit = safe_counts + output_bit
        available = source_bit < 64
        safe_source = torch.where(available, source_bit, 0)
        extracted = torch.bitwise_and(
            torch.bitwise_right_shift(bits, safe_source), one
        )
        result = torch.bitwise_or(
            result,
            torch.where(
                available,
                torch.bitwise_left_shift(extracted, output_bit),
                torch.zeros_like(result),
            ),
        )
    return torch.where(oversized, torch.zeros_like(result), result).view(
        torch.uint64
    )


def _raise_namespace_invalid(
    invalid: torch.Tensor,
    operation: str,
    message: str,
    *,
    index_error: bool = False,
) -> None:
    """Raise a graph-safe public error for an invalid Torch predicate."""
    try:
        _check_tensor_all(~invalid, lambda: f"{operation}: {message}")
    except RuntimeError as exception:
        if index_error:
            raise IndexError(f"{operation}: {message}") from exception
        raise ValueError(f"{operation}: {message}") from exception


def _validate_namespace_values(
    name: str,
    args: tuple[object, ...],
    kwargs: dict[str, object],
    expected_result_dtype: str | None,
) -> None:
    """Validate Torch values without scalarizing batched tensors."""
    if name in {"take", "take_along_axis"}:
        extent = _namespace.take_axis_extent(name, args, kwargs)
        if extent is not None:
            indices = typing.cast(torch.Tensor, args[1])
            if indices.dtype == torch.uint64:
                carrier = indices.view(torch.int64)
                invalid = (carrier < 0) | (carrier >= extent)
            elif indices.dtype in {torch.uint8, torch.uint16, torch.uint32}:
                invalid = indices.to(dtype=torch.int64) >= extent
            else:
                invalid = (indices < -extent) | (indices >= extent)
            _raise_namespace_invalid(
                torch.any(invalid),
                name,
                "index is out of bounds",
                index_error=True,
            )
    value: object | None = None
    message = ""
    if name == "pow" and _namespace.is_integer_dtype_name(
        expected_result_dtype
    ):
        value = args[1]
        message = "integer exponents must be non-negative"
    elif name in {"bitwise_left_shift", "bitwise_right_shift"}:
        value = args[1]
        message = "shift counts must be non-negative"
    elif name == "repeat":
        value = _namespace.namespace_argument_value(args, kwargs, "repeats", 1)
        message = "repeat counts must be non-negative"
    if isinstance(value, torch.Tensor) and value.dtype in {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }:
        _raise_namespace_invalid(torch.any(value < 0), name, message)


def _convert_wide_unsigned(
    value: object, dtype: torch.dtype, destination: torch.dtype
) -> object:
    """Convert matching direct tensor operands to an exact signed carrier."""
    if isinstance(value, torch.Tensor) and value.dtype == dtype:
        return value.to(dtype=destination)
    return value


def _unsupported_reduction_dtype_dispatch(
    name: str, args: tuple[object, ...], kwargs: dict[str, object]
) -> tuple[bool, object]:
    """Emulate Torch CPU reduction dtypes without native kernels."""
    if (
        name not in {"sum", "prod", "cumulative_sum", "cumulative_prod"}
        or not args
        or not isinstance(args[0], torch.Tensor)
    ):
        return False, None
    requested = kwargs.get("dtype")
    if requested not in {*_WIDE_UNSIGNED_DTYPES, torch.bool}:
        return False, None

    array = args[0].to(dtype=typing.cast(torch.dtype, requested))
    if requested == torch.uint64:
        carrier = array.view(torch.int64)
        carrier_dtype = torch.int64
    elif requested == torch.bool:
        carrier = array.to(dtype=torch.int64)
        carrier_dtype = torch.int64
    else:
        carrier_dtype = _WIDER_SIGNED_DTYPE[typing.cast(torch.dtype, requested)]
        carrier = array.to(dtype=carrier_dtype)
    call_kwargs = dict(kwargs)
    call_kwargs["dtype"] = carrier_dtype
    result = getattr(array_api_namespace, name)(
        carrier, *args[1:], **call_kwargs
    )
    if requested == torch.uint64:
        return True, result.view(torch.uint64)
    return True, result.to(dtype=typing.cast(torch.dtype, requested))


def _wide_unsigned_dispatch(
    name: str, args: tuple[object, ...], kwargs: dict[str, object]
) -> tuple[bool, object]:
    """Fill PyTorch CPU kernel gaps for advertised unsigned dtypes."""
    handled, result = _unsupported_reduction_dtype_dispatch(name, args, kwargs)
    if handled:
        return handled, result
    if not args or not isinstance(args[0], torch.Tensor):
        return False, None
    array = args[0]
    dtype = array.dtype
    if dtype not in _WIDE_UNSIGNED_DTYPES:
        return False, None

    signed_view = array.view(_SIGNED_VIEW_DTYPE[dtype])
    if name in {"count_nonzero", "nonzero"}:
        return True, getattr(array_api_namespace, name)(
            signed_view, *args[1:], **kwargs
        )
    if name in {"flip", "tril", "triu"}:
        result = getattr(array_api_namespace, name)(
            signed_view, *args[1:], **kwargs
        )
        return True, result.view(dtype)

    if name in {"sum", "prod", "cumulative_sum", "cumulative_prod"} and (
        kwargs.get("dtype") is None
    ):
        carrier = (
            array.view(torch.int64)
            if dtype == torch.uint64
            else array.to(dtype=torch.int64)
        )
        call_kwargs = dict(kwargs)
        call_kwargs["dtype"] = torch.int64
        result = getattr(array_api_namespace, name)(carrier, **call_kwargs)
        return True, result.view(torch.uint64)

    if dtype in _WIDER_SIGNED_DTYPE:
        destination = _WIDER_SIGNED_DTYPE[dtype]
        supported = {
            "abs",
            "add",
            "argmax",
            "argmin",
            "bitwise_invert",
            "bitwise_left_shift",
            "bitwise_right_shift",
            "clip",
            "diff",
            "floor_divide",
            "greater",
            "greater_equal",
            "less",
            "less_equal",
            "max",
            "maximum",
            "min",
            "minimum",
            "matmul",
            "negative",
            "positive",
            "pow",
            "remainder",
            "searchsorted",
            "sign",
            "square",
            "subtract",
            "tensordot",
            "vecdot",
        }
        if name not in supported:
            return False, None
        converted_args = tuple(
            _convert_wide_unsigned(value, dtype, destination) for value in args
        )
        converted_kwargs = {
            key: _convert_wide_unsigned(value, dtype, destination)
            for key, value in kwargs.items()
        }
        result = getattr(array_api_namespace, name)(
            *converted_args, **converted_kwargs
        )
        if name in {
            "argmax",
            "argmin",
            "greater",
            "greater_equal",
            "less",
            "less_equal",
            "searchsorted",
        }:
            return True, result
        return True, result.to(dtype=dtype)

    second = args[1] if len(args) > 1 else None
    if name in {"add", "subtract"} and isinstance(second, torch.Tensor):
        operation = torch.add if name == "add" else torch.subtract
        return True, operation(
            array.view(torch.int64), second.view(torch.int64)
        ).view(torch.uint64)
    if name == "negative":
        return True, torch.negative(array.view(torch.int64)).view(torch.uint64)
    if name == "bitwise_invert":
        return True, torch.bitwise_not(array.view(torch.int64)).view(
            torch.uint64
        )
    if name in {"abs", "positive"}:
        return True, array.clone()
    if name == "sign":
        return True, (array != 0).to(dtype=torch.uint64)
    if name in {"greater", "greater_equal", "less", "less_equal"} and (
        isinstance(second, torch.Tensor)
    ):
        operation = {
            "greater": torch.gt,
            "greater_equal": torch.ge,
            "less": torch.lt,
            "less_equal": torch.le,
        }[name]
        return True, operation(
            _ordered_unsigned(array), _ordered_unsigned(second)
        )
    if name in {"maximum", "minimum"} and isinstance(second, torch.Tensor):
        operation = torch.maximum if name == "maximum" else torch.minimum
        ordered = operation(_ordered_unsigned(array), _ordered_unsigned(second))
        return True, _restore_ordered_unsigned(ordered, torch.uint64)
    if name in {"max", "min", "argmax", "argmin"}:
        ordered = _ordered_unsigned(array)
        result = getattr(array_api_namespace, name)(ordered, **kwargs)
        if name in {"argmax", "argmin"}:
            return True, result
        return True, _restore_ordered_unsigned(result, torch.uint64)
    if name == "clip":
        ordered_args = list(args)
        ordered_kwargs = dict(kwargs)
        ordered_args[0] = _ordered_unsigned(array)
        for parameter, position in (("min", 1), ("max", 2)):
            value = kwargs.get(
                parameter, args[position] if len(args) > position else None
            )
            if not isinstance(value, torch.Tensor):
                continue
            ordered = _ordered_unsigned(value)
            if parameter in ordered_kwargs:
                ordered_kwargs[parameter] = ordered
            else:
                ordered_args[position] = ordered
        result = array_api_namespace.clip(*ordered_args, **ordered_kwargs)
        return True, _restore_ordered_unsigned(result, torch.uint64)
    if name in {"floor_divide", "remainder"} and isinstance(
        second, torch.Tensor
    ):
        quotient, remainder = _uint64_divmod(array, second)
        return True, quotient if name == "floor_divide" else remainder
    if name == "pow" and isinstance(second, torch.Tensor):
        return True, _uint64_power(array, second)
    if name == "square":
        return True, torch.square(signed_view).view(torch.uint64)
    if name in {"matmul", "tensordot", "vecdot"} and isinstance(
        second, torch.Tensor
    ):
        converted_args = (
            signed_view,
            second.view(torch.int64),
            *args[2:],
        )
        result = getattr(array_api_namespace, name)(*converted_args, **kwargs)
        return True, result.view(torch.uint64)
    if name == "searchsorted" and isinstance(second, torch.Tensor):
        converted_args = (
            _ordered_unsigned(array),
            _ordered_unsigned(second),
            *args[2:],
        )
        searchsorted = getattr(array_api_namespace, name)
        return True, searchsorted(*converted_args, **kwargs)
    if name in {"bitwise_left_shift", "bitwise_right_shift"} and isinstance(
        second, torch.Tensor
    ):
        return True, _uint64_shift(
            array, second, left=name == "bitwise_left_shift"
        )
    if name == "diff":
        converted_args = (array.view(torch.int64), *args[1:])
        converted_kwargs = {
            key: (
                value.view(torch.int64)
                if isinstance(value, torch.Tensor)
                and value.dtype == torch.uint64
                else value
            )
            for key, value in kwargs.items()
        }
        result = array_api_namespace.diff(*converted_args, **converted_kwargs)
        return True, result.view(torch.uint64)
    return False, None


class _TorchNamespace:
    """Forward the frozen namespace while normalizing known Torch gaps."""

    __name__ = array_api_namespace.__name__
    __array_api_version__ = array_api_namespace.__array_api_version__

    def __getattr__(self, name: str) -> object:
        return _namespace.checked_attribute(
            array_api_namespace,
            name,
            "torch",
            validate_dtype,
            resolve_device,
            _wide_unsigned_dispatch,
            _validate_namespace_values,
        )

    @staticmethod
    def _validate(
        name: str,
        args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> None:
        """Apply the shared backend and dense-CPU boundary to overrides."""
        _namespace.validate_array_arguments("torch", name, args, kwargs)

    @staticmethod
    def _indices(indices: torch.Tensor, operation: str) -> torch.Tensor:
        if indices.dtype not in {
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
            torch.uint16,
            torch.uint32,
            torch.uint64,
        }:
            raise TypeError(f"{operation}: indices must be integers")
        if indices.dtype == torch.uint64:
            message = (
                f"{operation}: unsigned index exceeds the signed carrier range"
            )
            try:
                _check_tensor_all(
                    indices.view(torch.int64) >= 0,
                    lambda: message,
                )
            except RuntimeError as exception:
                raise IndexError(message) from exception
        return indices.to(dtype=torch.int64)

    @staticmethod
    def _axis(axis: object, operation: str, *, allow_none: bool) -> None:
        """Reject array-valued controls before Torch can extract scalars."""
        if axis is None and allow_none:
            return
        if type(axis) is not int:
            expected = (
                "a Python integer or None"
                if allow_none
                else ("a Python integer")
            )
            raise TypeError(f"{operation}: axis must be {expected}")

    @staticmethod
    def _shape(shape: object) -> None:
        """Require a tuple of Python integers for reshape controls."""
        if not isinstance(shape, tuple) or any(
            type(extent) is not int for extent in shape
        ):
            raise TypeError("reshape: shape must be a tuple of Python integers")

    @classmethod
    def take(
        cls,
        array: torch.Tensor,
        indices: torch.Tensor,
        /,
        *,
        axis: int | None = None,
    ) -> torch.Tensor:
        """Accept every integer index width supported by the standard."""
        cls._validate("take", (array, indices), {"axis": axis})
        cls._axis(axis, "take", allow_none=True)
        if indices.ndim != 1:
            raise ValueError("take: indices must be one-dimensional")
        if axis is None and array.ndim != 1:
            raise ValueError(
                "take: axis must be specified unless the input is "
                "one-dimensional"
            )
        normalized_indices = cls._indices(indices, "take")
        _validate_namespace_values(
            "take", (array, indices), {"axis": axis}, None
        )
        source = (
            array.view(_SIGNED_VIEW_DTYPE[array.dtype])
            if array.dtype in _WIDE_UNSIGNED_DTYPES
            else array
        )
        result = array_api_namespace.take(source, normalized_indices, axis=axis)
        return result.view(array.dtype) if source is not array else result

    @classmethod
    def take_along_axis(
        cls,
        array: torch.Tensor,
        indices: torch.Tensor,
        /,
        *,
        axis: int = -1,
    ) -> torch.Tensor:
        """Accept every integer index width supported by the standard."""
        cls._validate("take_along_axis", (array, indices), {"axis": axis})
        cls._axis(axis, "take_along_axis", allow_none=False)
        normalized_indices = cls._indices(indices, "take_along_axis")
        _validate_namespace_values(
            "take_along_axis", (array, indices), {"axis": axis}, None
        )
        source = (
            array.view(_SIGNED_VIEW_DTYPE[array.dtype])
            if array.dtype in _WIDE_UNSIGNED_DTYPES
            else array
        )
        result = array_api_namespace.take_along_axis(
            source,
            normalized_indices,
            axis=axis,
        )
        return result.view(array.dtype) if source is not array else result

    @classmethod
    def reshape(
        cls,
        array: torch.Tensor,
        /,
        shape: tuple[int, ...],
        *,
        copy: bool | None = None,
    ) -> torch.Tensor:
        """Implement the standard reshape copy policy for PyTorch."""
        cls._validate("reshape", (array, shape), {"copy": copy})
        cls._shape(shape)
        if copy is not None and not isinstance(copy, bool):
            raise TypeError("reshape: copy must be a Boolean or None")
        if copy is False:
            try:
                return array.view(shape)
            except RuntimeError as exception:
                raise ValueError(
                    "reshape: copy=False cannot be honored"
                ) from exception
        source = array.clone() if copy is True else array
        return torch.reshape(source, shape)

    @classmethod
    def round(cls, array: torch.Tensor, /) -> torch.Tensor:
        """Round complex values componentwise around a missing CPU kernel."""
        cls._validate("round", (array,), {})
        if array.dtype.is_complex:
            return torch.complex(
                torch.round(torch.real(array)),
                torch.round(torch.imag(array)),
            )
        return torch.round(array)

    @staticmethod
    def _unique_components(
        array: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return sorted values, first indices, inverse groups, and counts."""
        flattened = torch.reshape(array, (-1,))
        if flattened.numel() == 0:
            empty_indices = torch.empty(
                (0,), dtype=torch.int64, device=array.device
            )
            return (
                flattened,
                empty_indices,
                torch.reshape(empty_indices, array.shape),
                empty_indices,
            )

        if array.dtype.is_complex:
            by_imaginary = torch.argsort(torch.imag(flattened), stable=True)
            order = by_imaginary[
                torch.argsort(torch.real(flattened[by_imaginary]), stable=True)
            ]
            sorted_values = flattened[order]
            starts = torch.cat(
                (
                    torch.ones((1,), dtype=torch.bool, device=array.device),
                    sorted_values[1:] != sorted_values[:-1],
                )
            )
            sorted_groups = torch.cumsum(starts, dim=0, dtype=torch.int64) - 1
            values = sorted_values[starts]
            first_indices = order[starts]
            counts = torch.bincount(sorted_groups, minlength=values.numel())
            inverse_indices = sorted_groups[torch.argsort(order)]
            return (
                values,
                first_indices,
                torch.reshape(inverse_indices, array.shape),
                counts,
            )

        values, inverse_indices, counts = torch.unique(
            array,
            sorted=True,
            return_inverse=True,
            return_counts=True,
        )
        flattened_inverse = torch.reshape(inverse_indices, (-1,))
        positions = torch.arange(
            flattened.numel(), dtype=torch.int64, device=array.device
        )
        first_indices = torch.full(
            (values.numel(),),
            flattened.numel(),
            dtype=torch.int64,
            device=array.device,
        ).scatter_reduce(
            0,
            flattened_inverse,
            positions,
            reduce="amin",
            include_self=True,
        )
        return values, first_indices, inverse_indices, counts

    @classmethod
    def unique_all(cls, array: torch.Tensor, /) -> object:
        """Return values, first indices, inverse indices, and counts."""
        cls._validate("unique_all", (array,), {})
        return array_api_namespace.UniqueAllResult(
            *cls._unique_components(array)
        )

    @classmethod
    def unique_counts(cls, array: torch.Tensor, /) -> object:
        """Return sorted unique values and their counts."""
        cls._validate("unique_counts", (array,), {})
        if not array.dtype.is_complex:
            values, counts = torch.unique(
                array, sorted=True, return_counts=True
            )
            return array_api_namespace.UniqueCountsResult(values, counts)
        values, _indices, _inverse_indices, counts = cls._unique_components(
            array
        )
        return array_api_namespace.UniqueCountsResult(values, counts)

    @classmethod
    def unique_inverse(cls, array: torch.Tensor, /) -> object:
        """Return sorted unique values and inverse group indices."""
        cls._validate("unique_inverse", (array,), {})
        if not array.dtype.is_complex:
            values, inverse_indices = torch.unique(
                array, sorted=True, return_inverse=True
            )
            return array_api_namespace.UniqueInverseResult(
                values, inverse_indices
            )
        values, _indices, inverse_indices, _counts = cls._unique_components(
            array
        )
        return array_api_namespace.UniqueInverseResult(values, inverse_indices)

    @classmethod
    def unique_values(cls, array: torch.Tensor, /) -> torch.Tensor:
        """Return sorted unique values."""
        cls._validate("unique_values", (array,), {})
        if not array.dtype.is_complex:
            return typing.cast(torch.Tensor, torch.unique(array, sorted=True))
        values, _indices, _inverse_indices, _counts = cls._unique_components(
            array
        )
        return values


_NAMESPACE = _TorchNamespace()


def namespace() -> object:
    """Return the frozen compatibility namespace without allocating arrays."""
    return _NAMESPACE


def einsum(subscripts: str, *operands: object) -> object:
    """Evaluate Einstein summation through the private native extension."""
    return torch.einsum(subscripts, *operands)


def kron(first: object, second: object) -> object:
    """Evaluate a Kronecker product through the private native extension."""
    return torch.kron(
        typing.cast(torch.Tensor, first), typing.cast(torch.Tensor, second)
    )


def resolve_device(device: object | None) -> object:
    """Normalize a supported PyTorch CPU device."""
    if (
        device is None
        or (type(device) is str and device == "cpu")
        or (isinstance(device, torch.device) and str(device) == "cpu")
    ):
        return torch.device("cpu")
    raise errors.DeviceError("torch backend: only the CPU device is supported")


def validate_dtype(dtype: object) -> None:
    """Reject dtypes outside the frozen PyTorch release surface."""
    name = _dtype.dtype_name(dtype)
    if name is None or dtype is not getattr(torch, name, None):
        raise errors.DTypeError(
            "torch backend: dtype is outside the supported release surface"
        )
    _dtype.require_supported_dtype("torch", dtype, "torch backend")


def asarray_preserving_graph(
    value: object,
    *,
    dtype: object | None = None,
    device: object | None = None,
    copy: bool | None = None,
) -> torch.Tensor:
    """Create a same-backend tensor through the private graph-aware path."""
    _namespace.validate_array_arguments("torch", "convert_array", (value,), {})
    if dtype is not None:
        validate_dtype(dtype)
    native_value = typing.cast(torch.Tensor, value)
    return array_api_namespace.asarray(
        native_value,
        dtype=dtype,
        device=resolve_device(device),
        copy=copy,
        requires_grad=native_value.requires_grad,
    )


def to_cpu(value: object) -> object:
    """Transfer a native tensor to CPU at an explicit boundary."""
    return typing.cast(torch.Tensor, value).to(device=torch.device("cpu"))


def create_key(seed: int, *, device: object | None = None) -> object:
    """Create replayable immutable PyTorch random state."""
    del device
    return _state.CounterKey("torch", seed)


def has_active_graph(value: object) -> bool:
    """Detect reverse- or forward-mode graph state on a tensor."""
    native_value = typing.cast(torch.Tensor, value)
    if native_value.requires_grad:
        return True
    unpacked = _forward_ad.unpack_dual(native_value)
    return unpacked.tangent is not None


def _validate_transform_arguments(
    args: object,
    kwargs: object,
    operation: str,
    argument: int | None = None,
) -> None:
    """Require dense CPU tensors and an optional real differentiable input."""
    from asc.core.namespace import array_namespace
    from asc.tree import tree_leaves

    arrays = [
        value
        for value in tree_leaves((args, kwargs))
        if _array_api_compat.compat.is_array_api_obj(value)
    ]
    if not any(isinstance(value, torch.Tensor) for value in arrays):
        raise errors.DeviceError(
            f"{operation}: at least one concrete PyTorch CPU array is required"
        )
    array_namespace(*arrays)
    if argument is not None:
        positional = typing.cast(tuple[object, ...], args)
        if argument >= len(positional) or not isinstance(
            positional[argument], torch.Tensor
        ):
            raise errors.DTypeError(
                f"{operation}: differentiable argument must be a positional "
                "PyTorch array"
            )
        if not positional[argument].dtype.is_floating_point:
            raise errors.DTypeError(
                f"{operation}: differentiable argument must be real floating"
            )


def owns_key(key: object) -> bool:
    """Return whether ``key`` is PyTorch state."""
    return isinstance(key, _state.CounterKey) and key.backend == "torch"


def _generator(key: object) -> tuple[torch.Generator, _state.CounterKey]:
    if not owns_key(key):
        raise errors.RandomStateError("torch random: incompatible state")
    native_key = typing.cast(_state.CounterKey, key)
    if not 0 <= native_key.counter < 2**32:
        raise errors.RandomStateError(
            "torch random: counter must fit in an unsigned 32-bit integer"
        )
    if native_key.counter == 2**32 - 1:
        raise errors.RandomStateError(
            "torch random: state is exhausted and cannot be advanced"
        )
    combined = (native_key.seed << 32) | native_key.counter
    combined ^= combined >> 30
    combined = (combined * 0xBF58476D1CE4E5B9) & (2**64 - 1)
    combined ^= combined >> 27
    combined = (combined * 0x94D049BB133111EB) & (2**64 - 1)
    derived_seed = combined ^ (combined >> 31)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(derived_seed)
    # Torch's CPU generator initializes its MT19937 payload from only the
    # low 32 seed bits. Preserve the other mixed half in the second payload
    # word so the full bijective 64-bit state selects a distinct substream.
    generator_state = generator.get_state()
    state_words = generator_state.view(torch.int64)
    if state_words.numel() != 632:
        raise errors.RandomStateError(
            "torch random: unsupported CPU generator state representation"
        )
    state_words[4] = derived_seed >> 32
    generator.set_state(generator_state)
    return generator, native_key


def _advanced(key: _state.CounterKey) -> _state.CounterKey:
    if key.counter >= 2**32 - 1:
        raise errors.RandomStateError(
            "torch random: state is exhausted and cannot be advanced"
        )
    return dataclasses.replace(key, counter=key.counter + 1)


def uniform(
    shape: asc_typing.Shape,
    *,
    key: object,
    low: float,
    high: float,
    dtype: object | None,
) -> tuple[asc_typing.NativeArray, object]:
    """Sample PyTorch without process-global random state."""
    generator, native_key = _generator(key)
    requested_dtype = _floating_dtype(dtype, "uniform")
    information = torch.finfo(requested_dtype)
    if (
        low < information.min
        or high > information.max
        or high - low > information.max
    ):
        raise errors.RandomStateError(
            "torch random: interval is not representable in the requested dtype"
        )
    low_value = torch.tensor(low, dtype=requested_dtype, device="cpu")
    high_value = torch.tensor(high, dtype=requested_dtype, device="cpu")
    if float(low_value) != low or float(high_value) != high:
        raise errors.RandomStateError(
            "torch random: interval endpoints must be exactly representable "
            "in the requested dtype"
        )
    unit = torch.rand(
        shape,
        generator=generator,
        dtype=requested_dtype,
        device="cpu",
    )
    result = low + (high - low) * unit
    upper = torch.nextafter(
        high_value,
        torch.tensor(-math.inf, dtype=requested_dtype, device="cpu"),
    )
    result = torch.clamp(result, min=low_value, max=upper)
    return typing.cast(asc_typing.NativeArray, result), _advanced(native_key)


def _floating_dtype(dtype: object | None, operation: str) -> torch.dtype:
    requested = (
        torch.get_default_dtype()
        if dtype is None
        else typing.cast(torch.dtype, dtype)
    )
    try:
        validate_dtype(requested)
    except errors.DTypeError as exception:
        raise errors.RandomStateError(
            f"torch {operation}: requested dtype is outside the supported "
            "release surface"
        ) from exception
    if not requested.is_floating_point:
        raise errors.RandomStateError(
            f"torch {operation}: dtype must be a real floating dtype"
        )
    return requested


def normal(
    shape: asc_typing.Shape,
    *,
    key: object,
    mean: float,
    std: float,
    dtype: object | None,
) -> tuple[asc_typing.NativeArray, object]:
    """Sample a PyTorch normal distribution from explicit state."""
    generator, native_key = _generator(key)
    result = torch.randn(
        shape,
        generator=generator,
        dtype=_floating_dtype(dtype, "normal"),
        device="cpu",
    )
    return typing.cast(asc_typing.NativeArray, mean + std * result), _advanced(
        native_key
    )


def truncated_normal(
    shape: asc_typing.Shape,
    *,
    key: object,
    mean: float,
    std: float,
    lower: float,
    upper: float,
    dtype: object | None,
) -> tuple[asc_typing.NativeArray, object]:
    """Sample a PyTorch truncated normal with bounded rejection work."""
    generator, native_key = _generator(key)
    requested = _floating_dtype(dtype, "truncated_normal")
    standardized_lower = (lower - mean) / std
    standardized_upper = (upper - mean) / std
    if not (
        math.isfinite(standardized_lower)
        and math.isfinite(standardized_upper)
        and standardized_lower < standardized_upper
    ):
        raise errors.RandomStateError(
            "torch truncated_normal: standardized bounds must be finite and "
            "ordered"
        )
    count = math.prod(shape)
    standard = torch.empty((count,), dtype=torch.float64, device="cpu")
    remaining = torch.arange(count, dtype=torch.int64, device="cpu")
    reflected = standardized_upper <= 0.0
    if reflected:
        proposal_lower = -standardized_upper
        proposal_upper = -standardized_lower
    else:
        proposal_lower = standardized_lower
        proposal_upper = standardized_upper
    for _ in range(1024):
        if remaining.numel() == 0:
            break
        size = remaining.numel()
        if proposal_lower >= 0.0:
            alpha = 0.5 * (proposal_lower + math.hypot(proposal_lower, 2.0))
            mass = -math.expm1(-alpha * (proposal_upper - proposal_lower))
            proposal = (
                proposal_lower
                - torch.log1p(
                    -torch.rand(
                        (size,),
                        generator=generator,
                        dtype=torch.float64,
                        device="cpu",
                    )
                    * mass
                )
                / alpha
            )
            accepted = torch.rand(
                (size,),
                generator=generator,
                dtype=torch.float64,
                device="cpu",
            ) <= torch.exp(-0.5 * torch.square(proposal - alpha))
        elif proposal_upper - proposal_lower <= 2.0:
            proposal = proposal_lower + (
                proposal_upper - proposal_lower
            ) * torch.rand(
                (size,),
                generator=generator,
                dtype=torch.float64,
                device="cpu",
            )
            accepted = torch.rand(
                (size,),
                generator=generator,
                dtype=torch.float64,
                device="cpu",
            ) <= torch.exp(-0.5 * torch.square(proposal))
        else:
            proposal = torch.randn(
                (size,),
                generator=generator,
                dtype=torch.float64,
                device="cpu",
            )
            accepted = (proposal >= proposal_lower) & (
                proposal <= proposal_upper
            )
        selected = -proposal[accepted] if reflected else proposal[accepted]
        standard[remaining[accepted]] = selected
        remaining = remaining[~accepted]
    if remaining.numel():
        raise errors.RandomStateError(
            "torch truncated_normal: sampling did not converge"
        )
    result = (mean + std * standard.reshape(shape)).to(dtype=requested)
    result = torch.clamp(
        result,
        min=torch.tensor(lower, dtype=requested, device="cpu"),
        max=torch.tensor(upper, dtype=requested, device="cpu"),
    )
    return typing.cast(asc_typing.NativeArray, result), _advanced(native_key)


def randint(
    shape: asc_typing.Shape,
    *,
    key: object,
    low: int,
    high: int,
    dtype: object | None,
) -> tuple[asc_typing.NativeArray, object]:
    """Sample signed PyTorch integers from explicit state."""
    generator, native_key = _generator(key)
    requested = (
        torch.int64 if dtype is None else typing.cast(torch.dtype, dtype)
    )
    try:
        validate_dtype(requested)
    except errors.DTypeError as exception:
        raise errors.RandomStateError(
            "torch randint: requested dtype is outside the supported release "
            "surface"
        ) from exception
    if requested not in {torch.int8, torch.int16, torch.int32, torch.int64}:
        raise errors.RandomStateError(
            "torch randint: dtype must be signed integer"
        )
    information = torch.iinfo(requested)
    if low < information.min or high > information.max + 1:
        raise errors.RandomStateError(
            "torch randint: bounds are outside the requested dtype range"
        )
    if requested == torch.int64 and high == information.max + 1:
        result = _torch_random.randint_max_endpoint(
            shape, generator=generator, low=low
        )
    else:
        result = torch.randint(
            low,
            high,
            shape,
            generator=generator,
            dtype=requested,
            device="cpu",
        )
    return typing.cast(asc_typing.NativeArray, result), _advanced(native_key)


def bernoulli(
    shape: asc_typing.Shape,
    *,
    key: object,
    probability: float,
) -> tuple[asc_typing.NativeArray, object]:
    """Sample Boolean PyTorch Bernoulli values from explicit state."""
    generator, native_key = _generator(key)
    result = torch.rand(shape, generator=generator, device="cpu") < probability
    return typing.cast(asc_typing.NativeArray, result), _advanced(native_key)


def gamma(
    shape: asc_typing.Shape,
    *,
    key: object,
    concentration: float,
    scale: float,
    dtype: object | None,
) -> tuple[asc_typing.NativeArray, object]:
    """Sample a PyTorch gamma distribution from explicit state."""
    generator, native_key = _generator(key)
    requested = _floating_dtype(dtype, "gamma")
    calculation_dtype = (
        torch.float32 if torch.finfo(requested).bits < 32 else requested
    )
    parameters = torch.full(
        shape, concentration, dtype=calculation_dtype, device="cpu"
    )
    result = (
        torch._standard_gamma(  # pyright: ignore[reportPrivateUsage, reportPrivateImportUsage]
            parameters, generator=generator
        )
        * scale
    )
    return typing.cast(
        asc_typing.NativeArray, result.to(dtype=requested)
    ), _advanced(native_key)


def exponential(
    shape: asc_typing.Shape,
    *,
    key: object,
    scale: float,
    dtype: object | None,
) -> tuple[asc_typing.NativeArray, object]:
    """Sample a PyTorch exponential distribution from explicit state."""
    generator, native_key = _generator(key)
    uniform_values = torch.rand(
        shape,
        generator=generator,
        dtype=_floating_dtype(dtype, "exponential"),
        device="cpu",
    )
    result = -scale * torch.log1p(-uniform_values)
    return typing.cast(asc_typing.NativeArray, result), _advanced(native_key)


def choice(
    population: object,
    shape: asc_typing.Shape,
    *,
    key: object,
    replace: bool,
    probabilities: object | None,
) -> tuple[asc_typing.NativeArray, object]:
    """Sample from a PyTorch population using explicit state."""
    generator, native_key = _generator(key)
    values = (
        torch.arange(population, device="cpu")
        if isinstance(population, int)
        else typing.cast(torch.Tensor, population)
    )
    count = 1
    for extent in shape:
        count *= extent
    if count == 0:
        result = torch.empty(
            shape,
            dtype=values.dtype,
            device=values.device,
        )
        return typing.cast(asc_typing.NativeArray, result), _advanced(
            native_key
        )
    weights = (
        torch.ones(values.shape[0], dtype=torch.float64, device="cpu")
        if probabilities is None
        else typing.cast(torch.Tensor, probabilities).to(dtype=torch.float64)
    )
    selected = torch.multinomial(
        weights, count, replacement=replace, generator=generator
    )
    result = values[selected].reshape(shape)
    return typing.cast(asc_typing.NativeArray, result), _advanced(native_key)


def permutation(
    value: object, *, key: object
) -> tuple[asc_typing.NativeArray, object]:
    """Return a functional PyTorch permutation."""
    generator, native_key = _generator(key)
    if isinstance(value, int):
        result = torch.randperm(value, generator=generator, device="cpu")
    else:
        native = typing.cast(torch.Tensor, value)
        order = torch.randperm(
            native.shape[0], generator=generator, device="cpu"
        )
        result = native[order]
    return typing.cast(asc_typing.NativeArray, result), _advanced(native_key)


def split_key(key: object, count: int) -> tuple[object, ...]:
    """Derive deterministic independent PyTorch child states."""
    _unused_generator, native_key = _generator(key)
    if count > 2**32:
        raise errors.RandomStateError(
            "torch random: cannot derive more than 2**32 distinct child states"
        )
    material = (
        native_key.seed.to_bytes(4, "little")
        + native_key.counter.to_bytes(4, "little")
        + b"asc-split"
    )
    digest = hashlib.blake2s(material, digest_size=8).digest()
    multiplier = int.from_bytes(digest[:4], "little") | 1
    offset = int.from_bytes(digest[4:], "little")
    return tuple(
        _state.CounterKey("torch", (multiplier * child_index + offset) % 2**32)
        for child_index in range(count)
    )


def index_add(
    array: asc_typing.NativeArray,
    indices: asc_typing.NativeArray,
    values: asc_typing.NativeArray,
    *,
    axis: int,
    update_shape: asc_typing.Shape,
) -> asc_typing.NativeArray:
    """Call the functional PyTorch indexed-add operation."""
    native_array = typing.cast(torch.Tensor, array)
    native_indices = typing.cast(torch.Tensor, indices)
    native_values = typing.cast(torch.Tensor, values)
    if native_indices.dtype not in {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }:
        raise errors.IndexContractError(
            "index_add: PyTorch indices must have a signed integer dtype"
        )
    try:
        _check_tensor_all(
            (native_indices >= 0) & (native_indices < native_array.shape[axis]),
            lambda: "index_add: index is out of bounds",
        )
    except RuntimeError as exception:
        raise errors.IndexContractError(
            "index_add: index is out of bounds"
        ) from exception
    broadcast_values = torch.broadcast_to(native_values, update_shape)
    result = torch.index_add(
        native_array,
        axis,
        native_indices.to(dtype=torch.int64),
        broadcast_values,
    )
    return typing.cast(asc_typing.NativeArray, result)


def index_update(
    array: asc_typing.NativeArray,
    indices: asc_typing.NativeArray,
    values: asc_typing.NativeArray,
    *,
    axis: int,
    update_shape: asc_typing.Shape,
    reduction: str,
) -> asc_typing.NativeArray:
    """Apply one functional indexed update with native PyTorch operators."""
    native_array = typing.cast(torch.Tensor, array)
    native_indices = typing.cast(torch.Tensor, indices)
    native_values = typing.cast(torch.Tensor, values)
    if native_indices.dtype not in {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }:
        raise errors.IndexUpdateError(
            "index update: PyTorch indices must be signed integers"
        )
    try:
        _check_tensor_all(
            (native_indices >= 0) & (native_indices < native_array.shape[axis]),
            lambda: "index update: index is out of bounds",
        )
    except RuntimeError as exception:
        raise errors.IndexUpdateError(
            "index update: index is out of bounds"
        ) from exception
    indices64 = native_indices.to(dtype=torch.int64)
    if reduction == "set":
        ordered = torch.sort(indices64).values
        try:
            _check_tensor_all(
                ordered[1:] != ordered[:-1],
                lambda: "index_set: duplicate indices are unsupported",
            )
        except RuntimeError as exception:
            raise errors.DuplicateIndexError(
                "index_set: duplicate indices have no deterministic set policy"
            ) from exception
        source = torch.broadcast_to(native_values, update_shape)
        if native_array.dtype in _WIDE_UNSIGNED_DTYPES:
            dtype = native_array.dtype
            carrier_dtype = _SIGNED_VIEW_DTYPE[dtype]
            result = torch.index_copy(
                native_array.view(carrier_dtype),
                axis,
                indices64,
                source.view(carrier_dtype),
            ).view(dtype)
        else:
            result = torch.index_copy(
                native_array,
                axis,
                indices64,
                source,
            )
        return typing.cast(asc_typing.NativeArray, result)
    source = torch.broadcast_to(native_values, update_shape)
    index_shape = [1] * source.ndim
    index_shape[axis] = indices64.numel()
    expanded = indices64.reshape(index_shape).expand(source.shape)
    reduce = {"add": "sum", "multiply": "prod", "min": "amin", "max": "amax"}[
        reduction
    ]
    if native_array.dtype not in _WIDE_UNSIGNED_DTYPES:
        return typing.cast(
            asc_typing.NativeArray,
            torch.scatter_reduce(
                native_array,
                axis,
                expanded,
                source,
                reduce=reduce,
                include_self=True,
            ),
        )

    dtype = native_array.dtype
    if dtype in _WIDER_SIGNED_DTYPE:
        carrier_dtype = _WIDER_SIGNED_DTYPE[dtype]
        destination_carrier = native_array.to(dtype=carrier_dtype)
        source_carrier = source.to(dtype=carrier_dtype)
        result = torch.scatter_reduce(
            destination_carrier,
            axis,
            expanded,
            source_carrier,
            reduce=reduce,
            include_self=True,
        ).to(dtype=dtype)
    elif reduction in {"min", "max"}:
        result = _restore_ordered_unsigned(
            torch.scatter_reduce(
                _ordered_unsigned(native_array),
                axis,
                expanded,
                _ordered_unsigned(source),
                reduce=reduce,
                include_self=True,
            ),
            dtype,
        )
    else:
        result = torch.scatter_reduce(
            native_array.view(torch.int64),
            axis,
            expanded,
            source.view(torch.int64),
            reduce=reduce,
            include_self=True,
        ).view(dtype)
    return typing.cast(asc_typing.NativeArray, result)


def check_index_bounds(invalid: object, operation: str) -> None:
    """Raise for invalid indices without scalarizing Torch BatchedTensors."""
    condition = ~typing.cast(torch.Tensor, invalid)
    try:
        _check_tensor_all(
            condition,
            lambda: f"{operation}: index is out of bounds",
        )
    except RuntimeError as exception:
        raise IndexError(f"{operation}: index is out of bounds") from exception


def value_and_grad(
    function: typing.Callable[..., object],
    argument: int,
) -> typing.Callable[..., tuple[object, object]]:
    """Build a PyTorch ``torch.func`` value-and-gradient callable."""
    native_transform = typing.cast(
        typing.Callable[..., typing.Callable[..., tuple[object, object]]],
        torch.func.grad_and_value,
    )(function, argnums=argument)

    @functools.wraps(function)
    def transformed(*args: object, **kwargs: object) -> tuple[object, object]:
        _validate_transform_arguments(args, kwargs, "value_and_grad", argument)
        gradient, value = native_transform(*args, **kwargs)
        return value, gradient

    return transformed


def grad(
    function: typing.Callable[..., object], argument: int
) -> typing.Callable[..., object]:
    """Build a PyTorch scalar-output gradient callable."""
    native = torch.func.grad(function, argnums=argument)

    @functools.wraps(function)
    def transformed(*args: object, **kwargs: object) -> object:
        _validate_transform_arguments(args, kwargs, "grad", argument)
        return native(*args, **kwargs)

    return transformed


def jacobian(
    function: typing.Callable[..., object], argument: int
) -> typing.Callable[..., object]:
    """Build a reverse-mode PyTorch Jacobian callable."""
    native = torch.func.jacrev(function, argnums=argument)

    @functools.wraps(function)
    def transformed(*args: object, **kwargs: object) -> object:
        _validate_transform_arguments(args, kwargs, "jacobian", argument)
        return native(*args, **kwargs)

    return transformed


def hessian(
    function: typing.Callable[..., object], argument: int
) -> typing.Callable[..., object]:
    """Build a PyTorch Hessian callable."""
    first = torch.func.jacrev(function, argnums=argument)
    native = torch.func.jacrev(first, argnums=argument)

    @functools.wraps(function)
    def transformed(*args: object, **kwargs: object) -> object:
        _validate_transform_arguments(args, kwargs, "hessian", argument)
        return native(*args, **kwargs)

    return transformed


def jvp(
    function: typing.Callable[..., object],
    primals: tuple[object, ...],
    tangents: tuple[object, ...],
) -> tuple[object, object]:
    """Evaluate a PyTorch forward-mode Jacobian-vector product."""
    _validate_transform_arguments((primals, tangents), {}, "jvp")
    _validate_real_transform_arrays(primals, "jvp primals")
    _validate_real_transform_arrays(tangents, "jvp tangents")
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"`torch\.jit\.script` is deprecated\..*",
            category=DeprecationWarning,
        )
        return typing.cast(
            tuple[object, object], torch.func.jvp(function, primals, tangents)
        )


def vjp(
    function: typing.Callable[..., object], primals: tuple[object, ...]
) -> tuple[object, typing.Callable[..., object]]:
    """Evaluate a PyTorch reverse-mode vector-Jacobian product setup."""
    _validate_transform_arguments(primals, {}, "vjp")
    _validate_real_transform_arrays(primals, "vjp primals")
    result = typing.cast(
        tuple[object, typing.Callable[..., object]],
        torch.func.vjp(function, *primals),
    )
    native_pullback = result[1]

    def pullback(*cotangents: object) -> object:
        _validate_transform_arguments(cotangents, {}, "vjp pullback")
        _validate_real_transform_arrays(cotangents, "vjp cotangents")
        return native_pullback(*cotangents)

    return result[0], pullback


def _validate_real_transform_arrays(
    values: tuple[object, ...], operation: str
) -> None:
    """Require Torch autodiff PyTree leaves to be real floating arrays."""
    from asc.tree import tree_leaves

    leaves = tree_leaves(values)
    if not leaves or any(
        not isinstance(value, torch.Tensor) for value in leaves
    ):
        raise errors.DTypeError(f"{operation}: operands must be PyTorch arrays")
    if any(not value.dtype.is_floating_point for value in leaves):
        raise errors.DTypeError(
            f"{operation}: operands must have real floating dtypes"
        )


def compile_function(
    function: typing.Callable[..., object],
) -> typing.Callable[..., object]:
    """Compile a function with PyTorch's native compiler."""
    return _torch.compile(function, backend="aot_eager", fullgraph=False)


def vmap(
    function: typing.Callable[..., object],
    in_axes: object,
    out_axes: object,
) -> typing.Callable[..., object]:
    """Vectorize a PyTorch callable over the supported axis subset."""
    native = typing.cast(
        typing.Callable[..., object],
        torch.vmap(function, in_dims=in_axes, out_dims=out_axes),
    )

    @functools.wraps(function)
    def transformed(*args: object, **kwargs: object) -> object:
        _validate_transform_arguments(args, kwargs, "vmap")
        return native(*args, **kwargs)

    return transformed
