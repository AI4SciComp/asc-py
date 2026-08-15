# Copyright 2026 AI4SciComp contributors.
# SPDX-License-Identifier: Apache-2.0
"""Portable validation for Python scalars cast to explicit dtypes."""

from __future__ import annotations

import math
import numbers
import struct

from asc import errors
from asc import typing as asc_typing


def _dtype_name(dtype: object) -> str:
    """Return the portable scalar dtype name without importing a backend."""
    name = getattr(dtype, "name", None)
    if isinstance(name, str):
        return name
    name = getattr(dtype, "__name__", None)
    if isinstance(name, str):
        return name
    return str(dtype).rsplit(".", maxsplit=1)[-1]


def _rounded_real(dtype: object, value: float) -> float:
    """Round one real value using Python-only IEEE conversion."""
    name = _dtype_name(dtype)
    if name in {"float64", "complex128"}:
        return value
    if name in {"float32", "complex64"}:
        return struct.unpack("!f", struct.pack("!f", value))[0]
    if name == "float16":
        return struct.unpack("!e", struct.pack("!e", value))[0]
    if name == "bfloat16":
        bits = struct.unpack("!I", struct.pack("!f", value))[0]
        least_significant = (bits >> 16) & 1
        rounded = (bits + 0x7FFF + least_significant) & 0xFFFF0000
        return struct.unpack("!f", struct.pack("!I", rounded))[0]
    raise TypeError(f"unsupported portable floating dtype {name!r}")


def normalize_real_scalar(
    namespace: asc_typing.ArrayNamespace,
    dtype: object,
    value: float,
    operation: str,
    parameter: str,
    *,
    device: object | None = None,
) -> float:
    """Round a prevalidated finite real without overflow or underflow."""
    del device
    try:
        if not namespace.isdtype(dtype, "real floating"):
            raise errors.DTypeError(
                f"{operation}: {parameter} requires a real floating dtype"
            )
        rounded = _rounded_real(dtype, value)
    except errors.DTypeError:
        raise
    except (
        AttributeError,
        OverflowError,
        RuntimeError,
        TypeError,
        ValueError,
        Warning,
    ) as exception:
        raise errors.DTypeError(
            f"{operation}: {parameter} is not representable in {dtype!r}"
        ) from exception
    if not math.isfinite(rounded) or (value != 0.0 and rounded == 0.0):
        raise errors.DTypeError(
            f"{operation}: {parameter} is not representable in {dtype!r}"
        )
    return rounded


def require_representable_scalar(
    namespace: asc_typing.ArrayNamespace,
    dtype: object | None,
    value: object,
    operation: str,
) -> None:
    """Reject numeric scalars outside an explicit dtype's finite range."""
    if not isinstance(value, numbers.Number):
        raise errors.DTypeError(
            f"{operation}: fill value must be a numeric Python scalar"
        )
    if dtype is None:
        return
    try:
        if namespace.isdtype(dtype, "bool"):
            if not isinstance(value, bool):
                raise errors.DTypeError(
                    f"{operation}: fill value is not representable in {dtype!r}"
                )
            return
        if namespace.isdtype(dtype, ("signed integer", "unsigned integer")):
            if not isinstance(value, numbers.Real):
                raise errors.DTypeError(
                    f"{operation}: fill value is not representable in {dtype!r}"
                )
            if isinstance(value, numbers.Integral):
                integer = int(value)
            else:
                numeric = float(value)
                if not math.isfinite(numeric) or not numeric.is_integer():
                    raise errors.DTypeError(
                        f"{operation}: fill value is not representable in "
                        f"{dtype!r}"
                    )
                integer = int(numeric)
            information = namespace.iinfo(dtype)
            if integer < int(information.min) or integer > int(information.max):
                raise errors.DTypeError(
                    f"{operation}: fill value is not representable in {dtype!r}"
                )
            return
        if namespace.isdtype(dtype, "real floating"):
            if not isinstance(value, numbers.Real):
                raise errors.DTypeError(
                    f"{operation}: fill value is not representable in {dtype!r}"
                )
            numeric = float(value)
            if math.isfinite(numeric):
                rounded = _rounded_real(dtype, numeric)
            else:
                rounded = numeric
            if not math.isfinite(rounded) and math.isfinite(numeric):
                raise errors.DTypeError(
                    f"{operation}: fill value is not representable in {dtype!r}"
                )
            if numeric != 0.0 and rounded == 0.0:
                raise errors.DTypeError(
                    f"{operation}: fill value is not representable in {dtype!r}"
                )
            return
        if namespace.isdtype(dtype, "complex floating"):
            numeric = complex(value)
            rounded_components = tuple(
                _rounded_real(dtype, component)
                if math.isfinite(component)
                else component
                for component in (numeric.real, numeric.imag)
            )
            if any(
                (math.isfinite(component) and not math.isfinite(rounded))
                or (component != 0.0 and rounded == 0.0)
                for component, rounded in zip(
                    (numeric.real, numeric.imag),
                    rounded_components,
                    strict=True,
                )
            ):
                raise errors.DTypeError(
                    f"{operation}: fill value is not representable in {dtype!r}"
                )
            return
    except errors.DTypeError:
        raise
    except (AttributeError, OverflowError, TypeError, ValueError) as exception:
        raise errors.DTypeError(
            f"{operation}: cannot validate fill value for dtype {dtype!r}"
        ) from exception
    raise errors.DTypeError(f"{operation}: dtype {dtype!r} is unsupported")


__all__: list[str] = []
