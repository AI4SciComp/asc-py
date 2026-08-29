# Copyright 2026 AI4SciComp contributors.
# SPDX-License-Identifier: Apache-2.0
"""Graph-safe portable activation functions."""

from __future__ import annotations

import math

from asc import _array_api_compat, errors
from asc import typing as asc_typing
from asc.core import namespace as namespace_module
from asc.core._scalar import normalize_real_scalar


def _real_namespace(array: object, operation: str) -> asc_typing.ArrayNamespace:
    xp = namespace_module.array_namespace(array)
    if not xp.isdtype(array.dtype, "real floating"):
        raise errors.DTypeError(
            f"{operation}: expected a real floating array; cast explicitly"
        )
    return xp


def _nonnegative_coefficient(
    value: object, operation: str, parameter: str
) -> float:
    """Validate an activation coefficient without coercing native arrays."""
    if (
        _array_api_compat.compat.is_array_api_obj(value)
        or isinstance(value, bool)
        or not isinstance(value, (int, float))
    ):
        raise ValueError(
            f"{operation}: {parameter} must be a finite non-negative "
            "Python real scalar"
        )
    try:
        result = float(value)
    except OverflowError as exception:
        raise ValueError(
            f"{operation}: {parameter} must be a finite non-negative "
            "Python real scalar"
        ) from exception
    if not math.isfinite(result) or result < 0:
        raise ValueError(
            f"{operation}: {parameter} must be a finite non-negative "
            "Python real scalar"
        )
    return result


def relu(array: object) -> object:
    """Return ``max(array, 0)`` elementwise."""
    xp = _real_namespace(array, "relu")
    return xp.maximum(array, xp.zeros_like(array))


def leaky_relu(array: object, *, negative_slope: float = 0.01) -> object:
    """Return a rectifier with an explicit negative slope."""
    xp = _real_namespace(array, "leaky_relu")
    negative_slope = _nonnegative_coefficient(
        negative_slope, "leaky_relu", "negative_slope"
    )
    negative_slope = normalize_real_scalar(
        xp,
        array.dtype,
        negative_slope,
        "leaky_relu",
        "negative_slope",
        device=_array_api_compat.compat.device(array),
    )
    if negative_slope == 0:
        return xp.maximum(array, xp.zeros_like(array))
    return xp.where(array >= 0, array, negative_slope * array)


def elu(array: object, *, alpha: float = 1.0) -> object:
    """Return the exponential linear unit."""
    xp = _real_namespace(array, "elu")
    alpha = _nonnegative_coefficient(alpha, "elu", "alpha")
    alpha = normalize_real_scalar(
        xp,
        array.dtype,
        alpha,
        "elu",
        "alpha",
        device=_array_api_compat.compat.device(array),
    )
    zero = xp.zeros_like(array)
    return xp.maximum(array, zero) + alpha * xp.expm1(xp.minimum(array, zero))


def gelu(array: object) -> object:
    """Return the tanh-approximation Gaussian error linear unit."""
    xp = _real_namespace(array, "gelu")
    coefficient = math.sqrt(2.0 / math.pi)
    bounded = xp.clip(array, -10.0, 10.0)
    inner = coefficient * (bounded + 0.044715 * bounded * bounded * bounded)
    approximation = 0.5 * bounded * (1.0 + xp.tanh(inner))
    zero = xp.zeros_like(array)
    return xp.where(
        array > 10.0, array, xp.where(array < -10.0, zero, approximation)
    )


def selu(array: object) -> object:
    """Return the self-normalizing exponential linear unit."""
    xp = _real_namespace(array, "selu")
    alpha = 1.6732632423543772
    scale = 1.0507009873554805
    zero = xp.zeros_like(array)
    return scale * (
        xp.maximum(array, zero) + alpha * xp.expm1(xp.minimum(array, zero))
    )


def sigmoid(array: object) -> object:
    """Return the logistic sigmoid."""
    xp = _real_namespace(array, "sigmoid")
    zero = xp.zeros_like(array)
    non_negative = array >= 0
    positive_input = xp.where(non_negative, array, zero)
    negative_input = xp.where(non_negative, zero, array)
    positive = 1.0 / (1.0 + xp.exp(-positive_input))
    exponential = xp.exp(negative_input)
    negative = exponential / (1.0 + exponential)
    return xp.where(non_negative, positive, negative)


def silu(array: object) -> object:
    """Return the sigmoid linear unit."""
    xp = _real_namespace(array, "silu")
    negative_infinity = xp.logical_and(xp.isinf(array), xp.signbit(array))
    finite_factor = xp.where(negative_infinity, xp.zeros_like(array), array)
    return finite_factor * sigmoid(array)


def softplus(array: object) -> object:
    """Return a numerically stable smooth rectifier."""
    xp = _real_namespace(array, "softplus")
    return xp.log1p(xp.exp(-xp.abs(array))) + xp.maximum(
        array, xp.zeros_like(array)
    )


def softsign(array: object) -> object:
    """Return ``array / (1 + abs(array))``."""
    xp = _real_namespace(array, "softsign")
    infinite = xp.isinf(array)
    finite_array = xp.where(infinite, xp.zeros_like(array), array)
    finite_result = finite_array / (1.0 + xp.abs(finite_array))
    return xp.where(infinite, xp.sign(array), finite_result)


def tanhshrink(array: object) -> object:
    """Return ``array - tanh(array)``."""
    xp = _real_namespace(array, "tanhshrink")
    return array - xp.tanh(array)


__all__ = [
    "elu",
    "gelu",
    "leaky_relu",
    "relu",
    "selu",
    "sigmoid",
    "silu",
    "softplus",
    "softsign",
    "tanhshrink",
]
