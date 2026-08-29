# Copyright 2026 AI4SciComp contributors.
# SPDX-License-Identifier: Apache-2.0
"""Backend-neutral regression metrics."""

from __future__ import annotations

import math
import typing

from asc import errors
from asc import typing as asc_typing
from asc.core import namespace as namespace_module

Axis = int | tuple[int, ...] | None
Reduction = typing.Literal["mean", "sum", "none"]


def _floating_exponent_limits(xp: object, dtype: object) -> tuple[int, int]:
    """Return the largest normal and smallest subnormal binary exponents."""
    info = xp.finfo(dtype)
    maximum = math.ceil(math.log2(float(info.max) / 2.0))
    minimum = math.floor(math.log2(float(info.smallest_normal))) + math.floor(
        math.log2(float(info.eps))
    )
    return maximum, minimum


def _inputs(
    prediction: object, target: object, operation: str
) -> tuple[asc_typing.ArrayNamespace, object, object]:
    """Validate and promote a pair of metric operands."""
    xp = namespace_module.array_namespace(prediction, target)
    if prediction.shape != target.shape:
        raise ValueError(
            f"{operation}: prediction and target shapes must match exactly"
        )
    if any(
        not xp.isdtype(value.dtype, "real floating")
        for value in (prediction, target)
    ):
        raise errors.DTypeError(
            f"{operation}: prediction and target must be real floating arrays"
        )
    try:
        dtype = xp.result_type(prediction, target)
        if int(xp.finfo(dtype).bits) < 32:
            dtype = xp.float32
        prediction = xp.astype(prediction, dtype, copy=True)
        target = xp.astype(target, dtype, copy=True)
    except (RuntimeError, TypeError, ValueError) as exception:
        raise errors.DTypeError(
            f"{operation}: operand dtypes do not have a portable promotion"
        ) from exception
    return xp, prediction, target


def _require_nonempty_reduction(
    shape: tuple[int, ...], axis: Axis, operation: str
) -> None:
    """Reject reductions over empty axes before backend-specific primitives."""
    if axis is None:
        axes = tuple(range(len(shape)))
    elif isinstance(axis, int) and not isinstance(axis, bool):
        axes = (axis,)
    elif isinstance(axis, tuple) and all(
        isinstance(entry, int) and not isinstance(entry, bool) for entry in axis
    ):
        axes = axis
    else:
        raise TypeError(
            f"{operation}: axis must be an integer, tuple of integers, or None"
        )
    normalized = tuple(
        entry + len(shape) if entry < 0 else entry for entry in axes
    )
    if any(entry < 0 or entry >= len(shape) for entry in normalized):
        return
    if any(shape[entry] == 0 for entry in normalized):
        raise ValueError(f"{operation}: cannot reduce a zero-sized axis")


def _validated_keepdims(keepdims: object, operation: str) -> bool:
    """Require the public reduction shape control to be a Python Boolean."""
    if type(keepdims) is not bool:
        raise TypeError(f"{operation}: keepdims must be a Python Boolean")
    return typing.cast(bool, keepdims)


def _scaled_root_mean_square(
    xp: asc_typing.ArrayNamespace,
    value: object,
    axis: Axis,
) -> tuple[object, object]:
    """Return a scale and stable normalized root mean square."""
    if int(xp.finfo(value.dtype).bits) < 32:
        value = xp.astype(value, xp.float32, copy=True)
    finite = xp.isfinite(value)
    finite_value = xp.where(finite, value, xp.zeros_like(value))
    scale = xp.max(xp.abs(finite_value), axis=axis, keepdims=True)
    safe_scale = xp.where(
        scale == xp.zeros_like(scale), xp.ones_like(scale), scale
    )
    normalized = xp.where(
        finite,
        finite_value / xp.broadcast_to(safe_scale, value.shape),
        value,
    )
    root_mean_square = xp.sqrt(
        xp.mean(normalized * normalized, axis=axis, keepdims=True)
    )
    return scale, root_mean_square


def _scaled_difference_sum_squares(
    xp: asc_typing.ArrayNamespace,
    first: object,
    second: object,
    axis: Axis,
) -> tuple[object, object, object]:
    """Return two scales and a sum of squares for a stable difference."""
    operand_scale, normalized = _scaled_normalized_difference(
        xp, first, second, axis
    )
    difference_scale, root_mean_square = _scaled_root_mean_square(
        xp, normalized, axis
    )
    return operand_scale, difference_scale, root_mean_square


def _scaled_normalized_difference(
    xp: asc_typing.ArrayNamespace,
    first: object,
    second: object,
    axis: Axis,
) -> tuple[object, object]:
    """Scale finite pairs while retaining non-finite residual semantics."""
    finite = xp.isfinite(first) & xp.isfinite(second)
    finite_first = xp.where(finite, first, xp.zeros_like(first))
    finite_second = xp.where(finite, second, xp.zeros_like(second))
    operand_scale = xp.max(
        xp.maximum(xp.abs(finite_first), xp.abs(finite_second)),
        axis=axis,
        keepdims=True,
    )
    safe_operand_scale = xp.where(
        operand_scale == xp.zeros_like(operand_scale),
        xp.ones_like(operand_scale),
        operand_scale,
    )
    broadcast_scale = xp.broadcast_to(safe_operand_scale, first.shape)
    finite_difference = (
        finite_first / broadcast_scale - finite_second / broadcast_scale
    )
    infinite_difference = (xp.isinf(first) | xp.isinf(second)) & (
        first != second
    )
    nonfinite_difference = xp.where(
        infinite_difference,
        xp.full_like(finite_difference, float("inf")),
        xp.full_like(finite_difference, float("nan")),
    )
    return operand_scale, xp.where(
        finite,
        finite_difference,
        nonfinite_difference,
    )


def _scaled_absolute_difference(
    xp: asc_typing.ArrayNamespace,
    first: object,
    second: object,
    axis: Axis,
) -> tuple[object, object]:
    """Return an operand scale and normalized absolute differences."""
    operand_scale, normalized = _scaled_normalized_difference(
        xp, first, second, axis
    )
    return operand_scale, xp.abs(normalized)


def _nonnegative_product(
    xp: asc_typing.ArrayNamespace, first: object, second: object
) -> object:
    """Multiply non-negative factors without avoidable overflow warnings."""
    zero = xp.zeros_like(first)
    one = xp.ones_like(first)
    maximum = xp.full_like(first, float(xp.finfo(first.dtype).max))
    finite_factors = xp.isfinite(first) & xp.isfinite(second)
    safe_second = xp.where(finite_factors & (second > one), second, one)
    representable = finite_factors & (
        (first == zero) | (second <= one) | (first <= maximum / safe_second)
    )
    product = xp.where(representable, first, zero) * xp.where(
        representable, second, one
    )
    result = xp.where(
        representable,
        product,
        xp.full_like(product, float("inf")),
    )
    has_nan = xp.isnan(first) | xp.isnan(second)
    return xp.where(has_nan, xp.full_like(result, float("nan")), result)


def _power_of_two(
    xp: asc_typing.ArrayNamespace, reference: object, exponent: object
) -> object:
    """Return exact dtype-native powers of two using standard operations."""
    maximum_exponent, minimum_exponent = _floating_exponent_limits(
        xp, reference.dtype
    )
    zero = xp.zeros_like(reference)
    one = xp.ones_like(reference)
    two = xp.full_like(reference, 2.0)

    positive = one
    factor = two
    bit = 1
    while bit <= maximum_exponent:
        quotient = xp.floor(xp.maximum(exponent, zero) / bit)
        use_factor = xp.remainder(quotient, two) == one
        positive = positive * xp.where(use_factor, factor, one)
        if bit * 2 <= maximum_exponent:
            factor = factor * factor
        bit *= 2

    negative = one
    factor = xp.full_like(reference, 0.5)
    bit = 1
    while bit <= -minimum_exponent:
        quotient = xp.floor(xp.maximum(-exponent, zero) / bit)
        use_factor = xp.remainder(quotient, two) == one
        negative = negative * xp.where(use_factor, factor, one)
        if bit * 2 <= -minimum_exponent:
            factor = factor * factor
        bit *= 2
    return positive * negative


def _standard_frexp(
    xp: asc_typing.ArrayNamespace, value: object
) -> tuple[object, object]:
    """Split positive values into mantissa/exponent on the frozen surface."""
    one = xp.ones_like(value)
    two = xp.full_like(value, 2.0)
    maximum_exponent, _ = _floating_exponent_limits(xp, value.dtype)
    exponent = xp.minimum(
        xp.floor(xp.log2(value)),
        xp.full_like(value, float(maximum_exponent)),
    )
    mantissa = value / _power_of_two(xp, value, exponent)
    too_small = mantissa < one
    too_large = mantissa >= two
    doubled = xp.where(too_small, mantissa, one) * two
    mantissa = xp.where(
        too_small,
        doubled,
        xp.where(too_large, mantissa / two, mantissa),
    )
    exponent = xp.where(
        too_small,
        exponent - one,
        xp.where(too_large, exponent + one, exponent),
    )
    return mantissa, exponent


def _stable_absolute_error(
    xp: asc_typing.ArrayNamespace,
    prediction: object,
    target: object,
    axis: Axis,
    reduction: Reduction,
    keepdims: bool,
) -> object:
    """Reduce absolute differences after scale-safe subtraction."""
    if reduction == "none":
        if axis is not None or keepdims:
            raise ValueError(
                "metric: axis and keepdims require reduction='mean' or 'sum'"
            )
        scale, normalized = _scaled_absolute_difference(
            xp, prediction, target, ()
        )
        return _nonnegative_product(xp, scale, normalized)
    if reduction not in {"mean", "sum"}:
        raise ValueError("metric: reduction must be 'mean', 'sum', or 'none'")
    scale, normalized = _scaled_absolute_difference(
        xp, prediction, target, axis
    )
    reduced = getattr(xp, reduction)(normalized, axis=axis, keepdims=True)
    result = _nonnegative_product(xp, scale, reduced)
    return _squeeze_reduction(xp, result, axis, keepdims)


def _stable_squared_error(
    xp: asc_typing.ArrayNamespace,
    prediction: object,
    target: object,
    axis: Axis,
    reduction: Reduction,
    keepdims: bool,
) -> object:
    """Reduce squared differences without squaring unscaled residuals."""
    if reduction == "none":
        if axis is not None or keepdims:
            raise ValueError(
                "metric: axis and keepdims require reduction='mean' or 'sum'"
            )
        scale, normalized = _scaled_absolute_difference(
            xp, prediction, target, ()
        )
        root = _nonnegative_product(xp, scale, normalized)
    elif reduction in {"mean", "sum"}:
        scale, normalized = _scaled_absolute_difference(
            xp, prediction, target, axis
        )
        normalized_square = normalized * normalized
        reduced = getattr(xp, reduction)(
            normalized_square, axis=axis, keepdims=True
        )
        root = _nonnegative_product(xp, scale, xp.sqrt(reduced))
    else:
        raise ValueError("metric: reduction must be 'mean', 'sum', or 'none'")
    maximum_root = xp.sqrt(xp.full_like(root, float(xp.finfo(root.dtype).max)))
    finite_square = root <= maximum_root
    safe_root = xp.where(finite_square, root, xp.zeros_like(root))
    result = xp.where(
        finite_square,
        safe_root * safe_root,
        xp.full_like(root, float("inf")),
    )
    result = xp.where(
        xp.isnan(root), xp.full_like(result, float("nan")), result
    )
    if reduction == "none":
        return result
    return _squeeze_reduction(xp, result, axis, keepdims)


def _stable_difference_norm_ratio(
    xp: asc_typing.ArrayNamespace,
    numerator_first: object,
    numerator_second: object,
    denominator_first: object,
    denominator_second: object,
    axis: Axis,
    *,
    keepdims: bool,
) -> tuple[object, object, object]:
    """Return a scale-safe ratio and zero flags for two differences."""
    numerator_outer, numerator_inner, numerator_norm = (
        _scaled_difference_sum_squares(
            xp, numerator_first, numerator_second, axis
        )
    )
    denominator_outer, denominator_inner, denominator_norm = (
        _scaled_difference_sum_squares(
            xp, denominator_first, denominator_second, axis
        )
    )
    has_nan = (
        xp.isnan(numerator_outer)
        | xp.isnan(numerator_inner)
        | xp.isnan(numerator_norm)
        | xp.isnan(denominator_outer)
        | xp.isnan(denominator_inner)
        | xp.isnan(denominator_norm)
    )
    numerator_is_zero = numerator_norm == xp.zeros_like(numerator_norm)
    denominator_is_zero = denominator_norm == xp.zeros_like(denominator_norm)
    safe_numerator_outer = xp.where(
        numerator_is_zero,
        xp.ones_like(numerator_outer),
        numerator_outer,
    )
    safe_numerator_inner = xp.where(
        numerator_is_zero,
        xp.ones_like(numerator_inner),
        numerator_inner,
    )
    safe_numerator_norm = xp.where(
        numerator_is_zero,
        xp.ones_like(numerator_norm),
        numerator_norm,
    )
    safe_denominator_outer = xp.where(
        denominator_is_zero,
        xp.ones_like(denominator_outer),
        denominator_outer,
    )
    safe_denominator_inner = xp.where(
        denominator_is_zero,
        xp.ones_like(denominator_inner),
        denominator_inner,
    )
    safe_denominator_norm = xp.where(
        denominator_is_zero,
        xp.ones_like(denominator_norm),
        denominator_norm,
    )
    mantissa = xp.ones_like(numerator_outer)
    exponent = None
    for value in (
        safe_numerator_outer,
        safe_numerator_inner,
        safe_numerator_norm,
    ):
        value_mantissa, value_exponent = _standard_frexp(xp, value)
        mantissa = mantissa * value_mantissa
        exponent = (
            value_exponent if exponent is None else exponent + value_exponent
        )
    for value in (
        safe_denominator_outer,
        safe_denominator_inner,
        safe_denominator_norm,
    ):
        value_mantissa, value_exponent = _standard_frexp(xp, value)
        mantissa = mantissa / value_mantissa
        exponent = exponent - value_exponent
    mantissa, mantissa_exponent = _standard_frexp(xp, mantissa)
    exponent = exponent + mantissa_exponent
    maximum = xp.full_like(mantissa, float(xp.finfo(mantissa.dtype).max))
    maximum_mantissa, maximum_exponent = _standard_frexp(xp, maximum)
    overflows = (exponent > maximum_exponent) | (
        (exponent == maximum_exponent) & (mantissa > maximum_mantissa)
    )
    safe_mantissa = xp.where(overflows, xp.ones_like(mantissa), mantissa)
    safe_exponent = xp.where(overflows, xp.zeros_like(exponent), exponent)
    ratio = safe_mantissa * _power_of_two(xp, safe_mantissa, safe_exponent)
    infinity = xp.full_like(ratio, float("inf"))
    ratio = xp.where(overflows, infinity, ratio)
    ratio = xp.where(
        numerator_is_zero,
        xp.zeros_like(ratio),
        xp.where(
            denominator_is_zero,
            infinity,
            ratio,
        ),
    )
    ratio = xp.where(has_nan, xp.full_like(ratio, float("nan")), ratio)
    return (
        _squeeze_reduction(xp, ratio, axis, keepdims),
        _squeeze_reduction(xp, numerator_is_zero, axis, keepdims),
        _squeeze_reduction(xp, denominator_is_zero, axis, keepdims),
    )


def _squeeze_reduction(
    xp: asc_typing.ArrayNamespace,
    value: object,
    axis: Axis,
    keepdims: bool,
) -> object:
    """Remove dimensions introduced by a stable keepdims reduction."""
    if keepdims or axis == ():
        return value
    squeeze_axis: tuple[int, ...] | int
    squeeze_axis = tuple(range(len(value.shape))) if axis is None else axis
    return xp.squeeze(value, axis=squeeze_axis)


def mean_absolute_error(
    prediction: object,
    target: object,
    *,
    axis: Axis = None,
    reduction: Reduction = "mean",
    keepdims: bool = False,
) -> object:
    """Return absolute-error reduction."""
    keepdims = _validated_keepdims(keepdims, "mean_absolute_error")
    xp, prediction, target = _inputs(prediction, target, "mean_absolute_error")
    if reduction in {"mean", "sum"}:
        _require_nonempty_reduction(
            prediction.shape, axis, "mean_absolute_error"
        )
    return _stable_absolute_error(
        xp, prediction, target, axis, reduction, keepdims
    )


def mean_squared_error(
    prediction: object,
    target: object,
    *,
    axis: Axis = None,
    reduction: Reduction = "mean",
    keepdims: bool = False,
) -> object:
    """Return squared-error reduction."""
    keepdims = _validated_keepdims(keepdims, "mean_squared_error")
    xp, prediction, target = _inputs(prediction, target, "mean_squared_error")
    if reduction in {"mean", "sum"}:
        _require_nonempty_reduction(
            prediction.shape, axis, "mean_squared_error"
        )
    return _stable_squared_error(
        xp, prediction, target, axis, reduction, keepdims
    )


def root_mean_squared_error(
    prediction: object,
    target: object,
    *,
    axis: Axis = None,
    keepdims: bool = False,
) -> object:
    """Return the square root of mean squared error."""
    keepdims = _validated_keepdims(keepdims, "root_mean_squared_error")
    xp, prediction, target = _inputs(
        prediction, target, "root_mean_squared_error"
    )
    _require_nonempty_reduction(
        prediction.shape, axis, "root_mean_squared_error"
    )
    outer_scale, inner_scale, normalized_norm = _scaled_difference_sum_squares(
        xp, prediction, target, axis
    )
    result = _nonnegative_product(
        xp,
        outer_scale,
        _nonnegative_product(xp, inner_scale, normalized_norm),
    )
    return _squeeze_reduction(xp, result, axis, keepdims)


def relative_l2_error(
    prediction: object,
    target: object,
    *,
    axis: Axis = None,
    keepdims: bool = False,
) -> object:
    """Return L2 error divided by target L2 norm.

    A zero target norm returns zero for an exact prediction and infinity
    otherwise.
    """
    keepdims = _validated_keepdims(keepdims, "relative_l2_error")
    xp, prediction, target = _inputs(prediction, target, "relative_l2_error")
    _require_nonempty_reduction(prediction.shape, axis, "relative_l2_error")
    zeros = xp.zeros_like(target)
    ratio, _, _ = _stable_difference_norm_ratio(
        xp,
        prediction,
        target,
        target,
        zeros,
        axis,
        keepdims=keepdims,
    )
    return ratio


def r2_score(
    prediction: object,
    target: object,
    *,
    axis: Axis = None,
    keepdims: bool = False,
) -> object:
    """Return R2 with a finite constant-target policy."""
    keepdims = _validated_keepdims(keepdims, "r2_score")
    xp, prediction, target = _inputs(prediction, target, "r2_score")
    _require_nonempty_reduction(prediction.shape, axis, "r2_score")
    target_scale = xp.max(xp.abs(target), axis=axis, keepdims=True)
    safe_target_scale = xp.where(
        target_scale == xp.zeros_like(target_scale),
        xp.ones_like(target_scale),
        target_scale,
    )
    normalized_target = target / xp.broadcast_to(
        safe_target_scale, target.shape
    )
    mean = (
        xp.mean(normalized_target, axis=axis, keepdims=True) * safe_target_scale
    )
    norm_ratio, residual_is_zero, target_is_constant = (
        _stable_difference_norm_ratio(
            xp,
            prediction,
            target,
            target,
            mean,
            axis,
            keepdims=keepdims,
        )
    )
    maximum_ratio = xp.sqrt(
        xp.full_like(norm_ratio, float(xp.finfo(norm_ratio.dtype).max))
    )
    finite_square = xp.abs(norm_ratio) <= maximum_ratio
    safe_ratio = xp.where(finite_square, norm_ratio, xp.zeros_like(norm_ratio))
    nonconstant = xp.where(
        finite_square,
        1.0 - safe_ratio * safe_ratio,
        xp.full_like(norm_ratio, float("-inf")),
    )
    result = xp.where(
        target_is_constant,
        xp.where(
            residual_is_zero,
            xp.ones_like(norm_ratio),
            xp.zeros_like(norm_ratio),
        ),
        nonconstant,
    )
    return xp.where(
        xp.isnan(norm_ratio), xp.full_like(result, float("nan")), result
    )


__all__ = [
    "mean_absolute_error",
    "mean_squared_error",
    "r2_score",
    "relative_l2_error",
    "root_mean_squared_error",
]
