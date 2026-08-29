# Copyright 2026 AI4SciComp contributors.
# SPDX-License-Identifier: Apache-2.0
"""Portable numerical comparison helpers."""

from __future__ import annotations

import math

from asc import _array_api_compat, errors
from asc.core import namespace as namespace_module


def _tolerances(
    xp: object,
    dtype: object,
    rtol: float | None,
    atol: float | None,
) -> tuple[float, float]:
    try:
        epsilon = float(xp.finfo(dtype).eps)
    except (TypeError, ValueError) as exception:
        raise errors.DTypeError(
            "isclose: arrays must have real or complex floating dtypes"
        ) from exception
    relative = (
        max(1e-8, 8.0 * epsilon) if rtol is None else _python_tolerance(rtol)
    )
    absolute = (
        max(1e-12, 8.0 * epsilon) if atol is None else _python_tolerance(atol)
    )
    if (
        isinstance(relative, bool)
        or isinstance(absolute, bool)
        or not math.isfinite(relative)
        or not math.isfinite(absolute)
        or relative < 0
        or absolute < 0
    ):
        raise ValueError("isclose: tolerances must be finite and non-negative")
    return relative, absolute


def _python_tolerance(value: object) -> float:
    """Return one built-in real tolerance without array scalarization."""
    if (
        _array_api_compat.compat.is_array_api_obj(value)
        or isinstance(value, bool)
        or not isinstance(value, (int, float))
    ):
        raise ValueError("isclose: tolerances must be Python real scalars")
    try:
        return float(value)
    except OverflowError as exception:
        raise ValueError(
            "isclose: tolerances must be finite and non-negative"
        ) from exception


def _positive_log(xp: object, value: object) -> object:
    """Return logs while mapping exact zeros to negative infinity."""
    zero = xp.zeros_like(value)
    positive = value > zero
    safe = xp.where(positive, value, xp.ones_like(value))
    return xp.where(
        positive,
        xp.log(safe),
        xp.full_like(value, float("-inf")),
    )


def _complex_log_magnitude(
    xp: object, real: object, imaginary: object
) -> object:
    """Return a complex magnitude log without square or sum overflow."""
    real_magnitude = xp.abs(real)
    imaginary_magnitude = xp.abs(imaginary)
    scale = xp.maximum(real_magnitude, imaginary_magnitude)
    safe_scale = xp.where(scale > 0, scale, xp.ones_like(scale))
    normalized = xp.hypot(
        real_magnitude / safe_scale,
        imaginary_magnitude / safe_scale,
    )
    return _positive_log(xp, scale) + _positive_log(xp, normalized)


def _log_tolerance(
    xp: object,
    difference: object,
    second: object,
    relative: float,
    absolute: float,
    *,
    second_log_magnitude: object | None = None,
) -> object:
    """Compute log(atol + rtol * abs(second)) without dtype overflow."""
    negative_infinity = xp.full_like(difference, float("-inf"))
    absolute_log = (
        xp.full_like(difference, math.log(absolute))
        if absolute > 0.0
        else negative_infinity
    )
    second_log = second_log_magnitude
    if second_log is None:
        second_log = _positive_log(xp, xp.abs(second))
    relative_log = (
        second_log + math.log(relative) if relative > 0.0 else negative_infinity
    )
    maximum = xp.maximum(absolute_log, relative_log)
    present = xp.isfinite(maximum)
    safe_maximum = xp.where(present, maximum, xp.zeros_like(maximum))
    terms = xp.exp(absolute_log - safe_maximum) + xp.exp(
        relative_log - safe_maximum
    )
    safe_terms = xp.where(terms > 0, terms, xp.ones_like(terms))
    return xp.where(
        present,
        safe_maximum + xp.log(safe_terms),
        negative_infinity,
    )


def _nonzero_magnitude(xp: object, value: object) -> object:
    """Detect stored nonzero values even when device arithmetic flushes them."""
    toward_negative = xp.nextafter(value, xp.full_like(value, float("-inf")))
    return xp.logical_not(xp.signbit(toward_negative))


def _positive_sum_leq(
    xp: object,
    left_first: object,
    left_second: object,
    right_first: object,
    right_second: object,
) -> object:
    """Compare two non-negative two-term sums without sum overflow."""
    zero = xp.zeros_like(left_first)
    maximum_value = xp.full_like(
        left_first, float(xp.finfo(left_first.dtype).max)
    )
    left_overflows = left_first > maximum_value - left_second
    right_overflows = right_first > maximum_value - right_second
    scale = xp.maximum(
        xp.maximum(left_first, left_second),
        xp.maximum(right_first, right_second),
    )
    safe_scale = xp.where(scale > 0, scale, xp.ones_like(scale))

    def expansion(
        first: object, second: object, divisor: object
    ) -> tuple[object, object, object]:
        normalized_first = first / divisor
        normalized_second = second / divisor
        high_term = xp.maximum(normalized_first, normalized_second)
        low_term = xp.minimum(normalized_first, normalized_second)
        total = high_term + low_term
        virtual = total - high_term
        error = (high_term - (total - virtual)) + (low_term - virtual)
        lost = xp.logical_or(
            xp.logical_and(
                normalized_first == 0,
                _nonzero_magnitude(xp, first),
            ),
            xp.logical_and(
                normalized_second == 0,
                _nonzero_magnitude(xp, second),
            ),
        )
        return total, error, lost

    def compare_expansions(
        first_pair: tuple[object, object, object],
        second_pair: tuple[object, object, object],
    ) -> object:
        left_high, left_low, left_lost = first_pair
        right_high, right_low, right_lost = second_pair
        high_less = left_high < right_high
        high_equal = left_high == right_high
        low_less_equal = left_low <= right_low
        result = xp.logical_or(
            high_less, xp.logical_and(high_equal, low_less_equal)
        )
        tied = xp.logical_and(high_equal, left_low == right_low)
        return xp.where(
            xp.logical_and(tied, left_lost != right_lost),
            right_lost,
            result,
        )

    # Use error-free addition directly whenever both sums fit.  Dividing by
    # an arbitrary scale can perturb an exact equality at one-ULP boundaries.
    safe_left_first = xp.where(left_overflows, zero, left_first)
    safe_left_second = xp.where(left_overflows, zero, left_second)
    safe_right_first = xp.where(right_overflows, zero, right_first)
    safe_right_second = xp.where(right_overflows, zero, right_second)
    one = xp.ones_like(scale)
    direct_result = compare_expansions(
        expansion(safe_left_first, safe_left_second, one),
        expansion(safe_right_first, safe_right_second, one),
    )
    scaled_result = compare_expansions(
        expansion(left_first, left_second, safe_scale),
        expansion(right_first, right_second, safe_scale),
    )
    return xp.where(
        left_overflows,
        xp.logical_and(right_overflows, scaled_result),
        xp.logical_or(right_overflows, direct_result),
    )


def _real_finite_close(
    xp: object,
    first: object,
    second: object,
    relative: float,
    absolute: float,
    fallback: object,
) -> object:
    """Compare finite reals using cancellation-free tolerance identities."""
    maximum = float(xp.finfo(first.dtype).max)
    if relative > 1.0 or absolute > maximum:
        return fallback
    zero = xp.zeros_like(first)
    absolute_value = xp.full_like(first, absolute)
    first_magnitude = xp.abs(first)
    second_magnitude = xp.abs(second)
    same_sign = xp.signbit(first) == xp.signbit(second)

    # Opposite signs turn the difference into x + y.  Moving the relative
    # term to the left avoids losing x when rtol*y rounds back to y.
    if relative == 1.0:
        opposite_close = (
            xp.logical_not(_nonzero_magnitude(xp, first_magnitude))
            if absolute == 0.0
            else _positive_sum_leq(
                xp,
                first_magnitude,
                zero,
                absolute_value,
                zero,
            )
        )
    else:
        opposite_close = _positive_sum_leq(
            xp,
            first_magnitude,
            (1.0 - relative) * second_magnitude,
            absolute_value,
            zero,
        )

    # Equal signs turn the difference into |x-y|.  Both branches below move
    # that subtraction across the inequality, leaving only positive sums.
    first_is_larger = first_magnitude >= second_magnitude
    coefficient = 1.0 + relative
    product_overflows = second_magnitude > maximum / coefficient
    safe_second = xp.where(product_overflows, zero, second_magnitude)
    larger_first_close = xp.logical_or(
        product_overflows,
        _positive_sum_leq(
            xp,
            first_magnitude,
            zero,
            absolute_value,
            coefficient * safe_second,
        ),
    )
    if relative == 1.0:
        larger_second_close = xp.ones_like(same_sign)
    else:
        larger_second_close = _positive_sum_leq(
            xp,
            (1.0 - relative) * second_magnitude,
            zero,
            absolute_value,
            first_magnitude,
        )
    same_sign_close = xp.where(
        first_is_larger,
        larger_first_close,
        larger_second_close,
    )
    return xp.where(same_sign, same_sign_close, opposite_close)


def isclose(
    first: object,
    second: object,
    *,
    rtol: float | None = None,
    atol: float | None = None,
    equal_nan: bool = False,
) -> object:
    """Return elementwise closeness under explicit tolerances."""
    xp = namespace_module.array_namespace(first, second)
    if not isinstance(equal_nan, bool):
        raise ValueError("isclose: equal_nan must be Boolean")
    try:
        dtype = xp.result_type(first, second)
        first = xp.astype(first, dtype, copy=True)
        second = xp.astype(second, dtype, copy=True)
    except (RuntimeError, TypeError, ValueError) as exception:
        raise errors.DTypeError(
            "isclose: input dtypes must promote under Array API rules"
        ) from exception
    relative, absolute = _tolerances(xp, dtype, rtol, atol)
    exact = first == second
    finite = xp.logical_and(xp.isfinite(first), xp.isfinite(second))
    safe_first = xp.where(finite, first, xp.zeros_like(first))
    safe_second = xp.where(finite, second, xp.zeros_like(second))
    working_first = safe_first
    working_second = safe_second
    if xp.isdtype(dtype, "real floating") and int(xp.finfo(dtype).bits) < 32:
        working_first = xp.astype(safe_first, xp.float32, copy=True)
        working_second = xp.astype(safe_second, xp.float32, copy=True)
    is_complex = xp.isdtype(dtype, "complex floating")
    if is_complex:
        first_real = xp.real(working_first)
        first_imaginary = xp.imag(working_first)
        second_real = xp.real(working_second)
        second_imaginary = xp.imag(working_second)
        component_scale = xp.maximum(
            xp.maximum(xp.abs(first_real), xp.abs(first_imaginary)),
            xp.maximum(xp.abs(second_real), xp.abs(second_imaginary)),
        )
        safe_component_scale = xp.where(
            component_scale > 0,
            component_scale,
            xp.ones_like(component_scale),
        )
        normalized_log_difference = _positive_log(xp, component_scale) + (
            _complex_log_magnitude(
                xp,
                first_real / safe_component_scale
                - second_real / safe_component_scale,
                first_imaginary / safe_component_scale
                - second_imaginary / safe_component_scale,
            )
        )
        maximum = xp.full_like(
            component_scale, float(xp.finfo(component_scale.dtype).max)
        )

        def subtraction_overflows(
            first_part: object, second_part: object
        ) -> object:
            opposite_signs = xp.signbit(first_part) != xp.signbit(second_part)
            return xp.logical_and(
                opposite_signs,
                xp.abs(first_part) > maximum - xp.abs(second_part),
            )

        unsafe_subtraction = xp.logical_or(
            subtraction_overflows(first_real, second_real),
            subtraction_overflows(first_imaginary, second_imaginary),
        )
        zero = xp.zeros_like(first_real)
        direct_log_difference = _complex_log_magnitude(
            xp,
            xp.where(unsafe_subtraction, zero, first_real)
            - xp.where(unsafe_subtraction, zero, second_real),
            xp.where(unsafe_subtraction, zero, first_imaginary)
            - xp.where(unsafe_subtraction, zero, second_imaginary),
        )
        log_difference = xp.where(
            unsafe_subtraction,
            normalized_log_difference,
            direct_log_difference,
        )
        second_log_magnitude = _complex_log_magnitude(
            xp, second_real, second_imaginary
        )
        log_tolerance = _log_tolerance(
            xp,
            component_scale,
            working_second,
            relative,
            absolute,
            second_log_magnitude=second_log_magnitude,
        )
    else:
        scale = xp.maximum(xp.abs(working_first), xp.abs(working_second))
        safe_scale = xp.where(scale > 0, scale, xp.ones_like(scale))
        normalized_difference = xp.abs(
            working_first / safe_scale - working_second / safe_scale
        )
        log_difference = _positive_log(xp, scale) + _positive_log(
            xp, normalized_difference
        )
        log_tolerance = _log_tolerance(
            xp, scale, working_second, relative, absolute
        )
    finite_close = log_difference <= log_tolerance
    if xp.isdtype(dtype, "real floating"):
        finite_close = _real_finite_close(
            xp,
            working_first,
            working_second,
            relative,
            absolute,
            finite_close,
        )
    result = xp.logical_or(
        exact,
        xp.logical_and(finite, finite_close),
    )
    if equal_nan:
        result = xp.logical_or(
            result,
            xp.logical_and(xp.isnan(first), xp.isnan(second)),
        )
    return result


def allclose(
    first: object,
    second: object,
    *,
    rtol: float | None = None,
    atol: float | None = None,
    equal_nan: bool = False,
) -> object:
    """Return a native 0-D Boolean array for whole-array closeness."""
    xp = namespace_module.array_namespace(first, second)
    return xp.all(
        isclose(first, second, rtol=rtol, atol=atol, equal_nan=equal_nan)
    )


def assert_allclose(
    first: object,
    second: object,
    *,
    rtol: float | None = None,
    atol: float | None = None,
    equal_nan: bool = False,
) -> None:
    """Raise ``AssertionError`` unless every pair of elements is close."""
    if not allclose(first, second, rtol=rtol, atol=atol, equal_nan=equal_nan):
        raise AssertionError(
            f"arrays are not close within rtol={rtol!r}, atol={atol!r}"
        )


__all__ = ["allclose", "assert_allclose", "isclose"]
