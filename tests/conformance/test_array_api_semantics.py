# Copyright 2026 AI4SciComp contributors.
# SPDX-License-Identifier: Apache-2.0
"""Executable semantic checks for every frozen Array API function family."""

from __future__ import annotations

import math

import numpy
import pytest

from asc import typing as asc_typing
from tests import helpers


def _array(value: object) -> numpy.ndarray:
    """Return a native CPU array for assertions without dtype coercion."""
    return numpy.asarray(value)


@pytest.mark.parametrize("backend", helpers.BACKENDS)
def test_constants_dtypes_and_creation(backend: asc_typing.BackendName) -> None:
    xp = helpers.namespace(backend)
    assert math.isclose(float(xp.e), math.e)
    assert math.isclose(float(xp.pi), math.pi)
    assert math.isinf(float(xp.inf)) and math.isnan(float(xp.nan))
    assert xp.newaxis is None
    assert xp.isdtype(xp.float32, "real floating")
    assert xp.isdtype(xp.int16, "signed integer")
    assert xp.can_cast(xp.int16, xp.int32)
    assert xp.result_type(xp.asarray(1, dtype=xp.int16), xp.int32) == xp.int32
    assert xp.finfo(xp.float32).eps > 0
    assert xp.iinfo(xp.int16).max == 32767
    base = xp.asarray([[1, 2], [3, 4]], dtype=xp.float32, copy=True)
    assert xp.astype(base, xp.float32, copy=True).shape == (2, 2)
    creations = (
        xp.empty((0, 2), dtype=xp.float32),
        xp.empty_like(base),
        xp.eye(2, dtype=xp.float32),
        xp.full((2,), 3, dtype=xp.float32),
        xp.full_like(base, 2),
        xp.linspace(0, 1, 3, dtype=xp.float32),
        xp.ones((2,), dtype=xp.float32),
        xp.ones_like(base),
        xp.zeros((2,), dtype=xp.float32),
        xp.zeros_like(base),
    )
    assert all(hasattr(value, "shape") for value in creations)
    grid = xp.meshgrid(
        xp.asarray([1, 2], dtype=xp.float32),
        xp.asarray([3, 4], dtype=xp.float32),
        indexing="ij",
    )
    assert tuple(value.shape for value in grid) == ((2, 2), (2, 2))
    numpy.testing.assert_array_equal(_array(xp.tril(base)), [[1, 0], [3, 4]])
    numpy.testing.assert_array_equal(_array(xp.triu(base)), [[1, 2], [0, 4]])


@pytest.mark.parametrize("backend", helpers.BACKENDS)
def test_arithmetic_exponential_and_rounding(
    backend: asc_typing.BackendName,
) -> None:
    xp = helpers.namespace(backend)
    left = xp.asarray([1.0, 2.0], dtype=xp.float32)
    right = xp.asarray([2.0, 4.0], dtype=xp.float32)
    operations = {
        "abs": xp.abs(-left),
        "add": xp.add(left, right),
        "divide": xp.divide(right, left),
        "floor_divide": xp.floor_divide(right, left),
        "multiply": xp.multiply(left, right),
        "negative": xp.negative(left),
        "positive": xp.positive(left),
        "pow": xp.pow(left, 2),
        "remainder": xp.remainder(right, 3),
        "square": xp.square(left),
        "subtract": xp.subtract(right, left),
    }
    assert tuple(operations) == (
        "abs",
        "add",
        "divide",
        "floor_divide",
        "multiply",
        "negative",
        "positive",
        "pow",
        "remainder",
        "square",
        "subtract",
    )
    positive = xp.asarray([0.5, 1.0], dtype=xp.float32)
    for value in (
        xp.exp(positive),
        xp.expm1(positive),
        xp.log(positive),
        xp.log1p(positive),
        xp.log2(positive),
        xp.log10(positive),
        xp.logaddexp(positive, positive),
        xp.sqrt(positive),
    ):
        assert bool(xp.all(xp.isfinite(value)))
    decimal = xp.asarray([-1.7, 1.2], dtype=xp.float32)
    for value in (
        xp.ceil(decimal),
        xp.clip(decimal, -1.0, 1.0),
        xp.floor(decimal),
        xp.round(decimal),
        xp.trunc(decimal),
    ):
        assert value.shape == (2,)


@pytest.mark.parametrize("backend", helpers.BACKENDS)
def test_trigonometric_hyperbolic_and_float_helpers(
    backend: asc_typing.BackendName,
) -> None:
    xp = helpers.namespace(backend)
    unit = xp.asarray([0.25, 0.5], dtype=xp.float32)
    for value in (
        xp.acos(unit),
        xp.asin(unit),
        xp.atan(unit),
        xp.atan2(unit, unit),
        xp.cos(unit),
        xp.sin(unit),
        xp.tan(unit),
        xp.acosh(unit + 1),
        xp.asinh(unit),
        xp.atanh(unit),
        xp.cosh(unit),
        xp.sinh(unit),
        xp.tanh(unit),
        xp.copysign(unit, -unit),
        xp.hypot(unit, unit),
        xp.maximum(unit, unit + 1),
        xp.minimum(unit, unit + 1),
        xp.nextafter(unit, unit + 1),
        xp.sign(unit),
    ):
        assert value.shape == (2,)
    complex_value = xp.asarray([1 + 2j], dtype=xp.complex64)
    numpy.testing.assert_allclose(_array(xp.real(complex_value)), [1])
    numpy.testing.assert_allclose(_array(xp.imag(complex_value)), [2])
    numpy.testing.assert_allclose(_array(xp.conj(complex_value)), [1 - 2j])
    special = xp.asarray([0.0, xp.inf, xp.nan], dtype=xp.float32)
    assert _array(xp.isfinite(special)).tolist() == [True, False, False]
    assert _array(xp.isinf(special)).tolist() == [False, True, False]
    assert _array(xp.isnan(special)).tolist() == [False, False, True]
    assert _array(xp.signbit(-unit)).tolist() == [True, True]


@pytest.mark.parametrize("backend", helpers.BACKENDS)
def test_logical_bitwise_search_sort_and_sets(
    backend: asc_typing.BackendName,
) -> None:
    xp = helpers.namespace(backend)
    left = xp.asarray([True, False], dtype=xp.bool)
    right = xp.asarray([True, True], dtype=xp.bool)
    assert _array(xp.logical_and(left, right)).tolist() == [True, False]
    assert _array(xp.logical_not(left)).tolist() == [False, True]
    assert _array(xp.logical_or(left, right)).tolist() == [True, True]
    assert _array(xp.logical_xor(left, right)).tolist() == [False, True]
    integers = xp.asarray([3, 1, 3, 2], dtype=xp.int32)
    for value in (
        xp.equal(integers, 3),
        xp.not_equal(integers, 3),
        xp.greater(integers, 1),
        xp.greater_equal(integers, 1),
        xp.less(integers, 3),
        xp.less_equal(integers, 3),
        xp.bitwise_and(integers, 1),
        xp.bitwise_or(integers, 1),
        xp.bitwise_xor(integers, 1),
        xp.bitwise_invert(integers),
        xp.bitwise_left_shift(integers, 1),
        xp.bitwise_right_shift(integers, 1),
    ):
        assert value.shape == (4,)
    assert int(xp.argmax(integers)) == 0
    assert int(xp.argmin(integers)) == 1
    assert int(xp.count_nonzero(integers == 3)) == 2
    assert xp.nonzero(integers == 3)[0].shape == (2,)
    query = xp.asarray(2, dtype=xp.int32)
    assert int(xp.searchsorted(xp.sort(integers), query)) == 1
    ones = xp.asarray([1, 1], dtype=xp.int32)
    zeros = xp.asarray([0, 0], dtype=xp.int32)
    numpy.testing.assert_array_equal(
        _array(xp.where(left, ones, zeros)), [1, 0]
    )
    numpy.testing.assert_array_equal(_array(xp.sort(integers)), [1, 2, 3, 3])
    numpy.testing.assert_array_equal(
        _array(xp.sort(integers, descending=True)), [3, 3, 2, 1]
    )
    assert xp.argsort(integers, stable=True).shape == (4,)
    assert xp.unique_values(integers).shape == (3,)
    assert xp.unique_counts(integers).counts.shape == (3,)
    assert xp.unique_inverse(integers).inverse_indices.shape == (4,)
    assert xp.unique_all(integers).values.shape == (3,)


@pytest.mark.parametrize("backend", helpers.BACKENDS)
def test_manipulation_reductions_and_linear_algebra(
    backend: asc_typing.BackendName,
) -> None:
    xp = helpers.namespace(backend)
    matrix = xp.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=xp.float32)
    vector = xp.asarray([1.0, 2.0], dtype=xp.float32)
    assert len(xp.broadcast_arrays(vector, matrix)) == 2
    assert xp.broadcast_to(vector, (2, 2)).shape == (2, 2)
    assert xp.concat((matrix, matrix), axis=0).shape == (4, 2)
    assert xp.expand_dims(vector, axis=0).shape == (1, 2)
    assert xp.flip(vector).shape == (2,)
    assert xp.moveaxis(xp.expand_dims(matrix, axis=0), 0, 2).shape == (2, 2, 1)
    assert xp.permute_dims(matrix, (1, 0)).shape == (2, 2)
    assert xp.repeat(vector, 2).shape == (4,)
    assert xp.reshape(matrix, (4,)).shape == (4,)
    assert xp.roll(vector, 1).shape == (2,)
    assert xp.squeeze(xp.expand_dims(vector, axis=0), axis=0).shape == (2,)
    assert xp.stack((vector, vector)).shape == (2, 2)
    assert xp.tile(vector, (2,)).shape == (4,)
    assert len(xp.unstack(matrix, axis=0)) == 2
    numpy.testing.assert_allclose(_array(xp.cumulative_prod(vector)), [1, 2])
    numpy.testing.assert_allclose(_array(xp.cumulative_sum(vector)), [1, 3])
    assert float(xp.max(vector)) == 2
    assert float(xp.mean(vector)) == pytest.approx(1.5)
    assert float(xp.min(vector)) == 1
    assert float(xp.prod(vector)) == 2
    assert float(xp.std(vector)) == pytest.approx(0.5)
    assert float(xp.sum(vector)) == 3
    assert float(xp.var(vector)) == pytest.approx(0.25)
    assert bool(xp.all(vector > 0)) and bool(xp.any(vector == 2))
    numpy.testing.assert_allclose(_array(xp.diff(vector)), [1])
    assert xp.matmul(matrix, vector).shape == (2,)
    assert xp.matrix_transpose(matrix).shape == (2, 2)
    assert xp.tensordot(matrix, vector, axes=1).shape == (2,)
    assert xp.vecdot(vector, vector).shape == ()


@pytest.mark.parametrize("backend", helpers.BACKENDS)
def test_indexing_and_native_array_metadata(
    backend: asc_typing.BackendName,
) -> None:
    xp = helpers.namespace(backend)
    array = xp.reshape(xp.arange(6, dtype=xp.float32), (2, 3))
    indices = xp.asarray([2, 0], dtype=xp.int16)
    numpy.testing.assert_array_equal(
        _array(xp.take(array, indices, axis=1)), [[2, 0], [5, 3]]
    )
    assert array[0, ...].shape == (3,)
    assert array[:, 1:].shape == (2, 2)
    assert array.shape == (2, 3)
    assert array.dtype == xp.float32
    assert array.ndim == 2
    observed_size = array.numel() if backend == "torch" else array.size
    assert observed_size == 6
    numpy.testing.assert_allclose(_array(array + 1), [[1, 2, 3], [4, 5, 6]])
    numpy.testing.assert_allclose(
        _array(array @ xp.ones((3, 1), dtype=xp.float32)), [[3], [12]]
    )
