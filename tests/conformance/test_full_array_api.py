# Copyright 2026 AI4SciComp contributors.
# SPDX-License-Identifier: Apache-2.0
"""Conformance inventory for the frozen Array API 2024.12 surface."""

from __future__ import annotations

import numpy
import pytest

import asc
from asc import typing as asc_typing
from asc.core._array_api import MANDATORY_SYMBOLS
from tests import helpers

CORE_SYMBOLS = {
    "e",
    "inf",
    "nan",
    "newaxis",
    "pi",
    "__array_api_version__",
    "__array_namespace_info__",
    "bool",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "float32",
    "float64",
    "complex64",
    "complex128",
    "arange",
    "asarray",
    "empty",
    "empty_like",
    "eye",
    "from_dlpack",
    "full",
    "full_like",
    "linspace",
    "meshgrid",
    "ones",
    "ones_like",
    "tril",
    "triu",
    "zeros",
    "zeros_like",
    "astype",
    "can_cast",
    "finfo",
    "iinfo",
    "isdtype",
    "result_type",
    "abs",
    "add",
    "divide",
    "floor_divide",
    "multiply",
    "negative",
    "positive",
    "pow",
    "reciprocal",
    "remainder",
    "square",
    "subtract",
    "exp",
    "expm1",
    "log",
    "log1p",
    "log2",
    "log10",
    "logaddexp",
    "sqrt",
    "acos",
    "asin",
    "atan",
    "atan2",
    "cos",
    "sin",
    "tan",
    "acosh",
    "asinh",
    "atanh",
    "cosh",
    "sinh",
    "tanh",
    "ceil",
    "clip",
    "floor",
    "round",
    "trunc",
    "equal",
    "greater",
    "greater_equal",
    "less",
    "less_equal",
    "logical_and",
    "logical_not",
    "logical_or",
    "logical_xor",
    "not_equal",
    "bitwise_and",
    "bitwise_invert",
    "bitwise_left_shift",
    "bitwise_or",
    "bitwise_right_shift",
    "bitwise_xor",
    "conj",
    "copysign",
    "hypot",
    "imag",
    "isfinite",
    "isinf",
    "isnan",
    "maximum",
    "minimum",
    "nextafter",
    "real",
    "sign",
    "signbit",
    "take",
    "take_along_axis",
    "broadcast_arrays",
    "broadcast_to",
    "concat",
    "expand_dims",
    "flip",
    "moveaxis",
    "permute_dims",
    "repeat",
    "reshape",
    "roll",
    "squeeze",
    "stack",
    "tile",
    "unstack",
    "argmax",
    "argmin",
    "count_nonzero",
    "nonzero",
    "searchsorted",
    "where",
    "unique_all",
    "unique_counts",
    "unique_inverse",
    "unique_values",
    "argsort",
    "sort",
    "cumulative_prod",
    "cumulative_sum",
    "max",
    "mean",
    "min",
    "prod",
    "std",
    "sum",
    "var",
    "all",
    "any",
    "diff",
    "matmul",
    "matrix_transpose",
    "tensordot",
    "vecdot",
}

assert CORE_SYMBOLS == MANDATORY_SYMBOLS


@pytest.mark.parametrize("backend", helpers.BACKENDS)
def test_frozen_symbol_inventory(backend: asc_typing.BackendName) -> None:
    namespace = helpers.namespace(backend)
    missing = sorted(
        name for name in CORE_SYMBOLS if not hasattr(namespace, name)
    )
    assert missing == []
    assert namespace.__array_api_version__ == "2024.12"
    info = asc.namespace_info(namespace)
    assert info.array_api_version == "2024.12"
    assert info.backend == backend


@pytest.mark.parametrize("backend", helpers.BACKENDS)
def test_creation_manipulation_and_reductions(
    backend: asc_typing.BackendName,
) -> None:
    xp = helpers.namespace(backend)
    value = xp.reshape(xp.arange(6, dtype=xp.float32), (2, 3))
    assert value.shape == (2, 3)
    numpy.testing.assert_allclose(
        helpers.as_numpy(xp.sum(value, axis=1)), [3, 12]
    )
    numpy.testing.assert_allclose(
        helpers.as_numpy(xp.concat((value, value), axis=0)),
        [[0, 1, 2], [3, 4, 5], [0, 1, 2], [3, 4, 5]],
    )
    indices = xp.asarray([[2, 1, 0], [0, 1, 2]], dtype=xp.int32)
    numpy.testing.assert_allclose(
        helpers.as_numpy(xp.take_along_axis(value, indices, axis=1)),
        [[2, 1, 0], [3, 4, 5]],
    )
    assert xp.reshape(xp.asarray(2.0, dtype=xp.float32), ()).shape == ()
    assert xp.empty((0,), dtype=xp.float32).shape == (0,)


@pytest.mark.parametrize("backend", helpers.BACKENDS)
def test_elementwise_sort_set_and_operators(
    backend: asc_typing.BackendName,
) -> None:
    xp = helpers.namespace(backend)
    value = xp.asarray([-0.0, 1.0, 2.0], dtype=xp.float32)
    numpy.testing.assert_allclose(helpers.as_numpy(xp.square(value)), [0, 1, 4])
    assert helpers.as_numpy(xp.signbit(value)).tolist() == [1.0, 0.0, 0.0]
    integers = xp.asarray([3, 1, 3, 2], dtype=xp.int32)
    numpy.testing.assert_allclose(
        helpers.as_numpy(xp.sort(integers)), [1, 2, 3, 3]
    )
    numpy.testing.assert_allclose(
        helpers.as_numpy(xp.unique_values(integers)), [1, 2, 3]
    )
    numpy.testing.assert_allclose(
        helpers.as_numpy((integers + 1) * 2), [8, 4, 8, 6]
    )
    matrix = xp.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=xp.float32)
    numpy.testing.assert_allclose(
        helpers.as_numpy(matrix @ xp.matrix_transpose(matrix)),
        [[5, 11], [11, 25]],
    )


@pytest.mark.parametrize("backend", helpers.BACKENDS)
def test_complex_nan_inf_and_noncontiguous_edges(
    backend: asc_typing.BackendName,
) -> None:
    xp = helpers.namespace(backend)
    complex_value = xp.asarray([1 + 2j, 3 - 4j], dtype=xp.complex64)
    numpy.testing.assert_allclose(
        numpy.asarray(xp.conj(complex_value)), [1 - 2j, 3 + 4j]
    )
    special = xp.asarray([xp.nan, xp.inf, -xp.inf], dtype=xp.float32)
    assert helpers.as_numpy(xp.isnan(special)).tolist() == [1.0, 0.0, 0.0]
    base = xp.reshape(xp.arange(12, dtype=xp.float32), (3, 4))
    sliced = base[:, ::2]
    numpy.testing.assert_allclose(
        helpers.as_numpy(xp.sum(sliced, axis=0)), [12, 18]
    )
