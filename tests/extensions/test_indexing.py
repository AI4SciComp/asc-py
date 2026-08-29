# Copyright 2026 AI4SciComp contributors.
# SPDX-License-Identifier: Apache-2.0

import numpy
import pytest

import asc
from asc import typing as asc_typing
from asc.extensions import indexing
from tests import helpers

INDEX_BACKENDS: tuple[asc_typing.BackendName, ...] = (
    "jax",
    "numpy",
    "torch",
)


@pytest.mark.parametrize("backend", INDEX_BACKENDS)
def test_index_add_accumulates_duplicates_without_mutation(
    backend: asc_typing.BackendName,
) -> None:
    array = helpers.float_array(backend, [10.0, 20.0, 30.0])
    indices = helpers.int_array(backend, [1, 1])
    values = helpers.float_array(backend, [2.0, 3.0])
    before = helpers.as_numpy(array).copy()

    result = indexing.index_add(array, indices, values)

    numpy.testing.assert_array_equal(
        helpers.as_numpy(result), [10.0, 25.0, 30.0]
    )
    numpy.testing.assert_array_equal(helpers.as_numpy(array), before)
    assert result is not array


@pytest.mark.parametrize("backend", INDEX_BACKENDS)
def test_index_add_validates_bounds_and_index_dtype(
    backend: asc_typing.BackendName,
) -> None:
    array = helpers.float_array(backend, [0.0, 0.0])
    values = helpers.float_array(backend, [1.0])
    with pytest.raises(asc.IndexContractError, match="out of bounds"):
        indexing.index_add(array, helpers.int_array(backend, [2]), values)
    with pytest.raises(asc.IndexContractError, match="integer dtype"):
        indexing.index_add(
            array,
            helpers.float_array(backend, [0.0]),
            values,
        )


def test_index_add_rejects_strict_backend() -> None:
    strict_array = helpers.float_array("array_api_strict", [0.0])
    strict_indices = helpers.int_array("array_api_strict", [0])
    strict_values = helpers.float_array("array_api_strict", [1.0])
    with pytest.raises(asc.UnsupportedCapabilityError):
        indexing.index_add(strict_array, strict_indices, strict_values)


@pytest.mark.backend("torch")
def test_index_add_rejects_mixed_backends() -> None:
    with pytest.raises(asc.MixedBackendError):
        indexing.index_add(
            helpers.float_array("numpy", [0.0]),
            helpers.int_array("torch", [0]),
            helpers.float_array("numpy", [1.0]),
        )


def test_index_add_validates_axis_rank_and_index_rank() -> None:
    array = helpers.float_array("numpy", [[0.0, 0.0]])
    indices = helpers.int_array("numpy", [0])
    values = helpers.float_array("numpy", [[1.0, 1.0]])
    with pytest.raises(asc.IndexContractError, match="axis must"):
        indexing.index_add(array, indices, values, axis=True)
    with pytest.raises(asc.IndexContractError, match="axis is out"):
        indexing.index_add(array, indices, values, axis=2)
    with pytest.raises(asc.IndexContractError, match="one-dimensional"):
        indexing.index_add(
            array,
            helpers.int_array("numpy", [[0]]),
            values,
        )
    with pytest.raises(asc.IndexContractError, match="rank at least one"):
        indexing.index_add(
            helpers.float_array("numpy", 0.0),
            indices,
            helpers.float_array("numpy", [1.0]),
        )


@pytest.mark.parametrize("backend", INDEX_BACKENDS)
def test_index_add_moves_values_with_nonzero_axis(
    backend: asc_typing.BackendName,
) -> None:
    result = indexing.index_add(
        helpers.float_array(backend, [[0.0, 0.0, 0.0]] * 2),
        helpers.int_array(backend, [0, 2]),
        helpers.float_array(backend, [[1.0, 2.0], [3.0, 4.0]]),
        axis=1,
    )
    numpy.testing.assert_array_equal(
        helpers.as_numpy(result),
        [[1.0, 0.0, 2.0], [3.0, 0.0, 4.0]],
    )


@pytest.mark.parametrize("backend", INDEX_BACKENDS)
def test_index_add_broadcasts_values_to_indexed_slices(
    backend: asc_typing.BackendName,
) -> None:
    result = indexing.index_add(
        helpers.float_array(backend, [[0.0, 0.0, 0.0]] * 2),
        helpers.int_array(backend, [0]),
        helpers.float_array(backend, [1.0, 2.0, 3.0]),
    )
    numpy.testing.assert_array_equal(
        helpers.as_numpy(result),
        [[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]],
    )


@pytest.mark.parametrize("backend", INDEX_BACKENDS)
@pytest.mark.parametrize("dtype_name", ["int8", "int16"])
def test_index_add_accepts_every_signed_index_width(
    backend: asc_typing.BackendName,
    dtype_name: str,
) -> None:
    selected = helpers.namespace(backend)
    dtype = getattr(selected, dtype_name)
    indices = selected.asarray([1], dtype=dtype)
    result = indexing.index_add(
        helpers.float_array(backend, [0.0, 0.0]),
        indices,
        helpers.float_array(backend, [2.0]),
    )
    numpy.testing.assert_array_equal(helpers.as_numpy(result), [0.0, 2.0])


@pytest.mark.parametrize("backend", INDEX_BACKENDS)
def test_index_add_rejects_unsigned_indices(
    backend: asc_typing.BackendName,
) -> None:
    selected = helpers.namespace(backend)
    indices = selected.asarray([0], dtype=selected.uint8)
    with pytest.raises(asc.IndexContractError, match="signed integer"):
        indexing.index_add(
            helpers.float_array(backend, [0.0]),
            indices,
            helpers.float_array(backend, [1.0]),
        )
