# Copyright 2026 AI4SciComp contributors.
# SPDX-License-Identifier: Apache-2.0
"""Property checks for portable numerical and data invariants."""

from __future__ import annotations

import numpy
from hypothesis import given, settings
from hypothesis import strategies as st

import asc
from asc import data
from tests import helpers

FINITE = st.floats(
    min_value=-100,
    max_value=100,
    allow_nan=False,
    allow_infinity=False,
    allow_subnormal=False,
    width=32,
)


@settings(max_examples=15, deadline=None)
@given(st.lists(FINITE, min_size=1, max_size=8))
def test_sum_square_and_padding_parity(values: list[float]) -> None:
    expected = numpy.sum(
        numpy.square(numpy.asarray(values, dtype=numpy.float32))
    )
    for backend in helpers.NATIVE_BACKENDS:
        xp = asc.backend(backend).xp
        array = xp.asarray(values, dtype=xp.float32)
        result = asc.sum_of_squares(array)
        numpy.testing.assert_allclose(
            numpy.asarray(result), expected, rtol=1e-5
        )
        padded = asc.ops.pad(array, 2, mode="wrap")
        numpy.testing.assert_allclose(
            numpy.asarray(padded), numpy.pad(values, 2, mode="wrap"), rtol=1e-5
        )


@settings(max_examples=12, deadline=None)
@given(
    st.integers(min_value=1, max_value=8),
    st.lists(st.integers(min_value=0, max_value=20), min_size=1, max_size=4),
)
def test_split_lengths_are_total_and_reproducible(
    length: int, raw_sizes: list[int]
) -> None:
    sizes = [0] * len(raw_sizes)
    for index in range(length):
        sizes[index % len(sizes)] += 1
    dataset = data.ArrayDataset(numpy.arange(length))
    first = data.split_dataset(dataset, sizes)
    second = data.split_dataset(dataset, sizes)
    assert [subset.indices for subset in first] == [
        subset.indices for subset in second
    ]
    assert sum(map(len, first)) == length


@settings(max_examples=15, deadline=None)
@given(st.lists(FINITE, min_size=1, max_size=8))
def test_index_add_matches_reference(values: list[float]) -> None:
    reference = numpy.zeros((len(values),), dtype=numpy.float32)
    indices = numpy.arange(len(values), dtype=numpy.int16)
    numpy.add.at(reference, indices, numpy.asarray(values, dtype=numpy.float32))
    for backend in helpers.NATIVE_BACKENDS:
        xp = asc.backend(backend).xp
        result = asc.index_add(
            xp.zeros((len(values),), dtype=xp.float32),
            xp.asarray(indices.tolist(), dtype=xp.int16),
            xp.asarray(values, dtype=xp.float32),
        )
        numpy.testing.assert_allclose(
            numpy.asarray(result), reference, rtol=1e-5
        )
