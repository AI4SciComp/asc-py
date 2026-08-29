# Copyright 2026 AI4SciComp contributors.
# SPDX-License-Identifier: Apache-2.0

import hypothesis
import hypothesis.strategies as strategies
import numpy

import asc
from tests import helpers


@hypothesis.given(
    strategies.lists(
        strategies.floats(
            min_value=-100.0,
            max_value=100.0,
            allow_nan=False,
            allow_infinity=False,
            width=32,
        ),
        min_size=0,
        max_size=20,
    )
)
def test_numpy_sum_of_squares_is_nonnegative(values: list[float]) -> None:
    array = helpers.float_array("numpy", values)

    result = asc.sum_of_squares(array)

    host_result = helpers.as_numpy(result)
    assert host_result.shape == ()
    assert float(host_result) >= 0.0
    numpy.testing.assert_allclose(
        host_result,
        numpy.sum(numpy.square(numpy.asarray(values, dtype=numpy.float32))),
        rtol=2e-6,
        atol=2e-6,
    )
