# Copyright 2026 AI4SciComp contributors.
# SPDX-License-Identifier: Apache-2.0

import array_api_strict
import numpy

import asc


def test_strict_revision_and_zero_dimensional_reduction() -> None:
    array_api_strict.set_array_api_strict_flags(api_version="2024.12")
    value = array_api_strict.asarray(
        [3.0, 4.0],
        dtype=array_api_strict.float32,
    )

    result = asc.sum_of_squares(value)

    assert result.shape == ()
    assert result.dtype == array_api_strict.float32
    numpy.testing.assert_allclose(numpy.asarray(result), 25.0)
