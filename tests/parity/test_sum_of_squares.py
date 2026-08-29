# Copyright 2026 AI4SciComp contributors.
# SPDX-License-Identifier: Apache-2.0

import numpy

import asc
from tests import helpers


def test_sum_of_squares_backend_parity() -> None:
    results = tuple(
        helpers.as_numpy(
            asc.sum_of_squares(helpers.float_array(backend, [-3.0, 4.0, 0.5]))
        )
        for backend in helpers.BACKENDS
    )

    for result in results[1:]:
        numpy.testing.assert_allclose(result, results[0], rtol=1e-6, atol=1e-6)
