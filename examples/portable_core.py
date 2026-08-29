# Copyright 2026 AI4SciComp contributors.
# SPDX-License-Identifier: Apache-2.0
"""Use asc-py with a native NumPy array and explicit creation context."""

import array_api_compat.numpy as numpy_namespace

import asc

values = numpy_namespace.asarray(
    [[1.0, 2.0], [3.0, 4.0]],
    dtype=numpy_namespace.float32,
)
reduction = asc.sum_of_squares(values, axis=1)

context = asc.CreationContext(
    numpy_namespace,
    "numpy",
    dtype=numpy_namespace.float32,
)
created = asc.create_full((2, 2), 3.0, context=context)

print(reduction)
print(created)
