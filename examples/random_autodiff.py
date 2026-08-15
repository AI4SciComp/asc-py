# Copyright 2026 AI4SciComp contributors.
# SPDX-License-Identifier: Apache-2.0
"""Advance explicit random state and differentiate a JAX computation."""

import numpy

import asc

selected = asc.backend("jax")
xp = selected.xp
state = asc.random_state(42, backend="jax")
sample, next_state = asc.random.normal((3,), state=state, dtype=xp.float32)
gradient = asc.grad(lambda value: xp.sum(value**2), backend="jax")(sample)

assert next_state != state
numpy.testing.assert_allclose(
    numpy.asarray(gradient), 2 * numpy.asarray(sample), rtol=1e-6
)
print(sample)
print(gradient)
