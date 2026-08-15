# Copyright 2026 AI4SciComp contributors.
# SPDX-License-Identifier: Apache-2.0
"""Execute examples independently for NumPy, Torch, and JAX extras."""

from __future__ import annotations

import numpy
import pytest

import asc


def test_numpy_example() -> None:
    selected = asc.backend("numpy")
    xp = selected.xp
    values = xp.asarray([1.0, 2.0, 3.0], dtype=xp.float32)
    result = asc.sum_of_squares(values)
    numpy.testing.assert_allclose(result, 14.0)


@pytest.mark.backend("torch")
def test_torch_example() -> None:
    import torch

    values = torch.asarray([1.0, 2.0, 3.0], dtype=torch.float32)
    result = asc.sum_of_squares(values)
    torch.testing.assert_close(result, torch.tensor(14.0))


@pytest.mark.backend("jax")
def test_jax_example() -> None:
    import jax.numpy as jnp

    values = jnp.asarray([1.0, 2.0, 3.0], dtype=jnp.float32)
    result = asc.sum_of_squares(values)
    numpy.testing.assert_allclose(numpy.asarray(result), 14.0)
