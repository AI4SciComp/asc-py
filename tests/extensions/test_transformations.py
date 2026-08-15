# Copyright 2026 AI4SciComp contributors.
# SPDX-License-Identifier: Apache-2.0

import numpy
import pytest

import asc
from asc import typing as asc_typing
from asc.extensions import autodiff, compilation
from tests import helpers

TRANSFORM_BACKENDS: tuple[asc_typing.BackendName, ...] = ("jax", "torch")


def _objective(value: asc_typing.NativeArray) -> asc_typing.NativeArray:
    return asc.sum_of_squares(value)


@pytest.mark.parametrize("backend", TRANSFORM_BACKENDS)
def test_value_and_grad_matches_analytic_gradient(
    backend: asc_typing.BackendName,
) -> None:
    value = helpers.float_array(backend, [2.0, 3.0])
    transformed = autodiff.value_and_grad(_objective, backend=backend)

    result, gradient = transformed(value)

    numpy.testing.assert_allclose(helpers.as_numpy(result), 13.0)
    numpy.testing.assert_allclose(helpers.as_numpy(gradient), [4.0, 6.0])


@pytest.mark.backend("jax")
def test_compiled_value_matches_eager() -> None:
    value = helpers.float_array("jax", [2.0, 3.0])
    compiled = compilation.compile_function(_objective, backend="jax")

    result = compiled(value)

    numpy.testing.assert_allclose(helpers.as_numpy(result), 13.0)


@pytest.mark.parametrize("backend", ["array_api_strict", "numpy"])
def test_unsupported_transformations_fail_clearly(
    backend: asc_typing.BackendName,
) -> None:
    with pytest.raises(asc.UnsupportedCapabilityError, match="autodiff"):
        autodiff.value_and_grad(_objective, backend=backend)
    with pytest.raises(asc.UnsupportedCapabilityError, match="compilation"):
        compilation.compile_function(_objective, backend=backend)


@pytest.mark.backend("torch")
def test_torch_compilation_is_unsupported() -> None:
    with pytest.raises(asc.UnsupportedCapabilityError, match="compilation"):
        compilation.compile_function(_objective, backend="torch")


@pytest.mark.backend("jax")
def test_value_and_grad_rejects_negative_argument() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        autodiff.value_and_grad(_objective, backend="jax", argument=-1)
