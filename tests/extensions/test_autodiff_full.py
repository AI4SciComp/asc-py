# Copyright 2026 AI4SciComp contributors.
# SPDX-License-Identifier: Apache-2.0
"""Autodiff and vectorization contracts for Torch and JAX."""

from __future__ import annotations

import numpy
import pytest

import asc


@pytest.mark.parametrize("backend", ("torch", "jax"))
def test_grad_jacobian_hessian_and_higher_derivative(backend: str) -> None:
    selected = asc.backend(backend)
    xp = selected.xp
    value = xp.asarray([1.0, 2.0], dtype=xp.float32)

    def scalar(array: object) -> object:
        return xp.sum(array**3)

    numpy.testing.assert_allclose(
        numpy.asarray(asc.grad(scalar, backend=backend)(value)), [3, 12]
    )
    result, gradient = asc.value_and_grad(scalar, backend=backend)(value)
    assert float(result) == pytest.approx(9)
    numpy.testing.assert_allclose(numpy.asarray(gradient), [3, 12])
    jacobian = asc.jacobian(lambda array: array**2, backend=backend)(value)
    numpy.testing.assert_allclose(numpy.asarray(jacobian), [[2, 0], [0, 4]])
    hessian = asc.hessian(scalar, backend=backend)(value)
    numpy.testing.assert_allclose(
        numpy.asarray(hessian), [[6, 0], [0, 12]], rtol=1e-5
    )
    second = asc.grad(
        lambda array: xp.sum(asc.grad(scalar, backend=backend)(array)),
        backend=backend,
    )(value)
    numpy.testing.assert_allclose(numpy.asarray(second), [6, 12], rtol=1e-5)


@pytest.mark.parametrize("backend", ("torch", "jax"))
def test_jvp_vjp_and_vmap(backend: str) -> None:
    selected = asc.backend(backend)
    xp = selected.xp
    value = xp.asarray([1.0, 2.0], dtype=xp.float32)
    tangent = xp.asarray([0.5, 1.0], dtype=xp.float32)
    forward = asc.jvp(
        lambda array: array**2, (value,), (tangent,), backend=backend
    )
    numpy.testing.assert_allclose(numpy.asarray(forward.primal), [1, 4])
    numpy.testing.assert_allclose(numpy.asarray(forward.tangent), [1, 4])
    reverse = asc.vjp(lambda array: array**2, value, backend=backend)
    cotangent = xp.asarray([1.0, 1.0], dtype=xp.float32)
    pulled = reverse.pullback(cotangent)
    gradient = pulled[0] if isinstance(pulled, tuple) else pulled
    numpy.testing.assert_allclose(numpy.asarray(gradient), [2, 4])
    mapped = asc.vmap(lambda item: item * item, backend=backend)(value)
    numpy.testing.assert_allclose(numpy.asarray(mapped), [1, 4])


@pytest.mark.parametrize("backend", ("torch", "jax"))
def test_vjp_accepts_structured_output_cotangents(backend: str) -> None:
    selected = asc.backend(backend)
    xp = selected.xp
    value = xp.asarray([1.0, 2.0], dtype=xp.float32)
    reverse = asc.vjp(lambda array: (array, array**2), value, backend=backend)
    cotangent = xp.ones_like(value)

    gradient = reverse.pullback((cotangent, cotangent))[0]

    numpy.testing.assert_allclose(numpy.asarray(gradient), [3.0, 5.0])


@pytest.mark.backend("jax")
@pytest.mark.backend("torch")
def test_jax_jit_and_capability_errors() -> None:
    selected = asc.backend("jax")
    xp = selected.xp
    value = xp.asarray([1.0, 2.0], dtype=xp.float32)
    compiled = asc.jit(lambda array: array * array + 1, backend="jax")
    numpy.testing.assert_allclose(numpy.asarray(compiled(value)), [2, 5])
    for backend in ("numpy", "torch", "array_api_strict"):
        with pytest.raises(asc.CapabilityNotSupportedError):
            asc.jit(lambda array: array, backend=backend)
    with pytest.raises(asc.CapabilityNotSupportedError):
        asc.grad(lambda array: array, backend="numpy")
    with pytest.raises(ValueError, match="equal length"):
        asc.jvp(lambda value: value, (value,), (), backend="jax")
    with pytest.raises(ValueError, match="primal"):
        asc.vjp(lambda: 1, backend="jax")
    for backend in ("torch", "jax"):
        native = asc.backend(backend).xp
        integers = native.asarray([1, 2], dtype=native.int32)
        with pytest.raises(asc.DTypeError, match="real floating"):
            asc.grad(
                lambda array, _native=native: _native.sum(array),
                backend=backend,
            )(integers)
        with pytest.raises(asc.DeviceError, match="CPU"):
            asc.vmap(lambda item: item, backend=backend)(1)


@pytest.mark.backend("jax")
def test_direct_jax_trace_cannot_bypass_cpu_proof() -> None:
    import jax
    import jax.numpy as jnp

    with pytest.raises(asc.CapabilityNotSupportedError, match="dense CPU"):
        jax.jit(lambda value: asc.sum_of_squares(value))(jnp.asarray([1.0]))


@pytest.mark.backend("torch")
def test_torch_jvp_and_vjp_require_real_floating_operands() -> None:
    selected = asc.backend("torch")
    xp = selected.xp
    floating = xp.asarray([1.0, 2.0], dtype=xp.float32)
    integer = xp.asarray([1, 2], dtype=xp.int32)
    complex_value = xp.asarray([1 + 1j, 2 + 0j], dtype=xp.complex64)

    for invalid in (integer, complex_value):
        with pytest.raises(asc.DTypeError, match="real floating"):
            asc.jvp(
                lambda value: value * value,
                (invalid,),
                (invalid,),
                backend="torch",
            )
        with pytest.raises(asc.DTypeError, match="real floating"):
            asc.vjp(lambda value: value * value, invalid, backend="torch")

    reverse = asc.vjp(lambda value: value * value, floating, backend="torch")
    with pytest.raises(asc.DTypeError, match="real floating"):
        reverse.pullback(integer)


@pytest.mark.backend("jax")
def test_jax_jvp_and_vjp_require_native_array_operands() -> None:
    selected = asc.backend("jax")
    value = selected.xp.asarray([1.0], dtype=selected.xp.float32)

    with pytest.raises(asc.DTypeError, match="every operand"):
        asc.jvp(lambda item: item, (value,), (1.0,), backend="jax")
    with pytest.raises(asc.DTypeError, match="every operand"):
        asc.vjp(lambda item: item, 1.0, backend="jax")
    reverse = asc.vjp(lambda item: item, value, backend="jax")
    with pytest.raises(asc.DTypeError, match="every operand"):
        reverse.pullback(1.0)
