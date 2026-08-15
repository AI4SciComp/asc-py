# Copyright 2026 AI4SciComp contributors.
# SPDX-License-Identifier: Apache-2.0
"""Native Torch and JAX automatic-differentiation transformations."""

from __future__ import annotations

import typing

from asc import errors
from asc import typing as asc_typing
from asc.extensions import _dispatch


class JVPResult(typing.NamedTuple):
    """Primal output and forward-mode tangent."""

    primal: object
    tangent: object


class VJPResult(typing.NamedTuple):
    """Primal output and reverse-mode pullback callable."""

    primal: object
    pullback: typing.Callable[..., object]


def _adapter(backend: asc_typing.BackendName, operation: str) -> object:
    if backend not in {"torch", "jax"}:
        raise errors.CapabilityNotSupportedError(
            f"{operation}: backend {backend!r} does not provide autodiff"
        )
    return _dispatch.load_backend(backend)


def _argument(argument: int, operation: str) -> int:
    if (
        isinstance(argument, bool)
        or not isinstance(argument, int)
        or argument < 0
    ):
        raise ValueError(
            f"{operation}: argument must be a non-negative integer"
        )
    return argument


def grad[**P, R](
    function: typing.Callable[P, R],
    *,
    backend: asc_typing.BackendName,
    argument: int = 0,
) -> typing.Callable[P, object]:
    """Return a scalar-output gradient transformation."""
    adapter = _adapter(backend, "grad")
    return typing.cast(
        typing.Callable[P, object],
        adapter.grad(function, _argument(argument, "grad")),
    )


def value_and_grad[**P, R](
    function: typing.Callable[P, R],
    *,
    backend: asc_typing.BackendName,
    argument: int = 0,
) -> typing.Callable[P, tuple[R, object]]:
    """Return value and gradient without duplicate function evaluation."""
    adapter = _adapter(backend, "value_and_grad")
    return typing.cast(
        typing.Callable[P, tuple[R, object]],
        adapter.value_and_grad(function, _argument(argument, "value_and_grad")),
    )


def jacobian[**P](
    function: typing.Callable[P, object],
    *,
    backend: asc_typing.BackendName,
    argument: int = 0,
) -> typing.Callable[P, object]:
    """Return a Jacobian transformation using output-shape then input-shape."""
    adapter = _adapter(backend, "jacobian")
    return typing.cast(
        typing.Callable[P, object],
        adapter.jacobian(function, _argument(argument, "jacobian")),
    )


def hessian[**P](
    function: typing.Callable[P, object],
    *,
    backend: asc_typing.BackendName,
    argument: int = 0,
) -> typing.Callable[P, object]:
    """Return a second-derivative transformation for a scalar output."""
    adapter = _adapter(backend, "hessian")
    return typing.cast(
        typing.Callable[P, object],
        adapter.hessian(function, _argument(argument, "hessian")),
    )


def jvp(
    function: typing.Callable[..., object],
    primals: tuple[object, ...],
    tangents: tuple[object, ...],
    *,
    backend: asc_typing.BackendName,
) -> JVPResult:
    """Evaluate a forward-mode Jacobian-vector product."""
    if len(primals) != len(tangents):
        raise ValueError("jvp: primals and tangents must have equal length")
    result = _adapter(backend, "jvp").jvp(function, primals, tangents)
    return JVPResult(result[0], result[1])


def vjp(
    function: typing.Callable[..., object],
    *primals: object,
    backend: asc_typing.BackendName,
) -> VJPResult:
    """Evaluate a reverse-mode transform and return its pullback."""
    if not primals:
        raise ValueError("vjp: at least one primal is required")
    result = _adapter(backend, "vjp").vjp(function, tuple(primals))
    return VJPResult(result[0], result[1])


__all__ = [
    "JVPResult",
    "VJPResult",
    "grad",
    "hessian",
    "jacobian",
    "jvp",
    "value_and_grad",
    "vjp",
]
