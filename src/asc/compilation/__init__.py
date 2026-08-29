# Copyright 2026 AI4SciComp contributors.
# SPDX-License-Identifier: Apache-2.0
"""Backend-native JIT and vectorized mapping transforms."""

from __future__ import annotations

import typing

from asc import errors
from asc import typing as asc_typing
from asc.extensions import _dispatch


def _adapter(backend: asc_typing.BackendName, operation: str) -> object:
    supported = {"jax"} if operation == "jit" else {"torch", "jax"}
    if backend not in supported:
        raise errors.CapabilityNotSupportedError(
            f"{operation}: backend {backend!r} does not provide compilation"
        )
    return _dispatch.load_backend(backend)


def jit[**P, R](
    function: typing.Callable[P, R],
    *,
    backend: asc_typing.BackendName,
) -> typing.Callable[P, R]:
    """Compile a callable with the declared backend capability."""
    compiled = _adapter(backend, "jit").compile_function(function)
    return typing.cast(typing.Callable[P, R], compiled)


def vmap[**P, R](
    function: typing.Callable[P, R],
    *,
    backend: asc_typing.BackendName,
    in_axes: object = 0,
    out_axes: object = 0,
) -> typing.Callable[P, R]:
    """Vectorize a callable over an explicit axis subset."""
    transformed = _adapter(backend, "vmap").vmap(function, in_axes, out_axes)
    return typing.cast(typing.Callable[P, R], transformed)


compile_function = jit

__all__ = ["compile_function", "jit", "vmap"]
