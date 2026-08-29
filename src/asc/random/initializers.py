# Copyright 2026 AI4SciComp contributors.
# SPDX-License-Identifier: Apache-2.0
"""Backend-native initializers driven by explicit random state."""

from __future__ import annotations

import math
import struct
import typing

from asc import errors
from asc.random._validation import finite_python_real, floating_parameters

if typing.TYPE_CHECKING:
    from asc.random import RandomState


def _shape(shape: tuple[int, ...], operation: str) -> tuple[int, ...]:
    if not isinstance(shape, tuple) or any(
        isinstance(extent, bool) or not isinstance(extent, int) or extent < 0
        for extent in shape
    ):
        raise errors.RandomStateError(
            f"{operation}: shape must contain non-negative integers"
        )
    return shape


def _state_backend(state: object, operation: str) -> str:
    """Return a validated native random-state backend name."""
    from asc.random import RandomState

    if not isinstance(state, RandomState):
        raise errors.RandomStateError(
            f"{operation}: state must be an asc.random.RandomState"
        )
    return state.backend


def _state_device(state: RandomState) -> object | None:
    """Return a concrete JAX state device, if the key exposes one."""
    device = getattr(state.key, "device", None)
    return device() if callable(device) else device


def _fans(shape: tuple[int, ...]) -> tuple[int, int]:
    if len(shape) < 2 or any(
        isinstance(extent, bool) or not isinstance(extent, int) or extent <= 0
        for extent in shape
    ):
        raise errors.RandomStateError(
            "initializer: shape must have at least two positive dimensions"
        )
    receptive = math.prod(shape[2:]) if len(shape) > 2 else 1
    return shape[1] * receptive, shape[0] * receptive


def _rounded_float(value: float, dtype_name: str) -> float:
    """Round a derived initializer bound to its requested floating dtype."""
    if "bfloat16" in dtype_name:
        bits = struct.unpack("!I", struct.pack("!f", value))[0]
        least_significant = (bits >> 16) & 1
        rounded = (bits + 0x7FFF + least_significant) & 0xFFFF0000
        return struct.unpack("!f", struct.pack("!I", rounded))[0]
    if "float16" in dtype_name:
        return struct.unpack("!e", struct.pack("!e", value))[0]
    if "float32" in dtype_name:
        return struct.unpack("!f", struct.pack("!f", value))[0]
    return value


def _uniform_limit(
    value: float,
    state: RandomState,
    dtype: object | None,
    operation: str,
) -> float:
    """Return an exactly representable backend initializer limit."""
    from asc.core.backend import backend as select_backend

    selected = select_backend(
        _state_backend(state, operation), device=_state_device(state)
    )
    requested_dtype = dtype
    if requested_dtype is None:
        requested_dtype = selected.xp.asarray(0.0, device=selected.device).dtype
    return _rounded_float(value, str(requested_dtype))


def _validate_truncated_bounds(
    lower: float,
    upper: float,
    state: RandomState,
    dtype: object | None,
) -> None:
    """Require bounds to be exactly representable in the output dtype."""
    from asc.core.backend import backend as select_backend

    selected = select_backend(state.backend, device=_state_device(state))
    requested_dtype = dtype
    if requested_dtype is None:
        requested_dtype = selected.xp.asarray(0.0, device=selected.device).dtype
    try:
        rounded = (
            _rounded_float(lower, str(requested_dtype)),
            _rounded_float(upper, str(requested_dtype)),
        )
    except (OverflowError, struct.error) as exception:
        raise errors.RandomStateError(
            "truncated_normal: bounds must be representable in the requested "
            "dtype"
        ) from exception
    if rounded != (lower, upper):
        raise errors.RandomStateError(
            "truncated_normal: bounds must be exactly representable in the "
            "requested dtype"
        )


def constant(
    shape: tuple[int, ...],
    value: object,
    *,
    backend: typing.Literal["numpy", "torch", "jax"] | object,
    dtype: object | None = None,
    device: object | None = None,
) -> object:
    """Return an explicitly placed constant array."""
    from asc.core.backend import backend as select_backend

    name = getattr(backend, "name", backend)
    effective_dtype = (
        getattr(backend, "dtype", None) if dtype is None else dtype
    )
    effective_device = (
        getattr(backend, "device", None) if device is None else device
    )
    selected = select_backend(
        name, device=effective_device, dtype=effective_dtype
    )
    return selected.full(_shape(shape, "constant"), value)


def truncated_normal(
    shape: tuple[int, ...],
    *,
    state: RandomState,
    mean: float = 0.0,
    std: float = 1.0,
    lower: float = -2.0,
    upper: float = 2.0,
    dtype: object | None = None,
) -> tuple[object, object]:
    """Sample a backend-native truncated normal initializer."""
    mean = finite_python_real(mean, "truncated_normal", "mean")
    std = finite_python_real(std, "truncated_normal", "std")
    lower = finite_python_real(lower, "truncated_normal", "lower")
    upper = finite_python_real(upper, "truncated_normal", "upper")
    if std <= 0 or lower >= upper:
        raise errors.RandomStateError(
            "truncated_normal: bounds and parameters must be finite, std must "
            "be positive, and lower must be less than upper"
        )
    parameters = floating_parameters(
        _state_backend(state, "truncated_normal"),
        dtype,
        "truncated_normal",
        mean=mean,
        std=std,
    )
    mean = parameters["mean"]
    std = parameters["std"]
    _validate_truncated_bounds(lower, upper, state, dtype)
    from asc.random import _call  # pyright: ignore[reportPrivateUsage]

    return _call(
        state,
        "truncated_normal",
        _shape(shape, "truncated_normal"),
        mean=mean,
        std=std,
        lower=lower,
        upper=upper,
        dtype=dtype,
    )


def glorot_uniform(
    shape: tuple[int, ...], *, state: RandomState, dtype: object | None = None
) -> tuple[object, object]:
    """Sample a Glorot/Xavier uniform initializer."""
    fan_in, fan_out = _fans(shape)
    limit = _uniform_limit(
        math.sqrt(6.0 / (fan_in + fan_out)),
        state,
        dtype,
        "glorot_uniform",
    )
    from asc.random import uniform

    return uniform(shape, state=state, low=-limit, high=limit, dtype=dtype)


def glorot_normal(
    shape: tuple[int, ...], *, state: RandomState, dtype: object | None = None
) -> tuple[object, object]:
    """Sample a Glorot/Xavier normal initializer."""
    fan_in, fan_out = _fans(shape)
    from asc.random import normal

    return normal(
        shape,
        state=state,
        std=math.sqrt(2.0 / (fan_in + fan_out)),
        dtype=dtype,
    )


def lecun_uniform(
    shape: tuple[int, ...], *, state: RandomState, dtype: object | None = None
) -> tuple[object, object]:
    """Sample a LeCun uniform initializer."""
    fan_in, _ = _fans(shape)
    limit = _uniform_limit(
        math.sqrt(3.0 / fan_in), state, dtype, "lecun_uniform"
    )
    from asc.random import uniform

    return uniform(shape, state=state, low=-limit, high=limit, dtype=dtype)


def lecun_normal(
    shape: tuple[int, ...], *, state: RandomState, dtype: object | None = None
) -> tuple[object, object]:
    """Sample a LeCun normal initializer."""
    fan_in, _ = _fans(shape)
    from asc.random import normal

    return normal(shape, state=state, std=math.sqrt(1.0 / fan_in), dtype=dtype)


def he_uniform(
    shape: tuple[int, ...], *, state: RandomState, dtype: object | None = None
) -> tuple[object, object]:
    """Sample a Kaiming/He uniform initializer."""
    fan_in, _ = _fans(shape)
    limit = _uniform_limit(math.sqrt(6.0 / fan_in), state, dtype, "he_uniform")
    from asc.random import uniform

    return uniform(shape, state=state, low=-limit, high=limit, dtype=dtype)


def he_normal(
    shape: tuple[int, ...], *, state: RandomState, dtype: object | None = None
) -> tuple[object, object]:
    """Sample a Kaiming/He normal initializer."""
    fan_in, _ = _fans(shape)
    from asc.random import normal

    return normal(shape, state=state, std=math.sqrt(2.0 / fan_in), dtype=dtype)


def orthogonal(
    shape: tuple[int, ...],
    *,
    state: RandomState,
    gain: float = 1.0,
    dtype: object | None = None,
) -> tuple[object, object]:
    """Sample an orthogonal matrix/tensor using native QR decomposition."""
    gain = finite_python_real(gain, "orthogonal", "gain")
    if len(shape) < 2 or any(
        isinstance(extent, bool) or not isinstance(extent, int) or extent <= 0
        for extent in shape
    ):
        raise errors.RandomStateError(
            "orthogonal: shape must have at least two positive dimensions "
            "and gain must be finite"
        )
    backend_name = _state_backend(state, "orthogonal")
    parameters = floating_parameters(
        backend_name, dtype, "orthogonal", gain=gain
    )
    gain = parameters["gain"]
    rows = shape[0]
    columns = math.prod(shape[1:])
    transposed = rows < columns
    matrix_shape = (columns, rows) if transposed else (rows, columns)
    from asc.core.backend import backend as select_backend
    from asc.random import normal

    selected = select_backend(
        backend_name, device=_state_device(state), dtype=dtype
    )
    effective_dtype = selected.asarray(0.0).dtype
    dtype_name = str(effective_dtype).rsplit(".", maxsplit=1)[-1]
    if dtype_name not in {"float32", "float64"}:
        raise errors.RandomStateError(
            "orthogonal: requested dtype is unsupported by portable CPU QR"
        )
    matrix, next_state = normal(matrix_shape, state=state, dtype=dtype)
    try:
        qr = selected.linalg.qr(matrix, mode="reduced")
    except (
        RuntimeError,
        TypeError,
        ValueError,
        NotImplementedError,
    ) as exception:
        raise errors.RandomStateError(
            "orthogonal: selected backend cannot perform CPU QR for the "
            "requested dtype"
        ) from exception
    diagonal = selected.linalg.diagonal(qr.R)
    signs = selected.xp.sign(diagonal)
    signs = selected.xp.where(signs == 0, 1.0, signs)
    q = qr.Q * signs
    if transposed:
        q = selected.xp.matrix_transpose(q)
    return gain * selected.xp.reshape(q, shape), next_state


__all__ = [
    "constant",
    "glorot_normal",
    "glorot_uniform",
    "he_normal",
    "he_uniform",
    "lecun_normal",
    "lecun_uniform",
    "orthogonal",
    "truncated_normal",
]
