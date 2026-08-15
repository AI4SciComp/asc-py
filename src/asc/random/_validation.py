# Copyright 2026 AI4SciComp contributors.
# SPDX-License-Identifier: Apache-2.0
"""Validation helpers for backend-neutral random parameters."""

from __future__ import annotations

import math

from asc import _array_api_compat, errors
from asc.core._scalar import normalize_real_scalar


def finite_python_real(value: object, operation: str, parameter: str) -> float:
    """Return one finite built-in real scalar without array coercion."""
    if (
        _array_api_compat.compat.is_array_api_obj(value)
        or isinstance(value, bool)
        or not isinstance(value, (int, float))
    ):
        raise errors.RandomStateError(
            f"{operation}: {parameter} must be a finite Python real scalar"
        )
    try:
        result = float(value)
    except OverflowError as exception:
        raise errors.RandomStateError(
            f"{operation}: {parameter} must be a finite Python real scalar"
        ) from exception
    if not math.isfinite(result):
        raise errors.RandomStateError(
            f"{operation}: {parameter} must be a finite Python real scalar"
        )
    return result


def floating_parameters(
    backend_name: str,
    dtype: object | None,
    operation: str,
    **parameters: float,
) -> dict[str, float]:
    """Round parameters to a finite, nonzero-preserving output dtype."""
    from asc.core.backend import backend as select_backend

    try:
        selected = select_backend(backend_name, dtype=dtype)
        requested_dtype = selected.dtype
        if requested_dtype is None:
            requested_dtype = selected.xp.asarray(
                0.0, device=selected.device
            ).dtype
        if not selected.xp.isdtype(requested_dtype, "real floating"):
            raise errors.RandomStateError(
                f"{operation}: dtype must be real floating"
            )
    except errors.RandomStateError:
        raise
    except (
        errors.AscError,
        AttributeError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exception:
        raise errors.RandomStateError(
            f"{operation}: backend rejected the requested floating dtype"
        ) from exception
    try:
        rounded = {
            name: normalize_real_scalar(
                selected.xp,
                requested_dtype,
                value,
                operation,
                name,
                device=selected.device,
            )
            for name, value in parameters.items()
        }
    except (
        errors.AscError,
        AttributeError,
        OverflowError,
        RuntimeError,
        TypeError,
        ValueError,
        Warning,
    ) as exception:
        raise errors.RandomStateError(
            f"{operation}: parameters must be representable in the output dtype"
        ) from exception
    if any(
        not math.isfinite(rounded[name])
        or (value != 0.0 and rounded[name] == 0.0)
        for name, value in parameters.items()
    ):
        raise errors.RandomStateError(
            f"{operation}: parameters must be representable in the output dtype"
        )
    return rounded


__all__: list[str] = []
