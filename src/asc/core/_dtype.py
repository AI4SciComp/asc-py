# Copyright 2026 AI4SciComp contributors.
# SPDX-License-Identifier: Apache-2.0
"""Import-safe helpers for the frozen portable dtype surface."""

from __future__ import annotations

import os
import sys
import typing

from asc import errors
from asc import typing as asc_typing

DTYPE_ARGUMENT_FUNCTIONS: typing.Final = frozenset(
    {
        "arange",
        "argsort",
        "asarray",
        "astype",
        "cumulative_prod",
        "cumulative_sum",
        "empty",
        "empty_like",
        "eye",
        "full",
        "full_like",
        "linspace",
        "mean",
        "ones",
        "ones_like",
        "prod",
        "stack",
        "std",
        "sum",
        "var",
        "zeros",
        "zeros_like",
    }
)

_COMMON_DTYPES: typing.Final = (
    "bool",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "float32",
    "float64",
    "complex64",
    "complex128",
)
_DTYPES: typing.Final = {
    "array_api_strict": _COMMON_DTYPES,
    "numpy": (*_COMMON_DTYPES[:9], "float16", *_COMMON_DTYPES[9:]),
    "torch": (
        *_COMMON_DTYPES[:9],
        "float16",
        "bfloat16",
        *_COMMON_DTYPES[9:],
    ),
    "jax": (
        *_COMMON_DTYPES[:9],
        "float16",
        "bfloat16",
        *_COMMON_DTYPES[9:],
    ),
}
_JAX_WIDE_DTYPES: typing.Final = frozenset(
    {"int64", "uint64", "float64", "complex128"}
)


def extension_result_type(
    xp: asc_typing.ArrayNamespace,
    *operands: object,
    operation: str,
) -> object:
    """Return the common explicit-promotion dtype used by asc extensions."""
    try:
        return xp.result_type(*operands)
    except (errors.DTypeError, TypeError) as exception:
        floating: list[object] = []
        integer_count = 0
        for operand in operands:
            dtype = getattr(operand, "dtype", None)
            if dtype is None:
                continue
            if xp.isdtype(
                dtype,
                ("real floating", "complex floating"),
            ):
                floating.append(dtype)
            elif xp.isdtype(
                dtype,
                ("signed integer", "unsigned integer"),
            ):
                integer_count += 1
        if floating and len(floating) + integer_count == len(operands):
            return xp.result_type(*floating)
        raise errors.DTypeError(
            f"{operation}: operands do not have a portable promotion"
        ) from exception


def active_jax_x64_enabled() -> bool:
    """Return the active JAX x64 state without importing optional JAX."""
    module = sys.modules.get("jax")
    if module is not None:
        config = getattr(module, "config", None)
        return bool(getattr(config, "x64_enabled", False))
    configured = os.environ.get("JAX_ENABLE_X64", "").strip().lower()
    return configured in {"1", "on", "true", "yes"}


def dtype_name(dtype: object) -> str | None:
    """Return the standard name of a native dtype object, when recognizable."""
    if isinstance(dtype, str):
        return None
    for attribute_name in ("name", "__name__"):
        try:
            name = getattr(dtype, attribute_name, None)
        except Exception:  # pylint: disable=broad-exception-caught
            return None
        if isinstance(name, str) and name:
            return "bool" if name == "bool_" else name
    try:
        text = str(dtype)
    except Exception:  # pylint: disable=broad-exception-caught
        return None
    candidate = text.rsplit(".", maxsplit=1)[-1]
    if candidate == "bool_":
        candidate = "bool"
    return (
        candidate
        if candidate
        in frozenset(_COMMON_DTYPES)
        | {
            "float16",
            "bfloat16",
        }
        else None
    )


def _has_backend_provenance(
    backend: asc_typing.BackendName,
    dtype: object,
    *,
    allow_non_native_endian: bool,
) -> bool:
    """Return whether a dtype has canonical ownership and byte order."""
    name = dtype_name(dtype)
    if name is None:
        return False
    if backend == "array_api_strict":
        module = sys.modules.get("array_api_strict")
        canonical = None if module is None else getattr(module, name, None)
        return (
            canonical is not None
            and type(dtype) is type(canonical)
            and dtype == canonical
        )
    if backend == "torch":
        module = sys.modules.get("torch")
        return module is not None and dtype is getattr(module, name, None)
    module = sys.modules.get("numpy" if backend == "numpy" else "jax.numpy")
    if module is None:
        return False
    try:
        native_dtype = module.dtype(dtype)
        if backend == "numpy":
            is_canonical = dtype is native_dtype or dtype is native_dtype.type
        else:
            canonical = getattr(module, name, None)
            canonical_dtype = module.dtype(canonical)
            is_canonical = native_dtype == canonical_dtype and (
                dtype is canonical
                or dtype is native_dtype
                or dtype is getattr(native_dtype, "type", None)
            )
        is_native = getattr(native_dtype, "isnative", True) is not False
        has_metadata = getattr(native_dtype, "metadata", None) is not None
    except Exception:  # pylint: disable=broad-exception-caught
        return False
    return (
        is_canonical
        and not has_metadata
        and (is_native or (backend == "numpy" and allow_non_native_endian))
    )


def supported_dtype_names(
    backend: asc_typing.BackendName,
    *,
    jax_x64_enabled: bool = True,
) -> tuple[str, ...]:
    """Return exact native dtype names in the frozen release surface."""
    names = _DTYPES[backend]
    if backend == "jax" and not jax_x64_enabled:
        return tuple(name for name in names if name not in _JAX_WIDE_DTYPES)
    return names


def is_supported_dtype(
    backend: asc_typing.BackendName,
    dtype: object,
    *,
    jax_x64_enabled: bool = True,
    allow_non_native_endian: bool = False,
) -> bool:
    """Return whether a dtype is an exact member of the release surface."""
    return _has_backend_provenance(
        backend,
        dtype,
        allow_non_native_endian=allow_non_native_endian,
    ) and dtype_name(dtype) in supported_dtype_names(
        backend,
        jax_x64_enabled=jax_x64_enabled,
    )


def require_supported_dtype(
    backend: asc_typing.BackendName,
    dtype: object,
    operation: str,
    *,
    jax_x64_enabled: bool = True,
) -> None:
    """Raise a stable dtype error unless a native dtype is release-supported."""
    if not is_supported_dtype(backend, dtype, jax_x64_enabled=jax_x64_enabled):
        name = dtype_name(dtype)
        if name is None:
            try:
                description = repr(dtype)
            except Exception:  # pylint: disable=broad-exception-caught
                description = "<unprintable dtype>"
        else:
            description = repr(name)
        raise errors.DTypeError(
            f"{operation}: dtype {description} is outside the supported "
            f"{backend} release surface"
        )


__all__ = [
    "DTYPE_ARGUMENT_FUNCTIONS",
    "active_jax_x64_enabled",
    "dtype_name",
    "extension_result_type",
    "is_supported_dtype",
    "require_supported_dtype",
    "supported_dtype_names",
]
