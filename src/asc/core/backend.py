# Copyright 2026 AI4SciComp contributors.
# SPDX-License-Identifier: Apache-2.0
"""Immutable backend discovery and explicit selection."""

from __future__ import annotations

import collections.abc
import dataclasses
import importlib
import types
import typing
import warnings

from asc import _array_api_compat, errors
from asc import typing as asc_typing
from asc.backends import capabilities
from asc.core import namespace as namespace_module
from asc.core._scalar import require_representable_scalar
from asc.extensions import _dispatch


def _native_array_leaves(
    value: object, seen: set[int] | None = None
) -> tuple[object, ...]:
    """Find native arrays inside supported Python container structures."""
    if _array_api_compat.compat.is_array_api_obj(value):
        return (value,)
    if seen is None:
        seen = set()
    marker = id(value)
    if marker in seen:
        return ()
    if isinstance(value, collections.abc.Mapping):
        seen.add(marker)
        return tuple(
            leaf
            for item in value.values()
            for leaf in _native_array_leaves(item, seen)
        )
    if isinstance(value, collections.abc.Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        seen.add(marker)
        return tuple(
            leaf for item in value for leaf in _native_array_leaves(item, seen)
        )
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        seen.add(marker)
        return tuple(
            leaf
            for field in dataclasses.fields(value)
            for leaf in _native_array_leaves(getattr(value, field.name), seen)
        )
    return ()


def _validated_shape(shape: object, operation: str) -> asc_typing.Shape:
    """Return one canonical tuple shape or raise a stable context error."""
    if not isinstance(shape, tuple) or any(
        isinstance(extent, bool) or not isinstance(extent, int) or extent < 0
        for extent in shape
    ):
        raise errors.ContextError(
            f"{operation}: shape must be a tuple of non-negative integers"
        )
    return typing.cast(asc_typing.Shape, shape)


class _BackendAdapter(typing.Protocol):
    """Small adapter surface needed to construct a :class:`Backend`."""

    def namespace(self) -> object:
        """Return a 2024.12-compatible namespace."""
        ...  # pylint: disable=unnecessary-ellipsis

    def resolve_device(self, device: object | None) -> object | None:
        """Normalize an explicit CPU device."""
        ...  # pylint: disable=unnecessary-ellipsis

    def validate_dtype(self, dtype: object) -> None:
        """Reject a dtype unavailable under active backend configuration."""
        ...  # pylint: disable=unnecessary-ellipsis


@dataclasses.dataclass(frozen=True, slots=True)
class Backend:
    """Immutable selected backend and its extension namespaces."""

    name: asc_typing.BackendName
    xp: asc_typing.ArrayNamespace
    device: object | None
    dtype: object | None
    capabilities: frozenset[capabilities.Capability]

    @property
    def linalg(self) -> object:
        """Return the backend's normalized linear-algebra namespace."""
        module = importlib.import_module("asc.linalg")
        return module.linalg_namespace(self)

    @property
    def fft(self) -> object:
        """Return the backend's normalized Fourier namespace."""
        module = importlib.import_module("asc.fft")
        return module.fft_namespace(self)

    @property
    def random(self) -> types.ModuleType:
        """Return the explicit-state random public module."""
        return importlib.import_module("asc.random")

    @property
    def updates(self) -> types.ModuleType:
        """Return the functional update public module."""
        return importlib.import_module("asc.updates")

    @property
    def conversion(self) -> types.ModuleType:
        """Return the explicit conversion public module."""
        return importlib.import_module("asc.conversion")

    @property
    def autodiff(self) -> types.ModuleType:
        """Return the automatic-differentiation public module."""
        return importlib.import_module("asc.autodiff")

    @property
    def compilation(self) -> types.ModuleType:
        """Return the compilation public module."""
        return importlib.import_module("asc.compilation")

    def asarray(self, value: object, *, copy: bool | None = None) -> object:
        """Create a native array under this backend's explicit context."""
        try:
            native_leaves = _native_array_leaves(value)
        except Exception as exception:  # pylint: disable=broad-exception-caught
            raise errors.ContextError(
                "Backend.asarray: value exposes an invalid array protocol"
            ) from exception
        if native_leaves and native_leaves[0] is value:
            from asc.conversion import convert_array

            return convert_array(value, self, copy=copy)
        if native_leaves:
            try:
                tuple(
                    namespace_module.array_namespace(leaf)
                    for leaf in native_leaves
                )
            except errors.AscError as exception:
                raise errors.ContextError(
                    "Backend.asarray: nested arrays must be supported dense "
                    "numeric CPU arrays"
                ) from exception
            raise errors.ContextError(
                "Backend.asarray: nested native arrays, including nested "
                "foreign arrays, must be stacked or converted explicitly "
                "before array construction"
            )
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                result = self.xp.asarray(
                    value,
                    dtype=self.dtype,
                    device=self.device,
                    copy=copy,
                )
            observed = namespace_module.identify_backend(
                namespace_module.array_namespace(result)
            )
        except errors.AscError as exception:
            raise errors.ContextError(
                "Backend.asarray: value did not produce a supported dense "
                "numeric CPU array"
            ) from exception
        except Exception as exception:  # pylint: disable=broad-exception-caught
            raise errors.ContextError(
                "Backend.asarray: backend rejected the value, dtype, device, "
                "or copy policy"
            ) from exception
        if observed != self.name:
            raise errors.ContextError(
                "Backend.asarray: backend returned an array from another "
                "namespace"
            )
        return result

    def zeros(self, shape: asc_typing.Shape) -> object:
        """Create zeros under this backend's explicit context."""
        validated_shape = _validated_shape(shape, "Backend.zeros")
        try:
            result = self.xp.zeros(
                validated_shape, dtype=self.dtype, device=self.device
            )
            observed = namespace_module.identify_backend(
                namespace_module.array_namespace(result)
            )
        except errors.AscError as exception:
            raise errors.ContextError(
                "Backend.zeros: result is not a supported dense numeric CPU "
                "array"
            ) from exception
        except (
            OverflowError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as exception:
            raise errors.ContextError(
                "Backend.zeros: backend rejected the shape, dtype, or device"
            ) from exception
        if observed != self.name:
            raise errors.ContextError(
                "Backend.zeros: backend returned an array from another "
                "namespace"
            )
        return result

    def ones(self, shape: asc_typing.Shape) -> object:
        """Create ones under this backend's explicit context."""
        validated_shape = _validated_shape(shape, "Backend.ones")
        try:
            result = self.xp.ones(
                validated_shape, dtype=self.dtype, device=self.device
            )
            observed = namespace_module.identify_backend(
                namespace_module.array_namespace(result)
            )
        except errors.AscError as exception:
            raise errors.ContextError(
                "Backend.ones: result is not a supported dense numeric CPU "
                "array"
            ) from exception
        except (
            OverflowError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as exception:
            raise errors.ContextError(
                "Backend.ones: backend rejected the shape, dtype, or device"
            ) from exception
        if observed != self.name:
            raise errors.ContextError(
                "Backend.ones: backend returned an array from another namespace"
            )
        return result

    def full(self, shape: asc_typing.Shape, fill_value: object) -> object:
        """Create a filled array under this backend's explicit context."""
        validated_shape = _validated_shape(shape, "Backend.full")
        if _array_api_compat.compat.is_array_api_obj(fill_value):
            raise errors.ContextError(
                "Backend.full: fill_value must be a Python scalar, not an array"
            )
        require_representable_scalar(
            self.xp, self.dtype, fill_value, "Backend.full"
        )

        def create() -> object:
            return self.xp.full(
                validated_shape,
                fill_value,
                dtype=self.dtype,
                device=self.device,
            )

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                result = create()
            observed = namespace_module.identify_backend(
                namespace_module.array_namespace(result)
            )
        except errors.AscError as exception:
            raise errors.ContextError(
                "Backend.full: result is not a supported dense numeric CPU "
                "array"
            ) from exception
        except (
            OverflowError,
            RuntimeError,
            TypeError,
            ValueError,
            Warning,
        ) as exception:
            raise errors.ContextError(
                "Backend.full: backend rejected the shape, dtype, device, or "
                "fill value"
            ) from exception
        if observed != self.name:
            raise errors.ContextError(
                "Backend.full: backend returned an array from another namespace"
            )
        return result


def backend(
    name: typing.Literal["numpy", "torch", "jax"],
    *,
    device: object | None = None,
    dtype: object | None = None,
) -> Backend:
    """Return an immutable explicit backend without changing global state."""
    if name not in {"numpy", "torch", "jax"}:
        raise errors.NamespaceError(
            "backend: name must be one of ('numpy', 'torch', 'jax'); "
            f"received {name!r}"
        )
    typed_name = typing.cast(asc_typing.BackendName, name)
    module = typing.cast(_BackendAdapter, _dispatch.load_backend(typed_name))
    xp = typing.cast(asc_typing.ArrayNamespace, module.namespace())
    namespace_module.validate_namespace_revision(xp)
    resolved_device = module.resolve_device(device)
    if dtype is not None:
        module.validate_dtype(dtype)
        try:
            supported_dtype = xp.isdtype(
                dtype,
                (
                    "bool",
                    "signed integer",
                    "unsigned integer",
                    "real floating",
                    "complex floating",
                ),
            )
        except (AttributeError, TypeError, ValueError) as exception:
            raise errors.DTypeError(
                "backend: dtype does not belong to the selected namespace"
            ) from exception
        if not supported_dtype:
            raise errors.DTypeError(
                "backend: dtype is unsupported by the selected namespace"
            )
    info = capabilities.backend_info(name)
    return Backend(
        name=typed_name,
        xp=xp,
        device=resolved_device,
        dtype=dtype,
        capabilities=info.capabilities,
    )


def backend_of(array: object) -> asc_typing.BackendName:
    """Identify a native array without copy, transfer, or materialization."""
    xp = namespace_module.array_namespace(array)
    return namespace_module.identify_backend(xp)


def is_array(value: object) -> bool:
    """Return whether a value is a supported native array type."""
    try:
        if not _array_api_compat.compat.is_array_api_obj(value):
            return False
        namespace_module.array_namespace(value)
    except Exception:  # pylint: disable=broad-exception-caught
        return False
    return True


def available_backends() -> tuple[str, ...]:
    """Report installed backends without importing optional packages."""
    return tuple(
        name
        for name in ("numpy", "torch", "jax")
        if capabilities.backend_info(name).installed
    )


__all__ = [
    "Backend",
    "available_backends",
    "backend",
    "backend_of",
    "is_array",
]
