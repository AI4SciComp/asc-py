# Copyright 2026 AI4SciComp contributors.
# SPDX-License-Identifier: Apache-2.0
"""Immutable, import-safe backend capability descriptions."""

from __future__ import annotations

import dataclasses
import enum
import importlib.metadata
import typing

from asc import _array_api_compat, errors
from asc import typing as asc_typing
from asc.core import _dtype
from asc.core import namespace as namespace_module


class Capability(enum.StrEnum):
    """Stable names for optional backend behavior."""

    ARRAY_API = "array_api"
    LINALG = "linalg"
    FFT = "fft"
    COMPLEX = "complex"
    FLOAT64 = "float64"
    DLPACK = "dlpack"
    RANDOM = "random"
    UPDATES = "updates"
    AUTODIFF = "autodiff"
    JIT = "jit"
    VMAP = "vmap"


@dataclasses.dataclass(frozen=True, slots=True)
class BackendInfo:
    """Installed state and declared CPU capabilities for one backend."""

    name: asc_typing.BackendName
    installed: bool
    version: str | None
    array_api_version: str
    devices: tuple[str, ...]
    dtypes: tuple[str, ...]
    capabilities: frozenset[Capability]

    @property
    def dtype_families(self) -> tuple[str, ...]:
        """Return compatibility dtype-family names."""
        return (
            "bool",
            "signed integer",
            "unsigned integer",
            "real floating",
            "complex floating",
        )

    @property
    def random(self) -> bool:
        """Return whether explicit random state is supported."""
        return Capability.RANDOM in self.capabilities

    @property
    def index_add(self) -> bool:
        """Return whether functional updates are supported."""
        return Capability.UPDATES in self.capabilities

    @property
    def autodiff(self) -> bool:
        """Return whether automatic differentiation is supported."""
        return Capability.AUTODIFF in self.capabilities

    @property
    def compilation(self) -> bool:
        """Return whether JIT compilation is supported."""
        return Capability.JIT in self.capabilities


@dataclasses.dataclass(frozen=True, slots=True)
class _CapabilitySpec:
    distribution: str
    extra: str | None
    capabilities: frozenset[Capability]


_STANDARD = frozenset(
    {
        Capability.ARRAY_API,
        Capability.LINALG,
        Capability.FFT,
        Capability.COMPLEX,
        Capability.FLOAT64,
        Capability.DLPACK,
    }
)
_CAPABILITIES: typing.Final = {
    "array_api_strict": _CapabilitySpec("array-api-strict", None, _STANDARD),
    "numpy": _CapabilitySpec(
        "numpy",
        None,
        _STANDARD | {Capability.RANDOM, Capability.UPDATES},
    ),
    "torch": _CapabilitySpec(
        "torch",
        "torch",
        _STANDARD
        | {
            Capability.RANDOM,
            Capability.UPDATES,
            Capability.AUTODIFF,
            Capability.VMAP,
        },
    ),
    "jax": _CapabilitySpec(
        "jax",
        "jax",
        _STANDARD
        | {
            Capability.RANDOM,
            Capability.UPDATES,
            Capability.AUTODIFF,
            Capability.JIT,
            Capability.VMAP,
        },
    ),
}


def _validated_name(value: str) -> asc_typing.BackendName:
    if value not in _CAPABILITIES:
        raise errors.NamespaceError(
            "backend_info: unsupported backend name "
            f"{value!r}; expected one of {tuple(_CAPABILITIES)!r}"
        )
    return typing.cast(asc_typing.BackendName, value)


def _name_from_value(value: object) -> asc_typing.BackendName:
    if isinstance(value, str):
        return _validated_name(value)
    if hasattr(value, "name") and value.name in _CAPABILITIES:
        return _validated_name(typing.cast(str, value.name))
    if _array_api_compat.compat.is_array_api_obj(value):
        return namespace_module.identify_backend(
            namespace_module.array_namespace(value)
        )
    return namespace_module.identify_backend(value)


def backend_info(value_or_name: object) -> BackendInfo:
    """Return immutable capabilities without importing optional backends."""
    name = _name_from_value(value_or_name)
    spec = _CAPABILITIES[name]
    try:
        version = importlib.metadata.version(spec.distribution)
    except importlib.metadata.PackageNotFoundError:
        version = None
    dtypes = _dtype.supported_dtype_names(
        name,
        jax_x64_enabled=name != "jax" or _dtype.active_jax_x64_enabled(),
    )
    capabilities = spec.capabilities
    if name == "jax" and not _dtype.active_jax_x64_enabled():
        capabilities = capabilities - {Capability.FLOAT64}
    return BackendInfo(
        name=name,
        installed=version is not None,
        version=version,
        array_api_version=namespace_module.ARRAY_API_VERSION,
        devices=("cpu",),
        dtypes=dtypes,
        capabilities=capabilities,
    )


def has_capability(value_or_name: object, capability: Capability | str) -> bool:
    """Return whether a backend declares a capability without emulation."""
    try:
        normalized = Capability(capability)
    except ValueError:
        return False
    info = backend_info(value_or_name)
    return info.installed and normalized in info.capabilities


def require_capability(
    value_or_name: object, capability: Capability | str
) -> None:
    """Raise an actionable error unless a capability is installed."""
    try:
        normalized = Capability(capability)
    except ValueError as exception:
        raise errors.CapabilityNotSupportedError(
            f"require_capability: unknown capability {capability!r}"
        ) from exception
    info = backend_info(value_or_name)
    if not info.installed:
        spec = _CAPABILITIES[info.name]
        suffix = "" if spec.extra is None else f"; install asc-py[{spec.extra}]"
        raise errors.BackendUnavailableError(
            f"require_capability: backend {info.name!r} is unavailable{suffix}"
        )
    if normalized not in info.capabilities:
        raise errors.CapabilityNotSupportedError(
            "require_capability: backend "
            f"{info.name!r} does not support {normalized.value!r}"
        )


__all__ = [
    "BackendInfo",
    "Capability",
    "backend_info",
    "has_capability",
    "require_capability",
]
