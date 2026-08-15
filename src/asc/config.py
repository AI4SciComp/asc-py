# Copyright 2026 AI4SciComp contributors.
# SPDX-License-Identifier: Apache-2.0
"""Immutable configuration values for explicit portable operations."""

from __future__ import annotations

import dataclasses
import enum
import typing

from asc import errors
from asc import typing as asc_typing
from asc.core import _dtype
from asc.core import namespace as namespace_module
from asc.core.device import is_cpu_device

_AMBIGUOUS_CSV_DELIMITERS = frozenset("0123456789+-.eEjJ()% nNaAiIfFtTyY")


def _is_supported_context_dtype(
    backend: asc_typing.BackendName, dtype: object
) -> bool:
    """Validate canonical native dtype identity for an array context."""
    return _dtype.is_supported_dtype(
        backend,
        dtype,
        jax_x64_enabled=(backend != "jax" or _dtype.active_jax_x64_enabled()),
    )


def _validate_extensions(value: object) -> tuple[ExtensionHandle, ...]:
    if not isinstance(value, tuple):
        raise errors.ContextError("context: extensions must be a tuple")
    raw_extensions = typing.cast(tuple[object, ...], value)
    if not all(isinstance(item, ExtensionHandle) for item in raw_extensions):
        raise errors.ContextError(
            "context: extensions must contain ExtensionHandle values"
        )
    return typing.cast(tuple[ExtensionHandle, ...], raw_extensions)


class PrecisionPolicy(enum.StrEnum):
    """Policy for a backend that cannot represent a requested dtype."""

    INHERIT = "inherit"
    STRICT = "strict"
    ALLOW_NARROWING = "allow_narrowing"


class CopyPolicy(enum.StrEnum):
    """Required ownership policy at an explicit conversion boundary."""

    ALWAYS = "always"
    IF_NEEDED = "if_needed"
    NEVER = "never"


@dataclasses.dataclass(frozen=True, slots=True)
class ArrayContext:
    """Explicit backend, dtype, device, precision, copy, and random policy."""

    backend: asc_typing.BackendName
    dtype: object | None = None
    device: object | None = "cpu"
    precision: PrecisionPolicy = PrecisionPolicy.INHERIT
    copy: CopyPolicy = CopyPolicy.IF_NEEDED
    random_state: object | None = None

    def __post_init__(self) -> None:
        """Validate the supported backend and dense-CPU release boundary."""
        if self.backend not in {"numpy", "torch", "jax", "array_api_strict"}:
            raise errors.ContextError(
                f"ArrayContext: unsupported backend {self.backend!r}"
            )
        if not is_cpu_device(self.device):
            raise errors.DeviceError(
                "ArrayContext: 0.1.0 supports only explicit CPU devices"
            )
        if not isinstance(self.precision, PrecisionPolicy):
            raise errors.ContextError(
                "ArrayContext: precision must be a PrecisionPolicy"
            )
        if not isinstance(self.copy, CopyPolicy):
            raise errors.ContextError("ArrayContext: copy must be a CopyPolicy")
        if self.dtype is not None and not _is_supported_context_dtype(
            self.backend, self.dtype
        ):
            raise errors.ContextError(
                "ArrayContext: dtype is outside the selected backend's "
                "supported release surface"
            )
        if self.random_state is not None:
            from asc.random import RandomState

            if not isinstance(self.random_state, RandomState):
                raise errors.ContextError(
                    "ArrayContext: random_state must be an asc.RandomState"
                )
        if (
            self.random_state is not None
            and self.random_state.backend != self.backend
        ):
            raise errors.ContextError(
                "ArrayContext: random_state must match the context backend"
            )


@dataclasses.dataclass(frozen=True, slots=True)
class DataLoaderConfig:
    """Validated immutable baseline loader configuration."""

    batch_size: int | None = 1
    shuffle: bool = False
    drop_last: bool = False

    def __post_init__(self) -> None:
        """Validate batch size and mutually consistent Boolean policies."""
        if self.batch_size is not None and (
            isinstance(self.batch_size, bool)
            or not isinstance(self.batch_size, int)
            or self.batch_size <= 0
        ):
            raise errors.DataLoaderError(
                "DataLoaderConfig: batch_size must be a positive integer "
                "or None"
            )
        if not isinstance(self.shuffle, bool) or not isinstance(
            self.drop_last, bool
        ):
            raise errors.DataLoaderError(
                "DataLoaderConfig: shuffle and drop_last must be Boolean"
            )
        if self.batch_size is None and self.drop_last:
            raise errors.DataLoaderError(
                "DataLoaderConfig: drop_last requires batching to be enabled"
            )


@dataclasses.dataclass(frozen=True, slots=True)
class NpyOptions:
    """Immutable NPY loading options."""

    mmap_mode: typing.Literal["r", "r+", "w+", "c"] | None = None
    allow_unsafe_pickle: bool = False

    def __post_init__(self) -> None:
        """Validate safe loading policy fields."""
        if self.mmap_mode not in {None, "r", "r+", "w+", "c"} or not isinstance(
            self.allow_unsafe_pickle, bool
        ):
            raise errors.ContextError("NpyOptions: invalid loading options")


@dataclasses.dataclass(frozen=True, slots=True)
class NpzOptions:
    """Immutable NPZ writing options."""

    compressed: bool = True

    def __post_init__(self) -> None:
        """Require an explicit Boolean compression policy."""
        if not isinstance(self.compressed, bool):
            raise errors.ContextError("NpzOptions: compressed must be Boolean")


@dataclasses.dataclass(frozen=True, slots=True)
class CsvOptions:
    """Immutable numeric CSV options."""

    delimiter: str = ","
    header: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        """Require an unambiguous delimiter and safe header fields."""
        if (
            not isinstance(self.delimiter, str)
            or len(self.delimiter) != 1
            or self.delimiter in {"\n", "\r"}
            or self.delimiter in _AMBIGUOUS_CSV_DELIMITERS
        ):
            raise errors.ContextError(
                "CsvOptions: delimiter must be one unambiguous character"
            )
        if self.header is not None and (
            not isinstance(self.header, tuple)
            or any(
                not isinstance(name, str)
                or not name
                or self.delimiter in name
                or "\n" in name
                or "\r" in name
                for name in self.header
            )
        ):
            raise errors.ContextError(
                "CsvOptions: header fields must be non-empty and contain no "
                "delimiter or newline"
            )


@dataclasses.dataclass(frozen=True, slots=True)
class Hdf5Options:
    """Immutable HDF5 storage options."""

    compression: str | None = None
    chunks: bool | tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        """Validate optional compression and chunk metadata."""
        if self.compression is not None and (
            not isinstance(self.compression, str) or not self.compression
        ):
            raise errors.ContextError(
                "Hdf5Options: compression must be a non-empty string or None"
            )
        if isinstance(self.chunks, tuple) and (
            not self.chunks
            or any(
                isinstance(extent, bool)
                or not isinstance(extent, int)
                or extent <= 0
                for extent in self.chunks
            )
        ):
            raise errors.ContextError(
                "Hdf5Options: chunks must contain positive integers"
            )
        if not isinstance(self.chunks, (bool, tuple, type(None))):
            raise errors.ContextError(
                "Hdf5Options: chunks must be Boolean, a shape tuple, or None"
            )
        if self.compression is not None and self.chunks is False:
            raise errors.ContextError(
                "Hdf5Options: compression is incompatible with chunks=False"
            )


@dataclasses.dataclass(frozen=True, slots=True)
class MatOptions:
    """Immutable MATLAB storage options."""

    do_compression: bool = False

    def __post_init__(self) -> None:
        """Require an explicit Boolean compression policy."""
        if not isinstance(self.do_compression, bool):
            raise errors.ContextError(
                "MatOptions: do_compression must be Boolean"
            )


@dataclasses.dataclass(frozen=True, slots=True)
class ExtensionHandle:
    """Named immutable handle for an explicitly selected extension."""

    name: str
    value: object

    def __post_init__(self) -> None:
        """Validate a stable non-empty extension name."""
        if (
            not isinstance(self.name, str)
            or not self.name
            or self.name.strip() != self.name
        ):
            raise errors.ContextError(
                "extension: name must be non-empty with no surrounding space"
            )


@dataclasses.dataclass(frozen=True, slots=True, init=False)
class CreationContext:
    """Explicit namespace and creation policy for operations without inputs."""

    namespace: asc_typing.ArrayNamespace
    backend: asc_typing.BackendName
    dtype: object | None = None
    device: object | None = None
    precision: PrecisionPolicy
    extensions: tuple[ExtensionHandle, ...] = ()

    def __init__(  # pylint: disable=too-many-arguments
        self,
        namespace: asc_typing.ArrayNamespace,
        backend: asc_typing.BackendName,
        *,
        dtype: object | None = None,
        device: object | None = None,
        precision: PrecisionPolicy | None = None,
        extensions: tuple[ExtensionHandle, ...] = (),
    ) -> None:
        """Create a context with a dtype-sensitive default precision policy."""
        effective_precision = precision
        if effective_precision is None:
            effective_precision = (
                PrecisionPolicy.STRICT
                if dtype is not None
                else PrecisionPolicy.INHERIT
            )
        object.__setattr__(self, "namespace", namespace)
        object.__setattr__(self, "backend", backend)
        object.__setattr__(self, "dtype", dtype)
        object.__setattr__(self, "device", device)
        object.__setattr__(self, "precision", effective_precision)
        object.__setattr__(self, "extensions", extensions)
        self.__post_init__()

    def __post_init__(self) -> None:
        """Validate namespace identity and immutable extension metadata."""
        namespace_module.validate_namespace_revision(self.namespace)
        observed = namespace_module.identify_backend(self.namespace)
        if observed != self.backend:
            raise errors.ContextError(
                "context: backend does not match namespace; "
                f"expected {self.backend!r}, observed {observed!r}"
            )
        if not is_cpu_device(self.device):
            raise errors.ContextError("context: only CPU devices are supported")
        if not isinstance(self.precision, PrecisionPolicy):
            raise errors.ContextError(
                "context: precision must be a PrecisionPolicy"
            )
        if self.dtype is not None:
            if self.backend != "array_api_strict" and not (
                _dtype.is_supported_dtype(
                    self.backend,
                    self.dtype,
                    jax_x64_enabled=(
                        self.backend != "jax" or _dtype.active_jax_x64_enabled()
                    ),
                )
            ):
                raise errors.ContextError(
                    "context: dtype is outside the selected backend's "
                    "supported release surface"
                )
            try:
                supported_dtype = bool(
                    self.namespace.isdtype(
                        self.dtype,
                        (
                            "bool",
                            "signed integer",
                            "unsigned integer",
                            "real floating",
                            "complex floating",
                        ),
                    )
                )
            except Exception as exception:  # pylint: disable=broad-exception-caught
                raise errors.ContextError(
                    "context: dtype does not belong to the selected namespace"
                ) from exception
            if not supported_dtype:
                raise errors.ContextError(
                    "context: dtype is unsupported by the selected namespace"
                )
            if not _dtype.is_supported_dtype(
                self.backend,
                self.dtype,
                jax_x64_enabled=(
                    self.backend != "jax" or _dtype.active_jax_x64_enabled()
                ),
            ):
                raise errors.ContextError(
                    "context: dtype is outside the selected backend's "
                    "supported release surface"
                )
        extensions = _validate_extensions(self.extensions)
        names = tuple(extension.name for extension in extensions)
        if len(names) != len(set(names)):
            raise errors.ContextError("context: extension names must be unique")


__all__ = [
    "ArrayContext",
    "CopyPolicy",
    "CreationContext",
    "CsvOptions",
    "DataLoaderConfig",
    "ExtensionHandle",
    "Hdf5Options",
    "MatOptions",
    "NpyOptions",
    "NpzOptions",
    "PrecisionPolicy",
]
