# Copyright 2026 AI4SciComp contributors.
# SPDX-License-Identifier: Apache-2.0
"""Array API namespace discovery and validation."""

from __future__ import annotations

import contextlib
import contextvars
import dataclasses
import functools
import importlib
import sys
import typing

from asc import _array_api_compat, errors
from asc import typing as asc_typing
from asc.core import _dtype
from asc.core._array_api import MANDATORY_SYMBOLS
from asc.core.device import is_cpu_device

ARRAY_API_VERSION: typing.Final = "2024.12"
_SUPPORTED_BACKENDS: typing.Final = (
    "array_api_strict",
    "jax",
    "numpy",
    "torch",
)
_TRUSTED_JAX_CPU_TRACE: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "asc_trusted_jax_cpu_trace", default=False
)


class _GraphInspector(typing.Protocol):  # pylint: disable=too-few-public-methods
    """Backend hook for graph state outside Array API metadata."""

    def has_active_graph(self, value: object) -> bool:
        """Return whether an array carries active autodiff state."""
        ...  # pylint: disable=unnecessary-ellipsis


@dataclasses.dataclass(frozen=True, slots=True)
class NamespaceInfo:
    """Frozen inspection metadata for one selected namespace."""

    name: str
    backend: asc_typing.BackendName
    array_api_version: str
    capabilities: tuple[tuple[str, object], ...]


@contextlib.contextmanager
def trusted_jax_cpu_trace() -> typing.Iterator[None]:
    """Allow abstract JAX values only inside an asc CPU-pinned transform."""
    token = _TRUSTED_JAX_CPU_TRACE.set(True)
    try:
        yield
    finally:
        _TRUSTED_JAX_CPU_TRACE.reset(token)


def is_trusted_jax_cpu_trace() -> bool:
    """Return whether execution is inside an ASC CPU-pinned JAX transform."""
    return _TRUSTED_JAX_CPU_TRACE.get()


def has_active_graph(value: object, backend: asc_typing.BackendName) -> bool:
    """Return whether a native array carries active autodiff state."""
    if backend == "torch":
        module = importlib.import_module("asc.backends.torch")
        inspector = typing.cast(_GraphInspector, module)
        return inspector.has_active_graph(value)
    value_type = type(value)
    return (
        value_type.__module__.startswith("jax")
        and "Tracer" in value_type.__name__
    )


def _validate_dense_cpu_array(
    value: object,
    namespace: asc_typing.ArrayNamespace,
    backend: asc_typing.BackendName,
    position: int,
    *,
    allow_non_cpu: bool = False,
    allow_non_native_numpy_dtype: bool = False,
) -> None:
    """Reject arrays outside the release's dense CPU boundary."""
    try:
        is_numeric = namespace.isdtype(
            value.dtype,
            (
                "bool",
                "signed integer",
                "unsigned integer",
                "real floating",
                "complex floating",
            ),
        )
    except (AttributeError, TypeError, ValueError):
        is_numeric = False
    is_supported_dtype = _dtype.is_supported_dtype(
        backend,
        value.dtype,
        jax_x64_enabled=backend != "jax" or _dtype.active_jax_x64_enabled(),
        allow_non_native_endian=(
            backend == "numpy" and allow_non_native_numpy_dtype
        ),
    )
    is_jax_key = backend == "jax" and str(value.dtype).startswith("key<")
    is_masked = hasattr(value, "mask")
    is_nested = getattr(value, "is_nested", False) is True
    is_quantized = (
        backend == "torch" and getattr(value, "is_quantized", False) is True
    )
    is_numpy_matrix = backend == "numpy" and any(
        base.__module__.startswith("numpy") and base.__name__ == "matrix"
        for base in type(value).__mro__
    )
    layout = getattr(value, "layout", None)
    is_non_dense_torch = (
        backend == "torch"
        and layout is not None
        and str(layout) != "torch.strided"
    )
    try:
        device = _array_api_compat.compat.device(value)
    except (AttributeError, RuntimeError, TypeError, ValueError) as exception:
        raise errors.UnsupportedCapabilityError(
            "array_namespace: input "
            f"{position} is not a supported dense CPU array"
        ) from exception
    is_abstract_jax_value = (
        backend == "jax" and device is None and "Tracer" in type(value).__name__
    )
    trusted_abstract = is_abstract_jax_value and _TRUSTED_JAX_CPU_TRACE.get()
    is_non_cpu = (
        is_abstract_jax_value and not trusted_abstract
    ) or not is_cpu_device(device)
    if (
        not ((is_numeric and is_supported_dtype) or is_jax_key)
        or is_masked
        or is_nested
        or is_quantized
        or is_numpy_matrix
        or is_non_dense_torch
        or (is_non_cpu and not allow_non_cpu)
    ):
        raise errors.UnsupportedCapabilityError(
            "array_namespace: input "
            f"{position} is not a supported dense CPU array"
        )


def array_metadata(
    value: object,
    position: int,
    *,
    allow_non_cpu: bool = False,
    allow_non_native_numpy_dtype: bool = False,
) -> tuple[asc_typing.ArrayNamespace, asc_typing.BackendName]:
    """Return validated namespace metadata for one native array."""
    try:
        is_array = _array_api_compat.compat.is_array_api_obj(value)
    except Exception as exception:  # pylint: disable=broad-exception-caught
        raise errors.NamespaceError(
            f"array_namespace: input {position} exposes an invalid array "
            "protocol"
        ) from exception
    if not is_array:
        raise errors.NamespaceError(
            f"array_namespace: input {position} is not a supported native array"
        )
    try:
        selected = _array_api_compat.compat.array_namespace(
            value,
            api_version=ARRAY_API_VERSION,
        )
    except Exception as exception:  # pylint: disable=broad-exception-caught
        raise errors.NamespaceError(
            "array_namespace: input "
            f"{position} does not provide Array API {ARRAY_API_VERSION}"
        ) from exception
    namespace = typing.cast(asc_typing.ArrayNamespace, selected)
    validate_namespace_revision(namespace)
    backend = identify_backend(namespace)
    native_types: dict[asc_typing.BackendName, tuple[str, tuple[str, ...]]] = {
        "array_api_strict": ("array_api_strict._array_object", ("Array",)),
        "jax": ("jax", ("Array",)),
        "numpy": ("numpy", ("ndarray", "generic")),
        "torch": ("torch", ("Tensor",)),
    }
    module_name, type_names = native_types[backend]
    module = sys.modules.get(module_name)
    loaded_types = tuple(
        candidate
        for type_name in type_names
        if isinstance(
            candidate := (
                None if module is None else getattr(module, type_name, None)
            ),
            type,
        )
    )
    is_native = bool(loaded_types) and isinstance(value, loaded_types)
    if backend == "jax" and not is_native:
        jax_module = sys.modules.get("jax")
        core = None if jax_module is None else getattr(jax_module, "core", None)
        tracer_type = None if core is None else getattr(core, "Tracer", None)
        is_native = isinstance(tracer_type, type) and isinstance(
            value, tracer_type
        )
    if not is_native:
        raise errors.NamespaceError(
            f"array_namespace: input {position} is not a supported native array"
        )
    _validate_dense_cpu_array(
        value,
        namespace,
        backend,
        position,
        allow_non_cpu=allow_non_cpu,
        allow_non_native_numpy_dtype=allow_non_native_numpy_dtype,
    )
    return namespace, backend


def identify_backend(namespace: object) -> asc_typing.BackendName:
    """Return the stable backend identity for a compatible namespace.

    Args:
        namespace: Native or compatibility Array API namespace.

    Returns:
        The corresponding public backend name.

    Raises:
        NamespaceError: If the namespace is unsupported.
    """
    predicates: tuple[
        tuple[asc_typing.BackendName, typing.Callable[[object], bool]], ...
    ] = (
        (
            "array_api_strict",
            _array_api_compat.compat.is_array_api_strict_namespace,
        ),
        (
            "jax",
            _array_api_compat.compat.is_jax_namespace,
        ),
        (
            "numpy",
            _array_api_compat.compat.is_numpy_namespace,
        ),
        (
            "torch",
            _array_api_compat.compat.is_torch_namespace,
        ),
    )
    for backend, predicate in predicates:
        try:
            matches = predicate(namespace)
        except (AttributeError, TypeError):
            matches = False
        if matches:
            return backend
    observed = getattr(namespace, "__name__", type(namespace).__name__)
    raise errors.NamespaceError(
        "namespace: unsupported namespace "
        f"{observed!r}; expected one of {_SUPPORTED_BACKENDS}"
    )


@functools.cache
def validate_namespace_revision(
    namespace: asc_typing.ArrayNamespace,
) -> None:
    """Validate that a namespace implements the normative API revision.

    Args:
        namespace: Namespace to validate.

    Raises:
        NamespaceError: If the namespace does not expose a sufficient revision.
    """
    revision = getattr(namespace, "__array_api_version__", None)
    if revision != ARRAY_API_VERSION:
        raise errors.NamespaceError(
            "namespace: Array API revision "
            f"{revision!r} does not satisfy {ARRAY_API_VERSION!r}"
        )
    missing = tuple(
        sorted(
            name for name in MANDATORY_SYMBOLS if not hasattr(namespace, name)
        )
    )
    if missing:
        raise errors.NamespaceError(
            "namespace: Array API 2024.12 surface is incomplete; missing "
            f"{missing!r}"
        )


def namespace_info(namespace: asc_typing.ArrayNamespace) -> NamespaceInfo:
    """Inspect a namespace without changing backend state."""
    validate_namespace_revision(namespace)
    backend = identify_backend(namespace)
    raw_capabilities: dict[str, object] = {}
    inspection_factory = getattr(namespace, "__array_namespace_info__", None)
    if inspection_factory is not None:
        inspection = inspection_factory()
        capability_function = getattr(inspection, "capabilities", None)
        if capability_function is not None:
            raw_capabilities = dict(capability_function())
    return NamespaceInfo(
        name=getattr(namespace, "__name__", type(namespace).__name__),
        backend=backend,
        array_api_version=ARRAY_API_VERSION,
        capabilities=tuple(sorted(raw_capabilities.items())),
    )


def array_namespace(
    *arrays: object,
    api_version: str = ARRAY_API_VERSION,
) -> asc_typing.ArrayNamespace:
    """Select one 2024.12-compatible namespace from native arrays.

    Python scalars cannot select a namespace. All arrays must belong to the
    same supported backend.

    Args:
        *arrays: One or more native backend arrays.
        api_version: Required revision. Only 2024.12 is supported.

    Returns:
        A compatible Array API namespace.

    Raises:
        MixedBackendError: If the arrays come from multiple backends.
        NamespaceError: If inputs are absent, unsupported, or not arrays.
    """
    if api_version != ARRAY_API_VERSION:
        raise errors.NamespaceError(
            "array_namespace: only Array API revision "
            f"{ARRAY_API_VERSION!r} is supported"
        )
    if not arrays:
        raise errors.NamespaceError(
            "array_namespace: at least one native array is required"
        )

    namespaces: list[asc_typing.ArrayNamespace] = []
    backends: list[asc_typing.BackendName] = []
    selecting_arrays: list[object] = []
    for position, value in enumerate(arrays):
        if value is None or type(value) in {bool, int, float, complex}:
            continue
        namespace, backend = array_metadata(value, position)
        selecting_arrays.append(value)
        namespaces.append(namespace)
        backends.append(backend)

    if not selecting_arrays:
        raise errors.NamespaceError(
            "array_namespace: at least one native array is required; "
            "Python scalars and None do not select a backend"
        )

    observed = tuple(dict.fromkeys(backends))
    if len(observed) != 1:
        raise errors.MixedBackendError(
            "array_namespace: mixed backends are not permitted; "
            f"observed {observed!r}"
        )

    try:
        selected = _array_api_compat.compat.array_namespace(
            *selecting_arrays,
            api_version=api_version,
        )
    except Exception as exception:  # pylint: disable=broad-exception-caught
        raise errors.NamespaceError(
            "array_namespace: compatible namespace selection failed"
        ) from exception
    namespace = typing.cast(asc_typing.ArrayNamespace, selected)
    validate_namespace_revision(namespace)
    backend = identify_backend(namespace)
    if backend in {"numpy", "torch", "jax"}:
        from asc.extensions import _dispatch

        return typing.cast(
            asc_typing.ArrayNamespace,
            _dispatch.load_backend(backend).namespace(),
        )
    return namespace
