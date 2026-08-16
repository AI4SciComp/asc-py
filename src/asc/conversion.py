# Copyright 2026 AI4SciComp contributors.
# SPDX-License-Identifier: Apache-2.0
"""Explicit dense CPU conversion boundaries."""

# Conversion and one-shot DLPack ownership policy intentionally remain in one
# module so validation cannot be separated accidentally from consumption.
# pylint: disable=too-many-lines

from __future__ import annotations

import dataclasses
import importlib
import inspect
import typing
import warnings

import numpy

from asc import _array_api_compat, config, errors
from asc import typing as asc_typing
from asc.core import _dtype
from asc.core import namespace as namespace_module
from asc.core.device import is_cpu_device

_CONVERSION_FAILURES = (
    BufferError,
    RuntimeError,
    TypeError,
    ValueError,
    Warning,
)
_DTYPE_NAMES = (
    "complex128",
    "complex64",
    "bfloat16",
    "float64",
    "float32",
    "float16",
    "uint64",
    "uint32",
    "uint16",
    "uint8",
    "int64",
    "int32",
    "int16",
    "int8",
    "bool",
)


def _dtype_name(dtype: object) -> str:
    """Normalize equivalent backend-native dtype spellings."""
    try:
        return numpy.dtype(dtype).name
    except TypeError:
        pass
    observed = str(dtype).lower()
    for name in _DTYPE_NAMES:
        if name in observed:
            return name
    return observed


class _ConversionNamespace(asc_typing.ArrayNamespace, typing.Protocol):
    """Array API functions required only at explicit conversion boundaries."""

    def asarray(
        self,
        value: object,
        *,
        dtype: object | None = None,
        device: object | None = None,
        copy: bool | None = None,
    ) -> asc_typing.NativeArray:
        """Create or view an array with an explicit copy policy."""
        ...  # pylint: disable=unnecessary-ellipsis

    def from_dlpack(
        self,
        value: object,
        *,
        device: object | None = None,
        copy: bool | None = None,
    ) -> asc_typing.NativeArray:
        """Import an interoperable array through DLPack."""
        ...  # pylint: disable=unnecessary-ellipsis


class _TorchConversionAdapter(typing.Protocol):
    """Internal PyTorch adapter surface for graph-preserving conversion."""

    def asarray_preserving_graph(
        self,
        value: object,
        *,
        dtype: object | None = None,
        device: object | None = None,
        copy: bool | None = None,
    ) -> asc_typing.NativeArray:
        """Create a tensor while preserving its reverse-mode state."""
        ...  # pylint: disable=unnecessary-ellipsis


# A single hook is the complete protocol surface.
# pylint: disable-next=too-few-public-methods
class _VersionedDLPackExporter(typing.Protocol):
    """Array supporting the version-negotiated DLPack protocol."""

    def __dlpack__(
        self, *, max_version: tuple[int, int] | None = None
    ) -> object:
        """Export a DLPack capsule with explicit version negotiation."""
        ...  # pylint: disable=unnecessary-ellipsis


@dataclasses.dataclass(slots=True)
class _DLPackCapsule:
    """One-consumer DLPack producer around an exported native capsule."""

    capsule: object
    device: tuple[int, int]
    dtype: object | None = None
    pointer: int | None = None
    consumed: bool = False

    def __dlpack_device__(self) -> tuple[int, int]:
        """Return the frozen DLPack device tuple."""
        return self.device

    def __dlpack__(
        self,
        stream: object | None = None,
        *,
        max_version: tuple[int, int] | None = None,
        dl_device: tuple[int, int] | None = None,
        copy: bool | None = None,
    ) -> object:
        """Transfer ownership of the wrapped capsule exactly once."""
        del stream
        if self.consumed:
            raise BufferError("DLPack capsule has already been consumed")
        if copy is not None and type(copy) is not bool:
            raise TypeError("DLPack copy must be a Boolean or None")
        if max_version is not None and (
            not isinstance(max_version, tuple)
            or len(max_version) != 2
            or any(
                isinstance(entry, bool)
                or not isinstance(entry, int)
                or entry < 0
                for entry in max_version
            )
        ):
            raise TypeError(
                "DLPack max_version must be a non-negative integer pair or None"
            )
        if dl_device is not None and (
            not isinstance(dl_device, tuple)
            or len(dl_device) != 2
            or any(
                isinstance(entry, bool) or not isinstance(entry, int)
                for entry in dl_device
            )
        ):
            raise TypeError("DLPack dl_device must be an integer pair or None")
        if copy is True:
            raise BufferError(
                "DLPack capsule cannot copy at export; copy after import"
            )
        if dl_device is not None and dl_device != self.device:
            raise BufferError("DLPack capsule cannot transfer devices")
        self.consumed = True
        return self.capsule


def _is_cpu(value: object) -> bool:
    return is_cpu_device(_array_api_compat.compat.device(value))


def _data_pointer(value: object) -> int | None:
    """Return a native CPU data pointer when the backend exposes one."""
    if isinstance(value, _DLPackCapsule):
        return value.pointer
    try:
        interface = getattr(value, "__array_interface__", None)
        if isinstance(interface, dict):
            data = interface.get("data")
            if isinstance(data, tuple) and data and isinstance(data[0], int):
                return data[0] or None
        for name in ("data_ptr", "unsafe_buffer_pointer"):
            pointer = getattr(value, name, None)
            if callable(pointer):
                result = pointer()
                if isinstance(result, int):
                    return result or None
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return None
    return None


def _detach_known_backend(
    source: asc_typing.NativeArray,
    backend: asc_typing.BackendName,
) -> asc_typing.NativeArray:
    """Detach an already-identified backend without repeating CPU validation."""
    if backend in {"numpy", "array_api_strict"}:
        return source
    if backend == "torch":
        return typing.cast(asc_typing.NativeArray, source.detach())
    module = importlib.import_module("jax.lax")
    return typing.cast(asc_typing.NativeArray, module.stop_gradient(source))


def _transfer_to_cpu(
    source: asc_typing.NativeArray,
    backend: asc_typing.BackendName,
) -> asc_typing.NativeArray:
    """Perform an authorized backend-native transfer to host CPU."""
    if backend == "array_api_strict":
        raise errors.ConversionError(
            "to_numpy: array-api-strict has no device-transfer extension"
        )
    from asc.extensions import _dispatch

    try:
        transferred = _dispatch.load_backend(backend).to_cpu(source)
        result = typing.cast(asc_typing.NativeArray, transferred)
        namespace_module.array_namespace(result)
        return result
    except errors.AscError:
        raise
    except (RuntimeError, TypeError, ValueError) as exception:
        raise errors.ConversionError(
            "to_numpy: backend failed the authorized CPU transfer"
        ) from exception


def _same_backend_asarray(
    target: _ConversionNamespace,
    source: asc_typing.NativeArray,
    destination: config.CreationContext,
    copy: bool | None,
) -> asc_typing.NativeArray:
    """Create a same-backend result while preserving backend graph state."""
    if destination.backend == "torch":
        from asc.extensions import _dispatch

        torch_adapter = typing.cast(
            _TorchConversionAdapter, _dispatch.load_backend("torch")
        )
        return torch_adapter.asarray_preserving_graph(
            source,
            dtype=destination.dtype,
            device=destination.device,
            copy=copy,
        )
    return target.asarray(
        source,
        dtype=destination.dtype,
        device=destination.device,
        copy=copy,
    )


def _shares_storage(
    source: asc_typing.NativeArray,
    result: asc_typing.NativeArray,
    backend: asc_typing.BackendName,
) -> bool:
    """Return whether a same-backend result aliases source storage."""
    if result is source:
        return True
    try:
        if backend == "numpy":
            return bool(numpy.shares_memory(source, result))
        if backend == "torch":
            source_storage = getattr(source, "untyped_storage", None)
            result_storage = getattr(result, "untyped_storage", None)
            if source_storage is None or result_storage is None:
                return False
            source_pointer = source_storage().data_ptr()
            result_pointer = result_storage().data_ptr()
            if source_pointer == 0 or result_pointer == 0:
                return False
            return bool(source_pointer == result_pointer)
    except (RuntimeError, TypeError, ValueError):
        return False
    return False


def _from_dlpack(
    target: _ConversionNamespace,
    source: asc_typing.NativeArray,
    *,
    device: object | None = None,
    copy: bool | None = None,
    allow_legacy_no_copy: bool = False,
) -> asc_typing.NativeArray:
    """Import DLPack, accommodating exporters without version support.

    NumPy versions before 2.1 cannot honor a requested DLPack protocol
    version. Some consumers request one automatically, so retry through a
    raw capsule while preserving the same zero-copy interchange protocol.
    """
    from asc.backends import _namespace

    try:
        with _namespace.trusted_dlpack_conversion():
            return target.from_dlpack(source, device=device, copy=copy)
    except TypeError as exception:
        if copy is False and not allow_legacy_no_copy:
            raise errors.ConversionError(
                "from_dlpack: destination cannot guarantee no-copy import"
            ) from exception
        with _namespace.trusted_dlpack_conversion():
            imported = target.from_dlpack(source)
        if copy is True:
            return target.asarray(imported, device=device, copy=True)
        return imported
    except NotImplementedError as exception:
        if copy is False:
            raise errors.ConversionError(
                "from_dlpack: destination cannot guarantee no-copy import"
            ) from exception
        exporter = typing.cast(_VersionedDLPackExporter, source)
        capsule = exporter.__dlpack__(max_version=None)
        wrapped = _DLPackCapsule(
            capsule,
            (1, 0),
            getattr(source, "dtype", None),
            _data_pointer(source),
        )
        with _namespace.trusted_dlpack_conversion():
            imported = target.from_dlpack(wrapped)
        if copy is True:
            return target.asarray(imported, device=device, copy=True)
        return imported


def _resolve_dlpack_view(
    source: asc_typing.NativeArray,
    backend: asc_typing.BackendName,
) -> asc_typing.NativeArray:
    """Materialize Torch lazy view bits before logical-value export."""
    if backend != "torch":
        return source
    value = source
    for predicate_name, resolver_name in (
        ("is_conj", "resolve_conj"),
        ("is_neg", "resolve_neg"),
    ):
        predicate = getattr(value, predicate_name, None)
        if callable(predicate) and predicate():
            resolver = getattr(value, resolver_name, None)
            if not callable(resolver):
                raise errors.ConversionError(
                    "conversion: Torch lazy view cannot be resolved safely"
                )
            value = typing.cast(asc_typing.NativeArray, resolver())
    return value


def _prepare_dlpack_export(
    source: asc_typing.NativeArray,
    backend: asc_typing.BackendName,
) -> asc_typing.NativeArray:
    """Materialize backend views that a DLPack consumer cannot import safely."""
    value = _resolve_dlpack_view(source, backend)
    if backend == "numpy" and _numpy_dlpack_requires_normalization(value):
        native_dtype = numpy.dtype(value.dtype).newbyteorder("=")
        value = typing.cast(
            asc_typing.NativeArray,
            numpy.array(value, dtype=native_dtype, copy=True, order="C"),
        )
    elif backend == "torch":
        contiguous = getattr(value, "contiguous", None)
        if callable(contiguous):
            value = typing.cast(asc_typing.NativeArray, contiguous())
    return value


def _prepare_dlpack_capsule_export(
    source: asc_typing.NativeArray,
    backend: asc_typing.BackendName,
) -> asc_typing.NativeArray:
    """Materialize layouts that cannot be represented safely by consumers."""
    value = _resolve_dlpack_view(source, backend)
    if backend == "numpy" and _numpy_dlpack_requires_normalization(value):
        dtype = numpy.dtype(value.dtype)
        value = typing.cast(
            asc_typing.NativeArray,
            numpy.array(
                value, dtype=dtype.newbyteorder("="), copy=True, order="C"
            ),
        )
    return value


def copy_opaque_cpu_dlpack_producer(producer: object) -> numpy.ndarray:
    """Copy an opaque CPU producer through NumPy before unsafe consumption."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            if isinstance(producer, (numpy.ndarray, numpy.generic)):
                imported = numpy.asarray(producer)
            else:
                try:
                    imported = numpy.from_dlpack(producer)
                except NotImplementedError:
                    exporter = typing.cast(_VersionedDLPackExporter, producer)
                    capsule = exporter.__dlpack__(max_version=None)
                    imported = numpy.from_dlpack(
                        _DLPackCapsule(
                            capsule,
                            (1, 0),
                            getattr(producer, "dtype", None),
                            _data_pointer(producer),
                        )
                    )
            return numpy.array(imported, copy=True, order="C")
    except _CONVERSION_FAILURES as exception:
        raise errors.ConversionError(
            "from_dlpack: opaque CPU producer could not be copied safely "
            "before destination import"
        ) from exception


def _destination_supports_source_dtype(
    destination: config.CreationContext, source_dtype: object
) -> bool:
    """Return whether DLPack can preserve a dtype in the destination."""
    source_name = _dtype_name(source_dtype)
    destination_dtype = getattr(destination.namespace, source_name, None)
    return destination_dtype is not None and _dtype.is_supported_dtype(
        destination.backend,
        destination_dtype,
        jax_x64_enabled=(
            destination.backend != "jax" or _dtype.active_jax_x64_enabled()
        ),
    )


def _interchange_source(
    source: asc_typing.NativeArray,
    source_namespace: asc_typing.ArrayNamespace,
    source_backend: asc_typing.BackendName,
    destination: config.CreationContext,
    operation: str,
    *,
    allow_copy: bool,
) -> asc_typing.NativeArray:
    """Cast before DLPack when the destination cannot preserve source dtype."""
    if _destination_supports_source_dtype(destination, source.dtype):
        return source
    if destination.dtype is None:
        raise errors.ConversionError(
            f"{operation}: source dtype is unsupported by the destination "
            "and would be narrowed or rejected"
        )
    if not allow_copy:
        raise errors.ConversionError(
            f"{operation}: requested dtype change requires a copy"
        )
    requested_name = _dtype_name(destination.dtype)
    source_dtype = getattr(source_namespace, requested_name, None)
    if source_dtype is None or not _dtype.is_supported_dtype(
        source_backend,
        source_dtype,
        jax_x64_enabled=(
            source_backend != "jax" or _dtype.active_jax_x64_enabled()
        ),
    ):
        raise errors.ConversionError(
            f"{operation}: source backend cannot safely cast to the requested "
            "interchange dtype"
        )
    try:
        astype = typing.cast(
            typing.Callable[..., asc_typing.NativeArray],
            source_namespace.astype,
        )
        return astype(source, source_dtype, copy=True)
    except (AttributeError, RuntimeError, TypeError, ValueError) as exception:
        raise errors.ConversionError(
            f"{operation}: source backend rejected the required pre-import cast"
        ) from exception


def _numpy_dlpack_requires_normalization(value: object) -> bool:
    """Return whether a NumPy DLPack producer requires a safe copy."""
    strides = getattr(value, "strides", None)
    flags = getattr(value, "flags", None)
    is_read_only = (
        flags is not None and getattr(flags, "writeable", True) is False
    )
    is_c_contiguous = (
        flags is not None and getattr(flags, "c_contiguous", False) is True
    )
    return (
        not is_c_contiguous
        or (strides is not None and any(stride < 0 for stride in strides))
        or not numpy.dtype(value.dtype).isnative
        or is_read_only
    )


def _accepts_copy_keyword(function: object) -> bool:
    """Return whether a DLPack importer declares a ``copy`` keyword."""
    try:
        parameters = inspect.signature(function).parameters
    except Exception:  # pylint: disable=broad-exception-caught
        return False
    copy_parameter = parameters.get("copy")
    return (
        copy_parameter is not None
        and copy_parameter.kind
        in {
            inspect.Parameter.KEYWORD_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        }
    ) or any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )


def _is_builtin_numpy_dlpack_importer(function: object) -> bool:
    """Return whether a wrapper delegates to NumPy's trusted importer."""
    try:
        return inspect.unwrap(function) is numpy.from_dlpack
    except Exception:  # pylint: disable=broad-exception-caught
        return False


def _require_cpu_dlpack_device(producer: object) -> None:
    """Validate DLPack CPU provenance without consuming the producer."""
    try:
        device_function = getattr(producer, "__dlpack_device__", None)
    except Exception as exception:  # pylint: disable=broad-exception-caught
        raise errors.ConversionError(
            "from_dlpack: producer device provenance is invalid"
        ) from exception
    if not callable(device_function):
        raise errors.ConversionError(
            "from_dlpack: producer lacks DLPack device provenance"
        )
    try:
        device = device_function()
    except Exception as exception:  # pylint: disable=broad-exception-caught
        raise errors.ConversionError(
            "from_dlpack: producer device provenance is invalid"
        ) from exception
    if (
        not isinstance(device, tuple)
        or len(device) != 2
        or any(
            isinstance(entry, bool) or not isinstance(entry, int)
            for entry in device
        )
        or device != (1, 0)
    ):
        raise errors.ConversionError(
            "from_dlpack: only CPU producers are supported"
        )


def _validate_result(
    result: asc_typing.NativeArray,
    source: asc_typing.NativeArray,
    destination: config.CreationContext,
) -> None:
    result_backend = namespace_module.identify_backend(
        namespace_module.array_namespace(result)
    )
    if result_backend != destination.backend:
        raise errors.ConversionError(
            "convert_array: destination returned another backend"
        )
    if result.shape != source.shape:
        raise errors.ConversionError("convert_array: shape was not preserved")
    requested_dtype = (
        source.dtype if destination.dtype is None else destination.dtype
    )
    source_name = _dtype_name(requested_dtype)
    result_name = _dtype_name(result.dtype)
    if result_name != source_name:
        raise errors.ConversionError(
            "convert_array: source or requested dtype was not preserved"
        )
    if not _is_cpu(result):
        raise errors.ConversionError("convert_array: result is not on CPU")


def _convert_array_context(
    source: asc_typing.NativeArray,
    *,
    destination: config.CreationContext,
    copy: config.CopyPolicy,
) -> asc_typing.NativeArray:
    """Convert or copy a dense CPU array under an explicit ownership policy.

    Args:
        source: Native dense CPU source array.
        destination: Explicit destination namespace, dtype, and device.
        copy: Required ownership policy. Cross-backend conversion requires
            :attr:`CopyPolicy.ALWAYS`.

    Returns:
        A native destination array with preserved shape and requested dtype.

    Raises:
        ConversionError: If device, graph, copy, dtype, or ownership rules fail.
        NamespaceError: If ``source`` is not a supported native array.
    """
    source_namespace, source_backend = namespace_module.array_metadata(
        source,
        0,
        allow_non_native_numpy_dtype=True,
    )
    if not _is_cpu(source):
        raise errors.ConversionError(
            "convert_array: only dense CPU sources are supported"
        )
    if (
        destination.device is not None
        and "cpu" not in str(destination.device).lower()
    ):
        raise errors.ConversionError(
            "convert_array: only CPU destinations are supported"
        )

    target = typing.cast(_ConversionNamespace, destination.namespace)
    same_backend = source_backend == destination.backend
    if not same_backend and copy is not config.CopyPolicy.ALWAYS:
        raise errors.ConversionError(
            "convert_array: cross-backend conversion requires CopyPolicy.ALWAYS"
        )
    if not same_backend and namespace_module.has_active_graph(
        source, source_backend
    ):
        raise errors.ConversionError(
            "convert_array: active autodiff graphs must be detached "
            "explicitly by the caller"
        )
    requires_same_backend_normalization = (
        same_backend
        and source_backend == "numpy"
        and not numpy.dtype(source.dtype).isnative
    )
    if requires_same_backend_normalization and copy is config.CopyPolicy.NEVER:
        raise errors.ConversionError(
            "convert_array: NumPy dtype normalization requires a copy"
        )
    if same_backend and copy is config.CopyPolicy.NEVER:
        requested_dtype = (
            source.dtype if destination.dtype is None else destination.dtype
        )
        source_device = _array_api_compat.compat.device(source)
        if requested_dtype != source.dtype or (
            destination.device is not None
            and destination.device != source_device
            and str(destination.device) != str(source_device)
        ):
            raise errors.ConversionError(
                "convert_array: dtype or device change requires a copy"
            )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            if same_backend:
                conversion_source = source
                if requires_same_backend_normalization:
                    conversion_source = _prepare_dlpack_export(
                        source, source_backend
                    )
                copy_value = {
                    config.CopyPolicy.ALWAYS: True,
                    config.CopyPolicy.IF_NEEDED: None,
                    config.CopyPolicy.NEVER: False,
                }[copy]
                result = _same_backend_asarray(
                    target,
                    conversion_source,
                    destination,
                    copy_value,
                )
            else:
                interchange_source = _interchange_source(
                    source,
                    source_namespace,
                    source_backend,
                    destination,
                    "convert_array",
                    allow_copy=True,
                )
                exported_source = _prepare_dlpack_export(
                    interchange_source, source_backend
                )
                preparation_copied = exported_source is not source
                imported = _from_dlpack(
                    target,
                    exported_source,
                    device=destination.device,
                    copy=None,
                )
                requires_final_copy = not preparation_copied or (
                    destination.dtype is not None
                    and _dtype_name(destination.dtype)
                    != _dtype_name(exported_source.dtype)
                )
                result = target.asarray(
                    imported,
                    dtype=destination.dtype,
                    device=destination.device,
                    copy=requires_final_copy,
                )
    except _CONVERSION_FAILURES as exception:
        raise errors.ConversionError(
            "convert_array: backend conversion failed under the requested "
            "dtype, device, and copy policy"
        ) from exception

    if (
        same_backend
        and copy is config.CopyPolicy.NEVER
        and not _shares_storage(source, result, source_backend)
    ):
        raise errors.ConversionError(
            "convert_array: backend could not prove aliasing for "
            "CopyPolicy.NEVER"
        )
    _validate_result(result, source, destination)
    return result


def _destination_context(
    destination: object,
    *,
    dtype: object | None,
    device: object | None,
) -> config.CreationContext:
    """Normalize legacy and comprehensive destination descriptions."""
    if isinstance(destination, config.CreationContext):
        name = destination.backend
        effective_dtype = destination.dtype if dtype is None else dtype
        effective_device = destination.device if device is None else device
        precision = destination.precision
        extensions = destination.extensions
    elif isinstance(destination, config.ArrayContext):
        name = destination.backend
        effective_dtype = destination.dtype if dtype is None else dtype
        effective_device = destination.device if device is None else device
        precision = destination.precision
        extensions = ()
    elif isinstance(destination, str):
        name = typing.cast(asc_typing.BackendName, destination)
        effective_dtype = dtype
        effective_device = device
        precision = None
        extensions = ()
    else:
        try:
            name = typing.cast(asc_typing.BackendName, destination.name)
            _destination_namespace = destination.xp
            bound_dtype = destination.dtype
            bound_device = destination.device
        except Exception as exception:  # pylint: disable=broad-exception-caught
            raise errors.ConversionError(
                "convert_array: destination must be a backend name, Backend, "
                "ArrayContext, or CreationContext"
            ) from exception
        effective_dtype = bound_dtype if dtype is None else dtype
        effective_device = bound_device if device is None else device
        precision = None
        extensions = ()
    from asc.core.backend import backend as select_backend

    if name == "array_api_strict":
        raise errors.ConversionError(
            "convert_array: array-api-strict is a conformance oracle, not a "
            "runtime conversion destination"
        )
    try:
        selected = select_backend(
            typing.cast(typing.Literal["numpy", "torch", "jax"], name),
            device=effective_device,
            dtype=effective_dtype,
        )
    except errors.AscError as exception:
        raise errors.ConversionError(
            f"convert_array: invalid destination context: {exception}"
        ) from exception
    return config.CreationContext(
        typing.cast(asc_typing.ArrayNamespace, selected.xp),
        selected.name,
        dtype=effective_dtype,
        device=selected.device,
        precision=precision,
        extensions=extensions,
    )


def _copy_policy(value: config.CopyPolicy | bool | None) -> config.CopyPolicy:
    if isinstance(value, config.CopyPolicy):
        return value
    if value is True:
        return config.CopyPolicy.ALWAYS
    if value is False:
        return config.CopyPolicy.NEVER
    if value is None:
        return config.CopyPolicy.IF_NEEDED
    raise errors.ConversionError(
        "convert_array: copy must be True, False, None, or CopyPolicy"
    )


def convert_array(
    source: asc_typing.NativeArray,
    destination: object,
    *,
    dtype: object | None = None,
    device: object | None = None,
    copy: config.CopyPolicy | bool | None = True,
) -> asc_typing.NativeArray:
    """Explicitly convert an array under dtype, device, and copy policies."""
    context = _destination_context(destination, dtype=dtype, device=device)
    return _convert_array_context(
        source, destination=context, copy=_copy_policy(copy)
    )


def detach(source: asc_typing.NativeArray) -> asc_typing.NativeArray:
    """Explicitly discard automatic-differentiation history."""
    backend = namespace_module.identify_backend(
        namespace_module.array_namespace(source)
    )
    return _detach_known_backend(source, backend)


def stop_gradient(source: asc_typing.NativeArray) -> asc_typing.NativeArray:
    """Alias for the explicit graph-boundary operation :func:`detach`."""
    return detach(source)


def copy_array(
    source: asc_typing.NativeArray,
    *,
    copy: config.CopyPolicy | bool | None = True,
) -> asc_typing.NativeArray:
    """Copy or alias an array while preserving backend, dtype, and device."""
    backend = namespace_module.identify_backend(
        namespace_module.array_namespace(source)
    )
    return convert_array(
        source,
        backend,
        dtype=source.dtype,
        device=_array_api_compat.compat.device(source),
        copy=copy,
    )


def to_device(
    source: asc_typing.NativeArray,
    device: object,
    *,
    copy: config.CopyPolicy | bool | None = None,
) -> asc_typing.NativeArray:
    """Explicitly transfer within one backend and never cross backends."""
    backend = namespace_module.identify_backend(
        namespace_module.array_namespace(source)
    )
    return convert_array(
        source,
        backend,
        dtype=source.dtype,
        device=device,
        copy=copy,
    )


def to_numpy(
    source: asc_typing.NativeArray,
    *,
    allow_detach: bool = False,
    allow_transfer: bool = False,
    copy: bool = True,
) -> asc_typing.NativeArray:
    """Cross the explicit NumPy host boundary under graph/transfer policy."""
    if not isinstance(allow_detach, bool) or not isinstance(
        allow_transfer, bool
    ):
        raise errors.ConversionError(
            "to_numpy: allow_detach and allow_transfer must be Booleans"
        )
    if not isinstance(copy, bool):
        raise errors.ConversionError("to_numpy: copy must be Boolean")
    _source_namespace, backend = namespace_module.array_metadata(
        source,
        0,
        allow_non_cpu=True,
        allow_non_native_numpy_dtype=True,
    )
    value = source
    if namespace_module.has_active_graph(source, backend):
        if not allow_detach:
            raise errors.ConversionError(
                "to_numpy: active autodiff graph requires allow_detach=True"
            )
        value = _detach_known_backend(source, backend)
    if not _is_cpu(value):
        if not allow_transfer:
            raise errors.ConversionError(
                "to_numpy: non-CPU source requires allow_transfer=True"
            )
        value = _transfer_to_cpu(value, backend)
    return convert_array(
        value,
        "numpy",
        dtype=None,
        device="cpu",
        copy=copy,
    )


def from_numpy(
    source: asc_typing.NativeArray,
    destination: object = "numpy",
    *,
    dtype: object | None = None,
    device: object | None = "cpu",
    copy: config.CopyPolicy | bool | None = True,
) -> asc_typing.NativeArray:
    """Convert an actual NumPy array to an explicit destination backend."""
    try:
        _source_namespace, backend = namespace_module.array_metadata(
            source,
            0,
            allow_non_native_numpy_dtype=True,
        )
    except errors.AscError as exception:
        raise errors.ConversionError(
            "from_numpy: source must be a native NumPy array"
        ) from exception
    if backend != "numpy":
        raise errors.ConversionError(
            "from_numpy: source must be a native NumPy array"
        )
    return convert_array(
        source,
        destination,
        dtype=dtype,
        device=device,
        copy=copy,
    )


def to_dlpack(
    source: asc_typing.NativeArray,
    *,
    allow_detach: bool = False,
) -> object:
    """Export a one-consumer DLPack capsule with explicit graph policy."""
    if not isinstance(allow_detach, bool):
        raise errors.ConversionError("to_dlpack: allow_detach must be Boolean")
    try:
        _source_namespace, backend = namespace_module.array_metadata(
            source,
            0,
            allow_non_native_numpy_dtype=True,
        )
    except errors.AscError as exception:
        raise errors.ConversionError(
            "to_dlpack: source must be a supported native array"
        ) from exception
    value = source
    if namespace_module.has_active_graph(source, backend):
        if not allow_detach:
            raise errors.ConversionError(
                "to_dlpack: active graph requires allow_detach=True"
            )
        value = detach(source)
    try:
        value = _prepare_dlpack_capsule_export(value, backend)
        exporter = getattr(value, "__dlpack__", None)
        if exporter is None:
            raise errors.CapabilityNotSupportedError(
                "to_dlpack: source does not implement the DLPack protocol"
            )
        device_function = getattr(value, "__dlpack_device__", None)
        device = (1, 0) if device_function is None else device_function()
        capsule = exporter()
    except errors.AscError:
        raise
    except (BufferError, RuntimeError, TypeError, ValueError) as exception:
        raise errors.ConversionError(
            "to_dlpack: backend rejected capsule export"
        ) from exception
    return _DLPackCapsule(
        capsule,
        typing.cast(tuple[int, int], device),
        getattr(value, "dtype", None),
        _data_pointer(value),
    )


def from_dlpack(  # pylint: disable=too-many-statements
    capsule: object,
    destination: object,
    *,
    dtype: object | None = None,
    device: object | None = "cpu",
    copy: config.CopyPolicy | bool | None = None,
) -> asc_typing.NativeArray:
    """Import a DLPack producer into an explicit backend."""
    context = _destination_context(destination, dtype=dtype, device=device)
    if not is_cpu_device(context.device):
        raise errors.ConversionError(
            "from_dlpack: only CPU destinations are supported"
        )
    target = typing.cast(_ConversionNamespace, context.namespace)
    policy = _copy_policy(copy)
    try:
        exporter = getattr(capsule, "__dlpack__", None)
    except Exception as exception:  # pylint: disable=broad-exception-caught
        raise errors.ConversionError(
            "from_dlpack: producer protocol metadata is invalid"
        ) from exception
    if not callable(exporter):
        raise errors.ConversionError(
            "from_dlpack: raw capsules lack device provenance; pass a DLPack "
            "protocol producer or the result of asc.to_dlpack"
        )
    _require_cpu_dlpack_device(capsule)
    producer = capsule
    preparation_copied = False
    try:
        native_producer = _array_api_compat.compat.is_array_api_obj(producer)
    except Exception as exception:  # pylint: disable=broad-exception-caught
        raise errors.ConversionError(
            "from_dlpack: producer array protocol metadata is invalid"
        ) from exception
    if native_producer:
        try:
            source_namespace, source_backend = namespace_module.array_metadata(
                producer,
                0,
                allow_non_cpu=True,
                allow_non_native_numpy_dtype=True,
            )
        except errors.AscError as exception:
            raise errors.ConversionError(
                "from_dlpack: producer is not a supported native array"
            ) from exception
        if namespace_module.has_active_graph(producer, source_backend):
            raise errors.ConversionError(
                "from_dlpack: active autodiff graphs require explicit detach "
                "before import"
            )
        prepared = _interchange_source(
            typing.cast(asc_typing.NativeArray, producer),
            source_namespace,
            source_backend,
            context,
            "from_dlpack",
            allow_copy=policy is not config.CopyPolicy.NEVER,
        )
        preparation_copied = prepared is not producer
        producer = prepared
        if source_backend == "numpy" and _numpy_dlpack_requires_normalization(
            producer
        ):
            if policy is config.CopyPolicy.NEVER:
                raise errors.ConversionError(
                    "from_dlpack: NumPy producer normalization requires a copy"
                )
            prepared = _prepare_dlpack_capsule_export(producer, source_backend)
            preparation_copied = preparation_copied or prepared is not producer
            producer = prepared
    try:
        known_dtype = getattr(producer, "dtype", None)
    except Exception as exception:  # pylint: disable=broad-exception-caught
        raise errors.ConversionError(
            "from_dlpack: producer dtype metadata is invalid"
        ) from exception
    known_name: str | None = None
    if known_dtype is not None:
        try:
            known_name = _dtype_name(known_dtype)
            destination_source_dtype = getattr(target, known_name, None)
            destination_supports_source = (
                destination_source_dtype is not None
                and _dtype.is_supported_dtype(
                    context.backend,
                    destination_source_dtype,
                    jax_x64_enabled=(
                        context.backend != "jax"
                        or _dtype.active_jax_x64_enabled()
                    ),
                )
            )
        except Exception as exception:  # pylint: disable=broad-exception-caught
            raise errors.ConversionError(
                "from_dlpack: producer dtype metadata is invalid"
            ) from exception
        if not destination_supports_source:
            raise errors.ConversionError(
                "from_dlpack: producer dtype is unsupported by the destination "
                "and would be narrowed or rejected"
            )
    elif context.backend == "jax" and not _dtype.active_jax_x64_enabled():
        raise errors.ConversionError(
            "from_dlpack: producer dtype metadata is required when JAX x64 "
            "is disabled"
        )
    if policy is config.CopyPolicy.NEVER and context.dtype is not None:
        if known_dtype is None:
            raise errors.ConversionError(
                "from_dlpack: producer dtype metadata is required to prove a "
                "no-copy dtype request"
            )
        if known_name is None or known_name != _dtype_name(context.dtype):
            raise errors.ConversionError(
                "from_dlpack: requested dtype change requires a copy"
            )
    if (
        policy is config.CopyPolicy.NEVER
        and not _accepts_copy_keyword(target.from_dlpack)
        and not (
            context.backend == "numpy"
            and _is_builtin_numpy_dlpack_importer(target.from_dlpack)
        )
    ):
        raise errors.ConversionError(
            "from_dlpack: destination cannot guarantee no-copy import"
        )
    expected_dtype = context.dtype if context.dtype is not None else known_dtype
    copy_value = {
        config.CopyPolicy.ALWAYS: True,
        config.CopyPolicy.IF_NEEDED: None,
        config.CopyPolicy.NEVER: False,
    }[policy]
    opaque_torch_producer = (
        context.backend == "torch"
        and not native_producer
        and not isinstance(producer, _DLPackCapsule)
    )
    if opaque_torch_producer and not _accepts_copy_keyword(exporter):
        raise errors.ConversionError(
            "from_dlpack: opaque Torch producers must support the copy keyword "
            "for safe import"
        )
    if opaque_torch_producer and policy is config.CopyPolicy.NEVER:
        raise errors.ConversionError(
            "from_dlpack: opaque producer layout cannot safely guarantee a "
            "no-copy Torch import"
        )
    if opaque_torch_producer:
        producer = copy_opaque_cpu_dlpack_producer(producer)
        preparation_copied = True
        opaque_torch_producer = False
    import_copy = copy_value
    if copy_value is True and not opaque_torch_producer:
        import_copy = None
    elif opaque_torch_producer and copy_value is None:
        import_copy = True
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            imported = _from_dlpack(
                target,
                typing.cast(asc_typing.NativeArray, producer),
                device=context.device,
                copy=import_copy,
                allow_legacy_no_copy=(
                    context.backend == "numpy"
                    and _is_builtin_numpy_dlpack_importer(target.from_dlpack)
                ),
            )
    except _CONVERSION_FAILURES as exception:
        raise errors.ConversionError(
            "from_dlpack: destination rejected the capsule or copy/device "
            "policy"
        ) from exception
    if policy is config.CopyPolicy.NEVER:
        source_pointer = _data_pointer(producer)
        imported_pointer = _data_pointer(imported)
        if (
            source_pointer is not None
            and imported_pointer is not None
            and source_pointer != imported_pointer
        ):
            raise errors.ConversionError(
                "from_dlpack: copy/device policy rejected a hidden copy under "
                "CopyPolicy.NEVER"
            )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            normalization_fulfills_copy = preparation_copied and (
                context.dtype is None
                or known_name == _dtype_name(context.dtype)
            )
            if (
                policy is config.CopyPolicy.ALWAYS
                and normalization_fulfills_copy
            ):
                final_copy = False
            elif opaque_torch_producer and import_copy is True:
                final_copy = True
            else:
                final_copy = copy_value
            result = target.asarray(
                imported,
                dtype=context.dtype,
                device=context.device,
                copy=final_copy,
            )
    except _CONVERSION_FAILURES as exception:
        raise errors.ConversionError(
            "from_dlpack: destination rejected the requested dtype or copy "
            "policy"
        ) from exception
    namespace_module.array_namespace(result)
    expected_name = _dtype_name(expected_dtype)
    result_name = _dtype_name(result.dtype)
    if expected_dtype is not None and result_name != expected_name:
        raise errors.ConversionError(
            "from_dlpack: destination narrowed or changed requested dtype"
        )
    return result


__all__ = [
    "convert_array",
    "copy_array",
    "detach",
    "from_dlpack",
    "from_numpy",
    "stop_gradient",
    "to_device",
    "to_dlpack",
    "to_numpy",
]
