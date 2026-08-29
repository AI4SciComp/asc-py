# Copyright 2026 AI4SciComp contributors.
# SPDX-License-Identifier: Apache-2.0
"""All explicit conversion, ownership, graph, and DLPack boundaries."""

from __future__ import annotations

import itertools
import os
import pathlib
import subprocess
import sys
import typing
import warnings

import numpy
import pytest

import asc
from asc import data
from asc import typing as asc_typing

BACKENDS = ("numpy", "torch", "jax")


@pytest.mark.parametrize(
    ("source_name", "target_name"), tuple(itertools.product(BACKENDS, repeat=2))
)
def test_every_backend_pair_float32(source_name: str, target_name: str) -> None:
    source = asc.backend(source_name).xp.asarray(
        [1.0, 2.0], dtype=asc.backend(source_name).xp.float32
    )
    result = asc.convert_array(source, target_name, copy=True)
    assert asc.backend_of(result) == target_name
    assert str(result.dtype).endswith("float32")
    numpy.testing.assert_allclose(numpy.asarray(result), [1, 2])


@pytest.mark.parametrize(
    ("source_name", "target_name"), tuple(itertools.product(BACKENDS, repeat=2))
)
def test_dlpack_every_backend_pair(source_name: str, target_name: str) -> None:
    source = asc.backend(source_name).xp.asarray(
        [1.0, 2.0], dtype=asc.backend(source_name).xp.float32
    )
    capsule = asc.to_dlpack(source)
    result = asc.from_dlpack(capsule, target_name, copy=True)
    assert asc.backend_of(result) == target_name
    numpy.testing.assert_allclose(numpy.asarray(result), [1, 2])
    with pytest.raises(asc.ConversionError, match="capsule"):
        asc.from_dlpack(capsule, target_name, copy=True)


def test_dlpack_never_verifies_shared_storage() -> None:
    source = numpy.arange(4, dtype=numpy.float32)

    result = asc.from_dlpack(
        asc.to_dlpack(source), "numpy", copy=asc.CopyPolicy.NEVER
    )

    assert numpy.shares_memory(source, result)


@pytest.mark.parametrize("backend", BACKENDS)
def test_same_backend_copy_alias_and_device(backend: str) -> None:
    selected = asc.backend(backend)
    source = selected.xp.asarray([1.0, 2.0], dtype=selected.xp.float32)
    copied = asc.copy_array(source, copy=True)
    assert copied is not source
    aliased = asc.copy_array(source, copy=False)
    assert aliased is source
    moved = asc.to_device(source, "cpu", copy=None)
    assert asc.backend_of(moved) == backend
    assert moved.dtype == source.dtype


@pytest.mark.backend("torch")
def test_numpy_boundaries_and_policy_errors() -> None:
    source = numpy.asarray([1.0, 2.0], dtype=numpy.float32)
    torch_value = asc.from_numpy(source, "torch")
    assert asc.backend_of(torch_value) == "torch"
    host = asc.to_numpy(torch_value)
    assert isinstance(host, numpy.ndarray)
    numpy.testing.assert_allclose(host, source)
    with pytest.raises(asc.ConversionError, match="source must"):
        asc.from_numpy(torch_value)
    with pytest.raises(asc.ConversionError, match="cross-backend"):
        asc.convert_array(source, "torch", copy=False)
    with pytest.raises(asc.ConversionError, match="destination"):
        asc.convert_array(source, object())
    with pytest.raises(asc.ConversionError, match="copy"):
        asc.convert_array(source, "numpy", copy="yes")
    with pytest.raises(asc.ConversionError, match="CPU"):
        asc.convert_array(source, "numpy", device="cuda")


@pytest.mark.backend("jax")
def test_dtype_narrowing_is_never_silent() -> None:
    source = numpy.asarray([1.0], dtype=numpy.float64)
    import jax

    if not jax.config.x64_enabled:
        with pytest.raises(asc.ConversionError, match="dtype"):
            asc.convert_array(source, "jax")
        with pytest.raises(asc.ConversionError, match="dtype"):
            asc.from_dlpack(asc.to_dlpack(source), "jax", copy=True)


@pytest.mark.backend("jax")
def test_native_dlpack_producer_dtype_is_never_silently_narrowed() -> None:
    program = """
import numpy as np
import asc
source = np.asarray([1.25], dtype=np.float64)
try:
    asc.from_dlpack(source, "jax", copy=True)
except asc.ConversionError as exception:
    assert "narrowed" in str(exception)
else:
    raise AssertionError("native producer dtype narrowing was accepted")
"""
    environment = dict(os.environ)
    environment["JAX_ENABLE_X64"] = "0"
    environment["JAX_PLATFORMS"] = "cpu"
    environment["PYTHONWARNINGS"] = "error"

    result = subprocess.run(
        [sys.executable, "-c", program],
        check=False,
        capture_output=True,
        env=environment,
        text=True,
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("context_kind", ("backend", "array_context"))
def test_dlpack_import_honors_destination_context_dtype(
    context_kind: str,
) -> None:
    source = numpy.asarray([1.25], dtype=numpy.float64)
    destination = (
        asc.backend("numpy", dtype=numpy.float32)
        if context_kind == "backend"
        else asc.ArrayContext("numpy", dtype=numpy.float32)
    )

    result = asc.from_dlpack(asc.to_dlpack(source), destination, copy=True)

    assert result.dtype == numpy.float32
    numpy.testing.assert_allclose(result, [1.25])


@pytest.mark.backend("jax")
def test_jax_creation_context_normalizes_default_cpu_overrides() -> None:
    destination = asc.backend("jax")
    context = asc.CreationContext(
        destination.xp,
        "jax",
        device=destination.device,
    )
    source = numpy.asarray([1.25], dtype=numpy.float32)

    converted = asc.from_numpy(source, context)
    imported = asc.from_dlpack(asc.to_dlpack(source), context, copy=True)

    assert asc.backend_of(converted) == "jax"
    assert asc.backend_of(imported) == "jax"
    numpy.testing.assert_allclose(numpy.asarray(converted), source)
    numpy.testing.assert_allclose(numpy.asarray(imported), source)


def test_strict_creation_context_is_not_a_runtime_destination() -> None:
    import array_api_strict

    array_api_strict.set_array_api_strict_flags(api_version="2024.12")
    context = asc.CreationContext(array_api_strict, "array_api_strict")

    with pytest.raises(asc.ConversionError, match="conformance oracle"):
        asc.convert_array(numpy.asarray([1]), context)


@pytest.mark.backend("torch")
def test_torch_graph_boundaries_are_explicit() -> None:
    import torch

    value = torch.tensor([1.0, 2.0], requires_grad=True)
    copied = asc.copy_array(value, copy=True)
    assert copied.requires_grad
    with pytest.raises(asc.ConversionError, match="graphs"):
        asc.convert_array(value, "numpy")
    with pytest.raises(asc.ConversionError, match="allow_detach"):
        asc.to_numpy(value)
    host = asc.to_numpy(value, allow_detach=True)
    numpy.testing.assert_allclose(host, [1, 2])
    with pytest.raises(asc.ConversionError, match="active graph"):
        asc.to_dlpack(value)
    detached = asc.detach(value)
    assert not detached.requires_grad
    assert asc.stop_gradient(value).grad_fn is None


@pytest.mark.backend("torch")
def test_torch_forward_mode_graph_is_rejected() -> None:
    import torch

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"`torch\.jit\.script` is deprecated\..*",
            category=DeprecationWarning,
        )
        with torch.autograd.forward_ad.dual_level():
            dual = torch.autograd.forward_ad.make_dual(
                torch.tensor([1.0]), torch.tensor([2.0])
            )
            with pytest.raises(asc.ConversionError, match="graphs"):
                asc.convert_array(dual, "numpy")


@pytest.mark.backend("torch")
def test_torch_lazy_conjugate_and_negative_views_export_logical_values(
    tmp_path: pathlib.Path,
) -> None:
    import torch

    negative_view = typing.cast(
        typing.Callable[[torch.Tensor], torch.Tensor],
        vars(torch)["_neg_view"],
    )
    sources = (
        (
            torch.tensor([1 + 2j, 3 - 4j], dtype=torch.complex64).conj(),
            numpy.asarray([1 - 2j, 3 + 4j], dtype=numpy.complex64),
        ),
        (
            negative_view(torch.tensor([1.0, -2.0], dtype=torch.float32)),
            numpy.asarray([-1.0, 2.0], dtype=numpy.float32),
        ),
    )
    for position, (source, expected) in enumerate(sources):
        converted = asc.convert_array(source, "numpy", copy=True)
        host = asc.to_numpy(source)
        imported = asc.from_dlpack(asc.to_dlpack(source), "numpy", copy=True)
        path = data.save_npy(
            tmp_path / f"view-{position}.npy",
            source,
            allow_transfer=True,
        )

        for result in (converted, host, imported, data.load_npy(path)):
            numpy.testing.assert_array_equal(result, expected)


@pytest.mark.backend("torch")
def test_negative_stride_numpy_view_is_compacted_before_torch_import() -> None:
    program = """
import numpy as np
import asc
source = np.arange(5, dtype=np.float32)[::-1]
result = asc.convert_array(source, "torch", copy=True)
np.testing.assert_array_equal(np.asarray(result), [4, 3, 2, 1, 0])
"""
    environment = dict(os.environ)
    environment["PYTHONWARNINGS"] = "error"

    completed = subprocess.run(
        [sys.executable, "-c", program],
        check=False,
        env=environment,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


@pytest.mark.backend("torch")
def test_negative_stride_numpy_view_is_compacted_before_dlpack_export() -> None:
    program = """
import numpy as np
import asc
source = np.arange(5, dtype=np.float32)[::-1]
capsule = asc.to_dlpack(source)
result = asc.from_dlpack(capsule, "torch", copy=True)
np.testing.assert_array_equal(np.asarray(result), [4, 3, 2, 1, 0])
"""
    completed = subprocess.run(
        [sys.executable, "-c", program],
        check=False,
        env={**os.environ, "PYTHONWARNINGS": "error"},
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


@pytest.mark.backend("torch")
def test_boundary_acknowledgements_require_actual_booleans() -> None:
    import torch

    graph = torch.tensor([1.0], requires_grad=True)
    host = numpy.asarray([1.0], dtype=numpy.float32)

    for value in (1, "false", object()):
        with pytest.raises(asc.ConversionError, match="Booleans"):
            asc.to_numpy(graph, allow_detach=value)  # type: ignore[arg-type]
        with pytest.raises(asc.ConversionError, match="Booleans"):
            asc.to_numpy(host, allow_transfer=value)  # type: ignore[arg-type]
        with pytest.raises(asc.ConversionError, match="Boolean"):
            asc.to_dlpack(graph, allow_detach=value)  # type: ignore[arg-type]


@pytest.mark.parametrize("destination", ("torch", "jax"))
def test_non_native_numpy_byte_order_is_normalized_for_dlpack(
    destination: str,
) -> None:
    source = numpy.asarray([1.25, -2.5], dtype=">f4")

    result = asc.convert_array(source, destination)

    assert result.dtype == asc.backend(destination).xp.float32
    numpy.testing.assert_array_equal(numpy.asarray(result), [1.25, -2.5])


@pytest.mark.parametrize("destination", ("torch", "jax"))
def test_noncompact_positive_numpy_strides_are_normalized(
    destination: str,
) -> None:
    source = numpy.arange(8, dtype=numpy.float32)[::2]

    result = asc.convert_array(source, destination)

    numpy.testing.assert_array_equal(numpy.asarray(result), [0, 2, 4, 6])


def test_numpy_boundaries_normalize_non_native_byte_order() -> None:
    source = numpy.asarray([1.25, -2.5], dtype=">f4")

    converted = asc.convert_array(source, "numpy")
    imported = asc.from_numpy(source, "numpy")
    hosted = asc.to_numpy(source)
    round_tripped = asc.from_dlpack(asc.to_dlpack(source), "numpy")

    for result in (converted, imported, hosted, round_tripped):
        assert result.dtype == numpy.dtype("float32")
        assert result.dtype.isnative
        numpy.testing.assert_array_equal(result, [1.25, -2.5])

    with pytest.raises(asc.ConversionError, match="requires a copy"):
        asc.convert_array(source, "numpy", copy=asc.CopyPolicy.NEVER)


@pytest.mark.backend("torch")
def test_empty_dlpack_import_accepts_no_copy_policy() -> None:
    import torch

    source = torch.empty((0,), dtype=torch.float32)

    result = asc.from_dlpack(
        asc.to_dlpack(source),
        "torch",
        copy=asc.CopyPolicy.NEVER,
    )

    assert result.shape == (0,)
    assert result.dtype == torch.float32


def test_from_dlpack_normalizes_non_native_numpy_producer() -> None:
    source = numpy.asarray([1.25, -2.5], dtype=">f4")

    for copy in (asc.CopyPolicy.ALWAYS, asc.CopyPolicy.IF_NEEDED):
        result = asc.from_dlpack(source, "numpy", copy=copy)
        assert result.dtype == numpy.dtype("float32")
        assert result.dtype.isnative
        numpy.testing.assert_array_equal(result, [1.25, -2.5])

    with pytest.raises(asc.ConversionError, match="requires a copy"):
        asc.from_dlpack(source, "numpy", copy=asc.CopyPolicy.NEVER)


def test_dlpack_normalizes_read_only_numpy_producer() -> None:
    source = numpy.asarray([1.25, -2.5], dtype=numpy.float32)
    source.setflags(write=False)

    exported = asc.from_dlpack(asc.to_dlpack(source), "numpy")
    direct = asc.from_dlpack(source, "numpy", copy=asc.CopyPolicy.IF_NEEDED)

    for result in (exported, direct):
        numpy.testing.assert_array_equal(result, source)
    with pytest.raises(asc.ConversionError, match="requires a copy"):
        asc.from_dlpack(source, "numpy", copy=asc.CopyPolicy.NEVER)


def test_from_dlpack_accepts_metadata_free_no_copy_producer() -> None:
    source = numpy.asarray([1.0, 2.0], dtype=numpy.float32)
    producer = _TrackingDLPackProducer(source)

    result = asc.from_dlpack(
        producer,
        "numpy",
        copy=asc.CopyPolicy.NEVER,
    )

    assert producer.consumed
    assert numpy.shares_memory(source, result)


def test_from_dlpack_rejects_device_before_consuming_producer() -> None:
    producer = _TrackingDLPackProducer(
        numpy.asarray([1.0], dtype=numpy.float32)
    )

    with pytest.raises(asc.ConversionError, match="destination context"):
        asc.from_dlpack(producer, "numpy", device="cuda")

    assert not producer.consumed


def test_from_dlpack_rejects_non_cpu_producer_before_consumption() -> None:
    producer = _NonCpuDLPackProducer(numpy.asarray([1.0], dtype=numpy.float32))

    with pytest.raises(asc.ConversionError, match="CPU producers"):
        asc.from_dlpack(producer, "numpy")

    assert not producer.consumed


def test_from_dlpack_rejects_never_dtype_change_before_consumption() -> None:
    producer = _TypedTrackingDLPackProducer(
        numpy.asarray([1.0], dtype=numpy.float32)
    )

    with pytest.raises(asc.ConversionError, match="dtype change"):
        asc.from_dlpack(
            producer,
            "numpy",
            dtype=numpy.float64,
            copy=asc.CopyPolicy.NEVER,
        )

    assert not producer.consumed


def test_from_dlpack_requires_dtype_metadata_for_typed_never_request() -> None:
    producer = _TrackingDLPackProducer(
        numpy.asarray([1.0], dtype=numpy.float32)
    )

    with pytest.raises(asc.ConversionError, match="dtype metadata"):
        asc.from_dlpack(
            producer,
            "numpy",
            dtype=numpy.float32,
            copy=asc.CopyPolicy.NEVER,
        )

    assert not producer.consumed


@pytest.mark.backend("torch")
def test_from_dlpack_rejects_unsupported_source_dtype_before_consumption() -> (
    None
):
    import torch

    class BFloat16Producer:
        """Track attempts to consume a Torch bfloat16 producer."""

        def __init__(self) -> None:
            self.value = torch.asarray([1.0], dtype=torch.bfloat16)
            self.consumed = False
            self.dtype = torch.bfloat16

        def __dlpack_device__(self) -> tuple[int, int]:
            return self.value.__dlpack_device__()

        def __dlpack__(self, *args: object, **kwargs: object) -> object:
            self.consumed = True
            return self.value.__dlpack__(*args, **kwargs)

    producer = BFloat16Producer()

    with pytest.raises(asc.ConversionError, match=r"unsupported.*destination"):
        asc.from_dlpack(producer, "numpy", copy=True)

    assert not producer.consumed


def test_from_dlpack_wraps_hostile_protocol_metadata() -> None:
    class HostileProtocol:
        """Raise while exposing the DLPack protocol method."""

        @property
        def __dlpack__(self) -> object:
            raise RuntimeError("hostile protocol")

    class HostileDevice:
        """Raise while exposing DLPack device provenance."""

        def __dlpack__(self) -> object:
            raise AssertionError("must not consume")

        @property
        def __dlpack_device__(self) -> object:
            raise RuntimeError("hostile device")

    class HostileDType(_TrackingDLPackProducer):
        """Raise while exposing producer dtype metadata."""

        @property
        def dtype(self) -> object:
            raise RuntimeError("hostile dtype")

    producers = (
        HostileProtocol(),
        HostileDevice(),
        HostileDType(numpy.asarray([1.0], dtype=numpy.float32)),
    )

    for producer in producers:
        with pytest.raises(asc.ConversionError):
            asc.from_dlpack(producer, "numpy")


def test_from_dlpack_normalization_satisfies_always_with_one_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import asc.conversion as conversion

    prepared_values: list[asc_typing.NativeArray] = []
    prepare = typing.cast(
        typing.Callable[
            [asc_typing.NativeArray, asc_typing.BackendName],
            asc_typing.NativeArray,
        ],
        vars(conversion)["_prepare_dlpack_capsule_export"],
    )

    def track_preparation(
        value: asc_typing.NativeArray,
        backend: asc_typing.BackendName,
    ) -> asc_typing.NativeArray:
        prepared = prepare(value, backend)
        prepared_values.append(prepared)
        return prepared

    monkeypatch.setattr(
        conversion, "_prepare_dlpack_capsule_export", track_preparation
    )
    source = numpy.asarray([1.0], dtype=">f4")

    result = asc.from_dlpack(source, "numpy", copy=asc.CopyPolicy.ALWAYS)

    assert prepared_values[0] is not source
    assert numpy.shares_memory(result, prepared_values[0])


def test_from_dlpack_normalization_allows_required_dtype_copy() -> None:
    source = numpy.asarray([1.0], dtype=">f4")

    result = asc.from_dlpack(
        source,
        "numpy",
        dtype=numpy.float64,
        copy=asc.CopyPolicy.ALWAYS,
    )

    assert result.dtype == numpy.float64
    numpy.testing.assert_array_equal(result, [1.0])


def test_from_dlpack_rejects_legacy_never_before_consumption(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected = asc.backend("numpy")
    producer = _TrackingDLPackProducer(
        numpy.asarray([1.0], dtype=numpy.float32)
    )

    def legacy_import(value: object) -> object:
        return numpy.from_dlpack(value)

    monkeypatch.setattr(
        selected.xp, "from_dlpack", legacy_import, raising=False
    )
    with pytest.raises(asc.ConversionError, match="guarantee no-copy"):
        asc.from_dlpack(producer, "numpy", copy=asc.CopyPolicy.NEVER)

    assert not producer.consumed


def test_from_dlpack_type_error_fallback_preserves_always_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected = asc.backend("numpy")
    source = numpy.asarray([1.0], dtype=numpy.float32)

    def legacy_import(value: object) -> object:
        return numpy.from_dlpack(value)

    monkeypatch.setattr(
        selected.xp, "from_dlpack", legacy_import, raising=False
    )
    result = asc.from_dlpack(source, "numpy", copy=asc.CopyPolicy.ALWAYS)

    assert not numpy.shares_memory(source, result)
    source[0] = 2.0
    numpy.testing.assert_array_equal(result, [1.0])


def test_from_dlpack_rejects_not_implemented_never_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected = asc.backend("numpy")
    producer = _TrackingDLPackProducer(
        numpy.asarray([1.0], dtype=numpy.float32)
    )

    def unsupported_import(
        value: object,
        *,
        device: object | None = None,
        copy: bool | None = None,
    ) -> object:
        del value, device, copy
        raise NotImplementedError

    monkeypatch.setattr(
        selected.xp, "from_dlpack", unsupported_import, raising=False
    )
    with pytest.raises(asc.ConversionError, match="guarantee no-copy"):
        asc.from_dlpack(producer, "numpy", copy=asc.CopyPolicy.NEVER)

    assert not producer.consumed


def test_numpy_detach_is_documented_no_op() -> None:
    source = numpy.asarray([1.0])
    assert asc.detach(source) is source
    assert asc.stop_gradient(source) is source


def test_capsule_rejects_copy_and_device_requests() -> None:
    source = numpy.asarray([1.0], dtype=numpy.float32)
    for copy in (0, 1, "yes", object()):
        capsule = asc.to_dlpack(source)
        with pytest.raises(TypeError, match="copy must be"):
            capsule.__dlpack__(copy=copy)
        assert not capsule.consumed
        capsule.__dlpack__()

    capsule = asc.to_dlpack(source)
    assert capsule.__dlpack_device__() == (1, 0)
    with pytest.raises(BufferError, match="copy"):
        capsule.__dlpack__(copy=True)
    second = asc.to_dlpack(source)
    with pytest.raises(BufferError, match="devices"):
        second.__dlpack__(dl_device=(2, 0))


class _TrackingDLPackProducer:
    """Record whether a one-shot NumPy DLPack producer was consumed."""

    def __init__(self, value: numpy.ndarray[typing.Any, typing.Any]) -> None:
        self.value = value
        self.consumed = False

    def __dlpack_device__(self) -> tuple[int, int]:
        return self.value.__dlpack_device__()

    def __dlpack__(self, *args: object, **kwargs: object) -> object:
        self.consumed = True
        return self.value.__dlpack__(*args, **kwargs)


class _NonCpuDLPackProducer(_TrackingDLPackProducer):
    """Protocol producer reporting an unsupported accelerator device."""

    def __dlpack_device__(self) -> tuple[int, int]:
        return (2, 0)


class _TypedTrackingDLPackProducer(_TrackingDLPackProducer):
    """One-shot producer with observable dtype metadata."""

    @property
    def dtype(self) -> object:
        return self.value.dtype


def test_same_backend_never_preflights_dtype_changes() -> None:
    source = numpy.asarray([1.0], dtype=numpy.float32)

    with pytest.raises(asc.ConversionError, match="dtype or device change"):
        asc.convert_array(
            source,
            "numpy",
            dtype=numpy.float64,
            copy=asc.CopyPolicy.NEVER,
        )


def test_malformed_destination_has_stable_conversion_error() -> None:
    class PartialDestination:
        """Backend-shaped object missing required destination metadata."""

        name = "numpy"
        xp = asc.backend("numpy").xp

    with pytest.raises(asc.ConversionError, match="destination must be"):
        asc.convert_array(
            numpy.asarray([1.0], dtype=numpy.float32),
            PartialDestination(),
        )
