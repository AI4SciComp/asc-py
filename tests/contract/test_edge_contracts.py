# Copyright 2026 AI4SciComp contributors.
# SPDX-License-Identifier: Apache-2.0
"""Failure-policy and diagnostic edge contracts."""

from __future__ import annotations

import importlib.metadata

import numpy
import pytest

import asc
from asc.core.namespace import identify_backend, validate_namespace_revision
from asc.extensions import _dispatch


@pytest.mark.backend("torch")
@pytest.mark.backend("jax")
def test_capability_records_properties_and_requirements(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    numpy_info = asc.backend_info("numpy")
    assert numpy_info.random and numpy_info.index_add
    assert not numpy_info.autodiff and not numpy_info.compilation
    assert numpy_info.dtype_families[-1] == "complex floating"
    torch_info = asc.backend_info(asc.backend("torch"))
    assert torch_info.autodiff and not torch_info.compilation
    jax_info = asc.backend_info(asc.backend("jax").xp.asarray([1]))
    assert jax_info.autodiff and jax_info.compilation
    with pytest.raises(
        asc.CapabilityNotSupportedError, match="does not support"
    ):
        asc.require_capability("numpy", asc.Capability.AUTODIFF)

    original = importlib.metadata.version

    def unavailable(distribution: str) -> str:
        if distribution == "torch":
            raise importlib.metadata.PackageNotFoundError(distribution)
        return original(distribution)

    monkeypatch.setattr(importlib.metadata, "version", unavailable)
    assert not asc.backend_info("torch").installed
    with pytest.raises(asc.BackendUnavailableError, match="install"):
        asc.require_capability("torch", "autodiff")


def test_extension_dispatch_strict_and_dependency_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(asc.UnsupportedCapabilityError, match="strict"):
        _dispatch.load_backend("array_api_strict")
    original = _dispatch.importlib.import_module

    def missing(name: str):
        if name == "asc.backends.torch":
            exception = ModuleNotFoundError("No module named 'torch'")
            exception.name = "torch"
            raise exception
        return original(name)

    monkeypatch.setattr(_dispatch.importlib, "import_module", missing)
    with pytest.raises(asc.BackendUnavailableError, match="torch"):
        _dispatch.load_backend("torch")


def test_namespace_surface_and_info_failures() -> None:
    class Incomplete:
        """Hashable incomplete namespace fixture."""

        __array_api_version__ = "2024.12"

    with pytest.raises(asc.NamespaceError, match="incomplete"):
        validate_namespace_revision(Incomplete())
    incomplete = Incomplete()
    incomplete.__array_api_version__ = "2023.12"
    with pytest.raises(asc.NamespaceError, match="does not satisfy"):
        validate_namespace_revision(incomplete)
    with pytest.raises(asc.NamespaceError, match="unsupported namespace"):
        identify_backend(Incomplete())
    info = asc.namespace_info(asc.backend("numpy").xp)
    assert isinstance(info.capabilities, tuple)


def test_operations_metric_comparison_and_signal_failures() -> None:
    values = numpy.asarray([1.0, 2.0], dtype=numpy.float32)
    integers = numpy.asarray([1, 2], dtype=numpy.int32)
    with pytest.raises(asc.DTypeError):
        asc.sum_of_squares(integers)
    for invalid in (
        lambda: asc.ops.diag(values.reshape(1, 2)),
        lambda: asc.ops.diag(values, offset=True),
        lambda: asc.ops.pad(values, ((1, 1), (1, 1))),
        lambda: asc.ops.pad(values, ((-1, 0),)),
        lambda: asc.ops.pad(values, 1, mode="linear_ramp"),
        lambda: asc.ops.pad(
            numpy.empty((0,), dtype=numpy.float32), 1, mode="edge"
        ),
        lambda: asc.ops.ravel_multi_index((integers,), (0,)),
        lambda: asc.ops.ravel_multi_index((values,), (2,)),
        lambda: asc.ops.ravel_multi_index((integers,), (2,)),
        lambda: asc.ops.unravel_index(integers, ()),
        lambda: asc.ops.unravel_index(values, (2,)),
        lambda: asc.ops.unravel_index(integers, (2,)),
    ):
        with pytest.raises((asc.AscError, IndexError, TypeError, ValueError)):
            invalid()
    with pytest.raises(asc.MixedBackendError):
        asc.ops.pad(values, 1, constant_values=numpy.asarray(1.0))
    with pytest.raises(asc.DTypeError):
        asc.ops.eps(integers)
    with pytest.raises(asc.DTypeError):
        asc.ops.eps(numpy.float32)
    with pytest.raises(ValueError, match="shapes"):
        asc.metrics.mean_absolute_error(values, values[:1])
    with pytest.raises(asc.DTypeError):
        asc.metrics.mean_squared_error(integers, integers)
    with pytest.raises(ValueError, match="reduction"):
        asc.metrics.mean_absolute_error(values, values, reduction="bad")
    with pytest.raises(ValueError, match="axis"):
        asc.metrics.mean_absolute_error(
            values, values, reduction="none", axis=0
        )
    exact_zero = numpy.zeros((2,), dtype=numpy.float32)
    assert float(asc.metrics.relative_l2_error(exact_zero, exact_zero)) == 0
    assert numpy.isinf(asc.metrics.relative_l2_error(values, exact_zero))
    assert float(asc.metrics.r2_score(exact_zero, exact_zero)) == 1
    assert float(asc.metrics.r2_score(values, exact_zero)) == 0
    with pytest.raises(ValueError, match="equal_nan"):
        asc.ops.isclose(values, values, equal_nan=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="tolerances"):
        asc.ops.isclose(values, values, rtol=-1)
    with pytest.raises(AssertionError, match="not close"):
        asc.ops.assert_allclose(values, values + 5)
    nan = numpy.asarray([numpy.nan], dtype=numpy.float32)
    assert asc.ops.allclose(nan, nan, equal_nan=True)
    kernel = numpy.asarray([1.0], dtype=numpy.float32)
    with pytest.raises(ValueError, match="one-dimensional"):
        asc.ops.convolve1d(values, kernel.reshape(1, 1))
    with pytest.raises(ValueError, match="axis"):
        asc.ops.convolve1d(values, kernel, axis=3)
    for boolean_axis in (False, True):
        with pytest.raises(ValueError, match="non-Boolean integer"):
            asc.ops.convolve1d(values, kernel, axis=boolean_axis)
        with pytest.raises(ValueError, match="non-Boolean integer"):
            asc.ops.moving_mean(values, 1, axis=boolean_axis)
    with pytest.raises(ValueError, match="empty"):
        asc.ops.convolve1d(values, kernel[:0])
    with pytest.raises(ValueError, match="mode"):
        asc.ops.convolve1d(values, kernel, mode="bad")
    numpy.testing.assert_array_equal(
        asc.ops.convolve1d(values, numpy.ones((3,), dtype=numpy.float32)),
        numpy.convolve(values, numpy.ones((3,), dtype=numpy.float32), "valid"),
    )
    with pytest.raises(ValueError, match="window"):
        asc.ops.moving_mean(values, 0)
    with pytest.raises(asc.DTypeError):
        asc.ops.moving_mean(integers, 1)


def test_update_validation_edges() -> None:
    array = numpy.zeros((2, 2), dtype=numpy.float32)
    indices = numpy.asarray([0], dtype=numpy.int16)
    values = numpy.ones((1, 2), dtype=numpy.float32)
    for call in (
        lambda: asc.index_add(array, indices, values, axis=True),
        lambda: asc.index_add(array, indices, values, axis=2),
        lambda: asc.index_add(numpy.asarray(0.0), indices, values),
        lambda: asc.index_add(array, indices.reshape(1, 1), values),
        lambda: asc.index_add(array, indices.astype(numpy.uint8), values),
        lambda: asc.index_add(array, indices, numpy.ones((3, 3))),
    ):
        with pytest.raises(asc.AscError):
            call()
    complex_array = numpy.zeros((2,), dtype=numpy.complex64)
    complex_values = numpy.ones((1,), dtype=numpy.complex64)
    with pytest.raises(asc.DTypeError, match="complex"):
        asc.index_min(complex_array, indices, complex_values)
    import array_api_strict as strict

    strict_indices = strict.asarray([0], dtype=strict.int32)
    strict_array = strict.asarray([0.0], dtype=strict.float32)
    with pytest.raises(asc.CapabilityNotSupportedError, match="strict"):
        asc.index_set(strict_array, strict_indices, strict_array)
