# Copyright 2026 AI4SciComp contributors.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for adversarial backend and boundary review findings."""

from __future__ import annotations

import collections.abc
import dataclasses
import importlib.metadata
import importlib.util
import os
import pathlib
import subprocess
import sys
import typing
import warnings

import numpy
import pytest

import asc
from asc import random, tree
from asc.extensions import random as compatibility_random


class _SpoofedNumpyDType:
    """Non-native object with deliberately misleading dtype metadata."""

    __module__ = "numpy"
    name = "float32"


class _SpoofedStrictDType:
    """Non-native object impersonating an array-api-strict dtype."""

    __module__ = "array_api_strict"
    name = "float32"


class _MalformedDType:
    """Hostile dtype object whose inspection hooks all fail."""

    @property
    def name(self) -> object:
        raise RuntimeError("name must not escape")

    def __str__(self) -> str:
        raise RuntimeError("str must not escape")

    def __repr__(self) -> str:
        raise RuntimeError("repr must not escape")


def test_exact_dtype_release_surface_rejects_numpy_extensions(
    tmp_path: pathlib.Path,
) -> None:
    import array_api_strict

    array_api_strict.set_array_api_strict_flags(api_version="2024.12")
    selected = asc.backend("numpy")
    discovered = asc.array_namespace(numpy.asarray([1], dtype=numpy.float32))
    assert discovered is selected.xp
    assert "float16" in asc.backend_info("numpy").dtypes
    assert "longdouble" not in asc.backend_info("numpy").dtypes

    for dtype in (numpy.longdouble, numpy.clongdouble):
        value = numpy.asarray([[1]], dtype=dtype)
        with pytest.raises(asc.DTypeError, match="release surface"):
            asc.backend("numpy", dtype=dtype)
        with pytest.raises(asc.DTypeError, match="release surface"):
            selected.xp.asarray([1], dtype=dtype)
        with pytest.raises(asc.DTypeError, match="release surface"):
            discovered.asarray([1], dtype=dtype)
        with pytest.raises(asc.ContextError, match="release surface"):
            asc.ArrayContext("numpy", dtype=dtype)
        with pytest.raises(asc.ContextError, match="release surface"):
            asc.CreationContext(selected.xp, "numpy", dtype=dtype)
        assert not asc.is_array(value)
        with pytest.raises(asc.DataFormatError, match="supported numeric"):
            asc.data.save_npy(tmp_path / f"{dtype.__name__}.npy", value)
        with pytest.raises(asc.DataFormatError, match="supported numeric"):
            asc.data.save_csv(tmp_path / f"{dtype.__name__}.csv", value)

    csv_path = tmp_path / "extended.csv"
    csv_path.write_text("1.0\n", encoding="utf-8")
    with pytest.raises(asc.DataFormatError, match="supported numeric"):
        asc.data.load_csv(csv_path, dtype=numpy.longdouble)

    state = random.random_state(1, backend="numpy")
    with pytest.raises(asc.RandomStateError, match="release surface"):
        random.uniform((2,), state=state, dtype=numpy.longdouble)
    with pytest.raises(asc.RandomStateError, match="release surface"):
        random.glorot_uniform((2, 2), state=state, dtype=numpy.longdouble)

    non_native = numpy.asarray([1.0], dtype=">f4")
    assert not asc.is_array(non_native)
    with pytest.raises(asc.DTypeError, match="release surface"):
        asc.backend("numpy", dtype=non_native.dtype)
    with pytest.raises(asc.DTypeError, match="release surface"):
        selected.xp.asarray([1.0], dtype=non_native.dtype)

    spoofed = _SpoofedNumpyDType()
    with pytest.raises(asc.DTypeError, match="release surface"):
        asc.backend("numpy", dtype=spoofed)
    with pytest.raises(asc.ContextError, match="release surface"):
        asc.ArrayContext("numpy", dtype=spoofed)
    with pytest.raises(asc.RandomStateError, match="release surface"):
        random.uniform((2,), state=state, dtype=spoofed)

    with pytest.raises(asc.ContextError, match="release surface"):
        asc.ArrayContext("array_api_strict", dtype=_SpoofedStrictDType())
    strict_context = asc.ArrayContext(
        "array_api_strict", dtype=array_api_strict.float32
    )
    assert strict_context.dtype is array_api_strict.float32
    malformed = _MalformedDType()
    with pytest.raises(asc.DTypeError, match="release surface"):
        asc.backend("numpy", dtype=malformed)
    with pytest.raises(asc.ContextError, match="release surface"):
        asc.ArrayContext("array_api_strict", dtype=malformed)
    with pytest.raises(asc.ContextError, match="selected namespace"):
        asc.CreationContext(
            array_api_strict,
            "array_api_strict",
            dtype=malformed,
        )


@pytest.mark.backend("torch")
def test_exact_dtype_release_surface_rejects_torch_float8() -> None:
    import torch

    selected = asc.backend("torch")
    numpy_backend = asc.backend("numpy")
    assert "float16" in asc.backend_info("torch").dtypes
    assert "bfloat16" in asc.backend_info("torch").dtypes
    with pytest.raises(asc.DTypeError, match="release surface"):
        asc.backend("numpy", dtype=torch.float32)
    with pytest.raises(asc.DTypeError, match="release surface"):
        numpy_backend.xp.asarray([1.0], dtype=torch.float32)
    for name in ("float8_e4m3fn", "float8_e5m2"):
        dtype = getattr(torch, name)
        with pytest.raises(asc.DTypeError, match="release surface"):
            asc.backend("torch", dtype=dtype)
        with pytest.raises(asc.DTypeError, match="release surface"):
            selected.xp.asarray([1], dtype=dtype)
        assert not asc.is_array(torch.asarray([1], dtype=dtype))
        with pytest.raises(asc.RandomStateError, match="release surface"):
            random.uniform(
                (1,),
                state=random.random_state(1, backend="torch"),
                dtype=dtype,
            )


@pytest.mark.parametrize("backend", ("numpy", "torch", "jax"))
def test_random_dtypes_require_canonical_native_objects(backend: str) -> None:
    state = random.random_state(1, backend=backend)
    calls = (
        lambda: random.uniform((1,), state=state, dtype="float32"),
        lambda: random.normal((1,), state=state, dtype="float32"),
        lambda: random.randint(0, 2, (1,), state=state, dtype="int32"),
        lambda: random.uniform((1,), state=state, dtype=_MalformedDType()),
    )

    for call in calls:
        with pytest.raises(asc.RandomStateError):
            call()


def test_invalid_context_dtypes_do_not_import_optional_backends() -> None:
    program = """
import sys
import asc
for name in ("torch", "jax"):
    assert name not in sys.modules
    try:
        asc.ArrayContext(name, dtype=object())
    except asc.ContextError:
        pass
    else:
        raise AssertionError(name)
    assert name not in sys.modules
"""

    subprocess.run([sys.executable, "-c", program], check=True)


def test_public_state_and_extension_records_validate_field_types() -> None:
    from asc.backends import _state

    version = importlib.metadata.version("numpy")
    valid_key = _state.CounterKey("numpy", seed=1)
    for arguments in (
        ("bad", valid_key, version),
        ("numpy", object(), version),
        ("numpy", valid_key, ""),
        ("numpy", valid_key, "0.invalid"),
    ):
        with pytest.raises(asc.RandomStateError):
            asc.RandomState(*arguments)  # type: ignore[arg-type]
    with pytest.raises(asc.ContextError, match=r"asc\.RandomState"):
        asc.ArrayContext("numpy", random_state=object())
    with pytest.raises(asc.ContextError, match="name"):
        asc.ExtensionHandle(3, object())  # type: ignore[arg-type]


@pytest.mark.backend("jax")
def test_jax_random_state_authenticates_tracers_and_trusted_context() -> None:
    import jax

    class FakeTracer:
        dtype = "key<fry>"
        device = jax.devices("cpu")[0]

    version = importlib.metadata.version("jax")
    with pytest.raises(asc.RandomStateError, match="invalid JAX key"):
        asc.RandomState("jax", FakeTracer(), version)

    def construct_state(key: object) -> object:
        return asc.RandomState("jax", key, version).key

    with pytest.raises(asc.RandomStateError, match="CPU-pinned transform"):
        jax.make_jaxpr(construct_state)(jax.random.key(1))


@pytest.mark.parametrize("backend", ("numpy", "torch", "jax"))
def test_linalg_array_operands_are_never_implicitly_coerced(
    backend: str,
) -> None:
    selected = asc.backend(backend)
    vector = selected.xp.asarray(
        [1.0, 2.0], dtype=selected.xp.float32, device=selected.device
    )

    for call in (
        lambda: selected.linalg.vecdot(vector, [1.0, 2.0]),
        lambda: asc.linalg.kron(vector, 2.0),
        lambda: asc.linalg.einsum("i,->i", vector, 2.0),
    ):
        with pytest.raises(asc.NamespaceError, match="native array"):
            call()


@pytest.mark.parametrize("backend", ("numpy", "torch", "jax"))
def test_boolean_padding_uses_a_false_default(backend: str) -> None:
    selected = asc.backend(backend)
    values = selected.xp.asarray(
        [True, False], dtype=selected.xp.bool, device=selected.device
    )

    result = asc.ops.pad(values, 1)

    numpy.testing.assert_array_equal(
        numpy.asarray(result), [False, True, False, False]
    )


@pytest.mark.parametrize("backend", ("numpy", "torch", "jax"))
def test_scalar_padding_rejects_negative_widths(backend: str) -> None:
    selected = asc.backend(backend)
    scalar = selected.xp.asarray(3, device=selected.device)

    with pytest.raises(ValueError, match="non-negative"):
        asc.ops.pad(scalar, -1)


@pytest.mark.parametrize("backend", ("numpy", "torch", "jax"))
def test_metrics_and_comparisons_avoid_representable_range_failures(
    backend: str,
) -> None:
    selected = asc.backend(backend)
    xp = selected.xp

    def array(values: list[float]) -> object:
        return xp.asarray(values, dtype=xp.float32, device=selected.device)

    large_rmse = asc.metrics.root_mean_squared_error(
        array([3e20]), array([0.0])
    )
    small_rmse = asc.metrics.root_mean_squared_error(
        array([1e-30]), array([0.0])
    )
    relative = asc.metrics.relative_l2_error(array([1e20]), array([1e-10]))
    score = asc.metrics.r2_score(array([1e20, 1e20]), array([1e-10, 2e-10]))
    close = asc.ops.isclose(array([1.0]), array([2.0]), rtol=1e40, atol=0.0)
    maximum = float(xp.finfo(xp.float32).max)
    opposite = array([-maximum, maximum])
    reversed_opposite = array([maximum, -maximum])
    extreme_relative = asc.metrics.relative_l2_error(
        opposite, reversed_opposite
    )
    extreme_score = asc.metrics.r2_score(opposite, reversed_opposite)
    extreme_close = asc.ops.isclose(
        array([-maximum]), array([maximum]), rtol=2.0, atol=0.0
    )

    assert float(large_rmse) == pytest.approx(3e20, rel=1e-6)
    assert float(small_rmse) == pytest.approx(1e-30, rel=1e-6)
    assert float(relative) == pytest.approx(1e30, rel=1e-6)
    assert float(score) == -float("inf")
    assert bool(xp.all(close))
    assert float(extreme_relative) == pytest.approx(2.0)
    assert float(extreme_score) == pytest.approx(-3.0)
    assert bool(xp.all(extreme_close))


@pytest.mark.parametrize("backend", ("numpy", "torch", "jax"))
def test_metrics_propagate_nan_inputs(backend: str) -> None:
    selected = asc.backend(backend)
    xp = selected.xp
    prediction = xp.asarray(
        [float("nan")], dtype=xp.float32, device=selected.device
    )
    target = xp.zeros_like(prediction)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        results = (
            asc.metrics.mean_absolute_error(prediction, target),
            asc.metrics.mean_squared_error(prediction, target),
            asc.metrics.root_mean_squared_error(prediction, target),
            asc.metrics.relative_l2_error(prediction, target),
            asc.metrics.r2_score(prediction, target),
        )

    assert all(bool(xp.isnan(result)) for result in results)


@pytest.mark.backend("torch")
@pytest.mark.backend("jax")
@pytest.mark.parametrize("backend", ("numpy", "torch", "jax"))
def test_error_metrics_preserve_infinite_residuals(backend: str) -> None:
    selected = asc.backend(backend)
    xp = selected.xp
    prediction = xp.asarray(
        [float("inf"), 1.0],
        dtype=xp.float32,
        device=selected.device,
    )
    target = xp.asarray([0.0, 1.0], dtype=xp.float32, device=selected.device)

    absolute = asc.metrics.mean_absolute_error(
        prediction,
        target,
        reduction="none",
    )
    squared = asc.metrics.mean_squared_error(
        prediction,
        target,
        reduction="none",
    )
    root_mean_square = asc.metrics.root_mean_squared_error(prediction, target)

    assert absolute.tolist() == [float("inf"), 0.0]
    assert squared.tolist() == [float("inf"), 0.0]
    assert bool(xp.isinf(root_mean_square))


def test_numpy_metrics_return_infinity_without_overflow_warnings() -> None:
    maximum = numpy.finfo(numpy.float32).max
    smallest = numpy.nextafter(numpy.float32(0), numpy.float32(1))
    prediction = numpy.asarray([maximum], dtype=numpy.float32)
    opposite = numpy.asarray([-maximum], dtype=numpy.float32)
    tiny_target = numpy.asarray([smallest], dtype=numpy.float32)

    rmse = asc.metrics.root_mean_squared_error(prediction, opposite)
    relative = asc.metrics.relative_l2_error(prediction, tiny_target)
    score = asc.metrics.r2_score(
        numpy.asarray([maximum, 0], dtype=numpy.float32),
        numpy.asarray([smallest, -smallest], dtype=numpy.float32),
    )

    assert numpy.isposinf(rmse)
    assert numpy.isposinf(relative)
    assert numpy.isneginf(score)


@pytest.mark.parametrize("backend", ("numpy", "torch", "jax"))
def test_r2_normalizes_extreme_target_mean(backend: str) -> None:
    selected = asc.backend(backend)
    xp = selected.xp
    maximum = float(xp.finfo(xp.float32).max)
    target = xp.asarray(
        [maximum, maximum / 2],
        dtype=xp.float32,
        device=selected.device,
    )

    score = asc.metrics.r2_score(target, target)

    assert float(score) == 1.0


@pytest.mark.parametrize("backend", ("numpy", "torch", "jax"))
def test_float16_metrics_do_not_overflow_the_reduction_count(
    backend: str,
) -> None:
    selected = asc.backend(backend)
    xp = selected.xp
    ones = xp.ones((100_000,), dtype=xp.float16, device=selected.device)
    zeros = xp.zeros_like(ones)
    target = xp.asarray(
        numpy.resize(
            numpy.asarray([0.0, 1.0], dtype=numpy.float16), 100_000
        ).tolist(),
        dtype=xp.float16,
        device=selected.device,
    )

    rmse = asc.metrics.root_mean_squared_error(ones, zeros)
    relative = asc.metrics.relative_l2_error(zeros, ones)
    score = asc.metrics.r2_score(zeros, target)

    assert float(rmse) == pytest.approx(1.0)
    assert float(relative) == pytest.approx(1.0)
    assert float(score) == pytest.approx(-1.0)


def test_sparse_float16_metrics_do_not_underflow_before_the_root() -> None:
    count = 33_554_432
    target = numpy.zeros((count,), dtype=numpy.float16)
    target[0] = 1.0
    prediction = numpy.zeros_like(target)

    rmse = asc.metrics.root_mean_squared_error(prediction, target)
    relative = asc.metrics.relative_l2_error(prediction, target)

    assert float(rmse) == pytest.approx(1.0 / numpy.sqrt(count), rel=2e-3)
    assert float(relative) == pytest.approx(1.0)


@pytest.mark.parametrize("backend", ("numpy", "torch", "jax"))
def test_exact_metrics_restore_zero_residuals(backend: str) -> None:
    selected = asc.backend(backend)
    target = selected.xp.asarray(
        [1.0, 2.0], dtype=selected.xp.float32, device=selected.device
    )

    assert float(asc.metrics.relative_l2_error(target, target)) == 0.0
    assert float(asc.metrics.r2_score(target, target)) == 1.0


@pytest.mark.parametrize("backend", ("numpy", "torch", "jax"))
def test_mae_and_mse_scale_before_reduction(backend: str) -> None:
    selected = asc.backend(backend)
    xp = selected.xp
    maximum = float(xp.finfo(xp.float32).max)
    positive = xp.asarray(
        [maximum, 0.0, 0.0, 0.0],
        dtype=xp.float32,
        device=selected.device,
    )
    opposite = xp.asarray(
        [-maximum, 0.0, 0.0, 0.0],
        dtype=xp.float32,
        device=selected.device,
    )
    sparse = xp.asarray(
        [1e20, *([0.0] * 999)],
        dtype=xp.float32,
        device=selected.device,
    )

    mae = asc.metrics.mean_absolute_error(positive, opposite)
    mse = asc.metrics.mean_squared_error(sparse, xp.zeros_like(sparse))

    assert float(mae) == pytest.approx(maximum / 2, rel=1e-6)
    assert float(mse) == pytest.approx(1e37, rel=1e-6)


@pytest.mark.parametrize("backend", ("numpy", "torch"))
def test_metrics_promote_mixed_floating_operands_before_widening(
    backend: str,
) -> None:
    selected = asc.backend(backend)
    xp = selected.xp
    narrow_zero = xp.asarray([0.0], dtype=xp.float16, device=selected.device)
    wide_small = xp.asarray([1e-100], dtype=xp.float64, device=selected.device)

    rmse = asc.metrics.root_mean_squared_error(narrow_zero, wide_small)
    score = asc.metrics.r2_score(wide_small, narrow_zero)

    assert rmse.dtype == xp.float64
    assert float(rmse) == pytest.approx(1e-100)
    assert score.dtype == xp.float64
    assert float(score) == 0.0


@pytest.mark.parametrize("backend", ("numpy", "torch", "jax"))
def test_isclose_preserves_small_tolerance_boundaries(backend: str) -> None:
    selected = asc.backend(backend)
    xp = selected.xp
    first = xp.asarray(
        [float(numpy.nextafter(numpy.float32(0), numpy.float32(1)))],
        dtype=xp.float32,
        device=selected.device,
    )
    second = xp.asarray(
        [-float(numpy.finfo(numpy.float32).tiny)],
        dtype=xp.float32,
        device=selected.device,
    )

    relative_result = asc.ops.isclose(first, second, rtol=1.0, atol=0.0)
    absolute_result = asc.ops.isclose(
        first,
        second,
        rtol=0.0,
        atol=float(numpy.finfo(numpy.float32).tiny),
    )

    assert not bool(xp.any(relative_result))
    assert not bool(xp.any(absolute_result))


@pytest.mark.parametrize("backend", ("numpy", "torch", "jax"))
def test_isclose_preserves_mixed_tolerance_boundaries(backend: str) -> None:
    selected = asc.backend(backend)
    xp = selected.xp
    first = xp.asarray([-1.0], dtype=xp.float32, device=selected.device)
    second = xp.asarray([1e20], dtype=xp.float32, device=selected.device)

    result = asc.ops.isclose(first, second, rtol=1.0, atol=0.5)

    assert not bool(xp.any(result))


@pytest.mark.parametrize(
    ("backend", "dtype"),
    (
        ("numpy", "float32"),
        ("numpy", "float64"),
        ("torch", "float32"),
        ("torch", "float64"),
        ("jax", "float32"),
    ),
)
def test_isclose_accepts_exact_absolute_tolerance_boundary(
    backend: str, dtype: str
) -> None:
    selected = asc.backend(backend)
    xp = selected.xp
    numpy_dtype = getattr(numpy, dtype)
    first_value = numpy_dtype(1.0)
    second_value = numpy.nextafter(first_value, numpy_dtype(2.0))
    tolerance = float(second_value - first_value)
    first = xp.asarray([float(first_value)], dtype=getattr(xp, dtype))
    second = xp.asarray([float(second_value)], dtype=getattr(xp, dtype))

    result = asc.ops.isclose(first, second, rtol=0.0, atol=tolerance)

    assert bool(xp.all(result))


@pytest.mark.parametrize("backend", ("numpy", "torch", "jax"))
def test_orthogonal_rejects_cpu_qr_dtypes_portably(backend: str) -> None:
    selected = asc.backend(backend)

    with pytest.raises(asc.RandomStateError, match="portable CPU QR"):
        asc.random.orthogonal(
            (2, 2),
            state=asc.random_state(1, backend=backend),
            dtype=selected.xp.float16,
        )


@pytest.mark.backend("jax")
def test_jax_transforms_reject_foreign_native_array_arguments() -> None:
    selected = asc.backend("jax")
    jax_value = selected.xp.asarray([1.0, 2.0], dtype=selected.xp.float32)
    numpy_value = numpy.asarray([3.0, 4.0], dtype=numpy.float32)

    def scalar(left: object, right: object) -> object:
        return selected.xp.sum(left + right)

    transformed = (
        asc.jit(scalar, backend="jax"),
        asc.grad(scalar, backend="jax"),
        asc.vmap(lambda left, right: left + right, backend="jax"),
    )
    for function in transformed:
        with pytest.raises(asc.MixedBackendError, match="mixed backends"):
            function(jax_value, numpy_value)


@pytest.mark.backend("torch")
def test_torch_transforms_reject_foreign_native_array_arguments() -> None:
    import torch

    torch_value = torch.asarray([1.0, 2.0], dtype=torch.float32)
    numpy_value = numpy.asarray([3.0, 4.0], dtype=numpy.float32)

    def scalar(left: torch.Tensor, _right: object) -> torch.Tensor:
        return torch.sum(left * left)

    transformed = (
        asc.grad(scalar, backend="torch"),
        asc.vmap(lambda left, _right: left * left, backend="torch"),
    )
    for function in transformed:
        with pytest.raises(asc.MixedBackendError, match="mixed backends"):
            function(torch_value, numpy_value)


@pytest.mark.backend("torch")
@pytest.mark.backend("jax")
def test_backend_bound_numerical_facades_reject_foreign_arrays() -> None:
    import torch

    numpy_backend = asc.backend("numpy")
    torch_value = torch.asarray([[2.0, 0.0], [0.0, 3.0]], dtype=torch.float32)
    with pytest.raises(asc.MixedBackendError, match="facade"):
        numpy_backend.linalg.det(torch_value)

    jax_backend = asc.backend("jax")
    numpy_value = numpy.asarray([1.0, 2.0], dtype=numpy.float32)
    with pytest.raises(asc.MixedBackendError, match="facade"):
        jax_backend.fft.fft(numpy_value)


@pytest.mark.backend("jax")
def test_numpy_namespace_rejects_jax_arrays_before_dispatch() -> None:
    jax = asc.backend("jax")
    foreign = jax.xp.asarray([1.0, 2.0], dtype=jax.xp.float32)

    with pytest.raises(asc.MixedBackendError, match=r"namespace\.add"):
        asc.backend("numpy").xp.add(foreign, foreign)


@pytest.mark.backend("torch")
@pytest.mark.backend("jax")
def test_asarray_rejects_foreign_arrays_in_custom_sequences() -> None:
    import torch

    class Sequence(collections.abc.Sequence[object]):
        def __init__(self, value: object) -> None:
            self.value = value

        def __len__(self) -> int:
            return 1

        @typing.overload
        def __getitem__(self, index: int) -> object: ...

        @typing.overload
        def __getitem__(
            self, index: slice
        ) -> collections.abc.Sequence[object]: ...

        def __getitem__(
            self, index: int | slice
        ) -> object | collections.abc.Sequence[object]:
            if isinstance(index, slice):
                return (self.value,)[index]
            if index == 0:
                return self.value
            raise IndexError(index)

    nested = Sequence(torch.tensor(2.0))
    for destination in ("numpy", "jax"):
        selected = asc.backend(destination)
        with pytest.raises(asc.MixedBackendError, match="cannot consume"):
            selected.xp.asarray(nested)
        with pytest.raises(asc.ContextError, match="nested native arrays"):
            selected.asarray(nested)


@pytest.mark.backend("jax")
def test_jax_namespace_creation_defaults_to_selected_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import jax.numpy

    selected = asc.backend("jax")
    original = jax.numpy.asarray
    observed: dict[str, object] = {}

    def recording_asarray(*args: object, **kwargs: object) -> object:
        observed["device"] = kwargs.get("device")
        return original(*args, **kwargs)

    monkeypatch.setattr(jax.numpy, "asarray", recording_asarray)

    result = selected.xp.asarray([1.0], dtype=selected.xp.float32)

    assert observed["device"] is selected.device
    result_device = result.device
    if callable(result_device):
        result_device = result_device()
    assert result_device.platform == "cpu"


@pytest.mark.parametrize("backend", ("torch", "jax"))
def test_asarray_requires_explicit_conversion_for_numpy_scalars(
    backend: str,
) -> None:
    selected = asc.backend(backend)
    scalar = numpy.float32(1.25)

    with pytest.raises(asc.MixedBackendError, match="cannot consume"):
        selected.xp.asarray(scalar)
    with pytest.raises(asc.ConversionError, match="cross-backend"):
        selected.asarray(scalar)
    with pytest.raises(asc.ContextError, match="nested native arrays"):
        selected.asarray([scalar, numpy.float32(2.5)])

    converted = selected.asarray(scalar, copy=True)

    assert asc.backend_of(converted) == backend
    numpy.testing.assert_array_equal(numpy.asarray(converted), 1.25)


@pytest.mark.backend("torch")
@pytest.mark.parametrize(
    ("name", "arguments"),
    (
        ("reshape", (numpy.asarray([1.0]), (1,))),
        (
            "take",
            (numpy.asarray([1.0]), numpy.asarray([0], dtype=numpy.int32)),
        ),
        (
            "take_along_axis",
            (numpy.asarray([1.0]), numpy.asarray([0], dtype=numpy.int32)),
        ),
        ("unique_all", (numpy.asarray([1.0]),)),
        ("unique_counts", (numpy.asarray([1.0]),)),
        ("unique_inverse", (numpy.asarray([1.0]),)),
        ("unique_values", (numpy.asarray([1.0]),)),
    ),
)
def test_torch_namespace_overrides_reject_foreign_arrays(
    name: str,
    arguments: tuple[object, ...],
) -> None:
    with pytest.raises(asc.MixedBackendError, match=rf"namespace\.{name}"):
        getattr(asc.backend("torch").xp, name)(*arguments)


@pytest.mark.backend("torch")
def test_torch_namespace_overrides_reject_non_cpu_arrays() -> None:
    import torch

    value = torch.empty((2,), device="meta")

    with pytest.raises(asc.CapabilityNotSupportedError, match="dense CPU"):
        asc.backend("torch").xp.reshape(value, (1, 2))


@pytest.mark.backend("jax")
@pytest.mark.backend("torch")
def test_torch_namespace_overrides_validate_array_valued_options() -> None:
    import torch

    selected = asc.backend("torch")
    value = torch.asarray([1.0])
    index = torch.asarray([0], dtype=torch.int32)
    foreign_boolean = asc.backend("jax").xp.asarray(True)
    foreign_axis = asc.backend("jax").xp.asarray(0)

    with pytest.raises(asc.MixedBackendError, match=r"namespace\.reshape"):
        selected.xp.reshape(value, (1,), copy=foreign_boolean)
    with pytest.raises(asc.MixedBackendError, match=r"namespace\.take"):
        selected.xp.take(value, index, axis=foreign_axis)


@pytest.mark.backend("torch")
def test_namespace_astype_rejects_non_cpu_device_requests() -> None:
    selected = asc.backend("torch")
    value = selected.xp.asarray([1.0], dtype=selected.xp.float32)

    with pytest.raises(asc.DeviceError, match="CPU"):
        selected.xp.astype(value, selected.xp.float32, device="meta")


@pytest.mark.backend("torch")
def test_namespace_from_dlpack_rejects_non_cpu_producers() -> None:
    import torch

    producer = torch.empty((1,), device="meta")

    with pytest.raises(asc.UnsupportedCapabilityError, match="dense CPU"):
        asc.backend("numpy").xp.from_dlpack(producer)


@pytest.mark.backend("jax")
def test_namespace_special_cases_still_validate_option_arrays() -> None:
    source = numpy.asarray([1.0], dtype=numpy.float32)
    foreign_option = asc.backend("jax").xp.asarray(False)

    with pytest.raises(asc.MixedBackendError, match=r"namespace\.from_dlpack"):
        asc.backend("numpy").xp.from_dlpack(source, copy=foreign_option)
    with pytest.raises(asc.MixedBackendError, match=r"namespace\.asarray"):
        asc.backend("jax").xp.asarray([1.0], copy=numpy.bool_(False))


@pytest.mark.backend("jax")
def test_jax_inferred_numpy_scalar_dtypes_do_not_narrow() -> None:
    program = """
import numpy as np
import asc

selected = asc.backend("jax")
value = np.uint64(2**32 + 1)
try:
    selected.xp.asarray(value)
except asc.MixedBackendError:
    pass
else:
    raise AssertionError("foreign NumPy scalar was silently converted")
try:
    asc.data.default_convert(value, backend="jax")
except asc.ConversionError:
    pass
else:
    raise AssertionError("default_convert silently narrowed a NumPy scalar")
explicit = asc.backend("jax", dtype=selected.xp.uint32).asarray(
    value, copy=True
)
assert explicit.dtype == selected.xp.uint32
"""
    environment = dict(os.environ)
    environment["JAX_ENABLE_X64"] = "0"
    environment["JAX_PLATFORMS"] = "cpu"

    subprocess.run(
        [sys.executable, "-c", program],
        check=True,
        env=environment,
    )


@pytest.mark.backend("jax")
def test_namespace_from_dlpack_rejects_unavailable_dtype_before_import() -> (
    None
):
    program = """
import numpy as np
import asc

class Producer:
    def __init__(self):
        self.value = np.asarray([2**63], dtype=np.uint64)
        self.dtype = self.value.dtype
        self.consumed = False

    def __dlpack_device__(self):
        return (1, 0)

    def __dlpack__(self, *args, **kwargs):
        self.consumed = True
        return self.value.__dlpack__(*args, **kwargs)

producer = Producer()
try:
    asc.backend("jax").xp.from_dlpack(producer)
except asc.DTypeError:
    pass
else:
    raise AssertionError("uint64 DLPack producer was silently narrowed")
assert not producer.consumed
"""
    environment = dict(os.environ)
    environment["JAX_ENABLE_X64"] = "0"
    environment["JAX_PLATFORMS"] = "cpu"

    subprocess.run(
        [sys.executable, "-c", program],
        check=True,
        env=environment,
    )


@pytest.mark.backend("jax")
def test_jax_namespace_rejects_positional_dtype_requests() -> None:
    program = """
import numpy as np
import asc

selected = asc.backend("jax")
operations = (
    lambda: selected.xp.asarray([1.0], selected.xp.float64),
    lambda: selected.xp.zeros((1,), selected.xp.float64),
)
for operation in operations:
    try:
        operation()
    except TypeError as exception:
        assert "positional" in str(exception)
    else:
        raise AssertionError("positional dtype extension was accepted")
for operation in (
    lambda: selected.xp.asarray([1.0], dtype=selected.xp.float64),
    lambda: selected.xp.zeros((1,), dtype=selected.xp.float64),
):
    try:
        operation()
    except asc.DTypeError:
        pass
    else:
        raise AssertionError("keyword float64 request was narrowed")
value = 2**32 + 1
converted = selected.xp.asarray(value, dtype=selected.xp.float32)
assert converted.dtype == selected.xp.float32
assert float(converted) == float(np.float32(value))
"""
    environment = dict(os.environ)
    environment["JAX_ENABLE_X64"] = "0"
    environment["JAX_PLATFORMS"] = "cpu"

    subprocess.run(
        [sys.executable, "-c", program],
        check=True,
        env=environment,
    )


@pytest.mark.backend("jax")
def test_from_dlpack_requires_dtype_provenance_for_narrow_jax() -> None:
    program = """
import numpy as np
import asc

class Producer:
    def __init__(self):
        self.value = np.asarray([1.25], dtype=np.float64)
        self.consumed = False

    def __dlpack_device__(self):
        return (1, 0)

    def __dlpack__(self, *args, **kwargs):
        self.consumed = True
        return self.value.__dlpack__(*args, **kwargs)

for dtype in (None, asc.backend("jax").xp.float32):
    producer = Producer()
    try:
        asc.from_dlpack(producer, "jax", dtype=dtype, copy=True)
    except asc.ConversionError as exception:
        assert "dtype metadata" in str(exception)
    else:
        raise AssertionError("untyped wide producer was silently narrowed")
    assert not producer.consumed
"""
    environment = dict(os.environ)
    environment["JAX_ENABLE_X64"] = "0"
    environment["JAX_PLATFORMS"] = "cpu"

    subprocess.run(
        [sys.executable, "-c", program],
        check=True,
        env=environment,
    )


@pytest.mark.backend("torch")
def test_namespace_from_dlpack_accepts_metadata_free_cpu_producers() -> None:
    class Producer:
        def __init__(self) -> None:
            self.value = numpy.asarray([1.0], dtype=numpy.float32)
            self.consumed = False

        def __dlpack_device__(self) -> tuple[int, int]:
            return (1, 0)

        def __dlpack__(self, *args: object, **kwargs: object) -> object:
            self.consumed = True
            return self.value.__dlpack__(*args, **kwargs)

    for destination in ("numpy", "torch"):
        producer = Producer()
        result = asc.backend(destination).xp.from_dlpack(producer)

        assert producer.consumed
        assert asc.backend_of(result) == destination
        numpy.testing.assert_array_equal(numpy.asarray(result), [1.0])


@pytest.mark.backend("torch")
def test_opaque_torch_dlpack_imports_copy_before_unsafe_layout() -> None:
    program = """
import numpy as np
import asc

class Producer:
    def __init__(self):
        self.value = np.arange(5, dtype=np.float32)[::-1]
        self.dtype = self.value.dtype
        self.consumed = False

    def __dlpack_device__(self):
        return (1, 0)

    def __dlpack__(self, *args, **kwargs):
        self.consumed = True
        return self.value.__dlpack__(*args, **kwargs)

for copy in (None, True):
    producer = Producer()
    result = asc.from_dlpack(producer, "torch", copy=copy)
    np.testing.assert_array_equal(np.asarray(result), [4, 3, 2, 1, 0])
    assert producer.consumed

producer = Producer()
try:
    asc.from_dlpack(producer, "torch", copy=False)
except asc.ConversionError:
    pass
else:
    raise AssertionError("opaque Torch no-copy import was not rejected")
assert not producer.consumed
"""
    subprocess.run([sys.executable, "-c", program], check=True)


@pytest.mark.backend("torch")
def test_torch_namespace_binds_keyword_dlpack_producer() -> None:
    program = """
import numpy as np
import torch
import asc

class Producer:
    def __init__(self):
        self.value = np.arange(5, dtype=np.float32)[::-1]
        self.dtype = self.value.dtype

    def __dlpack_device__(self):
        return (1, 0)

    def __dlpack__(self, *args, **kwargs):
        return self.value.__dlpack__(*args, **kwargs)

result = asc.backend("torch").xp.from_dlpack(ext_tensor=Producer())
np.testing.assert_array_equal(np.asarray(result), [4, 3, 2, 1, 0])

graph = torch.tensor([1.0], requires_grad=True)
try:
    asc.backend("torch").xp.from_dlpack(ext_tensor=graph)
except asc.ConversionError:
    pass
else:
    raise AssertionError("keyword graph producer was detached")
"""
    subprocess.run([sys.executable, "-c", program], check=True)


@pytest.mark.backend("torch")
def test_dlpack_internal_capsule_exemption_cannot_be_spoofed() -> None:
    program = """
import numpy as np
import asc

class Producer:
    _asc_preexported_dlpack = True

    def __init__(self):
        self.value = np.arange(5, dtype=np.float32)[::-1]
        self.dtype = self.value.dtype

    def __dlpack_device__(self):
        return (1, 0)

    def __dlpack__(self, *args, **kwargs):
        return self.value.__dlpack__(*args, **kwargs)

for call in (
    lambda: asc.from_dlpack(Producer(), "torch", copy=True),
    lambda: asc.backend("torch").xp.from_dlpack(Producer(), copy=None),
):
    result = call()
    np.testing.assert_array_equal(np.asarray(result), [4, 3, 2, 1, 0])
"""
    subprocess.run([sys.executable, "-c", program], check=True)


@pytest.mark.backend("torch")
def test_legacy_opaque_torch_producer_is_rejected_before_export() -> None:
    program = """
import numpy as np
import asc

class Producer:
    def __init__(self):
        self.value = np.arange(5, dtype=np.float32)[::-1]
        self.dtype = self.value.dtype
        self.consumed = False

    def __dlpack_device__(self):
        return (1, 0)

    def __dlpack__(self):
        self.consumed = True
        return self.value.__dlpack__()

for copy in (None, True):
    producer = Producer()
    try:
        asc.from_dlpack(producer, "torch", copy=copy)
    except asc.ConversionError as exception:
        assert "copy keyword" in str(exception)
    else:
        raise AssertionError("unsafe legacy producer was imported")
    assert not producer.consumed
"""
    subprocess.run([sys.executable, "-c", program], check=True)


@pytest.mark.backend("jax")
def test_jax_wide_integer_conversion_casts_before_dlpack_import() -> None:
    program = """
import numpy as np
import asc

selected = asc.backend("jax", dtype=asc.backend("jax").xp.float32)
value = np.int64(2**40 + 3)
expected = np.float32(value)
converted = asc.convert_array(np.asarray([value]), selected, copy=True)
collated = asc.data.default_collate([value, value], backend=selected)
direct = asc.from_dlpack(
    np.asarray([value]), selected, dtype=selected.dtype, copy=True
)
np.testing.assert_array_equal(np.asarray(converted), [expected])
np.testing.assert_array_equal(np.asarray(collated), [expected, expected])
np.testing.assert_array_equal(np.asarray(direct), [expected])
"""
    environment = dict(os.environ)
    environment["JAX_ENABLE_X64"] = "0"
    environment["JAX_PLATFORMS"] = "cpu"
    subprocess.run([sys.executable, "-c", program], check=True, env=environment)


@pytest.mark.backend("torch")
def test_legacy_dlpack_raw_capsule_retry_omits_device_defaults() -> None:
    class LegacyProducer:
        def __init__(self) -> None:
            self.value = numpy.asarray([1.0], dtype=numpy.float32)
            self.dtype = self.value.dtype

        def __dlpack_device__(self) -> tuple[int, int]:
            return (1, 0)

        def __dlpack__(
            self,
            *,
            max_version: tuple[int, int] | None = None,
            **_kwargs: object,
        ) -> object:
            if max_version is not None:
                raise NotImplementedError("legacy producer")
            return self.value.__dlpack__()

    producer = LegacyProducer()
    result = asc.from_dlpack(producer, "torch", copy=True)

    assert asc.backend_of(result) == "torch"
    numpy.testing.assert_array_equal(numpy.asarray(result), [1.0])
    producer.value[0] = 2.0
    numpy.testing.assert_array_equal(numpy.asarray(result), [1.0])


def test_legacy_dlpack_capsule_is_wrapped_for_numpy_import() -> None:
    class LegacyProducer:
        def __init__(self) -> None:
            self.value = numpy.asarray([1.0], dtype=numpy.float32)
            self.dtype = self.value.dtype

        def __dlpack_device__(self) -> tuple[int, int]:
            return (1, 0)

        def __dlpack__(
            self,
            *,
            max_version: tuple[int, int] | None = None,
            **_kwargs: object,
        ) -> object:
            if max_version is not None:
                raise NotImplementedError("legacy producer")
            return self.value.__dlpack__()

    producer = LegacyProducer()
    result = asc.from_dlpack(producer, "numpy", copy=True)

    assert asc.backend_of(result) == "numpy"
    numpy.testing.assert_array_equal(result, [1.0])
    producer.value[0] = 2.0
    numpy.testing.assert_array_equal(result, [1.0])


@pytest.mark.parametrize("enabled", ("0", "1"))
@pytest.mark.backend("jax")
def test_jax_float64_capability_tracks_x64_without_importing_jax(
    enabled: str,
) -> None:
    program = f"""
import sys
import asc
assert "jax" not in sys.modules
info = asc.backend_info("jax")
expected = bool({enabled!r} == "1")
assert (asc.Capability.FLOAT64 in info.capabilities) is expected
assert ("float64" in info.dtypes) is expected
assert ("complex128" in info.dtypes) is expected
assert asc.has_capability("jax", asc.Capability.FLOAT64) is expected
assert "jax" not in sys.modules
"""
    environment = dict(os.environ)
    environment["JAX_ENABLE_X64"] = enabled
    environment["JAX_PLATFORMS"] = "cpu"
    subprocess.run(
        [sys.executable, "-c", program],
        check=True,
        env=environment,
    )


def test_numpy_no_copy_accepts_a_distinct_shared_memmap_view(
    tmp_path: pathlib.Path,
) -> None:
    path = tmp_path / "mapped.npy"
    numpy.save(path, numpy.arange(4, dtype=numpy.float32))
    source = numpy.load(path, mmap_mode="r")

    result = asc.convert_array(source, "numpy", copy=asc.CopyPolicy.NEVER)

    assert result is not source
    assert numpy.shares_memory(source, result)
    numpy.testing.assert_array_equal(result, source)


def test_dependency_floor_job_installs_the_pinned_uv_version() -> None:
    repository = pathlib.Path(__file__).parents[2]
    workflow = (repository / ".github/workflows/ci.yml").read_text()
    floor_job = workflow.split("\n  dependency-floor:", maxsplit=1)[1]
    floor_job = floor_job.split("\n  platform-smoke:", maxsplit=1)[0]

    assert "astral-sh/setup-uv@" in floor_job
    assert "version: ${{ env.UV_VERSION }}" in floor_job


def test_ci_runs_full_independent_backend_profiles() -> None:
    repository = pathlib.Path(__file__).parents[2]
    workflow = (repository / ".github/workflows/ci.yml").read_text()
    profile_job = workflow.split("\n  independent-backends:", maxsplit=1)[1]
    profile_job = profile_job.split("\n  dependency-floor:", maxsplit=1)[0]

    assert "profile: base" in profile_job
    assert "profile: torch" in profile_job
    assert 'extra: "--extra torch"' in profile_job
    assert "profile: jax" in profile_job
    assert 'extra: "--extra jax"' in profile_job
    assert "ASC_TEST_PROFILE: ${{ matrix.profile }}" in profile_job
    assert "uv run pytest --no-cov" in profile_job
    assert "--all-extras" not in profile_job


def test_active_backend_profile_installs_only_declared_array_extras() -> None:
    profile = os.environ.get("ASC_TEST_PROFILE", "all")
    expected = {
        "base": {"jax": False, "torch": False},
        "jax": {"jax": True, "torch": False},
        "torch": {"jax": False, "torch": True},
        "all": {"jax": True, "torch": True},
    }[profile]

    for package, installed in expected.items():
        assert (importlib.util.find_spec(package) is not None) is installed


@pytest.mark.backend("jax")
def test_jax_unavailable_wide_dtypes_fail_before_creation() -> None:
    program = """
import jax.numpy as jnp
import asc
for dtype in (
    jnp.int64,
    jnp.uint64,
    jnp.float64,
    jnp.complex128,
    jnp.dtype("int64"),
    jnp.dtype("uint64"),
    jnp.dtype("float64"),
    int,
    float,
    complex,
):
    try:
        asc.backend("jax", dtype=dtype)
    except asc.DTypeError:
        pass
    else:
        raise AssertionError(dtype)

selected = asc.backend("jax")
value = jnp.asarray([1], dtype=jnp.int32)
namespace = asc.array_namespace(value)
assert namespace is selected.xp
for function in (
    lambda: selected.xp.asarray([1], dtype=selected.xp.float64),
    lambda: selected.xp.arange(2, dtype=selected.xp.int64),
    lambda: selected.xp.zeros((1,), dtype=selected.xp.uint64),
    lambda: namespace.astype(value, namespace.complex128),
    lambda: namespace.argsort(value, dtype=namespace.int64),
    lambda: namespace.sum(value, dtype=namespace.float64),
    lambda: namespace.prod(value, dtype=namespace.float64),
    lambda: namespace.cumulative_sum(value, dtype=namespace.int64),
    lambda: namespace.cumulative_prod(value, dtype=namespace.int64),
    lambda: namespace.mean(value, dtype=namespace.float64),
    lambda: namespace.std(value, dtype=namespace.float64),
    lambda: namespace.var(value, dtype=namespace.float64),
    lambda: namespace.stack((value,), dtype=namespace.int64),
):
    try:
        function()
    except asc.DTypeError:
        pass
    else:
        raise AssertionError(function)
"""
    environment = dict(os.environ)
    environment["JAX_ENABLE_X64"] = "0"
    environment["JAX_PLATFORMS"] = "cpu"
    environment["PYTHONWARNINGS"] = "error"
    subprocess.run(
        [sys.executable, "-c", program],
        check=True,
        env=environment,
    )


@pytest.mark.parametrize("backend", ("numpy", "torch", "jax"))
def test_complex_isclose_preserves_small_residuals(backend: str) -> None:
    selected = asc.backend(backend)
    xp = selected.xp
    first = xp.asarray(
        [30254.523 + 16.987034j],
        dtype=xp.complex64,
        device=selected.device,
    )
    second = xp.asarray(
        [30254.521 + 16.986917j],
        dtype=xp.complex64,
        device=selected.device,
    )

    result = asc.ops.isclose(first, second, rtol=1e-7, atol=0.0)

    assert bool(xp.all(result))


def test_complex_isclose_avoids_invalid_extreme_arithmetic() -> None:
    maximum = numpy.finfo(numpy.float32).max
    value = numpy.asarray([complex(maximum, maximum)], dtype=numpy.complex64)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = asc.ops.isclose(value, value)

    assert bool(numpy.all(result))


@pytest.mark.backend("jax")
def test_jax_fft_and_cast_reject_unavailable_wide_dtypes() -> None:
    program = """
import jax.numpy as jnp
import asc
from asc import data

selected = asc.backend("jax")
for function in (selected.fft.fftfreq, selected.fft.rfftfreq):
    try:
        function(4, dtype=jnp.float64)
    except asc.DTypeError:
        pass
    else:
        raise AssertionError(function)

value = jnp.asarray([1], dtype=jnp.int32)
for dtype in (jnp.int64, jnp.uint64, jnp.float64, jnp.complex128):
    try:
        data.CastDType(dtype).transform(value)
    except asc.DTypeError:
        pass
    else:
        raise AssertionError(dtype)
"""
    environment = dict(os.environ)
    environment["JAX_ENABLE_X64"] = "0"
    environment["JAX_PLATFORMS"] = "cpu"
    environment["PYTHONWARNINGS"] = "error"
    subprocess.run(
        [sys.executable, "-c", program],
        check=True,
        env=environment,
    )


@pytest.mark.parametrize("function_name", ("fftfreq", "rfftfreq"))
@pytest.mark.backend("torch")
def test_fft_frequency_device_overrides_use_backend_resolver(
    function_name: str,
) -> None:
    selected = asc.backend("torch")
    function = getattr(selected.fft, function_name)

    with pytest.raises(asc.DeviceError, match="CPU"):
        function(4, device="meta")

    result = function(4, device="cpu", dtype=selected.xp.float32)
    assert result.device.type == "cpu"


@pytest.mark.backend("torch")
def test_explicit_host_transfer_precedes_cpu_only_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    from asc.backends import torch as torch_adapter

    source = torch.empty((2,), dtype=torch.float32, device="meta")
    with pytest.raises(asc.ConversionError, match="allow_transfer"):
        asc.to_numpy(source)
    monkeypatch.setattr(
        torch_adapter,
        "to_cpu",
        lambda _value: torch.tensor([1.0, 2.0], dtype=torch.float32),
    )
    result = asc.to_numpy(source, allow_transfer=True)
    numpy.testing.assert_array_equal(result, [1.0, 2.0])


@pytest.mark.backend("torch")
@pytest.mark.backend("jax")
def test_quantized_torch_and_ambiguous_device_names_are_rejected() -> None:
    import torch

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        quantized = torch.quantize_per_tensor(
            torch.tensor([1.0, 2.0]),
            scale=0.1,
            zero_point=0,
            dtype=torch.qint8,
        )
    with pytest.raises(asc.UnsupportedCapabilityError, match="dense CPU"):
        asc.array_namespace(quantized)
    for backend in ("numpy", "torch", "jax"):
        with pytest.raises(asc.DeviceError, match="CPU"):
            asc.backend(backend, device="notcpu")
    with pytest.raises(asc.DeviceError, match="CPU"):
        asc.ArrayContext("numpy", device="notcpu")
    with pytest.raises(asc.ContextError, match="CPU"):
        asc.CreationContext(
            asc.backend("numpy").xp,
            "numpy",
            device="notcpu",
        )


@pytest.mark.backend("torch")
def test_torch_unique_all_handles_an_empty_array() -> None:
    selected = asc.backend("torch")
    empty = selected.xp.asarray([], dtype=selected.xp.float32)
    result = selected.xp.unique_all(empty)
    assert result.values.shape == (0,)
    assert result.indices.shape == (0,)
    assert result.inverse_indices.shape == (0,)
    assert result.counts.shape == (0,)


@pytest.mark.backend("torch")
def test_torch_unique_all_preserves_repeated_nan_groups() -> None:
    selected = asc.backend("torch")
    value = selected.xp.asarray(
        [float("nan"), 1.0, float("nan")], dtype=selected.xp.float32
    )

    result = selected.xp.unique_all(value)

    numpy.testing.assert_array_equal(numpy.asarray(result.indices), [1, 0, 2])
    numpy.testing.assert_array_equal(
        numpy.asarray(result.inverse_indices), [1, 0, 2]
    )
    numpy.testing.assert_array_equal(numpy.asarray(result.counts), [1, 1, 1])


@pytest.mark.backend("torch")
def test_torch_unique_operations_support_complex_arrays() -> None:
    selected = asc.backend("torch")
    value = selected.xp.asarray(
        [2 + 1j, 1 + 2j, 2 + 1j, 1 - 1j],
        dtype=selected.xp.complex64,
    )
    expected_values = numpy.asarray(
        [1 - 1j, 1 + 2j, 2 + 1j], dtype=numpy.complex64
    )

    all_result = selected.xp.unique_all(value)
    counts_result = selected.xp.unique_counts(value)
    inverse_result = selected.xp.unique_inverse(value)

    for observed in (
        all_result.values,
        counts_result.values,
        inverse_result.values,
        selected.xp.unique_values(value),
    ):
        numpy.testing.assert_array_equal(
            numpy.asarray(observed), expected_values
        )
    numpy.testing.assert_array_equal(
        numpy.asarray(all_result.indices), [3, 1, 0]
    )
    numpy.testing.assert_array_equal(
        numpy.asarray(all_result.inverse_indices), [2, 1, 2, 0]
    )
    numpy.testing.assert_array_equal(
        numpy.asarray(counts_result.counts), [1, 1, 2]
    )
    numpy.testing.assert_array_equal(
        numpy.asarray(inverse_result.inverse_indices), [2, 1, 2, 0]
    )


@pytest.mark.backend("torch")
def test_torch_unique_operations_do_not_scan_once_per_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    def reject_nonzero(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("unique operations must not scan every group")

    monkeypatch.setattr(torch, "nonzero", reject_nonzero)
    selected = asc.backend("torch")
    value = selected.xp.arange(30_000, dtype=selected.xp.int32)

    assert selected.xp.unique_values(value).shape == (30_000,)
    assert selected.xp.unique_counts(value).values.shape == (30_000,)
    assert selected.xp.unique_inverse(value).values.shape == (30_000,)
    assert selected.xp.unique_all(value).indices.shape == (30_000,)


def test_dataclass_init_false_fields_round_trip_and_map() -> None:
    @dataclasses.dataclass(frozen=True)
    class Record:
        value: int
        derived: int = dataclasses.field(init=False)

        def __post_init__(self) -> None:
            object.__setattr__(self, "derived", self.value * 2)

    source = Record(3)
    leaves, spec = tree.tree_flatten(source)
    restored = tree.tree_unflatten(spec, leaves)
    assert restored == source
    mapped = tree.tree_map(lambda value: value + 1, source)
    assert mapped.value == 4
    assert mapped.derived == 7


def test_missing_random_state_backend_maps_to_asc_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document = (
        '{"schema":1,"backend":"torch","version":"missing",'
        '"state":{"seed":1,"counter":0}}'
    )
    original = importlib.metadata.version

    def missing(name: str) -> str:
        if name == "torch":
            raise importlib.metadata.PackageNotFoundError(name)
        return original(name)

    monkeypatch.setattr(importlib.metadata, "version", missing)
    with pytest.raises(asc.BackendUnavailableError, match=r"asc-py\[torch\]"):
        random.RandomState.from_json(document)


@pytest.mark.parametrize("backend", ("numpy", "torch", "jax"))
def test_full_and_constant_padding_reject_unrepresentable_scalars(
    backend: str,
) -> None:
    selected = asc.backend(backend)
    xp = selected.xp
    configured = asc.backend(backend, dtype=xp.int8)
    context = asc.CreationContext(xp, backend, dtype=xp.int8)
    value = xp.asarray([1], dtype=xp.int8)

    with pytest.raises(asc.DTypeError, match="not representable"):
        configured.full((2,), 1000)
    with pytest.raises(asc.ContextError, match="not representable"):
        asc.create_full((2,), 1000, context=context)
    with pytest.raises(asc.DTypeError, match="not representable"):
        asc.ops.pad(value, 1, constant_values=1000)


@pytest.mark.parametrize("backend", ("numpy", "torch", "jax"))
def test_discrete_fills_require_exact_scalar_representability(
    backend: str,
) -> None:
    selected = asc.backend(backend)
    integer_backend = asc.backend(backend, dtype=selected.xp.int32)
    integer_context = asc.CreationContext(
        selected.xp, backend, dtype=selected.xp.int32
    )
    integer_array = selected.xp.asarray([1], dtype=selected.xp.int32)
    boolean_backend = asc.backend(backend, dtype=selected.xp.bool)
    boolean_context = asc.CreationContext(
        selected.xp, backend, dtype=selected.xp.bool
    )
    boolean_array = selected.xp.asarray([True], dtype=selected.xp.bool)

    for fill_value in (1.5, float("nan")):
        with pytest.raises(asc.DTypeError):
            integer_backend.full((1,), fill_value)
        with pytest.raises(asc.ContextError):
            asc.create_full((1,), fill_value, context=integer_context)
        with pytest.raises(asc.DTypeError):
            asc.ops.pad(integer_array, 1, constant_values=fill_value)
    for fill_value in (2, float("nan"), 1 + 2j):
        with pytest.raises(asc.DTypeError):
            boolean_backend.full((1,), fill_value)
        with pytest.raises(asc.ContextError):
            asc.create_full((1,), fill_value, context=boolean_context)
        with pytest.raises(asc.DTypeError):
            asc.ops.pad(boolean_array, 1, constant_values=fill_value)


def test_oversized_python_scalars_map_to_public_validation_errors() -> None:
    huge = 10**10_000

    with pytest.raises(asc.RandomStateError, match="finite Python real"):
        asc.random.normal(
            (1,),
            state=asc.random_state(1, backend="numpy"),
            mean=huge,
        )
    with pytest.raises(ValueError, match="finite non-negative"):
        asc.ops.elu(numpy.asarray([1.0], dtype=numpy.float32), alpha=huge)


def test_complete_make_gate_includes_isolated_required_profiles() -> None:
    repository = pathlib.Path(__file__).parents[2]
    makefile = (repository / "Makefile").read_text(encoding="utf-8")
    check_line = next(
        line for line in makefile.splitlines() if line.startswith("check:")
    )

    assert "docs-base" in check_line.split()
    assert "floor" in check_line.split()


@pytest.mark.backend("jax")
@pytest.mark.backend("torch")
def test_scalar_numerical_parameters_reject_native_arrays() -> None:
    numpy_backend = asc.backend("numpy")
    matrix = numpy.asarray([[1.0], [2.0]], dtype=numpy.float32)
    right = numpy.asarray([1.0, 2.0], dtype=numpy.float32)
    foreign_scalars = (
        asc.backend("jax").xp.asarray(0.5),
        asc.backend("torch").xp.asarray(0.5),
    )

    for scalar in foreign_scalars:
        for function in (
            numpy_backend.linalg.matrix_rank,
            numpy_backend.linalg.pinv,
        ):
            with pytest.raises(asc.MixedBackendError):
                function(matrix, rtol=scalar)
        with pytest.raises(asc.DTypeError, match="Python scalar"):
            numpy_backend.linalg.lstsq(matrix, right, rcond=scalar)
        for function in (
            numpy_backend.fft.fftfreq,
            numpy_backend.fft.rfftfreq,
        ):
            with pytest.raises(ValueError, match="Python real scalar"):
                function(4, d=scalar)
        with pytest.raises(ValueError, match="Python real scalar"):
            asc.ops.isclose(matrix, matrix, rtol=scalar)


def test_from_dlpack_rejects_raw_capsules_without_device_provenance() -> None:
    source = numpy.asarray([1.0], dtype=numpy.float32)
    raw_capsule = source.__dlpack__()

    with pytest.raises(asc.ConversionError, match="device provenance"):
        asc.from_dlpack(raw_capsule, "numpy", copy=True)


@pytest.mark.backend("jax")
def test_from_dlpack_propagates_no_copy_to_the_jax_importer() -> None:
    storage = bytearray(17)
    source = numpy.frombuffer(storage, dtype=numpy.float32, count=4, offset=1)

    with pytest.raises(asc.ConversionError, match="copy/device policy"):
        asc.from_dlpack(source, "jax", copy=asc.CopyPolicy.NEVER)


@pytest.mark.parametrize("backend", ("numpy", "torch", "jax"))
def test_lstsq_rejects_nonfinite_cutoffs(backend: str) -> None:
    selected = asc.backend(backend)
    matrix = selected.xp.asarray([[1.0], [2.0]], dtype=selected.xp.float32)
    right = selected.xp.asarray([1.0, 2.0], dtype=selected.xp.float32)

    for cutoff in (
        float("nan"),
        float("inf"),
        -float("inf"),
        10**10_000,
    ):
        with pytest.raises(asc.DTypeError, match="finite"):
            selected.linalg.lstsq(matrix, right, rcond=cutoff)


@pytest.mark.parametrize("backend", ("numpy", "torch", "jax"))
def test_lstsq_rejects_cutoffs_unrepresentable_in_promoted_dtype(
    backend: str,
) -> None:
    selected = asc.backend(backend)
    matrix = selected.xp.asarray(
        [[1.0, 0.0], [0.0, 1.0]],
        dtype=selected.xp.float32,
        device=selected.device,
    )
    right = selected.xp.asarray(
        [1.0, 2.0], dtype=selected.xp.float32, device=selected.device
    )

    for cutoff in (1e100, 1e-100):
        with pytest.raises(asc.DTypeError, match="not representable"):
            selected.linalg.lstsq(matrix, right, rcond=cutoff)


@pytest.mark.parametrize("backend", ("numpy", "torch", "jax"))
@pytest.mark.parametrize("function_name", ("fftfreq", "rfftfreq"))
def test_fft_frequency_validates_counts_and_dtype_scale(
    backend: str,
    function_name: str,
) -> None:
    selected = asc.backend(backend)
    function = getattr(selected.fft, function_name)

    for count in (True, 0, -1, 1.5):
        with pytest.raises(ValueError, match="positive integer"):
            function(count)
    for spacing in (1e-50, 1e40):
        with pytest.raises(ValueError, match="representable"):
            function(4, d=spacing, dtype=selected.xp.float32)
    with pytest.raises(ValueError, match="finite and nonzero"):
        function(4, d=10**10_000, dtype=selected.xp.float32)


@pytest.mark.parametrize("backend", ("numpy", "torch"))
@pytest.mark.parametrize(
    ("function_name", "expected"),
    (("fftfreq", -5e-309), ("rfftfreq", 5e-309)),
)
def test_fft_frequency_avoids_reciprocal_intermediate_overflow(
    backend: str,
    function_name: str,
    expected: float,
) -> None:
    selected = asc.backend(backend)

    result = getattr(selected.fft, function_name)(
        2,
        d=1e308,
        dtype=selected.xp.float64,
    )

    assert float(numpy.asarray(result)[1]) == expected


@pytest.mark.parametrize("backend", ("numpy", "torch", "jax"))
def test_activation_coefficients_must_survive_array_dtype(backend: str) -> None:
    selected = asc.backend(backend)
    values = selected.xp.asarray([-1.0, 1.0], dtype=selected.xp.float32)

    for call in (
        lambda: asc.ops.elu(values, alpha=1e40),
        lambda: asc.ops.leaky_relu(values, negative_slope=1e-50),
    ):
        with pytest.raises(asc.DTypeError, match="representable"):
            call()


@pytest.mark.backend("jax")
def test_jax_activation_coefficient_validation_is_transformation_safe() -> None:
    selected = asc.backend("jax")
    values = selected.xp.asarray([-1.0, 1.0], dtype=selected.xp.float32)

    def function(array: object) -> object:
        return asc.ops.elu(array) + asc.ops.leaky_relu(array)

    compiled = asc.jit(function, backend="jax")(values)
    gradient = asc.grad(
        lambda array: selected.xp.sum(function(array)),
        backend="jax",
    )(values)

    assert compiled.shape == values.shape
    assert gradient.shape == values.shape


@pytest.mark.parametrize("backend", ("numpy", "torch", "jax"))
def test_distribution_parameters_must_survive_output_dtype_conversion(
    backend: str,
) -> None:
    selected = asc.backend(backend)
    state = asc.random_state(7, backend=backend)
    calls = (
        lambda: asc.random.normal(
            (2,), state=state, mean=1e40, dtype=selected.xp.float32
        ),
        lambda: asc.random.normal(
            (2,), state=state, std=1e-50, dtype=selected.xp.float32
        ),
        lambda: asc.random.gamma(
            (2,),
            state=state,
            concentration=1e-50,
            dtype=selected.xp.float32,
        ),
        lambda: asc.random.gamma(
            (2,),
            state=state,
            concentration=1.0,
            scale=1e40,
            dtype=selected.xp.float32,
        ),
        lambda: asc.random.exponential(
            (2,), state=state, scale=1e40, dtype=selected.xp.float32
        ),
    )

    for call in calls:
        with pytest.raises(asc.RandomStateError, match="representable"):
            call()


@pytest.mark.backend("jax")
def test_jax_random_parameter_validation_is_jit_safe() -> None:
    state = asc.random_state(7, backend="jax")
    calls = (
        lambda current: asc.random.normal((2,), state=current),
        lambda current: asc.random.gamma(
            (2,), state=current, concentration=1.0
        ),
        lambda current: asc.random.exponential((2,), state=current),
        lambda current: asc.random.bernoulli((2,), state=current),
        lambda current: asc.random.truncated_normal((2,), state=current),
    )

    for call in calls:
        values, next_state = asc.jit(call, backend="jax")(state)
        assert values.shape == (2,)
        assert isinstance(next_state, asc.RandomState)


@pytest.mark.parametrize("backend", ("numpy", "torch", "jax"))
def test_floating_fills_reject_nonzero_underflow(backend: str) -> None:
    selected = asc.backend(backend)
    configured = asc.backend(backend, dtype=selected.xp.float32)
    context = asc.CreationContext(
        selected.xp,
        backend,
        dtype=selected.xp.float32,
        device=selected.device,
    )
    values = selected.xp.asarray([1.0], dtype=selected.xp.float32)

    calls = (
        lambda: configured.full((1,), 1e-50),
        lambda: asc.create_full((1,), 1e-50, context=context),
        lambda: asc.ops.pad(values, 1, constant_values=1e-50),
        lambda: asc.random.constant((1,), 1e-50, backend=configured),
    )
    for call in calls:
        with pytest.raises(asc.AscError, match="representable"):
            call()

    complex_backend = asc.backend(backend, dtype=selected.xp.complex64)
    with pytest.raises(asc.DTypeError, match="representable"):
        complex_backend.full((1,), 1e-50j)


@pytest.mark.parametrize("backend", ("numpy", "torch", "jax"))
def test_constant_honors_a_bound_backend_dtype(backend: str) -> None:
    selected = asc.backend(backend)
    configured = asc.backend(backend, dtype=selected.xp.float32)

    result = asc.random.constant((2,), 1.5, backend=configured)

    assert result.dtype == selected.xp.float32


def test_numpy_matrix_is_not_a_supported_native_array() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PendingDeprecationWarning)
        legacy = numpy.matrix([[1.0, 2.0]])

    assert not asc.is_array(legacy)
    with pytest.raises(asc.UnsupportedCapabilityError, match="dense CPU"):
        asc.backend_of(legacy)


@pytest.mark.parametrize("backend", ("numpy", "torch", "jax"))
def test_choice_normalizes_probabilities_within_tolerance(
    backend: str,
) -> None:
    selected = asc.backend(backend)
    xp = selected.xp
    dtype = (
        xp.float64
        if "float64" in asc.backend_info(backend).dtypes
        else xp.float32
    )
    probabilities = xp.asarray([0.5, 0.5000005], dtype=dtype)
    state = asc.random_state(7, backend=backend)

    sample, _ = asc.random.choice(
        2,
        (16,),
        state=state,
        probabilities=probabilities,
    )

    assert sample.shape == (16,)
    assert asc.backend_of(sample) == backend


@pytest.mark.parametrize("backend", ("numpy", "torch", "jax"))
def test_choice_rejects_overflowing_probability_totals(backend: str) -> None:
    selected = asc.backend(backend)
    maximum = float(selected.xp.finfo(selected.xp.float32).max)
    probabilities = selected.xp.asarray(
        [maximum, maximum],
        dtype=selected.xp.float32,
        device=selected.device,
    )

    with pytest.raises(asc.RandomStateError, match="sum to one"):
        asc.random.choice(
            2,
            (1,),
            state=asc.random_state(7, backend=backend),
            probabilities=probabilities,
        )


@pytest.mark.parametrize("backend", ("numpy", "torch", "jax"))
def test_choice_rejects_entries_above_one_within_sum_tolerance(
    backend: str,
) -> None:
    selected = asc.backend(backend)
    probabilities = selected.xp.asarray(
        [1.0000005, 0.0],
        dtype=selected.xp.float32,
        device=selected.device,
    )

    with pytest.raises(asc.RandomStateError, match=r"\[0, 1\]"):
        asc.random.choice(
            2,
            (1,),
            state=asc.random_state(7, backend=backend),
            probabilities=probabilities,
        )


@pytest.mark.backend("torch")
def test_torch_index_and_multi_index_operations_work_under_vmap() -> None:
    import torch

    destinations = torch.zeros((2, 3), dtype=torch.float32)
    indices = torch.tensor([[0], [1]], dtype=torch.int16)
    values = torch.tensor([[1.0], [2.0]], dtype=torch.float32)
    for operation in (
        asc.index_set,
        asc.index_add,
        asc.index_multiply,
        asc.index_min,
        asc.index_max,
    ):
        mapped = asc.vmap(
            operation,
            backend="torch",
            in_axes=(0, 0, 0),
        )
        assert mapped(destinations, indices, values).shape == (2, 3)

    duplicate_set = asc.vmap(
        asc.index_set,
        backend="torch",
        in_axes=(0, 0, 0),
    )
    with pytest.raises(asc.DuplicateIndexError, match="duplicate"):
        duplicate_set(
            destinations,
            torch.tensor([[0, 0], [1, 1]], dtype=torch.int8),
            torch.ones((2, 2), dtype=torch.float32),
        )

    unravel = asc.vmap(
        lambda value: asc.ops.unravel_index(value, (2, 2)),
        backend="torch",
    )
    coordinates = unravel(torch.tensor([0, 3], dtype=torch.int16))
    torch.testing.assert_close(coordinates[0], torch.tensor([0, 1]))
    torch.testing.assert_close(coordinates[1], torch.tensor([0, 1]))
    ravel = asc.vmap(
        lambda row, column: asc.ops.ravel_multi_index((row, column), (2, 2)),
        backend="torch",
        in_axes=(0, 0),
    )
    torch.testing.assert_close(
        ravel(
            torch.tensor([0, 1], dtype=torch.int16),
            torch.tensor([0, 1], dtype=torch.int16),
        ),
        torch.tensor([0, 3], dtype=torch.int64),
    )

    invalid_update = asc.vmap(
        asc.index_add,
        backend="torch",
        in_axes=(0, 0, 0),
    )
    with pytest.raises(asc.IndexUpdateError, match="out of bounds"):
        invalid_update(
            destinations,
            torch.tensor([[0], [3]], dtype=torch.int8),
            values,
        )
    with pytest.raises(IndexError, match="out of bounds"):
        unravel(torch.tensor([0, 4], dtype=torch.int16))


@pytest.mark.backend("torch")
def test_backend_asarray_routes_native_arrays_through_conversion_policy() -> (
    None
):
    import torch

    source = numpy.arange(3, dtype=numpy.float32)
    selected = asc.backend("torch")
    with pytest.raises(asc.ConversionError, match="cross-backend"):
        selected.asarray(source, copy=False)
    converted = selected.asarray(source, copy=True)
    assert asc.backend_of(converted) == "torch"

    primal = torch.tensor([1.0])
    tangent = torch.tensor([2.0])
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"`torch\.jit\.script` is deprecated\..*",
            category=DeprecationWarning,
        )
        with torch.autograd.forward_ad.dual_level():
            dual = torch.autograd.forward_ad.make_dual(primal, tangent)
            with pytest.raises(asc.ConversionError, match="active autodiff"):
                asc.backend("numpy").asarray(dual, copy=True)


@pytest.mark.backend("torch")
def test_from_dlpack_rejects_direct_active_graph_producers() -> None:
    import torch

    primal = torch.tensor([1.0])
    tangent = torch.tensor([2.0])
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"`torch\.jit\.script` is deprecated\..*",
            category=DeprecationWarning,
        )
        with torch.autograd.forward_ad.dual_level():
            dual = torch.autograd.forward_ad.make_dual(primal, tangent)
            with pytest.raises(asc.ConversionError, match="explicit detach"):
                asc.from_dlpack(dual, "numpy", copy=True)


@pytest.mark.backend("torch")
def test_namespace_from_dlpack_rejects_active_graph_producers() -> None:
    import torch

    primal = torch.tensor([1.0])
    tangent = torch.tensor([2.0])
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"`torch\.jit\.script` is deprecated\..*",
            category=DeprecationWarning,
        )
        with torch.autograd.forward_ad.dual_level():
            dual = torch.autograd.forward_ad.make_dual(primal, tangent)
            with pytest.raises(asc.ConversionError, match="active autodiff"):
                asc.backend("numpy").xp.from_dlpack(dual)


def test_backend_creation_rejects_inferred_unsupported_dtypes() -> None:
    selected = asc.backend("numpy")

    with pytest.raises(asc.ContextError, match="numeric CPU"):
        selected.full((1,), 2**100)
    with pytest.raises(asc.ContextError, match="numeric CPU"):
        selected.asarray(["not-numeric"])


@pytest.mark.parametrize("backend", ("numpy", "torch", "jax"))
def test_integer_convolution_preserves_promoted_dtype(backend: str) -> None:
    selected = asc.backend(backend)
    xp = selected.xp
    array = xp.asarray([1, 2], dtype=xp.int8)
    kernel = xp.asarray([1, 1, 1], dtype=xp.int8)

    for mode in ("valid", "same", "full"):
        result = asc.ops.convolve1d(array, kernel, mode=mode)
        assert result.dtype == xp.int8


@pytest.mark.parametrize("backend", ("numpy", "torch", "jax"))
def test_mixed_numeric_convolution_uses_extension_promotion(
    backend: str,
) -> None:
    selected = asc.backend(backend)
    xp = selected.xp
    array = xp.asarray([1, 2], dtype=xp.int16)
    kernel = xp.asarray([0.5, 1.0], dtype=xp.float32)

    result = asc.ops.convolve1d(array, kernel, mode="full")

    assert result.dtype == xp.float32
    numpy.testing.assert_allclose(numpy.asarray(result), [0.5, 2.0, 2.0])


@pytest.mark.parametrize("backend", ("numpy", "torch", "jax"))
def test_backend_constructors_validate_shapes_and_wrap_overflow(
    backend: str,
) -> None:
    selected = asc.backend(backend)
    invalid_shapes: tuple[object, ...] = ((True,), (-1,), [1])

    assert selected.zeros((2,)).shape == (2,)
    assert selected.ones((2,)).shape == (2,)

    for method_name in ("zeros", "ones", "full"):
        method = getattr(selected, method_name)
        for shape in invalid_shapes:
            arguments = (shape, 1) if method_name == "full" else (shape,)
            with pytest.raises(asc.ContextError, match="shape"):
                method(*arguments)

    with pytest.raises(asc.ContextError):
        selected.full((1,), 2**100)
    with pytest.raises(asc.ContextError):
        asc.random.constant((1,), 2**100, backend=selected)
    context = asc.CreationContext(
        selected.xp,
        typing.cast(asc.BackendName, backend),
        device=selected.device,
    )
    with pytest.raises(asc.ContextError):
        asc.create_full((1,), 2**100, context=context)


@pytest.mark.backend("torch")
def test_backend_asarray_rejects_nested_foreign_arrays() -> None:
    import torch

    with pytest.raises(asc.ContextError, match="nested foreign arrays"):
        asc.backend("numpy").asarray([torch.tensor(2.0)], copy=True)


def test_backend_asarray_rejects_nested_native_container_fields() -> None:
    @dataclasses.dataclass
    class Holder:
        value: object

    array = numpy.asarray([1.0])
    for nested in ({"value": array}, Holder(array)):
        with pytest.raises(asc.ContextError, match="nested native arrays"):
            asc.backend("numpy").asarray(nested)


@pytest.mark.parametrize("backend", ("numpy", "torch", "jax"))
def test_compatibility_random_rejects_oversized_bounds(backend: str) -> None:
    selected = asc.backend(backend)
    context = asc.CreationContext(
        selected.xp,
        typing.cast(asc.BackendName, backend),
        device=selected.device,
    )
    key = compatibility_random.create_key(0, context=context)

    with pytest.raises(asc.RandomStateError, match="finite real values"):
        compatibility_random.uniform((1,), key=key, high=10**10_000)


@pytest.mark.backend("jax")
def test_jax_vmap_translates_dynamic_validation_errors() -> None:
    selected = asc.backend("jax")
    xp = selected.xp
    unravel = asc.vmap(
        lambda index: asc.ops.unravel_index(index, (2, 2)), backend="jax"
    )
    with pytest.raises(IndexError, match=r"vmap:.*out of bounds"):
        unravel(xp.asarray([0, 4], dtype=xp.int32))

    state = asc.random_state(1, backend="jax")
    sample = asc.vmap(
        lambda probabilities: asc.random.choice(
            2,
            (1,),
            state=state,
            probabilities=probabilities,
        )[0],
        backend="jax",
    )
    probabilities = xp.asarray([[0.5, 0.5], [0.5, -0.5]], dtype=xp.float32)
    with pytest.raises(asc.RandomStateError, match=r"vmap:.*probabilities"):
        sample(probabilities)


def test_array_random_operations_validate_state_before_backend_access() -> None:
    invalid_state = typing.cast(random.RandomState, object())
    population = numpy.asarray([1, 2], dtype=numpy.int32)

    with pytest.raises(asc.RandomStateError, match=r"random\.choice: state"):
        random.choice(population, (1,), state=invalid_state)
    with pytest.raises(
        asc.RandomStateError, match=r"random\.permutation: state"
    ):
        random.permutation(population, state=invalid_state)


@pytest.mark.parametrize("backend", ("numpy", "torch", "jax"))
def test_metrics_reject_only_reduced_zero_sized_axes(backend: str) -> None:
    selected = asc.backend(backend)
    xp = selected.xp
    empty = xp.empty((0,), dtype=xp.float32, device=selected.device)
    operations = (
        asc.metrics.mean_absolute_error,
        asc.metrics.mean_squared_error,
        asc.metrics.root_mean_squared_error,
        asc.metrics.relative_l2_error,
        asc.metrics.r2_score,
    )

    for operation in operations:
        with pytest.raises(ValueError, match="zero-sized axis"):
            operation(empty, empty)
        assert operation(empty, empty, axis=()).shape == (0,)

    assert asc.metrics.mean_absolute_error(
        empty, empty, reduction="none"
    ).shape == (0,)
    assert asc.metrics.mean_squared_error(
        empty, empty, reduction="none"
    ).shape == (0,)

    empty_batch = xp.empty((0, 2), dtype=xp.float32, device=selected.device)
    for operation in operations:
        for axis in (False, True, (False,), (True,)):
            with pytest.raises(TypeError, match="axis must be"):
                operation(empty_batch, empty_batch, axis=axis)
        assert operation(empty_batch, empty_batch, axis=1).shape == (0,)
        with pytest.raises(ValueError, match="zero-sized axis"):
            operation(empty_batch, empty_batch, axis=0)


@pytest.mark.backend("torch")
@pytest.mark.parametrize("destination", ("numpy", "torch"))
def test_opaque_dlpack_copy_is_enforced_after_import(
    destination: str,
) -> None:
    class CopyIgnoringProducer:
        def __init__(self) -> None:
            self.value = numpy.asarray([1.0], dtype=numpy.float32)
            self.dtype = self.value.dtype

        def __dlpack_device__(self) -> tuple[int, int]:
            return (1, 0)

        def __dlpack__(self, *args: object, **kwargs: object) -> object:
            kwargs.pop("copy", None)
            return self.value.__dlpack__(*args, **kwargs)

    direct_producer = CopyIgnoringProducer()
    direct = asc.backend(destination).xp.from_dlpack(direct_producer, copy=True)
    direct_producer.value[0] = 2.0
    numpy.testing.assert_array_equal(numpy.asarray(direct), [1.0])

    public_producer = CopyIgnoringProducer()
    public = asc.from_dlpack(public_producer, destination, copy=True)
    public_producer.value[0] = 2.0
    numpy.testing.assert_array_equal(numpy.asarray(public), [1.0])


@pytest.mark.backend("torch")
def test_opaque_torch_default_import_enforces_its_safety_copy() -> None:
    class CopyIgnoringProducer:
        def __init__(self) -> None:
            self.value = numpy.asarray([1.0], dtype=numpy.float32)
            self.dtype = self.value.dtype

        def __dlpack_device__(self) -> tuple[int, int]:
            return (1, 0)

        def __dlpack__(self, *args: object, **kwargs: object) -> object:
            kwargs.pop("copy", None)
            return self.value.__dlpack__(*args, **kwargs)

    producer = CopyIgnoringProducer()
    result = asc.backend("torch").xp.from_dlpack(producer)
    producer.value[0] = 2.0
    numpy.testing.assert_array_equal(numpy.asarray(result), [1.0])


@pytest.mark.backend("torch")
def test_torch_overrides_reject_tensor_valued_scalar_controls() -> None:
    import torch

    selected = asc.backend("torch")
    xp = selected.xp
    array = xp.asarray([1.0, 2.0], dtype=xp.float32)
    indices = xp.asarray([0], dtype=xp.int64)
    tensor_zero = torch.tensor(0)

    with pytest.raises(TypeError, match="shape must be"):
        xp.reshape(array, (torch.tensor(2),))
    with pytest.raises(TypeError, match="axis must be"):
        xp.take(array, indices, axis=tensor_zero)
    with pytest.raises(TypeError, match="axis must be"):
        xp.take_along_axis(array, indices, axis=tensor_zero)


@pytest.mark.backend("jax")
def test_jax_asarray_rejects_allocating_python_inputs_with_copy_false() -> None:
    selected = asc.backend("jax")

    for value in (1.0, [1.0, 2.0]):
        with pytest.raises(ValueError, match="copy=False"):
            selected.xp.asarray(value, copy=False)
        with pytest.raises(asc.ContextError, match="copy policy"):
            selected.asarray(value, copy=False)

    native = selected.xp.asarray([1.0], dtype=selected.xp.float32)
    assert selected.xp.asarray(native, copy=False) is native
    for nested in ([native], (native,), [[native]], [native, native]):
        with pytest.raises(ValueError, match="copy=False"):
            selected.xp.asarray(nested, copy=False)


@pytest.mark.backend("torch")
@pytest.mark.backend("jax")
@pytest.mark.parametrize("backend", ("numpy", "torch", "jax"))
def test_namespace_from_dlpack_rejects_non_boolean_copy_before_export(
    backend: str,
) -> None:
    class Producer:
        def __init__(self) -> None:
            self.value = numpy.asarray([1.0], dtype=numpy.float32)
            self.dtype = self.value.dtype
            self.consumed = False

        def __dlpack_device__(self) -> tuple[int, int]:
            return (1, 0)

        def __dlpack__(self, *args: object, **kwargs: object) -> object:
            self.consumed = True
            return self.value.__dlpack__(*args, **kwargs)

    for copy in (0, 1, "yes", object()):
        producer = Producer()
        with pytest.raises(TypeError, match="copy must be"):
            asc.backend(backend).xp.from_dlpack(producer, copy=copy)
        assert not producer.consumed


@pytest.mark.backend("torch")
@pytest.mark.backend("jax")
@pytest.mark.parametrize("backend", ("numpy", "torch", "jax"))
def test_backend_rejects_array_valued_cpu_device(backend: str) -> None:
    with pytest.raises(asc.DeviceError, match="CPU"):
        asc.backend(backend, device=numpy.asarray("cpu"))


@pytest.mark.backend("torch")
@pytest.mark.backend("jax")
@pytest.mark.parametrize("backend", ("numpy", "torch", "jax"))
def test_namespace_rejects_same_backend_array_valued_controls(
    backend: str,
) -> None:
    selected = asc.backend(backend)
    xp = selected.xp
    matrix = xp.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=xp.float32)
    zero = xp.asarray(0, dtype=xp.int32)
    two = xp.asarray(2, dtype=xp.int32)
    four = xp.asarray(4, dtype=xp.int32)

    operations = (
        lambda: xp.sum(matrix, axis=zero),
        lambda: xp.eye(two, dtype=xp.float32),
        lambda: xp.reshape(matrix, (four,)),
    )
    for operation in operations:
        with pytest.raises(TypeError):
            operation()


def test_contexts_reject_spoofed_cpu_device_objects() -> None:
    class PlatformDevice:
        platform = "cpu"

    class TypedDevice:
        type = "cpu"
        index = None

    class StrictDevice:
        _device = "CPU_DEVICE"

    selected = asc.backend("numpy")
    for device in (PlatformDevice(), TypedDevice(), StrictDevice()):
        with pytest.raises(asc.DeviceError, match="CPU"):
            asc.ArrayContext("numpy", device=device)
        with pytest.raises(asc.ContextError, match="CPU"):
            asc.CreationContext(
                selected.xp,
                "numpy",
                device=device,
            )


@pytest.mark.backend("torch")
@pytest.mark.backend("jax")
def test_contexts_accept_authenticated_native_cpu_devices() -> None:
    import torch

    for backend, device in (
        ("torch", torch.device("cpu")),
        ("jax", asc.backend("jax").device),
    ):
        selected = asc.backend(backend)
        assert asc.ArrayContext(backend, device=device).device is device
        assert (
            asc.CreationContext(
                selected.xp,
                typing.cast(asc.BackendName, backend),
                device=device,
            ).device
            is device
        )


@pytest.mark.backend("torch")
def test_numpy_python_scalar_subclasses_retain_array_ownership() -> None:
    import torch

    for scalar in (numpy.float64(1.0), numpy.complex128(1.0 + 2.0j)):
        assert asc.is_array(scalar)
        assert asc.backend_of(scalar) == "numpy"
        with pytest.raises(asc.MixedBackendError):
            asc.array_namespace(torch.asarray([1.0]), scalar)


@pytest.mark.backend("torch")
@pytest.mark.parametrize("backend", ("numpy", "torch", "jax"))
def test_namespace_asarray_rejects_nested_native_arrays(backend: str) -> None:
    selected = asc.backend(backend)
    native = selected.xp.asarray(2.0, dtype=selected.xp.float32)

    for nested in ([native], [[native]], (native, native)):
        with pytest.raises(TypeError, match="stacked explicitly"):
            selected.xp.asarray(nested)

    if backend == "torch":
        import torch

        differentiable = torch.tensor(2.0, requires_grad=True)
        with pytest.raises(TypeError, match="stacked explicitly"):
            selected.xp.asarray([differentiable])


@pytest.mark.backend("torch")
@pytest.mark.backend("jax")
@pytest.mark.parametrize("backend", ("numpy", "torch", "jax"))
def test_namespace_validates_complete_python_control_types(
    backend: str,
) -> None:
    selected = asc.backend(backend)
    xp = selected.xp
    matrix = xp.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=xp.float32)
    zero = xp.asarray(0, dtype=xp.int32)
    one = xp.asarray(1, dtype=xp.int32)

    invalid_calls = (
        lambda: xp.moveaxis(matrix, zero, one),
        lambda: xp.tile(matrix, (one, 1)),
        lambda: xp.sum(matrix, keepdims=1),
        lambda: xp.std(matrix, correction=True),
        lambda: xp.linspace(0.0, 1.0, 2, endpoint=1),
    )
    for invalid_call in invalid_calls:
        with pytest.raises(TypeError):
            invalid_call()

    assert xp.moveaxis(matrix, 0, 1).shape == matrix.shape
    assert xp.tile(matrix, (1, 1)).shape == matrix.shape
    assert xp.sum(matrix, keepdims=True).shape == (1, 1)


@pytest.mark.backend("jax")
def test_jax_asarray_copy_false_rejects_native_dtype_changes() -> None:
    selected = asc.backend("jax")
    value = selected.xp.asarray([1], dtype=selected.xp.int32)

    with pytest.raises(ValueError, match=r"copy=False.*dtype"):
        selected.xp.asarray(value, dtype=selected.xp.float32, copy=False)
    assert (
        selected.xp.asarray(
            value,
            dtype=selected.xp.int32,
            device=selected.device,
            copy=False,
        )
        is value
    )


@pytest.mark.backend("torch")
def test_hostile_dlpack_signature_errors_are_contained() -> None:
    class HostileExporter:
        def __init__(self) -> None:
            self.called = False

        @property
        def __signature__(self) -> object:
            raise RuntimeError("hostile signature")

        def __call__(self, *_args: object, **_kwargs: object) -> object:
            self.called = True
            return numpy.asarray([1.0], dtype=numpy.float32).__dlpack__()

    class Producer:
        dtype = numpy.dtype("float32")

        def __init__(self) -> None:
            self.__dlpack__ = HostileExporter()

        def __dlpack_device__(self) -> tuple[int, int]:
            return (1, 0)

    for importer in (
        lambda value: asc.from_dlpack(value, "torch"),
        lambda value: asc.backend("torch").xp.from_dlpack(value),
    ):
        producer = Producer()
        with pytest.raises(asc.AscError):
            importer(producer)
        assert not producer.__dlpack__.called


@pytest.mark.backend("torch")
def test_nested_namespaces_reject_foreign_arrays_before_dispatch() -> None:
    import torch

    selected = asc.backend("numpy")
    matrix = torch.asarray([[1.0, 0.0], [0.0, 1.0]])

    with pytest.raises(asc.MixedBackendError):
        selected.xp.linalg.det(matrix)
    with pytest.raises(asc.MixedBackendError):
        selected.xp.fft.fft(matrix)


@pytest.mark.backend("torch")
@pytest.mark.backend("jax")
@pytest.mark.parametrize("backend", ("numpy", "torch", "jax"))
def test_standard_array_operands_cannot_be_python_containers(
    backend: str,
) -> None:
    xp = asc.backend(backend).xp
    array = xp.asarray([1.0, 2.0], dtype=xp.float32)

    for operation in (
        lambda: xp.sin(1.0),
        lambda: xp.sum([1.0, 2.0]),
        lambda: xp.add(1.0, 2.0),
        lambda: xp.matmul(array, [[1.0], [2.0]]),
    ):
        with pytest.raises(asc.NamespaceError):
            operation()

    assert xp.add(array, 1.0).shape == array.shape


@pytest.mark.backend("jax")
def test_jax_no_copy_requires_provably_identical_storage() -> None:
    import jax.numpy as jnp

    selected = asc.backend("jax")
    xp = selected.xp
    strong = xp.asarray([[1, 2]], dtype=xp.int32)
    weak = jnp.asarray(1)

    assert xp.asarray(strong, dtype=xp.int32, copy=False) is strong
    assert xp.astype(strong, xp.int32, copy=False) is strong
    assert xp.reshape(strong, strong.shape, copy=False) is strong
    for operation in (
        lambda: xp.asarray(weak, dtype=weak.dtype, copy=False),
        lambda: xp.astype(strong, xp.float32, copy=False),
        lambda: xp.reshape(strong, (2,), copy=False),
    ):
        with pytest.raises(ValueError, match="copy=False"):
            operation()


@pytest.mark.backend("torch")
def test_hostile_array_protocol_errors_do_not_consume_dlpack() -> None:
    class Producer:
        dtype = numpy.dtype("float32")

        def __init__(self) -> None:
            self.consumed = False

        @property
        def __array_namespace__(self) -> object:
            raise RuntimeError("hostile array protocol")

        def __dlpack_device__(self) -> tuple[int, int]:
            return (1, 0)

        def __dlpack__(self, *args: object, **kwargs: object) -> object:
            self.consumed = True
            return numpy.asarray([1.0], dtype=numpy.float32).__dlpack__(
                *args, **kwargs
            )

    for importer in (
        lambda value: asc.from_dlpack(value, "torch"),
        lambda value: asc.backend("torch").xp.from_dlpack(value),
    ):
        producer = Producer()
        with pytest.raises(asc.AscError):
            importer(producer)
        assert not producer.consumed


def test_numpy_namespace_rejects_native_keyword_extensions() -> None:
    xp = asc.backend("numpy").xp
    array = xp.asarray([1.0, 2.0], dtype=xp.float32)

    with pytest.raises(asc.DTypeError, match="frozen Array API signature"):
        xp.concat((array,), dtype=numpy.longdouble)
    with pytest.raises(TypeError, match="frozen Array API surface"):
        xp.argsort(array, kind="stable")
    with pytest.raises(TypeError, match="frozen Array API surface"):
        xp.clip(array, min=0.0, max=1.0, out=array)


@pytest.mark.backend("torch")
@pytest.mark.backend("jax")
@pytest.mark.parametrize("backend", ("numpy", "torch", "jax"))
def test_remaining_namespace_controls_reject_arrays_and_booleans(
    backend: str,
) -> None:
    xp = asc.backend(backend).xp
    matrix = xp.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=xp.float32)
    zero = xp.asarray(0, dtype=xp.int32)

    for operation in (
        lambda: xp.tril(matrix, k=zero),
        lambda: xp.triu(matrix, k=True),
        lambda: xp.isdtype(xp.float32, zero),
    ):
        with pytest.raises(TypeError):
            operation()


@pytest.mark.backend("torch")
@pytest.mark.backend("jax")
@pytest.mark.parametrize("backend", ("numpy", "torch", "jax"))
def test_linalg_controls_are_validated_before_native_dispatch(
    backend: str,
) -> None:
    xp = asc.backend(backend).xp
    matrix = xp.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=xp.float32)
    zero = xp.asarray(0, dtype=xp.int32)

    for operation in (
        lambda: xp.linalg.svd(matrix, full_matrices=1),
        lambda: xp.linalg.vector_norm(matrix, axis=zero),
        lambda: xp.linalg.matrix_power(matrix, zero),
    ):
        with pytest.raises(TypeError):
            operation()


@pytest.mark.backend("torch")
@pytest.mark.backend("jax")
@pytest.mark.parametrize("backend", ("numpy", "torch", "jax"))
def test_fft_controls_are_validated_before_native_dispatch(
    backend: str,
) -> None:
    xp = asc.backend(backend).xp
    array = xp.asarray([1.0, 2.0], dtype=xp.float32)
    zero = xp.asarray(0, dtype=xp.int32)

    for operation in (
        lambda: xp.fft.fft(array, n=True),
        lambda: xp.fft.fft(array, axis=zero),
        lambda: xp.fft.fft(array, norm="native-extension"),
    ):
        with pytest.raises(TypeError):
            operation()


@pytest.mark.backend("torch")
@pytest.mark.backend("jax")
@pytest.mark.parametrize("backend", ("numpy", "torch", "jax"))
def test_standard_namespace_enforces_dtype_domains_and_promotion(
    backend: str,
) -> None:
    xp = asc.backend(backend).xp
    integer = xp.asarray([1, 2], dtype=xp.int32)
    floating = xp.asarray([1.0, 2.0], dtype=xp.float32)

    for operation in (
        lambda: xp.sin(integer),
        lambda: xp.reciprocal(integer),
        lambda: xp.mean(integer),
        lambda: xp.add(integer, floating),
        lambda: xp.result_type(integer, floating),
    ):
        with pytest.raises(asc.DTypeError):
            operation()

    assert xp.sin(floating).dtype == xp.float32
    assert xp.add(integer, 1).dtype == xp.int32
    assert xp.can_cast(xp.int32, xp.float32) is False


@pytest.mark.backend("torch")
@pytest.mark.backend("jax")
@pytest.mark.parametrize("backend", ("numpy", "torch", "jax"))
def test_standard_namespace_enforces_shape_edge_contracts(
    backend: str,
) -> None:
    xp = asc.backend(backend).xp
    vector = xp.asarray([1, 2, 3, 4], dtype=xp.int32)
    matrix = xp.reshape(vector, (2, 2))
    indices = xp.asarray([0, 1], dtype=xp.int16)

    assert xp.can_cast(xp.int8, xp.int16) is True
    assert xp.can_cast(xp.int16, xp.int16) is True
    assert len(xp.meshgrid()) == 0
    with pytest.raises(ValueError, match="axis must be specified"):
        xp.take(matrix, indices)
    with pytest.raises(ValueError, match="one-dimensional"):
        xp.take(matrix, xp.reshape(indices, (1, 2)), axis=0)
    with pytest.raises(ValueError, match="at least one-dimensional"):
        xp.cumulative_prod(xp.asarray(2, dtype=xp.int32))
    with pytest.raises(ValueError, match="one-dimensional"):
        xp.searchsorted(matrix, matrix)
    with pytest.raises(ValueError, match="zero- or one-dimensional"):
        xp.repeat(vector, matrix)
    with pytest.raises(ValueError, match="broadcast to the selected axis"):
        xp.repeat(vector, indices)


def test_numpy_namespace_uses_the_frozen_promotion_lattice() -> None:
    xp = asc.backend("numpy").xp

    expected_promotions = (
        (xp.int8, xp.int16, xp.int16),
        (xp.uint8, xp.uint16, xp.uint16),
        (xp.int8, xp.uint8, xp.int16),
        (xp.int32, xp.uint32, xp.int64),
        (xp.float32, xp.float64, xp.float64),
        (xp.float32, xp.complex64, xp.complex64),
        (xp.float64, xp.complex64, xp.complex128),
        (xp.complex64, xp.complex128, xp.complex128),
    )
    for first, second, expected in expected_promotions:
        assert xp.result_type(first, second) == expected
        assert xp.result_type(second, first) == expected
    assert xp.result_type(xp.float32, xp.float32) == xp.float32

    with pytest.raises(asc.DTypeError):
        xp.result_type(xp.int64, xp.uint64)
    with pytest.raises(asc.DTypeError):
        xp.result_type(1, 2)

    boolean = xp.asarray([True], dtype=xp.bool)
    integer = xp.asarray([1], dtype=xp.int8)
    unsigned = xp.asarray([1], dtype=xp.uint8)
    floating = xp.asarray([1.0], dtype=xp.float32)
    complex_value = xp.asarray([1.0j], dtype=xp.complex64)

    assert xp.equal(boolean, True).dtype == xp.bool
    assert xp.add(floating, 1).dtype == xp.float32
    assert xp.add(floating, 1.0).dtype == xp.float32
    assert xp.add(floating, 1.0j).dtype == xp.complex64
    assert xp.add(complex_value, 1.0j).dtype == xp.complex64
    for operation in (
        lambda: xp.equal(integer, True),
        lambda: xp.equal(boolean, 1),
        lambda: xp.add(integer, 1.0),
        lambda: xp.add(integer, 1.0j),
        lambda: xp.add(integer, 128),
        lambda: xp.add(unsigned, -1),
    ):
        with pytest.raises(asc.DTypeError):
            operation()


@pytest.mark.backend("torch")
def test_optional_low_precision_dtypes_promote_portably() -> None:
    xp = asc.backend("torch").xp

    assert xp.result_type(xp.float16, xp.bfloat16) == xp.float32
    assert xp.result_type(xp.float16, xp.complex64) == xp.complex64


def test_numpy_namespace_validates_secondary_dtype_contracts() -> None:
    xp = asc.backend("numpy").xp
    integer = xp.asarray([1, 3], dtype=xp.int32)
    floating = xp.asarray([1.5, 2.5], dtype=xp.float32)
    complex_value = xp.asarray([1.0j, 2.0j], dtype=xp.complex64)
    integer_indices = xp.asarray([0, 1], dtype=xp.int16)

    assert xp.clip(integer, min=0, max=2).dtype == xp.int32
    assert xp.clip(floating, min=xp.zeros_like(floating)).dtype == xp.float32
    assert xp.searchsorted(integer, floating, sorter=integer_indices).shape == (
        2,
    )
    assert xp.take(integer, integer_indices).shape == (2,)
    assert xp.repeat(integer, integer_indices).shape == (1,)
    assert (
        xp.concat(
            (
                xp.astype(integer, xp.int8, copy=True),
                xp.astype(integer, xp.int16, copy=True),
            )
        ).dtype
        == xp.int16
    )
    assert xp.matmul(
        xp.reshape(integer, (1, 2)),
        xp.reshape(floating, (2, 1)),
    ).shape == (1, 1)

    for operation in (
        lambda: xp.clip(integer, min=0.5),
        lambda: xp.clip(floating, min=integer),
        lambda: xp.clip(floating, min="invalid"),
        lambda: xp.searchsorted(integer, floating, sorter=floating),
        lambda: xp.take(integer, floating),
        lambda: xp.repeat(integer, floating),
        lambda: xp.concat((integer, floating)),
        lambda: xp.stack((integer, floating)),
        lambda: xp.meshgrid(integer, floating),
        lambda: xp.astype(complex_value, xp.float32),
        lambda: xp.logical_and(integer, integer),
        lambda: xp.divide(integer, integer),
        lambda: xp.greater(complex_value, complex_value),
        lambda: xp.imag(floating),
        lambda: xp.signbit(integer),
    ):
        with pytest.raises(asc.DTypeError):
            operation()


@pytest.mark.backend("torch")
@pytest.mark.backend("jax")
@pytest.mark.parametrize("backend", ("numpy", "torch", "jax"))
def test_where_and_searchsorted_require_standard_array_operands(
    backend: str,
) -> None:
    xp = asc.backend(backend).xp
    condition = xp.asarray([True, False], dtype=xp.bool)
    integers = xp.asarray([1, 3], dtype=xp.int32)
    floating = xp.asarray([1.5], dtype=xp.float32)

    for operation in (
        lambda: xp.where(condition, 1, 2),
        lambda: xp.where(condition, [1, 2], integers),
        lambda: xp.searchsorted(integers, 2),
        lambda: xp.searchsorted(integers, [2]),
    ):
        with pytest.raises(asc.NamespaceError):
            operation()
    with pytest.raises(asc.DTypeError):
        xp.where(condition, integers, floating)

    assert xp.where(condition, integers, 0).dtype == xp.int32
    assert xp.searchsorted(integers, floating).shape == floating.shape


@pytest.mark.backend("torch")
@pytest.mark.backend("jax")
@pytest.mark.parametrize("backend", ("numpy", "torch", "jax"))
def test_fft_and_linalg_enforce_standard_dtype_domains(backend: str) -> None:
    xp = asc.backend(backend).xp
    integer_matrix = xp.asarray([[2, 0], [0, 2]], dtype=xp.int32)
    real = xp.asarray([1.0, 2.0], dtype=xp.float32)
    complex_value = xp.astype(real, xp.complex64, copy=True)

    for operation in (
        lambda: xp.linalg.cholesky(integer_matrix),
        lambda: xp.fft.fft(real),
        lambda: xp.fft.rfft(complex_value),
    ):
        with pytest.raises(asc.DTypeError):
            operation()

    assert xp.fft.fft(complex_value).dtype == xp.complex64
    assert xp.fft.rfft(real).dtype == xp.complex64


def test_dataclass_tree_reconstruction_does_not_repeat_post_init() -> None:
    @dataclasses.dataclass(frozen=True)
    class ProcessedRecord:
        value: int

        def __post_init__(self) -> None:
            object.__setattr__(self, "value", self.value * 2)

    source = ProcessedRecord(1)
    leaves, spec = tree.tree_flatten(source)

    restored = tree.tree_unflatten(spec, leaves)
    mapped = tree.tree_map(lambda value: value, source)
    replaced = tree.tree_replace(source, ("value",), 3)

    assert restored.value == 2
    assert mapped.value == 2
    assert replaced.value == 3


def test_portable_ops_and_metrics_use_only_the_frozen_array_api() -> None:
    import array_api_strict as strict

    strict.set_array_api_strict_flags(api_version="2024.12")
    first = strict.asarray([1.0, 2.0], dtype=strict.float32)
    second = strict.asarray([1.0, 3.0], dtype=strict.float32)
    infinities = strict.asarray(
        [-float("inf"), float("inf")], dtype=strict.float32
    )

    numpy.testing.assert_array_equal(
        numpy.asarray(asc.ops.silu(infinities)), [0.0, numpy.inf]
    )
    numpy.testing.assert_array_equal(
        numpy.asarray(asc.ops.isclose(first, second)), [True, False]
    )
    assert asc.ops.allclose(first, first)
    asc.ops.assert_allclose(first, first)
    assert numpy.isfinite(
        numpy.asarray(asc.metrics.relative_l2_error(first, second))
    )
    assert numpy.isfinite(numpy.asarray(asc.metrics.r2_score(first, second)))


@pytest.mark.backend("jax")
def test_allclose_preserves_jax_compilation() -> None:
    selected = asc.backend("jax")
    values = selected.xp.asarray([1.0, 2.0], dtype=selected.xp.float32)

    compiled = asc.jit(asc.ops.allclose, backend="jax")(values, values)

    assert compiled.shape == ()
    assert bool(compiled)


@pytest.mark.backend("torch")
@pytest.mark.backend("jax")
@pytest.mark.parametrize("backend", ("numpy", "torch", "jax"))
def test_metrics_require_boolean_keepdims(backend: str) -> None:
    selected = asc.backend(backend)
    values = selected.xp.asarray([1.0, 2.0], dtype=selected.xp.float32)
    functions = (
        asc.metrics.mean_absolute_error,
        asc.metrics.mean_squared_error,
        asc.metrics.root_mean_squared_error,
        asc.metrics.relative_l2_error,
        asc.metrics.r2_score,
    )

    for function in functions:
        for keepdims in (0, 1, "", "yes"):
            with pytest.raises(TypeError, match="keepdims must be"):
                function(
                    values,
                    values,
                    keepdims=typing.cast(bool, keepdims),
                )


@pytest.mark.backend("torch")
@pytest.mark.backend("jax")
@pytest.mark.parametrize("backend", ("numpy", "torch", "jax"))
def test_namespace_rejects_unrepresentable_integer_fills(backend: str) -> None:
    selected = asc.backend(backend)
    xp = selected.xp
    reference = xp.zeros((1,), dtype=xp.uint8)

    for fill_value in (-1, 256, 300):
        with pytest.raises(asc.DTypeError, match="not representable"):
            xp.full((1,), fill_value, dtype=xp.uint8)
        with pytest.raises(asc.DTypeError, match="not representable"):
            xp.full_like(reference, fill_value)

    assert xp.full((1,), 255, dtype=xp.uint8).tolist() == [255]
    assert xp.full_like(reference, 255).tolist() == [255]


@pytest.mark.backend("torch")
@pytest.mark.backend("jax")
@pytest.mark.parametrize("backend", ("numpy", "torch", "jax"))
def test_integer_control_arrays_use_portable_index_carriers(
    backend: str,
) -> None:
    selected = asc.backend(backend)
    xp = selected.xp
    values = xp.asarray([10, 20], dtype=xp.int32)
    boundaries = xp.asarray([10, 20], dtype=xp.int32)

    for dtype_name in ("int8", "int16", "int32", "uint8", "uint16", "uint32"):
        if not hasattr(xp, dtype_name):
            continue
        dtype = getattr(xp, dtype_name)
        counts = xp.asarray([1, 2], dtype=dtype)
        sorter = xp.asarray([0, 1], dtype=dtype)

        assert xp.repeat(values, counts).tolist() == [10, 20, 20]
        assert xp.searchsorted(
            boundaries,
            boundaries,
            sorter=sorter,
        ).tolist() == [0, 1]

    if "uint64" in asc.backend_info(backend).dtypes:
        counts = xp.asarray([1, 2], dtype=xp.uint64)
        sorter = xp.asarray([0, 1], dtype=xp.uint64)
        assert xp.repeat(values, counts).tolist() == [10, 20, 20]
        assert xp.searchsorted(
            boundaries,
            boundaries,
            sorter=sorter,
        ).tolist() == [0, 1]


@pytest.mark.backend("torch")
def test_torch_take_accepts_unsigned_index_arrays() -> None:
    xp = asc.backend("torch").xp
    vector = xp.asarray([10, 20], dtype=xp.int32)
    matrix = xp.asarray([[10, 20], [30, 40]], dtype=xp.int32)

    for dtype_name in ("uint8", "uint16", "uint32", "uint64"):
        dtype = getattr(xp, dtype_name)
        indices = xp.asarray([1, 0], dtype=dtype)
        matrix_indices = xp.asarray([[1, 0], [0, 1]], dtype=dtype)

        assert xp.take(vector, indices).tolist() == [20, 10]
        assert xp.take_along_axis(
            matrix,
            matrix_indices,
            axis=1,
        ).tolist() == [[20, 10], [30, 40]]

    oversized = xp.asarray([2**63], dtype=xp.uint64)
    with pytest.raises(IndexError, match="carrier range"):
        xp.take(vector, oversized)


def test_public_array_detection_contains_hostile_protocol_failures() -> None:
    class HostileArrayProtocol:
        @property
        def __array_namespace__(self) -> object:
            raise RuntimeError("hostile protocol must not escape")

    value = HostileArrayProtocol()

    assert not asc.is_array(value)
    with pytest.raises(asc.NamespaceError, match="invalid array protocol"):
        asc.array_namespace(value)
    with pytest.raises(asc.NamespaceError, match="invalid array protocol"):
        asc.backend_of(value)
    with pytest.raises(asc.ContextError, match="invalid array protocol"):
        asc.backend("numpy").asarray(value)
    with pytest.raises(asc.ContextError, match="invalid array protocol"):
        asc.backend("numpy").asarray([value])


@pytest.mark.backend("torch")
@pytest.mark.backend("jax")
@pytest.mark.parametrize("backend", ("numpy", "torch", "jax"))
def test_standard_dispatch_applies_portable_promotion(backend: str) -> None:
    xp = asc.backend(backend).xp
    signed = xp.asarray([-1], dtype=xp.int16)
    unsigned = xp.asarray([65535], dtype=xp.uint16)
    condition = xp.asarray([True], dtype=xp.bool)

    added = xp.add(signed, unsigned)
    concatenated = xp.concat((signed, unsigned))
    stacked = xp.stack((signed, unsigned))
    selected = xp.where(condition, signed, unsigned)

    assert added.dtype == xp.int32
    assert added.tolist() == [65534]
    assert xp.equal(signed, unsigned).tolist() == [False]
    assert xp.less(signed, unsigned).tolist() == [True]
    assert concatenated.dtype == xp.int32
    assert concatenated.tolist() == [-1, 65535]
    assert stacked.dtype == xp.int32
    assert stacked.tolist() == [[-1], [65535]]
    assert selected.dtype == xp.int32
    assert selected.tolist() == [-1]


@pytest.mark.backend("torch")
@pytest.mark.parametrize("dtype_name", ("uint16", "uint32", "uint64"))
def test_torch_advertised_wide_unsigned_operations(dtype_name: str) -> None:
    xp = asc.backend("torch").xp
    dtype = getattr(xp, dtype_name)
    values = xp.asarray([0, 2, 7], dtype=dtype)
    twos = xp.asarray([2, 2, 2], dtype=dtype)

    assert xp.add(values, twos).tolist() == [2, 4, 9]
    assert xp.maximum(values, twos).tolist() == [2, 2, 7]
    assert xp.clip(values, min=twos).tolist() == [2, 2, 7]
    assert xp.floor_divide(values, twos).tolist() == [0, 1, 3]
    assert xp.remainder(values, twos).tolist() == [0, 0, 1]
    assert xp.bitwise_left_shift(values, twos).tolist() == [0, 8, 28]
    assert xp.bitwise_right_shift(values, twos).tolist() == [0, 0, 1]
    assert xp.max(values).dtype == dtype
    assert xp.max(values).tolist() == 7
    assert xp.argmax(values).tolist() == 2
    assert xp.square(values).tolist() == [0, 4, 49]
    assert xp.count_nonzero(values).tolist() == 2
    assert xp.nonzero(values)[0].tolist() == [1, 2]
    assert xp.flip(values, axis=0).tolist() == [7, 2, 0]
    assert xp.take(values, xp.asarray([2, 0], dtype=xp.int16)).tolist() == [
        7,
        0,
    ]
    assert xp.take_along_axis(
        values, xp.asarray([2, 0], dtype=xp.int16), axis=0
    ).tolist() == [7, 0]
    assert xp.searchsorted(
        values, xp.asarray([2, 6], dtype=dtype)
    ).tolist() == [1, 2]

    matrix = xp.asarray([[1, 2], [3, 4]], dtype=dtype)
    identity = xp.asarray([[1, 0], [0, 1]], dtype=dtype)
    assert xp.tril(matrix).tolist() == [[1, 0], [3, 4]]
    assert xp.triu(matrix).tolist() == [[1, 2], [0, 4]]
    assert xp.matmul(matrix, identity).tolist() == matrix.tolist()
    assert xp.tensordot(matrix, identity, axes=1).tolist() == matrix.tolist()
    assert xp.vecdot(values, twos).tolist() == 18


@pytest.mark.backend("torch")
@pytest.mark.parametrize("dtype_name", ("uint16", "uint32", "uint64"))
def test_torch_wide_unsigned_index_updates(dtype_name: str) -> None:
    xp = asc.backend("torch").xp
    dtype = getattr(xp, dtype_name)
    source = xp.asarray([5, 10, 20], dtype=dtype)
    indices = xp.asarray([0, 2], dtype=xp.int16)
    values = xp.asarray([3, 4], dtype=dtype)

    expected = (
        (asc.index_set, [3, 10, 4]),
        (asc.index_add, [8, 10, 24]),
        (asc.index_multiply, [15, 10, 80]),
        (asc.index_min, [3, 10, 4]),
        (asc.index_max, [5, 10, 20]),
    )
    for operation, result in expected:
        updated = operation(source, indices, values)
        assert updated.dtype == dtype
        assert updated.tolist() == result
        assert source.tolist() == [5, 10, 20]


@pytest.mark.backend("torch")
def test_torch_uint64_index_extrema_preserve_unsigned_ordering() -> None:
    xp = asc.backend("torch").xp
    source = xp.asarray([2**63 + 1, 2**63 + 5], dtype=xp.uint64)
    indices = xp.asarray([0, 1], dtype=xp.int8)
    values = xp.asarray([2**63 + 3, 2**63 + 2], dtype=xp.uint64)

    assert asc.index_min(source, indices, values).tolist() == [
        2**63 + 1,
        2**63 + 2,
    ]
    assert asc.index_max(source, indices, values).tolist() == [
        2**63 + 3,
        2**63 + 5,
    ]


@pytest.mark.backend("torch")
@pytest.mark.parametrize("dtype_name", ("float16", "bfloat16"))
def test_torch_low_precision_gamma_uses_supported_calculation_dtype(
    dtype_name: str,
) -> None:
    xp = asc.backend("torch").xp
    dtype = getattr(xp, dtype_name)
    samples, _next_state = asc.random.gamma(
        (32,),
        state=asc.random_state(7, backend="torch"),
        concentration=2.0,
        scale=0.5,
        dtype=dtype,
    )

    assert samples.dtype == dtype
    assert xp.all(xp.isfinite(samples)).tolist()
    assert xp.all(samples >= xp.zeros_like(samples)).tolist()


def test_data_loader_configuration_is_frozen_after_validation() -> None:
    loader = asc.data.DataLoader(
        asc.data.ArrayDataset(numpy.arange(4)), batch_size=2
    )
    expected = [batch.tolist() for batch in loader]

    for attribute, value in (
        ("batch_size", 0),
        ("shuffle", True),
        ("drop_last", True),
        ("sampler", object()),
    ):
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(loader, attribute, value)
    with pytest.raises(dataclasses.FrozenInstanceError):
        del loader.batch_size

    assert len(loader) == 2
    assert [batch.tolist() for batch in loader] == expected


def test_dataset_statistics_detach_first_sample_extrema() -> None:
    source = numpy.asarray([[1.0, 2.0]], dtype=numpy.float32)
    statistics = asc.data.dataset_statistics(asc.data.ArrayDataset(source))
    assert isinstance(statistics, asc.data.Statistics)
    minimum = typing.cast(numpy.ndarray, statistics.minimum)
    maximum = typing.cast(numpy.ndarray, statistics.maximum)

    assert not numpy.shares_memory(source, minimum)
    assert not numpy.shares_memory(source, maximum)
    source[...] = 100.0
    numpy.testing.assert_array_equal(minimum, [1.0, 2.0])
    numpy.testing.assert_array_equal(maximum, [1.0, 2.0])


@pytest.mark.backend("torch")
def test_torch_uint64_reductions_preserve_modular_values() -> None:
    xp = asc.backend("torch").xp
    values = xp.asarray([2**63, 1, 2], dtype=xp.uint64)

    total = xp.sum(values)
    cumulative = xp.cumulative_sum(values)
    product = xp.prod(xp.asarray([2**63, 2], dtype=xp.uint64))

    assert total.dtype == xp.uint64
    assert total.tolist() == 2**63 + 3
    assert cumulative.dtype == xp.uint64
    assert cumulative.tolist() == [2**63, 2**63 + 1, 2**63 + 3]
    assert product.dtype == xp.uint64
    assert product.tolist() == 0


@pytest.mark.backend("torch")
@pytest.mark.parametrize("source_name", ("int8", "uint64"))
@pytest.mark.parametrize("result_name", ("uint16", "uint32", "uint64"))
def test_torch_reductions_dispatch_from_explicit_result_dtype(
    source_name: str, result_name: str
) -> None:
    xp = asc.backend("torch").xp
    values = xp.asarray([1, 0, 2], dtype=getattr(xp, source_name))
    result_dtype = getattr(xp, result_name)

    total = xp.sum(values, dtype=result_dtype)
    cumulative = xp.cumulative_sum(values, dtype=result_dtype)

    assert total.dtype == result_dtype
    assert total.tolist() == 3
    assert cumulative.dtype == result_dtype
    assert cumulative.tolist() == [1, 1, 3]


@pytest.mark.backend("torch")
def test_torch_reductions_emulate_explicit_boolean_result_dtype() -> None:
    xp = asc.backend("torch").xp
    values = xp.asarray([1, 0, 2], dtype=xp.int8)

    total = xp.sum(values, dtype=xp.bool)
    product = xp.prod(values, dtype=xp.bool)
    cumulative_total = xp.cumulative_sum(values, dtype=xp.bool)
    cumulative_product = xp.cumulative_prod(values, dtype=xp.bool)

    assert total.dtype == xp.bool
    assert total.tolist() is True
    assert product.dtype == xp.bool
    assert product.tolist() is False
    assert cumulative_total.dtype == xp.bool
    assert cumulative_total.tolist() == [True, True, True]
    assert cumulative_product.dtype == xp.bool
    assert cumulative_product.tolist() == [True, False, False]


@pytest.mark.backend("torch")
def test_torch_uint64_diff_converts_boundary_arrays_to_carriers() -> None:
    xp = asc.backend("torch").xp
    values = xp.asarray([2**63, 2**63 + 2], dtype=xp.uint64)
    prepend = xp.asarray([2**63 - 1], dtype=xp.uint64)
    append = xp.asarray([2**63 + 5], dtype=xp.uint64)

    result = xp.diff(values, prepend=prepend, append=append)

    assert result.dtype == xp.uint64
    assert result.tolist() == [1, 2, 3]


@pytest.mark.backend("torch")
@pytest.mark.backend("jax")
@pytest.mark.parametrize("backend", ("numpy", "torch", "jax"))
def test_backend_namespace_hides_native_extensions(backend: str) -> None:
    xp = asc.backend(backend).xp

    for name in ("array", "tensor", "histogram", "median", "random"):
        assert not hasattr(xp, name)
        with pytest.raises(AttributeError, match="frozen Array API surface"):
            getattr(xp, name)


@pytest.mark.backend("torch")
@pytest.mark.backend("jax")
@pytest.mark.parametrize("backend", ("numpy", "torch", "jax"))
def test_named_linalg_extensions_do_not_expand_backend_namespace(
    backend: str,
) -> None:
    xp = asc.backend(backend).xp
    matrix = xp.asarray([[1, 2], [3, 4]], dtype=xp.int32)

    contraction = asc.linalg.einsum("ij,jk->ik", matrix, matrix)
    product = asc.linalg.kron(matrix, matrix)

    assert contraction.tolist() == [[7, 10], [15, 22]]
    assert product.shape == (4, 4)
    assert not hasattr(xp, "einsum")
    assert not hasattr(xp, "kron")


@pytest.mark.backend("jax")
def test_jax_x64_disabled_rejects_unavailable_promotions() -> None:
    import jax

    with jax.enable_x64(False):
        xp = asc.backend("jax").xp
        signed = xp.asarray([[-1]], dtype=xp.int32)
        unsigned = xp.asarray([[2**32 - 1]], dtype=xp.uint32)

        with pytest.raises(asc.DTypeError, match="release surface"):
            xp.result_type(xp.int32, xp.uint32)
        with pytest.raises(asc.DTypeError, match="release surface"):
            xp.linalg.matmul(signed, unsigned)


@pytest.mark.backend("torch")
@pytest.mark.parametrize("backend", ("numpy", "torch"))
def test_linalg_applies_portable_promotion(backend: str) -> None:
    xp = asc.backend(backend).xp
    signed = xp.asarray([[-1]], dtype=xp.int32)
    unsigned = xp.asarray([[2**32 - 1]], dtype=xp.uint32)

    result = xp.linalg.matmul(signed, unsigned)

    assert result.dtype == xp.int64
    assert result.tolist() == [[-(2**32 - 1)]]


@pytest.mark.backend("torch")
@pytest.mark.backend("jax")
@pytest.mark.parametrize("backend", ("numpy", "torch", "jax"))
def test_linalg_rank_and_optional_controls_are_portable(backend: str) -> None:
    xp = asc.backend(backend).xp
    vector = xp.asarray([1, 0], dtype=xp.int32)
    matrix = xp.asarray([[1, 0], [0, 1]], dtype=xp.int32)
    floating_matrix = xp.astype(matrix, xp.float32, copy=True)

    assert xp.linalg.matrix_rank(matrix).tolist() == 2
    for operation in (
        lambda: xp.linalg.outer(matrix, vector),
        lambda: xp.linalg.matrix_rank(vector),
        lambda: xp.linalg.cholesky(floating_matrix, upper=None),
        lambda: xp.linalg.cross(
            xp.asarray([1.0, 0.0, 0.0], dtype=xp.float32),
            xp.asarray([0.0, 1.0, 0.0], dtype=xp.float32),
            axis=None,
        ),
        lambda: xp.linalg.matrix_norm(floating_matrix, keepdims=None),
    ):
        with pytest.raises((TypeError, ValueError)):
            operation()


@pytest.mark.backend("torch")
@pytest.mark.backend("jax")
@pytest.mark.parametrize("backend", ("numpy", "torch", "jax"))
def test_normalized_linalg_methods_preflight_invalid_ranks_and_axes(
    backend: str,
) -> None:
    xp = asc.backend(backend).xp
    vector = xp.asarray([1.0, 2.0], dtype=xp.float32)
    matrix = xp.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=xp.float32)
    cross_vector = xp.asarray([1.0, 2.0, 3.0], dtype=xp.float32)

    operations = (
        lambda: xp.linalg.eigh(vector),
        lambda: xp.linalg.eig(vector),
        lambda: xp.linalg.qr(vector),
        lambda: xp.linalg.slogdet(vector),
        lambda: xp.linalg.svd(vector),
        lambda: xp.linalg.lstsq(vector, vector),
        lambda: xp.linalg.lstsq(matrix, xp.asarray(1.0, dtype=xp.float32)),
        lambda: xp.linalg.cross(cross_vector, cross_vector, axis=1),
    )

    for operation in operations:
        with pytest.raises(ValueError):
            operation()


@pytest.mark.backend("torch")
@pytest.mark.backend("jax")
@pytest.mark.parametrize("backend", ("numpy", "torch", "jax"))
def test_triangular_construction_requires_matrix_rank(backend: str) -> None:
    xp = asc.backend(backend).xp
    vector = xp.asarray([1, 2], dtype=xp.int32)

    with pytest.raises(ValueError, match="two-dimensional"):
        xp.tril(vector)
    with pytest.raises(ValueError, match="two-dimensional"):
        xp.triu(vector)


@pytest.mark.backend("torch")
@pytest.mark.backend("jax")
@pytest.mark.parametrize("backend", ("numpy", "torch", "jax"))
def test_clip_normalizes_admitted_mixed_integer_bounds(backend: str) -> None:
    xp = asc.backend(backend).xp
    values = xp.asarray([-2, 3], dtype=xp.int8)
    lower = xp.asarray([0, 1], dtype=xp.int16)

    result = xp.clip(values, min=lower)

    assert result.dtype == xp.int8
    assert result.tolist() == [0, 3]


@pytest.mark.backend("torch")
@pytest.mark.backend("jax")
@pytest.mark.parametrize("backend", ("numpy", "torch", "jax"))
def test_clip_compares_wide_bounds_before_result_narrowing(
    backend: str,
) -> None:
    xp = asc.backend(backend).xp
    values = xp.asarray([1, 1], dtype=xp.int8)
    lower = xp.asarray([200, -200], dtype=xp.int16)
    upper = xp.asarray([-200, 200], dtype=xp.int16)

    clipped_below = xp.clip(values, min=lower)
    clipped_above = xp.clip(values, max=upper)
    clipped_scalar = xp.clip(values[:1], min=200)

    assert clipped_below.dtype == xp.int8
    assert clipped_below.tolist() == [-56, 1]
    assert clipped_above.dtype == xp.int8
    assert clipped_above.tolist() == [56, 1]
    assert clipped_scalar.dtype == xp.int8
    assert clipped_scalar.tolist() == [-56]


@pytest.mark.backend("torch")
@pytest.mark.backend("jax")
@pytest.mark.parametrize("backend", ("torch", "jax"))
def test_xp_fft_frequency_uses_namespace_default_dtype(backend: str) -> None:
    setup = (
        "import torch\ntorch.set_default_dtype(torch.float64)"
        if backend == "torch"
        else ""
    )
    program = f"""
{setup}
import asc

selected = asc.backend({backend!r})
xp = selected.xp
expected = xp.asarray(0.0, device=selected.device).dtype
assert xp.fft.fftfreq(4).dtype == expected
assert xp.fft.rfftfreq(4).dtype == expected
assert selected.fft.fftfreq(4).dtype == expected
assert selected.fft.rfftfreq(4).dtype == expected
"""
    environment = dict(os.environ)
    environment["JAX_ENABLE_X64"] = "1"
    environment["JAX_PLATFORMS"] = "cpu"
    subprocess.run(
        [sys.executable, "-c", program],
        check=True,
        env=environment,
    )


@pytest.mark.backend("torch")
@pytest.mark.parametrize("dtype_name", ("complex64", "complex128"))
def test_torch_round_handles_complex_arrays(dtype_name: str) -> None:
    xp = asc.backend("torch").xp
    values = xp.asarray(
        [1.25 + 2.75j, -1.5 - 2.5j],
        dtype=getattr(xp, dtype_name),
    )

    rounded = xp.round(values)

    assert rounded.dtype == getattr(xp, dtype_name)
    assert rounded.tolist() == [1.0 + 3.0j, -2.0 - 2.0j]


@pytest.mark.backend("torch")
def test_torch_reshape_copy_false_is_safe_under_vmap() -> None:
    xp = asc.backend("torch").xp
    values = xp.reshape(xp.arange(12, dtype=xp.float32), (3, 4))
    reshape_rows = asc.vmap(
        lambda row: xp.reshape(row, (2, 2), copy=False),
        backend="torch",
    )

    result = reshape_rows(values)

    assert result.shape == (3, 2, 2)
    assert xp.reshape(result, (12,)).tolist() == list(range(12))


@pytest.mark.backend("torch")
def test_torch_uint64_take_indices_are_safe_under_vmap() -> None:
    xp = asc.backend("torch").xp
    values = xp.asarray([[10, 20, 30], [40, 50, 60]], dtype=xp.int32)
    indices = xp.asarray([[2, 0], [1, 2]], dtype=xp.uint64)
    mapped_take = asc.vmap(
        lambda row, row_indices: xp.take(row, row_indices),
        backend="torch",
    )
    mapped_take_along_axis = asc.vmap(
        lambda row, row_indices: xp.take_along_axis(row, row_indices, axis=0),
        backend="torch",
    )

    assert mapped_take(values, indices).tolist() == [[30, 10], [50, 60]]
    assert mapped_take_along_axis(values, indices).tolist() == [
        [30, 10],
        [50, 60],
    ]


@pytest.mark.backend("torch")
@pytest.mark.backend("jax")
@pytest.mark.parametrize("backend", ("numpy", "torch", "jax"))
def test_namespace_inspection_advertises_only_usable_cpu_device(
    backend: str,
) -> None:
    selected = asc.backend(backend)
    inspection = selected.xp.__array_namespace_info__()

    devices = inspection.devices()

    assert len(devices) == 1
    selected.xp.zeros((1,), device=devices[0])
    selected.xp.zeros((1,), device=inspection.default_device())


@pytest.mark.parametrize(
    "initializer",
    (random.glorot_uniform, random.lecun_uniform, random.he_uniform),
)
def test_uniform_initializers_reject_malformed_state(
    initializer: typing.Callable[..., object],
) -> None:
    malformed = typing.cast(random.RandomState, object())

    with pytest.raises(asc.RandomStateError, match="state"):
        initializer((2, 2), state=malformed)


@pytest.mark.backend("torch")
@pytest.mark.backend("jax")
@pytest.mark.parametrize("backend", ("numpy", "torch", "jax"))
def test_linalg_trace_accepts_native_dtype_controls(backend: str) -> None:
    selected = asc.backend(backend)
    xp = selected.xp
    matrix = xp.asarray([[1, 2], [3, 4]], dtype=xp.int8)

    for namespace in (xp.linalg, selected.linalg):
        result = namespace.trace(matrix, dtype=xp.int32)

        assert result.dtype == xp.int32
        assert result.tolist() == 5


@pytest.mark.backend("torch")
@pytest.mark.backend("jax")
@pytest.mark.parametrize("backend", ("numpy", "torch", "jax"))
@pytest.mark.parametrize("function_name", ("fftfreq", "rfftfreq"))
@pytest.mark.parametrize("dtype_name", ("int32", "complex64"))
def test_fft_frequency_facades_reject_nonreal_dtypes(
    backend: str,
    function_name: str,
    dtype_name: str,
) -> None:
    selected = asc.backend(backend)
    dtype = getattr(selected.xp, dtype_name)

    for namespace in (selected.xp.fft, selected.fft):
        with pytest.raises(asc.DTypeError, match="real floating"):
            getattr(namespace, function_name)(4, dtype=dtype)


@pytest.mark.backend("torch")
@pytest.mark.backend("jax")
@pytest.mark.parametrize("backend", ("numpy", "torch", "jax"))
def test_namespace_inspection_reports_the_active_dtype_surface(
    backend: str,
) -> None:
    selected = asc.backend(backend)
    inspection = selected.xp.__array_namespace_info__()

    reported = inspection.dtypes()
    integral = inspection.dtypes(kind="integral")

    assert set(reported) == set(asc.backend_info(backend).dtypes)
    assert all(
        selected.xp.isdtype(dtype, "integral") for dtype in integral.values()
    )
    assert {"uint16", "uint32"} <= set(integral)


@pytest.mark.backend("torch")
@pytest.mark.backend("jax")
@pytest.mark.parametrize("backend", ("numpy", "torch", "jax"))
def test_namespace_rejects_keyword_only_controls_passed_positionally(
    backend: str,
) -> None:
    selected = asc.backend(backend)
    xp = selected.xp
    values = xp.asarray([1.25, 2.75], device=selected.device)

    for call in (
        lambda: xp.sum(values, 0),
        lambda: xp.round(values, 1),
        lambda: xp.argsort(values, 0),
        lambda: xp.reshape(values, (2,), False),
    ):
        with pytest.raises(TypeError, match="positional"):
            call()

    assert xp.clip(values, 0.0, 2.0).shape == (2,)
    assert xp.reshape(values, (2,)).shape == (2,)


@pytest.mark.backend("torch")
@pytest.mark.backend("jax")
@pytest.mark.parametrize("backend", ("numpy", "torch", "jax"))
def test_linalg_rejects_keyword_only_controls_passed_positionally(
    backend: str,
) -> None:
    selected = asc.backend(backend)
    xp = selected.xp
    matrix = xp.asarray(
        [[2.0, 0.0], [0.0, 2.0]],
        dtype=xp.float32,
        device=selected.device,
    )
    vector = xp.asarray(
        [1.0, 2.0, 3.0], dtype=xp.float32, device=selected.device
    )

    calls = (
        lambda: selected.linalg.cholesky(matrix, False),
        lambda: selected.linalg.cross(vector, vector, -1),
        lambda: selected.linalg.diagonal(matrix, 0),
        lambda: selected.linalg.matrix_norm(matrix, 1),
        lambda: selected.linalg.matrix_rank(matrix, None),
        lambda: selected.linalg.pinv(matrix, None),
        lambda: selected.linalg.tensordot(matrix, matrix, 1),
        lambda: selected.linalg.trace(matrix, 0),
        lambda: selected.linalg.vecdot(vector, vector, -1),
        lambda: selected.linalg.vector_norm(vector, None),
    )
    for call in calls:
        with pytest.raises(TypeError, match="positionally"):
            call()

    assert selected.linalg.matrix_power(matrix, 2).shape == (2, 2)
    assert selected.linalg.matmul(matrix, matrix).shape == (2, 2)


@pytest.mark.backend("torch")
@pytest.mark.backend("jax")
@pytest.mark.parametrize("backend", ("numpy", "torch", "jax"))
def test_linalg_normalizes_none_norm_orders(backend: str) -> None:
    selected = asc.backend(backend)
    xp = selected.xp
    matrix = xp.asarray(
        [[1.0, 2.0], [3.0, 4.0]],
        dtype=xp.float32,
        device=selected.device,
    )
    vector = xp.asarray([3.0, 4.0], dtype=xp.float32, device=selected.device)

    matrix_norm = selected.linalg.matrix_norm(matrix, ord=None)
    vector_norm = selected.linalg.vector_norm(vector, ord=None)

    assert float(matrix_norm) == pytest.approx(numpy.sqrt(30.0))
    assert float(vector_norm) == pytest.approx(5.0)


def test_pytree_registration_rejects_an_explicit_empty_name() -> None:
    class EmptyNameNode:
        pass

    with pytest.raises(ValueError, match="non-empty"):
        tree.register_pytree_node(
            EmptyNameNode,
            lambda _value: ((), None),
            lambda _metadata, _values: EmptyNameNode(),
            name="",
        )


@pytest.mark.backend("torch")
@pytest.mark.backend("jax")
@pytest.mark.parametrize(
    ("backend", "dtype_name"),
    (
        ("numpy", "float16"),
        ("torch", "float16"),
        ("torch", "bfloat16"),
        ("jax", "float16"),
        ("jax", "bfloat16"),
    ),
)
@pytest.mark.parametrize(
    "function_name",
    (
        "cholesky",
        "eig",
        "eigh",
        "eigvals",
        "eigvalsh",
        "inv",
        "lstsq",
        "matrix_rank",
        "pinv",
        "qr",
        "solve",
        "svd",
        "svdvals",
    ),
)
def test_low_precision_cpu_linalg_rejects_unsupported_kernels_eagerly(
    backend: str,
    dtype_name: str,
    function_name: str,
) -> None:
    selected = asc.backend(backend)
    xp = selected.xp
    dtype = getattr(xp, dtype_name)
    matrix = xp.asarray(
        [[4.0, 1.0], [1.0, 3.0]],
        dtype=dtype,
        device=selected.device,
    )
    vector = xp.asarray([1.0, 2.0], dtype=dtype, device=selected.device)

    with pytest.raises(
        asc.CapabilityNotSupportedError, match="low-precision CPU"
    ):
        function = getattr(selected.linalg, function_name)
        if function_name in {"lstsq", "solve"}:
            function(matrix, vector)
        else:
            function(matrix)


@pytest.mark.backend("torch")
@pytest.mark.parametrize("backend", ("numpy", "torch"))
@pytest.mark.parametrize("function_name", ("det", "slogdet"))
def test_numpy_and_torch_low_precision_determinants_reject_eagerly(
    backend: str,
    function_name: str,
) -> None:
    selected = asc.backend(backend)
    matrix = selected.xp.asarray(
        [[4.0, 1.0], [1.0, 3.0]],
        dtype=selected.xp.float16,
        device=selected.device,
    )

    with pytest.raises(
        asc.CapabilityNotSupportedError, match="low-precision CPU"
    ):
        getattr(selected.linalg, function_name)(matrix)


@pytest.mark.backend("torch")
@pytest.mark.backend("jax")
@pytest.mark.parametrize(
    ("backend", "dtype_name"),
    (
        ("numpy", "float16"),
        ("torch", "float16"),
        ("torch", "bfloat16"),
        ("jax", "float16"),
        ("jax", "bfloat16"),
    ),
)
def test_low_precision_linalg_rejects_solver_dependent_controls(
    backend: str,
    dtype_name: str,
) -> None:
    selected = asc.backend(backend)
    matrix = selected.xp.asarray(
        [[4.0, 1.0], [1.0, 3.0]],
        dtype=getattr(selected.xp, dtype_name),
        device=selected.device,
    )

    for call in (
        lambda: selected.linalg.matrix_norm(matrix, ord=2),
        lambda: selected.linalg.matrix_norm(matrix, ord="nuc"),
        lambda: selected.linalg.matrix_power(matrix, -1),
    ):
        with pytest.raises(
            asc.CapabilityNotSupportedError, match="low-precision CPU"
        ):
            call()

    assert selected.linalg.matrix_norm(matrix).shape == ()
    assert selected.linalg.matrix_power(matrix, 2).shape == (2, 2)


@pytest.mark.backend("torch")
@pytest.mark.backend("jax")
@pytest.mark.parametrize(
    ("backend", "dtype_name"),
    (
        ("torch", "float16"),
        ("torch", "bfloat16"),
        ("jax", "float16"),
        ("jax", "bfloat16"),
    ),
)
@pytest.mark.parametrize("function_name", ("ihfft", "rfft", "rfftn"))
def test_low_precision_real_fft_rejects_unsupported_cpu_kernels_eagerly(
    backend: str,
    dtype_name: str,
    function_name: str,
) -> None:
    selected = asc.backend(backend)
    values = selected.xp.asarray(
        [1.0, 2.0, 3.0, 4.0],
        dtype=getattr(selected.xp, dtype_name),
        device=selected.device,
    )

    with pytest.raises(
        asc.CapabilityNotSupportedError, match="low-precision CPU"
    ):
        getattr(selected.fft, function_name)(values)


def test_numpy_float16_real_fft_remains_supported() -> None:
    selected = asc.backend("numpy")
    values = selected.xp.asarray(
        [1.0, 2.0, 3.0, 4.0], dtype=selected.xp.float16
    )

    for function_name in ("ihfft", "rfft", "rfftn"):
        result = getattr(selected.fft, function_name)(values)
        assert result.shape


@pytest.mark.backend("torch")
@pytest.mark.backend("jax")
@pytest.mark.parametrize("backend", ("numpy", "torch", "jax"))
@pytest.mark.parametrize("function_name", ("take", "take_along_axis"))
@pytest.mark.parametrize("index", (3, -4))
def test_take_operations_reject_out_of_bounds_indices(
    backend: str, function_name: str, index: int
) -> None:
    selected = asc.backend(backend)
    xp = selected.xp
    values = xp.asarray([10, 20, 30], dtype=xp.int32, device=selected.device)
    indices = xp.asarray([index], dtype=xp.int32, device=selected.device)

    with pytest.raises(IndexError, match="out of bounds"):
        if function_name == "take":
            xp.take(values, indices)
        else:
            xp.take_along_axis(values, indices, axis=0)


@pytest.mark.backend("torch")
@pytest.mark.backend("jax")
@pytest.mark.parametrize("backend", ("numpy", "torch", "jax"))
def test_integer_power_rejects_negative_scalar_and_array_exponents(
    backend: str,
) -> None:
    selected = asc.backend(backend)
    xp = selected.xp
    bases = xp.asarray([2, 3], dtype=xp.int32, device=selected.device)
    exponents = xp.asarray([-1, 2], dtype=xp.int32, device=selected.device)

    for exponent in (-1, exponents):
        with pytest.raises(ValueError, match="non-negative"):
            xp.pow(bases, exponent)


@pytest.mark.backend("torch")
@pytest.mark.backend("jax")
@pytest.mark.parametrize("backend", ("numpy", "torch", "jax"))
def test_namespace_rejects_negative_shift_and_repeat_counts(
    backend: str,
) -> None:
    selected = asc.backend(backend)
    xp = selected.xp
    values = xp.asarray([1, 2], dtype=xp.int32, device=selected.device)
    counts = xp.asarray([-1, 1], dtype=xp.int32, device=selected.device)

    for call in (
        lambda: xp.bitwise_left_shift(values, -1),
        lambda: xp.bitwise_right_shift(values, counts),
        lambda: xp.repeat(values, -1),
        lambda: xp.repeat(values, counts),
    ):
        with pytest.raises(ValueError, match="non-negative"):
            call()


@pytest.mark.backend("jax")
def test_jax_take_and_integer_power_checks_survive_jit() -> None:
    selected = asc.backend("jax")
    xp = selected.xp
    values = xp.asarray([10, 20, 30], dtype=xp.int32)
    bases = xp.asarray([2, 3], dtype=xp.int32)
    take = asc.jit(lambda indices: xp.take(values, indices), backend="jax")
    power = asc.jit(lambda exponents: xp.pow(bases, exponents), backend="jax")

    with pytest.raises(IndexError, match=r"jit:.*out of bounds"):
        take(xp.asarray([3], dtype=xp.int32))
    with pytest.raises(ValueError, match=r"jit:.*non-negative"):
        power(xp.asarray([-1, 2], dtype=xp.int32))


@pytest.mark.backend("torch")
def test_torch_take_checks_survive_vmap() -> None:
    selected = asc.backend("torch")
    xp = selected.xp
    values = xp.asarray([[10, 20, 30], [40, 50, 60]], dtype=xp.int32)
    indices = xp.asarray([[0], [3]], dtype=xp.int32)
    take = asc.vmap(
        lambda row, row_indices: xp.take(row, row_indices),
        backend="torch",
    )

    with pytest.raises(IndexError, match="out of bounds"):
        take(values, indices)


@pytest.mark.backend("torch")
def test_torch_uint64_oversized_shifts_produce_zero() -> None:
    selected = asc.backend("torch")
    xp = selected.xp
    values = xp.asarray([1, 1, 1, 1], dtype=xp.uint64)
    shifts = xp.asarray([64, 65, 2**63, 2**64 - 1], dtype=xp.uint64)

    assert xp.bitwise_left_shift(values, shifts).tolist() == [0, 0, 0, 0]
    assert xp.bitwise_right_shift(values, shifts).tolist() == [0, 0, 0, 0]


def test_same_dtype_binary_linalg_skips_redundant_casts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import array_api_compat.numpy as native_namespace

    selected = asc.backend("numpy")
    xp = selected.xp
    matrix = xp.asarray([[2.0, 0.0], [0.0, 2.0]], dtype=xp.float32)
    vector = xp.asarray([2.0, 4.0], dtype=xp.float32)

    def reject_cast(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("same-dtype linalg operands must not be copied")

    monkeypatch.setattr(native_namespace, "astype", reject_cast)

    assert selected.linalg.matmul(matrix, matrix).shape == (2, 2)
    assert selected.linalg.outer(vector, vector).shape == (2, 2)
    assert selected.linalg.solve(matrix, vector).shape == (2,)
    assert asc.linalg.kron(matrix, matrix).shape == (4, 4)
    assert asc.linalg.einsum("ij,jk->ik", matrix, matrix).shape == (2, 2)
    assert selected.linalg.lstsq(matrix, vector).solution.shape == (2,)


@pytest.mark.backend("torch")
@pytest.mark.backend("jax")
@pytest.mark.parametrize("backend", ("numpy", "torch", "jax"))
def test_fft_rejects_nonpositive_transform_lengths(backend: str) -> None:
    selected = asc.backend(backend)
    xp = selected.xp
    values = xp.asarray([1.0, 2.0], dtype=xp.float32)

    for count in (0, -1):
        with pytest.raises(ValueError, match="strictly positive"):
            selected.fft.rfft(values, n=count)
    for shape in ((), (0,), (-1,)):
        with pytest.raises(ValueError, match="strictly positive"):
            selected.fft.rfftn(values, s=shape)


@pytest.mark.backend("torch")
@pytest.mark.backend("jax")
@pytest.mark.parametrize("backend", ("numpy", "torch", "jax"))
def test_linalg_rejects_invalid_tolerances_and_matrix_norm_orders(
    backend: str,
) -> None:
    selected = asc.backend(backend)
    xp = selected.xp
    matrix = xp.asarray([[1.0, 0.0], [0.0, 0.5]], dtype=xp.float32)
    complex_tolerance = xp.asarray(0.25 + 0.0j, dtype=xp.complex64)
    integer_tolerance = xp.asarray(0, dtype=xp.int32)
    nonscalar_tolerance = xp.asarray([0.25], dtype=xp.float32)
    scalar_tolerance = xp.asarray(0.25, dtype=xp.float32)

    for function in (selected.linalg.matrix_rank, selected.linalg.pinv):
        for invalid_tolerance in (complex_tolerance, integer_tolerance):
            with pytest.raises(asc.DTypeError, match="real floating"):
                function(matrix, rtol=invalid_tolerance)
        with pytest.raises(ValueError, match="batch shape"):
            function(matrix, rtol=nonscalar_tolerance)
        assert function(matrix, rtol=scalar_tolerance).shape in {(), (2, 2)}
    for order in (0, 3, -3, 0.5, float("nan")):
        with pytest.raises(ValueError, match="numeric ord"):
            selected.linalg.matrix_norm(matrix, ord=order)
    for order in (
        -2,
        -1,
        1,
        2,
        float("-inf"),
        float("inf"),
        "fro",
        "nuc",
        None,
    ):
        assert selected.linalg.matrix_norm(matrix, ord=order).shape == ()
