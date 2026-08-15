# Copyright 2026 AI4SciComp contributors.
# SPDX-License-Identifier: Apache-2.0
"""CPU parity tests for linalg, Fourier, operations, and metrics."""

from __future__ import annotations

import numpy
import pytest

import asc
from asc import metrics, ops
from asc import typing as asc_typing
from asc.linalg import EigResult, LstsqResult, einsum, gkron, kron, lstsq
from tests import helpers

NATIVE_BACKENDS: tuple[asc_typing.BackendName, ...] = helpers.NATIVE_BACKENDS


@pytest.mark.parametrize("backend", NATIVE_BACKENDS)
def test_linalg_complete_surface(backend: asc_typing.BackendName) -> None:
    selected = asc.backend(backend)
    xp = selected.xp
    matrix = xp.asarray([[4.0, 1.0], [1.0, 3.0]], dtype=xp.float32)
    vector = xp.asarray([1.0, 2.0], dtype=xp.float32)
    assert selected.linalg.cholesky(matrix).shape == (2, 2)
    assert selected.linalg.det(matrix).shape == ()
    assert selected.linalg.diagonal(matrix).shape == (2,)
    assert selected.linalg.eigh(matrix).eigenvectors.shape == (2, 2)
    assert selected.linalg.eigvalsh(matrix).shape == (2,)
    assert selected.linalg.inv(matrix).shape == (2, 2)
    assert selected.linalg.matrix_norm(matrix).shape == ()
    assert selected.linalg.matrix_power(matrix, 2).shape == (2, 2)
    assert selected.linalg.matrix_rank(matrix).shape == ()
    assert selected.linalg.matrix_transpose(matrix).shape == (2, 2)
    assert selected.linalg.outer(vector, vector).shape == (2, 2)
    assert selected.linalg.pinv(matrix).shape == (2, 2)
    assert selected.linalg.qr(matrix).Q.shape == (2, 2)
    assert selected.linalg.slogdet(matrix).logabsdet.shape == ()
    assert selected.linalg.solve(matrix, vector).shape == (2,)
    assert selected.linalg.svd(matrix).S.shape == (2,)
    assert selected.linalg.svdvals(matrix).shape == (2,)
    assert selected.linalg.tensordot(matrix, vector, axes=1).shape == (2,)
    assert selected.linalg.trace(matrix).shape == ()
    assert selected.linalg.vecdot(vector, vector).shape == ()
    assert selected.linalg.vector_norm(vector).shape == ()
    result = lstsq(matrix, vector)
    assert isinstance(result, LstsqResult)
    assert result.solution.shape == (2,)
    assert result.singular_values.shape == (2,)


@pytest.mark.parametrize("backend", NATIVE_BACKENDS)
def test_eigh_honors_the_selected_triangle(
    backend: asc_typing.BackendName,
) -> None:
    selected = asc.backend(backend)
    xp = selected.xp
    matrix = xp.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=xp.float32)

    lower = selected.linalg.eigh(matrix, UPLO="L").eigenvalues
    upper = selected.linalg.eigh(matrix, UPLO="U").eigenvalues

    numpy.testing.assert_allclose(
        numpy.asarray(lower), [-0.85410196, 5.854102], rtol=1e-5
    )
    numpy.testing.assert_allclose(numpy.asarray(upper), [0.0, 5.0], atol=1e-6)


@pytest.mark.parametrize("backend", NATIVE_BACKENDS)
def test_general_eigen_batched_and_tensor_products(
    backend: asc_typing.BackendName,
) -> None:
    selected = asc.backend(backend)
    xp = selected.xp
    rotation = xp.asarray([[0.0, -1.0], [1.0, 0.0]], dtype=xp.float32)
    result = selected.linalg.eig(rotation)
    assert isinstance(result, EigResult)
    numpy.testing.assert_allclose(
        numpy.sort(numpy.abs(numpy.asarray(result.eigenvalues))),
        [1, 1],
        rtol=1e-5,
    )
    batch = xp.stack((rotation, rotation), axis=0)
    assert selected.linalg.det(batch).shape == (2,)
    vector = xp.asarray([1.0, 2.0], dtype=xp.float32)
    numpy.testing.assert_allclose(
        helpers.as_numpy(einsum("i,i->", vector, vector)), 5
    )
    assert kron(rotation, rotation).shape == (4, 4)
    expanded = gkron((rotation, rotation))
    assert tuple(value.shape for value in expanded) == ((4, 2), (4, 2))
    with pytest.raises(ValueError, match="non-empty"):
        gkron(())
    with pytest.raises(ValueError, match="two-dimensional"):
        gkron((vector,))


@pytest.mark.parametrize("backend", NATIVE_BACKENDS)
def test_linalg_promotes_mixed_operands_before_dispatch(
    backend: asc_typing.BackendName,
) -> None:
    selected = asc.backend(backend)
    xp = selected.xp
    integer = xp.asarray([1, 2], dtype=xp.int16)
    floating = xp.asarray([3.0, 4.0], dtype=xp.float32)

    contraction = einsum("i,i->", integer, floating)
    product = kron(integer, floating)

    assert contraction.dtype == xp.float32
    assert product.dtype == xp.float32
    numpy.testing.assert_allclose(numpy.asarray(contraction), 11.0)
    numpy.testing.assert_allclose(numpy.asarray(product), [3.0, 4.0, 6.0, 8.0])

    if "float64" in asc.backend_info(backend).dtypes:
        matrix_dtype = xp.float32
        right_dtype = xp.float64
        expected_dtype = xp.float64
    else:
        matrix_dtype = xp.float16
        right_dtype = xp.float32
        expected_dtype = xp.float32
    matrix = xp.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=matrix_dtype)
    right = xp.asarray([2.0, 3.0], dtype=right_dtype)

    solution = lstsq(matrix, right).solution

    assert solution.dtype == expected_dtype
    numpy.testing.assert_allclose(numpy.asarray(solution), [2.0, 3.0])


@pytest.mark.parametrize("backend", NATIVE_BACKENDS)
def test_kron_supports_boolean_operands_portably(
    backend: asc_typing.BackendName,
) -> None:
    selected = asc.backend(backend)
    xp = selected.xp
    first = xp.asarray([True, False], dtype=xp.bool)
    second = xp.asarray([True, True], dtype=xp.bool)

    result = kron(first, second)

    numpy.testing.assert_array_equal(
        numpy.asarray(result), [True, True, False, False]
    )


@pytest.mark.parametrize("backend", NATIVE_BACKENDS)
def test_einsum_supports_boolean_contractions_portably(
    backend: asc_typing.BackendName,
) -> None:
    selected = asc.backend(backend)
    xp = selected.xp
    first = xp.asarray(
        [[True, False, True], [False, False, True]], dtype=xp.bool
    )
    second = xp.asarray(
        [[False, True], [True, False], [True, False]], dtype=xp.bool
    )

    result = einsum("ij,jk->ik", first, second)

    assert result.dtype == xp.bool
    numpy.testing.assert_array_equal(
        numpy.asarray(result), [[True, True], [True, False]]
    )


@pytest.mark.parametrize("backend", NATIVE_BACKENDS)
def test_fft_complete_surface_and_round_trips(
    backend: asc_typing.BackendName,
) -> None:
    selected = asc.backend(backend)
    xp = selected.xp
    value = xp.asarray([1.0, 2.0, 3.0, 4.0], dtype=xp.float32)
    complex_value = xp.astype(value, xp.complex64, copy=True)
    transformed = selected.fft.fft(complex_value, norm="backward")
    restored = selected.fft.ifft(transformed, norm="backward")
    numpy.testing.assert_allclose(
        numpy.asarray(restored).real, [1, 2, 3, 4], rtol=1e-5
    )
    real = selected.fft.rfft(value)
    numpy.testing.assert_allclose(
        helpers.as_numpy(selected.fft.irfft(real, n=4)), [1, 2, 3, 4], rtol=1e-5
    )
    for forward, inverse, source in (
        ("fftn", "ifftn", complex_value),
        ("rfftn", "irfftn", value),
    ):
        frequency = getattr(selected.fft, forward)(source)
        output = getattr(selected.fft, inverse)(frequency, s=(4,), axes=(0,))
        numpy.testing.assert_allclose(
            numpy.asarray(output).real, [1, 2, 3, 4], rtol=1e-5
        )
    assert selected.fft.hfft(selected.fft.ihfft(value)).shape == (4,)
    assert selected.fft.fftshift(value).shape == (4,)
    assert selected.fft.ifftshift(value).shape == (4,)
    assert selected.fft.fftfreq(4, dtype=xp.float32).dtype == xp.float32
    assert selected.fft.rfftfreq(4, dtype=xp.float32).shape == (3,)


@pytest.mark.parametrize("backend", NATIVE_BACKENDS)
def test_half_precision_fft_frequencies_scale_before_narrowing(
    backend: asc_typing.BackendName,
) -> None:
    selected = asc.backend(backend)
    xp = selected.xp

    for function in (selected.fft.fftfreq, selected.fft.rfftfreq):
        frequencies = function(140_000, dtype=xp.float16)
        assert frequencies.dtype == xp.float16
        assert bool(xp.all(xp.isfinite(frequencies)))
        assert float(xp.max(xp.abs(frequencies))) <= 0.5


@pytest.mark.parametrize("backend", NATIVE_BACKENDS)
def test_fft_frequencies_reject_unrepresentable_bins(
    backend: asc_typing.BackendName,
) -> None:
    selected = asc.backend(backend)
    spacing = float(numpy.nextafter(numpy.float16(0), numpy.float16(1)))

    for function in (selected.fft.fftfreq, selected.fft.rfftfreq):
        with pytest.raises(ValueError, match="bins are not representable"):
            function(1000, d=spacing, dtype=selected.xp.float16)


@pytest.mark.parametrize("backend", NATIVE_BACKENDS)
def test_portable_operations_parity(backend: asc_typing.BackendName) -> None:
    xp = asc.backend(backend).xp
    vector = xp.asarray([1.0, 2.0, 3.0], dtype=xp.float32)
    numpy.testing.assert_allclose(
        helpers.as_numpy(ops.diag(vector, offset=1)), numpy.diag([1, 2, 3], 1)
    )
    assert ops.flatten(xp.reshape(vector, (1, 3))).shape == (3,)
    assert ops.ravel(vector, copy=True).shape == (3,)
    coordinates = (
        xp.asarray([0, 1], dtype=xp.int32),
        xp.asarray([1, 0], dtype=xp.int32),
    )
    flat = ops.ravel_multi_index(coordinates, (2, 2))
    numpy.testing.assert_allclose(helpers.as_numpy(flat), [1, 2])
    recovered = ops.unravel_index(flat, (2, 2))
    numpy.testing.assert_allclose(helpers.as_numpy(recovered[0]), [0, 1])
    source = xp.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=xp.float32)
    for mode in ("constant", "edge", "reflect", "symmetric", "wrap"):
        result = ops.pad(source, ((1, 1), (1, 1)), mode=mode)
        expected = numpy.pad(numpy.asarray(source), ((1, 1), (1, 1)), mode=mode)
        numpy.testing.assert_allclose(helpers.as_numpy(result), expected)
    singleton = xp.asarray([5.0], dtype=xp.float32)
    numpy.testing.assert_array_equal(
        helpers.as_numpy(ops.pad(singleton, ((2, 3),), mode="reflect")),
        numpy.pad([5.0], ((2, 3),), mode="reflect"),
    )
    kernel = xp.asarray([1.0, 1.0], dtype=xp.float32)
    for mode in ("valid", "same", "full"):
        numpy.testing.assert_allclose(
            helpers.as_numpy(ops.convolve1d(vector, kernel, mode=mode)),
            numpy.convolve([1, 2, 3], [1, 1], mode=mode),
        )
    short = xp.asarray([1.0, 2.0], dtype=xp.float32)
    long_kernel = xp.asarray([1.0, 1.0, 1.0, 1.0], dtype=xp.float32)
    numpy.testing.assert_allclose(
        helpers.as_numpy(ops.convolve1d(short, long_kernel, mode="same")),
        numpy.convolve([1, 2], [1, 1, 1, 1], mode="same"),
    )
    numpy.testing.assert_allclose(
        helpers.as_numpy(ops.convolve1d(short, long_kernel, mode="valid")),
        numpy.convolve([1, 2], [1, 1, 1, 1], mode="valid"),
    )
    numpy.testing.assert_allclose(
        helpers.as_numpy(ops.moving_mean(vector, 2)), [1.5, 2.5]
    )
    with pytest.raises(ValueError, match="must not exceed"):
        ops.moving_mean(vector, 4)


@pytest.mark.parametrize("backend", NATIVE_BACKENDS)
def test_diag_preserves_nonfinite_values_only_on_the_diagonal(
    backend: asc_typing.BackendName,
) -> None:
    selected = asc.backend(backend)
    values = selected.xp.asarray(
        [float("inf"), float("nan")],
        dtype=selected.xp.float32,
        device=selected.device,
    )

    result = numpy.asarray(ops.diag(values))

    assert numpy.isposinf(result[0, 0])
    assert numpy.isnan(result[1, 1])
    assert result[0, 1] == 0.0
    assert result[1, 0] == 0.0


@pytest.mark.parametrize("backend", NATIVE_BACKENDS)
def test_moving_mean_widens_low_precision_calculation(
    backend: asc_typing.BackendName,
) -> None:
    selected = asc.backend(backend)
    values = selected.xp.ones((100_000,), dtype=selected.xp.float16)

    result = ops.moving_mean(values, values.shape[0])

    assert result.dtype == values.dtype
    numpy.testing.assert_array_equal(helpers.as_numpy(result), [1.0])


@pytest.mark.parametrize("backend", NATIVE_BACKENDS)
def test_multi_index_operations_widen_narrow_integer_inputs(
    backend: asc_typing.BackendName,
) -> None:
    xp = asc.backend(backend).xp
    coordinate = xp.asarray([15], dtype=xp.int8)

    flat = ops.ravel_multi_index((coordinate, coordinate), (16, 16))
    recovered = ops.unravel_index(xp.asarray([127], dtype=xp.int8), (16, 16))

    numpy.testing.assert_array_equal(numpy.asarray(flat), [255])
    numpy.testing.assert_array_equal(numpy.asarray(recovered[0]), [7])
    numpy.testing.assert_array_equal(numpy.asarray(recovered[1]), [15])


@pytest.mark.backend("jax")
def test_multi_index_operations_keep_bounds_checks_inside_jax_jit() -> None:
    selected = asc.backend("jax")
    xp = selected.xp
    ravelled = asc.jit(
        lambda first, second: ops.ravel_multi_index((first, second), (16, 16)),
        backend="jax",
    )
    unravelled = asc.jit(
        lambda indices: ops.unravel_index(indices, (16, 16)),
        backend="jax",
    )

    coordinate = xp.asarray([15], dtype=xp.int8)
    numpy.testing.assert_array_equal(
        numpy.asarray(ravelled(coordinate, coordinate)), [255]
    )
    recovered = unravelled(xp.asarray([127], dtype=xp.int8))
    numpy.testing.assert_array_equal(numpy.asarray(recovered[0]), [7])
    numpy.testing.assert_array_equal(numpy.asarray(recovered[1]), [15])
    with pytest.raises(IndexError, match="ravel_multi_index"):
        ravelled(
            xp.asarray([16], dtype=xp.int32),
            xp.asarray([0], dtype=xp.int32),
        )
    with pytest.raises(IndexError, match="unravel_index"):
        unravelled(xp.asarray([256], dtype=xp.int32))


@pytest.mark.parametrize("backend", NATIVE_BACKENDS)
def test_isclose_treats_only_equal_nonfinite_values_as_close(
    backend: asc_typing.BackendName,
) -> None:
    xp = asc.backend(backend).xp
    first = xp.asarray(
        [numpy.inf, -numpy.inf, numpy.inf, numpy.nan], dtype=xp.float32
    )
    second = xp.asarray(
        [numpy.inf, -numpy.inf, -numpy.inf, numpy.nan], dtype=xp.float32
    )

    result = ops.isclose(first, second, equal_nan=True)

    numpy.testing.assert_array_equal(
        numpy.asarray(result), [True, True, False, True]
    )
    assert ops.allclose(first[:2], second[:2])


@pytest.mark.parametrize("backend", NATIVE_BACKENDS)
def test_activations_metrics_and_metadata(
    backend: asc_typing.BackendName,
) -> None:
    xp = asc.backend(backend).xp
    value = xp.asarray([-2.0, 0.0, 2.0], dtype=xp.float32)
    for name in (
        "elu",
        "gelu",
        "leaky_relu",
        "relu",
        "selu",
        "sigmoid",
        "silu",
        "softplus",
        "softsign",
        "tanhshrink",
    ):
        result = getattr(ops, name)(value)
        assert result.shape == value.shape
        assert bool(xp.all(xp.isfinite(result)))
    prediction = xp.asarray([1.0, 2.0, 4.0], dtype=xp.float32)
    target = xp.asarray([1.0, 3.0, 2.0], dtype=xp.float32)
    assert float(
        metrics.mean_absolute_error(prediction, target)
    ) == pytest.approx(1.0)
    assert float(
        metrics.mean_squared_error(prediction, target)
    ) == pytest.approx(5 / 3)
    assert float(
        metrics.root_mean_squared_error(prediction, target)
    ) == pytest.approx(numpy.sqrt(5 / 3))
    assert metrics.relative_l2_error(prediction, target).shape == ()
    assert metrics.r2_score(prediction, target).shape == ()
    assert ops.eps(value) > 0
    assert ops.tiny(value) > 0
    minimum, maximum = ops.finite_range(value)
    assert minimum < 0 < maximum
    assert bool(xp.all(ops.isclose(value, value)))
    assert ops.allclose(value, value)
    ops.assert_allclose(value, value)


@pytest.mark.parametrize("backend", NATIVE_BACKENDS)
@pytest.mark.parametrize("scale", (1e20, 1e-30))
def test_dimensionless_metrics_use_scale_stable_reductions(
    backend: asc_typing.BackendName,
    scale: float,
) -> None:
    xp = asc.backend(backend).xp
    relative_target = xp.asarray([scale], dtype=xp.float32)
    relative_prediction = xp.asarray([2 * scale], dtype=xp.float32)
    target = xp.asarray([scale, 2 * scale], dtype=xp.float32)
    prediction = xp.asarray([scale, scale], dtype=xp.float32)

    assert float(
        metrics.relative_l2_error(relative_prediction, relative_target)
    ) == pytest.approx(1.0)
    assert float(metrics.r2_score(prediction, target)) == pytest.approx(-1.0)


@pytest.mark.parametrize("backend", NATIVE_BACKENDS)
def test_activation_coefficients_are_python_scalars_and_zero_slope_is_stable(
    backend: asc_typing.BackendName,
) -> None:
    xp = asc.backend(backend).xp
    value = xp.asarray([-xp.inf, -1.0, 1.0], dtype=xp.float32)
    coefficient = xp.asarray(0.5, dtype=xp.float32)

    numpy.testing.assert_array_equal(
        numpy.asarray(ops.leaky_relu(value, negative_slope=0)),
        [0.0, 0.0, 1.0],
    )
    with pytest.raises(ValueError, match="Python real scalar"):
        ops.leaky_relu(value, negative_slope=coefficient)
    with pytest.raises(ValueError, match="Python real scalar"):
        ops.elu(value, alpha=coefficient)


@pytest.mark.parametrize("backend", NATIVE_BACKENDS)
def test_stable_activation_limits_at_infinity(
    backend: asc_typing.BackendName,
) -> None:
    xp = asc.backend(backend).xp
    values = xp.asarray([-numpy.inf, numpy.inf, numpy.nan], dtype=xp.float32)

    silu_values = numpy.asarray(ops.silu(values))
    softsign_values = numpy.asarray(ops.softsign(values))

    numpy.testing.assert_array_equal(silu_values[:2], [0.0, numpy.inf])
    numpy.testing.assert_array_equal(softsign_values[:2], [-1.0, 1.0])
    assert numpy.isnan(silu_values[2])
    assert numpy.isnan(softsign_values[2])


@pytest.mark.parametrize("backend", ("torch", "jax"))
def test_activation_and_signal_gradients(
    backend: asc_typing.BackendName,
) -> None:
    xp = asc.backend(backend).xp
    value = xp.asarray([-0.5, 0.5, 1.5], dtype=xp.float32)
    kernel = xp.asarray([0.25, 0.75], dtype=xp.float32)

    def objective(array: object) -> object:
        return xp.sum(
            ops.silu(array) + ops.convolve1d(array, kernel, mode="same")
        )

    gradient = asc.grad(objective, backend=backend)(value)
    assert gradient.shape == value.shape
    assert bool(xp.all(xp.isfinite(gradient)))


@pytest.mark.backend("jax")
def test_jax_fft_and_operations_compile() -> None:
    xp = asc.backend("jax").xp
    value = xp.asarray([1.0, 2.0, 3.0, 4.0], dtype=xp.float32)
    compiled = asc.jit(
        lambda array: asc.backend("jax").fft.irfft(
            asc.backend("jax").fft.rfft(ops.relu(array)), n=4
        ),
        backend="jax",
    )
    numpy.testing.assert_allclose(
        helpers.as_numpy(compiled(value)), [1, 2, 3, 4]
    )


@pytest.mark.parametrize("backend", NATIVE_BACKENDS)
def test_operation_error_contracts(backend: asc_typing.BackendName) -> None:
    xp = asc.backend(backend).xp
    floating = xp.asarray([1.0, 2.0], dtype=xp.float32)
    integer = xp.asarray([0, 1], dtype=xp.int32)
    with pytest.raises(ValueError, match="one-dimensional"):
        ops.diag(xp.reshape(floating, (1, 2)))
    with pytest.raises(IndexError, match="out of bounds"):
        ops.ravel_multi_index((integer + 2,), (2,))
    with pytest.raises(IndexError, match="out of bounds"):
        ops.unravel_index(integer + 2, (2,))
    with pytest.raises(asc.DTypeError):
        ops.moving_mean(integer, 1)
    with pytest.raises(ValueError, match="kernel"):
        ops.convolve1d(floating, xp.asarray([], dtype=xp.float32))
    empty = xp.asarray([], dtype=xp.float32)
    for mode in ("valid", "same", "full"):
        with pytest.raises(ValueError, match="input signal"):
            ops.convolve1d(empty, floating, mode=mode)
    with pytest.raises(ValueError, match="tolerances"):
        ops.isclose(floating, floating, rtol=-1)
    with pytest.raises(AssertionError):
        ops.assert_allclose(floating, floating + 1)
