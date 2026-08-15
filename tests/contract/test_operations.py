# Copyright 2026 AI4SciComp contributors.
# SPDX-License-Identifier: Apache-2.0

import typing

import numpy
import pytest

import asc
from asc import typing as asc_typing
from tests import helpers


class _JaxConfig(typing.Protocol):  # pylint: disable=too-few-public-methods
    """Typed configuration query used by the x64 contract test."""

    def read(self, name: str) -> bool:
        """Read a named JAX configuration flag."""
        ...  # pylint: disable=unnecessary-ellipsis


@pytest.mark.parametrize("backend", helpers.BACKENDS)
@pytest.mark.parametrize(
    ("shape", "axis", "keepdims", "expected_shape", "expected"),
    [
        ((2, 2), None, False, (), 30.0),
        ((2, 2), 0, True, (1, 2), [[10.0, 20.0]]),
        ((2, 2), (0, 1), True, (1, 1), [[30.0]]),
    ],
)
def test_sum_of_squares_contract(
    backend: asc_typing.BackendName,
    shape: tuple[int, ...],
    axis: asc_typing.Axis,
    keepdims: bool,
    expected_shape: tuple[int, ...],
    expected: float | list[list[float]],
) -> None:
    value = helpers.float_array(backend, [[1.0, 2.0], [3.0, 4.0]])
    before = helpers.as_numpy(value).copy()

    result = asc.sum_of_squares(value, axis=axis, keepdims=keepdims)

    assert result.shape == expected_shape
    assert asc.backend_info(result).name == backend
    numpy.testing.assert_allclose(helpers.as_numpy(result), expected)
    numpy.testing.assert_array_equal(helpers.as_numpy(value), before)
    assert value.shape == shape


@pytest.mark.parametrize("backend", helpers.BACKENDS)
def test_sum_of_squares_empty_and_dtype_error(
    backend: asc_typing.BackendName,
) -> None:
    selected = helpers.namespace(backend)
    empty = selected.empty((0,), dtype=selected.float32)

    result = asc.sum_of_squares(empty)

    assert result.shape == ()
    assert float(helpers.as_numpy(result)) == 0.0
    integer = helpers.int_array(backend, [1, 2])
    with pytest.raises(asc.DTypeError, match="real floating-point"):
        asc.sum_of_squares(integer)


@pytest.mark.parametrize("backend", helpers.BACKENDS)
@pytest.mark.parametrize(
    ("axis", "keepdims", "message"),
    [
        (True, False, "axis must be"),
        ((True,), False, "axis must be"),
        (None, "x", "keepdims must be"),
    ],
)
def test_sum_of_squares_rejects_invalid_reduction_controls(
    backend: asc_typing.BackendName,
    axis: object,
    keepdims: object,
    message: str,
) -> None:
    value = helpers.float_array(backend, [[1.0, 2.0], [3.0, 4.0]])

    with pytest.raises(TypeError, match=message):
        asc.sum_of_squares(
            value,
            axis=typing.cast(asc_typing.Axis, axis),
            keepdims=typing.cast(bool, keepdims),
        )


@pytest.mark.parametrize("backend", helpers.BACKENDS)
@pytest.mark.parametrize("axis", ((0, 0), (0, -2), 2, -3))
def test_sum_of_squares_rejects_invalid_axis_values(
    backend: asc_typing.BackendName,
    axis: asc_typing.Axis,
) -> None:
    value = helpers.float_array(backend, [[1.0, 2.0], [3.0, 4.0]])

    with pytest.raises(ValueError, match="axis"):
        asc.sum_of_squares(value, axis=axis)


@pytest.mark.parametrize("backend", helpers.BACKENDS)
@pytest.mark.parametrize("shape", [(), (0,), (2, 0), (2, 3)])
def test_create_full_contract(
    backend: asc_typing.BackendName,
    shape: tuple[int, ...],
) -> None:
    selected = helpers.namespace(backend)
    context = asc.CreationContext(
        namespace=selected,
        backend=backend,
        dtype=selected.float32,
    )

    result = asc.create_full(shape, 2.5, context=context)

    assert result.shape == shape
    assert result.dtype == selected.float32
    assert asc.backend_info(result).name == backend
    numpy.testing.assert_array_equal(
        helpers.as_numpy(result), numpy.full(shape, 2.5)
    )


@pytest.mark.parametrize("shape", [[1], (-1,), (True,), (1.5,)])
def test_create_full_rejects_invalid_shapes(shape: object) -> None:
    selected = helpers.namespace("numpy")
    context = asc.CreationContext(
        namespace=selected,
        backend="numpy",
        dtype=selected.float32,
    )
    with pytest.raises(asc.ContextError, match="shape"):
        asc.create_full(
            typing.cast(asc_typing.Shape, shape),
            1.0,
            context=context,
        )


@pytest.mark.parametrize("fill_value", [numpy.asarray(1.0)])
def test_create_full_rejects_array_valued_fills(fill_value: object) -> None:
    selected = helpers.namespace("numpy")
    context = asc.CreationContext(selected, "numpy")
    with pytest.raises(asc.ContextError, match="array-valued fill"):
        asc.create_full((1,), fill_value, context=context)


def test_create_full_rejects_non_numeric_fill_without_explicit_dtype() -> None:
    selected = helpers.namespace("numpy")
    context = asc.CreationContext(selected, "numpy")

    with pytest.raises(asc.ContextError, match="representable"):
        asc.create_full((1,), "value", context=context)


@pytest.mark.backend("torch")
def test_create_full_rejects_foreign_array_valued_fill() -> None:
    selected = helpers.namespace("numpy")
    context = asc.CreationContext(selected, "numpy")
    with pytest.raises(asc.ContextError, match="array-valued fill"):
        asc.create_full(
            (1,), helpers.float_array("torch", 1.0), context=context
        )


@pytest.mark.backend("jax")
def test_jax_float64_respects_caller_owned_x64_configuration() -> None:
    import jax

    selected = helpers.namespace("jax")
    jax_config = typing.cast(_JaxConfig, jax.config)
    if jax_config.read("jax_enable_x64"):
        context = asc.CreationContext(
            selected,
            "jax",
            dtype=selected.float64,
        )
        result = asc.create_full((2,), 1.0, context=context)
        assert result.dtype == selected.float64
    else:
        with pytest.raises(asc.ContextError, match="release surface"):
            asc.CreationContext(
                selected,
                "jax",
                dtype=selected.float64,
            )


@pytest.mark.backend("jax")
def test_checked_jax_context_maps_disabled_x64_as_backend_rejection() -> None:
    import jax

    with jax.enable_x64():
        selected = asc.backend("jax")
        context = asc.CreationContext(
            selected.xp,
            "jax",
            dtype=selected.xp.float64,
        )

    with (
        jax.enable_x64(False),
        pytest.raises(asc.ContextError, match="backend rejected"),
    ):
        asc.create_full((2,), 1.0, context=context)


@pytest.mark.backend("jax")
def test_jax_ml_dtypes_bfloat16_has_canonical_provenance() -> None:
    import ml_dtypes

    selected = asc.backend("jax", dtype=ml_dtypes.bfloat16)

    result = selected.xp.asarray(
        [1.0], dtype=ml_dtypes.bfloat16, device=selected.device
    )

    assert result.dtype == selected.xp.bfloat16
    assert asc.is_array(result)


@pytest.mark.backend("jax")
def test_stale_jax_float64_is_rejected_when_x64_becomes_disabled() -> None:
    import jax
    import jax.numpy

    with jax.enable_x64():
        value = jax.numpy.asarray([1.0], dtype=jax.numpy.float64)

    with jax.enable_x64(False):
        assert not asc.is_array(value)
        with pytest.raises(asc.UnsupportedCapabilityError, match="dense CPU"):
            asc.array_namespace(value)
        with pytest.raises(asc.UnsupportedCapabilityError, match="dense CPU"):
            asc.sum_of_squares(value)
