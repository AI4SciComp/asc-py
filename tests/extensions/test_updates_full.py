# Copyright 2026 AI4SciComp contributors.
# SPDX-License-Identifier: Apache-2.0
"""Functional update semantics, parity, aliasing, gradients, and JIT."""

from __future__ import annotations

import numpy
import pytest

import asc

BACKENDS = ("numpy", "torch", "jax")


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    ("name", "expected"),
    (
        ("index_set", [[10.0, 20.0], [3.0, 4.0], [30.0, 40.0]]),
        ("index_add", [[11.0, 22.0], [3.0, 4.0], [35.0, 46.0]]),
        ("index_multiply", [[10.0, 40.0], [3.0, 4.0], [150.0, 240.0]]),
        ("index_min", [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        ("index_max", [[10.0, 20.0], [3.0, 4.0], [30.0, 40.0]]),
    ),
)
def test_all_updates_no_mutation(
    backend: str, name: str, expected: object
) -> None:
    selected = asc.backend(backend)
    xp = selected.xp
    source = xp.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=xp.float32)
    original = numpy.asarray(source).copy()
    indices = xp.asarray([0, 2], dtype=xp.int16)
    values = xp.asarray([[10.0, 20.0], [30.0, 40.0]], dtype=xp.float32)
    result = getattr(asc, name)(source, indices, values)
    numpy.testing.assert_allclose(numpy.asarray(result), expected)
    numpy.testing.assert_allclose(numpy.asarray(source), original)
    assert result is not source


@pytest.mark.parametrize("backend", BACKENDS)
def test_axis_broadcast_and_promotion(backend: str) -> None:
    selected = asc.backend(backend)
    xp = selected.xp
    source = xp.asarray([[1, 2, 3], [4, 5, 6]], dtype=xp.int32)
    indices = xp.asarray([0, 2], dtype=xp.int8)
    values = xp.asarray([[0.5, 1.5], [2.5, 3.5]], dtype=xp.float32)
    result = asc.scatter_add(source, indices, values, axis=1)
    numpy.testing.assert_allclose(
        numpy.asarray(result), [[1.5, 2, 4.5], [6.5, 5, 9.5]]
    )
    one_row = asc.index_add(
        xp.zeros((2, 3), dtype=xp.float32),
        xp.asarray([1], dtype=xp.int32),
        xp.asarray([1.0, 2.0, 3.0], dtype=xp.float32),
    )
    numpy.testing.assert_allclose(
        numpy.asarray(one_row), [[0, 0, 0], [1, 2, 3]]
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_updates_widen_narrow_indices_before_bounds_checks(
    backend: str,
) -> None:
    selected = asc.backend(backend)
    xp = selected.xp
    source = xp.zeros((256,), dtype=xp.float32)
    indices = xp.asarray([0], dtype=xp.int8)
    values = xp.asarray([2.0], dtype=xp.float32)

    result = asc.index_add(source, indices, values)

    assert float(numpy.asarray(result)[0]) == pytest.approx(2.0)


@pytest.mark.parametrize("backend", BACKENDS)
def test_update_validation(backend: str) -> None:
    selected = asc.backend(backend)
    xp = selected.xp
    source = xp.ones((2, 2), dtype=xp.float32)
    values = xp.ones((2, 2), dtype=xp.float32)
    with pytest.raises(asc.IndexUpdateError, match="signed"):
        asc.index_add(source, xp.asarray([0, 1], dtype=xp.uint8), values)
    with pytest.raises(asc.IndexUpdateError, match="out of bounds"):
        asc.index_add(source, xp.asarray([0, 2], dtype=xp.int32), values)
    with pytest.raises(asc.DuplicateIndexError):
        asc.index_set(source, xp.asarray([0, 0], dtype=xp.int32), values)
    with pytest.raises(asc.IndexUpdateError, match="one-dimensional"):
        asc.index_add(source, xp.asarray([[0]], dtype=xp.int32), values)
    with pytest.raises(asc.IndexUpdateError, match="axis"):
        asc.index_add(source, xp.asarray([0], dtype=xp.int32), values, axis=4)
    with pytest.raises(asc.DTypeError):
        asc.index_max(
            xp.asarray([1 + 1j], dtype=xp.complex64),
            xp.asarray([0], dtype=xp.int32),
            xp.asarray([2 + 1j], dtype=xp.complex64),
        )


@pytest.mark.parametrize("backend", ("torch", "jax"))
def test_update_gradients(backend: str) -> None:
    selected = asc.backend(backend)
    xp = selected.xp
    source = xp.asarray([1.0, 2.0, 3.0], dtype=xp.float32)
    indices = xp.asarray([0, 2], dtype=xp.int32)
    values = xp.asarray([4.0, 5.0], dtype=xp.float32)

    def objective(array: object) -> object:
        return xp.sum(asc.index_add(array, indices, values) ** 2)

    gradient = asc.grad(objective, backend=backend)(source)
    numpy.testing.assert_allclose(
        numpy.asarray(gradient), [10, 4, 16], rtol=1e-5
    )


@pytest.mark.backend("jax")
def test_jax_updates_jit_checks_bounds_and_duplicates() -> None:
    selected = asc.backend("jax")
    xp = selected.xp
    source = xp.zeros((3,), dtype=xp.float32)
    values = xp.asarray([1.0, 2.0], dtype=xp.float32)
    add = asc.jit(
        lambda array, index, update: asc.scatter_add(array, index, update),
        backend="jax",
    )
    result = add(source, xp.asarray([0, 2], dtype=xp.int32), values)
    numpy.testing.assert_allclose(numpy.asarray(result), [1, 0, 2])
    with pytest.raises(asc.IndexUpdateError):
        add(source, xp.asarray([0, 4], dtype=xp.int32), values)
    set_values = asc.jit(
        lambda array, index, update: asc.scatter_set(array, index, update),
        backend="jax",
    )
    with pytest.raises(asc.DuplicateIndexError, match="duplicate"):
        set_values(source, xp.asarray([0, 0], dtype=xp.int32), values)

    vectorized_set = asc.vmap(
        asc.index_set,
        backend="jax",
        in_axes=(0, 0, 0),
    )
    with pytest.raises(asc.DuplicateIndexError, match="duplicate"):
        vectorized_set(
            xp.zeros((2, 3), dtype=xp.float32),
            xp.asarray([[0, 0], [1, 1]], dtype=xp.int32),
            xp.ones((2, 2), dtype=xp.float32),
        )


@pytest.mark.backend("jax")
def test_jax_non_set_updates_do_not_build_quadratic_duplicate_matrices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from asc.backends import jax as jax_adapter

    def reject_eye(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("non-set updates must not build an N x N eye")

    monkeypatch.setattr(vars(jax_adapter)["_jax_numpy"], "eye", reject_eye)
    selected = asc.backend("jax")
    xp = selected.xp
    size = 4096
    source = xp.ones((size,), dtype=xp.float32)
    indices = xp.arange(size, dtype=xp.int32)
    values = xp.ones((size,), dtype=xp.float32)

    for name in ("index_add", "index_multiply", "index_min", "index_max"):
        result = getattr(asc, name)(source, indices, values)
        assert result.shape == source.shape


@pytest.mark.backend("torch")
def test_mixed_backends_rejected_before_update() -> None:
    numpy_value = numpy.asarray([1.0, 2.0])
    torch = asc.backend("torch")
    with pytest.raises(asc.MixedBackendError):
        asc.index_add(
            numpy_value,
            torch.xp.asarray([0], dtype=torch.xp.int32),
            torch.xp.asarray([1.0]),
        )
