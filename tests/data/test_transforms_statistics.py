# Copyright 2026 AI4SciComp contributors.
# SPDX-License-Identifier: Apache-2.0
"""Backend-neutral transform and streaming-statistics contracts."""

from __future__ import annotations

import numpy
import pytest

import asc
from asc import data


@pytest.mark.parametrize("backend", ("numpy", "torch", "jax"))
def test_scalers_fit_transform_inverse_and_constant_policy(
    backend: str,
) -> None:
    selected = asc.backend(backend)
    xp = selected.xp
    values = xp.asarray([[1.0, 4.0], [3.0, 4.0]], dtype=xp.float32)
    standard, normalized = data.StandardScaler().fit_transform(values)
    numpy.testing.assert_allclose(
        numpy.asarray(normalized), [[-1, 0], [1, 0]], atol=1e-6
    )
    numpy.testing.assert_allclose(
        numpy.asarray(standard.inverse_transform(normalized)),
        numpy.asarray(values),
        atol=1e-6,
    )
    minimum, scaled = data.MinMaxScaler((-1.0, 1.0)).fit_transform(values)
    numpy.testing.assert_allclose(
        numpy.asarray(scaled), [[-1, -1], [1, -1]], atol=1e-6
    )
    numpy.testing.assert_allclose(
        numpy.asarray(minimum.inverse_transform(scaled)),
        numpy.asarray(values),
        atol=1e-6,
    )
    with pytest.raises(asc.DataSpecError, match="fit"):
        data.StandardScaler().transform(values)
    with pytest.raises(asc.DTypeError):
        data.MinMaxScaler().fit(xp.asarray([1, 2], dtype=xp.int32))


@pytest.mark.parametrize("backend", ("numpy", "torch", "jax"))
def test_fitted_scalers_preserve_sample_shape_and_reject_bad_broadcasts(
    backend: str,
) -> None:
    selected = asc.backend(backend)
    xp = selected.xp
    training = xp.asarray([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]], dtype=xp.float32)
    sample = xp.asarray([2.0, 3.0, 4.0], dtype=xp.float32)
    incompatible = xp.asarray([[2.0], [3.0], [4.0]], dtype=xp.float32)

    for scaler in (data.StandardScaler(), data.MinMaxScaler()):
        fitted = scaler.fit(training)
        transformed = fitted.transform(sample)
        assert transformed.shape == sample.shape
        assert fitted.inverse_transform(transformed).shape == sample.shape
        with pytest.raises(asc.DataSpecError, match="shape"):
            fitted.transform(incompatible)


@pytest.mark.parametrize("backend", ("numpy", "torch", "jax"))
def test_fitted_scalers_reject_nonfloating_transform_inputs(
    backend: str,
) -> None:
    selected = asc.backend(backend)
    xp = selected.xp
    training = xp.asarray([[1.0], [3.0]], dtype=xp.float32)
    integer_data = xp.asarray([[1], [3]], dtype=xp.int32)

    for scaler in (data.StandardScaler(), data.MinMaxScaler()):
        fitted = scaler.fit(training)
        for function in (fitted.transform, fitted.inverse_transform):
            with pytest.raises(asc.DTypeError, match="real floating"):
                function(integer_data)


@pytest.mark.parametrize("backend", ("numpy", "torch", "jax"))
def test_fitted_scalers_enforce_singleton_feature_dimensions(
    backend: str,
) -> None:
    selected = asc.backend(backend)
    xp = selected.xp
    training = xp.asarray([[1.0], [3.0]], dtype=xp.float32)
    wrong_sample = xp.ones((4,), dtype=xp.float32)
    wrong_batch = xp.ones((3, 4), dtype=xp.float32)

    for scaler in (data.StandardScaler(), data.MinMaxScaler()):
        fitted = scaler.fit(training)
        for value in (wrong_sample, wrong_batch):
            with pytest.raises(asc.DataSpecError, match="shape"):
                fitted.transform(value)
            with pytest.raises(asc.DataSpecError, match="shape"):
                fitted.inverse_transform(value)


@pytest.mark.parametrize("backend", ("numpy", "torch", "jax"))
def test_scalar_statistic_scalers_enforce_fitted_input_rank(
    backend: str,
) -> None:
    selected = asc.backend(backend)
    xp = selected.xp
    training = xp.asarray([1.0, 3.0], dtype=xp.float32, device=selected.device)
    scalar_sample = xp.asarray(2.0, dtype=xp.float32, device=selected.device)
    fitted_rank = xp.asarray(
        [2.0, 4.0], dtype=xp.float32, device=selected.device
    )
    incompatible = xp.asarray(
        [[2.0, 4.0]], dtype=xp.float32, device=selected.device
    )

    for scaler in (data.StandardScaler(), data.MinMaxScaler()):
        fitted = scaler.fit(training)
        assert fitted.transform(scalar_sample).shape == ()
        assert fitted.transform(fitted_rank).shape == fitted_rank.shape
        for function in (fitted.transform, fitted.inverse_transform):
            with pytest.raises(asc.DataSpecError, match="shape"):
                function(incompatible)


@pytest.mark.parametrize("backend", ("numpy", "torch", "jax"))
def test_scalers_reject_empty_reduction_axes(backend: str) -> None:
    selected = asc.backend(backend)
    values = selected.xp.empty((0, 2), dtype=selected.xp.float32)

    for scaler in (data.StandardScaler(), data.MinMaxScaler()):
        with pytest.raises(asc.DataSpecError, match="nonzero extents"):
            scaler.fit(values)


@pytest.mark.parametrize(
    ("backend", "value"),
    (
        ("numpy", 2.0**-149),
        ("torch", 2.0**-149),
        ("jax", 2.0**127),
    ),
)
def test_standard_scaler_constant_extremes_use_unit_scale(
    backend: str,
    value: float,
) -> None:
    selected = asc.backend(backend)
    values = selected.xp.asarray(
        [[value], [value]],
        dtype=selected.xp.float32,
        device=selected.device,
    )

    fitted, transformed = data.StandardScaler().fit_transform(values)
    restored = fitted.inverse_transform(transformed)

    numpy.testing.assert_array_equal(numpy.asarray(transformed), 0.0)
    numpy.testing.assert_array_equal(
        numpy.asarray(restored), numpy.asarray(values)
    )


@pytest.mark.parametrize("backend", ("numpy", "torch", "jax"))
@pytest.mark.parametrize(
    "feature_range",
    (
        (0.0, 1e-50),
        (0.0, 1e40),
        (-1e40, 1e40),
        (1.0, 1.0 + 2**-25),
    ),
)
def test_minmax_feature_range_must_survive_fitted_dtype(
    backend: str,
    feature_range: tuple[float, float],
) -> None:
    selected = asc.backend(backend)
    values = selected.xp.asarray([0.0, 1.0], dtype=selected.xp.float32)

    with pytest.raises(asc.DataSpecError, match="feature_range"):
        data.MinMaxScaler(feature_range).fit(values)


def test_minmax_constructor_maps_oversized_scalars_to_data_error() -> None:
    with pytest.raises(asc.DataSpecError, match="feature_range"):
        data.MinMaxScaler((0, 10**10_000))


def test_composition_field_and_conversion_transforms() -> None:
    identity = data.Identity()
    value = {"x": numpy.asarray([1.0]), "label": "one"}
    assert identity.transform(value) is value
    assert identity.inverse_transform(value) is value
    shifted = data.LambdaTransform(lambda item: item + 1, lambda item: item - 1)
    composed = data.Compose((shifted, shifted))
    fitted, result = composed.fit_transform(1)
    assert result == 3
    assert fitted.inverse_transform(result) == 1
    with pytest.raises(asc.CapabilityNotSupportedError, match="inverse"):
        data.LambdaTransform(lambda item: item).inverse_transform(1)
    selected = data.SelectFields(("label", "x"))
    assert tuple(selected.transform(value)) == ("label", "x")
    renamed = data.RenameFields({"label": "target"})
    output = renamed.transform(value)
    assert tuple(output) == ("x", "target")
    assert renamed.inverse_transform(output)["label"] == "one"
    numpy_tree = data.ToBackend("numpy").transform(value)
    assert asc.backend_of(numpy_tree["x"]) == "numpy"
    assert asc.backend_of(data.ToDevice("cpu").transform(numpy_tree)["x"]) == (
        "numpy"
    )
    cast = data.CastDType(numpy.float64).transform(value)
    assert cast["x"].dtype == numpy.float64


def test_transform_validation_and_collisions() -> None:
    for factory in (
        lambda: data.Compose((object(),)),
        lambda: data.SelectFields(()),
        lambda: data.SelectFields(("x", "x")),
        lambda: data.RenameFields({}),
        lambda: data.RenameFields({"x": "z", "y": "z"}),
        lambda: data.MinMaxScaler((1.0, 1.0)),
        lambda: data.MinMaxScaler((float("nan"), 1.0)),
    ):
        with pytest.raises(asc.DataSpecError):
            factory()
    with pytest.raises(asc.DataSpecError, match="mapping"):
        data.SelectFields(("x",)).transform(1)
    with pytest.raises(asc.DataSpecError, match="missing"):
        data.SelectFields(("x",)).transform({"y": 1})
    with pytest.raises(asc.DataSpecError, match="mapping"):
        data.RenameFields({"x": "z"}).transform(1)
    with pytest.raises(asc.DataSpecError, match="source"):
        data.RenameFields({"x": "z"}).transform({"y": 1})
    with pytest.raises(asc.DataSpecError, match="collide"):
        data.RenameFields({"x": "y"}).transform({"x": 1, "y": 2})


@pytest.mark.parametrize("backend", ("numpy", "torch", "jax"))
def test_streaming_statistics_arrays_and_mappings(backend: str) -> None:
    selected = asc.backend(backend)
    xp = selected.xp
    values = xp.asarray([[1.0, 2.0], [3.0, 6.0], [5.0, 10.0]], dtype=xp.float32)
    statistics = data.dataset_statistics(data.ArrayDataset(values))
    assert isinstance(statistics, data.Statistics)
    assert statistics.count == 3
    numpy.testing.assert_allclose(numpy.asarray(statistics.mean), [3, 6])
    numpy.testing.assert_allclose(
        numpy.asarray(statistics.variance), [8 / 3, 32 / 3], rtol=1e-5
    )
    mapping = data.MappingDataset(
        {
            "x": data.ArrayDataset(values),
            "y": data.ArrayDataset(values + 1),
        }
    )
    by_field = data.dataset_statistics(mapping, fields=("y",))
    assert isinstance(by_field, dict)
    assert tuple(by_field) == ("y",)
    assert by_field["y"].count == 3


def test_streaming_statistics_errors() -> None:
    with pytest.raises(asc.DatasetError, match="empty"):
        data.dataset_statistics(data.ArrayDataset(numpy.empty((0, 2))))
    with pytest.raises(asc.DTypeError):
        data.dataset_statistics(data.ArrayDataset(numpy.asarray([[1], [2]])))
    with pytest.raises(asc.DataSpecError, match="shape"):
        data.dataset_statistics(
            data.TransformDataset(
                data.ArrayDataset(numpy.asarray([1.0, 2.0])),
                lambda value: numpy.ones((int(value),)),
            )
        )
    mapping = data.MappingDataset(
        {"x": data.ArrayDataset(numpy.ones((2, 1), dtype=numpy.float32))}
    )
    with pytest.raises(asc.DataSpecError, match="missing"):
        data.dataset_statistics(mapping, fields=("y",))
    with pytest.raises(asc.DataSpecError, match="fields"):
        data.dataset_statistics(mapping, fields=())
    with pytest.raises(asc.DataSpecError, match="fields"):
        data.dataset_statistics(
            data.ArrayDataset(numpy.ones((2, 1), dtype=numpy.float32)),
            fields=("x",),
        )
