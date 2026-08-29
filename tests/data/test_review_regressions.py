# Copyright 2026 AI4SciComp contributors.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for atomic data orchestration and persisted trees."""

from __future__ import annotations

import collections
import json
import os
import pathlib
import subprocess
import sys
import typing

import numpy
import pytest

import asc
from asc import data
from asc.data import dataset as data_dataset
from asc.data import io as data_io


def test_composite_datasets_reject_boolean_scalar_indices() -> None:
    class PermissiveDataset(data.Dataset[object]):
        def __len__(self) -> int:
            return 2

        def __getitem__(self, index: object) -> object:
            return index

    source = PermissiveDataset()
    composites = (
        data.TupleDataset(source),
        data.MappingDataset({"value": source}),
        data.TransformDataset(source, lambda value: value),
    )

    for dataset in composites:
        with pytest.raises(IndexError, match="Boolean indices"):
            dataset[True]


@pytest.mark.parametrize("num_samples", (0, 1))
def test_weighted_sampler_rejects_malformed_state(num_samples: int) -> None:
    malformed = typing.cast(asc.RandomState, object())

    with pytest.raises(asc.DataLoaderError, match="state"):
        data.WeightedRandomSampler([1.0], num_samples, state=malformed)


@pytest.mark.backend("jax")
def test_weighted_sampler_uses_the_jax_state_device() -> None:
    program = """
import asc
from asc import data
import jax

devices = jax.devices("cpu")
assert len(devices) == 2
state = asc.random_state(7, backend="jax", device=devices[1])
sampler = data.WeightedRandomSampler([1.0, 2.0], 8, state=state)
samples = list(sampler)
assert len(samples) == 8
assert set(samples) <= {0, 1}
"""
    environment = dict(os.environ)
    environment["JAX_PLATFORMS"] = "cpu"
    environment["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"

    subprocess.run(
        [sys.executable, "-c", program],
        check=True,
        env=environment,
    )


@pytest.mark.backend("jax")
def test_jax_random_state_round_trips_device_and_batched_keys() -> None:
    program = """
import importlib.metadata

import asc
import jax
import numpy

devices = jax.devices("cpu")
assert len(devices) == 2
configured = asc.backend(
    "jax",
    device=devices[1],
    dtype=asc.backend("jax").xp.float32,
)

constant = asc.random.constant((2,), 1.0, backend=configured)
state = asc.random_state(11, backend=configured)
assert constant.device is devices[1]
assert state.key.device is devices[1]

restored = asc.RandomState.from_json(state.to_json())
assert restored.key.device is devices[1]
numpy.testing.assert_array_equal(
    numpy.asarray(jax.random.key_data(restored.key)),
    numpy.asarray(jax.random.key_data(state.key)),
)

batched_key = jax.device_put(
    jax.random.split(jax.random.key(13), 3),
    devices[1],
)
batched = asc.RandomState(
    "jax",
    batched_key,
    importlib.metadata.version("jax"),
)
_, batched = asc.vmap(
    lambda current: asc.random.normal((1,), state=current),
    backend="jax",
)(batched)
batched_restored = asc.RandomState.from_json(batched.to_json())
assert batched_restored.key.shape == (3,)
assert batched_restored.key.device is devices[1]
numpy.testing.assert_array_equal(
    numpy.asarray(jax.random.key_data(batched_restored.key)),
    numpy.asarray(jax.random.key_data(batched.key)),
)

assert asc.random.constant(
    (1,), 1.0, backend=configured, device=devices[0]
).device is devices[0]
assert asc.random_state(
    17, backend=configured, device=devices[0]
).key.device is devices[0]
"""
    environment = dict(os.environ)
    environment["JAX_PLATFORMS"] = "cpu"
    environment["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"

    subprocess.run(
        [sys.executable, "-c", program],
        check=True,
        env=environment,
    )


def test_array_dataset_index_dtype_uses_maximum_valid_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    maxima: list[int] = []

    def record_maximum(
        xp: object,
        backend: str,
        maximum: int,
        operation: str,
    ) -> object:
        del backend, operation
        maxima.append(maximum)
        return xp.int32

    monkeypatch.setattr(data_dataset, "safe_index_dtype", record_maximum)
    dataset = data.ArrayDataset(numpy.arange(3))

    dataset[::-1]
    dataset[numpy.asarray([0], dtype=numpy.int32)]
    dataset[[0]]

    assert maxima == [2, 2, 2]


def test_filtered_stream_remains_iterable_for_loading_and_statistics() -> None:
    class Stream(data.IterableDataset[object]):
        def __iter__(self):
            return iter(
                numpy.asarray([value], dtype=numpy.float32)
                for value in range(6)
            )

    filtered = data.FilteredDataset(
        Stream(), lambda sample: int(sample[0]) % 2 == 0
    )
    nested = data.FilteredDataset(filtered, lambda sample: int(sample[0]) < 5)

    batches = list(data.DataLoader(nested, batch_size=2))
    assert [batch.shape for batch in batches] == [(2, 1), (1, 1)]
    numpy.testing.assert_array_equal(batches[0], [[0.0], [2.0]])
    statistics = data.dataset_statistics(nested)
    assert statistics.count == 3
    numpy.testing.assert_array_equal(statistics.mean, [2.0])
    with pytest.raises(TypeError, match="stable length"):
        len(data.DataLoader(nested, batch_size=2))
    with pytest.raises(asc.DatasetError, match="finite"):
        data.DataModule().add_dataset("train", "stream", nested)


def test_malformed_npz_metadata_is_wrapped(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "malformed-metadata.npz"
    numpy.savez(
        path,
        values=numpy.arange(2),
        __asc_metadata__=numpy.asarray([[1, 2]], dtype=numpy.uint8),
    )

    with pytest.raises(asc.DataFormatError, match="one-dimensional"):
        data.load_npz(path)


@pytest.mark.parametrize("format_name", ("hdf5", "mat"))
def test_object_leaves_are_rejected_during_tree_load(
    tmp_path: pathlib.Path,
    format_name: str,
) -> None:
    if format_name == "hdf5":
        import h5py

        path = data.save_hdf5(tmp_path / "unsafe.h5", numpy.asarray([1.0]))
        with h5py.File(path, "r+") as file:
            group = file["leaves"]
            del group["0"]
            group.create_dataset(
                "0",
                data=numpy.asarray(["unsafe"], dtype=object),
                dtype=h5py.string_dtype(encoding="utf-8"),
            )
    else:
        import scipy.io

        path = data.save_mat(tmp_path / "unsafe.mat", numpy.asarray([1.0]))
        payload = {
            name: value
            for name, value in scipy.io.loadmat(path).items()
            if not name.startswith("__")
        }
        unsafe = numpy.empty((1, 1), dtype=object)
        unsafe[0, 0] = "unsafe"
        payload["asc_leaf_0"] = unsafe
        scipy.io.savemat(path, payload, appendmat=False)

    with pytest.raises(asc.DataFormatError, match="object arrays"):
        getattr(data, f"load_{format_name}")(path)


def test_register_splits_preflights_every_stage() -> None:
    dataset = data.ArrayDataset(numpy.arange(10))
    existing = data.ArrayDataset(numpy.arange(2))
    module = data.DataModule()
    module.add_dataset("validation", "experiment", existing)

    with pytest.raises(asc.DatasetError, match="duplicate"):
        module.register_splits(
            dataset,
            name="experiment",
            train=0.6,
            validation=0.2,
            test=0.2,
        )

    assert dict(module.datasets("train")) == {}
    assert module.get_dataset("validation", "experiment") is existing
    assert dict(module.datasets("test")) == {}


@pytest.mark.parametrize("format_name", ("hdf5", "mat"))
def test_persisted_tree_reconstruction_errors_are_wrapped(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    format_name: str,
) -> None:
    suffix = ".h5" if format_name == "hdf5" else ".mat"
    path = getattr(data, f"save_{format_name}")(
        tmp_path / f"tree{suffix}", {"x": numpy.arange(2)}
    )

    def fail(_spec: object, _leaves: object) -> object:
        raise RuntimeError("unavailable custom node")

    monkeypatch.setattr(data_io, "tree_unflatten", fail)
    with pytest.raises(asc.DataFormatError, match="tree"):
        getattr(data, f"load_{format_name}")(path)


@pytest.mark.parametrize("format_name", ("hdf5", "mat"))
def test_safe_tree_loaders_reject_custom_callbacks_before_reconstruction(
    tmp_path: pathlib.Path,
    format_name: str,
) -> None:
    called: list[object] = []

    class UnsafeNode:
        pass

    tag = f"tests.UnsafePersistedNode.{format_name}"

    def reconstruct(metadata: object, leaves: object) -> UnsafeNode:
        called.append((metadata, leaves))
        return UnsafeNode()

    asc.tree.register_pytree_node(
        UnsafeNode,
        lambda _value: ((), None),
        reconstruct,
        name=tag,
    )
    forged = asc.tree.TreeSpec(
        "custom",
        (tag, {"attacker": True}),
        (asc.tree.TreeSpec("leaf"),),
    ).to_json()

    if format_name == "hdf5":
        import h5py

        path = data.save_hdf5(tmp_path / "forged.h5", numpy.asarray([1.0]))
        with h5py.File(path, "r+") as file:
            file.attrs["tree_spec"] = forged
    else:
        import scipy.io

        path = data.save_mat(tmp_path / "forged.mat", numpy.asarray([1.0]))
        payload = {
            name: value
            for name, value in scipy.io.loadmat(path).items()
            if not name.startswith("__")
        }
        payload["asc_tree_spec"] = numpy.asarray(forged)
        scipy.io.savemat(path, payload, appendmat=False)

    with pytest.raises(asc.DataFormatError, match="custom PyTree"):
        getattr(data, f"load_{format_name}")(path)
    assert called == []


@pytest.mark.parametrize("format_name", ("npz", "hdf5", "mat"))
def test_loaders_reject_non_finite_json_metadata(
    tmp_path: pathlib.Path,
    format_name: str,
) -> None:
    document = '{"value":NaN}'
    if format_name == "npz":
        path = tmp_path / "non-finite.npz"
        numpy.savez(
            path,
            value=numpy.arange(2),
            __asc_metadata__=numpy.frombuffer(
                document.encode(), dtype=numpy.uint8
            ),
        )
    elif format_name == "hdf5":
        import h5py

        path = data.save_hdf5(tmp_path / "non-finite.h5", numpy.arange(2))
        with h5py.File(path, "r+") as file:
            file.attrs["metadata"] = document
    else:
        import scipy.io

        path = data.save_mat(tmp_path / "non-finite.mat", numpy.arange(2))
        payload = {
            name: value
            for name, value in scipy.io.loadmat(path).items()
            if not name.startswith("__")
        }
        payload["asc_metadata"] = numpy.asarray(document)
        scipy.io.savemat(path, payload, appendmat=False)

    with pytest.raises(asc.DataFormatError, match="strict JSON"):
        getattr(data, f"load_{format_name}")(path)


@pytest.mark.parametrize("format_name", ("hdf5", "mat"))
def test_tree_loaders_revalidate_numeric_dtype_families(
    tmp_path: pathlib.Path,
    format_name: str,
) -> None:
    if format_name == "hdf5":
        import h5py

        path = data.save_hdf5(tmp_path / "text.h5", numpy.arange(2))
        with h5py.File(path, "r+") as file:
            del file["leaves"]["0"]
            file["leaves"].create_dataset(
                "0", data=numpy.asarray([b"unsafe", b"text"])
            )
    else:
        import scipy.io

        path = data.save_mat(tmp_path / "datetime.mat", numpy.arange(2))
        payload = {
            name: value
            for name, value in scipy.io.loadmat(path).items()
            if not name.startswith("__")
        }
        payload["asc_dtypes"] = numpy.asarray(json.dumps(["<M8[D]"]))
        scipy.io.savemat(path, payload, appendmat=False)

    with pytest.raises(asc.DataFormatError, match="numeric"):
        getattr(data, f"load_{format_name}")(path)


def test_npz_save_rejects_names_that_alias_archive_members(
    tmp_path: pathlib.Path,
) -> None:
    for arrays in (
        {"x": numpy.asarray([1]), "x.npy": numpy.asarray([2])},
        {"__asc_metadata__.npy": numpy.asarray([1])},
    ):
        with pytest.raises(asc.DataFormatError, match="alias"):
            data.save_npz(tmp_path / "alias.npz", arrays)


def test_schema_records_freeze_inputs_and_validate_roles() -> None:
    field = data.FieldSpec("x", [1], "float32", "numpy", "cpu", role="input")
    fields = [(("x",), field)]
    structure = asc.tree.tree_structure({"x": numpy.asarray([1.0])})
    spec = data.DataSpec(
        structure,
        typing.cast(tuple[tuple[asc.tree.Path, data.FieldSpec], ...], fields),
    )
    fields.append((("y",), field))

    assert field.shape == (1,)
    assert field.role is data.SemanticRole.INPUT
    assert len(spec.fields) == 1
    with pytest.raises(asc.DataSpecError, match="role"):
        data.FieldSpec("x", (1,), "float32", "numpy", "cpu", role="not-a-role")


def test_compose_validates_its_complete_protocol() -> None:
    class TransformOnly:
        def transform(self, value: object) -> object:
            return value

    with pytest.raises(asc.DataSpecError, match="Transform protocol"):
        data.Compose((typing.cast(data.Transform, TransformOnly()),))


def test_targeted_transforms_preserve_mapping_and_named_tuple_types() -> None:
    class Samples(data.Dataset[object]):
        def __init__(self, sample: object) -> None:
            self.sample = sample

        def __len__(self) -> int:
            return 1

        def __getitem__(self, index: object) -> object:
            del index
            return self.sample

    class Pair(typing.NamedTuple):
        input: int
        target: int

    mapping = collections.defaultdict(list, {"input": 1, "target": 2})
    mapped = data.TransformDataset(
        Samples(mapping),
        lambda value: typing.cast(int, value) + 10,
        target="input",
    )[0]
    paired = data.TransformDataset(
        Samples(Pair(1, 2)),
        lambda value: typing.cast(int, value) + 10,
        target="input",
    )[0]

    assert isinstance(mapped, collections.defaultdict)
    assert mapped.default_factory is list
    assert mapped["input"] == 11
    assert isinstance(paired, Pair)
    assert paired == Pair(11, 2)


@pytest.mark.parametrize("backend", ("numpy", "torch", "jax"))
def test_scalers_and_streaming_statistics_are_extreme_value_stable(
    backend: str,
) -> None:
    selected = asc.backend(backend)
    values = selected.xp.asarray([-3e38, 3e38], dtype=selected.xp.float32)

    standard = data.StandardScaler().fit(values)
    standardized = standard.transform(values)
    minmax = data.MinMaxScaler().fit(values)
    normalized = minmax.transform(values)

    class Samples(data.Dataset[object]):
        def __len__(self) -> int:
            return 2

        def __getitem__(self, index: object) -> object:
            return values[index]

    statistics = data.dataset_statistics(Samples())
    assert isinstance(statistics, data.Statistics)
    numpy.testing.assert_allclose(numpy.asarray(standardized), [-1.0, 1.0])
    numpy.testing.assert_allclose(numpy.asarray(normalized), [0.0, 1.0])
    numpy.testing.assert_allclose(
        numpy.asarray(standard.inverse_transform(standardized)),
        [-3e38, 3e38],
        rtol=1e-6,
    )
    numpy.testing.assert_allclose(
        numpy.asarray(minmax.inverse_transform(normalized)),
        [-3e38, 3e38],
        rtol=1e-6,
    )
    assert float(numpy.asarray(statistics.mean)) == 0.0
    assert numpy.isposinf(numpy.asarray(statistics.variance))
    assert float(numpy.asarray(statistics.std)) == pytest.approx(3e38)


def test_standard_scaler_does_not_underflow_sparse_float16_moments() -> None:
    count = 33_554_432
    values = numpy.zeros((count, 1), dtype=numpy.float16)
    values[0, 0] = 1.0

    fitted = data.StandardScaler().fit(values)
    transformed = fitted.transform(values[:2])

    expected_mean = 1.0 / count
    expected_scale = numpy.sqrt(expected_mean * (1.0 - expected_mean))
    assert fitted.mean_.dtype == numpy.float32
    assert fitted.scale_.dtype == numpy.float32
    assert float(fitted.mean_[0]) == pytest.approx(expected_mean, rel=2e-3)
    assert float(fitted.scale_[0]) == pytest.approx(expected_scale, rel=2e-3)
    assert not bool(fitted.constant_[0])
    assert float(transformed[0, 0]) == pytest.approx(
        (1.0 - expected_mean) / expected_scale,
        rel=2e-3,
    )
    assert float(transformed[1, 0]) == pytest.approx(
        -expected_mean / expected_scale,
        rel=2e-3,
    )


def test_streaming_statistics_do_not_store_counts_in_float16() -> None:
    class Samples(data.IterableDataset[object]):
        def __iter__(self) -> typing.Iterator[object]:
            yield from (
                numpy.asarray(0.0, dtype=numpy.float16) for _ in range(65_505)
            )
            yield numpy.asarray(1.0, dtype=numpy.float16)

    statistics = data.dataset_statistics(Samples())
    expected_mean = 1.0 / 65_506
    expected_variance = expected_mean * (1.0 - expected_mean)

    assert statistics.count == 65_506
    assert float(numpy.asarray(statistics.mean)) == pytest.approx(
        expected_mean, rel=2e-3
    )
    assert float(numpy.asarray(statistics.variance)) == pytest.approx(
        expected_variance, rel=2e-3
    )


def test_streaming_statistics_accumulate_float16_moments_in_float32() -> None:
    class Samples(data.IterableDataset[object]):
        def __iter__(self) -> typing.Iterator[object]:
            for index in range(4_096):
                yield numpy.asarray(index % 2, dtype=numpy.float16)

    statistics = data.dataset_statistics(Samples())

    assert statistics.mean.dtype == numpy.float32
    assert statistics.variance.dtype == numpy.float32
    assert float(statistics.mean) == pytest.approx(0.5)
    assert float(statistics.variance) == pytest.approx(0.25)
    assert float(statistics.std) == pytest.approx(0.5)


@pytest.mark.parametrize("backend", ("numpy", "torch", "jax"))
def test_streaming_statistics_adapt_after_an_initial_zero(backend: str) -> None:
    selected = asc.backend(backend)

    class Samples(data.IterableDataset[object]):
        def __iter__(self) -> typing.Iterator[object]:
            for value in (0.0, 1e-30):
                yield selected.xp.asarray(
                    value,
                    dtype=selected.xp.float32,
                    device=selected.device,
                )

    statistics = data.dataset_statistics(Samples())

    assert float(numpy.asarray(statistics.mean)) == pytest.approx(5e-31)
    assert float(numpy.asarray(statistics.std)) == pytest.approx(5e-31)


def test_data_loader_rejects_invalid_random_state_during_construction() -> None:
    dataset = data.ArrayDataset(numpy.arange(4))

    for shuffle in (False, True):
        with pytest.raises(asc.DataLoaderError, match="state"):
            data.DataLoader(dataset, shuffle=shuffle, state=object())
