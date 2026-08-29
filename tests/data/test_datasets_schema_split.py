# Copyright 2026 AI4SciComp contributors.
# SPDX-License-Identifier: Apache-2.0
"""Dataset composition, metadata, validation, and splitting contracts."""

from __future__ import annotations

import numpy
import pytest

import asc
from asc import data


@pytest.mark.parametrize("backend", ("numpy", "torch", "jax"))
def test_array_tuple_mapping_datasets(backend: str) -> None:
    selected = asc.backend(backend)
    xp = selected.xp
    array = xp.reshape(xp.arange(12, dtype=xp.float32), (3, 4))
    dataset = data.ArrayDataset(array, field={"role": "input"})
    assert len(dataset) == 3
    numpy.testing.assert_allclose(numpy.asarray(dataset[-1]), [8, 9, 10, 11])
    assert dataset[:2].shape == (2, 4)
    index_array = xp.asarray([2, 0], dtype=xp.int16)
    numpy.testing.assert_allclose(
        numpy.asarray(dataset[index_array]), [[8, 9, 10, 11], [0, 1, 2, 3]]
    )
    columns = data.ArrayDataset(array, sample_axis=1)
    assert len(columns) == 4
    assert columns[0].shape == (3,)
    tuple_dataset = data.TupleDataset(dataset, dataset)
    mapping_dataset = data.MappingDataset({"x": dataset, "y": dataset})
    assert len(tuple_dataset[0]) == 2
    assert tuple(mapping_dataset[0]) == ("x", "y")
    assert len(tuple_dataset[[2, 0]]) == 2
    assert len(mapping_dataset[:2]) == 2


@pytest.mark.parametrize("sample_axis", (True, 1.0, "0"))
def test_array_dataset_requires_a_non_boolean_integer_axis(
    sample_axis: object,
) -> None:
    with pytest.raises(asc.DatasetError, match="non-Boolean integer"):
        data.ArrayDataset(
            numpy.ones((2, 3), dtype=numpy.float32),
            sample_axis=sample_axis,  # type: ignore[arg-type]
        )


@pytest.mark.backend("torch")
def test_array_dataset_slices_are_views_and_foreign_indices_are_rejected() -> (
    None
):
    source = numpy.arange(12, dtype=numpy.float32).reshape(3, 4)
    dataset = data.ArrayDataset(source)

    sliced = dataset[:]

    assert numpy.shares_memory(source, sliced)
    torch = asc.backend("torch")
    foreign = torch.xp.asarray([2, 0], dtype=torch.xp.int16)
    with pytest.raises(asc.MixedBackendError):
        dataset[foreign]


@pytest.mark.parametrize("backend", ("numpy", "torch", "jax"))
def test_array_dataset_normalizes_native_negative_indices(backend: str) -> None:
    selected = asc.backend(backend)
    xp = selected.xp
    dataset = data.ArrayDataset(xp.arange(4, dtype=xp.float32))
    indices = xp.asarray([-1, 0], dtype=xp.int8)

    numpy.testing.assert_array_equal(numpy.asarray(dataset[indices]), [3, 0])
    with pytest.raises(IndexError, match="outside"):
        dataset[xp.asarray([4], dtype=xp.int8)]


@pytest.mark.backend("jax")
def test_array_dataset_native_index_bounds_are_jax_trace_safe() -> None:
    selected = asc.backend("jax")
    xp = selected.xp
    dataset = data.ArrayDataset(xp.arange(5, dtype=xp.float32))
    compiled = asc.jit(lambda indices: dataset[indices], backend="jax")
    vectorized = asc.vmap(
        lambda index: dataset[xp.reshape(index, (1,))], backend="jax"
    )

    numpy.testing.assert_array_equal(
        numpy.asarray(compiled(xp.asarray([1, -1], dtype=xp.int32))), [1, 4]
    )
    numpy.testing.assert_array_equal(
        numpy.asarray(vectorized(xp.asarray([1, 2], dtype=xp.int32))),
        [[1], [2]],
    )
    with pytest.raises(IndexError, match=r"jit:.*ArrayDataset.*out of bounds"):
        compiled(xp.asarray([5], dtype=xp.int32))
    with pytest.raises(IndexError, match=r"vmap:.*ArrayDataset.*out of bounds"):
        vectorized(xp.asarray([1, 5], dtype=xp.int32))


@pytest.mark.parametrize("backend", ("numpy", "torch", "jax"))
def test_array_dataset_normalizes_negative_step_slices(backend: str) -> None:
    selected = asc.backend(backend)
    dataset = data.ArrayDataset(
        selected.xp.arange(5, dtype=selected.xp.float32)
    )

    numpy.testing.assert_array_equal(numpy.asarray(dataset[::-2]), [4, 2, 0])


def test_concat_subset_transform_filter_and_zip() -> None:
    first = data.ArrayDataset(numpy.arange(6).reshape(3, 2))
    second = data.ArrayDataset(numpy.arange(4).reshape(2, 2) + 10)
    concatenated = data.ConcatDataset((first, second))
    assert len(concatenated) == 5
    numpy.testing.assert_array_equal(concatenated[-1], [12, 13])
    assert len(concatenated[1:4]) == 3
    subset = data.Subset(concatenated, [4, 0, 2])
    numpy.testing.assert_array_equal(subset[0], [12, 13])
    assert len(subset[::-1]) == 3
    transformed = data.TransformDataset(subset, lambda value: value + 1)
    numpy.testing.assert_array_equal(transformed[1], [1, 2])
    mapping = data.MappingDataset({"x": first, "y": first})
    field_transform = data.TransformDataset(
        mapping, lambda value: value * 2, target="x"
    )
    numpy.testing.assert_array_equal(field_transform[1]["x"], [4, 6])
    tuple_transform = data.TransformDataset(
        data.TupleDataset(first, first),
        lambda value: value - 1,
        target="target",
    )
    numpy.testing.assert_array_equal(tuple_transform[1][1], [1, 2])
    filtered = data.FilteredDataset(first, lambda sample: int(sample[0]) >= 2)
    assert len(filtered) == 2
    assert len(list(filtered)) == 2
    strict_zip = data.ZipDataset(first, first)
    minimum_zip = data.ZipDataset(first, second, policy="min_size")
    assert len(strict_zip) == 3
    assert len(minimum_zip) == 2
    assert len(strict_zip[:2]) == 2


class Stream(data.IterableDataset[int]):
    """Small repeatable iterable fixture."""

    def __iter__(self):
        """Yield the fixture values."""
        return iter(range(6))


def test_iterable_filter_and_dataset_errors() -> None:
    filtered = data.FilteredDataset(Stream(), lambda value: value % 2 == 0)
    assert list(filtered) == [0, 2, 4]
    with pytest.raises(TypeError, match="finite length"):
        len(filtered)
    with pytest.raises(TypeError, match="not indexable"):
        filtered[0]
    first = data.ArrayDataset(numpy.arange(3))
    second = data.ArrayDataset(numpy.arange(2))
    with pytest.raises(asc.DatasetError, match="counts"):
        data.TupleDataset(first, second)
    with pytest.raises(asc.DatasetError, match="strict"):
        data.ZipDataset(first, second)
    with pytest.raises(asc.DatasetError, match="integers"):
        data.Subset(first, [True])
    with pytest.raises(IndexError):
        first[5]
    with pytest.raises(asc.DatasetError, match="field"):
        data.TransformDataset(
            data.MappingDataset({"x": first}), lambda x: x, target="missing"
        )[0]


def test_filtered_map_access_reuses_precomputed_indices() -> None:
    class CountingDataset(data.Dataset[int]):
        def __init__(self) -> None:
            self.length_calls = 0

        def __len__(self) -> int:
            self.length_calls += 1
            return 8

        def __getitem__(self, index: object) -> int:
            if not isinstance(index, int):
                raise IndexError(index)
            return index

    source = CountingDataset()
    filtered = data.FilteredDataset(source, lambda value: value % 2 == 0)
    construction_calls = source.length_calls

    assert filtered[0] == 0
    assert filtered[-1] == 6
    assert filtered[1:3] == [2, 4]
    assert list(data.DataLoader(filtered, batch_size=None)) == [0, 2, 4, 6]
    assert source.length_calls == construction_calls


@pytest.mark.parametrize("backend", ("numpy", "torch", "jax"))
def test_schema_inference_and_validation(backend: str) -> None:
    selected = asc.backend(backend)
    xp = selected.xp
    sample = {
        "x": xp.asarray([1.0, 2.0], dtype=xp.float32),
        "y": xp.asarray(3, dtype=xp.int32),
    }
    spec = data.infer_data_spec(sample)
    assert len(spec.fields) == 2
    assert spec.fields[0][1].shape == (2,)
    data.validate_sample(sample, spec)
    dataset = data.MappingDataset(
        {
            "x": xp.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=xp.float32),
            "y": xp.asarray([3, 4], dtype=xp.int32),
        }
    )
    assert data.validate_dataset(dataset) == spec
    with pytest.raises(asc.DataSpecError, match="structure"):
        data.validate_sample((sample["x"], sample["y"]), spec)
    bad = {**sample, "x": xp.asarray([1.0], dtype=xp.float32)}
    with pytest.raises(asc.DataSpecError, match="shape"):
        data.validate_sample(bad, spec)
    with pytest.raises(asc.DataSpecError, match="max_samples"):
        data.validate_dataset(dataset, max_samples=-1)


def test_schema_inference_counts_toward_the_sample_limit() -> None:
    class ChangingDataset(data.Dataset[object]):
        def __init__(self) -> None:
            self.accesses: list[int] = []

        def __len__(self) -> int:
            return 2

        def __getitem__(self, index: object) -> object:
            if not isinstance(index, int):
                raise IndexError(index)
            self.accesses.append(index)
            size = len(self.accesses) if index == 0 else 1
            return numpy.ones((size,), dtype=numpy.float32)

    dataset = ChangingDataset()

    spec = data.validate_dataset(dataset, max_samples=1)

    assert spec.fields[0][1].shape == (1,)
    assert dataset.accesses == [0]

    uninspected = ChangingDataset()
    with pytest.raises(asc.DataSpecError, match="max_samples"):
        data.validate_dataset(uninspected, max_samples=0)
    assert uninspected.accesses == []

    explicit = data.infer_data_spec(numpy.ones((1,), dtype=numpy.float32))
    assert (
        data.validate_dataset(uninspected, explicit, max_samples=0) is explicit
    )
    assert uninspected.accesses == []


def test_validation_accepts_a_compatible_safely_restored_tree_spec() -> None:
    sample = {"x": numpy.asarray([1.0], dtype=numpy.float32)}
    inferred = data.infer_data_spec(sample)
    restored = asc.tree.TreeSpec.from_json(inferred.structure.to_json())
    spec = data.DataSpec(restored, inferred.fields)

    data.validate_sample(sample, spec)


def test_field_spec_validation_and_empty_dataset() -> None:
    field = data.FieldSpec(
        "x",
        (2, 3),
        "float32",
        "numpy",
        "cpu",
        dimensions=("row", "column"),
        role=data.SemanticRole.INPUT,
    )
    assert field.role is data.SemanticRole.INPUT
    for factory in (
        lambda: data.FieldSpec("", (), "float32", "numpy", "cpu"),
        lambda: data.FieldSpec(1, (), "float32", "numpy", "cpu"),
        lambda: data.FieldSpec("x", (-1,), "float32", "numpy", "cpu"),
        lambda: data.FieldSpec("x", (2,), "", "numpy", "cpu"),
        lambda: data.FieldSpec("x", (2,), 3, "numpy", "cpu"),
        lambda: data.FieldSpec("x", (2,), "float32", 3, "cpu"),
        lambda: data.FieldSpec("x", (2,), "float32", "numpy", 3),
        lambda: data.FieldSpec(
            "x", (2,), "float32", "numpy", "cpu", dimensions=(1,)
        ),
        lambda: data.FieldSpec(
            "x", (2,), "float32", "numpy", "cpu", dimensions=("",)
        ),
        lambda: data.FieldSpec(
            "x", (2,), "float32", "numpy", "cpu", dimensions=("a", "a")
        ),
    ):
        with pytest.raises(asc.DataSpecError):
            factory()
    empty = data.ArrayDataset(numpy.empty((0, 2)))
    with pytest.raises(asc.DataSpecError, match="empty"):
        data.validate_dataset(empty)


def test_data_spec_rejects_duplicate_paths() -> None:
    field = data.FieldSpec("x", (2,), "float32", "numpy", "cpu")
    structure = data.infer_data_spec(numpy.zeros((2,))).structure

    with pytest.raises(asc.DataSpecError, match="unique"):
        data.DataSpec(structure, (((), field), ((), field)))


def test_data_spec_rejects_invalid_or_nonleaf_paths() -> None:
    sample = {"x": numpy.zeros((2,), dtype=numpy.float32)}
    inferred = data.infer_data_spec(sample)
    field = inferred.fields[0][1]

    for path in (("y",), (), (True,), (object(),), ([],)):
        with pytest.raises(asc.DataSpecError):
            data.DataSpec(inferred.structure, ((path, field),))
    with pytest.raises(asc.DataSpecError, match="TreeSpec"):
        data.DataSpec(object(), inferred.fields)


def test_deterministic_splits_and_kfolds() -> None:
    dataset = data.ArrayDataset(numpy.arange(20))
    splits = data.split_dataset(dataset, (0.5, 0.3, 0.2))
    assert tuple(map(len, splits)) == (10, 6, 4)
    integer_splits = data.split_dataset(dataset, (5, 15))
    assert tuple(map(len, integer_splits)) == (5, 15)
    state = asc.random_state(10, backend="numpy")
    first = data.train_validation_test_split(
        dataset, train=0.6, validation=0.2, test=0.2, state=state
    )
    second = data.train_validation_test_split(
        dataset, train=0.6, validation=0.2, test=0.2, state=state
    )
    assert first.train.indices == second.train.indices
    folds = data.kfold_indices(10, 3, state=state)
    assert len(folds) == 3
    assert sorted(
        index for _, validation in folds for index in validation
    ) == list(range(10))
    for call in (
        lambda: data.split_dataset(dataset, (0.3, 0.3)),
        lambda: data.split_dataset(dataset, (2, 2)),
        lambda: data.split_dataset(dataset, (1, 0.0)),
        lambda: data.split_dataset(dataset, (0, 1.0)),
        lambda: data.kfold_indices(1, 2),
        lambda: data.kfold_indices(4, 1),
    ):
        with pytest.raises(asc.DataError):
            call()
