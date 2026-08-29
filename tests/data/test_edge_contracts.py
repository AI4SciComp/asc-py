# Copyright 2026 AI4SciComp contributors.
# SPDX-License-Identifier: Apache-2.0
"""Collation, dataset, and loader failure-path contracts."""

from __future__ import annotations

import collections
import dataclasses

import numpy
import pytest

import asc
from asc import data


@dataclasses.dataclass(frozen=True)
class Item:
    """Nested sample fixture."""

    value: object


class OtherItem(Item):
    """Distinct dataclass type fixture."""


@dataclasses.dataclass(frozen=True)
class EmptyItem:
    """Empty dataclass fixture."""


class Stream(data.IterableDataset[object]):
    """Optionally sized iterable fixture."""

    def __init__(self, values: tuple[object, ...]) -> None:
        """Store values."""
        self.values = values

    def __iter__(self):
        """Yield values."""
        return iter(self.values)


class UnsizedBatches:
    """Reiterable batch collection without a stable length."""

    def __iter__(self):
        """Return a fresh iterator for every loader pass."""
        return iter(((0,), (1,)))


def test_recursive_conversion_and_collation_edges() -> None:
    Pair = collections.namedtuple("Pair", "left right")
    converted = data.default_convert(
        Item({"pair": Pair(1, [2]), "metadata": None}), backend="numpy"
    )
    assert asc.backend_of(converted.value["pair"].left) == "numpy"
    unknown = object()
    assert data.default_convert(unknown) is unknown
    values = numpy.asarray([1.0], dtype=numpy.float32)
    invalid_samples = (
        [values, 1],
        ["x", b"x"],
        [Item(1), OtherItem(1)],
        [(1, 2), (1,)],
        [[1], [1, 2]],
        [object(), object()],
    )
    for samples in invalid_samples:
        with pytest.raises(asc.CollationError):
            data.default_collate(samples)
    with pytest.raises(asc.CollationError, match="dimension"):
        data.uncollate(numpy.asarray(1.0))
    for batch in ({}, EmptyItem(), (), object()):
        with pytest.raises(asc.CollationError):
            data.uncollate(batch)
    assert data.uncollate([]) == []
    with pytest.raises(asc.CollationError, match="dimensions"):
        data.uncollate({"x": numpy.ones((2,)), "y": numpy.ones((3,))})
    with pytest.raises(asc.CollationError, match="lengths"):
        data.uncollate({"x": numpy.ones((2,)), "name": ["a"]})
    assert data.uncollate({"empty": [], "x": numpy.ones((2,))}) == [
        {"empty": [], "x": 1.0},
        {"empty": [], "x": 1.0},
    ]


def test_loader_custom_paths_and_unsized_errors() -> None:
    dataset = data.ArrayDataset(numpy.arange(5))
    custom = data.DataLoader(
        dataset,
        batch_sampler=((0, 2), (4,)),
        collate_fn=lambda samples: tuple(int(value) for value in samples),
    )
    assert list(custom) == [(0, 2), (4,)]
    unsized_batches = data.DataLoader(dataset, batch_sampler=UnsizedBatches())
    with pytest.raises(TypeError, match="stable length"):
        len(unsized_batches)
    assert len(list(unsized_batches)) == 2
    assert len(list(unsized_batches)) == 2
    with pytest.raises(asc.DataLoaderError, match="reiterable"):
        data.DataLoader(
            dataset,
            batch_sampler=(batch for batch in ((0,), (1,))),
        )
    assert list(data.DataLoader(dataset, batch_size=None)) == list(range(5))
    stream = Stream((1, 2, 3))
    assert list(data.DataLoader(stream, batch_size=None)) == [1, 2, 3]
    with pytest.raises(TypeError, match="streaming"):
        len(data.DataLoader(stream))
    with pytest.raises(asc.DataLoaderError, match="excludes"):
        data.DataLoader(dataset, batch_size=2, batch_sampler=((0,),))
    with pytest.raises(asc.DataLoaderError, match="streaming"):
        data.DataLoader(stream, sampler=data.SequentialSampler(dataset))
    with pytest.raises(asc.DataLoaderError, match="batching"):
        data.DataLoader(dataset, batch_size=None, drop_last=True)


def test_dataset_index_and_composition_edges() -> None:
    array = data.ArrayDataset(numpy.arange(6).reshape(3, 2))
    for index in (True, 4, -4, [0, True], numpy.asarray([0.0])):
        with pytest.raises(IndexError):
            array[index]
    with pytest.raises(asc.DatasetError, match="axis"):
        data.ArrayDataset(numpy.arange(2), sample_axis=2)
    with pytest.raises(asc.DatasetError, match="field"):
        data.MappingDataset({"": array})
    with pytest.raises(asc.DatasetError, match="counts"):
        data.TupleDataset(array, data.ArrayDataset(numpy.arange(4)))
    with pytest.raises(asc.DatasetError, match="least one"):
        data.TupleDataset()
    with pytest.raises(asc.DatasetError, match="least one"):
        data.MappingDataset({})
    with pytest.raises(asc.DatasetError, match="empty"):
        data.ConcatDataset(())
    with pytest.raises(asc.DatasetError, match="least one"):
        data.ZipDataset()
    with pytest.raises(asc.DatasetError, match="indices"):
        data.Subset(array, (True,))
