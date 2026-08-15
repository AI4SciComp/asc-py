# Copyright 2026 AI4SciComp contributors.
# SPDX-License-Identifier: Apache-2.0
"""Collation, sampling, and deterministic single-process loading."""

from __future__ import annotations

import collections
import dataclasses

import numpy
import pytest

import asc
from asc import data


@dataclasses.dataclass(frozen=True)
class Sample:
    """Nested sample fixture."""

    values: object
    label: str


@dataclasses.dataclass(frozen=True, slots=True)
class DerivedSample:
    """Dataclass with state excluded from its generated constructor."""

    value: object
    doubled: object = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        """Derive the non-init field."""
        object.__setattr__(self, "doubled", self.value * 2)


@dataclasses.dataclass(frozen=True)
class EmptySample:
    """Dataclass fixture with no recoverable batch leaves."""


@dataclasses.dataclass(frozen=True)
class ProcessedSample:
    """Dataclass whose persistent init field is processed exactly once."""

    value: object

    def __post_init__(self) -> None:
        """Apply a visible one-time transformation to the init field."""
        object.__setattr__(self, "value", self.value * 2)


@pytest.mark.parametrize("backend", ("numpy", "torch", "jax"))
def test_recursive_collate_uncollate_and_convert(backend: str) -> None:
    selected = asc.backend(backend)
    xp = selected.xp
    Pair = collections.namedtuple("Pair", "x y")
    samples = [
        {
            "record": Sample(
                xp.asarray([index, index + 1], dtype=xp.float32), str(index)
            ),
            "pair": Pair(index, [index + 1]),
        }
        for index in range(3)
    ]
    batch = data.default_collate(samples, backend=backend)
    assert batch["record"].values.shape == (3, 2)
    recovered = data.uncollate(batch)
    assert len(recovered) == 3
    numpy.testing.assert_allclose(
        numpy.asarray(recovered[2]["record"].values), [2, 3]
    )
    converted = data.default_convert({"x": 2.0, "name": "x"}, backend=backend)
    assert asc.backend_of(converted["x"]) == backend
    assert converted["name"] == "x"


def test_dataclass_data_reconstruction_does_not_repeat_post_init() -> None:
    samples = [ProcessedSample(1), ProcessedSample(2)]

    converted = data.default_convert(samples[0])
    batch = data.default_collate(samples)
    recovered = data.uncollate(batch)

    assert numpy.asarray(converted.value).item() == 2
    numpy.testing.assert_array_equal(numpy.asarray(batch.value), [2, 4])
    assert [numpy.asarray(item.value).item() for item in recovered] == [2, 4]


@pytest.mark.parametrize("backend", ("numpy", "torch", "jax"))
def test_scalar_conversion_preserves_configured_backend_dtype(
    backend: str,
) -> None:
    selected = asc.backend(backend)
    xp = selected.xp
    dtype = (
        xp.float64
        if "float64" in asc.backend_info(backend).dtypes
        else xp.float32
    )
    configured = asc.backend(backend, dtype=dtype)

    converted = data.default_convert(1.0, backend=configured)
    collated = data.default_collate([1.0, 2.0], backend=configured)

    assert converted.dtype == dtype
    assert collated.dtype == dtype


@pytest.mark.parametrize("backend", ("numpy", "torch", "jax"))
def test_numpy_scalar_conversion_uses_the_explicit_backend(
    backend: str,
) -> None:
    selected = asc.backend(backend)
    configured = asc.backend(backend, dtype=selected.xp.float32)

    converted = data.default_convert(numpy.float64(1.25), backend=configured)
    collated = data.default_collate(
        [numpy.float64(1.25), numpy.float64(2.5)], backend=configured
    )

    assert asc.backend_of(converted) == backend
    assert asc.backend_of(collated) == backend
    assert converted.dtype == selected.xp.float32
    assert collated.dtype == selected.xp.float32


@pytest.mark.parametrize("backend", ("numpy", "torch", "jax"))
def test_numpy_scalar_conversion_preserves_inferred_dtype(backend: str) -> None:
    scalar = numpy.float16(1.25)

    converted = data.default_convert(scalar, backend=backend)
    collated = data.default_collate([scalar, scalar], backend=backend)

    assert str(converted.dtype).rsplit(".", maxsplit=1)[-1] == "float16"
    assert str(collated.dtype).rsplit(".", maxsplit=1)[-1] == "float16"


@pytest.mark.backend("torch")
def test_numpy_uint64_scalar_conversion_preserves_full_range() -> None:
    scalar = numpy.uint64(2**64 - 1)

    converted = data.default_convert(scalar, backend="torch")
    collated = data.default_collate([scalar, scalar], backend="torch")

    assert str(converted.dtype) == "torch.uint64"
    assert str(collated.dtype) == "torch.uint64"
    assert numpy.asarray(converted).item() == 2**64 - 1


@pytest.mark.parametrize("backend", ("numpy", "torch", "jax"))
def test_dataclass_non_init_fields_convert_collate_and_uncollate(
    backend: str,
) -> None:
    selected = asc.backend(backend)
    xp = selected.xp
    converted = data.default_convert(DerivedSample(2.0), backend=backend)
    assert asc.backend_of(converted.value) == backend
    assert asc.backend_of(converted.doubled) == backend
    samples = [
        DerivedSample(xp.asarray([value], dtype=xp.float32))
        for value in (1.0, 3.0)
    ]

    batch = data.default_collate(samples, backend=backend)
    recovered = data.uncollate(batch)

    numpy.testing.assert_array_equal(
        numpy.asarray(batch.doubled), [[2.0], [6.0]]
    )
    numpy.testing.assert_array_equal(numpy.asarray(recovered[1].value), [3.0])
    numpy.testing.assert_array_equal(numpy.asarray(recovered[1].doubled), [6.0])


def test_mapping_types_survive_convert_collate_and_uncollate() -> None:
    samples = (
        collections.Counter({"x": 1}),
        collections.Counter({"x": 2}),
    )

    converted = data.default_convert(samples[0])
    batch = data.default_collate(samples)
    recovered = data.uncollate(batch)

    assert isinstance(converted, collections.Counter)
    assert isinstance(batch, collections.Counter)
    assert all(isinstance(sample, collections.Counter) for sample in recovered)
    assert tuple(int(sample["x"]) for sample in recovered) == (1, 2)


def test_collation_requires_one_concrete_mapping_type() -> None:
    ordered = collections.OrderedDict((("x", 1),))

    with pytest.raises(asc.CollationError, match="mapping keys"):
        data.default_collate((ordered, {"x": 2}))


@pytest.mark.parametrize("sample", ([], (), {}, EmptySample(), {"x": []}))
def test_collation_rejects_leafless_samples(sample: object) -> None:
    with pytest.raises(asc.CollationError, match="leafless"):
        data.default_collate((sample, sample))


def test_collation_preserves_nested_leafless_structures_with_sibling_data() -> (
    None
):
    samples = (
        {"x": numpy.asarray([1]), "mapping": {}, "sequence": []},
        {"x": numpy.asarray([2]), "mapping": {}, "sequence": []},
    )

    batch = data.default_collate(samples)
    recovered = data.uncollate(batch)

    assert batch["mapping"] == {}
    assert batch["sequence"] == []
    assert [sample["mapping"] for sample in recovered] == [{}, {}]
    assert [sample["sequence"] for sample in recovered] == [[], []]


def test_collation_rejects_structure_dtype_and_backend_mismatch() -> None:
    with pytest.raises(asc.CollationError, match="empty"):
        data.default_collate([])
    with pytest.raises(asc.CollationError, match="shapes"):
        data.default_collate([numpy.ones((1,)), numpy.ones((2,))])
    with pytest.raises(asc.CollationError, match="dtypes"):
        data.default_collate(
            [
                numpy.ones((1,), dtype=numpy.float32),
                numpy.ones((1,), dtype=numpy.float64),
            ]
        )
    with pytest.raises(asc.CollationError, match="keys"):
        data.default_collate([{"a": 1}, {"b": 1}])
    with pytest.raises(asc.CollationError, match="type"):
        data.default_collate([1, 1.0])
    with pytest.raises(asc.CollationError):
        data.uncollate(numpy.asarray(1))


@pytest.mark.parametrize("samples", ([1, True], [True, 1]))
def test_collation_requires_exact_scalar_types_in_every_order(
    samples: list[object],
) -> None:
    with pytest.raises(asc.CollationError, match="stable Python type"):
        data.default_collate(samples)


@pytest.mark.backend("torch")
def test_collation_rejects_mixed_backend_requests() -> None:
    with pytest.raises(asc.MixedBackendError):
        data.default_collate([numpy.asarray([1])], backend="torch")
    with pytest.raises(asc.MixedBackendError):
        data.default_convert(numpy.asarray([1]), backend="torch")


def test_samplers_exact_counts_and_replay() -> None:
    source = list(range(8))
    state = asc.random_state(3, backend="numpy")
    assert list(data.SequentialSampler(source)) == source
    random_sampler = data.RandomSampler(source, state=state)
    assert sorted(random_sampler) == source
    assert list(random_sampler) == list(random_sampler)
    replacement = data.RandomSampler(
        source, state=state, replacement=True, num_samples=12
    )
    assert len(list(replacement)) == 12
    subset = data.SubsetRandomSampler([2, 4, 6], state=state)
    assert sorted(subset) == [2, 4, 6]
    weighted = data.WeightedRandomSampler([0, 1, 2], 8, state=state)
    assert set(weighted).issubset({1, 2})
    batches = data.BatchSampler(data.SequentialSampler(source), 3)
    assert list(batches) == [[0, 1, 2], [3, 4, 5], [6, 7]]
    dropped = data.BatchSampler(
        data.SequentialSampler(source), 3, drop_last=True
    )
    assert list(dropped) == [[0, 1, 2], [3, 4, 5]]


def test_empty_replacement_sampler_with_zero_samples_is_empty() -> None:
    sampler = data.RandomSampler(
        [],
        state=asc.random_state(3, backend="numpy"),
        replacement=True,
        num_samples=0,
    )

    assert len(sampler) == 0
    assert list(sampler) == []


def test_empty_replacement_sampler_rejects_positive_sample_count() -> None:
    with pytest.raises(asc.DataLoaderError, match="empty"):
        data.RandomSampler(
            [],
            state=asc.random_state(3, backend="numpy"),
            replacement=True,
            num_samples=1,
        )


@pytest.mark.parametrize("backend", ("numpy", "torch", "jax"))
def test_weighted_sampler_normalizes_large_finite_weights(backend: str) -> None:
    sampler = data.WeightedRandomSampler(
        [1e308, 1e308],
        16,
        state=asc.random_state(3, backend=backend),
    )

    samples = list(sampler)

    assert len(samples) == 16
    assert set(samples) <= {0, 1}


@pytest.mark.parametrize("backend", ("numpy", "torch", "jax"))
def test_weighted_sampler_rejects_probabilities_that_round_to_zero(
    backend: str,
) -> None:
    with pytest.raises(asc.DataLoaderError, match="representable"):
        data.WeightedRandomSampler(
            [1e-300, 1.0],
            2,
            state=asc.random_state(3, backend=backend),
            replacement=False,
        )


def test_sampler_validation() -> None:
    state = asc.random_state(1, backend="numpy")
    invalid = (
        lambda: data.RandomSampler([1], state=state, replacement=1),
        lambda: data.RandomSampler([1], state=state, num_samples=-1),
        lambda: data.RandomSampler([1], state=state, num_samples=2),
        lambda: data.SubsetRandomSampler([True], state=state),
        lambda: data.WeightedRandomSampler([], 1, state=state),
        lambda: data.WeightedRandomSampler([0, 0], 1, state=state),
        lambda: data.WeightedRandomSampler([1], -1, state=state),
        lambda: data.WeightedRandomSampler(
            [1], 2, state=state, replacement=False
        ),
        lambda: data.BatchSampler(data.SequentialSampler([1]), 0),
        lambda: data.BatchSampler(data.SequentialSampler([1]), 1, drop_last=1),
    )
    for factory in invalid:
        with pytest.raises(asc.DataLoaderError):
            factory()


@pytest.mark.parametrize("backend", ("numpy", "torch", "jax"))
def test_map_loader_backend_preservation_and_repeated_iteration(
    backend: str,
) -> None:
    selected = asc.backend(backend)
    xp = selected.xp
    dataset = data.ArrayDataset(
        xp.reshape(xp.arange(10, dtype=xp.float32), (5, 2))
    )
    loader = data.DataLoader(dataset, batch_size=2)
    assert len(loader) == 3
    first = list(loader)
    second = list(loader)
    assert [batch.shape for batch in first] == [(2, 2), (2, 2), (1, 2)]
    assert all(asc.backend_of(batch) == backend for batch in first)
    for left, right in zip(first, second, strict=True):
        numpy.testing.assert_allclose(numpy.asarray(left), numpy.asarray(right))
    shuffled = data.DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        state=asc.random_state(4, backend=backend),
    )
    assert len(list(shuffled)) == 3
    unbatched = data.DataLoader(dataset, batch_size=None)
    assert len(list(unbatched)) == 5


@pytest.mark.backend("torch")
def test_torch_collation_preserves_the_autodiff_graph() -> None:
    import torch

    first = torch.tensor([1.0, 2.0], requires_grad=True)
    second = torch.tensor([3.0, 4.0], requires_grad=True)
    batch = data.default_collate([first, second])
    batch.sum().backward()
    torch.testing.assert_close(first.grad, torch.ones_like(first))
    torch.testing.assert_close(second.grad, torch.ones_like(second))


class SizedStream(data.IterableDataset[int]):
    """Sized iterable dataset fixture."""

    def __iter__(self):
        """Yield the fixture values."""
        return iter(range(5))

    def __len__(self):
        """Return the fixture size."""
        return 5


def test_iterable_loader_empty_drop_and_validation() -> None:
    loader = data.DataLoader(SizedStream(), batch_size=2)
    assert len(loader) == 3
    assert [numpy.asarray(batch).tolist() for batch in loader] == [
        [0, 1],
        [2, 3],
        [4],
    ]
    dropped = data.DataLoader(SizedStream(), batch_size=2, drop_last=True)
    assert len(list(dropped)) == 2
    empty = data.DataLoader(
        data.ArrayDataset(numpy.empty((0, 2))), batch_size=2
    )
    assert list(empty) == []
    state = asc.random_state(1, backend="numpy")
    invalid = (
        lambda: data.DataLoader(SizedStream(), shuffle=True, state=state),
        lambda: data.DataLoader(
            data.ArrayDataset(numpy.arange(3)), shuffle=True
        ),
        lambda: data.DataLoader(
            data.ArrayDataset(numpy.arange(3)), batch_size=0
        ),
        lambda: data.DataLoader(
            data.ArrayDataset(numpy.arange(3)),
            shuffle=True,
            sampler=data.SequentialSampler([1]),
            state=state,
        ),
        lambda: data.DataLoader(data.ArrayDataset(numpy.arange(3)), shuffle=1),
        lambda: data.DataLoader(
            data.ArrayDataset(numpy.arange(3)),
            batch_size=None,
            drop_last=True,
        ),
    )
    for factory in invalid:
        with pytest.raises(asc.DataLoaderError):
            factory()
