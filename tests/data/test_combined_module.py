# Copyright 2026 AI4SciComp contributors.
# SPDX-License-Identifier: Apache-2.0
"""CombinedLoader and DataModule lifecycle contracts."""

from __future__ import annotations

import collections.abc

import numpy
import pytest

import asc
from asc import data
from asc.data.module import Stage


class SizedLoader(collections.abc.Iterable[int]):
    """Small reiterable loader fixture."""

    def __init__(self, values: collections.abc.Iterable[int]) -> None:
        """Store a repeatable immutable sequence."""
        self.values = tuple(values)

    def __iter__(self):
        """Yield fixture batches."""
        return iter(self.values)

    def __len__(self):
        """Return fixture batch count."""
        return len(self.values)


@pytest.mark.parametrize(
    ("mode", "length", "expected"),
    (
        ("min_size", 2, [(1, 10), (2, 20)]),
        ("max_size", 3, [(1, 10), (2, 20), (None, 30)]),
        ("max_size_cycle", 3, [(1, 10), (2, 20), (1, 30)]),
    ),
)
def test_parallel_combined_modes(
    mode: str, length: int, expected: list[tuple[int | None, int]]
) -> None:
    combined = data.CombinedLoader(
        {"short": SizedLoader((1, 2)), "long": SizedLoader((10, 20, 30))},
        mode=mode,
    )
    assert len(combined) == length
    batches = list(combined)
    assert [(batch.data["short"], batch.data["long"]) for batch in batches] == (
        expected
    )
    assert [batch.batch_index for batch in batches] == list(range(length))
    assert all(batch.loader_index is None for batch in batches)
    assert list(combined) == batches


def test_sequential_limits_and_combined_errors() -> None:
    loaders = {"a": SizedLoader((1, 2, 3)), "b": SizedLoader((4, 5))}
    sequential = data.CombinedLoader(
        loaders, mode="sequential", limits={"a": 2, "b": 1}
    )
    assert len(sequential) == 3
    assert [(batch.data, batch.loader_index) for batch in sequential] == [
        (1, 0),
        (2, 0),
        (4, 1),
    ]
    for factory in (
        lambda: data.CombinedLoader({}, mode="min_size"),
        lambda: data.CombinedLoader({"a": 1}),
        lambda: data.CombinedLoader(loaders, mode="bad"),
        lambda: data.CombinedLoader(loaders, limits={"a": 1}),
        lambda: data.CombinedLoader(loaders, limits=-1),
    ):
        with pytest.raises(asc.DataLoaderError):
            factory()
    cycled = data.CombinedLoader(
        {"empty": SizedLoader(()), "full": SizedLoader((1,))},
        mode="max_size_cycle",
    )
    with pytest.raises(asc.DataLoaderError, match="empty"):
        list(cycled)
    with pytest.raises(asc.DataLoaderError, match="one-shot"):
        data.CombinedLoader({"stream": (value for value in range(2))})


def test_sequential_limits_do_not_consume_an_extra_batch() -> None:
    consumed: list[int] = []

    class TrackingLoader(collections.abc.Iterable[int]):
        def __iter__(self):
            for value in range(4):
                consumed.append(value)
                yield value

        def __len__(self) -> int:
            return 4

    empty = data.CombinedLoader(
        {"loader": TrackingLoader()}, mode="sequential", limits=0
    )
    assert list(empty) == []
    assert consumed == []

    limited = data.CombinedLoader(
        {"loader": TrackingLoader()}, mode="sequential", limits=2
    )
    assert [batch.data for batch in limited] == [0, 1]
    assert consumed == [0, 1]


class TrackingModule(data.DataModule):
    """Record protected lifecycle hook execution."""

    def __init__(self) -> None:
        """Initialize an empty registry and event log."""
        super().__init__()
        self.events: list[str] = []

    def _prepare_data(self) -> None:
        self.events.append("prepare")

    def _setup(self, stage: Stage) -> None:
        self.events.append(f"setup:{stage}")

    def _teardown(self, stage: Stage) -> None:
        self.events.append(f"teardown:{stage}")


def test_data_module_registry_lifecycle_loaders_and_splits() -> None:
    module = TrackingModule()
    dataset = data.ArrayDataset(
        numpy.arange(12, dtype=numpy.float32).reshape(6, 2)
    )
    module.add_dataset("train", "primary", dataset)
    assert module.get_dataset("train", "primary") is dataset
    assert tuple(module.datasets("train")) == ("primary",)
    with pytest.raises(TypeError):
        module.datasets("train")["x"] = dataset
    module.prepare_data()
    module.prepare_data()
    module.setup("train")
    module.setup("train")
    module.teardown("train")
    module.teardown("train")
    assert module.events == ["prepare", "setup:train", "teardown:train"]
    assert len(module.loader("train", "primary", batch_size=4)) == 2
    assert len(module.combined_loader("train", limits=1)) == 1
    spec = module.specs("train")["primary"]
    assert spec is not None and spec.fields[0][1].shape == (2,)
    assert module.diagnostics()["datasets"]["train"] == {"primary": 6}
    splits = module.register_splits(
        dataset,
        name="partition",
        train=0.5,
        validation=0.25,
        test=0.25,
        state=asc.random_state(3, backend="numpy"),
    )
    assert sum(map(len, splits)) == 6
    assert module.remove_dataset("train", "partition") is splits.train


def test_data_module_validation_errors() -> None:
    module = data.DataModule()
    dataset = data.ArrayDataset(numpy.arange(3))
    for stage in ("bad", "", "TRAIN"):
        with pytest.raises(asc.DatasetError, match="stage"):
            module.datasets(stage)  # type: ignore[arg-type]
    with pytest.raises(asc.DatasetError, match="Dataset"):
        module.add_dataset("train", "bad", object())  # type: ignore[arg-type]
    with pytest.raises(asc.DatasetError, match="name"):
        module.add_dataset("train", "", dataset)
    module.add_dataset("train", "x", dataset)
    with pytest.raises(asc.DatasetError, match="duplicate"):
        module.add_dataset("train", "x", dataset)
    with pytest.raises(asc.DatasetError, match="unknown"):
        module.get_dataset("test", "x")
    with pytest.raises(asc.DatasetError, match="unknown"):
        module.remove_dataset("test", "x")
    with pytest.raises(asc.DataLoaderError, match="unknown"):
        module.combined_loader("train", configs={"bad": {}})
