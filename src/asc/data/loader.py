# Copyright 2026 AI4SciComp contributors.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic single-process data loading."""

from __future__ import annotations

import collections.abc
import dataclasses
import typing

from asc import errors
from asc.data.collate import default_collate
from asc.data.dataset import Dataset, IterableDataset, is_map_style_dataset
from asc.data.sampler import (
    BatchSampler,
    RandomSampler,
    Sampler,
    SequentialSampler,
)
from asc.random import RandomState


class DataLoader(collections.abc.Iterable[object]):
    """Batch map or iterable datasets in the current process."""

    __slots__ = (
        "backend",
        "batch_sampler",
        "batch_size",
        "collate_fn",
        "dataset",
        "drop_last",
        "sampler",
        "shuffle",
        "state",
    )

    dataset: Dataset[object] | IterableDataset[object]
    batch_size: int | None
    shuffle: bool
    sampler: Sampler | None
    batch_sampler: (
        collections.abc.Iterable[collections.abc.Sequence[int]] | None
    )
    drop_last: bool
    collate_fn: (
        typing.Callable[[collections.abc.Sequence[object]], object] | None
    )
    state: RandomState | None
    backend: object | None

    def __init__(  # pylint: disable=too-many-arguments
        self,
        dataset: Dataset[object] | IterableDataset[object],
        *,
        batch_size: int | None = 1,
        shuffle: bool = False,
        sampler: Sampler | None = None,
        batch_sampler: (
            collections.abc.Iterable[collections.abc.Sequence[int]] | None
        ) = None,
        drop_last: bool = False,
        collate_fn: (
            typing.Callable[[collections.abc.Sequence[object]], object] | None
        ) = None,
        state: RandomState | None = None,
        backend: object | None = None,
    ) -> None:
        if not isinstance(shuffle, bool) or not isinstance(drop_last, bool):
            raise errors.DataLoaderError(
                "DataLoader: shuffle and drop_last must be Boolean"
            )
        if batch_size is not None and (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or batch_size <= 0
        ):
            raise errors.DataLoaderError(
                "DataLoader: batch_size must be a positive integer or None"
            )
        if batch_size is None and drop_last:
            raise errors.DataLoaderError(
                "DataLoader: drop_last requires batching to be enabled"
            )
        if shuffle and sampler is not None:
            raise errors.DataLoaderError(
                "DataLoader: shuffle and sampler are mutually exclusive"
            )
        if batch_sampler is not None and (
            batch_size != 1 or shuffle or sampler is not None or drop_last
        ):
            raise errors.DataLoaderError(
                "DataLoader: batch_sampler excludes batch_size, shuffle, "
                "sampler, and drop_last"
            )
        if isinstance(batch_sampler, collections.abc.Iterator):
            raise errors.DataLoaderError(
                "DataLoader: batch_sampler must be reiterable, not a "
                "one-shot iterator"
            )
        if not is_map_style_dataset(dataset) and (
            shuffle or sampler is not None or batch_sampler is not None
        ):
            raise errors.DataLoaderError(
                "DataLoader: streaming datasets do not support shuffle or "
                "samplers"
            )
        if shuffle and state is None:
            raise errors.DataLoaderError(
                "DataLoader: shuffle requires explicit random state"
            )
        if state is not None and not isinstance(state, RandomState):
            raise errors.DataLoaderError(
                "DataLoader: state must be an asc.RandomState or None"
            )
        object.__setattr__(self, "dataset", dataset)
        object.__setattr__(self, "batch_size", batch_size)
        object.__setattr__(self, "shuffle", shuffle)
        object.__setattr__(self, "sampler", sampler)
        object.__setattr__(self, "batch_sampler", batch_sampler)
        object.__setattr__(self, "drop_last", drop_last)
        object.__setattr__(self, "state", state)
        object.__setattr__(self, "backend", backend)
        object.__setattr__(self, "collate_fn", collate_fn)

    def __setattr__(self, name: str, _value: object) -> None:
        """Prevent configuration changes after construction."""
        raise dataclasses.FrozenInstanceError(
            f"cannot assign to field {name!r}"
        )

    def __delattr__(self, name: str) -> None:
        """Prevent configuration deletion after construction."""
        raise dataclasses.FrozenInstanceError(f"cannot delete field {name!r}")

    def _collate(self, samples: collections.abc.Sequence[object]) -> object:
        if self.collate_fn is not None:
            return self.collate_fn(samples)
        return default_collate(samples, backend=self.backend)

    def _map_batches(self) -> typing.Iterator[object]:
        dataset = typing.cast(Dataset[object], self.dataset)
        if self.batch_sampler is not None:
            for indices in self.batch_sampler:
                yield self._collate([dataset[index] for index in indices])
            return
        sampler = self.sampler
        if sampler is None:
            sampler = (
                RandomSampler(
                    dataset, state=typing.cast(RandomState, self.state)
                )
                if self.shuffle
                else SequentialSampler(dataset)
            )
        if self.batch_size is None:
            yield from (dataset[index] for index in sampler)
            return
        for indices in BatchSampler(
            sampler, self.batch_size, drop_last=self.drop_last
        ):
            yield self._collate([dataset[index] for index in indices])

    def _iterable_batches(self) -> typing.Iterator[object]:
        dataset = typing.cast(IterableDataset[object], self.dataset)
        if self.batch_size is None:
            yield from dataset
            return
        batch: list[object] = []
        for sample in dataset:
            batch.append(sample)
            if len(batch) == self.batch_size:
                yield self._collate(batch)
                batch = []
        if batch and not self.drop_last:
            yield self._collate(batch)

    def __iter__(self) -> typing.Iterator[object]:
        """Return a new independent deterministic iterator."""
        return (
            self._map_batches()
            if is_map_style_dataset(self.dataset)
            else self._iterable_batches()
        )

    def __len__(self) -> int:
        """Return stable batch count when the dataset has a length."""
        if is_map_style_dataset(self.dataset):
            if self.batch_sampler is not None:
                try:
                    return len(self.batch_sampler)  # type: ignore[arg-type]
                except TypeError as exception:
                    raise TypeError(
                        "DataLoader: batch_sampler has no stable length"
                    ) from exception
            count = (
                len(self.sampler)
                if self.sampler is not None
                else len(self.dataset)
            )
        else:
            try:
                count = len(self.dataset)  # type: ignore[arg-type]
            except TypeError as exception:
                raise TypeError(
                    "DataLoader: streaming dataset has no stable length"
                ) from exception
        if self.batch_size is None:
            return count
        if self.drop_last:
            return count // self.batch_size
        return (count + self.batch_size - 1) // self.batch_size


__all__ = ["DataLoader"]
