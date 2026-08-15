# Copyright 2026 AI4SciComp contributors.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic samplers driven by explicit random state."""

from __future__ import annotations

import abc
import collections.abc
import math
import typing

from asc import errors
from asc.core.backend import backend as select_backend
from asc.random import RandomState, choice, permutation


class Sampler(abc.ABC):
    """Narrow sampler protocol for Python integer indices."""

    @abc.abstractmethod
    def __iter__(self) -> typing.Iterator[int]:
        """Return a fresh deterministic index iterator."""

    @abc.abstractmethod
    def __len__(self) -> int:
        """Return the exact number of yielded indices."""


def _tolist(array: object) -> list[int]:
    from asc.conversion import to_numpy

    native = to_numpy(array, copy=True)
    return [int(value) for value in native.reshape(-1).tolist()]  # type: ignore[attr-defined]


def _validated_state(state: object, operation: str) -> RandomState:
    """Require explicit sampler state at configuration time."""
    if not isinstance(state, RandomState):
        raise errors.DataLoaderError(
            f"{operation}: state must be an asc.RandomState"
        )
    return state


class SequentialSampler(Sampler):
    """Yield every index exactly once in ascending order."""

    def __init__(self, data_source: collections.abc.Sized) -> None:
        self.data_source = data_source

    def __iter__(self) -> typing.Iterator[int]:
        return iter(range(len(self.data_source)))

    def __len__(self) -> int:
        return len(self.data_source)


class RandomSampler(Sampler):
    """Sample a finite data source with explicit replacement semantics."""

    def __init__(
        self,
        data_source: collections.abc.Sized,
        *,
        state: RandomState,
        replacement: bool = False,
        num_samples: int | None = None,
    ) -> None:
        if not isinstance(replacement, bool):
            raise errors.DataLoaderError(
                "RandomSampler: replacement must be Boolean"
            )
        self.data_source = data_source
        self.state = _validated_state(state, "RandomSampler")
        self.replacement = replacement
        self.num_samples = (
            len(data_source) if num_samples is None else num_samples
        )
        if (
            isinstance(self.num_samples, bool)
            or not isinstance(self.num_samples, int)
            or self.num_samples < 0
        ):
            raise errors.DataLoaderError(
                "RandomSampler: num_samples must be a non-negative integer"
            )
        if replacement and self.num_samples > 0 and len(data_source) == 0:
            raise errors.DataLoaderError(
                "RandomSampler: cannot draw positive samples with replacement "
                "from an empty data source"
            )
        if not replacement and self.num_samples > len(data_source):
            raise errors.DataLoaderError(
                "RandomSampler: without replacement num_samples cannot "
                "exceed length"
            )

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self) -> typing.Iterator[int]:
        if self.num_samples == 0:
            return iter(())
        if self.replacement:
            values, _ = choice(
                len(self.data_source),
                (self.num_samples,),
                state=self.state,
                replace=True,
            )
        else:
            values, _ = permutation(len(self.data_source), state=self.state)
        return iter(_tolist(values)[: self.num_samples])


class SubsetRandomSampler(Sampler):
    """Yield only supplied indices in explicit-state random order."""

    def __init__(
        self, indices: collections.abc.Sequence[int], *, state: RandomState
    ) -> None:
        if any(
            isinstance(index, bool) or not isinstance(index, int)
            for index in indices
        ):
            raise errors.DataLoaderError(
                "SubsetRandomSampler: indices must contain only integers"
            )
        self.indices = tuple(indices)
        self.state = _validated_state(state, "SubsetRandomSampler")

    def __len__(self) -> int:
        return len(self.indices)

    def __iter__(self) -> typing.Iterator[int]:
        order, _ = permutation(len(self.indices), state=self.state)
        return iter(self.indices[index] for index in _tolist(order))


class WeightedRandomSampler(Sampler):
    """Sample validated non-negative finite weights."""

    def __init__(
        self,
        weights: collections.abc.Sequence[float],
        num_samples: int,
        *,
        state: RandomState,
        replacement: bool = True,
    ) -> None:
        if not isinstance(replacement, bool):
            raise errors.DataLoaderError(
                "WeightedRandomSampler: replacement must be Boolean"
            )
        try:
            self.weights = tuple(
                float(weight)
                if isinstance(weight, (int, float))
                and not isinstance(weight, bool)
                else math.nan
                for weight in weights
            )
        except OverflowError as exception:
            raise errors.DataLoaderError(
                "WeightedRandomSampler: weights must be finite Python real "
                "scalars"
            ) from exception
        self.num_samples = num_samples
        self.state = _validated_state(state, "WeightedRandomSampler")
        self.replacement = replacement
        if (
            not self.weights
            or any(
                not math.isfinite(weight) or weight < 0
                for weight in self.weights
            )
            or not any(weight > 0 for weight in self.weights)
        ):
            raise errors.DataLoaderError(
                "WeightedRandomSampler: weights must be finite, non-negative, "
                "and have positive sum"
            )
        if (
            isinstance(num_samples, bool)
            or not isinstance(num_samples, int)
            or num_samples < 0
        ):
            raise errors.DataLoaderError(
                "WeightedRandomSampler: num_samples must be non-negative"
            )
        if not replacement and num_samples > sum(
            weight > 0 for weight in self.weights
        ):
            raise errors.DataLoaderError(
                "WeightedRandomSampler: insufficient positive weights without "
                "replacement"
            )
        maximum = max(self.weights)
        scaled = tuple(weight / maximum for weight in self.weights)
        total = math.fsum(scaled)
        self._probabilities = tuple(weight / total for weight in scaled)
        minimum_float32 = 2.0**-149
        if any(
            weight > 0 and probability <= minimum_float32 / 2
            for weight, probability in zip(
                self.weights, self._probabilities, strict=True
            )
        ):
            raise errors.DataLoaderError(
                "WeightedRandomSampler: every positive normalized weight must "
                "be representable in the portable float32 probability dtype"
            )

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self) -> typing.Iterator[int]:
        if self.num_samples == 0:
            return iter(())
        state_device = getattr(self.state.key, "device", None)
        selected = select_backend(self.state.backend, device=state_device)
        probabilities = selected.xp.asarray(
            self._probabilities,
            dtype=selected.xp.float32,
            device=selected.device,
        )
        values, _ = choice(
            len(self.weights),
            (self.num_samples,),
            state=self.state,
            replace=self.replacement,
            probabilities=probabilities,
        )
        return iter(_tolist(values))


class BatchSampler(collections.abc.Iterable[list[int]]):
    """Group any finite sampler into validated batches."""

    def __init__(
        self, sampler: Sampler, batch_size: int, *, drop_last: bool = False
    ) -> None:
        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or batch_size <= 0
        ):
            raise errors.DataLoaderError(
                "BatchSampler: batch_size must be a positive integer"
            )
        if not isinstance(drop_last, bool):
            raise errors.DataLoaderError(
                "BatchSampler: drop_last must be Boolean"
            )
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        return (len(self.sampler) + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> typing.Iterator[list[int]]:
        batch: list[int] = []
        for index in self.sampler:
            batch.append(index)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if batch and not self.drop_last:
            yield batch


__all__ = [
    "BatchSampler",
    "RandomSampler",
    "Sampler",
    "SequentialSampler",
    "SubsetRandomSampler",
    "WeightedRandomSampler",
]
