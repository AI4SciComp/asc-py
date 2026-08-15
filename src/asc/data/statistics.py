# Copyright 2026 AI4SciComp contributors.
# SPDX-License-Identifier: Apache-2.0
"""Stable streaming statistics over finite or iterable datasets."""

from __future__ import annotations

import collections.abc
import dataclasses
import itertools
import typing

from asc import _array_api_compat, errors
from asc.core import namespace as namespace_module
from asc.data.dataset import Dataset, IterableDataset, is_map_style_dataset


@dataclasses.dataclass(frozen=True, slots=True)
class Statistics:
    """Population statistics accumulated with scaled online moments."""

    count: int
    minimum: object
    maximum: object
    mean: object
    variance: object
    std: object


def _samples(
    dataset: Dataset[object] | IterableDataset[object],
) -> typing.Iterator[object]:
    if is_map_style_dataset(dataset):
        map_dataset = typing.cast(Dataset[object], dataset)
        yield from (map_dataset[index] for index in range(len(map_dataset)))
    else:
        yield from typing.cast(IterableDataset[object], dataset)


class _Accumulator:
    """Mutable private scale-normalized state used during one streaming pass."""

    def __init__(self, operation: str) -> None:
        self.operation = operation
        self.count = 0
        self.minimum: object | None = None
        self.maximum: object | None = None
        self.mean: object | None = None
        self.m2: object | None = None
        self.scale: object | None = None
        self.xp: object | None = None

    def update(self, value: object) -> None:
        """Incorporate one equally shaped real-floating array."""
        current_xp = namespace_module.array_namespace(value)
        if not current_xp.isdtype(value.dtype, "real floating"):
            raise errors.DTypeError(
                f"{self.operation}: statistics require real floating arrays"
            )
        if int(current_xp.finfo(value.dtype).bits) < 32:
            value = current_xp.astype(value, current_xp.float32, copy=True)
        if self.count == 0:
            self.xp = current_xp
            self.minimum = current_xp.astype(value, value.dtype, copy=True)
            self.maximum = current_xp.astype(value, value.dtype, copy=True)
            self.scale = current_xp.abs(value)
            safe_scale = current_xp.where(
                self.scale == 0,
                current_xp.ones_like(self.scale),
                self.scale,
            )
            self.mean = value / safe_scale
            self.m2 = current_xp.zeros_like(value)
            self.count = 1
            return
        assert (
            self.mean is not None
            and self.m2 is not None
            and self.scale is not None
        )
        namespace_module.array_namespace(typing.cast(object, self.mean), value)
        if value.shape != self.mean.shape:
            raise errors.DataSpecError(
                f"{self.operation}: every sample field must have one shape"
            )
        self.minimum = current_xp.minimum(self.minimum, value)
        self.maximum = current_xp.maximum(self.maximum, value)
        next_scale = current_xp.maximum(self.scale, current_xp.abs(value))
        safe_next_scale = current_xp.where(
            next_scale == 0,
            current_xp.ones_like(next_scale),
            next_scale,
        )
        ratio = self.scale / safe_next_scale
        adjusted_mean = self.mean * ratio
        adjusted_m2 = self.m2 * ratio * ratio
        normalized = value / safe_next_scale
        self.count += 1
        delta = normalized - adjusted_mean
        inverse_count = current_xp.asarray(
            1.0 / self.count,
            dtype=self.mean.dtype,
            device=_array_api_compat.compat.device(self.mean),
        )
        self.mean = adjusted_mean + delta * inverse_count
        self.m2 = adjusted_m2 + delta * (normalized - self.mean)
        self.scale = next_scale

    def finish(self) -> Statistics:
        """Freeze the accumulated population statistics."""
        if self.count == 0 or self.xp is None or self.scale is None:
            raise errors.DatasetError(
                f"{self.operation}: dataset must not be empty"
            )
        assert self.mean is not None and self.m2 is not None
        inverse_count = self.xp.asarray(
            1.0 / self.count,
            dtype=self.m2.dtype,
            device=_array_api_compat.compat.device(self.m2),
        )
        normalized_variance = self.xp.maximum(
            self.m2 * inverse_count, self.xp.zeros_like(self.m2)
        )
        mean = self.mean * self.scale
        std = self.xp.sqrt(normalized_variance) * self.scale
        maximum = self.xp.full_like(
            normalized_variance, float(self.xp.finfo(self.scale.dtype).max)
        )
        limit_scale = self.xp.maximum(self.scale, self.xp.ones_like(self.scale))
        threshold = maximum / limit_scale / limit_scale
        overflow = normalized_variance > threshold
        safe_variance = (
            self.xp.minimum(normalized_variance, threshold)
            * self.scale
            * self.scale
        )
        variance = self.xp.where(
            overflow,
            self.xp.full_like(normalized_variance, float("inf")),
            safe_variance,
        )
        return Statistics(
            self.count,
            self.minimum,
            self.maximum,
            mean,
            variance,
            std,
        )


def _statistics(values: typing.Iterable[object], operation: str) -> Statistics:
    accumulator = _Accumulator(operation)
    for value in values:
        accumulator.update(value)
    return accumulator.finish()


def dataset_statistics(
    dataset: Dataset[object] | IterableDataset[object],
    *,
    fields: collections.abc.Sequence[str] | None = None,
) -> Statistics | dict[str, Statistics]:
    """Compute elementwise streaming statistics, optionally by mapping field."""
    iterator = _samples(dataset)
    try:
        first = next(iterator)
    except StopIteration as exception:
        raise errors.DatasetError(
            "dataset_statistics: dataset must not be empty"
        ) from exception
    if fields is None and not isinstance(first, collections.abc.Mapping):
        return _statistics(
            itertools.chain((first,), iterator), "dataset_statistics"
        )
    if not isinstance(first, collections.abc.Mapping):
        raise errors.DataSpecError(
            "dataset_statistics: fields require mapping samples"
        )
    selected = tuple(first.keys()) if fields is None else tuple(fields)
    if (
        not selected
        or len(set(selected)) != len(selected)
        or any(not isinstance(field, str) or not field for field in selected)
    ):
        raise errors.DataSpecError(
            "dataset_statistics: fields must be unique non-empty strings"
        )
    missing = [field for field in selected if field not in first]
    if missing:
        raise errors.DataSpecError(
            f"dataset_statistics: missing fields {missing!r}"
        )
    accumulators = {
        field: _Accumulator(f"dataset_statistics[{field}]")
        for field in selected
    }
    for position, sample in enumerate(itertools.chain((first,), iterator)):
        if not isinstance(sample, collections.abc.Mapping):
            raise errors.DataSpecError(
                "dataset_statistics: every sample must be a mapping; "
                f"sample {position} is {type(sample).__name__}"
            )
        missing = [field for field in selected if field not in sample]
        if missing:
            raise errors.DataSpecError(
                "dataset_statistics: sample "
                f"{position} is missing fields {missing!r}"
            )
        for field, accumulator in accumulators.items():
            accumulator.update(sample[field])
    return {
        field: accumulator.finish()
        for field, accumulator in accumulators.items()
    }


__all__ = ["Statistics", "dataset_statistics"]
