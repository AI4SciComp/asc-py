# Copyright 2026 AI4SciComp contributors.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic finite-dataset splitting."""

from __future__ import annotations

import math
import typing

from asc import errors
from asc.data.dataset import Dataset, Subset
from asc.random import RandomState, permutation


class DatasetSplits(typing.NamedTuple):
    """Named train, validation, and test subsets."""

    train: Subset
    validation: Subset
    test: Subset


def _indices(length: int, state: RandomState | None) -> list[int]:
    if state is None:
        return list(range(length))
    values, _ = permutation(length, state=state)
    from asc.conversion import to_numpy

    native = to_numpy(values, copy=True)
    return [int(value) for value in native.tolist()]  # type: ignore[attr-defined]


def _lengths(total: int, sizes: typing.Sequence[int | float]) -> list[int]:
    if not sizes:
        raise errors.DataSplitError("split_dataset: sizes must not be empty")
    if all(
        isinstance(size, int) and not isinstance(size, bool) for size in sizes
    ):
        result = [int(size) for size in sizes]
        if any(size < 0 for size in result) or sum(result) != total:
            raise errors.DataSplitError(
                "split_dataset: integer sizes must be non-negative and sum "
                "to dataset length"
            )
        return result
    if not all(isinstance(size, float) for size in sizes):
        raise errors.DataSplitError(
            "split_dataset: sizes must be all integers or all fractions"
        )
    fractions = [float(size) for size in sizes]
    if any(
        not math.isfinite(size) or size < 0 for size in fractions
    ) or not math.isclose(sum(fractions), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise errors.DataSplitError(
            "split_dataset: fractions must be finite, non-negative, and sum "
            "to one"
        )
    result = [math.floor(total * fraction) for fraction in fractions]
    for index in range(total - sum(result)):
        result[index % len(result)] += 1
    return result


def split_dataset(
    dataset: Dataset[object],
    sizes: typing.Sequence[int | float],
    *,
    state: RandomState | None = None,
) -> tuple[Subset, ...]:
    """Return non-overlapping subsets with deterministic remainders."""
    lengths = _lengths(len(dataset), sizes)
    indices = _indices(len(dataset), state)
    result: list[Subset] = []
    offset = 0
    for length in lengths:
        result.append(Subset(dataset, indices[offset : offset + length]))
        offset += length
    return tuple(result)


def train_validation_test_split(
    dataset: Dataset[object],
    *,
    train: float,
    validation: float,
    test: float,
    state: RandomState | None = None,
) -> DatasetSplits:
    """Return named deterministic train/validation/test subsets."""
    result = split_dataset(dataset, (train, validation, test), state=state)
    return DatasetSplits(result[0], result[1], result[2])


def kfold_indices(
    length: int,
    folds: int,
    *,
    state: RandomState | None = None,
) -> tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]:
    """Return deterministic ``(train, validation)`` indices for K folds."""
    if isinstance(length, bool) or not isinstance(length, int) or length < 1:
        raise errors.DataSplitError("kfold_indices: length must be positive")
    if (
        isinstance(folds, bool)
        or not isinstance(folds, int)
        or not 2 <= folds <= length
    ):
        raise errors.DataSplitError(
            "kfold_indices: folds must be an integer in [2, length]"
        )
    ordered = _indices(length, state)
    sizes = _lengths(length, [1.0 / folds] * folds)
    result: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    offset = 0
    for size in sizes:
        validation = tuple(ordered[offset : offset + size])
        validation_set = set(validation)
        train = tuple(index for index in ordered if index not in validation_set)
        result.append((train, validation))
        offset += size
    return tuple(result)


__all__ = [
    "DatasetSplits",
    "kfold_indices",
    "split_dataset",
    "train_validation_test_split",
]
