# Copyright 2026 AI4SciComp contributors.
# SPDX-License-Identifier: Apache-2.0
"""PyTree-preserving combined loader modes."""

from __future__ import annotations

import collections.abc
import itertools
import typing

from asc import errors
from asc.tree import TreeSpec, tree_flatten, tree_unflatten


class CombinedBatch(typing.NamedTuple):
    """Combined data plus stable batch and optional loader indices."""

    data: object
    batch_index: int
    loader_index: int | None


Mode = typing.Literal["min_size", "max_size_cycle", "max_size", "sequential"]


class CombinedLoader(collections.abc.Iterable[CombinedBatch]):
    """Combine an arbitrary tree of finite loaders under one mode."""

    def __init__(
        self,
        loaders: object,
        *,
        mode: Mode = "min_size",
        limits: int | object | None = None,
    ) -> None:
        if mode not in {"min_size", "max_size_cycle", "max_size", "sequential"}:
            raise errors.DataLoaderError(
                f"CombinedLoader: unsupported mode {mode!r}"
            )
        flattened, spec = tree_flatten(loaders)
        if not flattened:
            raise errors.DataLoaderError(
                "CombinedLoader: loaders tree must contain at least one leaf"
            )
        if not all(
            isinstance(loader, collections.abc.Iterable) for loader in flattened
        ):
            raise errors.DataLoaderError(
                "CombinedLoader: every tree leaf must be iterable"
            )
        if any(iter(loader) is loader for loader in flattened):
            raise errors.DataLoaderError(
                "CombinedLoader: loader leaves must be reiterable, not "
                "one-shot iterators"
            )
        self.loaders = loaders
        self.mode = mode
        self._flattened = tuple(flattened)
        self._spec = spec
        self._limits = self._normalize_limits(limits, spec)

    def _normalize_limits(
        self, limits: int | object | None, spec: TreeSpec
    ) -> tuple[int | None, ...]:
        if limits is None:
            return (None,) * len(self._flattened)
        if isinstance(limits, int) and not isinstance(limits, bool):
            values = [limits] * len(self._flattened)
        else:
            values, limit_spec = tree_flatten(limits)
            if limit_spec != spec:
                raise errors.DataLoaderError(
                    "CombinedLoader: tree-shaped limits must match loaders"
                )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in values
        ):
            raise errors.DataLoaderError(
                "CombinedLoader: limits must be non-negative integers"
            )
        return tuple(typing.cast(int, value) for value in values)

    def _lengths(self) -> tuple[int, ...]:
        lengths: list[int] = []
        for position, (loader, limit) in enumerate(
            zip(self._flattened, self._limits, strict=True)
        ):
            try:
                length = len(loader)  # type: ignore[arg-type]
            except TypeError as exception:
                raise TypeError(
                    "CombinedLoader: loader leaf "
                    f"{position} has no stable length"
                ) from exception
            lengths.append(length if limit is None else min(length, limit))
        return tuple(lengths)

    def __len__(self) -> int:
        """Return the exact combined length before or after iteration."""
        lengths = self._lengths()
        if self.mode == "min_size":
            return min(lengths)
        if self.mode in {"max_size", "max_size_cycle"}:
            return max(lengths)
        return sum(lengths)

    def _parallel(self) -> typing.Iterator[CombinedBatch]:
        lengths = self._lengths()
        iterators = [iter(loader) for loader in self._flattened]
        steps = len(self)
        if (
            self.mode == "max_size_cycle"
            and any(length == 0 for length in lengths)
            and steps
        ):
            raise errors.DataLoaderError(
                "CombinedLoader: max_size_cycle cannot cycle an empty loader"
            )
        for batch_index in range(steps):
            outputs: list[object] = []
            for index, iterator in enumerate(iterators):
                length = lengths[index]
                if self.mode == "max_size" and batch_index >= length:
                    outputs.append(None)
                    continue
                if (
                    self.mode == "max_size_cycle"
                    and batch_index
                    and batch_index % length == 0
                ):
                    iterators[index] = iter(self._flattened[index])
                    iterator = iterators[index]
                try:
                    outputs.append(next(iterator))
                except StopIteration:
                    if self.mode == "max_size":
                        outputs.append(None)
                    elif self.mode == "max_size_cycle":
                        iterators[index] = iter(self._flattened[index])
                        try:
                            outputs.append(next(iterators[index]))
                        except StopIteration as exception:
                            raise errors.DataLoaderError(
                                "CombinedLoader: cycled loader became empty"
                            ) from exception
                    else:
                        return
            yield CombinedBatch(
                tree_unflatten(self._spec, outputs), batch_index, None
            )

    def _sequential(self) -> typing.Iterator[CombinedBatch]:
        for loader_index, (loader, limit) in enumerate(
            zip(self._flattened, self._limits, strict=True)
        ):
            values = (
                loader if limit is None else itertools.islice(loader, limit)
            )
            for batch_index, value in enumerate(values):
                yield CombinedBatch(value, batch_index, loader_index)

    def __iter__(self) -> typing.Iterator[CombinedBatch]:
        """Return a fresh iterator and reset every child loader."""
        return (
            self._sequential()
            if self.mode == "sequential"
            else self._parallel()
        )


__all__ = ["CombinedBatch", "CombinedLoader", "Mode"]
