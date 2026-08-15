# Copyright 2026 AI4SciComp contributors.
# SPDX-License-Identifier: Apache-2.0
"""Backend-neutral dataset abstractions and composition."""

from __future__ import annotations

import abc
import bisect
import collections
import collections.abc
import importlib
import typing

from asc import _array_api_compat, errors
from asc import typing as asc_typing
from asc.core import namespace as namespace_module
from asc.core._indexing import safe_index_dtype
from asc.extensions import _dispatch
from asc.tree import tree_replace

Index = int | slice | collections.abc.Sequence[int] | object


class Dataset[SampleT](abc.ABC):
    """Typed finite map-style dataset interface."""

    @abc.abstractmethod
    def __len__(self) -> int:
        """Return a stable non-negative sample count."""

    @abc.abstractmethod
    def __getitem__(self, index: Index) -> SampleT | list[SampleT]:
        """Return one sample or a deterministic collection of samples."""


class IterableDataset[SampleT](abc.ABC):
    """Typed streaming dataset with optional length."""

    @abc.abstractmethod
    def __iter__(self) -> typing.Iterator[SampleT]:
        """Return a fresh sample iterator."""


def _normalize_scalar(index: int, length: int, operation: str) -> int:
    if isinstance(index, bool):
        raise IndexError(f"{operation}: Boolean indices are not supported")
    normalized = index + length if index < 0 else index
    if normalized < 0 or normalized >= length:
        raise IndexError(
            f"{operation}: index {index} is outside dataset length {length}"
        )
    return normalized


def _python_indices(
    index: Index, length: int, operation: str
) -> list[int] | None:
    if isinstance(index, bool):
        raise IndexError(f"{operation}: Boolean indices are not supported")
    if isinstance(index, int):
        return None
    if isinstance(index, slice):
        return list(range(*index.indices(length)))
    if isinstance(index, collections.abc.Sequence) and not isinstance(
        index, (str, bytes)
    ):
        if any(
            isinstance(item, bool) or not isinstance(item, int)
            for item in index
        ):
            raise IndexError(
                f"{operation}: index sequences must contain only integers"
            )
        return [_normalize_scalar(item, length, operation) for item in index]
    if _array_api_compat.compat.is_array_api_obj(index):
        xp = namespace_module.array_namespace(index)
        if len(index.shape) != 1 or not xp.isdtype(
            index.dtype, "signed integer"
        ):
            raise IndexError(
                f"{operation}: native index arrays must be 1-D signed integers"
            )
        from asc.conversion import to_numpy

        try:
            host = to_numpy(index, copy=True)
        except errors.ConversionError as exception:
            raise IndexError(
                f"{operation}: index arrays must be dense CPU values without "
                "a graph"
            ) from exception
        return [
            _normalize_scalar(int(item), length, operation)
            for item in host.tolist()
        ]
    return None


def _check_array_index_bounds(
    invalid: object, backend: asc_typing.BackendName
) -> None:
    """Validate native indices without scalarizing traced or batched values."""
    if backend == "jax" and "Tracer" in type(invalid).__name__:
        checkify = importlib.import_module("jax.experimental.checkify")
        checkify.check(~invalid, "asc ArrayDataset index out of bounds")
        return
    if backend == "torch":
        adapter = _dispatch.load_backend("torch")
        try:
            adapter.check_index_bounds(invalid, "ArrayDataset")
        except IndexError as exception:
            raise IndexError(
                "ArrayDataset: native index array contains an index outside "
                "the dataset"
            ) from exception
        return
    if bool(invalid):
        raise IndexError(
            "ArrayDataset: native index array contains an index outside the "
            "dataset"
        )


class ArrayDataset(Dataset[object]):
    """View samples along one axis of a native array."""

    def __init__(
        self,
        array: object,
        *,
        sample_axis: int = 0,
        field: object | None = None,
    ) -> None:
        namespace_module.array_namespace(array)
        if isinstance(sample_axis, bool) or not isinstance(sample_axis, int):
            raise errors.DatasetError(
                "ArrayDataset: sample_axis must be a non-Boolean integer"
            )
        ndim = len(array.shape)
        axis = sample_axis + ndim if sample_axis < 0 else sample_axis
        if axis < 0 or axis >= ndim:
            raise errors.DatasetError(
                "ArrayDataset: sample_axis is outside the array rank"
            )
        self._array = array
        self._sample_axis = axis
        self._field = field

    @property
    def array(self) -> object:
        """Return the caller-owned native array without copying it."""
        return self._array

    @property
    def sample_axis(self) -> int:
        """Return the normalized sample axis."""
        return self._sample_axis

    @property
    def field(self) -> object | None:
        """Return optional immutable field metadata."""
        return self._field

    def __len__(self) -> int:
        return typing.cast(int, self._array.shape[self._sample_axis])

    def __getitem__(self, index: Index) -> object:
        if isinstance(index, int):
            index = _normalize_scalar(index, len(self), "ArrayDataset")
        elif isinstance(index, slice):
            if index.step is not None and index.step < 0:
                start, stop, step = index.indices(len(self))
                xp = namespace_module.array_namespace(self._array)
                backend = namespace_module.identify_backend(xp)
                native_indices = xp.arange(
                    start,
                    stop,
                    step,
                    dtype=safe_index_dtype(
                        xp,
                        backend,
                        len(self) - 1,
                        "ArrayDataset",
                    ),
                    device=_array_api_compat.compat.device(self._array),
                )
                return xp.take(
                    self._array,
                    native_indices,
                    axis=self._sample_axis,
                )
        elif _array_api_compat.compat.is_array_api_obj(index):
            xp = namespace_module.array_namespace(self._array, index)
            backend = namespace_module.identify_backend(xp)
            if len(index.shape) != 1 or not xp.isdtype(
                index.dtype, "signed integer"
            ):
                raise IndexError(
                    "ArrayDataset: native index arrays must be 1-D signed "
                    "integers"
                )
            dtype = safe_index_dtype(
                xp,
                backend,
                len(self) - 1,
                "ArrayDataset",
            )
            widened = xp.astype(index, dtype, copy=True)
            normalized = xp.where(widened < 0, widened + len(self), widened)
            _check_array_index_bounds(
                xp.any((normalized < 0) | (normalized >= len(self))), backend
            )
            return xp.take(
                self._array,
                typing.cast(asc_typing.NativeArray, normalized),
                axis=self._sample_axis,
            )
        else:
            indices = _python_indices(index, len(self), "ArrayDataset")
            if indices is None:
                raise IndexError(
                    "ArrayDataset: index must be integer, slice, sequence, "
                    "or native index array"
                )
            xp = namespace_module.array_namespace(self._array)
            native_indices = xp.asarray(
                indices,
                dtype=safe_index_dtype(
                    xp,
                    namespace_module.identify_backend(xp),
                    len(self) - 1,
                    "ArrayDataset",
                ),
                device=_array_api_compat.compat.device(self._array),
            )
            return xp.take(self._array, native_indices, axis=self._sample_axis)
        selector: list[object] = [slice(None)] * len(self._array.shape)
        selector[self._sample_axis] = index
        try:
            return self._array[tuple(selector)]  # type: ignore[index]
        except (IndexError, TypeError) as exception:
            raise IndexError(
                f"ArrayDataset: unsupported index type {type(index).__name__}"
            ) from exception


class TupleDataset(Dataset[tuple[object, ...]]):
    """Aligned sequence of datasets or native arrays."""

    def __init__(self, *fields: Dataset[object] | object) -> None:
        if not fields:
            raise errors.DatasetError(
                "TupleDataset: at least one field is required"
            )
        self._datasets = tuple(_as_dataset(field) for field in fields)
        _validate_aligned(self._datasets, "TupleDataset")

    def __len__(self) -> int:
        return len(self._datasets[0])

    def __getitem__(
        self, index: Index
    ) -> tuple[object, ...] | list[tuple[object, ...]]:
        indices = _python_indices(index, len(self), "TupleDataset")
        if indices is not None:
            return [
                typing.cast(tuple[object, ...], self[item]) for item in indices
            ]
        return tuple(dataset[index] for dataset in self._datasets)


class MappingDataset(Dataset[dict[str, object]]):
    """Ordered aligned named fields with stable mapping output."""

    def __init__(
        self, fields: collections.abc.Mapping[str, Dataset[object] | object]
    ) -> None:
        if not fields:
            raise errors.DatasetError(
                "MappingDataset: at least one field is required"
            )
        if any(not isinstance(name, str) or not name for name in fields):
            raise errors.DatasetError(
                "MappingDataset: field names must be non-empty strings"
            )
        self._datasets = collections.OrderedDict(
            (name, _as_dataset(field)) for name, field in fields.items()
        )
        _validate_aligned(tuple(self._datasets.values()), "MappingDataset")

    def __len__(self) -> int:
        return len(next(iter(self._datasets.values())))

    def __getitem__(
        self, index: Index
    ) -> dict[str, object] | list[dict[str, object]]:
        indices = _python_indices(index, len(self), "MappingDataset")
        if indices is not None:
            return [
                typing.cast(dict[str, object], self[item]) for item in indices
            ]
        return {
            name: dataset[index] for name, dataset in self._datasets.items()
        }


def _as_dataset(value: Dataset[object] | object) -> Dataset[object]:
    if isinstance(value, Dataset):
        return value
    return ArrayDataset(value)


def _validate_aligned(
    datasets: tuple[Dataset[object], ...], operation: str
) -> None:
    lengths = tuple(len(dataset) for dataset in datasets)
    if len(set(lengths)) != 1:
        raise errors.DatasetError(
            f"{operation}: field sample counts must match; observed {lengths!r}"
        )


class ConcatDataset(Dataset[object]):
    """Concatenate non-empty finite datasets without copying samples."""

    def __init__(
        self, datasets: collections.abc.Iterable[Dataset[object]]
    ) -> None:
        self.datasets = tuple(datasets)
        if not self.datasets:
            raise errors.DatasetError(
                "ConcatDataset: datasets must not be empty"
            )
        total = 0
        cumulative: list[int] = []
        for dataset in self.datasets:
            total += len(dataset)
            cumulative.append(total)
        self._cumulative = tuple(cumulative)

    def __len__(self) -> int:
        return self._cumulative[-1]

    def __getitem__(self, index: Index) -> object:
        indices = _python_indices(index, len(self), "ConcatDataset")
        if indices is not None:
            return [self[item] for item in indices]
        if not isinstance(index, int):
            raise IndexError(
                "ConcatDataset: index must be integer, slice, or sequence"
            )
        normalized = _normalize_scalar(index, len(self), "ConcatDataset")
        dataset_index = bisect.bisect_right(self._cumulative, normalized)
        offset = (
            0 if dataset_index == 0 else self._cumulative[dataset_index - 1]
        )
        return self.datasets[dataset_index][normalized - offset]


class Subset(Dataset[object]):
    """Indexed dataset view that never copies source samples."""

    def __init__(
        self, dataset: Dataset[object], indices: collections.abc.Sequence[int]
    ) -> None:
        self.dataset = dataset
        if any(
            isinstance(index, bool) or not isinstance(index, int)
            for index in indices
        ):
            raise errors.DatasetError(
                "Subset: indices must contain only integers"
            )
        self.indices = tuple(
            _normalize_scalar(index, len(dataset), "Subset")
            for index in indices
        )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: Index) -> object:
        selected = _python_indices(index, len(self), "Subset")
        if selected is not None:
            return [self.dataset[self.indices[item]] for item in selected]
        if not isinstance(index, int):
            raise IndexError(
                "Subset: index must be integer, slice, or sequence"
            )
        normalized = _normalize_scalar(index, len(self), "Subset")
        return self.dataset[self.indices[normalized]]


class TransformDataset(Dataset[object]):
    """Apply a transform lazily to a sample, input, target, or named field."""

    def __init__(
        self,
        dataset: Dataset[object],
        transform: typing.Callable[[object], object],
        *,
        target: str = "sample",
    ) -> None:
        self.dataset = dataset
        self.transform = transform
        self.target = target

    def __len__(self) -> int:
        return len(self.dataset)

    def _apply(self, sample: object) -> object:
        if self.target == "sample":
            return self.transform(sample)
        if isinstance(sample, collections.abc.Mapping):
            key = self.target
            if key not in sample:
                raise errors.DatasetError(
                    f"TransformDataset: sample has no field {key!r}"
                )
            try:
                return tree_replace(sample, (key,), self.transform(sample[key]))
            except (KeyError, TypeError, ValueError) as exception:
                raise errors.DatasetError(
                    "TransformDataset: mapping type cannot be reconstructed"
                ) from exception
        if isinstance(sample, tuple) and self.target in {"input", "target"}:
            position = 0 if self.target == "input" else 1
            if len(sample) <= position:
                raise errors.DatasetError(
                    f"TransformDataset: tuple has no {self.target} position"
                )
            result = list(sample)
            result[position] = self.transform(result[position])
            if hasattr(type(sample), "_fields"):
                field = type(sample)._fields[position]
                return sample._replace(**{field: result[position]})
            return tuple(result)
        raise errors.DatasetError(
            f"TransformDataset: target {self.target!r} is not valid for sample"
        )

    def __getitem__(self, index: Index) -> object:
        selected = _python_indices(index, len(self), "TransformDataset")
        if selected is not None:
            return [self._apply(self.dataset[item]) for item in selected]
        return self._apply(self.dataset[index])


class FilteredDataset(Dataset[object], IterableDataset[object]):
    """Deterministic map view or lazy iterable predicate view."""

    def __init__(
        self,
        dataset: Dataset[object] | IterableDataset[object],
        predicate: typing.Callable[[object], bool],
    ) -> None:
        self.dataset = dataset
        self.predicate = predicate
        map_dataset = typing.cast(Dataset[object], dataset)
        self._indices = (
            tuple(
                index
                for index in range(len(map_dataset))
                if predicate(map_dataset[index])
            )
            if is_map_style_dataset(dataset)
            else None
        )

    def __len__(self) -> int:
        if self._indices is None:
            raise TypeError(
                "FilteredDataset: streaming view has no finite length"
            )
        return len(self._indices)

    def __getitem__(self, index: Index) -> object:
        if self._indices is None or not isinstance(self.dataset, Dataset):
            raise TypeError("FilteredDataset: streaming view is not indexable")
        selected = _python_indices(index, len(self), "FilteredDataset")
        if selected is not None:
            return [
                self.dataset[self._indices[position]] for position in selected
            ]
        if not isinstance(index, int):
            raise IndexError(
                "FilteredDataset: index must be integer, slice, or sequence"
            )
        normalized = _normalize_scalar(index, len(self), "FilteredDataset")
        return self.dataset[self._indices[normalized]]

    @property
    def is_map_style(self) -> bool:
        """Return whether this filtered view supports indexed access."""
        return self._indices is not None

    def __iter__(self) -> typing.Iterator[object]:
        if is_map_style_dataset(self.dataset):
            map_dataset = typing.cast(Dataset[object], self.dataset)
            yield from (
                map_dataset[index]
                for index in typing.cast(tuple[int, ...], self._indices)
            )
        else:
            iterable_dataset = typing.cast(
                IterableDataset[object], self.dataset
            )
            yield from (
                sample for sample in iterable_dataset if self.predicate(sample)
            )


def is_map_style_dataset(
    dataset: Dataset[object] | IterableDataset[object],
) -> bool:
    """Return whether a possibly filtered dataset supports indexed access."""
    return isinstance(dataset, Dataset) and (
        not isinstance(dataset, FilteredDataset) or dataset.is_map_style
    )


class ZipDataset(Dataset[tuple[object, ...]]):
    """Zip finite datasets with strict or minimum-size length policy."""

    def __init__(
        self,
        *datasets: Dataset[object],
        policy: typing.Literal["strict", "min_size"] = "strict",
    ) -> None:
        if not datasets:
            raise errors.DatasetError(
                "ZipDataset: at least one dataset is required"
            )
        if policy not in {"strict", "min_size"}:
            raise errors.DatasetError(
                "ZipDataset: policy must be 'strict' or 'min_size'"
            )
        lengths = tuple(len(dataset) for dataset in datasets)
        if policy == "strict" and len(set(lengths)) != 1:
            raise errors.DatasetError(
                f"ZipDataset: strict lengths must match; observed {lengths!r}"
            )
        self.datasets = datasets
        self.policy = policy
        self._length = lengths[0] if policy == "strict" else min(lengths)

    def __len__(self) -> int:
        return self._length

    def __getitem__(
        self, index: Index
    ) -> tuple[object, ...] | list[tuple[object, ...]]:
        selected = _python_indices(index, len(self), "ZipDataset")
        if selected is not None:
            return [
                typing.cast(tuple[object, ...], self[item]) for item in selected
            ]
        if not isinstance(index, int):
            raise IndexError(
                "ZipDataset: index must be integer, slice, or sequence"
            )
        normalized = _normalize_scalar(index, len(self), "ZipDataset")
        return tuple(dataset[normalized] for dataset in self.datasets)


__all__ = [
    "ArrayDataset",
    "ConcatDataset",
    "Dataset",
    "FilteredDataset",
    "Index",
    "IterableDataset",
    "MappingDataset",
    "Subset",
    "TransformDataset",
    "TupleDataset",
    "ZipDataset",
]
