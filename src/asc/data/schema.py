# Copyright 2026 AI4SciComp contributors.
# SPDX-License-Identifier: Apache-2.0
"""Immutable dataset metadata and path-aware schema validation."""

from __future__ import annotations

import dataclasses
import enum

from asc import _array_api_compat, errors
from asc.core.backend import backend_of
from asc.data.dataset import Dataset
from asc.tree import Path, TreeSpec, tree_flatten, tree_map_with_path


class SemanticRole(enum.StrEnum):
    """Portable semantic role for a dataset field."""

    INPUT = "input"
    TARGET = "target"
    WEIGHT = "weight"
    METADATA = "metadata"
    UNSPECIFIED = "unspecified"


@dataclasses.dataclass(frozen=True, slots=True)
class FieldSpec:
    """Immutable metadata for one sample leaf."""

    name: str
    shape: tuple[int, ...]
    dtype: str
    backend: str
    device: str
    dimensions: tuple[str, ...] = ()
    role: SemanticRole = SemanticRole.UNSPECIFIED

    def __post_init__(self) -> None:
        """Validate names and optional dimension labels."""
        try:
            shape = tuple(self.shape)
            dimensions = tuple(self.dimensions)
            role = SemanticRole(self.role)
        except (TypeError, ValueError) as exception:
            raise errors.DataSpecError(
                "FieldSpec: shape, dimensions, and role are invalid"
            ) from exception
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "dimensions", dimensions)
        object.__setattr__(self, "role", role)
        if not isinstance(self.name, str) or not self.name:
            raise errors.DataSpecError(
                "FieldSpec: name must be a non-empty string"
            )
        if any(
            isinstance(extent, bool)
            or not isinstance(extent, int)
            or extent < 0
            for extent in self.shape
        ):
            raise errors.DataSpecError(
                "FieldSpec: shape must contain non-negative integers"
            )
        if any(
            not isinstance(value, str) or not value
            for value in (self.dtype, self.backend, self.device)
        ):
            raise errors.DataSpecError(
                "FieldSpec: dtype, backend, and device must be non-empty "
                "strings"
            )
        if any(
            not isinstance(dimension, str) or not dimension
            for dimension in self.dimensions
        ):
            raise errors.DataSpecError(
                "FieldSpec: dimensions must contain non-empty strings"
            )
        if self.dimensions and len(self.dimensions) != len(self.shape):
            raise errors.DataSpecError(
                "FieldSpec: dimensions must match the rank of shape"
            )
        if len(set(self.dimensions)) != len(self.dimensions):
            raise errors.DataSpecError(
                "FieldSpec: dimension names must be unique"
            )


@dataclasses.dataclass(frozen=True, slots=True)
class DataSpec:
    """Immutable tree structure and ordered array-field metadata."""

    structure: TreeSpec
    fields: tuple[tuple[Path, FieldSpec], ...]

    def __post_init__(self) -> None:
        """Reject malformed metadata and paths outside the tree structure."""
        if not isinstance(self.structure, TreeSpec):
            raise errors.DataSpecError(
                "DataSpec: structure must be a valid TreeSpec"
            )
        try:
            self.structure.to_json()
        except (TypeError, ValueError) as exception:
            raise errors.DataSpecError(
                "DataSpec: structure must be a valid TreeSpec"
            ) from exception
        try:
            fields = tuple((tuple(path), field) for path, field in self.fields)
        except (TypeError, ValueError) as exception:
            raise errors.DataSpecError(
                "DataSpec: fields must contain path and FieldSpec pairs"
            ) from exception
        if any(not isinstance(field, FieldSpec) for _, field in fields):
            raise errors.DataSpecError(
                "DataSpec: every field entry must contain a FieldSpec"
            )
        if any(
            any(
                isinstance(entry, bool) or not isinstance(entry, (int, str))
                for entry in path
            )
            for path, _ in fields
        ):
            raise errors.DataSpecError(
                "DataSpec: path components must be strings or non-Boolean "
                "integers"
            )
        object.__setattr__(self, "fields", fields)
        paths = tuple(path for path, _ in fields)
        if len(paths) != len(set(paths)):
            raise errors.DataSpecError(
                "DataSpec: array field paths must be unique"
            )
        leaf_paths = set(_tree_leaf_paths(self.structure))
        if any(path not in leaf_paths for path in paths):
            raise errors.DataSpecError(
                "DataSpec: every field path must identify a tree leaf"
            )


def _tree_leaf_paths(
    structure: TreeSpec, prefix: Path = ()
) -> tuple[Path, ...]:
    """Return the stable paths represented by a validated tree structure."""
    if structure.kind == "leaf":
        return (prefix,)
    entries = (
        structure.metadata
        if structure.kind == "mapping"
        else structure.metadata[1:]
        if structure.kind in {"dataclass", "namedtuple"}
        else tuple(range(len(structure.children)))
    )
    return tuple(
        path
        for entry, child in zip(entries, structure.children, strict=True)
        for path in _tree_leaf_paths(child, (*prefix, entry))
    )


def _path_name(path: Path) -> str:
    return ".".join(str(entry) for entry in path) or "value"


def infer_data_spec(sample: object) -> DataSpec:
    """Infer shape, dtype, backend, and device without copying array leaves."""
    _, structure = tree_flatten(sample)
    fields: list[tuple[Path, FieldSpec]] = []

    def inspect(path: Path, value: object) -> object:
        if _array_api_compat.compat.is_array_api_obj(value):
            fields.append(
                (
                    path,
                    FieldSpec(
                        name=_path_name(path),
                        shape=tuple(value.shape),
                        dtype=str(value.dtype),
                        backend=backend_of(value),
                        device=str(_array_api_compat.compat.device(value)),
                    ),
                )
            )
        return value

    tree_map_with_path(inspect, sample)
    return DataSpec(structure, tuple(fields))


def validate_sample(sample: object, spec: DataSpec) -> None:
    """Validate one sample and report the exact mismatching path."""
    observed = infer_data_spec(sample)
    if not observed.structure.is_compatible(spec.structure):
        raise errors.DataSpecError(
            "validate_sample: sample tree structure does not match DataSpec"
        )
    expected = dict(spec.fields)
    actual = dict(observed.fields)
    if expected.keys() != actual.keys():
        raise errors.DataSpecError(
            "validate_sample: array leaf paths do not match DataSpec"
        )
    for path, field in expected.items():
        current = actual[path]
        differences = tuple(
            name
            for name in ("shape", "dtype", "backend", "device")
            if getattr(field, name) != getattr(current, name)
        )
        if differences:
            raise errors.DataSpecError(
                "validate_sample: path "
                f"{path!r} mismatches fields {differences!r}; expected "
                f"{field!r}, "
                f"observed {current!r}"
            )


def validate_dataset(
    dataset: Dataset[object],
    spec: DataSpec | None = None,
    *,
    max_samples: int | None = None,
) -> DataSpec:
    """Validate a finite dataset, optionally bounding inspected samples."""
    if max_samples is not None and (
        isinstance(max_samples, bool)
        or not isinstance(max_samples, int)
        or max_samples < 0
    ):
        raise errors.DataSpecError(
            "validate_dataset: max_samples must be a non-negative integer"
        )
    length = len(dataset)
    if length == 0:
        if spec is None:
            raise errors.DataSpecError(
                "validate_dataset: cannot infer a schema from an empty dataset"
            )
        return spec
    count = length if max_samples is None else min(length, max_samples)
    if spec is None:
        if count == 0:
            raise errors.DataSpecError(
                "validate_dataset: max_samples must permit one sample when "
                "inferring a schema"
            )
        expected = infer_data_spec(dataset[0])
        first_validation_index = 1
    else:
        expected = spec
        first_validation_index = 0
    for index in range(first_validation_index, count):
        try:
            validate_sample(dataset[index], expected)
        except errors.DataSpecError as exception:
            raise errors.DataSpecError(
                f"validate_dataset: sample {index} violates the schema"
            ) from exception
    return expected


__all__ = [
    "DataSpec",
    "FieldSpec",
    "SemanticRole",
    "infer_data_spec",
    "validate_dataset",
    "validate_sample",
]
