# Copyright 2026 AI4SciComp contributors.
# SPDX-License-Identifier: Apache-2.0
"""Recursive backend-preserving conversion, collation, and uncollation."""

from __future__ import annotations

import collections
import collections.abc
import dataclasses
import typing

import numpy

from asc import _array_api_compat, errors
from asc.core import namespace as namespace_module
from asc.core.backend import Backend, backend_of
from asc.core.backend import backend as select_backend


def _required_init_variables(value: object) -> tuple[str, ...]:
    """Return required InitVar names absent from an instance."""
    persistent_fields = {field.name for field in dataclasses.fields(value)}
    return tuple(
        field.name
        for field in value.__dataclass_fields__.values()
        if field.name not in persistent_fields
        and field.init
        and field.default is dataclasses.MISSING
        and field.default_factory is dataclasses.MISSING
    )


def _reconstruct_dataclass(
    template: object, values: collections.abc.Mapping[str, object]
) -> object:
    """Restore persistent fields without rerunning dataclass initialization."""
    required_init_variables = _required_init_variables(template)
    if required_init_variables:
        raise errors.CollationError(
            "data collation: dataclasses with required InitVar parameters are "
            f"unsupported; observed {required_init_variables!r}"
        )
    result = object.__new__(type(template))
    for field in dataclasses.fields(template):
        object.__setattr__(result, field.name, values[field.name])
    return result


def _reconstruct_mapping(
    template: collections.abc.Mapping[object, object],
    items: collections.abc.Iterable[tuple[object, object]],
) -> object:
    """Rebuild a supported mapping without changing its concrete type."""
    materialized = tuple(items)
    node_type = type(template)
    if node_type is dict:
        return dict(materialized)
    try:
        if isinstance(template, collections.defaultdict):
            return node_type(template.default_factory, materialized)
        try:
            return node_type(dict(materialized))
        except TypeError:
            return node_type(materialized)
    except (TypeError, ValueError) as exception:
        raise errors.CollationError(
            "data collation: mapping type must reconstruct from its items"
        ) from exception


def _has_collatable_leaf(value: object) -> bool:
    """Return whether a supported structured sample carries cardinality."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return any(
            _has_collatable_leaf(getattr(value, field.name))
            for field in dataclasses.fields(value)
        )
    if isinstance(value, collections.abc.Mapping):
        return any(_has_collatable_leaf(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return any(_has_collatable_leaf(item) for item in value)
    return True


def default_convert(
    value: object,
    *,
    backend: typing.Literal["numpy", "torch", "jax"] | object = "numpy",
) -> object:
    """Convert Python/NumPy scalar leaves under an explicit backend."""
    selected = (
        backend
        if isinstance(backend, Backend)
        else select_backend(getattr(backend, "name", backend))
    )
    if isinstance(value, numpy.generic):
        return selected.asarray(numpy.asarray(value), copy=True)
    if _array_api_compat.compat.is_array_api_obj(value):
        observed = backend_of(value)
        if observed != selected.name:
            raise errors.MixedBackendError(
                "default_convert: native array backend "
                f"{observed!r} does not match explicit backend "
                f"{selected.name!r}; "
                "use tree_to_backend first"
            )
        return value
    if isinstance(value, (bool, int, float, complex)):
        return selected.asarray(value)
    if isinstance(value, (str, bytes, type(None))):
        return value
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _reconstruct_dataclass(
            value,
            {
                field.name: default_convert(
                    getattr(value, field.name), backend=selected
                )
                for field in dataclasses.fields(value)
            },
        )
    if isinstance(value, collections.abc.Mapping):
        return _reconstruct_mapping(
            value,
            (
                (key, default_convert(item, backend=selected))
                for key, item in value.items()
            ),
        )
    if isinstance(value, tuple):
        converted = tuple(
            default_convert(item, backend=selected) for item in value
        )
        return (
            type(value)(*converted)
            if hasattr(type(value), "_fields")
            else converted
        )
    if isinstance(value, list):
        return [default_convert(item, backend=selected) for item in value]
    return value


def _default_collate(
    samples: collections.abc.Sequence[object],
    *,
    backend: typing.Literal["numpy", "torch", "jax"] | object | None = None,
) -> object:
    """Recursively stack strict samples after the root cardinality check."""
    if not samples:
        raise errors.CollationError(
            "default_collate: samples must not be empty"
        )
    first = samples[0]
    if isinstance(first, numpy.generic):
        if not all(type(item) is type(first) for item in samples):
            raise errors.CollationError(
                "default_collate: scalar leaves must have one stable NumPy "
                "scalar type"
            )
        selected = (
            backend
            if isinstance(backend, Backend)
            else select_backend(
                "numpy"
                if backend is None
                else getattr(backend, "name", backend)
            )
        )
        return selected.asarray(numpy.asarray(tuple(samples)), copy=True)
    if _array_api_compat.compat.is_array_api_obj(first):
        if not all(
            _array_api_compat.compat.is_array_api_obj(item) for item in samples
        ):
            raise errors.CollationError(
                "default_collate: array and non-array leaves cannot be mixed"
            )
        if any(
            item.shape != first.shape or item.dtype != first.dtype
            for item in samples[1:]
        ):
            raise errors.CollationError(
                "default_collate: array shapes and dtypes must match exactly"
            )
        try:
            xp = namespace_module.array_namespace(*samples)
            if backend is not None:
                expected = getattr(backend, "name", backend)
                observed = namespace_module.identify_backend(xp)
                if observed != expected:
                    raise errors.MixedBackendError(
                        "default_collate: array backend does not match the "
                        "explicit collation backend"
                    )
            return xp.stack(tuple(samples), axis=0)
        except errors.AscError:
            raise
        except (RuntimeError, TypeError, ValueError) as exception:
            raise errors.CollationError(
                "default_collate: native arrays must share shape, dtype, "
                "and backend"
            ) from exception
    if isinstance(first, (bool, int, float, complex)):
        if not all(type(item) is type(first) for item in samples):
            raise errors.CollationError(
                "default_collate: scalar leaves must have one stable Python "
                "type"
            )
        selected = (
            backend
            if isinstance(backend, Backend)
            else select_backend(
                "numpy"
                if backend is None
                else getattr(backend, "name", backend)
            )
        )
        return selected.asarray(tuple(samples))
    if isinstance(first, (str, bytes, type(None))):
        if not all(isinstance(item, type(first)) for item in samples):
            raise errors.CollationError(
                "default_collate: metadata leaves must have matching types"
            )
        return list(samples)
    if dataclasses.is_dataclass(first) and not isinstance(first, type):
        if not all(type(item) is type(first) for item in samples):
            raise errors.CollationError(
                "default_collate: dataclass sample types must match"
            )
        return _reconstruct_dataclass(
            first,
            {
                field.name: _default_collate(
                    [getattr(item, field.name) for item in samples],
                    backend=backend,
                )
                for field in dataclasses.fields(first)
            },
        )
    if isinstance(first, collections.abc.Mapping):
        keys = tuple(first.keys())
        if not all(
            type(item) is type(first) and tuple(item.keys()) == keys
            for item in samples
        ):
            raise errors.CollationError(
                "default_collate: mapping keys and order must match"
            )
        return _reconstruct_mapping(
            first,
            (
                (
                    key,
                    _default_collate(
                        [
                            typing.cast(
                                collections.abc.Mapping[object, object], item
                            )[key]
                            for item in samples
                        ],
                        backend=backend,
                    ),
                )
                for key in keys
            ),
        )
    if isinstance(first, tuple):
        if not all(
            type(item) is type(first) and len(item) == len(first)
            for item in samples
        ):
            raise errors.CollationError(
                "default_collate: tuple types and lengths must match"
            )
        values = tuple(
            _default_collate([item[index] for item in samples], backend=backend)  # type: ignore[index]
            for index in range(len(first))
        )
        return (
            type(first)(*values) if hasattr(type(first), "_fields") else values
        )
    if isinstance(first, list):
        if not all(
            isinstance(item, list) and len(item) == len(first)
            for item in samples
        ):
            raise errors.CollationError(
                "default_collate: list sample lengths must match"
            )
        return [
            _default_collate([item[index] for item in samples], backend=backend)  # type: ignore[index]
            for index in range(len(first))
        ]
    raise errors.CollationError(
        f"default_collate: unsupported leaf type {type(first).__name__}"
    )


def default_collate(
    samples: collections.abc.Sequence[object],
    *,
    backend: typing.Literal["numpy", "torch", "jax"] | object | None = None,
) -> object:
    """Recursively stack strict, structurally matching samples."""
    if not samples:
        raise errors.CollationError(
            "default_collate: samples must not be empty"
        )
    if not _has_collatable_leaf(samples[0]):
        raise errors.CollationError(
            "default_collate: leafless samples cannot preserve batch "
            "cardinality"
        )
    return _default_collate(samples, backend=backend)


def _batch_sizes(value: object) -> list[int]:
    """Collect every recoverable batch extent in a collated tree."""
    if _array_api_compat.compat.is_array_api_obj(value):
        shape = value.shape
        if not shape:
            raise errors.CollationError(
                "uncollate: array leaves must have a leading batch dimension"
            )
        return [typing.cast(int, shape[0])]
    if isinstance(value, collections.abc.Mapping):
        return [size for item in value.values() for size in _batch_sizes(item)]
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        fields = dataclasses.fields(value)
        return [
            size
            for field in fields
            for size in _batch_sizes(getattr(value, field.name))
        ]
    if isinstance(value, tuple):
        return [size for item in value for size in _batch_sizes(item)]
    if isinstance(value, list):
        if not value:
            return []
        if all(isinstance(item, (str, bytes, type(None))) for item in value):
            return [len(value)]
        return [size for item in value for size in _batch_sizes(item)]
    raise errors.CollationError(
        f"uncollate: unsupported collated leaf {type(value).__name__}"
    )


def _extract(value: object, index: int, size: int) -> object:
    if _array_api_compat.compat.is_array_api_obj(value):
        if value.shape[0] != size:
            raise errors.CollationError(
                "uncollate: array batch dimensions must agree"
            )
        return value[index]  # type: ignore[index]
    if isinstance(value, collections.abc.Mapping):
        return _reconstruct_mapping(
            value,
            ((key, _extract(item, index, size)) for key, item in value.items()),
        )
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _reconstruct_dataclass(
            value,
            {
                field.name: _extract(getattr(value, field.name), index, size)
                for field in dataclasses.fields(value)
            },
        )
    if isinstance(value, tuple):
        items = tuple(_extract(item, index, size) for item in value)
        return type(value)(*items) if hasattr(type(value), "_fields") else items
    if isinstance(value, list):
        if not value:
            return []
        if all(isinstance(item, (str, bytes, type(None))) for item in value):
            if len(value) != size:
                raise errors.CollationError(
                    "uncollate: metadata batch lengths must agree"
                )
            return value[index]
        return [_extract(item, index, size) for item in value]
    raise errors.CollationError(
        f"uncollate: unsupported collated leaf {type(value).__name__}"
    )


def uncollate(batch: object) -> list[object]:
    """Recover samples from the supported inverse collation contract."""
    if isinstance(batch, list) and not batch:
        return []
    sizes = _batch_sizes(batch)
    if not sizes:
        raise errors.CollationError(
            "uncollate: batch tree has no recoverable batch cardinality"
        )
    if len(set(sizes)) != 1:
        raise errors.CollationError(
            "uncollate: all batch dimensions and metadata lengths must agree"
        )
    size = sizes[0]
    return [_extract(batch, index, size) for index in range(size)]


__all__ = ["default_collate", "default_convert", "uncollate"]
