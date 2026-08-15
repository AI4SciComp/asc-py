# Copyright 2026 AI4SciComp contributors.
# SPDX-License-Identifier: Apache-2.0
"""Immutable backend-neutral transforms and feature scalers."""

from __future__ import annotations

import collections.abc
import dataclasses
import math
import typing

from asc import _array_api_compat, errors
from asc.core import namespace as namespace_module
from asc.core._scalar import normalize_real_scalar
from asc.core.backend import backend as select_backend
from asc.tree import tree_map


class Transform(typing.Protocol):
    """Typed fit/transform protocol with functional learned state."""

    def fit(self, data: object) -> Transform:
        """Return a transform fitted to data."""
        ...  # pylint: disable=unnecessary-ellipsis

    def transform(self, data: object) -> object:
        """Transform data without modifying it."""
        ...  # pylint: disable=unnecessary-ellipsis

    def inverse_transform(self, data: object) -> object:
        """Invert a supported transformation."""
        ...  # pylint: disable=unnecessary-ellipsis

    def fit_transform(self, data: object) -> tuple[Transform, object]:
        """Fit and return both learned transform and transformed value."""
        ...  # pylint: disable=unnecessary-ellipsis


class _TransformMixin:
    def fit(self, data: object) -> Transform:
        del data
        return typing.cast(Transform, self)

    def inverse_transform(self, data: object) -> object:
        raise errors.CapabilityNotSupportedError(
            f"{type(self).__name__}.inverse_transform: inverse is not available"
        )

    def fit_transform(self, data: object) -> tuple[Transform, object]:
        fitted = self.fit(data)
        return fitted, fitted.transform(data)


@dataclasses.dataclass(frozen=True, slots=True)
class Identity(_TransformMixin):
    """Return every value unchanged."""

    def transform(self, data: object) -> object:
        """Return data unchanged without copying or mutating it."""
        return data

    def inverse_transform(self, data: object) -> object:
        """Return data unchanged without copying or mutating it."""
        return data


@dataclasses.dataclass(frozen=True, slots=True)
class LambdaTransform(_TransformMixin):
    """Wrap explicit transform and optional inverse callables."""

    function: typing.Callable[[object], object]
    inverse: typing.Callable[[object], object] | None = None

    def transform(self, data: object) -> object:
        """Apply the configured callable without changing its result."""
        return self.function(data)

    def inverse_transform(self, data: object) -> object:
        """Apply the inverse callable or raise a capability error."""
        if self.inverse is None:
            return _TransformMixin.inverse_transform(self, data)
        return self.inverse(data)


@dataclasses.dataclass(frozen=True, slots=True)
class Compose(_TransformMixin):
    """Apply a fixed tuple of transforms in order."""

    transforms: tuple[Transform, ...]

    def __init__(self, transforms: collections.abc.Iterable[Transform]) -> None:
        values = tuple(transforms)
        if any(
            any(
                not callable(getattr(value, method, None))
                for method in (
                    "fit",
                    "transform",
                    "inverse_transform",
                    "fit_transform",
                )
            )
            for value in values
        ):
            raise errors.DataSpecError(
                "Compose: every item must implement the Transform protocol"
            )
        object.__setattr__(self, "transforms", values)

    def fit(self, data: object) -> Transform:
        """Fit each transform against the preceding transformed output."""
        fitted: list[Transform] = []
        current = data
        for transform in self.transforms:
            learned = transform.fit(current)
            fitted.append(learned)
            current = learned.transform(current)
        return Compose(fitted)

    def transform(self, data: object) -> object:
        """Apply every transform in declared order without mutation."""
        result = data
        for transform in self.transforms:
            result = transform.transform(result)
        return result

    def inverse_transform(self, data: object) -> object:
        """Apply every inverse transform in reverse declared order."""
        result = data
        for transform in reversed(self.transforms):
            result = transform.inverse_transform(result)
        return result


@dataclasses.dataclass(frozen=True, slots=True)
class SelectFields(_TransformMixin):
    """Select named mapping fields in an explicit stable order."""

    fields: tuple[str, ...]

    def __init__(self, fields: collections.abc.Iterable[str]) -> None:
        values = tuple(fields)
        if (
            not values
            or len(set(values)) != len(values)
            or any(not isinstance(field, str) or not field for field in values)
        ):
            raise errors.DataSpecError(
                "SelectFields: fields must be unique non-empty strings"
            )
        object.__setattr__(self, "fields", values)

    def transform(self, data: object) -> object:
        """Return selected mapping fields in stable configured order."""
        if not isinstance(data, collections.abc.Mapping):
            raise errors.DataSpecError("SelectFields: data must be a mapping")
        missing = [field for field in self.fields if field not in data]
        if missing:
            raise errors.DataSpecError(
                f"SelectFields: missing fields {missing!r}"
            )
        return {field: data[field] for field in self.fields}


@dataclasses.dataclass(frozen=True, slots=True)
class RenameFields(_TransformMixin):
    """Rename mapping fields without mutating the source mapping."""

    names: tuple[tuple[str, str], ...]

    def __init__(self, names: collections.abc.Mapping[str, str]) -> None:
        pairs = tuple(names.items())
        if (
            not pairs
            or any(
                not isinstance(source, str)
                or not source
                or not isinstance(target, str)
                or not target
                for source, target in pairs
            )
            or len({target for _, target in pairs}) != len(pairs)
        ):
            raise errors.DataSpecError(
                "RenameFields: names must be non-empty and destinations unique"
            )
        object.__setattr__(self, "names", pairs)

    def transform(self, data: object) -> object:
        """Return a mapping with configured field names replaced."""
        if not isinstance(data, collections.abc.Mapping):
            raise errors.DataSpecError("RenameFields: data must be a mapping")
        mapping = dict(self.names)
        if any(source not in data for source in mapping):
            raise errors.DataSpecError(
                "RenameFields: every source field must exist"
            )
        output_names = tuple(mapping.get(name, name) for name in data)
        if len(set(output_names)) != len(output_names):
            raise errors.DataSpecError(
                "RenameFields: renamed fields collide with retained fields"
            )
        return {mapping.get(name, name): value for name, value in data.items()}

    def inverse_transform(self, data: object) -> object:
        """Reverse configured field names without mutating the mapping."""
        return RenameFields(
            {target: source for source, target in self.names}
        ).transform(data)


@dataclasses.dataclass(frozen=True, slots=True)
class ToBackend(_TransformMixin):
    """Explicitly convert every native array leaf to one backend."""

    destination: object
    dtype: object | None = None
    device: object | None = "cpu"
    copy: bool | None = True

    def transform(self, data: object) -> object:
        """Convert every native leaf under explicit copy and device policy."""
        from asc.tree import tree_to_backend

        return tree_to_backend(
            data,
            self.destination,
            dtype=self.dtype,
            device=self.device,
            copy=self.copy,
        )


@dataclasses.dataclass(frozen=True, slots=True)
class ToDevice(_TransformMixin):
    """Explicitly move every native array leaf within its backend."""

    device: object
    copy: bool | None = None

    def transform(self, data: object) -> object:
        """Move native leaves within their backends under explicit policy."""
        from asc.tree import tree_to_device

        return tree_to_device(data, self.device, copy=self.copy)


@dataclasses.dataclass(frozen=True, slots=True)
class CastDType(_TransformMixin):
    """Cast every native array leaf within its existing backend."""

    dtype: object
    copy: bool = True

    def transform(self, data: object) -> object:
        """Cast native leaves and preserve non-array leaves and structure."""

        def cast(value: object) -> object:
            if not _array_api_compat.compat.is_array_api_obj(value):
                return value
            xp = namespace_module.array_namespace(value)
            name = namespace_module.identify_backend(xp)
            if name != "array_api_strict":
                select_backend(name, dtype=self.dtype)
            return xp.astype(value, self.dtype, copy=self.copy)

        return tree_map(cast, data)


def _normalized_reduction_axes(
    axis: int | tuple[int, ...] | None,
    ndim: int,
    operation: str,
) -> tuple[int, ...]:
    """Normalize fitted reduction axes for later statistic alignment."""
    if axis is None:
        raw_axes = tuple(range(ndim))
    elif isinstance(axis, int) and not isinstance(axis, bool):
        raw_axes = (axis,)
    else:
        raw_axes = axis
    if not isinstance(raw_axes, tuple) or any(
        isinstance(item, bool) or not isinstance(item, int) for item in raw_axes
    ):
        raise errors.DataSpecError(f"{operation}: axis is invalid")
    normalized = tuple(item + ndim if item < 0 else item for item in raw_axes)
    if any(item < 0 or item >= ndim for item in normalized) or len(
        set(normalized)
    ) != len(normalized):
        raise errors.DataSpecError(f"{operation}: axis is invalid")
    return normalized


def _nonempty_reduction_axes(
    data: object,
    axis: int | tuple[int, ...] | None,
    operation: str,
) -> tuple[int, ...]:
    """Normalize axes and reject reductions over an empty extent."""
    axes = _normalized_reduction_axes(axis, len(data.shape), operation)
    if any(data.shape[item] == 0 for item in axes):
        raise errors.DataSpecError(
            f"{operation}: reduction axes must have nonzero extents"
        )
    return axes


def _aligned_statistic(
    data: object,
    statistic: object,
    axis: int | tuple[int, ...] | None,
    fitted_ndim: int | None,
    operation: str,
) -> object:
    """Align one reduced statistic without changing the input shape."""
    if fitted_ndim is None:
        raise errors.DataSpecError(
            f"{operation}: fitted input-rank metadata is unavailable"
        )
    xp = namespace_module.array_namespace(data, statistic)
    data_shape = tuple(data.shape)
    statistic_shape = tuple(statistic.shape)
    axes = set(_normalized_reduction_axes(axis, fitted_ndim, operation))
    expected_rank = fitted_ndim - len(axes)
    if len(statistic_shape) != expected_rank:
        raise errors.DataSpecError(
            f"{operation}: fitted state has an invalid shape"
        )
    if len(data_shape) == expected_rank and len(data_shape) != fitted_ndim:
        if data_shape != statistic_shape:
            raise errors.DataSpecError(
                f"{operation}: input shape is incompatible with fitted state"
            )
        aligned = statistic
    elif len(data_shape) == fitted_ndim:
        position = 0
        target_shape: list[int] = []
        for dimension in range(fitted_ndim):
            if dimension in axes:
                target_shape.append(1)
            else:
                statistic_extent = statistic_shape[position]
                if data_shape[dimension] != statistic_extent:
                    raise errors.DataSpecError(
                        f"{operation}: input shape is incompatible with fitted "
                        "state"
                    )
                target_shape.append(statistic_extent)
                position += 1
        aligned = xp.reshape(statistic, tuple(target_shape))
    else:
        raise errors.DataSpecError(
            f"{operation}: input shape is incompatible with fitted state"
        )
    try:
        return xp.broadcast_to(aligned, data_shape)
    except (RuntimeError, TypeError, ValueError) as exception:
        raise errors.DataSpecError(
            f"{operation}: input shape is incompatible with fitted state"
        ) from exception


def _require_real_floating_data(
    xp: object, data: object, operation: str
) -> None:
    """Reject scaler arithmetic outside the portable floating domain."""
    if not xp.isdtype(data.dtype, "real floating"):
        raise errors.DTypeError(f"{operation}: data must be real floating")


@dataclasses.dataclass(frozen=True, slots=True)
class StandardScaler(_TransformMixin):
    """Per-feature standardization with zero-variance scale set to one."""

    axis: int | tuple[int, ...] | None = 0
    mean_: object | None = None
    scale_: object | None = None
    fitted_ndim_: int | None = None
    magnitude_: object | None = None
    constant_: object | None = None

    def fit(self, data: object) -> Transform:
        """Return a fitted scaler with backend-native mean and scale."""
        xp = namespace_module.array_namespace(data)
        if not xp.isdtype(data.dtype, "real floating"):
            raise errors.DTypeError(
                "StandardScaler.fit: data must be real floating"
            )
        _nonempty_reduction_axes(data, self.axis, "StandardScaler.fit")
        calculation = (
            xp.astype(data, xp.float32, copy=True)
            if int(xp.finfo(data.dtype).bits) < 32
            else data
        )
        magnitude = xp.max(xp.abs(calculation), axis=self.axis, keepdims=False)
        magnitude = xp.where(magnitude == 0, xp.ones_like(magnitude), magnitude)
        aligned_magnitude = _aligned_statistic(
            calculation,
            magnitude,
            self.axis,
            len(calculation.shape),
            "StandardScaler.fit",
        )
        normalized = calculation / aligned_magnitude
        normalized_mean = xp.mean(normalized, axis=self.axis, keepdims=False)
        normalized_scale = xp.std(normalized, axis=self.axis, keepdims=False)
        mean = normalized_mean * magnitude
        scale = normalized_scale * magnitude
        constant = scale == xp.zeros_like(scale)
        scale = xp.where(constant, xp.ones_like(scale), scale)
        return StandardScaler(
            axis=self.axis,
            mean_=mean,
            scale_=scale,
            fitted_ndim_=len(calculation.shape),
            magnitude_=magnitude,
            constant_=constant,
        )

    def _state(self) -> tuple[object, object, object, object]:
        if (
            self.mean_ is None
            or self.scale_ is None
            or self.magnitude_ is None
            or self.constant_ is None
        ):
            raise errors.DataSpecError(
                "StandardScaler: call fit and use the returned scaler first"
            )
        return self.mean_, self.scale_, self.magnitude_, self.constant_

    def transform(self, data: object) -> object:
        """Standardize data without mutation using fitted native state."""
        mean, scale, magnitude, constant = self._state()
        xp = namespace_module.array_namespace(
            data, mean, scale, magnitude, constant
        )
        _require_real_floating_data(xp, data, "StandardScaler.transform")
        mean = _aligned_statistic(
            data,
            mean,
            self.axis,
            self.fitted_ndim_,
            "StandardScaler.transform",
        )
        scale = _aligned_statistic(
            data,
            scale,
            self.axis,
            self.fitted_ndim_,
            "StandardScaler.transform",
        )
        magnitude = _aligned_statistic(
            data,
            magnitude,
            self.axis,
            self.fitted_ndim_,
            "StandardScaler.transform",
        )
        constant = _aligned_statistic(
            data,
            constant,
            self.axis,
            self.fitted_ndim_,
            "StandardScaler.transform",
        )
        safe_magnitude = xp.where(constant, xp.ones_like(magnitude), magnitude)
        safe_scale = xp.where(constant, xp.ones_like(scale), scale)
        return (data / safe_magnitude - mean / safe_magnitude) / (
            safe_scale / safe_magnitude
        )

    def inverse_transform(self, data: object) -> object:
        """Restore standardized data using fitted native state."""
        mean, scale, magnitude, constant = self._state()
        xp = namespace_module.array_namespace(
            data, mean, scale, magnitude, constant
        )
        _require_real_floating_data(
            xp, data, "StandardScaler.inverse_transform"
        )
        mean = _aligned_statistic(
            data,
            mean,
            self.axis,
            self.fitted_ndim_,
            "StandardScaler.inverse_transform",
        )
        scale = _aligned_statistic(
            data,
            scale,
            self.axis,
            self.fitted_ndim_,
            "StandardScaler.inverse_transform",
        )
        magnitude = _aligned_statistic(
            data,
            magnitude,
            self.axis,
            self.fitted_ndim_,
            "StandardScaler.inverse_transform",
        )
        constant = _aligned_statistic(
            data,
            constant,
            self.axis,
            self.fitted_ndim_,
            "StandardScaler.inverse_transform",
        )
        safe_magnitude = xp.where(constant, xp.ones_like(magnitude), magnitude)
        safe_scale = xp.where(constant, xp.ones_like(scale), scale)
        normalized = (
            data * (safe_scale / safe_magnitude) + mean / safe_magnitude
        )
        return normalized * safe_magnitude


@dataclasses.dataclass(frozen=True, slots=True)
class MinMaxScaler(_TransformMixin):
    """Per-feature affine scaling with a stable constant-feature policy."""

    feature_range: tuple[float, float] = (0.0, 1.0)
    axis: int | tuple[int, ...] | None = 0
    minimum_: object | None = None
    span_: object | None = None
    fitted_ndim_: int | None = None
    magnitude_: object | None = None

    def __post_init__(self) -> None:
        try:
            low, high = self.feature_range
            if (
                isinstance(low, bool)
                or isinstance(high, bool)
                or not isinstance(low, (int, float))
                or not isinstance(high, (int, float))
            ):
                raise TypeError
            low = float(low)
            high = float(high)
        except (OverflowError, TypeError, ValueError) as exception:
            raise errors.DataSpecError(
                "MinMaxScaler: feature_range must satisfy low < high"
            ) from exception
        if not math.isfinite(low) or not math.isfinite(high) or low >= high:
            raise errors.DataSpecError(
                "MinMaxScaler: feature_range must satisfy low < high"
            )
        object.__setattr__(self, "feature_range", (low, high))

    def fit(self, data: object) -> Transform:
        """Return a fitted scaler with backend-native minimum and span."""
        xp = namespace_module.array_namespace(data)
        if not xp.isdtype(data.dtype, "real floating"):
            raise errors.DTypeError(
                "MinMaxScaler.fit: data must be real floating"
            )
        device = _array_api_compat.compat.device(data)
        try:
            low = normalize_real_scalar(
                xp,
                data.dtype,
                self.feature_range[0],
                "MinMaxScaler.fit",
                "feature_range low",
                device=device,
            )
            high = normalize_real_scalar(
                xp,
                data.dtype,
                self.feature_range[1],
                "MinMaxScaler.fit",
                "feature_range high",
                device=device,
            )
            normalize_real_scalar(
                xp,
                data.dtype,
                high - low,
                "MinMaxScaler.fit",
                "feature_range width",
                device=device,
            )
        except errors.DTypeError as exception:
            raise errors.DataSpecError(
                "MinMaxScaler.fit: feature_range is not representable in "
                "the fitted dtype"
            ) from exception
        if low >= high:
            raise errors.DataSpecError(
                "MinMaxScaler.fit: feature_range collapses in the fitted dtype"
            )
        _nonempty_reduction_axes(data, self.axis, "MinMaxScaler.fit")
        minimum = xp.min(data, axis=self.axis, keepdims=False)
        maximum = xp.max(data, axis=self.axis, keepdims=False)
        magnitude = xp.maximum(xp.abs(minimum), xp.abs(maximum))
        magnitude = xp.where(magnitude == 0, xp.ones_like(magnitude), magnitude)
        span = maximum / magnitude - minimum / magnitude
        span = xp.where(span == 0, xp.ones_like(span), span)
        return MinMaxScaler(
            feature_range=(low, high),
            axis=self.axis,
            minimum_=minimum,
            span_=span,
            magnitude_=magnitude,
            fitted_ndim_=len(data.shape),
        )

    def _state(self) -> tuple[object, object, object]:
        if (
            self.minimum_ is None
            or self.span_ is None
            or self.magnitude_ is None
        ):
            raise errors.DataSpecError(
                "MinMaxScaler: call fit and use the returned scaler first"
            )
        return self.minimum_, self.span_, self.magnitude_

    def transform(self, data: object) -> object:
        """Scale data into the feature range using fitted native state."""
        minimum, span, magnitude = self._state()
        xp = namespace_module.array_namespace(data, minimum, span, magnitude)
        _require_real_floating_data(xp, data, "MinMaxScaler.transform")
        minimum = _aligned_statistic(
            data,
            minimum,
            self.axis,
            self.fitted_ndim_,
            "MinMaxScaler.transform",
        )
        span = _aligned_statistic(
            data,
            span,
            self.axis,
            self.fitted_ndim_,
            "MinMaxScaler.transform",
        )
        magnitude = _aligned_statistic(
            data,
            magnitude,
            self.axis,
            self.fitted_ndim_,
            "MinMaxScaler.transform",
        )
        low, high = self.feature_range
        normalized = data / magnitude - minimum / magnitude
        return low + (normalized / span) * (high - low)

    def inverse_transform(self, data: object) -> object:
        """Restore scaled data from the configured feature range."""
        minimum, span, magnitude = self._state()
        xp = namespace_module.array_namespace(data, minimum, span, magnitude)
        _require_real_floating_data(xp, data, "MinMaxScaler.inverse_transform")
        minimum = _aligned_statistic(
            data,
            minimum,
            self.axis,
            self.fitted_ndim_,
            "MinMaxScaler.inverse_transform",
        )
        span = _aligned_statistic(
            data,
            span,
            self.axis,
            self.fitted_ndim_,
            "MinMaxScaler.inverse_transform",
        )
        magnitude = _aligned_statistic(
            data,
            magnitude,
            self.axis,
            self.fitted_ndim_,
            "MinMaxScaler.inverse_transform",
        )
        low, high = self.feature_range
        normalized_minimum = minimum / magnitude
        normalized = normalized_minimum + (data - low) * (span / (high - low))
        return normalized * magnitude


__all__ = [
    "CastDType",
    "Compose",
    "Identity",
    "LambdaTransform",
    "MinMaxScaler",
    "RenameFields",
    "SelectFields",
    "StandardScaler",
    "ToBackend",
    "ToDevice",
    "Transform",
]
