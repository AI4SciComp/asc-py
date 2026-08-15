# Copyright 2026 AI4SciComp contributors.
# SPDX-License-Identifier: Apache-2.0
"""Portable Fourier namespace facade."""

from __future__ import annotations

import collections.abc
import functools
import math
import typing

from asc import _array_api_compat, errors
from asc import typing as asc_typing
from asc.backends import _namespace as backend_namespace
from asc.core import namespace as namespace_module
from asc.core._indexing import safe_index_dtype
from asc.core._scalar import normalize_real_scalar
from asc.extensions import _dispatch

_ONE_DIMENSIONAL_TRANSFORMS: typing.Final = frozenset(
    {"fft", "hfft", "ifft", "ihfft", "irfft", "rfft"}
)
_N_DIMENSIONAL_TRANSFORMS: typing.Final = frozenset(
    {"fftn", "ifftn", "irfftn", "rfftn"}
)
_SHIFT_TRANSFORMS: typing.Final = frozenset({"fftshift", "ifftshift"})
_REAL_INPUT_TRANSFORMS: typing.Final = frozenset({"ihfft", "rfft", "rfftn"})
_COMPLEX_INPUT_TRANSFORMS: typing.Final = frozenset(
    {"fft", "fftn", "hfft", "ifft", "ifftn", "irfft", "irfftn"}
)
_STANDARD_FFT_FUNCTIONS: typing.Final = (
    _ONE_DIMENSIONAL_TRANSFORMS
    | _N_DIMENSIONAL_TRANSFORMS
    | _SHIFT_TRANSFORMS
    | {"fftfreq", "rfftfreq"}
)
_FFT_KEYWORDS: typing.Final = {
    **{
        name: frozenset({"n", "axis", "norm"})
        for name in _ONE_DIMENSIONAL_TRANSFORMS
    },
    **{
        name: frozenset({"s", "axes", "norm"})
        for name in _N_DIMENSIONAL_TRANSFORMS
    },
    **{name: frozenset({"axes"}) for name in _SHIFT_TRANSFORMS},
}


def _integer_sequence(value: object, *, allow_integer: bool) -> bool:
    """Return whether a value is an exact FFT integer control."""
    if allow_integer and type(value) is int:
        return True
    return (
        isinstance(value, collections.abc.Sequence)
        and not isinstance(value, (str, bytes, bytearray))
        and all(type(item) is int for item in value)
    )


def _validate_fft_controls(
    name: str,
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> None:
    """Validate the frozen FFT signature before native dispatch."""
    if name not in _FFT_KEYWORDS:
        raise errors.CapabilityNotSupportedError(
            f"fft.{name}: selected symbol is outside the frozen FFT surface"
        )
    if len(args) != 1:
        raise TypeError(f"fft.{name}: exactly one native array is required")
    unexpected = set(kwargs).difference(_FFT_KEYWORDS[name])
    if unexpected:
        parameter = sorted(unexpected)[0]
        raise TypeError(
            f"fft.{name}: {parameter} is outside the frozen FFT signature"
        )
    if name in _ONE_DIMENSIONAL_TRANSFORMS:
        count = kwargs.get("n")
        if count is not None and type(count) is not int:
            raise TypeError(f"fft.{name}: n must be a Python integer or None")
        if count is not None and count <= 0:
            raise ValueError(f"fft.{name}: n must be strictly positive")
        axis = kwargs.get("axis", -1)
        if type(axis) is not int:
            raise TypeError(f"fft.{name}: axis must be a Python integer")
    elif name in _N_DIMENSIONAL_TRANSFORMS:
        shape = kwargs.get("s")
        if shape is not None and not _integer_sequence(
            shape, allow_integer=False
        ):
            raise TypeError(
                f"fft.{name}: s must be an integer sequence or None"
            )
        if shape is not None:
            validated_shape = typing.cast(collections.abc.Sequence[int], shape)
            if not validated_shape or any(
                count <= 0 for count in validated_shape
            ):
                raise ValueError(
                    f"fft.{name}: every transform length in s must be strictly "
                    "positive"
                )
        axes = kwargs.get("axes")
        if axes is not None and not _integer_sequence(
            axes, allow_integer=False
        ):
            raise TypeError(
                f"fft.{name}: axes must be an integer sequence or None"
            )
    else:
        axes = kwargs.get("axes")
        if axes is not None and not _integer_sequence(axes, allow_integer=True):
            raise TypeError(
                f"fft.{name}: axes must be an integer, integer sequence, or "
                "None"
            )
    if "norm" in kwargs and (
        type(kwargs["norm"]) is not str
        or kwargs["norm"] not in {"backward", "ortho", "forward"}
    ):
        raise TypeError(
            f"fft.{name}: norm must be 'backward', 'ortho', or 'forward'"
        )


class FFTNamespace:
    """Delegate the frozen FFT extension for one immutable backend."""

    def __init__(
        self,
        native: object,
        *,
        namespace: asc_typing.ArrayNamespace,
        backend: asc_typing.BackendName,
        device: object | None,
        dtype: object | None,
        default_dtype: object,
        dtype_validator: typing.Callable[[object], None] | None = None,
        device_resolver: typing.Callable[[object | None], object | None]
        | None = None,
    ) -> None:
        self._native = native
        self._namespace = namespace
        self._backend = backend
        self._device = device
        self._dtype = dtype
        self._default_dtype = default_dtype
        self._dtype_validator = dtype_validator
        self._device_resolver = device_resolver

    def _frequency_dtype(self, dtype: object | None) -> object:
        """Return a validated real dtype for a frequency constructor."""
        effective_dtype = self._dtype if dtype is None else dtype
        if effective_dtype is None:
            effective_dtype = self._default_dtype
        if self._dtype_validator is not None:
            self._dtype_validator(effective_dtype)
        try:
            is_real = self._namespace.isdtype(effective_dtype, "real floating")
        except (AttributeError, TypeError, ValueError) as exception:
            raise errors.DTypeError(
                "fft frequency dtype does not belong to the selected backend"
            ) from exception
        if not is_real:
            raise errors.DTypeError(
                "fft frequency dtype must be a real floating dtype"
            )
        return effective_dtype

    def _frequency_device(self, device: object | None) -> object | None:
        """Return a normalized CPU device for a frequency constructor."""
        effective_device = self._device if device is None else device
        if self._device_resolver is not None:
            return self._device_resolver(effective_device)
        return effective_device

    @staticmethod
    def _frequency_count(value: object) -> int:
        """Require a positive non-Boolean Python sample count."""
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError("fft frequency count must be a positive integer")
        return value

    def _frequency_scale(
        self,
        value: object,
        count: int,
        dtype: object,
        device: object | None,
    ) -> float:
        """Return a representable scale without intermediate overflow."""
        if (
            _array_api_compat.compat.is_array_api_obj(value)
            or isinstance(value, bool)
            or not isinstance(value, (int, float))
        ):
            raise ValueError(
                "fft frequency spacing must be a Python real scalar"
            )
        try:
            spacing = float(value)
        except OverflowError as exception:
            raise ValueError(
                "fft frequency spacing must be finite and nonzero"
            ) from exception
        if not math.isfinite(spacing) or spacing == 0.0:
            raise ValueError("fft frequency spacing must be finite and nonzero")
        try:
            rounded = normalize_real_scalar(
                typing.cast(asc_typing.ArrayNamespace, self._namespace),
                dtype,
                spacing,
                "fft frequency",
                "spacing",
                device=device,
            )
            scale = (1.0 / count) / rounded
            normalize_real_scalar(
                typing.cast(asc_typing.ArrayNamespace, self._namespace),
                dtype,
                scale,
                "fft frequency",
                "reciprocal spacing scale",
                device=device,
            )
        except errors.DTypeError as exception:
            raise ValueError(
                "fft frequency spacing is not representable in the requested "
                "dtype"
            ) from exception
        return scale

    def _frequency_values(
        self,
        start: int,
        stop: int,
        *,
        dtype: object,
        device: object | None,
        scale: float,
    ) -> object:
        """Scale wide integer bins before narrowing to the output dtype."""
        maximum = max(abs(start), abs(stop - 1), 0)
        dtype_maximum = float(self._namespace.finfo(dtype).max)
        if maximum and abs(scale) > dtype_maximum / maximum:
            raise ValueError(
                "fft frequency bins are not representable in the requested "
                "dtype"
            )
        index_dtype = safe_index_dtype(
            self._namespace,
            self._backend,
            maximum,
            "fft frequency",
        )
        indices = self._namespace.arange(
            start,
            stop,
            dtype=index_dtype,
            device=device,
        )
        computation_dtype = (
            self._namespace.float32
            if int(self._namespace.finfo(dtype).bits) < 32
            else dtype
        )
        scaled = (
            self._namespace.astype(indices, computation_dtype, copy=True)
            * scale
        )
        return self._namespace.astype(scaled, dtype, copy=True)

    def __getattr__(self, name: str) -> object:
        """Return a native 2024.12 FFT function."""
        if name not in _STANDARD_FFT_FUNCTIONS:
            raise errors.CapabilityNotSupportedError(
                f"fft.{name}: selected symbol is outside the frozen FFT surface"
            )
        try:
            function = getattr(self._native, name)
        except AttributeError as exception:
            raise errors.CapabilityNotSupportedError(
                f"fft.{name}: selected backend does not provide this capability"
            ) from exception
        if not callable(function):
            return function

        @functools.wraps(function)
        def checked(*args: object, **kwargs: object) -> object:
            from asc.tree import tree_leaves

            _validate_fft_controls(name, args, kwargs)
            if not _array_api_compat.compat.is_array_api_obj(args[0]):
                raise errors.NamespaceError(
                    f"fft.{name}: the first operand must be a native array"
                )
            arrays = [
                leaf
                for leaf in tree_leaves((args, kwargs))
                if _array_api_compat.compat.is_array_api_obj(leaf)
            ]
            if not arrays:
                raise errors.NamespaceError(
                    f"fft.{name}: at least one native array is required"
                )
            xp = namespace_module.array_namespace(*arrays)
            observed = namespace_module.identify_backend(xp)
            if observed != self._backend:
                raise errors.MixedBackendError(
                    f"fft.{name}: the {self._backend!r} facade cannot consume "
                    f"{observed!r} arrays; convert explicitly"
                )
            category = (
                "real floating"
                if name in _REAL_INPUT_TRANSFORMS
                else "complex floating"
                if name in _COMPLEX_INPUT_TRANSFORMS
                else "floating"
            )
            dtype_name = backend_namespace.require_dtype_category(
                f"fft.{name}", args[0], category
            )
            if (
                name in _REAL_INPUT_TRANSFORMS
                and observed in {"jax", "torch"}
                and dtype_name in {"bfloat16", "float16"}
            ):
                raise errors.CapabilityNotSupportedError(
                    f"fft.{name}: low-precision CPU kernel is unavailable "
                    f"for {dtype_name}"
                )
            return function(*args, **kwargs)

        return checked

    def fftfreq(
        self,
        n: int,
        /,
        *,
        d: float = 1.0,
        device: object | None = None,
        dtype: object | None = None,
    ) -> object:
        """Return frequency bins on an explicit backend device and dtype."""
        count = self._frequency_count(n)
        effective_dtype = self._frequency_dtype(dtype)
        effective_device = self._frequency_device(device)
        scale = self._frequency_scale(
            d, count, effective_dtype, effective_device
        )
        positive_stop = (count - 1) // 2 + 1
        positive = self._frequency_values(
            0,
            positive_stop,
            dtype=effective_dtype,
            device=effective_device,
            scale=scale,
        )
        negative = self._frequency_values(
            -(count // 2),
            0,
            dtype=effective_dtype,
            device=effective_device,
            scale=scale,
        )
        return self._namespace.concat((positive, negative), axis=0)

    def rfftfreq(
        self,
        n: int,
        /,
        *,
        d: float = 1.0,
        device: object | None = None,
        dtype: object | None = None,
    ) -> object:
        """Return real-transform frequency bins on the selected backend."""
        count = self._frequency_count(n)
        effective_dtype = self._frequency_dtype(dtype)
        effective_device = self._frequency_device(device)
        scale = self._frequency_scale(
            d, count, effective_dtype, effective_device
        )
        return self._frequency_values(
            0,
            count // 2 + 1,
            dtype=effective_dtype,
            device=effective_device,
            scale=scale,
        )


def fft_namespace(selected_backend: object) -> FFTNamespace:
    """Return the normalized FFT namespace for a :class:`Backend`."""
    xp = selected_backend.xp
    candidate = xp.fft
    if isinstance(candidate, FFTNamespace):
        # Reconfigure the owning module's facade without exposing its raw
        # namespace through the public surface.
        # pylint: disable-next=protected-access
        candidate = candidate._native  # pyright: ignore[reportPrivateUsage]
    adapter = _dispatch.load_backend(selected_backend.name)

    def validate_dtype(dtype: object) -> None:
        adapter.validate_dtype(dtype)
        try:
            is_real = xp.isdtype(dtype, "real floating")
        except (AttributeError, TypeError, ValueError) as exception:
            raise errors.DTypeError(
                "fft frequency dtype does not belong to the selected backend"
            ) from exception
        if not is_real:
            raise errors.DTypeError(
                "fft frequency dtype must be a real floating dtype"
            )

    return FFTNamespace(
        candidate,
        namespace=xp,
        backend=selected_backend.name,
        device=selected_backend.device,
        dtype=selected_backend.dtype,
        default_dtype=xp.asarray(0.0, device=selected_backend.device).dtype,
        dtype_validator=validate_dtype,
        device_resolver=adapter.resolve_device,
    )


__all__ = ["FFTNamespace", "fft_namespace"]
