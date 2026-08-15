# Copyright 2026 AI4SciComp contributors.
# SPDX-License-Identifier: Apache-2.0
"""Narrow public typing contracts used by portable code."""

from __future__ import annotations

from typing import Any, Literal, Protocol, TypeVar, runtime_checkable

ArrayT = TypeVar("ArrayT")
Axis = int | tuple[int, ...] | None
BackendName = Literal["array_api_strict", "jax", "numpy", "torch"]
Shape = tuple[int, ...]


class NativeArray(Protocol):  # pylint: disable=too-few-public-methods
    """Observable array metadata returned by public operations."""

    @property
    def dtype(self) -> object:
        """Return the native dtype."""
        ...  # pylint: disable=unnecessary-ellipsis

    @property
    def shape(self) -> Shape:
        """Return the array shape."""
        ...  # pylint: disable=unnecessary-ellipsis


@runtime_checkable
class ArrayNamespace(Protocol):
    """Minimal Array API namespace surface used by the portable core."""

    __array_api_version__: str
    __name__: str

    def __getattr__(self, name: str) -> Any:
        """Expose frozen standard members omitted from this narrow protocol."""
        ...  # pylint: disable=unnecessary-ellipsis

    def full(
        self,
        shape: Shape,
        fill_value: object,
        *,
        dtype: object | None = None,
        device: object | None = None,
    ) -> object:
        """Create an array filled with one value."""
        ...  # pylint: disable=unnecessary-ellipsis

    def isdtype(self, dtype: object, kind: str | tuple[str, ...]) -> bool:
        """Return whether a dtype belongs to a named family."""
        ...  # pylint: disable=unnecessary-ellipsis

    def square(self, value: ArrayT) -> ArrayT:
        """Square an array elementwise."""
        ...  # pylint: disable=unnecessary-ellipsis

    def sum(
        self,
        value: ArrayT,
        *,
        axis: Axis = None,
        dtype: object | None = None,
        keepdims: bool = False,
    ) -> ArrayT:
        """Sum array elements."""
        ...  # pylint: disable=unnecessary-ellipsis


class CreationContextLike(Protocol):  # pylint: disable=too-few-public-methods
    """Structural contract needed by context-based creation."""

    @property
    def namespace(self) -> ArrayNamespace:
        """Return the selected namespace."""
        ...  # pylint: disable=unnecessary-ellipsis

    @property
    def backend(self) -> BackendName:
        """Return the stable backend name."""
        ...  # pylint: disable=unnecessary-ellipsis

    @property
    def dtype(self) -> object | None:
        """Return the explicitly requested dtype, if any."""
        ...  # pylint: disable=unnecessary-ellipsis

    @property
    def device(self) -> object | None:
        """Return the explicit CPU device, if any."""
        ...  # pylint: disable=unnecessary-ellipsis


__all__ = [
    "ArrayNamespace",
    "ArrayT",
    "Axis",
    "BackendName",
    "CreationContextLike",
    "NativeArray",
    "Shape",
]
