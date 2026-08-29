# Copyright 2026 AI4SciComp contributors.
# SPDX-License-Identifier: Apache-2.0
"""Strictly typed boundary around :mod:`array_api_compat`."""

from __future__ import annotations

import typing

import array_api_compat


class _CompatibilityModule(typing.Protocol):
    """Subset of compatibility helpers required by asc-py."""

    def array_namespace(
        self, *arrays: object, api_version: str | None = None
    ) -> object:
        """Select a compatibility namespace."""
        ...  # pylint: disable=unnecessary-ellipsis

    def device(self, array: object) -> object:
        """Return an array device."""
        ...  # pylint: disable=unnecessary-ellipsis

    def is_array_api_obj(self, value: object) -> bool:
        """Return whether a value is a recognized native array."""
        ...  # pylint: disable=unnecessary-ellipsis

    def is_array_api_strict_namespace(self, namespace: object) -> bool:
        """Return whether a namespace belongs to array-api-strict."""
        ...  # pylint: disable=unnecessary-ellipsis

    def is_jax_namespace(self, namespace: object) -> bool:
        """Return whether a namespace belongs to JAX."""
        ...  # pylint: disable=unnecessary-ellipsis

    def is_numpy_namespace(self, namespace: object) -> bool:
        """Return whether a namespace belongs to NumPy."""
        ...  # pylint: disable=unnecessary-ellipsis

    def is_torch_namespace(self, namespace: object) -> bool:
        """Return whether a namespace belongs to PyTorch."""
        ...  # pylint: disable=unnecessary-ellipsis


compat = typing.cast(_CompatibilityModule, array_api_compat)
