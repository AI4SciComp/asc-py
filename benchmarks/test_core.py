# Copyright 2026 AI4SciComp contributors.
# SPDX-License-Identifier: Apache-2.0
"""Reproducible steady-state portable-core benchmarks."""

from __future__ import annotations

import typing

import array_api_strict
import jax.numpy
import numpy
import pytest
import torch

import asc
from asc import typing as asc_typing

_BACKENDS: typing.Final[tuple[asc_typing.BackendName, ...]] = (
    "array_api_strict",
    "jax",
    "numpy",
    "torch",
)

array_api_strict.set_array_api_strict_flags(api_version="2024.12")


class _CreationNamespace(typing.Protocol):
    """Minimal namespace creation surface needed by benchmarks."""

    float32: object

    def asarray(
        self,
        value: object,
        *,
        dtype: object,
    ) -> asc_typing.NativeArray:
        """Create one native benchmark array."""
        ...  # pylint: disable=unnecessary-ellipsis


def _float_array(
    backend: asc_typing.BackendName,
) -> asc_typing.NativeArray:
    if backend == "array_api_strict":
        namespace = array_api_strict
    elif backend == "jax":
        namespace = jax.numpy
    elif backend == "numpy":
        namespace = numpy
    else:
        namespace = torch
    selected = typing.cast(_CreationNamespace, namespace)
    return selected.asarray([1.0] * 4096, dtype=selected.float32)


class BenchmarkFixture(typing.Protocol):  # pylint: disable=too-few-public-methods
    """Subset of pytest-benchmark used by this suite."""

    def pedantic[T](
        self,
        target: typing.Callable[[], T],
        *,
        rounds: int,
        iterations: int,
        warmup_rounds: int,
    ) -> T:
        """Benchmark a zero-argument callable under fixed settings."""
        ...  # pylint: disable=unnecessary-ellipsis


def _synchronize(value: asc_typing.NativeArray) -> None:
    blocker = getattr(value, "block_until_ready", None)
    if callable(blocker):
        blocker()


@pytest.mark.parametrize("backend", _BACKENDS)
def test_sum_of_squares_steady_state(
    benchmark: BenchmarkFixture,
    backend: asc_typing.BackendName,
) -> None:
    """Measure a fixed float32 reduction after one warm-up call."""
    value = _float_array(backend)
    _synchronize(asc.sum_of_squares(value))

    def operation() -> asc_typing.NativeArray:
        result = asc.sum_of_squares(value)
        _synchronize(result)
        return result

    result = benchmark.pedantic(
        operation,
        rounds=20,
        iterations=10,
        warmup_rounds=2,
    )
    assert result.shape == ()
