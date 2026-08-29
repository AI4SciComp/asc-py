# Copyright 2026 AI4SciComp contributors.
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import typing

import numpy
import pytest

import asc
from asc import typing as asc_typing
from asc.extensions import random
from tests import helpers

RANDOM_BACKENDS: tuple[asc_typing.BackendName, ...] = helpers.NATIVE_BACKENDS


class _JaxConfig(typing.Protocol):  # pylint: disable=too-few-public-methods
    """Typed configuration query used by random tests."""

    def read(self, name: str) -> bool:
        """Read a named JAX configuration flag."""
        ...  # pylint: disable=unnecessary-ellipsis


class _JaxModule(typing.Protocol):  # pylint: disable=too-few-public-methods
    """Typed JAX device query used by random tests."""

    def devices(self, backend: str | None = None) -> list[object]:
        """Return devices for one backend."""
        ...  # pylint: disable=unnecessary-ellipsis


class _DeviceArray(typing.Protocol):  # pylint: disable=too-few-public-methods
    """Array exposing its current device."""

    @property
    def device(self) -> object:
        """Return the array device."""
        ...  # pylint: disable=unnecessary-ellipsis


def _set_attribute(value: object, name: str, replacement: object) -> None:
    setattr(value, name, replacement)


@pytest.mark.parametrize("backend", RANDOM_BACKENDS)
def test_random_replay_progression_and_bounds(
    backend: asc_typing.BackendName,
) -> None:
    selected = helpers.namespace(backend)
    context = asc.CreationContext(
        selected,
        backend,
        dtype=selected.float32,
    )
    initial = random.create_key(42, context=context)

    first, advanced = random.uniform(
        (4, 3),
        key=initial,
        low=-2.0,
        high=3.0,
        dtype=selected.float32,
    )
    replay, _ = random.uniform(
        (4, 3),
        key=initial,
        low=-2.0,
        high=3.0,
        dtype=selected.float32,
    )
    second, _ = random.uniform(
        (4, 3),
        key=advanced,
        low=-2.0,
        high=3.0,
        dtype=selected.float32,
    )

    first_host = helpers.as_numpy(first)
    numpy.testing.assert_array_equal(first_host, helpers.as_numpy(replay))
    assert not numpy.array_equal(first_host, helpers.as_numpy(second))
    assert numpy.all(first_host >= -2.0)
    assert numpy.all(first_host < 3.0)
    assert first.shape == (4, 3)
    assert asc.backend_info(first).name == backend
    if dataclasses.is_dataclass(initial):
        with pytest.raises(dataclasses.FrozenInstanceError):
            _set_attribute(initial, "counter", 9)


def test_random_rejects_strict_invalid_seed_shape_bounds_and_key() -> None:
    selected = helpers.namespace("array_api_strict")
    context = asc.CreationContext(selected, "array_api_strict")
    with pytest.raises(
        asc.UnsupportedCapabilityError, match="does not provide"
    ):
        random.create_key(1, context=context)

    numpy_context = asc.CreationContext(
        helpers.namespace("numpy"),
        "numpy",
    )
    with pytest.raises(asc.RandomStateError, match="non-negative"):
        random.create_key(-1, context=numpy_context)
    with pytest.raises(asc.RandomStateError, match="non-negative"):
        random.create_key(True, context=numpy_context)
    key = random.create_key(1, context=numpy_context)
    with pytest.raises(asc.RandomStateError, match="shape"):
        random.uniform((-1,), key=key)
    with pytest.raises(asc.RandomStateError, match="shape"):
        random.uniform(
            typing.cast(asc_typing.Shape, [1]),
            key=key,
        )
    with pytest.raises(asc.RandomStateError, match="strictly less"):
        random.uniform((1,), key=key, low=1.0, high=1.0)
    with pytest.raises(asc.RandomStateError, match="not recognized"):
        random.uniform((1,), key=object())


@pytest.mark.parametrize("backend", RANDOM_BACKENDS)
def test_random_rejects_seed_outside_common_uint32_range(
    backend: asc_typing.BackendName,
) -> None:
    context = asc.CreationContext(helpers.namespace(backend), backend)
    with pytest.raises(asc.RandomStateError, match="32-bit"):
        random.create_key(2**32, context=context)


@pytest.mark.parametrize("backend", RANDOM_BACKENDS)
@pytest.mark.parametrize("dtype_name", ["int32", "bool"])
def test_random_rejects_non_floating_dtype(
    backend: asc_typing.BackendName,
    dtype_name: str,
) -> None:
    selected = helpers.namespace(backend)
    dtype = getattr(selected, dtype_name)
    context = asc.CreationContext(selected, backend)
    key = random.create_key(1, context=context)
    with pytest.raises(asc.RandomStateError, match="floating dtype"):
        random.uniform((2,), key=key, dtype=dtype)


@pytest.mark.parametrize("backend", RANDOM_BACKENDS)
def test_random_default_dtype_is_real_floating(
    backend: asc_typing.BackendName,
) -> None:
    selected = helpers.namespace(backend)
    context = asc.CreationContext(selected, backend)
    key = random.create_key(1, context=context)
    sample, _ = random.uniform((1,), key=key)
    assert selected.isdtype(sample.dtype, "real floating")


@pytest.mark.backend("jax")
def test_jax_random_rejects_unavailable_float64() -> None:
    import jax
    import jax.numpy

    jax_config = typing.cast(_JaxConfig, jax.config)
    selected = helpers.namespace("jax")
    context = asc.CreationContext(selected, "jax")
    key = random.create_key(1, context=context)
    if jax_config.read("jax_enable_x64"):
        sample, _ = random.uniform(
            (1,),
            key=key,
            dtype=jax.numpy.float64,
        )
        assert sample.dtype == jax.numpy.float64
    else:
        with pytest.raises(asc.RandomStateError, match="backend rejected"):
            random.uniform((1,), key=key, dtype=jax.numpy.float64)


@pytest.mark.backend("jax")
def test_jax_random_key_honors_explicit_cpu_device() -> None:
    import jax

    jax_module = typing.cast(_JaxModule, jax)
    for device in (jax_module.devices("cpu")[0], "cpu"):
        context = asc.CreationContext(
            helpers.namespace("jax"),
            "jax",
            device=device,
        )
        key = random.create_key(1, context=context)
        assert "cpu" in str(typing.cast(_DeviceArray, key).device).lower()
