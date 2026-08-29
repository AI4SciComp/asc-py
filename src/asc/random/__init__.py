# Copyright 2026 AI4SciComp contributors.
# SPDX-License-Identifier: Apache-2.0
"""Explicit, backend-native random state and distributions."""

from __future__ import annotations

import dataclasses
import functools
import importlib
import importlib.metadata
import json
import math
import typing
import warnings

from asc import errors
from asc import typing as asc_typing
from asc.backends import _state
from asc.core import namespace as namespace_module
from asc.extensions import _dispatch
from asc.extensions.random import create_key as _create_key
from asc.random._validation import finite_python_real, floating_parameters
from asc.random.initializers import (
    constant,
    glorot_normal,
    glorot_uniform,
    he_normal,
    he_uniform,
    lecun_normal,
    lecun_uniform,
    orthogonal,
    truncated_normal,
)


@functools.lru_cache(maxsize=3)
def _installed_backend_version(backend: str) -> str:
    """Return cached distribution provenance for immutable state checks."""
    try:
        return importlib.metadata.version(backend)
    except importlib.metadata.PackageNotFoundError as exception:
        raise errors.BackendUnavailableError(
            f"RandomState: backend {backend!r} is unavailable"
        ) from exception


class _RandomAdapter(typing.Protocol):
    """Backend distribution and state operations."""

    def split_key(self, key: object, count: int) -> tuple[object, ...]:
        """Split one key into independent children."""
        ...  # pylint: disable=unnecessary-ellipsis

    def register_random_state(self, state_type: type[object]) -> None:
        """Register the state record with a backend tree system if needed."""
        ...  # pylint: disable=unnecessary-ellipsis

    def owns_key(self, key: object) -> bool:
        """Return whether a key belongs to this backend."""
        ...  # pylint: disable=unnecessary-ellipsis

    def is_tracer(self, value: object) -> bool:
        """Return whether a value is an authenticated backend tracer."""
        ...  # pylint: disable=unnecessary-ellipsis


def _nested_json_shape(value: object) -> tuple[int, ...]:
    """Return the rectangular shape of integer-only nested JSON lists."""
    if not isinstance(value, list):
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 0 <= value < 2**32
        ):
            raise ValueError("invalid JAX key word")
        return ()
    if not value:
        return (0,)
    child_shape = _nested_json_shape(value[0])
    if any(_nested_json_shape(child) != child_shape for child in value[1:]):
        raise ValueError("ragged JAX key data")
    return (len(value), *child_shape)


def _flatten_jax_key_data(
    value: object, shape: tuple[int, ...]
) -> tuple[int, ...]:
    """Validate and flatten nested uint32 key data against an explicit shape."""
    if not shape:
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 0 <= value < 2**32
        ):
            raise ValueError("invalid JAX key word")
        return (value,)
    if not isinstance(value, list) or len(value) != shape[0]:
        raise ValueError("JAX key data does not match its declared shape")
    return tuple(
        word
        for child in value
        for word in _flatten_jax_key_data(child, shape[1:])
    )


def _validated_jax_key_data(
    value: object, raw_shape: object
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return flat uint32 words and the complete JAX key-data shape."""
    if raw_shape is None:
        shape = _nested_json_shape(value)
    else:
        if not isinstance(raw_shape, list) or any(
            isinstance(extent, bool)
            or not isinstance(extent, int)
            or extent < 0
            for extent in raw_shape
        ):
            raise ValueError("invalid JAX key-data shape")
        shape = tuple(raw_shape)
    if not shape or shape[-1] not in {2, 4}:
        raise ValueError("invalid JAX key-data word count")
    return _flatten_jax_key_data(value, shape), shape


def _jax_key_device_payload(key: object) -> dict[str, object]:
    """Return stable CPU device identity for a concrete JAX key."""
    device = getattr(key, "device", None)
    if callable(device):
        device = device()
    identifier = getattr(device, "id", None)
    if (
        getattr(device, "platform", None) != "cpu"
        or isinstance(identifier, bool)
        or not isinstance(identifier, int)
        or identifier < 0
    ):
        raise errors.RandomStateError(
            "RandomState.to_json: JAX key has invalid CPU device provenance"
        )
    return {"platform": "cpu", "id": identifier}


@dataclasses.dataclass(frozen=True, slots=True)
class RandomState:
    """Immutable backend-specific random state with version provenance."""

    backend: typing.Literal["numpy", "torch", "jax"]
    key: object
    version: str

    def __post_init__(self) -> None:
        """Validate backend provenance and immutable native key metadata."""
        if self.backend not in {"numpy", "torch", "jax"}:
            raise errors.RandomStateError(
                f"RandomState: unsupported backend {self.backend!r}"
            )
        if (
            not isinstance(self.version, str)
            or not self.version
            or self.version.strip() != self.version
        ):
            raise errors.RandomStateError(
                "RandomState: version must be a non-empty trimmed string"
            )
        installed = _installed_backend_version(self.backend)
        if installed != self.version:
            raise errors.RandomStateError(
                "RandomState: version does not match the installed backend"
            )
        if self.backend in {"numpy", "torch"}:
            native_key = self.key
            if not isinstance(native_key, _state.CounterKey):
                raise errors.RandomStateError(
                    "RandomState: invalid counter key for the selected backend"
                )
            key_backend = getattr(native_key, "backend", None)
            seed = getattr(native_key, "seed", None)
            counter = getattr(native_key, "counter", None)
            if (
                key_backend != self.backend
                or isinstance(seed, bool)
                or not isinstance(seed, int)
                or not 0 <= seed < 2**32
                or isinstance(counter, bool)
                or not isinstance(counter, int)
                or not 0 <= counter < 2**32
            ):
                raise errors.RandomStateError(
                    "RandomState: invalid counter key for the selected backend"
                )
            return
        adapter = typing.cast(
            _RandomAdapter,
            _dispatch.load_backend(
                typing.cast(asc_typing.BackendName, self.backend)
            ),
        )
        is_tracer = adapter.is_tracer(self.key)
        if not (adapter.owns_key(self.key) or is_tracer) or not str(
            getattr(self.key, "dtype", "")
        ).startswith("key<"):
            raise errors.RandomStateError(
                "RandomState: invalid JAX key for the selected backend"
            )
        if is_tracer and not namespace_module.is_trusted_jax_cpu_trace():
            raise errors.RandomStateError(
                "RandomState: JAX tracers are accepted only inside an ASC "
                "CPU-pinned transform"
            )
        device = getattr(self.key, "device", None)
        if not is_tracer and getattr(device, "platform", None) != "cpu":
            raise errors.RandomStateError(
                "RandomState: JAX keys must reside on a CPU device"
            )

    def split(self, count: int = 2) -> tuple[RandomState, ...]:
        """Return deterministic independent child states."""
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise errors.RandomStateError(
                "RandomState.split: count must be a positive integer"
            )
        adapter = typing.cast(
            _RandomAdapter,
            _dispatch.load_backend(
                typing.cast(asc_typing.BackendName, self.backend)
            ),
        )
        return tuple(
            RandomState(self.backend, key, self.version)
            for key in adapter.split_key(self.key, count)
        )

    def spawn(self, count: int) -> tuple[RandomState, ...]:
        """Alias for deterministic child-state derivation."""
        return self.split(count)

    def to_json(self) -> str:
        """Serialize state safely for the same backend/version contract."""
        if isinstance(self.key, _state.CounterKey):
            payload: dict[str, object] = {
                "seed": self.key.seed,
                "counter": self.key.counter,
            }
        else:
            jax = importlib.import_module("jax")
            numpy = importlib.import_module("numpy")
            data = jax.random.key_data(self.key)
            native_data = numpy.asarray(data)
            payload = {
                "device": _jax_key_device_payload(self.key),
                "key_data": native_data.tolist(),
                "key_data_shape": list(native_data.shape),
                "key_impl": str(jax.random.key_impl(self.key)),
            }
        return json.dumps(
            {
                "schema": 1,
                "backend": self.backend,
                "version": self.version,
                "state": payload,
            },
            sort_keys=True,
            separators=(",", ":"),
        )

    @classmethod
    def from_json(cls, document: str) -> RandomState:
        """Restore safe JSON state only for the installed backend version."""
        try:
            payload = json.loads(document)
            if (
                not isinstance(payload["schema"], int)
                or isinstance(payload["schema"], bool)
                or payload["schema"] != 1
            ):
                raise ValueError("unsupported schema")
            backend = payload["backend"]
            version = payload["version"]
            state = payload["state"]
        except (
            KeyError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ) as exception:
            raise errors.RandomStateError(
                "RandomState.from_json: malformed or unsupported state document"
            ) from exception
        if type(backend) is not str or backend not in {
            "numpy",
            "torch",
            "jax",
        }:
            raise errors.RandomStateError(
                f"RandomState.from_json: unsupported backend {backend!r}"
            )
        try:
            installed = importlib.metadata.version(backend)
        except importlib.metadata.PackageNotFoundError as exception:
            raise errors.BackendUnavailableError(
                "RandomState.from_json: optional backend is unavailable; "
                f"install asc-py[{backend}]"
            ) from exception
        if installed != version:
            raise errors.RandomStateError(
                "RandomState.from_json: state version does not match installed "
                f"{backend} version {installed!r}"
            )
        try:
            if not isinstance(state, dict):
                raise TypeError("state must be an object")
            if backend in {"numpy", "torch"}:
                seed = state["seed"]
                counter = state["counter"]
                if (
                    isinstance(seed, bool)
                    or not isinstance(seed, int)
                    or not 0 <= seed < 2**32
                    or isinstance(counter, bool)
                    or not isinstance(counter, int)
                    or not 0 <= counter < 2**32
                ):
                    raise ValueError("invalid counter state")
                key = _state.CounterKey(
                    typing.cast(typing.Literal["numpy", "torch"], backend),
                    seed,
                    counter,
                )
            else:
                key_data = state["key_data"]
                flat_key_data, key_data_shape = _validated_jax_key_data(
                    key_data, state.get("key_data_shape")
                )
                key_impl = state.get("key_impl")
                if key_impl is not None and (
                    not isinstance(key_impl, str)
                    or not key_impl
                    or key_impl.strip() != key_impl
                ):
                    raise ValueError("invalid JAX key data")
                jax = importlib.import_module("jax")
                jnp = importlib.import_module("jax.numpy")
                device_payload = state.get("device")
                cpu_devices = jax.devices("cpu")
                if device_payload is None:
                    selected_device = cpu_devices[0]
                else:
                    if (
                        not isinstance(device_payload, dict)
                        or device_payload.get("platform") != "cpu"
                    ):
                        raise ValueError("invalid JAX key device")
                    device_id = device_payload.get("id")
                    if isinstance(device_id, bool) or not isinstance(
                        device_id, int
                    ):
                        raise ValueError("invalid JAX key device")
                    selected_device = next(
                        (
                            candidate
                            for candidate in cpu_devices
                            if getattr(candidate, "id", None) == device_id
                        ),
                        None,
                    )
                    if selected_device is None:
                        raise ValueError(
                            "serialized JAX CPU device is unavailable"
                        )
                data = jnp.reshape(
                    jnp.asarray(flat_key_data, dtype=jnp.uint32),
                    key_data_shape,
                )
                data = jax.device_put(data, selected_device)
                key = jax.random.wrap_key_data(data, impl=key_impl)
        except (KeyError, TypeError, ValueError) as exception:
            raise errors.RandomStateError(
                "RandomState.from_json: invalid backend state payload"
            ) from exception
        result = cls(backend, key, version)
        if backend == "jax":
            adapter = typing.cast(_RandomAdapter, _dispatch.load_backend("jax"))
            adapter.register_random_state(cls)
        return result


def random_state(
    seed: int,
    *,
    backend: typing.Literal["numpy", "torch", "jax"] | object,
    device: object | None = None,
) -> RandomState:
    """Create explicit state without touching a process-global generator."""
    name = getattr(backend, "name", backend)
    if name not in {"numpy", "torch", "jax"}:
        raise errors.RandomStateError(
            "random_state: backend must be 'numpy', 'torch', or 'jax'"
        )
    from asc.core.backend import backend as select_backend

    effective_device = (
        getattr(backend, "device", None) if device is None else device
    )
    selected = select_backend(
        typing.cast(typing.Literal["numpy", "torch", "jax"], name),
        device=effective_device,
    )
    from asc.config import CreationContext

    context = CreationContext(
        typing.cast(asc_typing.ArrayNamespace, selected.xp),
        selected.name,
        device=selected.device,
    )
    key = _create_key(seed, context=context)
    version = importlib.metadata.version(typing.cast(str, name))
    result = RandomState(typing.cast(typing.Any, name), key, version)
    if name == "jax":
        adapter = typing.cast(_RandomAdapter, _dispatch.load_backend("jax"))
        adapter.register_random_state(RandomState)
    return result


def _shape(shape: object, operation: str) -> asc_typing.Shape:
    if not isinstance(shape, tuple) or any(
        isinstance(extent, bool) or not isinstance(extent, int) or extent < 0
        for extent in shape
    ):
        raise errors.RandomStateError(
            f"{operation}: shape must be a tuple of non-negative integers"
        )
    return typing.cast(asc_typing.Shape, shape)


def _call(
    state: RandomState,
    method: str,
    *args: object,
    **kwargs: object,
) -> tuple[asc_typing.NativeArray, RandomState]:
    if not isinstance(state, RandomState):
        raise errors.RandomStateError(
            f"random.{method}: state must be an asc.random.RandomState"
        )
    adapter = _dispatch.load_backend(
        typing.cast(asc_typing.BackendName, state.backend)
    )
    function = getattr(adapter, method)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result, key = function(*args, key=state.key, **kwargs)
    except errors.AscError:
        raise
    except (
        AttributeError,
        OverflowError,
        RuntimeError,
        TypeError,
        ValueError,
        Warning,
    ) as exception:
        raise errors.RandomStateError(
            f"random.{method}: backend rejected state or distribution arguments"
        ) from exception
    return typing.cast(asc_typing.NativeArray, result), dataclasses.replace(
        state, key=key
    )


def _floating_parameters(
    state: RandomState,
    dtype: object | None,
    operation: str,
    **parameters: float,
) -> dict[str, float]:
    """Validate state and normalize representable distribution parameters."""
    if not isinstance(state, RandomState):
        raise errors.RandomStateError(
            f"{operation}: state must be an asc.random.RandomState"
        )
    return floating_parameters(state.backend, dtype, operation, **parameters)


def random(
    shape: asc_typing.Shape,
    *,
    state: RandomState,
    dtype: object | None = None,
) -> tuple[asc_typing.NativeArray, RandomState]:
    """Sample the half-open unit interval."""
    return uniform(shape, state=state, low=0.0, high=1.0, dtype=dtype)


def uniform(
    shape: asc_typing.Shape,
    *,
    state: RandomState,
    low: float = 0.0,
    high: float = 1.0,
    dtype: object | None = None,
) -> tuple[asc_typing.NativeArray, RandomState]:
    """Sample a finite half-open uniform interval."""
    low = finite_python_real(low, "random.uniform", "low")
    high = finite_python_real(high, "random.uniform", "high")
    if low >= high:
        raise errors.RandomStateError(
            "random.uniform: bounds must be finite with low < high"
        )
    return _call(
        state,
        "uniform",
        _shape(shape, "random.uniform"),
        low=low,
        high=high,
        dtype=dtype,
    )


def normal(
    shape: asc_typing.Shape,
    *,
    state: RandomState,
    mean: float = 0.0,
    std: float = 1.0,
    dtype: object | None = None,
) -> tuple[asc_typing.NativeArray, RandomState]:
    """Sample a normal distribution with positive finite deviation."""
    mean = finite_python_real(mean, "random.normal", "mean")
    std = finite_python_real(std, "random.normal", "std")
    if std <= 0:
        raise errors.RandomStateError(
            "random.normal: mean must be finite and std must be positive finite"
        )
    parameters = _floating_parameters(
        state,
        dtype,
        "random.normal",
        mean=mean,
        std=std,
    )
    return _call(
        state,
        "normal",
        _shape(shape, "random.normal"),
        mean=parameters["mean"],
        std=parameters["std"],
        dtype=dtype,
    )


def standard_normal(
    shape: asc_typing.Shape,
    *,
    state: RandomState,
    dtype: object | None = None,
) -> tuple[asc_typing.NativeArray, RandomState]:
    """Sample a zero-mean, unit-deviation normal distribution."""
    return normal(shape, state=state, dtype=dtype)


def randint(
    low: int,
    high: int,
    shape: asc_typing.Shape,
    *,
    state: RandomState,
    dtype: object | None = None,
) -> tuple[asc_typing.NativeArray, RandomState]:
    """Sample signed integers from ``[low, high)``."""
    if (
        any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in (low, high)
        )
        or low >= high
    ):
        raise errors.RandomStateError(
            "random.randint: low and high must be integers with low < high"
        )
    return _call(
        state,
        "randint",
        _shape(shape, "random.randint"),
        low=low,
        high=high,
        dtype=dtype,
    )


def bernoulli(
    shape: asc_typing.Shape,
    *,
    state: RandomState,
    probability: float = 0.5,
) -> tuple[asc_typing.NativeArray, RandomState]:
    """Sample Boolean Bernoulli outcomes."""
    probability = finite_python_real(
        probability, "random.bernoulli", "probability"
    )
    if not 0.0 <= probability <= 1.0:
        raise errors.RandomStateError(
            "random.bernoulli: probability must be finite in [0, 1]"
        )
    if not isinstance(state, RandomState):
        raise errors.RandomStateError(
            "random.bernoulli: state must be an asc.random.RandomState"
        )
    namespace = _dispatch.load_backend(state.backend).namespace()
    probability = _floating_parameters(
        state,
        namespace.float32,
        "random.bernoulli",
        probability=probability,
    )["probability"]
    return _call(
        state,
        "bernoulli",
        _shape(shape, "random.bernoulli"),
        probability=probability,
    )


def gamma(
    shape: asc_typing.Shape,
    *,
    state: RandomState,
    concentration: float,
    scale: float = 1.0,
    dtype: object | None = None,
) -> tuple[asc_typing.NativeArray, RandomState]:
    """Sample a gamma distribution with positive finite parameters."""
    concentration = finite_python_real(
        concentration, "random.gamma", "concentration"
    )
    scale = finite_python_real(scale, "random.gamma", "scale")
    if concentration <= 0 or scale <= 0:
        raise errors.RandomStateError(
            "random.gamma: concentration and scale must be positive finite"
        )
    parameters = _floating_parameters(
        state,
        dtype,
        "random.gamma",
        concentration=concentration,
        scale=scale,
    )
    return _call(
        state,
        "gamma",
        _shape(shape, "random.gamma"),
        concentration=parameters["concentration"],
        scale=parameters["scale"],
        dtype=dtype,
    )


def exponential(
    shape: asc_typing.Shape,
    *,
    state: RandomState,
    scale: float = 1.0,
    dtype: object | None = None,
) -> tuple[asc_typing.NativeArray, RandomState]:
    """Sample an exponential distribution with positive finite scale."""
    scale = finite_python_real(scale, "random.exponential", "scale")
    if scale <= 0:
        raise errors.RandomStateError(
            "random.exponential: scale must be positive and finite"
        )
    parameters = _floating_parameters(
        state,
        dtype,
        "random.exponential",
        scale=scale,
    )
    return _call(
        state,
        "exponential",
        _shape(shape, "random.exponential"),
        scale=parameters["scale"],
        dtype=dtype,
    )


def choice(
    population: object,
    shape: asc_typing.Shape,
    *,
    state: RandomState,
    replace: bool = True,
    probabilities: object | None = None,
) -> tuple[asc_typing.NativeArray, RandomState]:
    """Sample a one-dimensional population with explicit replacement policy."""
    if not isinstance(state, RandomState):
        raise errors.RandomStateError(
            "random.choice: state must be an asc.random.RandomState"
        )
    if not isinstance(replace, bool):
        raise errors.RandomStateError(
            "random.choice: replace must be a Boolean"
        )
    sample_shape = _shape(shape, "random.choice")
    if isinstance(population, int):
        if isinstance(population, bool) or population <= 0:
            raise errors.RandomStateError(
                "random.choice: integer population must be positive"
            )
        population_size = population
    else:
        try:
            xp = namespace_module.array_namespace(population)
        except errors.AscError as exception:
            raise errors.RandomStateError(
                "random.choice: population must be a positive integer or "
                "one-dimensional native array"
            ) from exception
        if namespace_module.identify_backend(xp) != state.backend:
            raise errors.RandomStateError(
                "random.choice: array population must match the state backend"
            )
        if len(population.shape) != 1 or population.shape[0] == 0:
            raise errors.RandomStateError(
                "random.choice: array population must be non-empty and 1-D"
            )
        population_size = population.shape[0]
    available: object = population_size
    if probabilities is not None:
        try:
            probability_xp = namespace_module.array_namespace(probabilities)
        except errors.AscError as exception:
            raise errors.RandomStateError(
                "random.choice: probabilities must be a native array"
            ) from exception
        if namespace_module.identify_backend(probability_xp) != state.backend:
            raise errors.RandomStateError(
                "random.choice: probabilities must match the state backend"
            )
        if (
            len(probabilities.shape) != 1
            or probabilities.shape[0] != population_size
            or not probability_xp.isdtype(probabilities.dtype, "real floating")
        ):
            raise errors.RandomStateError(
                "random.choice: probabilities must be a real floating 1-D "
                "array matching the population"
            )
        valid_entries = (
            probability_xp.isfinite(probabilities)
            & (probabilities >= 0)
            & (probabilities <= 1.0)
        )
        safe_probabilities = probability_xp.where(
            valid_entries,
            probabilities,
            probability_xp.zeros_like(probabilities),
        )
        calculation_dtype = (
            probability_xp.float32
            if int(probability_xp.finfo(probabilities.dtype).bits) < 32
            else probabilities.dtype
        )
        safe_probabilities = probability_xp.astype(
            safe_probabilities, calculation_dtype, copy=True
        )
        probability_total = probability_xp.sum(safe_probabilities)
        valid = probability_xp.all(valid_entries) & (
            probability_xp.abs(probability_total - 1.0) <= 1e-6
        )
        if state.backend == "jax" and "Tracer" in type(valid).__name__:
            checkify = importlib.import_module("jax.experimental.checkify")
            checkify.check(valid, "asc random.choice invalid probabilities")
        elif not bool(valid):
            raise errors.RandomStateError(
                "random.choice: probabilities must be finite, in [0, 1], and "
                "sum to one"
            )
        safe_total = probability_xp.where(
            valid,
            probability_total,
            probability_xp.ones_like(probability_total),
        )
        probabilities = safe_probabilities / safe_total
        available = probability_xp.count_nonzero(probabilities)
    if not replace:
        enough = available >= math.prod(sample_shape)
        if state.backend == "jax" and "Tracer" in type(enough).__name__:
            checkify = importlib.import_module("jax.experimental.checkify")
            checkify.check(
                enough, "asc random.choice insufficient positive population"
            )
        elif not bool(enough):
            raise errors.RandomStateError(
                "random.choice: sampling without replacement exceeds the "
                "positive population"
            )
    return _call(
        state,
        "choice",
        population,
        sample_shape,
        replace=replace,
        probabilities=probabilities,
    )


def permutation(
    value: object,
    *,
    state: RandomState,
) -> tuple[asc_typing.NativeArray, RandomState]:
    """Return a functional permutation of an integer range or first axis."""
    if not isinstance(state, RandomState):
        raise errors.RandomStateError(
            "random.permutation: state must be an asc.random.RandomState"
        )
    if isinstance(value, int) and (isinstance(value, bool) or value < 0):
        raise errors.RandomStateError(
            "random.permutation: integer input must be non-negative"
        )
    if not isinstance(value, int):
        try:
            xp = namespace_module.array_namespace(value)
        except errors.AscError as exception:
            raise errors.RandomStateError(
                "random.permutation: value must be an integer or native array"
            ) from exception
        if namespace_module.identify_backend(xp) != state.backend:
            raise errors.RandomStateError(
                "random.permutation: array must match the state backend"
            )
        if len(value.shape) == 0:
            raise errors.RandomStateError(
                "random.permutation: array must have rank at least one"
            )
    return _call(state, "permutation", value)


__all__ = [
    "RandomState",
    "bernoulli",
    "choice",
    "constant",
    "exponential",
    "gamma",
    "glorot_normal",
    "glorot_uniform",
    "he_normal",
    "he_uniform",
    "lecun_normal",
    "lecun_uniform",
    "normal",
    "orthogonal",
    "permutation",
    "randint",
    "random",
    "random_state",
    "standard_normal",
    "truncated_normal",
    "uniform",
]
