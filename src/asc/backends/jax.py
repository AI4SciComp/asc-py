# Copyright 2026 AI4SciComp contributors.
# SPDX-License-Identifier: Apache-2.0
"""Lazy JAX extension adapter."""

from __future__ import annotations

import math
import struct
import typing

import jax
import jax.numpy
import numpy

from asc import _array_api_compat, errors
from asc import typing as asc_typing
from asc.backends import _namespace
from asc.core import _dtype


class _JaxRandom(typing.Protocol):
    """Typed subset of :mod:`jax.random` used by this adapter."""

    def key(self, seed: int) -> object:
        """Create a native key."""
        ...  # pylint: disable=unnecessary-ellipsis

    def split(self, key: object) -> tuple[object, object]:
        """Split a native key."""
        ...  # pylint: disable=unnecessary-ellipsis

    def uniform(
        self,
        key: object,
        *,
        shape: asc_typing.Shape,
        dtype: object,
        minval: float,
        maxval: float,
    ) -> object:
        """Sample a uniform native array."""
        ...  # pylint: disable=unnecessary-ellipsis


class _JaxCore(typing.Protocol):
    """Typed tracer identity exposed by :mod:`jax.core`."""

    Tracer: type[object]


class _JaxModule(typing.Protocol):
    """Typed dynamic transformation surface used by this adapter."""

    Array: type[object]
    core: _JaxCore
    random: _JaxRandom

    def device_put(self, value: object, device: object) -> object:
        """Place a value on an explicit device."""
        ...  # pylint: disable=unnecessary-ellipsis

    def devices(self, backend: str | None = None) -> list[object]:
        """Return devices for a backend."""
        ...  # pylint: disable=unnecessary-ellipsis

    def value_and_grad(
        self,
        function: typing.Callable[..., object],
        *,
        argnums: int,
    ) -> typing.Callable[..., tuple[object, object]]:
        """Build a value-and-gradient transform."""
        ...  # pylint: disable=unnecessary-ellipsis

    def jit(
        self,
        function: typing.Callable[..., object],
    ) -> typing.Callable[..., object]:
        """Build a compiled callable."""
        ...  # pylint: disable=unnecessary-ellipsis


class _JaxNumpy(typing.Protocol):
    """Typed dynamic creation surface used by this adapter."""

    floating: object
    signedinteger: object

    def asarray(self, value: object) -> asc_typing.NativeArray:
        """Create an array using JAX defaults."""
        ...  # pylint: disable=unnecessary-ellipsis

    def broadcast_to(
        self, value: object, shape: asc_typing.Shape
    ) -> asc_typing.NativeArray:
        """Broadcast a value to an update shape."""
        ...  # pylint: disable=unnecessary-ellipsis

    def issubdtype(self, first: object, second: object) -> bool:
        """Return whether one dtype belongs to a dtype family."""
        ...  # pylint: disable=unnecessary-ellipsis


class _CheckifyValidation(typing.Protocol):
    """Validation result returned by :mod:`jax.experimental.checkify`."""

    def throw(self) -> None:
        """Raise when a functionalized check failed."""
        ...  # pylint: disable=unnecessary-ellipsis


_jax = typing.cast(_JaxModule, jax)
_jax_numpy = typing.cast(_JaxNumpy, jax.numpy)
_RANDOM_STATE_REGISTRATION = [False]


def _raise_namespace_invalid(
    invalid: object,
    operation: str,
    message: str,
    *,
    index_error: bool = False,
) -> None:
    """Raise or stage a public error for an invalid JAX predicate."""
    if "Tracer" in type(invalid).__name__:
        from jax.experimental import checkify

        checkify.check(
            _jax_numpy.logical_not(invalid),
            f"asc {operation} {message}",
        )
        return
    if not bool(invalid):
        return
    if index_error:
        raise IndexError(f"{operation}: {message}")
    raise ValueError(f"{operation}: {message}")


def _validate_namespace_values(
    name: str,
    args: tuple[object, ...],
    kwargs: dict[str, object],
    expected_result_dtype: str | None,
) -> None:
    """Validate JAX values eagerly or through functionalized checks."""
    if name in {"take", "take_along_axis"}:
        extent = _namespace.take_axis_extent(name, args, kwargs)
        if extent is not None:
            indices = args[1]
            invalid = _jax_numpy.greater_equal(indices, extent)
            if _namespace.is_signed_integer_array(indices):
                invalid = _jax_numpy.logical_or(
                    invalid, _jax_numpy.less(indices, -extent)
                )
            _raise_namespace_invalid(
                _jax_numpy.any(invalid),
                name,
                "index is out of bounds",
                index_error=True,
            )
    value: object | None = None
    message = ""
    if name == "pow" and _namespace.is_integer_dtype_name(
        expected_result_dtype
    ):
        value = args[1]
        message = "integer exponents must be non-negative"
    elif name in {"bitwise_left_shift", "bitwise_right_shift"}:
        value = args[1]
        message = "shift counts must be non-negative"
    elif name == "repeat":
        value = _namespace.namespace_argument_value(args, kwargs, "repeats", 1)
        message = "repeat counts must be non-negative"
    if _array_api_compat.compat.is_jax_array(
        value
    ) and _namespace.is_signed_integer_array(value):
        _raise_namespace_invalid(
            _jax_numpy.any(_jax_numpy.less(value, 0)), name, message
        )


class _JaxNamespace:
    """Forward the frozen namespace after validating explicit dtypes."""

    __name__ = jax.numpy.__name__
    __array_api_version__ = jax.numpy.__array_api_version__

    def __getattr__(self, name: str) -> object:
        return _namespace.checked_attribute(
            jax.numpy,
            name,
            "jax",
            validate_dtype,
            resolve_device,
            validate_values=_validate_namespace_values,
        )


_JAX_NAMESPACE = _JaxNamespace()


def _rounded_float(value: float, dtype: object) -> float:
    """Round a Python float without creating a traced JAX scalar."""
    name = jax.numpy.dtype(dtype).name
    if name == "float64":
        return value
    if name == "float32":
        return struct.unpack("!f", struct.pack("!f", value))[0]
    if name == "float16":
        return struct.unpack("!e", struct.pack("!e", value))[0]
    if name == "bfloat16":
        bits = struct.unpack("!I", struct.pack("!f", value))[0]
        least_significant = (bits >> 16) & 1
        rounded = (bits + 0x7FFF + least_significant) & 0xFFFF0000
        return struct.unpack("!f", struct.pack("!I", rounded))[0]
    raise errors.RandomStateError(
        f"jax random: unsupported floating dtype {name!r}"
    )


def register_random_state(state_type: type[object]) -> None:
    """Register asc random state as a JAX PyTree after JAX is selected."""
    if _RANDOM_STATE_REGISTRATION[0]:
        return

    def flatten(state: object) -> tuple[tuple[object, ...], tuple[str, str]]:
        return (state.key,), (state.backend, state.version)

    def unflatten(
        metadata: tuple[str, str], leaves: tuple[object, ...]
    ) -> object:
        if type(leaves[0]) is object:
            state = object.__new__(state_type)
            object.__setattr__(state, "backend", metadata[0])
            object.__setattr__(state, "key", leaves[0])
            object.__setattr__(state, "version", metadata[1])
            return state
        return state_type(metadata[0], leaves[0], metadata[1])

    jax.tree_util.register_pytree_node(state_type, flatten, unflatten)
    _RANDOM_STATE_REGISTRATION[0] = True


def _validate_cpu_arguments(
    args: object,
    kwargs: object,
    operation: str,
    argument: int | None = None,
) -> None:
    """Prove concrete JAX inputs are dense CPU arrays before tracing."""
    from asc.core.namespace import array_namespace
    from asc.tree import tree_leaves

    arrays = [
        value
        for value in tree_leaves((args, kwargs))
        if _array_api_compat.compat.is_array_api_obj(value)
    ]
    if not any(isinstance(value, _jax.Array) for value in arrays):
        raise errors.DeviceError(
            f"{operation}: at least one concrete JAX CPU array is required"
        )
    array_namespace(*arrays)
    if argument is not None:
        positional = typing.cast(tuple[object, ...], args)
        if argument >= len(positional) or not isinstance(
            positional[argument], _jax.Array
        ):
            raise errors.DTypeError(
                f"{operation}: differentiable argument must be a positional "
                "JAX array"
            )
        if not _jax_numpy.issubdtype(
            positional[argument].dtype, _jax_numpy.floating
        ):
            raise errors.DTypeError(
                f"{operation}: differentiable argument must be real floating"
            )


def namespace() -> object:
    """Return JAX's validated frozen native Array API namespace."""
    return _JAX_NAMESPACE


def einsum(subscripts: str, *operands: object) -> object:
    """Evaluate Einstein summation through the private native extension."""
    return jax.numpy.einsum(subscripts, *operands)


def kron(first: object, second: object) -> object:
    """Evaluate a Kronecker product through the private native extension."""
    return jax.numpy.kron(first, second)


def resolve_device(device: object | None) -> object:
    """Normalize a supported JAX CPU device string or native object."""
    cpu_devices = _jax.devices("cpu")
    if device is None or (type(device) is str and device == "cpu"):
        return cpu_devices[0]
    if any(device is candidate for candidate in cpu_devices):
        return device
    raise errors.DeviceError("jax backend: only the CPU device is supported")


def validate_dtype(dtype: object) -> None:
    """Reject dtypes outside the active frozen JAX release surface."""
    try:
        native_dtype = jax.numpy.dtype(dtype)
    except Exception as exception:  # pylint: disable=broad-exception-caught
        raise errors.DTypeError(
            "jax backend: dtype is outside the supported release surface"
        ) from exception
    name = _dtype.dtype_name(native_dtype)
    canonical = None if name is None else getattr(jax.numpy, name, None)
    try:
        canonical_dtype = jax.numpy.dtype(canonical)
    except Exception as exception:  # pylint: disable=broad-exception-caught
        raise errors.DTypeError(
            "jax backend: dtype is outside the supported release surface"
        ) from exception
    if (
        native_dtype != canonical_dtype
        or (
            dtype is not canonical
            and dtype is not native_dtype
            and dtype is not getattr(native_dtype, "type", None)
        )
        or getattr(native_dtype, "metadata", None) is not None
    ):
        raise errors.DTypeError(
            "jax backend: dtype is outside the supported release surface"
        )
    x64_enabled = bool(getattr(jax.config, "x64_enabled", False))
    _dtype.require_supported_dtype(
        "jax",
        native_dtype,
        "jax backend",
        jax_x64_enabled=x64_enabled,
    )


def to_cpu(value: object) -> object:
    """Transfer a native JAX array to an explicit CPU device."""
    return _jax.device_put(value, _jax.devices("cpu")[0])


def create_key(seed: int, *, device: object | None = None) -> object:
    """Create a native JAX random key on the selected CPU device."""
    key = _jax.random.key(seed)
    if device is not None:
        selected_device = device
        if isinstance(device, str):
            selected_device = _jax.devices("cpu")[0]
        key = _jax.device_put(key, selected_device)
    native_key = typing.cast(asc_typing.NativeArray, key)
    observed_device = getattr(native_key, "device", None)
    if callable(observed_device):
        observed_device = observed_device()
    if getattr(observed_device, "platform", None) != "cpu":
        raise errors.RandomStateError(
            "jax random: key was not created on a CPU device"
        )
    return key


def owns_key(key: object) -> bool:
    """Return whether ``key`` is native JAX random state."""
    return isinstance(key, _jax.Array)


def is_tracer(value: object) -> bool:
    """Return whether ``value`` is an authenticated JAX tracer."""
    return isinstance(value, _jax.core.Tracer)


def uniform(
    shape: asc_typing.Shape,
    *,
    key: object,
    low: float,
    high: float,
    dtype: object | None,
) -> tuple[asc_typing.NativeArray, object]:
    """Split a JAX key and sample with the derived key."""
    if not isinstance(key, _jax.Array):
        raise errors.RandomStateError("jax random: incompatible key")
    advanced_key, sample_key = _jax.random.split(key)
    requested_dtype = dtype
    if requested_dtype is None:
        requested_dtype = _jax_numpy.asarray(0.0).dtype
    _validate_random_dtype(requested_dtype, "uniform")
    if not _jax_numpy.issubdtype(requested_dtype, _jax_numpy.floating):
        raise errors.RandomStateError(
            "jax random: dtype must be a real floating dtype"
        )
    information = jax.numpy.finfo(requested_dtype)
    if (
        low < float(information.min)
        or high > float(information.max)
        or high - low > float(information.max)
    ):
        raise errors.RandomStateError(
            "jax random: interval is not representable in the requested dtype"
        )
    if (
        _rounded_float(low, requested_dtype) != low
        or _rounded_float(high, requested_dtype) != high
    ):
        raise errors.RandomStateError(
            "jax random: interval endpoints must be exactly representable "
            "in the requested dtype"
        )
    low_value = jax.numpy.asarray(low, dtype=requested_dtype)
    high_value = jax.numpy.asarray(high, dtype=requested_dtype)
    result = _jax.random.uniform(
        sample_key,
        shape=shape,
        dtype=requested_dtype,
        minval=low,
        maxval=high,
    )
    upper = jax.numpy.nextafter(
        high_value, jax.numpy.asarray(-numpy.inf, dtype=requested_dtype)
    )
    result = jax.numpy.clip(result, low_value, upper)
    return typing.cast(asc_typing.NativeArray, result), advanced_key


def _split(key: object) -> tuple[object, object]:
    if not owns_key(key):
        raise errors.RandomStateError("jax random: incompatible state")
    return _jax.random.split(key)


def _floating_dtype(dtype: object | None, operation: str) -> object:
    requested = _jax_numpy.asarray(0.0).dtype if dtype is None else dtype
    _validate_random_dtype(requested, operation)
    if not _jax_numpy.issubdtype(requested, _jax_numpy.floating):
        raise errors.RandomStateError(
            f"jax {operation}: dtype must be floating"
        )
    return requested


def _validate_random_dtype(dtype: object, operation: str) -> None:
    """Translate an unavailable JAX dtype into the random error surface."""
    try:
        validate_dtype(dtype)
    except errors.DTypeError as exception:
        raise errors.RandomStateError(
            f"jax {operation}: backend rejected the unavailable requested dtype"
        ) from exception


def normal(
    shape: asc_typing.Shape,
    *,
    key: object,
    mean: float,
    std: float,
    dtype: object | None,
) -> tuple[asc_typing.NativeArray, object]:
    """Sample a JAX normal distribution from explicit state."""
    advanced, sample_key = _split(key)
    result = mean + std * jax.random.normal(
        sample_key, shape=shape, dtype=_floating_dtype(dtype, "normal")
    )
    return typing.cast(asc_typing.NativeArray, result), advanced


def truncated_normal(
    shape: asc_typing.Shape,
    *,
    key: object,
    mean: float,
    std: float,
    lower: float,
    upper: float,
    dtype: object | None,
) -> tuple[asc_typing.NativeArray, object]:
    """Sample a JAX truncated normal distribution from explicit state."""
    advanced, sample_key = _split(key)
    requested = _floating_dtype(dtype, "truncated_normal")
    standardized_lower = (lower - mean) / std
    standardized_upper = (upper - mean) / std
    if not (
        math.isfinite(standardized_lower)
        and math.isfinite(standardized_upper)
        and standardized_lower < standardized_upper
    ):
        raise errors.RandomStateError(
            "jax truncated_normal: standardized bounds must be finite and "
            "ordered"
        )
    standard = jax.random.truncated_normal(
        sample_key,
        standardized_lower,
        standardized_upper,
        shape=shape,
        dtype=requested,
    )
    result = mean + std * standard
    result = jax.numpy.clip(
        result,
        jax.numpy.asarray(lower, dtype=requested),
        jax.numpy.asarray(upper, dtype=requested),
    )
    return typing.cast(asc_typing.NativeArray, result), advanced


def randint(
    shape: asc_typing.Shape,
    *,
    key: object,
    low: int,
    high: int,
    dtype: object | None,
) -> tuple[asc_typing.NativeArray, object]:
    """Sample signed JAX integers from explicit state."""
    advanced, sample_key = _split(key)
    requested = _jax_numpy.asarray(0).dtype if dtype is None else dtype
    _validate_random_dtype(requested, "randint")
    if not _jax_numpy.issubdtype(requested, _jax_numpy.signedinteger):
        raise errors.RandomStateError(
            "jax randint: dtype must be signed integer"
        )
    information = jax.numpy.iinfo(requested)
    if low < int(information.min) or high > int(information.max) + 1:
        raise errors.RandomStateError(
            "jax randint: bounds are outside the requested dtype range"
        )
    if high == int(information.max) + 1:
        if low == int(information.min):
            unsigned = getattr(jax.numpy, f"uint{information.bits}")
            bits = jax.random.bits(
                sample_key,
                shape=shape,
                dtype=unsigned,
            )
            result = jax.lax.bitcast_convert_type(bits, requested)
        elif low < 0:
            unsigned = getattr(jax.numpy, f"uint{information.bits}")
            pending = jax.numpy.ones(shape, dtype=jax.numpy.bool_)
            initial = jax.numpy.zeros(shape, dtype=requested)

            def needs_sample(carry: tuple[object, object, object]) -> object:
                return jax.numpy.any(carry[2])

            def sample_pending(
                carry: tuple[object, object, object],
            ) -> tuple[object, object, object]:
                current_key, current, current_pending = carry
                next_key, proposal_key = jax.random.split(current_key)
                bits = jax.random.bits(
                    proposal_key,
                    shape=shape,
                    dtype=unsigned,
                )
                proposals = jax.lax.bitcast_convert_type(bits, requested)
                accepted = current_pending & (proposals >= low)
                return (
                    next_key,
                    jax.numpy.where(accepted, proposals, current),
                    current_pending & ~accepted,
                )

            _, result, _ = jax.lax.while_loop(
                needs_sample,
                sample_pending,
                (sample_key, initial, pending),
            )
        else:
            reflected = jax.random.randint(
                sample_key,
                shape,
                int(information.min),
                -low,
                dtype=requested,
            )
            result = jax.numpy.bitwise_not(reflected)
    else:
        result = jax.random.randint(
            sample_key, shape, low, high, dtype=requested
        )
    return typing.cast(asc_typing.NativeArray, result), advanced


def bernoulli(
    shape: asc_typing.Shape,
    *,
    key: object,
    probability: float,
) -> tuple[asc_typing.NativeArray, object]:
    """Sample Boolean JAX Bernoulli values from explicit state."""
    advanced, sample_key = _split(key)
    result = jax.random.bernoulli(sample_key, probability, shape=shape)
    return typing.cast(asc_typing.NativeArray, result), advanced


def gamma(
    shape: asc_typing.Shape,
    *,
    key: object,
    concentration: float,
    scale: float,
    dtype: object | None,
) -> tuple[asc_typing.NativeArray, object]:
    """Sample a JAX gamma distribution from explicit state."""
    advanced, sample_key = _split(key)
    result = scale * jax.random.gamma(
        sample_key,
        concentration,
        shape=shape,
        dtype=_floating_dtype(dtype, "gamma"),
    )
    return typing.cast(asc_typing.NativeArray, result), advanced


def exponential(
    shape: asc_typing.Shape,
    *,
    key: object,
    scale: float,
    dtype: object | None,
) -> tuple[asc_typing.NativeArray, object]:
    """Sample a JAX exponential distribution from explicit state."""
    advanced, sample_key = _split(key)
    result = scale * jax.random.exponential(
        sample_key,
        shape=shape,
        dtype=_floating_dtype(dtype, "exponential"),
    )
    return typing.cast(asc_typing.NativeArray, result), advanced


def choice(
    population: object,
    shape: asc_typing.Shape,
    *,
    key: object,
    replace: bool,
    probabilities: object | None,
) -> tuple[asc_typing.NativeArray, object]:
    """Sample from a JAX population using explicit state."""
    advanced, sample_key = _split(key)
    result = jax.random.choice(
        sample_key,
        population,
        shape=shape,
        replace=replace,
        p=probabilities,
    )
    return typing.cast(asc_typing.NativeArray, result), advanced


def permutation(
    value: object, *, key: object
) -> tuple[asc_typing.NativeArray, object]:
    """Return a functional JAX permutation."""
    advanced, sample_key = _split(key)
    result = jax.random.permutation(sample_key, value)
    return typing.cast(asc_typing.NativeArray, result), advanced


def split_key(key: object, count: int) -> tuple[object, ...]:
    """Derive independent native JAX child keys."""
    if not owns_key(key):
        raise errors.RandomStateError("jax random: incompatible state")
    return tuple(jax.random.split(key, count))


def index_add(
    array: asc_typing.NativeArray,
    indices: asc_typing.NativeArray,
    values: asc_typing.NativeArray,
    *,
    axis: int,
    update_shape: asc_typing.Shape,
) -> asc_typing.NativeArray:
    """Apply JAX's functional ``at[].add`` update."""
    native_array = typing.cast(jax.Array, array)
    native_indices = typing.cast(jax.Array, indices)
    native_values = typing.cast(jax.Array, values)
    if not _jax_numpy.issubdtype(
        native_indices.dtype, _jax_numpy.signedinteger
    ):
        raise errors.IndexContractError(
            "index_add: JAX indices must have a signed integer dtype"
        )
    valid = (native_indices >= 0) & (native_indices < native_array.shape[axis])
    if isinstance(native_indices, jax.core.Tracer):
        from jax.experimental import checkify

        checkify.check(_jax_numpy.all(valid), "asc index_add out of bounds")
    elif not bool(_jax_numpy.all(valid)):
        raise errors.IndexContractError("index_add: index is out of bounds")
    selector: list[object] = [slice(None)] * native_array.ndim
    selector[axis] = native_indices
    broadcast_values = _jax_numpy.broadcast_to(native_values, update_shape)
    result = native_array.at[tuple(selector)].add(broadcast_values)
    return typing.cast(asc_typing.NativeArray, result)


def index_update(
    array: asc_typing.NativeArray,
    indices: asc_typing.NativeArray,
    values: asc_typing.NativeArray,
    *,
    axis: int,
    update_shape: asc_typing.Shape,
    reduction: str,
) -> asc_typing.NativeArray:
    """Apply one native JAX functional indexed update."""
    native_array = typing.cast(jax.Array, array)
    native_indices = typing.cast(jax.Array, indices)
    native_values = typing.cast(jax.Array, values)
    if not _jax_numpy.issubdtype(
        native_indices.dtype, _jax_numpy.signedinteger
    ):
        raise errors.IndexUpdateError(
            "index update: JAX indices must be signed integers"
        )
    valid = (native_indices >= 0) & (native_indices < native_array.shape[axis])
    duplicate: object | None = None
    if reduction == "set":
        sorted_indices = _jax_numpy.sort(native_indices)
        duplicate = _jax_numpy.any(sorted_indices[1:] == sorted_indices[:-1])
    if isinstance(native_indices, jax.core.Tracer):
        from jax.experimental import checkify

        checkify.check(_jax_numpy.all(valid), "asc index update out of bounds")
        if reduction == "set":
            assert duplicate is not None
            checkify.check(~duplicate, "asc index_set duplicate indices")
    else:
        if not bool(_jax_numpy.all(valid)):
            raise errors.IndexUpdateError(
                "index update: index is out of bounds"
            )
        if reduction == "set":
            assert duplicate is not None
            if bool(duplicate):
                raise errors.DuplicateIndexError(
                    "index_set: duplicate indices have no deterministic set "
                    "policy"
                )
    selector: list[object] = [slice(None)] * native_array.ndim
    selector[axis] = native_indices
    target = native_array.at[tuple(selector)]
    broadcast_values = _jax_numpy.broadcast_to(native_values, update_shape)
    operation = getattr(target, reduction)
    return typing.cast(asc_typing.NativeArray, operation(broadcast_values))


def value_and_grad(
    function: typing.Callable[..., object],
    argument: int,
) -> typing.Callable[..., tuple[object, object]]:
    """Build a native JAX value-and-gradient callable."""
    native = _jax.value_and_grad(function, argnums=argument)

    def transformed(*args: object, **kwargs: object) -> tuple[object, object]:
        from asc.core.namespace import trusted_jax_cpu_trace

        _validate_cpu_arguments(args, kwargs, "value_and_grad", argument)
        with trusted_jax_cpu_trace():
            return native(*args, **kwargs)

    return transformed


def _jax_transform(
    transform: typing.Callable[..., typing.Callable[..., object]],
    function: typing.Callable[..., object],
    argument: int,
) -> typing.Callable[..., object]:
    native = transform(function, argnums=argument)

    def transformed(*args: object, **kwargs: object) -> object:
        from asc.core.namespace import trusted_jax_cpu_trace

        _validate_cpu_arguments(args, kwargs, "autodiff", argument)
        with trusted_jax_cpu_trace():
            return native(*args, **kwargs)

    return transformed


def _validate_transform_operands(
    values: tuple[object, ...], operation: str
) -> None:
    """Require native real-floating arrays at JVP/VJP boundaries."""
    from asc.tree import tree_leaves

    leaves = tree_leaves(values)
    if not leaves or any(
        not _array_api_compat.compat.is_array_api_obj(value) for value in leaves
    ):
        raise errors.DTypeError(
            f"{operation}: every operand must be a native real-floating array"
        )
    _validate_cpu_arguments(tuple(leaves), {}, operation)
    if any(
        not _jax_numpy.issubdtype(value.dtype, _jax_numpy.floating)
        for value in leaves
    ):
        raise errors.DTypeError(
            f"{operation}: every operand must be a native real-floating array"
        )


def grad(
    function: typing.Callable[..., object], argument: int
) -> typing.Callable[..., object]:
    """Build a native JAX scalar-output gradient callable."""
    return _jax_transform(jax.grad, function, argument)


def jacobian(
    function: typing.Callable[..., object], argument: int
) -> typing.Callable[..., object]:
    """Build a native JAX Jacobian callable."""
    return _jax_transform(jax.jacrev, function, argument)


def hessian(
    function: typing.Callable[..., object], argument: int
) -> typing.Callable[..., object]:
    """Build a native JAX Hessian callable."""
    return _jax_transform(jax.hessian, function, argument)


def jvp(
    function: typing.Callable[..., object],
    primals: tuple[object, ...],
    tangents: tuple[object, ...],
) -> tuple[object, object]:
    """Evaluate a native JAX forward-mode product."""
    from asc.core.namespace import trusted_jax_cpu_trace

    _validate_transform_operands((*primals, *tangents), "jvp")
    with trusted_jax_cpu_trace():
        return typing.cast(
            tuple[object, object], jax.jvp(function, primals, tangents)
        )


def vjp(
    function: typing.Callable[..., object], primals: tuple[object, ...]
) -> tuple[object, typing.Callable[..., object]]:
    """Evaluate a native JAX reverse-mode product setup."""
    from asc.core.namespace import trusted_jax_cpu_trace

    _validate_transform_operands(primals, "vjp")
    with trusted_jax_cpu_trace():
        result = typing.cast(
            tuple[object, typing.Callable[..., object]],
            jax.vjp(function, *primals),
        )
    native_pullback = result[1]

    def pullback(*cotangents: object) -> object:
        _validate_transform_operands(cotangents, "vjp pullback")
        with trusted_jax_cpu_trace():
            return native_pullback(*cotangents)

    return result[0], pullback


def compile_function(
    function: typing.Callable[..., object],
) -> typing.Callable[..., object]:
    """Compile a function with JAX JIT."""
    from asc.core.namespace import trusted_jax_cpu_trace

    def traced(*args: object, **kwargs: object) -> object:
        with trusted_jax_cpu_trace():
            return function(*args, **kwargs)

    from jax.experimental import checkify

    native = jax.jit(checkify.checkify(traced))

    def checked(*args: object, **kwargs: object) -> object:
        _validate_cpu_arguments(args, kwargs, "jit")
        with trusted_jax_cpu_trace():
            validation, result = native(*args, **kwargs)
        _throw_checkify_error(validation, "jit")
        return result

    return checked


def vmap(
    function: typing.Callable[..., object],
    in_axes: object,
    out_axes: object,
) -> typing.Callable[..., object]:
    """Vectorize a JAX callable over the supported axis subset."""
    from jax.experimental import checkify

    def traced(*args: object, **kwargs: object) -> object:
        from asc.core.namespace import trusted_jax_cpu_trace

        with trusted_jax_cpu_trace():
            return function(*args, **kwargs)

    native = checkify.checkify(
        jax.vmap(traced, in_axes=in_axes, out_axes=out_axes)
    )

    def transformed(*args: object, **kwargs: object) -> object:
        from asc.core.namespace import trusted_jax_cpu_trace

        _validate_cpu_arguments(args, kwargs, "vmap")
        with trusted_jax_cpu_trace():
            validation, result = native(*args, **kwargs)
        _throw_checkify_error(validation, "vmap")
        return result

    return transformed


def _throw_checkify_error(
    validation: _CheckifyValidation, operation: str
) -> None:
    """Translate functional JAX checks into the stable public hierarchy."""
    from jax.experimental import checkify

    try:
        validation.throw()
    except checkify.JaxRuntimeError as exception:
        message = str(exception)
        if "random.choice" in message:
            raise errors.RandomStateError(
                f"{operation}: {message}"
            ) from exception
        if "index_set duplicate indices" in message:
            raise errors.DuplicateIndexError(
                f"{operation}: {message}"
            ) from exception
        if any(
            operation_name in message
            for operation_name in (
                "ArrayDataset",
                "ravel_multi_index",
                "take index",
                "take_along_axis index",
                "unravel_index",
            )
        ):
            raise IndexError(f"{operation}: {message}") from exception
        if any(
            operation_name in message
            for operation_name in (
                "pow integer exponents",
                "repeat repeat counts",
                "shift counts",
            )
        ):
            raise ValueError(f"{operation}: {message}") from exception
        raise errors.IndexUpdateError(f"{operation}: {message}") from exception
