# Copyright 2026 AI4SciComp contributors.
# SPDX-License-Identifier: Apache-2.0
"""Explicit random-state, distribution, sampling, and initializer tests."""

from __future__ import annotations

import importlib.metadata
import json

import numpy
import pytest

import asc
from asc import random
from tests import helpers

BACKENDS = helpers.NATIVE_BACKENDS


@pytest.mark.parametrize("backend", BACKENDS)
def test_state_replay_split_spawn_and_serialization(backend: str) -> None:
    first = random.random_state(42, backend=backend)
    second = random.random_state(42, backend=backend)
    first_value, first_next = random.random((8,), state=first)
    second_value, _second_next = random.random((8,), state=second)
    numpy.testing.assert_allclose(
        numpy.asarray(first_value), numpy.asarray(second_value)
    )
    assert first_next != first or backend == "jax"
    assert len(first.split(3)) == 3
    assert len(first.spawn(2)) == 2
    restored = random.RandomState.from_json(first.to_json())
    replay, _ = random.random((8,), state=restored)
    numpy.testing.assert_allclose(
        numpy.asarray(replay), numpy.asarray(first_value)
    )
    document = json.loads(first.to_json())
    assert document["schema"] == 1


@pytest.mark.backend("jax")
def test_non_default_jax_prng_state_round_trip() -> None:
    import jax

    key = jax.random.key(42, impl="rbg")
    state = random.RandomState("jax", key, importlib.metadata.version("jax"))

    restored = random.RandomState.from_json(state.to_json())

    assert str(jax.random.key_impl(restored.key)) == "rbg"
    numpy.testing.assert_array_equal(
        numpy.asarray(jax.random.key_data(restored.key)),
        numpy.asarray(jax.random.key_data(key)),
    )

    legacy_payload = json.loads(state.to_json())
    legacy_payload["state"].pop("device")
    legacy_payload["state"].pop("key_data_shape")
    legacy = random.RandomState.from_json(json.dumps(legacy_payload))
    numpy.testing.assert_array_equal(
        numpy.asarray(jax.random.key_data(legacy.key)),
        numpy.asarray(jax.random.key_data(key)),
    )

    empty = random.RandomState(
        "jax",
        jax.random.split(key, 0),
        importlib.metadata.version("jax"),
    )
    empty_restored = random.RandomState.from_json(empty.to_json())
    assert empty_restored.key.shape == (0,)


@pytest.mark.backend("jax")
def test_jax_random_state_rejects_malformed_key_shape_and_device() -> None:
    state = random.random_state(42, backend="jax")
    valid = json.loads(state.to_json())

    malformed_states = (
        {**valid["state"], "key_data_shape": [3]},
        {**valid["state"], "key_data_shape": [2, 2]},
        {**valid["state"], "key_data": [True, 1]},
        {**valid["state"], "device": {"platform": "gpu", "id": 0}},
        {**valid["state"], "device": {"platform": "cpu", "id": True}},
        {**valid["state"], "device": {"platform": "cpu", "id": 10**9}},
    )
    for malformed_state in malformed_states:
        document = {**valid, "state": malformed_state}
        with pytest.raises(asc.RandomStateError, match="state payload"):
            random.RandomState.from_json(json.dumps(document))


@pytest.mark.parametrize("backend", BACKENDS)
def test_all_distributions_have_native_shape_dtype(backend: str) -> None:
    selected = asc.backend(backend)
    xp = selected.xp
    state = random.random_state(7, backend=backend)
    calls = (
        lambda current: random.uniform(
            (32,), state=current, low=-2, high=3, dtype=xp.float32
        ),
        lambda current: random.normal(
            (32,), state=current, mean=1, std=2, dtype=xp.float32
        ),
        lambda current: random.standard_normal(
            (32,), state=current, dtype=xp.float32
        ),
        lambda current: random.randint(
            2, 9, (32,), state=current, dtype=xp.int32
        ),
        lambda current: random.bernoulli(
            (32,), state=current, probability=0.25
        ),
        lambda current: random.gamma(
            (32,), state=current, concentration=2, scale=3, dtype=xp.float32
        ),
        lambda current: random.exponential(
            (32,), state=current, scale=2, dtype=xp.float32
        ),
    )
    for call in calls:
        values, state = call(state)
        assert values.shape == (32,)
        assert asc.backend_of(values) == backend
    assert bool(xp.all((calls[0](state)[0] >= -2) & (calls[0](state)[0] < 3)))


@pytest.mark.parametrize("backend", BACKENDS)
def test_choice_permutation_and_probabilities(backend: str) -> None:
    selected = asc.backend(backend)
    xp = selected.xp
    state = random.random_state(9, backend=backend)
    probabilities = xp.asarray([0.0, 0.25, 0.75], dtype=xp.float32)
    values, state = random.choice(
        3, (20,), state=state, probabilities=probabilities
    )
    assert values.shape == (20,)
    assert not bool(xp.any(values == 0))
    population = xp.asarray([10, 20, 30], dtype=xp.int32)
    selected_values, state = random.choice(
        population, (2,), state=state, replace=False
    )
    assert selected_values.shape == (2,)
    order, _ = random.permutation(population, state=state)
    assert sorted(int(item) for item in numpy.asarray(order)) == [10, 20, 30]


@pytest.mark.backend("torch")
def test_torch_choice_supports_zero_sized_output_shapes() -> None:
    selected = asc.backend("torch")
    population = selected.xp.asarray([10, 20, 30], dtype=selected.xp.int16)
    state = random.random_state(9, backend="torch")

    values, next_state = random.choice(population, (2, 0, 3), state=state)

    assert values.shape == (2, 0, 3)
    assert values.dtype == population.dtype
    assert next_state != state


@pytest.mark.parametrize("backend", BACKENDS)
def test_initializers_and_orthogonality(backend: str) -> None:
    selected = asc.backend(backend)
    xp = selected.xp
    state = random.random_state(11, backend=backend)
    assert random.constant(
        (2, 3), 4, backend=backend, dtype=xp.float32
    ).shape == (2, 3)
    names = (
        "glorot_uniform",
        "glorot_normal",
        "lecun_uniform",
        "lecun_normal",
        "he_uniform",
        "he_normal",
    )
    for name in names:
        values, state = getattr(random, name)(
            (4, 3), state=state, dtype=xp.float32
        )
        assert values.shape == (4, 3)
        assert bool(xp.all(xp.isfinite(values)))
    bounded, state = random.truncated_normal(
        (256,), state=state, lower=-0.5, upper=0.25, dtype=xp.float32
    )
    assert bool(xp.all((bounded >= -0.5) & (bounded <= 0.25)))
    orthogonal, _ = random.orthogonal((5, 3), state=state, dtype=xp.float32)
    gram = xp.matrix_transpose(orthogonal) @ orthogonal
    numpy.testing.assert_allclose(
        numpy.asarray(gram), numpy.eye(3), atol=1e-4, rtol=1e-4
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_truncated_normal_far_tail_finishes_and_stays_bounded(
    backend: str,
) -> None:
    selected = asc.backend(backend)
    values, _ = random.truncated_normal(
        (64,),
        state=random.random_state(19, backend=backend),
        lower=10.0,
        upper=11.0,
        dtype=selected.xp.float32,
    )
    assert bool(selected.xp.all((values >= 10.0) & (values <= 11.0)))


def test_numpy_float16_normal_allows_expected_tail_overflow() -> None:
    with numpy.errstate(all="raise"):
        values, _ = random.normal(
            (10,),
            state=random.random_state(0, backend="numpy"),
            std=float(numpy.finfo(numpy.float16).max),
            dtype=numpy.float16,
        )

    assert values.dtype == numpy.float16
    assert numpy.isinf(values).any()


@pytest.mark.parametrize("backend", BACKENDS)
def test_truncated_normal_rejects_collapsed_standardized_bounds(
    backend: str,
) -> None:
    xp = asc.backend(backend).xp

    with pytest.raises(asc.RandomStateError, match="standardized bounds"):
        random.truncated_normal(
            (2,),
            state=random.random_state(1, backend=backend),
            mean=float(numpy.finfo(numpy.float32).max),
            std=float(numpy.nextafter(numpy.float32(0), numpy.float32(1))),
            lower=-1.0,
            upper=1.0,
            dtype=xp.float32,
        )


@pytest.mark.parametrize("backend", BACKENDS)
def test_distribution_validation(backend: str) -> None:
    selected = asc.backend(backend)
    xp = selected.xp
    state = random.random_state(1, backend=backend)
    invalid_calls = (
        lambda: random.uniform((2,), state=state, low=numpy.nan),
        lambda: random.uniform((2,), state=state, low=2, high=1),
        lambda: random.normal((2,), state=state, std=0),
        lambda: random.randint(2, 2, (2,), state=state),
        lambda: random.bernoulli((2,), state=state, probability=2),
        lambda: random.gamma((2,), state=state, concentration=0),
        lambda: random.exponential((2,), state=state, scale=numpy.inf),
        lambda: random.choice(0, (1,), state=state),
        lambda: random.choice(2, (3,), state=state, replace=False),
        lambda: random.permutation(-1, state=state),
        lambda: random.uniform((True,), state=state),
        lambda: random.uniform((2,), state=state, dtype=xp.int32),
        lambda: random.truncated_normal((2,), state=state, std=-1),
        lambda: random.orthogonal((1,), state=state),
    )
    for call in invalid_calls:
        with pytest.raises(asc.RandomStateError):
            call()
    bad_probabilities = xp.asarray([0.2, 0.2], dtype=xp.float32)
    with pytest.raises(asc.RandomStateError, match="sum"):
        random.choice(2, (1,), state=state, probabilities=bad_probabilities)


@pytest.mark.parametrize("backend", BACKENDS)
def test_distribution_parameters_reject_native_scalar_arrays(
    backend: str,
) -> None:
    selected = asc.backend(backend)
    scalar = selected.xp.asarray(0.5, dtype=selected.xp.float32)
    state = random.random_state(1, backend=backend)
    calls = (
        lambda: random.uniform((1,), state=state, low=scalar),
        lambda: random.normal((1,), state=state, mean=scalar),
        lambda: random.bernoulli((1,), state=state, probability=scalar),
        lambda: random.gamma((1,), state=state, concentration=scalar),
        lambda: random.exponential((1,), state=state, scale=scalar),
        lambda: random.truncated_normal((1,), state=state, mean=scalar),
        lambda: random.orthogonal((2, 2), state=state, gain=scalar),
    )

    for call in calls:
        with pytest.raises(asc.RandomStateError, match="Python real scalar"):
            call()


@pytest.mark.backend("torch")
@pytest.mark.parametrize(
    "operation",
    ("uniform", "normal", "gamma", "exponential"),
)
def test_torch_random_translates_foreign_dtype_errors(operation: str) -> None:
    state = random.random_state(1, backend="torch")
    kwargs: dict[str, object] = {
        "shape": (1,),
        "state": state,
        "dtype": numpy.float32,
    }
    if operation == "gamma":
        kwargs["concentration"] = 1.0

    with pytest.raises(
        asc.RandomStateError, match=r"release surface|backend rejected"
    ):
        getattr(random, operation)(**kwargs)


@pytest.mark.parametrize("backend", BACKENDS)
def test_uniform_rejects_intervals_unrepresentable_in_output_dtype(
    backend: str,
) -> None:
    selected = asc.backend(backend)
    with pytest.raises(asc.RandomStateError, match="representable"):
        random.uniform(
            (2,),
            state=random.random_state(1, backend=backend),
            low=-3e38,
            high=3e38,
            dtype=selected.xp.float32,
        )
    with pytest.raises(asc.RandomStateError, match="endpoints"):
        random.uniform(
            (2,),
            state=random.random_state(1, backend=backend),
            low=1.0,
            high=1.0 + 2**-25,
            dtype=selected.xp.float32,
        )


@pytest.mark.parametrize("backend", BACKENDS)
def test_uniform_preserves_half_open_adjacent_float32_bounds(
    backend: str,
) -> None:
    selected = asc.backend(backend)
    high = float(numpy.nextafter(numpy.float32(1), numpy.float32(numpy.inf)))

    values, _ = random.uniform(
        (10_000,),
        state=random.random_state(7, backend=backend),
        low=1.0,
        high=high,
        dtype=selected.xp.float32,
    )
    widened = numpy.asarray(values).astype(numpy.float64)

    assert bool(numpy.all(widened >= 1.0))
    assert bool(numpy.all(widened < high))


@pytest.mark.parametrize("backend", BACKENDS)
def test_truncated_normal_rejects_unrepresentable_output_bounds(
    backend: str,
) -> None:
    selected = asc.backend(backend)

    with pytest.raises(asc.RandomStateError, match="exactly representable"):
        random.truncated_normal(
            (8,),
            state=random.random_state(7, backend=backend),
            lower=1.0 + 2**-25,
            upper=1.0 + 2**-24,
            dtype=selected.xp.float32,
        )


@pytest.mark.parametrize("backend", BACKENDS)
def test_initializer_parameters_must_survive_output_dtype(
    backend: str,
) -> None:
    selected = asc.backend(backend)
    state = random.random_state(7, backend=backend)

    for call in (
        lambda: random.truncated_normal(
            (2,), state=state, std=1e-50, dtype=selected.xp.float32
        ),
        lambda: random.truncated_normal(
            (2,), state=state, mean=1e40, dtype=selected.xp.float32
        ),
        lambda: random.orthogonal(
            (2, 2), state=state, gain=1e40, dtype=selected.xp.float32
        ),
    ):
        with pytest.raises(asc.RandomStateError, match="representable"):
            call()


@pytest.mark.parametrize("backend", BACKENDS)
def test_bernoulli_rejects_probability_below_portable_precision(
    backend: str,
) -> None:
    with pytest.raises(asc.RandomStateError, match="representable"):
        random.bernoulli(
            (2,),
            state=random.random_state(7, backend=backend),
            probability=1e-50,
        )


def test_numpy_uniform_preserves_low_precision_half_open_bound() -> None:
    values, _ = random.uniform(
        (100_000,),
        state=random.random_state(7, backend="numpy"),
        dtype=numpy.float16,
    )

    assert bool(numpy.all(values >= numpy.float16(0)))
    assert bool(numpy.all(values < numpy.float16(1)))


@pytest.mark.backend("torch")
def test_torch_counter_states_do_not_collapse_seed_and_counter_pairs() -> None:
    from asc.backends import _state

    version = importlib.metadata.version("torch")
    pairs = (((1, 1), (2, 0)), ((0, 78_505), (0, 96_158)))
    for first_pair, second_pair in pairs:
        first, _ = random.random(
            (16,),
            state=random.RandomState(
                "torch", _state.CounterKey("torch", *first_pair), version
            ),
        )
        second, _ = random.random(
            (16,),
            state=random.RandomState(
                "torch", _state.CounterKey("torch", *second_pair), version
            ),
        )

        assert not bool(asc.backend("torch").xp.all(first == second))


@pytest.mark.backend("torch")
def test_torch_full_endpoint_randint_is_uniform_with_negative_low() -> None:
    import torch

    low = -(2**62)
    values, _ = random.randint(
        low,
        2**63,
        (100_000,),
        state=random.random_state(17, backend="torch"),
        dtype=torch.int64,
    )
    negative_fraction = float((values < 0).to(torch.float64).mean())

    assert bool(torch.all(values >= low))
    assert 0.32 < negative_fraction < 0.35


@pytest.mark.parametrize("backend", ("numpy", "torch"))
def test_counter_state_rejects_draw_when_successor_would_overflow(
    backend: str,
) -> None:
    from asc.backends import _state

    state = random.RandomState(
        backend,  # type: ignore[arg-type]
        _state.CounterKey(backend, seed=1, counter=2**32 - 1),  # type: ignore[arg-type]
        importlib.metadata.version(backend),
    )

    with pytest.raises(asc.RandomStateError, match="exhausted"):
        random.uniform((1,), state=state)


@pytest.mark.backend("jax")
def test_jax_randint_rejects_bounds_outside_requested_dtype() -> None:
    selected = asc.backend("jax")
    state = random.random_state(1, backend="jax")

    for low, high in ((0, 1000), (-129, 127)):
        with pytest.raises(asc.RandomStateError, match="dtype range"):
            random.randint(
                low,
                high,
                (4,),
                state=state,
                dtype=selected.xp.int8,
            )


@pytest.mark.backend("jax")
@pytest.mark.parametrize("dtype_name", ("int8", "int16", "int32"))
def test_jax_randint_supports_max_plus_one_endpoint(dtype_name: str) -> None:
    selected = asc.backend("jax")
    dtype = getattr(selected.xp, dtype_name)
    information = selected.xp.iinfo(dtype)
    state = random.random_state(1, backend="jax")

    singleton, state = random.randint(
        int(information.max),
        int(information.max) + 1,
        (8,),
        state=state,
        dtype=dtype,
    )
    full_range, _ = random.randint(
        int(information.min),
        int(information.max) + 1,
        (32,),
        state=state,
        dtype=dtype,
    )

    assert singleton.dtype == dtype
    assert bool(selected.xp.all(singleton == information.max))
    assert full_range.dtype == dtype
    assert full_range.shape == (32,)


@pytest.mark.backend("jax")
def test_jax_full_endpoint_randint_is_uniform_with_negative_low() -> None:
    selected = asc.backend("jax")
    low = -(2**30)
    values, _ = random.randint(
        low,
        2**31,
        (100_000,),
        state=random.random_state(17, backend="jax"),
        dtype=selected.xp.int32,
    )
    negative_fraction = float(
        selected.xp.mean(
            selected.xp.astype(values < 0, selected.xp.float32, copy=True)
        )
    )

    assert bool(selected.xp.all(values >= low))
    assert 0.32 < negative_fraction < 0.35


@pytest.mark.backend("torch")
def test_torch_randint_supports_int64_max_plus_one_endpoint() -> None:
    selected = asc.backend("torch")
    information = selected.xp.iinfo(selected.xp.int64)
    state = random.random_state(1, backend="torch")

    singleton, state = random.randint(
        int(information.max),
        int(information.max) + 1,
        (8,),
        state=state,
        dtype=selected.xp.int64,
    )
    full_range, _ = random.randint(
        int(information.min),
        int(information.max) + 1,
        (128,),
        state=state,
    )

    assert singleton.dtype == selected.xp.int64
    assert bool(selected.xp.all(singleton == information.max))
    assert full_range.dtype == selected.xp.int64
    assert full_range.shape == (128,)


def test_seed_and_state_document_validation() -> None:
    for seed in (-1, True, 2**32, 2**100):
        with pytest.raises(asc.RandomStateError, match="32-bit"):
            random.random_state(seed, backend="numpy")
    with pytest.raises(asc.RandomStateError, match="backend"):
        random.random_state(1, backend="invalid")
    for document in (
        "not json",
        '{"schema":2}',
        '{"schema":true,"backend":"numpy","version":"1","state":{}}',
        '{"schema":1.0,"backend":"numpy","version":"1","state":{}}',
        '{"schema":1,"backend":[],"version":"1","state":{}}',
        '{"schema":1,"backend":{},"version":"1","state":{}}',
        '{"schema":1,"backend":"bad","version":"1","state":{}}',
        '{"schema":1,"backend":"numpy","version":"0","state":{"seed":1,"counter":0}}',
    ):
        with pytest.raises(asc.RandomStateError):
            random.RandomState.from_json(document)


@pytest.mark.backend("jax")
def test_jax_random_state_progresses_under_jit() -> None:
    state = random.random_state(5, backend="jax")
    compiled = asc.jit(
        lambda current: random.uniform((4,), state=current), backend="jax"
    )
    values, next_state = compiled(state)
    assert values.shape == (4,)
    assert isinstance(next_state, random.RandomState)
    assert (
        len(asc.jit(lambda current: current.split(2), backend="jax")(state))
        == 2
    )


def test_random_does_not_advance_global_numpy_state() -> None:
    numpy.random.seed(123)
    expected = numpy.random.random(3)
    numpy.random.seed(123)
    state = random.random_state(3, backend="numpy")
    random.normal((10,), state=state)
    numpy.testing.assert_allclose(numpy.random.random(3), expected)


@pytest.mark.backend("torch")
def test_torch_random_allocations_ignore_process_default_device() -> None:
    import torch

    state = random.random_state(7, backend="torch")
    calls = (
        lambda: random.uniform((2,), state=state),
        lambda: random.normal((2,), state=state),
        lambda: random.truncated_normal((2,), state=state),
        lambda: random.randint(0, 3, (2,), state=state),
        lambda: random.bernoulli((2,), state=state),
        lambda: random.gamma((2,), state=state, concentration=1.0),
        lambda: random.exponential((2,), state=state),
        lambda: random.choice(3, (2,), state=state),
        lambda: random.permutation(3, state=state),
    )
    previous = torch.get_default_device()
    try:
        torch.set_default_device("meta")
        for call in calls:
            sample, _next_state = call()
            assert sample.device == torch.device("cpu")
    finally:
        torch.set_default_device(previous)


@pytest.mark.backend("torch")
def test_torch_split_derives_distinct_child_states() -> None:
    children = random.random_state(1, backend="torch").split(100_000)
    seeds = {child.key.seed for child in children}

    assert len(seeds) == len(children)
    first, _ = random.random((16,), state=children[18_630])
    second, _ = random.random((16,), state=children[40_098])
    assert not bool(asc.backend("torch").xp.all(first == second))
