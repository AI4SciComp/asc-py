# Copyright 2026 AI4SciComp contributors.
# SPDX-License-Identifier: Apache-2.0

import contextlib
import itertools
import typing
import warnings

import numpy
import pytest

import asc
from asc import typing as asc_typing
from tests import helpers

BACKEND_PAIRS = tuple(itertools.product(helpers.BACKENDS, repeat=2))


class _JaxConfig(typing.Protocol):  # pylint: disable=too-few-public-methods
    """Typed configuration query used by conversion tests."""

    def read(self, name: str) -> bool:
        """Read a named JAX configuration flag."""
        ...  # pylint: disable=unnecessary-ellipsis


class _TorchForwardAD(typing.Protocol):
    """Typed forward-mode construction surface used by tests."""

    def dual_level(self) -> contextlib.AbstractContextManager[object]:
        """Enter a forward-mode level."""
        ...  # pylint: disable=unnecessary-ellipsis

    def make_dual(self, primal: object, tangent: object) -> object:
        """Create a forward-mode dual tensor."""
        ...  # pylint: disable=unnecessary-ellipsis


@pytest.mark.parametrize(("source_backend", "target_backend"), BACKEND_PAIRS)
def test_conversion_preserves_value_shape_dtype_and_backend(
    source_backend: asc_typing.BackendName,
    target_backend: asc_typing.BackendName,
) -> None:
    source = helpers.float_array(source_backend, [[1.0, 2.0], [3.0, 4.0]])
    target_namespace = helpers.namespace(target_backend)
    context = asc.CreationContext(
        target_namespace,
        target_backend,
        dtype=target_namespace.float32,
    )
    if target_backend == "array_api_strict":
        with pytest.raises(asc.ConversionError, match="conformance oracle"):
            asc.convert_array(
                source,
                destination=context,
                copy=asc.CopyPolicy.ALWAYS,
            )
        return

    result = asc.convert_array(
        source,
        destination=context,
        copy=asc.CopyPolicy.ALWAYS,
    )

    assert result.shape == source.shape
    assert result.dtype == target_namespace.float32
    assert asc.backend_info(result).name == target_backend
    assert result is not source
    numpy.testing.assert_array_equal(
        helpers.as_numpy(result),
        [[1.0, 2.0], [3.0, 4.0]],
    )


@pytest.mark.parametrize("backend", helpers.BACKENDS)
def test_same_backend_never_copy_aliases(
    backend: asc_typing.BackendName,
) -> None:
    source = helpers.float_array(backend, [1.0, 2.0])
    target_namespace = helpers.namespace(backend)
    context = asc.CreationContext(target_namespace, backend)

    if backend == "array_api_strict":
        with pytest.raises(asc.ConversionError, match="conformance oracle"):
            asc.convert_array(
                source,
                destination=context,
                copy=asc.CopyPolicy.NEVER,
            )
    else:
        result = asc.convert_array(
            source,
            destination=context,
            copy=asc.CopyPolicy.NEVER,
        )
        assert result is source


@pytest.mark.backend("torch")
def test_cross_backend_requires_copy_and_rejects_active_graph() -> None:
    import torch

    source = helpers.float_array("numpy", [1.0])
    target_namespace = helpers.namespace("torch")
    context = asc.CreationContext(target_namespace, "torch")
    with pytest.raises(
        asc.ConversionError, match=r"requires CopyPolicy\.ALWAYS"
    ):
        asc.convert_array(
            source,
            destination=context,
            copy=asc.CopyPolicy.IF_NEEDED,
        )

    differentiable = torch.tensor([1.0], requires_grad=True)
    numpy_namespace = helpers.namespace("numpy")
    numpy_context = asc.CreationContext(numpy_namespace, "numpy")
    with pytest.raises(
        asc.ConversionError, match="must be detached explicitly"
    ):
        asc.convert_array(
            differentiable,
            destination=numpy_context,
            copy=asc.CopyPolicy.ALWAYS,
        )


@pytest.mark.backend("torch")
def test_cross_backend_rejects_torch_forward_mode_dual() -> None:
    import torch

    torch_forward_ad = typing.cast(_TorchForwardAD, torch.autograd.forward_ad)
    destination = asc.CreationContext(
        helpers.namespace("numpy"),
        "numpy",
    )
    with torch_forward_ad.dual_level():
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"`torch\.jit\.script`.*",
                category=DeprecationWarning,
            )
            dual = torch_forward_ad.make_dual(
                torch.tensor([1.0]),
                torch.tensor([2.0]),
            )
        with pytest.raises(
            asc.ConversionError,
            match="must be detached explicitly",
        ):
            asc.convert_array(
                dual,
                destination=destination,
                copy=asc.CopyPolicy.ALWAYS,
            )


@pytest.mark.parametrize(
    ("copy", "is_copy"),
    [
        (asc.CopyPolicy.ALWAYS, True),
        (asc.CopyPolicy.IF_NEEDED, False),
        (asc.CopyPolicy.NEVER, False),
    ],
)
@pytest.mark.backend("torch")
def test_torch_same_backend_conversion_preserves_graph_without_warning(
    copy: asc.CopyPolicy,
    is_copy: bool,
) -> None:
    import torch

    source = torch.tensor([1.0], requires_grad=True)
    destination = asc.CreationContext(helpers.namespace("torch"), "torch")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = asc.convert_array(
            source,
            destination=destination,
            copy=copy,
        )

    assert (result is not source) is is_copy
    native_result = typing.cast(torch.Tensor, result)
    assert native_result.requires_grad is True
    if is_copy:
        assert native_result.grad_fn is not None


@pytest.mark.backend("jax")
def test_jax_conversion_rejects_unavailable_float64_context() -> None:
    import jax
    import jax.numpy

    jax_config = typing.cast(_JaxConfig, jax.config)
    if jax_config.read("jax_enable_x64"):
        destination = asc.CreationContext(
            helpers.namespace("jax"),
            "jax",
            dtype=jax.numpy.float64,
        )
        source = helpers.float_array("numpy", [1.0])
        result = asc.convert_array(
            source,
            destination=destination,
            copy=asc.CopyPolicy.ALWAYS,
        )
        assert result.dtype == jax.numpy.float64
    else:
        with pytest.raises(asc.ContextError, match="release surface"):
            asc.CreationContext(
                helpers.namespace("jax"),
                "jax",
                dtype=jax.numpy.float64,
            )
