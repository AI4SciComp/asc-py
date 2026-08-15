# Copyright 2026 AI4SciComp contributors.
# SPDX-License-Identifier: Apache-2.0

import numpy
import pytest

import asc
from asc import typing as asc_typing
from tests import helpers


class _ProtocolArrayImpostor:
    """Array-protocol object that is not a native backend array."""

    dtype = numpy.dtype("float32")
    shape = (1,)
    device = "cpu"

    def __array_namespace__(self, *, api_version: str | None = None) -> object:
        del api_version
        return asc.backend("numpy").xp


@pytest.mark.parametrize("backend", helpers.BACKENDS)
def test_namespace_identity_and_revision(
    backend: asc_typing.BackendName,
) -> None:
    array = helpers.float_array(backend, [1.0, 2.0])
    namespace = asc.array_namespace(array)

    assert asc.backend_info(namespace).name == backend
    assert namespace.__array_api_version__ == asc.ARRAY_API_VERSION


def test_namespace_rejects_missing_and_scalar_inputs() -> None:
    with pytest.raises(
        asc.NamespaceError,
        match="at least one native array is required",
    ):
        asc.array_namespace()
    with pytest.raises(
        asc.NamespaceError,
        match="Python scalars and None do not select a backend",
    ):
        asc.array_namespace(1.0)

    array = helpers.float_array("numpy", [1.0])
    assert asc.array_namespace(array, 1.0, None) is not None


def test_namespace_rejects_protocol_only_array_impostors() -> None:
    impostor = _ProtocolArrayImpostor()

    assert not asc.is_array(impostor)
    with pytest.raises(asc.NamespaceError, match="supported native array"):
        asc.array_namespace(impostor)


def test_namespace_accepts_native_numpy_zero_dimensional_scalar() -> None:
    scalar = numpy.asarray([1.0], dtype=numpy.float32)[0]

    assert asc.is_array(scalar)
    assert asc.backend_of(scalar) == "numpy"


def test_namespace_rejects_another_revision() -> None:
    array = helpers.float_array("numpy", [1.0])
    with pytest.raises(
        asc.NamespaceError,
        match=r"only Array API revision '2024\.12' is supported",
    ):
        asc.array_namespace(array, api_version="2025.12")


@pytest.mark.backend("torch")
def test_namespace_rejects_mixed_backends_before_computation() -> None:
    left = helpers.float_array("numpy", [1.0])
    right = helpers.float_array("torch", [2.0])
    with pytest.raises(
        asc.MixedBackendError,
        match=r"observed \('numpy', 'torch'\)",
    ):
        asc.array_namespace(left, right)


def test_backend_info_rejects_an_unknown_value() -> None:
    with pytest.raises(asc.NamespaceError, match="unsupported backend name"):
        asc.backend_info("cupy")
    with pytest.raises(asc.NamespaceError, match="unsupported namespace"):
        asc.backend_info(object())


def test_namespace_rejects_masked_arrays() -> None:
    with pytest.raises(
        asc.UnsupportedCapabilityError,
        match="dense CPU",
    ):
        asc.array_namespace(numpy.ma.array([1.0], mask=[False]))


@pytest.mark.parametrize(
    "array",
    (
        numpy.asarray(["value"]),
        numpy.asarray([object()], dtype=object),
    ),
)
def test_namespace_rejects_non_numeric_arrays(array: object) -> None:
    with pytest.raises(asc.UnsupportedCapabilityError, match="dense CPU"):
        asc.array_namespace(array)


@pytest.mark.backend("torch")
def test_namespace_rejects_unsupported_torch_arrays() -> None:
    import torch

    arrays = (
        torch.sparse_coo_tensor(
            torch.tensor([[0, 1]]),
            torch.tensor([1.0, 2.0]),
            (2,),
            check_invariants=True,
        ),
        torch.empty((1,), device="meta"),
    )
    for array in arrays:
        with pytest.raises(
            asc.UnsupportedCapabilityError,
            match="dense CPU",
        ):
            asc.array_namespace(array)
