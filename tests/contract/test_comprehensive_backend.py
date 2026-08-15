# Copyright 2026 AI4SciComp contributors.
# SPDX-License-Identifier: Apache-2.0
"""Comprehensive backend, configuration, diagnostics, and error contracts."""

from __future__ import annotations

import dataclasses
import importlib
import json
import logging
import sys

import numpy
import pytest

import asc
from asc import config
from tests import helpers


@pytest.mark.parametrize("name", ("numpy", "torch", "jax"))
def test_backend_object_and_capabilities(name: str) -> None:
    selected = asc.backend(name)
    assert selected.name == name
    if name == "numpy":
        assert selected.device is None
    else:
        assert selected.device is not None
    assert selected.xp.__array_api_version__ == "2024.12"
    assert selected.linalg is not None
    assert selected.fft is not None
    assert selected.random.__name__ == "asc.random"
    assert selected.updates.__name__ == "asc.updates"
    assert selected.conversion.__name__ == "asc.conversion"
    assert selected.autodiff.__name__ == "asc.autodiff"
    assert selected.compilation.__name__ == "asc.compilation"
    assert selected.zeros((2,)).shape == (2,)
    assert selected.ones((2,)).shape == (2,)
    assert selected.full((2,), 3).shape == (2,)
    assert selected.asarray([1, 2], copy=True).shape == (2,)
    assert asc.backend_of(selected.asarray([1])) == name
    assert asc.is_array(selected.asarray([1]))
    assert asc.has_capability(name, asc.Capability.ARRAY_API)
    asc.require_capability(name, "linalg")
    with pytest.raises(dataclasses.FrozenInstanceError):
        selected.name = "numpy"  # type: ignore[misc]


def test_discovery_and_capability_errors() -> None:
    assert "numpy" in asc.available_backends()
    assert not asc.is_array(object())
    assert not asc.has_capability("numpy", "not-a-capability")
    with pytest.raises(asc.CapabilityNotSupportedError, match="unknown"):
        asc.require_capability("numpy", "not-a-capability")
    with pytest.raises(asc.NamespaceError, match="name"):
        asc.backend("invalid")  # type: ignore[arg-type]
    with pytest.raises(asc.NamespaceError, match="unsupported"):
        asc.backend_info("invalid")
    with pytest.raises(asc.DTypeError):
        asc.backend("numpy", dtype=object())
    with pytest.raises(asc.DeviceError):
        asc.backend("numpy", device="cuda")


def test_configuration_records_validate_and_freeze() -> None:
    state = asc.random_state(1, backend="numpy")
    context = config.ArrayContext(
        "numpy",
        dtype=numpy.float32,
        random_state=state,
        precision=config.PrecisionPolicy.STRICT,
        copy=config.CopyPolicy.ALWAYS,
    )
    assert context.backend == "numpy"
    assert config.DataLoaderConfig(batch_size=None).batch_size is None
    assert config.NpyOptions(mmap_mode="r").mmap_mode == "r"
    assert config.NpzOptions(compressed=False).compressed is False
    assert config.CsvOptions(delimiter=";").delimiter == ";"
    assert config.Hdf5Options(chunks=(2, 2)).chunks == (2, 2)
    assert config.MatOptions(do_compression=True).do_compression
    invalid = [
        lambda: config.ArrayContext("numpy", device="cuda"),
        lambda: config.ArrayContext("numpy", precision="strict"),
        lambda: config.ArrayContext("numpy", copy="always"),
        lambda: config.DataLoaderConfig(batch_size=0),
        lambda: config.DataLoaderConfig(batch_size=None, drop_last=True),
        lambda: config.DataLoaderConfig(shuffle=1),
        lambda: config.NpyOptions(mmap_mode="bad"),
        lambda: config.NpzOptions(compressed=1),
        lambda: config.CsvOptions(delimiter="xx"),
        lambda: config.CsvOptions(delimiter="e"),
        lambda: config.CsvOptions(delimiter="I"),
        lambda: config.CsvOptions(delimiter="y"),
        lambda: config.CsvOptions(header=("",)),
        lambda: config.CsvOptions(header=("x,y",)),
        lambda: config.CsvOptions(header=("x\ny",)),
        lambda: config.Hdf5Options(chunks=(0,)),
        lambda: config.Hdf5Options(compression=""),
        lambda: config.Hdf5Options(compression="gzip", chunks=False),
        lambda: config.MatOptions(do_compression=1),
    ]
    if helpers.has_backend("torch"):
        invalid.append(
            lambda: config.ArrayContext(
                "numpy", random_state=asc.random_state(1, backend="torch")
            )
        )
    for factory in invalid:
        with pytest.raises(asc.AscError):
            factory()


def test_creation_context_and_extension_validation() -> None:
    selected = asc.backend("numpy")
    extension = config.ExtensionHandle("random", object())
    context = config.CreationContext(
        selected.xp,
        "numpy",
        dtype=selected.xp.float32,
        extensions=(extension,),
    )
    assert context.precision is config.PrecisionPolicy.STRICT
    assert (
        config.CreationContext(selected.xp, "numpy").precision
        is config.PrecisionPolicy.INHERIT
    )
    with pytest.raises(asc.ContextError, match="backend"):
        config.CreationContext(selected.xp, "jax")
    with pytest.raises(asc.ContextError, match="dtype"):
        config.CreationContext(selected.xp, "numpy", dtype=object())
    with pytest.raises(asc.ContextError, match="unique"):
        config.CreationContext(
            selected.xp, "numpy", extensions=(extension, extension)
        )
    with pytest.raises(asc.ContextError, match="name"):
        config.ExtensionHandle(" bad ", object())


def test_diagnostics_are_small_and_import_safe() -> None:
    before = set(sys.modules)
    document = asc.diagnostics()
    assert document["asc_version"] == "0.1.0"
    assert document["array_api_version"] == "2024.12"
    assert set(document["backends"]) == {"numpy", "torch", "jax"}
    json.dumps(document)
    for optional_module in ("torch", "jax", "h5py", "scipy"):
        assert (optional_module in sys.modules) == (optional_module in before)


def test_logging_is_library_safe() -> None:
    module = importlib.import_module("asc.logging")
    logger = module.get_logger()
    assert logger.name == "asc"
    assert any(
        isinstance(handler, logging.NullHandler) for handler in logger.handlers
    )
    assert logging.getLogger().handlers == logging.getLogger().handlers


def test_error_hierarchy_and_public_manifest() -> None:
    error_types = [
        asc.BackendError,
        asc.BackendUnavailableError,
        asc.CapabilityNotSupportedError,
        asc.ConversionError,
        asc.DTypeError,
        asc.DeviceError,
        asc.RandomStateError,
        asc.IndexUpdateError,
        asc.DuplicateIndexError,
        asc.DataError,
        asc.DatasetError,
        asc.DataSpecError,
        asc.CollationError,
        asc.DataLoaderError,
        asc.DataSplitError,
        asc.DataFormatError,
    ]
    assert all(
        issubclass(error_type, asc.AscError) for error_type in error_types
    )
    assert set(asc.PUBLIC_EXPORTS).issubset(set(asc.__all__))
