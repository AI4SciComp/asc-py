# Copyright 2026 AI4SciComp contributors.
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import typing

import pytest

import asc
from tests import helpers


def _set_attribute(value: object, name: str, replacement: object) -> None:
    setattr(value, name, replacement)


def test_creation_context_is_frozen_and_validates_identity() -> None:
    selected = helpers.namespace("numpy")
    context = asc.CreationContext(selected, "numpy")

    assert context.precision is asc.PrecisionPolicy.INHERIT
    assert (
        asc.CreationContext(selected, "numpy", dtype=selected.float32).precision
        is asc.PrecisionPolicy.STRICT
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        _set_attribute(context, "backend", "torch")
    with pytest.raises(asc.ContextError, match="does not match namespace"):
        asc.CreationContext(selected, "torch")
    with pytest.raises(asc.ContextError, match="only CPU"):
        asc.CreationContext(selected, "numpy", device="cuda:0")


def test_extension_handles_are_validated_and_unique() -> None:
    selected = helpers.namespace("numpy")
    extension = asc.ExtensionHandle("random", object())
    context = asc.CreationContext(
        selected,
        "numpy",
        extensions=(extension,),
    )

    assert context.extensions == (extension,)
    with pytest.raises(asc.ContextError, match="non-empty"):
        asc.ExtensionHandle(" bad ", object())
    with pytest.raises(asc.ContextError, match="must be unique"):
        asc.CreationContext(
            selected,
            "numpy",
            extensions=(extension, extension),
        )
    with pytest.raises(asc.ContextError, match="must be a tuple"):
        asc.CreationContext(
            selected,
            "numpy",
            extensions=typing.cast(tuple[asc.ExtensionHandle, ...], []),
        )
    with pytest.raises(asc.ContextError, match="ExtensionHandle values"):
        asc.CreationContext(
            selected,
            "numpy",
            extensions=typing.cast(
                tuple[asc.ExtensionHandle, ...],
                (object(),),
            ),
        )


def test_backend_info_is_frozen_and_complete() -> None:
    info = asc.backend_info("numpy")

    assert info.installed
    assert info.version is not None
    assert info.devices == ("cpu",)
    assert info.dtype_families == (
        "bool",
        "signed integer",
        "unsigned integer",
        "real floating",
        "complex floating",
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        _set_attribute(info, "name", "jax")
