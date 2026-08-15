# Copyright 2026 AI4SciComp contributors.
# SPDX-License-Identifier: Apache-2.0
"""Shared immutable state for counter-based backend random adapters."""

from __future__ import annotations

import dataclasses
import typing

CounterBackend = typing.Literal["numpy", "torch"]


@dataclasses.dataclass(frozen=True, slots=True)
class CounterKey:
    """Replayable seed and substream counter for stateful random backends."""

    backend: CounterBackend
    seed: int
    counter: int = 0
