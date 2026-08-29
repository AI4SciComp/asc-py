# Copyright 2026 AI4SciComp contributors.
# SPDX-License-Identifier: Apache-2.0
"""PyTorch-only random helpers for intervals wider than signed int64."""

from __future__ import annotations

import math

import torch

from asc import errors
from asc import typing as asc_typing


def _random_int64_bits(
    shape: asc_typing.Shape,
    generator: torch.Generator,
) -> torch.Tensor:
    """Sample every signed 64-bit bit pattern with equal probability."""
    lower_bits = torch.randint(
        0, 2**32, shape, generator=generator, dtype=torch.int64, device="cpu"
    )
    upper_bits = torch.randint(
        0, 2**32, shape, generator=generator, dtype=torch.int64, device="cpu"
    )
    return torch.bitwise_or(upper_bits << 32, lower_bits)


def randint_max_endpoint(
    shape: asc_typing.Shape,
    *,
    generator: torch.Generator,
    low: int,
) -> torch.Tensor:
    """Sample ``[low, 2**63)`` without overflowing Torch's range width."""
    minimum = torch.iinfo(torch.int64).min
    if low == minimum:
        return _random_int64_bits(shape, generator)
    if low >= 0:
        reflected = torch.randint(
            minimum,
            -low,
            shape,
            generator=generator,
            dtype=torch.int64,
            device="cpu",
        )
        return torch.bitwise_not(reflected)

    count = math.prod(shape)
    result = torch.empty(count, dtype=torch.int64, device="cpu")
    remaining = torch.arange(count, dtype=torch.int64, device="cpu")
    for _ in range(1024):
        if remaining.numel() == 0:
            return result.reshape(shape)
        proposals = _random_int64_bits((remaining.numel(),), generator)
        accepted = proposals >= low
        result[remaining[accepted]] = proposals[accepted]
        remaining = remaining[~accepted]
    raise errors.RandomStateError("torch randint: sampling did not converge")


__all__ = ["randint_max_endpoint"]
