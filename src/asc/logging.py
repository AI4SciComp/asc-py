# Copyright 2026 AI4SciComp contributors.
# SPDX-License-Identifier: Apache-2.0
"""Library-safe logging helpers with no root-logger configuration."""

import logging as stdlib_logging

_LOGGER = stdlib_logging.getLogger("asc")
if not any(
    isinstance(handler, stdlib_logging.NullHandler)
    for handler in _LOGGER.handlers
):
    _LOGGER.addHandler(stdlib_logging.NullHandler())


def get_logger(name: str | None = None) -> stdlib_logging.Logger:
    """Return the package logger or one of its descendants.

    Args:
        name: Optional child name such as ``"conversion"``.

    Returns:
        A standard-library logger. No handlers are added by this function.
    """
    if name is None:
        return _LOGGER
    return _LOGGER.getChild(name)


__all__ = ["get_logger"]
