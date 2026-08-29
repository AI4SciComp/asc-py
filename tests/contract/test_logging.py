# Copyright 2026 AI4SciComp contributors.
# SPDX-License-Identifier: Apache-2.0

import logging

from asc import logging as asc_logging


def test_logging_is_library_safe() -> None:
    root_handlers = tuple(logging.getLogger().handlers)

    package_logger = asc_logging.get_logger()
    child_logger = asc_logging.get_logger("conversion")

    assert tuple(logging.getLogger().handlers) == root_handlers
    assert child_logger.parent is package_logger
    assert (
        sum(
            isinstance(handler, logging.NullHandler)
            for handler in package_logger.handlers
        )
        == 1
    )
