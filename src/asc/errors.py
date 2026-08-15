# Copyright 2026 AI4SciComp contributors.
# SPDX-License-Identifier: Apache-2.0
"""Stable public exception hierarchy for :mod:`asc`."""


class AscError(Exception):
    """Base class for every asc contract error."""


class BackendError(AscError):
    """Base class for backend discovery and execution errors."""


class NamespaceError(BackendError):
    """Raised when an Array API namespace cannot be selected."""


class MixedBackendError(NamespaceError):
    """Raised when one operation receives multiple native backends."""


class BackendUnavailableError(BackendError):
    """Raised when a requested optional backend is not installed."""


class CapabilityNotSupportedError(BackendError):
    """Raised when a backend does not implement a requested capability."""


# Compatibility spelling retained from the pre-release vertical slice.
UnsupportedCapabilityError = CapabilityNotSupportedError


class ContextError(AscError):
    """Raised when an immutable context is inconsistent."""


class ConversionError(AscError):
    """Raised when an explicit conversion cannot satisfy its contract."""


class DTypeError(AscError):
    """Raised when an operation does not support an input dtype."""


class DeviceError(AscError):
    """Raised when an explicit device request cannot be satisfied."""


class RandomStateError(AscError):
    """Raised when explicit random state or sampling input is invalid."""


class IndexUpdateError(AscError):
    """Raised when a functional index or scatter update is invalid."""


class DuplicateIndexError(IndexUpdateError):
    """Raised when duplicate indices make a set update ambiguous."""


# Compatibility spelling retained from the pre-release vertical slice.
IndexContractError = IndexUpdateError


class DataError(AscError):
    """Base class for dataset, loader, schema, and format errors."""


class DatasetError(DataError):
    """Raised when a dataset contract is invalid."""


class DataSpecError(DataError):
    """Raised when data does not match an immutable schema."""


class CollationError(DataError):
    """Raised when samples cannot be collated without losing structure."""


class DataLoaderError(DataError):
    """Raised for invalid loader configuration or iteration behavior."""


class DataSplitError(DataError):
    """Raised when a deterministic split contract is invalid."""


class DataFormatError(DataError):
    """Raised for unsafe, unsupported, or malformed persistence data."""


__all__ = [
    "AscError",
    "BackendError",
    "BackendUnavailableError",
    "CapabilityNotSupportedError",
    "CollationError",
    "ContextError",
    "ConversionError",
    "DTypeError",
    "DataError",
    "DataFormatError",
    "DataLoaderError",
    "DataSpecError",
    "DataSplitError",
    "DatasetError",
    "DeviceError",
    "DuplicateIndexError",
    "IndexContractError",
    "IndexUpdateError",
    "MixedBackendError",
    "NamespaceError",
    "RandomStateError",
    "UnsupportedCapabilityError",
]
