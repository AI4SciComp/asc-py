# Exceptions

All package-defined contract failures derive from {py:class}`asc.AscError`.
Catch the narrowest applicable class:

| Family | Public exceptions | Typical recovery |
|---|---|---|
| Backend | `BackendUnavailableError`, `CapabilityNotSupportedError` | Install the named extra or select a supported capability |
| Array | `NamespaceError`, `MixedBackendError`, `DTypeError`, `DeviceError` | Align backend, dtype, layout, and CPU placement |
| Conversion | `ConversionError` | Request copy, transfer, or detach explicitly |
| Updates/random | `IndexContractError`, `DuplicateIndexError`, `RandomStateError` | Correct indices, duplicates, state, shape, or bounds |
| Data | `DatasetError`, `DataSpecError`, `DataLoaderError`, `DataFormatError` | Correct structure, configuration, or safe file format |

Exceptions name the failing operation and avoid including array contents or
sensitive data. Backend exceptions are never hidden behind a NumPy fallback.
