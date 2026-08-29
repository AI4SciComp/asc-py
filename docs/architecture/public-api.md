# Public API for 0.1.0

Status: Frozen. Last reviewed: 2026-08-11.

The distribution is `asc-py`; users write `import asc`. The public contract is
the union of names exported by the modules below and the functionality matrix.
Underscore-prefixed names are implementation details.

## Backend and standard namespaces

Top-level discovery exports `backend`, `backend_of`, `is_array`,
`available_backends`, `backend_info`, `array_namespace`, `has_capability`, and
`require_capability`. `Backend` is immutable and exposes `xp`, `linalg`, `fft`,
creation, random, update, conversion, autodiff, and compilation facades.
`ARRAY_API_VERSION` is exactly `"2024.12"`.

Standard operations are used through `Backend.xp` or `array_namespace`; asc
does not mirror the complete standard at top level. Inputs must be dense native
arrays from one backend. Python scalars and `None` do not select a namespace.

## Explicit extensions

- `asc.ops`: diagonal construction, flatten/ravel, multi-index conversion,
  padding, activations, 1-D convolution, moving mean, comparisons, and numeric
  metadata.
- `asc.linalg`: `einsum`, `kron`, `gkron`, stable `lstsq`, and normalized
  decomposition result records in addition to `Backend.linalg`.
- `asc.metrics`: MAE, MSE, RMSE, relative L2 error, and R2 score.
- `asc.conversion`: backend conversion, NumPy and DLPack boundaries, copying,
  device transfer, detach, and stop-gradient.
- `asc.updates`: functional indexed and scatter set/add/multiply/min/max.
- `asc.random`: explicit `RandomState`, distributions, sampling, splitting,
  serialization, and initializers.
- `asc.autodiff`: `grad`, `value_and_grad`, `jacobian`, `hessian`, `jvp`, and
  `vjp` for declared Torch/JAX capabilities.
- `asc.compilation`: `jit` and `vmap` for declared capabilities.
- `asc.tree`: stable `TreeSpec`, mapping/path operations, safe registration and
  JSON, plus explicit array-tree conversion helpers.

## Data

`asc.data` exports map and iterable dataset bases; array, tuple, mapping,
concat, subset, transform, filter, and zip datasets; immutable schema records
and validators; deterministic splits and samplers; recursive conversion,
collation, and uncollation; single-process `DataLoader`; four-mode
`CombinedLoader`; `DataModule`; transforms and scalers; streaming statistics;
and NPY, NPZ, CSV, HDF5, and MATLAB persistence.

## Configuration and errors

Frozen configuration includes `ArrayContext`, `PrecisionPolicy`, copy policy,
`DataLoaderConfig`, and I/O option records. `AscError` is the stable exception
root, with backend, capability, array, update, random, data, loader, split, and
format subclasses. `diagnostics()` reports package and discoverability metadata
without importing optional backends. `__version__` is derived from installed
package metadata, and `PUBLIC_EXPORTS` freezes the supported export manifest.

All cross-backend, host, device, ownership, dtype, and graph changes are named
and explicit. Unsupported behavior raises an asc exception; no operation
silently detaches, transfers, converts, mutates, or falls back through NumPy.

Downstream repositories must also follow the
[released public API boundary](downstream-public-api.md), including its
versioning, deprecation, and dependency-direction rules.
