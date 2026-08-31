# Changelog

All notable changes are recorded here. The project follows semantic versioning
for its documented public API after the 0.1.0 contract is released.

## [Unreleased]

### Added

- Initial repository, architecture, portability contract, and packaging
  foundation for asc-py 0.1.0.
- Portable namespace discovery, explicit creation, and sum-of-squares core over
  NumPy, PyTorch, JAX, and array-api-strict native arrays.
- Immutable backend metadata, configuration, stable errors, logging, typing,
  and a typed-package marker.
- Explicit DLPack CPU conversion with required copy policy.
- Replayable random state, functional indexed addition, native autodiff, and
  native compilation extensions.
- Contract, conformance, parity, property, benchmark, documentation, and CI
  infrastructure.
- Corrected the JAX range to `>=0.6,<0.11`; 0.11 exposes only Array API 2025.12.
- Established conservative floors at array-api-compat 1.13 and PyTorch 2.13,
  with a version-neutral DLPack capsule path for NumPy 1.26 interoperability.
- Renamed the unreleased import package from `asc_py` to `asc` by explicit
  project direction.
- Tightened dense-CPU discovery, explicit dtype and seed validation, autodiff
  conversion guards, and cross-backend indexed-update parity.
- Added the complete Python Array API 2024.12 validation surface, normalized
  linalg/FFT, portable operations, metrics, conversion, functional updates,
  explicit randomness, autodiff/JIT/vmap, and PyTrees.
- Added datasets, schemas, deterministic splits and samplers, recursive
  collation, single-process loading, combined loading, DataModule lifecycle,
  transforms, streaming statistics, and safe atomic NPY/NPZ/CSV/HDF5/MAT I/O.
- Added full dependency-floor and CPython 3.12-3.14 suites, independent
  installation-profile checks, executable examples, strict documentation,
  artifact/license inspection, and 168-ID release traceability.
- Hardened explicit device and dtype validation, authorized host transfers,
  bounded far-tail random sampling, convolution parity, PyTree persistence,
  optional-backend restore errors, and atomic split registration.
- Added the Sphinx/PyData/MyST documentation site, generated public API
  inventory, strict cross-reference and docstring gates, 35 doctests,
  independent backend examples, scheduled link checking, and a pinned
  least-privilege GitHub Pages workflow.
- Enforced JAX wide-dtype failure across creation, FFT, and transforms;
  completed Torch complex/NaN unique semantics; and preserved destination
  dtypes and scalar-edge behavior across DLPack, MATLAB, HDF5, and sampling.

## [0.1.0] - 2026-08-31

The first release establishes the frozen semantic boundary documented in
`docs/architecture/portability-contract.md`.
