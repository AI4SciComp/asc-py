# Release notes: 0.1.0

The first release defines a native-array foundation over Python Array API
2024.12. It includes explicit backend/context selection, normalized linalg and
FFT, portable numerical operations, functional updates, explicit random state,
Torch/JAX autodiff, JAX JIT, PyTrees, and a complete backend-neutral data and
safe persistence package.

NumPy is required; Torch, JAX, HDF5, and MATLAB are independent extras. The
supported execution boundary is dense CPU on Linux x86-64. Provisional platform
workflows exist for Windows and macOS. See the [release-readiness
report](../reports/release-readiness-v0.1.0.md), [compatibility
matrix](../reference/support-matrix.md), [citation file](../../CITATION.cff),
and [license](../../LICENSE).

## Deprecations

Version 0.1.0 is the first release and contains no deprecated public APIs.
Future deprecations will appear here, in the changelog, and in migration notes.
