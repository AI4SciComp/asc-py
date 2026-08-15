# asc-py

`asc-py` is the standards-first array and data foundation for AI4SciComp. It
keeps NumPy, PyTorch, and JAX arrays native while enforcing a frozen Python
Array API 2024.12 contract, explicit state, and explicit conversion boundaries.

NumPy is required. Torch, JAX, HDF5, and MATLAB support are independent lazy
extras. Version 0.1.0 supports dense CPU arrays and never selects a mutable
global backend, silently transfers arrays, or falls back through NumPy.

```{toctree}
:maxdepth: 2
:caption: Getting started

getting-started/installation
getting-started/quickstart
```

```{toctree}
:maxdepth: 2
:caption: User guide

user-guide/backend-selection
user-guide/array-api
user-guide/dtype-device
user-guide/conversion
user-guide/random
user-guide/autodiff-jit
user-guide/data
```

```{toctree}
:maxdepth: 2
:caption: Tutorials

tutorials/portable-computation
tutorials/data-pipeline
```

```{toctree}
:maxdepth: 2
:caption: Reference

api/index
reference/support-matrix
reference/functionality-matrix
reference/exceptions
reference/configuration
specification
```

```{toctree}
:hidden:

architecture/support-matrix
specification/functionality-matrix
```

```{toctree}
:maxdepth: 2
:caption: Development

development/architecture
development/contributing
development/testing
development/documentation
architecture/overview
architecture/portability-contract
architecture/public-api
architecture/dependency-policy
architecture/decisions/0001-import-namespace
architecture/decisions/0002-array-api-revision
architecture/decisions/0003-native-array-architecture
architecture/decisions/0004-toolchain-and-documentation
architecture/decisions/0005-license-and-support-boundary
architecture/decisions/0006-rename-import-package
architecture/decisions/0007-comprehensive-foundation-scope
architecture/decisions/0008-sphinx-documentation
reference/deepmlt-inspiration-audit
```

```{toctree}
:maxdepth: 2
:caption: Release

release/changelog
release/migration
release/release-notes
reports/release-readiness-v0.1.0
```
