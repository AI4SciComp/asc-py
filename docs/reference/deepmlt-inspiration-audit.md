# DeepMLT Inspiration Audit

Status: Complete read-only inventory. Audited: 2026-08-11.

## Source and method

The audit inspected `escapetiger/deepmlt` `main` at commit
`7dab7906038a2cfbac83801843b4a9148ad40ece` (2024-07-02) in a shallow `/tmp`
clone. It was not modified or vendored. DeepMLT has no repository license file
or packaging metadata, so no source, fixtures, or data are copied.

## Array API inventory

`deepmlt/dtensor` presents a broad dynamic tensor facade. Its public vocabulary
includes creation (`tensor`, `as_tensor`, empty/zeros/ones/full families,
`eye`, `arange`, `linspace`, `meshgrid`), attributes and contexts, indexing,
shape manipulation, arithmetic, comparisons, transcendental functions,
reductions, sorting/searching, linalg, padding/convolution, activations,
randomness/initializers, conversion and persistence, and PyTorch autodiff.

The corresponding backend files are `api/dtensor/numpy.py` and
`api/dtensor/pytorch.py`. Standard creation, elementwise, manipulation,
reduction, search, sorting, and core linalg vocabulary maps to `Backend.xp` or
`Backend.linalg`. The comprehensive asc extension contract adopts and
normalizes `diag`, flatten/ravel, multi-index conversion, padding, activations,
convolution/moving mean, comparisons, `einsum`, `kron`, `gkron`, least squares,
random distributions/initializers, functional updates, and autodiff. It adds
JAX and FFT. Native arrays are always retained.

## Data API inventory

DeepMLT's `data/` exposes:

- `Dataset`, `TensorDataset`, `SequenceTensorDataset`,
  `MappingTensorDataset`, `ConcatDataset`, `Subset`, and `partition`;
- `BatchLoader` and its single-process iterator;
- `TreeLoader` with `min_size`, `max_size_cycle`, `max_size`, and `sequential`
  modes plus per-loader limits;
- `DataModule` with stage dictionaries and loader construction;
- NPY, HDF5, and MATLAB save/load helpers.

asc adopts these useful concepts but strengthens them: typed map and iterable
datasets, immutable metadata/schema validation, deterministic explicit-state
splits and samplers, recursive backend-preserving collation, validated
single-process loading, PyTree-preserving `CombinedLoader`, four-stage
`DataModule`, transforms, streaming statistics, and safe atomic NPY/NPZ/CSV,
HDF5, and MATLAB I/O. No mutable default mappings, assertions as validation,
global NumPy shuffling, default pickle, eager optional imports, or constructor
I/O are retained.

## Designs explicitly rejected

DeepMLT's `_DTensorEngine` reads `DEEPMLT_BACKEND`, defaults to PyTorch,
dynamically imports a registry, and allows global/thread-local switching.
`dtensor.Tensor` is replaced with backend subclasses. PyTorch default context
calls backend-wide dtype/device setters. These designs are rejected in favor of
native arrays, immutable `Backend`/`ArrayContext` values, and input-driven
namespace discovery.

DeepMLT's PyTorch `to_numpy` silently performs detach and CPU transfer; its
padding and random paths may convert through NumPy; random helpers may use
global state. asc requires named conversion/I/O boundaries, explicit graph and
transfer policies, backend-native state, and no hidden NumPy fallback.

Geometry, PDE problems, numerical solvers, physics data generation, modules,
neural operators, optimizers, trainers, visualization, and experiment drivers
remain outside asc-py and belong in domain packages.

## Migration rules

- Replace backend setters with `backend(name)` or infer from native inputs.
- Replace wrapper tensors with native arrays and standard namespaces.
- Pass dtype/device/copy/precision/random policies explicitly.
- Retain the return of functional updates; input arrays are never mutated.
- Expect reproducibility only within one backend/version/configuration.
- Use explicit detach/transfer flags before NumPy or persistence boundaries.
- Replace `BatchLoader`, `TreeLoader`, and stage dictionaries with validated
  asc data objects and immutable loader configuration.
- Treat mixed backends, unsupported devices/layouts, unsafe object loading, and
  duplicate nondeterministic set updates as errors before computation.

The audit adopts behavioral questions, not implementation: backend parity,
native identity, exact dtype/device, no mutation, deterministic state
progression, graph integrity, loader reset/length behavior, and persistence
round trips are all exercised by asc's own tests.
