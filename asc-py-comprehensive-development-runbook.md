# `asc-py` Comprehensive Functionality Contract and Codex Runbook

**Repository:** `AI4SciComp/asc-py`<br>
**Target release:** `0.1.0`<br>
**Prepared:** 2026-08-10<br>
**Status:** Normative replacement for the earlier v0.1 functionality table
**Primary Codex skill:** `$port-scientific-python`

## 1. Outcome

Implement `asc-py` as the reusable Python foundation of AI4SciComp. The package
must provide:

- NumPy as the required numerical backend;
- PyTorch and JAX as independent, lazily imported optional backends;
- a standards-first array API using native backend arrays;
- explicit backend objects inspired by DeepMLT, without a mutable global backend;
- portable extensions for functionality outside the Python Array API;
- a backend-neutral data module for scientific and machine-learning data;
- errors, logging, configuration, typing, diagnostics, packaging, tests,
  benchmarks, documentation, and CI.

One Codex goal must implement every row marked **Required** below. Internal
milestones control dependency order; they do not reduce the promised scope.

## 2. Scope boundary

The `data` package belongs in `asc-py`, but domain data generation does not.

| Belongs in `asc-py` | Belongs in `asc-xde-py` |
|---|---|
| Array/backend infrastructure | PDE and kinetic-equation solvers |
| Generic datasets and iterable datasets | Meshes, basis functions, and quadrature |
| Samplers, collation, batching, and combined loaders | DG/FEM/FDM/spectral discretizations |
| Train/validation/test data organization | PDE-specific data-generation pipelines |
| Generic transforms, scaling, and statistics | Physics-informed sampling rules |
| NPY/NPZ/CSV and optional HDF5/MAT I/O | Neural operators and research models |
| PyTrees and explicit tree conversion | Training loops, optimizers, and AutoML |
| Backend-neutral metrics | Experiment-specific plotting and reporting |

Do not include geometry, PDE operators, solvers, neural-network layers, models,
optimizers, trainers, or visualization in `asc-py` 0.1.0.

## 3. Dependency and installation contract

| Installation | Required packages | Purpose |
|---|---|---|
| `pip install asc-py` | NumPy plus only justified lightweight infrastructure | General scientific computing and data handling |
| `pip install "asc-py[torch]"` | Base plus PyTorch | PyTorch arrays, autodiff, compilation, and data |
| `pip install "asc-py[jax]"` | Base plus JAX | JAX arrays, autodiff, JIT, and data |
| `pip install "asc-py[io-hdf5]"` | Base plus HDF5 dependency | HDF5 persistence |
| `pip install "asc-py[io-mat]"` | Base plus SciPy | MATLAB MAT persistence |
| `pip install "asc-py[all]"` | All optional runtime features | Users explicitly needing every backend/format |
| Development groups | Test, lint, type, docs, benchmark, and build tools | Contributors and CI only |

NumPy must always be available. PyTorch and JAX must never be co-required.
`import asc` must not import either optional backend. A missing optional feature
must raise an actionable error naming the appropriate installation extra.

The provisional portability contract is Python Array API `2024.12`. Codex must
verify it against the selected minimum and maximum NumPy, PyTorch, JAX, and
compatibility-layer versions before freezing it. The public standard is newer,
but a package may only promise the newest revision satisfied by its complete
supported dependency range.

## 4. Normative functionality matrix

### 4.1 Backend discovery, selection, and context

| ID | Public API | Requirement | Coverage |
|---|---|---|---|
| B01 | `backend(name, *, device=None, dtype=None)` | Return an immutable backend object; accepted names are `numpy`, `torch`, and `jax`; never mutate process-global state. | Required: NumPy, Torch, JAX |
| B02 | `array_namespace(*arrays)` | Return one Array API-compatible namespace inferred from native arrays; reject irreconcilable or mixed backends. | Required: NumPy, Torch, JAX, `array-api-strict` |
| B03 | `backend_of(array)` | Identify the backend without copying, moving, detaching, or materializing the array. | Required: all backends |
| B04 | `is_array(value)` | Recognize declared native array types without importing all optional backends. | Required: all backends |
| B05 | `available_backends()` | Report discoverable backends without eager initialization or device allocation. | Required |
| B06 | `backend_info(name)` | Report installed/version/availability status, devices, dtypes, Array API revision, and capabilities. | Required |
| B07 | `Backend` | Frozen object exposing `name`, `xp`, `linalg`, `fft`, `device`, `dtype`, `capabilities`, creation helpers, random API, update API, conversion API, autodiff, and compilation hooks. | Required |
| B08 | `ArrayContext` | Frozen configuration carrying backend, dtype, device, precision policy, copy policy, and optional random state. | Required |
| B09 | Dedicated adapter modules | Provide `asc.backends.numpy`, `asc.backends.torch`, and `asc.backends.jax`; Torch/JAX modules fail lazily and actionably when unavailable. | Required |
| B10 | Capability query | `has_capability(array_or_backend, capability)` and `require_capability(...)`; no false emulation through NumPy. | Required |
| B11 | Namespace inspection | Expose frozen standard revision and namespace metadata without changing backend state. | Required |
| B12 | Mixed-backend policy | Ordinary mathematical APIs reject mixed native backends; only explicit conversion APIs may cross the boundary. | Required |

### 4.2 Frozen Python Array API surface

These operations are accessed through `Backend.xp` or
`array_namespace(*arrays)`. `asc-py` must validate and normalize the selected
standard semantics; it must not create redundant top-level wrappers for every
standard function.

| ID | Standard family | Mandatory operations at the frozen revision | Coverage |
|---|---|---|---|
| A01 | Constants and namespace | `e`, `inf`, `nan`, `newaxis`, `pi`, `__array_api_version__`, and namespace inspection. | Required: all backends |
| A02 | Dtypes | `bool`, signed/unsigned integer families supported by the backend, `float32`, `float64`, `complex64`, and `complex128`, with an honest support matrix. | Required by capability |
| A03 | Creation | `arange`, `asarray`, `empty`, `empty_like`, `eye`, `from_dlpack`, `full`, `full_like`, `linspace`, `meshgrid`, `ones`, `ones_like`, `tril`, `triu`, `zeros`, `zeros_like`. | Required: all backends |
| A04 | Dtype functions | `astype`, `can_cast`, `finfo`, `iinfo`, `isdtype`, and `result_type`. | Required: all backends |
| A05 | Arithmetic elementwise | `abs`, `add`, `divide`, `floor_divide`, `multiply`, `negative`, `positive`, `pow`, `remainder`, `square`, and `subtract`. | Required: all backends |
| A06 | Exponential/logarithmic | `exp`, `expm1`, `log`, `log1p`, `log2`, `log10`, `logaddexp`, and `sqrt`. | Required: all backends |
| A07 | Trigonometric | `acos`, `asin`, `atan`, `atan2`, `cos`, `sin`, and `tan`. | Required: all backends |
| A08 | Hyperbolic | `acosh`, `asinh`, `atanh`, `cosh`, `sinh`, and `tanh`. | Required: all backends |
| A09 | Rounding and clipping | `ceil`, `clip`, `floor`, `round`, and `trunc`. | Required: all backends |
| A10 | Comparison/logical | `equal`, `greater`, `greater_equal`, `less`, `less_equal`, `logical_and`, `logical_not`, `logical_or`, `logical_xor`, and `not_equal`. | Required: all backends |
| A11 | Bitwise | `bitwise_and`, `bitwise_invert`, `bitwise_left_shift`, `bitwise_or`, `bitwise_right_shift`, and `bitwise_xor`. | Required: all backends |
| A12 | Floating/complex helpers | `conj`, `copysign`, `hypot`, `imag`, `isfinite`, `isinf`, `isnan`, `maximum`, `minimum`, `nextafter`, `real`, `sign`, and `signbit`. | Required by dtype capability |
| A13 | Indexing | Standard basic/advanced indexing semantics plus `take` and `take_along_axis` at the frozen revision. | Required: all backends |
| A14 | Manipulation | `broadcast_arrays`, `broadcast_to`, `concat`, `expand_dims`, `flip`, `moveaxis`, `permute_dims`, `repeat`, `reshape`, `roll`, `squeeze`, `stack`, `tile`, and `unstack`. | Required: all backends |
| A15 | Searching | `argmax`, `argmin`, `count_nonzero`, `nonzero`, `searchsorted`, and `where`. | Required: all backends |
| A16 | Sets | `unique_all`, `unique_counts`, `unique_inverse`, and `unique_values`. | Required: all backends |
| A17 | Sorting | `argsort` and `sort`, including descending/stable behavior promised by the frozen revision. | Required: all backends |
| A18 | Statistics/reductions | `cumulative_prod`, `cumulative_sum`, `max`, `mean`, `min`, `prod`, `std`, `sum`, and `var`. | Required: all backends |
| A19 | Utility | `all`, `any`, and `diff`. | Required: all backends |
| A20 | Core linear algebra | `matmul`, `matrix_transpose`, `tensordot`, and `vecdot`. | Required: all backends |
| A21 | Array operators | Validate standard arithmetic, comparison, bitwise, matrix multiplication, indexing, shape, dtype, device, size, and dimensionality behavior of native arrays. | Required: all backends |
| A22 | Edge cases | Zero-dimensional, empty, singleton, broadcast, non-contiguous where applicable, NaN/Inf, signed zero, complex, and scalar interactions. | Required: all applicable APIs |

### 4.3 Linear algebra and Fourier namespaces

| ID | Namespace/API | Mandatory operations | Coverage |
|---|---|---|---|
| L01 | `backend.linalg` | `cholesky`, `cross`, `det`, `diagonal`, `eigh`, `eigvalsh`, `inv`, `matmul`, `matrix_norm`, `matrix_power`, `matrix_rank`, `matrix_transpose`, `outer`, `pinv`, `qr`, `slogdet`, `solve`, `svd`, `svdvals`, `tensordot`, `trace`, `vecdot`, and `vector_norm`. | Required: NumPy, Torch, JAX CPU |
| L02 | Additional eigen API | `eig` and `eigvals` with explicit complex-output and backend-version semantics. | Required where the frozen backend range supports it; otherwise fail via capability |
| L03 | Least squares | `lstsq` returning a stable named result with solution, residuals, rank, and singular values. | Required: NumPy, Torch, JAX CPU |
| L04 | Einstein/Kronecker | Portable `einsum`, `kron`, and DeepMLT-inspired `gkron`. | Required: all backends |
| L05 | Decomposition results | Multi-result operations return documented named records with consistent field names, shapes, and dtypes. | Required |
| L06 | Batched semantics | Matrix decompositions and solves support standard leading batch dimensions where native backends do. | Required and tested |
| F01 | `backend.fft` | `fft`, `ifft`, `fftn`, `ifftn`, `rfft`, `irfft`, `rfftn`, `irfftn`, `hfft`, `ihfft`, `fftfreq`, `rfftfreq`, `fftshift`, and `ifftshift`. | Required: NumPy, Torch, JAX CPU |
| F02 | FFT semantics | Freeze normalization, axes, complex dtype, round-trip tolerance, frequency dtype/device, JIT, and gradient behavior. | Required |

### 4.4 Portable extensions inspired by DeepMLT

| ID | Public API | Required semantics | Coverage |
|---|---|---|---|
| E01 | `asc.ops.diag` | Construct a matrix from a one-dimensional diagonal; distinct from extracting a diagonal. | Required: all backends |
| E02 | `asc.ops.flatten`, `ravel` | Return one-dimensional arrays with explicit copy/alias policy; never promise views portably. | Required |
| E03 | `ravel_multi_index`, `unravel_index` | C-order index conversion with frozen bounds behavior; document unsupported order modes. | Required |
| E04 | `pad` | Portable constant, edge, reflect, symmetric, and wrap modes; explicitly reject unsupported statistical modes. | Required |
| E05 | Activation functions | `elu`, `gelu`, `leaky_relu`, `relu`, `selu`, `sigmoid`, `silu`, `softplus`, `softsign`, and `tanhshrink`, with stable formulas and gradients. | Required: all backends; gradients Torch/JAX |
| E06 | Signal helpers | `convolve1d` with explicit `valid`/`same`/`full` modes and `moving_mean`; axis, padding, dtype, and gradient behavior frozen. | Required: all backends |
| E07 | Numerical comparison | `allclose`, `isclose`, and test-only `assert_allclose` with dtype-aware defaults and overridable tolerances. | Required |
| E08 | Numerical metadata | `eps`, `tiny`, and finite-range helpers based on active dtype. | Required |
| E09 | Metrics | `mean_absolute_error`, `mean_squared_error`, `root_mean_squared_error`, `relative_l2_error`, and `r2_score`; axis/reduction/zero-denominator policy frozen. | Required: all backends |
| E10 | No hidden fallback | No extension may convert through NumPy or host memory unless it is an explicitly named conversion/I/O boundary. | Required invariant |

### 4.5 Explicit conversion, devices, and functional updates

| ID | Public API | Required semantics | Coverage |
|---|---|---|---|
| C01 | `convert_array(x, destination, *, dtype=None, device=None, copy=True)` | Explicit cross-backend conversion with documented copy, ownership, layout, dtype, device, and gradient-detachment behavior. | Required: every backend pair |
| C02 | `from_numpy` / `to_numpy` | Explicit NumPy boundaries; `to_numpy` never silently detaches or transfers unless the caller acknowledges those policies. | Required |
| C03 | `to_dlpack` / `from_dlpack` | Explicit DLPack interchange with ownership/lifetime, device, copy, and unsupported-layout rules. | Required where capability exists |
| C04 | `copy_array` | Logical copy with backend-preserving dtype/device; distinguish `copy=None`, `True`, and `False` when supported. | Required |
| C05 | `to_device` | Explicit same-backend device transfer; no implicit CPU/GPU movement. | Required by device capability |
| C06 | `detach` / `stop_gradient` | Explicit graph-boundary operations; names and docs must make loss of gradient history unmistakable. | Required: Torch/JAX; NumPy no-op only if explicitly documented |
| U01 | `index_set`, `index_add`, `index_multiply`, `index_min`, `index_max` | Functional return semantics with no mutation observable through the input. | Required: all backends |
| U02 | `scatter_set`, `scatter_add`, `scatter_multiply`, `scatter_min`, `scatter_max` | Freeze axis, broadcasting, bounds, dtype, duplicate indices, gradients, and JIT. Duplicate indices for `set` are rejected unless a deterministic policy is proven. | Required: all backends |
| U03 | Update safety | Input aliasing and mutation checks prove that NumPy/Torch adapters do not mutate input arrays to imitate JAX. | Required |

### 4.6 Randomness and initialization

| ID | Public API | Required semantics | Coverage |
|---|---|---|---|
| R01 | `random_state(seed, backend=...)` / `RandomState` | Explicit immutable or safely encapsulated state; no hidden global seeding. | Required: all backends |
| R02 | State operations | `split`, `spawn`, and reproducible state serialization within the same backend/version contract. | Required |
| R03 | Basic distributions | `random`, `uniform`, `normal`, `standard_normal`, `randint`, and `bernoulli`. | Required: all backends |
| R04 | Additional distributions | `gamma` and `exponential`. | Required: all backends |
| R05 | Sampling | `choice` and functional `permutation`; sampling with/without replacement and probability validation. | Required: all backends |
| R06 | Initializers | `constant`, `uniform`, `normal`, `truncated_normal`, `glorot_uniform`, `glorot_normal`, `lecun_uniform`, `lecun_normal`, `he_uniform`, `he_normal`, and `orthogonal`. | Required: all backends |
| R07 | Reproducibility | Deterministic replay only within one declared backend/version/configuration; cross-backend statistical equivalence, not identical bitstreams. | Required and documented |
| R08 | JIT/gradient behavior | State progression remains explicit under JAX JIT; random generation is non-differentiable unless a specific reparameterized API says otherwise. | Required |

### 4.7 Automatic differentiation and compilation

| ID | Public API | Required semantics | Coverage |
|---|---|---|---|
| D01 | `grad` | Scalar-output gradient with explicit differentiable argument selection. | Required: Torch/JAX; NumPy raises capability error |
| D02 | `value_and_grad` | Return value and gradient without duplicate evaluation where backend permits. | Required: Torch/JAX |
| D03 | `jacobian` / `hessian` | Stable shape convention, argument selection, vectorization policy, and complex-number limitations. | Required: Torch/JAX |
| D04 | `jvp` / `vjp` | Forward- and reverse-mode products with stable result records. | Required where frozen Torch/JAX versions support them |
| D05 | Higher derivatives | At least second derivatives for smooth representative functions; unsupported nondifferentiable paths fail honestly. | Required test |
| J01 | `jit` | JAX JIT required; Torch compilation required only after a stable frozen contract is verified; NumPy raises capability error. | Required by capability |
| J02 | `vmap` | Vectorized mapping for JAX and Torch where supported, with explicit `in_axes`/`out_axes` subset. | Required by capability |
| J03 | Graph integrity | Tests detect `.item()`, NumPy conversion, host extraction, mutation, data-dependent Python control flow, and graph detachment in promised compiled paths. | Required |

### 4.8 PyTree/data-tree utilities

| ID | Public API | Required semantics | Coverage |
|---|---|---|---|
| T01 | `tree_flatten` / `tree_unflatten` | Stable leaf order and `TreeSpec` for tuples, lists, mappings, named tuples, and dataclasses. | Required |
| T02 | `tree_leaves` / `tree_structure` | Introspection without mutation. | Required |
| T03 | `tree_map` / `tree_map_with_path` | One or more trees with strict structure matching and informative path errors. | Required |
| T04 | `tree_all` / `tree_any` | Predicate reductions over leaves. | Required |
| T05 | `tree_get` / `tree_replace` | Path-based immutable access and replacement. | Required |
| T06 | Custom nodes | `register_pytree_node` with thread-safe registration and duplicate-registration errors; no mutable public registry exposure. | Required |
| T07 | Tree serialization | Versioned, safe `TreeSpec` JSON serialization; no arbitrary-code deserialization. | Required |
| T08 | Array-tree helpers | `tree_array_namespace`, `tree_to_backend`, `tree_to_device`, and `tree_to_numpy`, all explicit and mixed-backend safe. | Required |

### 4.9 Dataset abstractions and composition

| ID | Public API | Required semantics | Coverage |
|---|---|---|---|
| DS01 | `Dataset` | Typed map-style abstract interface with `__len__` and integer/slice/index-array `__getitem__` contract. | Required |
| DS02 | `IterableDataset` | Typed streaming interface with explicit absence or availability of length. | Required |
| DS03 | `ArrayDataset` | Samples along a configurable leading sample axis; preserve native backend arrays and optional field metadata. | Required: all backends |
| DS04 | `TupleDataset` | Aligned sequence of arrays/fields with validated sample counts. | Required |
| DS05 | `MappingDataset` | Ordered named fields with validated sample counts and stable mapping output. | Required |
| DS06 | `ConcatDataset` | Concatenate datasets with negative-index, slice, and boundary correctness. | Required |
| DS07 | `Subset` | Indexed view without copying source data; scalar/slice/index-sequence support. | Required |
| DS08 | `TransformDataset` | Apply sample, input, target, or field transforms lazily without modifying the source dataset. | Required |
| DS09 | `FilteredDataset` | Deterministic index view for map datasets; lazy predicate mode for iterable datasets. | Required |
| DS10 | `ZipDataset` | Combine aligned datasets with strict/min-size policies and explicit length semantics. | Required |
| DS11 | Dataset metadata | Immutable `FieldSpec`, `DataSpec`, dimension names, shape excluding sample axis, dtype, backend, device, and semantic role metadata. | Required |
| DS12 | Schema validation | `infer_data_spec`, `validate_sample`, and `validate_dataset`; path-aware errors for shape/dtype/structure mismatch. | Required |

### 4.10 Splitting, sampling, collation, and loading

| ID | Public API | Required semantics | Coverage |
|---|---|---|---|
| S01 | `split_dataset` | Integer or fractional non-overlapping splits; deterministic remainder allocation; optional explicit random state. | Required |
| S02 | `train_validation_test_split` | Named train/validation/test subsets with reproducible shuffling and validation of fractions. | Required |
| S03 | `kfold_indices` | Deterministic K-fold train/validation index generation with optional shuffle/state. | Required |
| S04 | `SequentialSampler` | Every index exactly once in order. | Required |
| S05 | `RandomSampler` | With/without replacement, explicit random state, and exact requested sample count. | Required |
| S06 | `SubsetRandomSampler` | Samples only from supplied indices with explicit state. | Required |
| S07 | `WeightedRandomSampler` | Validated nonnegative finite weights and explicit replacement semantics. | Required |
| S08 | `BatchSampler` | Batch any sampler with validated batch size and `drop_last`. | Required |
| CO01 | `default_convert` | Convert Python/NumPy scalar leaves only according to explicit backend/context; preserve strings and metadata. | Required |
| CO02 | `default_collate` | Stack arrays/scalars and recursively collate mappings, tuples, named tuples, and dataclasses; strict structure and mixed-backend validation. | Required: all backends |
| CO03 | `uncollate` | Recover a sequence of samples when the collated structure satisfies the supported inverse contract. | Required |
| DL01 | `DataLoader` | Map- and iterable-dataset batching with `batch_size`, `shuffle`, `sampler`, `batch_sampler`, `drop_last`, `collate_fn`, and explicit random state. | Required |
| DL02 | Loader validation | Enforce mutually exclusive arguments and stable length semantics; never use mutable default configuration. | Required |
| DL03 | Single-process baseline | Correct deterministic single-process loader is required for 0.1.0. Multiprocessing must not be faked or silently enabled. | Required |
| DL04 | Backend preservation | Collation preserves backend/dtype/device and gradients where stacking naturally supports them. | Required: all backends |
| DL05 | Empty/error behavior | Freeze empty dataset, incomplete final batch, iterator reset, repeated iteration, and exception propagation semantics. | Required |

### 4.11 Combined loaders and `DataModule`

| ID | Public API | Required semantics | Coverage |
|---|---|---|---|
| CL01 | `CombinedLoader` | Accept an arbitrary PyTree of loaders and reconstruct the same structure on output. | Required |
| CL02 | `min_size` mode | Stop when the shortest loader is exhausted. | Required |
| CL03 | `max_size_cycle` mode | Stop after the longest loader and cycle shorter loaders deterministically. | Required |
| CL04 | `max_size` mode | Stop after the longest loader and emit `None` for exhausted leaves. | Required |
| CL05 | `sequential` mode | Exhaust loaders sequentially and report batch and loader indices. | Required |
| CL06 | Per-loader limits | Validated scalar or tree-shaped limits with correct `len`, reset, and repeated iteration. | Required |
| DM01 | `DataModule` | Base class/implementation for `train`, `validation`, `test`, and `predict` stages. | Required |
| DM02 | Dataset registry | `add_dataset`, `get_dataset`, `remove_dataset`, `datasets(stage)`, with duplicate-name and invalid-stage errors. | Required |
| DM03 | Lifecycle | Idempotent `prepare_data`, `setup(stage)`, and `teardown(stage)` hooks with no import-time or constructor I/O. | Required |
| DM04 | Loader factories | `loader(stage, name, **config)` and `combined_loader(stage, mode, configs)` using validated immutable configuration. | Required |
| DM05 | Reproducible split setup | Helper to register deterministic train/validation/test splits from one source dataset. | Required |
| DM06 | Metadata | Stage/dataset specs and diagnostics without loading all samples when avoidable. | Required |

### 4.12 Transforms, statistics, and persistence

| ID | Public API | Required semantics | Coverage/dependency |
|---|---|---|---|
| P01 | Transform protocol | Typed `fit`, `transform`, optional `inverse_transform`, and `fit_transform`; immutable learned state where practical. | Required |
| P02 | Composition | `Compose`, `Identity`, `LambdaTransform`, `SelectFields`, and `RenameFields`. | Required |
| P03 | Backend conversion transform | `ToBackend`, `ToDevice`, and `CastDType`; all conversions explicit. | Required |
| P04 | `StandardScaler` | Per-feature mean/std fitting with axis policy, zero-variance behavior, transform, and inverse transform. | Required: all backends |
| P05 | `MinMaxScaler` | Configurable output interval, constant-feature policy, transform, and inverse transform. | Required: all backends |
| P06 | Statistics | `dataset_statistics` with count, min, max, mean, variance/std, and optional field selection using a stable streaming algorithm. | Required |
| IO01 | NPY | `save_npy` / `load_npy` for one array/tree leaf, optional memory mapping where NumPy supports it, and explicit backend conversion. | Required: base |
| IO02 | NPZ | `save_npz` / `load_npz` for named arrays plus safe JSON metadata; compressed and uncompressed modes. | Required: base |
| IO03 | CSV | `save_csv` / `load_csv` for documented two-dimensional numeric/tabular subset with header and dtype policy. | Required: base |
| IO04 | HDF5 | `save_hdf5` / `load_hdf5` for nested named array trees, metadata, compression/chunk options, and lazy optional dependency. | Required with `io-hdf5` extra |
| IO05 | MATLAB | `save_mat` / `load_mat` for documented numeric tree subset, MATLAB metadata cleanup, and lazy SciPy dependency. | Required with `io-mat` extra |
| IO06 | Atomic writes | Write temporary file in the destination directory and replace atomically where the platform supports it; never leave a successful-looking partial file. | Required |
| IO07 | Safe loading | No default pickle or arbitrary-code deserialization; object arrays require explicit unsafe opt-in and prominent warning, or are rejected. | Required |
| IO08 | Tree/backend policy | Save through explicit host conversion; load to NumPy by default and convert to another backend only when requested explicitly. | Required |
| IO09 | Round trips | Preserve documented values, shapes, dtypes, field names, tree structure, and metadata across each supported format. | Required |

### 4.13 Configuration, errors, logging, typing, and diagnostics

| ID | Public API | Requirement |
|---|---|---|
| Q01 | Immutable config | `ArrayContext`, `PrecisionPolicy`, `DataLoaderConfig`, and I/O option records are frozen and validated. |
| Q02 | Error root | `AscError` is the stable package exception root. |
| Q03 | Backend errors | `BackendError`, `BackendUnavailableError`, `MixedBackendError`, and `CapabilityNotSupportedError`. |
| Q04 | Array errors | `ConversionError`, `DTypeError`, `DeviceError`, `RandomStateError`, `IndexUpdateError`, and `DuplicateIndexError`. |
| Q05 | Data errors | `DataError`, `DatasetError`, `DataSpecError`, `CollationError`, `DataLoaderError`, `DataSplitError`, and `DataFormatError`. |
| Q06 | Error quality | Messages identify the public operation, offending value/type/shape, expected contract, and recovery action without leaking sensitive data. |
| Q07 | Logging | Package logger installs `NullHandler`; no root configuration, import-time output, or logging of entire user arrays by default. |
| Q08 | Typing | Every public API is typed; narrow protocols are used for arrays, namespaces, datasets, samplers, transforms, and random state; ship `py.typed`. |
| Q09 | Diagnostics | `diagnostics()` returns package/Python/backend/dependency/capability information without eagerly importing or initializing optional backends. |
| Q10 | Versioning | Runtime version from package metadata, semantic versioning, changelog, deprecation policy, and public-export manifest. |

## 5. DeepMLT adoption map

| DeepMLT concept | `asc-py` decision |
|---|---|
| Unified dense-tensor vocabulary | Adopt the vocabulary through native arrays and standards-first namespaces. |
| NumPy and PyTorch adapter modules | Adopt and add JAX. |
| Backend registry | Redesign as internal, explicit, and non-mutable after initialization. |
| Default tensor context | Redesign as immutable `ArrayContext`; no global setter. |
| Dynamic `Tensor` inheritance | Reject. Use native `numpy.ndarray`, `torch.Tensor`, and JAX arrays. |
| Global `set_dtensor_backend()` | Reject. Use `backend(...)` or infer from inputs. |
| Mathematical operation catalog | Adopt standard operations through `xp`; adopt nonstandard operations in narrow extension namespaces. |
| Random utilities and initializers | Adopt with explicit backend-native random state. |
| `jacobian` and `hessian` | Adopt through the autodiff extension. |
| Tensor/sequence/mapping datasets | Adopt with stronger validation and native-array semantics. |
| Concatenation, subsets, and partitioning | Adopt and generalize. |
| `BatchLoader` | Redesign around samplers, collation, explicit random state, and deterministic single-process behavior. |
| `TreeLoader` modes | Adopt as `CombinedLoader` with PyTree-preserving output. |
| `DataModule` | Adopt and expand to validated lifecycle and stage management. |
| NPY/HDF5/MAT I/O | Adopt with optional extras, safe loading, atomic writes, and explicit conversion. |
| Geometry, PDE, solver, module, optimizer, trainer, vision | Move to `asc-xde-py` or future dedicated packages. |
| Hidden NumPy fallback, CPU copies, and graph detach | Reject. |

## 6. Required repository architecture

```text
asc-py/
├── AGENTS.md
├── pyproject.toml
├── README.md
├── CHANGELOG.md
├── CONTRIBUTING.md
├── SECURITY.md
├── CITATION.cff
├── src/asc/
│   ├── __init__.py
│   ├── py.typed
│   ├── backends/
│   │   ├── _protocols.py
│   │   ├── numpy.py
│   │   ├── torch.py
│   │   └── jax.py
│   ├── core/
│   ├── ops/
│   ├── linalg/
│   ├── fft/
│   ├── random/
│   ├── updates/
│   ├── autodiff/
│   ├── compilation/
│   ├── conversion/
│   ├── metrics/
│   ├── tree/
│   ├── data/
│   │   ├── dataset.py
│   │   ├── schema.py
│   │   ├── split.py
│   │   ├── sampler.py
│   │   ├── collate.py
│   │   ├── loader.py
│   │   ├── combined.py
│   │   ├── module.py
│   │   ├── transforms.py
│   │   ├── statistics.py
│   │   └── io.py
│   ├── config.py
│   ├── errors.py
│   ├── logging.py
│   ├── diagnostics.py
│   └── typing.py
├── tests/
│   ├── contract/
│   ├── conformance/
│   ├── parity/
│   ├── properties/
│   ├── data/
│   ├── packaging/
│   └── integration/
├── benchmarks/
├── docs/
├── examples/
├── .github/workflows/
└── .agent/execplans/
```

The dependency direction is:

```text
public API → portable core → capability protocols
                              ↑
                   backend implementations

data API → tree + portable core + explicit conversion/I/O boundaries
```

## 7. Acceptance and traceability rules

Create `docs/specification/functionality-matrix.md` by copying Section 4. It must
add these columns for every ID:

| ID | Public symbol(s) | Source file(s) | Test file(s) | Documentation | Backend matrix | Status |
|---|---|---|---|---|---|---|

Codex may mark an ID complete only when implementation, tests, documentation,
and all declared CPU backend checks pass. A skipped required backend is not a
pass. Capability-gated rows must prove both the supported path and the explicit
unsupported error path.

Every public operation must test, as applicable:

- values, shapes, dtypes, devices, and return types;
- scalar, zero-dimensional, empty, singleton, and broadcast inputs;
- invalid arguments and stable public exceptions;
- mixed-backend rejection;
- no hidden copy, host transfer, conversion, mutation, or graph detach;
- gradient correctness for Torch/JAX;
- JAX eager/JIT agreement and Torch eager/compiled agreement if promised;
- deterministic same-backend random replay and statistical invariants;
- dataset slicing, repeated iteration, split disjointness, collation structure,
  loader reset, and I/O round trips;
- minimum and newest supported dependency environments.

## 8. Google Python style and engineering gates

Use the current official Google Python Style Guide and compare it with the
supplied snapshot. The current official guide governs if they differ. Enforce:

- Pyink or a demonstrably Google-compatible formatter at an 80-column target;
- Pylint, plus Ruff for complementary bug, import, modernization, performance,
  and Google-docstring checks;
- strict Pyright for maintained source and public APIs;
- full package/module imports except the guide's typing exceptions;
- Google-style `Args`, `Returns`, `Yields`, and `Raises` docstrings;
- no mutable default arguments or unjustified mutable global state;
- built-in exceptions for ordinary precondition errors and the stable public
  hierarchy for package/domain failures;
- small focused functions, explicit resource management, descriptive names,
  and no import-time I/O or backend initialization.

Required gates are formatting, Ruff, Pylint, strict Pyright, pytest with coverage,
Array API conformance, all-backend parity, docs with warnings treated as errors,
wheel/sdist build and clean installation, pre-commit, dependency/license audit,
and the `$port-scientific-python` portable-core audit.

## 9. Single all-in-one Codex directive

Place this file at the root of the `asc-py` repository, start Codex there, enter
Goal mode if available, and paste the following directive:

```text
$port-scientific-python

Develop AI4SciComp/asc-py autonomously from its current state to a locally
release-ready version 0.1.0.

Read, in full, AGENTS.md and asc-py-comprehensive-development-runbook.md before
editing. Treat the runbook's complete Section 4 functionality matrix as a
normative release contract. Implement every row marked Required. Do not replace
the matrix with representative examples, a vertical-slice-only result, stubs,
empty protocols, TODOs, or documentation-only claims.

Use https://github.com/escapetiger/deepmlt as a read-only historical reference.
Inventory its array and data APIs, but do not copy its mutable global backend,
dynamic Tensor dispatch, hidden NumPy fallbacks, implicit host transfers, graph
detachment, global random state, or domain-specific solver/model modules.

The architecture must use native NumPy, PyTorch, and JAX arrays. NumPy is the
required backend. PyTorch and JAX are independent lazy optional extras; neither
may require the other. Include the complete backend-neutral data package defined
in the runbook. Keep PDE solvers, physics-specific data generation, neural
operators, training, optimizers, and AutoML out of this repository.

Before production code:

1. Inspect the repository, Git state, existing user changes, organization
   conventions, packaging, tests, docs, and CI.
2. Verify the newest explicit Python Array API revision supported by the entire
   proposed dependency range; provisionally use 2024.12 and record any change in
   an ADR with primary-source evidence.
3. Resolve the import namespace and license only from repository/organization
   evidence; ask once if either remains genuinely ambiguous.
4. Write the architecture, portability contract, public API, dependency policy,
   support matrix, DeepMLT audit, ADRs, AGENTS.md, and a living ExecPlan.
5. Copy every functionality ID into
   docs/specification/functionality-matrix.md with source, test, documentation,
   backend, and status columns.

Implement the complete contract in dependency order: repository foundation;
backend/context vertical slice; full Array API validation; linalg/FFT and
portable operations; conversion and functional updates; randomness; autodiff and
compilation; PyTrees; datasets/schema/splits; samplers/collation/loaders;
CombinedLoader and DataModule; transforms/statistics/I/O; errors/logging/typing/
diagnostics; full parity/property/conformance tests; benchmarks/docs/CI;
packaging and final adversarial release audit.

Milestones are internal sequencing only. Continue automatically through the next
incomplete functionality ID after each milestone passes. Do not stop after a
representative slice and do not ask for routine approvals. Pause only for an
unresolved license/import decision, permission boundary, destructive action,
unsupported public promise, or irreversible external action.

Follow the current official Google Python Style Guide. Use typed public APIs,
Google-style docstrings, an 80-column target, Pyink or a compatible formatter,
Ruff, Pylint, strict Pyright, pytest, property tests, pre-commit, and a src layout.
Run the Port Scientific Python audit during discovery, after the first slice, and
before handoff.

For each functionality ID, implement source, public export, tests, API docs, and
an executable example where appropriate. Prove NumPy/Torch/JAX CPU parity for
every declared portable operation. Test optional installations independently:
base, torch, jax, HDF5, MAT, and all. Never report an unavailable or skipped
required matrix entry as passed.

Do not push, create a PR, tag, publish, release, change repository settings, or
modify DeepMLT without explicit authorization. You may prepare local workflows
and release artifacts.

Finish only when every required matrix row is complete and traceable; all format,
lint, type, unit, property, conformance, parity, data, I/O, documentation,
packaging, and audit gates pass; wheel and sdist install cleanly in isolated
environments; and the final report lists exact commands/results, skips, known
limitations, benchmarks, dependency/license findings, and external actions still
requiring authorization.
```

## 10. Definition of done

Codex must not declare completion until:

- every Required ID in Section 4 is `complete` with source/test/docs traceability;
- NumPy-only base installation works without Torch or JAX;
- Torch-only and JAX-only extras work independently;
- all promised NumPy/Torch/JAX CPU parity and Array API conformance tests pass;
- data structures preserve native arrays and explicit backend semantics;
- NPY/NPZ/CSV base I/O and optional HDF5/MAT round trips pass;
- no hidden fallback, device transfer, dtype change, mutation, global backend,
  global seed change, host extraction, or gradient detach remains;
- formatting, Ruff, Pylint, strict Pyright, coverage, docs, pre-commit, build,
  clean-install smoke tests, and portability audit pass;
- wheel/sdist contents, metadata, license, `py.typed`, and public exports pass;
- the final release-readiness report states all limitations honestly;
- no remote mutation or publication was performed without separate approval.

## 11. Primary references

- [Python Array API standard](https://data-apis.org/array-api/)
- [Array API 2024.12 specification](https://data-apis.org/array-api/2024.12/)
- [array-api-compat](https://data-apis.org/array-api-compat/)
- [NumPy Array API compatibility](https://numpy.org/doc/stable/reference/array_api.html)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- [DeepMLT historical repository](https://github.com/escapetiger/deepmlt)
- [AI4SciComp asc-py](https://github.com/AI4SciComp/asc-py)
