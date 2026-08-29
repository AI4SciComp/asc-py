# Functionality Matrix

Status: Normative 0.1.0 traceability ledger. Last reviewed: 2026-08-11.

This ledger copies every functionality ID from Section 4 of the development
runbook and DOC-001 through DOC-018 from its normative documentation addendum.
A row is `Complete` only after its implementation, public export, tests,
documentation, and every declared CPU backend path pass. `In progress` is not
release evidence, and a required skip is a failure.

## Backend discovery, selection, and context

| ID | Public symbol(s) | Source file(s) | Test file(s) | Documentation | Backend matrix | Status |
|---|---|---|---|---|---|---|
| B01 | `backend` | `src/asc/core/backend.py` | `tests/contract/test_comprehensive_backend.py` | `docs/api/index.rst` | NumPy/Torch/JAX | Complete |
| B02 | `array_namespace` | `src/asc/core/namespace.py` | `tests/contract/test_namespace.py` | `docs/architecture/portability-contract.md` | NumPy/Torch/JAX/strict | Complete |
| B03 | `backend_of` | `src/asc/core/backend.py` | `tests/contract/test_comprehensive_backend.py` | `docs/api/index.rst` | All | Complete |
| B04 | `is_array` | `src/asc/core/backend.py` | `tests/contract/test_comprehensive_backend.py` | `docs/api/index.rst` | All | Complete |
| B05 | `available_backends` | `src/asc/core/backend.py` | `tests/contract/test_comprehensive_backend.py` | `docs/architecture/support-matrix.md` | All | Complete |
| B06 | `backend_info` | `src/asc/core/backend.py` | `tests/contract/test_comprehensive_backend.py` | `docs/architecture/support-matrix.md` | All | Complete |
| B07 | `Backend` | `src/asc/core/backend.py` | `tests/contract/test_comprehensive_backend.py` | `docs/architecture/public-api.md` | NumPy/Torch/JAX | Complete |
| B08 | `ArrayContext` | `src/asc/config.py` | `tests/contract/test_configuration.py` | `docs/api/index.rst` | All | Complete |
| B09 | Backend modules | `src/asc/backends/` | `tests/contract/test_comprehensive_backend.py` | `docs/architecture/dependency-policy.md` | NumPy/Torch/JAX | Complete |
| B10 | `has_capability`, `require_capability` | `src/asc/backends/capabilities.py` | `tests/contract/test_comprehensive_backend.py` | `docs/architecture/support-matrix.md` | All | Complete |
| B11 | `namespace_info` | `src/asc/core/namespace.py` | `tests/contract/test_comprehensive_backend.py` | `docs/user-guide/array-api.md` | All | Complete |
| B12 | Mixed-backend policy | `src/asc/core/namespace.py` | `tests/contract/test_namespace.py` | `docs/architecture/portability-contract.md` | All | Complete |

## Frozen Python Array API surface

The standard operations below are exposed by `Backend.xp` and
`array_namespace`; asc does not duplicate them as top-level wrappers.

| ID | Public symbol(s) | Source file(s) | Test file(s) | Documentation | Backend matrix | Status |
|---|---|---|---|---|---|---|
| A01 | Constants/namespace | `src/asc/core/namespace.py` | `tests/conformance/test_array_api_semantics.py` | `docs/user-guide/array-api.md` | All | Complete |
| A02 | Standard dtypes | `src/asc/core/namespace.py` | `tests/conformance/test_array_api_semantics.py` | `docs/architecture/support-matrix.md` | By capability | Complete |
| A03 | Creation family | Backend `xp` | `tests/conformance/test_array_api_semantics.py` | `docs/api/index.rst` | All | Complete |
| A04 | Dtype functions | Backend `xp` | `tests/conformance/test_array_api_semantics.py` | `docs/api/index.rst` | All | Complete |
| A05 | Arithmetic elementwise | Backend `xp` | `tests/conformance/test_array_api_semantics.py` | `docs/api/index.rst` | All | Complete |
| A06 | Exponential/logarithmic | Backend `xp` | `tests/conformance/test_array_api_semantics.py` | `docs/api/index.rst` | All | Complete |
| A07 | Trigonometric | Backend `xp` | `tests/conformance/test_array_api_semantics.py` | `docs/api/index.rst` | All | Complete |
| A08 | Hyperbolic | Backend `xp` | `tests/conformance/test_array_api_semantics.py` | `docs/api/index.rst` | All | Complete |
| A09 | Rounding/clipping | Backend `xp` | `tests/conformance/test_array_api_semantics.py` | `docs/api/index.rst` | All | Complete |
| A10 | Comparison/logical | Backend `xp` | `tests/conformance/test_array_api_semantics.py` | `docs/api/index.rst` | All | Complete |
| A11 | Bitwise | Backend `xp` | `tests/conformance/test_array_api_semantics.py` | `docs/api/index.rst` | All | Complete |
| A12 | Floating/complex helpers | Backend `xp` | `tests/conformance/test_array_api_semantics.py` | `docs/architecture/support-matrix.md` | By dtype | Complete |
| A13 | Indexing/`take` | Backend `xp` | `tests/conformance/test_array_api_semantics.py` | `docs/user-guide/array-api.md` | All | Complete |
| A14 | Manipulation | Backend `xp` | `tests/conformance/test_array_api_semantics.py` | `docs/api/index.rst` | All | Complete |
| A15 | Searching | Backend `xp` | `tests/conformance/test_array_api_semantics.py` | `docs/api/index.rst` | All | Complete |
| A16 | Sets | Backend `xp` | `tests/conformance/test_array_api_semantics.py` | `docs/api/index.rst` | All | Complete |
| A17 | Sorting | Backend `xp` | `tests/conformance/test_array_api_semantics.py` | `docs/api/index.rst` | All | Complete |
| A18 | Statistics/reductions | Backend `xp` | `tests/conformance/test_array_api_semantics.py` | `docs/api/index.rst` | All | Complete |
| A19 | Utility | Backend `xp` | `tests/conformance/test_array_api_semantics.py` | `docs/api/index.rst` | All | Complete |
| A20 | Core linear algebra | Backend `xp` | `tests/conformance/test_array_api_semantics.py` | `docs/api/index.rst` | All | Complete |
| A21 | Native array operators | Backend native arrays | `tests/conformance/test_array_api_semantics.py` | `docs/user-guide/array-api.md` | All | Complete |
| A22 | Standard edge cases | Backend native arrays | `tests/conformance/test_array_api_semantics.py` | `docs/user-guide/array-api.md` | Applicable APIs | Complete |

## Linear algebra and Fourier namespaces

| ID | Public symbol(s) | Source file(s) | Test file(s) | Documentation | Backend matrix | Status |
|---|---|---|---|---|---|---|
| L01 | `Backend.linalg` | `src/asc/linalg/` | `tests/parity/test_linalg_fft_ops.py` | `docs/api/index.rst` | NumPy/Torch/JAX; low precision by CPU capability | Complete |
| L02 | `eig`, `eigvals` | `src/asc/linalg/` | `tests/parity/test_linalg_fft_ops.py` | `docs/architecture/support-matrix.md` | By capability | Complete |
| L03 | `lstsq`, `LstsqResult` | `src/asc/linalg/` | `tests/parity/test_linalg_fft_ops.py` | `docs/api/index.rst` | NumPy/Torch/JAX; low precision by CPU capability | Complete |
| L04 | `einsum`, `kron`, `gkron` | `src/asc/linalg/` | `tests/parity/test_linalg_fft_ops.py` | `docs/api/index.rst` | All | Complete |
| L05 | Decomposition records | `src/asc/linalg/` | `tests/parity/test_linalg_fft_ops.py` | `docs/api/index.rst` | All | Complete |
| L06 | Batched decompositions | `src/asc/linalg/` | `tests/parity/test_linalg_fft_ops.py` | `docs/api/index.rst` | Native support | Complete |
| F01 | `Backend.fft` | `src/asc/fft/` | `tests/parity/test_linalg_fft_ops.py` | `docs/api/index.rst` | NumPy/Torch/JAX; low precision by CPU capability | Complete |
| F02 | FFT semantics | `src/asc/fft/` | `tests/parity/test_linalg_fft_ops.py` | `docs/architecture/portability-contract.md` | NumPy/Torch/JAX; low precision by CPU capability | Complete |

## Portable operations

| ID | Public symbol(s) | Source file(s) | Test file(s) | Documentation | Backend matrix | Status |
|---|---|---|---|---|---|---|
| E01 | `ops.diag` | `src/asc/ops/` | `tests/parity/test_linalg_fft_ops.py` | `docs/user-guide/array-api.md` | All | Complete |
| E02 | `ops.flatten`, `ops.ravel` | `src/asc/ops/` | `tests/parity/test_linalg_fft_ops.py` | `docs/user-guide/array-api.md` | All | Complete |
| E03 | `ravel_multi_index`, `unravel_index` | `src/asc/ops/` | `tests/parity/test_linalg_fft_ops.py` | `docs/user-guide/array-api.md` | All | Complete |
| E04 | `pad` | `src/asc/ops/` | `tests/parity/test_linalg_fft_ops.py` | `docs/user-guide/array-api.md` | All | Complete |
| E05 | Activations | `src/asc/ops/activations.py` | `tests/parity/test_linalg_fft_ops.py` | `docs/user-guide/array-api.md` | All + Torch/JAX grad | Complete |
| E06 | `convolve1d`, `moving_mean` | `src/asc/ops/signal.py` | `tests/parity/test_linalg_fft_ops.py` | `docs/user-guide/array-api.md` | All | Complete |
| E07 | `allclose`, `isclose`, `assert_allclose` | `src/asc/ops/comparison.py` | `tests/parity/test_linalg_fft_ops.py` | `docs/user-guide/array-api.md` | All | Complete |
| E08 | `eps`, `tiny`, finite range | `src/asc/ops/numeric.py` | `tests/parity/test_linalg_fft_ops.py` | `docs/user-guide/array-api.md` | All | Complete |
| E09 | Metrics | `src/asc/metrics/` | `tests/parity/test_linalg_fft_ops.py` | `docs/api/index.rst` | All | Complete |
| E10 | No hidden fallback | `src/asc/ops/`, `src/asc/metrics/` | `tests/contract/test_graph_integrity.py` | `docs/architecture/portability-contract.md` | All | Complete |

## Conversion, devices, and functional updates

| ID | Public symbol(s) | Source file(s) | Test file(s) | Documentation | Backend matrix | Status |
|---|---|---|---|---|---|---|
| C01 | `convert_array` | `src/asc/conversion.py` | `tests/contract/test_conversion_full.py` | `docs/user-guide/conversion.md` | Every pair | Complete |
| C02 | `from_numpy`, `to_numpy` | `src/asc/conversion.py` | `tests/contract/test_conversion_full.py` | `docs/user-guide/conversion.md` | All | Complete |
| C03 | `to_dlpack`, `from_dlpack` | `src/asc/conversion.py` | `tests/contract/test_conversion_full.py` | `docs/user-guide/conversion.md` | By capability | Complete |
| C04 | `copy_array` | `src/asc/conversion.py` | `tests/contract/test_conversion_full.py` | `docs/user-guide/conversion.md` | All | Complete |
| C05 | `to_device` | `src/asc/conversion.py` | `tests/contract/test_conversion_full.py` | `docs/user-guide/conversion.md` | By capability | Complete |
| C06 | `detach`, `stop_gradient` | `src/asc/conversion.py` | `tests/contract/test_conversion_full.py` | `docs/user-guide/conversion.md` | Torch/JAX/NumPy | Complete |
| U01 | Functional index updates | `src/asc/updates/` | `tests/extensions/test_updates_full.py` | `docs/user-guide/array-api.md` | All | Complete |
| U02 | Functional scatter updates | `src/asc/updates/` | `tests/extensions/test_updates_full.py` | `docs/user-guide/array-api.md` | All | Complete |
| U03 | Update safety | `src/asc/updates/` | `tests/extensions/test_updates_full.py` | `docs/architecture/portability-contract.md` | All | Complete |

## Randomness and initialization

| ID | Public symbol(s) | Source file(s) | Test file(s) | Documentation | Backend matrix | Status |
|---|---|---|---|---|---|---|
| R01 | `random_state`, `RandomState` | `src/asc/random/` | `tests/extensions/test_random_full.py` | `docs/user-guide/random.md` | NumPy/Torch/JAX | Complete |
| R02 | `split`, `spawn`, serialization | `src/asc/random/` | `tests/extensions/test_random_full.py` | `docs/user-guide/random.md` | All | Complete |
| R03 | Basic distributions | `src/asc/random/` | `tests/extensions/test_random_full.py` | `docs/user-guide/random.md` | All | Complete |
| R04 | `gamma`, `exponential` | `src/asc/random/` | `tests/extensions/test_random_full.py` | `docs/user-guide/random.md` | All | Complete |
| R05 | `choice`, `permutation` | `src/asc/random/` | `tests/extensions/test_random_full.py` | `docs/user-guide/random.md` | All | Complete |
| R06 | Initializers | `src/asc/random/initializers.py` | `tests/extensions/test_random_full.py` | `docs/user-guide/random.md` | All | Complete |
| R07 | Reproducibility | `src/asc/random/` | `tests/extensions/test_random_full.py` | `docs/architecture/portability-contract.md` | Per backend/version | Complete |
| R08 | JIT/gradient behavior | `src/asc/random/` | `tests/extensions/test_random_full.py` | `docs/user-guide/random.md` | JAX + all errors | Complete |

## Automatic differentiation and compilation

| ID | Public symbol(s) | Source file(s) | Test file(s) | Documentation | Backend matrix | Status |
|---|---|---|---|---|---|---|
| D01 | `grad` | `src/asc/autodiff/` | `tests/extensions/test_autodiff_full.py` | `docs/user-guide/autodiff-jit.md` | Torch/JAX; NumPy error | Complete |
| D02 | `value_and_grad` | `src/asc/autodiff/` | `tests/extensions/test_autodiff_full.py` | `docs/user-guide/autodiff-jit.md` | Torch/JAX | Complete |
| D03 | `jacobian`, `hessian` | `src/asc/autodiff/` | `tests/extensions/test_autodiff_full.py` | `docs/user-guide/autodiff-jit.md` | Torch/JAX | Complete |
| D04 | `jvp`, `vjp` | `src/asc/autodiff/` | `tests/extensions/test_autodiff_full.py` | `docs/user-guide/autodiff-jit.md` | By capability | Complete |
| D05 | Higher derivatives | `src/asc/autodiff/` | `tests/extensions/test_autodiff_full.py` | `docs/user-guide/autodiff-jit.md` | Torch/JAX | Complete |
| J01 | `jit` | `src/asc/compilation/` | `tests/extensions/test_autodiff_full.py` | `docs/user-guide/autodiff-jit.md` | JAX; Torch capability error | Complete |
| J02 | `vmap` | `src/asc/compilation/` | `tests/extensions/test_autodiff_full.py` | `docs/user-guide/autodiff-jit.md` | JAX/Torch capability | Complete |
| J03 | Graph integrity | Portable paths | `tests/contract/test_graph_integrity.py` | `docs/architecture/portability-contract.md` | Promised compiled paths | Complete |

## PyTree/data-tree utilities

| ID | Public symbol(s) | Source file(s) | Test file(s) | Documentation | Backend matrix | Status |
|---|---|---|---|---|---|---|
| T01 | `tree_flatten`, `tree_unflatten`, `TreeSpec` | `src/asc/tree/` | `tests/contract/test_tree_full.py` | `docs/api/index.rst` | Backend-neutral | Complete |
| T02 | `tree_leaves`, `tree_structure` | `src/asc/tree/` | `tests/contract/test_tree_full.py` | `docs/api/index.rst` | Backend-neutral | Complete |
| T03 | `tree_map`, `tree_map_with_path` | `src/asc/tree/` | `tests/contract/test_tree_full.py` | `docs/api/index.rst` | Backend-neutral | Complete |
| T04 | `tree_all`, `tree_any` | `src/asc/tree/` | `tests/contract/test_tree_full.py` | `docs/api/index.rst` | Backend-neutral | Complete |
| T05 | `tree_get`, `tree_replace` | `src/asc/tree/` | `tests/contract/test_tree_full.py` | `docs/api/index.rst` | Backend-neutral | Complete |
| T06 | `register_pytree_node` | `src/asc/tree/` | `tests/contract/test_tree_full.py` | `docs/api/index.rst` | Backend-neutral | Complete |
| T07 | TreeSpec JSON | `src/asc/tree/` | `tests/contract/test_tree_full.py` | `docs/api/index.rst` | Backend-neutral | Complete |
| T08 | Array-tree helpers | `src/asc/tree/` | `tests/contract/test_tree_full.py` | `docs/api/index.rst` | All | Complete |

## Dataset abstractions and composition

| ID | Public symbol(s) | Source file(s) | Test file(s) | Documentation | Backend matrix | Status |
|---|---|---|---|---|---|---|
| DS01 | `Dataset` | `src/asc/data/dataset.py` | `tests/data/test_datasets_schema_split.py` | `docs/user-guide/data.md` | Backend-neutral | Complete |
| DS02 | `IterableDataset` | `src/asc/data/dataset.py` | `tests/data/test_datasets_schema_split.py` | `docs/user-guide/data.md` | Backend-neutral | Complete |
| DS03 | `ArrayDataset` | `src/asc/data/dataset.py` | `tests/data/test_datasets_schema_split.py` | `docs/user-guide/data.md` | All | Complete |
| DS04 | `TupleDataset` | `src/asc/data/dataset.py` | `tests/data/test_datasets_schema_split.py` | `docs/user-guide/data.md` | All | Complete |
| DS05 | `MappingDataset` | `src/asc/data/dataset.py` | `tests/data/test_datasets_schema_split.py` | `docs/user-guide/data.md` | All | Complete |
| DS06 | `ConcatDataset` | `src/asc/data/dataset.py` | `tests/data/test_datasets_schema_split.py` | `docs/user-guide/data.md` | Backend-neutral | Complete |
| DS07 | `Subset` | `src/asc/data/dataset.py` | `tests/data/test_datasets_schema_split.py` | `docs/user-guide/data.md` | Backend-neutral | Complete |
| DS08 | `TransformDataset` | `src/asc/data/dataset.py` | `tests/data/test_datasets_schema_split.py` | `docs/user-guide/data.md` | Backend-neutral | Complete |
| DS09 | `FilteredDataset` | `src/asc/data/dataset.py` | `tests/data/test_datasets_schema_split.py` | `docs/user-guide/data.md` | Backend-neutral | Complete |
| DS10 | `ZipDataset` | `src/asc/data/dataset.py` | `tests/data/test_datasets_schema_split.py` | `docs/user-guide/data.md` | Backend-neutral | Complete |
| DS11 | `FieldSpec`, `DataSpec` | `src/asc/data/schema.py` | `tests/data/test_datasets_schema_split.py` | `docs/user-guide/data.md` | All | Complete |
| DS12 | Schema validation | `src/asc/data/schema.py` | `tests/data/test_datasets_schema_split.py` | `docs/user-guide/data.md` | All | Complete |

## Splitting, sampling, collation, and loading

| ID | Public symbol(s) | Source file(s) | Test file(s) | Documentation | Backend matrix | Status |
|---|---|---|---|---|---|---|
| S01 | `split_dataset` | `src/asc/data/split.py` | `tests/data/test_datasets_schema_split.py` | `docs/user-guide/data.md` | Backend-neutral | Complete |
| S02 | `train_validation_test_split` | `src/asc/data/split.py` | `tests/data/test_datasets_schema_split.py` | `docs/user-guide/data.md` | Backend-neutral | Complete |
| S03 | `kfold_indices` | `src/asc/data/split.py` | `tests/data/test_datasets_schema_split.py` | `docs/user-guide/data.md` | Backend-neutral | Complete |
| S04 | `SequentialSampler` | `src/asc/data/sampler.py` | `tests/data/test_collate_loader_sampler.py` | `docs/user-guide/data.md` | Backend-neutral | Complete |
| S05 | `RandomSampler` | `src/asc/data/sampler.py` | `tests/data/test_collate_loader_sampler.py` | `docs/user-guide/data.md` | Backend-neutral | Complete |
| S06 | `SubsetRandomSampler` | `src/asc/data/sampler.py` | `tests/data/test_collate_loader_sampler.py` | `docs/user-guide/data.md` | Backend-neutral | Complete |
| S07 | `WeightedRandomSampler` | `src/asc/data/sampler.py` | `tests/data/test_collate_loader_sampler.py` | `docs/user-guide/data.md` | Backend-neutral | Complete |
| S08 | `BatchSampler` | `src/asc/data/sampler.py` | `tests/data/test_collate_loader_sampler.py` | `docs/user-guide/data.md` | Backend-neutral | Complete |
| CO01 | `default_convert` | `src/asc/data/collate.py` | `tests/data/test_collate_loader_sampler.py` | `docs/user-guide/data.md` | All | Complete |
| CO02 | `default_collate` | `src/asc/data/collate.py` | `tests/data/test_collate_loader_sampler.py` | `docs/user-guide/data.md` | All | Complete |
| CO03 | `uncollate` | `src/asc/data/collate.py` | `tests/data/test_collate_loader_sampler.py` | `docs/user-guide/data.md` | All | Complete |
| DL01 | `DataLoader` | `src/asc/data/loader.py` | `tests/data/test_collate_loader_sampler.py` | `docs/user-guide/data.md` | All | Complete |
| DL02 | Loader validation | `src/asc/data/loader.py` | `tests/data/test_collate_loader_sampler.py` | `docs/user-guide/data.md` | Backend-neutral | Complete |
| DL03 | Single-process baseline | `src/asc/data/loader.py` | `tests/data/test_collate_loader_sampler.py` | `docs/user-guide/data.md` | Backend-neutral | Complete |
| DL04 | Backend preservation | `src/asc/data/collate.py` | `tests/data/test_collate_loader_sampler.py` | `docs/user-guide/data.md` | All | Complete |
| DL05 | Empty/error behavior | `src/asc/data/loader.py` | `tests/data/test_collate_loader_sampler.py` | `docs/user-guide/data.md` | All | Complete |

## Combined loaders and DataModule

| ID | Public symbol(s) | Source file(s) | Test file(s) | Documentation | Backend matrix | Status |
|---|---|---|---|---|---|---|
| CL01 | `CombinedLoader` | `src/asc/data/combined.py` | `tests/data/test_combined_module.py` | `docs/user-guide/data.md` | Backend-neutral | Complete |
| CL02 | `min_size` | `src/asc/data/combined.py` | `tests/data/test_combined_module.py` | `docs/user-guide/data.md` | Backend-neutral | Complete |
| CL03 | `max_size_cycle` | `src/asc/data/combined.py` | `tests/data/test_combined_module.py` | `docs/user-guide/data.md` | Backend-neutral | Complete |
| CL04 | `max_size` | `src/asc/data/combined.py` | `tests/data/test_combined_module.py` | `docs/user-guide/data.md` | Backend-neutral | Complete |
| CL05 | `sequential` | `src/asc/data/combined.py` | `tests/data/test_combined_module.py` | `docs/user-guide/data.md` | Backend-neutral | Complete |
| CL06 | Per-loader limits | `src/asc/data/combined.py` | `tests/data/test_combined_module.py` | `docs/user-guide/data.md` | Backend-neutral | Complete |
| DM01 | `DataModule` | `src/asc/data/module.py` | `tests/data/test_combined_module.py` | `docs/user-guide/data.md` | Backend-neutral | Complete |
| DM02 | Dataset registry | `src/asc/data/module.py` | `tests/data/test_combined_module.py` | `docs/user-guide/data.md` | Backend-neutral | Complete |
| DM03 | Lifecycle hooks | `src/asc/data/module.py` | `tests/data/test_combined_module.py` | `docs/user-guide/data.md` | Backend-neutral | Complete |
| DM04 | Loader factories | `src/asc/data/module.py` | `tests/data/test_combined_module.py` | `docs/user-guide/data.md` | All | Complete |
| DM05 | Reproducible split setup | `src/asc/data/module.py` | `tests/data/test_combined_module.py` | `docs/user-guide/data.md` | Backend-neutral | Complete |
| DM06 | Metadata/diagnostics | `src/asc/data/module.py` | `tests/data/test_combined_module.py` | `docs/user-guide/data.md` | All | Complete |

## Transforms, statistics, and persistence

| ID | Public symbol(s) | Source file(s) | Test file(s) | Documentation | Backend matrix | Status |
|---|---|---|---|---|---|---|
| P01 | `Transform` protocol | `src/asc/data/transforms.py` | `tests/data/test_transforms_statistics.py` | `docs/user-guide/data.md` | Backend-neutral | Complete |
| P02 | Composition transforms | `src/asc/data/transforms.py` | `tests/data/test_transforms_statistics.py` | `docs/user-guide/data.md` | Backend-neutral | Complete |
| P03 | Conversion transforms | `src/asc/data/transforms.py` | `tests/data/test_transforms_statistics.py` | `docs/user-guide/data.md` | All | Complete |
| P04 | `StandardScaler` | `src/asc/data/transforms.py` | `tests/data/test_transforms_statistics.py` | `docs/user-guide/data.md` | All | Complete |
| P05 | `MinMaxScaler` | `src/asc/data/transforms.py` | `tests/data/test_transforms_statistics.py` | `docs/user-guide/data.md` | All | Complete |
| P06 | `dataset_statistics` | `src/asc/data/statistics.py` | `tests/data/test_transforms_statistics.py` | `docs/user-guide/data.md` | All | Complete |
| IO01 | `save_npy`, `load_npy` | `src/asc/data/io.py` | `tests/data/test_io.py` | `docs/user-guide/data.md` | Base | Complete |
| IO02 | `save_npz`, `load_npz` | `src/asc/data/io.py` | `tests/data/test_io.py` | `docs/user-guide/data.md` | Base | Complete |
| IO03 | `save_csv`, `load_csv` | `src/asc/data/io.py` | `tests/data/test_io.py` | `docs/user-guide/data.md` | Base | Complete |
| IO04 | `save_hdf5`, `load_hdf5` | `src/asc/data/io.py` | `tests/data/test_io.py` | `docs/user-guide/data.md` | `io-hdf5` | Complete |
| IO05 | `save_mat`, `load_mat` | `src/asc/data/io.py` | `tests/data/test_io.py` | `docs/user-guide/data.md` | `io-mat` | Complete |
| IO06 | Atomic writes | `src/asc/data/io.py` | `tests/data/test_io.py` | `docs/user-guide/data.md` | Supported platforms | Complete |
| IO07 | Safe loading | `src/asc/data/io.py` | `tests/data/test_io.py` | `docs/user-guide/data.md` | All formats | Complete |
| IO08 | Tree/backend policy | `src/asc/data/io.py` | `tests/data/test_io.py` | `docs/user-guide/data.md` | All | Complete |
| IO09 | Round trips | `src/asc/data/io.py` | `tests/data/test_io.py` | `docs/user-guide/data.md` | Every format | Complete |

## Configuration, errors, logging, typing, and diagnostics

| ID | Public symbol(s) | Source file(s) | Test file(s) | Documentation | Backend matrix | Status |
|---|---|---|---|---|---|---|
| Q01 | Frozen configuration records | `src/asc/config.py` | `tests/contract/test_configuration.py` | `docs/api/index.rst` | All | Complete |
| Q02 | `AscError` | `src/asc/errors.py` | `tests/contract/test_comprehensive_backend.py` | `docs/api/index.rst` | Backend-neutral | Complete |
| Q03 | Backend errors | `src/asc/errors.py` | `tests/contract/test_comprehensive_backend.py` | `docs/api/index.rst` | Backend-neutral | Complete |
| Q04 | Array errors | `src/asc/errors.py` | `tests/contract/test_comprehensive_backend.py` | `docs/api/index.rst` | Backend-neutral | Complete |
| Q05 | Data errors | `src/asc/errors.py` | `tests/contract/test_comprehensive_backend.py` | `docs/api/index.rst` | Backend-neutral | Complete |
| Q06 | Error quality | All public modules | `tests/contract/test_comprehensive_backend.py` | `docs/api/index.rst` | All | Complete |
| Q07 | Logging | `src/asc/logging.py` | `tests/contract/test_logging.py` | `AGENTS.md` | Backend-neutral | Complete |
| Q08 | Public typing | `src/asc/typing.py`, `src/asc/py.typed` | `tests/contract/test_comprehensive_backend.py` | `docs/api/index.rst` | All | Complete |
| Q09 | `diagnostics` | `src/asc/diagnostics.py` | `tests/contract/test_comprehensive_backend.py` | `docs/architecture/support-matrix.md` | All | Complete |
| Q10 | Version/export manifest | `src/asc/_version.py`, `src/asc/__init__.py` | `tests/contract/test_version.py` | `CHANGELOG.md`, `docs/api/index.rst` | Backend-neutral | Complete |

## Automatic documentation

| ID | Public symbol(s) | Source file(s) | Test file(s) | Documentation | Backend matrix | CI evidence | Status |
|---|---|---|---|---|---|---|---|
| DOC-001 | Sphinx project | `docs/conf.py`, `docs/index.md` | `tests/docs/test_sphinx_configuration.py` | `docs/development/documentation.md` | Base | `.github/workflows/docs.yml` build | Complete |
| DOC-002 | `docs` extra | `pyproject.toml` | `tests/docs/test_sphinx_configuration.py` | `docs/getting-started/installation.md` | Base without Torch/JAX | `.github/workflows/docs.yml` build | Complete |
| DOC-003 | Napoleon parsing | `docs/conf.py` | `tests/docs/test_public_docstrings.py` | `docs/development/documentation.md` | Installed package | `.github/workflows/docs.yml` build | Complete |
| DOC-004 | Autosummary API | `docs/_inventory.py`, `docs/api/index.rst` | `tests/docs/test_public_api_inventory.py` | `docs/api/index.rst` | Base | `.github/workflows/docs.yml` build | Complete |
| DOC-005 | API completeness | `docs/_inventory.py` | `tests/docs/test_public_api_inventory.py`, `tests/docs/test_public_docstrings.py` | `docs/api/index.rst` | All public modules | `.github/workflows/docs.yml` build | Complete |
| DOC-006 | Types/signatures | `docs/conf.py`, `src/asc/typing.py` | `tests/docs/test_public_docstrings.py` | `docs/api/index.rst` | All public modules | `.github/workflows/docs.yml` build | Complete |
| DOC-007 | Cross-references | `docs/conf.py` | `tests/docs/test_sphinx_configuration.py` | `docs/development/documentation.md` | Base | `.github/workflows/docs.yml` build | Complete |
| DOC-008 | Executable examples | `examples/`, `docs/tutorials/` | `tests/docs/test_documented_examples.py` | `docs/getting-started/quickstart.md` | NumPy/Torch/JAX independent | `.github/workflows/docs.yml` backend jobs | Complete |
| DOC-009 | Optional safety | `docs/conf.py`, `.github/workflows/docs.yml` | `tests/docs/test_sphinx_configuration.py`, `tests/docs/test_documented_examples.py` | `docs/getting-started/installation.md` | Base/Torch/JAX independent | `.github/workflows/docs.yml` isolated jobs | Complete |
| DOC-010 | Scientific semantics | Public source docstrings | `tests/docs/test_public_docstrings.py` | `docs/user-guide/array-api.md`, `docs/user-guide/dtype-device.md` | NumPy/Torch/JAX | `.github/workflows/docs.yml` backend jobs | Complete |
| DOC-011 | Data documentation | `src/asc/data/` | `tests/docs/test_public_api_inventory.py`, `tests/docs/test_documented_examples.py` | `docs/user-guide/data.md`, `docs/tutorials/data-pipeline.md` | Base + format extras | `.github/workflows/docs.yml` build | Complete |
| DOC-012 | Compatibility pages | `docs/reference/` | `tests/docs/test_sphinx_configuration.py` | `docs/reference/support-matrix.md` | All declared backends | `.github/workflows/docs.yml` build | Complete |
| DOC-013 | Local commands | `Makefile` | `tests/docs/test_sphinx_configuration.py` | `docs/development/documentation.md` | Base | `.github/workflows/docs.yml` build | Complete |
| DOC-014 | PR documentation gate | `.github/workflows/docs.yml` | `tests/docs/test_sphinx_configuration.py` | `docs/development/testing.md` | Base/Torch/JAX independent | `.github/workflows/docs.yml` PR jobs | Complete |
| DOC-015 | Pages publication | `.github/workflows/docs.yml` | `tests/docs/test_sphinx_configuration.py` | `docs/development/documentation.md` | Authorized default branch | `.github/workflows/docs.yml` opt-in deploy | Complete |
| DOC-016 | External links | `.github/workflows/docs.yml`, `Makefile` | `tests/docs/test_sphinx_configuration.py` | `docs/development/documentation.md` | Scheduled/manual | `.github/workflows/docs.yml` linkcheck | Complete |
| DOC-017 | Release docs | `CHANGELOG.md`, `CITATION.cff`, `docs/release/` | `tests/docs/test_sphinx_configuration.py` | `docs/release/release-notes.md` | Backend-neutral | `.github/workflows/docs.yml` build | Complete |
| DOC-018 | Traceability | `scripts/audit_release.py`, `docs/specification/functionality-matrix.md` | `tests/docs/test_public_api_inventory.py` | `docs/reference/functionality-matrix.md` | All required IDs | `.github/workflows/docs.yml` build | Complete |
