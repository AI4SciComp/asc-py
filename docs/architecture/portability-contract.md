# Portability Contract

Status: Normative for 0.1.0. Last reviewed: 2026-08-11.

## Standard and backend selection

asc freezes Python Array API revision 2024.12. `array_namespace(*values)`
ignores Python scalars and `None` when at least one native array is present,
then requires every selecting array to use one backend. An empty/no-array call,
mixed backends, unknown arrays, masked/sparse/nested layouts, and unprovable CPU
placement fail before computation.

NumPy is required. PyTorch and JAX are independent lazy extras. Native arrays
cross public boundaries; asc has no wrapper tensor, active backend, global
default context, or implicit device selection.
Backend-bound linear-algebra and FFT facades validate all native leaves against
their selected backend before native dispatch.

## Dtype and scalar semantics

Standard promotion follows the selected 2024.12 namespace. Backend-specific
extensions compute one result dtype before dispatch and either cast operands
explicitly under that rule or reject the combination. They never rely on
backend scatter/assignment casts. Complex values are supported where the
standard and backend capability allow them; mathematically real-only APIs fail
with `DTypeError`. Low-precision linear-algebra and real-input FFT operations
whose CPU kernels are unavailable fail eagerly with
`CapabilityNotSupportedError` rather than leaking backend-native exceptions.
Explicit JAX int64/uint64/float64/complex128 requests fail when x64 is disabled
rather than narrowing with a warning.

Python scalars may participate only where the public signature allows them.
Array-valued scalar parameters are validated as arrays and never implicitly
cross a backend boundary.

## Devices, layouts, ownership, and graphs

The release execution boundary is dense CPU. No ordinary operation transfers,
copies, converts, casts, detaches, synchronizes, or materializes an array unless
its name and arguments state that action. Conversion and I/O document copy,
ownership, DLPack lifetime, host transfer, device, dtype, and graph policies.
Active reverse- or forward-mode graphs cannot cross a backend or host boundary
without explicit detach/stop-gradient acknowledgement.

Package JIT/vmap/autodiff entry points validate concrete CPU inputs before
transformation and pin or prove CPU execution. An abstract JAX tracer is not
accepted merely because its device cannot be observed. Promised graph paths
contain no `.item()`, Python data-dependent control flow, NumPy conversion,
mutation, or silent detach.

## Functional updates

Index and scatter updates return new logical values and leave inputs unchanged.
Indices are signed integers, axes are normalized, bounds are checked, values
broadcast to the indexed slice shape, and one promoted result dtype is used.
Duplicate additive/multiplicative/min/max updates have documented reduction
semantics. Duplicate `set` indices raise `DuplicateIndexError`.

## Randomness

Random state is explicit and backend-native or safely encapsulated. Every
operation returns progressed state; no function seeds or consumes a global
generator. Seeds and counters use validated common ranges; Torch derives each
substream by mixing the complete `(seed, counter)` pair instead of adding its
components. Bounds and probabilities must be finite, valid, and representable
in the output dtype.
Replay is promised only within the same backend, dependency version, device,
dtype, and configuration; cross-backend tests prove distributional semantics,
not identical bitstreams.

## Data and persistence

Datasets, schemas, trees, samplers, collation, and loaders preserve native
leaves and structure. Single-process loading is deterministic; multiprocessing
is not exposed. Loader configuration is immutable and mutually exclusive
arguments, missing shuffle state, and invalid random-state objects fail eagerly.

Persistence is a named host boundary. Save functions require explicit host
conversion permission for non-NumPy/accelerator/graph arrays. Loads return
NumPy unless a destination is requested. Writes are atomic where supported.
Pickle and arbitrary-code deserialization are disabled by default; object arrays
are rejected unless a prominently documented unsafe opt-in applies.

## Errors and evidence

All public failures derive from `AscError` and identify the operation, observed
contract violation, expected recovery, and safe non-sensitive context.
Capability absence is never emulated through NumPy. A functionality ID is
complete only when implementation, export, docs, examples where applicable,
and every declared CPU test pass without unexpected skips.
