# Data workflows

{py:mod}`asc.data` preserves native array leaves and deterministic PyTree
structure. Map and iterable datasets compose through array, tuple, mapping,
concat, subset, transform, filter, and zip views without copying source
storage. `FieldSpec` and `DataSpec` validate shape, dtype, backend, device, and
semantic roles with path-aware errors. Every `DataSpec` path must be unique,
contain only supported path components, and identify a leaf in its `TreeSpec`.

Splits and random samplers consume explicit {py:class}`asc.RandomState`.
Split sizes must be uniformly integer counts or uniformly floating fractions;
mixed representations are rejected.
Weighted sampling normalizes finite weights without overflowing, and impossible
positive draws from an empty population fail during sampler construction. Its
portable probability dtype is float32; a positive normalized weight that would
round to zero in that dtype is rejected instead of becoming unsampleable.
Collation recursively handles arrays, scalars, mappings, sequences, named
tuples, and dataclasses, including non-init fields, while requiring matching
structure and concrete mapping types. Entirely leafless samples are rejected
because they cannot retain batch cardinality; empty nested containers are
preserved when a sibling leaf supplies cardinality. `DataLoader` is single-process
and deterministic. Custom batch
samplers must be reiterable, and `drop_last` requires batching to be enabled.
Random state is validated during loader construction, including when a supplied
state is not consumed because shuffling is disabled.
`CombinedLoader` supplies four finite combination modes and rejects one-shot
loader leaves so every iteration is fresh. Sequential limits consume exactly
the yielded batches. `DataModule` manages named stages and idempotent lifecycle
hooks. Scalar conversion and collation preserve the dtype configured on a
supplied {py:class}`asc.Backend`.

Native `ArrayDataset` index arrays must use the dataset backend and stay native;
positive-step slices use backend slicing, while negative-step slices use a
backend-native index vector for Torch parity without a host index list. Supported
mapping subclasses retain their concrete type through PyTree and collation
round trips and through field-targeted transforms; named tuples retain their
type as well. Dataclasses with required `InitVar` parameters are rejected
because those constructor-only values cannot be recovered from an instance.

Transforms are functional. Conversion transforms expose backend, dtype,
device, and copy policies; fitted scalers return learned immutable state and
reject reductions over empty axes. `MinMaxScaler` rounds its feature range to
the fitted array dtype and rejects endpoints that overflow, underflow, or
collapse at that precision. Both fitted scalers normalize intermediate
reductions, differences, and spans so finite extreme values do not overflow.
`StandardScaler` performs float16 and bfloat16 moment fitting in backend-native
float32 and retains that state, so representable standard deviations and small
means do not disappear on long reductions; its transform output follows normal
Array API promotion against the fitted state.
`StandardScaler` retains its constant-feature mask and applies the documented
unit scale directly, avoiding overflow or underflow in forward/inverse round
trips for constant finite features.
Streaming statistics use one-pass scale-normalized online moments; a variance
that cannot fit the input dtype is reported as positive infinity while its mean
and standard deviation remain independently accurate. Sample counts remain
Python integers and are applied as typed, accumulator-device scalar
reciprocals, so float16 accumulation does not overflow when the number of
samples exceeds 65,504. An initial all-zero sample retains a zero scale so a
later small nonzero sample can establish the correct normalization. Float16 and
bfloat16 samples use float32 moment state so long streams do not lose updates to
low-precision rounding.

NPY, NPZ, and CSV require only NumPy. HDF5 requires `asc-py[io-hdf5]`; MATLAB
requires `asc-py[io-mat]`. Writes are atomic where supported. Save operations
are explicit host boundaries and require graph/transfer permission. Loads
return NumPy unless `destination=` requests an explicit conversion. Pickle and
object arrays are rejected by default. NPY, NPZ, HDF5, and MATLAB accept only
Boolean and numeric arrays; safe loads enforce the same dtype boundary as
writes. Single-array NPY/CSV inputs and every NPZ value must already be native
arrays; Python scalars and containers are rejected rather than implicitly
coerced. NPZ logical names that would alias another `.npy` archive member are
rejected before writing; NUL-containing names are also rejected because ZIP
member names would truncate them. CSV delimiters must not occur in formatted
real or complex numbers or accepted case-insensitive NaN/infinity spellings,
and headers may contain neither the delimiter nor a newline. HDF5/MAT metadata
is strict JSON without non-finite numbers. MATLAB
files record each accepted leaf dtype so Boolean and float16 values round-trip
exactly. HDF5 storage options are applied only to non-scalar leaves because
scalar datasets cannot be chunked or compressed; `chunks=False` explicitly
selects contiguous storage and is incompatible with compression. Persisted
HDF5 filter and per-leaf chunk failures are translated to `DataFormatError`.
Safe HDF5 loads require self-contained hard-linked datasets; external links,
virtual datasets, and externally stored raw data are rejected before reading.
Persisted mapping keys must be unique, so malformed trees cannot overwrite
leaves during reconstruction. Every serialized tree node is validated for kind-specific
metadata and child-count invariants before traversal. HDF5 and MATLAB reject
custom PyTree nodes: safe loaders never execute registered reconstruction
callbacks from persisted tags.

Live {py:class}`asc.tree.TreeSpec` equality includes container identity and is
strict and transitive. A specification restored from safe JSON intentionally
lacks Python type objects; schema validation uses `TreeSpec.is_compatible` to
check that serialized structure while retaining strict live-type comparisons.
