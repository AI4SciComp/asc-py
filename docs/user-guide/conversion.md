# Conversion, ownership, and graphs

{py:func}`asc.convert_array` is the explicit backend boundary. It requires a
copy for cross-backend conversion, preserves shape, validates the requested
dtype and CPU device, and rejects active autodiff graphs. Same-backend copies
preserve graph history. Because cross-backend conversion already owns a new
allocation, strided NumPy views and lazy or non-contiguous Torch views are
materialized before DLPack export so consumers never observe unsupported
layout metadata. Direct DLPack export likewise compacts negative-stride NumPy
views, which cannot be consumed safely by every declared backend.

```python
import asc

numpy_backend = asc.backend("numpy")
x = numpy_backend.xp.asarray([1.0, 2.0])
copy = asc.convert_array(x, "numpy", copy=True)
```

{py:func}`asc.to_numpy` is the host boundary. Non-CPU inputs require
`allow_transfer=True`; active graphs require `allow_detach=True`. These flags
authorize only the named action and do not broaden the package's normal CPU
execution contract. {py:func}`asc.detach` and {py:func}`asc.stop_gradient`
make graph loss visible.
Acknowledgement flags accept actual Booleans only; truthy strings or integers
never authorize graph detachment or a device transfer.

DLPack export returns a one-consumer producer. Import/export validates shape,
dtype, device, copy policy, ownership, and graph state. A dtype declared by the
destination backend or context applies even when the separate `dtype=` argument
is omitted. Native producers' declared dtypes are also preserved; unavailable
wide JAX dtypes fail instead of narrowing. `CopyPolicy.NEVER` is passed to the
DLPack importer itself, so it cannot be satisfied by a hidden import copy
followed by a no-copy view. `CopyPolicy.ALWAYS` may allocate during import or in
the validated destination conversion after consuming a previously materialized
capsule. Once consumed, a capsule cannot be reused. Raw capsules without the
producer protocol are rejected because their device provenance cannot be
verified; pass a native protocol producer or the result of
{py:func}`asc.to_dlpack`.

Direct native DLPack producers are inspected before their capsule is consumed.
An active reverse- or forward-mode graph is rejected because DLPack does not
carry that graph state; callers must detach explicitly before import.

Destination contexts are normalized through the selected backend adapter, so a
portable `device="cpu"` override resolves to the backend's native CPU device.
`array-api-strict` remains a conformance oracle and is not a runtime conversion
destination, including when wrapped in a `CreationContext`.
