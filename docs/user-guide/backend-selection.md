# Backend selection

Use {py:func}`asc.backend` when an operation creates arrays without array
inputs. The returned immutable {py:class}`asc.Backend` contains the standard
namespace (`xp`), normalized `linalg` and `fft` namespaces, a resolved CPU
device, and capability metadata.

Use {py:func}`asc.array_namespace` when arrays already exist. Python scalars and
`None` do not select a namespace; native arrays must all have the same backend.
Sparse, masked, nested, distributed, quantized, accelerator, and abstract
unplaced inputs fail before computation.

```python
import asc

selected = asc.backend("numpy")
xp = selected.xp
x = xp.asarray([1.0, 2.0], dtype=xp.float32)
assert asc.array_namespace(x) is xp
```

NumPy is always available. Selecting `torch` or `jax` requires the matching
extra. {py:func}`asc.available_backends` and {py:func}`asc.backend_info` inspect
installation metadata without importing optional backend packages. JAX 64-bit
dtypes and the `float64` capability are reported only when x64 is configured or
active.

`Backend.asarray` accepts Python numeric data directly. Native-array inputs use
the same explicit ownership, graph, dtype, device, and cross-backend rules as
{py:func}`asc.convert_array`; cross-backend input therefore requires
`copy=True`. Native arrays nested inside Python containers must be explicitly
converted and stacked before construction. `Backend.zeros`, `Backend.ones`,
and `Backend.full` require tuple shapes with non-Boolean, non-negative integer
extents. Inferred results are validated as supported dense numeric CPU arrays,
and native creation failures are normalized to {py:class}`asc.ContextError`.
