# Dtypes and devices

Version 0.1.0 supports dense CPU execution. Creation contexts accept `None`,
the exact string `"cpu"`, or a recognized native CPU device. Misspelled or
accelerator device requests raise an asc device or capability error; ordinary
operations never transfer data.

Standard operations follow Array API promotion. Extension operations compute a
single promoted result dtype before backend dispatch so Torch, JAX, and NumPy
cannot apply incompatible update casts. Real-only metrics, distributions, and
initializers reject Boolean, integer, and complex inputs as applicable.

JAX `int64`, `uint64`, `float64`, and `complex128` require
`JAX_ENABLE_X64=1`. An explicit request fails before creation, FFT frequency
construction, or dtype transforms when x64 is disabled instead of warning and
silently narrowing. The validated `Backend.xp` namespace applies this rule to
direct creation, casting, stacking, and dtype-selecting reduction calls as well
as asc extension entry points. Consult the
[support matrix](../reference/support-matrix.md) before relying on a conditional
dtype or transformation.
