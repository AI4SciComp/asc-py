# Support and compatibility

The frozen standard is Python Array API 2024.12. A capability is supported only
after its required job passes without a skip.

| Component | Supported range | Notes |
|---|---|---|
| CPython | 3.12–3.14 | Linux x86-64 is validated locally |
| NumPy | 1.26.4–2.5.x | Required backend |
| PyTorch | 2.13.x | Independent `torch` extra |
| JAX/JAXlib | 0.6–0.10.x | Independent `jax` extra |
| array-api-compat | 1.13.x | Normalizes Array API 2024.12 |
| HDF5 | h5py 3.11–3.x | Independent `io-hdf5` extra |
| MATLAB | SciPy 1.13–1.x | Independent `io-mat` extra |
| Documentation | Sphinx 9.1, MyST 5.1, PyData Theme 0.19 | Independent `docs` extra; no Torch/JAX |

Dense CPU is the 0.1.0 execution boundary. `float32` is portable. JAX requires
x64 mode for `int64`, `uint64`, `float64`, and `complex128`; explicit requests
fail instead of narrowing. Complex support is operation-specific, and
real-only APIs fail explicitly. Torch and JAX support autodiff, JAX supports
JIT, and every unavailable capability raises an asc exception rather than
being emulated.

The normative detailed matrix, platform status, and evidence are maintained in
the [architecture support matrix](../architecture/support-matrix.md).
