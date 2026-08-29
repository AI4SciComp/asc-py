# Benchmarks

Reproducible pytest-benchmark cases live here. Benchmarks separate import,
first-call or compilation, warm-up, and steady-state timing; synchronize
asynchronous backends; and record shape, dtype, device, dependency versions,
hardware, and memory where material.

Run `make benchmark`. Benchmark results are evidence, not normal test gates or
unsupported speedup claims.

The initial suite measures steady-state `sum_of_squares` over 4,096 float32
values for every supported namespace. JAX results are synchronized explicitly;
compilation and first-call costs are excluded from this steady-state case.
