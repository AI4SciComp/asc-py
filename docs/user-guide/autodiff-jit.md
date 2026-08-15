# Autodiff, JIT, and vectorization

Install `asc-py[torch]` or `asc-py[jax]` before using automatic
differentiation. Both optional backends provide gradients, value-and-gradient,
Jacobians, Hessians, JVPs, VJPs, and supported vectorization over real floating
dense CPU arrays. NumPy raises {py:class}`asc.CapabilityNotSupportedError`.

```python
import asc

selected = asc.backend("jax")
xp = selected.xp
x = xp.asarray([1.0, 2.0], dtype=xp.float32)
gradient = asc.grad(lambda value: xp.sum(value**2), backend="jax")(x)
```

Results remain backend-native and retain graphs for higher derivatives on
smooth functions. Complex differentiation is outside 0.1.0. Concrete inputs
are validated as one dense CPU backend before transformation; foreign native
array leaves raise `MixedBackendError`. Torch JVP primal/tangent arrays and VJP
primal/cotangent arrays must all have real floating dtypes and fail with
`DTypeError` otherwise. Abstract JAX tracers cannot independently assert CPU
placement.

JAX is the only supported JIT backend. Torch compilation and NumPy JIT fail
explicitly. `vmap` follows the capability matrix. User functions must avoid
Python data-dependent control flow when executed under JAX tracing. Dynamic
checks inside JAX `jit` and `vmap` are translated back to the same documented
`asc` or built-in exception type used by eager execution.
