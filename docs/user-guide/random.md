# Randomness

Randomness is explicit and backend-native. {py:func}`asc.random_state` accepts
a seed in `[0, 2**32)`. Each distribution consumes a state and returns both a
sample and progressed immutable state; no process-global generator is read or
modified. Torch allocations explicitly target CPU and do not follow a
process-global default device. Torch child-state splitting uses a deterministic
permutation of the 32-bit seed space, so every supported split returns distinct
child streams.

```{doctest}
>>> import asc
>>> selected = asc.backend("numpy")
>>> state = asc.random_state(7, backend="numpy")
>>> sample, next_state = asc.random.uniform(
...     (3,), state=state, dtype=selected.xp.float32
... )
>>> sample.shape
(3,)
>>> next_state != state
True
```

Bounds, probabilities, shapes, and dtype families are validated before
dispatch. Probability vectors accepted within the documented sum tolerance are
normalized natively before sampling. Uniform intervals and integer bounds must
be representable in the requested output dtype. Signed integer sampling accepts
the full half-open dtype interval, including a maximum-plus-one exclusive upper
endpoint. Floating samples preserve the
half-open upper bound after low-precision conversion. Uniform endpoints must
themselves be exactly representable in that dtype. Truncated-normal bounds
follow the same exact representability rule, and results are clamped to those
accepted inclusive bounds after output-dtype conversion. Normal, truncated
normal, gamma, exponential, and orthogonal-gain parameters must remain finite
after conversion to the output dtype; nonzero parameters may not underflow to
zero. Unbounded NumPy distributions may return signed infinity when a finite
tail sample overflows a low-precision output dtype; this expected conversion is
not promoted to a warning-based API failure. Truncated-normal standardized
bounds must remain finite and distinct after parameter normalization.
Orthogonal initialization accepts float32 and, where enabled, float64;
lower-precision CPU QR is rejected consistently before native decomposition.
Bernoulli probabilities use a portable float32 sampling precision and
reject a nonzero probability that would round to zero. Counter states use
distinct backend substreams for distinct supported `(seed, counter)` pairs.
Replay is promised only for the same backend, dependency version, device,
dtype, and configuration.
Cross-backend bitstream identity is not promised. Serialized JAX state records
the PRNG implementation so supported non-default `rbg` keys restore without
changing their representation. Sampling is neither reparameterized nor
differentiable; JAX compiled paths use explicit state and trace-safe scalar
contract checks for every supported distribution.
Foreign or malformed backend dtype objects fail with `RandomStateError`; native
adapter exceptions do not leak through the public distribution API.
