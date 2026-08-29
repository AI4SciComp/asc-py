# Array API and portable operations

asc freezes Python Array API revision 2024.12. Standard creation, elementwise,
manipulation, indexing, reduction, set, sorting, linear algebra, and FFT
operations are available through `Backend.xp`, `Backend.linalg`, and
`Backend.fft`. Native arrays—not wrapper tensors—cross every public boundary.
The linear-algebra and FFT facades are bound to their immutable backend.
Linear-algebra operands must all be native arrays from that backend; Python
containers and scalars are never coerced into arrays by a delegated call.

{py:mod}`asc.ops` supplies portable functions that are absent from the standard:
padding, activations, signals, comparisons, indexing helpers, and numeric
limits. {py:mod}`asc.metrics` supplies regression metrics. These operations
preserve backend and device, use standard promotion, do not mutate inputs, and
contain no host fallback. Smooth real floating paths preserve Torch/JAX graphs
and the declared JAX JIT paths.
MAE, MSE, RMSE, and dimensionless norm metrics promote both operands, then scale
them before subtraction and reduction. This prevents avoidable overflow or
underflow for representable results. Their normalized square reductions use
means instead of count-sized sums and promote float16/bfloat16 calculation to
backend-native float32. Long low-precision inputs therefore neither overflow
solely because a reduction contains more than 65,504 terms nor underflow before
a representable root is taken. Exact predictions return zero relative error and
an R2 score of one, including for nonzero targets. Reductions over a zero-sized
axis are rejected before backend dispatch, while `reduction="none"` and an empty
axis tuple preserve empty elementwise results.

Explicitly typed `Backend.full`, {py:func}`asc.create_full`, and constant
padding validate numeric scalars against the destination dtype before native
dispatch. Out-of-range values and nonzero values that would underflow to zero
fail instead of wrapping or narrowing.
The default constant padding value is Boolean `False`, which also represents
numeric zero and therefore works for every supported numeric dtype.

Multi-index helpers widen signed coordinates before bounds and stride
arithmetic, and JAX keeps dynamic bounds checks inside compiled execution.
Portable `einsum` and `lstsq` promote all operands to one result dtype before
native dispatch; `kron` follows the same promotion and supplies a portable
Boolean logical-product path. Least-squares `rcond` must be finite and
representable in the promoted operand dtype. FFT sample
counts must be positive integers, and spacing, its reciprocal scale, and every
resulting frequency bin must remain finite and representable in the requested
output dtype. The reciprocal is evaluated without an overflowing `n * spacing`
intermediate. These values and comparison
tolerances are Python scalar policies; native arrays are rejected rather than
implicitly scalarized. Finite differences and tolerances use exact
representable arithmetic where it is safe and scale-normalized logarithmic
forms only when necessary, so boundary rounding remains precise without
turning large values into warning-based backend failures.
Comparisons treat exactly equal infinities as close
without performing invalid non-finite subtraction.
The normalized `eigh` result accepts the standard `UPLO="L"` or `UPLO="U"`
selection and uses only that triangle consistently on every backend.
Frequency bins are formed with a wide signed index dtype and scaled before
narrowing, so valid float16 results do not overflow merely because the sample
count exceeds the output dtype's largest integer. Relative L2 and R2 use
dimensionless scaled reductions to avoid intermediate square overflow or
underflow for finite floating inputs.
Stable activation formulas preserve the mathematical limits of SiLU and
softsign at positive and negative infinity. ELU and leaky-ReLU coefficients
must remain finite without nonzero underflow in the input array dtype; their
validation remains usable inside JAX JIT and autodiff transformations.

`convolve1d` follows NumPy output-size semantics: `same` returns the longer
operand length, and `valid` returns `max(n, m) - min(n, m) + 1`, including when
the kernel is longer than the signal. Its reductions explicitly retain the
operands' promoted dtype, including for narrow integer arrays.
`moving_mean` has stricter valid-window semantics and rejects a window longer
than the selected axis because no complete moving window exists.

Functional `index_*` and `scatter_*` operations accept signed integer indices,
normalize axes, widen indices before bounds checks, promote destination and
values once, and broadcast values to the indexed slice. Duplicate arithmetic
updates reduce deterministically; duplicate `set` indices raise
{py:class}`asc.DuplicateIndexError`.
Torch bounds and duplicate checks stay native under `vmap`; multi-index helper
bounds checks follow the same transformation-safe rule.
