# Tutorial: portable computation

This tutorial computes an energy, applies a functional update, and preserves
the input on every supported backend.

The energy is

\[
E(x) = \sum_i x_i^2.
\]

```{doctest}
>>> import asc
>>> selected = asc.backend("numpy")
>>> xp = selected.xp
>>> x = xp.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=xp.float32)
>>> asc.sum_of_squares(x, axis=1).tolist()
[5.0, 25.0]
>>> rows = xp.asarray([0], dtype=xp.int16)
>>> delta = xp.asarray([[5.0, 6.0]], dtype=xp.float32)
>>> y = asc.index_add(x, rows, delta, axis=0)
>>> y.tolist()
[[6.0, 8.0], [3.0, 4.0]]
>>> x.tolist()
[[1.0, 2.0], [3.0, 4.0]]
```

Replace `"numpy"` with an installed `"torch"` or `"jax"` backend. The shape
and mathematical result remain portable; exact floating-point rounding and
random bitstreams are backend-specific.
