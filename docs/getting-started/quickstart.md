# Quickstart

Select a backend for creation, then use its native Array API namespace:

```{doctest}
>>> import asc
>>> selected = asc.backend("numpy")
>>> xp = selected.xp
>>> values = xp.asarray([1.0, 2.0, 3.0], dtype=xp.float32)
>>> float(asc.sum_of_squares(values))
14.0
```

Nonstandard operations are grouped by purpose. Functional updates return a new
array and preserve the original:

```{doctest}
>>> indices = xp.asarray([1], dtype=xp.int32)
>>> increments = xp.asarray([10.0], dtype=xp.float32)
>>> updated = asc.index_add(values, indices, increments)
>>> updated.tolist()
[1.0, 12.0, 3.0]
>>> values.tolist()
[1.0, 2.0, 3.0]
```

Use {py:mod}`asc.random` for explicit random state, {py:mod}`asc.data` for
backend-neutral data pipelines, and {py:mod}`asc.conversion` only at named
backend or host boundaries.
