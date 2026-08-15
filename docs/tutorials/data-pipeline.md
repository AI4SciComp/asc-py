# Tutorial: deterministic data pipeline

Create native samples, split with explicit state, and batch without changing
their backend:

```{doctest}
>>> import numpy as np
>>> import asc
>>> features = np.arange(12, dtype=np.float32).reshape(6, 2)
>>> dataset = asc.data.ArrayDataset(features)
>>> state = asc.random_state(11, backend="numpy")
>>> splits = asc.data.train_validation_test_split(
...     dataset, train=0.5, validation=0.25, test=0.25, state=state
... )
>>> tuple(map(len, splits))
(4, 1, 1)
>>> loader = asc.data.DataLoader(splits.train, batch_size=2)
>>> [batch.shape for batch in loader]
[(2, 2), (2, 2)]
```

Collation stacks only structurally compatible leaves. For named fields, use
`MappingDataset`; for synchronized inputs and targets, use `TupleDataset`.
Persist final numerical trees with the safe NPY/NPZ/CSV base formats or install
one format-specific extra.
