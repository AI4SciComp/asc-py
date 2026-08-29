# Copyright 2026 AI4SciComp contributors.
# SPDX-License-Identifier: Apache-2.0
"""Use the standard namespace and a functional indexed update."""

import asc

selected = asc.backend("numpy")
xp = selected.xp
values = xp.reshape(xp.arange(6, dtype=xp.float32), (2, 3))
indices = xp.asarray([0, 2], dtype=xp.int16)
increments = xp.asarray([[10.0, 20.0], [30.0, 40.0]], dtype=xp.float32)

updated = asc.index_add(values, indices, increments, axis=1)
assert updated.tolist() == [[10.0, 1.0, 22.0], [33.0, 4.0, 45.0]]
assert values.tolist() == [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]
print(updated)
