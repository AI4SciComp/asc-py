# Copyright 2026 AI4SciComp contributors.
# SPDX-License-Identifier: Apache-2.0
"""Build a deterministic native-array data pipeline."""

import numpy

import asc

features = numpy.arange(12, dtype=numpy.float32).reshape(6, 2)
targets = numpy.arange(6, dtype=numpy.int32)
dataset = asc.data.MappingDataset(
    {
        "features": asc.data.ArrayDataset(features),
        "target": asc.data.ArrayDataset(targets),
    }
)
state = asc.random_state(7, backend="numpy")
loader = asc.data.DataLoader(dataset, batch_size=2, shuffle=True, state=state)
batches = list(loader)

assert len(batches) == 3
assert batches[0]["features"].shape == (2, 2)
assert sum(len(batch["target"]) for batch in batches) == len(dataset)
print(batches[0])
