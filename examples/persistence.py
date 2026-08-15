# Copyright 2026 AI4SciComp contributors.
# SPDX-License-Identifier: Apache-2.0
"""Round-trip safe NPZ data and JSON metadata atomically."""

import pathlib
import tempfile

import numpy

import asc

values = numpy.arange(6, dtype=numpy.float32).reshape(3, 2)
with tempfile.TemporaryDirectory() as directory:
    path = pathlib.Path(directory) / "experiment.npz"
    asc.data.save_npz(path, {"values": values}, metadata={"units": "m"})
    restored = asc.data.load_npz(path)

numpy.testing.assert_array_equal(restored.arrays["values"], values)
assert restored.metadata == {"units": "m"}
print(restored.metadata)
