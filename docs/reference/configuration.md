# Configuration

{py:class}`asc.ArrayContext` records backend, dtype, CPU device, precision,
copy, and optional random-state policy. {py:class}`asc.CreationContext` pairs a
validated namespace with creation dtype/device policy. Both are immutable and
reject inconsistent backend or random-state values eagerly.

`CopyPolicy` distinguishes mandatory copy, copy-if-needed, and no-copy
boundaries. `PrecisionPolicy` records whether a backend may inherit defaults,
must preserve explicit precision, or may narrow where an API expressly permits
it. Version 0.1.0 normally fails rather than silently narrows.

`DataLoaderConfig`, `NpyOptions`, `NpzOptions`, `CsvOptions`, `Hdf5Options`, and
`MatOptions` validate data and persistence settings before I/O. Option records
do not perform imports, allocate arrays, or mutate global configuration.
`CsvOptions` rejects delimiters that conflict with formatted numbers and header
fields containing delimiters or newlines. `Hdf5Options` rejects compression
together with `chunks=False`, because filtered HDF5 datasets require chunking.
