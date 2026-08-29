# ASC Python research-sidecar pilot

This authored state models `asc-py` as five bounded semantic contexts. ASC OS
is invoked from a separate local tool environment; it is not listed in
`pyproject.toml`, imported by `src/asc`, or included in package artifacts.

The root cover is:

```text
CTX-ROOT <- CTX-API + CTX-BACKENDS + CTX-DATA + CTX-RELEASE
```

Required overlaps make native arrays, explicit conversion, optional backend
imports, data/persistence types, reproducibility, documentation, support, and
compatibility policies agree. These are declarative compatibility checks, not
a formal proof of a sheaf condition.

Generated Codex and Claude projections remain ignored local output under
`.ai/generated/`. Their source hashes, generator version, deterministic
comparison, and deliberate-conflict result are recorded in the draft pull
request and post-runbook evidence report.

The pilot neither implements the planned `asc-no` repository nor changes
runtime, source, tests, package metadata, publication state, or release state.
