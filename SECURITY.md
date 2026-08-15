# Security Policy

## Supported Versions

| Version | Security support |
| --- | --- |
| `0.1.x` | Supported only after public 0.1.0 publication |
| unreleased code | No security support promise |

Publication is blocked until private vulnerability reporting is enabled and
verified. No response or remediation deadline is promised before release.

## Reporting a Vulnerability

Use the repository Security page at
<https://github.com/AI4SciComp/asc-py/security> and select private vulnerability
reporting when available. Do not disclose suspected vulnerabilities publicly.
If the form is unavailable, contact repository owner `@escapetiger` through a
previously established private channel.

Include the affected version/commit, platform and dependency versions, smallest
reproducer, impact, and whether the issue involves untrusted input, numerical
integrity, credentials, package metadata, or the software supply chain.

## Trust Boundaries

asc-py validates documented array contracts; it is not a sandbox. Callers own
validation outside supported shapes, dtypes, CPU devices, and dense layouts.
Diagnostic metadata is bounded and excludes array contents and secrets.
Conversion is explicit but does not authenticate data. Numerical correctness is
not cryptographic integrity.

Runtime dependencies are bounded in `pyproject.toml`, development resolution is
locked, and GitHub Actions must be pinned by full commit SHA. Report any lock,
artifact, license, checksum, action, SBOM, or attestation mismatch as a supply-
chain concern.
