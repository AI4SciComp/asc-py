# Functionality traceability

The [normative functionality ledger](../specification/functionality-matrix.md)
maps all comprehensive and documentation IDs to implementation, tests,
documentation, backend coverage, and completion status. A row is complete only
when every referenced path exists and every declared local gate passes without
a required skip.

The release audit compares that ledger with both normative runbooks. The public
API inventory additionally compares installed `__all__` declarations with the
autosummary source and rejects duplicate, missing, or internal targets.
