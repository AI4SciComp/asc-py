# Migration notes

## 0.1.0 namespace decision

Early local discovery material used the import name `asc_py`. Before any
release, organization direction changed the public import to `asc` while the
distribution remained `asc-py`:

```python
import asc
```

There is no compatibility alias because no public release used `asc_py` and an
alias would create two identities. The unrelated PyPI project named `asc`
cannot coexist in the same environment.

## Future compatibility changes

Incompatible post-0.1 changes require an ADR, changelog and migration entry,
tests, and a deprecation period unless immediate security or data-corruption
risk makes that unsafe. Patch releases may correct behavior that violated the
documented contract.
