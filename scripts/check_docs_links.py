# Copyright 2026 AI4SciComp contributors.
# SPDX-License-Identifier: Apache-2.0
"""Validate repository-local links in Markdown documentation."""

from __future__ import annotations

import pathlib
import re
import sys
import urllib.parse

_LINK = re.compile(r"(?<!!)\[[^]]+\]\(([^)]+)\)")


def _markdown_files(root: pathlib.Path) -> tuple[pathlib.Path, ...]:
    candidates = tuple(root.glob("*.md")) + tuple((root / "docs").rglob("*.md"))
    return tuple(sorted(candidates))


def _local_target(source: pathlib.Path, raw_target: str) -> pathlib.Path | None:
    target = raw_target.strip().strip("<>")
    parsed = urllib.parse.urlsplit(target)
    if parsed.scheme or parsed.netloc or not parsed.path:
        return None
    return (source.parent / urllib.parse.unquote(parsed.path)).resolve()


def main() -> int:
    """Return nonzero when a repository-local Markdown link is broken."""
    root = pathlib.Path(__file__).resolve().parents[1]
    errors: list[str] = []
    for source in _markdown_files(root):
        text = source.read_text(encoding="utf-8")
        for match in _LINK.finditer(text):
            target = _local_target(source, match.group(1))
            if target is not None and not target.exists():
                relative_source = source.relative_to(root)
                errors.append(
                    f"{relative_source}: missing local link {match.group(1)!r}"
                )
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print("documentation links: all repository-local targets exist")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
