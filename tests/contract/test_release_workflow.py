# Copyright 2026 AI4SciComp contributors.
# SPDX-License-Identifier: Apache-2.0
"""Verify the immutable, least-privilege release workflow contract."""

from __future__ import annotations

import pathlib
import re


def test_release_workflow_uses_one_tested_artifact_set() -> None:
    repository = pathlib.Path(__file__).parents[2]
    workflow = (repository / ".github" / "workflows" / "release.yml").read_text(
        encoding="utf-8"
    )

    assert '      - "v*"' in workflow
    assert "release tag {actual!r} must equal {expected!r}" in workflow
    assert "run: make build" in workflow
    assert "name: distributions" in workflow
    assert workflow.count("name: distributions") == 3
    assert "needs: build" in workflow
    assert "needs: publish-pypi" in workflow
    assert "--verify-tag" in workflow


def test_release_workflow_uses_oidc_and_least_privilege() -> None:
    repository = pathlib.Path(__file__).parents[2]
    workflow = (repository / ".github" / "workflows" / "release.yml").read_text(
        encoding="utf-8"
    )
    publish_job = workflow.split("\n  publish-pypi:", maxsplit=1)[1]
    publish_job = publish_job.split("\n  github-release:", maxsplit=1)[0]

    assert "environment: pypi" in publish_job
    assert "id-token: write" in publish_job
    assert "uv publish --trusted-publishing always" in publish_job
    assert "secrets." not in publish_job
    assert "contents: write" not in publish_job

    actions = re.findall(r"uses: [^@\s]+@([^\s]+)", workflow)
    assert actions
    assert all(re.fullmatch(r"[0-9a-f]{40}", revision) for revision in actions)
