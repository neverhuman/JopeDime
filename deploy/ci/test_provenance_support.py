#!/usr/bin/env python3
"""Provenance bind: PR base vs exact-head for runner-change PRs."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]


def load_bind():
    path = ROOT / "deploy/runner/provenance_bind.py"
    spec = importlib.util.spec_from_file_location("provenance_bind", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bind = load_bind()

COMMON_TEMPLATE = """#!/bin/bash
JOPE_RUNNER_DEPLOYED=(
  /opt/jope-runner/common.sh
  /opt/jope-runner/run.sh
  /opt/jope-runner/disk-guard.sh
  /opt/jope-runner/prune.sh
  /opt/jope-runner/job-started.sh
  /opt/jope-runner/job-policy.py
  /etc/systemd/system/jope-runner@.service
  /etc/systemd/system/jope-runner@.service.d/runtime-max.conf
  /etc/systemd/system/jope-runner@.timer
  /etc/systemd/system/jope-runner-prune.service
  /etc/systemd/system/jope-runner-prune.timer
  /etc/systemd/system/jope-docker-proxy.service
  /etc/tmpfiles.d/jope-exclusive-cpu.conf
)
# marker={marker}
"""

DEPLOYED_REL = [
    "deploy/runner/common.sh",
    "deploy/runner/run.sh",
    "deploy/runner/disk-guard.sh",
    "deploy/runner/prune.sh",
    "deploy/runner/job-started.sh",
    "deploy/runner/job-policy.py",
    "deploy/runner/jope-runner@.service",
    "deploy/runner/jope-runner@.service.d/runtime-max.conf",
    "deploy/runner/jope-runner@.timer",
    "deploy/runner/jope-runner-prune.service",
    "deploy/runner/jope-runner-prune.timer",
    "deploy/runner/jope-docker-proxy.service",
    "deploy/runner/jope-exclusive-cpu.conf",
]

BASH_DIGEST = r"""
mapfile -t deployed < <(sed -n '/^JOPE_RUNNER_DEPLOYED=(/,/^)/p' deploy/runner/common.sh   | grep -oE '/(opt/jope-runner|etc/systemd/system|etc/tmpfiles.d)/[^ ]+')
files=()
for d in "${deployed[@]}"; do
  rel=${d#/opt/jope-runner/}
  rel=${rel#/etc/systemd/system/}
  rel=${rel#/etc/tmpfiles.d/}
  files+=("deploy/runner/$rel")
done
cat "${files[@]}" | sha256sum | cut -d' ' -f1
"""


def write_tree(repo: Path, marker: str) -> None:
    for rel in DEPLOYED_REL:
        path = repo / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if rel.endswith("common.sh"):
            path.write_text(COMMON_TEMPLATE.format(marker=marker))
        else:
            path.write_text(f"{rel}\n{marker}\n")
    (repo / "deploy/runner/entrypoint.sh").write_text(f"entrypoint\n{marker}\n")
    (repo / "deploy/runner/mac-capacity.json").write_text(json.dumps(
        {"runner_label": "jope-mac-ci"} if marker.startswith("head") else {}))


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()


def digest_of(repo: Path, rev=None) -> str:
    return bind.scripts_digest(repo, rev)
