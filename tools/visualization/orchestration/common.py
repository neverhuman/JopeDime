"""Configuration and admission checks for optional rendering orchestration.

Importing a module or asking for help never starts rendering or opens a network
connection. Artifact locations are mandatory at execution, never inferred from
this source directory. Historical runtime hashes remain deliberate admission pins.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import re

REPO = Path(__file__).resolve().parents[3]
_UNSET = Path("/__JAIN_ARTIFACT_ROOT_NOT_CONFIGURED__/artifacts/h256")


def product_roots():
    roots = [REPO]
    marker = REPO / ".git"
    if marker.is_file():
        line = marker.read_text(encoding="utf-8").strip()
        if line.startswith("gitdir: "):
            gitdir = (REPO / line[8:]).resolve()
            common = gitdir / "commondir"
            if common.is_file():
                roots.append((gitdir / common.read_text().strip()).resolve().parent)
    return tuple(roots)


def external_path(path):
    path = Path(path).expanduser()
    if not path.is_absolute():
        raise ValueError("JAIN artifact paths must be absolute")
    path = path.resolve()
    if any(path == root or root in path.parents for root in product_roots()):
        raise ValueError("Render outputs must stay outside the product checkout and its worktrees")
    return path


def artifact_root(*, required=False):
    value = os.environ.get("JAIN_ARTIFACT_ROOT")
    if not value:
        if required:
            raise ValueError("Set JAIN_ARTIFACT_ROOT to the external h256 artifact directory")
        return _UNSET
    return external_path(value)


def artifact_family(name):
    return external_path(artifact_root() / name)


def validate_configuration():
    root = artifact_root(required=True)
    if not root.is_dir():
        raise ValueError("JAIN_ARTIFACT_ROOT must be an existing external artifact directory")
    return root


def require_render_enabled(*, local=False, uhd=False):
    validate_configuration()
    flag = "JAIN_ENABLE_LOCAL_RENDER" if local else "JAIN_ENABLE_REMOTE_RENDER"
    if os.environ.get(flag) != "1":
        raise ValueError(f"Rendering is disabled; set {flag}=1 only for an authorized run")
    if uhd and os.environ.get("JAIN_ENABLE_UHD_RENDER") != "1":
        raise ValueError("UHD is stopped; renewed authorization and JAIN_ENABLE_UHD_RENDER=1 are required")
    if local:
        value = os.environ.get("JAIN_BLENDER")
        if not value:
            raise ValueError("Set JAIN_BLENDER to the checksum-pinned local Blender executable")
        runtime = Path(value).expanduser()
        if not runtime.is_absolute() or not runtime.resolve().is_file():
            raise ValueError("JAIN_BLENDER must be an existing absolute executable path")


def configured_hosts():
    hosts = tuple(os.environ.get("JAIN_RENDER_HOSTS", "xbabe3,xbabe1").split(","))
    if len(hosts) != 2 or len(set(hosts)) != 2 or any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", h) for h in hosts):
        raise ValueError("JAIN_RENDER_HOSTS must contain two distinct SSH aliases separated by a comma")
    return hosts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("field", choices=("repo", "root", "dendrite", "cuda", "path"))
    parser.add_argument("path", nargs="?")
    args = parser.parse_args()
    root = validate_configuration()
    if args.field == "path":
        if not args.path:
            parser.error("path requires an absolute path argument")
        print(external_path(args.path))
        return
    print({"repo": REPO, "root": root, "dendrite": external_path(root / "variable-attribution-v1/dendrite-final-v1"),
           "cuda": REPO / "docs/plans/blender-fused-first-render/language-icons/nvidia-cuda-toolkit.jpg"}[args.field])


if __name__ == "__main__":
    main()
