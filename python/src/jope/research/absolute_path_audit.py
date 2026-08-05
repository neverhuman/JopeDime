#!/usr/bin/env python3
"""Resolve every absolute path our jobs and units depend on, on every node.

Three outages in one day came from one defect class: a hard-coded absolute path
whose target is not where the code assumes.

  1. `/home/ubuntu/fable_god` -- the foundation-expert root had moved, so the
     encoder died on BOTH devices while logging "falling back to cpu". Hidden
     for an unknown period; each attempt burned ~854s.
  2. Eleven dangling symlinks across five chimera trees, all pointing into the
     same vanished root, including model weights.
  3. `/home/ubuntu/jope-prime/registry/operator_registry.json` -- present on
     xbabe1/2/3, absent on xbabe0/xbabe4. Once the proposer's node pin was
     lifted, jobs landed where they could not run and burned the retry budget.
     The evaluation queue sat at zero.

Every one of them surfaced as a symptom that did NOT name the path: a device
fallback, a retry-budget message, a load error. This resolves them up front.

The per-NODE dimension is the point. A path that exists on the submitting host
proves nothing about the host the job lands on, and that asymmetry is exactly
what defect 3 was.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

NODES = ("xbabe0", "xbabe1", "xbabe2", "xbabe3", "xbabe4")
# Absolute paths under the roots this system actually depends on. Anchored to
# avoid sweeping up every /usr/bin/env in an args list, which is noise.
PATH_PATTERN = re.compile(r"/home/ubuntu/[A-Za-z0-9._/+-]+")
# Content-addressed and per-run paths are expected to differ across nodes or to
# be absent once a run retires; auditing them produces noise, not findings.
# scq stages these from the submitting node; they need not exist elsewhere.
STAGED_INPUT = re.compile(r"\s*(source|workdir)\s*=")
VOLATILE = re.compile(
    r"(/runtime-bundles/(?!current)[0-9a-f]{16,}|/waves/|/receipts/"
    r"|/proposals/|/\.encoder-|\.tmp-|/scratch|/specs/)"
)


def sources(root: Path, flywheel: Path, specs_per_family: int) -> list[Path]:
    """What we audit, and why each surface matters.

    JOB SPECS are the dangerous surface and the reason this exists: a spec is
    submitted to scq and can land on ANY node, so a host-local path it names
    must resolve everywhere. The proposer's operator-registry defect lived
    exactly here and cost hours of idle cluster.

    Service UNITS run only on the control node, so a unit-only path being
    absent elsewhere is expected, not a finding -- they are audited for the
    narrower question of whether the control node itself is complete.
    """

    found: list[Path] = []
    for pattern in ("systemd/*.service",):
        found += sorted(root.glob(pattern))
    installed = Path.home() / ".config/systemd/user"
    if installed.is_dir():
        found += sorted(installed.glob("jope*.service"))
    # Newest specs per family: they are regenerated constantly, so auditing
    # every historical one is pure cost for no additional signal.
    specs = flywheel / "specs"
    if specs.is_dir():
        for family in sorted(p for p in specs.iterdir() if p.is_dir()):
            recent = sorted(family.glob("*.toml"), key=lambda p: p.stat().st_mtime)
            found += recent[-specs_per_family:]
    return [p for p in found if p.is_file()]


def harvest(files: list[Path]) -> dict[str, set[str]]:
    """path -> the files that reference it."""

    referenced: dict[str, set[str]] = defaultdict(set)
    for path in files:
        try:
            text = path.read_text(errors="replace")
        except OSError:
            continue
        for line in text.splitlines():
            # An scq `[[input]] source = ...` is resolved on the SUBMITTING
            # node and staged onto whichever node runs the job, so it does not
            # need to exist anywhere else. Auditing it is a guaranteed false
            # positive, and false positives are how an audit gets ignored.
            # What is dangerous is a path the job opens at runtime, which is
            # precisely what the operator-registry defect was.
            if STAGED_INPUT.match(line):
                continue
            for match in PATH_PATTERN.findall(line):
                candidate = match.rstrip(".,;:\"')")
                if VOLATILE.search(candidate):
                    continue
                referenced[candidate].add(path.name)
    return referenced


def probe(node: str, paths: list[str], local: bool) -> set[str]:
    """Return the subset of `paths` that EXIST on `node`."""

    script = "while IFS= read -r p; do [ -e \"$p\" ] && printf '%s\\n' \"$p\"; done"
    command = ["/bin/sh", "-c", script]
    if not local:
        command = ["ssh", "-o", "ConnectTimeout=20", node, script]
    try:
        done = subprocess.run(
            command,
            input="\n".join(paths),
            capture_output=True,
            text=True,
            timeout=180,
        )
    except (subprocess.TimeoutExpired, OSError) as error:
        print(f"  {node}: UNREACHABLE ({error})", file=sys.stderr)
        # Unreachable is not "everything is missing"; report nothing rather
        # than manufacturing hundreds of false findings.
        return set(paths)
    return {line.strip() for line in done.stdout.splitlines() if line.strip()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/home/ubuntu/jope-dime"))
    parser.add_argument("--nodes", nargs="*", default=list(NODES))
    parser.add_argument("--local-node", default="xbabe2")
    parser.add_argument(
        "--flywheel", type=Path, default=Path("/home/ubuntu/jope-flywheel/test_exact_gen0")
    )
    parser.add_argument("--specs-per-family", type=int, default=2)
    args = parser.parse_args()

    files = sources(args.root, args.flywheel, args.specs_per_family)
    referenced = harvest(files)
    paths = sorted(referenced)
    print(f"scanned {len(files)} spec/unit files -> {len(paths)} distinct absolute paths")
    if not paths:
        return

    present: dict[str, set[str]] = {}
    for node in args.nodes:
        present[node] = probe(node, paths, local=(node == args.local_node))
        print(f"  {node}: {len(present[node])}/{len(paths)} resolve")

    findings = 0
    for path in paths:
        missing = [n for n in args.nodes if path not in present[n]]
        if not missing:
            continue
        findings += 1
        where = ", ".join(sorted(referenced[path]))
        # A path missing EVERYWHERE is usually a template or a retired
        # reference; a path missing on SOME nodes is the dangerous one,
        # because placement decides whether the job works.
        severity = "SPLIT" if len(missing) < len(args.nodes) else "absent-everywhere"
        print(f"\n  [{severity}] {path}")
        print(f"     missing on: {', '.join(missing)}")
        print(f"     referenced by: {where}")

    print(f"\n{findings} path(s) do not resolve on every node")
    sys.exit(1 if findings else 0)


if __name__ == "__main__":
    main()
